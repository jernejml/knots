// `knots run` — one-shot pipeline: per-frame inference + per-board stitching.
//
// Walks args.input_dir of frame PNGs, groups them by board, and for each
// board runs YOLO11-seg inference on every frame then raster-unions the
// per-frame polygons into a single {board}.json. No intermediate per-frame
// JSON is written — predictions live in memory only.
//
// Frame-selection knobs (FramesFilter + SplitsFilter) and inference /
// stitching knobs are populated by cli_options::AddRunOptions before this
// function is invoked from main's callback.
//
// Resume: a board whose {board}.json already exists is skipped (its frames
// are not even loaded) unless --force is passed. For finer-grained resume
// (per-frame), use `knots infer` + `knots stitch` separately.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "knots/cli_util.hpp"
#include "knots/commands.hpp"
#include "knots/inference.hpp"
#include "knots/pipeline.hpp"
#include "knots/stitching.hpp"

namespace fs = std::filesystem;

namespace knots {

int CmdRun(const RunArgs& args) {
    if (!fs::exists(args.inference.model)) {
        std::cerr << "model not found: " << args.inference.model << "\n";
        return 1;
    }
    if (!fs::is_directory(args.input_dir)) {
        std::cerr << "input dir not found: " << args.input_dir << "\n";
        return 1;
    }
    fs::create_directories(args.output_dir);

    try {
        const auto explicit_stems =
            cli::CollectExplicitStems(args.frames.frames, args.frames.frames_file);
        std::unordered_set<int> boards_filter;
        if (!args.splits.partitions_json.empty()) {
            boards_filter = cli::LoadBoardsInSplit(args.splits.partitions_json, args.splits.split);
            if (boards_filter.empty()) {
                std::cerr << "warning: no boards match split " << args.splits.split << " in "
                          << args.splits.partitions_json << "\n";
            }
        }
        const auto by_board =
            pipeline::CollectFramesByBoard(args.input_dir, ".png", explicit_stems, boards_filter);
        if (by_board.empty()) {
            std::cerr << "no frames to process\n";
            return 0;
        }
        size_t total_frames = 0;
        for (const auto& [_, frames] : by_board) total_frames += frames.size();

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "knots");
        std::string ep;
        Ort::Session session =
            MakeSession(env, args.inference.model, !args.inference.cpu_only, ep);

        std::cerr << "knots run: " << by_board.size() << " board(s) / " << total_frames
                  << " frame(s)  ep=" << ep << "  conf=" << args.inference.conf
                  << "  stride=" << args.stitch.stride_px
                  << "  force=" << (args.force ? "true" : "false") << "\n";

        size_t n_boards_out = 0, n_boards_skipped = 0;
        size_t total_knots = 0;
        pipeline::InferStats infer_stats;
        const size_t heartbeat = std::max<size_t>(20, total_frames / 20);
        size_t global_idx = 0;
        auto frame_done = [&]() {
            ++global_idx;
            if (global_idx % heartbeat == 0 || global_idx == total_frames) {
                std::cerr << "  ... " << global_idx << "/" << total_frames << " frames\n";
            }
        };

        for (const auto& [board, frames] : by_board) {
            fs::path out_path = args.output_dir / (std::to_string(board) + ".json");
            if (fs::exists(out_path) && !args.force) {
                ++n_boards_skipped;
                for (size_t i = 0; i < frames.size(); ++i) frame_done();
                continue;
            }

            auto fp_list = pipeline::InferBoardFrames(session, args.input_dir, frames,
                                                     args.inference.conf, infer_stats, frame_done);
            if (fp_list.empty()) {
                std::cerr << "  WARN board " << board << ": no usable frames, skipping stitch\n";
                continue;
            }
            total_knots += StitchBoardToJson(board, std::move(fp_list), args.stitch.stride_px,
                                             args.stitch.simplify_eps_px, out_path);
            ++n_boards_out;
        }

        std::cerr << "\nResult\n"
                  << "  boards: written=" << n_boards_out << "  skipped=" << n_boards_skipped
                  << "\n"
                  << "  frames: processed=" << infer_stats.processed
                  << "  unread=" << infer_stats.unread << "  failed=" << infer_stats.failed << "\n"
                  << "  total per-board polygons=" << total_knots << "\n"
                  << "  output dir: " << args.output_dir << "\n";
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

}  // namespace knots
