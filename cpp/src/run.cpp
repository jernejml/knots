// `knots run` — one-shot pipeline: per-frame inference + per-board stitching.
//
// Walks --input-dir of frame PNGs, groups them by board, and for each board
// runs YOLO11-seg inference on every frame then raster-unions the per-frame
// polygons into a single {board}.json. No intermediate per-frame JSON is
// written — predictions live in memory only.
//
// Frame-selection knobs match `knots infer`:
//   --frames LIST            comma-separated stems
//   --frames-file FILE       one stem per line ('#' comments allowed)
//   --splits-csv + --split   restrict to a split from analysis/splits.csv
//
// Resume: a board whose {board}.json already exists is skipped (its frames
// are not even loaded) unless --force is passed. For finer-grained resume
// (per-frame), use `knots infer` + `knots stitch` separately.
//
// Stitching knobs match `knots stitch`: --stride-px, --simplify-eps.

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

namespace {

struct Args {
    fs::path model;
    fs::path input_dir;
    fs::path output_dir;
    std::string frames_csv;
    fs::path frames_file;
    fs::path splits_csv;
    std::string split;
    float conf = 0.25f;
    int stride_px = 320;
    float simplify_eps_px = 1.0f;
    bool prefer_cuda = true;
    bool force = false;
};

void PrintUsage() {
    std::cerr << "usage: knots run --model M --input-dir IN --output-dir OUT [opts]\n"
                 "  --frames LIST            comma-separated frame stems, e.g. 0_5,100_3\n"
                 "  --frames-file FILE       one frame stem per line\n"
                 "  --splits-csv PATH        analysis/splits.csv\n"
                 "  --split {train,val,test} with --splits-csv: restrict to this split\n"
                 "  --conf C                 confidence threshold (default 0.25)\n"
                 "  --stride-px N            frame stride in board px (default 320)\n"
                 "  --simplify-eps F         approxPolyDP eps in board px (default 1.0)\n"
                 "  --cpu                    force CPU execution provider\n"
                 "  --force                  overwrite existing per-board outputs\n";
}

bool ParseArgs(int argc, char** argv, Args& out) {
    int i = 1;
    while (i < argc) {
        std::string a = argv[i];
        if (a == "--model" && cli::RequireNext(a, i, argc)) {
            out.model = argv[++i];
        } else if (a == "--input-dir" && cli::RequireNext(a, i, argc)) {
            out.input_dir = argv[++i];
        } else if (a == "--output-dir" && cli::RequireNext(a, i, argc)) {
            out.output_dir = argv[++i];
        } else if (a == "--frames" && cli::RequireNext(a, i, argc)) {
            out.frames_csv = argv[++i];
        } else if (a == "--frames-file" && cli::RequireNext(a, i, argc)) {
            out.frames_file = argv[++i];
        } else if (a == "--splits-csv" && cli::RequireNext(a, i, argc)) {
            out.splits_csv = argv[++i];
        } else if (a == "--split" && cli::RequireNext(a, i, argc)) {
            out.split = argv[++i];
        } else if (a == "--conf" && cli::RequireNext(a, i, argc)) {
            out.conf = std::stof(argv[++i]);
        } else if (a == "--stride-px" && cli::RequireNext(a, i, argc)) {
            out.stride_px = std::stoi(argv[++i]);
        } else if (a == "--simplify-eps" && cli::RequireNext(a, i, argc)) {
            out.simplify_eps_px = std::stof(argv[++i]);
        } else if (a == "--cpu") {
            out.prefer_cuda = false;
        } else if (a == "--force") {
            out.force = true;
        } else if (a == "--help" || a == "-h") {
            return false;
        } else {
            std::cerr << "unrecognised arg: " << a << "\n";
            return false;
        }
        ++i;
    }
    if (out.model.empty() || out.input_dir.empty() || out.output_dir.empty()) {
        std::cerr << "--model, --input-dir, --output-dir are required\n";
        return false;
    }
    if (!out.splits_csv.empty() && out.split.empty()) {
        std::cerr << "--splits-csv requires --split {train,val,test}\n";
        return false;
    }
    return true;
}

}  // namespace

int CmdRun(int argc, char** argv) {
    Args args;
    if (!ParseArgs(argc, argv, args)) {
        PrintUsage();
        return 2;
    }
    if (!fs::exists(args.model)) {
        std::cerr << "model not found: " << args.model << "\n";
        return 1;
    }
    if (!fs::is_directory(args.input_dir)) {
        std::cerr << "input dir not found: " << args.input_dir << "\n";
        return 1;
    }
    fs::create_directories(args.output_dir);

    try {
        std::vector<std::string> explicit_stems;
        if (!args.frames_csv.empty()) explicit_stems = cli::ParseFramesList(args.frames_csv);
        if (!args.frames_file.empty()) {
            auto from_file = cli::ParseFramesFile(args.frames_file);
            explicit_stems.insert(explicit_stems.end(), from_file.begin(), from_file.end());
        }
        std::unordered_set<int> boards_filter;
        if (!args.splits_csv.empty()) {
            boards_filter = cli::LoadBoardsInSplit(args.splits_csv, args.split);
            if (boards_filter.empty()) {
                std::cerr << "warning: no boards match split " << args.split << " in "
                          << args.splits_csv << "\n";
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
        Ort::Session session = MakeSession(env, args.model, args.prefer_cuda, ep);

        std::cerr << "knots run: " << by_board.size() << " board(s) / " << total_frames
                  << " frame(s)  ep=" << ep << "  conf=" << args.conf
                  << "  stride=" << args.stride_px
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

            auto fp_list = pipeline::InferBoardFrames(session, args.input_dir, frames, args.conf,
                                                     infer_stats, frame_done);
            if (fp_list.empty()) {
                std::cerr << "  WARN board " << board << ": no usable frames, skipping stitch\n";
                continue;
            }
            total_knots += StitchBoardToJson(board, std::move(fp_list), args.stride_px,
                                             args.simplify_eps_px, out_path);
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
