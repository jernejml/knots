// `knots gt-stitch` — per-board GT polygon stitching from per-frame bbox labels.
//
// Reads YOLO bbox labels (cls cx cy w h) from args.labels_dir, projects each
// bbox as a 4-vertex rectangle, translates by (frame_idx * stride_px, 0),
// and uses the SAME raster-union pipeline as `knots stitch` so prediction
// and GT per-board polygons are bit-identical when the inputs match.
//
// Frame dimensions are read from args.images_dir (frame width/height needed
// to denormalise YOLO coords). All frames within a board should have the
// same height in this dataset; the stitcher warns if they don't.

#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "knots/cli_util.hpp"
#include "knots/commands.hpp"
#include "knots/pipeline.hpp"
#include "knots/stitching.hpp"

namespace fs = std::filesystem;

namespace knots {

int CmdGtStitch(const GtStitchArgs& args) {
    if (!fs::is_directory(args.labels_dir)) {
        std::cerr << "labels dir not found: " << args.labels_dir << "\n";
        return 1;
    }
    if (!fs::is_directory(args.images_dir)) {
        std::cerr << "images dir not found: " << args.images_dir << "\n";
        return 1;
    }
    fs::create_directories(args.output_dir);

    try {
        const auto boards_filter = cli::BuildBoardsFilter(
            args.boards.boards, args.boards.boards_file,
            args.splits.partitions_json, args.splits.split);

        const auto by_board =
            pipeline::CollectFramesByBoard(args.labels_dir, ".txt", {}, boards_filter);

        std::cerr << "knots gt-stitch: " << by_board.size() << " board(s)"
                  << "  stride=" << args.stitch.stride_px
                  << "  force=" << (args.force ? "true" : "false") << "\n";

        size_t n_boards_out = 0, n_skipped = 0, total_polys = 0;
        for (const auto& [board, frames] : by_board) {
            fs::path out_path = args.output_dir / (std::to_string(board) + ".json");
            if (fs::exists(out_path) && !args.force) {
                ++n_skipped;
                continue;
            }

            auto fp_list =
                pipeline::LoadGtBoardFrames(args.labels_dir, args.images_dir, frames);
            if (fp_list.empty()) continue;

            total_polys += StitchBoardToJson(board, std::move(fp_list), args.stitch.stride_px,
                                             args.stitch.simplify_eps_px, out_path);
            ++n_boards_out;
        }

        std::cerr << "\nResult\n"
                  << "  boards processed=" << n_boards_out << "  skipped=" << n_skipped << "\n"
                  << "  total per-board GT polygons=" << total_polys << "\n"
                  << "  output dir: " << args.output_dir << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

}  // namespace knots
