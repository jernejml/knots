// `knots gt-stitch` — per-board GT polygon stitching from per-frame bbox labels.
//
// Reads YOLO bbox labels (cls cx cy w h) from --labels-dir, projects each bbox
// as a 4-vertex rectangle, translates by (frame_idx * stride_px, 0), and uses
// the SAME raster-union pipeline as `knots stitch` so prediction and GT
// per-board polygons are bit-identical when the inputs match.
//
// Frame dimensions are read from --images-dir (frame width/height needed to
// denormalise YOLO coords). All frames within a board should have the same
// height in this dataset; the stitcher warns if they don't.
//
// Optional filtering:
//   --boards LIST | --boards-file FILE
//   --splits-csv PATH + --split {train,val,test}

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

namespace {

struct GtStitchArgs {
    fs::path labels_dir;
    fs::path images_dir;
    fs::path output_dir;
    int stride_px = 320;
    float simplify_eps_px = 1.0f;
    std::string boards_csv;
    fs::path boards_file;
    fs::path splits_csv;
    std::string split;
    bool force = false;
};

void PrintUsage() {
    std::cerr << "usage: knots gt-stitch --labels-dir L --images-dir I --output-dir O [opts]\n"
                 "  --labels-dir DIR         YOLO bbox labels (per-frame .txt)\n"
                 "  --images-dir DIR         frame PNGs (for frame dimensions)\n"
                 "  --output-dir DIR         per-board GT JSONs\n"
                 "  --stride-px N            frame stride (default 320)\n"
                 "  --simplify-eps F         approxPolyDP eps in board px (default 1.0)\n"
                 "  --boards LIST            comma-separated board IDs to restrict to\n"
                 "  --boards-file FILE       one board ID per line ('#' comments allowed)\n"
                 "  --splits-csv PATH        analysis/splits.csv\n"
                 "  --split {train,val,test} with --splits-csv: pick boards in this split\n"
                 "  --force                  overwrite existing outputs\n";
}

bool ParseArgs(int argc, char** argv, GtStitchArgs& out) {
    int i = 1;
    while (i < argc) {
        std::string a = argv[i];
        if (a == "--labels-dir" && cli::RequireNext(a, i, argc)) {
            out.labels_dir = argv[++i];
        } else if (a == "--images-dir" && cli::RequireNext(a, i, argc)) {
            out.images_dir = argv[++i];
        } else if (a == "--output-dir" && cli::RequireNext(a, i, argc)) {
            out.output_dir = argv[++i];
        } else if (a == "--stride-px" && cli::RequireNext(a, i, argc)) {
            out.stride_px = std::stoi(argv[++i]);
        } else if (a == "--simplify-eps" && cli::RequireNext(a, i, argc)) {
            out.simplify_eps_px = std::stof(argv[++i]);
        } else if (a == "--boards" && cli::RequireNext(a, i, argc)) {
            out.boards_csv = argv[++i];
        } else if (a == "--boards-file" && cli::RequireNext(a, i, argc)) {
            out.boards_file = argv[++i];
        } else if (a == "--splits-csv" && cli::RequireNext(a, i, argc)) {
            out.splits_csv = argv[++i];
        } else if (a == "--split" && cli::RequireNext(a, i, argc)) {
            out.split = argv[++i];
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
    if (out.labels_dir.empty() || out.images_dir.empty() || out.output_dir.empty()) {
        std::cerr << "--labels-dir, --images-dir, --output-dir are required\n";
        return false;
    }
    if (!out.splits_csv.empty() && out.split.empty()) {
        std::cerr << "--splits-csv requires --split {train,val,test}\n";
        return false;
    }
    return true;
}

}  // namespace

int CmdGtStitch(int argc, char** argv) {
    GtStitchArgs args;
    if (!ParseArgs(argc, argv, args)) {
        PrintUsage();
        return 2;
    }
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
        std::unordered_set<int> boards_filter;
        if (!args.boards_csv.empty()) boards_filter = cli::ParseBoardsList(args.boards_csv);
        if (!args.boards_file.empty()) boards_filter = cli::ParseBoardsFile(args.boards_file);
        if (!args.splits_csv.empty())
            boards_filter = cli::LoadBoardsInSplit(args.splits_csv, args.split);

        const auto by_board =
            pipeline::CollectFramesByBoard(args.labels_dir, ".txt", {}, boards_filter);

        std::cerr << "knots gt-stitch: " << by_board.size() << " board(s)"
                  << "  stride=" << args.stride_px << "  force=" << (args.force ? "true" : "false")
                  << "\n";

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

            total_polys += StitchBoardToJson(board, std::move(fp_list), args.stride_px,
                                             args.simplify_eps_px, out_path);
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
