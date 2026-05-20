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

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/imgcodecs.hpp>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "knots/cli_util.hpp"
#include "knots/commands.hpp"
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

const std::regex kLabelFileRe(R"(^(\d+)_(\d+)\.txt$)");

// Parse YOLO bbox labels for one frame; emit each box as a 4-vertex
// rectangle polygon in frame-local pixel coords, clipped to image bounds.
std::vector<std::vector<cv::Point>> ParseYoloBboxesAsPolys(const fs::path& label_path, int w,
                                                           int h) {
    std::vector<std::vector<cv::Point>> polys;
    std::ifstream f(label_path);
    if (!f) return polys;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        int cls;
        float cx, cy, bw, bh;
        if (!(iss >> cls >> cx >> cy >> bw >> bh)) continue;
        const float fx = cx * w, fy = cy * h;
        const float fbw = bw * w, fbh = bh * h;
        const int x1 = std::max(0, static_cast<int>(std::round(fx - fbw / 2)));
        const int y1 = std::max(0, static_cast<int>(std::round(fy - fbh / 2)));
        const int x2 = std::min(w, static_cast<int>(std::round(fx + fbw / 2)));
        const int y2 = std::min(h, static_cast<int>(std::round(fy + fbh / 2)));
        if (x2 <= x1 || y2 <= y1) continue;
        polys.push_back({{x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}});
    }
    return polys;
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
        // Resolve which boards we want.
        std::unordered_set<int> boards_filter;
        if (!args.boards_csv.empty()) boards_filter = cli::ParseBoardsList(args.boards_csv);
        if (!args.boards_file.empty()) boards_filter = cli::ParseBoardsFile(args.boards_file);
        if (!args.splits_csv.empty())
            boards_filter = cli::LoadBoardsInSplit(args.splits_csv, args.split);

        // Walk labels dir; group label files by board.
        std::map<int, std::vector<int>> board_frames;
        for (const auto& entry : fs::directory_iterator(args.labels_dir)) {
            if (!entry.is_regular_file()) continue;
            std::string name = entry.path().filename().string();
            std::smatch m;
            if (!std::regex_match(name, m, kLabelFileRe)) continue;
            const int board = std::stoi(m[1]);
            const int frame_idx = std::stoi(m[2]);
            if (!boards_filter.empty() && !boards_filter.count(board)) continue;
            board_frames[board].push_back(frame_idx);
        }

        std::cerr << "knots gt-stitch: " << board_frames.size() << " board(s)"
                  << "  stride=" << args.stride_px << "  force=" << (args.force ? "true" : "false")
                  << "\n";

        size_t n_boards_out = 0, n_skipped = 0, total_polys = 0;
        for (auto& [board, frames] : board_frames) {
            fs::path out_path = args.output_dir / (std::to_string(board) + ".json");
            if (fs::exists(out_path) && !args.force) {
                ++n_skipped;
                continue;
            }

            std::vector<FramePolys> fp_list;
            fp_list.reserve(frames.size());
            for (int frame_idx : frames) {
                const std::string stem = std::to_string(board) + "_" + std::to_string(frame_idx);
                fs::path img_path = args.images_dir / (stem + ".png");
                cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_COLOR);
                if (img.empty()) {
                    std::cerr << "  WARN unreadable image " << img_path << " — skipping frame\n";
                    continue;
                }
                FramePolys fp;
                fp.frame_idx = frame_idx;
                fp.width = img.cols;
                fp.height = img.rows;
                fp.polygons =
                    ParseYoloBboxesAsPolys(args.labels_dir / (stem + ".txt"), img.cols, img.rows);
                fp_list.push_back(std::move(fp));
            }
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
