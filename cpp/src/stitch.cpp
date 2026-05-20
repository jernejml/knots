// `knots stitch` — per-board polygon stitching from per-frame inference JSONs.
//
// For every board with per-frame JSONs in --input-dir:
//   1. Allocate a board-sized mask (board_width × board_height).
//   2. For each frame's polygons, translate vertices by (frame_idx * STRIDE_PX,
//      0) and fillPoly onto the mask.
//   3. findContours(RETR_EXTERNAL) extracts merged contours — overlapping or
//      touching polygons from the 50% frame overlap merge automatically.
//   4. Simplify each contour with approxPolyDP and write one {board}.json with
//      the resulting per-board polygon list.
//
// Algorithm per CLAUDE.md:
//   "translate each polygon by `frame_idx * STRIDE_PX` (frames are 640 px
//    wide but advance by only 320 px) and take the raster union via
//    cv::fillPoly + cv::findContours"

#include "knots/commands.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

namespace knots {

namespace {

struct StitchArgs {
    fs::path input_dir;
    fs::path output_dir;
    int stride_px = 320;
    float simplify_eps_px = 1.0f;
    bool force = false;
};

void PrintUsage() {
    std::cerr <<
        "usage: knots stitch --input-dir IN --output-dir OUT [opts]\n"
        "  --input-dir DIR        per-frame JSONs from `knots infer`\n"
        "  --output-dir DIR       per-board JSONs (one .json per board)\n"
        "  --stride-px N          frame stride in pixels (default 320)\n"
        "  --simplify-eps F       approxPolyDP eps in board px (default 1.0)\n"
        "  --force                overwrite existing outputs\n";
}

bool RequireNext(const std::string& flag, int i, int argc) {
    if (i + 1 >= argc) {
        std::cerr << "missing value for " << flag << "\n";
        return false;
    }
    return true;
}

bool ParseArgs(int argc, char** argv, StitchArgs& out) {
    int i = 1;  // skip subcommand name
    while (i < argc) {
        std::string a = argv[i];
        if (a == "--input-dir" && RequireNext(a, i, argc))     { out.input_dir = argv[++i]; }
        else if (a == "--output-dir" && RequireNext(a, i, argc)){ out.output_dir = argv[++i]; }
        else if (a == "--stride-px" && RequireNext(a, i, argc)) { out.stride_px = std::stoi(argv[++i]); }
        else if (a == "--simplify-eps" && RequireNext(a, i, argc)) { out.simplify_eps_px = std::stof(argv[++i]); }
        else if (a == "--force")                                { out.force = true; }
        else if (a == "--help" || a == "-h")                    { return false; }
        else {
            std::cerr << "unrecognised arg: " << a << "\n";
            return false;
        }
        ++i;
    }
    if (out.input_dir.empty() || out.output_dir.empty()) {
        std::cerr << "--input-dir and --output-dir are required\n";
        return false;
    }
    return true;
}

struct FramePolys {
    int frame_idx;
    int width;
    int height;
    std::vector<std::vector<cv::Point>> polygons;  // in frame-local px coords
};

const std::regex kJsonStemRe(R"(^(\d+)_(\d+)$)");

}  // namespace


int CmdStitch(int argc, char** argv) {
    StitchArgs args;
    if (!ParseArgs(argc, argv, args)) {
        PrintUsage();
        return 2;
    }
    if (!fs::is_directory(args.input_dir)) {
        std::cerr << "input dir not found: " << args.input_dir << "\n";
        return 1;
    }
    fs::create_directories(args.output_dir);

    try {
        // Walk input dir; group frame JSONs by board id.
        std::map<int, std::vector<FramePolys>> by_board;
        size_t n_files = 0, n_unparseable = 0;
        for (const auto& entry : fs::directory_iterator(args.input_dir)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() != ".json") continue;
            const std::string stem = entry.path().stem().string();
            std::smatch m;
            if (!std::regex_match(stem, m, kJsonStemRe)) continue;
            const int board = std::stoi(m[1]);
            const int frame_idx = std::stoi(m[2]);

            nlohmann::json j;
            try {
                std::ifstream f(entry.path());
                f >> j;
            } catch (const std::exception& e) {
                std::cerr << "  WARN cannot parse " << stem << ".json: " << e.what() << "\n";
                ++n_unparseable;
                continue;
            }

            FramePolys fp;
            fp.frame_idx = frame_idx;
            fp.width = j["image_size"][0].get<int>();
            fp.height = j["image_size"][1].get<int>();
            for (const auto& det : j["detections"]) {
                std::vector<cv::Point> poly;
                for (const auto& pt : det["polygon"]) {
                    poly.emplace_back(pt[0].get<int>(), pt[1].get<int>());
                }
                if (poly.size() >= 3) fp.polygons.push_back(std::move(poly));
            }
            by_board[board].push_back(std::move(fp));
            ++n_files;
        }

        std::cerr << "knots stitch: " << n_files << " frame JSON(s) across "
                  << by_board.size() << " board(s)"
                  << (n_unparseable ? " (" + std::to_string(n_unparseable) + " unparseable)" : "")
                  << "  stride=" << args.stride_px
                  << "  force=" << (args.force ? "true" : "false") << "\n";

        size_t n_boards_out = 0, n_skipped = 0, total_polys = 0;
        size_t n_mixed_heights = 0;

        for (auto& [board, frames] : by_board) {
            fs::path out_path = args.output_dir / (std::to_string(board) + ".json");
            if (fs::exists(out_path) && !args.force) {
                ++n_skipped;
                continue;
            }

            std::sort(frames.begin(), frames.end(),
                      [](const auto& a, const auto& b) { return a.frame_idx < b.frame_idx; });

            const int board_height = frames.front().height;
            for (const auto& f : frames) {
                if (f.height != board_height) {
                    ++n_mixed_heights;
                    std::cerr << "  WARN board " << board << " frame " << f.frame_idx
                              << ": height=" << f.height << " vs board " << board_height
                              << " — using board height for mask\n";
                }
            }
            const int max_frame_idx = frames.back().frame_idx;
            const int board_width = max_frame_idx * args.stride_px + frames.back().width;

            cv::Mat mask = cv::Mat::zeros(board_height, board_width, CV_8U);
            for (const auto& f : frames) {
                const int dx = f.frame_idx * args.stride_px;
                for (const auto& poly : f.polygons) {
                    std::vector<cv::Point> shifted;
                    shifted.reserve(poly.size());
                    for (const auto& p : poly) {
                        shifted.emplace_back(p.x + dx, p.y);
                    }
                    std::vector<std::vector<cv::Point>> tmp{std::move(shifted)};
                    cv::fillPoly(mask, tmp, cv::Scalar(255));
                }
            }

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

            nlohmann::json jb;
            jb["board"] = board;
            jb["board_height"] = board_height;
            jb["board_width"] = board_width;
            jb["stride_px"] = args.stride_px;
            jb["knots"] = nlohmann::json::array();
            for (const auto& contour : contours) {
                std::vector<cv::Point> simplified;
                cv::approxPolyDP(contour, simplified, args.simplify_eps_px, /*closed=*/true);
                if (simplified.size() < 3) continue;
                nlohmann::json knot;
                knot["polygon"] = nlohmann::json::array();
                for (const auto& p : simplified) {
                    knot["polygon"].push_back({p.x, p.y});
                }
                jb["knots"].push_back(knot);
            }
            total_polys += jb["knots"].size();

            std::ofstream of(out_path);
            of << jb.dump(2);
            ++n_boards_out;
        }

        std::cerr << "\nResult\n"
                  << "  boards processed=" << n_boards_out
                  << "  skipped=" << n_skipped << "\n"
                  << "  total per-board polygons=" << total_polys
                  << (n_mixed_heights ? "  (" + std::to_string(n_mixed_heights) + " mixed-height warnings)" : "")
                  << "\n"
                  << "  output dir: " << args.output_dir << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

}  // namespace knots
