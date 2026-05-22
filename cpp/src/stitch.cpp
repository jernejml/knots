// `knots stitch` — per-board polygon stitching from per-frame inference JSONs.
//
// Reads {board}_{frame}.json files written by `knots infer`, groups by board,
// projects each frame's polygons into board coords via stride_px, raster-
// unions overlapping shapes, and writes one {board}.json per board.

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <regex>
#include <string>
#include <vector>

#include "knots/commands.hpp"
#include "knots/stitching.hpp"

namespace fs = std::filesystem;

namespace knots {

namespace {

const std::regex kJsonStemRe(R"(^(\d+)_(\d+)$)");

}  // namespace

int CmdStitch(const StitchArgs& args) {
    if (!fs::is_directory(args.input_dir)) {
        std::cerr << "input dir not found: " << args.input_dir << "\n";
        return 1;
    }
    fs::create_directories(args.output_dir);

    try {
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

        std::cerr << "knots stitch: " << n_files << " frame JSON(s) across " << by_board.size()
                  << " board(s)"
                  << (n_unparseable ? " (" + std::to_string(n_unparseable) + " unparseable)" : "")
                  << "  stride=" << args.stitch.stride_px
                  << "  force=" << (args.force ? "true" : "false") << "\n";

        size_t n_boards_out = 0, n_skipped = 0, total_polys = 0;
        for (auto& [board, frames] : by_board) {
            fs::path out_path = args.output_dir / (std::to_string(board) + ".json");
            if (fs::exists(out_path) && !args.force) {
                ++n_skipped;
                continue;
            }
            total_polys += StitchBoardToJson(board, std::move(frames), args.stitch.stride_px,
                                             args.stitch.simplify_eps_px, out_path);
            ++n_boards_out;
        }

        std::cerr << "\nResult\n"
                  << "  boards processed=" << n_boards_out << "  skipped=" << n_skipped << "\n"
                  << "  total per-board polygons=" << total_polys << "\n"
                  << "  output dir: " << args.output_dir << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

}  // namespace knots
