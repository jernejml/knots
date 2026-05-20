#include "knots/stitching.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

namespace knots {

StitchedBoard StitchBoardPolygons(int board, std::vector<FramePolys> frames, int stride_px,
                                  float simplify_eps_px) {
    StitchedBoard out{0, 0, {}};
    if (frames.empty()) return out;

    std::sort(frames.begin(), frames.end(),
              [](const auto& a, const auto& b) { return a.frame_idx < b.frame_idx; });

    out.board_height = frames.front().height;
    for (const auto& f : frames) {
        if (f.height != out.board_height) {
            std::cerr << "  WARN board " << board << " frame " << f.frame_idx
                      << ": height=" << f.height << " vs board " << out.board_height
                      << " — using board height for mask\n";
        }
    }
    const int max_frame_idx = frames.back().frame_idx;
    out.board_width = max_frame_idx * stride_px + frames.back().width;

    cv::Mat mask = cv::Mat::zeros(out.board_height, out.board_width, CV_8U);
    for (const auto& f : frames) {
        const int dx = f.frame_idx * stride_px;
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

    out.polygons.reserve(contours.size());
    for (const auto& contour : contours) {
        std::vector<cv::Point> simplified;
        cv::approxPolyDP(contour, simplified, simplify_eps_px, /*closed=*/true);
        if (simplified.size() < 3) continue;
        out.polygons.push_back(std::move(simplified));
    }
    return out;
}

size_t StitchBoardToJson(int board, std::vector<FramePolys> frames, int stride_px,
                         float simplify_eps_px, const std::filesystem::path& out_path) {
    // Preserve original behaviour: no output file when there are no frames.
    if (frames.empty()) return 0;
    auto sb = StitchBoardPolygons(board, std::move(frames), stride_px, simplify_eps_px);

    nlohmann::json jb;
    jb["board"] = board;
    jb["board_height"] = sb.board_height;
    jb["board_width"] = sb.board_width;
    jb["stride_px"] = stride_px;
    jb["knots"] = nlohmann::json::array();
    for (const auto& poly : sb.polygons) {
        nlohmann::json knot;
        knot["polygon"] = nlohmann::json::array();
        for (const auto& p : poly) {
            knot["polygon"].push_back({p.x, p.y});
        }
        jb["knots"].push_back(knot);
    }

    std::ofstream of(out_path);
    of << jb.dump(2);
    return sb.polygons.size();
}

}  // namespace knots
