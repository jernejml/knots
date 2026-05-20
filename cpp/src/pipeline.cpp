#include "knots/pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <regex>
#include <sstream>

#include "knots/cli_util.hpp"
#include "knots/inference.hpp"

namespace fs = std::filesystem;

namespace knots::pipeline {

namespace {

// Parse YOLO bboxes from a label file; emit each as a 4-vertex rectangle
// polygon in frame-local pixel coords, clipped to image bounds.
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

FramesByBoard CollectFramesByBoard(const fs::path& dir, const std::string& extension,
                                   const std::vector<std::string>& explicit_stems,
                                   const std::unordered_set<int>& boards_filter) {
    FramesByBoard by_board;

    auto admit = [&](const std::string& stem) {
        int b = -1, fi = -1;
        if (!cli::ParseFrameStem(stem, b, fi)) return;
        if (!boards_filter.empty() && !boards_filter.count(b)) return;
        by_board[b].emplace_back(fi, stem);
    };

    if (!explicit_stems.empty()) {
        for (const auto& s : explicit_stems) admit(s);
    } else if (fs::is_directory(dir)) {
        const std::string suffix = extension;  // includes the dot, e.g. ".png"
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() != suffix) continue;
            admit(entry.path().stem().string());
        }
    }

    for (auto& [_, frames] : by_board) {
        std::sort(frames.begin(), frames.end());
    }
    return by_board;
}

std::vector<FramePolys> InferBoardFrames(Ort::Session& session, const fs::path& images_dir,
                                         const BoardFrames& frames, float conf_threshold,
                                         InferStats& stats, FrameDoneFn frame_done) {
    std::vector<FramePolys> out;
    out.reserve(frames.size());
    for (const auto& [frame_idx, stem] : frames) {
        fs::path img_path = images_dir / (stem + ".png");
        cv::Mat image = cv::imread(img_path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "  WARN unreadable: " << img_path << "\n";
            ++stats.unread;
            if (frame_done) frame_done();
            continue;
        }
        try {
            auto dets = InferFrame(session, image, conf_threshold);
            FramePolys fp;
            fp.frame_idx = frame_idx;
            fp.width = image.cols;
            fp.height = image.rows;
            fp.polygons.reserve(dets.size());
            for (auto& d : dets) {
                if (d.polygon.size() >= 3) {
                    fp.polygons.push_back(std::move(d.polygon));
                }
            }
            out.push_back(std::move(fp));
            ++stats.processed;
        } catch (const std::exception& e) {
            std::cerr << "  WARN inference failed for " << stem << ": " << e.what() << "\n";
            ++stats.failed;
        }
        if (frame_done) frame_done();
    }
    return out;
}

std::vector<FramePolys> LoadGtBoardFrames(const fs::path& labels_dir, const fs::path& images_dir,
                                          const BoardFrames& frames) {
    std::vector<FramePolys> out;
    out.reserve(frames.size());
    for (const auto& [frame_idx, stem] : frames) {
        fs::path img_path = images_dir / (stem + ".png");
        cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "  WARN unreadable image " << img_path << " — skipping frame\n";
            continue;
        }
        FramePolys fp;
        fp.frame_idx = frame_idx;
        fp.width = img.cols;
        fp.height = img.rows;
        fp.polygons = ParseYoloBboxesAsPolys(labels_dir / (stem + ".txt"), img.cols, img.rows);
        out.push_back(std::move(fp));
    }
    return out;
}

}  // namespace knots::pipeline
