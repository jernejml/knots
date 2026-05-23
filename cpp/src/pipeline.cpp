#include "knots/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <regex>
#include <sstream>

#include "knots/cli_util.hpp"
#include "knots/inference.hpp"

namespace fs = std::filesystem;

namespace knots::pipeline {

namespace {

// Write a per-frame JSON next to the source frame. Schema matches what
// LoadBoardFromFrameJsons consumes: image_size + a list of detections each
// carrying bbox / confidence / class / polygon. The bbox/conf/class are
// informational (the stitching path uses only the polygon) but live here
// for debug consumers.
void WritePerFrameJson(const fs::path& out_path, const std::string& stem,
                       int img_w, int img_h, const std::string& session_ep,
                       float conf_threshold, const std::vector<Detection>& dets) {
    nlohmann::json j;
    j["frame"] = stem;
    j["image_size"] = {img_w, img_h};
    j["session_ep"] = session_ep;
    j["conf_threshold"] = conf_threshold;
    j["detections"] = nlohmann::json::array();
    for (const auto& d : dets) {
        nlohmann::json jd;
        jd["bbox"] = {d.bbox.x, d.bbox.y, d.bbox.x + d.bbox.width, d.bbox.y + d.bbox.height};
        jd["confidence"] = d.confidence;
        jd["class"] = d.cls;
        nlohmann::json poly = nlohmann::json::array();
        for (const auto& p : d.polygon) poly.push_back({p.x, p.y});
        jd["polygon"] = poly;
        j["detections"].push_back(std::move(jd));
    }
    std::ofstream f(out_path);
    f << j.dump(2);
}

// Parse YOLO bboxes from a label file; emit each as a 4-vertex rectangle
// polygon in frame-local pixel coords, clipped to image bounds.
//
// Defensive against accidentally being pointed at YOLO-seg labels
// (cls x1 y1 x2 y2 x3 y3 ...): a seg line has 7+ tokens, so reading the
// first 5 as a bbox silently misuses the first two polygon vertices as
// (cx, cy, w, h). We check for extra tokens and skip such lines with a
// per-file warning rather than producing nonsense rectangles.
std::vector<std::vector<cv::Point>> ParseYoloBboxesAsPolys(const fs::path& label_path, int w,
                                                          int h) {
    std::vector<std::vector<cv::Point>> polys;
    std::ifstream f(label_path);
    if (!f) return polys;
    std::string line;
    bool warned_seg = false;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        int cls;
        float cx, cy, bw, bh;
        if (!(iss >> cls >> cx >> cy >> bw >> bh)) continue;
        std::string extra;
        if (iss >> extra) {
            if (!warned_seg) {
                std::cerr << "  WARN " << label_path << ": extra tokens past the YOLO "
                          << "bbox 5-tuple (looks like seg-format) — skipping; "
                          << "pass YOLO bbox labels for GT\n";
                warned_seg = true;
            }
            continue;
        }
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
                                         InferStats& stats, FrameDoneFn frame_done,
                                         const fs::path& dump_per_frame_dir,
                                         const std::string& session_ep) {
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
            const auto t0 = std::chrono::steady_clock::now();
            auto dets = InferFrame(session, image, conf_threshold);
            const auto t1 = std::chrono::steady_clock::now();
            stats.total_inference_sec += std::chrono::duration<double>(t1 - t0).count();
            if (!dump_per_frame_dir.empty()) {
                WritePerFrameJson(dump_per_frame_dir / (stem + ".json"), stem,
                                  image.cols, image.rows, session_ep, conf_threshold, dets);
            }
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

std::vector<FramePolys> LoadBoardFromFrameJsons(const fs::path& jsons_dir,
                                                const BoardFrames& frames,
                                                FrameDoneFn frame_done) {
    std::vector<FramePolys> out;
    out.reserve(frames.size());
    for (const auto& [frame_idx, stem] : frames) {
        fs::path j_path = jsons_dir / (stem + ".json");
        nlohmann::json j;
        try {
            std::ifstream f(j_path);
            if (!f) {
                std::cerr << "  WARN missing frame JSON: " << j_path << "\n";
                if (frame_done) frame_done();
                continue;
            }
            f >> j;
        } catch (const std::exception& e) {
            std::cerr << "  WARN cannot parse " << j_path << ": " << e.what() << "\n";
            if (frame_done) frame_done();
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
        out.push_back(std::move(fp));
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
