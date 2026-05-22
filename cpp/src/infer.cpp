// `knots infer` — per-frame YOLOv11-seg inference.
//
// Walks args.input_dir of frame PNGs, runs YOLO11-seg through ONNX Runtime
// on each, writes one {frame_id}.json per frame to args.output_dir. The
// ONNX session is created once and reused across all frames (single CUDA
// warmup; no per-frame ORT init cost).
//
// Frame-selection knobs (FramesFilter + SplitsFilter) and inference knobs
// are populated by cli_options::AddInferOptions before this function fires.
//
// Runs are resumable: frames whose JSON already exists are skipped unless
// args.force is true.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <unordered_set>
#include <vector>

#include "knots/cli_util.hpp"
#include "knots/commands.hpp"
#include "knots/inference.hpp"
#include "knots/pipeline.hpp"

namespace fs = std::filesystem;

namespace knots {

namespace {

void WriteJson(const fs::path& out_path, const std::string& frame_id, const cv::Mat& image,
               const std::string& ep, float conf, const std::vector<Detection>& dets) {
    nlohmann::json j;
    j["frame"] = frame_id;
    j["image_size"] = {image.cols, image.rows};
    j["session_ep"] = ep;
    j["conf_threshold"] = conf;
    j["detections"] = nlohmann::json::array();
    for (const auto& d : dets) {
        nlohmann::json jd;
        jd["bbox"] = {d.bbox.x, d.bbox.y, d.bbox.x + d.bbox.width, d.bbox.y + d.bbox.height};
        jd["confidence"] = d.confidence;
        jd["class"] = d.cls;
        nlohmann::json poly = nlohmann::json::array();
        for (const auto& p : d.polygon) {
            poly.push_back({p.x, p.y});
        }
        jd["polygon"] = poly;
        j["detections"].push_back(jd);
    }
    std::ofstream f(out_path);
    f << j.dump(2);
}

}  // namespace

int CmdInfer(const InferArgs& args) {
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
        if (!args.splits.splits_csv.empty()) {
            boards_filter = cli::LoadBoardsInSplit(args.splits.splits_csv, args.splits.split);
            if (boards_filter.empty()) {
                std::cerr << "warning: no boards match split " << args.splits.split << " in "
                          << args.splits.splits_csv << "\n";
            }
        }
        const auto by_board =
            pipeline::CollectFramesByBoard(args.input_dir, ".png", explicit_stems, boards_filter);

        // Flatten to a deterministic stems list (board-major, frame-minor).
        std::vector<std::string> stems;
        for (const auto& [_, frames] : by_board) {
            for (const auto& [__, stem] : frames) stems.push_back(stem);
        }
        if (stems.empty()) {
            std::cerr << "no frames to process\n";
            return 0;
        }

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "knots");
        std::string ep;
        Ort::Session session =
            MakeSession(env, args.inference.model, !args.inference.cpu_only, ep);

        std::cerr << "knots infer: " << stems.size() << " frame(s)  ep=" << ep
                  << "  conf=" << args.inference.conf
                  << "  force=" << (args.force ? "true" : "false") << "\n";

        const size_t heartbeat = std::max<size_t>(20, stems.size() / 20);
        size_t processed = 0, skipped = 0, errors = 0, total_dets = 0;

        for (size_t i = 0; i < stems.size(); ++i) {
            const std::string& stem = stems[i];
            fs::path img_path = args.input_dir / (stem + ".png");
            fs::path out_path = args.output_dir / (stem + ".json");

            if (fs::exists(out_path) && !args.force) {
                ++skipped;
                continue;
            }
            cv::Mat image = cv::imread(img_path.string(), cv::IMREAD_COLOR);
            if (image.empty()) {
                std::cerr << "  WARN unreadable: " << img_path << "\n";
                ++errors;
                continue;
            }
            try {
                auto dets = InferFrame(session, image, args.inference.conf);
                WriteJson(out_path, stem, image, ep, args.inference.conf, dets);
                total_dets += dets.size();
                ++processed;
            } catch (const std::exception& e) {
                std::cerr << "  WARN inference failed for " << stem << ": " << e.what() << "\n";
                ++errors;
            }
            if ((i + 1) % heartbeat == 0 || (i + 1) == stems.size()) {
                std::cerr << "  ... " << (i + 1) << "/" << stems.size() << " frames\n";
            }
        }

        std::cerr << "\nResult\n"
                  << "  processed=" << processed << "  skipped=" << skipped << "  errors=" << errors
                  << "\n"
                  << "  detections=" << total_dets << " across " << processed << " frame(s)\n"
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
