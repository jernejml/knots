// `knots infer` — per-frame YOLOv11-seg inference.
//
// Bulk mode: take an input directory of frame PNGs, run YOLO11-seg through
// ONNX Runtime on each, write one {frame_id}.json per frame to the output
// directory. The ONNX session is created once and reused across all frames
// (single CUDA warmup; no per-frame ORT init cost).
//
// Filtering frames to process:
//   - default: every {board}_{frame}.png in --input-dir
//   - --frames LIST           comma-separated stems (e.g. "0_5,100_3")
//   - --frames-file FILE      one stem per line ('#' comments allowed)
//   - --splits-csv + --split  pick a split (train|val|test) from
//                             analysis/splits.csv
//
// Runs are resumable: frames whose JSON already exists are skipped unless
// --force is passed.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <regex>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "knots/cli_util.hpp"
#include "knots/commands.hpp"
#include "knots/inference.hpp"

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
    bool prefer_cuda = true;
    bool force = false;
};

void PrintUsage() {
    std::cerr << "usage: knots infer --model M --input-dir IN --output-dir OUT [opts]\n"
                 "  --frames LIST            comma-separated frame stems, e.g. 0_5,100_3\n"
                 "  --frames-file FILE       one frame stem per line\n"
                 "  --splits-csv PATH        analysis/splits.csv\n"
                 "  --split {train,val,test} with --splits-csv: restrict to this split\n"
                 "  --conf C                 confidence threshold (default 0.25)\n"
                 "  --cpu                    force CPU execution provider\n"
                 "  --force                  overwrite existing outputs\n";
}

bool ParseArgs(int argc, char** argv, Args& out) {
    int i = 1;  // skip subcommand name
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

const std::regex kFrameFileRe(R"(^(\d+)_(\d+)\.png$)");

std::vector<std::string> CollectFrameStems(const Args& args) {
    std::vector<std::string> explicit_stems;
    if (!args.frames_csv.empty()) {
        explicit_stems = cli::ParseFramesList(args.frames_csv);
    }
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
    std::set<std::string> dedup;
    if (!explicit_stems.empty()) {
        for (const auto& s : explicit_stems) {
            int b = -1, fi = -1;
            if (!cli::ParseFrameStem(s, b, fi)) {
                std::cerr << "warning: unrecognised frame stem: " << s << "\n";
                continue;
            }
            if (!boards_filter.empty() && !boards_filter.count(b)) continue;
            dedup.insert(s);
        }
    } else {
        for (const auto& entry : fs::directory_iterator(args.input_dir)) {
            if (!entry.is_regular_file()) continue;
            std::string fname = entry.path().filename().string();
            std::smatch m;
            if (!std::regex_match(fname, m, kFrameFileRe)) continue;
            std::string stem = entry.path().stem().string();
            int b = -1, fi = -1;
            if (!cli::ParseFrameStem(stem, b, fi)) continue;
            if (!boards_filter.empty() && !boards_filter.count(b)) continue;
            dedup.insert(stem);
        }
    }
    return {dedup.begin(), dedup.end()};
}

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

int CmdInfer(int argc, char** argv) {
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
        const auto stems = CollectFrameStems(args);
        if (stems.empty()) {
            std::cerr << "no frames to process\n";
            return 0;
        }

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "knots");
        std::string ep;
        Ort::Session session = MakeSession(env, args.model, args.prefer_cuda, ep);

        std::cerr << "knots infer: " << stems.size() << " frame(s)  ep=" << ep
                  << "  conf=" << args.conf << "  force=" << (args.force ? "true" : "false")
                  << "\n";

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
                auto dets = InferFrame(session, image, args.conf);
                WriteJson(out_path, stem, image, ep, args.conf, dets);
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
