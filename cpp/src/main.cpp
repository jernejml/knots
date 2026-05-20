// knots — C++ inference for wood-knot polygon extraction.
//
// Step 2: load a YOLOv11-seg ONNX model and run it on one frame. Decodes the
// post-NMS output (4 bbox + 1 conf + 1 cls + 32 mask coefs) and the mask
// prototypes (32×160×160) into per-detection polygons in source-image
// coordinates, then writes a JSON blob to stdout or to a file.
//
// Usage:
//   knots <model.onnx> <frame.png> [out.json] [--conf C] [--cpu]
//
// Cross-frame dedup and per-board aggregation come later — this step
// produces the per-frame predictions that those stages will consume.

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>

#include "knots/inference.hpp"

namespace fs = std::filesystem;

namespace {

struct Args {
    fs::path model;
    fs::path image;
    fs::path output;        // empty -> stdout
    float conf = 0.25f;
    bool prefer_cuda = true;
};

void PrintUsage(const char* prog) {
    std::cerr
        << "usage: " << prog << " <model.onnx> <frame.png> [out.json] [opts]\n"
        << "  --conf C     confidence threshold (default 0.25)\n"
        << "  --cpu        force CPU execution provider\n";
}

bool ParseArgs(int argc, char** argv, Args& out) {
    if (argc < 3) return false;
    out.model = argv[1];
    out.image = argv[2];

    int i = 3;
    while (i < argc) {
        std::string a = argv[i];
        if (a == "--conf" && i + 1 < argc) {
            out.conf = std::stof(argv[++i]);
        } else if (a == "--cpu") {
            out.prefer_cuda = false;
        } else if (!a.empty() && a[0] != '-' && out.output.empty()) {
            out.output = a;
        } else {
            std::cerr << "unrecognised arg: " << a << "\n";
            return false;
        }
        ++i;
    }
    return true;
}

}  // namespace


int main(int argc, char** argv) {
    Args args;
    if (!ParseArgs(argc, argv, args)) {
        PrintUsage(argv[0]);
        return 2;
    }
    if (!fs::exists(args.model)) {
        std::cerr << "model not found: " << args.model << "\n";
        return 1;
    }
    if (!fs::exists(args.image)) {
        std::cerr << "image not found: " << args.image << "\n";
        return 1;
    }

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "knots");
        std::string active_provider;
        Ort::Session session = knots::MakeSession(
            env, args.model, args.prefer_cuda, active_provider);

        cv::Mat image = cv::imread(args.image.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "could not decode image: " << args.image << "\n";
            return 1;
        }

        auto detections = knots::InferFrame(session, image, args.conf);

        nlohmann::json j;
        j["frame"] = args.image.stem().string();
        j["image_size"] = {image.cols, image.rows};
        j["session_ep"] = active_provider;
        j["conf_threshold"] = args.conf;
        j["detections"] = nlohmann::json::array();
        for (const auto& d : detections) {
            nlohmann::json jd;
            jd["bbox"] = {d.bbox.x, d.bbox.y,
                          d.bbox.x + d.bbox.width,
                          d.bbox.y + d.bbox.height};
            jd["confidence"] = d.confidence;
            jd["class"] = d.cls;
            nlohmann::json poly = nlohmann::json::array();
            for (const auto& p : d.polygon) {
                poly.push_back({p.x, p.y});
            }
            jd["polygon"] = poly;
            j["detections"].push_back(jd);
        }

        if (args.output.empty()) {
            std::cout << j.dump(2) << "\n";
        } else {
            std::ofstream f(args.output);
            f << j.dump(2);
        }
        std::cerr << detections.size() << " detection(s)"
                  << " (EP=" << active_provider << ")\n";
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
