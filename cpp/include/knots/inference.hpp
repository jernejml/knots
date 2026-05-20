#pragma once

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace knots {

struct LetterboxParams {
    float scale;   // resized_dim = original_dim * scale (uniform on both axes)
    int pad_x;     // px added on the left of the resized image
    int pad_y;     // px added on the top of the resized image
    int net_size;  // square model input size (e.g. 640)
};

struct Detection {
    cv::Rect2f bbox;  // original-image coords
    float confidence;
    int cls;
    std::vector<cv::Point2i> polygon;  // original-image coords (px)
};

// Build an ONNX Runtime session. If `prefer_cuda` is set, tries the CUDA EP
// first and falls back to CPU if it fails (e.g., no GPU, no --gpus all).
// Reports the active EP via `active_provider_out`.
Ort::Session MakeSession(const Ort::Env& env, const std::filesystem::path& model_path,
                         bool prefer_cuda, std::string& active_provider_out);

// Letterbox `src` (BGR) into `dst` of size `target_size`×`target_size`,
// centered, padded with `pad_value`. Returns the params needed to project
// any point in `dst` back to `src` coordinates.
LetterboxParams Letterbox(const cv::Mat& src, cv::Mat& dst, int target_size, int pad_value = 114);

// Run YOLO11-seg inference on one BGR frame. Decodes the post-NMS ultralytics
// export (output0: [1, N, 38] = bbox(4)+conf(1)+cls(1)+mask_coefs(32);
// output1: [1, 32, 160, 160] mask prototypes) into a list of detections with
// polygons in src image coordinates.
//
// `conf_threshold`     filters detections below this confidence.
// `mask_threshold`     binarisation threshold on sigmoid(mask).
// `simplify_eps_px`    cv::approxPolyDP epsilon in letterboxed pixels.
std::vector<Detection> InferFrame(Ort::Session& session, const cv::Mat& bgr_image,
                                  float conf_threshold = 0.25f, float mask_threshold = 0.5f,
                                  float simplify_eps_px = 1.0f);

}  // namespace knots
