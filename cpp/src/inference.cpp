#include "knots/inference.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace knots {

Ort::Session MakeSession(const Ort::Env& env,
                         const std::filesystem::path& model_path,
                         bool prefer_cuda,
                         std::string& active_provider_out) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    active_provider_out = "CPU";
    if (prefer_cuda) {
        try {
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = 0;
            opts.AppendExecutionProvider_CUDA(cuda_opts);
            active_provider_out = "CUDA:0";
        } catch (const Ort::Exception& e) {
            std::cerr << "(CUDA EP unavailable: " << e.what() << " — using CPU)\n";
        }
    }
    return Ort::Session(env, model_path.c_str(), opts);
}


LetterboxParams Letterbox(const cv::Mat& src, cv::Mat& dst,
                          int target_size, int pad_value) {
    const int h0 = src.rows;
    const int w0 = src.cols;
    const float scale = std::min(static_cast<float>(target_size) / h0,
                                 static_cast<float>(target_size) / w0);
    const int nh = static_cast<int>(std::round(h0 * scale));
    const int nw = static_cast<int>(std::round(w0 * scale));
    const int pad_x = (target_size - nw) / 2;
    const int pad_y = (target_size - nh) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

    dst = cv::Mat(target_size, target_size, src.type(),
                  cv::Scalar(pad_value, pad_value, pad_value));
    resized.copyTo(dst(cv::Rect(pad_x, pad_y, nw, nh)));

    return LetterboxParams{scale, pad_x, pad_y, target_size};
}


namespace {

// Map a 2D point from letterboxed (net) space back to source-image space.
cv::Point2f InverseLetterboxPoint(const cv::Point2f& pt,
                                  const LetterboxParams& lb) {
    return cv::Point2f((pt.x - lb.pad_x) / lb.scale,
                       (pt.y - lb.pad_y) / lb.scale);
}

// Same, but for a bbox.
cv::Rect2f InverseLetterboxBox(float x1, float y1, float x2, float y2,
                               const LetterboxParams& lb) {
    cv::Point2f tl = InverseLetterboxPoint({x1, y1}, lb);
    cv::Point2f br = InverseLetterboxPoint({x2, y2}, lb);
    return cv::Rect2f(tl, br);
}

}  // namespace


std::vector<Detection> InferFrame(Ort::Session& session,
                                  const cv::Mat& bgr_image,
                                  float conf_threshold,
                                  float mask_threshold,
                                  float simplify_eps_px) {
    constexpr int kNetSize = 640;
    constexpr int kProtoSize = 160;
    constexpr int kNumMaskCoefs = 32;
    constexpr int kDetCols = 38;  // 4 bbox + 1 conf + 1 cls + 32 mask coefs

    cv::Mat letterboxed;
    LetterboxParams lb = Letterbox(bgr_image, letterboxed, kNetSize);

    // Build input tensor: NCHW float32, BGR→RGB, normalized to [0,1].
    cv::Mat blob;
    cv::dnn::blobFromImage(letterboxed, blob, 1.0 / 255.0,
                           cv::Size(), cv::Scalar(),
                           /*swapRB=*/true, /*crop=*/false, CV_32F);

    std::vector<int64_t> input_shape = {1, 3, kNetSize, kNetSize};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        reinterpret_cast<float*>(blob.data),
        blob.total(),
        input_shape.data(),
        input_shape.size());

    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name_ptr = session.GetInputNameAllocated(0, alloc);
    auto out0_name_ptr = session.GetOutputNameAllocated(0, alloc);
    auto out1_name_ptr = session.GetOutputNameAllocated(1, alloc);
    const char* in_names[] = {in_name_ptr.get()};
    const char* out_names[] = {out0_name_ptr.get(), out1_name_ptr.get()};

    auto outputs = session.Run(Ort::RunOptions{nullptr},
                               in_names, &input_tensor, 1,
                               out_names, 2);

    const Ort::Value& out0 = outputs[0];
    const Ort::Value& out1 = outputs[1];
    const auto out0_shape = out0.GetTensorTypeAndShapeInfo().GetShape();
    const auto out1_shape = out1.GetTensorTypeAndShapeInfo().GetShape();
    if (out0_shape.size() != 3 || out0_shape[2] != kDetCols) {
        throw std::runtime_error(
            "unexpected output0 shape — expected [1, N, 38] post-NMS YOLO11-seg");
    }
    if (out1_shape.size() != 4
        || out1_shape[1] != kNumMaskCoefs
        || out1_shape[2] != kProtoSize
        || out1_shape[3] != kProtoSize) {
        throw std::runtime_error(
            "unexpected output1 shape — expected [1, 32, 160, 160] mask prototypes");
    }

    const float* det_data = out0.GetTensorData<float>();
    const float* proto_data = out1.GetTensorData<float>();
    const int n_dets = static_cast<int>(out0_shape[1]);

    // Mask prototypes reshaped to (32, 25600) for matmul against mask coefs.
    // We only read this Mat — const_cast is to satisfy cv::Mat's non-const ptr,
    // not to mutate. The underlying buffer is owned by `out1`.
    cv::Mat protos(kNumMaskCoefs, kProtoSize * kProtoSize, CV_32F,
                   const_cast<float*>(proto_data));

    std::vector<Detection> detections;
    detections.reserve(n_dets);

    const int src_w = bgr_image.cols;
    const int src_h = bgr_image.rows;

    for (int i = 0; i < n_dets; ++i) {
        const float* row = det_data + static_cast<ptrdiff_t>(i) * kDetCols;
        const float conf = row[4];
        if (conf < conf_threshold) continue;

        const float x1 = row[0], y1 = row[1], x2 = row[2], y2 = row[3];
        const int cls = static_cast<int>(row[5]);
        const float* mc = row + 6;

        // mc (1×32) · protos (32×25600) = mask_flat (1×25600), reshape to 160×160.
        cv::Mat mc_mat(1, kNumMaskCoefs, CV_32F, const_cast<float*>(mc));
        cv::Mat mask_flat = mc_mat * protos;
        cv::Mat mask = mask_flat.reshape(0, kProtoSize);

        // Sigmoid + binarise.
        cv::Mat exp_neg;
        cv::exp(-mask, exp_neg);
        cv::Mat sig = 1.0 / (1.0 + exp_neg);
        cv::Mat mask_bin;
        cv::compare(sig, mask_threshold, mask_bin, cv::CMP_GT);  // CV_8U 0/255

        // Upsample to net size.
        cv::Mat mask_net;
        cv::resize(mask_bin, mask_net, cv::Size(kNetSize, kNetSize), 0, 0,
                   cv::INTER_LINEAR);

        // Crop to bbox: suppress mask outside the predicted bbox. SAM-style
        // safety, also drops spurious mask blobs in distant frame regions.
        const int bx1 = std::clamp(static_cast<int>(std::floor(x1)), 0, kNetSize);
        const int by1 = std::clamp(static_cast<int>(std::floor(y1)), 0, kNetSize);
        const int bx2 = std::clamp(static_cast<int>(std::ceil(x2)), 0, kNetSize);
        const int by2 = std::clamp(static_cast<int>(std::ceil(y2)), 0, kNetSize);
        const int bw = bx2 - bx1, bh = by2 - by1;
        if (bw <= 0 || bh <= 0) continue;

        cv::Mat cropped = cv::Mat::zeros(kNetSize, kNetSize, CV_8U);
        const cv::Rect roi(bx1, by1, bw, bh);
        mask_net(roi).copyTo(cropped(roi));

        // Threshold to a clean {0,255} after the linear resize.
        cv::Mat clean;
        cv::compare(cropped, 127, clean, cv::CMP_GT);

        // Extract largest contour.
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(clean, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        if (contours.empty()) continue;
        auto largest_it = std::max_element(
            contours.begin(), contours.end(),
            [](const auto& a, const auto& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });
        if (cv::contourArea(*largest_it) < 4.0) continue;

        // Simplify.
        std::vector<cv::Point> simplified;
        cv::approxPolyDP(*largest_it, simplified, simplify_eps_px, /*closed=*/true);
        if (simplified.size() < 3) continue;

        // Inverse-letterbox polygon vertices into source-image coords.
        std::vector<cv::Point2i> poly_src;
        poly_src.reserve(simplified.size());
        for (const auto& p : simplified) {
            const cv::Point2f q = InverseLetterboxPoint(cv::Point2f(p.x, p.y), lb);
            poly_src.emplace_back(
                std::clamp(static_cast<int>(std::round(q.x)), 0, src_w - 1),
                std::clamp(static_cast<int>(std::round(q.y)), 0, src_h - 1));
        }

        Detection d;
        d.bbox = InverseLetterboxBox(x1, y1, x2, y2, lb);
        d.confidence = conf;
        d.cls = cls;
        d.polygon = std::move(poly_src);
        detections.push_back(std::move(d));
    }

    return detections;
}

}  // namespace knots
