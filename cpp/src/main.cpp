// knots — C++ inference pipeline for wood-knot polygon extraction.
//
// Step 1 (smoke test): load a YOLOv11-seg ONNX model and print its session
// metadata: available execution providers and the names/shapes/types of
// every input and output tensor. This validates that ONNX Runtime is wired
// up correctly and that the exported model matches the shapes we expect
// before we start writing the inference + postprocessing logic.
//
// Expected output for an ultralytics-exported YOLO11-seg model with
// `nms=True`, fixed imgsz=640:
//   inputs (1):   images   shape=[1, 3, 640, 640]   type=float32
//   outputs (2):  output0  shape=[1, N_keep, 38]    type=float32
//                 output1  shape=[1, 32, 160, 160]  type=float32

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include <onnxruntime_cxx_api.h>

namespace fs = std::filesystem;

namespace {

std::string ShapeToString(const std::vector<int64_t>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

// Map ONNX element-type enum to a short, human-readable string. Only the
// types we actually expect to encounter are listed; everything else falls
// through to "other(<int>)" so unexpected models are visibly different
// rather than silently OK.
std::string TypeName(ONNXTensorElementDataType t) {
    switch (t) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return "uint8";
        default:
            return "other(" + std::to_string(static_cast<int>(t)) + ")";
    }
}

// Note: takes Ort::TypeInfo by const-ref. Pulling the shape-info as a free
// expression (`session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo()`)
// is a use-after-free — the shape-info is a non-owning view into the
// TypeInfo, which is destroyed at the end of the temporary's lifetime.
void DumpTensorInfo(const Ort::TypeInfo& type_info, const std::string& name) {
    const auto info = type_info.GetTensorTypeAndShapeInfo();
    std::cout << "  " << name
              << "  shape=" << ShapeToString(info.GetShape())
              << "  type=" << TypeName(info.GetElementType()) << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <model.onnx>\n";
        return 2;
    }
    const fs::path model_path = argv[1];
    if (!fs::exists(model_path)) {
        std::cerr << "model not found: " << model_path << "\n";
        return 1;
    }

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "knots-smoke");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        Ort::Session session(env, model_path.c_str(), opts);

        Ort::AllocatorWithDefaultOptions alloc;

        std::cout << "model: " << model_path.string() << "\n";
        std::cout << "providers:";
        for (const auto& p : Ort::GetAvailableProviders()) {
            std::cout << " " << p;
        }
        std::cout << "\n\n";

        const size_t n_inputs = session.GetInputCount();
        std::cout << "inputs (" << n_inputs << "):\n";
        for (size_t i = 0; i < n_inputs; ++i) {
            const auto name = session.GetInputNameAllocated(i, alloc);
            const auto type_info = session.GetInputTypeInfo(i);
            DumpTensorInfo(type_info, name.get());
        }

        const size_t n_outputs = session.GetOutputCount();
        std::cout << "outputs (" << n_outputs << "):\n";
        for (size_t i = 0; i < n_outputs; ++i) {
            const auto name = session.GetOutputNameAllocated(i, alloc);
            const auto type_info = session.GetOutputTypeInfo(i);
            DumpTensorInfo(type_info, name.get());
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
