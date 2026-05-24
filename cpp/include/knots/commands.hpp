#pragma once

// Subcommand entry points and the Args structs they consume. main.cpp builds
// one CLI11 app, wires per-subcommand options (via cli_options.hpp) into a
// populated struct, and dispatches to CmdXxx(args) from the parser callback.
//
// The structs are intentionally per-subcommand even though they overlap.
// Small shared option groups (InferenceOpts / StitchOpts / BoardsFilter /
// SplitsFilter / FramesFilter) are embedded as fields so the AddXxxOpts
// helpers in cli_options.hpp can register the same flag set in multiple
// subcommands without sharing top-level Args type.

#include <filesystem>
#include <string>
#include <vector>

namespace knots {

// Default frame stride: 640 px frame width on a 320 px stride means each
// physical knot is seen by two adjacent frames. Same constant is used by
// all stitching paths.
inline constexpr int kStridePxDefault = 320;

// -- Shared option groups ----------------------------------------------------

// --model / --conf / --cpu. The `cpu_only` flag is the storage for `--cpu`;
// call sites pass `!cpu_only` to MakeSession as `prefer_cuda`.
struct InferenceOpts {
    std::filesystem::path model;
    float conf = 0.25f;
    bool cpu_only = false;
};

// --stride-px / --simplify-eps. Used by every command that calls into
// StitchBoardPolygons.
struct StitchOpts {
    int stride_px = kStridePxDefault;
    float simplify_eps_px = 1.0f;
};

// --boards LIST / --boards-file PATH. The list is parsed to ints by CLI11's
// `->delimiter(',')` transform; --boards-file wins if both are set.
struct BoardsFilter {
    std::vector<int> boards;
    std::filesystem::path boards_file;
};

// --partitions-json PATH / --split {train,val,test}. Validation that
// --partitions-json requires --split happens in the subcommand callback.
struct SplitsFilter {
    std::filesystem::path partitions_json;
    std::string split;
};

// --frames LIST / --frames-file PATH. CLI11 splits --frames into the vector
// via `->delimiter(',')`; --frames-file is appended at command time.
// Both empty means "no explicit frame selection, walk --input-dir".
struct FramesFilter {
    std::vector<std::string> frames;
    std::filesystem::path frames_file;
};

// -- Per-subcommand Args -----------------------------------------------------

struct RunArgs {
    InferenceOpts inference;
    StitchOpts stitch;
    FramesFilter frames;
    SplitsFilter splits;
    std::filesystem::path input_dir;
    std::filesystem::path output_dir;
    // --dump-per-frame DIR: also write one {stem}.json per frame to DIR
    // (the cached inference output, useful for re-stitching with different
    // params). Empty = skip.
    std::filesystem::path dump_per_frame_dir;
    // --from-frame-jsons DIR: skip inference; read per-frame JSONs from DIR
    // instead. Mutually exclusive with --dump-per-frame and makes --model /
    // --input-dir optional. Empty = run inference as usual.
    std::filesystem::path from_frame_jsons_dir;
    bool force = false;
};

struct EvalArgs {
    std::filesystem::path pred_dir;  // required: per-board prediction JSONs
    std::filesystem::path gt_dir;    // required: per-board GT JSONs
    BoardsFilter boards;
    std::filesystem::path out_json;
    float match_iou = 0.5f;
    bool no_write = false;

    // Optional GT rebuild: when --labels-dir / --images-dir are provided,
    // eval first stitches per-board GT polygons from the raw YOLO bbox
    // labels into --gt-dir (skip-if-exists), then runs the comparison.
    // Replaces the deprecated `knots gt-stitch` subcommand.
    std::filesystem::path labels_dir;
    std::filesystem::path images_dir;
    StitchOpts stitch;
};

// -- Entry points ------------------------------------------------------------

// Per-board pipeline. Default: walks --input-dir of frame PNGs, runs YOLO11-seg
// through ONNX Runtime per frame, raster-unions the per-frame polygons into
// one {board}.json. Two debug flags:
//   --dump-per-frame DIR     also write per-frame inference JSONs
//   --from-frame-jsons DIR   skip inference; read per-frame JSONs from DIR
int CmdRun(const RunArgs& args);

// Test mode: greedy bbox-IoU matching + per-pair mask IoU, P/R/F1 plus
// extras (FP) and missing (FN). Prints a per-board table and writes an
// aggregate JSON. Consumes two dirs of stitched per-board JSONs. If
// --labels-dir / --images-dir are passed, eval rebuilds missing GT under
// --gt-dir first (formerly the separate `knots gt-stitch` step).
int CmdEval(const EvalArgs& args);

}  // namespace knots
