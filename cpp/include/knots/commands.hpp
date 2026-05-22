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
// `->delimiter(',')` transform; --boards-file wins if both are set (per
// BuildBoardsFilter semantics).
struct BoardsFilter {
    std::vector<int> boards;
    std::filesystem::path boards_file;
};

// --splits-csv PATH / --split {train,val,test}. Validation that --splits-csv
// requires --split happens in the subcommand callback.
struct SplitsFilter {
    std::filesystem::path splits_csv;
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
    bool force = false;
};

struct InferArgs {
    InferenceOpts inference;
    FramesFilter frames;
    SplitsFilter splits;
    std::filesystem::path input_dir;
    std::filesystem::path output_dir;
    bool force = false;
};

struct StitchArgs {
    StitchOpts stitch;
    std::filesystem::path input_dir;
    std::filesystem::path output_dir;
    bool force = false;
};

struct GtStitchArgs {
    StitchOpts stitch;
    BoardsFilter boards;
    SplitsFilter splits;
    std::filesystem::path labels_dir;
    std::filesystem::path images_dir;
    std::filesystem::path output_dir;
    bool force = false;
};

struct EvalArgs {
    // Mode A: compare two dirs of stitched per-board JSONs.
    std::filesystem::path pred_dir;
    std::filesystem::path gt_dir;

    // Mode B: in-process infer + GT-stitch + compare.
    InferenceOpts inference;
    StitchOpts stitch;
    SplitsFilter splits;
    std::filesystem::path images_dir;
    std::filesystem::path labels_dir;

    // Shared.
    BoardsFilter boards;
    std::filesystem::path out_json;
    float match_iou = 0.5f;
    bool no_write = false;
};

// -- Entry points ------------------------------------------------------------

// One-shot pipeline: per-frame inference + per-board stitching in one pass.
// Groups frames by board, infers each frame, raster-unions the per-frame
// polygons into one {board}.json. No intermediate per-frame JSON.
int CmdRun(const RunArgs& args);

// Per-frame inference: walks input dir of PNGs, runs YOLO11-seg through
// ONNX Runtime, writes one {frame}.json per frame.
int CmdInfer(const InferArgs& args);

// Per-board stitching: reads per-frame JSONs (output by CmdInfer), translates
// each polygon by `frame_idx * stride_px`, rasterises onto a board-sized
// mask, extracts merged contours, writes one {board}.json per board.
int CmdStitch(const StitchArgs& args);

// Per-board ground-truth stitching: reads per-frame YOLO bboxes from
// --labels-dir, projects each bbox as a 4-vertex rectangle, runs the same
// raster-union pipeline as CmdStitch. Output format matches CmdStitch.
int CmdGtStitch(const GtStitchArgs& args);

// Test mode: greedy bbox-IoU matching + per-pair mask IoU, P/R/F1 plus
// extras (FP) and missing (FN). Prints a per-board table and writes an
// aggregate JSON. Mode A consumes stitched dirs; Mode B runs inference
// in-process. The mode is implicit from which flags are populated.
int CmdEval(const EvalArgs& args);

}  // namespace knots
