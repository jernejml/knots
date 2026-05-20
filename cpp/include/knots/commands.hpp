#pragma once

// Subcommand entry points for the `knots` binary. The top-level `main()`
// dispatches on argv[1] to one of these.

namespace knots {

// Per-frame inference: walks --input-dir of PNGs, runs YOLO11-seg through
// ONNX Runtime, writes one {frame}.json per frame to --output-dir.
int CmdInfer(int argc, char** argv);

// Per-board stitching: walks --input-dir of per-frame JSONs (output by
// CmdInfer), translates each polygon by `frame_idx * stride_px`, rasterises
// onto a board-sized mask, extracts merged contours, writes one
// {board}.json per board to --output-dir.
int CmdStitch(int argc, char** argv);

// Per-board ground-truth stitching: reads per-frame YOLO bboxes from
// --labels-dir, projects each bbox as a 4-vertex rectangle, and runs the
// same raster-union pipeline as CmdStitch. Output is bit-identical-format
// to CmdStitch's so the eval script can compare apples to apples.
int CmdGtStitch(int argc, char** argv);

}  // namespace knots
