#pragma once

#include <filesystem>
#include <opencv2/core.hpp>
#include <vector>

namespace knots {

// Per-frame contribution to a board's polygon set. Polygons are in
// frame-local pixel coords; the stitcher translates them by
// (frame_idx * stride_px, 0) into board coords.
struct FramePolys {
    int frame_idx;
    int width;   // frame pixel width
    int height;  // frame pixel height
    std::vector<std::vector<cv::Point>> polygons;
};

// Output of the raster-union pipeline: per-board polygons in board-coord px
// plus the canvas dimensions used.
struct StitchedBoard {
    int board_height;
    int board_width;
    std::vector<std::vector<cv::Point>> polygons;
};

// Raster-union per-frame polygons into a per-board polygon set, returned
// in memory (no I/O).
//
// Algorithm (per CLAUDE.md): allocate a board-sized mask, fillPoly each
// polygon translated by (frame_idx * stride_px, 0), then findContours +
// approxPolyDP. RETR_EXTERNAL gives one contour per merged region —
// overlapping or touching polygons fuse automatically.
//
// `frames` is taken by value; it is sorted by frame_idx in place. Returns
// an empty StitchedBoard (height=0, width=0, polygons={}) if `frames` is
// empty.
StitchedBoard StitchBoardPolygons(int board, std::vector<FramePolys> frames, int stride_px,
                                  float simplify_eps_px);

// Convenience wrapper: stitches + writes the JSON described in
// knots/cmd_stitch.cpp (board, board_height, board_width, stride_px, knots[]).
// Returns the count of knots written.
size_t StitchBoardToJson(int board, std::vector<FramePolys> frames, int stride_px,
                         float simplify_eps_px, const std::filesystem::path& out_path);

}  // namespace knots
