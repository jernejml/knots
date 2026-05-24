#pragma once

// Shared per-board inference / GT-load loops used by `knots run` and
// `knots eval` (the latter for its inline GT rebuild). Each function
// operates on one board's frames and returns FramePolys ready for
// StitchBoardPolygons.

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "knots/stitching.hpp"

namespace knots::pipeline {

// (frame_idx, stem) tuples for one board, sorted by frame_idx.
using BoardFrames = std::vector<std::pair<int, std::string>>;
using FramesByBoard = std::map<int, BoardFrames>;

// Walk `dir` for files matching {board}_{frame}<extension>, group by board.
// If `explicit_stems` is non-empty, use that list as the candidate set
// instead of scanning `dir` (stems must still parse as {board}_{frame}).
// `boards_filter` is applied after stem parsing — empty means no filter.
//
// Returns FramesByBoard sorted by frame_idx within each board.
FramesByBoard CollectFramesByBoard(const std::filesystem::path& dir,
                                   const std::string& extension,
                                   const std::vector<std::string>& explicit_stems,
                                   const std::unordered_set<int>& boards_filter);

// Counters surfaced by InferBoardFrames so callers can print a summary.
// `total_inference_sec` is the sum of wall-clock InferFrame calls (CPU/GPU
// session.Run + pre/post processing), excluding image I/O and JSON write.
struct InferStats {
    size_t processed = 0;
    size_t unread = 0;
    size_t failed = 0;
    double total_inference_sec = 0.0;
};

// Called once per frame after it has been processed, skipped, or failed.
// Caller maintains any global progress state.
using FrameDoneFn = std::function<void()>;

// For each frame in `frames`, imread {images_dir}/{stem}.png and run
// InferFrame. Returns one FramePolys per readable frame, containing only
// detections with polygon.size() >= 3.
//
// Unread or inference-failed frames bump the corresponding counter in
// `stats` and are skipped. `frame_done` (if non-null) is called once per
// frame regardless of outcome.
//
// If `dump_per_frame_dir` is non-empty, a per-frame JSON is also written
// to `<dump_per_frame_dir>/<stem>.json`. `session_ep` and `conf_threshold`
// are recorded in the JSON for traceability.
std::vector<FramePolys> InferBoardFrames(Ort::Session& session,
                                         const std::filesystem::path& images_dir,
                                         const BoardFrames& frames, float conf_threshold,
                                         InferStats& stats, FrameDoneFn frame_done = {},
                                         const std::filesystem::path& dump_per_frame_dir = {},
                                         const std::string& session_ep = {});

// For each frame in `frames`, read {jsons_dir}/{stem}.json (written earlier
// by InferBoardFrames with dump_per_frame_dir set) and rehydrate FramePolys
// from the `image_size` + per-detection `polygon` fields. Frames whose JSON
// is missing or unparseable are warned and skipped.
std::vector<FramePolys> LoadBoardFromFrameJsons(const std::filesystem::path& jsons_dir,
                                                const BoardFrames& frames,
                                                FrameDoneFn frame_done = {});

// For each frame in `frames`, imread (for dimensions) then parse YOLO
// bboxes from {labels_dir}/{stem}.txt, projecting each as a 4-vertex
// rectangle polygon in frame-local pixel coords. Frames with unreadable
// images are warned and skipped.
std::vector<FramePolys> LoadGtBoardFrames(const std::filesystem::path& labels_dir,
                                          const std::filesystem::path& images_dir,
                                          const BoardFrames& frames);

// One-shot per-board GT stitching. Walks labels_dir for {board}_{frame}.txt
// files, groups by board, runs LoadGtBoardFrames + StitchBoardToJson for
// each board into gt_dir/{board}.json. boards_filter restricts the boards
// processed (empty = all). Existing gt_dir/{board}.json files are skipped
// unless `force` is true.
//
// Called by `knots eval` when --labels-dir / --images-dir are passed, so
// eval can rebuild missing GT on the fly without a separate gt-stitch step.
struct GtStitchStats {
    size_t written = 0;
    size_t skipped = 0;
    size_t total_polys = 0;
};

GtStitchStats StitchGtForBoards(const std::filesystem::path& labels_dir,
                                const std::filesystem::path& images_dir,
                                const std::filesystem::path& gt_dir,
                                const std::unordered_set<int>& boards_filter,
                                int stride_px,
                                float simplify_eps_px,
                                bool force);

}  // namespace knots::pipeline
