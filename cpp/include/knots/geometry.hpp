#pragma once

// Pure geometry / matching primitives shared by the evaluation pipeline. Kept
// in their own translation unit so the unit tests can link against them
// without pulling in ONNX Runtime / I/O dependencies.

#include <opencv2/core.hpp>

#include <optional>
#include <vector>

namespace knots {

using Polygon = std::vector<cv::Point>;

// Axis-aligned bounding box of `poly`. Convention: cv::Rect width/height use
// the *exclusive* form (max - min), matching how cv::Rect is normally used
// for ROIs. MaskIou below allocates its mask with an inclusive +1 because it
// rasterises onto an integer grid; the two callers don't interact, but the
// difference is intentional. Tests pin the current behaviour so any future
// "fix" is conscious.
cv::Rect PolygonBbox(const Polygon& poly);

// Standard bbox IoU. Disjoint → 0; identical → 1. Both sides use the same
// exclusive convention as PolygonBbox.
float BboxIou(const cv::Rect& a, const cv::Rect& b);

// Rasterised IoU between two polygons. Polygons with <3 vertices score 0.
// Internally allocates a CV_8U mask covering the union bbox; cost is O(area).
float MaskIou(const Polygon& a, const Polygon& b);

struct Match {
    int pred_idx;
    int gt_idx;
    float bbox_iou;
};

// Greedy bbox-IoU matcher. Sorts all (pred, gt) candidate pairs by descending
// IoU, claims each pair the first time both sides are unmatched, stops when
// no pair clears `threshold`. Returns at most min(preds, gts) matches.
std::vector<Match> GreedyMatch(const std::vector<Polygon>& preds,
                                const std::vector<Polygon>& gts, float threshold);

// Optional-returning division. nullopt iff den <= 0.
std::optional<float> SafeDiv(double num, double den);

// Harmonic mean of precision and recall. nullopt if either is nullopt or both
// are zero (otherwise the formula divides by zero).
std::optional<float> F1FromPR(std::optional<float> p, std::optional<float> r);

}  // namespace knots
