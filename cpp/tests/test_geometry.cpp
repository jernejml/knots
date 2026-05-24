// Unit tests for the pure geometry / matching primitives in
// knots/geometry.hpp. No ORT or I/O dependencies — keeps the test binary
// fast to link and free of CUDA/runtime surprises.

#include <gtest/gtest.h>

#include "knots/geometry.hpp"

using knots::BboxIou;
using knots::CountInstancesByIou;
using knots::F1FromPR;
using knots::GreedyMatch;
using knots::MaskIou;
using knots::Match;
using knots::Polygon;
using knots::PolygonBbox;
using knots::SafeDiv;

// -- PolygonBbox -------------------------------------------------------------

TEST(PolygonBbox, SquareReturnsExclusiveWidthHeight) {
    // Pins the current "exclusive" convention: a square from (0,0) to (10,10)
    // has width = 10 (max - min), not 11. Tests guarding the smell flagged in
    // the project review: MaskIou's bbox uses an inclusive +1 instead.
    Polygon p = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    auto r = PolygonBbox(p);
    EXPECT_EQ(r.x, 0);
    EXPECT_EQ(r.y, 0);
    EXPECT_EQ(r.width, 10);
    EXPECT_EQ(r.height, 10);
}

TEST(PolygonBbox, TriangleSpansMinMax) {
    Polygon p = {{2, 3}, {7, 5}, {4, 9}};
    auto r = PolygonBbox(p);
    EXPECT_EQ(r.x, 2);
    EXPECT_EQ(r.y, 3);
    EXPECT_EQ(r.width, 5);   // 7 - 2
    EXPECT_EQ(r.height, 6);  // 9 - 3
}

TEST(PolygonBbox, SinglePointDegenerate) {
    Polygon p = {{4, 4}};
    auto r = PolygonBbox(p);
    EXPECT_EQ(r.x, 4);
    EXPECT_EQ(r.y, 4);
    EXPECT_EQ(r.width, 0);
    EXPECT_EQ(r.height, 0);
}

// -- BboxIou -----------------------------------------------------------------

TEST(BboxIou, IdenticalIsOne) {
    cv::Rect a(0, 0, 10, 10);
    EXPECT_FLOAT_EQ(BboxIou(a, a), 1.0f);
}

TEST(BboxIou, DisjointIsZero) {
    cv::Rect a(0, 0, 5, 5);
    cv::Rect b(10, 10, 5, 5);
    EXPECT_FLOAT_EQ(BboxIou(a, b), 0.0f);
}

TEST(BboxIou, TouchingIsZero) {
    // Edge-touching is 0 intersection area under the exclusive convention.
    cv::Rect a(0, 0, 10, 10);
    cv::Rect b(10, 0, 10, 10);
    EXPECT_FLOAT_EQ(BboxIou(a, b), 0.0f);
}

TEST(BboxIou, EquallySizedHalfOverlap) {
    // 10x10 boxes shifted by 5 along x: overlap area = 5*10 = 50;
    // union area = 100 + 100 - 50 = 150; IoU = 1/3.
    cv::Rect a(0, 0, 10, 10);
    cv::Rect b(5, 0, 10, 10);
    EXPECT_NEAR(BboxIou(a, b), 1.0f / 3.0f, 1e-5);
}

TEST(BboxIou, ContainmentIsRatioOfAreas) {
    // Smaller (5x5=25) inside larger (10x10=100): IoU = 25/100 = 0.25.
    cv::Rect a(2, 2, 5, 5);
    cv::Rect b(0, 0, 10, 10);
    EXPECT_NEAR(BboxIou(a, b), 0.25f, 1e-5);
}

// -- MaskIou -----------------------------------------------------------------

TEST(MaskIou, IdenticalSquaresAreOne) {
    Polygon p = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    EXPECT_FLOAT_EQ(MaskIou(p, p), 1.0f);
}

TEST(MaskIou, DisjointIsZero) {
    Polygon a = {{0, 0}, {5, 0}, {5, 5}, {0, 5}};
    Polygon b = {{20, 20}, {25, 20}, {25, 25}, {20, 25}};
    EXPECT_FLOAT_EQ(MaskIou(a, b), 0.0f);
}

TEST(MaskIou, EmptyPolygonReturnsZero) {
    Polygon empty = {};
    Polygon valid = {{0, 0}, {5, 0}, {5, 5}};
    EXPECT_FLOAT_EQ(MaskIou(empty, valid), 0.0f);
    EXPECT_FLOAT_EQ(MaskIou(valid, empty), 0.0f);
}

TEST(MaskIou, TooFewVerticesReturnsZero) {
    Polygon line = {{0, 0}, {10, 0}};
    Polygon valid = {{0, 0}, {5, 0}, {5, 5}};
    EXPECT_FLOAT_EQ(MaskIou(line, valid), 0.0f);
}

TEST(MaskIou, HalfOverlapApproxOneThird) {
    // Same construction as the BboxIou half-overlap case; rasterised IoU
    // should land close (allow some tolerance because cv::fillPoly's edges
    // include integer boundary pixels).
    Polygon a = {{0, 0}, {100, 0}, {100, 100}, {0, 100}};
    Polygon b = {{50, 0}, {150, 0}, {150, 100}, {50, 100}};
    EXPECT_NEAR(MaskIou(a, b), 1.0f / 3.0f, 0.02f);
}

// -- GreedyMatch -------------------------------------------------------------

TEST(GreedyMatch, EmptyInputsReturnEmpty) {
    EXPECT_TRUE(GreedyMatch({}, {}, 0.5f).empty());

    Polygon p = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    EXPECT_TRUE(GreedyMatch({p}, {}, 0.5f).empty());
    EXPECT_TRUE(GreedyMatch({}, {p}, 0.5f).empty());
}

TEST(GreedyMatch, PerfectOneToOnePairing) {
    Polygon a = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    Polygon b = {{100, 100}, {110, 100}, {110, 110}, {100, 110}};

    std::vector<Polygon> preds = {a, b};
    std::vector<Polygon> gts = {a, b};
    auto matches = GreedyMatch(preds, gts, 0.5f);
    ASSERT_EQ(matches.size(), 2u);
    // Both matches must have IoU ≈ 1 and each side must appear exactly once.
    std::vector<int> p_seen, g_seen;
    for (const auto& m : matches) {
        EXPECT_NEAR(m.bbox_iou, 1.0f, 1e-5);
        p_seen.push_back(m.pred_idx);
        g_seen.push_back(m.gt_idx);
    }
    std::sort(p_seen.begin(), p_seen.end());
    std::sort(g_seen.begin(), g_seen.end());
    EXPECT_EQ(p_seen, (std::vector<int>{0, 1}));
    EXPECT_EQ(g_seen, (std::vector<int>{0, 1}));
}

TEST(GreedyMatch, ThresholdFiltersWeakPairs) {
    // Two preds and two gts: pred 0 overlaps gt 0 at IoU=1 (perfect),
    // pred 1 overlaps gt 1 at IoU=1/3 (the half-overlap construction).
    // Threshold 0.5 should accept only the perfect pair.
    Polygon p0 = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    Polygon p1 = {{100, 0}, {110, 0}, {110, 10}, {100, 10}};
    Polygon g1 = {{105, 0}, {115, 0}, {115, 10}, {105, 10}};

    auto matches = GreedyMatch({p0, p1}, {p0, g1}, 0.5f);
    ASSERT_EQ(matches.size(), 1u);
    EXPECT_EQ(matches[0].pred_idx, 0);
    EXPECT_EQ(matches[0].gt_idx, 0);
    EXPECT_NEAR(matches[0].bbox_iou, 1.0f, 1e-5);
}

TEST(GreedyMatch, GreedyClaimsHighestIouFirst) {
    // Two preds compete for the same GT; the higher-IoU pred wins, the
    // other gets nothing (no second GT to fall back to).
    Polygon gt = {{0, 0}, {100, 0}, {100, 100}, {0, 100}};
    Polygon good = {{0, 0}, {100, 0}, {100, 100}, {0, 100}};      // IoU 1.0
    Polygon weaker = {{50, 0}, {150, 0}, {150, 100}, {50, 100}};  // IoU 1/3
    auto matches = GreedyMatch({weaker, good}, {gt}, 0.1f);
    ASSERT_EQ(matches.size(), 1u);
    EXPECT_EQ(matches[0].pred_idx, 1);
    EXPECT_EQ(matches[0].gt_idx, 0);
    EXPECT_NEAR(matches[0].bbox_iou, 1.0f, 1e-5);
}

// -- SafeDiv -----------------------------------------------------------------

TEST(SafeDiv, ZeroDenominatorReturnsNullopt) {
    EXPECT_FALSE(SafeDiv(5.0, 0.0).has_value());
    EXPECT_FALSE(SafeDiv(0.0, 0.0).has_value());
    EXPECT_FALSE(SafeDiv(5.0, -1.0).has_value());
}

TEST(SafeDiv, NormalDivision) {
    auto v = SafeDiv(3.0, 4.0);
    ASSERT_TRUE(v.has_value());
    EXPECT_FLOAT_EQ(*v, 0.75f);
}

// -- F1FromPR ----------------------------------------------------------------

TEST(F1FromPR, NulloptInputsReturnNullopt) {
    EXPECT_FALSE(F1FromPR(std::nullopt, 0.5f).has_value());
    EXPECT_FALSE(F1FromPR(0.5f, std::nullopt).has_value());
    EXPECT_FALSE(F1FromPR(std::nullopt, std::nullopt).has_value());
}

TEST(F1FromPR, BothZeroReturnsNullopt) { EXPECT_FALSE(F1FromPR(0.0f, 0.0f).has_value()); }

TEST(F1FromPR, HarmonicMean) {
    // P=0.5, R=0.5 → F1=0.5; P=1, R=0 → F1=0 (avoid div-by-zero exception).
    auto v = F1FromPR(0.5f, 0.5f);
    ASSERT_TRUE(v.has_value());
    EXPECT_NEAR(*v, 0.5f, 1e-5);

    v = F1FromPR(0.8f, 0.6f);
    ASSERT_TRUE(v.has_value());
    EXPECT_NEAR(*v, 2.0f * 0.8f * 0.6f / (0.8f + 0.6f), 1e-5);
}

// -- CountInstancesByIou -----------------------------------------------------

TEST(CountInstancesByIou, EmptyIsZero) { EXPECT_EQ(CountInstancesByIou({}, 0.5f), 0); }

TEST(CountInstancesByIou, SingleBoxIsOne) {
    EXPECT_EQ(CountInstancesByIou({cv::Rect(0, 0, 10, 10)}, 0.5f), 1);
}

TEST(CountInstancesByIou, IdenticalBoxesCollapse) {
    // Two identical boxes (IoU 1.0) are one instance — the cross-frame
    // duplicate case the GT union is *supposed* to merge.
    cv::Rect a(0, 0, 10, 10);
    EXPECT_EQ(CountInstancesByIou({a, a}, 0.5f), 1);
}

TEST(CountInstancesByIou, DisjointStaySeparate) {
    EXPECT_EQ(CountInstancesByIou({cv::Rect(0, 0, 10, 10), cv::Rect(100, 100, 10, 10)}, 0.5f), 2);
}

TEST(CountInstancesByIou, LowIouStaysSeparate) {
    // Half-overlap: IoU = 1/3 < 0.5 → two distinct instances. This is exactly
    // the over-merge the raster-union would wrongly fuse into one polygon.
    cv::Rect a(0, 0, 10, 10);
    cv::Rect b(5, 0, 10, 10);
    EXPECT_EQ(CountInstancesByIou({a, b}, 0.5f), 2);
}

TEST(CountInstancesByIou, HighIouMerges) {
    // Shift 1 of a 10px box → IoU ~0.82 > 0.5 → one instance.
    cv::Rect a(0, 0, 10, 10);
    cv::Rect b(1, 0, 10, 10);
    EXPECT_EQ(CountInstancesByIou({a, b}, 0.5f), 1);
}

TEST(CountInstancesByIou, DuplicatePairPlusDistinct) {
    cv::Rect dup(0, 0, 10, 10);
    cv::Rect other(50, 0, 10, 10);
    EXPECT_EQ(CountInstancesByIou({dup, dup, other}, 0.5f), 2);
}

TEST(CountInstancesByIou, TransitiveMerge) {
    // a~b and b~c each clear 0.5, but a~c (IoU 0.25) does not; union-find still
    // collapses all three into one component.
    cv::Rect a(0, 0, 10, 10);
    cv::Rect b(3, 0, 10, 10);  // IoU(a,b) = 7/13 ~0.54
    cv::Rect c(6, 0, 10, 10);  // IoU(b,c) ~0.54 ; IoU(a,c) = 0.25
    EXPECT_EQ(CountInstancesByIou({a, b, c}, 0.5f), 1);
}
