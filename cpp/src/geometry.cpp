#include "knots/geometry.hpp"

#include <algorithm>
#include <opencv2/imgproc.hpp>

namespace knots {

cv::Rect PolygonBbox(const Polygon& poly) {
    int x0 = poly.front().x, y0 = poly.front().y, x1 = x0, y1 = y0;
    for (const auto& p : poly) {
        if (p.x < x0) x0 = p.x;
        if (p.y < y0) y0 = p.y;
        if (p.x > x1) x1 = p.x;
        if (p.y > y1) y1 = p.y;
    }
    return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

float BboxIou(const cv::Rect& a, const cv::Rect& b) {
    const int ix1 = std::max(a.x, b.x);
    const int iy1 = std::max(a.y, b.y);
    const int ix2 = std::min(a.x + a.width, b.x + b.width);
    const int iy2 = std::min(a.y + a.height, b.y + b.height);
    const int iw = ix2 - ix1, ih = iy2 - iy1;
    if (iw <= 0 || ih <= 0) return 0.f;
    const float inter = static_cast<float>(iw) * ih;
    const float ua = static_cast<float>(a.width) * a.height;
    const float ub = static_cast<float>(b.width) * b.height;
    const float uni = ua + ub - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

float MaskIou(const Polygon& a, const Polygon& b) {
    if (a.size() < 3 || b.size() < 3) return 0.f;
    const cv::Rect ba = PolygonBbox(a);
    const cv::Rect bb = PolygonBbox(b);
    const int x0 = std::min(ba.x, bb.x);
    const int y0 = std::min(ba.y, bb.y);
    const int x1 = std::max(ba.x + ba.width, bb.x + bb.width);
    const int y1 = std::max(ba.y + ba.height, bb.y + bb.height);
    const int w = std::max(1, x1 - x0 + 1);
    const int h = std::max(1, y1 - y0 + 1);

    cv::Mat ma = cv::Mat::zeros(h, w, CV_8U);
    cv::Mat mb = cv::Mat::zeros(h, w, CV_8U);
    Polygon sa, sb;
    sa.reserve(a.size());
    sb.reserve(b.size());
    for (const auto& p : a) sa.emplace_back(p.x - x0, p.y - y0);
    for (const auto& p : b) sb.emplace_back(p.x - x0, p.y - y0);
    std::vector<Polygon> tmpa{std::move(sa)};
    std::vector<Polygon> tmpb{std::move(sb)};
    cv::fillPoly(ma, tmpa, cv::Scalar(1));
    cv::fillPoly(mb, tmpb, cv::Scalar(1));

    cv::Mat inter_m, union_m;
    cv::bitwise_and(ma, mb, inter_m);
    cv::bitwise_or(ma, mb, union_m);
    const double inter = cv::countNonZero(inter_m);
    const double uni = cv::countNonZero(union_m);
    return uni > 0.0 ? static_cast<float>(inter / uni) : 0.f;
}

std::vector<Match> GreedyMatch(const std::vector<Polygon>& preds,
                                const std::vector<Polygon>& gts, float threshold) {
    std::vector<cv::Rect> pb, gb;
    pb.reserve(preds.size());
    gb.reserve(gts.size());
    for (const auto& p : preds) pb.push_back(PolygonBbox(p));
    for (const auto& g : gts) gb.push_back(PolygonBbox(g));

    struct Cand {
        float iou;
        int i;
        int j;
    };
    std::vector<Cand> cands;
    for (size_t i = 0; i < pb.size(); ++i) {
        for (size_t j = 0; j < gb.size(); ++j) {
            const float iou = BboxIou(pb[i], gb[j]);
            if (iou >= threshold) {
                cands.push_back({iou, static_cast<int>(i), static_cast<int>(j)});
            }
        }
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.iou > b.iou; });

    std::vector<bool> mp(preds.size(), false), mg(gts.size(), false);
    std::vector<Match> out;
    for (const auto& c : cands) {
        if (mp[c.i] || mg[c.j]) continue;
        mp[c.i] = true;
        mg[c.j] = true;
        out.push_back({c.i, c.j, c.iou});
    }
    return out;
}

int CountInstancesByIou(const std::vector<cv::Rect>& boxes, float iou_thresh) {
    const int n = static_cast<int>(boxes.size());
    if (n == 0) return 0;
    std::vector<int> parent(n);
    for (int i = 0; i < n; ++i) parent[i] = i;
    auto find = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (BboxIou(boxes[i], boxes[j]) >= iou_thresh) {
                const int ri = find(i), rj = find(j);
                if (ri != rj) parent[ri] = rj;
            }
        }
    }
    int components = 0;
    for (int i = 0; i < n; ++i) {
        if (find(i) == i) ++components;
    }
    return components;
}

std::optional<float> SafeDiv(double num, double den) {
    if (den <= 0.0) return std::nullopt;
    return static_cast<float>(num / den);
}

std::optional<float> F1FromPR(std::optional<float> p, std::optional<float> r) {
    if (!p || !r) return std::nullopt;
    const float s = *p + *r;
    if (s <= 0.f) return std::nullopt;
    return 2.f * (*p) * (*r) / s;
}

}  // namespace knots
