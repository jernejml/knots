// `knots eval` — test mode: compares per-board prediction polygons against
// per-board GT polygons.
//
// Inputs are two directories of per-board JSONs in the schema written by
// `knots stitch` and `knots gt-stitch`. The algorithm matches the Python
// reference in scripts/eval_boards.py:
//
//   1. For each board, compute pairwise bbox IoU between prediction and GT
//      polygons.
//   2. Greedy-match at --match-iou (default 0.5): pop the highest-IoU pair,
//      mark both sides as matched, repeat until no pair remains above
//      threshold.
//   3. For each matched pair, compute mask IoU by rasterizing both polygons
//      onto a shared ROI mask via cv::fillPoly.
//   4. Count TP / FP (extras) / FN (missing); derive P / R / F1 and mean IoU.
//
// Aggregate metrics span every board common to both directories. Per-board
// rows print to stdout; a JSON dump is written unless --no-write is set.
//
// Note on the GT side: GT polygons are derived from per-frame YOLO bboxes
// projected and unioned by the same raster-union pipeline used for the
// predictions. Mask IoU therefore tops out below 1.0 even for a perfect
// detection — the metric reflects relative model quality given that GT
// shape, not absolute knot-edge agreement.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "knots/commands.hpp"

namespace fs = std::filesystem;

namespace knots {

namespace {

using Polygon = std::vector<cv::Point>;

struct EvalArgs {
    fs::path pred_dir;
    fs::path gt_dir;
    fs::path out_json;
    float match_iou = 0.5f;
    std::string boards_csv;
    fs::path boards_file;
    bool no_write = false;
};

void PrintUsage() {
    std::cerr << "usage: knots eval --pred-dir P --gt-dir G [opts]\n"
                 "  --pred-dir DIR        per-board prediction JSONs (`knots stitch`)\n"
                 "  --gt-dir DIR          per-board GT JSONs (`knots gt-stitch`)\n"
                 "  --out PATH            JSON output path (default analysis/eval_boards.json)\n"
                 "  --match-iou F         bbox IoU threshold for matching (default 0.5)\n"
                 "  --boards LIST         comma-separated board IDs to restrict to\n"
                 "  --boards-file FILE    one board ID per line ('#' comments allowed)\n"
                 "  --no-write            skip JSON output; print to stdout only\n";
}

bool RequireNext(const std::string& flag, int i, int argc) {
    if (i + 1 >= argc) {
        std::cerr << "missing value for " << flag << "\n";
        return false;
    }
    return true;
}

bool ParseArgs(int argc, char** argv, EvalArgs& out) {
    int i = 1;
    while (i < argc) {
        std::string a = argv[i];
        if (a == "--pred-dir" && RequireNext(a, i, argc)) {
            out.pred_dir = argv[++i];
        } else if (a == "--gt-dir" && RequireNext(a, i, argc)) {
            out.gt_dir = argv[++i];
        } else if (a == "--out" && RequireNext(a, i, argc)) {
            out.out_json = argv[++i];
        } else if (a == "--match-iou" && RequireNext(a, i, argc)) {
            out.match_iou = std::stof(argv[++i]);
        } else if (a == "--boards" && RequireNext(a, i, argc)) {
            out.boards_csv = argv[++i];
        } else if (a == "--boards-file" && RequireNext(a, i, argc)) {
            out.boards_file = argv[++i];
        } else if (a == "--no-write") {
            out.no_write = true;
        } else if (a == "--help" || a == "-h") {
            return false;
        } else {
            std::cerr << "unrecognised arg: " << a << "\n";
            return false;
        }
        ++i;
    }
    if (out.pred_dir.empty() || out.gt_dir.empty()) {
        std::cerr << "--pred-dir and --gt-dir are required\n";
        return false;
    }
    return true;
}

std::unordered_set<int> ParseBoardsCsv(const std::string& s) {
    std::unordered_set<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        try {
            out.insert(std::stoi(tok));
        } catch (...) {
        }
    }
    return out;
}

std::unordered_set<int> ParseBoardsFile(const fs::path& p) {
    std::unordered_set<int> out;
    std::ifstream f(p);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    std::string line;
    while (std::getline(f, line)) {
        auto a = line.find_first_not_of(" \t");
        if (a == std::string::npos) continue;
        if (line[a] == '#') continue;
        try {
            out.insert(std::stoi(line.substr(a)));
        } catch (...) {
        }
    }
    return out;
}

std::set<int> ListBoardIds(const fs::path& dir) {
    std::set<int> out;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        if (e.path().extension() != ".json") continue;
        const std::string stem = e.path().stem().string();
        if (stem.empty()) continue;
        if (!std::all_of(stem.begin(), stem.end(), [](unsigned char c) { return std::isdigit(c); }))
            continue;
        try {
            out.insert(std::stoi(stem));
        } catch (...) {
        }
    }
    return out;
}

std::vector<Polygon> LoadBoardPolygons(const fs::path& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path.string());
    nlohmann::json j;
    f >> j;
    std::vector<Polygon> out;
    if (!j.contains("knots")) return out;
    for (const auto& k : j["knots"]) {
        if (!k.contains("polygon")) continue;
        Polygon poly;
        for (const auto& pt : k["polygon"]) {
            poly.emplace_back(pt[0].get<int>(), pt[1].get<int>());
        }
        if (poly.size() >= 3) out.push_back(std::move(poly));
    }
    return out;
}

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

// Rasterise both polygons onto a shared ROI mask; return mask IoU.
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

struct Match {
    int pred_idx;
    int gt_idx;
    float bbox_iou;
};

// Greedy bbox-IoU matching. Pair list is sorted by descending IoU; we walk it,
// claiming each pair the first time both sides are unmatched.
std::vector<Match> GreedyMatch(const std::vector<Polygon>& preds, const std::vector<Polygon>& gts,
                               float threshold) {
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

// Optional metrics: nullopt means "denominator was zero". Serialised as JSON
// null and printed as "n/a" for parity with eval_boards.py.
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

nlohmann::json OptToJson(std::optional<float> v) {
    if (!v) return nullptr;
    return *v;
}

std::string FmtOpt(std::optional<float> v) {
    if (!v) return "  n/a";
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3) << *v;
    return ss.str();
}

struct BoardResult {
    int board;
    int pred_count;
    int gt_count;
    int tp;
    int fp;
    int fn;
    std::optional<float> precision;
    std::optional<float> recall;
    std::optional<float> f1;
    float mean_iou;
    std::vector<float> matched_ious;
};

}  // namespace

int CmdEval(int argc, char** argv) {
    EvalArgs args;
    if (!ParseArgs(argc, argv, args)) {
        PrintUsage();
        return 2;
    }
    if (!fs::is_directory(args.pred_dir)) {
        std::cerr << "missing pred dir: " << args.pred_dir << "\n";
        return 1;
    }
    if (!fs::is_directory(args.gt_dir)) {
        std::cerr << "missing gt dir: " << args.gt_dir << "\n";
        return 1;
    }
    if (args.out_json.empty()) {
        args.out_json = fs::path("analysis") / "eval_boards.json";
    }

    try {
        // Resolve board id set: intersection of dirs, optionally filtered.
        std::set<int> pred_ids = ListBoardIds(args.pred_dir);
        std::set<int> gt_ids = ListBoardIds(args.gt_dir);

        std::set<int> common;
        std::set_intersection(pred_ids.begin(), pred_ids.end(), gt_ids.begin(), gt_ids.end(),
                              std::inserter(common, common.begin()));
        std::vector<int> only_pred, only_gt;
        std::set_difference(pred_ids.begin(), pred_ids.end(), gt_ids.begin(), gt_ids.end(),
                            std::back_inserter(only_pred));
        std::set_difference(gt_ids.begin(), gt_ids.end(), pred_ids.begin(), pred_ids.end(),
                            std::back_inserter(only_gt));
        if (!only_pred.empty()) {
            std::cout << "warning: " << only_pred.size() << " board(s) in pred dir but not GT\n";
        }
        if (!only_gt.empty()) {
            std::cout << "warning: " << only_gt.size() << " board(s) in GT dir but not pred\n";
        }

        std::unordered_set<int> filter;
        if (!args.boards_csv.empty()) filter = ParseBoardsCsv(args.boards_csv);
        if (!args.boards_file.empty()) filter = ParseBoardsFile(args.boards_file);
        std::vector<int> boards;
        for (int b : common) {
            if (!filter.empty() && !filter.count(b)) continue;
            boards.push_back(b);
        }
        if (boards.empty()) {
            std::cerr << "no boards to evaluate (empty intersection)\n";
            return 1;
        }

        std::cout << "knots eval: " << boards.size() << " board(s)  match-iou=" << args.match_iou
                  << "\n"
                  << "  pred: " << args.pred_dir.string() << "/\n"
                  << "  gt:   " << args.gt_dir.string() << "/\n\n";

        std::vector<BoardResult> per_board;
        per_board.reserve(boards.size());
        for (int board : boards) {
            const auto preds = LoadBoardPolygons(args.pred_dir / (std::to_string(board) + ".json"));
            const auto gts = LoadBoardPolygons(args.gt_dir / (std::to_string(board) + ".json"));
            const auto matches = GreedyMatch(preds, gts, args.match_iou);

            BoardResult r;
            r.board = board;
            r.pred_count = static_cast<int>(preds.size());
            r.gt_count = static_cast<int>(gts.size());
            r.tp = static_cast<int>(matches.size());
            r.fp = r.pred_count - r.tp;
            r.fn = r.gt_count - r.tp;

            double iou_sum = 0.0;
            r.matched_ious.reserve(matches.size());
            for (const auto& m : matches) {
                const float iou = MaskIou(preds[m.pred_idx], gts[m.gt_idx]);
                r.matched_ious.push_back(iou);
                iou_sum += iou;
            }
            r.mean_iou = matches.empty() ? 0.f : static_cast<float>(iou_sum / matches.size());
            r.precision = SafeDiv(r.tp, r.tp + r.fp);
            r.recall = SafeDiv(r.tp, r.tp + r.fn);
            r.f1 = F1FromPR(r.precision, r.recall);
            per_board.push_back(std::move(r));
        }

        // Aggregate.
        long total_pred = 0, total_gt = 0, total_tp = 0, total_fp = 0, total_fn = 0;
        double all_iou_sum = 0.0;
        size_t all_iou_n = 0;
        double per_board_iou_sum = 0.0;
        size_t per_board_iou_n = 0;
        for (const auto& b : per_board) {
            total_pred += b.pred_count;
            total_gt += b.gt_count;
            total_tp += b.tp;
            total_fp += b.fp;
            total_fn += b.fn;
            for (float v : b.matched_ious) {
                all_iou_sum += v;
                ++all_iou_n;
            }
            if (b.tp > 0) {
                per_board_iou_sum += b.mean_iou;
                ++per_board_iou_n;
            }
        }
        auto agg_p = SafeDiv(total_tp, total_tp + total_fp);
        auto agg_r = SafeDiv(total_tp, total_tp + total_fn);
        auto agg_f1 = F1FromPR(agg_p, agg_r);
        const float mean_iou_micro = all_iou_n ? static_cast<float>(all_iou_sum / all_iou_n) : 0.f;
        const float mean_iou_macro =
            per_board_iou_n ? static_cast<float>(per_board_iou_sum / per_board_iou_n) : 0.f;

        // Stdout table.
        const std::string header =
            "  board  pred    gt    tp    fp    fn       P       R      F1    mIoU";
        std::cout << header << "\n";
        std::cout << std::string(header.size(), '-') << "\n";
        for (const auto& b : per_board) {
            std::cout << std::setw(7) << b.board << "  " << std::setw(4) << b.pred_count << "  "
                      << std::setw(4) << b.gt_count << "  " << std::setw(4) << b.tp << "  "
                      << std::setw(4) << b.fp << "  " << std::setw(4) << b.fn << "  "
                      << std::setw(6) << FmtOpt(b.precision) << "  " << std::setw(6)
                      << FmtOpt(b.recall) << "  " << std::setw(6) << FmtOpt(b.f1) << "   "
                      << std::fixed << std::setprecision(3) << b.mean_iou << "\n";
        }
        std::cout << "\nAggregate\n"
                  << "  boards=" << per_board.size() << "  match-iou=" << args.match_iou << "\n"
                  << "  total: pred=" << total_pred << "  gt=" << total_gt << "  tp=" << total_tp
                  << "  fp=" << total_fp << "  fn=" << total_fn << "\n"
                  << "  P=" << FmtOpt(agg_p) << "  R=" << FmtOpt(agg_r) << "  F1=" << FmtOpt(agg_f1)
                  << "\n"
                  << "  mean IoU micro=" << std::fixed << std::setprecision(3) << mean_iou_micro
                  << "  macro=" << std::fixed << std::setprecision(3) << mean_iou_macro << "\n";

        if (!args.no_write) {
            nlohmann::json out;
            out["aggregate"] = {
                {"boards", per_board.size()},
                {"match_iou_threshold", args.match_iou},
                {"total_pred", total_pred},
                {"total_gt", total_gt},
                {"total_tp", total_tp},
                {"total_fp", total_fp},
                {"total_fn", total_fn},
                {"precision", OptToJson(agg_p)},
                {"recall", OptToJson(agg_r)},
                {"f1", OptToJson(agg_f1)},
                {"mean_iou_micro", mean_iou_micro},
                {"mean_iou_macro", mean_iou_macro},
            };
            out["per_board"] = nlohmann::json::array();
            for (const auto& b : per_board) {
                nlohmann::json jb;
                jb["board"] = b.board;
                jb["pred_count"] = b.pred_count;
                jb["gt_count"] = b.gt_count;
                jb["tp"] = b.tp;
                jb["fp"] = b.fp;
                jb["fn"] = b.fn;
                jb["precision"] = OptToJson(b.precision);
                jb["recall"] = OptToJson(b.recall);
                jb["f1"] = OptToJson(b.f1);
                jb["mean_iou"] = b.mean_iou;
                jb["matched_ious"] = nlohmann::json::array();
                for (float v : b.matched_ious) {
                    jb["matched_ious"].push_back(std::round(v * 10000.f) / 10000.f);
                }
                out["per_board"].push_back(std::move(jb));
            }
            fs::create_directories(args.out_json.parent_path());
            std::ofstream of(args.out_json);
            of << out.dump(2);
            std::cout << "\nwrote " << args.out_json.string() << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

}  // namespace knots
