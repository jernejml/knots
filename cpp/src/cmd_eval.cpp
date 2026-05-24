// `knots eval` — test mode. Compares two dirs of stitched per-board JSONs
// (predictions vs ground truth):
//
//   knots eval --pred-dir P --gt-dir G [--labels-dir L --images-dir I] [opts]
//
//   1. If --labels-dir / --images-dir are given, first (re)stitch any missing
//      per-board GT into --gt-dir from the raw YOLO bbox labels.
//   2. For each board, greedy-match prediction and GT polygons by bbox IoU at
//      args.match_iou (highest-IoU pair first, until none clears threshold).
//   3. For each matched pair, compute mask IoU by rasterising both polygons.
//   4. Count TP / FP / FN; derive P / R / F1 and mean IoU.
//
// Per-board rows print to stdout; a JSON dump is written unless --no-write.
//
// Two caveats this report surfaces, both rooted in how GT is built (per-frame
// YOLO bboxes projected and fused by the same raster-union as predictions):
//   - Mask IoU tops out below 1.0 even for a perfect detection, since GT shape
//     is a union of rectangles, not the true knot edge.
//   - The union also fuses distinct-but-touching knots into one polygon. The
//     `gtMrg` column / `gt_merged_away` field report how many distinct GT knots
//     (estimated by IoU-clustering the raw annotations, in the GT stitch step)
//     the union swallowed — P / R / F1 are blind to those, because GT lost them
//     too.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "knots/cli_util.hpp"
#include "knots/commands.hpp"
#include "knots/geometry.hpp"
#include "knots/pipeline.hpp"

namespace fs = std::filesystem;

namespace knots {

namespace {

struct BoardData {
    std::vector<Polygon> preds;
    std::vector<Polygon> gts;
    std::optional<int> gt_distinct;  // gt_stats.distinct_estimate, if the GT JSON has it
};

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

nlohmann::json ReadJson(const fs::path& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path.string());
    nlohmann::json j;
    f >> j;
    return j;
}

std::vector<Polygon> PolygonsFromJson(const nlohmann::json& j) {
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

// gt_stats.distinct_estimate, written by the GT stitch step. Absent on GT JSONs
// produced before that step existed; eval then just omits the merge delta.
std::optional<int> ReadDistinctEstimate(const nlohmann::json& j) {
    if (!j.contains("gt_stats")) return std::nullopt;
    const auto& s = j["gt_stats"];
    if (!s.contains("distinct_estimate") || !s["distinct_estimate"].is_number_integer()) {
        return std::nullopt;
    }
    return s["distinct_estimate"].get<int>();
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
    std::optional<int> gt_distinct;     // estimated distinct physical knots in GT
    std::optional<int> gt_merged_away;  // gt_distinct - gt_count (union over-merge), if known
    std::vector<float> matched_ious;
};

// Load per-board polygons from two directories of JSON; return only boards
// present in both, intersected with the optional --boards / --boards-file
// filter.
std::map<int, BoardData> CollectData(const EvalArgs& args) {
    if (!fs::is_directory(args.pred_dir)) {
        throw std::runtime_error("missing pred dir: " + args.pred_dir.string());
    }
    if (!fs::is_directory(args.gt_dir)) {
        throw std::runtime_error("missing gt dir: " + args.gt_dir.string());
    }

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

    std::unordered_set<int> filter(args.boards.boards.begin(), args.boards.boards.end());
    if (!args.boards.boards_file.empty()) filter = cli::ParseBoardsFile(args.boards.boards_file);

    std::map<int, BoardData> data;
    for (int board : common) {
        if (!filter.empty() && !filter.count(board)) continue;
        const auto gt_json = ReadJson(args.gt_dir / (std::to_string(board) + ".json"));
        BoardData bd;
        bd.preds = PolygonsFromJson(ReadJson(args.pred_dir / (std::to_string(board) + ".json")));
        bd.gts = PolygonsFromJson(gt_json);
        bd.gt_distinct = ReadDistinctEstimate(gt_json);
        data.emplace(board, std::move(bd));
    }
    return data;
}

}  // namespace

int CmdEval(const EvalArgs& args_in) {
    EvalArgs args = args_in;
    if (args.out_json.empty()) {
        args.out_json = fs::path("out") / "analysis" / "eval_boards.json";
    }

    // --labels-dir + --images-dir are paired; either both set or neither.
    const bool rebuild_gt = !args.labels_dir.empty() || !args.images_dir.empty();
    if (rebuild_gt && (args.labels_dir.empty() || args.images_dir.empty())) {
        std::cerr << "--labels-dir and --images-dir must be set together (GT rebuild)\n";
        return 2;
    }

    try {
        // Rebuild missing per-board GT before the comparison if asked.
        if (rebuild_gt) {
            if (!fs::is_directory(args.labels_dir)) {
                throw std::runtime_error("missing labels dir: " + args.labels_dir.string());
            }
            if (!fs::is_directory(args.images_dir)) {
                throw std::runtime_error("missing images dir: " + args.images_dir.string());
            }
            std::unordered_set<int> boards_filter(args.boards.boards.begin(),
                                                  args.boards.boards.end());
            if (!args.boards.boards_file.empty()) {
                boards_filter = cli::ParseBoardsFile(args.boards.boards_file);
            }
            std::cout << "knots eval: stitching GT from " << args.labels_dir.string() << " → "
                      << args.gt_dir.string() << " (skip-if-exists)\n";
            auto gt = pipeline::StitchGtForBoards(args.labels_dir, args.images_dir, args.gt_dir,
                                                  boards_filter, args.stitch.stride_px,
                                                  args.stitch.simplify_eps_px, /*force=*/false);
            std::cout << "  gt: written=" << gt.written << "  skipped=" << gt.skipped
                      << "  polygons=" << gt.total_polys << "\n\n";
        }

        auto data = CollectData(args);
        if (data.empty()) {
            std::cerr << "no boards to evaluate (empty intersection)\n";
            return 1;
        }

        std::cout << "knots eval: " << data.size() << " board(s)  match-iou=" << args.match_iou
                  << "\n"
                  << "  pred: " << args.pred_dir.string() << "/\n"
                  << "  gt:   " << args.gt_dir.string() << "/\n\n";

        std::vector<BoardResult> per_board;
        per_board.reserve(data.size());
        for (const auto& [board, bp] : data) {
            const auto& preds = bp.preds;
            const auto& gts = bp.gts;
            const auto matches = GreedyMatch(preds, gts, args.match_iou);

            BoardResult r;
            r.board = board;
            r.pred_count = static_cast<int>(preds.size());
            r.gt_count = static_cast<int>(gts.size());
            r.tp = static_cast<int>(matches.size());
            r.fp = r.pred_count - r.tp;
            r.fn = r.gt_count - r.tp;
            r.gt_distinct = bp.gt_distinct;
            if (bp.gt_distinct) {
                r.gt_merged_away = std::max(0, *bp.gt_distinct - r.gt_count);
            }

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
        long total_gt_distinct = 0, total_merged_away = 0;
        bool any_gt_stats = false;
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
            if (b.gt_distinct) {
                total_gt_distinct += *b.gt_distinct;
                total_merged_away += b.gt_merged_away.value_or(0);
                any_gt_stats = true;
            }
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
            "  board  pred    gt    tp    fp    fn       P       R      F1    mIoU  gtMrg";
        std::cout << header << "\n";
        std::cout << std::string(header.size(), '-') << "\n";
        for (const auto& b : per_board) {
            std::cout << std::setw(7) << b.board << "  " << std::setw(4) << b.pred_count << "  "
                      << std::setw(4) << b.gt_count << "  " << std::setw(4) << b.tp << "  "
                      << std::setw(4) << b.fp << "  " << std::setw(4) << b.fn << "  "
                      << std::setw(6) << FmtOpt(b.precision) << "  " << std::setw(6)
                      << FmtOpt(b.recall) << "  " << std::setw(6) << FmtOpt(b.f1) << "   "
                      << std::fixed << std::setprecision(3) << b.mean_iou << "  " << std::setw(5)
                      << (b.gt_merged_away ? std::to_string(*b.gt_merged_away) : std::string("-"))
                      << "\n";
        }
        std::cout << "\nAggregate\n"
                  << "  boards=" << per_board.size() << "  match-iou=" << args.match_iou << "\n"
                  << "  total: pred=" << total_pred << "  gt=" << total_gt << "  tp=" << total_tp
                  << "  fp=" << total_fp << "  fn=" << total_fn << "\n"
                  << "  P=" << FmtOpt(agg_p) << "  R=" << FmtOpt(agg_r) << "  F1=" << FmtOpt(agg_f1)
                  << "\n"
                  << "  mean IoU micro=" << std::fixed << std::setprecision(3) << mean_iou_micro
                  << "  macro=" << std::fixed << std::setprecision(3) << mean_iou_macro << "\n";
        if (any_gt_stats) {
            const double pct =
                total_gt_distinct > 0
                    ? 100.0 * static_cast<double>(total_merged_away) / total_gt_distinct
                    : 0.0;
            std::cout << "  union over-merge: " << total_merged_away << " of ~" << total_gt_distinct
                      << " distinct GT knots (" << std::fixed << std::setprecision(1) << pct
                      << "%) fused into a neighbour — P/R/F1 above can't see these\n";
        }

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
                {"gt_distinct_total",
                 any_gt_stats ? nlohmann::json(total_gt_distinct) : nlohmann::json(nullptr)},
                {"gt_merged_away_total",
                 any_gt_stats ? nlohmann::json(total_merged_away) : nlohmann::json(nullptr)},
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
                jb["gt_distinct_estimate"] =
                    b.gt_distinct ? nlohmann::json(*b.gt_distinct) : nlohmann::json(nullptr);
                jb["gt_merged_away"] =
                    b.gt_merged_away ? nlohmann::json(*b.gt_merged_away) : nlohmann::json(nullptr);
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
