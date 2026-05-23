// `knots eval` — test mode. Two ways to invoke the same comparison algorithm:
//
// Mode A (compare two dirs of stitched per-board JSONs):
//   knots eval --pred-dir P --gt-dir G [opts]
//
// Mode B (one-shot: infer + GT-stitch + compare in-process):
//   knots eval --model M --images-dir I --labels-dir L [opts]
//
// The mode is auto-detected from which flags are populated. The matching /
// IoU / aggregate code is identical:
//
//   1. For each board, compute pairwise bbox IoU between prediction and GT
//      polygons.
//   2. Greedy-match at args.match_iou: pop the highest-IoU pair, mark both
//      sides as matched, repeat until no pair remains above threshold.
//   3. For each matched pair, compute mask IoU by rasterising both polygons
//      onto a shared ROI mask via cv::fillPoly.
//   4. Count TP / FP / FN; derive P / R / F1 and mean IoU.
//
// Per-board rows print to stdout; a JSON dump is written unless no_write.
//
// Note on the GT side: GT polygons are derived from per-frame YOLO bboxes
// projected and unioned by the same raster-union pipeline used for the
// predictions. Mask IoU therefore tops out below 1.0 even for a perfect
// detection — the metric reflects relative model quality given that GT
// shape, not absolute knot-edge agreement.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
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
#include "knots/inference.hpp"
#include "knots/pipeline.hpp"
#include "knots/stitching.hpp"

namespace fs = std::filesystem;

namespace knots {

namespace {

using BoardPolygons = std::pair<std::vector<Polygon>, std::vector<Polygon>>;  // (preds, gts)

bool HasModeAFlags(const EvalArgs& a) {
    return !a.pred_dir.empty() || !a.gt_dir.empty();
}

bool HasModeBFlags(const EvalArgs& a) {
    return !a.inference.model.empty() || !a.images_dir.empty() || !a.labels_dir.empty();
}

// CLI11 doesn't enforce mode-coherence (the two modes share an --out flag
// and would clash if expressed as option groups), so the callback validates
// here and returns a usage hint on failure.
bool ValidateMode(const EvalArgs& a) {
    const bool mode_a = HasModeAFlags(a);
    const bool mode_b = HasModeBFlags(a);
    if (mode_a && mode_b) {
        std::cerr << "mode A (--pred-dir/--gt-dir) and mode B (--model/--images-dir/"
                     "--labels-dir) are mutually exclusive\n";
        return false;
    }
    if (!mode_a && !mode_b) {
        std::cerr << "either Mode A (--pred-dir + --gt-dir) or Mode B (--model + "
                     "--images-dir + --labels-dir) is required\n";
        return false;
    }
    if (mode_a && (a.pred_dir.empty() || a.gt_dir.empty())) {
        std::cerr << "Mode A requires both --pred-dir and --gt-dir\n";
        return false;
    }
    if (mode_b && (a.inference.model.empty() || a.images_dir.empty() || a.labels_dir.empty())) {
        std::cerr << "Mode B requires --model, --images-dir, and --labels-dir\n";
        return false;
    }
    return true;
}

// -- Mode A helpers ----------------------------------------------------------

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

std::vector<Polygon> LoadBoardPolygonsFromJson(const fs::path& path) {
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

// Mode A: load per-board polygons from two directories of JSON.
std::map<int, BoardPolygons> CollectModeAData(const EvalArgs& args) {
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

    std::map<int, BoardPolygons> data;
    for (int board : common) {
        if (!filter.empty() && !filter.count(board)) continue;
        auto preds = LoadBoardPolygonsFromJson(args.pred_dir / (std::to_string(board) + ".json"));
        auto gts = LoadBoardPolygonsFromJson(args.gt_dir / (std::to_string(board) + ".json"));
        data.emplace(board, BoardPolygons{std::move(preds), std::move(gts)});
    }
    return data;
}

// Mode B: walk labels dir, run inference on each frame, stitch in memory,
// build GT polygons from YOLO bboxes (same stitch pipeline). Returns the
// per-board (preds, gts) map plus the EP used (for the banner) via out-param.
std::map<int, BoardPolygons> CollectModeBData(const EvalArgs& args, std::string& ep_out,
                                              size_t& frames_total_out,
                                              pipeline::InferStats& stats_out) {
    if (!fs::exists(args.inference.model)) {
        throw std::runtime_error("model not found: " + args.inference.model.string());
    }
    if (!fs::is_directory(args.images_dir)) {
        throw std::runtime_error("images dir not found: " + args.images_dir.string());
    }
    if (!fs::is_directory(args.labels_dir)) {
        throw std::runtime_error("labels dir not found: " + args.labels_dir.string());
    }

    const auto boards_filter = cli::BuildBoardsFilter(
        args.boards.boards, args.boards.boards_file,
        args.splits.partitions_json, args.splits.split);

    const auto by_board =
        pipeline::CollectFramesByBoard(args.labels_dir, ".txt", {}, boards_filter);
    if (by_board.empty()) {
        throw std::runtime_error("no boards/frames to evaluate");
    }
    frames_total_out = 0;
    for (const auto& [_, frames] : by_board) frames_total_out += frames.size();

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "knots");
    Ort::Session session = MakeSession(env, args.inference.model, !args.inference.cpu_only, ep_out);

    const size_t heartbeat = std::max<size_t>(20, frames_total_out / 20);
    size_t global_idx = 0;
    auto frame_done = [&]() {
        ++global_idx;
        if (global_idx % heartbeat == 0 || global_idx == frames_total_out) {
            std::cerr << "  ... " << global_idx << "/" << frames_total_out << " frames\n";
        }
    };

    std::map<int, BoardPolygons> data;
    for (const auto& [board, frames] : by_board) {
        auto pred_fp = pipeline::InferBoardFrames(session, args.images_dir, frames,
                                                  args.inference.conf, stats_out, frame_done);
        auto gt_fp = pipeline::LoadGtBoardFrames(args.labels_dir, args.images_dir, frames);

        auto pred_sb = StitchBoardPolygons(board, std::move(pred_fp), args.stitch.stride_px,
                                           args.stitch.simplify_eps_px);
        auto gt_sb = StitchBoardPolygons(board, std::move(gt_fp), args.stitch.stride_px,
                                         args.stitch.simplify_eps_px);

        data.emplace(board, BoardPolygons{std::move(pred_sb.polygons), std::move(gt_sb.polygons)});
    }
    return data;
}

}  // namespace

int CmdEval(const EvalArgs& args_in) {
    if (!ValidateMode(args_in)) return 2;
    EvalArgs args = args_in;
    if (args.out_json.empty()) {
        args.out_json = fs::path("out") / "analysis" / "eval_boards.json";
    }
    const bool mode_b = HasModeBFlags(args);

    try {
        std::map<int, BoardPolygons> data;
        std::string ep;
        size_t frames_total = 0;
        pipeline::InferStats infer_stats;

        if (mode_b) {
            data = CollectModeBData(args, ep, frames_total, infer_stats);
        } else {
            data = CollectModeAData(args);
        }

        if (data.empty()) {
            std::cerr << "no boards to evaluate (empty intersection)\n";
            return 1;
        }

        // Banner.
        if (mode_b) {
            std::cout << "knots eval (Mode B): " << data.size() << " board(s) / " << frames_total
                      << " frame(s)  match-iou=" << args.match_iou
                      << "  conf=" << args.inference.conf << "  ep=" << ep << "\n"
                      << "  images: " << args.images_dir.string() << "/\n"
                      << "  labels: " << args.labels_dir.string() << "/\n\n";
        } else {
            std::cout << "knots eval: " << data.size() << " board(s)  match-iou=" << args.match_iou
                      << "\n"
                      << "  pred: " << args.pred_dir.string() << "/\n"
                      << "  gt:   " << args.gt_dir.string() << "/\n\n";
        }

        std::vector<BoardResult> per_board;
        per_board.reserve(data.size());
        for (const auto& [board, bp] : data) {
            const auto& preds = bp.first;
            const auto& gts = bp.second;
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
        if (mode_b) {
            std::cout << "  frames: processed=" << infer_stats.processed
                      << "  unread=" << infer_stats.unread << "  failed=" << infer_stats.failed
                      << "\n";
            if (infer_stats.processed > 0) {
                const double avg_ms =
                    1000.0 * infer_stats.total_inference_sec / infer_stats.processed;
                std::cout << "  inference: total=" << std::fixed << std::setprecision(2)
                          << infer_stats.total_inference_sec << "s  avg=" << std::fixed
                          << std::setprecision(1) << avg_ms << " ms/frame\n";
            }
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
            };
            if (mode_b) {
                out["aggregate"]["frames_processed"] = infer_stats.processed;
                out["aggregate"]["inference_total_sec"] =
                    std::round(infer_stats.total_inference_sec * 1000.0) / 1000.0;
                if (infer_stats.processed > 0) {
                    out["aggregate"]["inference_avg_ms_per_frame"] =
                        std::round(1000.0 * infer_stats.total_inference_sec /
                                   infer_stats.processed * 100.0) /
                        100.0;
                }
            }
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
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

}  // namespace knots
