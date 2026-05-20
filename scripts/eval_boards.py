#!/usr/bin/env python3
"""Evaluate per-board polygon predictions against per-board GT polygons.

Both inputs are directories of `{board}.json` files with the schema written
by `knots stitch` / `knots gt-stitch`. The script:

  1. Per board, computes pairwise bbox IoU between prediction polygons and
     GT polygons.
  2. Greedy-matches at --match-iou (default 0.5): pop the highest IoU pair,
     mark both as matched, repeat until no pair remains above threshold.
  3. For each matched pair, computes mask IoU by rasterizing both polygons
     onto a shared ROI mask via PIL.
  4. Counts TP / FP / FN, derives P / R / F1, mean IoU on the TP set.

Aggregate metrics span every board common to both dirs. Per-board metrics
are printed and (optionally) written as JSON.

Note on the GT side: the GT polygons are derived from per-frame YOLO bboxes
projected and unioned by the same raster-union pipeline used for the
predictions. Mask IoU therefore tops out below 1.0 even for a perfect
detection — predictions follow knot shape while GT polygons are unions of
axis-aligned rectangles. The metric measures relative model quality given
that GT shape, not absolute knot-edge agreement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def polygon_bbox(poly: list[list[int]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def mask_iou(poly_a: list[list[int]], poly_b: list[list[int]]) -> float:
    """Rasterize both polygons in their union ROI and return mask IoU."""
    if len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0
    ax1, ay1, ax2, ay2 = polygon_bbox(poly_a)
    bx1, by1, bx2, by2 = polygon_bbox(poly_b)
    x0, y0 = min(ax1, bx1), min(ay1, by1)
    x1, y1 = max(ax2, bx2), max(ay2, by2)
    w, h = max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)
    pa = [(p[0] - x0, p[1] - y0) for p in poly_a]
    pb = [(p[0] - x0, p[1] - y0) for p in poly_b]
    img_a = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img_a).polygon(pa, fill=1)
    img_b = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img_b).polygon(pb, fill=1)
    arr_a = np.asarray(img_a, dtype=bool)
    arr_b = np.asarray(img_b, dtype=bool)
    inter = int(np.logical_and(arr_a, arr_b).sum())
    union = int(np.logical_or(arr_a, arr_b).sum())
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def greedy_match(preds: list[list[list[int]]],
                 gts: list[list[list[int]]],
                 threshold: float) -> list[tuple[int, int, float]]:
    """Greedy bbox-IoU matching. Returns [(pred_idx, gt_idx, bbox_iou)]."""
    pairs: list[tuple[float, int, int]] = []
    pred_bboxes = [polygon_bbox(p) for p in preds]
    gt_bboxes = [polygon_bbox(g) for g in gts]
    for i, pb in enumerate(pred_bboxes):
        for j, gb in enumerate(gt_bboxes):
            iou = bbox_iou(pb, gb)
            if iou >= threshold:
                pairs.append((iou, i, j))
    pairs.sort(key=lambda t: -t[0])
    matched_p: set[int] = set()
    matched_g: set[int] = set()
    out: list[tuple[int, int, float]] = []
    for iou, i, j in pairs:
        if i in matched_p or j in matched_g:
            continue
        matched_p.add(i); matched_g.add(j)
        out.append((i, j, iou))
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def safe_div(num: float, den: float) -> float | None:
    return (num / den) if den > 0 else None


def f1_from_pr(p: float | None, r: float | None) -> float | None:
    if p is None or r is None or (p + r) == 0:
        return None
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def select_boards(args: argparse.Namespace,
                  pred_dir: Path,
                  gt_dir: Path) -> list[int]:
    pred_ids = {int(p.stem) for p in pred_dir.glob("*.json") if p.stem.isdigit()}
    gt_ids = {int(p.stem) for p in gt_dir.glob("*.json") if p.stem.isdigit()}
    common = pred_ids & gt_ids
    only_pred = pred_ids - gt_ids
    only_gt = gt_ids - pred_ids
    if only_pred:
        print(f"warning: {len(only_pred)} board(s) in pred dir but not GT: "
              f"{sorted(only_pred)[:5]}{'...' if len(only_pred) > 5 else ''}")
    if only_gt:
        print(f"warning: {len(only_gt)} board(s) in GT dir but not pred: "
              f"{sorted(only_gt)[:5]}{'...' if len(only_gt) > 5 else ''}")

    if args.boards:
        wanted = {int(s.strip()) for s in args.boards.split(",") if s.strip()}
        common &= wanted
    if args.boards_file is not None:
        wanted: set[int] = set()
        for line in args.boards_file.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                try:
                    wanted.add(int(s))
                except ValueError:
                    pass
        common &= wanted
    return sorted(common)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boards-pred-dir", type=Path,
                    default=REPO_ROOT / "boards_out",
                    help="Per-board predictions (`knots stitch` output).")
    ap.add_argument("--boards-gt-dir", type=Path,
                    default=REPO_ROOT / "boards_gt",
                    help="Per-board GT polygons (`knots gt-stitch` output).")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "analysis" / "eval_boards.json",
                    help="JSON output with per-board + aggregate metrics.")
    ap.add_argument("--match-iou", type=float, default=0.5,
                    help="Bbox-IoU threshold for greedy matching (default 0.5).")
    ap.add_argument("--boards", type=str, default="",
                    help="Comma-separated board IDs to restrict to.")
    ap.add_argument("--boards-file", type=Path, default=None,
                    help="File with one board ID per line; '#' comments allowed.")
    ap.add_argument("--no-write", action="store_true",
                    help="Skip JSON output; print to stdout only.")
    args = ap.parse_args()

    if not args.boards_pred_dir.is_dir():
        raise SystemExit(f"missing pred dir: {args.boards_pred_dir}")
    if not args.boards_gt_dir.is_dir():
        raise SystemExit(f"missing GT dir: {args.boards_gt_dir}")

    boards = select_boards(args, args.boards_pred_dir, args.boards_gt_dir)
    if not boards:
        raise SystemExit("no boards to evaluate (empty intersection)")

    print(f"eval_boards: {len(boards)} board(s)  match-iou={args.match_iou}")
    print(f"  pred: {rel_to_root(args.boards_pred_dir)}/")
    print(f"  gt:   {rel_to_root(args.boards_gt_dir)}/")
    print()

    per_board: list[dict] = []
    for board in boards:
        pred = json.loads((args.boards_pred_dir / f"{board}.json").read_text())
        gt = json.loads((args.boards_gt_dir / f"{board}.json").read_text())
        preds = [k["polygon"] for k in pred.get("knots", [])]
        gts = [k["polygon"] for k in gt.get("knots", [])]

        matches = greedy_match(preds, gts, args.match_iou)
        tp = len(matches)
        fp = len(preds) - tp
        fn = len(gts) - tp
        ious = [mask_iou(preds[i], gts[j]) for (i, j, _) in matches]
        mean_iou = (sum(ious) / len(ious)) if ious else 0.0
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = f1_from_pr(precision, recall)

        per_board.append({
            "board": board,
            "pred_count": len(preds),
            "gt_count": len(gts),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": mean_iou,
            "matched_ious": [round(v, 4) for v in ious],
        })

    # Aggregate.
    total_pred = sum(b["pred_count"] for b in per_board)
    total_gt = sum(b["gt_count"] for b in per_board)
    total_tp = sum(b["tp"] for b in per_board)
    total_fp = sum(b["fp"] for b in per_board)
    total_fn = sum(b["fn"] for b in per_board)
    all_ious = [iou for b in per_board for iou in b["matched_ious"]]
    per_board_means = [b["mean_iou"] for b in per_board if b["tp"] > 0]
    agg_p = safe_div(total_tp, total_tp + total_fp)
    agg_r = safe_div(total_tp, total_tp + total_fn)
    aggregate = {
        "boards": len(per_board),
        "match_iou_threshold": args.match_iou,
        "total_pred": total_pred,
        "total_gt": total_gt,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": agg_p,
        "recall": agg_r,
        "f1": f1_from_pr(agg_p, agg_r),
        "mean_iou_micro": (sum(all_ious) / len(all_ious)) if all_ious else 0.0,
        "mean_iou_macro": (sum(per_board_means) / len(per_board_means))
                          if per_board_means else 0.0,
    }

    # Stdout summary.
    header = (f"{'board':>6}  {'pred':>4}  {'gt':>4}  {'tp':>4}  "
              f"{'fp':>4}  {'fn':>4}  {'P':>6}  {'R':>6}  {'F1':>6}  {'mIoU':>6}")
    print(header)
    print("-" * len(header))
    for b in per_board:
        def fmt(v: float | None) -> str:
            return f"{v:.3f}" if v is not None else "  n/a"
        print(f"{b['board']:>6}  {b['pred_count']:>4}  {b['gt_count']:>4}  "
              f"{b['tp']:>4}  {b['fp']:>4}  {b['fn']:>4}  "
              f"{fmt(b['precision']):>6}  {fmt(b['recall']):>6}  "
              f"{fmt(b['f1']):>6}  {b['mean_iou']:.3f}")

    print()
    print("Aggregate")
    print(f"  boards={aggregate['boards']}  match-iou={args.match_iou}")
    print(f"  total: pred={total_pred}  gt={total_gt}  tp={total_tp}  fp={total_fp}  fn={total_fn}")
    def fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "n/a"
    print(f"  P={fmt(aggregate['precision'])}  "
          f"R={fmt(aggregate['recall'])}  "
          f"F1={fmt(aggregate['f1'])}")
    print(f"  mean IoU micro={aggregate['mean_iou_micro']:.3f}  "
          f"macro={aggregate['mean_iou_macro']:.3f}")

    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump({"aggregate": aggregate, "per_board": per_board}, f, indent=2)
        print(f"\nwrote {rel_to_root(args.output)}")


if __name__ == "__main__":
    main()
