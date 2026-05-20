#!/usr/bin/env python3
"""Per-frame analysis of the wood-board dataset.

For every frame in data/images/ this script computes:

  * Frame geometry: width, height, black background bands above and below
    the wood strip (detected via row-brightness fraction).
  * Dark-wood quantification inside the strip but outside any annotation
    bbox, gated by both luminance and chroma (so brown stains count but
    background leakage does not).
  * Per-annotation features: aspect ratio, size bucket, full edge-distance
    set, edge-touch flags and touch span.
  * Clustering of annotations within a frame via axial-gap union-find
    with a hybrid (absolute + relative) near-touching threshold.

Outputs (written to --output-dir, default analysis/):
  frames.csv / frames.json            one row per frame
  annotations.csv / annotations.json  one row per annotation

Stdout is intentionally terse: a config banner, file counts, and a flag
block listing anomalies (non-640 widths, mixed heights inside a board,
frames where the wood strip could not be detected).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
from PIL import Image
from scipy.ndimage import label as ndi_label


def rel_to_root(path: Path) -> str:
    """Render path relative to the repo root for stdout; fall back to absolute."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


FRAME_RE = re.compile(r"^(?P<board>\d+)_(?P<frame>\d+)\.png$")
EDGES = ("left", "right", "top", "bottom")
BUCKET_NAMES = ("tiny", "small", "medium", "large")
CONNECTIVITY_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Read-only input dir containing images/ and labels/.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "analysis",
        help="Output dir for frames/annotations CSV+JSON.",
    )
    p.add_argument(
        "--bg-max", type=int, default=30, help="Pixels with luminance <= this count as background."
    )
    p.add_argument(
        "--row-bright-frac",
        type=float,
        default=0.5,
        help="Row is 'wood' if >= this fraction of its pixels > bg-max.",
    )
    p.add_argument(
        "--dark-wood-max",
        type=int,
        default=120,
        help="Upper luminance bound for a dark-wood pixel.",
    )
    p.add_argument(
        "--min-chroma",
        type=int,
        default=15,
        help="Min max(RGB) - min(RGB) for a dark pixel to count as dark wood.",
    )
    p.add_argument(
        "--bucket-thresholds",
        type=str,
        default="0.25,1,4",
        help="Three comma-separated %% cutoffs (tiny|small|medium|large).",
    )
    p.add_argument(
        "--near-T-min-px",
        type=float,
        default=5.0,
        help="Floor (px) of the hybrid near-touching threshold.",
    )
    p.add_argument(
        "--near-T-rel",
        type=float,
        default=0.25,
        help="Multiplier of min(min_dim_a, min_dim_b) for the hybrid threshold.",
    )
    p.add_argument(
        "--edge-touch-tol-px",
        type=float,
        default=1.0,
        help="Box is touching an edge if its distance to that edge is <= this.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most this many frames (0 = all). Useful for smoke tests.",
    )
    return p.parse_args()


def parse_bucket_thresholds(spec: str) -> list[float]:
    parts = [float(x.strip()) for x in spec.split(",")]
    if len(parts) != 3:
        raise SystemExit("--bucket-thresholds must be exactly 3 values, e.g. '0.25,1,4'")
    if any(parts[i] >= parts[i + 1] for i in range(len(parts) - 1)):
        raise SystemExit("--bucket-thresholds must be strictly increasing")
    return [p / 100.0 for p in parts]


def list_frames(images_dir: Path) -> list[tuple[int, int, Path]]:
    out: list[tuple[int, int, Path]] = []
    for entry in images_dir.iterdir():
        m = FRAME_RE.match(entry.name)
        if m:
            out.append((int(m["board"]), int(m["frame"]), entry))
    out.sort()
    return out


def load_annotations(label_path: Path, width: int, height: int) -> list[dict]:
    """Parse one YOLO label file into dicts with pixel-space bboxes."""
    annots: list[dict] = []
    if not label_path.exists():
        return annots
    with label_path.open() as fh:
        for idx, line in enumerate(fh):
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w_n, h_n = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            xc = cx * width
            yc = cy * height
            bw = w_n * width
            bh = h_n * height
            annots.append(
                {
                    "annot_index": idx,
                    "class": cls,
                    "cx": cx,
                    "cy": cy,
                    "w": w_n,
                    "h": h_n,
                    "x1": xc - bw / 2,
                    "y1": yc - bh / 2,
                    "x2": xc + bw / 2,
                    "y2": yc + bh / 2,
                    "w_px": bw,
                    "h_px": bh,
                }
            )
    return annots


def detect_strip(lum: np.ndarray, bg_max: int, row_frac: float) -> tuple[int, int]:
    """Return (top_band_px, bottom_band_px). (-1, -1) signals detection failure."""
    height, _ = lum.shape
    bright_frac = (lum > bg_max).mean(axis=1)
    wood_row = bright_frac >= row_frac
    if not wood_row.any():
        return -1, -1
    idx = np.where(wood_row)[0]
    top = int(idx.min())
    bottom_band = height - 1 - int(idx.max())
    return top, bottom_band


def size_bucket(area_frac: float, thresholds: list[float]) -> str:
    for name, cutoff in zip(BUCKET_NAMES[:-1], thresholds):
        if area_frac < cutoff:
            return name
    return BUCKET_NAMES[-1]


def edge_features(a: dict, width: int, height: int, tol: float) -> dict:
    dists = {
        "left": a["x1"],
        "right": width - a["x2"],
        "top": a["y1"],
        "bottom": height - a["y2"],
    }
    nearest_edge, min_dist = min(dists.items(), key=lambda kv: kv[1])
    touches = {edge: (d <= tol) for edge, d in dists.items()}
    spans: list[float] = []
    if touches["left"] or touches["right"]:
        spans.append(a["y2"] - a["y1"])
    if touches["top"] or touches["bottom"]:
        spans.append(a["x2"] - a["x1"])
    touch_span = max(spans) if spans else 0.0
    return {
        "dist_left_px": float(dists["left"]),
        "dist_right_px": float(dists["right"]),
        "dist_top_px": float(dists["top"]),
        "dist_bottom_px": float(dists["bottom"]),
        "min_edge_dist_px": float(min_dist),
        "nearest_edge": nearest_edge,
        "touches_left_edge": bool(touches["left"]),
        "touches_right_edge": bool(touches["right"]),
        "touches_top_edge": bool(touches["top"]),
        "touches_bottom_edge": bool(touches["bottom"]),
        "touch_span_px": float(touch_span),
    }


def axial_gap(a: dict, b: dict) -> float:
    gx = max(0.0, max(a["x1"], b["x1"]) - min(a["x2"], b["x2"]))
    gy = max(0.0, max(a["y1"], b["y1"]) - min(a["y2"], b["y2"]))
    return float(np.hypot(gx, gy))


def cluster_metrics(annots: list[dict], near_T_min_px: float, near_T_rel: float) -> dict:
    n = len(annots)
    if n == 0:
        return {
            "touching_pairs": 0,
            "near_pairs": 0,
            "cluster_count": 0,
            "largest_cluster_size": 0,
            "singleton_count": 0,
            "cluster_sizes": [],
        }
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    touching = 0
    near = 0
    for i in range(n):
        ai = annots[i]
        min_dim_a = min(ai["w_px"], ai["h_px"])
        for j in range(i + 1, n):
            aj = annots[j]
            gap = axial_gap(ai, aj)
            min_dim = min(min_dim_a, aj["w_px"], aj["h_px"])
            threshold = max(near_T_min_px, near_T_rel * min_dim)
            if gap == 0.0:
                touching += 1
            if gap <= threshold:
                near += 1
                union(i, j)

    counts: dict[int, int] = {}
    for i in range(n):
        r = find(i)
        counts[r] = counts.get(r, 0) + 1
    sizes = sorted(counts.values(), reverse=True)
    return {
        "touching_pairs": touching,
        "near_pairs": near,
        "cluster_count": sum(1 for s in sizes if s >= 2),
        "largest_cluster_size": sizes[0] if sizes else 0,
        "singleton_count": sum(1 for s in sizes if s == 1),
        "cluster_sizes": sizes,
    }


def annotation_pixel_mask(annots: list[dict], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for a in annots:
        x1 = max(0, int(np.floor(a["x1"])))
        y1 = max(0, int(np.floor(a["y1"])))
        x2 = min(width, int(np.ceil(a["x2"])))
        y2 = min(height, int(np.ceil(a["y2"])))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask


def contrast_stats(
    lab_l: np.ndarray,
    strip_top: int,
    strip_bot: int,
    annot_mask: np.ndarray,
    annots: list[dict],
    width: int,
    height: int,
) -> tuple[float | None, list[tuple[float | None, float | None]]]:
    """LAB-L contrast between each knot and the surrounding (unannotated) wood.

    Returns (wood_l_mean, [(knot_l_mean, knot - wood) for each annotation]).

    Both elements may be None when the wood strip wasn't detected or contains
    no wood pixels after annotations are masked out. Contrast is signed: a
    dark knot on light wood reads negative; a light knot on dark wood reads
    positive. PIL's LAB-L is uint8 in [0, 255], not the standard [0, 100] —
    fine for *relative* comparisons but don't reuse this scale outside.
    """
    n = len(annots)
    if strip_top < 0 or strip_bot < strip_top:
        return None, [(None, None)] * n
    strip_mask = np.zeros_like(lab_l, dtype=bool)
    strip_mask[strip_top : strip_bot + 1, :] = True
    wood_mask = strip_mask & ~annot_mask
    if not wood_mask.any():
        return None, [(None, None)] * n
    wood_l_mean = float(lab_l[wood_mask].mean())

    per_annot: list[tuple[float | None, float | None]] = []
    for a in annots:
        x1 = max(0, int(np.floor(a["x1"])))
        y1 = max(0, int(np.floor(a["y1"])))
        x2 = min(width, int(np.ceil(a["x2"])))
        y2 = min(height, int(np.ceil(a["y2"])))
        if x2 > x1 and y2 > y1:
            knot_l = float(lab_l[y1:y2, x1:x2].mean())
            per_annot.append((knot_l, knot_l - wood_l_mean))
        else:
            per_annot.append((None, None))
    return wood_l_mean, per_annot


def dark_wood_stats(
    rgb: np.ndarray,
    lum: np.ndarray,
    strip_top: int,
    strip_bot: int,
    annot_mask: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    height, width = lum.shape
    if strip_top < 0 or strip_bot < strip_top:
        return {
            "dark_wood_px": 0,
            "dark_wood_frac": 0.0,
            "dark_blob_count": 0,
            "largest_dark_blob_px": 0,
            "dark_blob_sizes": [],
        }
    strip_mask = np.zeros((height, width), dtype=bool)
    strip_mask[strip_top : strip_bot + 1, :] = True
    inspect = strip_mask & ~annot_mask
    chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    dark_mask = (
        inspect & (lum > args.bg_max) & (lum <= args.dark_wood_max) & (chroma >= args.min_chroma)
    )
    dark_px = int(dark_mask.sum())
    denom = int(inspect.sum())
    dark_frac = dark_px / denom if denom > 0 else 0.0

    labeled, n_blobs = ndi_label(dark_mask, structure=CONNECTIVITY_4)
    if n_blobs == 0:
        sizes: list[int] = []
    else:
        counts = np.bincount(labeled.ravel())
        sizes = sorted((int(c) for c in counts[1:]), reverse=True)
    return {
        "dark_wood_px": dark_px,
        "dark_wood_frac": float(dark_frac),
        "dark_blob_count": len(sizes),
        "largest_dark_blob_px": sizes[0] if sizes else 0,
        "dark_blob_sizes": sizes,
    }


def build_annotation_rows(
    board: int,
    frame: int,
    annots: list[dict],
    width: int,
    height: int,
    bucket_thresholds: list[float],
    edge_tol: float,
    per_annot_contrast: list[tuple[float | None, float | None]],
) -> tuple[list[dict], int]:
    rows: list[dict] = []
    edge_touching = 0
    frame_area = float(width * height)
    for a, (knot_l, contrast) in zip(annots, per_annot_contrast):
        area_px = a["w_px"] * a["h_px"]
        area_frac = area_px / frame_area if frame_area > 0 else 0.0
        aspect = (a["w"] / a["h"]) if a["h"] > 0 else 0.0
        bucket = size_bucket(area_frac, bucket_thresholds)
        edge = edge_features(a, width, height, edge_tol)
        if any(edge[f"touches_{e}_edge"] for e in EDGES):
            edge_touching += 1
        rows.append(
            {
                "board": board,
                "frame": frame,
                "annot_index": a["annot_index"],
                "class": a["class"],
                "cx": a["cx"],
                "cy": a["cy"],
                "w": a["w"],
                "h": a["h"],
                "w_px": float(a["w_px"]),
                "h_px": float(a["h_px"]),
                "area_px": float(area_px),
                "area_frac": float(area_frac),
                "aspect_ratio": float(aspect),
                "size_bucket": bucket,
                "knot_l_mean": knot_l,
                "knot_wood_contrast": contrast,
                **edge,
            }
        )
    return rows, edge_touching


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        for r in rows:
            out = []
            for c in cols:
                v = r.get(c)
                if isinstance(v, list):
                    out.append(json.dumps(v))
                elif isinstance(v, bool):
                    out.append(int(v))
                elif isinstance(v, float):
                    out.append(f"{v:.6g}")
                else:
                    out.append(v)
            writer.writerow(out)


def write_json(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        json.dump(rows, fh, indent=2)


def main() -> None:
    args = parse_args()
    bucket_thresholds = parse_bucket_thresholds(args.bucket_thresholds)

    images_dir = args.data_dir / "images"
    labels_dir = args.data_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"Expected {images_dir} and {labels_dir} to exist.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = list_frames(images_dir)
    if args.limit > 0:
        frame_paths = frame_paths[: args.limit]

    print(f"analyze_dataset: {len(frame_paths)} frames from {rel_to_root(images_dir)}/")
    print(
        f"  bg_max={args.bg_max}  row_bright_frac={args.row_bright_frac}  "
        f"dark_wood_max={args.dark_wood_max}  min_chroma={args.min_chroma}"
    )
    print(
        f"  buckets%={args.bucket_thresholds}  "
        f"near_T=max({args.near_T_min_px}px, {args.near_T_rel}*min_dim)  "
        f"edge_tol={args.edge_touch_tol_px}px"
    )
    print(f"  output={rel_to_root(args.output_dir)}/")

    frames_out: list[dict] = []
    annots_out: list[dict] = []
    height_by_board: dict[int, int] = {}
    width_mismatch: list[tuple[int, int, int]] = []
    height_mismatch: list[tuple[int, int, int, int]] = []
    strip_fail: list[tuple[int, int]] = []

    for board, frame, img_path in frame_paths:
        with Image.open(img_path) as img:
            rgb = np.asarray(img.convert("RGB"))
            lum = np.asarray(img.convert("L"))
            lab_l = np.asarray(img.convert("LAB"))[:, :, 0]
        height, width = lum.shape

        if board not in height_by_board:
            height_by_board[board] = height
        elif height_by_board[board] != height:
            height_mismatch.append((board, frame, height_by_board[board], height))

        if width != 640:
            width_mismatch.append((board, frame, width))

        top_band, bot_band = detect_strip(lum, args.bg_max, args.row_bright_frac)
        strip_detected = top_band >= 0
        if strip_detected:
            strip_top = top_band
            strip_bot = height - 1 - bot_band
            strip_height = strip_bot - strip_top + 1
        else:
            strip_fail.append((board, frame))
            strip_top = -1
            strip_bot = -1
            strip_height = 0

        label_path = labels_dir / f"{board}_{frame}.txt"
        annots = load_annotations(label_path, width, height)
        annot_mask = annotation_pixel_mask(annots, width, height)

        dark = dark_wood_stats(rgb, lum, strip_top, strip_bot, annot_mask, args)
        wood_l_mean, per_annot_contrast = contrast_stats(
            lab_l,
            strip_top,
            strip_bot,
            annot_mask,
            annots,
            width,
            height,
        )
        clust = cluster_metrics(annots, args.near_T_min_px, args.near_T_rel)
        ann_rows, edge_touching_count = build_annotation_rows(
            board,
            frame,
            annots,
            width,
            height,
            bucket_thresholds,
            args.edge_touch_tol_px,
            per_annot_contrast,
        )
        annots_out.extend(ann_rows)

        valid_contrasts = [c for (_k, c) in per_annot_contrast if c is not None]
        min_abs_contrast = min(valid_contrasts, key=abs) if valid_contrasts else None

        frames_out.append(
            {
                "board": board,
                "frame": frame,
                "width": width,
                "height": height,
                "top_band_px": top_band if strip_detected else -1,
                "bottom_band_px": bot_band if strip_detected else -1,
                "top_band_frac": (top_band / height) if strip_detected and height > 0 else -1.0,
                "bottom_band_frac": (bot_band / height) if strip_detected and height > 0 else -1.0,
                "strip_height_px": strip_height,
                "annot_count": len(annots),
                **dark,
                "wood_l_mean": wood_l_mean,
                "min_abs_knot_wood_contrast": min_abs_contrast,
                **clust,
                "edge_touching_annot_count": edge_touching_count,
            }
        )

    out_dir = args.output_dir
    write_csv(out_dir / "frames.csv", frames_out)
    write_json(out_dir / "frames.json", frames_out)
    write_csv(out_dir / "annotations.csv", annots_out)
    write_json(out_dir / "annotations.json", annots_out)

    print()
    print(f"frames written:      {len(frames_out)} -> frames.csv, frames.json")
    print(f"annotations written: {len(annots_out)} -> annotations.csv, annotations.json")

    print()
    print("Anomalies")
    print(f"  width != 640:              {len(width_mismatch)}")
    for b, f, w in width_mismatch[:5]:
        print(f"    board {b} frame {f}: width={w}")
    if len(width_mismatch) > 5:
        print(f"    ... +{len(width_mismatch) - 5} more")

    boards_with_mixed_h = sorted({b for b, _, _, _ in height_mismatch})
    print(f"  boards with mixed heights: {len(boards_with_mixed_h)}")
    for b, f, h0, h1 in height_mismatch[:5]:
        print(f"    board {b} frame {f}: expected H={h0}, got H={h1}")
    if len(height_mismatch) > 5:
        print(f"    ... +{len(height_mismatch) - 5} more")

    print(f"  frames with no wood strip: {len(strip_fail)}")
    for b, f in strip_fail[:5]:
        print(f"    board {b} frame {f}")
    if len(strip_fail) > 5:
        print(f"    ... +{len(strip_fail) - 5} more")


if __name__ == "__main__":
    main()
