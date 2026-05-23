#!/usr/bin/env python3
"""Per-frame inspection of the wood-board dataset — manual debug tool.

Walks data/{images,labels}/ and prints one row per frame to stdout:

    board  frame  w  h  top  bot  dark%  annot  clust  edge

  top/bot     background-band height in px above/below the wood strip
              ('--' when no wood strip could be detected)
  dark%       fraction of in-strip non-annotation pixels that look like
              dark wood (luminance + chroma gated)
  annot       annotation count
  clust       number of connected groups of >=2 near-touching annotations
  edge        annotations within --edge-touch-tol-px of any frame edge

Use --board N to restrict to one board, --limit N to cap rows. Not wired
into run.sh: the pipeline (prepare → train → infer → eval) doesn't need
this. It exists for human eyeballing when a board misbehaves.

Math salvaged from the deleted analyze_dataset.py.

Run inside the knots-data image:

    docker run --rm -v "$PWD/data:/work/data:ro" knots-data \\
        python3 scripts/inspect_dataset.py --board 23
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAME_RE = re.compile(r"^(?P<board>\d+)_(?P<frame>\d+)\.png$")


def parse_yolo_bboxes(label_path: Path, w: int, h: int) -> list[tuple[float, float, float, float]]:
    """Pixel-space (x1, y1, x2, y2) from a YOLO bbox label file."""
    if not label_path.exists():
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx_n, cy_n, w_n, h_n = (
            float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        )
        cx, cy, bw, bh = cx_n * w, cy_n * h, w_n * w, h_n * h
        boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return boxes


def detect_strip(lum: np.ndarray, bg_max: int, row_frac: float) -> tuple[int, int]:
    """(top_band_px, bottom_band_px). (-1, -1) when no wood row was found."""
    bright_frac = (lum > bg_max).mean(axis=1)
    wood_row = bright_frac >= row_frac
    if not wood_row.any():
        return -1, -1
    idx = np.where(wood_row)[0]
    return int(idx.min()), lum.shape[0] - 1 - int(idx.max())


def dark_wood_frac(
    rgb: np.ndarray,
    lum: np.ndarray,
    strip_top: int,
    strip_bot: int,
    boxes: list[tuple[float, float, float, float]],
    bg_max: int,
    dark_wood_max: int,
    min_chroma: int,
) -> float:
    """In-strip, non-annotation pixels that look like dark wood, as a fraction.

    A pixel counts when its luminance is in (bg_max, dark_wood_max] AND its
    chroma (max(RGB) - min(RGB)) is >= min_chroma — chroma gating rules
    out neutral-grey background bleed-through and keeps brown stains.
    """
    if strip_top < 0 or strip_bot < strip_top:
        return 0.0
    h, w = lum.shape
    annot_mask = np.zeros((h, w), dtype=bool)
    for (x1, y1, x2, y2) in boxes:
        xi1 = max(0, int(np.floor(x1)))
        yi1 = max(0, int(np.floor(y1)))
        xi2 = min(w, int(np.ceil(x2)))
        yi2 = min(h, int(np.ceil(y2)))
        if xi2 > xi1 and yi2 > yi1:
            annot_mask[yi1:yi2, xi1:xi2] = True
    strip_mask = np.zeros_like(lum, dtype=bool)
    strip_mask[strip_top : strip_bot + 1, :] = True
    inspect = strip_mask & ~annot_mask
    denom = int(inspect.sum())
    if denom == 0:
        return 0.0
    chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    dark = inspect & (lum > bg_max) & (lum <= dark_wood_max) & (chroma >= min_chroma)
    return int(dark.sum()) / denom


def cluster_count(
    boxes: list[tuple[float, float, float, float]],
    near_T_min_px: float,
    near_T_rel: float,
) -> int:
    """Connected groups of >=2 near-touching annotations.

    Two boxes are 'near' when their axial gap is <=
    max(near_T_min_px, near_T_rel * min(min_dim_a, min_dim_b)). Identical
    threshold formula to prepare.py and to the deleted analyze_dataset.py.
    """
    n = len(boxes)
    if n < 2:
        return 0
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

    def axial_gap(a, b) -> float:
        gx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
        gy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
        return (gx * gx + gy * gy) ** 0.5

    for i in range(n):
        ai = boxes[i]
        mdi = min(ai[2] - ai[0], ai[3] - ai[1])
        for j in range(i + 1, n):
            aj = boxes[j]
            mdj = min(aj[2] - aj[0], aj[3] - aj[1])
            threshold = max(near_T_min_px, near_T_rel * min(mdi, mdj))
            if axial_gap(ai, aj) <= threshold:
                union(i, j)

    counts: dict[int, int] = defaultdict(int)
    for i in range(n):
        counts[find(i)] += 1
    return sum(1 for c in counts.values() if c >= 2)


def edge_touch_count(
    boxes: list[tuple[float, float, float, float]],
    w: int,
    h: int,
    tol: float,
) -> int:
    """Boxes within `tol` px of any frame edge."""
    n_touch = 0
    for (x1, y1, x2, y2) in boxes:
        if x1 <= tol or (w - x2) <= tol or y1 <= tol or (h - y2) <= tol:
            n_touch += 1
    return n_touch


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data",
                   help="Read-only input dir containing images/ and labels/.")
    p.add_argument("--board", type=int, default=None,
                   help="Restrict to one board. Default: every board.")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N frames (0 = no limit).")
    p.add_argument("--bg-max", type=int, default=30,
                   help="Pixels with luminance <= this count as background.")
    p.add_argument("--row-bright-frac", type=float, default=0.5,
                   help="Row is 'wood' if >= this fraction of its pixels > bg-max.")
    p.add_argument("--dark-wood-max", type=int, default=120,
                   help="Upper luminance bound for a dark-wood pixel.")
    p.add_argument("--min-chroma", type=int, default=15,
                   help="Min max(RGB)-min(RGB) for a dark pixel to count as dark wood.")
    p.add_argument("--near-T-min-px", type=float, default=5.0,
                   help="Floor (px) of the hybrid near-touching threshold.")
    p.add_argument("--near-T-rel", type=float, default=0.25,
                   help="Multiplier of min(min_dim_a, min_dim_b) for the threshold.")
    p.add_argument("--edge-touch-tol-px", type=float, default=1.0,
                   help="Box touches an edge if its distance to that edge is <= this.")
    args = p.parse_args()

    images_dir = args.data_dir / "images"
    labels_dir = args.data_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"expected {images_dir} and {labels_dir} to exist")

    frame_paths: list[tuple[int, int, Path]] = []
    for entry in images_dir.iterdir():
        m = FRAME_RE.match(entry.name)
        if not m:
            continue
        board = int(m["board"])
        if args.board is not None and board != args.board:
            continue
        frame_paths.append((board, int(m["frame"]), entry))
    frame_paths.sort()
    if args.limit > 0:
        frame_paths = frame_paths[: args.limit]
    if not frame_paths:
        raise SystemExit("no frames matched")

    header = (
        f"{'board':>5}  {'frame':>5}  {'w':>4}  {'h':>4}  "
        f"{'top':>4}  {'bot':>4}  {'dark':>6}  "
        f"{'annot':>5}  {'clust':>5}  {'edge':>4}"
    )
    print(header)
    print("-" * len(header))

    prev_board: int | None = None
    for board, frame, img_path in frame_paths:
        with Image.open(img_path) as img:
            rgb = np.asarray(img.convert("RGB"))
            lum = np.asarray(img.convert("L"))
        h, w = lum.shape

        top, bot = detect_strip(lum, args.bg_max, args.row_bright_frac)
        strip_top = top
        strip_bot = (h - 1 - bot) if top >= 0 else -1

        boxes = parse_yolo_bboxes(labels_dir / f"{board}_{frame}.txt", w, h)
        dark = dark_wood_frac(
            rgb, lum, strip_top, strip_bot, boxes,
            args.bg_max, args.dark_wood_max, args.min_chroma,
        )
        n_clust = cluster_count(boxes, args.near_T_min_px, args.near_T_rel)
        n_edge = edge_touch_count(boxes, w, h, args.edge_touch_tol_px)

        if prev_board is not None and board != prev_board:
            print()
        prev_board = board

        top_s = str(top) if top >= 0 else "--"
        bot_s = str(bot) if bot >= 0 else "--"
        print(
            f"{board:>5}  {frame:>5}  {w:>4}  {h:>4}  "
            f"{top_s:>4}  {bot_s:>4}  {dark * 100:>5.1f}%  "
            f"{len(boxes):>5}  {n_clust:>5}  {n_edge:>4}"
        )


if __name__ == "__main__":
    main()
