#!/usr/bin/env python3
"""Convert YOLO bbox annotations into polygon (YOLO-seg) annotations by
prompting SAM2 with the bbox plus its centre point.

Inputs (read-only):
    --data-dir/images/{board}_{frame}.png
    --data-dir/labels/{board}_{frame}.txt   (YOLO bbox format)

Output:
    --output-dir/{board}_{frame}.txt        (YOLO-seg polygon format)

Each output line is one knot:
    0 x1/W y1/H x2/W y2/H ... xN/W yN/H
(class always 0; coords normalised to image size).

SAM2 is prompted per bbox with `bboxes=[box], points=[centre], labels=[1]`.
The centre-point prompt anchors SAM2 to "the object is here", reducing two
failure modes: oversized polygons when the bbox is loose, and zero-mask
returns on low-contrast knots.

When SAM2 returns no usable mask, the original bbox is emitted as a 4-vertex
rectangle (`bbox_to_polygon` fallback) so no annotation is lost. The count
of such fallbacks is reported at the end — watch this number; it's the
quality signal for whether SAM2 is actually helping (see project notes).

Runs are resumable: frames whose output already exists are skipped unless
--force is set. Empty-bbox frames touch an empty output file so resume
logic is consistent.

Requires CUDA. Run in the `knots-train` Docker image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_yolo_bboxes(
    label_path: Path, w: int, h: int
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Return (boxes, raw_line_count). Boxes are pixel-space (x1,y1,x2,y2),
    clipped to image and with any degenerate (zero w/h after rounding)
    entries silently dropped. The line-count delta lets the caller report
    how many degenerates were skipped."""
    if not label_path.exists():
        return [], 0
    raw_lines = [ln for ln in label_path.read_text().splitlines() if ln.strip()]
    boxes: list[tuple[int, int, int, int]] = []
    for line in raw_lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cx_n, cy_n, w_n, h_n = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        cx, cy = cx_n * w, cy_n * h
        bw, bh = w_n * w, h_n * h
        x1 = max(0, int(round(cx - bw / 2)))
        y1 = max(0, int(round(cy - bh / 2)))
        x2 = min(w, int(round(cx + bw / 2)))
        y2 = min(h, int(round(cy + bh / 2)))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes, len(raw_lines)


def mask_to_polygon(mask: np.ndarray, eps_px: float = 1.0) -> list[tuple[int, int]] | None:
    """Largest external contour, simplified with approxPolyDP."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 4:
        return None
    simplified = cv2.approxPolyDP(biggest, eps_px, closed=True)
    if len(simplified) < 3:
        return None
    return [(int(p[0][0]), int(p[0][1])) for p in simplified]


def polygon_to_yolo_seg(poly: list[tuple[int, int]], w: int, h: int, cls: int = 0) -> str:
    parts = [str(cls)]
    for x, y in poly:
        parts.append(f"{x / w:.6f}")
        parts.append(f"{y / h:.6f}")
    return " ".join(parts)


def clip_mask_to_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Zero out pixels outside the bbox. Defensive: if SAM segments wood
    grain past the bbox edge, we still trust the bbox as ground truth."""
    x1, y1, x2, y2 = bbox
    h, w = mask.shape
    clipped = np.zeros_like(mask)
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c > x1c and y2c > y1c:
        clipped[y1c:y2c, x1c:x2c] = mask[y1c:y2c, x1c:x2c]
    return clipped


def bbox_to_polygon(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    """Fallback when SAM returns an empty mask: emit the bbox as a 4-vertex
    rectangle polygon. Noisy is better than missing."""
    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def run_sam_on_frame(
    model, image_bgr: np.ndarray, bboxes: list[tuple[int, int, int, int]]
) -> list[np.ndarray]:
    """Return one boolean mask per bbox. Per-bbox calls keep the
    (bbox, point) prompt pairing unambiguous; the per-frame overhead is
    small compared to SAM inference itself."""
    masks: list[np.ndarray] = []
    h, w = image_bgr.shape[:2]
    for x1, y1, x2, y2 in bboxes:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        results = model.predict(
            image_bgr,
            bboxes=[[x1, y1, x2, y2]],
            points=[[cx, cy]],
            labels=[1],
            verbose=False,
        )
        r = results[0]
        if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
            masks.append(np.zeros((h, w), dtype=bool))
            continue
        m = r.masks.data[0].cpu().numpy() > 0.5
        masks.append(m)
    return masks


def load_sam(model_name: str, device: str):
    """Late import so --help works without ultralytics/CUDA installed."""
    from ultralytics import SAM

    print(f"loading SAM '{model_name}' on {device} ...", file=sys.stderr)
    model = SAM(model_name)
    model.to(device)
    return model


def list_frames(images_dir: Path, board_filter: set[int] | None) -> list[Path]:
    """Sorted PNG paths, optionally filtered to a set of board IDs."""
    paths = sorted(images_dir.glob("*.png"))
    if board_filter is None:
        return paths
    out: list[Path] = []
    for p in paths:
        stem = p.stem  # "{board}_{frame}"
        head = stem.split("_", 1)[0]
        if head.isdigit() and int(head) in board_filter:
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Read-only input dir containing images/ and labels/.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "labels_seg",
        help="Output dir for YOLO-seg polygon labels.",
    )
    ap.add_argument(
        "--model", default="sam2.1_s.pt", help="Ultralytics SAM checkpoint name or local path."
    )
    ap.add_argument("--device", default="cuda:0", help="Torch device.")
    ap.add_argument(
        "--force", action="store_true", help="Re-run frames whose output label already exists."
    )
    ap.add_argument(
        "--boards",
        type=str,
        default="",
        help="Comma-separated board IDs to restrict to (default: all).",
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="Process at most N frames (0 = all). Smoke test knob."
    )
    args = ap.parse_args()

    images_dir = args.data_dir / "images"
    labels_dir = args.data_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"Expected {images_dir} and {labels_dir} to exist.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    board_filter = (
        {int(x) for x in args.boards.split(",") if x.strip()} if args.boards.strip() else None
    )

    frames = list_frames(images_dir, board_filter)
    if args.limit > 0:
        frames = frames[: args.limit]

    boards_str = "all" if board_filter is None else ",".join(str(b) for b in sorted(board_filter))
    print(
        f"sam_polygons: {len(frames)} frames from {rel_to_root(images_dir)}/  -> {rel_to_root(args.output_dir)}/"
    )
    print(f"  model={args.model}  device={args.device}  force={args.force}  boards={boards_str}")

    model = load_sam(args.model, args.device)

    n_processed = 0
    n_skipped = 0
    n_empty = 0
    n_degenerate_skipped = 0
    n_bboxes_total = 0
    n_fallback_polygons = 0
    heartbeat_every = max(50, len(frames) // 20)  # ~20 lines of progress

    for i, img_path in enumerate(frames, start=1):
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        out_label = args.output_dir / f"{stem}.txt"

        if out_label.exists() and not args.force:
            n_skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARN unreadable: {img_path.name}", file=sys.stderr)
            continue
        h, w = img.shape[:2]

        bboxes, n_raw = parse_yolo_bboxes(label_path, w, h)
        n_degenerate_skipped += n_raw - len(bboxes)

        if not bboxes:
            out_label.write_text("")
            n_empty += 1
            continue

        n_bboxes_total += len(bboxes)
        masks = run_sam_on_frame(model, img, bboxes)
        lines: list[str] = []
        for mask, bbox in zip(masks, bboxes):
            poly = mask_to_polygon(clip_mask_to_bbox(mask, bbox))
            if poly is None:
                poly = bbox_to_polygon(bbox)
                n_fallback_polygons += 1
            lines.append(polygon_to_yolo_seg(poly, w, h))
        out_label.write_text("\n".join(lines) + "\n")
        n_processed += 1

        if i % heartbeat_every == 0 or i == len(frames):
            print(f"  ... {i}/{len(frames)} frames", file=sys.stderr)

    fallback_pct = (100.0 * n_fallback_polygons / n_bboxes_total) if n_bboxes_total else 0.0
    print()
    print("Result")
    print(f"  processed={n_processed}  skipped={n_skipped}  empty={n_empty}")
    print(f"  degenerate_bboxes_skipped={n_degenerate_skipped}")
    print(f"  fallback_polygons={n_fallback_polygons} / {n_bboxes_total} ({fallback_pct:.1f}%)")
    print(f"  output dir: {rel_to_root(args.output_dir)}/")


if __name__ == "__main__":
    main()
