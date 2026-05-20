#!/usr/bin/env python3
"""Render side-by-side comparison composites for frames.

For each requested frame, builds a composite image with the requested panels
stacked vertically (default) or horizontally. Built-in panels:
    raw          original image, no overlay
    bbox         image + YOLO bbox rectangles
    polygon      image + YOLO-seg polygons (translucent fill + outline)
    predictions  image + polygons from C++-inference JSON output

Adding a new panel type is one function + one PANELS-dict entry.

Inputs (all read-only):
    --data-dir/images/{frame_id}.png
    --bbox-labels-dir/{frame_id}.txt    (YOLO bbox: cls cx cy w h, normalised)
    --seg-labels-dir/{frame_id}.txt     (YOLO-seg: cls x1 y1 x2 y2 ... xN yN)
    --predictions-dir/{frame_id}.json   (cpp/knots output)

Output:
    --output-dir/{frame_id}.png

Frame IDs are {board}_{frame}, e.g. 0_5. Exactly one of --frame, --frames,
--frames-file is required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_hex_color(s: str) -> tuple[int, int, int]:
    raw = s.strip().lstrip("#")
    if len(raw) != 6:
        raise argparse.ArgumentTypeError(f"expected #rrggbb, got {s!r}")
    return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))


def parse_yolo_bboxes(label_path: Path, w: int, h: int) -> list[tuple[int, int, int, int]]:
    if not label_path.exists():
        return []
    boxes: list[tuple[int, int, int, int]] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx = float(parts[1]) * w
        cy = float(parts[2]) * h
        bw = float(parts[3]) * w
        bh = float(parts[4]) * h
        x1 = max(0, int(round(cx - bw / 2)))
        y1 = max(0, int(round(cy - bh / 2)))
        x2 = min(w, int(round(cx + bw / 2)))
        y2 = min(h, int(round(cy + bh / 2)))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def parse_predictions_json(path: Path) -> list[list[tuple[int, int]]]:
    """Read C++/knots JSON output, return polygons in source-image px coords.

    The JSON schema is {"detections": [{"polygon": [[x, y], ...]}, ...]}.
    Coordinates are already in source-image pixel space (the C++ binary
    inverse-letterboxes before writing), so no resize is needed.
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    polygons: list[list[tuple[int, int]]] = []
    for det in data.get("detections", []):
        pts = det.get("polygon", [])
        poly = [(int(round(x)), int(round(y))) for x, y in pts]
        if len(poly) >= 3:
            polygons.append(poly)
    return polygons


def parse_yolo_seg(label_path: Path, w: int, h: int) -> list[list[tuple[int, int]]]:
    if not label_path.exists():
        return []
    polygons: list[list[tuple[int, int]]] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:  # class + at least 3 (x,y) pairs
            continue
        coords = parts[1:]
        if len(coords) % 2 != 0:
            continue
        pts: list[tuple[int, int]] = []
        for i in range(0, len(coords), 2):
            x = int(round(float(coords[i]) * w))
            y = int(round(float(coords[i + 1]) * h))
            pts.append((x, y))
        if len(pts) >= 3:
            polygons.append(pts)
    return polygons


def render_raw(image: Image.Image, **_kwargs) -> Image.Image:
    return image.copy()


def render_bbox(
    image: Image.Image,
    *,
    bboxes: list[tuple[int, int, int, int]],
    bbox_color: tuple[int, int, int],
    **_kwargs,
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for x1, y1, x2, y2 in bboxes:
        draw.rectangle([(x1, y1), (x2 - 1, y2 - 1)], outline=bbox_color, width=2)
    return out


def render_polygon(
    image: Image.Image,
    *,
    polygons: list[list[tuple[int, int]]],
    polygon_color: tuple[int, int, int],
    polygon_alpha: float,
    **_kwargs,
) -> Image.Image:
    if not polygons:
        return image.copy()
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    fill_alpha = max(0, min(255, int(round(255 * polygon_alpha))))
    fill = polygon_color + (fill_alpha,)
    outline = polygon_color + (255,)
    o_draw = ImageDraw.Draw(overlay)
    for poly in polygons:
        if len(poly) >= 3:
            o_draw.polygon(poly, fill=fill, outline=outline)
    composed = Image.alpha_composite(base, overlay).convert("RGB")
    return composed


def render_predictions(
    image: Image.Image,
    *,
    predictions: list[list[tuple[int, int]]],
    prediction_color: tuple[int, int, int],
    prediction_alpha: float,
    **_kwargs,
) -> Image.Image:
    # Same draw path as render_polygon but a separate function so the color /
    # alpha knobs and the source field are visibly distinct in stack traces
    # and in --help. Source is JSON instead of YOLO-seg .txt.
    if not predictions:
        return image.copy()
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    fill_alpha = max(0, min(255, int(round(255 * prediction_alpha))))
    fill = prediction_color + (fill_alpha,)
    outline = prediction_color + (255,)
    o_draw = ImageDraw.Draw(overlay)
    for poly in predictions:
        if len(poly) >= 3:
            o_draw.polygon(poly, fill=fill, outline=outline)
    composed = Image.alpha_composite(base, overlay).convert("RGB")
    return composed


PANELS = {
    "raw": render_raw,
    "bbox": render_bbox,
    "polygon": render_polygon,
    "predictions": render_predictions,
}


def add_label(panel: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    if font is not None:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        tw, th = 6 * len(text), 11
    pad = 3
    box_w, box_h = tw + 2 * pad, th + 2 * pad
    draw.rectangle([(0, 0), (box_w, box_h)], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return panel


def compose(panels: list[Image.Image], direction: str, sep_px: int = 2) -> Image.Image:
    if direction == "vertical":
        width = max(p.width for p in panels)
        total_h = sum(p.height for p in panels) + sep_px * (len(panels) - 1)
        canvas = Image.new("RGB", (width, total_h), (255, 255, 255))
        y = 0
        for p in panels:
            canvas.paste(p, (0, y))
            y += p.height + sep_px
        return canvas
    # horizontal
    height = max(p.height for p in panels)
    total_w = sum(p.width for p in panels) + sep_px * (len(panels) - 1)
    canvas = Image.new("RGB", (total_w, height), (255, 255, 255))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + sep_px
    return canvas


def load_frame_ids(args: argparse.Namespace) -> list[str]:
    if args.frame is not None:
        return [args.frame]
    if args.frames is not None:
        return [s.strip() for s in args.frames.split(",") if s.strip()]
    ids: list[str] = []
    for line in args.frames_file.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            ids.append(s)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Read-only input root containing images/.",
    )
    ap.add_argument(
        "--bbox-labels-dir",
        type=Path,
        default=None,
        help="YOLO bbox labels dir (default: --data-dir/labels).",
    )
    ap.add_argument(
        "--seg-labels-dir",
        type=Path,
        default=REPO_ROOT / "labels_seg",
        help="YOLO-seg polygon labels dir.",
    )
    ap.add_argument(
        "--predictions-dir",
        type=Path,
        default=REPO_ROOT / "cpp_out",
        help="C++ inference JSON outputs (one .json per frame).",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "viz", help="Composites are written here."
    )

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frame", type=str, default=None, help="Single frame ID, e.g. 0_5.")
    src.add_argument(
        "--frames", type=str, default=None, help="Comma-separated frame IDs, e.g. 0_5,100_3."
    )
    src.add_argument(
        "--frames-file",
        type=Path,
        default=None,
        help="File with one frame ID per line; '#' comments allowed.",
    )

    ap.add_argument(
        "--panels",
        type=str,
        default="raw,bbox,polygon",
        help=f"Comma list of panel names in render order. " f"Available: {','.join(PANELS)}.",
    )
    ap.add_argument("--stack", choices=("vertical", "horizontal"), default="vertical")
    ap.add_argument(
        "--label-panels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay a small text label per panel.",
    )
    ap.add_argument("--bbox-color", type=parse_hex_color, default=parse_hex_color("#ff3030"))
    ap.add_argument("--polygon-color", type=parse_hex_color, default=parse_hex_color("#ffff00"))
    ap.add_argument(
        "--polygon-alpha", type=float, default=0.35, help="Filled-polygon transparency in [0, 1]."
    )
    ap.add_argument(
        "--prediction-color",
        type=parse_hex_color,
        default=parse_hex_color("#00e676"),
        help="Color for model-prediction polygons (distinct from GT yellow).",
    )
    ap.add_argument("--prediction-alpha", type=float, default=0.35)
    args = ap.parse_args()

    images_dir = args.data_dir / "images"
    bbox_dir = args.bbox_labels_dir or (args.data_dir / "labels")
    seg_dir = args.seg_labels_dir
    preds_dir = args.predictions_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel_names = [s.strip() for s in args.panels.split(",") if s.strip()]
    for name in panel_names:
        if name not in PANELS:
            raise SystemExit(f"unknown panel {name!r}; available: {','.join(PANELS)}")

    frame_ids = load_frame_ids(args)
    if not frame_ids:
        raise SystemExit("no frame IDs collected.")

    print(
        f"compare_frames: {len(frame_ids)} frame(s)  panels={','.join(panel_names)}  stack={args.stack}"
    )
    print(f"  output={rel_to_root(args.output_dir)}/")

    n_ok = 0
    for fid in frame_ids:
        img_path = images_dir / f"{fid}.png"
        if not img_path.exists():
            print(f"  WARN missing image: {fid}", file=sys.stderr)
            continue
        with Image.open(img_path) as img:
            base = img.convert("RGB")
        w, h = base.size

        bboxes = parse_yolo_bboxes(bbox_dir / f"{fid}.txt", w, h)
        polygons = parse_yolo_seg(seg_dir / f"{fid}.txt", w, h)
        predictions = parse_predictions_json(preds_dir / f"{fid}.json")

        panel_imgs = []
        for name in panel_names:
            panel = PANELS[name](
                base,
                bboxes=bboxes,
                polygons=polygons,
                predictions=predictions,
                bbox_color=args.bbox_color,
                polygon_color=args.polygon_color,
                polygon_alpha=args.polygon_alpha,
                prediction_color=args.prediction_color,
                prediction_alpha=args.prediction_alpha,
            )
            if args.label_panels:
                panel = add_label(panel, name)
            panel_imgs.append(panel)

        composite = compose(panel_imgs, args.stack)
        out_path = args.output_dir / f"{fid}.png"
        composite.save(out_path)
        n_ok += 1

    print(f"wrote {n_ok}/{len(frame_ids)} composites to {rel_to_root(args.output_dir)}/")


if __name__ == "__main__":
    main()
