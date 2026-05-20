#!/usr/bin/env python3
"""Render comparison composites for frames or stitched boards.

Two scopes selected via --scope:

  --scope frame   One composite per frame ID (e.g. 0_5).
                  Panels: raw, bbox, polygon, predictions.
  --scope board   One composite per board, built by stitching the board's
                  frames at (frame_idx * stride_px, 0). Adjacent frames
                  overlap by 50 % (stride 320 of 640); pasting later frames
                  on top is harmless because both views show the same
                  pixels. Panels: raw, bbox, predictions.

Inputs (all read-only):
    --data-dir/images/{board}_{frame}.png
    --bbox-labels-dir/{board}_{frame}.txt    YOLO bbox labels (both scopes)
    --seg-labels-dir/{board}_{frame}.txt     YOLO-seg labels (frame scope)
    --predictions-dir/{board}_{frame}.json   per-frame inference (frame scope)
    --boards-pred-dir/{board}.json           per-board polygons (board scope)

Output:
    --output-dir/{id}.png  (frame ID for frame scope, board ID for board scope)

IDs (mutually exclusive):
    --id ID                 single ID; '0_5' for frame scope, '0' for board scope
    --ids LIST              comma-separated
    --ids-file PATH         one ID per line; '#' comments allowed
    --all                   board scope only; discover from --boards-pred-dir
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]

STRIDE_PX_DEFAULT = 320
FRAME_RE = re.compile(r"^(\d+)_(\d+)\.png$")

FRAME_PANELS = ("raw", "bbox", "polygon", "predictions")
BOARD_PANELS = ("raw", "bbox", "predictions")


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_hex_color(s: str) -> tuple[int, int, int]:
    raw = s.strip().lstrip("#")
    if len(raw) != 6:
        raise argparse.ArgumentTypeError(f"expected #rrggbb, got {s!r}")
    return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))


def parse_yolo_bboxes(label_path: Path, w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Frame-local (x1, y1, x2, y2) per YOLO bbox line; empty list if no file."""
    if not label_path.exists():
        return []
    out: list[tuple[int, int, int, int]] = []
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
            out.append((x1, y1, x2, y2))
    return out


def parse_yolo_seg(label_path: Path, w: int, h: int) -> list[list[tuple[int, int]]]:
    """Frame-local polygon vertices per YOLO-seg line."""
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


def parse_per_frame_predictions(path: Path) -> list[list[tuple[int, int]]]:
    """Polygons from `knots infer` JSON (already in source-image px coords)."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    out: list[list[tuple[int, int]]] = []
    for det in data.get("detections", []):
        poly = [(int(round(x)), int(round(y))) for x, y in det.get("polygon", [])]
        if len(poly) >= 3:
            out.append(poly)
    return out


def parse_per_board_predictions(path: Path) -> list[list[tuple[int, int]]]:
    """Polygons from `knots stitch` JSON (already in board-coord px)."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"  WARN invalid JSON in {path}", file=sys.stderr)
        return []
    out: list[list[tuple[int, int]]] = []
    for knot in data.get("knots", []):
        poly = [(int(x), int(y)) for x, y in knot.get("polygon", [])]
        if len(poly) >= 3:
            out.append(poly)
    return out


# ---------------------------------------------------------------------------
# Canvas builders
# ---------------------------------------------------------------------------


def load_frames_for_board(images_dir: Path, board: int) -> list[tuple[int, Path]]:
    result: list[tuple[int, Path]] = []
    for entry in images_dir.iterdir():
        m = FRAME_RE.match(entry.name)
        if m and int(m.group(1)) == board:
            result.append((int(m.group(2)), entry))
    result.sort(key=lambda t: t[0])
    return result


def build_frame_canvas(
    images_dir: Path, frame_id: str
) -> tuple[Image.Image, list[tuple[int, int, int]]] | None:
    """Returns (canvas, [(frame_idx=0, w, h)]) so the overlay path is the
    same as for board scope (which yields one entry per stitched frame)."""
    path = images_dir / f"{frame_id}.png"
    if not path.exists():
        print(f"  WARN missing image: {frame_id}", file=sys.stderr)
        return None
    with Image.open(path) as img:
        rgb = img.convert("RGB")
    return rgb, [(0, rgb.width, rgb.height)]


def build_board_canvas(
    images_dir: Path, board: int, stride_px: int
) -> tuple[Image.Image, list[tuple[int, int, int]]] | None:
    """Returns (stitched canvas, [(frame_idx, w, h), ...])."""
    frames = load_frames_for_board(images_dir, board)
    if not frames:
        print(f"  WARN board {board}: no frames found", file=sys.stderr)
        return None

    with Image.open(frames[0][1]) as first:
        board_h = first.height
    last_idx, last_path = frames[-1]
    with Image.open(last_path) as last_img:
        last_w = last_img.width
    board_w = last_idx * stride_px + last_w

    canvas = Image.new("RGB", (board_w, board_h), (0, 0, 0))
    info: list[tuple[int, int, int]] = []
    for frame_idx, path in frames:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            if rgb.height != board_h:
                print(
                    f"  WARN board {board} frame {frame_idx}: "
                    f"H={rgb.height} vs board H={board_h}",
                    file=sys.stderr,
                )
            canvas.paste(rgb, (frame_idx * stride_px, 0))
            info.append((frame_idx, rgb.width, rgb.height))
    return canvas, info


# ---------------------------------------------------------------------------
# Overlay collection (everything lives in canvas-coord space after this)
# ---------------------------------------------------------------------------


def collect_bboxes(
    scope: str,
    *,
    bbox_dir: Path,
    frame_id: str | None,
    board: int | None,
    frame_info: list[tuple[int, int, int]],
    stride_px: int,
) -> list[tuple[int, int, int, int]]:
    """Bboxes in canvas coords."""
    out: list[tuple[int, int, int, int]] = []
    if scope == "frame":
        assert frame_id is not None
        _, w, h = frame_info[0]
        return parse_yolo_bboxes(bbox_dir / f"{frame_id}.txt", w, h)
    assert board is not None
    for frame_idx, w, h in frame_info:
        boxes = parse_yolo_bboxes(bbox_dir / f"{board}_{frame_idx}.txt", w, h)
        dx = frame_idx * stride_px
        for x1, y1, x2, y2 in boxes:
            out.append((x1 + dx, y1, x2 + dx, y2))
    return out


def collect_seg_polygons(
    seg_dir: Path, frame_id: str, frame_info: list[tuple[int, int, int]]
) -> list[list[tuple[int, int]]]:
    """YOLO-seg polygons (frame scope only)."""
    _, w, h = frame_info[0]
    return parse_yolo_seg(seg_dir / f"{frame_id}.txt", w, h)


def collect_predictions(
    scope: str,
    *,
    predictions_dir: Path,
    boards_pred_dir: Path,
    frame_id: str | None,
    board: int | None,
) -> list[list[tuple[int, int]]]:
    """Model-output polygons in canvas coords."""
    if scope == "frame":
        assert frame_id is not None
        return parse_per_frame_predictions(predictions_dir / f"{frame_id}.json")
    assert board is not None
    return parse_per_board_predictions(boards_pred_dir / f"{board}.json")


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_raw(canvas: Image.Image, **_kwargs) -> Image.Image:
    return canvas.copy()


def render_bbox(
    canvas: Image.Image,
    *,
    bboxes: list[tuple[int, int, int, int]],
    bbox_color: tuple[int, int, int],
    **_kwargs,
) -> Image.Image:
    out = canvas.copy()
    draw = ImageDraw.Draw(out)
    for x1, y1, x2, y2 in bboxes:
        draw.rectangle([(x1, y1), (x2 - 1, y2 - 1)], outline=bbox_color, width=2)
    return out


def render_polygons(
    canvas: Image.Image,
    polygons: list[list[tuple[int, int]]],
    color: tuple[int, int, int],
    alpha: float,
) -> Image.Image:
    if not polygons:
        return canvas.copy()
    base = canvas.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    fill_alpha = max(0, min(255, int(round(255 * alpha))))
    fill = color + (fill_alpha,)
    outline = color + (255,)
    o_draw = ImageDraw.Draw(overlay)
    for poly in polygons:
        if len(poly) >= 3:
            o_draw.polygon(poly, fill=fill, outline=outline)
    return Image.alpha_composite(base, overlay).convert("RGB")


def render_polygon_panel(
    canvas: Image.Image,
    *,
    seg_polygons: list[list[tuple[int, int]]],
    polygon_color: tuple[int, int, int],
    polygon_alpha: float,
    **_kwargs,
) -> Image.Image:
    return render_polygons(canvas, seg_polygons, polygon_color, polygon_alpha)


def render_predictions_panel(
    canvas: Image.Image,
    *,
    predictions: list[list[tuple[int, int]]],
    prediction_color: tuple[int, int, int],
    prediction_alpha: float,
    **_kwargs,
) -> Image.Image:
    return render_polygons(canvas, predictions, prediction_color, prediction_alpha)


PANEL_FUNCS = {
    "raw": render_raw,
    "bbox": render_bbox,
    "polygon": render_polygon_panel,
    "predictions": render_predictions_panel,
}


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------


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
    height = max(p.height for p in panels)
    total_w = sum(p.width for p in panels) + sep_px * (len(panels) - 1)
    canvas = Image.new("RGB", (total_w, height), (255, 255, 255))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + sep_px
    return canvas


def downscale(img: Image.Image, max_width: int) -> Image.Image:
    if max_width <= 0 or img.width <= max_width:
        return img
    new_h = int(round(img.height * max_width / img.width))
    return img.resize((max_width, new_h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# ID resolution
# ---------------------------------------------------------------------------


def load_ids(args: argparse.Namespace) -> list[str]:
    """Return IDs as strings ('0_5' for frame scope, '0' for board scope).
    Board-scope IDs are validated to be digits; frame IDs to '{int}_{int}'."""
    if args.id is not None:
        return [args.id]
    if args.ids is not None:
        return [s.strip() for s in args.ids.split(",") if s.strip()]
    if args.ids_file is not None:
        ids: list[str] = []
        for line in args.ids_file.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                ids.append(s)
        return ids
    if args.all:
        if args.scope != "board":
            raise SystemExit("--all is only valid with --scope board")
        if not args.boards_pred_dir.is_dir():
            raise SystemExit(f"--all needs --boards-pred-dir to exist: {args.boards_pred_dir}")
        ids = [
            e.stem
            for e in args.boards_pred_dir.iterdir()
            if e.suffix == ".json" and e.stem.isdigit()
        ]
        return sorted(ids, key=int)
    return []


def validate_ids(scope: str, ids: list[str]) -> None:
    if scope == "frame":
        bad = [i for i in ids if not re.match(r"^\d+_\d+$", i)]
        if bad:
            raise SystemExit(f"frame IDs must look like '0_5'; got: {bad[:5]}")
    else:
        bad = [i for i in ids if not i.isdigit()]
        if bad:
            raise SystemExit(f"board IDs must be integers; got: {bad[:5]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--scope",
        choices=("frame", "board"),
        default="frame",
        help="What to render: per-frame or per-board composites.",
    )

    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument(
        "--bbox-labels-dir",
        type=Path,
        default=None,
        help="YOLO bbox labels dir (default: --data-dir/labels).",
    )
    ap.add_argument(
        "--seg-labels-dir",
        type=Path,
        default=REPO_ROOT / "out" / "labels" / "seg",
        help="YOLO-seg labels dir (frame scope only).",
    )
    ap.add_argument(
        "--predictions-dir",
        type=Path,
        default=REPO_ROOT / "out" / "frames",
        help="Per-frame `knots infer` JSON outputs (frame scope only).",
    )
    ap.add_argument(
        "--boards-pred-dir",
        type=Path,
        default=REPO_ROOT / "out" / "boards" / "pred",
        help="Per-board `knots stitch` JSON outputs (board scope only).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Composites are written here. Default: out/viz/ (frame) or out/viz/boards/ (board).",
    )

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--id", type=str, default=None, help="Single ID.")
    src.add_argument("--ids", type=str, default=None, help="Comma-separated IDs.")
    src.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help="File with one ID per line; '#' comments allowed.",
    )
    src.add_argument(
        "--all",
        action="store_true",
        help="Board scope: every board with a JSON in --boards-pred-dir.",
    )

    ap.add_argument(
        "--panels",
        type=str,
        default=None,
        help="Comma list of panel names. "
        f"Frame: {','.join(FRAME_PANELS)}. Board: {','.join(BOARD_PANELS)}.",
    )
    ap.add_argument("--stack", choices=("vertical", "horizontal"), default="vertical")
    ap.add_argument("--label-panels", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--stride-px",
        type=int,
        default=STRIDE_PX_DEFAULT,
        help="Frame stride for board stitching (default 320).",
    )
    ap.add_argument(
        "--max-width",
        type=int,
        default=4096,
        help="Downscale composite to this width. 0 = no downscale. Board scope only.",
    )

    ap.add_argument("--bbox-color", type=parse_hex_color, default=parse_hex_color("#ff3030"))
    ap.add_argument(
        "--polygon-color",
        type=parse_hex_color,
        default=parse_hex_color("#ffff00"),
        help="Color for YOLO-seg GT polygons (frame scope only).",
    )
    ap.add_argument("--polygon-alpha", type=float, default=0.35)
    ap.add_argument(
        "--prediction-color",
        type=parse_hex_color,
        default=parse_hex_color("#00e676"),
        help="Color for model predictions (frame: per-frame JSON, board: per-board JSON).",
    )
    ap.add_argument("--prediction-alpha", type=float, default=0.35)
    args = ap.parse_args()

    images_dir = args.data_dir / "images"
    bbox_dir = args.bbox_labels_dir or (args.data_dir / "labels")
    if args.output_dir is None:
        args.output_dir = (
            REPO_ROOT / "out" / "viz"
            if args.scope == "frame"
            else REPO_ROOT / "out" / "viz" / "boards"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    allowed = FRAME_PANELS if args.scope == "frame" else BOARD_PANELS
    default_panels = "raw,bbox,polygon" if args.scope == "frame" else "raw,bbox,predictions"
    panels_str = args.panels or default_panels
    panel_names = [s.strip() for s in panels_str.split(",") if s.strip()]
    for name in panel_names:
        if name not in allowed:
            raise SystemExit(
                f"unknown panel {name!r} for --scope {args.scope}; "
                f"available: {','.join(allowed)}"
            )

    ids = load_ids(args)
    if not ids:
        raise SystemExit("no IDs collected.")
    validate_ids(args.scope, ids)

    print(
        f"compare ({args.scope}): {len(ids)} item(s)  panels={','.join(panel_names)}  "
        f"stack={args.stack}"
    )
    print(f"  output={rel_to_root(args.output_dir)}/")

    n_ok = 0
    for item_id in ids:
        if args.scope == "frame":
            built = build_frame_canvas(images_dir, item_id)
            board_int = None
            frame_id = item_id
        else:
            board_int = int(item_id)
            built = build_board_canvas(images_dir, board_int, args.stride_px)
            frame_id = None
        if built is None:
            continue
        canvas, frame_info = built

        bboxes = collect_bboxes(
            args.scope,
            bbox_dir=bbox_dir,
            frame_id=frame_id,
            board=board_int,
            frame_info=frame_info,
            stride_px=args.stride_px,
        )
        seg_polygons = (
            collect_seg_polygons(args.seg_labels_dir, frame_id, frame_info)
            if args.scope == "frame" and "polygon" in panel_names
            else []
        )
        predictions = (
            collect_predictions(
                args.scope,
                predictions_dir=args.predictions_dir,
                boards_pred_dir=args.boards_pred_dir,
                frame_id=frame_id,
                board=board_int,
            )
            if "predictions" in panel_names
            else []
        )

        panel_imgs: list[Image.Image] = []
        for name in panel_names:
            panel = PANEL_FUNCS[name](
                canvas,
                bboxes=bboxes,
                seg_polygons=seg_polygons,
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
        if args.scope == "board":
            composite = downscale(composite, args.max_width)
        composite.save(args.output_dir / f"{item_id}.png")
        n_ok += 1

    print(f"wrote {n_ok}/{len(ids)} composites to {rel_to_root(args.output_dir)}/")


if __name__ == "__main__":
    main()
