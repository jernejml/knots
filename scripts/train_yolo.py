#!/usr/bin/env python3
"""Train YOLOv11-seg on SAM-derived polygon labels, then export to ONNX.

Builds a YOLO-shaped staging dir (symlinks back to data/images/ and the
chosen SAM polygon-labels dir), writes split list files derived from
out/analysis/partitions.json, calls ultralytics' training loop, and finally
exports a checkpoint (best or last, see --export-weights) to ONNX next to the
.pt. The C++ runtime consumes the policy-neutral path `out/models/model.onnx`,
refreshed as a symlink to the latest export; run_meta records which checkpoint
it came from.

Staging layout:
    out/yolo_dataset/
        dataset.yaml
        images/   (symlinks into data/images/)
        labels/   (symlinks into out/labels/seg/)
        train.txt val.txt test.txt   (paths relative to repo root)

Evaluate the trained model with scripts/tools/eval_yolo.py.

Augmentation: ultralytics defaults. Pass augmentation kwargs via the
script (a future change) only after seeing the train/val curve.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from stage_util import (
    add_config_arg,
    apply_config_defaults,
    load_config_section,
    save_run_meta,
    stage_timer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE = "train_yolo"
FILENAME_RE = re.compile(r"^(\d+)_(\d+)\.png$")
# C++ runtime reads from this fixed, policy-neutral path; the symlink is
# refreshed each train pass to point at the just-exported ONNX. Which
# checkpoint produced it (best or last) is recorded in run_meta, not the name.
LATEST_ONNX_LINK = REPO_ROOT / "out" / "models" / "model.onnx"
VALID_SPLITS = ("train", "val", "test")


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def ensure_symlink(link: Path, target: Path) -> None:
    """Create a relative symlink at `link` pointing at `target`. Idempotent."""
    rel_target = Path(os.path.relpath(target.resolve(), link.parent.resolve()))
    if link.is_symlink() or link.exists():
        if link.is_symlink() and Path(os.readlink(link)) == rel_target:
            return
        link.unlink()
    link.symlink_to(rel_target)


def refresh_latest_onnx_link(target: Path) -> None:
    """Update LATEST_ONNX_LINK to point at `target` via a relative symlink."""
    LATEST_ONNX_LINK.parent.mkdir(parents=True, exist_ok=True)
    target_resolved = target.resolve()
    if LATEST_ONNX_LINK.resolve(strict=False) == target_resolved:
        return
    if LATEST_ONNX_LINK.is_symlink() or LATEST_ONNX_LINK.exists():
        LATEST_ONNX_LINK.unlink()
    rel = os.path.relpath(target_resolved, LATEST_ONNX_LINK.parent.resolve())
    LATEST_ONNX_LINK.symlink_to(rel)


def load_board_splits(partitions_json: Path) -> dict[int, str]:
    """Parse out/analysis/partitions.json → {board: split}.

    File shape is {"train": [board_ids], "val": [...], "test": [...]};
    we invert it to a per-board lookup for the staging pass below.
    """
    if not partitions_json.is_file():
        raise SystemExit(f"missing {partitions_json}; run scripts/prepare.py first.")
    data = json.loads(partitions_json.read_text())
    out: dict[int, str] = {}
    for split, board_ids in data.items():
        if split not in VALID_SPLITS:
            raise SystemExit(f"unexpected split {split!r} in {partitions_json}")
        for bid in board_ids:
            out[int(bid)] = split
    return out


def build_staging(args: argparse.Namespace) -> Path:
    images_dir = args.data_dir / "images"
    seg_dir = args.seg_labels_dir
    if not images_dir.is_dir():
        raise SystemExit(f"{images_dir} not found.")
    if not seg_dir.is_dir():
        raise SystemExit(f"{seg_dir} not found.")

    board_to_split = load_board_splits(args.partitions_json)

    args.staging_dir.mkdir(parents=True, exist_ok=True)
    images_link_dir = args.staging_dir / "images"
    labels_link_dir = args.staging_dir / "labels"
    images_link_dir.mkdir(exist_ok=True)
    labels_link_dir.mkdir(exist_ok=True)

    by_split: dict[str, list[str]] = {s: [] for s in VALID_SPLITS}
    n_paired = 0
    n_no_label = 0
    n_no_split = 0
    for img_path in sorted(images_dir.glob("*.png")):
        m = FILENAME_RE.match(img_path.name)
        if not m:
            continue
        board = int(m.group(1))
        split = board_to_split.get(board)
        if split is None:
            n_no_split += 1
            continue

        lbl_path = seg_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            n_no_label += 1
            continue

        ensure_symlink(images_link_dir / img_path.name, img_path)
        ensure_symlink(labels_link_dir / lbl_path.name, lbl_path)
        # Path relative to repo root (= cwd in our Docker setup).
        by_split[split].append(str(images_link_dir.relative_to(REPO_ROOT) / img_path.name))
        n_paired += 1

    for split in VALID_SPLITS:
        (args.staging_dir / f"{split}.txt").write_text("\n".join(by_split[split]) + "\n")

    yaml_path = args.staging_dir / "dataset.yaml"
    yaml_path.write_text(
        "# Auto-generated by scripts/train_yolo.py — do not commit.\n"
        "# Paths are relative to cwd (repo root in our Docker setup).\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "test: test.txt\n"
        "names:\n"
        "  0: knot\n"
    )

    print(f"staging at {rel_to_root(args.staging_dir)}/")
    print(f"  paired (image, label) symlinks: {n_paired}")
    if n_no_label:
        print(f"  WARN: {n_no_label} frame(s) missing in {rel_to_root(seg_dir)}/ — skipped")
    if n_no_split:
        print(f"  WARN: {n_no_split} frame(s) whose board is not in partitions.json — skipped")
    print(
        f"  train={len(by_split['train'])}  val={len(by_split['val'])}  test={len(by_split['test'])}"
    )
    return yaml_path


def main() -> None:
    # Two-stage parse: pre-parser grabs --config so its values can pre-fill
    # the real parser's defaults before --help / parse_args runs.
    pre = argparse.ArgumentParser(add_help=False)
    add_config_arg(pre)
    pre_args, _ = pre.parse_known_args()

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_arg(ap)
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Read-only input root containing images/.",
    )
    ap.add_argument(
        "--seg-labels-dir",
        type=Path,
        default=REPO_ROOT / "out" / "labels" / "seg",
        help="SAM polygon labels (YOLO-seg format).",
    )
    ap.add_argument(
        "--partitions-json",
        type=Path,
        default=REPO_ROOT / "out" / "analysis" / "partitions.json",
        help="Board-level split assignment from prepare.py.",
    )
    ap.add_argument(
        "--staging-dir",
        type=Path,
        default=REPO_ROOT / "out" / "yolo_dataset",
        help="Where the YOLO scaffold (symlinks + dataset.yaml) lives.",
    )
    ap.add_argument(
        "--model", default="yolo11m-seg.pt", help="Starting checkpoint or model variant."
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--name", default="train", help="Run name (suffix in <project>/).")
    ap.add_argument(
        "--project",
        type=Path,
        default=REPO_ROOT / "out" / "runs" / "segment",
        help="Output project dir; default = out/runs/segment/.",
    )
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--resume", action="store_true", help="Resume the last checkpoint of this run name."
    )
    ap.add_argument(
        "--setup-only", action="store_true", help="Build the staging dir and exit (no training)."
    )
    # -- ONNX export knobs (the export runs as the last step of this script) --
    ap.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version for the export; 17 has wider runtime compatibility than 19+.",
    )
    ap.add_argument(
        "--simplify",
        action="store_true",
        help="Run onnxslim on the exported graph (requires onnxslim installed).",
    )
    ap.add_argument(
        "--export-device",
        default="cpu",
        help="Device for the ONNX export; CPU is fine and avoids GPU init.",
    )
    ap.add_argument(
        "--export-weights",
        choices=("best", "last"),
        default="last",
        help="Which checkpoint to export: 'best' (highest val fitness) or 'last' "
        "(final epoch). Default 'last' — val labels are SAM pseudo-labels, so the "
        "best-val selection signal is weak.",
    )
    apply_config_defaults(ap, load_config_section(pre_args.config, STAGE))
    args = ap.parse_args()

    with stage_timer(STAGE) as timing:
        yaml_path = build_staging(args)
        if args.setup_only:
            print("--setup-only: skipping training")
            return

        from ultralytics import YOLO

        print(f"loading {args.model} …")
        model = YOLO(args.model)

        print(
            f"training: epochs={args.epochs}  imgsz={args.imgsz}  "
            f"batch={args.batch}  patience={args.patience}  device={args.device}"
        )
        train_kwargs = dict(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            name=args.name,
            device=args.device,
            workers=args.workers,
            resume=args.resume,
        )
        train_kwargs["project"] = str(args.project)
        model.train(**train_kwargs)
        save_dir = Path(model.trainer.save_dir)
        # `model.trainer` exposes both checkpoints as .best / .last; pick the
        # one the caller asked for. Load it fresh rather than exporting the
        # in-memory weights, which hold the final epoch regardless of choice.
        checkpoint = Path(getattr(model.trainer, args.export_weights))

        print(
            f"\nexporting {args.export_weights} checkpoint {rel_to_root(checkpoint)} → ONNX "
            f"(opset={args.opset}, simplify={args.simplify}, device={args.export_device})"
        )
        export_model = YOLO(str(checkpoint))
        exported = Path(
            export_model.export(
                format="onnx",
                imgsz=args.imgsz,
                opset=args.opset,
                simplify=args.simplify,
                dynamic=False,
                half=False,
                nms=True,
                device=args.export_device,
            )
        )
        size_mb = exported.stat().st_size / 1e6
        print(f"exported → {rel_to_root(exported)}  ({size_mb:.1f} MB)")
        refresh_latest_onnx_link(exported)
        print(
            f"latest pointer: {rel_to_root(LATEST_ONNX_LINK)} → " f"{os.readlink(LATEST_ONNX_LINK)}"
        )

    meta_path = save_run_meta(
        save_dir,
        STAGE,
        args,
        elapsed_sec=timing["elapsed_sec"],
        extra={
            "export_weights": args.export_weights,
            "source_checkpoint": rel_to_root(checkpoint),
            "published_onnx": rel_to_root(LATEST_ONNX_LINK),
        },
    )
    print(f"run meta written: {rel_to_root(meta_path)}")


if __name__ == "__main__":
    main()
