#!/usr/bin/env python3
"""Evaluate a trained YOLOv11-seg model on a held-out split.

Defaults: load the most-recent best.pt under out/runs/segment/*/weights/ and
evaluate on the test split. Pass --split val to compare against in-training
val numbers, or --split train to gauge overfitting.

Train was used to fit the model. Val drove early-stopping, so val isn't
strictly "unseen". Test was never used for model selection — that's the
honest generalization number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Sibling pipeline scripts in scripts/ own stage_util; we're one level deeper.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage_util import (  # noqa: E402  (sys.path bump must precede this)
    add_config_arg,
    apply_config_defaults,
    load_config_section,
    save_run_meta,
    stage_timer,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE = "eval_yolo"


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def latest_best_weights(runs_dir: Path) -> Path:
    candidates = list(runs_dir.glob("*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(f"no best.pt under {runs_dir}/. Train first or pass --weights.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    add_config_arg(pre)
    pre_args, _ = pre.parse_known_args()

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_arg(ap)
    ap.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to best.pt; default = most recent under --runs-dir.",
    )
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=REPO_ROOT / "out" / "runs" / "segment",
        help="Where to look for runs (auto-pick latest best.pt).",
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "out" / "yolo_dataset" / "dataset.yaml",
        help="dataset.yaml from train_yolo.py staging.",
    )
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument(
        "--name", default=None, help="Output subdir name; default = ultralytics auto-naming."
    )
    apply_config_defaults(ap, load_config_section(pre_args.config, STAGE))
    args = ap.parse_args()

    with stage_timer(STAGE) as timing:
        save_dir = _run(args)

    save_run_meta(Path(save_dir), STAGE, args, elapsed_sec=timing["elapsed_sec"])


def _run(args: argparse.Namespace) -> Path:
    weights = args.weights or latest_best_weights(args.runs_dir)
    print(f"weights: {rel_to_root(weights)}")
    print(f"data:    {rel_to_root(args.data)}")
    print(f"split:   {args.split}\n")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    kwargs: dict = {
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
    }
    if args.name:
        kwargs["name"] = args.name
    metrics = model.val(**kwargs)

    print("\n=== Summary ===")
    print(f"  split: {args.split}")
    print(
        f"  box:  P={metrics.box.mp:.3f}  R={metrics.box.mr:.3f}  "
        f"mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}"
    )
    print(
        f"  mask: P={metrics.seg.mp:.3f}  R={metrics.seg.mr:.3f}  "
        f"mAP50={metrics.seg.map50:.3f}  mAP50-95={metrics.seg.map:.3f}"
    )
    print(f"  outputs (incl. sample renders): {metrics.save_dir}")
    return Path(metrics.save_dir)


if __name__ == "__main__":
    main()
