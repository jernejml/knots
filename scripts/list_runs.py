#!/usr/bin/env python3
"""Tabulate every YOLO training run under out/runs/segment/ for comparison.

Reads each run's run_meta_train_yolo.json (resolved args, elapsed, git_sha)
and, if present, the co-located eval_boards.json (aggregate P / R / F1 / IoU).
Prints one row per run; defaults sort newest-first.

Picks up runs left behind by ./run.sh as well as anything written by
train_yolo.py directly — both produce run_meta_train_yolo.json at the
ultralytics save_dir.

Read-only; no side effects. Skip --config / --timer / run_meta dance —
this is an inspection tool, not a pipeline stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SORT_KEYS = ("mtime", "name", "f1", "miou", "elapsed", "epochs")


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def fmt_elapsed(s: float | None) -> str:
    if s is None:
        return "--"
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m{int(s):02d}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m"


def fmt_opt(v: float | None, prec: int = 3) -> str:
    return f"{v:.{prec}f}" if isinstance(v, (int, float)) else "--"


def short_sha(sha: str | None) -> str:
    if not sha:
        return "--"
    return sha.split("-")[0][:8] + ("*" if sha.endswith("-dirty") else "")


def collect_row(run_dir: Path) -> dict:
    meta_path = run_dir / "run_meta_train_yolo.json"
    meta = json.loads(meta_path.read_text())
    args = meta.get("args", {})
    row: dict = {
        "name": run_dir.name,
        "mtime": meta_path.stat().st_mtime,
        "model": args.get("model", "--"),
        "imgsz": args.get("imgsz", "--"),
        "batch": args.get("batch", "--"),
        "epochs": args.get("epochs", "--"),
        "elapsed": meta.get("elapsed_sec"),
        "git_sha": meta.get("git_sha"),
        "f1": None,
        "miou": None,
        "precision": None,
        "recall": None,
    }
    eval_path = run_dir / "eval_boards.json"
    if eval_path.is_file():
        try:
            agg = json.loads(eval_path.read_text()).get("aggregate", {})
            row["f1"] = agg.get("f1")
            row["miou"] = agg.get("mean_iou_micro")
            row["precision"] = agg.get("precision")
            row["recall"] = agg.get("recall")
        except Exception as e:
            print(f"WARN: cannot read {rel_to_root(eval_path)}: {e}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=REPO_ROOT / "out" / "runs" / "segment",
        help="Where to look for training runs.",
    )
    ap.add_argument("--sort", choices=SORT_KEYS, default="mtime")
    ap.add_argument("--order", choices=("asc", "desc"), default="desc")
    ap.add_argument("--limit", type=int, default=0, help="0 = print all.")
    args = ap.parse_args()

    if not args.runs_dir.is_dir():
        raise SystemExit(f"no such dir: {args.runs_dir}")

    metas = sorted(args.runs_dir.glob("*/run_meta_train_yolo.json"))
    if not metas:
        print(f"no training runs under {rel_to_root(args.runs_dir)}/")
        return

    rows = [collect_row(p.parent) for p in metas]
    sort_key = args.sort
    # None values always sort to the bottom regardless of order — otherwise
    # 'desc' bubbles every un-evaluated run to the top, which is the opposite
    # of what you want when looking for best metrics.
    valid = [r for r in rows if r[sort_key] is not None]
    missing = [r for r in rows if r[sort_key] is None]
    valid.sort(key=lambda r: r[sort_key], reverse=(args.order == "desc"))
    rows = valid + missing
    if args.limit > 0:
        rows = rows[: args.limit]

    header = (
        f"  {'name':<14}  {'model':<18}  {'imgsz':>5}  {'batch':>5}  "
        f"{'epochs':>6}  {'elapsed':>7}  "
        f"{'P':>5}  {'R':>5}  {'F1':>5}  {'mIoU':>5}  {'sha':<10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(
            f"  {r['name']:<14}  {str(r['model']):<18}  {str(r['imgsz']):>5}  "
            f"{str(r['batch']):>5}  {str(r['epochs']):>6}  "
            f"{fmt_elapsed(r['elapsed']):>7}  "
            f"{fmt_opt(r['precision']):>5}  {fmt_opt(r['recall']):>5}  "
            f"{fmt_opt(r['f1']):>5}  {fmt_opt(r['miou']):>5}  "
            f"{short_sha(r['git_sha']):<10}"
        )

    print(f"\n{len(rows)} run(s) under {rel_to_root(args.runs_dir)}/")


if __name__ == "__main__":
    main()
