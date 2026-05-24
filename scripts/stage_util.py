"""Shared per-stage plumbing: TOML config loading, wall-time timing, run metadata.

Each pipeline script (prepare, sam_polygons, train_yolo) reads its inputs
from a `[<stage>]` TOML section, runs its work inside a `stage_timer(...)`
block, and dumps a `run_meta.json` next to its output artefacts. CLI flags
continue to override config values, so the quick-iteration loop "edit a
number on the command line" still works.

Wiring pattern (see scripts/train_yolo.py for the worked example):

    pre = argparse.ArgumentParser(add_help=False)
    add_config_arg(pre)
    pre_args, _ = pre.parse_known_args()

    ap = argparse.ArgumentParser(...)
    add_config_arg(ap)               # so --help shows it
    ap.add_argument("--epochs", type=int, default=100)
    ...
    apply_config_defaults(ap, load_config_section(pre_args.config, "train_yolo"))
    args = ap.parse_args()

    with stage_timer("train_yolo") as t:
        ...do work...

    save_run_meta(out_dir, "train_yolo", args, elapsed_sec=t["elapsed_sec"])

Caveats
- Config values are applied as argparse *defaults*. argparse only runs `type=`
  on values it sees on the command line, so for non-scalar custom types
  (e.g. parse_ratios) keep config values as the post-`type()` form — or write
  the type to accept both. The helper coerces str→type for the common cases
  (Path, int, float, bool); other type collisions raise loudly.
- Unknown keys in a config section raise SystemExit so typos don't silently
  no-op. Sections you don't recognise are ignored — one shared config file
  can drive many stages.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tomllib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML config file. Values under the stage's [section] become "
        "argparse defaults; CLI flags still override.",
    )


def load_config_section(path: Path | None, section: str) -> dict[str, Any]:
    """Return the [section] table from `path`, or {} if no config or no section."""
    if path is None:
        return {}
    if not path.is_file():
        raise SystemExit(f"--config: not a file: {path}")
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    val = data.get(section, {})
    if not isinstance(val, dict):
        raise SystemExit(f"--config: [{section}] must be a table, got {type(val).__name__}")
    return val


def apply_config_defaults(parser: argparse.ArgumentParser, cfg: dict[str, Any]) -> None:
    """Set argparse defaults from `cfg`. Unknown keys raise.

    Coercion rules for `action.type` (when it's a callable other than `str`):
      - scalar string  → coerce via action.type (so `data_dir = "data"` becomes Path)
      - list of strings on an `action="append"` flag → map element-wise
      - anything else is passed through (TOML ints / floats / bools / pre-parsed
        lists are trusted)
    """
    actions_by_dest = {a.dest: a for a in parser._actions if a.dest != argparse.SUPPRESS}
    unknown = sorted(set(cfg) - set(actions_by_dest))
    if unknown:
        raise SystemExit(f"--config: unknown keys {unknown}; known: {sorted(actions_by_dest)}")
    coerced: dict[str, Any] = {}
    for k, v in cfg.items():
        action = actions_by_dest[k]
        coerced[k] = _coerce_config_value(k, v, action)
    parser.set_defaults(**coerced)


def _coerce_config_value(key: str, v: Any, action: argparse.Action) -> Any:
    is_append = type(action).__name__ == "_AppendAction"
    has_type = callable(action.type) and action.type is not str
    try:
        if isinstance(v, str) and has_type:
            return action.type(v)
        if isinstance(v, list) and is_append and has_type:
            return [action.type(x) if isinstance(x, str) else x for x in v]
    except Exception as e:
        raise SystemExit(f"--config: failed to coerce {key}={v!r}: {e}")
    return v


def iter_with_progress(items, label: str = "items", every: int | None = None):
    """Yield from `items` while printing '  ... i/n label' to stderr.

    `every` is the heartbeat stride. Default `None` derives ~20 ticks across
    the run (`max(20, n // 20)`); pass a fixed int (e.g. `every=500`) when the
    work is fast enough that you want a coarser rhythm regardless of dataset
    size. The final tick at `i == n` always fires.

    Works for any sized iterable that supports `len()`. The heartbeat fires
    *after* each yielded body, so `continue`-skipped items still count.
    """
    n = len(items)
    step = every if every is not None else max(20, n // 20)
    for i, item in enumerate(items, start=1):
        yield item
        if i % step == 0 or i == n:
            print(f"  ... {i}/{n} {label}", file=sys.stderr)


@contextmanager
def stage_timer(name: str) -> Iterator[dict[str, float]]:
    """Wall-time the enclosed block; print 'elapsed=…' to stderr on exit.

    Yields a dict that gains an 'elapsed_sec' key after the block ends, so the
    caller can fold it into run_meta.json without re-measuring.
    """
    out: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        elapsed = time.perf_counter() - t0
        out["elapsed_sec"] = elapsed
        print(f"[{name}] elapsed={elapsed:.3f}s ({_fmt_elapsed(elapsed)})", file=sys.stderr)


def save_run_meta(
    out_dir: Path,
    stage: str,
    args: argparse.Namespace,
    *,
    elapsed_sec: float | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Dump resolved args + timing + git SHA to `out_dir/run_meta_<stage>.json`.

    Stage suffix avoids clobbering when multiple stages share an output dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "stage": stage,
        "args": {k: _json_safe(v) for k, v in vars(args).items()},
    }
    if elapsed_sec is not None:
        payload["elapsed_sec"] = round(elapsed_sec, 3)
    sha = _git_sha()
    if sha is not None:
        payload["git_sha"] = sha
    if extra:
        payload["extra"] = extra
    path = out_dir / f"run_meta_{stage}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _json_safe(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    return v


def _git_sha() -> str | None:
    """`<sha>` or `<sha>-dirty`. None if not a git checkout / git unavailable."""
    repo = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            timeout=2,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo,
                text=True,
                timeout=2,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except Exception:
        return None
    return f"{sha}-dirty" if dirty else sha


def _fmt_elapsed(s: float) -> str:
    if s < 60:
        return f"{s:.2f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m{s:04.1f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m{int(s):02d}s"
