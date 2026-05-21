#!/usr/bin/env bash
# End-to-end pipeline for the knots project.
#
# With no arguments, runs the full chain on the default test split:
#   1. Build docker images (skipped if already present locally).
#   2. Analyse frames + annotations.
#   3. Aggregate per-board features.
#   4. Stratified train/val/test split.
#   5. SAM2 polygon upgrade of the rectangle labels.
#   6. YOLOv11-seg training.
#   7. ONNX export.
#   8. Test-mode evaluation (Mode B: model + images-dir + labels-dir).
#
# All artefacts land under ./out/. Re-runs are idempotent: each script skips
# work whose outputs already exist (see each script's --force flag for how
# to override).
#
# Usage:
#   ./run.sh                       # full pipeline, --split test
#   SPLIT=val ./run.sh             # evaluate on val instead of test
#   SKIP_BUILD=1 ./run.sh          # don't even check whether images need building
#   SKIP_TRAIN=1 ./run.sh          # reuse existing best.pt under out/runs/segment/
#   CONFIG=configs/foo.toml ./run.sh   # use a different TOML; empty disables --config
#                                  # (default: configs/default.toml)
#
# Requires Docker + NVIDIA Container Toolkit on the host (CUDA 12.8 base).
#
# The PyTorch stages (sam_polygons, train_yolo, export_onnx) need --ipc=host
# so DataLoader workers can use the host's /dev/shm; the Docker default of
# 64 MB causes the workers to crash with a 'bus error' once batches start
# prefetching. Equivalent: --shm-size=8g (bounded but more explicit).

set -euo pipefail

cd "$(dirname "$0")"   # cd to repo root

SPLIT="${SPLIT:-test}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
CONFIG="${CONFIG:-configs/default.toml}"

# CONFIG_ARGS is appended to every Python invocation. Empty CONFIG opts out of
# --config entirely (scripts fall back to argparse defaults). The configs dir
# is bind-mounted read-only into every container so /work/$CONFIG resolves.
CONFIG_ARGS=()
if [[ -n "$CONFIG" ]]; then
    [[ -f "$CONFIG" ]] || { echo "CONFIG=$CONFIG not found on host" >&2; exit 2; }
    CONFIG_ARGS=(--config "/work/$CONFIG")
fi

log() { printf '\n=== %s ===\n' "$*" >&2; }

image_exists() { docker image inspect "$1" >/dev/null 2>&1; }

build_image() {
    local tag="$1" file="$2"
    if [[ "$SKIP_BUILD" == "1" ]] || image_exists "$tag"; then
        log "skip: $tag already present (SKIP_BUILD=1 to force-skip; rebuild manually if stale)"
        return
    fi
    log "build $tag"
    docker build -f "$file" -t "$tag" .
}

# --- 1. Build images --------------------------------------------------------

build_image knots-data  docker/Dockerfile.data
build_image knots-train docker/Dockerfile.train
build_image knots-infer docker/Dockerfile.infer

# --- 2-4. Analysis + per-board features + stratified split ------------------

log "analyse frames + annotations"
docker run --rm \
    -v "$PWD/data:/work/data:ro" \
    -v "$PWD/out:/work/out" \
    -v "$PWD/configs:/work/configs:ro" \
    knots-data python3 scripts/analyze_dataset.py "${CONFIG_ARGS[@]}"

log "aggregate per-board features"
docker run --rm \
    -v "$PWD/data:/work/data:ro" \
    -v "$PWD/out:/work/out" \
    -v "$PWD/configs:/work/configs:ro" \
    knots-data python3 scripts/board_features.py "${CONFIG_ARGS[@]}"

log "stratified train/val/test split"
docker run --rm \
    -v "$PWD/data:/work/data:ro" \
    -v "$PWD/out:/work/out" \
    -v "$PWD/configs:/work/configs:ro" \
    knots-data python3 scripts/make_splits.py "${CONFIG_ARGS[@]}"

# --- 5. SAM2 polygon upgrade ------------------------------------------------

log "SAM2 polygon upgrade (slow — minutes on GPU)"
docker run --rm --gpus all --ipc=host \
    -v "$PWD/data:/work/data:ro" \
    -v "$PWD/out:/work/out" \
    -v "$PWD/configs:/work/configs:ro" \
    knots-train python3 scripts/sam_polygons.py "${CONFIG_ARGS[@]}"

# --- 6. YOLO training -------------------------------------------------------

if [[ "$SKIP_TRAIN" == "1" ]]; then
    log "skip training (SKIP_TRAIN=1) — expects existing best.pt under out/runs/segment/"
else
    log "train YOLOv11-seg (longest stage — hours on a single GPU)"
    docker run --rm --gpus all --ipc=host \
        -v "$PWD/data:/work/data:ro" \
        -v "$PWD/out:/work/out" \
        -v "$PWD/configs:/work/configs:ro" \
        knots-train python3 scripts/train_yolo.py "${CONFIG_ARGS[@]}"
fi

# --- 7. ONNX export ---------------------------------------------------------

log "export to ONNX"
docker run --rm --gpus all --ipc=host \
    -v "$PWD/data:/work/data:ro" \
    -v "$PWD/out:/work/out" \
    -v "$PWD/configs:/work/configs:ro" \
    knots-train python3 scripts/export_onnx.py "${CONFIG_ARGS[@]}"

# --- 8. Test-mode evaluation ------------------------------------------------

log "knots eval (Mode B): infer + GT-stitch + match in one pass, split=$SPLIT"
docker run --rm --gpus all \
    -v "$PWD/data:/work/data:ro" \
    -v "$PWD/out:/work/out" \
    -v "$PWD/configs:/work/configs:ro" \
    knots-infer knots eval \
    --model /work/out/models/best.onnx \
    --images-dir /work/data/images \
    --labels-dir /work/data/labels \
    --splits-csv /work/out/analysis/splits.csv --split "$SPLIT"

log "done — metrics in out/analysis/eval_boards.json"
