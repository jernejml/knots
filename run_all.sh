#!/usr/bin/env bash
# End-to-end pipeline for the knots project.
#
# Runs the six stages on the default test split:
#   1. Build docker images (skipped if already present locally).
#   2. Analyze dataset + stratified split.
#   3. SAM2 polygon upgrade of the rectangle labels.
#   4. YOLOv11-seg training.
#   5. ONNX export.
#   6. Test-mode evaluation (Mode B: model + images-dir + labels-dir).
#
# All artefacts land under ./out/. Re-runs are idempotent: scripts skip
# work whose outputs already exist (see each script's --force flag for
# how to override).
#
# Usage:
#   ./run_all.sh                # full pipeline, --split test
#   SPLIT=val ./run_all.sh      # evaluate on val instead of test
#   SKIP_TRAIN=1 ./run_all.sh   # reuse existing best.pt
#
# Requires Docker + NVIDIA Container Toolkit on the host (CUDA 12.8 base).

set -euo pipefail

cd "$(dirname "$0")"   # cd to repo root

SPLIT="${SPLIT:-test}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

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

# --- 2. Analysis + splits ---------------------------------------------------

log "analyze dataset + build stratified split"
docker run --rm \
    -v "$PWD/data:/work/data:ro" \
    -v "$PWD/out:/work/out" \
    knots-data make all

# --- 3. SAM2 polygon upgrade ------------------------------------------------

log "SAM2 polygon upgrade (slow — minutes on GPU)"
docker run --rm --gpus all \
    -v "$PWD/data:/work/data:ro" \
    -v "$PWD/out:/work/out" \
    knots-train python3 scripts/sam_polygons.py

# --- 4. YOLO training -------------------------------------------------------

if [[ "$SKIP_TRAIN" == "1" ]]; then
    log "skip training (SKIP_TRAIN=1) — expects existing best.pt under out/runs/segment/"
else
    log "train YOLOv11-seg (longest stage — hours on a single GPU)"
    docker run --rm --gpus all \
        -v "$PWD/data:/work/data:ro" \
        -v "$PWD/out:/work/out" \
        knots-train python3 scripts/train_yolo.py
fi

# --- 5. ONNX export ---------------------------------------------------------

log "export to ONNX"
docker run --rm --gpus all \
    -v "$PWD/data:/work/data:ro" \
    -v "$PWD/out:/work/out" \
    knots-train python3 scripts/export_onnx.py

# --- 6. Test-mode evaluation ------------------------------------------------

log "knots eval (Mode B): infer + GT-stitch + match in one pass, split=$SPLIT"
docker run --rm --gpus all \
    -v "$PWD/data:/work/data:ro" \
    -v "$PWD/out:/work/out" \
    knots-infer knots eval \
    --model /work/out/models/best.onnx \
    --images-dir /work/data/images \
    --labels-dir /work/data/labels \
    --splits-csv /work/out/analysis/splits.csv --split "$SPLIT"

log "done — metrics in out/analysis/eval_boards.json"
