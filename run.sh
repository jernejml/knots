#!/usr/bin/env bash
# End-to-end pipeline for the knots project.
#
# Stages (canonical order):
#   clean      wipe out/, cpp/build/, host Python caches
#   nuke       clean + docker image rm knots-{data,train,infer}
#   prepare    prepare.py (board partitions)     [knots-data]
#   sam        sam_polygons.py                   [knots-train, GPU]
#   train      train_yolo.py (incl. ONNX export) [knots-train, GPU]
#   infer      knots run (per-board polygons)    [knots-infer, GPU]
#   eval       knots eval (rebuilds GT + metrics)[knots-infer]
#   viz        stitched board overlays (JPEG)    [knots-data]
#
# infer and eval map to the two task-brief deliverables: infer produces
# "polygon bounds of all knots on each individual board" (writes
# out/boards/pred/<board>.json); eval is the "test mode which compares the
# outputs to already annotated scans" (writes eval_boards.json next to the
# latest train run, or under out/analysis/ if there isn't one).
#
# eval consumes the per-board JSONs that infer writes (out/boards/pred/) and
# rebuilds the matching per-board GT (out/boards/gt/) from raw labels in the
# same pass. Inference runs exactly once per pipeline pass and eval is cheap
# (seconds, no GPU). ./run.sh eval needs infer to have run first.
#
# Usage:
#   ./run.sh                       # full pipeline (all pipeline stages)
#   ./run.sh all                   # same, explicit
#   ./run.sh sam train infer eval  # subset; canonical order is enforced
#                                  # regardless of how you type the tokens
#   ./run.sh infer                 # produce per-board polygons; reuses the
#                                  # existing out/models/model.onnx
#   ./run.sh eval                  # metrics; same model, plus GT labels
#   ./run.sh clean all             # nuke local artefacts, then re-run from zero
#   ./run.sh nuke all              # also remove docker images, then re-run
#   ./run.sh --help                # stage list + env-var help
#
# Image builds are demand-driven: only images that the selected stages need
# are built (e.g. `./run.sh infer eval` skips knots-data and knots-train).
#
# clean / nuke prompt for confirmation when stdin is a TTY. Skip the prompt
# with FORCE=1 (useful in CI or one-shot scripts).
#
# Env-var knobs:
#   SPLIT=test ./run.sh eval       # restrict infer/eval to a named split.
#                                  # Default 'all' processes every board
#                                  # under data/images/. Use SPLIT=test for
#                                  # a clean held-out metric signal during
#                                  # model iteration; the brief's two
#                                  # deliverables both want all boards.
#   SKIP_BUILD=1 ./run.sh          # skip the image-build phase entirely
#   FORCE=1 ./run.sh clean         # suppress the destructive-action prompt
#   CONFIG=configs/foo.toml ./run.sh   # alternate TOML; empty disables
#                                  # --config (default: configs/default.toml)
#   RUN_NAME=iter5 ./run.sh        # ultralytics 'name' for the train run;
#                                  # the exported ONNX and eval JSON
#                                  # co-locate under out/runs/segment/$RUN_NAME/.
#
# All artefacts land under ./out/. Re-runs are idempotent: each script skips
# work whose outputs already exist (see each script's --force flag for how
# to override).
#
# Requires Docker + NVIDIA Container Toolkit on the host (CUDA 12.8 base).
#
# The PyTorch stages (sam, train) need --ipc=host so DataLoader
# workers can use the host's /dev/shm; the Docker default of 64 MB causes
# the workers to crash with a 'bus error' once batches start prefetching.
# Equivalent: --shm-size=8g (bounded but more explicit).

set -euo pipefail

cd "$(dirname "$0")"   # cd to repo root

# --- Config & flags ---------------------------------------------------------

SPLIT="${SPLIT:-all}"
SKIP_BUILD="${SKIP_BUILD:-0}"
FORCE="${FORCE:-0}"
CONFIG="${CONFIG:-configs/default.toml}"
RUN_NAME="${RUN_NAME:-}"

# CONFIG_ARGS is appended to every Python invocation. Empty CONFIG opts out
# of --config entirely (scripts fall back to argparse defaults). The configs
# dir is bind-mounted read-only into every container so /work/$CONFIG resolves.
CONFIG_ARGS=()
if [[ -n "$CONFIG" ]]; then
    [[ -f "$CONFIG" ]] || { echo "CONFIG=$CONFIG not found on host" >&2; exit 2; }
    CONFIG_ARGS=(--config "/work/$CONFIG")
fi

# Empty RUN_NAME leaves train_yolo's config/argparse default in place.
TRAIN_NAME_ARGS=()
[[ -n "$RUN_NAME" ]] && TRAIN_NAME_ARGS=(--name "$RUN_NAME")

# --- Stage registry ---------------------------------------------------------

# Canonical order. clean/nuke run before any pipeline work; `all` expands to
# the pipeline subset only (you have to ask for clean/nuke explicitly).
PIPELINE_STAGES=(prepare sam train infer eval viz)
ALL_STAGES=(clean nuke "${PIPELINE_STAGES[@]}")

# Map each pipeline stage to the docker image it needs. clean/nuke don't
# have entries here; the image-build loop guards against missing keys.
declare -A STAGE_IMAGE=(
    [prepare]=knots-data
    [sam]=knots-train
    [train]=knots-train
    [infer]=knots-infer
    [eval]=knots-infer
    [viz]=knots-data
)

CLEAN_TARGETS=(out cpp/build scripts/__pycache__ .ruff_cache)
NUKE_IMAGES=(knots-data knots-train knots-infer)

# --- Helpers ----------------------------------------------------------------

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

# After training, locate the run dir ultralytics wrote into. Used to place
# the eval JSON alongside the run's weights/ONNX so each run owns its artefacts.
# Returns "" if no run_meta_train_yolo.json is found.
find_latest_train_run_dir() {
    local latest
    latest=$(ls -1t out/runs/segment/*/run_meta_train_yolo.json 2>/dev/null | head -1)
    [[ -n "$latest" ]] && dirname "$latest"
}

# Returns 0 if $1 appears in the remaining args. Used to test stage membership
# without iterating manually each time.
in_list() {
    local needle="$1"; shift
    local x
    for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
    return 1
}

# --- Stage functions --------------------------------------------------------

stage_clean() {
    log "clean: removing ${CLEAN_TARGETS[*]}"
    rm -rf "${CLEAN_TARGETS[@]}"
}

stage_nuke() {
    # The pre-dispatch confirmation block already covered both the file wipe
    # and the image removal. stage_clean is idempotent so calling it again
    # when a prior `clean` stage already ran is a cheap no-op.
    stage_clean
    log "nuke: docker image rm ${NUKE_IMAGES[*]}"
    docker image rm -f "${NUKE_IMAGES[@]}" 2>/dev/null || true
}

stage_prepare() {
    log "prepare: stratified board partitions → out/analysis/partitions.json"
    docker run --rm \
        -v "$PWD/data:/work/data:ro" \
        -v "$PWD/out:/work/out" \
        -v "$PWD/configs:/work/configs:ro" \
        knots-data python3 scripts/prepare.py "${CONFIG_ARGS[@]}"
}

stage_sam() {
    log "SAM2 polygon upgrade (slow — minutes on GPU)"
    docker run --rm --gpus all --ipc=host \
        -v "$PWD/data:/work/data:ro" \
        -v "$PWD/out:/work/out" \
        -v "$PWD/configs:/work/configs:ro" \
        knots-train python3 scripts/sam_polygons.py "${CONFIG_ARGS[@]}"
}

stage_train() {
    log "train YOLOv11-seg (longest stage — hours on a single GPU)"
    docker run --rm --gpus all --ipc=host \
        -v "$PWD/data:/work/data:ro" \
        -v "$PWD/out:/work/out" \
        -v "$PWD/configs:/work/configs:ro" \
        knots-train python3 scripts/train_yolo.py \
        "${CONFIG_ARGS[@]}" "${TRAIN_NAME_ARGS[@]}"
}

stage_infer() {
    # Produces the task brief's primary deliverable: per-board polygons
    # written to out/boards/pred/<board>.json. SPLIT honoured for iteration;
    # default 'all' is what the brief asks for.
    local splits_args=()
    if [[ "$SPLIT" != "all" ]]; then
        splits_args=(--partitions-json /work/out/analysis/partitions.json --split "$SPLIT")
    fi
    log "knots run: per-frame inference + per-board stitching, split=$SPLIT"
    docker run --rm --gpus all \
        -v "$PWD/data:/work/data:ro" \
        -v "$PWD/out:/work/out" \
        -v "$PWD/configs:/work/configs:ro" \
        knots-infer knots run \
        --model /work/out/models/model.onnx \
        --input-dir /work/data/images \
        --output-dir /work/out/boards/pred \
        "${splits_args[@]}"
    log "done — per-board polygons in out/boards/pred/"
}

stage_viz() {
    # Reviewer-friendly per-board JPEGs: source frames stitched into one wide
    # image with predicted polygons (green) and GT polygons (red) overlaid.
    # GT overlay is opportunistic — falls back to pred-only if out/boards/gt/
    # is empty (precommitted-model demo case).
    log "visualize_boards: stitched overlays of pred + GT polygons"
    docker run --rm \
        -v "$PWD/data:/work/data:ro" \
        -v "$PWD/out:/work/out" \
        -v "$PWD/configs:/work/configs:ro" \
        knots-data python3 scripts/visualize_boards.py "${CONFIG_ARGS[@]}"
    log "done — board overlays in out/boards/viz/"
}

stage_eval() {
    # Compare out/boards/pred/ (from `infer`) with out/boards/gt/. Eval
    # rebuilds GT from data/labels into out/boards/gt before comparing
    # (skip-if-exists), so a separate gt-stitch step is no longer needed.
    # No model, no inference, no GPU — bbox match + mask IoU on the cached
    # per-board JSONs. Cheap; safe to re-run while tweaking thresholds.
    #
    # Locate the training run dir so the eval JSON co-locates with the
    # weights/ONNX/run_meta_*.json files. If nothing's there, eval falls
    # back to its built-in default of out/analysis/eval_boards.json.
    local run_dir eval_out_args=()
    run_dir=$(find_latest_train_run_dir || true)
    if [[ -n "$run_dir" ]]; then
        eval_out_args=(--out "/work/${run_dir}/eval_boards.json")
    fi

    log "knots eval: rebuild GT + compare against out/boards/pred"
    docker run --rm \
        -v "$PWD/data:/work/data:ro" \
        -v "$PWD/out:/work/out" \
        knots-infer knots eval \
        --pred-dir /work/out/boards/pred \
        --gt-dir /work/out/boards/gt \
        --labels-dir /work/data/labels \
        --images-dir /work/data/images \
        "${eval_out_args[@]}"

    if [[ -n "$run_dir" ]]; then
        log "done — metrics in ${run_dir}/eval_boards.json"
    else
        log "done — metrics in out/analysis/eval_boards.json (no train run dir found)"
    fi
}

# --- Dispatch ---------------------------------------------------------------

print_help() {
    cat <<EOF
Usage: ./run.sh [STAGE ...]

Stages (canonical order; tokens may be given in any order on the command line):
  clean      wipe out/, cpp/build/, host Python caches
  nuke       clean + docker image rm knots-{data,train,infer}
  prepare    prepare.py (board partitions)     [knots-data]
  sam        sam_polygons.py                   [knots-train, GPU]
  train      train_yolo.py (incl. ONNX export) [knots-train, GPU]
  infer      knots run (per-board polygons)    [knots-infer, GPU]
  eval       knots eval (rebuilds GT + metrics)[knots-infer]
  viz        stitched board overlays (JPEG)    [knots-data]
  all        every pipeline stage above (also the default with no args)

Examples:
  ./run.sh                     # full pipeline
  ./run.sh infer               # per-board polygons; reuses out/models/model.onnx
  ./run.sh infer eval viz      # polygons + metrics + reviewer-friendly overlays
  ./run.sh eval                # metrics; rebuilds GT under out/boards/gt/ if missing
  ./run.sh train infer eval    # re-train (re-exports ONNX), re-run both modes
  ./run.sh clean all           # wipe artefacts, then re-run from zero
  FORCE=1 ./run.sh clean       # skip the destructive-action prompt (CI)
  SPLIT=test ./run.sh eval     # restrict to test split (default: all boards)

Env vars: SPLIT, CONFIG, RUN_NAME, SKIP_BUILD, FORCE (see file header).
EOF
}

# Build the canonical, deduped stage list from positional args.
declare -a STAGES=()
if [[ $# -eq 0 ]]; then
    STAGES=("${PIPELINE_STAGES[@]}")
else
    declare -A want=()
    for arg in "$@"; do
        case "$arg" in
            -h|--help) print_help; exit 0 ;;
            all) for s in "${PIPELINE_STAGES[@]}"; do want["$s"]=1; done ;;
            clean|nuke|prepare|sam|train|infer|viz|eval)
                want["$arg"]=1 ;;
            *) echo "unknown stage: $arg" >&2; echo >&2; print_help >&2; exit 2 ;;
        esac
    done
    for s in "${ALL_STAGES[@]}"; do
        [[ -n "${want[$s]:-}" ]] && STAGES+=("$s")
    done
fi

# --- Destructive-stage confirmation ----------------------------------------
# Show one combined prompt that covers both clean and nuke (if either is in
# the stage list). Skip the prompt entirely if FORCE=1 or stdin is not a TTY
# (the second case keeps CI scriptable without surprises).

destructive_files=()
destructive_images=()
if in_list clean "${STAGES[@]}" || in_list nuke "${STAGES[@]}"; then
    for t in "${CLEAN_TARGETS[@]}"; do
        [[ -e "$t" ]] && destructive_files+=("$t")
    done
    if in_list nuke "${STAGES[@]}"; then
        for img in "${NUKE_IMAGES[@]}"; do
            image_exists "$img" && destructive_images+=("$img")
        done
    fi
    if (( ${#destructive_files[@]} + ${#destructive_images[@]} > 0 )); then
        echo "About to delete:" >&2
        (( ${#destructive_files[@]}  > 0 )) && echo "  files:  ${destructive_files[*]}"  >&2
        (( ${#destructive_images[@]} > 0 )) && echo "  images: ${destructive_images[*]}" >&2
        if [[ "$FORCE" != "1" && -t 0 ]]; then
            read -r -p "Proceed? [y/N] " ans
            [[ "$ans" =~ ^[Yy]$ ]] || { echo "aborted" >&2; exit 1; }
        fi
    else
        log "clean/nuke: nothing to remove"
    fi
fi

# Demand-driven image builds: only build the images the selected stages need.
# clean/nuke have no STAGE_IMAGE entry, so guard the lookup with :-.
declare -A needed=()
for s in "${STAGES[@]}"; do
    img="${STAGE_IMAGE[$s]:-}"
    [[ -n "$img" ]] && needed["$img"]=1
done

[[ -n "${needed[knots-data]:-}"  ]] && build_image knots-data  docker/Dockerfile.data
[[ -n "${needed[knots-train]:-}" ]] && build_image knots-train docker/Dockerfile.train
[[ -n "${needed[knots-infer]:-}" ]] && build_image knots-infer docker/Dockerfile.infer

# Run each stage in canonical order.
for s in "${STAGES[@]}"; do
    "stage_$s"
done
