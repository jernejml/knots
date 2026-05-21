#!/usr/bin/env bash
# End-to-end pipeline for the knots project.
#
# Stages (canonical order):
#   clean      wipe out/, cpp/build/, host Python caches
#   nuke       clean + docker image rm knots-{data,train,infer}
#   analyze    analyze_dataset.py        [knots-data]
#   features   board_features.py         [knots-data]
#   split      make_splits.py            [knots-data]
#   sam        sam_polygons.py           [knots-train, GPU]
#   train      train_yolo.py             [knots-train, GPU]
#   export     export_onnx.py            [knots-train, GPU]
#   infer      knots eval (metrics)      [knots-infer, GPU]
#
# Usage:
#   ./run.sh                       # full pipeline (all pipeline stages)
#   ./run.sh all                   # same, explicit
#   ./run.sh sam train infer       # subset; canonical order is enforced
#                                  # regardless of how you type the tokens
#   ./run.sh infer                 # just metrics against the existing
#                                  # out/models/best.onnx
#   ./run.sh clean all             # nuke local artefacts, then re-run from zero
#   ./run.sh nuke all              # also remove docker images, then re-run
#   ./run.sh --help                # stage list + env-var help
#
# Image builds are demand-driven: only images that the selected stages need
# are built (e.g. `./run.sh infer` skips knots-data and knots-train).
#
# clean / nuke prompt for confirmation when stdin is a TTY. Skip the prompt
# with FORCE=1 (useful in CI or one-shot scripts).
#
# Env-var knobs:
#   SPLIT=val ./run.sh infer       # evaluate on val instead of test
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
# The PyTorch stages (sam, train, export) need --ipc=host so DataLoader
# workers can use the host's /dev/shm; the Docker default of 64 MB causes
# the workers to crash with a 'bus error' once batches start prefetching.
# Equivalent: --shm-size=8g (bounded but more explicit).

set -euo pipefail

cd "$(dirname "$0")"   # cd to repo root

# --- Config & flags ---------------------------------------------------------

SPLIT="${SPLIT:-test}"
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
PIPELINE_STAGES=(analyze features split sam train export infer)
ALL_STAGES=(clean nuke "${PIPELINE_STAGES[@]}")

# Map each pipeline stage to the docker image it needs. clean/nuke don't
# have entries here; the image-build loop guards against missing keys.
declare -A STAGE_IMAGE=(
    [analyze]=knots-data
    [features]=knots-data
    [split]=knots-data
    [sam]=knots-train
    [train]=knots-train
    [export]=knots-train
    [infer]=knots-infer
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
# the eval JSON alongside best.pt/best.onnx so each run owns its artefacts.
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

stage_analyze() {
    log "analyze frames + annotations"
    docker run --rm \
        -v "$PWD/data:/work/data:ro" \
        -v "$PWD/out:/work/out" \
        -v "$PWD/configs:/work/configs:ro" \
        knots-data python3 scripts/analyze_dataset.py "${CONFIG_ARGS[@]}"
}

stage_features() {
    log "aggregate per-board features"
    docker run --rm \
        -v "$PWD/data:/work/data:ro" \
        -v "$PWD/out:/work/out" \
        -v "$PWD/configs:/work/configs:ro" \
        knots-data python3 scripts/board_features.py "${CONFIG_ARGS[@]}"
}

stage_split() {
    log "stratified train/val/test split"
    docker run --rm \
        -v "$PWD/data:/work/data:ro" \
        -v "$PWD/out:/work/out" \
        -v "$PWD/configs:/work/configs:ro" \
        knots-data python3 scripts/make_splits.py "${CONFIG_ARGS[@]}"
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

stage_export() {
    log "export to ONNX"
    docker run --rm --gpus all --ipc=host \
        -v "$PWD/data:/work/data:ro" \
        -v "$PWD/out:/work/out" \
        -v "$PWD/configs:/work/configs:ro" \
        knots-train python3 scripts/export_onnx.py "${CONFIG_ARGS[@]}"
}

stage_infer() {
    # Locate the training run dir whose model we're about to eval, so the
    # eval JSON co-locates with the weights/exported ONNX/run_meta_*.json
    # files. If nothing is found, eval falls back to its built-in default
    # of out/analysis/eval_boards.json.
    local run_dir eval_out_args=()
    run_dir=$(find_latest_train_run_dir || true)
    if [[ -n "$run_dir" ]]; then
        eval_out_args=(--out "/work/${run_dir}/eval_boards.json")
    fi

    log "knots eval (Mode B): infer + GT-stitch + match in one pass, split=$SPLIT"
    docker run --rm --gpus all \
        -v "$PWD/data:/work/data:ro" \
        -v "$PWD/out:/work/out" \
        -v "$PWD/configs:/work/configs:ro" \
        knots-infer knots eval \
        --model /work/out/models/best.onnx \
        --images-dir /work/data/images \
        --labels-dir /work/data/labels \
        --splits-csv /work/out/analysis/splits.csv --split "$SPLIT" \
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
  analyze    analyze_dataset.py        [knots-data]
  features   board_features.py         [knots-data]
  split      make_splits.py            [knots-data]
  sam        sam_polygons.py           [knots-train, GPU]
  train      train_yolo.py             [knots-train, GPU]
  export     export_onnx.py            [knots-train, GPU]
  infer      knots eval (metrics)      [knots-infer, GPU]
  all        every pipeline stage above (also the default with no args)

Examples:
  ./run.sh                     # full pipeline
  ./run.sh infer               # just metrics; reuses out/models/best.onnx
  ./run.sh train export infer  # re-train, re-export, re-eval
  ./run.sh clean all           # wipe artefacts, then re-run from zero
  FORCE=1 ./run.sh clean       # skip the destructive-action prompt (CI)
  SPLIT=val ./run.sh infer     # evaluate on val split

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
            clean|nuke|analyze|features|split|sam|train|export|infer)
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
