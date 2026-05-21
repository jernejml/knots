#!/usr/bin/env bash
# prerequisites.sh — host readiness check for the knots pipeline.
#
# Default mode runs the load-bearing layers (1-4): shell/OS, docker, NVIDIA
# stack, and a docker --gpus all probe. Any failure here means run.sh will
# not work; the script exits 1 and tells you what to fix.
#
# --full additionally runs layer 5 (resources) and layer 6 (repo state).
# Those checks emit WARNs only — the pipeline can still run, but you might
# OOM or have nothing to process.
#
# The layer-4 probe runs `docker run --rm --gpus all nvidia/cuda:...
# nvidia-smi`. That's the only check that proves the full chain (docker
# daemon → NVIDIA Container Toolkit → driver → CUDA runtime) is wired
# correctly; individual layer-2/3 checks can all pass while this one fails
# (e.g. if /etc/docker/daemon.json was never updated by
# `nvidia-ctk runtime configure`). On a cold host it pulls a ~150 MB base
# image once, so the script asks before doing so.
#
# Usage:
#   ./prerequisites.sh                  # default: layers 1-4
#   ./prerequisites.sh --full           # also layers 5-6 (warn-only)
#   ./prerequisites.sh --skip-docker-probe
#   ./prerequisites.sh -y               # auto-accept the base-image pull
#
# Exit codes:
#   0 — all FAIL-class checks passed (may include WARNs in --full mode)
#   1 — at least one critical check failed
#   2 — invalid arguments / not run from repo root

set -uo pipefail   # no -e: we want to gather all failures, not stop at the first

cd "$(dirname "$0")"

# --- Flag parsing -----------------------------------------------------------

FULL=0
SKIP_PROBE=0
ASSUME_YES=0

print_help() {
    cat <<'EOF'
Usage: ./prerequisites.sh [OPTIONS]

Default mode runs layers 1-4 (load-bearing). FAIL → exit 1.

Options:
  --full                 also run layer 5 (resources) and layer 6 (repo state)
  --skip-docker-probe    skip the docker --gpus all probe (layer 4)
  -y, --yes              auto-accept the layer-4 base-image pull prompt
  -h, --help             show this message
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --full)              FULL=1 ;;
        --skip-docker-probe) SKIP_PROBE=1 ;;
        -y|--yes)            ASSUME_YES=1 ;;
        -h|--help)           print_help; exit 0 ;;
        *) echo "unknown flag: $1" >&2; print_help >&2; exit 2 ;;
    esac
    shift
done

# --- Repo-root sanity (cheap, before any layer) -----------------------------

if [[ ! -f docker/Dockerfile.data || ! -f run.sh ]]; then
    echo "prerequisites.sh: not in repo root (docker/Dockerfile.data not found)" >&2
    exit 2
fi

# --- Output helpers ---------------------------------------------------------

if [[ -t 1 ]]; then
    C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'
    C_DIM=$'\033[2m';    C_RESET=$'\033[0m'
else
    C_GREEN=""; C_YELLOW=""; C_RED=""; C_DIM=""; C_RESET=""
fi

FAILS=0
WARNS=0

section() { printf '\n=== %s ===\n' "$*"; }
ok()      { printf '  [%sOK%s]   %s\n'   "$C_GREEN"  "$C_RESET" "$*"; }
warn()    { printf '  [%sWARN%s] %s\n'   "$C_YELLOW" "$C_RESET" "$*"; WARNS=$((WARNS+1)); }
fail()    { printf '  [%sFAIL%s] %s\n'   "$C_RED"    "$C_RESET" "$*"; FAILS=$((FAILS+1)); }
info()    { printf '  [%sINFO%s] %s\n'   "$C_DIM"    "$C_RESET" "$*"; }

# --- Layer 1: Shell & OS ----------------------------------------------------

check_bash_version() {
    local v="${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]}"
    if (( BASH_VERSINFO[0] >= 4 )); then
        ok "bash $v"
    else
        fail "bash $v — need ≥ 4.0 (run.sh uses declare -A and other bash-4 features)"
    fi
}

check_linux() {
    local kernel; kernel=$(uname -s)
    if [[ "$kernel" == "Linux" ]]; then
        ok "OS: Linux $(uname -r)"
    else
        fail "OS: $kernel — docker --gpus all only works on Linux (and WSL2 on Windows)"
    fi
}

# --- Layer 2: Docker --------------------------------------------------------

check_docker_daemon() {
    if ! command -v docker >/dev/null; then
        fail "docker not installed — see https://docs.docker.com/engine/install/"
        return
    fi
    if ! docker info >/dev/null 2>&1; then
        local err; err=$(docker info 2>&1 || true)
        if grep -qiE 'permission denied|/var/run/docker\.sock' <<<"$err"; then
            fail "docker installed but user lacks permission — add yourself to the 'docker' group, or use rootless docker"
        elif grep -qiE 'cannot connect|connection refused|no such file' <<<"$err"; then
            fail "docker daemon not running — try: sudo systemctl start docker (or systemctl --user start docker for rootless)"
        else
            fail "docker info failed: $(head -1 <<<"$err")"
        fi
        return
    fi
    local cver sver
    cver=$(docker version --format '{{.Client.Version}}' 2>/dev/null || echo '?')
    sver=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo '?')
    ok "docker daemon reachable (client $cver, server $sver)"
    # Check server version ≥ 20.10 (BuildKit default).
    local sv_major; sv_major=${sver%%.*}
    if [[ "$sv_major" =~ ^[0-9]+$ ]] && (( sv_major < 20 )); then
        warn "docker server $sver — recommend ≥ 20.10 for BuildKit defaults"
    fi
}

# --- Layer 3: NVIDIA --------------------------------------------------------

check_nvidia_driver() {
    if ! command -v nvidia-smi >/dev/null; then
        fail "nvidia-smi not found — NVIDIA driver missing"
        return
    fi
    if ! nvidia-smi >/dev/null 2>&1; then
        fail "nvidia-smi present but fails to run — driver likely broken; inspect 'nvidia-smi' output"
        return
    fi
    local driver; driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    if [[ -z "$driver" ]]; then
        warn "couldn't parse driver version from nvidia-smi"
    else
        local major=${driver%%.*}
        if (( major >= 570 )); then
            ok "NVIDIA driver $driver (CUDA 12.8 supported)"
        elif (( major >= 525 )); then
            warn "NVIDIA driver $driver — CUDA 12.8 may work via forward-compat but driver ≥ 570 recommended"
        else
            fail "NVIDIA driver $driver — CUDA 12.8 needs driver ≥ 570 (forward-compat unlikely to cover this gap)"
        fi
    fi
    local gpu; gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    [[ -n "$gpu" ]] && info "GPU: $gpu"
}

check_nvidia_container_toolkit() {
    if command -v nvidia-ctk >/dev/null; then
        local v; v=$(nvidia-ctk --version 2>/dev/null | head -1)
        ok "NVIDIA Container Toolkit: $v"
    elif command -v nvidia-container-cli >/dev/null; then
        ok "nvidia-container-cli present (older NCT release)"
    else
        fail "NVIDIA Container Toolkit not found — install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    fi
}

# --- Layer 4: End-to-end GPU probe -----------------------------------------

check_docker_gpu_probe() {
    if (( SKIP_PROBE )); then
        info "docker GPU probe: skipped (--skip-docker-probe)"
        return
    fi
    # Don't bother probing if the upstream layers already failed.
    if ! docker info >/dev/null 2>&1; then
        warn "docker GPU probe: skipped (docker daemon unreachable)"
        return
    fi
    if ! command -v nvidia-smi >/dev/null; then
        warn "docker GPU probe: skipped (no NVIDIA driver on host)"
        return
    fi

    local img="nvidia/cuda:12.8.0-base-ubuntu24.04"
    if ! docker image inspect "$img" >/dev/null 2>&1; then
        cat >&2 <<EOF

  The end-to-end GPU probe needs base image '$img'.
  It's not cached locally, so it will pull (~150 MB).

  Why this probe matters: it's the only check that proves the full chain
  (docker daemon → nvidia-container-toolkit → driver → CUDA runtime) is
  wired correctly. Individual layer 1-3 checks can all pass while this
  one fails — e.g. when the daemon was never reconfigured to register
  the nvidia runtime (\`sudo nvidia-ctk runtime configure --runtime=docker\`
  + daemon restart).

EOF
        if (( ASSUME_YES )); then
            info "proceeding with pull (-y / --yes)"
        elif [[ ! -t 0 ]]; then
            warn "docker GPU probe: skipped — non-interactive shell and image not cached. Re-run with -y or --skip-docker-probe."
            return
        else
            local ans
            read -r -p "  Pull and run? [y/N] " ans
            if [[ ! "$ans" =~ ^[Yy]$ ]]; then
                warn "docker GPU probe: skipped — user declined pull"
                return
            fi
        fi
    fi

    if docker run --rm --gpus all "$img" nvidia-smi >/dev/null 2>&1; then
        ok "docker GPU probe: --gpus all → nvidia-smi succeeded"
    else
        local err; err=$(docker run --rm --gpus all "$img" nvidia-smi 2>&1 || true)
        fail "docker GPU probe failed: $(tail -3 <<<"$err" | tr '\n' ' | ')"
    fi
}

# --- Layer 5: Resources (warn-only) -----------------------------------------

check_disk() {
    local docker_root avail
    if docker info >/dev/null 2>&1; then
        docker_root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo "/var/lib/docker")
    else
        docker_root="/var/lib/docker"
    fi
    if [[ ! -d "$docker_root" ]]; then
        warn "disk: docker root '$docker_root' not found, skipping check"
        return
    fi
    avail=$(df --output=avail -BG "$docker_root" 2>/dev/null | tail -1 | tr -dc '0-9')
    if [[ -z "$avail" ]]; then
        warn "disk: couldn't query free space at $docker_root"
        return
    fi
    if (( avail >= 30 )); then
        ok "disk: ${avail}G free at $docker_root"
    elif (( avail >= 10 )); then
        warn "disk: only ${avail}G free at $docker_root (≥ 30G recommended; the three images total ~15G + cache)"
    else
        warn "disk: ${avail}G free at $docker_root — likely insufficient for a clean build"
    fi
}

check_gpu_memory() {
    if ! command -v nvidia-smi >/dev/null; then return; fi
    local mb; mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [[ -z "$mb" || ! "$mb" =~ ^[0-9]+$ ]]; then
        warn "GPU memory: couldn't parse from nvidia-smi"
        return
    fi
    local gb=$((mb / 1024))
    if (( gb >= 8 )); then
        ok "GPU memory: ${gb}G (sufficient for SAM2-L + YOLO11n at batch=32)"
    elif (( gb >= 4 )); then
        warn "GPU memory: ${gb}G — may OOM on SAM2-L or YOLO batch=32; reduce batch in configs/default.toml"
    else
        warn "GPU memory: ${gb}G — training likely won't fit; inference may still work"
    fi
}

check_ram() {
    local kb; kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null)
    if [[ -z "$kb" ]]; then warn "RAM: couldn't read /proc/meminfo"; return; fi
    local gb=$((kb / 1024 / 1024))
    if   (( gb >= 16 )); then ok   "RAM: ${gb}G"
    elif (( gb >= 8  )); then warn "RAM: ${gb}G — workable, 16G recommended for training"
    else                       warn "RAM: ${gb}G — tight; training may swap heavily"
    fi
}

check_shm() {
    if [[ ! -d /dev/shm ]]; then
        warn "/dev/shm: missing — PyTorch DataLoaders will bus-error in training"
        return
    fi
    local mb; mb=$(df -m /dev/shm 2>/dev/null | tail -1 | awk '{print $2}')
    if [[ -z "$mb" || ! "$mb" =~ ^[0-9]+$ ]]; then
        warn "/dev/shm: couldn't size"; return
    fi
    local gb=$((mb / 1024))
    if (( gb >= 4 )); then ok "/dev/shm: ${gb}G"
    else                   warn "/dev/shm: ${gb}G — small; sam/train may bus-error even with --ipc=host"
    fi
}

# --- Layer 6: Repo state (warn-only) ---------------------------------------

check_repo_data() {
    if [[ ! -d data/images ]]; then
        warn "data/images/ missing — pipeline has nothing to process"
    else
        local n; n=$(find data/images -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
        if (( n > 0 )); then ok "data/images/: $n PNG frame(s)"
        else                 warn "data/images/ exists but contains no PNGs"
        fi
    fi
    if [[ ! -d data/labels ]]; then
        warn "data/labels/ missing — train / eval / gt-stitch can't run"
    else
        local n; n=$(find data/labels -maxdepth 1 -name '*.txt' 2>/dev/null | wc -l)
        if (( n > 0 )); then ok "data/labels/: $n label file(s)"
        else                 warn "data/labels/ exists but contains no .txt files"
        fi
    fi
}

# --- Driver -----------------------------------------------------------------

section "Layer 1 — Shell & OS"
check_bash_version
check_linux

section "Layer 2 — Docker"
check_docker_daemon

section "Layer 3 — NVIDIA"
check_nvidia_driver
check_nvidia_container_toolkit

section "Layer 4 — End-to-end GPU probe"
check_docker_gpu_probe

if (( FULL )); then
    section "Layer 5 — Resources (warn-only)"
    check_disk
    check_gpu_memory
    check_ram
    check_shm

    section "Layer 6 — Repo state (warn-only)"
    check_repo_data
fi

printf '\n=== Summary ===\n'
if (( FAILS > 0 )); then
    printf '  %sFAIL%s: %d critical, %d warning(s)\n' "$C_RED" "$C_RESET" "$FAILS" "$WARNS"
    printf '  Fix the FAIL items above before running ./run.sh.\n'
    exit 1
elif (( WARNS > 0 )); then
    printf '  %sOK%s with %d warning(s) — should be runnable; review WARNs above.\n' "$C_GREEN" "$C_RESET" "$WARNS"
    exit 0
else
    printf '  %sOK%s — all checks passed.\n' "$C_GREEN" "$C_RESET"
    exit 0
fi
