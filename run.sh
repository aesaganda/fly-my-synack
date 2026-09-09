#!/usr/bin/env bash
# Build and run the whole stack natively, no Docker.
#
#   ./run.sh setup            create .venv and install everything
#   ./run.sh test             run the test suite
#   ./run.sh sim [args...]    python run.py [args...]   (default: dry_land, 5000 steps)
#   ./run.sh web              serve the browser UI on http://localhost:8000
#   ./run.sh compare          cross-environment comparison
#   ./run.sh shell            drop into the venv's python
#
# `setup` runs automatically the first time, so `./run.sh test` on a clean
# checkout does the right thing.
#
# Environment overrides:
#   PIP_INDEX_URL   package mirror, if PyPI is blocked
#   TORCH_INDEX     torch wheel index (default: CPU-only build on Linux)
#   PYTHON          interpreter to build the venv with
#   PORT            web UI port (default 8000)

set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv
PY_BIN="$VENV/bin/python"
STAMP="$VENV/.deps-installed"

# flygym 2.1.0 requires >=3.12,<3.15. 3.12 is listed first deliberately: it is
# the version this stack was verified on, and the newest interpreters often lack
# wheels for mujoco/torch.
pick_python() {
    if [[ -n "${PYTHON:-}" ]]; then echo "$PYTHON"; return; fi
    for v in python3.12 python3.13 python3.14; do
        command -v "$v" >/dev/null 2>&1 && { echo "$v"; return; }
    done
    echo "no python3.12/3.13/3.14 found (flygym needs >=3.12,<3.15)." >&2
    echo "  macOS:  brew install python@3.12" >&2
    echo "  Debian: apt install python3.12 python3.12-venv" >&2
    exit 1
}

# Kept as two plain words rather than an array: macOS ships bash 3.2, which has
# no `mapfile` and no `${arr[@]+...}` niceties.
IDX=""
[[ -n "${PIP_INDEX_URL:-}" ]] && IDX="--index-url $PIP_INDEX_URL"

setup() {
    local py; py="$(pick_python)"
    echo "==> building $VENV with $py ($("$py" --version 2>&1))"
    [[ -d $VENV ]] || "$py" -m venv "$VENV"

    # shellcheck disable=SC2086  # $IDX is deliberately word-split
    "$PY_BIN" -m pip install -q --upgrade pip $IDX

    # torch is not in requirements.txt: on Linux the default wheel bundles ~3 GB
    # of CUDA libraries even for CPU-only use, so pull the +cpu build there.
    # macOS wheels have no CUDA and no +cpu variant, so use the normal index.
    local torch_index="${TORCH_INDEX:-}"
    if [[ -z "$torch_index" && "$(uname -s)" != "Darwin" ]]; then
        torch_index="https://download.pytorch.org/whl/cpu"
    fi
    echo "==> installing torch${torch_index:+ from $torch_index}"
    if [[ -n "$torch_index" ]]; then
        "$PY_BIN" -m pip install -q --index-url "$torch_index" \
            ${PIP_INDEX_URL:+--extra-index-url "$PIP_INDEX_URL"} torch==2.14.0
    else
        # shellcheck disable=SC2086
        "$PY_BIN" -m pip install -q $IDX torch==2.14.0
    fi

    echo "==> installing the rest"
    # shellcheck disable=SC2086
    "$PY_BIN" -m pip install -q $IDX -r requirements.txt

    touch "$STAMP"
    echo "==> ready: $("$PY_BIN" -c 'import flygym,mujoco,torch,importlib.metadata as m;
print("flygym", m.version("flygym"), "| mujoco", m.version("mujoco"), "| torch", torch.__version__)')"
}

# Python does not trust a TLS-inspecting corporate proxy even where curl does,
# which shows up as CERTIFICATE_VERIFY_FAILED on neuPrint fetches. On macOS the
# system roots include the proxy CA, so export them if nothing is set already.
ensure_ca_bundle() {
    [[ -n "${REQUESTS_CA_BUNDLE:-}" ]] && return
    [[ "$(uname -s)" == "Darwin" ]] || return
    local ca="$VENV/system-ca.pem"
    if [[ ! -s $ca ]]; then
        security find-certificate -a -p \
            /System/Library/Keychains/SystemRootCertificates.keychain > "$ca" 2>/dev/null || return
        security find-certificate -a -p /Library/Keychains/System.keychain >> "$ca" 2>/dev/null || true
    fi
    export REQUESTS_CA_BUNDLE="$PWD/$ca"
}

[[ -f $STAMP ]] || setup
ensure_ca_bundle

# Keep runs and checkpoints out of /data, which only exists in the container.
export FLY_RUNS_DIR="${FLY_RUNS_DIR:-$PWD/runs}"
export FLY_CHECKPOINT_DIR="${FLY_CHECKPOINT_DIR:-$PWD/checkpoints}"
export FLY_ENV_DIR="${FLY_ENV_DIR:-$PWD/environments}"
CONNECTOME_DIR="${FLY_CONNECTOME_DIR:-$PWD/data/connectome}"
mkdir -p "$FLY_RUNS_DIR" "$FLY_CHECKPOINT_DIR" "$CONNECTOME_DIR"

cmd="${1:-setup}"; shift || true
case "$cmd" in
    setup)   setup ;;
    test)    "$PY_BIN" -m pytest tests/ "${@:--q}" ;;
    # Defaults are set by assigning to "$@", not by "${@:-a b c}" - the latter
    # collapses the whole default into a SINGLE argv word, which argparse
    # rejects.
    sim)     [[ $# -eq 0 ]] && set -- --env dry_land --steps 5000
             "$PY_BIN" run.py --connectome-dir "$CONNECTOME_DIR" "$@" ;;
    compare) [[ $# -eq 0 ]] && set -- dry_land submerged_water windy
             "$PY_BIN" run.py --connectome-dir "$CONNECTOME_DIR" \
                 --compare-envs "$@" --steps "${STEPS:-12000}" ;;
    web)     export FLY_CONNECTOME_DIR="$CONNECTOME_DIR"
             echo "==> http://localhost:${PORT:-8000}"
             "$PY_BIN" -m uvicorn web.app:app --host 0.0.0.0 --port "${PORT:-8000}" ;;
    shell)   exec "$PY_BIN" "$@" ;;
    *)       echo "unknown command: $cmd" >&2
             sed -n '2,20p' "$0" >&2; exit 2 ;;
esac
