#!/bin/bash
# Build the Python environment on a Trillium login node (run once, and again whenever uv.lock changes):
#   cd $HOME/cf-avg && bash scripts/trillium/setup_env.sh
#
# Interpreter: the Alliance python/3.13 module (built in StdEnv/2023's Gentoo layer, the same library world as
# libOSMesa, which off-screen video rendering needs). Packages: the exact versions in uv.lock, installed by uv.
set -euo pipefail
REPO=${REPO:-$HOME/cf-avg}
cd "$REPO"

# A previously activated venv would make `which python` point at the wrong interpreter
deactivate 2>/dev/null || true

if ! type module &>/dev/null && [ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]; then
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
fi
module load StdEnv/2023 python/3.13
PY=$(which python)
echo "interpreter: $PY ($($PY --version))"
case "$PY" in /cvmfs/*) ;; *) echo "ERROR: expected the python/3.13 module's interpreter, got $PY"; exit 1;; esac

# uv itself: a user-space binary in ~/.local/bin (login nodes have internet)
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi
echo "uv $(uv --version)"

rm -rf .venv
uv sync --python "$PY"
source .venv/bin/activate
python - <<'EOF'
import sys, torch, mujoco, gymnasium
assert ".local/share/uv/python" not in sys.base_prefix, f"venv is on uv's standalone Python: {sys.base_prefix}"
print(f"venv OK on {sys.base_prefix}")
print(f"torch {torch.__version__}  mujoco {mujoco.__version__}  gymnasium {gymnasium.__version__}")
EOF
echo
echo "Next: bash scripts/trillium/smoke_test.sh"
