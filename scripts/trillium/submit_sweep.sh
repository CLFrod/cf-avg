#!/bin/bash
# Submit hyperparameter sweeps, one job (one node, 186 runs) per config:
#   bash $HOME/cf-avg/scripts/trillium/submit_sweep.sh                                   # every configs/sweep_*.json
#   bash $HOME/cf-avg/scripts/trillium/submit_sweep.sh configs/sweep_walker2d-v4.json    # just these
# Configs come from scripts/sweep/make_sweep_configs.py. Results go to $SCRATCH/results_sweep/<env>/ (the config's
# name without sweep_, e.g. results_sweep/walker2d-v4/h000-Walker2d-v4-0-<date>/); rank them with
#   python scripts/sweep/sweep_summary.py --root $SCRATCH/results_sweep --top 5
set -euo pipefail
REPO=${REPO:-$HOME/cf-avg}
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
CONFIGS=("$@")
[ ${#CONFIGS[@]} -gt 0 ] || CONFIGS=(configs/sweep_*.json)
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .json)
    bash "$HERE/submit.sh" --job-name="$name" --export=ALL,CONFIG="$(realpath "$cfg")",RESULTS_DIR="$SCRATCH/results_sweep/${name#sweep_}"
done
