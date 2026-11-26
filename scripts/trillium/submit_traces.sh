#!/bin/bash
# Submit the critic eligibility trace study, one job (one node) per config.
#   bash $HOME/cf-avg/scripts/trillium/submit_traces.sh                                          # every configs/traces/traces_<env>_<shift>.json
#   bash $HOME/cf-avg/scripts/trillium/submit_traces.sh configs/traces/traces_hopper_*.json      # just these
# Configs come from scripts/configs/make_trace_configs.py. Every config of an environment shares
# $SCRATCH/results_traces/<env>/ (the config's name between traces_ and the last _<shift>).
set -euo pipefail
REPO=${REPO:-$HOME/cf-avg}
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    for cfg in configs/traces/traces_*.json; do
        case "$cfg" in *traces_smoke*) ;; *) CONFIGS+=("$cfg") ;; esac
    done
fi
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .json)
    env=$(python3 -c "import json, sys; print(next(iter(json.load(open(sys.argv[1]))['envs'])))" "$cfg")
    env=${env#dmc-}; env=${env%-v*}; env=${env,,}
    bash "$HERE/submit.sh" --job-name="$name" --export=ALL,CONFIG="$(realpath "$cfg")",RESULTS_DIR="$SCRATCH/results_traces/$env"
done
