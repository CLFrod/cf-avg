#!/bin/bash
# Submit the critic eligibility trace study, one job (one node) per config.
#   bash $HOME/cf-avg/scripts/trillium/submit_traces.sh                                          # every configs/traces/traces_<env>_<shift>.json
#   bash $HOME/cf-avg/scripts/trillium/submit_traces.sh configs/traces/traces_hopper_*.json      # just these
# Configs come from scripts/configs/make_trace_configs.py. Every config of an environment shares
# $SCRATCH/results_traces/<env>/ (the config's name between traces_ and the next _).
# HOURS=6 caps the job's walltime at 6 h (default: avg_multi.sh's 24 h); runs checkpoint and stop half an hour before.
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
    env=${name#traces_}; env=${env%%_*}
    bash "$HERE/submit.sh" --job-name="$name" ${HOURS:+--time="$HOURS:00:00"}         --export=ALL,CONFIG="$(realpath "$cfg")",RESULTS_DIR="$SCRATCH/results_traces/$env"${HOURS:+,TIME_LIMIT=$(awk "BEGIN{print $HOURS - 0.5}")}
done
