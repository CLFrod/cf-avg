#!/bin/bash
# Submit the eligibility trace study, one job (one node) per config.
#   bash $HOME/cf-avg/scripts/trillium/submit_etr.sh                                                  # every configs/eligibility_trace_real_2026-11-27/etr_*.json
#   bash $HOME/cf-avg/scripts/trillium/submit_etr.sh configs/eligibility_trace_real_2026-11-27/etr_hopper_*.json # just these
# Configs come from scripts/configs/make_eligibility_trace_configs.py. Every config of an environment shares
# $SCRATCH/results_etr/<env>/ (the config's name between etr_ and the last _).
# HOURS=6 caps the job's walltime at 6 h (default: avg_multi.sh's 24 h); runs checkpoint and stop half an hour before.
set -euo pipefail
REPO=${REPO:-$HOME/cf-avg}
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=(configs/eligibility_trace_real_2026-11-27/etr_*.json)
fi
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .json)
    env=${name#etr_}; env=${env%_*}
    bash "$HERE/submit.sh" --job-name="$name" ${HOURS:+--time="$HOURS:00:00"} \
        --export=ALL,CONFIG="$(realpath "$cfg")",RESULTS_DIR="$SCRATCH/results_etr/$env"${HOURS:+,TIME_LIMIT=$(awk "BEGIN{print $HOURS - 0.5}")}
done
