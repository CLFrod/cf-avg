#!/bin/bash
# Submit the shift study, one job (one node) per config.
#   bash $HOME/cf-avg/scripts/trillium/submit_study.sh                                              # every configs/shift_study/study_*.json
#   bash $HOME/cf-avg/scripts/trillium/submit_study.sh configs/shift_study/study_reacher_core.json  # just these
# Configs come from scripts/configs/make_shift_study_configs.py. Both halves of an environment share
# $SCRATCH/results_study/<env>/ (the config's name without study_ and _core/_body).
set -euo pipefail
REPO=${REPO:-$HOME/cf-avg}
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    for cfg in configs/shift_study/study_*.json; do
        case "$cfg" in *study_smoke_*) ;; *) CONFIGS+=("$cfg") ;; esac
    done
fi
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .json)
    env=${name#study_}; env=${env%_core}; env=${env%_body}
    bash "$HERE/submit.sh" --job-name="$name" --export=ALL,CONFIG="$(realpath "$cfg")",RESULTS_DIR="$SCRATCH/results_study/$env"
done
