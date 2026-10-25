#!/bin/bash
# Submit the experiment job from anywhere:
#   bash $HOME/cf-avg/scripts/trillium/submit.sh
# Slurm's log must be written to scratch (home is read-only on compute nodes), so the job's working directory is
# set to $SCRATCH/avg_jobs. Any extra arguments go to sbatch, e.g. the continuation job:
#   bash submit.sh --dependency=afterany:<jobid> --export=ALL,RESUME_FROM="$SCRATCH/results/*-2026*"
set -euo pipefail
JOB_DIR="$SCRATCH/avg_jobs"
mkdir -p "$JOB_DIR"
exec sbatch --chdir="$JOB_DIR" "$@" "$(cd "$(dirname "$0")" && pwd)/avg_multi.sh"
