#!/bin/bash
# Run many seeds of avg_ablation on one full Trillium CPU node (192 cores), one pinned core per run.
#
# Submit from a directory in $SCRATCH (home is read-only on compute nodes, so Slurm's output file must
# land in scratch):
#   cd $SCRATCH/avg_jobs && sbatch $HOME/cf-avg/scripts/trillium/avg_multi.sbatch
#
# Edit NAME / RUN_ARGS below, or override at submit time, e.g.
#   sbatch --export=ALL,NAME=moon_earth,N_RUNS=188 avg_multi.sbatch
# Several environments with their own hyperparameters: point CONFIG at a YAML (see configs/avg_envs.yaml);
# RUN_ARGS then act as overrides for every experiment in it (leave empty for none):
#   sbatch --export=ALL,CONFIG=$HOME/cf-avg/configs/avg_envs.yaml avg_multi.sbatch
# Runs stop cleanly at TIME_LIMIT hours with a checkpoint. To continue them in a new job:
#   sbatch --export=ALL,RESUME_FROM="$SCRATCH/results/gravity-*-20261024_*" avg_multi.sbatch
# (add --dependency=afterany:<jobid> to queue the continuation behind the first job automatically)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=24:00:00
#SBATCH --job-name=avg_multi
#SBATCH --output=avg_multi_%j.out
#SBATCH --mail-type=FAIL

set -euo pipefail

REPO=${REPO:-$HOME/cf-avg}                     # read-only on compute nodes: code + .venv only
RESULTS_DIR=${RESULTS_DIR:-$SCRATCH/results}   # all outputs go to scratch
NAME=${NAME:-avg_multi}
N_RUNS=${N_RUNS:-188}
START_SEED=${START_SEED:-0}
RESERVE_CORES=${RESERVE_CORES:-4}              # launcher + OS + file I/O
CONFIG=${CONFIG:-}                             # multi-experiment YAML/JSON; empty = single experiment from RUN_ARGS
RESUME_FROM=${RESUME_FROM:-}                   # glob of run folders to continue from their checkpoints
TIME_LIMIT=${TIME_LIMIT:-23.5}                 # hours: runs checkpoint and exit cleanly before the 24 h walltime
RUN_ARGS=${RUN_ARGS-}
if [ -z "$CONFIG$RESUME_FROM" ] && [ -z "$RUN_ARGS" ]; then
    RUN_ARGS="--env Hopper-v4 --N 12000000 --normalize_obs --pnorm --scaled_td"
fi
MODE_ARGS=()
[ -n "$CONFIG" ] && MODE_ARGS=(--config "$CONFIG")
[ -n "$RESUME_FROM" ] && MODE_ARGS=(--resume_from "$RESUME_FROM")

cd "$REPO"
source .venv/bin/activate

export PYTHONDONTWRITEBYTECODE=1   # can't write __pycache__ to read-only home
# No internet on compute nodes: keep --wandb_mode disabled (default) or use offline + `wandb sync` from a login node
# Only needed for --n_eval > 0 (video recording); compute nodes are headless.
# Check that OSMesa is available (`ldconfig -p | grep -i osmesa`) before relying on this.
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa

echo "Job $SLURM_JOB_ID on $(hostname): $(nproc) CPUs in allocation"
python -m incremental_rl.avg_multi_experiment "${MODE_ARGS[@]}" \
    --name "$NAME" --results_dir "$RESULTS_DIR" \
    --n_runs "$N_RUNS" --start_seed "$START_SEED" \
    --cores_per_run 1 --reserve_cores "$RESERVE_CORES" \
    --time_limit "$TIME_LIMIT" $RUN_ARGS
