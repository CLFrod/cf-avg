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
#   sbatch --export=ALL,CONFIG=$HOME/cf-avg/configs/avg_envs.yaml,RUN_ARGS= avg_multi.sbatch
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
RUN_ARGS=${RUN_ARGS-"--env Hopper-v4 --N 12000000 --normalize_obs --pnorm --scaled_td"}
CONFIG=${CONFIG:-}                             # multi-experiment YAML/JSON; empty = single experiment from RUN_ARGS
CONFIG_ARG=()
[ -n "$CONFIG" ] && CONFIG_ARG=(--config "$CONFIG")

cd "$REPO"
source .venv/bin/activate

export PYTHONDONTWRITEBYTECODE=1   # can't write __pycache__ to read-only home
# No internet on compute nodes: keep --wandb_mode disabled (default) or use offline + `wandb sync` from a login node
# Only needed for --n_eval > 0 (video recording); compute nodes are headless.
# Check that OSMesa is available (`ldconfig -p | grep -i osmesa`) before relying on this.
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa

echo "Job $SLURM_JOB_ID on $(hostname): $(nproc) CPUs in allocation"
python -m incremental_rl.avg_multi_experiment "${CONFIG_ARG[@]}" \
    --name "$NAME" --results_dir "$RESULTS_DIR" \
    --n_runs "$N_RUNS" --start_seed "$START_SEED" \
    --cores_per_run 1 --reserve_cores "$RESERVE_CORES" \
    $RUN_ARGS
