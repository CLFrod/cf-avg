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
# More runs than one node holds: submit one job per slice, e.g. SHARD=1/2 and SHARD=2/2 (same CONFIG).
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
SHARD=${SHARD:-}                               # K/N: run only one of N interleaved slices (one job per node)
MODE_ARGS=()
[ -n "$CONFIG" ] && MODE_ARGS=(--config "$CONFIG")
[ -n "$RESUME_FROM" ] && MODE_ARGS=(--resume_from "$RESUME_FROM")
[ -n "$SHARD" ] && MODE_ARGS+=(--shard "$SHARD")

cd "$REPO"
source .venv/bin/activate

export PYTHONDONTWRITEBYTECODE=1   # can't write __pycache__ to read-only home
export MPLCONFIGDIR=$SCRATCH/.cache/matplotlib   # matplotlib's font cache: ~/.cache is read-only on compute nodes
mkdir -p "$MPLCONFIGDIR" "$RESULTS_DIR"
# No internet on compute nodes: keep --wandb_mode disabled (default) or use offline + `wandb sync` from a login node
# Videos (--n_eval > 0) render off-screen on the CPU through OSMesa, which is what Trillium supports.
# libOSMesa must be loadable: if it comes from a module, load it here (find it with `module spider mesa`):
# module load mesa/<version>
echo "Job $SLURM_JOB_ID on $(hostname): $(nproc) CPUs in allocation"

# Render self-check. N.B. with MUJOCO_GL=osmesa and no libOSMesa, `import mujoco` itself fails, which would kill
# every run at startup. So: try OSMesa; if it fails, fall back to MUJOCO_GL=disable (no GL at all): training
# runs normally, videos are skipped (each run logs "[video] recording failed").
export MUJOCO_GL=osmesa
if python - <<'EOF'
import gymnasium as gym
e = gym.make("Hopper-v4", render_mode="rgb_array", width=640, height=480); e.reset(seed=0); f = e.render(); e.close()
assert f.shape == (480, 640, 3), f.shape
print(f"OSMesa render OK: frame {f.shape}")
EOF
then :; else
    echo "OSMesa render FAILED: libOSMesa not loadable (see traceback above). Falling back to MUJOCO_GL=disable: NO VIDEOS."
    export MUJOCO_GL=disable
fi
python -m incremental_rl.avg_multi_experiment "${MODE_ARGS[@]}" \
    --name "$NAME" --results_dir "$RESULTS_DIR" \
    --n_runs "$N_RUNS" --start_seed "$START_SEED" \
    --cores_per_run 1 --reserve_cores "$RESERVE_CORES" \
    --time_limit "$TIME_LIMIT" $RUN_ARGS
