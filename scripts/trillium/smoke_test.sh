#!/bin/bash
# Light end-to-end check for a Trillium *login node* (within its limits: 3 cores, < 1 GB, ~90 s).
# Exercises everything the real job uses, at toy scale:
#   1. OSMesa off-screen rendering
#   2. the real config resolves (every env + mod validated)
#   3. one tiny run per env (Hopper / Ant / Humanoid) with the paper hyperparameters, a physics + termination
#      switch mid-run, videos, metrics.csv, checkpoints -> written to $SCRATCH like the real job
#   4. resume: extend one finished run by 1500 steps from its checkpoint
#
#   cd $HOME/cf-avg && bash scripts/trillium/smoke_test.sh
#
# Results are left in $SCRATCH/results/smoke_<time>/ for inspection (delete when done).
set -euo pipefail

REPO=${REPO:-$HOME/cf-avg}
OUT=${OUT:-$SCRATCH/results/smoke_$(date +%Y%m%d_%H%M%S)}
N=${N:-3000}
[ -n "${MUJOCO_GL-}" ] || export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-$SCRATCH/.cache/matplotlib}
mkdir -p "$OUT" "$MPLCONFIGDIR"
cd "$REPO"
source .venv/bin/activate
tic=$(date +%s)
step() { echo; echo "== $*"; }

step "1/4 OSMesa render check (MUJOCO_GL=$MUJOCO_GL)"
if python - <<'EOF'
import gymnasium as gym
e = gym.make("Hopper-v4", render_mode="rgb_array", width=640, height=480); e.reset(seed=0); f = e.render(); e.close()
assert f.shape == (480, 640, 3) and f.mean() > 0, f.shape
print(f"render OK: frame {f.shape}")
EOF
then VIDEOS=yes; else
    VIDEOS=no
    echo
    echo "render FAILED: libOSMesa is not loadable. Find its module with \`module spider mesa\` / \`module spider osmesa\`,"
    echo "load it, and rerun. Continuing with MUJOCO_GL=disable (training works, no videos) so the other checks still run."
    export MUJOCO_GL=disable
fi

step "2/4 real config resolves"
python -m incremental_rl.avg_multi_experiment --config configs/trillium_dynamics_3env.json --dry_run 2>/dev/null | head -1

step "3/4 three tiny runs (one per env) with a physics + termination switch at step $((N / 2)), videos, checkpoints"
cat > "$OUT/smoke.json" <<EOF
{"defaults": {"N": $N, "n_runs": 1, "checkpoint": $((N / 3)), "gamma": 0.99, "beta1": 0.0,
              "normalize_obs": true, "pnorm": true, "scaled_td": true, "n_eval": 1, "video_every": $((N / 2))},
 "envs": {"Hopper-v4":   {"actor_lr": 1.1e-5, "critic_lr": 7.7e-5, "alpha_lr": 0.3,
                          "schedule": [{"step": $((N / 2)), "mods": {"gravity": -1.62}, "env_kwargs": {"healthy_angle_range": [-0.5, 0.5]}}]},
          "Ant-v4":      {"actor_lr": 0.0063, "critic_lr": 0.0087, "alpha_lr": 0.07,
                          "schedule": [{"step": $((N / 2)), "mods": {"gravity": -1.62}, "env_kwargs": {"healthy_z_range": [0.2, 2.5]}}]},
          "Humanoid-v4": {"actor_lr": 0.0063, "critic_lr": 0.0087, "alpha_lr": 0.07,
                          "schedule": [{"step": $((N / 2)), "mods": {"gravity": -1.62}, "env_kwargs": {"healthy_z_range": [1.0, 3.5]}}]}},
 "experiments": [{"name": "smoke"}]}
EOF
python -m incremental_rl.avg_multi_experiment --config "$OUT/smoke.json" --results_dir "$OUT" --n_workers 3 2>/dev/null \
    | grep -E "^\[(done|FAILED)|finished in"

step "4/4 resume: extend the Hopper run to $((N + 1500)) steps from its checkpoint"
python -m incremental_rl.avg_multi_experiment --resume_from "$OUT/smoke-Hopper-v4-*" --results_dir "$OUT" --N $((N + 1500)) 2>/dev/null \
    | grep -E "resume from|^\[(done|FAILED)"

step "summary"
export VIDEOS
python - "$OUT" <<'EOF'
import sys, os, csv, glob, json
out = sys.argv[1]
ok = True
for d in sorted(glob.glob(f"{out}/smoke-*")):
    files = os.listdir(d)
    rows = list(csv.DictReader(open(glob.glob(f"{d}/*_metrics.csv")[0])))
    meta = json.load(open(f"{d}/checkpoint.json"))
    videos = sorted(f.split("_video_")[1][:-4] for f in files if f.endswith(".mp4"))
    log = open(f"{d}/run.log").read()
    switched = "[env_kwargs] step" in log and "[env_mods] step" in log
    rate = float(rows[-1]["step"]) / float(rows[-1]["wall_time"])
    want_videos = os.environ.get("VIDEOS", "yes") == "yes"
    status = "OK " if ((videos or not want_videos) and switched and meta["complete"]) else "BAD"
    ok &= status == "OK "
    print(f"  {status} {os.path.basename(d):34s} steps {meta['step']:5d} complete={meta['complete']}  episodes {len(rows):4d}  "
          f"{rate:4.0f} steps/s  videos at {videos}  switch applied={switched}")
print("\nALL OK" + ("" if want_videos else " (except rendering: no videos until libOSMesa is loadable)")
      if ok else "\nSOMETHING FAILED: see run.log in the folders above")
print(f"results: {out}")
EOF
echo "took $(( $(date +%s) - tic )) s"
