#!/bin/bash
# Light end-to-end check for a Trillium *login node* (within its limits: 3 cores, < 1 GB, ~90 s).
# Exercises everything the real job uses, at toy scale:
#   1. OSMesa off-screen rendering
#   2. the real config resolves (every env + mod validated)
#   3. one tiny run per env (Hopper / Ant / Humanoid) with the paper hyperparameters, a physics + termination
#      switch mid-run, videos, metrics and dense CSVs, model snapshots, checkpoints -> written to $SCRATCH like the real job
#   4. resume: extend one finished run by 1500 steps from its checkpoint
#   5. plot_runs reads the gzipped metrics
# The summary also checks the layout: runs/<run>/ holds the data, artifacts/<run>/ the models, videos and plots.
#
#   cd $HOME/cf-avg && bash scripts/trillium/smoke_test.sh
#
# Results are left in $SCRATCH/results/smoke_<time>/ for inspection (delete when done).
# Off the cluster: SCRATCH=/some/tmp REPO=$PWD MUJOCO_GL=glfw bash scripts/trillium/smoke_test.sh
set -euo pipefail

REPO=${REPO:-$HOME/cf-avg}
OUT=${OUT:-$SCRATCH/results/smoke_$(date +%Y%m%d_%H%M%S)}
N=${N:-3000}
[ -n "${MUJOCO_GL-}" ] || export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-$SCRATCH/.cache/matplotlib}
mkdir -p "$OUT" "$MPLCONFIGDIR"
cd "$REPO"
# The venv is built on the Alliance python module (see README): same Gentoo-layer libraries as libOSMesa.
# `module` is a shell function from the login profile; if this script runs in a shell without it, initialise
# Lmod the way the Alliance profile does.
if ! type module &>/dev/null && [ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]; then
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
fi
if type module &>/dev/null; then
    module load StdEnv/2023 python/3.13
else
    echo "note: no \`module\` command in this shell (not on an Alliance cluster?); using .venv as is"
fi
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate   # Scripts/: a Windows venv
tic=$(date +%s)
step() { echo; echo "== $*"; }

step "0/5 interpreter: the venv must be on the Alliance python module, not uv's standalone build"
python - <<'EOF'
import sys
print(f"  python {sys.version.split()[0]}  base: {sys.base_prefix}")
if ".local/share/uv/python" in sys.base_prefix:
    print("  WARNING: this venv is on uv's standalone Python; libOSMesa (Gentoo layer) cannot load into it.")
    print("  Rebuild: deactivate; module load StdEnv/2023 python/3.13; rm -rf .venv; uv sync --python \"$(which python)\"")
EOF

step "1/5 render check (MUJOCO_GL=$MUJOCO_GL)"
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

step "2/5 real config resolves"
# (captured rather than piped to `head`: head closing the pipe early would SIGPIPE python and, with pipefail, abort the script)
plan=$(python -m incremental_rl.avg_multi_experiment --config configs/trillium_dynamics_3env_24m.json --dry_run 2>/dev/null)
echo "${plan%%$'\n'*}"

step "3/5 three tiny runs (one per env) with a physics + termination switch at step $((N / 2)), videos, checkpoints"
cat > "$OUT/smoke.json" <<EOF
{"defaults": {"N": $N, "n_runs": 1, "checkpoint": $((N / 3)), "gamma": 0.99, "beta1": 0.0,
              "normalize_obs": true, "pnorm": true, "scaled_td": true, "n_eval": 1, "video_every": $((N / 2)),
              "save_model": true, "dense_window": [200, 300]},
 "envs": {"Hopper-v4":   {"actor_lr": 1.1e-5, "critic_lr": 7.7e-5, "alpha_lr": 0.3,
                          "schedule": [{"step": $((N / 2)), "mods": {"gravity": -1.62}, "env_kwargs": {"healthy_angle_range": [-0.5, 0.5]}}]},
          "Ant-v4":      {"actor_lr": 0.0063, "critic_lr": 0.0087, "alpha_lr": 0.07,
                          "schedule": [{"step": $((N / 2)), "mods": {"gravity": -1.62}, "env_kwargs": {"healthy_z_range": [0.2, 2.5]}}]},
          "Humanoid-v4": {"actor_lr": 0.0063, "critic_lr": 0.0087, "alpha_lr": 0.07,
                          "schedule": [{"step": $((N / 2)), "mods": {"gravity": -1.62}, "env_kwargs": {"healthy_z_range": [1.0, 3.5]}}]}},
 "experiments": [{"name": "smoke"}]}
EOF
python -m incremental_rl.avg_multi_experiment --config "$OUT/smoke.json" --results_dir "$OUT" --n_workers 3 2>/dev/null \
    | grep -E "^\[(done|FAILED)|finished in" || true   # a failure shows up as BAD in the summary below

step "4/5 resume: extend the Hopper run to $((N + 1500)) steps from its checkpoint"
python -m incremental_rl.avg_multi_experiment --resume_from "$OUT/runs/smoke-Hopper-v4-*" --results_dir "$OUT" --N $((N + 1500)) 2>/dev/null \
    | grep -E "resume from|^\[(done|FAILED)" || true

step "5/5 plot_runs reads the gzipped metrics"
python -m incremental_rl.plot_runs "smoke=$OUT/runs/smoke-*" --metrics return entropy --x_tick 500 --window 500 --out "$OUT/plot_runs.png" 2>/dev/null || true
[ -s "$OUT/plot_runs.png" ] && PLOT=yes || PLOT=no
echo "  plot written: $PLOT"

step "summary"
export VIDEOS PLOT N
python - "$OUT" <<'EOF'
import sys, os, csv, glob, gzip, json
out, N = sys.argv[1], int(os.environ["N"])
ok = os.environ["PLOT"] == "yes"
want_videos = os.environ.get("VIDEOS", "yes") == "yes"
HEAVY = (".pt", ".mp4", ".png")
dirs = sorted(glob.glob(f"{out}/runs/smoke-*"))
for d in dirs:
    name = os.path.basename(d)
    data, art = os.listdir(d), os.listdir(f"{out}/artifacts/{name}")
    metrics = [f for f in data if f.endswith("_metrics.csv.gz")]
    rows = list(csv.DictReader(gzip.open(f"{d}/{metrics[0]}", "rt", newline=""))) if metrics else []
    dense = [f for f in data if f.endswith("_dense.csv.gz")]
    dense_rows = sum(1 for _ in gzip.open(f"{d}/{dense[0]}", "rt")) - 1 if dense else 0
    meta = json.load(open(f"{d}/checkpoint.json"))
    videos = sorted(f.split("_video_")[1][:-4] for f in art if f.endswith(".mp4"))
    log = open(f"{d}/run.log").read()
    switched = "[env_kwargs] step" in log and "[env_mods] step" in log
    steps = [int(r["step"]) for r in rows if r["step"] != "step"]
    want_steps = N + 1500 if "Hopper" in name else N
    # data on one side, models/videos/plots on the other, nothing left unzipped
    split = (not [f for f in data if f.endswith(HEAVY + (".csv", ".tmp"))] and "checkpoint.pt" in art
             and any(f.endswith("_model.pt") for f in art)   # (no learning-curve PNG: runs this short have no points)
             and not [f for f in art if not f.endswith(HEAVY)] and any(f.endswith("_returns.txt") for f in data))
    csv_ok = (len(steps) == len(rows) > 0 and steps == sorted(steps) and steps[-1] == meta["step"] == want_steps
              and dense_rows == 500)
    rate = float(rows[-1]["step"]) / float(rows[-1]["wall_time"]) if rows else 0
    good = (videos or not want_videos) and switched and meta["complete"] and split and csv_ok and "Traceback" not in log
    ok &= bool(good)
    print(f"  {'OK ' if good else 'BAD'} {name:34s} steps {meta['step']:5d} complete={meta['complete']}  episodes {len(rows):4d}  "
          f"{rate:4.0f} steps/s  videos at {videos}  switch applied={switched}  layout={split}  csv.gz={csv_ok} (dense rows {dense_rows})")
ok &= len(dirs) == 3 and not glob.glob(f"{out}/smoke-*")
print("\nALL OK" + ("" if want_videos else " (except rendering: no videos until libOSMesa is loadable)")
      if ok else "\nSOMETHING FAILED: see run.log in the folders above")
print(f"results: {out}")
sys.exit(0 if ok else 1)
EOF
echo "took $(( $(date +%s) - tic )) s"
