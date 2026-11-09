#!/bin/bash
# Login-node smoke test of a lever study config (within its limits: 3 cores, < 1 GB, ~2 min):
#   every condition of the smoke config once, on Hopper gear2x, 3000 steps with the switch at 1500, then the last
#   lever run resumed from its checkpoint. Checks that every run completes, the hook fires at the switch, load_stats
#   finds its files on $SCRATCH, and the dense/metrics logs are written.
#
#   cd $HOME/cf-avg && bash scripts/trillium/smoke_levers.sh                      # batch 1: configs/levers_smoke.json
#   CONFIG=configs/priv_smoke.json bash scripts/trillium/smoke_levers.sh          # batch 2
#
# Results are left in $SCRATCH/results/smoke_levers_<time>/ (delete when done).
set -euo pipefail

REPO=${REPO:-$HOME/cf-avg}
CONFIG=${CONFIG:-configs/levers_smoke.json}
PREFIX=$(basename "$CONFIG" _smoke.json)                                  # levers | priv
OUT=${OUT:-$SCRATCH/results/smoke_${PREFIX}_$(date +%Y%m%d_%H%M%S)}
[ -n "${MUJOCO_GL-}" ] || export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-$SCRATCH/.cache/matplotlib}
mkdir -p "$OUT" "$MPLCONFIGDIR"
cd "$REPO"
if ! type module &>/dev/null && [ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]; then
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
fi
type module &>/dev/null && module load StdEnv/2023 python/3.13
source .venv/bin/activate
tic=$(date +%s)
step() { echo; echo "== $*"; }

step "1/3 the three real configs resolve"
for e in hopper ant humanoid; do
    plan=$(python -m incremental_rl.avg_multi_experiment --config configs/levers_$e.json --dry_run 2>/dev/null)
    echo "  levers_$e: ${plan%%$'\n'*}"
done

step "2/3 every condition once (3 at a time)"
python -m incremental_rl.avg_multi_experiment --config "$CONFIG" --results_dir "$OUT" --n_workers 3 2>/dev/null \
    | grep -E "^\[FAILED|finished in" || true

LAST=$(ls -d "$OUT"/*-gear2x-Hopper-v4-* | grep -v -E "/(fix|scratch|legacy)-" | tail -1)
step "3/3 resume $(basename "$LAST") by 1500 steps"
python -m incremental_rl.avg_multi_experiment --resume_from "$LAST" --results_dir "$OUT" --N 4500 2>/dev/null \
    | grep -E "resume from|^\[(done|FAILED)" || true

step "summary"
python - "$OUT" <<'EOF'
import sys, os, csv, glob, json
out = sys.argv[1]
ok = True
for d in sorted(glob.glob(f"{out}/*-gear2x-Hopper-v4-*")):
    name = os.path.basename(d).split("-gear2x")[0]
    meta = json.load(open(f"{d}/checkpoint.json"))
    rows = list(csv.DictReader(open(glob.glob(f"{d}/*_metrics.csv")[0])))
    log = open(f"{d}/run.log").read()
    fired = "[levers] step 1500" in log
    dense = bool(glob.glob(f"{d}/*_dense.csv"))
    want = name != "scratch"  # the from-scratch run has no switch, so no hook and no dense window
    actions = json.load(open(f"{d}/multi_experiment.json"))["hyperparams"]["on_shift"]
    loaded = not any(a["action"] == "load_stats" for a in actions) or "'load_stats'" in log
    clean = "Traceback" not in log
    good = meta["complete"] and fired == want and dense == want and loaded and clean and len(rows[0]) >= 50
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {name:20s} steps {meta['step']:4d}  episodes {len(rows):3d}  columns {len(rows[0])}  "
          f"hook fired={fired}  dense={dense}" + ("" if loaded else "  STATS NOT LOADED") + ("" if clean else "  TRACEBACK"))
print("\nALL OK" if ok else "\nSOMETHING FAILED: see run.log in the folders above")
print(f"results: {out}")
EOF
echo "took $(( $(date +%s) - tic )) s"
