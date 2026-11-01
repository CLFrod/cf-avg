#!/bin/bash
# Progress of one or more results folders: per experiment and env, how many runs are complete, how far the rest
# are, when they last checkpointed (a run not complete and not checkpointed for an hour = its job is gone), and
# an ETA from each run's own steps/s. Also shows your Slurm queue and whether a utilisation monitor is live.
#   bash scripts/trillium/progress.sh $SCRATCH/results_dynamics_18m $SCRATCH/results_limb_18m
#   bash scripts/trillium/progress.sh -v $SCRATCH/results_dynamics_18m      # one line per run as well
#   watch -n 300 bash scripts/trillium/progress.sh $SCRATCH/results_*       # refresh every 5 min
verbose=0
if [ "${1:-}" = "-v" ]; then verbose=1; shift; fi
[ $# -gt 0 ] || set -- .
PY=python3; "$PY" -c pass &>/dev/null || PY=python      # whichever interpreter actually runs here

if command -v squeue &>/dev/null; then
    echo "== squeue -u $USER"
    squeue -u "$USER" -o "  %.10i %-12j %.3t %.11M %.11l %R" 2>/dev/null | sed 1d
    echo
fi

for dir in "$@"; do
    dir=${dir%/}
    [ -d "$dir" ] || { echo "== $dir: no such directory"; continue; }
    echo "== $dir"
    VERBOSE=$verbose "$PY" - "$dir" <<'EOF'
import os, sys, json, glob, time
from datetime import datetime
d = sys.argv[1]
verbose = os.environ.get("VERBOSE") == "1"
now = time.time()
folders = sorted(f for f in glob.glob(os.path.join(d, "*")) if os.path.isdir(f) and not os.path.basename(f).startswith("monitor_"))
runs = []
for f in folders:
    name = os.path.basename(f)
    parts = name.split("-")                       # <experiment>-<env>-<seed>-<stamp>
    exp, env, seed = (parts[0], "-".join(parts[1:-2]), parts[-2]) if len(parts) >= 4 else (name, "?", "?")
    try:
        with open(os.path.join(f, "checkpoint.json")) as fh:
            ck = json.load(fh)
    except (OSError, ValueError):
        runs.append({"name": name, "exp": exp, "env": env, "seed": seed, "state": "no checkpoint", "step": 0, "N": 0, "age": None, "eta": None, "complete": False})
        continue
    step, N, wall = ck.get("step", 0), ck.get("N", 0), ck.get("wall_time", 0) or 0
    try:
        age = now - datetime.strptime(ck["saved_at"], "%Y-%m-%d %H:%M:%S").timestamp()
    except (KeyError, ValueError):
        age = None
    complete = bool(ck.get("complete")) or (N and step >= N)
    eta = (N - step) / (step / wall) if (not complete and step and wall) else None
    if complete:
        state = "complete"
    elif age is not None and age > 3600:
        state = "STALLED"           # no checkpoint for an hour: the job finished / died / hit walltime
    else:
        state = "running"
    runs.append({"name": name, "exp": exp, "env": env, "seed": seed, "state": state, "step": step, "N": N,
                 "age": age, "eta": eta, "complete": complete, "wall": wall})

def hms(s):
    if s is None: return "-"
    s = int(s); return f"{s // 3600}:{s % 3600 // 60:02d}h" if s >= 3600 else f"{s // 60}m"

if not runs:
    print("  no run folders"); sys.exit()
n_done = sum(r["complete"] for r in runs)
n_stalled = sum(r["state"] == "STALLED" for r in runs)
print(f"  {n_done} / {len(runs)} runs complete, {len(runs) - n_done} not" + (f" ({n_stalled} STALLED: no checkpoint for > 1 h)" if n_stalled else ""))
groups = {}
for r in runs:
    groups.setdefault((r["exp"], r["env"]), []).append(r)
print(f"  {'experiment/env':30s} {'done':>9s} {'min':>6s} {'median':>6s} {'max':>6s}   {'last ckpt':>9s} {'ETA':>7s}")
for (exp, env), rs in sorted(groups.items()):
    steps = sorted(r["step"] for r in rs)
    N = max((r["N"] for r in rs), default=0)
    pct = lambda s: f"{100 * s / N:5.1f}%" if N else f"{s}"
    done = sum(r["complete"] for r in rs)
    ages = [r["age"] for r in rs if r["age"] is not None and not r["complete"]]
    etas = [r["eta"] for r in rs if r["eta"] is not None]
    flag = "  STALLED" if any(r["state"] == "STALLED" for r in rs) else ""
    print(f"  {exp + '/' + env:30s} {done:3d}/{len(rs):<3d}   {pct(steps[0]):>6s} {pct(steps[len(steps) // 2]):>6s} {pct(steps[-1]):>6s}   "
          f"{hms(min(ages)) if ages else '-':>9s} {hms(max(etas)) if etas else '-':>7s}{flag}")
if verbose:
    print()
    for r in sorted(runs, key=lambda r: (r["exp"], r["env"], int(r["seed"]) if r["seed"].isdigit() else 0)):
        print(f"    {r['name']:48s} {r['state']:13s} {r['step']:>10,d} / {r['N']:<10,d} last ckpt {hms(r['age']):>6s} ago   ETA {hms(r['eta'])}")
EOF
    # Utilisation monitor(s) in this folder: live if node.csv was written in the last minute
    for m in "$dir"/monitor_*/node.csv; do
        [ -f "$m" ] || continue
        age=$(( $(date +%s) - $(stat -c %Y "$m") ))
        rows=$(( $(wc -l < "$m") - 1 ))
        if [ "$age" -lt 60 ]; then st="live"; else st="stopped ${age}s ago"; fi
        echo "  monitor $(basename "$(dirname "$m")"): $rows samples, $st"
    done
    echo
done
