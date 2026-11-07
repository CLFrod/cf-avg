"""
Peak return of every run whose dynamics never changed mid-run (no `schedule` in its multi_experiment.json).

    python scripts/analysis/peak_returns.py $SCRATCH/results [--env Ant Hopper] [--window 100000] [--csv peaks.csv]

Peak = highest trailing-window mean of episodic return (window in steps), plus the single best episode.
"""
import os, sys, csv, glob, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from incremental_rl.utils import binned_curve


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir")
    p.add_argument("--env", nargs="*", default=None, help="Keep only these envs (prefix match, e.g. Ant Hopper)")
    p.add_argument("--window", type=int, default=100000, help="Trailing window in steps for the smoothed peak")
    p.add_argument("--x_tick", type=int, default=10000, help="Step spacing at which the window is evaluated")
    p.add_argument("--csv", default=None, help="Also write the per-run table here")
    return p.parse_args()


def load_run(run_dir):
    with open(os.path.join(run_dir, "multi_experiment.json")) as f:
        info = json.load(f)
    hyp = info["hyperparams"]
    if hyp.get("schedule"):
        return None
    metrics = glob.glob(os.path.join(run_dir, "*_metrics.csv"))
    if not metrics:
        return None
    with open(metrics[0], newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r and r["partial"] == "0"]
    if not rows:
        return None
    steps = np.array([float(r["step"]) for r in rows])
    rets = np.array([float(r["return"]) for r in rows])
    meta_path = os.path.join(run_dir, "checkpoint.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"complete": None}
    return dict(experiment=info["experiment"], env=hyp["env"], seed=info["seed"], folder=os.path.basename(run_dir),
                steps=steps, returns=rets, N=hyp["N"], complete=meta["complete"])


def summarise(run, window, x_tick):
    smooth, x = binned_curve(run["steps"], run["returns"], x_tick=x_tick, window_len=window)
    i, j = int(np.argmax(smooth)), int(np.argmax(run["returns"]))
    tail = run["returns"][run["steps"] > run["steps"][-1] - window]
    return dict(experiment=run["experiment"], env=run["env"], seed=run["seed"],
                peak_smoothed=smooth[i], peak_step=int(x[i]),
                peak_episode=run["returns"][j], peak_episode_step=int(run["steps"][j]),
                final_smoothed=tail.mean(), episodes=len(run["returns"]),
                steps_done=int(run["steps"][-1]), complete=run["complete"], folder=run["folder"])


def main():
    args = parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(args.results_dir, "*", "multi_experiment.json"))):
        run = load_run(os.path.dirname(d))
        if run is None:
            continue
        if args.env and not any(run["env"].startswith(e) for e in args.env):
            continue
        rows.append(summarise(run, args.window, args.x_tick))
    if not rows:
        sys.exit(f"no unshifted runs found under {args.results_dir}")
    rows.sort(key=lambda r: (r["env"], r["experiment"], r["seed"]))

    print(f"{'env':12s} {'experiment':21s} {'seed':>4s} {'peak(smooth)':>13s} {'@step':>10s} {'best ep':>10s} "
          f"{'final':>10s} {'steps':>10s} done")
    for r in rows:
        print(f"{r['env']:12s} {r['experiment']:21s} {r['seed']:4d} {r['peak_smoothed']:13.1f} {r['peak_step']:10,d} "
              f"{r['peak_episode']:10.1f} {r['final_smoothed']:10.1f} {r['steps_done']:10,d} { {True: 'yes', False: 'NO', None: '?'}[r['complete']] }")

    print(f"\nper condition (mean +- std over seeds, window {args.window:,} steps)")
    groups = {}
    for r in rows:
        groups.setdefault((r["env"], r["experiment"]), []).append(r)
    for (env, exp), g in groups.items():
        peaks = np.array([r["peak_smoothed"] for r in g])
        finals = np.array([r["final_smoothed"] for r in g])
        print(f"{env:12s} {exp:21s} n={len(g)}  peak {peaks.mean():9.1f} +- {peaks.std():7.1f}   "
              f"final {finals.mean():9.1f} +- {finals.std():7.1f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
