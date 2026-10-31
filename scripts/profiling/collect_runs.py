#!/usr/bin/env python3
"""
Gather every run's throughput over time from its metrics.csv into CSVs that line up with the node monitor's clock.

    python scripts/profiling/collect_runs.py --monitor_dir $SCRATCH/results/monitor_<jobid> --results_dir $SCRATCH/results
    python scripts/profiling/collect_runs.py "$SCRATCH/results/base-*-2026*" --out profile_csv

Run folders come from the positional globs and/or --results_dir; with --monitor_dir only the runs the monitor
actually saw in procs.csv are kept (i.e. the runs of that job). Written to --out (default: --monitor_dir, else .):
  run_summary.csv   one row per run: experiment, env, seed, run id, start time, steps done, N, complete, wall time,
                    mean steps/s, start-up time (wall time before the first episode finished), episodes, resumed
  runs_binned.csv   one row per run per --bin seconds of run wall time: steps/s in that bin, cumulative step,
                    with t_local / t_unix of the bin end so it can be overlaid on node.csv / cpus.csv
  runs.csv          every episode row of every run (large: ~16k episodes per 12M-step Ant run) with steps/s since
                    the previous episode and since the start. Skipped with --no_episodes.

Time alignment: the run id prefix is the run's start (local time, the same clock the monitor's t_local uses), and
metrics.csv's wall_time is seconds since that start. A resumed run keeps its original id, so its wall_time
continues from the saved value: for those the absolute times are not meaningful and `resumed` is 1.
"""
import os, sys, csv, glob, json, argparse
from datetime import datetime


def run_folders(patterns, results_dir):
    dirs = []
    for p in patterns:
        dirs += [d for d in glob.glob(p) if os.path.isdir(d)]
    if results_dir:
        dirs += [os.path.join(results_dir, d) for d in os.listdir(results_dir)
                 if os.path.isdir(os.path.join(results_dir, d))]
    out = []
    for d in sorted(set(os.path.normpath(d) for d in dirs)):
        if glob.glob(os.path.join(d, "*_metrics.csv")):
            out.append(d)
    return out


def runs_seen_by_monitor(monitor_dir):
    path = os.path.join(monitor_dir, "procs.csv")
    with open(path, newline="") as f:
        return {r["run"] for r in csv.DictReader(f) if r["role"] == "run" and r["run"]}


def load_run(d):
    metrics = sorted(glob.glob(os.path.join(d, "*_metrics.csv")))[0]
    run_id = os.path.basename(metrics)[: -len("_metrics.csv")]
    info = {"experiment": "", "seed": "", "env": "", "N": ""}
    try:
        with open(os.path.join(d, "multi_experiment.json")) as f:
            me = json.load(f)
        info["experiment"] = me.get("experiment", "")
        info["seed"] = me.get("seed", "")
        hyp = me.get("hyperparams", {})
        info["env"], info["N"] = hyp.get("env", ""), hyp.get("N", "")
    except (OSError, ValueError):
        parts = os.path.basename(d).split("-")        # <experiment>-<env>-<seed>-<stamp>
        if len(parts) >= 4:
            info["experiment"], info["env"], info["seed"] = parts[0], "-".join(parts[1:-2]), parts[-2]
    meta = {}
    try:
        with open(os.path.join(d, "checkpoint.json")) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        pass
    resumed = 0
    try:
        with open(os.path.join(d, "run.log"), errors="replace") as f:
            resumed = int(any("[resume] continuing" in line for line in f))
    except OSError:
        pass
    try:
        start = datetime.strptime(run_id[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        start = None
    with open(metrics, newline="") as f:
        rows = [(int(r["step"]), float(r["wall_time"]), int(r.get("partial", 0) or 0), float(r["ep_duration"]))
                for r in csv.DictReader(f)]
    return {"dir": d, "name": os.path.basename(d), "run_id": run_id, "start": start, "rows": rows,
            "meta": meta, "resumed": resumed, **info}


def stamp(start, wall):
    if start is None:
        return "", ""
    t = start.timestamp() + wall
    return datetime.fromtimestamp(t).isoformat(timespec="seconds"), f"{t:.1f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("globs", nargs="*", help="run folder globs")
    p.add_argument("--results_dir", default=None, help="take every run folder directly under this directory")
    p.add_argument("--monitor_dir", default=None, help="keep only runs listed in this monitor output's procs.csv")
    p.add_argument("--out", default=None, help="output directory (default: --monitor_dir or .)")
    p.add_argument("--bin", type=float, default=60.0, help="bin width in seconds of run wall time")
    p.add_argument("--no_episodes", action="store_true", help="skip the per-episode runs.csv")
    args = p.parse_args()
    out = args.out or args.monitor_dir or "."
    os.makedirs(out, exist_ok=True)

    dirs = run_folders(args.globs, args.results_dir)
    if args.monitor_dir:
        seen = runs_seen_by_monitor(args.monitor_dir)
        dirs = [d for d in dirs if os.path.basename(d) in seen]
        print(f"{len(seen)} runs seen by the monitor, {len(dirs)} of them found")
    if not dirs:
        sys.exit("no run folders with a metrics.csv found")

    f_sum = open(os.path.join(out, "run_summary.csv"), "w", newline="")
    w_sum = csv.DictWriter(f_sum, fieldnames=[
        "run", "experiment", "env", "seed", "run_id", "start_local", "end_local", "steps", "N", "complete",
        "wall_time_s", "mean_steps_per_sec", "startup_s", "episodes", "resumed"])
    w_sum.writeheader()
    f_bin = open(os.path.join(out, "runs_binned.csv"), "w", newline="")
    w_bin = csv.DictWriter(f_bin, fieldnames=[
        "run", "experiment", "env", "seed", "bin_end_wall_s", "t_local", "t_unix", "step", "steps_per_sec"])
    w_bin.writeheader()
    f_ep = w_ep = None
    if not args.no_episodes:
        f_ep = open(os.path.join(out, "runs.csv"), "w", newline="")
        w_ep = csv.DictWriter(f_ep, fieldnames=[
            "run", "experiment", "env", "seed", "episode", "step", "wall_time_s", "t_local", "t_unix", "partial",
            "ep_duration_s", "steps_per_sec", "steps_per_sec_cum"])
        w_ep.writeheader()

    for d in dirs:
        r = load_run(d)
        rows = r["rows"]
        if not rows:
            continue
        base = {"run": r["name"], "experiment": r["experiment"], "env": r["env"], "seed": r["seed"]}
        last_step, last_wall = rows[-1][0], rows[-1][1]
        first_step, first_wall, _, first_dur = rows[0]
        meta = r["meta"]
        w_sum.writerow({
            **base, "run_id": r["run_id"],
            "start_local": r["start"].isoformat(timespec="seconds") if r["start"] else "",
            "end_local": stamp(r["start"], last_wall)[0],
            "steps": meta.get("step", last_step), "N": meta.get("N", r["N"]), "complete": int(meta.get("complete", 0)),
            "wall_time_s": meta.get("wall_time", last_wall),
            "mean_steps_per_sec": f"{last_step / last_wall:.1f}" if last_wall > 0 else "",
            "startup_s": f"{max(0.0, first_wall - first_dur):.1f}",
            "episodes": len(rows), "resumed": r["resumed"],
        })

        # bins of wall time: the last episode row inside each bin marks the bin's progress
        bin_w = args.bin
        prev_step, prev_wall, cur_bin = 0, 0.0, 0
        bin_last = None       # (step, wall) of the last row seen in the current bin
        for step, wall, partial, dur in rows:
            b = int(wall // bin_w)
            if b != cur_bin and bin_last is not None:
                s, wl = bin_last
                dt = wl - prev_wall
                tl, tu = stamp(r["start"], (cur_bin + 1) * bin_w)
                w_bin.writerow({**base, "bin_end_wall_s": (cur_bin + 1) * bin_w, "t_local": tl, "t_unix": tu,
                                "step": s, "steps_per_sec": f"{(s - prev_step) / dt:.1f}" if dt > 0 else ""})
                prev_step, prev_wall = s, wl
                bin_last = None
            cur_bin = b
            bin_last = (step, wall)
        if bin_last is not None:
            s, wl = bin_last
            dt = wl - prev_wall
            tl, tu = stamp(r["start"], wl)
            w_bin.writerow({**base, "bin_end_wall_s": (cur_bin + 1) * bin_w, "t_local": tl, "t_unix": tu,
                            "step": s, "steps_per_sec": f"{(s - prev_step) / dt:.1f}" if dt > 0 else ""})

        if w_ep is not None:
            ps, pw = 0, 0.0
            for i, (step, wall, partial, dur) in enumerate(rows, 1):
                tl, tu = stamp(r["start"], wall)
                dt = wall - pw
                w_ep.writerow({**base, "episode": i, "step": step, "wall_time_s": wall, "t_local": tl, "t_unix": tu,
                               "partial": partial, "ep_duration_s": dur,
                               "steps_per_sec": f"{(step - ps) / dt:.1f}" if dt > 0 else "",
                               "steps_per_sec_cum": f"{step / wall:.1f}" if wall > 0 else ""})
                ps, pw = step, wall
        print(f"  {r['name']}: {last_step} steps in {last_wall:.0f}s ({last_step / max(last_wall, 1e-9):.0f} steps/s)"
              + (", resumed" if r["resumed"] else ""))

    for f in (f_sum, f_bin, f_ep):
        if f:
            f.close()
    print(f"wrote run_summary.csv, runs_binned.csv{'' if args.no_episodes else ', runs.csv'} -> {out}")


if __name__ == "__main__":
    main()
