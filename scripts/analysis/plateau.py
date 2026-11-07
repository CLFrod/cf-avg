"""
When does learning plateau? For one condition (default: base) per env, over the runs without a mid-run shift:

  * per seed: step of the smoothed peak, and the first step at which the curve reaches 80 / 90 / 95 % of that peak
  * the across-seed mean curve: same milestones
  * a table at every --every steps: mean return, std across seeds, and the std of episode returns inside each run
    (within-run), both as a fraction of the mean so envs are comparable

    python scripts/analysis/plateau.py results_trillium/results_24M [--experiment base] [--out plateau.png]
"""
import os, sys, glob, argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from peak_returns import load_run

FRACTIONS = (0.8, 0.9, 0.95)
COLORS = {"Ant-v4": "#2a78d6", "Hopper-v4": "#eb6834", "Humanoid-v4": "#1baf7a"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir")
    p.add_argument("--experiment", default="base")
    p.add_argument("--env", nargs="*", default=None, help="Keep only these envs (prefix match)")
    p.add_argument("--window", type=int, default=500000, help="Trailing window in steps for the smoothed curve")
    p.add_argument("--x_tick", type=int, default=50000)
    p.add_argument("--every", type=int, default=1000000, help="Row spacing of the milestone table, in steps")
    p.add_argument("--out", default=None, help="Output PNG (default: show)")
    return p.parse_args()


def windowed(run, grid, window):
    """ Mean and std of the episode returns inside the trailing window ending at each grid point """
    steps, rets = run["steps"], run["returns"]
    mean, std = np.full(len(grid), np.nan), np.full(len(grid), np.nan)
    for k, end in enumerate(grid):
        r = rets[(steps > end - window) & (steps <= end)]
        if len(r):
            mean[k], std[k] = r.mean(), r.std()
    return mean, std


def milestones(curve, grid):
    peak_i = int(np.nanargmax(curve))
    out = {"peak_step": int(grid[peak_i]), "peak": float(curve[peak_i])}
    for f in FRACTIONS:
        hit = np.flatnonzero(curve >= f * curve[peak_i])
        out[f] = int(grid[hit[0]]) if len(hit) else None
    return out


def fmt_step(s):
    return "   -   " if s is None else f"{s / 1e6:5.2f}M"


def analyse(env, runs, args):
    N = min(r["steps"][-1] for r in runs)
    grid = np.arange(args.x_tick, N + 1, args.x_tick)
    per_seed = [windowed(r, grid, args.window) for r in runs]
    means = np.array([m for m, _ in per_seed])
    within = np.array([s for _, s in per_seed])
    mean_curve, across_std = np.nanmean(means, axis=0), np.nanstd(means, axis=0)

    print(f"\n{'=' * 100}\n{env}  {args.experiment}  n={len(runs)} seeds   window {args.window // 1000}k steps")
    print(f"{'seed':>6s} {'peak':>8s} {'@':>7s}  " + "  ".join(f"{int(f * 100)}%".rjust(7) for f in FRACTIONS))
    for r, m in zip(runs, means):
        ms = milestones(m, grid)
        print(f"{r['seed']:6d} {ms['peak']:8.0f} {fmt_step(ms['peak_step'])}  " + "  ".join(fmt_step(ms[f]) for f in FRACTIONS))
    ms = milestones(mean_curve, grid)
    print(f"{'mean':>6s} {ms['peak']:8.0f} {fmt_step(ms['peak_step'])}  " + "  ".join(fmt_step(ms[f]) for f in FRACTIONS))

    print(f"\n{'step':>7s} {'mean':>8s} {'across-seed std':>16s} {'within-run std':>15s}")
    for end in np.arange(args.every, N + 1, args.every):
        k = int(np.searchsorted(grid, end))
        k = min(k, len(grid) - 1)
        w = np.nanmean(within[:, k])
        print(f"{fmt_step(end):>7s} {mean_curve[k]:8.0f} {across_std[k]:8.0f} ({across_std[k] / mean_curve[k]:4.0%})"
              f" {w:8.0f} ({w / mean_curve[k]:4.0%})")
    return dict(grid=grid, means=means, mean=mean_curve, across=across_std, within=np.nanmean(within, axis=0), ms=ms)


def plot(results, args):
    envs = list(results)
    fig, axes = plt.subplots(2, len(envs), figsize=(5 * len(envs) + 1, 7.5), squeeze=False)
    for c, env in enumerate(envs):
        d, col = results[env], COLORS.get(env, "#52514e")
        x = d["grid"] / 1e6
        ax = axes[0, c]
        for m in d["means"]:
            ax.plot(x, m, color=col, lw=0.8, alpha=0.35)
        ax.plot(x, d["mean"], color=col, lw=2, label="mean over seeds")
        ax.fill_between(x, d["mean"] - d["across"], d["mean"] + d["across"], color=col, alpha=0.15, lw=0)
        for f, ls in zip(FRACTIONS, (":", "--", "-")):
            if d["ms"][f] is not None:
                ax.axvline(d["ms"][f] / 1e6, color="#8a8984", lw=1, ls=ls, label=f"{int(f * 100)}% of peak")
        ax.set_title(f"{env}  {args.experiment}  (n = {len(d['means'])})", fontsize=11)
        ax.set_ylabel(f"return (trailing {args.window // 1000}k mean)")
        ax.legend(fontsize=7, frameon=False, loc="lower right")

        ax = axes[1, c]
        across, within = d["across"] / d["mean"], d["within"] / d["mean"]
        ax.plot(x, across, color=col, lw=1.8, label="across seeds")
        ax.plot(x, within, color=col, lw=1.8, ls="--", label="within run (episodes)")
        ax.set_ylim(0, min(1.5, np.nanmax(np.r_[across, within]) * 1.1))
        ax.set_ylabel("std / mean")
        ax.set_xlabel("steps (M)")
        ax.legend(fontsize=7, frameon=False)
        for a in axes[:, c]:
            a.yaxis.grid(True, color="#e6e5e1", lw=0.8)
            a.set_axisbelow(True)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    if args.out:
        matplotlib.use("Agg")
    runs = {}
    for d in sorted(glob.glob(os.path.join(args.results_dir, "*", "multi_experiment.json"))):
        run = load_run(os.path.dirname(d))
        if run is None or run["experiment"] != args.experiment:
            continue
        if args.env and not any(run["env"].startswith(e) for e in args.env):
            continue
        runs.setdefault(run["env"], []).append(run)
    if not runs:
        sys.exit(f"no '{args.experiment}' runs under {args.results_dir}")
    results = {env: analyse(env, sorted(rs, key=lambda r: r["seed"]), args) for env, rs in sorted(runs.items())}
    fig = plot(results, args)
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nwrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
