"""
Plot per-episode metrics (from <run_id>_metrics.csv) against steps, aggregated over seeds.

Each positional argument is a group of runs: a glob over run folders or metrics files, optionally
prefixed with a label. Every group is drawn as mean +- std across its runs, one subplot per metric,
using the same trailing-window binning as the per-run learning curve.

Examples:
    python -m incremental_rl.plot_runs "results/moon_earth-*"
    python -m incremental_rl.plot_runs "moon=results/moon_earth-*" "earth=results/baseline-*" \\
        --metrics return critic_loss entropy actor_grad_norm --vlines 6000000 --out compare.png
"""
import os, csv, glob
import argparse, warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from incremental_rl.utils import binned_curve

DEFAULT_METRICS = ['return', 'ep_len', 'actor_loss', 'critic_loss', 'td_error', 'entropy',
                   'mod_action', 'actor_grad_norm', 'critic_grad_norm', 'actor_weight_norm', 'critic_weight_norm']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('groups', nargs='+', help='"glob" or "label=glob" of run folders / metrics.csv files')
    parser.add_argument('--metrics', nargs='+', default=None, help=f"Columns to plot (default: all of {DEFAULT_METRICS})")
    parser.add_argument('--x_tick', default=10000, type=int, help="Bin spacing in steps")
    parser.add_argument('--window', default=10000, type=int, help="Trailing averaging window in steps")
    parser.add_argument('--vlines', default=[], type=int, nargs='*', help="Steps to mark with a vertical line")
    parser.add_argument('--per_seed', action='store_true', help="Also draw each run faintly")
    parser.add_argument('--include_partial', action='store_true', help="Keep the final partial episode")
    parser.add_argument('--out', default=None, help="Output PNG (default: show interactively)")
    return parser.parse_args()


def find_metrics_files(pattern):
    files = []
    for path in sorted(glob.glob(pattern)):
        if os.path.isdir(path):
            files += sorted(glob.glob(os.path.join(path, "*_metrics.csv")))
        elif path.endswith(".csv"):
            files.append(path)
    return files


def load_run(path, include_partial):
    """ metrics.csv -> structured array (one field per column). csv module rather than np.genfromtxt, which renames 'return' """
    with open(path, newline='') as f:
        reader = csv.reader(f)
        names = next(reader)
        rows = [[float(v) for v in r] for r in reader if r]
    data = np.array([tuple(r) for r in rows], dtype=[(n, float) for n in names])
    if not include_partial and 'partial' in names:
        data = data[data['partial'] == 0]
    return data


def group_curves(runs, metric, x_tick, window):
    """ Bin each run, then align on the common step grid -> (x, per-run matrix with NaN past a run's end) """
    curves = [binned_curve(r['step'], r[metric], x_tick, window) for r in runs if metric in r.dtype.names]
    curves = [(m, x) for m, x in curves if len(x)]
    if not curves:
        return None, None
    x_all = np.arange(x_tick, max(x[-1] for _, x in curves) + 1, x_tick)
    mat = np.full((len(curves), len(x_all)), np.nan)
    for i, (m, x) in enumerate(curves):
        mat[i, np.searchsorted(x_all, x)] = m
    return x_all, mat


def main():
    args = parse_args()
    if args.out:
        matplotlib.use("Agg")

    groups = []
    for g in args.groups:
        label, pattern = g.split("=", 1) if "=" in g else (g, g)
        files = find_metrics_files(pattern)
        if not files:
            print(f"No metrics files match {pattern}")
            continue
        groups.append((label, [load_run(f, args.include_partial) for f in files]))
    if not groups:
        raise SystemExit("Nothing to plot")

    available = set.intersection(*[set(r.dtype.names) for _, runs in groups for r in runs])
    metrics = args.metrics or [m for m in DEFAULT_METRICS if m in available]
    ncol = min(3, len(metrics))
    nrow = -(-len(metrics) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 3.6 * nrow), squeeze=False)

    for ax, metric in zip(axes.flat, metrics):
        for label, runs in groups:
            x, mat = group_curves(runs, metric, args.x_tick, args.window)
            if x is None:
                continue
            with np.errstate(all='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN bins past the shortest run
                mean, std = np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)
            line, = ax.plot(x, mean, label=f"{label} (n={len(runs)})")
            ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.2, linewidth=0)
            if args.per_seed:
                for row in mat:
                    ax.plot(x, row, color=line.get_color(), alpha=0.15, linewidth=0.8)
        for v in args.vlines:
            ax.axvline(v, color="orange", linestyle="--", alpha=0.6)
        ax.set_title(metric)
        ax.set_xlabel("steps")
        ax.grid(alpha=0.3)
    axes.flat[0].legend()
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
