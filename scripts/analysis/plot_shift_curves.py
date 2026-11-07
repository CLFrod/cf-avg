"""
Full learning curves of every shifted experiment, one image per env: a panel per experiment with the per-seed
curves, their mean, shaded segments, the dynamics in effect written in each segment, and (with --baseline) the
peak of the from-scratch run with the same dynamics as a dashed level.

    python scripts/analysis/plot_shift_curves.py results_trillium/results_24M --baseline results_trillium/results_24M --out_dir figs
"""
import os, sys, argparse, textwrap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from shifts import (load_runs, segments, baseline_peak_lookup, by_experiment, mean_curve, state_key, panel_grid,
                    style, caption, ENV_COLORS, GREY, LIGHT)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir")
    p.add_argument("--baseline", default=None)
    p.add_argument("--env", nargs="*", default=None)
    p.add_argument("--window", type=int, default=100000)
    p.add_argument("--x_tick", type=int, default=10000)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--incomplete", action="store_true")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def draw(ax, exp, runs, args, scratch, color):
    x, mean, std, ys = mean_curve(runs, args.window, args.x_tick)
    for y in ys:
        ax.plot(x / 1e6, y, color=color, lw=0.7, alpha=0.3)
    ax.plot(x / 1e6, mean, color=color, lw=1.8)
    top = np.nanmax(ys) * 1.28
    ax.set_ylim(min(0, np.nanmin(ys)), top)
    for i, (start, end, state, label, change) in enumerate(segments(runs[0])):
        if i % 2:
            ax.axvspan(start / 1e6, end / 1e6, color=LIGHT, alpha=0.5, lw=0, zorder=0)
        if i:
            ax.axvline(start / 1e6, color=GREY, lw=1, ls="--")
        ax.text((start + end) / 2e6, top * 0.98, "\n".join(textwrap.wrap(label, 22)), ha="center", va="top",
                fontsize=6.5, color="#52514e")
        ref = scratch.get(state_key(runs[0]["env"], state)) if scratch else None
        if ref:
            ax.hlines(ref, start / 1e6, end / 1e6, color="#52514e", lw=0.9, ls=(0, (4, 2)))
    ax.set_title(f"{exp}  (n = {len(runs)})", fontsize=9)
    style(ax)


def main():
    args = parse_args()
    if args.out_dir:
        matplotlib.use("Agg")
        os.makedirs(args.out_dir, exist_ok=True)
    runs = load_runs(args.results_dir, envs=args.env, complete_only=not args.incomplete)
    if not runs:
        sys.exit(f"no shifted runs under {args.results_dir}")
    scratch = baseline_peak_lookup(args.baseline, args.window, args.x_tick, args.env) if args.baseline else None
    for env in sorted({r["env"] for r in runs}):
        groups = by_experiment([r for r in runs if r["env"] == env])
        fig, axes = panel_grid(plt, len(groups), cols=args.cols)
        for ax, (exp, g) in zip(axes, groups.items()):
            draw(ax, exp, g, args, scratch, ENV_COLORS.get(env, "#52514e"))
        for ax in axes[-args.cols:]:
            ax.set_xlabel("steps (M)", fontsize=8)
        for ax in axes[::args.cols]:
            ax.set_ylabel("return", fontsize=8)
        fig.suptitle(f"{env}: learning curves across dynamics shifts", fontsize=12)
        caption(fig, f"Thin lines: seeds. Thick: mean. Shaded bands: segments of constant dynamics; the text in each gives "
                     f"the dynamics in effect (g in m/s², x = scale on the default, z/angle = termination bound, "
                     f"ctrl/healthy = reward weights). Return is the trailing {args.window // 1000}k-step mean of training "
                     f"returns." + ("  Dashed level: peak of the from-scratch run with the same dynamics." if scratch else ""))
        if args.out_dir:
            out = os.path.join(args.out_dir, f"shift_curves_{env}.png")
            fig.savefig(out, dpi=140)
            print(f"wrote {out}")
            plt.close(fig)
    if not args.out_dir:
        plt.show()


if __name__ == "__main__":
    main()
