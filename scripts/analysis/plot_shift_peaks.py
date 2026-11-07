"""
Peak and final return of every segment of every shifted experiment, one image per env: a panel per experiment,
a pair of bars (peak, final) per segment, mean over seeds with std whiskers and one dot per seed, the segment's
dynamics as the x label, and (with --baseline) the from-scratch peak of the same dynamics as a short dashed line.

    python scripts/analysis/plot_shift_peaks.py results_trillium/results_24M --baseline results_trillium/results_24M --out_dir figs
"""
import os, sys, argparse, textwrap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from shifts import (load_runs, segments, segment_stats, baseline_peak_lookup, by_experiment, panel_grid, style,
                    caption, ENV_COLORS, GREY)

FINAL_COLOR = "#b5b3ab"


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


def bar(ax, pos, vals, color, label=None):
    ax.bar(pos, vals.mean(), width=0.36, color=color, alpha=0.85, zorder=2, label=label)
    ax.errorbar(pos, vals.mean(), yerr=vals.std(), fmt="none", ecolor="#52514e", capsize=3, lw=1, zorder=3)
    jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else [0]
    ax.scatter(pos + jitter, vals, s=14, color=color, edgecolor="white", linewidth=0.8, zorder=4)
    ax.annotate(f"{vals.mean():.0f}", (pos, max(vals.max(), vals.mean() + vals.std())), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color="#52514e")


def draw(ax, exp, runs, args, scratch, color):
    stats = [segment_stats(r, args.window, args.x_tick, 1000000, scratch) for r in runs]
    segs = segments(runs[0])
    labels, drew_ref = [], False
    for i, (start, end, state, label, change) in enumerate(segs):
        peak = np.array([s[i]["peak"] for s in stats])
        final = np.array([s[i]["final"] for s in stats])
        bar(ax, i - 0.2, peak, color, "peak" if i == 0 else None)
        bar(ax, i + 0.2, final, FINAL_COLOR, "final" if i == 0 else None)
        ref = stats[0][i]["scratch_peak"]
        if ref:
            ax.hlines(ref, i - 0.45, i + 0.45, color="#52514e", lw=1, ls=(0, (4, 2)), zorder=5,
                      label=None if drew_ref else "from scratch")
            drew_ref = True
        labels.append(f"{start / 1e6:g}-{end / 1e6:g}M\n" + "\n".join(textwrap.wrap(label, 18)))
    ax.set_xticks(range(len(segs)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.margins(y=0.18)
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
        fig, axes = panel_grid(plt, len(groups), cols=args.cols, h=3.4)
        for ax, (exp, g) in zip(axes, groups.items()):
            draw(ax, exp, g, args, scratch, ENV_COLORS.get(env, "#52514e"))
        for ax in axes[::args.cols]:
            ax.set_ylabel("return", fontsize=8)
        fig.suptitle(f"{env}: peak and final return per segment", fontsize=12)
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="upper right", fontsize=8, frameon=False, ncol=3, bbox_to_anchor=(0.99, 0.985))
        caption(fig, f"Per segment of constant dynamics: peak = highest trailing {args.window // 1000}k-step mean return "
                     f"inside the segment, final = its last value. Bar: mean over seeds. Whisker: ±1 std. Dot: one seed. "
                     f"x labels: the segment's steps and the dynamics in effect (g in m/s², x = scale on the default, "
                     f"z/angle = termination bound, ctrl/healthy = reward weights)."
                     + ("  Dashed: peak of the from-scratch run with the same dynamics." if scratch else ""))
        if args.out_dir:
            out = os.path.join(args.out_dir, f"shift_peaks_{env}.png")
            fig.savefig(out, dpi=140)
            print(f"wrote {out}")
            plt.close(fig)
    if not args.out_dir:
        plt.show()


if __name__ == "__main__":
    main()
