"""
Recovery after each dynamics switch, one image per env: a panel per switch (experiment x switch), curves aligned
at the switch (x = steps since the switch), mean over seeds with a std band, the pre-switch level as a dotted
line, and (with --baseline) the from-scratch peak of the new dynamics as a dashed line. Each panel's title says
what the dynamics were before and after the switch.

    python scripts/analysis/plot_shift_recovery.py results_trillium/results_24M --baseline results_trillium/results_24M --out_dir figs
"""
import os, sys, argparse, textwrap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from shifts import (load_runs, segments, segment_stats, baseline_peak_lookup, by_experiment, smoothed, state_key,
                    panel_grid, style, caption, ENV_COLORS, GREY)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir")
    p.add_argument("--baseline", default=None)
    p.add_argument("--env", nargs="*", default=None)
    p.add_argument("--window", type=int, default=100000)
    p.add_argument("--x_tick", type=int, default=10000)
    p.add_argument("--before", type=int, default=1000000, help="Steps shown before the switch")
    p.add_argument("--after", type=int, default=6000000, help="Steps shown after the switch")
    p.add_argument("--horizon", type=int, default=1000000, help="Steps after the switch in which the drop is measured")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--incomplete", action="store_true")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def aligned(runs, switch, args):
    """ Smoothed curves of all seeds on a common grid of steps relative to `switch` """
    rel = np.arange(-args.before, args.after + 1, args.x_tick)
    ys = []
    for r in runs:
        c, x = smoothed(r, args.window, args.x_tick)
        lookup = dict(zip(x.astype(int), c))
        ys.append([lookup.get(int(switch + d), np.nan) for d in rel])
    return rel, np.array(ys)


def draw(ax, exp, runs, k, args, scratch, color):
    segs = segments(runs[0])
    start, end, state, label, change = segs[k]
    rel, ys = aligned(runs, start, args)
    ys[:, rel > end - start] = np.nan
    mean, std = np.nanmean(ys, axis=0), np.nanstd(ys, axis=0)
    ax.fill_between(rel / 1e6, mean - std, mean + std, color=color, alpha=0.15, lw=0)
    ax.plot(rel / 1e6, mean, color=color, lw=1.8)
    ax.axvline(0, color=GREY, lw=1, ls="--")
    stats = [segment_stats(r, args.window, args.x_tick, args.horizon)[k] for r in runs]
    pre = np.mean([s["pre"] for s in stats if s["pre"] is not None])
    edge = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.axhline(pre, color="#52514e", lw=0.9, ls=":")
    ax.text(0.01, pre, "pre", fontsize=6.5, color="#52514e", va="bottom", transform=edge)
    ref = scratch.get(state_key(runs[0]["env"], state)) if scratch else None
    if ref:
        ax.axhline(ref, color="#52514e", lw=0.9, ls=(0, (4, 2)))
        ax.text(0.99, ref, "scratch", fontsize=6.5, color="#52514e", va="bottom", ha="right", transform=edge)
    rec = [s["recover"] for s in stats if s["recover"] is not None]
    adapt = [s["adapt"] for s in stats if s["adapt"] is not None]
    drops = [s["drop"] for s in stats if s["drop"] is not None]
    note = (f"drop {np.mean(drops) / abs(pre):+.0%}   recovered {len(rec)}/{len(runs)}"
            + (f" in {np.mean(rec) / 1e6:.1f}M" if rec else "")
            + (f"   90% of peak at {np.mean(adapt) / 1e6:.1f}M" if adapt else ""))
    ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=6.5, color="#52514e")
    ax.set_title(f"{exp} · switch {k} at {start / 1e6:g}M\n{segs[k - 1][3]}  →  {label}", fontsize=8)
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
        panels = [(exp, g, k) for exp, g in groups.items() for k in range(1, len(g[0]["schedule"]) + 1)]
        fig, axes = panel_grid(plt, len(panels), cols=args.cols, h=3.2)
        for ax, (exp, g, k) in zip(axes, panels):
            draw(ax, exp, g, k, args, scratch, ENV_COLORS.get(env, "#52514e"))
        for ax in axes[-args.cols:]:
            ax.set_xlabel("steps since the switch (M)", fontsize=8)
        for ax in axes[::args.cols]:
            ax.set_ylabel("return", fontsize=8)
        fig.suptitle(f"{env}: recovery after each dynamics switch", fontsize=12)
        caption(fig, f"Mean over seeds ± 1 std of the trailing {args.window // 1000}k-step mean return, aligned at the "
                     f"switch. Dotted: return just before the switch (pre). Drop: lowest value within "
                     f"{args.horizon // 1000000}M steps after the switch, relative to pre. Recovered: seeds that got back to "
                     f"pre inside the segment, and the mean steps it took. 90% of peak: steps until the segment's own peak × 0.9."
                     + ("  Dashed: peak of the from-scratch run with the new dynamics." if scratch else ""))
        if args.out_dir:
            out = os.path.join(args.out_dir, f"shift_recovery_{env}.png")
            fig.savefig(out, dpi=140)
            print(f"wrote {out}")
            plt.close(fig)
    if not args.out_dir:
        plt.show()


if __name__ == "__main__":
    main()
