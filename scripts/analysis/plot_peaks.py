"""
Bar chart of peak (and final) return per condition, one image per env: mean over seeds, std error bar,
one dot per seed. Only runs without a mid-run dynamics shift (see peak_returns.py).

    python scripts/analysis/plot_peaks.py $SCRATCH/results --out_dir figures
    -> figures/peaks_Ant-v4.png, figures/peaks_Hopper-v4.png, ...
"""
import os, sys, glob, argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from peak_returns import load_run, summarise

CONDITION_ORDER = ["base", "moon", "moon_nohealthy", "moon_mid", "moon_adapted", "moon_wide_lowhealthy",
                   "moon_wide_nohealthy", "gear2x", "slippery", "heavy"]
COLORS = {"base": "#2a78d6", "moon": "#eb6834", "moon_nohealthy": "#c9512a", "moon_mid": "#f0906a",
          "moon_adapted": "#1baf7a", "moon_wide_lowhealthy": "#6250d6", "moon_wide_nohealthy": "#9d8de6",
          "gear2x": "#eda100", "slippery": "#e87ba4", "heavy": "#008300"}
EXTRA_COLORS = ["#e34948", "#7a7a7a", "#00a3b8"]
METRICS = {"peak": ("peak_smoothed", "Peak return"), "final": ("final_smoothed", "Final return")}
BOUND = {"Hopper-v4": {"mid": "angle ±0.35", "wide": "angle ±0.5"},
         "Ant-v4": {"mid": "z ≤ 1.5 m", "wide": "z ≤ 2.5 m"},
         "Humanoid-v4": {"mid": "z ≤ 2.75 m", "wide": "z ≤ 3.5 m"}}
LABELS = {
    "base": "base",
    "moon": "moon\ng = 1.62",
    "moon_nohealthy": "moon\nhealthy 0",
    "moon_mid": {env: f"moon\n{b['mid']}" for env, b in BOUND.items()},
    "moon_adapted": {env: f"moon\n{b['wide']}" for env, b in BOUND.items()},
    "moon_wide_lowhealthy": {env: f"moon\n{b['wide']}\nhealthy 0.2×" for env, b in BOUND.items()},
    "moon_wide_nohealthy": {env: f"moon\n{b['wide']}\nhealthy 0" for env, b in BOUND.items()},
    "gear2x": "gear ×2\ntorque",
    "slippery": "slippery\nfriction ×0.3",
    "heavy": "heavy\nmass ×2",
}
CAPTION = ("Bar: mean over seeds.  Whisker: ±1 std.  Dot: one seed.  g in m/s² (Earth 9.81); all moon bars use g = 1.62.\n"
           "angle / z: relaxed termination bound.  healthy: per-step survival reward (default 1, Humanoid 5).\n"
           "Training return smoothed over the last {window}k steps; peak = its maximum, final = its value at the end.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir")
    p.add_argument("--env", nargs="*", default=None, help="Keep only these envs (prefix match, e.g. Ant Hopper)")
    p.add_argument("--metrics", nargs="+", default=["peak", "final"], choices=list(METRICS))
    p.add_argument("--window", type=int, default=100000, help="Trailing window in steps (same as peak_returns.py)")
    p.add_argument("--x_tick", type=int, default=10000)
    p.add_argument("--incomplete", action="store_true", help="Include runs that have not reached N steps")
    p.add_argument("--out_dir", default=None, help="Write peaks_<env>.png here (default: show interactively)")
    return p.parse_args()


def collect(args):
    rows = []
    for d in sorted(glob.glob(os.path.join(args.results_dir, "*", "multi_experiment.json"))):
        run = load_run(os.path.dirname(d))
        if run is None or (args.env and not any(run["env"].startswith(e) for e in args.env)):
            continue
        if not args.incomplete and run["complete"] is False:
            continue
        rows.append(summarise(run, args.window, args.x_tick))
    return rows


def ordered(names):
    known = [c for c in CONDITION_ORDER if c in names]
    return known + sorted(set(names) - set(known))


def label_for(cond, env):
    label = LABELS.get(cond, cond)
    return label.get(env, cond) if isinstance(label, dict) else label


def draw_panel(ax, rows, key, conditions, color_of, env):
    for i, cond in enumerate(conditions):
        vals = np.array([r[key] for r in rows if r["experiment"] == cond])
        if not len(vals):
            continue
        ax.bar(i, vals.mean(), width=0.6, color=color_of[cond], alpha=0.85, zorder=2)
        ax.errorbar(i, vals.mean(), yerr=vals.std(), fmt="none", ecolor="#52514e", capsize=4, lw=1.2, zorder=3)
        jitter = np.linspace(-0.15, 0.15, len(vals)) if len(vals) > 1 else [0]
        ax.scatter(i + jitter, vals, s=28, color=color_of[cond], edgecolor="white", linewidth=1.2, zorder=4)
        top = max(vals.max(), vals.mean() + vals.std())
        ax.annotate(f"{vals.mean():.0f}", (i, top), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color="#52514e", zorder=5)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([label_for(c, env) for c in conditions], fontsize=8)
    ax.yaxis.grid(True, color="#e6e5e1", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(y=0.12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")


def seed_count(env_rows, conditions):
    n = {sum(r["experiment"] == c for r in env_rows) for c in conditions}
    return f"n = {min(n)} seeds" if len(n) == 1 else f"n = {min(n)}–{max(n)} seeds"


def plot_env(env, env_rows, conditions, color_of, args):
    fig, axes = plt.subplots(len(args.metrics), 1, figsize=(1.1 * len(conditions) + 1.5, 3.4 * len(args.metrics) + 1.3),
                             squeeze=False, sharex=True)
    for ax, metric in zip(axes[:, 0], args.metrics):
        key, label = METRICS[metric]
        draw_panel(ax, env_rows, key, conditions, color_of, env)
        ax.set_ylabel(label)
    fig.suptitle(f"{env}: return per dynamics condition ({seed_count(env_rows, conditions)})", fontsize=12)
    fig.text(0.5, 0.01, CAPTION.format(window=args.window // 1000), ha="center", va="bottom",
             fontsize=8, color="#8a8984", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def main():
    args = parse_args()
    if args.out_dir:
        matplotlib.use("Agg")
        os.makedirs(args.out_dir, exist_ok=True)
    rows = collect(args)
    if not rows:
        sys.exit(f"no unshifted runs found under {args.results_dir}")
    conditions = ordered({r["experiment"] for r in rows})
    color_of = {c: COLORS.get(c, EXTRA_COLORS[i % len(EXTRA_COLORS)]) for i, c in enumerate(conditions)}

    for env in sorted({r["env"] for r in rows}):
        fig = plot_env(env, [r for r in rows if r["env"] == env], conditions, color_of, args)
        if args.out_dir:
            out = os.path.join(args.out_dir, f"peaks_{env}.png")
            fig.savefig(out, dpi=150)
            print(f"wrote {out}")
            plt.close(fig)
    if not args.out_dir:
        plt.show()


if __name__ == "__main__":
    main()
