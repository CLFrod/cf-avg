"""One recovery-curve figure per method and env: the method against the two base runs only.

    python scripts/analysis/variant_curves.py [--cache results_trillium/lever_results/analysis/cache] [--out .../analysis/variants] [--base fix|legacy] [--batch 1|2]

Each figure has one panel per shift. Lines: base run with the legacy G reset (dotted red), base run with the
G-truncation fix (green) and the method (blue), mean +- SE over seeds of the 50k-window return. Dashed vertical line
= the switch, dotted horizontal line = 80% of the paired base run's final return (--base fix|legacy); the legend gives each run's time to reach it.
Needs the cache written by lever_recovery.py.
"""
import argparse, glob, os, re, sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(__file__))
from lever_recovery import WIN, band, style, fmt_steps, first_reach

SWITCH = {"Hopper-v4": 3_000_000, "Ant-v4": 6_000_000, "Humanoid-v4": 6_000_000}
SHIFTS = {"Hopper-v4": ["gear2x", "heavy"], "Ant-v4": ["gear2x", "limb_heavy"], "Humanoid-v4": ["gear2x", "heavy"]}
METHODS = [  # (condition, doc number, name)
    ("L02_shrink_obs", 1, "Observation normaliser soft reset"),
    ("L02_hard_obs", 2, "Observation normaliser hard reset"),
    ("L03_cap_obs", 3, "Observation normaliser window"),
    ("L04_shrink_scaler", 4, "TD error scaler soft reset"),
    ("L05_cap_scaler", 5, "TD error scaler window"),
    ("L06_warm_start", 6, "Use values from previous runs"),
    ("L07_reset_adam", 7, "Adam optimizer reset"),
    ("L08_critic_x3", 8, "Critic step size increase (3x)"),
    ("L08_critic_x10", 9, "Critic step size increase (10x)"),
    ("L09_actor_freeze", 10, "Actor freeze"),
    ("L10_actor_delayed", 11, "Delayed actor step size increase"),
    ("L11_adaptive", 12, "Adaptive step size"),
    ("L12_beta2", 13, "Adam beta2 = 0.99"),
    ("S01_stats_stack", 14, "Multi statistics reset"),
    ("S02_sink", 15, "Every method at once"),
]
METHODS_BATCH2 = [  # the privileged methods, numbered as in the Shift Recovery Methods doc
    ("L02_shrink_obs", "01", "Observation normaliser soft reset"),
    ("P1_alpha_x0.1", "16a", "Entropy coefficient cut (0.1x)"),
    ("P1_alpha_x0.3", "16b", "Entropy coefficient cut (0.3x)"),
    ("P2_alpha_track", "17", "Entropy tracking"),
    ("P3_reward_offset", "18", "Reward offset"),
    ("P4_priv_obs", "19", "Use observation normalizer values from previous runs"),
    ("P5_obs_alpha", "20", "Previous-run normalizer + entropy cut (0.1x)"),
    ("P6_obs_track", "21", "Previous-run normalizer + entropy tracking"),
    ("P7_obs_offset", "22", "Previous-run normalizer + reward offset"),
    ("P8_obs_alpha_offset", "23", "Previous-run normalizer + entropy cut (0.1x) + reward offset"),
    ("P9_obs_track_offset", "24", "Previous-run normalizer + entropy tracking + reward offset"),
    ("P10_obs_track_adam", "25", "Previous-run normalizer + entropy tracking + optimizer reset"),
]
METHODS_BATCH3 = [  # Greedy AVG study (Greedy AC actor update from step 0, legacy G reset)
    ("GAC", "26", "Greedy AVG"),
    ("GACx_obs_soft", "27", "Greedy AVG + observation normaliser soft reset"),
    ("GACx_obs_window", "28", "Greedy AVG + observation normaliser window"),
    ("GACx_priv_obs", "29", "Greedy AVG + observation normalizer values from previous runs"),
    ("GACx_reward_offset", "30", "Greedy AVG + reward offset"),
]
REF_LABEL = {"legacy": "base run (legacy G reset)", "fix": "base run (G-truncation fix)"}
VARIANT_COLOR = "#2a78d6"


def runs(cache, cond, env, shift):
    files = sorted(f for f in glob.glob(f"{cache}/{cond}-{shift}-{env}-*.npz") if "_dense" not in f)
    return [np.load(f) for f in files]


def t80_label(name, zs, S, thresh):
    """`name (80% in t)`: mean over seeds of the first time the windowed return reaches thresh; `>horizon` if censored"""
    ts = [first_reach(z["x"], z["return"], S, thresh, int(z["N"]) - S) for z in zs]
    cens = sum(c for _, c in ts)
    t = np.mean([t for t, _ in ts])
    return f"{name} (80% in {'>' if cens else ''}{fmt_steps(t)})"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="results_trillium/lever_results/analysis/cache")
    p.add_argument("--out", default="results_trillium/lever_results/analysis/variants")
    p.add_argument("--base", default="fix", help="Paired base condition (green); the other of fix/legacy is drawn in red if present")
    p.add_argument("--batch", type=int, default=1, choices=(1, 2, 3), help="1: the batch-1 methods and stacks; 2: the privileged methods; 3: the Greedy AVG study")
    args = p.parse_args()
    lists = {1: METHODS, 2: METHODS_BATCH2, 3: METHODS_BATCH3}
    methods = [(c, (f"{n:02d}" if isinstance(n, int) else n), name) for c, n, name in lists[args.batch]]
    other = "legacy" if args.base == "fix" else "fix"
    refs = [(other, REF_LABEL[other], "#e34948", ":"), (args.base, REF_LABEL[args.base], "#1baf7a", "-")]
    for env, shifts in SHIFTS.items():
        S = SWITCH[env]
        os.makedirs(os.path.join(args.out, env), exist_ok=True)
        data = {(c, s): runs(args.cache, c, env, s) for s in shifts for c in [args.base, other] + [m[0] for m in methods]}
        for cond, num, name in methods:
            fig, axes = plt.subplots(1, len(shifts), figsize=(6.5 * len(shifts), 5), sharey=True)
            for ax, shift in zip(axes, shifts):
                fix = data[(args.base, shift)]
                x, N = fix[0]["x"], int(fix[0]["N"])
                tgt = np.mean([np.nanmean(z["return"][x > N - 1_000_000]) for z in fix])
                for ref, label, col, ls in refs:
                    zs = data[(ref, shift)]
                    if not zs:
                        continue
                    band(ax, x, np.array([z["return"] for z in zs]), col, ls, t80_label(label, zs, S, 0.8 * tgt), lw=1.8)
                zs = data[(cond, shift)]
                band(ax, x, np.array([z["return"] for z in zs]), VARIANT_COLOR, "-", t80_label(name, zs, S, 0.8 * tgt), lw=1.8)
                ax.axvline(S, color="#8a8983", lw=0.8, ls="--")
                ax.axhline(0.8 * tgt, color="#8a8983", lw=0.8, ls=":")
                ax.set_xlim(S - 500_000, N)
                ax.set_title(f"{env.split('-')[0]} {shift}", fontsize=11)
                ax.set_xlabel("steps")
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
                ax.legend(fontsize=8, frameon=False, loc="lower right")
                style(ax)
            axes[0].set_ylabel(f"return ({WIN // 1000}k window, mean ± SE over {len(zs)} seeds)")
            fig.suptitle(f"{num.lstrip('0')}. {name}: recovery after the switch at {fmt_steps(S)} (dotted: 80% of the base run's final return)", fontsize=12)
            fig.tight_layout()
            slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            path = os.path.join(args.out, env, f"{num}_{slug}.png")
            fig.savefig(path, dpi=140)
            plt.close(fig)
            print(path)


if __name__ == "__main__":
    main()
