"""Learning curves per lever for the shift study, from the caches lever_recovery.py wrote.

    python scripts/analysis/lever_curves.py [--root results_trillium/results_study_runs_analysis] [--base legacy]

For every environment folder under --root and every lever, writes <env>/lever_curves/curves_<lever>.png: one panel
per shift with the shifted run without the lever (<base>-<shift>), the shifted run with the lever (<lever>-<shift>)
and the no-shift run (<base>-none).
"""
import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lever_recovery as lr

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results_trillium/results_study_runs_analysis")
    p.add_argument("--base", default="legacy")
    args = p.parse_args()
    for cache in sorted(glob.glob(os.path.join(args.root, "*", "cache"))):
        runs = {}
        for f in glob.glob(os.path.join(cache, "*.npz")):
            m = lr.RUN_RE.match(os.path.basename(f)[:-4])
            if m:
                runs.setdefault((m["cond"], m["shift"]), []).append(f)
        env = next(lr.RUN_RE.match(os.path.basename(f)[:-4])["env"] for fs in runs.values() for f in fs)
        load = lambda cond, shift: [np.load(f) for f in sorted(runs.get((cond, shift), []))]
        shifts = [s for s in lr.SHIFTS.split("|") if (args.base, s) in runs and s != "none"]
        levers = sorted({c for c, _ in runs} - {args.base, "scratch"})
        out = os.path.join(os.path.dirname(cache), "lever_curves")
        os.makedirs(out, exist_ok=True)
        none = load(args.base, "none")
        for lever in levers:
            ncol = (len(shifts) + 1) // 2
            fig, axes = plt.subplots(2, ncol, figsize=(4.6 * ncol, 8), sharey=True, squeeze=False)
            for ax, shift in zip(axes.flat, shifts):
                b, l = load(args.base, shift), load(lever, shift)
                x, N = b[0]["x"], int(b[0]["N"])
                if none:
                    lr.band(ax, x, np.array([c["return"] for c in none]), "#8a8983", ":", "no shift, no lever", lw=1.2)
                lr.band(ax, x, np.array([c["return"] for c in b]), "#0b0b0b", "-", f"{shift}, no lever", lw=1.6)
                if l:
                    lr.band(ax, x, np.array([c["return"] for c in l]), "#2a78d6", "-", f"{shift}, {lever}", lw=1.6)
                ax.axvline(N // 2, color="#8a8983", lw=0.8, ls="--")
                ax.set_xlim(0, N)
                ax.set_title(shift, fontsize=10)
                ax.legend(fontsize=7, frameon=False, loc="upper left")
                lr.style(ax)
            for ax in axes.flat[len(shifts):]:
                ax.axis("off")
            for ax in axes[:, 0]:
                ax.set_ylabel("return (mean ± SE over seeds)")
            for ax in axes[-1]:
                ax.set_xlabel("steps")
            fig.suptitle(f"{env}: {lever} against the same shifted run without it, per shift", fontsize=12)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"curves_{lever}.png"), dpi=120)
            plt.close(fig)
        print(f"wrote {out}: {len(levers)} levers x {len(shifts)} shifts")
