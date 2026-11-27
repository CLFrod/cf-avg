"""Cross-environment overview of the shift study, from the per-env folders lever_recovery.py wrote.

    python scripts/analysis/study_overview.py [--root results_trillium/results_study_runs_analysis] [--base legacy]

Reads <root>/<env>/{classes,baseline,recovery_summary}.csv and writes into <root>: classes.csv and class_grid.png
(recovery class per env and shift), baseline.csv, levers.csv (every lever on every env/shift, paired against the
base run), lever_scoreboard.csv (per lever: all pairs, per env, per class), lever_grid_<lever>.png and overview.md.
"""
import argparse, csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm

CLASS_NAME = {0: "unaffected", 1: "adapts", 2: "head start", 3: "scratch rate", 4: "impaired", 5: "scratch never learned"}
CLASS_COLOR = ["#d9d8d3", "#1baf7a", "#2a78d6", "#eda100", "#e34948", "#b9a3e3"]
SHIFT_ORDER = ["heavy", "gear2x", "gain2x", "stiff", "loose", "all_flipped", "payload", "heavy_object", "limb_heavy", "weak_joint",
               "dead_joint", "flipped_joint", "stiff_springs", "rotor"]


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def write(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def grid(ax, M, envs, shifts, txt, **kw):
    im = ax.imshow(M, aspect="auto", **kw)
    ax.set_xticks(range(len(shifts)))
    ax.set_xticklabels(shifts, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(envs)))
    ax.set_yticklabels(envs, fontsize=9)
    for i in range(len(envs)):
        for j in range(len(shifts)):
            ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=7.5, color="#0b0b0b")
    return im


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results_trillium/results_study_runs_analysis")
    p.add_argument("--base", default="legacy")
    args = p.parse_args()
    classes, baseline, summary = [], [], []
    for d in sorted(glob.glob(os.path.join(args.root, "*", "recovery_summary.csv"))):
        d = os.path.dirname(d)
        summary += read(os.path.join(d, "recovery_summary.csv"))
        for name, rows in (("classes.csv", classes), ("baseline.csv", baseline)):
            if os.path.exists(os.path.join(d, name)):
                rows += read(os.path.join(d, name))
    write(os.path.join(args.root, "classes.csv"), classes)
    write(os.path.join(args.root, "baseline.csv"), baseline)
    envs = sorted({r["env"] for r in classes})
    shifts = [s for s in SHIFT_ORDER if any(r["shift"] == s for r in classes)]
    cls = {(r["env"], r["shift"]): r for r in classes}
    T = {(r["env"], r["shift"], r["cond"], r["metric"]): r for r in summary}

    # recovery class per env and shift
    M = np.full((len(envs), len(shifts)), np.nan)
    txt = [["" for _ in shifts] for _ in envs]
    for i, e in enumerate(envs):
        for j, s in enumerate(shifts):
            r = cls.get((e, s))
            if r:
                M[i, j] = int(r["class"])
                txt[i][j] = (f"{r['class']}\ndip {num(r['dip_frac']):.2f}\nspd {num(r['speed_ratio']):.2f} asy {num(r['asym_ratio']):.2f}" if int(r["class"]) < 5
                             else f"5\ndip {num(r['dip_frac']):.2f}\nkept {num(r['asym_kept']):.2f}")
    fig, ax = plt.subplots(figsize=(1.25 * len(shifts) + 3, 0.75 * len(envs) + 2))
    grid(ax, M, envs, shifts, txt, cmap=ListedColormap(CLASS_COLOR), vmin=-0.5, vmax=5.5)
    ax.set_title("Recovery class of the base agent per environment and shift (0 unaffected, 1 adapts, 2 head start, 3 scratch rate, 4 impaired, 5 from-scratch never learned)\n"
                 "cell: class, dip as a fraction of pre-shift return, speed ratio and asymptote ratio against from-scratch", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.root, "class_grid.png"), dpi=140)
    plt.close(fig)

    # every lever on every pair, paired against the base run
    levers_rows, conds = [], sorted({r["cond"] for r in summary if r["cond"] not in (args.base, "scratch")})
    for (e, s), c in sorted(cls.items()):
        H, floor = num(c["N"]) - num(c["S"]), num(c["floor"])
        for cond in conds:
            if (e, s, cond, "regret_fix") not in T:
                continue
            g = lambda k, f="diff": num(T[(e, s, cond, k)][f])
            base_asym, base_early = num(T[(e, s, args.base, "asym")]["mean"]), num(T[(e, s, args.base, "early_1M")]["mean"])
            levers_rows.append({
                "env": e, "shift": s, "class": int(c["class"]), "cond": cond,
                "d_regret": g("regret_fix"), "regret_wins": g("regret_fix", "wins"), "regret_sig": T[(e, s, cond, "regret_fix")]["sig"],
                "d_t80_frac_H": g("t80_fix") / H, "t80_wins": g("t80_fix", "wins"),
                "d_early_frac": g("early_1M") / abs(base_early - floor), "early_wins": g("early_1M", "wins"), "early_sig": T[(e, s, cond, "early_1M")]["sig"],
                "asym_ratio": (num(T[(e, s, cond, "asym")]["mean"]) - floor) / (base_asym - floor), "asym_sig": T[(e, s, cond, "asym")]["sig"],
                "n_pairs": g("regret_fix", "n_pairs"),
            })
    write(os.path.join(args.root, "levers.csv"), levers_rows)

    def score(rows, group, key):
        v = lambda k: np.array([r[k] for r in rows], float)
        return {"group": group, "key": key, "cond": rows[0]["cond"], "pairs": len(rows), "d_regret": np.nanmean(v("d_regret")),
                "regret_better": int((v("d_regret") < 0).sum()), "regret_sig_better": sum(r["regret_sig"] != "" and r["d_regret"] < 0 for r in rows),
                "regret_sig_worse": sum(r["regret_sig"] != "" and r["d_regret"] > 0 for r in rows),
                "d_t80_frac_H": np.nanmean(v("d_t80_frac_H")), "d_early_frac": np.nanmean(v("d_early_frac")),
                "seed_wins_early": int(np.nansum(v("early_wins"))), "seed_pairs": int(np.nansum(v("n_pairs"))),
                "asym_ratio": np.nanmean(v("asym_ratio")), "worst_asym_ratio": np.nanmin(v("asym_ratio"))}

    board = []
    for cond in conds:
        rows = [r for r in levers_rows if r["cond"] == cond]
        board.append(score(rows, "all", "all"))
        for k in sorted({r["class"] for r in rows}):
            board.append(score([r for r in rows if r["class"] == k], "class", k))
        for e in envs:
            sub = [r for r in rows if r["env"] == e]
            if sub:
                board.append(score(sub, "env", e))
    write(os.path.join(args.root, "lever_scoreboard.csv"), board)

    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#d9d8d3", "#2a78d6"])
    for cond in conds:
        M = np.full((len(envs), len(shifts)), np.nan)
        txt = [["" for _ in shifts] for _ in envs]
        for r in levers_rows:
            if r["cond"] == cond:
                i, j = envs.index(r["env"]), shifts.index(r["shift"])
                M[i, j] = -100 * r["d_regret"]
                txt[i][j] = f"{M[i, j]:+.0f}\n{int(r['regret_wins'])}/{int(r['n_pairs'])}{r['regret_sig']}"
        fig, ax = plt.subplots(figsize=(1.1 * len(shifts) + 3, 0.6 * len(envs) + 2))
        im = grid(ax, M, envs, shifts, txt, cmap=cmap, norm=TwoSlopeNorm(0, -15, 15))
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="regret reduction, points")
        ax.set_title(f"{cond}: reduction in post-shift regret against the {args.base} run, in points of the {args.base} asymptote (blue = better)\n"
                     f"cell: value, seeds where the lever beat {args.base}, paired t-test", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.root, f"lever_grid_{cond}.png"), dpi=140)
        plt.close(fig)

    md = ["# Shift study overview\n", f"Base run: `{args.base}`. Per-environment tables and figures are in the environment folders.\n",
          "## Baselines (no shift)\n", "| env | seeds | return, whole run | return, before the halfway step | return, end of run |", "|---|---|---|---|---|"]
    for e in sorted({r["env"] for r in baseline}):
        b = [r for r in baseline if r["env"] == e]
        ms = lambda k: (np.mean([num(r[k]) for r in b]), np.std([num(r[k]) for r in b], ddof=1) / np.sqrt(len(b)))
        md.append(f"| {e} | {len(b)} | " + " | ".join(f"{m:.4g} ± {se:.3g}" for m, se in (ms("whole_run"), ms("mid"), ms("final"))) + " |")
    md += ["\n## Recovery classes\n", "| env | " + " | ".join(shifts) + " |", "|---|" + "---|" * len(shifts)]
    for e in envs:
        md.append(f"| {e} | " + " | ".join(str(cls[(e, s)]["class"]) if (e, s) in cls else "" for s in shifts) + " |")
    counts = {k: sum(int(r["class"]) == k for r in classes) for k in range(6)}
    md.append("\n" + ", ".join(f"class {k} ({CLASS_NAME[k]}): {n}" for k, n in counts.items()) + f"; {len(classes)} env/shift pairs.\n")
    md += ["## Levers against the base run\n", "| lever | scope | pairs | Δ regret | pairs better (sig. better / sig. worse) | Δ t80, share of horizon | Δ early return | seeds ahead early | asymptote ratio (mean, worst) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in board:
        if r["group"] in ("all", "class"):
            scope = "all" if r["group"] == "all" else f"class {r['key']}"
            md.append(f"| {r['cond']} | {scope} | {r['pairs']} | {r['d_regret']:+.3f} | {r['regret_better']} ({r['regret_sig_better']} / {r['regret_sig_worse']}) | "
                      f"{r['d_t80_frac_H']:+.3f} | {r['d_early_frac']:+.3f} | {r['seed_wins_early']}/{r['seed_pairs']} | {r['asym_ratio']:.2f}, {r['worst_asym_ratio']:.2f} |")
    with open(os.path.join(args.root, "overview.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {args.root}: {len(classes)} pairs, {len(levers_rows)} lever comparisons")
