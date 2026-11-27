"""Critic eligibility trace study: recovery of every critic_lambda x critic_lr cell against the reference cell.

    python scripts/analysis/trace_recovery.py --root results_traces_runs/results_traces/hopper --out results_trillium/results_traces_runs_analysis/hopper

Runs are named <cell>_legacy-none, <cell>_legacy-<shift> and <cell>_scratch-<shift> (make_trace_configs.py); the
reference cell lam0_clr1x is the shift study's own run. Uses lever_recovery.py's curves and recovery metrics, with
the reference cell's asymptotes as the common targets. Writes cells.csv (one row per shift and cell, paired by seed
against the reference), baseline.csv, grid.csv (per cell over all shifts), summary.md and heatmaps.
"""
import argparse, csv, glob, json, os, re
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import lever_recovery as lr

RUN_RE = re.compile(rf"^lam(?P<lam>[\d.]+)(?P<on>on)?_clr(?P<clr>[\d.]+)x_(?P<kind>legacy|scratch)-(?P<shift>{lr.SHIFTS})-.+-(?P<seed>\d+)-\d{{8}}_\d{{6}}$")
REF = (0.0, 1.0)
T_CRIT = {2: 4.303, 3: 3.182, 4: 2.776}


def load(job):
    return lr.load_run(*job)


def mean_se(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def heat(path, M, lams, clrs, title, txt, vmax, lower_better=False):
    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#d9d8d3", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(1.5 * len(clrs) + 2.5, 0.9 * len(lams) + 1.8))
    im = ax.imshow(-M if lower_better else M, cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    ax.set_xticks(range(len(clrs)))
    ax.set_xticklabels([f"{c:g}x" for c in clrs])
    ax.set_yticks(range(len(lams)))
    ax.set_yticklabels([f"{l:g}" for l in lams])
    ax.set_xlabel("critic_lr (multiple of tuned)")
    ax.set_ylabel("critic_lambda")
    for i in range(len(lams)):
        for j in range(len(clrs)):
            ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    cache = os.path.join(args.out, "cache")
    os.makedirs(cache, exist_ok=True)
    root = os.path.join(args.root, "runs") if os.path.isdir(os.path.join(args.root, "runs")) else args.root
    runs, unfinished = {}, 0
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        m = RUN_RE.match(os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, "checkpoint.json")):
            continue
        meta = json.load(open(os.path.join(d, "checkpoint.json")))
        if meta["step"] < 0.99 * meta["N"]:
            unfinished += 1
            continue
        g = m.groupdict()
        runs[(g["shift"], (float(g["lam"]), float(g["clr"])), g["kind"], int(g["seed"]))] = d
    print(f"{len(runs)} finished runs, {unfinished} unfinished skipped")
    with Pool(args.workers) as pool:
        curves = dict(zip(runs, pool.map(load, [(d, cache) for d in runs.values()], chunksize=8)))
    cells = sorted({k[1] for k in runs})
    lams, clrs = sorted({c[0] for c in cells}), sorted({c[1] for c in cells}, reverse=True)
    shifts = sorted({k[0] for k in runs if k[0] != "none"})
    get = lambda shift, cell, kind: {k[3]: v for k, v in curves.items() if k[:3] == (shift, cell, kind)}
    N = int(next(iter(curves.values()))["N"])
    S = N // 2
    x = next(iter(curves.values()))["x"]
    last = x > N - lr.u(1_000_000)

    baseline = []
    for cell in cells:
        for seed, c in get("none", cell, "legacy").items():
            R = c["return"]
            baseline.append({"lam": cell[0], "clr": cell[1], "seed": seed, "whole_run": float(np.nanmean(R)),
                             "mid": float(np.nanmean(R[(x > S - lr.u(200_000)) & (x <= S)])), "final": float(np.nanmean(R[last]))})

    rows = []
    for shift in shifts:
        ref_leg, ref_scr = get(shift, REF, "legacy"), {s: lr.align_scratch(c, S) for s, c in get(shift, REF, "scratch").items()}
        tgt_fix = float(np.mean([np.nanmean(c["return"][last]) for c in ref_leg.values()]))
        tgt_scr = float(np.mean([np.nanmean(c["return"][last]) for c in ref_scr.values()]))
        lr.FLOOR = float(np.mean([np.nanmean(c["return"][(x > S) & (x <= S + lr.WIN)]) for c in ref_scr.values()]))
        ref_m = {s: lr.recovery(c, S, tgt_fix, tgt_scr, True) for s, c in ref_leg.items()}
        for cell in cells:
            leg, scr = get(shift, cell, "legacy"), {s: lr.align_scratch(c, S) for s, c in get(shift, cell, "scratch").items()}
            if not leg:
                continue
            m = {s: lr.recovery(c, S, tgt_fix, tgt_scr, True) for s, c in leg.items()}
            own_scr = float(np.mean([np.nanmean(c["return"][last]) for c in scr.values()]))
            m_own = {s: lr.recovery(c, S, tgt_fix, own_scr, True) for s, c in leg.items()}  # t80 against the cell's own scratch
            ms = {s: lr.recovery(c, S, tgt_fix, own_scr, False) for s, c in scr.items()}
            mean = lambda d, k: float(np.mean([v[k] for v in d.values()]))
            dip_frac = mean(m, "dip") / abs(mean(m, "pre") - lr.FLOOR)
            speed = mean(m_own, "t80_scr") / mean(ms, "t80_scr")
            aratio = (mean(m, "asym") - lr.FLOOR) / (own_scr - lr.FLOOR)
            row = {"shift": shift, "lam": cell[0], "clr": cell[1], "n": len(m), "pre": mean(m, "pre"), "dip_frac": dip_frac,
                   "t80_ref": mean(m, "t80_fix"), "early_1M": mean(m, "early_1M"), "asym": mean(m, "asym"), "regret": mean(m, "regret_fix"),
                   "scratch_asym": own_scr, "speed_ratio": speed, "asym_ratio": aratio, "class": lr.recovery_class(dip_frac, speed, aratio),
                   "ref_asym": tgt_fix, "ref_scratch_asym": tgt_scr, "floor": lr.FLOOR}
            for k in ("regret_fix", "asym", "early_1M"):
                d = np.array([m[s][k] - ref_m[s][k] for s in m if s in ref_m])
                dm, dse = mean_se(d)
                better = d < 0 if k == "regret_fix" else d > 0
                t = dm / dse if dse and dse > 0 else 0.0
                row.update({f"d_{k}": dm, f"d_{k}_se": dse, f"wins_{k}": int(better.sum()),
                            f"sig_{k}": "*" if cell != REF and abs(t) > T_CRIT.get(len(d) - 1, np.inf) else ""})
            rows.append(row)

    def write(name, data):
        with open(os.path.join(args.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    write("cells.csv", rows)
    write("baseline.csv", baseline)

    grid = []
    for cell in cells:
        rs = [r for r in rows if (r["lam"], r["clr"]) == cell]
        b = [r for r in baseline if (r["lam"], r["clr"]) == cell]
        d = np.array([r["d_regret_fix"] for r in rs])
        grid.append({"lam": cell[0], "clr": cell[1], "baseline_final": np.mean([r["final"] for r in b]), "baseline_final_se": mean_se([r["final"] for r in b])[1],
                     "baseline_mid": np.mean([r["mid"] for r in b]), "baseline_whole": np.mean([r["whole_run"] for r in b]),
                     "d_regret": d.mean(), "d_regret_se": mean_se(d)[1], "improved": int((d < -0.02).sum()), "harmed": int((d > 0.02).sum()),
                     "unaffected": int((np.abs(d) <= 0.02).sum()), "sig_better": sum(r["sig_regret_fix"] != "" and r["d_regret_fix"] < 0 for r in rs),
                     "sig_worse": sum(r["sig_regret_fix"] != "" and r["d_regret_fix"] > 0 for r in rs),
                     "seed_wins": sum(r["wins_regret_fix"] for r in rs), "seed_pairs": sum(r["n"] for r in rs),
                     "asym_vs_ref": np.mean([(r["asym"] - r["floor"]) / (r["ref_asym"] - r["floor"]) for r in rs]),
                     "scratch_vs_ref": np.mean([(r["scratch_asym"] - r["floor"]) / (r["ref_scratch_asym"] - r["floor"]) for r in rs]),
                     "pre": np.mean([r["pre"] for r in rs]), "dip_frac": np.mean([r["dip_frac"] for r in rs]),
                     "speed_ratio": np.mean([r["speed_ratio"] for r in rs]), "classes": " ".join(str(r["class"]) for r in rs)})
    write("grid.csv", grid)
    G = {(g["lam"], g["clr"]): g for g in grid}
    ref = G[REF]

    def mat(fn):
        return np.array([[fn(G[(l, c)]) for c in clrs] for l in lams])

    M = mat(lambda g: 100 * (g["baseline_final"] / ref["baseline_final"] - 1))
    heat(os.path.join(args.out, "heat_baseline_final.png"), M, lams, clrs, "No-shift baseline, final return: % vs lam0_clr1x (blue = better)",
         [[f"{G[(l, c)]['baseline_final']:.0f}\n{M[i, j]:+.0f}%" for j, c in enumerate(clrs)] for i, l in enumerate(lams)], 40)
    M = mat(lambda g: 100 * g["d_regret"])
    heat(os.path.join(args.out, "heat_regret.png"), M, lams, clrs, f"Post-shift regret vs lam0_clr1x, points, mean over {len(shifts)} shifts (blue = better)",
         [[f"{M[i, j]:+.1f}\n{G[(l, c)]['improved']}/{G[(l, c)]['unaffected']}/{G[(l, c)]['harmed']}" for j, c in enumerate(clrs)] for i, l in enumerate(lams)], 15, lower_better=True)
    M = mat(lambda g: 100 * (g["asym_vs_ref"] - 1))
    heat(os.path.join(args.out, "heat_asym.png"), M, lams, clrs, "Post-shift final return: % vs lam0_clr1x, mean over shifts (blue = better)",
         [[f"{M[i, j]:+.0f}%" for j in range(len(clrs))] for i in range(len(lams))], 40)
    M = mat(lambda g: 100 * (g["scratch_vs_ref"] - 1))
    heat(os.path.join(args.out, "heat_scratch.png"), M, lams, clrs, "From-scratch final return in the shifted dynamics: % vs lam0_clr1x (blue = better)",
         [[f"{M[i, j]:+.0f}%" for j in range(len(clrs))] for i in range(len(lams))], 40)

    # regret per shift and cell
    fig, axes = plt.subplots(2, (len(shifts) + 1) // 2, figsize=(3.2 * ((len(shifts) + 1) // 2), 6.5), squeeze=False)
    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#d9d8d3", "#2a78d6"])
    for ax, shift in zip(axes.flat, shifts):
        R = {(r["lam"], r["clr"]): r for r in rows if r["shift"] == shift}
        M = np.array([[-100 * R[(l, c)]["d_regret_fix"] for c in clrs] for l in lams])
        ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -20, 20), aspect="auto")
        for i, l in enumerate(lams):
            for j, c in enumerate(clrs):
                ax.text(j, i, f"{-M[i, j]:+.0f}{R[(l, c)]['sig_regret_fix']}", ha="center", va="center", fontsize=7.5)
        ax.set_xticks(range(len(clrs)))
        ax.set_xticklabels([f"{c:g}x" for c in clrs], fontsize=7)
        ax.set_yticks(range(len(lams)))
        ax.set_yticklabels([f"{l:g}" for l in lams], fontsize=7)
        ax.set_title(f"{shift} (class {R[REF]['class']})", fontsize=9)
    for ax in axes.flat[len(shifts):]:
        ax.axis("off")
    fig.suptitle("Change in post-shift regret vs lam0_clr1x, points (blue = better); rows critic_lambda, columns critic_lr", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "heat_regret_by_shift.png"), dpi=140)
    plt.close(fig)

    # return curves per shift: reference and the best cell on regret
    best = min((g for g in grid if (g["lam"], g["clr"]) != REF), key=lambda g: g["d_regret"])
    bc = (best["lam"], best["clr"])
    fig, axes = plt.subplots(2, (len(shifts) + 1) // 2, figsize=(4 * ((len(shifts) + 1) // 2), 7), squeeze=False)
    for ax, shift in zip(axes.flat, shifts):
        for cell, kind, col, ls in ((REF, "legacy", "#0b0b0b", "-"), (REF, "scratch", "#52514e", "--"), (bc, "legacy", "#2a78d6", "-"), (bc, "scratch", "#2a78d6", "--")):
            cs = get(shift, cell, kind)
            mat_ = np.array([(lr.align_scratch(c, S) if kind == "scratch" else c)["return"] for c in cs.values()])
            lr.band(ax, x, mat_, col, ls, f"lam{cell[0]:g}_clr{cell[1]:g}x {kind}", lw=1.3)
        ax.axvline(S, color="#8a8983", lw=0.8, ls="--")
        ax.set_xlim(S - lr.u(500_000), N)
        ax.set_title(shift, fontsize=9)
        lr.style(ax)
    axes.flat[0].legend(fontsize=6.5, frameon=False)
    for ax in axes.flat[len(shifts):]:
        ax.axis("off")
    fig.suptitle(f"Return through the switch: reference cell and lam{bc[0]:g}_clr{bc[1]:g}x (lowest mean regret), mean ± SE over seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "curves_ref_vs_best.png"), dpi=130)
    plt.close(fig)

    # learning curves, one figure per cell: the shifted run with and without the trace settings, a panel per shift
    cdir = os.path.join(args.out, "curves")
    os.makedirs(cdir, exist_ok=True)
    ret = lambda shift, cell: np.array([c["return"] for c in get(shift, cell, "legacy").values()])
    for cell in cells:
        if cell == REF:
            continue
        name = f"lam{cell[0]:g}_clr{cell[1]:g}x"
        panels = ["none"] + shifts
        ncol = (len(panels) + 1) // 2
        fig, axes = plt.subplots(2, ncol, figsize=(4.6 * ncol, 8), sharey=True, squeeze=False)
        for ax, shift in zip(axes.flat, panels):
            if shift != "none":
                lr.band(ax, x, ret("none", REF), "#8a8983", ":", "no shift, no lever", lw=1.2)
                ax.axvline(S, color="#8a8983", lw=0.8, ls="--")
            lr.band(ax, x, ret(shift, REF), "#0b0b0b", "-", "no lever" if shift == "none" else f"{shift}, no lever", lw=1.6)
            lr.band(ax, x, ret(shift, cell), "#2a78d6", "-", name if shift == "none" else f"{shift}, {name}", lw=1.6)
            ax.set_xlim(0, N)
            ax.set_title("no shift" if shift == "none" else shift, fontsize=10)
            ax.legend(fontsize=7, frameon=False, loc="upper left")
            lr.style(ax)
        for ax in axes.flat[len(panels):]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("return (50k window, mean ± SE)")
        for ax in axes[-1]:
            ax.set_xlabel("steps")
        fig.suptitle(f"{name} (critic_lambda {cell[0]:g}, critic_lr {cell[1]:g}x tuned) against the same run without it (critic_lambda 0, tuned critic_lr), per shift", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(cdir, f"curves_{name}.png"), dpi=120)
        plt.close(fig)

    md = ["# Critic eligibility trace: recovery per critic_lambda x critic_lr cell\n",
          f"Reference cell: lam0_clr1x (the shift study's run). {len(shifts)} shifts, switch at {S}, {N} steps. Regret, t80 and floors use the reference cell's "
          "asymptotes, so cells are compared on one yardstick; change in regret is paired by seed against the reference.\n",
          "| lambda | critic_lr | baseline final | baseline mid | Δ regret (mean ± SE over shifts) | improved / unaffected / harmed | sig. better / worse | seeds ahead | post-shift final vs ref | scratch final vs ref | classes |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in grid:
        md.append(f"| {g['lam']:g} | {g['clr']:g}x | {g['baseline_final']:.0f} ± {g['baseline_final_se']:.0f} | {g['baseline_mid']:.0f} | {g['d_regret']:+.3f} ± {g['d_regret_se']:.3f} | "
                  f"{g['improved']} / {g['unaffected']} / {g['harmed']} | {g['sig_better']} / {g['sig_worse']} | {g['seed_wins']}/{g['seed_pairs']} | "
                  f"{g['asym_vs_ref']:.2f} | {g['scratch_vs_ref']:.2f} | {g['classes']} |")
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
