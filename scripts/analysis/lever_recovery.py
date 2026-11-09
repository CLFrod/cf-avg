"""Recovery analysis of the lever study: per-run recovery metrics, paired comparisons to the fix-only split, figures.

    python scripts/analysis/lever_recovery.py [--root results_trillium/lever_results/results] [--out .../analysis] [--dense]

Writes recovery_metrics.csv (one row per run), recovery_summary.csv (per condition, with paired deltas vs fix),
summary.md, heatmaps across all env/shift pairs, recovery curves and recovery-time plots per env/shift, and a
diagnostics grid per condition. --dense adds per-step plots of the first 100k steps after the switch.
"""
import argparse, csv, glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

RUN_RE = re.compile(r"^(?P<cond>.+?)-(?P<shift>gear2x|heavy|limb_heavy)-(?P<env>Hopper-v4|Ant-v4|Humanoid-v4)-(?P<seed>\d+)-\d{8}_\d{6}$")
TICK, WIN = 10_000, 50_000
DTICK = 1_000
COLS = ["return", "ep_len", "td_error", "td_signed", "td_abs_raw", "td_sigma", "critic_loss", "actor_loss", "entropy",
        "actor_grad_norm", "critic_grad_norm", "dq_da_norm", "actor_update_ratio", "critic_update_ratio",
        "actor_weight_norm", "critic_weight_norm", "scaler_G_count", "scaler_G2_mean", "scaler_reward_var", "obs_count",
        "obs_mean_drift", "obs_var_drift", "lr_mult_actor", "lr_mult_critic", "mod_action"]
DENSE_COLS = ["reward", "td_signed", "td_sigma", "actor_grad_norm", "critic_grad_norm", "dq_da_norm",
              "actor_update_ratio", "critic_update_ratio", "entropy", "terminated"]
BATCH1_GROUPS = {
    "Statistics levers": ["L02_shrink_obs", "L02_hard_obs", "L03_cap_obs", "L04_shrink_scaler", "L05_cap_scaler", "L06_warm_start"],
    "Optimiser & step-size levers": ["L07_reset_adam", "L08_critic_x3", "L08_critic_x10", "L09_actor_freeze", "L10_actor_delayed", "L11_adaptive", "L12_beta2"],
    "Stacks & G-reset reference": ["S01_stats_stack", "S02_sink", "legacy"],
}
BATCH2_GROUPS = {
    "Privileged singles": ["L02_shrink_obs", "P1_alpha_x0.1", "P1_alpha_x0.3", "P2_alpha_track", "P3_reward_offset", "P4_priv_obs"],
    "Stacks on privileged obs stats": ["P5_obs_alpha", "P6_obs_track", "P7_obs_offset", "P8_obs_alpha_offset", "P9_obs_track_offset", "P10_obs_track_adam"],
}
GROUPS = BATCH1_GROUPS  # replaced by select_groups() once the conditions are known
ORDER = [c for g in GROUPS.values() for c in g]


def select_groups(conds):
    """Pick the panel layout whose conditions appear in the results"""
    global GROUPS, ORDER
    for g in (BATCH2_GROUPS, BATCH1_GROUPS):
        if any(c in conds for c in g[list(g)[-1]]):
            GROUPS = g
            break
    ORDER = [c for g in GROUPS.values() for c in g]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REF_STYLE = {"fix": ("#0b0b0b", "-"), "scratch": ("#52514e", "--"), "legacy": ("#8a8983", ":")}
LOWER_BETTER = {"dip", "dip_frac", "t_min", "t80_fix", "t80_scr", "t90_pre", "regret_fix", "cg_peak", "cg_settle", "td_peak", "td_settle"}
T_CRIT = {2: (4.303, 9.925), 3: (3.182, 5.841), 4: (2.776, 4.604)}  # df -> two-sided 0.05, 0.01
DIAG = ["return", "ep_len", "td_abs_raw", "td_sigma", "critic_loss", "entropy", "actor_grad_norm", "critic_grad_norm",
        "dq_da_norm", "actor_update_ratio", "critic_update_ratio", "obs_count", "scaler_G_count", "lr_mult_actor",
        "lr_mult_critic", "critic_weight_norm"]
LOGY = {"td_abs_raw", "td_sigma", "critic_loss", "actor_grad_norm", "critic_grad_norm", "dq_da_norm", "actor_update_ratio",
        "critic_update_ratio", "obs_count", "scaler_G_count"}


def color_of(cond):
    if cond in REF_STYLE:
        return REF_STYLE[cond]
    for members in GROUPS.values():
        if cond in members:
            return PALETTE[members.index(cond)], "-"
    return "#999999", "-"


def discover(root):
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        m = RUN_RE.match(os.path.basename(d))
        if m and os.path.exists(os.path.join(d, "checkpoint.json")):
            g = m.groupdict()
            runs.setdefault((g["env"], g["shift"]), {}).setdefault(g["cond"], []).append((int(g["seed"]), d))
    return runs


def read_csv(path):
    with open(path) as f:
        names = f.readline().strip().split(",")
    return names, np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def window_means(steps, mat, names, cols, lo, nb, tick, win_bins):
    """Trailing-window means of `cols` on a regular grid of `nb` bins of `tick` steps starting at `lo`"""
    idx = np.clip((steps - lo - 1) // tick, 0, nb - 1).astype(int)
    out = {}
    for c in cols:
        if c not in names:
            continue
        v = mat[:, names.index(c)]
        ok = np.isfinite(v)
        s, n = np.bincount(idx[ok], weights=v[ok], minlength=nb), np.bincount(idx[ok], minlength=nb).astype(float)
        cs, cn = np.cumsum(np.r_[0.0, s]), np.cumsum(np.r_[0.0, n])
        ws, wn = np.r_[cs[1:win_bins], cs[win_bins:] - cs[:-win_bins]], np.r_[cn[1:win_bins], cn[win_bins:] - cn[:-win_bins]]
        out[c] = np.where(wn > 0, ws / np.maximum(wn, 1), np.nan)
    return out


def load_run(folder, cache_dir):
    cache = os.path.join(cache_dir, os.path.basename(folder) + ".npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    meta = json.load(open(os.path.join(folder, "checkpoint.json")))
    N = int(meta["N"])
    names, mat = read_csv(glob.glob(os.path.join(folder, "*_metrics.csv"))[0])
    mat = mat[mat[:, names.index("partial")] == 0]
    nb = N // TICK
    cur = window_means(mat[:, names.index("step")], mat, names, COLS, 0, nb, TICK, WIN // TICK)
    cur["x"] = np.arange(1, nb + 1) * TICK
    cur["N"] = np.array(N)
    np.savez(cache, **cur)
    return cur


def load_dense(folder, S, cache_dir):
    cache = os.path.join(cache_dir, os.path.basename(folder) + "_dense.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    files = glob.glob(os.path.join(folder, "*_dense.csv"))
    if not files:
        return None
    names, mat = read_csv(files[0])
    names.append("td_abs")
    mat = np.c_[mat, np.abs(mat[:, names.index("td_signed")])]
    lo, nb = S - 20_000, 220
    cur = window_means(mat[:, names.index("step")], mat, names, DENSE_COLS + ["td_abs"], lo, nb, DTICK, 1)
    cur["x"] = lo + np.arange(1, nb + 1) * DTICK
    np.savez(cache, **cur)
    return cur


def switch_step(folders):
    for _, d in folders:
        sched = json.load(open(os.path.join(d, "multi_experiment.json")))["hyperparams"]["schedule"]
        if sched:
            return int(sched[0]["step"])
    raise ValueError("no scheduled run found")


def align_scratch(cur, S):
    """Shift a from-scratch run so its step 0 sits at the switch; keeps the common grid"""
    k = S // TICK
    out = {"x": cur["x"], "N": cur["N"]}
    for c in COLS:
        if c in cur:
            a = np.full_like(cur[c], np.nan)
            a[k:] = cur[c][: len(a) - k]
            out[c] = a
    return out


def first_reach(x, v, S, thresh, H):
    i = np.where((x >= S + WIN) & (v >= thresh))[0]
    return (float(x[i[0]] - S), 0) if len(i) else (float(H), 1)


def settle(x, v, S, H):
    """Peak ratio of a diagnostic after the switch vs its pre-switch median, and steps until it is back within 1.5x"""
    pre = np.nanmedian(v[(x > S - 200_000) & (x <= S)])
    early = (x >= S + WIN) & (x <= S + 500_000)
    if not np.isfinite(pre) or pre <= 0 or not np.any(np.isfinite(v[early])):
        return np.nan, np.nan, np.nan
    i_peak = np.where(early)[0][np.nanargmax(v[early])]
    back = np.where((np.arange(len(x)) > i_peak) & (v <= 1.5 * pre))[0]
    return float(np.nanmax(v[early]) / pre), float(x[back[0]] - S) if len(back) else float(H), float(len(back) == 0)


def recovery(cur, S, tgt_fix, tgt_scr, has_pre):
    x, R, N = cur["x"], cur["return"], int(cur["N"])
    H = N - S
    post = x >= S + WIN  # the first window after the switch still holds pre-switch episodes
    early = post & (x <= S + 500_000)
    m = {"pre": np.nanmean(R[(x > S - 200_000) & (x <= S)]) if has_pre else np.nan}
    m["dip"] = m["pre"] - np.nanmin(R[early])
    m["dip_frac"] = m["dip"] / abs(m["pre"]) if has_pre else np.nan
    m["t_min"] = float(x[early][np.nanargmin(R[early])] - S)
    m["t80_fix"], m["t80_fix_cens"] = first_reach(x, R, S, 0.8 * tgt_fix, H)
    m["t80_scr"], m["t80_scr_cens"] = first_reach(x, R, S, 0.8 * tgt_scr, H)
    m["t90_pre"], m["t90_pre_cens"] = first_reach(x, R, S, 0.9 * m["pre"], H) if has_pre else (np.nan, np.nan)
    for k, w in (("early_500k", 500_000), ("early_1M", 1_000_000), ("early_2M", 2_000_000)):
        m[k] = np.nanmean(R[post & (x <= S + w)])
    m["asym"] = np.nanmean(R[x > N - 1_000_000])
    m["regret_fix"] = np.nanmean(np.maximum(tgt_fix - R[post], 0)) / abs(tgt_fix)
    m["cg_peak"], m["cg_settle"], m["cg_settle_cens"] = settle(x, cur["critic_grad_norm"], S, H) if has_pre else (np.nan,) * 3
    m["td_peak"], m["td_settle"], m["td_settle_cens"] = settle(x, cur["td_abs_raw"], S, H) if has_pre else (np.nan,) * 3
    return m


def mean_se(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan


def band(ax, x, mat, color, ls, label, lw=1.6):
    with np.errstate(all="ignore"):
        mean = np.nanmean(mat, axis=0)
        se = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(mat), axis=0)) if len(mat) > 1 else np.zeros_like(mean)
    ax.plot(x, mean, color=color, ls=ls, lw=lw, label=label)
    ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.15, lw=0)


def style(ax):
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fmt_steps(v):
    return f"{v / 1e6:.2f}M" if np.isfinite(v) else "n/a"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="results_trillium/lever_results/results")
    p.add_argument("--out", default="results_trillium/lever_results/analysis")
    p.add_argument("--dense", action="store_true", help="also plot the per-step dense logs (slow on first run)")
    p.add_argument("--only", nargs="*", default=None, help="restrict to these env or shift names")
    args = p.parse_args()
    cache_dir = os.path.join(args.out, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    runs = discover(args.root)
    if args.only:
        runs = {k: v for k, v in runs.items() if k[0] in args.only or k[1] in args.only}
    select_groups({c for conds in runs.values() for c in conds})
    print(f"{sum(len(s) for c in runs.values() for s in c.values())} runs in {len(runs)} env/shift pairs")

    per_run, summary, md = [], [], ["# Lever study batch 1: recovery after the physics switch\n",
                                    f"Return curves use a {WIN // 1000}k-step trailing window evaluated every {TICK // 1000}k steps. "
                                    "`t80_fix` is the number of steps after the switch until the curve first reaches 80% of the fix-only "
                                    "runs' post-shift asymptote (mean return over the final 1M steps, averaged over seeds); a run that never "
                                    "gets there is counted at the horizon and marked censored. `early_1M` is the mean return over the first 1M "
                                    "post-switch steps. Deltas are paired by seed against fix; wins count seeds where the lever beat fix; "
                                    "`*`/`**` mark a paired t-test at 0.05/0.01 (df = seeds - 1). From-scratch runs are aligned so their step 0 is the switch.\n"]
    heat, gfix = {}, {}
    for (env, shift), conds in sorted(runs.items()):
        S = switch_step(conds["fix"])
        print(f"== {env} {shift}: switch at {S}, {len(conds)} conditions")
        curves = {}
        for cond, folders in conds.items():
            curves[cond] = {}
            for seed, d in sorted(folders):
                cur = load_run(d, cache_dir)
                curves[cond][seed] = align_scratch(cur, S) if cond == "scratch" else cur
        N = int(next(iter(curves["fix"].values()))["N"])
        x = next(iter(curves["fix"].values()))["x"]
        H = N - S
        tgt_fix = float(np.mean([np.nanmean(c["return"][x > N - 1_000_000]) for c in curves["fix"].values()]))
        tgt_scr = float(np.mean([np.nanmean(c["return"][x > N - 1_000_000]) for c in curves["scratch"].values()]))
        metrics = {cond: {seed: recovery(c, S, tgt_fix, tgt_scr, cond != "scratch") for seed, c in cs.items()} for cond, cs in curves.items()}
        for cond, ms in metrics.items():
            for seed, m in ms.items():
                per_run.append({"env": env, "shift": shift, "cond": cond, "seed": seed, "S": S, "N": N, **m})

        keys = [k for k in next(iter(metrics["fix"].values())) if not k.endswith("_cens")]
        rows = {}
        for cond, ms in metrics.items():
            for k in keys:
                mean, se = mean_se([m[k] for m in ms.values()])
                row = {"env": env, "shift": shift, "cond": cond, "n": len(ms), "metric": k, "mean": mean, "se": se,
                       "censored": int(np.nansum([m.get(k + "_cens", 0) for m in ms.values()]))}
                fm, _ = mean_se([m[k] for m in metrics["fix"].values()])
                row["fix_mean"] = fm
                pairs = [(ms[s][k], metrics["fix"][s][k]) for s in ms if s in metrics["fix"]]
                diffs = np.array([a - b for a, b in pairs if np.isfinite(a) and np.isfinite(b)])
                if cond != "fix" and len(diffs) >= 2:
                    row["diff"], row["diff_se"] = mean_se(diffs)
                    better = diffs < 0 if k in LOWER_BETTER else diffs > 0
                    row["wins"], row["n_pairs"] = int(better.sum()), len(diffs)
                    t = row["diff"] / row["diff_se"] if row["diff_se"] and row["diff_se"] > 0 else 0.0
                    c05, c01 = T_CRIT.get(len(diffs) - 1, (np.inf, np.inf))
                    row["sig"] = "**" if abs(t) > c01 else "*" if abs(t) > c05 else ""
                else:
                    row.update(diff=np.nan, diff_se=np.nan, wins=np.nan, n_pairs=len(diffs), sig="")
                summary.append(row)
                rows[(cond, k)] = row
        for k in ("early_1M", "t80_fix", "asym", "regret_fix"):
            heat.setdefault(k, {})[(env, shift)] = {c: rows[(c, k)] for c in conds if c != "fix"}
        heat.setdefault("H", {})[(env, shift)] = H

        # markdown table
        md.append(f"\n## {env}, {shift} (switch at {fmt_steps(S)}, horizon {fmt_steps(H)}; fix asymptote {tgt_fix:.0f}, scratch asymptote {tgt_scr:.0f})\n")
        md.append("| condition | pre-shift | dip | t80_fix (cens.) | Δ t80 vs fix | early_1M | Δ vs fix | asymptote | regret |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for cond in ["fix", "legacy", "scratch"] + [c for c in ORDER if c in conds and c not in ("legacy",)]:
            if cond not in conds:
                continue
            r = {k: rows[(cond, k)] for k in keys}
            d80 = r["t80_fix"]
            d1m = r["early_1M"]
            dcol = lambda rr, f: "" if not np.isfinite(rr["diff"]) else f"{f(rr['diff'])} ({rr['wins']}/{rr['n_pairs']}){rr['sig']}"
            md.append(f"| {cond} | {r['pre']['mean']:.0f} | {r['dip']['mean']:.0f} | {fmt_steps(d80['mean'])} ({d80['censored']}/{d80['n']}) | "
                      f"{dcol(d80, lambda v: ('+' if v > 0 else '') + fmt_steps(abs(v)) if v >= 0 else '-' + fmt_steps(abs(v)))} | "
                      f"{d1m['mean']:.0f} ± {d1m['se']:.0f} | {dcol(d1m, lambda v: f'{v:+.0f}')} | {r['asym']['mean']:.0f} ± {r['asym']['se']:.0f} | {r['regret_fix']['mean']:.2f} |")

        md.append(f"\nPolicy response to the switch (pre = last 200k steps before it, post = 50k..300k after it):\n")
        md.append("| condition | entropy pre → post | actor grad norm post/pre | ep_len pre → post | critic loss post/pre |")
        md.append("|---|---|---|---|---|")
        for cond in ["fix", "legacy"] + [c for c in ORDER if c in conds and c != "legacy"]:
            if cond not in conds:
                continue
            def win(k, lo, hi):
                return np.mean([np.nanmean(c[k][(x > S + lo) & (x <= S + hi)]) for c in curves[cond].values()])
            e0, e1 = win("entropy", -200_000, 0), win("entropy", 50_000, 300_000)
            g0, g1 = win("actor_grad_norm", -200_000, 0), win("actor_grad_norm", 50_000, 300_000)
            l0, l1 = win("ep_len", -200_000, 0), win("ep_len", 50_000, 300_000)
            c0, c1 = win("critic_loss", -200_000, 0), win("critic_loss", 50_000, 300_000)
            md.append(f"| {cond} | {e0:.1f} → {e1:.1f} | ×{g1 / g0:.2f} | {l0:.0f} → {l1:.0f} | ×{c1 / c0:.2f} |")

        # recovery curves, one panel per lever group
        fig, axes = plt.subplots(1, len(GROUPS), figsize=(6 * len(GROUPS), 5), sharey=True)
        for ax, (title, members) in zip(axes, GROUPS.items()):
            for ref in ("fix", "scratch"):
                col, ls = REF_STYLE[ref]
                band(ax, x, np.array([c["return"] for c in curves[ref].values()]), col, ls, ref, lw=1.8)
            for cond in members:
                if cond in curves:
                    col, ls = color_of(cond)
                    band(ax, x, np.array([c["return"] for c in curves[cond].values()]), col, ls, cond)
            ax.axvline(S, color="#8a8983", lw=0.8, ls="--")
            ax.axhline(0.8 * tgt_fix, color="#8a8983", lw=0.8, ls=":")
            ax.set_xlim(S - 500_000, N)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("steps")
            ax.legend(fontsize=8, frameon=False)
            style(ax)
        axes[0].set_ylabel(f"return ({WIN // 1000}k window, mean ± SE over seeds)")
        fig.suptitle(f"{env} {shift}: recovery after the switch at {fmt_steps(S)} (dotted: 80% of fix asymptote)", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"curves_{env}_{shift}.png"), dpi=140)
        plt.close(fig)

        # recovery time and early return per condition, seeds as dots
        conds_plot = [c for c in ORDER if c in conds] + ["scratch"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(conds_plot) + 1.5), sharey=True)
        yl = np.arange(len(conds_plot))
        for ax, k, label in zip(axes, ("t80_fix", "early_1M"), ("steps after switch to reach 80% of fix asymptote", "mean return, first 1M steps after switch")):
            fv = [metrics["fix"][s][k] for s in metrics["fix"]]
            ax.axvspan(np.mean(fv) - np.std(fv, ddof=1) / np.sqrt(len(fv)), np.mean(fv) + np.std(fv, ddof=1) / np.sqrt(len(fv)), color="#0b0b0b", alpha=0.08, lw=0)
            ax.axvline(np.mean(fv), color="#0b0b0b", lw=1.2, label="fix (mean ± SE)")
            for yi, cond in zip(yl, conds_plot):
                col, _ = color_of(cond)
                vals = np.array([metrics[cond][s][k] for s in metrics[cond]])
                cens = np.array([metrics[cond][s].get(k + "_cens", 0) for s in metrics[cond]], bool)
                ax.scatter(vals[~cens], np.full((~cens).sum(), yi), color=col, s=28, zorder=3, edgecolor="white", lw=0.8)
                ax.scatter(vals[cens], np.full(cens.sum(), yi), color=col, s=40, marker=">", zorder=3, edgecolor="white", lw=0.8)
                ax.plot([vals.mean()], [yi], marker="|", color=col, ms=16, mew=2.2, zorder=4)
            ax.set_yticks(yl)
            ax.set_yticklabels(conds_plot, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel(label)
            style(ax)
        axes[0].set_xlim(0, H * 1.05)
        axes[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
        axes[0].legend(fontsize=8, frameon=False, loc="lower right")
        fig.suptitle(f"{env} {shift}: recovery time and early return per condition (dots: seeds, bar: mean, ▸: never reached)", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"recovery_times_{env}_{shift}.png"), dpi=140)
        plt.close(fig)

        # mechanism panels: what the switch does to the policy and critic, per lever group
        mech = ["ep_len", "entropy", "actor_grad_norm", "critic_loss"]
        fig, axes = plt.subplots(len(mech), len(GROUPS), figsize=(6 * len(GROUPS), 3.2 * len(mech)), sharex=True)
        for j, (title, members) in enumerate(GROUPS.items()):
            for i, k in enumerate(mech):
                ax = axes[i, j]
                for ref in ("fix", "scratch"):
                    band(ax, x, np.array([c[k] for c in curves[ref].values()]), *REF_STYLE[ref], ref, lw=1.8)
                for cond in members:
                    if cond in curves:
                        band(ax, x, np.array([c[k] for c in curves[cond].values()]), *color_of(cond), cond)
                ax.axvline(S, color="#8a8983", lw=0.8, ls="--")
                ax.set_xlim(S - 300_000, min(N, S + 2_000_000))
                if k in LOGY:
                    ax.set_yscale("log")
                if j == 0:
                    ax.set_ylabel(k)
                if i == 0:
                    ax.set_title(title, fontsize=11)
                style(ax)
            axes[0, j].legend(fontsize=7, frameon=False, ncol=2)
        fig.suptitle(f"{env} {shift}: policy and critic diagnostics through the switch ({WIN // 1000}k window, mean ± SE)", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"mechanism_{env}_{shift}.png"), dpi=130)
        plt.close(fig)
        gfix.setdefault((env, shift), {ref: curves[ref] for ref in ("fix", "legacy") if ref in curves})

        # diagnostics grid, lever vs fix
        ddir = os.path.join(args.out, "diag", f"{env}_{shift}")
        os.makedirs(ddir, exist_ok=True)
        for cond in conds:
            if cond == "fix":
                continue
            fig, axes = plt.subplots(4, 4, figsize=(18, 12))
            for ax, k in zip(axes.flat, DIAG):
                if k not in next(iter(curves["fix"].values())):
                    ax.axis("off")
                    continue
                band(ax, x, np.array([c[k] for c in curves["fix"].values()]), *REF_STYLE["fix"], "fix")
                if cond == "scratch" or k in next(iter(curves[cond].values())):
                    band(ax, x, np.array([c[k] for c in curves[cond].values()]), *color_of(cond), cond)
                ax.axvline(S, color="#8a8983", lw=0.8, ls="--")
                ax.set_xlim(S - 500_000, min(N, S + 3_000_000))
                if k in LOGY:
                    ax.set_yscale("log")
                ax.set_title(k, fontsize=10)
                style(ax)
            axes.flat[0].legend(fontsize=8, frameon=False)
            fig.suptitle(f"{env} {shift}: {cond} vs fix around the switch ({WIN // 1000}k window, mean ± SE)", fontsize=12)
            fig.tight_layout()
            fig.savefig(os.path.join(ddir, f"{cond}.png"), dpi=120)
            plt.close(fig)

        if args.dense:
            dense = {}
            for cond, folders in conds.items():
                if cond == "scratch":
                    continue
                dense[cond] = [d for d in (load_dense(f, S, cache_dir) for _, f in sorted(folders)) if d is not None]
            dcols = ["reward", "td_signed", "td_abs", "td_sigma", "critic_grad_norm", "actor_grad_norm", "dq_da_norm", "critic_update_ratio", "actor_update_ratio", "entropy"]
            fig, axes = plt.subplots(len(dcols), len(GROUPS), figsize=(6 * len(GROUPS), 2.3 * len(dcols)), sharex=True)
            dx = dense["fix"][0]["x"]
            for j, (title, members) in enumerate(GROUPS.items()):
                for i, k in enumerate(dcols):
                    ax = axes[i, j]
                    band(ax, dx, np.array([d[k] for d in dense["fix"]]), *REF_STYLE["fix"], "fix")
                    for cond in members:
                        if cond in dense and dense[cond]:
                            band(ax, dx, np.array([d[k] for d in dense[cond]]), *color_of(cond), cond)
                    ax.axvline(S, color="#8a8983", lw=0.8, ls="--")
                    ax.set_xlim(S - 20_000, S + 100_000)
                    if k in {"td_abs", "td_sigma", "critic_grad_norm", "actor_grad_norm", "dq_da_norm", "critic_update_ratio", "actor_update_ratio"}:
                        ax.set_yscale("log")
                    if j == 0:
                        ax.set_ylabel(k, fontsize=9)
                    if i == 0:
                        ax.set_title(title, fontsize=11)
                    style(ax)
                axes[0, j].legend(fontsize=7, frameon=False, ncol=2)
            fig.suptitle(f"{env} {shift}: per-step dynamics around the switch (1k-step bins, mean ± SE over seeds)", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.985))
            fig.savefig(os.path.join(args.out, f"dense_{env}_{shift}.png"), dpi=120)
            plt.close(fig)

    # tables
    with open(os.path.join(args.out, "recovery_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run[0].keys()))
        w.writeheader()
        w.writerows(per_run)
    with open(os.path.join(args.out, "recovery_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # the G-reset fix itself: legacy vs fix over the whole run, with the scaler statistics it changes
    pairs = sorted(k for k in heat["early_1M"])
    gk = ["return", "td_sigma", "scaler_G2_mean", "scaler_G_count"]
    fig, axes = plt.subplots(len(gk), len(pairs), figsize=(3.6 * len(pairs), 2.8 * len(gk)))
    for j, pr in enumerate(pairs):
        for i, k in enumerate(gk):
            ax = axes[i, j]
            for ref in ("legacy", "fix"):
                if ref not in gfix[pr]:
                    continue
                cs = gfix[pr][ref]
                x = next(iter(cs.values()))["x"]
                band(ax, x, np.array([c[k] for c in cs.values()]), *REF_STYLE[ref], ref, lw=1.4)
            if k != "return":
                ax.set_yscale("log")
            if i == 0:
                ax.set_title(f"{pr[0].split('-')[0]} {pr[1]}", fontsize=10)
            if j == 0:
                ax.set_ylabel(k)
            style(ax)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Effect of resetting G on truncation (fix) vs the original code (legacy), whole run, mean ± SE over 5 seeds", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "gfix_legacy_vs_fix.png"), dpi=130)
    plt.close(fig)

    # heatmaps: change vs fix across all env/shift pairs
    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#d9d8d3", "#2a78d6"])
    specs = {
        "early_1M": ("mean return over the first 1M post-switch steps: % change vs fix (blue = better)", lambda r, H: 100 * r["diff"] / abs(r["fix_mean"]), 50),
        "t80_fix": ("steps to 80% of the fix asymptote: % of horizon saved vs fix (blue = faster)", lambda r, H: -100 * r["diff"] / H, 30),
        "asym": ("post-shift asymptote (last 1M steps): % change vs fix (blue = better)", lambda r, H: 100 * r["diff"] / abs(r["fix_mean"]), 30),
    }
    conds_heat = [c for c in ORDER if c != "legacy"] + (["legacy"] if "legacy" in ORDER else []) + ["scratch"]
    for k, (title, fn, vmax) in specs.items():
        M = np.full((len(conds_heat), len(pairs)), np.nan)
        txt = [["" for _ in pairs] for _ in conds_heat]
        for j, pr in enumerate(pairs):
            for i, c in enumerate(conds_heat):
                r = heat[k][pr].get(c)
                if r and np.isfinite(r["diff"]):
                    M[i, j] = fn(r, heat["H"][pr])
                    txt[i][j] = f"{M[i, j]:+.0f}%\n{r['wins']}/{r['n_pairs']}{r['sig']}"
        fig, ax = plt.subplots(figsize=(2.1 * len(pairs) + 3, 0.5 * len(conds_heat) + 1.8))
        im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels([f"{e.split('-')[0]}\n{s}" for e, s in pairs], fontsize=9)
        ax.set_yticks(range(len(conds_heat)))
        ax.set_yticklabels(conds_heat, fontsize=9)
        for i in range(len(conds_heat)):
            for j in range(len(pairs)):
                ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=7.5, color="#0b0b0b")
        sizes = [len([c for c in m if c != "legacy"]) for m in GROUPS.values()]
        for g in list(np.cumsum(sizes)) + [len(conds_heat) - 1]:  # group boundaries and the references
            if 0 < g < len(conds_heat):
                ax.axhline(g - 0.5, color="white", lw=2)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="% vs fix")
        ax.set_title(title + "\ncell text: value, seeds where lever beat fix, paired t-test", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"heatmap_{k}.png"), dpi=140)
        plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
