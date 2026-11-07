#!/usr/bin/env python3
"""
Figures and a summary table from one or more node-monitor folders (node_monitor.py + collect_runs.py output).

    python scripts/profiling/plot_profile.py results_trillium/profiling/dynamics_18m/monitor_2722892 \
        results_trillium/profiling/limb_18m/monitor_2722894 --out results_trillium/profiling/figures

Per monitor folder (prefixed with its label, by default the parent folder's name):
  <label>_core_average.png   the headline: average busy % per core over the job, all cores and the cores running a job
  <label>_utilisation.png    cores in use and live runs, node CPU split, memory -- over the job
  <label>_cpu_heatmap.png    busy % of every CPU over time, rows grouped by socket / CCD: which cores idle, and when
  <label>_runs.png           per-run process behaviour over the job: CPU %, resident memory, involuntary switches
  <label>_throughput.png     steps/s of each env over run wall time, and time-to-complete per run
  <label>_load_vs_speed.png  steps/s of each env against how many runs were alive on the node: the cost of sharing
And over all folders given:
  summary.csv                the headline efficiency numbers per job (also printed)
  compare_cores.png          cores in use over time, one line per job (only with >= 2 folders)
"""
import os, sys, csv, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ------------------------------------------------------------------------------------------ palette (light mode)
RESERVED_CORES = 4   # cores avg_multi.sh keeps free (RESERVE_CORES): launcher, OS, utilisation monitor

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ENV_SLOT = {"Hopper-v4": 0, "Ant-v4": 1, "Humanoid-v4": 2}      # fixed: an env keeps its hue in every figure
MB_PER_GB = 1024.0

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.solid_joinstyle": "round", "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK2,
    "savefig.dpi": 160, "figure.constrained_layout.use": True,
})


# ------------------------------------------------------------------------------------------ loading
def read_csv(path):
    """-> (header, list of rows as str lists)"""
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        hdr = next(r)
        return hdr, [row for row in r if row]


def col(hdr, rows, name):
    i = hdr.index(name)
    out = np.full(len(rows), np.nan)
    for k, row in enumerate(rows):
        v = row[i] if i < len(row) else ""
        if v != "":
            out[k] = float(v)
    return out


class Job:
    def __init__(self, mdir, label):
        self.dir, self.label = mdir, label
        with open(os.path.join(mdir, "node_info.json"), encoding="utf-8") as f:
            self.info = json.load(f)
        self.t0 = self.info["started_unix"]
        hdr, rows = read_csv(os.path.join(mdir, "node.csv"))
        self.node_hdr, self.node_rows = hdr, rows
        self.hours = col(hdr, rows, "elapsed_s") / 3600
        self.ncpu = self.info["logical_cpus"]
        self.sockets = self.info["sockets"]

    def node(self, name):
        return col(self.node_hdr, self.node_rows, name)

    def cpu_matrix(self, bin_minutes=1):
        """per-CPU busy %, time-binned -> (bin hours, matrix cpu x bins, topology rows); rows ordered by
        socket / L3 group / cpu"""
        hdr, rows = read_csv(os.path.join(self.dir, "cpus.csv"))
        el = np.array([float(r[2]) for r in rows]) / 3600
        m = np.array([[float(x) for x in r[3:]] for r in rows])      # time x cpu
        thdr, trows = read_csv(os.path.join(self.dir, "cpu_topology.csv"))
        topo = sorted(((int(r[thdr.index("socket")]), int(r[thdr.index("l3_group")]), int(r[thdr.index("cpu")]))
                       for r in trows))
        order = [c for _, _, c in topo]
        w = bin_minutes / 60
        nb = int(np.ceil(el[-1] / w))
        idx = np.minimum((el / w).astype(int), nb - 1)
        binned = np.full((nb, m.shape[1]), np.nan)
        for b in range(nb):
            sel = idx == b
            if sel.any():
                binned[b] = m[sel].mean(0)
        return (np.arange(nb) + 0.5) * w, binned[:, order].T, topo

    def proc_percentiles(self, name, q=(1, 50, 99)):
        """per-run process samples -> (tick hours, percentiles of `name` across the live runs at each tick)"""
        hdr, rows = read_csv(os.path.join(self.dir, "procs.csv"))
        ir, it, iv, ifs = hdr.index("role"), hdr.index("elapsed_s"), hdr.index(name), hdr.index("first_sample")
        by_t = {}
        for r in rows:
            if r[ir] == "run" and r[ifs] == "0" and r[iv] != "":
                by_t.setdefault(float(r[it]), []).append(float(r[iv]))
        ts = sorted(by_t)
        P = np.array([np.percentile(by_t[t], q) for t in ts])
        return np.array(ts) / 3600, P

    def env_finish_hours(self):
        """{env: (first run finished, last run finished)} in hours since job start, from the last process sample
        at which each run was alive (resolution: one --proc_interval)"""
        hdr, rows = read_csv(os.path.join(self.dir, "procs.csv"))
        ir, it, ie, irun = hdr.index("role"), hdr.index("elapsed_s"), hdr.index("env"), hdr.index("run")
        last = {}
        for r in rows:
            if r[ir] == "run":
                last[(r[ie], r[irun])] = float(r[it])
        out = {}
        for (env, _), t in last.items():
            lo, hi = out.get(env, (np.inf, 0))
            out[env] = (min(lo, t), max(hi, t))
        return {env: (lo / 3600, hi / 3600) for env, (lo, hi) in out.items()}

    def run_summary(self):
        hdr, rows = read_csv(os.path.join(self.dir, "run_summary.csv"))
        g = lambda name: [r[hdr.index(name)] for r in rows]
        return {"env": g("env"), "wall_h": np.array(g("wall_time_s"), float) / 3600,
                "sps": np.array(g("mean_steps_per_sec"), float), "complete": np.array(g("complete"), int),
                "startup": np.array(g("startup_s"), float)}

    def throughput_by_env(self):
        """-> {env: (bin hours, median, q25, q75, n runs)} from runs_binned.csv"""
        hdr, rows = read_csv(os.path.join(self.dir, "runs_binned.csv"))
        ie, ib, isps = hdr.index("env"), hdr.index("bin_end_wall_s"), hdr.index("steps_per_sec")
        acc = {}
        for r in rows:
            if r[isps] != "":
                acc.setdefault(r[ie], {}).setdefault(float(r[ib]), []).append(float(r[isps]))
        out = {}
        for env, bins in acc.items():
            bs = sorted(bins)
            vals = [bins[b] for b in bs]
            out[env] = (np.array(bs) / 3600, np.array([np.median(v) for v in vals]),
                        np.array([np.percentile(v, 25) for v in vals]), np.array([np.percentile(v, 75) for v in vals]),
                        np.array([len(v) for v in vals]))
        return out


# ------------------------------------------------------------------------------------------ helpers
def hairline(ax, y, text, color=AXIS, below=False, left=False):
    ax.axhline(y, color=color, linewidth=1, zorder=1)
    ax.annotate(text, (0 if left else 1, y), xycoords=("axes fraction", "data"),
                xytext=(2 if left else -2, -3 if below else 3), textcoords="offset points",
                ha="left" if left else "right", va="top" if below else "bottom", fontsize=8, color=MUTED)


def end_label(ax, x, y, text, color=INK2, dy=0):
    ax.annotate(text, (x[-1], y[-1]), xytext=(4, dy), textcoords="offset points", ha="left", va="center",
                fontsize=8, color=color)


def mark_env_finishes(ax, job, y_frac=0.97):
    """A vertical line where the last run of each env finished, labelled with the env and the hour"""
    fin = job.env_finish_hours()
    for k, env in enumerate(sorted(fin, key=lambda e: fin[e][1])):
        lo, hi = fin[env]
        c = SERIES[ENV_SLOT.get(env, 3)]
        ax.axvline(hi, color=c, linewidth=1, zorder=1)
        ax.annotate(f"{env}, {hi:.1f} h", (hi, y_frac), xycoords=("data", "axes fraction"),
                    xytext=(-3, 0), textcoords="offset points", rotation=90, ha="right", va="top", fontsize=8,
                    color=INK2)


def style_time_axis(ax, hours):
    ax.set_xlim(0, hours[-1])
    ax.set_xlabel("hours since job start")
    ax.tick_params(length=0)


# ------------------------------------------------------------------------------------------ figures
def fig_utilisation(job, out):
    h = job.hours
    cores, runs = job.node("cores_in_use"), job.node("n_runs")
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fig.suptitle(f"{job.label}: node utilisation over the job  ({job.info['hostname']}, "
                 f"{job.ncpu} cores, {job.info['mem_total_mb'] / MB_PER_GB:.0f} GB)", fontsize=11, ha="left", x=0.01,
                 color=INK)

    ax = axs[0]
    ax.plot(h, cores, color=SERIES[0], label="cores in use (busy-time sum over all CPUs)")
    m = ~np.isnan(runs)
    ax.plot(h[m], runs[m], color=SERIES[1], label="live run processes")
    hairline(ax, job.ncpu, f"{job.ncpu} cores on the node", left=True)
    mark_env_finishes(ax, job, y_frac=0.99)
    ax.set_ylim(0, job.ncpu * 1.08)
    ax.set_ylabel("cores")
    ax.set_title("Cores in use and runs alive")
    ax.legend(loc="lower left")

    ax = axs[1]
    for k, (name, lab) in enumerate([("cpu_busy_pct", "busy"), ("cpu_system_pct", "system"),
                                     ("cpu_iowait_pct", "I/O wait")]):
        ax.plot(h, job.node(name), color=SERIES[k], label=lab)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of node CPU time")
    ax.set_title("Node CPU time split (user = busy minus system)")
    ax.legend(loc="center left")

    ax = axs[2]
    total = job.info["mem_total_mb"] / MB_PER_GB
    used = job.node("mem_used_mb")
    ax.plot(h, used / MB_PER_GB, color=SERIES[0], label="node memory used")
    cg = job.node("cg_mem_mb")
    if not np.all(np.isnan(cg)):
        ax.plot(h, cg / MB_PER_GB, color=SERIES[1], label="job cgroup (Slurm accounting)")
    rss = job.node("runs_rss_mb")
    m = ~np.isnan(rss)
    ax.plot(h[m], rss[m] / MB_PER_GB, color=SERIES[2], label="sum of run RSS")
    hairline(ax, total, f"{total:.0f} GB on the node")
    k = int(np.nanargmax(used))
    ax.annotate(f"peak {used[k] / MB_PER_GB:.0f} GB = {100 * used[k] / job.info['mem_total_mb']:.0f} % of the node",
                (h[k], used[k] / MB_PER_GB), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=8,
                color=INK2)
    ax.set_ylim(0, total * 1.08)
    ax.set_ylabel("GB")
    ax.set_title("Memory")
    ax.legend(loc="center left")
    style_time_axis(ax, h)
    fig.savefig(os.path.join(out, f"{job.label}_utilisation.png"))
    plt.close(fig)


def fig_cpu_heatmap(job, out):
    hb, M, topo = job.cpu_matrix(bin_minutes=1)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.grid(False)
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=100, interpolation="nearest",
                   extent=(0, hb[-1] + hb[0], M.shape[0], 0))
    sockets = [s for s, _, _ in topo]
    l3s = [l for _, l, _ in topo]
    for i in range(1, len(topo)):
        if sockets[i] != sockets[i - 1]:
            ax.axhline(i, color=INK, linewidth=1)
        elif l3s[i] != l3s[i - 1]:
            ax.axhline(i, color=SURFACE, linewidth=0.6)
    ticks = [i for i in range(len(topo)) if i % 16 == 0]
    ax.set_yticks([t + 0.5 for t in ticks])
    ax.set_yticklabels([f"cpu{topo[t][2]}" for t in ticks])
    ax.tick_params(length=0)
    for s in sorted(set(sockets)):
        rows = [i for i, x in enumerate(sockets) if x == s]
        ax.annotate(f"socket {s}", (1, (rows[0] + rows[-1]) / 2 + 0.5), xycoords=("axes fraction", "data"),
                    xytext=(8, 0), textcoords="offset points", rotation=270, ha="center", va="center",
                    fontsize=8.5, color=INK2)
    # name the idle rows (mean busy < 50 % over the second quarter of the job)
    mid = (hb > hb[-1] * 0.25) & (hb < hb[-1] * 0.5)
    idle = [i for i in range(M.shape[0]) if np.nanmean(M[i, mid]) < 50]
    if idle:
        groups, cur = [], [idle[0]]
        for i in idle[1:]:
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                groups.append(cur)
                cur = [i]
        groups.append(cur)
        for g in groups:
            cpus = [topo[i][2] for i in g]
            ax.annotate(f"idle: cpu{cpus[0]}-{cpus[-1]} ({len(g)})", (hb[-1] * 0.375, (g[0] + g[-1]) / 2 + 0.5),
                        xytext=(0, 0), textcoords="offset points", ha="center", va="center", fontsize=8, color=INK2,
                        bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.9))
    ax.set_xlabel("hours since job start")
    ax.set_ylabel("logical CPU, grouped by socket and CCD (8-core L3 groups)")
    ax.set_title(f"{job.label}: busy % of every CPU, 1-minute means")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cb.set_label("busy %", color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelcolor=INK2)
    fig.savefig(os.path.join(out, f"{job.label}_cpu_heatmap.png"))
    plt.close(fig)


def fig_core_average(job, out):
    """One line: average busy % over all cores through the job; a second over just the cores that ran a job"""
    hb, M, topo = job.cpu_matrix(bin_minutes=1)
    mid = (hb > hb[-1] * 0.25) & (hb < hb[-1] * 0.5)
    working = np.array([np.nanmean(M[i, mid]) >= 50 for i in range(M.shape[0])])
    n_all, n_work = M.shape[0], int(working.sum())
    all_mean, work_mean = np.nanmean(M, axis=0), np.nanmean(M[working], axis=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hb, work_mean, color=SERIES[1], label=f"the {n_work} cores that ran a job")
    ax.plot(hb, all_mean, color=SERIES[0], label=f"all {n_all} cores on the node")
    ceiling = 100 * n_work / n_all
    hairline(ax, ceiling, "", below=True)
    idle = n_all - n_work   # 4 are reserved for the launcher/OS/monitor (avg_multi.sh RESERVE_CORES); any more had no run
    cores = f"{n_work} used, {min(idle, RESERVED_CORES)} reserved" + (f", {idle - RESERVED_CORES} unused" if idle > RESERVED_CORES else "") + " cores"
    ax.annotate(f"{np.nanmedian(all_mean[mid]):.1f} %  (ceiling {ceiling:.1f} %: {cores})",
                (hb[mid][len(hb[mid]) // 2], np.nanmedian(all_mean[mid])),
                xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.annotate(f"{np.nanmedian(work_mean[mid]):.1f} %", (hb[mid][len(hb[mid]) // 2], np.nanmedian(work_mean[mid])),
                xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    mark_env_finishes(ax, job, y_frac=0.99)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Average Core Utilization (1-minute means)")
    ax.set_title(f"{job.label}: Job Average Core Utilization")
    ax.legend(loc="lower left")
    style_time_axis(ax, hb)
    ax.set_xlabel("Hours Since Job Start")
    fig.savefig(os.path.join(out, f"{job.label}_core_average.png"))
    plt.close(fig)


def fig_runs(job, out):
    panels = [("cpu_pct", "CPU % per run process (100 = one full core)", (0, 110), (1, 50, 99), "1st-99th percentile"),
              ("rss_mb", "Resident memory per run process (MB)", None, (0, 50, 100), "min-max"),
              ("invol_ctxt_per_s", "Involuntary context switches per second per run", None, (50, 50, 99), "median-99th")]
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fig.suptitle(f"{job.label}: the run processes themselves (30 s samples, all live runs)", fontsize=11,
                 ha="left", x=0.01, color=INK)
    for ax, (name, title, ylim, q, band) in zip(axs, panels):
        t, P = job.proc_percentiles(name, q)
        ax.fill_between(t, P[:, 0], P[:, 2], color=SEQ[1], linewidth=0, label=f"{band} across runs")
        ax.plot(t, P[:, 1], color=SERIES[0], label="median run")
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, None)
        ax.legend(loc="upper right" if name == "invol_ctxt_per_s" else "lower right")
    style_time_axis(axs[-1], job.hours)
    fig.savefig(os.path.join(out, f"{job.label}_runs.png"))
    plt.close(fig)


def fig_throughput(job, out):
    tp = job.throughput_by_env()
    rs = job.run_summary()
    envs = sorted(tp, key=lambda e: ENV_SLOT.get(e, 9))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2.4, 1]})
    fig.suptitle(f"{job.label}: per-run training speed", fontsize=11, ha="left", x=0.01, color=INK)
    for env in envs:
        hb, med, q25, q75, n = tp[env]
        c = SERIES[ENV_SLOT.get(env, 3)]
        keep = n >= max(3, n.max() // 2)        # drop the ragged end where only a few runs are still going
        ax.fill_between(hb[keep], q25[keep], q75[keep], color=c, alpha=0.18, linewidth=0)
        ax.plot(hb[keep], med[keep], color=c, label=env)
        end_label(ax, hb[keep], med[keep], f"{med[keep][-1]:.0f}", color=INK2)
    ax.set_ylim(0, None)
    ax.set_xlim(0, None)
    ax.set_xlabel("hours since the run started")
    ax.set_ylabel("steps / s  (median, interquartile band)")
    ax.set_title("Steps per second over a run, per environment")
    ax.legend(loc="lower left")
    ax.tick_params(length=0)

    rng = np.random.default_rng(0)
    for k, env in enumerate(envs):
        sel = np.array([e == env for e in rs["env"]])
        x = k + rng.uniform(-0.18, 0.18, sel.sum())
        ax2.scatter(x, rs["wall_h"][sel], s=28, color=SERIES[ENV_SLOT.get(env, 3)], edgecolors=SURFACE,
                    linewidths=1, zorder=3)
        ax2.annotate(f"{np.median(rs['wall_h'][sel]):.1f} h", (k, rs["wall_h"][sel].max()), xytext=(0, 6),
                     textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax2.set_xticks(range(len(envs)))
    ax2.set_xticklabels(envs)
    ax2.set_ylim(0, rs["wall_h"].max() * 1.15)
    ax2.set_ylabel("hours to finish")
    ax2.set_title("Time to complete, each run")
    ax2.tick_params(length=0)
    fig.savefig(os.path.join(out, f"{job.label}_throughput.png"))
    plt.close(fig)


def fig_load_vs_speed(job, out, step=20):
    """Median steps/s per env against how many runs were alive on the node at the time (runs_binned.csv joined
    to node.csv): the per-run cost of sharing the node"""
    hdr, rows = read_csv(os.path.join(job.dir, "runs_binned.csv"))
    ie, it, isps = hdr.index("env"), hdr.index("t_unix"), hdr.index("steps_per_sec")
    nr = job.node("n_runs")
    m = ~np.isnan(nr)
    live_at = lambda t: np.interp(t, job.hours[m] * 3600, nr[m])
    acc = {}
    for r in rows:
        if r[isps] == "" or r[it] == "":
            continue
        live = live_at(float(r[it]) - job.t0)
        b = int(live // step) * step
        acc.setdefault(r[ie], {}).setdefault(b, []).append(float(r[isps]))
    envs = sorted(acc, key=lambda e: ENV_SLOT.get(e, 9))
    fig, ax = plt.subplots(figsize=(7, 4))
    for env in envs:
        bins = sorted(b for b, v in acc[env].items() if len(v) >= 10)
        x = np.array(bins) + step / 2
        med = np.array([np.median(acc[env][b]) for b in bins])
        q25 = np.array([np.percentile(acc[env][b], 25) for b in bins])
        q75 = np.array([np.percentile(acc[env][b], 75) for b in bins])
        c = SERIES[ENV_SLOT.get(env, 3)]
        ax.fill_between(x, q25, q75, color=c, alpha=0.18, linewidth=0)
        ax.plot(x, med, color=c, marker="o", markersize=5, markeredgecolor=SURFACE, label=env)
        if len(med) > 1:
            ax.annotate(f"{med[0]:.0f} to {med[-1]:.0f}", (x[-1], med[-1]), xytext=(6, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, int(np.nanmax(nr)) + step)
    ax.set_ylim(0, None)
    ax.set_xlabel(f"runs alive on the node (bins of {step})")
    ax.set_ylabel("steps / s per run  (median, interquartile band)")
    ax.set_title(f"{job.label}: per-run speed against node load")
    ax.legend(loc="lower left")
    ax.tick_params(length=0)
    fig.savefig(os.path.join(out, f"{job.label}_load_vs_speed.png"))
    plt.close(fig)


def fig_compare(jobs, out):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for k, job in enumerate(jobs):
        ax.plot(job.hours, job.node("cores_in_use"), color=SERIES[k], label=job.label)
    ncpu = max(j.ncpu for j in jobs)
    hairline(ax, ncpu, f"{ncpu} cores on the node")
    ax.set_ylim(0, ncpu * 1.08)
    ax.set_xlim(0, max(j.hours[-1] for j in jobs))
    ax.set_xlabel("hours since job start")
    ax.set_ylabel("cores in use")
    ax.set_title("Cores in use, each job")
    ax.legend(loc="lower left")
    ax.tick_params(length=0)
    fig.savefig(os.path.join(out, "compare_cores.png"))
    plt.close(fig)


# ------------------------------------------------------------------------------------------ summary
def summarise(job):
    h, cores = job.hours, job.node("cores_in_use")
    runs = job.node("n_runs")
    n_launched = int(np.nanmax(runs))
    dur = h[-1]
    core_hours = np.trapezoid(cores, h) if hasattr(np, "trapezoid") else np.trapz(cores, h)
    steady = (h > 0.25) & (cores >= 0.95 * n_launched)
    first_finish = h[np.where(runs < n_launched)[0][0]] if np.any(runs < n_launched) else dur
    rs = job.run_summary()
    _, P = job.proc_percentiles("cpu_pct", (1, 50))
    _, R = job.proc_percentiles("rss_mb", (50, 100))
    mem_peak = np.nanmax(job.node("mem_used_mb"))
    cg_peak = np.nanmax(job.node("cg_mem_peak_mb"))
    med = lambda name: round(float(np.median(job.node(name)[steady])), 2) if steady.any() else ""
    row = {
        "job": job.label, "node": job.info["hostname"], "cpu": job.info["cpu_model"], "cores": job.ncpu,
        "sockets": job.sockets, "duration_h": round(dur, 2), "runs": n_launched,
        "runs_complete": int(rs["complete"].sum()),
        "steady_state_cores_in_use": round(float(np.median(cores[steady])), 1) if steady.any() else "",
        "steady_state_node_busy_pct": med("cpu_busy_pct"),
        "idle_cores_by_design": job.ncpu - n_launched,
        "core_hours_used": round(float(core_hours), 1), "core_hours_allocated": round(job.ncpu * dur, 1),
        "core_hours_used_pct": round(100 * core_hours / (job.ncpu * dur), 1),
        "core_hours_used_pct_of_launched": round(100 * core_hours / (n_launched * dur), 1),
        "first_run_finished_h": round(float(first_finish), 2), "tail_h": round(float(dur - first_finish), 2),
        "run_cpu_pct_median": round(float(np.median(P[:, 1])), 1), "run_cpu_pct_p1": round(float(np.min(P[:, 0])), 1),
        "run_rss_mb_median": round(float(np.median(R[:, 0])), 0), "run_rss_mb_max": round(float(np.max(R[:, 1])), 0),
        "node_mem_peak_gb": round(mem_peak / MB_PER_GB, 1),
        "node_mem_peak_pct": round(100 * mem_peak / job.info["mem_total_mb"], 1),
        "cgroup_mem_peak_gb": round(cg_peak / MB_PER_GB, 1) if not np.isnan(cg_peak) else "",
        "swap_used_mb_max": int(np.nanmax(job.node("swap_used_mb"))),
        "system_pct_median": med("cpu_system_pct"),
        "iowait_pct_median": med("cpu_iowait_pct"),
        "startup_s_median": round(float(np.median(rs["startup"])), 1),
    }
    for env in sorted(set(rs["env"]), key=lambda e: ENV_SLOT.get(e, 9)):
        sel = np.array([e == env for e in rs["env"]])
        row[f"{env}_runs"] = int(sel.sum())
        row[f"{env}_steps_per_sec_mean"] = round(float(rs["sps"][sel].mean()), 1)
        row[f"{env}_hours_min"] = round(float(rs["wall_h"][sel].min()), 2)
        row[f"{env}_hours_max"] = round(float(rs["wall_h"][sel].max()), 2)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("monitor_dirs", nargs="+")
    p.add_argument("--labels", nargs="*", default=None, help="one per folder (default: parent folder name)")
    p.add_argument("--out", default=None, help="output directory (default: <first monitor dir>/figures)")
    args = p.parse_args()
    labels = args.labels or [os.path.basename(os.path.dirname(os.path.abspath(d))) for d in args.monitor_dirs]
    out = args.out or os.path.join(args.monitor_dirs[0], "figures")
    os.makedirs(out, exist_ok=True)

    jobs, rows = [], []
    for d, label in zip(args.monitor_dirs, labels):
        print(f"{label}: {d}")
        job = Job(d, label)
        jobs.append(job)
        for fn in (fig_core_average, fig_utilisation, fig_cpu_heatmap, fig_runs, fig_throughput, fig_load_vs_speed):
            try:
                fn(job, out)
                print(f"  {fn.__name__[4:]}.png")
            except (FileNotFoundError, ValueError) as e:
                print(f"  {fn.__name__[4:]}: skipped ({e})")
        rows.append(summarise(job))
    if len(jobs) > 1:
        fig_compare(jobs, out)
        print("  compare_cores.png")

    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    width = max(len(str(r.get(k, ""))) for r in rows for k in keys) + 2
    print(f"\n{'metric':36s}" + "".join(f"{r['job']:>{width}s}" for r in rows))
    for k in keys:
        if k != "job":
            print(f"{k:36s}" + "".join(f"{str(r.get(k, '')):>{width}s}" for r in rows))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
