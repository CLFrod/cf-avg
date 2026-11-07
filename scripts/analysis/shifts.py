"""
Runs with mid-run dynamics shifts: load them, split each into its segments, say exactly what changed at every
switch (physics mods and reward/termination kwargs, against the env's defaults), and measure per segment:

    peak / final    highest and last value of the smoothed return inside the segment
    pre             smoothed return just before the switch (the previous segment's final)
    drop            lowest smoothed return within --horizon steps after the switch, relative to pre
    recover         steps after the switch until the smoothed return is back at pre (never: -)
    adapt           steps after the switch until 90 % of the segment's own peak
    vs scratch      segment peak / peak of the from-scratch baseline with the same dynamics (--baseline results dir)

    python scripts/analysis/shifts.py results_trillium/results_24M --baseline results_trillium/results_24M --csv shifts.csv

The other shift scripts (plot_shift_*.py) import from here.
"""
import os, sys, csv, glob, json, argparse, inspect
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

EARTH = -9.81
SCALE_NAMES = {"mass_scale": "mass", "friction_scale": "friction", "gear_scale": "gear", "damping_scale": "damping",
               "armature_scale": "armature", "stiffness_scale": "stiffness"}
NAMED_FIELDS = {"mass": "body", "friction": "geom", "damping": "joint", "gear": "actuator"}
KWARG_SHORT = {"healthy_z_range": "z", "healthy_angle_range": "angle", "healthy_reward": "healthy",
               "ctrl_cost_weight": "ctrl", "forward_reward_weight": "fwd"}
_defaults_cache = {}


def env_defaults(env_id):
    """ Default reward/termination kwargs and the model's named physics values, read from the env itself """
    if env_id in _defaults_cache:
        return _defaults_cache[env_id]
    import gymnasium as gym
    u = gym.make(env_id).unwrapped
    m = u.model
    kw = {k: p.default for k, p in inspect.signature(type(u).__init__).parameters.items() if p.default is not inspect._empty}
    named = {
        "mass": {(m.body(i).name or str(i)): float(m.body_mass[i]) for i in range(1, m.nbody)},
        "friction": {m.geom(i).name: float(m.geom_friction[i, 0]) for i in range(m.ngeom)},
        "damping": {m.joint(j).name: float(m.dof_damping[m.jnt_dofadr[j]]) for j in range(m.njnt)},
        "gear": {(m.actuator(a).name or m.joint(m.actuator_trnid[a, 0]).name): float(m.actuator_gear[a, 0]) for a in range(m.nu)},
    }
    _defaults_cache[env_id] = (kw, named)
    return kw, named


def run_meta(run_dir):
    with open(os.path.join(run_dir, "multi_experiment.json")) as f:
        info = json.load(f)
    return info, info["hyperparams"]


def load_run(run_dir):
    info, hyp = run_meta(run_dir)
    metrics = glob.glob(os.path.join(run_dir, "*_metrics.csv"))
    if not metrics:
        return None
    with open(metrics[0], newline="") as f:
        rows = [r for r in csv.DictReader(f) if r and r["partial"] == "0"]
    if not rows:
        return None
    meta_path = os.path.join(run_dir, "checkpoint.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"complete": None}
    return dict(experiment=info["experiment"], env=hyp["env"], seed=info["seed"], folder=os.path.basename(run_dir),
                steps=np.array([float(r["step"]) for r in rows]), returns=np.array([float(r["return"]) for r in rows]),
                N=hyp["N"], complete=meta["complete"], env_mods=hyp.get("env_mods") or {},
                env_kwargs=hyp.get("env_kwargs") or {}, schedule=sorted(hyp.get("schedule") or [], key=lambda e: e["step"]))


def load_runs(results_dir, shifted=True, envs=None, complete_only=True):
    runs = []
    for d in sorted(glob.glob(os.path.join(results_dir, "*", "multi_experiment.json"))):
        _, hyp = run_meta(os.path.dirname(d))
        if bool(hyp.get("schedule")) != shifted or (envs and not any(hyp["env"].startswith(e) for e in envs)):
            continue
        run = load_run(os.path.dirname(d))
        if run is None or (complete_only and run["complete"] is False):
            continue
        runs.append(run)
    return runs


def normalise_state(env, mods, kwargs):
    """ Dynamics in effect, with everything at its default removed, so equal states compare equal """
    kw_defaults, named = env_defaults(env)
    state = {}
    for k, v in mods.items():
        if k == "gravity" and (v == EARTH or (isinstance(v, list) and v[2] == EARTH)):
            continue
        if k in SCALE_NAMES and v == 1.0:
            continue
        if k in NAMED_FIELDS:
            v = {n: x for n, x in v.items() if not np.isclose(x, named[k].get(n, np.nan))}
            if not v:
                continue
        state[k] = v
    for k, v in kwargs.items():
        d = kw_defaults.get(k)
        if d is not None and np.all(np.isclose(np.ravel(v), np.ravel(d))):
            continue
        state["kw:" + k] = v
    return state


def state_key(env, state):
    return env, json.dumps(state, sort_keys=True)


def fmt(v):
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def describe_named(env, key, values):
    """ 'mass: 13 bodies x0.51-1.79' or 'friction: floor 0.3, 4 geoms x0.36-0.51' """
    _, named = env_defaults(env)
    parts, factors = [], []
    for n, x in values.items():
        if key == "friction" and n == "floor":
            parts.append(f"floor {x:g}")
        elif named[key].get(n):
            factors.append(x / named[key][n])
    if factors:
        lo, hi = min(factors), max(factors)
        rng = f"x{lo:.2f}" if np.isclose(lo, hi, rtol=0.01) else f"x{lo:.2f}-{hi:.2f}"
        names = [n for n in values if named[key].get(n) and not (key == "friction" and n == "floor")]
        kind = NAMED_FIELDS[key]
        what = ", ".join(names) if len(names) <= 3 else f"{len(names)} {'bodies' if kind == 'body' else kind + 's'}"
        parts.append(f"{what} {rng}")
    return f"{key}: " + ", ".join(parts)


def describe_state(env, state):
    """ Short label of the dynamics in effect, e.g. 'g=1.62, z<=1.5, ctrl=0.25' or 'base' """
    out = []
    for k, v in state.items():
        if k == "gravity":
            out.append(f"g={abs(v[2] if isinstance(v, list) else v):g}")
        elif k in SCALE_NAMES:
            out.append(f"{SCALE_NAMES[k]} x{v:g}")
        elif k in NAMED_FIELDS:
            out.append(describe_named(env, k, v))
        elif k.startswith("kw:"):
            name = k[3:]
            short = KWARG_SHORT.get(name, name)
            if isinstance(v, (list, tuple)) and len(v) == 2:
                out.append(f"{short}<={v[1]:g}" if name == "healthy_z_range" else f"{short} {fmt(v)}")
            else:
                out.append(f"{short}={fmt(v)}")
        else:
            out.append(f"{k}={fmt(v)}")
    return ", ".join(out) if out else "base"


def describe_change(env, before, after):
    """ What a switch did: every item of `after` that differs from `before`, as 'key: old -> new' """
    out = []
    for k in sorted(set(before) | set(after)):
        b, a = before.get(k), after.get(k)
        if b == a:
            continue
        if k in NAMED_FIELDS:
            out.append(describe_named(env, k, a) if a else f"{k}: reset")
            continue
        label = SCALE_NAMES.get(k, k[3:] if k.startswith("kw:") else k)
        b_s = "default" if b is None else fmt(b)
        a_s = "default" if a is None else fmt(a)
        if k == "gravity":
            b_s, a_s = (fmt(abs(x[2] if isinstance(x, list) else x)) if x is not None else "9.81" for x in (b, a))
        out.append(f"{label} {b_s} -> {a_s}")
    return "; ".join(out)


def segments(run):
    """ [(start, end, state, label, change)] for the run: one per stretch of constant dynamics """
    env = run["env"]
    mods, kwargs = dict(run["env_mods"]), dict(run["env_kwargs"])
    state = normalise_state(env, mods, kwargs)
    bounds = [0] + [e["step"] for e in run["schedule"]] + [run["N"]]
    segs = [(0, bounds[1], state, describe_state(env, state), "")]
    for i, e in enumerate(run["schedule"]):
        for k in e.get("mods") or {}:
            if k in SCALE_NAMES:               # a whole-field scale overwrites the named values set earlier
                mods.pop(SCALE_NAMES[k], None)
        mods.update(e.get("mods") or {})
        kwargs.update(e.get("env_kwargs") or {})
        new = normalise_state(env, mods, kwargs)
        segs.append((e["step"], bounds[i + 2], new, describe_state(env, new), describe_change(env, state, new)))
        state = new
    return segs


def smoothed(run, window, x_tick):
    """ Same as incremental_rl.utils.binned_curve (trailing-window mean every x_tick steps), via cumulative sums """
    steps, rets = run["steps"], run["returns"]
    ends = np.arange(x_tick, steps[-1] + 1, x_tick)
    lo = np.searchsorted(steps, np.maximum(0, ends - window), side="right")
    hi = np.searchsorted(steps, ends, side="left")
    cs = np.concatenate([[0.0], np.cumsum(rets)])
    n = hi - lo
    keep = n > 0
    return (cs[hi[keep]] - cs[lo[keep]]) / n[keep], ends[keep]


def first_at_or_above(curve, x, start, end, level):
    """ Steps after `start` until the curve first reaches `level` inside (start, end], or None """
    hit = np.flatnonzero((x > start) & (x <= end) & (curve >= level))
    return int(x[hit[0]] - start) if len(hit) else None


def segment_stats(run, window, x_tick, horizon, baseline_peaks=None):
    """ One dict per segment of the run (see module docstring) """
    curve, x = smoothed(run, window, x_tick)
    out = []
    for i, (start, end, state, label, change) in enumerate(segments(run)):
        inside = (x > start) & (x <= end)
        if not inside.any():
            continue
        c, xs = curve[inside], x[inside]
        p = int(np.argmax(c))
        row = dict(experiment=run["experiment"], env=run["env"], seed=run["seed"], segment=i, start=start, end=end,
                   condition=label, change=change, peak=float(c[p]), peak_step=int(xs[p]), final=float(c[-1]),
                   pre=None, drop=None, recover=None, adapt=first_at_or_above(curve, x, start, end, 0.9 * c[p]),
                   scratch_peak=None)
        if i > 0:
            before = (x <= start) & (x > start - window)
            if before.any():
                pre = float(curve[before][-1])
                early = c[xs <= start + horizon]
                row.update(pre=pre, drop=float(early.min() - pre) if len(early) else None,
                           recover=first_at_or_above(curve, x, start, end, pre))
        if baseline_peaks:
            row["scratch_peak"] = baseline_peaks.get(state_key(run["env"], state))
        out.append(row)
    return out


def baseline_peak_lookup(results_dir, window, x_tick, envs=None):
    """ {(env, state_key): mean peak over seeds} of the unshifted runs in results_dir """
    peaks = {}
    for run in load_runs(results_dir, shifted=False, envs=envs):
        curve, _ = smoothed(run, window, x_tick)
        key = state_key(run["env"], normalise_state(run["env"], run["env_mods"], run["env_kwargs"]))
        peaks.setdefault(key, []).append(float(curve.max()))
    return {k: float(np.mean(v)) for k, v in peaks.items()}


ENV_COLORS = {"Ant-v4": "#2a78d6", "Hopper-v4": "#eb6834", "Humanoid-v4": "#1baf7a"}
GREY, LIGHT = "#8a8984", "#e6e5e1"


def panel_grid(plt, n, cols=4, w=4.6, h=3.0, extra_h=1.2):
    rows = max(1, -(-n // cols))
    fig, axes = plt.subplots(rows, cols, figsize=(w * cols, h * rows + extra_h), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def style(ax):
    ax.yaxis.grid(True, color=LIGHT, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(labelsize=8)


def caption(fig, text, width=170):
    """ Grey caption at the bottom; reserves room for it based on the figure height """
    import textwrap
    lines = textwrap.wrap(text, width)
    fig.text(0.5, 0.01, "\n".join(lines), ha="center", va="bottom", fontsize=7.5, color=GREY, linespacing=1.5)
    fig.tight_layout(rect=(0, (0.25 + 0.17 * len(lines)) / fig.get_figheight(), 1, 0.97))


def by_experiment(runs):
    """ {experiment: [runs]}, two-segment experiments first, then by name """
    groups = {}
    for r in runs:
        groups.setdefault(r["experiment"], []).append(r)
    return dict(sorted(groups.items(), key=lambda kv: (len(kv[1][0]["schedule"]), kv[0])))


def mean_curve(runs, window, x_tick):
    """ Across-seed mean and std of the smoothed curves on a common grid (shortest run) """
    curves = [smoothed(r, window, x_tick) for r in runs]
    n = min(len(c) for c, _ in curves)
    ys = np.array([c[:n] for c, _ in curves])
    return curves[0][1][:n], ys.mean(axis=0), ys.std(axis=0), ys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir")
    p.add_argument("--baseline", default=None, help="Results dir with from-scratch runs of the same conditions")
    p.add_argument("--env", nargs="*", default=None)
    p.add_argument("--window", type=int, default=100000)
    p.add_argument("--x_tick", type=int, default=10000)
    p.add_argument("--horizon", type=int, default=1000000, help="Steps after a switch in which the drop is measured")
    p.add_argument("--incomplete", action="store_true")
    p.add_argument("--csv", default=None)
    return p.parse_args()


def fmt_steps(s):
    return "    -  " if s is None else f"{s / 1e6:5.2f}M"


def main():
    args = parse_args()
    runs = load_runs(args.results_dir, envs=args.env, complete_only=not args.incomplete)
    if not runs:
        sys.exit(f"no shifted runs under {args.results_dir}")
    scratch = baseline_peak_lookup(args.baseline, args.window, args.x_tick, args.env) if args.baseline else None
    rows = [r for run in runs for r in segment_stats(run, args.window, args.x_tick, args.horizon, scratch)]

    groups = {}
    for r in rows:
        groups.setdefault((r["env"], r["experiment"]), {}).setdefault(r["segment"], []).append(r)
    for (env, exp), segs in sorted(groups.items()):
        n = len(next(iter(segs.values())))
        print(f"\n{env}  {exp}  (n = {n} seeds, mean +- std; window {args.window // 1000}k)")
        for i, g in sorted(segs.items()):
            r0 = g[0]
            print(f"  segment {i}: {r0['start'] / 1e6:g}M-{r0['end'] / 1e6:g}M  [{r0['condition']}]"
                  + (f"   change: {r0['change']}" if r0["change"] else ""))
            peak, final = np.array([r["peak"] for r in g]), np.array([r["final"] for r in g])
            line = f"      peak {peak.mean():7.0f} +- {peak.std():5.0f}   final {final.mean():7.0f} +- {final.std():5.0f}"
            if i > 0 and g[0]["pre"] is not None:
                pre, drop = np.array([r["pre"] for r in g]), np.array([r["drop"] for r in g])
                rec = [r["recover"] for r in g]
                adapt = [r["adapt"] for r in g]
                line += (f"   pre {pre.mean():7.0f}   drop {drop.mean():7.0f} ({drop.mean() / abs(pre.mean()):+.0%})"
                         f"   recover {np.mean([x for x in rec if x is not None]) / 1e6 if any(x is not None for x in rec) else float('nan'):5.2f}M"
                         f" ({sum(x is not None for x in rec)}/{n})   adapt90 {np.mean(adapt) / 1e6:5.2f}M")
            if g[0]["scratch_peak"]:
                line += f"   vs scratch {peak.mean() / g[0]['scratch_peak']:.0%} (scratch peak {g[0]['scratch_peak']:.0f})"
            print(line)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
