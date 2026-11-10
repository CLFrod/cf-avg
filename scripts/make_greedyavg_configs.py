"""Greedy AVG (Greedy AC from step 0) on the dynamics-shift suite, and Greedy AVG with the recovery methods.

    python scripts/make_greedyavg_configs.py [--stats_root '$SCRATCH/results'] [--targets configs/privileged_targets.json]

Writes, per env and per G-reset world (legacy: gshifts_<env>.json, glevers_<env>.json; fix: gshifts_fix_<env>.json,
glevers_fix_<env>.json), plus gshifts_smoke.json and glevers_smoke.json:
  gshifts: AVG base vs Greedy AVG on the five switches of configs/trillium_dynamics_3env_24m.json (moon, gear2x,
           slippery, heavy, there-and-back moon), 10 seeds: 10 conditions x 10 seeds = 100 runs.
  glevers: Greedy AVG alone (the pairing base, analyse with --base GAC_full), Greedy AVG + each recovery method fired
           at the shift (obs soft/hard reset, obs window, Adam reset, statistics stack, privileged obs stats, reward
           offset) and the AVG base, on the lever study's two shifts: 9 x 10 seeds x 2 shifts = 180 runs.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from make_lever_configs import SEEDS as SEEDS5, LR, RUN, SHIFTS, SHRINK_OBS, SHRINK_SCALER, RESET_ADAM
from make_privileged_configs import levers as priv_levers

SEEDS = SEEDS5 + [24601, 7331, 90210, 41275, 5812]
GAC_FULL = {"greedy_ac": [30, 0.1]}  # N=30, top 3, plain Sarsa critic
SUITE = {  # the original dynamics-shift suite's switches: name -> mods applied at the switch
    "moon": {"gravity": -1.62},
    "gear2x": {"gear_scale": 2.0},
    "slippery": {"friction_scale": 0.3},
    "heavy": {"mass_scale": 2.0},
}


def gshifts(env, legacy, smoke=False):
    N, switch = RUN[env]
    base = "legacy" if legacy else "fix"
    experiments = []
    switches = {"gear2x": SUITE["gear2x"]} if smoke else SUITE
    for shift, mods in switches.items():
        schedule = [{"step": switch, "mods": mods}]
        experiments.append({"name": f"{base}-{shift}", "schedule": schedule})
        experiments.append({"name": f"GAC_full-{shift}", "schedule": schedule, **GAC_FULL})
    if not smoke:  # there and back: base, moon, base in thirds
        schedule = [{"step": N // 3, "mods": SUITE["moon"]}, {"step": 2 * N // 3, "mods": {"gravity": -9.81}}]
        experiments.append({"name": f"{base}-moon_back", "schedule": schedule})
        experiments.append({"name": "GAC_full-moon_back", "schedule": schedule, **GAC_FULL})
    comment = [
        f"Greedy AVG vs AVG on the dynamics-shift suite, {env}, {'legacy G reset' if legacy else 'G-truncation fix'}: {N // 1_000_000}M steps, switch at {switch // 1_000_000}M",
        "(there-and-back: base, moon, base in thirds). Shifts: moon gravity, gear2x, slippery (friction 0.3x), heavy (mass 2x), moon and back.",
        f"Greedy AVG = Greedy AC (N=30, top 3, plain Sarsa critic) from step 0. 10 seeds. Analyse with lever_recovery.py --base {base}.",
    ]
    return _cfg(env, N, switch, experiments, legacy, comment, smoke, "gshifts")


def glevers(env, legacy, stats_root, targets, smoke=False):
    N, switch = RUN[env]
    base = "legacy" if legacy else "fix"
    experiments = []
    for shift, (mods, _) in SHIFTS[env].items():
        if smoke and shift != "gear2x":
            continue
        schedule = [{"step": switch, "mods": mods}]
        priv = priv_levers(env, shift, stats_root, targets[env][shift])
        methods = {
            "GACx_obs_soft": {"on_shift": [SHRINK_OBS]},
            "GACx_obs_hard": {"on_shift": [{"action": "shrink_obs_norm", "count": 0}]},
            "GACx_obs_window": {"obs_norm_cap": 100_000},
            "GACx_adam": {"on_shift": [RESET_ADAM]},
            "GACx_stats_stack": {"on_shift": [SHRINK_OBS, SHRINK_SCALER, RESET_ADAM]},
            "GACx_priv_obs": priv["P4_priv_obs"],
            "GACx_reward_offset": priv["P3_reward_offset"],
        }
        experiments.append({"name": f"{base}-{shift}", "schedule": schedule})
        experiments.append({"name": f"GAC_full-{shift}", "schedule": schedule, **GAC_FULL})
        for name, extra in methods.items():
            experiments.append({"name": f"{name}-{shift}", "schedule": schedule, **GAC_FULL, **extra})
    comment = [
        f"Greedy AVG with the recovery methods, {env}, {'legacy G reset' if legacy else 'G-truncation fix'}: {N // 1_000_000}M steps, switch at {switch // 1_000_000}M.",
        "Per shift: AVG base, Greedy AVG alone (pairing base), Greedy AVG + obs soft reset / obs hard reset / obs window (always on) / Adam reset /",
        "statistics stack / privileged obs stats / reward offset = 9 conditions x 10 seeds. Greedy AVG = Greedy AC (N=30, top 3, plain Sarsa critic) from step 0.",
        "Analyse with lever_recovery.py --base GAC_full (methods vs Greedy AVG) and --base " + base + " (everything vs AVG).",
    ]
    return _cfg(env, N, switch, experiments, legacy, comment, smoke, "glevers")


def _cfg(env, N, switch, experiments, legacy, comment, smoke, prefix):
    cfg = {
        "_comment": comment,
        "defaults": {
            "N": N, "seeds": SEEDS, "gamma": 0.99, "beta1": 0.0, "normalize_obs": True, "pnorm": True, "scaled_td": True,
            "legacy_g_reset": legacy, "n_eval": 2, "video_every": switch, "save_model": True, "save_model_every": 1_000_000,
            "dense_window": [20_000, 200_000],
        },
        "envs": {env: LR[env]},
        "experiments": experiments,
    }
    if smoke:
        for e in experiments:
            for entry in e["schedule"]:
                entry["step"] = 1500
            for a in e.get("on_shift", []):
                if "steps" in a:
                    a["steps"] = 800
        cfg["_comment"] = [f"Smoke test of configs/{prefix}_<env>.json: every condition once on Hopper gear2x, 3000 steps, switch at 1500.",
                           f"Login node, ~1 min: CONFIG=configs/{prefix}_smoke.json bash scripts/trillium/smoke_levers.sh"]
        cfg["defaults"].update(N=3000, seeds=SEEDS[:1], n_eval=1, video_every=1500, dense_window=[500, 1000], checkpoint=1000)
    return cfg


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stats_root", default="$SCRATCH/results", help="Folder holding the batch-1 run folders (scratch-<shift>-<env>-<seed>-*)")
    p.add_argument("--targets", default=os.path.join(os.path.dirname(__file__), "..", "configs", "privileged_targets.json"))
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "configs"))
    args = p.parse_args()
    targets = json.load(open(args.targets))
    out = []
    for legacy, tag in ((True, ""), (False, "_fix")):
        for env in RUN:
            e = env.split("-")[0].lower()
            out.append((f"gshifts{tag}_{e}", gshifts(env, legacy)))
            out.append((f"glevers{tag}_{e}", glevers(env, legacy, args.stats_root, targets)))
    out.append(("gshifts_smoke", gshifts("Hopper-v4", True, smoke=True)))
    out.append(("glevers_smoke", glevers("Hopper-v4", True, args.stats_root, targets, smoke=True)))
    for name, cfg in out:
        path = os.path.join(args.out, f"{name}.json")
        with open(path, "w") as f:
            json.dump(cfg, f, indent=1)
        print(f"{path}: {len(cfg['experiments'])} experiments x {len(cfg['defaults']['seeds'])} seeds")
