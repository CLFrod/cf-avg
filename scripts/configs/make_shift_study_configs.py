"""Shift study: writes configs/shift_study_2026-11-26/study_<env>_core.json and study_<env>_body.json for every environment
with tuned hyperparameters. Each config fills one Trillium node.

    python scripts/configs/make_shift_study_configs.py [--env Hopper-v4 ...] [--n_seeds 5]
    bash scripts/trillium/submit_study.sh                                   # one job per config

Per environment, standard AVG with the legacy G reset on every run:
    legacy-none            baseline: no shift, the whole run
    legacy-<shift>         base -> shifted at the halfway step
    scratch-<shift>        the shifted dynamics from step 0 to the end
    <lever>-<shift>        base -> shifted with one recovery lever fired at the switch
<env>_core holds the baseline and the CORE shifts (the same on every environment); <env>_body holds the shifts
of that body (named bodies and joints, springs, rotor inertia). The levers are the self-contained ones with the
best regret over the six combos of lever batch 1.
--smoke writes study_smoke_<env>_<part>.json instead: every condition once, 3000 steps, switch at 1500.
"""
import argparse, json, os

SEEDS = [18606, 16858, 63790, 15097, 62693]

# env -> hyperparameters, steps per run (the switch is at half) and the shifts only this body has
ENVS = {
    "Hopper-v4": {
        "hyp": {"actor_lr": 1.1e-5, "critic_lr": 7.7e-5, "alpha_lr": 0.3, "gamma": 0.99, "beta1": 0.0},
        "N": 6_000_000,
        "shifts": {
            "payload": {"mass": {"torso": 7.33}},
            "limb_heavy": {"mass": {"leg": 8.34}},
            "weak_joint": {"gear": {"foot_joint": 50}},
            "dead_joint": {"gear": {"foot_joint": 0}},
            "flipped_joint": {"gear": {"foot_joint": -200}},
        },
    },
    "Ant-v4": {
        "hyp": {"actor_lr": 0.0063, "critic_lr": 0.0087, "alpha_lr": 0.07, "gamma": 0.99, "beta1": 0.0},
        "N": 12_000_000,
        "shifts": {
            "payload": {"mass": {"torso": 0.654}},
            "limb_heavy": {"mass": {"front_left_leg": 0.1175, "aux_1": 0.1175, "4": 0.2028}},
            "weak_joint": {"gear": {"hip_1": 37.5}},
            "dead_joint": {"gear": {"hip_1": 0}},
            "flipped_joint": {"gear": {"hip_1": -150}},
        },
    },
    "Humanoid-v4": {
        "hyp": {"actor_lr": 0.0063, "critic_lr": 0.0087, "alpha_lr": 0.07, "gamma": 0.99, "beta1": 0.0},
        "N": 12_000_000,
        "core": {"stiff": {"damping_scale": 2.0}},  # x4 and up is numerically unstable on this body
        "shifts": {
            "payload": {"mass": {"torso": 17.82}},
            "limb_heavy": {"mass": {"right_thigh": 14.26, "right_shin": 8.267, "right_foot": 5.301}},
            "weak_joint": {"gear": {"right_knee": 50}},
            "dead_joint": {"gear": {"right_knee": 0}},
            "flipped_joint": {"gear": {"right_knee": -200}},
            "stiff_springs": {"stiffness_scale": 3.0},
        },
    },
    "HalfCheetah-v4": {  # sweep draw h045
        "hyp": {"actor_lr": 4e-6, "critic_lr": 0.0033, "alpha_lr": 0.0005, "gamma": 0.97, "beta1": 0.9},
        "N": 10_000_000,
        "shifts": {
            "payload": {"mass": {"torso": 12.50}},
            "limb_heavy": {"mass": {"bthigh": 4.63, "bshin": 4.76, "bfoot": 3.29}},
            "weak_joint": {"gear": {"bthigh": 30}},
            "dead_joint": {"gear": {"bthigh": 0}},
            "flipped_joint": {"gear": {"bthigh": -120}},
            "stiff_springs": {"stiffness_scale": 3.0},
        },
    },
    "HumanoidStandup-v4": {  # sweep draw h012
        "hyp": {"actor_lr": 3e-6, "critic_lr": 0.0049, "alpha_lr": 0.05, "gamma": 0.97, "beta1": 0.0},
        "N": 6_000_000,
        "core": {"stiff": {"damping_scale": 2.0}},
        "shifts": {
            "payload": {"mass": {"torso": 17.82}},
            "limb_heavy": {"mass": {"right_thigh": 14.26, "right_shin": 8.267, "right_foot": 5.301}},
            "weak_joint": {"gear": {"right_knee": 50}},
            "dead_joint": {"gear": {"right_knee": 0}},
            "flipped_joint": {"gear": {"right_knee": -200}},
            "stiff_springs": {"stiffness_scale": 3.0},
        },
    },
    "Pusher-v5": {  # sweep draw h019
        "hyp": {"actor_lr": 0.00099, "critic_lr": 0.002, "alpha_lr": 0.003, "gamma": 0.95, "beta1": 0.0},
        "N": 2_000_000,
        "shifts": {
            "heavy_object": {"mass": {"object": 0.5}},
            "limb_heavy": {"mass": {"r_forearm_link": 2.53}},
            "weak_joint": {"gear": {"r_elbow_flex_joint": 0.25}},
            "dead_joint": {"gear": {"r_elbow_flex_joint": 0}},
            "flipped_joint": {"gear": {"r_elbow_flex_joint": -1}},
            "rotor": {"armature_scale": 3.0},
        },
    },
    "Reacher-v4": {  # sweep draw h019
        "hyp": {"actor_lr": 0.00099, "critic_lr": 0.002, "alpha_lr": 0.003, "gamma": 0.95, "beta1": 0.0},
        "N": 1_000_000,
        "shifts": {
            "payload": {"mass": {"fingertip": 0.042}},
            "limb_heavy": {"mass": {"body1": 0.107}},
            "weak_joint": {"gear": {"joint1": 50}},
            "dead_joint": {"gear": {"joint1": 0}},
            "flipped_joint": {"gear": {"joint1": -200}},
            "rotor": {"armature_scale": 3.0},
        },
    },
    "dmc-finger-spin": {  # sweep draw h058
        "hyp": {"actor_lr": 0.0024, "critic_lr": 0.00015, "alpha_lr": 0.0005, "gamma": 0.97, "beta1": 0.9},
        "N": 4_000_000,
        "shifts": {
            "heavy_object": {"mass": {"spinner": 4.92}},
            "limb_heavy": {"mass": {"distal": 2.06}},
            "weak_joint": {"gear": {"distal": 3.75}},
            "dead_joint": {"gear": {"distal": 0}},
            "flipped_joint": {"gear": {"distal": -15}},
        },
    },
    "InvertedDoublePendulum-v4": {  # long sweep draw h048; one actuator, so no single-joint shifts
        "hyp": {"actor_lr": 0.0052, "critic_lr": 0.00069, "alpha_lr": 0.1, "gamma": 0.99, "beta1": 0.9},
        "N": 4_000_000,
        "shifts": {
            "payload": {"mass": {"cart": 20.94}},
            "limb_heavy": {"mass": {"pole2": 12.60}},
        },
    },
    "dmc-quadruped-run": {  # long sweep draw h000; position servos, so strength is gain_scale, not gear_scale
        "hyp": {"actor_lr": 0.0048, "critic_lr": 0.00065, "alpha_lr": 0.0004, "gamma": 0.97, "beta1": 0.9},
        "N": 12_000_000,
        "core": {"gear2x": None, "gain2x": {"gain_scale": 2.0}},
        "shifts": {
            "payload": {"mass": {"torso": 135.72}},
            "limb_heavy": {"mass": {"hip_front_left": 12.855, "knee_front_left": 8.766, "ankle_front_left": 4.608, "toe_front_left": 3.216}},
            "dead_joint": {"gear": {"lift_front_left": 0}},
            "flipped_joint": {"gear": {"lift_front_left": -1}},
        },
    },
}

# shifts every environment gets; an env's "core" entry overrides a value, or drops the shift with None
CORE = {
    "heavy": {"mass_scale": 2.0},
    "gear2x": {"gear_scale": 2.0},
    "stiff": {"damping_scale": 5.0},
    "loose": {"damping_scale": 0.2},
    "all_flipped": {"gear_scale": -1.0},
}


def levers(post):
    """name -> extra arguments; the timed levers last a tenth of the post-shift steps, 200k at most"""
    boost = min(200_000, post // 10)
    return {
        "L02_shrink_obs": {"on_shift": [{"action": "shrink_obs_norm", "count": 1000}]},
        "L07_reset_adam": {"on_shift": [{"action": "reset_adam", "net": "both"}]},
        "L08_critic_x3": {"on_shift": [{"action": "lr_boost", "net": "critic", "k": 3, "steps": boost}]},
        "L09_actor_freeze": {"on_shift": [{"action": "lr_boost", "net": "actor", "k": 0, "steps": boost // 4, "mode": "hold"}]},
    }


def short(env):
    return env.replace("dmc-", "").split("-v")[0].lower()


def config(env, part, n_seeds, smoke=False):
    spec = ENVS[env]
    N = 3000 if smoke else spec["N"]
    switch = N // 2
    boost = min(200_000, (N - switch) // 10)
    core = {k: v for k, v in {**CORE, **spec.get("core", {})}.items() if v is not None}
    shifts = core if part == "core" else spec["shifts"]
    experiments = [{"name": "legacy-none"}] if part == "core" else []
    for shift, mods in shifts.items():
        schedule = [{"step": switch, "mods": mods}]
        experiments.append({"name": f"legacy-{shift}", "schedule": schedule})
        experiments.append({"name": f"scratch-{shift}", "env_mods": mods})
        for name, extra in levers(N - switch).items():
            experiments.append({"name": f"{name}-{shift}", "schedule": schedule, **extra})
    seeds = SEEDS[:1] if smoke else SEEDS[:n_seeds]
    cfg = {
        "_comment": [
            f"Shift study on {env}, {part} shifts: {N} steps per run, physics switch at {switch}, {len(seeds)} seed(s), legacy G reset in every run.",
            f"{len(shifts)} shifts ({', '.join(shifts)}). Per shift: legacy (base -> shifted), scratch (shifted from step 0), {len(levers(N - switch))} levers."
            + (" Plus legacy-none, the baseline with no shift." if part == "core" else f" The baseline is in study_{short(env)}_core.json."),
            f"{len(experiments)} conditions x {len(seeds)} seed(s) = {len(experiments) * len(seeds)} runs, one Trillium node.",
            "Generated by scripts/configs/make_shift_study_configs.py; regenerate rather than edit.",
        ],
        "defaults": {
            "N": N,
            "seeds": seeds,
            "normalize_obs": True,
            "pnorm": True,
            "scaled_td": True,
            "legacy_g_reset": True,
            "n_eval": 1 if smoke else 2,
            "video_every": switch,
            "save_model": True,
            "save_model_every": 1_000_000,
            "dense_window": [boost // 10, boost],
        },
        "envs": {env: spec["hyp"]},
        "experiments": experiments,
    }
    if smoke:
        cfg["defaults"]["checkpoint"] = 1000
    return cfg


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", nargs="*", default=list(ENVS), help="Default: every env in ENVS")
    p.add_argument("--n_seeds", default=5, type=int, help=f"1..{len(SEEDS)}")
    p.add_argument("--smoke", action="store_true", help="Write the smoke versions instead")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "configs", "shift_study_2026-11-26"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for env in args.env:
        for part in ("core", "body"):
            cfg = config(env, part, args.n_seeds, args.smoke)
            path = os.path.join(args.out, f"study_{'smoke_' if args.smoke else ''}{short(env)}_{part}.json")
            with open(path, "w") as f:
                json.dump(cfg, f, indent=1)
            runs = len(cfg["experiments"]) * len(cfg["defaults"]["seeds"])
            assert runs <= 186, f"{path}: {runs} runs do not fit one node"
            print(f"{path}: {len(cfg['experiments'])} conditions, {runs} runs")
