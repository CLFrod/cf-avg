"""
Environment modifications for the Gymnasium MuJoCo locomotion envs (Ant-v4, Hopper-v4, Humanoid-v4, ...).
See docs/env_modifications.md for every key, and the names/defaults of each env.

Two kinds:
  env_kwargs  reward / termination: constructor arguments, fixed for the run   -> make_env(env_id, env_kwargs)
  mods        physics: fields of the MuJoCo model, settable at any time        -> apply(env, mods)

A Schedule applies mods at given steps without any per-step check in the training loop:

    schedule = Schedule(env, [{"step": 3000000, "mods": {"gravity": -1.62}}])
    for t in schedule.steps(N):      # plain range() iteration inside each segment
        ...

Scale keys (mass_scale, friction_scale, ...) are relative to the model's original values, so applying
`mass_scale: 0.8` and later `mass_scale: 1.0` restores the default.
"""
import copy
import numpy as np
import gymnasium as gym
import mujoco

OPT_KEYS = ("gravity", "wind", "density", "viscosity", "timestep")
SCALE_KEYS = {
    "mass_scale": "body_mass",
    "friction_scale": "geom_friction",
    "damping_scale": "dof_damping",
    "armature_scale": "dof_armature",
    "stiffness_scale": "jnt_stiffness",
    "gear_scale": "actuator_gear",
}
NAMED_KEYS = ("mass", "friction", "damping", "gear")
ALL_KEYS = OPT_KEYS + tuple(SCALE_KEYS) + NAMED_KEYS
BASE_FIELDS = ("body_mass", "body_inertia", "geom_friction", "dof_damping", "dof_armature", "jnt_stiffness",
               "actuator_gear")


def make_env(env_id, env_kwargs=None, **make_kwargs):
    """ gym.make with the reward/termination kwargs. `max_episode_steps` in env_kwargs goes to gym.make itself """
    env_kwargs = dict(env_kwargs or {})
    if "max_episode_steps" in env_kwargs:
        make_kwargs["max_episode_steps"] = env_kwargs.pop("max_episode_steps")
    # Gymnasium's range arguments are tuples; JSON gives lists
    env_kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in env_kwargs.items()}
    return gym.make(env_id, **make_kwargs, **env_kwargs)


def _base(u):
    """ Original values of every field a mod can touch, snapshotted on first use and kept on the (unwrapped) env.
    MjModel itself cannot hold extra attributes """
    base = getattr(u, "_mods_base", None)
    if base is None:
        model = u.model
        base = {f: np.array(getattr(model, f), copy=True) for f in BASE_FIELDS}
        base["opt"] = {k: copy.deepcopy(getattr(model.opt, k)) for k in OPT_KEYS}
        u._mods_base = base
    return base


def _index(model, kind, name):
    """ Resolve a body/geom/joint/actuator name (or integer index) to its index """
    if isinstance(name, int) or (isinstance(name, str) and name.isdigit()):
        return int(name)
    try:
        return getattr(model, kind)(name).id
    except KeyError:
        if kind == "actuator":  # Ant's and Hopper's actuators are unnamed: accept the joint they drive
            for a in range(model.nu):
                if model.joint(model.actuator_trnid[a, 0]).name == name:
                    return a
        names = [getattr(model, kind)(i).name for i in range(getattr(model, f"n{kind}"))]
        raise KeyError(f"unknown {kind} {name!r}; known: {[n for n in names if n]}")


def apply(env, mods):
    """ Apply physics mods to env's MuJoCo model. Returns {key: value applied} for logging """
    if not mods:
        return {}
    u = env.unwrapped
    model, data = u.model, u.data
    base = _base(u)
    applied = {}
    for key, value in mods.items():
        if key not in ALL_KEYS:
            raise KeyError(f"unknown env mod {key!r}; known: {list(ALL_KEYS)}")

        if key == "gravity":
            g = base["opt"]["gravity"].copy()
            if np.isscalar(value):
                g[2] = value
            else:
                g[:] = value
            model.opt.gravity[:] = g
            applied[key] = model.opt.gravity.tolist()
        elif key == "wind":
            model.opt.wind[:] = value
            applied[key] = model.opt.wind.tolist()
        elif key in OPT_KEYS:                       # density, viscosity, timestep
            setattr(model.opt, key, float(value))
            applied[key] = float(value)

        elif key == "mass_scale":
            model.body_mass[:] = base["body_mass"] * value
            model.body_inertia[:] = base["body_inertia"] * value
            applied[key] = value
        elif key == "friction_scale":
            model.geom_friction[:] = base["geom_friction"] * np.asarray(value, dtype=float)
            applied[key] = value
        elif key in SCALE_KEYS:                     # damping, armature, stiffness, gear
            field = SCALE_KEYS[key]
            getattr(model, field)[:] = base[field] * value
            applied[key] = value

        elif key == "mass":
            for name, m in value.items():
                i = _index(model, "body", name)
                ratio = m / base["body_mass"][i]
                model.body_mass[i] = m
                model.body_inertia[i] = base["body_inertia"][i] * ratio
            applied[key] = dict(value)
        elif key == "friction":
            for name, f in value.items():
                i = _index(model, "geom", name)
                if np.isscalar(f):
                    model.geom_friction[i, 0] = f
                else:
                    model.geom_friction[i] = f
            applied[key] = dict(value)
        elif key == "damping":
            for name, d in value.items():
                j = _index(model, "joint", name)
                model.dof_damping[model.jnt_dofadr[j]] = d
            applied[key] = dict(value)
        elif key == "gear":
            for name, g in value.items():
                model.actuator_gear[_index(model, "actuator", name), 0] = g
            applied[key] = dict(value)

    if any(k in mods for k in ("mass_scale", "mass")):
        mujoco.mj_setConst(model, data)             # recompute derived quantities (subtree masses, etc.)
    return applied


def describe(env):
    """ Current values of the physics quantities mods can change, for logging """
    m = env.unwrapped.model
    return {
        "gravity": m.opt.gravity.tolist(), "timestep": m.opt.timestep, "wind": m.opt.wind.tolist(),
        "density": m.opt.density, "viscosity": m.opt.viscosity,
        "total_mass": float(m.body_mass.sum()), "floor_friction": m.geom_friction[0].tolist(),
        "mean_damping": float(m.dof_damping.mean()), "gear": sorted(set(m.actuator_gear[:, 0].tolist())),
    }


class Schedule:
    """ Physics mods applied at given steps. entries: [{"step": int, "mods": {...}}, ...] """

    def __init__(self, env, entries, log=print):
        self.env, self.log = env, log
        entries = sorted(entries or [], key=lambda e: e["step"])
        for e in entries:
            if e["step"] <= 0:
                raise ValueError(f"schedule step must be > 0 (got {e['step']}); use env_mods for step 0")
            if "mods" not in e:
                raise ValueError(f"schedule entry at step {e['step']} has no `mods`")
        if len({e["step"] for e in entries}) != len(entries):
            raise ValueError("schedule has duplicate steps")
        self.entries = entries

    @property
    def switch_steps(self):
        return [e["step"] for e in self.entries]

    def steps(self, n):
        """ Yields 0..n-1, applying each entry's mods just before its step. No per-step check: each segment
        between switches is a plain range() """
        prev = 0
        for e in self.entries:
            if e["step"] >= n:
                break
            yield from range(prev, e["step"])
            applied = apply(self.env, e["mods"])
            self.log(f"[env_mods] step {e['step']}: applied {applied} -> {describe(self.env)}")
            prev = e["step"]
        yield from range(prev, n)


def validate(env_id, env_kwargs=None, env_mods=None, schedule=None):
    """ Build the env once and apply every mod, so bad names/keys fail before a run (or a 188-run job) starts """
    env = make_env(env_id, env_kwargs)
    try:
        apply(env, env_mods)
        for e in Schedule(env, schedule).entries:
            apply(env, e["mods"])
    finally:
        env.close()
