"""
Resume support for avg_ablation: everything needed to continue a run from an episode boundary.

A run folder gets `checkpoint.pt` (torch.save of the full state, overwritten each time) and `checkpoint.json`
(step / N / episode / wall time, readable without torch). Saved every --checkpoint steps at the next episode
end, when --time_limit stops the run, and at the end of the run (so a finished run can also be extended with
a larger --N).

Restores: actor, critic, both Adam optimizers, the agent's update counter and return accumulator, the TD-error
scaler's running statistics, the NormalizeObservation running mean/var, and the torch / numpy / env / action-space
RNG states. Physics mods are re-applied by Schedule.steps(start=...). The in-progress episode at save time is
not stored: resumed runs start a fresh episode at the saved step.
"""
import os, json, time
import numpy as np
import torch
from gymnasium.wrappers import NormalizeObservation

CKPT_FILE, META_FILE = "checkpoint.pt", "checkpoint.json"
SCALER_STATS = ("gamma_rms", "return_sq_rms", "reward_rms", "return_rms")


def _find_wrapper(env, cls):
    w = env
    while w is not None:
        if isinstance(w, cls):
            return w
        w = getattr(w, "env", None)
    return None


def meta_path(run_dir):
    return os.path.join(run_dir, META_FILE)


def read_meta(run_dir):
    """ checkpoint.json as a dict, or None if the run has no checkpoint """
    try:
        with open(meta_path(run_dir)) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def save(run_dir, *, step, episode, agent, env, rets, ep_steps, wall_time, run_id, N, complete=False):
    norm = _find_wrapper(env, NormalizeObservation)
    scaler = agent.td_error_scaler
    state = {
        "step": step, "episode": episode, "run_id": run_id, "N": N, "wall_time": wall_time,
        "agent": {
            "actor": agent.actor.state_dict(), "critic": agent.Q.state_dict(),
            "popt": agent.popt.state_dict(), "qopt": agent.qopt.state_dict(),
            "steps": agent.steps, "G": agent.G,
            "td_scaler": {k: {"n": getattr(scaler, k).n, "m": getattr(scaler, k).m, "s": getattr(scaler, k).s}
                          for k in SCALER_STATS},
        },
        "obs_rms": None if norm is None else {
            "mean": norm.obs_rms.mean.copy(), "var": norm.obs_rms.var.copy(), "count": norm.obs_rms.count},
        "rng": {
            "torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "env": env.unwrapped.np_random.bit_generator.state,
            "action_space": env.action_space.np_random.bit_generator.state,
        },
        "rets": list(rets), "ep_steps": list(ep_steps),
    }
    path = os.path.join(run_dir, CKPT_FILE)
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)          # atomic: a kill mid-write leaves the previous checkpoint intact
    with open(meta_path(run_dir), "w") as f:
        json.dump({"step": step, "N": N, "episode": episode, "wall_time": round(wall_time, 1), "run_id": run_id,
                   "complete": complete, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)


def load(run_dir):
    path = os.path.join(run_dir, CKPT_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no {CKPT_FILE} in {run_dir}")
    return torch.load(path, weights_only=False)


def restore(state, agent, env):
    """ Load a saved state into a freshly built agent/env. Call after env.reset(seed=...) so the env has an RNG """
    a = state["agent"]
    agent.actor.load_state_dict(a["actor"])
    agent.Q.load_state_dict(a["critic"])
    agent.popt.load_state_dict(a["popt"])
    agent.qopt.load_state_dict(a["qopt"])
    agent.steps, agent.G = a["steps"], a["G"]
    for k, d in a["td_scaler"].items():
        rs = getattr(agent.td_error_scaler, k)
        rs.n, rs.m, rs.s = d["n"], d["m"], d["s"]

    norm = _find_wrapper(env, NormalizeObservation)
    if state["obs_rms"] is not None:
        if norm is None:
            raise ValueError("checkpoint has NormalizeObservation stats but the env is not normalized (--normalize_obs)")
        norm.obs_rms.mean[:] = state["obs_rms"]["mean"]
        norm.obs_rms.var[:] = state["obs_rms"]["var"]
        norm.obs_rms.count = state["obs_rms"]["count"]
    elif norm is not None:
        raise ValueError("env is normalized (--normalize_obs) but the checkpoint has no NormalizeObservation stats")

    rng = state["rng"]
    torch.set_rng_state(rng["torch"])
    np.random.set_state(rng["numpy"])
    env.unwrapped.np_random.bit_generator.state = rng["env"]
    env.action_space.np_random.bit_generator.state = rng["action_space"]
