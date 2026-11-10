"""On-shift levers: statistics resets, optimiser resets and step-size schedules fired at a physics switch"""
import glob, math, os
import torch
from gymnasium.wrappers import NormalizeObservation
from incremental_rl import checkpoint as ckpt

ACTIONS = ("shrink_obs_norm", "shrink_td_scaler", "load_stats", "reset_adam", "lr_boost", "set_beta2",
           "set_alpha", "alpha_track", "reward_offset")


class CappedNormalizeObservation(NormalizeObservation):
    """NormalizeObservation whose running stats forget with a window of `cap` samples"""

    def __init__(self, env, cap, epsilon=1e-8):
        super().__init__(env, epsilon)
        self.cap = cap

    def observation(self, observation):
        out = super().observation(observation)
        if self.obs_rms.count > self.cap:
            self.obs_rms.count = self.cap
        return out


def _load_one(pattern):
    """torch.load of the single file matching a path with env vars, ~ and wildcards"""
    matches = sorted(glob.glob(os.path.expandvars(os.path.expanduser(pattern))))
    if len(matches) != 1:
        raise FileNotFoundError(f"{pattern!r} matches {len(matches)} files, need exactly 1: {matches[:5]}")
    return torch.load(matches[0], weights_only=False)


def _nets(a):
    net = a.get("net", "both")
    return ("actor", "critic") if net == "both" else (net,)


class Levers:
    """Applies --on_shift actions at each switch, and the per-step schedules they start"""

    def __init__(self, agent, env, cfg):
        self.agent, self.env = agent, env
        self.actions = list(cfg.on_shift or [])
        unknown = {a["action"] for a in self.actions} - set(ACTIONS)
        if unknown:
            raise ValueError(f"unknown on_shift action(s) {sorted(unknown)}; known: {ACTIONS}")
        self.lr_actions = [a for a in self.actions if a["action"] == "lr_boost"]
        self.beta2_actions = [a for a in self.actions if a["action"] == "set_beta2"]
        self.alpha_actions = [a for a in self.actions if a["action"] == "set_alpha"]
        self.track_actions = [a for a in self.actions if a["action"] == "alpha_track"]
        self.offset_actions = [a for a in self.actions if a["action"] == "reward_offset"]
        self.adaptive = cfg.adaptive_lr  # (short window, long window, k_max)
        self.scheduled = bool(self.lr_actions or self.beta2_actions or self.adaptive or self.alpha_actions or self.track_actions)
        self.fired = []
        self.opts = {"actor": agent.popt, "critic": agent.qopt}
        self.base_lr = {"actor": cfg.actor_lr, "critic": cfg.critic_lr}
        self.betas = tuple(cfg.betas)
        self.mult = {"actor": 1.0, "critic": 1.0}
        self.cur_b2 = self.betas[1]
        self.ema = {"short": 0.0, "long": 0.0, "n": 0}
        self.base_alpha = agent.alpha
        self.alpha_mult = 1.0
        self.track = {"mult": 1.0, "h": 0.0, "n": 0}  # entropy-tracking multiplier and its entropy average
        self._reset_stats()

    def on_shift(self, step):
        self.fired.append(step)
        for a in self.actions:
            getattr(self, "act_" + a["action"])(a)
        print(f"[levers] step {step}: fired {[a['action'] for a in self.actions]}")

    def act_shrink_obs_norm(self, a):
        rms = ckpt.find_wrapper(self.env, NormalizeObservation).obs_rms
        count = a.get("count", 1000)
        if count <= 0:
            rms.mean[:], rms.var[:], rms.count = 0.0, 1.0, 1e-4
        else:
            rms.count = min(rms.count, count)

    def act_shrink_td_scaler(self, a):
        self.agent.td_error_scaler.shrink(a.get("steps", 1000), a.get("episodes", 10))

    def act_load_stats(self, a):
        """Normaliser stats from `path` (a model snapshot or checkpoint.pt); scaler stats from `scaler_path` or `path`"""
        state = _load_one(a["path"])
        norm = ckpt.find_wrapper(self.env, NormalizeObservation)
        rms = state.get("obs_rms")
        if rms is not None and norm is not None:
            norm.obs_rms.mean[:], norm.obs_rms.var[:], norm.obs_rms.count = rms["mean"], rms["var"], a.get("count", rms["count"])
        if a.get("obs_only"):
            return
        if "scaler_path" in a:
            state = _load_one(a["scaler_path"])
        scaler = state.get("agent", {}).get("td_scaler")
        if scaler:
            for k, d in scaler.items():
                rs = getattr(self.agent.td_error_scaler, k)
                rs.n, rs.m, rs.s = d["n"], d["m"], d["s"]

    def act_reset_adam(self, a):
        for net in _nets(a):
            self.opts[net].state.clear()

    def act_lr_boost(self, a):
        pass  # schedule, read in step()

    def act_set_beta2(self, a):
        pass  # schedule, read in step()

    def act_set_alpha(self, a):
        pass  # schedule, read in step()

    def act_alpha_track(self, a):
        self.track = {"mult": 1.0, "h": 0.0, "n": 0}

    def act_reward_offset(self, a):
        pass  # schedule, read in reward_offset()

    def reward_offset(self, t):
        """Constant added to the reward the agent learns from at step t (decays to 0 over `steps`); the logged return is unchanged"""
        c = 0.0
        for a in self.offset_actions:
            for f in reversed(self.fired):
                u = t - f
                if 0 <= u < a["steps"]:
                    c += a["offset"] * (1.0 if a.get("mode", "decay") == "hold" else 1 - u / a["steps"])
                    break
        self._acc["reward_offset"] += c
        self._n_offset += 1
        return c

    def step(self, t):
        """Set the step sizes and entropy coefficient for update t+1; call after agent.update at step t"""
        if not self.scheduled:
            return
        if self.adaptive:
            self._update_ema(self.agent.last_td_abs)
        for net in ("actor", "critic"):
            m = self._boost(net, t, self.lr_actions) * self._adaptive_mult()
            if m != self.mult[net]:
                self.mult[net] = m
                for g in self.opts[net].param_groups:
                    g["lr"] = self.base_lr[net] * m
            self._acc[f"lr_mult_{net}"] += self.mult[net]
        m = self._boost(None, t, self.alpha_actions) * self._track_mult(t)
        if m != self.alpha_mult:
            self.alpha_mult = m
            self.agent.alpha = self.base_alpha * m
        self._acc["alpha_mult"] += self.alpha_mult
        self._n += 1
        self._apply_beta2(t)

    def _boost(self, net, t, actions):
        """Product over actions; within an action the most recent firing wins (a second switch restarts it)"""
        m = 1.0
        for a in actions:
            if net is not None and net not in _nets(a):
                continue
            for f in reversed(self.fired):
                u = t - f - a.get("delay", 0)
                if 0 <= u < a["steps"]:
                    k = a["k"]
                    m *= k if a.get("mode", "decay") == "hold" else 1 + (k - 1) * (1 - u / a["steps"])
                    break
        return m

    def _track_mult(self, t):
        """Drive the policy entropy to `target`: alpha falls while the entropy average is above it, rises below, within [min, max]"""
        for a in self.track_actions:
            for f in reversed(self.fired):
                u = t - f
                if 0 <= u < a["steps"]:
                    tr, w = self.track, a.get("window", 1000)
                    tr["n"] += 1
                    tr["h"] += (self.agent.last_entropy - tr["h"]) / min(tr["n"], w)  # Gaussian entropy, same quantity as the target
                    if tr["n"] >= w:
                        tr["mult"] *= math.exp(-a.get("eta", 1e-4) * (tr["h"] - a["target"]))
                        tr["mult"] = min(max(tr["mult"], a.get("min", 0.02)), a.get("max", 2.0))
                    return tr["mult"]
                rel = a.get("release", 100_000)
                if a["steps"] <= u < a["steps"] + rel:  # ease back to the base coefficient
                    return 1 + (self.track["mult"] - 1) * (1 - (u - a["steps"]) / rel)
        return 1.0

    def _apply_beta2(self, t):
        b2 = self.betas[1]
        for a in self.beta2_actions:
            if any(0 <= t - f < a["steps"] for f in self.fired):
                b2 = a["beta2"]
        if b2 != self.cur_b2:
            self.cur_b2 = b2
            for opt in self.opts.values():
                for g in opt.param_groups:
                    g["betas"] = (self.betas[0], b2)

    def _update_ema(self, x):
        short, long, _ = self.adaptive
        self.ema["n"] += 1
        self.ema["short"] += (x - self.ema["short"]) / short
        self.ema["long"] += (x - self.ema["long"]) / long

    def _adaptive_mult(self):
        if not self.adaptive:
            return 1.0
        short, long, k_max = self.adaptive
        n = self.ema["n"]
        if n < long:
            return 1.0  # long window not filled yet
        # bias-corrected averages (both start at 0), so a constant TD error gives exactly 1
        s = self.ema["short"] / (1 - (1 - 1 / short) ** n)
        l = self.ema["long"] / (1 - (1 - 1 / long) ** n)
        return float(min(max(s / max(l, 1e-8), 1.0), k_max))

    def _reset_stats(self):
        self._acc = {"lr_mult_actor": 0.0, "lr_mult_critic": 0.0, "alpha_mult": 0.0, "reward_offset": 0.0}
        self._n = 0
        self._n_offset = 0

    def pop_stats(self):
        n = max(1, self._n)
        out = {k: (v / n if self._n else 1.0) for k, v in self._acc.items() if k != "reward_offset"}
        out["reward_offset"] = self._acc["reward_offset"] / max(1, self._n_offset)
        self._reset_stats()
        return out

    def state(self):
        return {"fired": list(self.fired), "mult": dict(self.mult), "cur_b2": self.cur_b2, "ema": dict(self.ema),
                "alpha_mult": self.alpha_mult, "track": dict(self.track)}

    def load(self, state):
        if state is None:
            return
        self.fired, self.mult, self.cur_b2, self.ema = state["fired"], state["mult"], state["cur_b2"], state["ema"]
        self.alpha_mult, self.track = state.get("alpha_mult", 1.0), state.get("track", self.track)
        self.agent.alpha = self.base_alpha * self.alpha_mult
