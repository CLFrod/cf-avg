"""On-shift levers: statistics resets, optimiser resets and step-size schedules fired at a physics switch"""
import glob, os
import torch
from gymnasium.wrappers import NormalizeObservation
from incremental_rl import checkpoint as ckpt

ACTIONS = ("shrink_obs_norm", "shrink_td_scaler", "load_stats", "reset_adam", "lr_boost", "set_beta2")


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
        self.adaptive = cfg.adaptive_lr  # (short window, long window, k_max)
        self.scheduled = bool(self.lr_actions or self.beta2_actions or self.adaptive)
        self.fired = []
        self.opts = {"actor": agent.popt, "critic": agent.qopt}
        self.base_lr = {"actor": cfg.actor_lr, "critic": cfg.critic_lr}
        self.betas = tuple(cfg.betas)
        self.mult = {"actor": 1.0, "critic": 1.0}
        self.cur_b2 = self.betas[1]
        self.ema = {"short": 0.0, "long": 0.0, "n": 0}
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
            norm.obs_rms.mean[:], norm.obs_rms.var[:], norm.obs_rms.count = rms["mean"], rms["var"], rms["count"]
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

    def step(self, t):
        """Set the step sizes for update t+1; call after agent.update at step t"""
        if not self.scheduled:
            return
        if self.adaptive:
            self._update_ema(self.agent.last_td_abs)
        for net in ("actor", "critic"):
            m = self._boost(net, t) * self._adaptive_mult()
            if m != self.mult[net]:
                self.mult[net] = m
                for g in self.opts[net].param_groups:
                    g["lr"] = self.base_lr[net] * m
            self._acc[f"lr_mult_{net}"] += self.mult[net]
        self._n += 1
        self._apply_beta2(t)

    def _boost(self, net, t):
        m = 1.0
        for a in self.lr_actions:
            if net not in _nets(a):
                continue
            for f in self.fired:
                u = t - f - a.get("delay", 0)
                if 0 <= u < a["steps"]:
                    k = a["k"]
                    m *= k if a.get("mode", "decay") == "hold" else 1 + (k - 1) * (1 - u / a["steps"])
        return m

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
        if not self.adaptive or self.ema["n"] < self.adaptive[1]:
            return 1.0  # long window not filled yet
        ratio = self.ema["short"] / max(self.ema["long"], 1e-8)
        return float(min(max(ratio, 1.0), self.adaptive[2]))

    def _reset_stats(self):
        self._acc = {"lr_mult_actor": 0.0, "lr_mult_critic": 0.0}
        self._n = 0

    def pop_stats(self):
        n = max(1, self._n)
        out = {k: (v / n if self._n else 1.0) for k, v in self._acc.items()}
        self._reset_stats()
        return out

    def state(self):
        return {"fired": list(self.fired), "mult": dict(self.mult), "cur_b2": self.cur_b2, "ema": dict(self.ema)}

    def load(self, state):
        if state is None:
            return
        self.fired, self.mult, self.cur_b2, self.ema = state["fired"], state["mult"], state["cur_b2"], state["ema"]
