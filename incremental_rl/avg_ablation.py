import torch
import argparse, traceback
import csv, json, os, time, wandb

import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F

from torch.distributions import Normal, Independent
from incremental_rl.logger import Logger
from incremental_rl.td_error_scaler import TDErrorScaler
from gymnasium.wrappers import NormalizeObservation, ClipAction
from incremental_rl.utils import (
    orthogonal_weight_init,
    human_format_numbers,
    set_one_thread,
)
from incremental_rl.experiment_tracker import ExperimentTracker, record_video
from incremental_rl.env_mods import (
    make_env,
    Schedule,
    set_env_kwargs,
    apply as apply_env_mods,
    describe as describe_env,
)
from incremental_rl import checkpoint as ckpt
from incremental_rl.levers import Levers, CappedNormalizeObservation

# Skip torch.distributions' per-call argument validation
torch.distributions.Distribution.set_default_validate_args(False)


class Actor(nn.Module):
    """Continous MLP Actor for Soft Actor-Critic"""

    def __init__(self, obs_dim, action_dim, device, n_hid, pnorm):
        super(Actor, self).__init__()
        self.pnorm = pnorm
        self.device = device
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

        # Two hidden layers
        self.phi = nn.Sequential(
            nn.Linear(obs_dim, n_hid),
            nn.LeakyReLU(),
            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),
        )

        self.mu = nn.Linear(n_hid, action_dim)
        self.log_std = nn.Linear(n_hid, action_dim)

        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, obs):
        phi = self.phi(obs.to(self.device))

        if self.pnorm:
            phi_norm = torch.norm(phi, dim=1).view((-1, 1))
            phi = phi / phi_norm

        mu = self.mu(phi)
        log_std = self.log_std(phi)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        # std = exp(0.5 * log_std): the original passed exp(log_std) as the covariance
        dist = Independent(Normal(mu, torch.exp(0.5 * log_std)), 1)

        action_pre = dist.rsample()
        lprob = dist.log_prob(action_pre)
        lprob -= (2 * (np.log(2) - action_pre - F.softplus(-2 * action_pre))).sum(
            axis=1
        )

        # Tanh only after the log-prob of the pre-tanh sample
        action = torch.tanh(action_pre)
        action_info = {
            "mu": mu,
            "log_std": log_std,
            "dist": dist,
            "lprob": lprob,
            "action_pre": action_pre,
        }

        return action, action_info


class Q(nn.Module):
    def __init__(self, obs_dim, action_dim, device, n_hid, pnorm):
        super(Q, self).__init__()
        self.pnorm = pnorm
        self.device = device

        # Two hidden layers
        self.phi = nn.Sequential(
            nn.Linear(obs_dim + action_dim, n_hid),
            nn.LeakyReLU(),
            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),
        )
        self.q = nn.Linear(n_hid, 1)
        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, obs, action):
        x = torch.cat((obs, action), -1).to(self.device)
        phi = self.phi(x)

        if self.pnorm:
            phi_norm = torch.norm(phi, dim=1).view((-1, 1))
            phi = phi / phi_norm
        return self.q(phi).view(-1)


def _total_norm(tensors):
    """L2 norm over a collection of tensors (e.g. all grads of a network), as a float"""
    return torch.sqrt(sum(torch.norm(t) ** 2 for t in tensors)).item()


def _update_ratio(before, params):
    """||params - before|| / ||before||"""
    params = list(params)
    moved = sum(torch.norm(p.detach() - b) ** 2 for p, b in zip(params, before))
    return (torch.sqrt(moved) / torch.sqrt(sum(torch.norm(b) ** 2 for b in before))).item()


class ObsNormDrift:
    """Per-episode change of the NormalizeObservation running mean/var"""

    KEYS = ("obs_mean_drift", "obs_var_drift", "obs_count")

    def __init__(self, env):
        self.norm = ckpt.find_wrapper(env, NormalizeObservation)
        self.prev = self.state()

    def state(self):
        if self.norm is None:
            return None
        return self.norm.obs_rms.mean.copy(), self.norm.obs_rms.var.copy()

    def load(self, state):
        if state is not None:
            self.prev = tuple(np.array(x) for x in state)

    def pop(self):
        if self.norm is None:
            return dict.fromkeys(self.KEYS, float("nan"))
        now = self.state()
        out = {
            "obs_mean_drift": float(np.linalg.norm(now[0] - self.prev[0])),
            "obs_var_drift": float(np.linalg.norm(now[1] - self.prev[1])),
            "obs_count": float(self.norm.obs_rms.count),
        }
        self.prev = now
        return out


class DenseLog:
    """Per-step rows inside --dense_window around each physics switch, in <run_id>_dense.csv"""

    def __init__(self, args, expt, windows):
        self.windows = windows
        self.path = f"{args.results_dir}/{expt.run_id}_dense.csv"
        self.enabled = bool(windows) and not args.do_not_save
        self.file = self.writer = None

    def active(self, t):
        return self.enabled and any(lo <= t < hi for lo, hi in self.windows)

    def write(self, t, episode, ep_step, reward, terminated, truncated, stats):
        row = {
            "step": t + 1,
            "episode": episode,
            "ep_step": ep_step,
            "reward": reward,
            "terminated": int(terminated),
            "truncated": int(truncated),
            **stats,
        }
        if self.writer is None:
            append = os.path.exists(self.path)
            self.file = open(self.path, "a" if append else "w", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            if not append:
                self.writer.writeheader()
        self.writer.writerow(row)

    def flush(self):
        if self.file:
            self.file.flush()


class AVG:
    def __init__(self, cfg):
        self.cfg = cfg
        self.steps = 0

        self.actor = Actor(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            device=cfg.device,
            n_hid=cfg.nhid_actor,
            pnorm=cfg.pnorm,
        )
        self.Q = Q(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            device=cfg.device,
            n_hid=cfg.nhid_critic,
            pnorm=cfg.pnorm,
        )

        # fused=True: ~3x faster per step on CPU
        self.popt = torch.optim.Adam(
            self.actor.parameters(),
            lr=cfg.actor_lr,
            betas=cfg.betas,
            weight_decay=cfg.l2_actor,
            fused=True,
        )
        self.qopt = torch.optim.Adam(
            self.Q.parameters(),
            lr=cfg.critic_lr,
            betas=cfg.betas,
            weight_decay=cfg.l2_critic,
            fused=True,
        )

        self.alpha = cfg.alpha_lr
        self.gamma = cfg.gamma
        self.device = cfg.device

        self.td_error_scaler = TDErrorScaler(*(cfg.td_scaler_cap or (None, None)))
        self.G = 0
        self.last_td_abs = 0.0
        self.last_lprob = 0.0

        # Per-episode accumulators for logging (see pop_episode_stats); dense = sample every step
        self.stats_every = cfg.stats_every
        self.dense = False
        self._reset_episode_stats()

    def to_tensor(self, obs):
        """numpy obs -> (1, obs_dim) float32 tensor; tensors pass through"""
        if isinstance(obs, torch.Tensor):
            return obs
        return torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)

    def compute_action(self, obs):
        action, action_info = self.actor(self.to_tensor(obs))
        return action, action_info

    def update(self, obs, action, next_obs, reward, done, truncated=False, **kwargs):
        obs, next_obs = self.to_tensor(obs), self.to_tensor(next_obs)
        dist, lprob = kwargs["dist"], kwargs["lprob"]

        #### Return scaling
        r_ent = reward - self.alpha * lprob.detach().item()
        self.G += r_ent
        episode_over = done or (truncated and not self.cfg.legacy_g_reset)
        if episode_over:
            self.td_error_scaler.update(
                reward=r_ent, gamma=0 if done else self.gamma, G=self.G
            )
            self.G = 0
        else:
            self.td_error_scaler.update(reward=r_ent, gamma=self.gamma, G=None)
        ####

        #### Q loss
        q = self.Q(
            obs, action.detach()
        )  # N.B: Gradient should NOT pass through action here
        with torch.no_grad():
            next_action, action_info = self.actor(next_obs)
            next_lprob = action_info["lprob"]
            q2 = self.Q(next_obs, next_action)
            target_V = q2 - self.alpha * next_lprob

        delta = reward + (1 - done) * self.gamma * target_V - q
        delta_raw = delta.item()
        self.last_td_abs = abs(delta_raw)
        self.last_lprob = lprob.detach().item()
        sigma = float(self.td_error_scaler.sigma)
        if self.cfg.scaled_td:
            delta /= sigma
        qloss = delta**2

        # Policy loss, with the reparametrized action
        q_pi = self.Q(obs, action)
        ploss = self.alpha * lprob - q_pi

        self.steps += 1
        sample = self.steps % self.stats_every == 0 or self.dense
        if sample:
            action.retain_grad()  # d(ploss)/d(action) = -dQ/da
            actor_before = [p.detach().clone() for p in self.actor.parameters()]
            critic_before = [p.detach().clone() for p in self.Q.parameters()]

        self.popt.zero_grad()
        ploss.backward()
        self.popt.step()

        self.qopt.zero_grad()
        qloss.backward()
        self.qopt.step()

        acc = self._acc
        acc["actor_loss"] += ploss.item()
        acc["actor_loss_ent"] += self.alpha * lprob.item()
        acc["actor_loss_q"] += q_pi.item()
        acc["critic_loss"] += qloss.item()
        acc["td_error"] += delta.abs().item()
        acc["td_signed"] += delta_raw
        acc["td_abs_raw"] += abs(delta_raw)
        acc["entropy"] += dist.entropy().item()
        acc["mod_action"] += action.abs().mean().item()
        acc["td_sigma"] += sigma
        self._acc_n += 1
        if not sample:
            return None

        s = {
            "actor_grad_norm": _total_norm(p.grad for p in self.actor.parameters()),
            "critic_grad_norm": _total_norm(p.grad for p in self.Q.parameters()),
            "dq_da_norm": action.grad.norm().item(),
            "actor_update_ratio": _update_ratio(actor_before, self.actor.parameters()),
            "critic_update_ratio": _update_ratio(critic_before, self.Q.parameters()),
        }
        for name, layer in self._layers().items():
            s[f"{name}_grad"] = _total_norm(p.grad for p in layer.parameters())
        for k, v in s.items():
            acc[k] += v
        self._acc_norm_n += 1
        if not self.dense:
            return None
        s.update(td_signed=delta_raw, td_sigma=sigma, entropy=dist.entropy().item())
        return s

    STAT_KEYS = [
        "actor_loss",
        "actor_loss_ent",
        "actor_loss_q",
        "critic_loss",
        "td_error",
        "td_signed",
        "td_abs_raw",
        "entropy",
        "mod_action",
        "td_sigma",
    ]
    LAYER_NAMES = [
        "actor_l1",
        "actor_l2",
        "actor_mu",
        "actor_logstd",
        "critic_l1",
        "critic_l2",
        "critic_out",
    ]
    NORM_KEYS = [
        "actor_grad_norm",
        "critic_grad_norm",
        "dq_da_norm",
        "actor_update_ratio",
        "critic_update_ratio",
    ] + [f"{n}_grad" for n in LAYER_NAMES]

    def _layers(self):
        mods = [
            self.actor.phi[0],
            self.actor.phi[2],
            self.actor.mu,
            self.actor.log_std,
            self.Q.phi[0],
            self.Q.phi[2],
            self.Q.q,
        ]
        return dict(zip(self.LAYER_NAMES, mods))

    def _reset_episode_stats(self):
        # Fixed key set so every logged episode has the same columns
        self._acc = {k: 0.0 for k in self.STAT_KEYS + self.NORM_KEYS}
        self._acc_n, self._acc_norm_n = 0, 0

    def pop_episode_stats(self):
        """Per-episode means of the accumulated stats, plus current weight norms and scaler stats"""
        n = max(1, self._acc_n)
        stats = {k: self._acc[k] / n for k in self.STAT_KEYS}
        for k in self.NORM_KEYS:
            stats[k] = (
                self._acc[k] / self._acc_norm_n if self._acc_norm_n else float("nan")
            )
        stats["actor_weight_norm"] = _total_norm(
            p.data for p in self.actor.parameters()
        )
        stats["critic_weight_norm"] = _total_norm(p.data for p in self.Q.parameters())
        for name, layer in self._layers().items():
            stats[f"{name}_w"] = _total_norm(p.data for p in layer.parameters())
        sc = self.td_error_scaler
        stats["scaler_reward_var"] = float(sc.reward_rms.variance)
        stats["scaler_gamma_var"] = float(sc.gamma_rms.variance)
        stats["scaler_G2_mean"] = float(sc.return_sq_rms.mean)
        stats["scaler_G_count"] = float(sc.return_sq_rms.n)
        stats["num_updates"] = self.steps
        self._reset_episode_stats()
        return stats

    def save(self, model_dir, unique_str, env=None):
        """Networks, optimizers and (with env) the observation normaliser stats"""
        model = {
            "actor": self.actor.state_dict(),
            "critic": self.Q.state_dict(),
            "policy_opt": self.popt.state_dict(),
            "critic_opt": self.qopt.state_dict(),
            "obs_rms": ckpt.obs_rms_state(env) if env is not None else None,
        }
        torch.save(model, "%s/%s.pt" % (model_dir, unique_str))


def make_video_recorder(args, train_env, agent, schedule, expt):
    """record(step) writes --n_eval episodes to <run_id>_video_<step>.mp4"""
    state = {"env": None, "disabled": False}

    def record(step):
        if args.n_eval <= 0 or args.do_not_save or state["disabled"]:
            return
        rng = torch.get_rng_state(), np.random.get_state()
        try:
            if state["env"] is None:
                e = make_env(
                    args.env,
                    args.env_kwargs,
                    render_mode="rgb_array",
                    width=640,
                    height=480,
                )
                e = gym.wrappers.FlattenObservation(e)
                if args.normalize_obs:
                    e = NormalizeObservation(e)
                    e.update_running_mean = False
                e = ClipAction(e)
                e.reset(seed=args.seed + 10_000)
                state["env"] = e
            e = state["env"]
            apply_env_mods(e, schedule.current_mods)
            set_env_kwargs(e, schedule.current_kwargs)
            train_norm = ckpt.find_wrapper(train_env, NormalizeObservation)
            if train_norm is not None:
                eval_norm = ckpt.find_wrapper(e, NormalizeObservation)
                eval_norm.obs_rms.mean[:] = train_norm.obs_rms.mean
                eval_norm.obs_rms.var[:] = train_norm.obs_rms.var
                eval_norm.obs_rms.count = train_norm.obs_rms.count
            fname = f"{args.results_dir}/{expt.run_id}_video_{human_format_numbers(step)}.mp4"
            print(f"[video] step {step}: recording {args.n_eval} episode(s) -> {fname}")
            record_video(e, agent, num_episodes=args.n_eval, video_filename=fname)
        except Exception as ex:
            print(
                f"[video] recording failed ({type(ex).__name__}: {ex}); videos disabled for this run"
            )
            state["disabled"] = True
        finally:
            torch.set_rng_state(rng[0])
            np.random.set_state(rng[1])

    return record


def episode_row(episode, step, ep_len, ret, ep_duration, wall_time, stats, partial=0):
    """One metrics.csv row: x-axis columns first, then the agent's per-episode stats"""
    row = {
        "episode": episode,
        "step": step,
        "ep_len": ep_len,
        "return": ret,
        "partial": partial,
        "ep_duration": round(ep_duration, 3),
        "wall_time": round(wall_time, 1),
    }
    row.update(stats)
    return row


def main(args):
    tic = time.time()

    # Mark the physics switches on the learning curve unless told otherwise
    if args.vlines is None:
        args.vlines = sorted(e["step"] for e in args.schedule)

    # Resume from args.resume's checkpoint, keeping its run id
    saved = ckpt.load(args.resume) if args.resume else None
    if saved is not None and saved["step"] >= args.N:
        print(
            f"[resume] {args.resume} is already complete ({saved['step']} >= N={args.N}); nothing to do"
        )
        return saved["ep_steps"], saved["rets"]
    expt = ExperimentTracker(args, run_id=saved["run_id"] if saved else None)
    L = Logger(args.results_dir, prefix=f"{expt.run_id}_", use_tb=False)

    # See docs/env_modifications.md
    env = make_env(
        args.env, args.env_kwargs, render_mode="rgb_array", width=640, height=480
    )
    applied = apply_env_mods(env, args.env_mods)
    schedule = Schedule(
        env, args.schedule, initial_mods=args.env_mods, initial_kwargs=args.env_kwargs
    )
    print(f"[env_mods] step 0: applied {applied} -> {describe_env(env)}")
    if schedule.entries:
        print(f"[env_mods] schedule: {schedule.entries}")

    env = gym.wrappers.FlattenObservation(env)
    if args.normalize_obs:
        env = CappedNormalizeObservation(env, args.obs_norm_cap) if args.obs_norm_cap else NormalizeObservation(env)
    env = ClipAction(env)

    #### Reproducibility
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    ####

    # Learner
    args.obs_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    # args.action_shape = env.action_space.shape
    agent = AVG(args)
    levers = Levers(agent, env, args)

    # Weights & Biases; start a new wandb run to track this script
    wandb.init(
        project="avg",
        config=vars(args),
        name=expt.exp_name,
        entity="gauthamv",
        mode=args.wandb_mode,
        dir=f"{args.results_dir}",
    )

    # Videos every --video_every steps, at each physics switch, and at the end
    record = make_video_recorder(args, env, agent, schedule, expt)
    video_steps = (
        set(range(args.video_every, args.N, args.video_every))
        if args.video_every
        else set()
    )
    video_steps |= set(schedule.switch_steps)
    switch_steps = set(schedule.switch_steps)

    def on_event(s):
        record(s)
        if s in switch_steps:
            levers.on_shift(s)

    # Extra instrumentation: normaliser drift per episode, per-step rows around each switch
    drift = ObsNormDrift(env)
    windows = (
        [(s - args.dense_window[0], s + args.dense_window[1]) for s in schedule.switch_steps]
        if args.dense_window
        else []
    )
    dense = DenseLog(args, expt, windows)

    # Interaction
    rets, ep_steps = [], []
    i_episode, ret, step = 0, 0, 0
    t0 = 0
    if saved is not None:
        ckpt.restore(saved, agent, env)
        drift.load(saved.get("extra", {}).get("obs_drift"))
        levers.load(saved.get("extra", {}).get("levers"))
        t0, i_episode = saved["step"], saved["episode"]
        rets, ep_steps = list(saved["rets"]), list(saved["ep_steps"])
        tic = time.time() - saved["wall_time"]
        expt.step_on_save = t0
        print(
            f"[resume] continuing {expt.run_id} from step {t0} (episode {i_episode}, {saved['wall_time']/3600:.2f} h so far)"
        )
    proc_tic, last_ckpt, stopped_early = time.time(), t0, False
    terminated, truncated = False, False
    obs, _ = env.reset()
    obs_t = agent.to_tensor(obs)
    ep_tic = time.time()
    try:
        for t in schedule.steps(args.N, start=t0, at=video_steps, callback=on_event):
            action, action_info = agent.compute_action(obs_t)
            sim_action = action.detach().cpu().view(-1).numpy()

            # Receive reward and next state
            next_obs, reward, terminated, truncated, _ = env.step(sim_action)
            next_obs_t = agent.to_tensor(next_obs)

            agent.dense = dense.active(t)
            step_stats = agent.update(
                obs_t,
                action,
                next_obs_t,
                reward + levers.reward_offset(t),
                terminated,
                truncated=truncated,
                **action_info,
            )

            # Log
            ret += reward
            step += 1
            if step_stats:
                dense.write(t, i_episode + 1, step, reward, terminated, truncated, step_stats)
            levers.step(t)

            obs_t = next_obs_t

            if (
                args.save_model
                and args.save_model_every
                and t % args.save_model_every == 0
                and t > 0
            ):
                agent.save(
                    model_dir=args.results_dir,
                    unique_str=f"{expt.run_id}_model_{human_format_numbers(t)}",
                    env=env,
                )

            # Termination
            if terminated or truncated:
                rets.append(ret)
                ep_steps.append(step)
                i_episode += 1
                row = episode_row(
                    i_episode,
                    t + 1,
                    step,
                    ret,
                    time.time() - ep_tic,
                    time.time() - tic,
                    {**agent.pop_episode_stats(), **drift.pop(), **levers.pop_stats()},
                )
                dense.flush()
                if args.debug:
                    for k, v in row.items():
                        L.log(f"train/{k}", v, t)
                    L.log("train/episode_return", ret, t)
                    L.log("train/duration", row["ep_duration"], t)
                    L.print_log(t)
                expt.log_episode(row)
                expt.checkpoint(t, rets, ep_steps)

                # Checkpoint every args.checkpoint steps, or stop cleanly at the time limit
                out_of_time = (
                    args.time_limit and time.time() - proc_tic > args.time_limit * 3600
                )
                if not args.do_not_save and (
                    t + 1 - last_ckpt >= args.checkpoint or out_of_time
                ):
                    ckpt.save(
                        args.results_dir,
                        step=t + 1,
                        episode=i_episode,
                        agent=agent,
                        env=env,
                        rets=rets,
                        ep_steps=ep_steps,
                        wall_time=time.time() - tic,
                        run_id=expt.run_id,
                        N=args.N,
                        extra={"obs_drift": drift.state(), "levers": levers.state()},
                    )
                    last_ckpt = t + 1
                if out_of_time:
                    stopped_early = True
                    print(
                        f"[time_limit] {args.time_limit} h reached at step {t + 1}; checkpoint saved. "
                        f"Continue with --resume {args.results_dir}"
                    )
                    record(t + 1)
                    break

                ep_tic = time.time()
                obs, _ = env.reset()
                obs_t = agent.to_tensor(obs)
                ret, step = 0, 0
    except Exception as e:
        print(e)
        print(
            "Exiting this run, storing partial logs in the database for future debugging..."
        )
        traceback.print_exc()

    if not (terminated or truncated):
        # Partial episode, logged for plotting only
        print(
            "Appending partial episode #{}, length: {}, Total Steps: {}".format(
                i_episode + 1, step, t + 1
            )
        )
        rets.append(ret)
        ep_steps.append(step)
        expt.log_episode(
            episode_row(
                i_episode + 1,
                t + 1,
                step,
                ret,
                time.time() - ep_tic,
                time.time() - tic,
                {**agent.pop_episode_stats(), **drift.pop(), **levers.pop_stats()},
                partial=1,
            )
        )

    # Save returns and learning curve before exiting run
    dense.flush()
    expt.finish(rets, ep_steps)
    if not args.do_not_save and not stopped_early:
        # Final checkpoint; marks the run complete and allows extending it with a larger --N
        ckpt.save(
            args.results_dir,
            step=t + 1,
            episode=i_episode,
            agent=agent,
            env=env,
            rets=rets,
            ep_steps=ep_steps,
            wall_time=time.time() - tic,
            run_id=expt.run_id,
            N=args.N,
            complete=t + 1 >= args.N,
            extra={"obs_drift": drift.state(), "levers": levers.state()},
        )

    if args.save_model:
        agent.save(
            model_dir=args.results_dir, unique_str=f"{expt.run_id}_model", env=env
        )

    print("Run with id: {} took {:.3f}s!".format(expt.run_id, time.time() - tic))
    wandb.finish()

    # Final video (a time-limit stop already recorded one)
    if not stopped_early:
        record(t + 1)

    return ep_steps, rets


def get_parser(add_help=True):
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument(
        "--env", default="Hopper-v4", type=str, help="e.g., 'Hopper-v4'"
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed for random number generator"
    )
    parser.add_argument(
        "--N", default=20001000, type=int, help="# timesteps for the run"
    )
    # SAVG params
    parser.add_argument(
        "--actor_lr", default=0.00006, type=float, help="Actor step size"
    )
    parser.add_argument(
        "--critic_lr", default=0.00087, type=float, help="Critic step size"
    )
    parser.add_argument(
        "--beta1", default=0.9, type=float, help="Beta1 parameter of Adam optimizer"
    )
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
    parser.add_argument(
        "--alpha_lr", default=0.6, type=float, help="Entropy Coefficient for AVG"
    )
    parser.add_argument("--l2_actor", default=0, type=float, help="L2 Regularization")
    parser.add_argument("--l2_critic", default=0, type=float, help="L2 Regularization")
    parser.add_argument("--nhid_actor", default=256, type=int)
    parser.add_argument("--nhid_critic", default=256, type=int)
    # Miscellaneous
    parser.add_argument(
        "--checkpoint",
        default=50000,
        type=int,
        help="Save plots and rets every checkpoint",
    )
    parser.add_argument(
        "--results_dir", default="./results", type=str, help="Location to store results"
    )
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--do_not_save", action="store_true", default=False)
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=False,
        help="Save <run_id>_model_<step>.pt every --save_model_every steps and <run_id>_model.pt at the end",
    )
    parser.add_argument(
        "--save_model_every",
        default=1000000,
        type=int,
        help="Model snapshot interval in steps when --save_model (0 = only at the end)",
    )
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--description", default="", type=str)
    parser.add_argument(
        "--wandb_mode",
        default="disabled",
        type=str,
        help="Either online, offline, or disabled",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    # Abaltions args
    parser.add_argument("--normalize_obs", action="store_true", default=False)
    parser.add_argument("--pnorm", action="store_true", default=False)
    parser.add_argument("--scaled_td", action="store_true", default=False)
    parser.add_argument(
        "--legacy_g_reset",
        action="store_true",
        default=False,
        help="Reset the scaler's return accumulator G only on termination, as in the original code (default: also on truncation)",
    )
    # Levers (see incremental_rl/levers.py)
    parser.add_argument(
        "--on_shift",
        default=[],
        type=json.loads,
        help='Actions fired at each schedule switch (see incremental_rl/levers.py ACTIONS), e.g. \'[{"action": "reset_adam"}, '
             '{"action": "set_alpha", "k": 0.1, "steps": 300000}, {"action": "reward_offset", "offset": 4.5, "steps": 300000}]\'',
    )
    parser.add_argument(
        "--obs_norm_cap", default=0, type=int, help="Observation normaliser as a moving average over this many steps (0 = off)"
    )
    parser.add_argument(
        "--td_scaler_cap",
        default=None,
        type=int,
        nargs=2,
        metavar=("STEPS", "EPISODES"),
        help="TD-error scaler stats as moving averages over STEPS steps and EPISODES episodes",
    )
    parser.add_argument(
        "--adaptive_lr",
        default=None,
        type=float,
        nargs=3,
        metavar=("SHORT", "LONG", "KMAX"),
        help="Scale both step sizes by clip(short / long moving average of |TD error|, 1, KMAX)",
    )
    parser.add_argument(
        "--n_eval",
        default=0,
        type=int,
        help="Episodes per video. Videos are recorded every --video_every steps, at each physics switch, and at the end",
    )
    parser.add_argument(
        "--video_every",
        default=0,
        type=int,
        help="Record a video every this many steps (0 = only at switches/end)",
    )
    parser.add_argument(
        "--n_threads",
        default=1,
        type=int,
        help="Torch intra-op threads (1 is fastest for incremental updates)",
    )
    # Logging
    parser.add_argument(
        "--stats_every",
        default=100,
        type=int,
        help="Sample gradient norms every k updates (averaged per episode)",
    )
    parser.add_argument(
        "--dense_window",
        default=None,
        type=int,
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help="Log every step to <run_id>_dense.csv from BEFORE steps before to AFTER steps after each switch",
    )
    parser.add_argument(
        "--vlines",
        default=None,
        type=int,
        nargs="*",
        help="Steps marked with a vertical line on the learning curve (default: the schedule's switch steps)",
    )
    # Resume
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="Run folder with a checkpoint.pt to continue from (keeps its run id and metrics.csv)",
    )
    parser.add_argument(
        "--time_limit",
        default=0.0,
        type=float,
        help="Hours of wall time after which the run checkpoints and exits cleanly (0 = no limit)",
    )
    # Environment modifications (JSON strings; see docs/env_modifications.md)
    parser.add_argument(
        "--env_kwargs",
        default={},
        type=json.loads,
        help='Reward/termination constructor kwargs, e.g. \'{"ctrl_cost_weight": 0.25, "max_episode_steps": 500}\'',
    )
    parser.add_argument(
        "--env_mods",
        default={},
        type=json.loads,
        help='Physics mods applied at step 0, e.g. \'{"gravity": -1.62, "mass_scale": 0.8}\'',
    )
    parser.add_argument(
        "--schedule",
        default=[],
        type=json.loads,
        help='Physics mods applied at given steps, e.g. \'[{"step": 3000000, "mods": {"gravity": -3.71}}]\'',
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    # Adam
    args.betas = [args.beta1, 0.999]

    # CPU/GPU use for the run
    if torch.cuda.is_available() and "cuda" in args.device:
        args.device = torch.device(args.device)
        print("Using CUDA GPU")
    else:
        args.device = torch.device("cpu")
        print("Using CPU")

    if not (args.normalize_obs or args.pnorm or args.scaled_td):
        args.algo = "avg_basic"
    elif args.pnorm and not (args.normalize_obs or args.scaled_td):
        args.algo = "avg_pnorm_only"
    elif args.normalize_obs and not (args.pnorm or args.scaled_td):
        args.algo = "avg_norm_obs_only"
    elif (args.normalize_obs and args.pnorm) and not args.scaled_td:
        args.algo = "avg_norm"
    elif args.normalize_obs and args.pnorm and args.scaled_td:
        args.algo = "avg_norm_scaled"
    else:
        args.algo = "avg_scaled_variant"
    if args.legacy_g_reset:
        args.algo += "_legacyG"
    print(args.algo)

    # Start experiment
    set_one_thread()
    torch.set_num_threads(args.n_threads)
    main(args)
