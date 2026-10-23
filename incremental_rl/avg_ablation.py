import torch
import argparse, traceback
import time, wandb

import numpy as np
from collections import defaultdict
import torch.nn as nn
import gymnasium as gym
from gymnasium import Env
import torch.nn.functional as F

from torch.distributions import Normal, Independent
from incremental_rl.logger import Logger
from incremental_rl.td_error_scaler import TDErrorScaler
from gymnasium.wrappers import NormalizeObservation, ClipAction
from incremental_rl.utils import orthogonal_weight_init, human_format_numbers, set_one_thread
from incremental_rl.experiment_tracker import ExperimentTracker, record_video

# Skip torch.distributions' per-call argument validation (shape/positive-definiteness checks on every step)
torch.distributions.Distribution.set_default_validate_args(False)


class Actor(nn.Module):
    """ Continous MLP Actor for Soft Actor-Critic """
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
            phi = phi/phi_norm

        mu = self.mu(phi)
        log_std = self.log_std(phi)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
   
        # Diagonal Gaussian. Equivalent to MultivariateNormal(mu, diag_embed(log_std.exp())) (same samples for the
        # same RNG state, log_prob/entropy equal to fp32 precision) but ~3x cheaper: no Cholesky/triangular solve.
        # N.B: exp(log_std) was passed as the *covariance* in the original, hence std = exp(0.5 * log_std).
        dist = Independent(Normal(mu, torch.exp(0.5 * log_std)), 1)

        action_pre = dist.rsample()
        lprob = dist.log_prob(action_pre)
        lprob -= (2 * (np.log(2) - action_pre - F.softplus(-2 * action_pre))).sum(axis=1)
        
        # N.B: Tanh must be applied _only_ after lprob estimation of dist sampled action!! 
        #   A mistake here can break learning :/ 
        action = torch.tanh(action_pre)
        action_info = {'mu': mu, 'log_std': log_std, 'dist': dist, 'lprob': lprob, 'action_pre': action_pre}

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
            phi = phi/phi_norm
        return self.q(phi).view(-1)
       

def _total_norm(tensors):
    """ L2 norm over a collection of tensors (e.g. all grads of a network), as a float """
    return torch.sqrt(sum(torch.norm(t) ** 2 for t in tensors)).item()


class AVG:
    def __init__(self, cfg):
        self.cfg = cfg
        self.steps = 0  

        self.actor = Actor(obs_dim=cfg.obs_dim, action_dim=cfg.action_dim, device=cfg.device, 
                           n_hid=cfg.nhid_actor, pnorm=cfg.pnorm)
        self.Q = Q(obs_dim=cfg.obs_dim, action_dim=cfg.action_dim, device=cfg.device,
                   n_hid=cfg.nhid_critic, pnorm=cfg.pnorm)

        # fused=True: single-kernel Adam update (~3x faster per step on CPU than the per-parameter Python loop)
        self.popt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr,
                                    betas=cfg.betas, weight_decay=cfg.l2_actor, fused=True)
        self.qopt = torch.optim.Adam(self.Q.parameters(), lr=cfg.critic_lr,
                                    betas=cfg.betas, weight_decay=cfg.l2_critic, fused=True)

        self.alpha = cfg.alpha_lr
        self.gamma = cfg.gamma
        self.device = cfg.device

        self.td_error_scaler = TDErrorScaler()
        self.G = 0

        # Per-episode accumulators for logging (see pop_episode_stats)
        self.stats_every = cfg.stats_every
        self._reset_episode_stats()

    def to_tensor(self, obs):
        """ numpy obs -> (1, obs_dim) float32 tensor. Passes tensors through, so callers can convert once per step """
        if isinstance(obs, torch.Tensor):
            return obs
        return torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)

    def compute_action(self, obs):
        action, action_info = self.actor(self.to_tensor(obs))
        return action, action_info

    def update(self, obs, action, next_obs, reward, done, **kwargs):
        obs, next_obs = self.to_tensor(obs), self.to_tensor(next_obs)
        dist, lprob = kwargs['dist'], kwargs['lprob']

        #### Return scaling
        r_ent = reward - self.alpha * lprob.detach().item()
        self.G += r_ent        
        if done:
            self.td_error_scaler.update(reward=r_ent, gamma=0, G=self.G)
            self.G = 0
        else:
            self.td_error_scaler.update(reward=r_ent, gamma=self.cfg.gamma, G=None)
        ####

        #### Q loss
        q = self.Q(obs, action.detach())    # N.B: Gradient should NOT pass through action here
        with torch.no_grad():
            next_action, action_info = self.actor(next_obs)
            next_lprob = action_info['lprob']
            q2 = self.Q(next_obs, next_action)
            target_V = q2 - self.alpha * next_lprob

        delta = reward + (1 - done) *  self.gamma * target_V - q
        if self.cfg.scaled_td:
            delta /= self.td_error_scaler.sigma
        qloss = delta ** 2
        ####

        # Policy loss
        ploss = self.alpha * lprob - self.Q(obs, action) # N.B: USE reparametrized action

        self.steps += 1
        # Gradient norms are the expensive stats: sample them every stats_every updates instead of every step
        sample_norms = self.steps % self.stats_every == 0

        self.popt.zero_grad()
        ploss.backward()
        if sample_norms:
            self._acc['actor_grad_norm'] += _total_norm(p.grad for p in self.actor.parameters())
        self.popt.step()

        self.qopt.zero_grad()
        qloss.backward()
        if sample_norms:
            self._acc['critic_grad_norm'] += _total_norm(p.grad for p in self.Q.parameters())
            self._acc_norm_n += 1
        self.qopt.step()

        ### Accumulate cheap per-step stats; averaged per episode by pop_episode_stats()
        acc = self._acc
        acc['actor_loss'] += ploss.item()
        acc['critic_loss'] += qloss.item()
        acc['td_error'] += delta.abs().item()
        acc['entropy'] += dist.entropy().item()
        acc['mod_action'] += action.abs().mean().item()
        acc['td_sigma'] += float(self.td_error_scaler.sigma)
        self._acc_n += 1

    STAT_KEYS = ['actor_loss', 'critic_loss', 'td_error', 'entropy', 'mod_action', 'td_sigma']
    NORM_KEYS = ['actor_grad_norm', 'critic_grad_norm']

    def _reset_episode_stats(self):
        # Fixed key set so every logged episode has the same columns
        self._acc = {k: 0.0 for k in self.STAT_KEYS + self.NORM_KEYS}
        self._acc_n, self._acc_norm_n = 0, 0

    def pop_episode_stats(self):
        """ Means of the stats accumulated since the last call (i.e. over the episode), plus current weight norms.
        Grad norms are NaN for episodes shorter than stats_every (no sample taken) """
        n = max(1, self._acc_n)
        stats = {k: self._acc[k] / n for k in self.STAT_KEYS}
        for k in self.NORM_KEYS:
            stats[k] = self._acc[k] / self._acc_norm_n if self._acc_norm_n else float('nan')
        stats['actor_weight_norm'] = _total_norm(p.data for p in self.actor.parameters())
        stats['critic_weight_norm'] = _total_norm(p.data for p in self.Q.parameters())
        stats['num_updates'] = self.steps
        self._reset_episode_stats()
        return stats

    def save(self, model_dir, unique_str):
        model = {
            "actor": self.actor.state_dict(),
            "critic": self.Q.state_dict(),
            "policy_opt": self.popt.state_dict(),
            "critic_opt": self.qopt.state_dict(),
        }
        torch.save(
            model, '%s/%s.pt' % (model_dir, unique_str)
        )


def episode_row(episode, step, ep_len, ret, ep_duration, wall_time, stats, partial=0):
    """ One metrics.csv row: x-axis columns first, then the agent's per-episode stats """
    row = {'episode': episode, 'step': step, 'ep_len': ep_len, 'return': ret, 'partial': partial,
           'ep_duration': round(ep_duration, 3), 'wall_time': round(wall_time, 1)}
    row.update(stats)
    return row


def main(args):
    tic = time.time()

    expt = ExperimentTracker(args)
    L = Logger(args.results_dir, prefix=f"{expt.run_id}_", use_tb=False)

    # Env
    env = gym.make(args.env, render_mode = "rgb_array", width=640, height=480)
    model = env.unwrapped.model
    #print(f"Friction Default: {model.geom_friction}")
    #print(f"Inertia Default: {model.body_inertia}")
    #print(f"Mass Default Default: {model.body_mass}")
    #print(f"Actuator Gear Default: {model.actuator_gear}")
    #print(f"Actuator Gain Default: {model.actuator_gainprm}")
    #print(f"Joint Damping Default: {model.dof_damping}")
    #print(f"Gravity Default: {model.opt.gravity}")
    #
    #print(f"Friction Default: {len(model.geom_friction)}")
    #print(f"Inertia Default: {len(model.body_inertia)}")
    #print(f"Mass Default Default: {len(model.body_mass)}")
    #print(f"Actuator Gear Default: {len(model.actuator_gear)}")
    #print(f"Actuator Gain Default: {len(model.actuator_gainprm)}")
    #print(f"Joint Damping Default: {len(model.dof_damping)}")
    #print(f"Gravity Default: {len(model.opt.gravity)}")
    
    env = gym.wrappers.FlattenObservation(env)
    if args.normalize_obs:
        env = NormalizeObservation(env)
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
    args.obs_dim =  env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    # args.action_shape = env.action_space.shape
    agent = AVG(args)

    # Weights & Biases; start a new wandb run to track this script
    wandb.init(project="avg", config=vars(args), name=expt.exp_name, entity="gauthamv", 
               mode=args.wandb_mode, dir=f"{args.results_dir}")

    # Interaction
    rets, ep_steps = [], []
    i_episode, ret, step = 0, 0, 0
    terminated, truncated = False, False
    obs, _ = env.reset()
    obs_t = agent.to_tensor(obs)   # convert each observation once; it is reused as next step's obs
    ep_tic = time.time()
    try:
        for t in range(args.N):
            if t == 3000000:
                # moon gravity:
                model.opt.gravity[-1] = -1.62
            elif t == 6000000:
                # mars gravity:
                model.opt.gravity[-1] = -3.71
            elif t == 9000000:
                # k2-18b gravity (estimate)
                model.opt.gravity[-1] = -11.95
            # N.B: Action is a torch.Tensor
            action, action_info = agent.compute_action(obs_t)
            sim_action = action.detach().cpu().view(-1).numpy()

            # Receive reward and next state
            next_obs, reward, terminated, truncated, _ = env.step(sim_action)
            next_obs_t = agent.to_tensor(next_obs)

            agent.update(obs_t, action, next_obs_t, reward, terminated, **action_info)

            # Log
            ret += reward
            step += 1

            obs_t = next_obs_t

            if t % args.checkpoint == 0 and args.save_model:
                agent.save(model_dir=args.results_dir, unique_str=f"{expt.run_id}_model_{human_format_numbers(t)}")

            # Termination
            if terminated or truncated:
                rets.append(ret)
                ep_steps.append(step)
                i_episode += 1
                row = episode_row(i_episode, t + 1, step, ret, time.time() - ep_tic, time.time() - tic,
                                  agent.pop_episode_stats())
                if args.debug:
                    for k, v in row.items():
                        L.log(f'train/{k}', v, t)
                    L.log('train/episode_return', ret, t)
                    L.log('train/duration', row['ep_duration'], t)
                    L.print_log(t)
                expt.log_episode(row)
                expt.checkpoint(t, rets, ep_steps)

                ep_tic = time.time()
                obs, _ = env.reset()
                obs_t = agent.to_tensor(obs)
                ret, step = 0, 0
    except Exception as e:
        print(e)
        print("Exiting this run, storing partial logs in the database for future debugging...")
        traceback.print_exc()

    if not (terminated or truncated):
        # N.B: We're adding a partial episode just to make plotting easier. But this data point shouldn't be used
        print("Appending partial episode #{}, length: {}, Total Steps: {}".format(i_episode+1, step, t+1))
        rets.append(ret)
        ep_steps.append(step)
        expt.log_episode(episode_row(i_episode + 1, t + 1, step, ret, time.time() - ep_tic, time.time() - tic,
                                     agent.pop_episode_stats(), partial=1))

    # Save returns and learning curve before exiting run
    expt.finish(rets, ep_steps)

    if args.save_model:
        agent.save(model_dir=args.results_dir, unique_str=f"{expt.run_id}_model")

    print("Run with id: {} took {:.3f}s!".format(expt.run_id, time.time()-tic))
    wandb.finish()

    # Eval
    if args.n_eval:
        fname = f"{args.results_dir}/{expt.run_id}.mp4"
        fname = fname.replace("dm_control/", "")
        record_video(env, agent, num_episodes=args.n_eval, video_filename=fname)

    return ep_steps, rets


def get_parser(add_help=True):
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument('--env', default="Hopper-v4", type=str, help="e.g., 'Hopper-v4'")
    parser.add_argument('--seed', default=42, type=int, help="Seed for random number generator")       
    parser.add_argument('--N', default=20001000, type=int, help="# timesteps for the run")
    # SAVG params
    parser.add_argument('--actor_lr', default=0.00006, type=float, help="Actor step size")
    parser.add_argument('--critic_lr', default=0.00087, type=float, help="Critic step size")
    parser.add_argument('--beta1', default=0.9, type=float, help="Beta1 parameter of Adam optimizer")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor")
    parser.add_argument('--alpha_lr', default=0.6, type=float, help="Entropy Coefficient for AVG")
    parser.add_argument('--l2_actor', default=0, type=float, help="L2 Regularization")
    parser.add_argument('--l2_critic', default=0, type=float, help="L2 Regularization")    
    parser.add_argument('--nhid_actor', default=256, type=int)
    parser.add_argument('--nhid_critic', default=256, type=int)
    # Miscellaneous
    parser.add_argument('--checkpoint', default=50000, type=int, help="Save plots and rets every checkpoint")
    parser.add_argument('--results_dir', default="./results", type=str, help="Location to store results")
    parser.add_argument('--device', default="cpu", type=str)
    parser.add_argument('--do_not_save', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--description', default='', type=str)
    parser.add_argument('--wandb_mode', default='disabled', type=str, help="Either online, offline, or disabled")
    parser.add_argument('--debug', action='store_true', default=False)
    # Abaltions args
    parser.add_argument('--normalize_obs', action='store_true', default=False)    
    parser.add_argument('--pnorm', action='store_true', default=False)
    parser.add_argument('--scaled_td', action='store_true', default=False)
    parser.add_argument("--n_eval", default=0, type=int, help="Number of evaluation episodes to record")
    parser.add_argument("--n_threads", default=1, type=int, help="Torch intra-op threads (1 is fastest for incremental updates)")
    # Logging
    parser.add_argument("--stats_every", default=100, type=int, help="Sample gradient norms every k updates (averaged per episode)")
    parser.add_argument("--vlines", default=[3000000, 6000000, 9000000], type=int, nargs='*',
                        help="Steps marked with a vertical line on the learning curve (e.g. gravity switches)")
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
    print(args.algo)
    
    # Start experiment
    set_one_thread()
    torch.set_num_threads(args.n_threads)
    main(args)