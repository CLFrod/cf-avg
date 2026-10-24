import torch
import os
import csv, time
import numpy as np
from datetime import datetime
from incremental_rl.utils import learning_curve, save_args, save_returns, get_git_hash
from gymnasium import Env


class ExperimentTracker:
    def __init__(self, args, run_id=None):
        self.args = args
        self.args.git_hash = get_git_hash()
        #### Unique IDs (a resumed run keeps its original id, so its files are continued rather than recreated)
        self.resumed = run_id is not None
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + f"-{args.algo}-{args.env}_seed-{args.seed}"
        self.lc_path = f"{args.results_dir}/{self.run_id}_learning_curve.png"
        self.rets_path = f"{args.results_dir}/{self.run_id}_returns.txt"
        self.args_path = f"{args.results_dir}/{self.run_id}_args.json"
        self.metrics_path = f"{args.results_dir}/{self.run_id}_metrics.csv"
        self._metrics_file, self._metrics_writer = None, None

        # Autogenerate a expt name
        self.exp_name = "{}-avg-aa-{:.5f}-ca-{:.5f}-beta1-{}".format(args.env, args.actor_lr, args.critic_lr, args.beta1)
        self.exp_name += "-ent-{:.5f}-seed-{}".format(args.alpha_lr, args.seed)
        if args.description:
            self.exp_name = args.description + f"-{self.exp_name}"

        if not self.args.do_not_save:
            os.makedirs(args.results_dir, exist_ok=True)
            save_args(args, self.args_path)
            self.step_on_save = 0

    def log_episode(self, row):
        """ Append one episode's metrics as a CSV row (header written on first call; flushed so a killed run keeps its data) """
        if self.args.do_not_save:
            return
        if self._metrics_writer is None:
            append = self.resumed and os.path.exists(self.metrics_path)
            self._metrics_file = open(self.metrics_path, 'a' if append else 'w', newline='')
            self._metrics_writer = csv.DictWriter(self._metrics_file, fieldnames=list(row.keys()))
            if not append:
                self._metrics_writer.writeheader()
        self._metrics_writer.writerow(row)
        self._metrics_file.flush()

    def checkpoint(self, step, rets, ep_lens):
        """ Save returns + learning curve every args.checkpoint steps """
        if self.args.do_not_save or step - self.step_on_save < self.args.checkpoint:
            return
        self._save_curve(rets, ep_lens)
        self.step_on_save = step

    def finish(self, rets, ep_lens):
        if self._metrics_file is not None:
            self._metrics_file.close()
        if not self.args.do_not_save and rets:
            self._save_curve(rets, ep_lens)

    def _save_curve(self, rets, ep_lens):
        save_returns(ep_lens=ep_lens, rets=rets, save_path=self.rets_path)
        learning_curve(ep_lens=ep_lens, rets=rets, save_path=self.lc_path, vlines=getattr(self.args, 'vlines', None))


# Function to record video
def record_video(env: Env, policy, num_episodes=10, video_filename='video.mp4'):
    import cv2  # imported here, not at module level: opencv needs libGL, which headless compute nodes may lack
    # Define video codec and create VideoWriter object
    print(video_filename)
    video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))

    for episode in range(num_episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        step = 0
        tic = time.time()
        while not (terminated or truncated):
            # Get action from policy
            with torch.no_grad():
                action, action_info = policy.compute_action(obs)
            sim_action = action.cpu().view(-1).numpy()
            next_obs, reward, terminated, truncated, _ = env.step(sim_action)

            # Render the environment
            img = env.render()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            step += 1

            obs = next_obs
            video.write(img)  # Write the frame to video
        
        # Pause in last frame for 300ms
        for _ in range(10):
            video.write(np.zeros((480, 640, 3), dtype=np.uint8))  # Write the frame to video
        print("Episode {} rendering complete, Time taken: {:.2f}".format(
            episode+1, time.time() - tic))

    video.release()  # Release the video writer