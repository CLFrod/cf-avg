import torch
import os, cv2
import time, json
import shutil, subprocess
import numpy as np
from datetime import datetime
from incremental_rl.utils import learning_curve, save_args, save_returns, get_git_hash
from gymnasium import Env


class ExperimentTracker:
    def __init__(self, args, flip_steps=()):
        self.args = args
        self.args.git_hash = get_git_hash()
        self.flip_steps = list(flip_steps)
        #### Unique IDs
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"-{args.algo}-{args.env}_seed-{args.seed}"
        self.lc_path = f"{args.results_dir}/{self.run_id}_learning_curve_smoothed.png"
        self.rets_path = f"{args.results_dir}/{self.run_id}_returns.txt"
        self.args_path = f"{args.results_dir}/{self.run_id}_args.json"
        self.metrics_path = f"{args.results_dir}/{self.run_id}_metrics.json"

        # Autogenerate a expt name
        self.exp_name = "{}-avg-aa-{:.5f}-ca-{:.5f}-beta1-{}".format(args.env, args.actor_lr, args.critic_lr, args.beta1)
        self.exp_name += "-ent-{:.5f}-seed-{}".format(args.alpha_lr, args.seed)
        if args.description:
            self.exp_name = args.description + f"-{self.exp_name}"

        if not self.args.do_not_save:
            os.makedirs(args.results_dir, exist_ok=True)
            save_args(args, self.args_path)
            self.step_on_save = 0

    def learning_curve(self, rets, ep_lens):
        save_returns(ep_lens=ep_lens, rets=rets, save_path=self.rets_path)
        learning_curve(ep_lens=ep_lens, rets=rets, save_path=self.lc_path, flip_steps=self.flip_steps)

    def log_episode_metrics(self, stats):
        log_string = json.dumps(stats)
        with open(self.metrics_path, 'a') as f:
            f.write(log_string + '\n')

    def dump(self, step, rets, ep_lens, stats):
        if self.args.do_not_save:
            return

        self.log_episode_metrics(stats)

        if step - self.step_on_save >= self.args.checkpoint:
            save_returns(ep_lens=ep_lens, rets=rets, save_path=self.rets_path)
            learning_curve(ep_lens=ep_lens, rets=rets, save_path=self.lc_path, flip_steps=self.flip_steps)
            self.step_on_save = step


# Function to record video
def _open_video_writer(path):
    for fourcc in ('avc1', 'mp4v'):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), 30.0, (640, 480))
        if writer.isOpened():
            return writer, fourcc
        writer.release()
    raise RuntimeError("Could not open a video writer for {}".format(path))


def _transcode_to_h264(path):
    if shutil.which('ffmpeg') is None:
        print("WARNING: mp4v video written but ffmpeg not found to transcode to H.264: {}".format(path))
        return
    tmp = path + '.h264.mp4'
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', path,
           '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', tmp]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        raise RuntimeError("ffmpeg transcode failed for {}: {}".format(path, e))
    os.replace(tmp, path)


def record_video(env: Env, policy, num_episodes=10, video_filename='video.mp4'):
    # Define video codec and create VideoWriter object
    print(video_filename)
    video, codec = _open_video_writer(video_filename)

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
    if codec != 'avc1':
        _transcode_to_h264(video_filename)