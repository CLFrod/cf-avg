"""
Launch many independent avg_ablation runs (one per seed) in parallel.

Each run is a separate `python -m incremental_rl.avg_ablation` subprocess, so runs are fully
isolated (own env, own torch RNG, own MuJoCo model). Every avg_ablation hyperparameter is
accepted here and forwarded to all runs.

Example:
    python -m incremental_rl.avg_multi_experiment --n_runs 10 --n_workers 5 --name moon_earth \
        --env Hopper-v4 --N 1000000 --actor_lr 0.0003 --alpha_lr 0.1 --normalize_obs --pnorm --scaled_td
"""
import os, sys
import time, json
import argparse
import subprocess
import numpy as np
from datetime import datetime
from incremental_rl.avg_ablation import get_parser


def add_launcher_args(parser):
    group = parser.add_argument_group("multi-experiment")
    group.add_argument('--n_runs', default=10, type=int, help="Number of avg_ablation runs to launch")
    group.add_argument('--n_workers', default=os.cpu_count(), type=int, help="Max runs executing concurrently")
    group.add_argument('--start_seed', default=0, type=int, help="Run i uses seed start_seed + i")
    group.add_argument('--seeds', default=None, type=int, nargs='+', help="Explicit seed list (overrides n_runs/start_seed)")
    group.add_argument('--results_dir', default="./results", type=str, help="Parent dir; each experiment gets a subfolder")
    group.add_argument('--name', default="avg_multi", type=str,
                       help="Experiment name: used for the subfolder and as each run's --description")
    return parser


def parse_args():
    # Full parser (launcher + every avg_ablation hyperparameter) for --help and to reject typos up front,
    # rather than having each of the N runs crash on an unrecognised argument
    full_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                          parents=[get_parser(add_help=False)], conflict_handler='resolve')
    add_launcher_args(full_parser)
    hyp = full_parser.parse_args()

    # Split off launcher-only args; everything else is forwarded verbatim to avg_ablation
    args, ablation_args = add_launcher_args(argparse.ArgumentParser(add_help=False)).parse_known_args()
    if '--seed' in ablation_args:
        full_parser.error("--seed is set per run; use --seeds or --start_seed instead")
    if '--description' not in ablation_args:
        ablation_args += ['--description', args.name]

    hyp = vars(hyp)
    for k in ['n_runs', 'n_workers', 'start_seed', 'seeds', 'results_dir', 'name', 'seed']:
        hyp.pop(k)
    hyp['description'] = ablation_args[ablation_args.index('--description') + 1]
    args.hyperparams = hyp

    if args.seeds is None:
        args.seeds = list(range(args.start_seed, args.start_seed + args.n_runs))
    args.n_workers = max(1, min(args.n_workers, len(args.seeds)))
    return args, ablation_args


def launch(seed, ablation_args, run_dir):
    cmd = [sys.executable, "-m", "incremental_rl.avg_ablation",
           "--seed", str(seed), "--results_dir", run_dir, *ablation_args]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"   # Avoid N processes each grabbing every core
    env["MKL_NUM_THREADS"] = "1"
    env["MPLBACKEND"] = "Agg"      # learning_curve() plots; no GUI windows from workers
    log_f = open(os.path.join(run_dir, "run.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    return proc, log_f


def main():
    tic = time.time()
    args, ablation_args = parse_args()

    # One folder per run: results/<name>-<seed>-<date>/. All runs in a launch share the same date stamp
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dirs = {seed: os.path.join(args.results_dir, f"{args.name}-{seed}-{stamp}") for seed in args.seeds}
    for seed, run_dir in run_dirs.items():
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "multi_experiment.json"), "w") as f:
            json.dump({"seed": seed, "all_seeds": args.seeds, "n_workers": args.n_workers,
                       "ablation_args": ablation_args, "hyperparams": args.hyperparams}, f, indent=2)

    print(f"Launching {len(args.seeds)} runs ({args.n_workers} at a time) -> "
          f"{os.path.join(args.results_dir, f'{args.name}-<seed>-{stamp}')}")
    print("Hyperparameters:")
    for k, v in args.hyperparams.items():
        print(f"  {k}: {v}")

    pending = list(args.seeds)
    running = {}    # seed -> (proc, log_file, start_time)
    status = {}     # seed -> return code
    try:
        while pending or running:
            while pending and len(running) < args.n_workers:
                seed = pending.pop(0)
                proc, log_f = launch(seed, ablation_args, run_dirs[seed])
                running[seed] = (proc, log_f, time.time())
                print(f"[start] seed {seed} (pid {proc.pid})")

            for seed, (proc, log_f, t0) in list(running.items()):
                if proc.poll() is not None:
                    log_f.close()
                    status[seed] = proc.returncode
                    del running[seed]
                    tag = "done" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
                    print(f"[{tag}] seed {seed} in {time.time()-t0:.1f}s "
                          f"({len(status)}/{len(args.seeds)} finished)")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted, terminating running workers...")
        for proc, log_f, _ in running.values():
            proc.terminate()
            proc.wait()
            log_f.close()
        raise

    summarize(run_dirs, status)
    print("All runs finished in {:.1f}s".format(time.time() - tic))


def summarize(run_dirs, status):
    failed = [s for s, rc in status.items() if rc != 0]
    if failed:
        print(f"Failed seeds: {failed} (see run.log in each run folder)")

    # Per-run returns files are written by save_returns() as a (2, n_episodes) array: [ep_lens; rets]
    final_rets = {}
    for seed, run_dir in run_dirs.items():
        for fname in os.listdir(run_dir):
            if fname.endswith("_returns.txt"):
                try:
                    data = np.loadtxt(os.path.join(run_dir, fname)).reshape(2, -1)
                    final_rets[seed] = float(np.mean(data[1, -10:]))
                except Exception as e:
                    print(f"Could not parse {fname}: {e}")
    if final_rets:
        vals = np.array(list(final_rets.values()))
        print(f"Mean of last-10-episode returns over {len(vals)} runs: {vals.mean():.2f} +- {vals.std():.2f}")


if __name__ == "__main__":
    main()
