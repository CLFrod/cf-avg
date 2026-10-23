"""
Launch many independent avg_ablation runs (one per seed) in parallel.

Each run is a separate `python -m incremental_rl.avg_ablation` subprocess, so runs are fully
isolated (own env, own torch RNG, own MuJoCo model). Every avg_ablation hyperparameter is
accepted here and forwarded to all runs.

On Linux each run is pinned to its own block of --cores_per_run physical cores, taken from the
CPUs this process is allowed to use (i.e. the Slurm allocation). By default the number of runs
executing at once is however many blocks fit, after setting aside --reserve_cores for the launcher/OS.

Example:
    python -m incremental_rl.avg_multi_experiment --n_runs 10 --name moon_earth \
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
    group.add_argument('--n_workers', default=None, type=int,
                       help="Max runs executing concurrently (default: as many core blocks as are available)")
    group.add_argument('--cores_per_run', default=1, type=int, help="Physical cores reserved (and pinned) per run")
    group.add_argument('--reserve_cores', default=0, type=int, help="Cores left free for the launcher/OS")
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
    for k in ['n_runs', 'n_workers', 'cores_per_run', 'reserve_cores', 'start_seed', 'seeds',
              'results_dir', 'name', 'seed']:
        hyp.pop(k)
    hyp['description'] = ablation_args[ablation_args.index('--description') + 1]
    args.hyperparams = hyp

    if args.seeds is None:
        args.seeds = list(range(args.start_seed, args.start_seed + args.n_runs))
    if hyp['n_threads'] > args.cores_per_run:
        print(f"WARNING: --n_threads {hyp['n_threads']} > --cores_per_run {args.cores_per_run}; runs will oversubscribe")

    # Contiguous blocks keep a run's cores on one L3 cache (8 cores per CCD on Trillium's EPYC 9655).
    # Reserved cores are taken as whole blocks from the start so the remaining blocks stay aligned
    cores = physical_cores()
    cpr = args.cores_per_run
    slots = [cores[i:i + cpr] for i in range(0, len(cores) - cpr + 1, cpr)]
    slots = slots[-(-args.reserve_cores // cpr):]
    if not slots:
        full_parser.error(f"No room for a {cpr}-core run on {len(cores)} cores with {args.reserve_cores} reserved")
    n_workers = args.n_workers or len(slots)
    if n_workers > len(slots):
        print(f"WARNING: {n_workers} workers but only {len(slots)} core blocks; extra runs are not pinned")
    args.n_workers = max(1, min(n_workers, len(args.seeds)))
    args.slots = slots
    return args, ablation_args


def pinning_supported():
    return hasattr(os, "sched_setaffinity")


def physical_cores():
    """ CPUs this process may run on, keeping one logical CPU per physical core (drops SMT siblings) """
    if not pinning_supported():  # Windows/macOS: no pinning, just count
        return list(range(os.cpu_count()))
    cores, seen = [], set()
    for cpu in sorted(os.sched_getaffinity(0)):
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
                siblings = f.read().strip()
        except OSError:
            siblings = str(cpu)
        if siblings not in seen:
            seen.add(siblings)
            cores.append(cpu)
    return cores


def launch(seed, ablation_args, run_dir, n_threads, slot):
    cmd = [sys.executable, "-m", "incremental_rl.avg_ablation",
           "--seed", str(seed), "--results_dir", run_dir, *ablation_args]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(n_threads)   # Avoid N processes each grabbing every core
    env["MKL_NUM_THREADS"] = str(n_threads)
    env["MPLBACKEND"] = "Agg"                 # learning_curve() plots; no GUI windows from workers
    # Pin in the child before exec, so torch/OpenMP threads are created with the right affinity
    pin = (lambda: os.sched_setaffinity(0, slot)) if slot and pinning_supported() else None
    log_f = open(os.path.join(run_dir, "run.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, preexec_fn=pin)
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

    print(f"{len(args.slots)} core blocks of {args.cores_per_run} available, {args.reserve_cores} cores reserved")
    print(f"Launching {len(args.seeds)} runs ({args.n_workers} at a time) -> "
          f"{os.path.join(args.results_dir, f'{args.name}-<seed>-{stamp}')}")
    print("Hyperparameters:")
    for k, v in args.hyperparams.items():
        print(f"  {k}: {v}")

    pending = list(args.seeds)
    running = {}    # seed -> (proc, log_file, start_time, slot)
    status = {}     # seed -> return code
    free_slots = list(args.slots)
    try:
        while pending or running:
            while pending and len(running) < args.n_workers:
                seed = pending.pop(0)
                slot = free_slots.pop(0) if free_slots else None
                proc, log_f = launch(seed, ablation_args, run_dirs[seed], args.hyperparams['n_threads'], slot)
                running[seed] = (proc, log_f, time.time(), slot)
                print(f"[start] seed {seed} (pid {proc.pid}, cores {slot if pinning_supported() else 'unpinned'})")

            for seed, (proc, log_f, t0, slot) in list(running.items()):
                if proc.poll() is not None:
                    log_f.close()
                    if slot:
                        free_slots.append(slot)
                    status[seed] = proc.returncode
                    del running[seed]
                    tag = "done" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
                    print(f"[{tag}] seed {seed} in {time.time()-t0:.1f}s "
                          f"({len(status)}/{len(args.seeds)} finished)")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted, terminating running workers...")
        for proc, log_f, _, _ in running.values():
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
