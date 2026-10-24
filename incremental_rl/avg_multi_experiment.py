"""
Launch many independent avg_ablation runs in parallel: one or more experiments (env + hyperparameters),
each over a set of seeds.

Each run is a separate `python -m incremental_rl.avg_ablation` subprocess, so runs are fully
isolated (own env, own torch RNG, own MuJoCo model). Every avg_ablation hyperparameter is
accepted here and validated up front.

Single experiment from the command line:
    python -m incremental_rl.avg_multi_experiment --n_runs 10 --name moon_earth \\
        --env Hopper-v4 --N 1000000 --actor_lr 0.0003 --alpha_lr 0.1 --normalize_obs --pnorm --scaled_td

Several experiments from a YAML/JSON config (see configs/avg_envs.yaml):
    python -m incremental_rl.avg_multi_experiment --config configs/avg_envs.yaml
    python -m incremental_rl.avg_multi_experiment --config configs/avg_envs.yaml --N 2000 --do_not_save  # smoke test

Config layout: `defaults` (any avg_ablation argument, plus `n_runs`/`seeds`) and a list of `experiments`,
each with a unique `name`, an `env`, and any arguments that differ from the defaults. Precedence, lowest to
highest: avg_ablation defaults < config `defaults` < experiment entry < arguments given on this command line.
`n_runs` is the number of instances (seeds start_seed .. start_seed+n-1); `seeds` can also be an explicit list.

On Linux each run is pinned to its own block of --cores_per_run physical cores, taken from the
CPUs this process is allowed to use (i.e. the Slurm allocation). By default the number of runs
executing at once is however many blocks fit, after setting aside --reserve_cores for the launcher/OS.
Jobs are queued round-robin over experiments, so every experiment makes progress when there are more
runs than cores.

Results: <results_dir>/<experiment name>-<env>-<seed>-<launch time>/ per run.
"""

import os, sys, glob
import time, json
import argparse
import subprocess
import numpy as np
from datetime import datetime
from incremental_rl.avg_ablation import get_parser
from incremental_rl import env_mods, checkpoint


def add_launcher_args(parser):
    group = parser.add_argument_group("multi-experiment")
    group.add_argument(
        "--config",
        default=None,
        type=str,
        help="YAML/JSON file with `defaults` and `experiments`",
    )
    group.add_argument(
        "--n_runs",
        default=10,
        type=int,
        help="Seeds per experiment (unless the experiment sets `seeds`)",
    )
    group.add_argument(
        "--n_workers",
        default=None,
        type=int,
        help="Max runs executing concurrently (default: as many core blocks as are available)",
    )
    group.add_argument(
        "--cores_per_run",
        default=1,
        type=int,
        help="Physical cores reserved (and pinned) per run",
    )
    group.add_argument(
        "--reserve_cores",
        default=0,
        type=int,
        help="Cores left free for the launcher/OS",
    )
    group.add_argument(
        "--start_seed", default=0, type=int, help="Run i uses seed start_seed + i"
    )
    group.add_argument(
        "--seeds",
        default=None,
        type=int,
        nargs="+",
        help="Explicit seed list (overrides n_runs/start_seed)",
    )
    group.add_argument(
        "--results_dir",
        default="./results",
        type=str,
        help="Parent dir; each run gets a subfolder",
    )
    group.add_argument(
        "--name",
        default="avg_multi",
        type=str,
        help="Experiment name (command-line mode): run folder prefix and each run's --description",
    )
    group.add_argument(
        "--resume_from", default=None, type=str, nargs="+", metavar="GLOB",
        help="Continue the runs in these folders from their checkpoints (e.g. 'results/gravity-*-20261024_*')",
    )
    group.add_argument(
        "--dry_run", action="store_true", help="Print the resolved experiments and exit"
    )
    return parser


def parse_args():
    # Full parser (launcher + every avg_ablation hyperparameter) for --help and to reject typos up front,
    # rather than having each of the N runs crash on an unrecognised argument
    full_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[get_parser(add_help=False)],
        conflict_handler="resolve",
    )
    add_launcher_args(full_parser)
    full_parser.parse_args()

    # Split off launcher-only args; everything else overrides the config / is forwarded to avg_ablation
    args, cli_overrides = add_launcher_args(
        argparse.ArgumentParser(add_help=False)
    ).parse_known_args()
    if "--seed" in cli_overrides:
        full_parser.error(
            "--seed is set per run; use --seeds / --start_seed, or `seeds` in the config"
        )

    if args.resume_from:
        if args.config:
            full_parser.error("--resume_from and --config are exclusive")
        args.jobs = resume_jobs(args.resume_from, cli_overrides, full_parser)
    else:
        if args.config:
            experiments = load_experiments(args.config)
        else:
            experiments = [{"name": args.name}]
        if "--env" in cli_overrides and args.config:
            print(
                "WARNING: --env on the command line overrides the env of every experiment in the config"
            )
        args.jobs = build_jobs(experiments, args, cli_overrides, full_parser)
    n_threads = max(j["hyp"]["n_threads"] for j in args.jobs)
    if n_threads > args.cores_per_run:
        print(
            f"WARNING: --n_threads {n_threads} > --cores_per_run {args.cores_per_run}; runs will oversubscribe"
        )

    # Contiguous blocks keep a run's cores on one L3 cache (8 cores per CCD on Trillium's EPYC 9655).
    # Reserved cores are taken as whole blocks from the start so the remaining blocks stay aligned
    cores = physical_cores()
    cpr = args.cores_per_run
    slots = [cores[i : i + cpr] for i in range(0, len(cores) - cpr + 1, cpr)]
    slots = slots[-(-args.reserve_cores // cpr) :]
    if not slots:
        full_parser.error(
            f"No room for a {cpr}-core run on {len(cores)} cores with {args.reserve_cores} reserved"
        )
    n_workers = args.n_workers or len(slots)
    if n_workers > len(slots):
        print(
            f"WARNING: {n_workers} workers but only {len(slots)} core blocks; extra runs are not pinned"
        )
    args.n_workers = max(1, min(n_workers, len(args.jobs)))
    args.slots = slots
    return args


def load_experiments(path):
    """Config file -> list of experiment dicts with `defaults` merged in"""
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            import yaml

            cfg = yaml.safe_load(f)
        else:
            cfg = json.load(f)
    defaults = cfg.get("defaults") or {}
    experiments = cfg.get("experiments") or []
    if not experiments:
        raise SystemExit(f"{path}: no `experiments` listed")
    merged, names = [], set()
    for i, exp in enumerate(experiments):
        exp = {**defaults, **exp}
        name = exp.get("name") or exp.get("env")
        if not exp.get("env"):
            raise SystemExit(f"{path}: experiment #{i + 1} ({name!r}) has no `env`")
        if (name, exp["env"]) in names:
            raise SystemExit(f"{path}: duplicate experiment {name!r} on {exp['env']}")
        names.add((name, exp["env"]))
        exp["name"] = name
        merged.append(exp)
    return merged


JSON_KEYS = ("env_kwargs", "env_mods", "schedule")   # structured values: passed to avg_ablation as JSON strings


def hyp_to_argv(hyp):
    """{actor_lr: 1e-5, pnorm: True, vlines: [1, 2]} -> ['--actor_lr', '1e-05', '--pnorm', '--vlines', '1', '2']"""
    argv = []
    for k, v in hyp.items():
        if v is None or v is False:
            continue
        argv.append(f"--{k}")
        if k in JSON_KEYS:
            argv.append(json.dumps(v))
        elif v is not True:
            argv += [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]
    return argv


def resolve_seeds(exp, args):
    seeds = exp.get(
        "seeds", exp.get("n_runs")
    )  # n_runs: count only; seeds: count or explicit list
    start = exp.get("start_seed", args.start_seed)
    if isinstance(seeds, int):
        return list(range(start, start + seeds))
    if seeds:
        return [int(s) for s in seeds]
    if args.seeds:
        return list(args.seeds)
    return list(range(args.start_seed, args.start_seed + args.n_runs))


def build_jobs(experiments, args, cli_overrides, full_parser):
    """One job per (experiment, seed), each with a validated avg_ablation argv and the resolved hyperparameters"""
    validator = get_parser(add_help=False)
    validator.exit_on_error = False
    jobs_per_exp = []
    for exp in experiments:
        name = exp["name"]
        hyp = {
            k: v
            for k, v in exp.items()
            if k not in ("name", "seeds", "n_runs", "start_seed")
        }
        if "seed" in hyp:
            full_parser.error(f"experiment {name!r}: use `seeds`, not `seed`")
        argv = (
            hyp_to_argv(hyp) + cli_overrides
        )  # later options win, so the command line overrides
        if "--description" not in argv:
            argv += ["--description", name]
        try:
            ns, unknown = validator.parse_known_args(argv)
        except argparse.ArgumentError as e:
            full_parser.error(f"experiment {name!r}: {e}")
        if unknown:
            full_parser.error(f"experiment {name!r}: unrecognised arguments {unknown}")
        resolved = vars(ns)
        resolved.pop("seed")
        try:  # build the env once and apply every mod, so a bad kwarg/body/geom name fails here, not in 30 runs
            env_mods.validate(resolved["env"], resolved["env_kwargs"], resolved["env_mods"], resolved["schedule"])
        except Exception as e:
            full_parser.exit(2, f"error: experiment {name!r} ({resolved['env']}): {type(e).__name__}: {e}\n")
        label = f"{name}/{resolved['env']}"  # what the progress lines and summary call this experiment
        jobs_per_exp.append(
            [
                {"name": name, "label": label, "seed": s, "argv": argv, "hyp": resolved}
                for s in resolve_seeds(exp, args)
            ]
        )

    # Round-robin over experiments so each one progresses when runs outnumber cores
    jobs = []
    for i in range(max(len(j) for j in jobs_per_exp)):
        jobs += [j[i] for j in jobs_per_exp if i < len(j)]
    return jobs


def resume_jobs(patterns, cli_overrides, full_parser):
    """Jobs that continue existing run folders (launched by this script) from their checkpoints"""
    validator = get_parser(add_help=False)
    validator.exit_on_error = False
    dirs = sorted(d for p in patterns for d in glob.glob(p) if os.path.isdir(d))
    if not dirs:
        full_parser.error(f"--resume_from: no run folders match {patterns}")
    jobs, skipped = [], []
    for d in dirs:
        try:
            with open(os.path.join(d, "multi_experiment.json")) as f:
                info = json.load(f)
        except FileNotFoundError:
            skipped.append((d, "no multi_experiment.json (not launched by this script)"))
            continue
        meta = checkpoint.read_meta(d)
        if meta is None:
            skipped.append((d, "no checkpoint"))
            continue
        argv = list(info["ablation_args"]) + cli_overrides + ["--resume", d]
        try:
            ns, unknown = validator.parse_known_args(argv)
        except argparse.ArgumentError as e:
            full_parser.error(f"{d}: {e}")
        if unknown:
            full_parser.error(f"{d}: unrecognised arguments {unknown}")
        if meta["step"] >= ns.N:
            skipped.append((d, f"complete ({meta['step']} steps)"))
            continue
        jobs.append({"name": info["experiment"], "label": f"{info['experiment']}/{ns.env}", "seed": info["seed"],
                     "argv": argv, "hyp": vars(ns), "run_dir": d, "resume_step": meta["step"]})
    for d, why in skipped:
        print(f"[skip] {d}: {why}")
    if not jobs:
        full_parser.exit(0, "nothing to resume\n")
    return jobs


def pinning_supported():
    return hasattr(os, "sched_setaffinity")


def physical_cores():
    """CPUs this process may run on, keeping one logical CPU per physical core (drops SMT siblings)"""
    if not hasattr(os, "sched_setaffinity"):  # Windows/macOS: no pinning, just count
        return list(range(os.cpu_count()))
    cores, seen = [], set()
    for cpu in sorted(os.sched_getaffinity(0)):
        try:
            with open(
                f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            ) as f:
                siblings = f.read().strip()
        except OSError:
            siblings = str(cpu)
        if siblings not in seen:
            seen.add(siblings)
            cores.append(cpu)
    return cores


def launch(job, slot):
    cmd = [
        sys.executable,
        "-m",
        "incremental_rl.avg_ablation",
        "--seed",
        str(job["seed"]),
        "--results_dir",
        job["run_dir"],
        *job["argv"],
    ]
    n_threads = job["hyp"]["n_threads"]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(
        n_threads
    )  # Avoid N processes each grabbing every core
    env["MKL_NUM_THREADS"] = str(n_threads)
    env["MPLBACKEND"] = "Agg"  # learning_curve() plots; no GUI windows from workers
    # Pin in the child before exec, so torch/OpenMP threads are created with the right affinity
    pin = (
        (lambda: os.sched_setaffinity(0, slot))
        if slot and pinning_supported()
        else None
    )
    log_f = open(os.path.join(job["run_dir"], "run.log"), "a" if "resume_step" in job else "w")
    proc = subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, preexec_fn=pin
    )
    return proc, log_f


def describe(jobs):
    """Per experiment: env, seeds, and the hyperparameters that differ from avg_ablation's defaults"""
    defaults = vars(get_parser(add_help=False).parse_args([]))
    by_exp = {}
    for j in jobs:
        by_exp.setdefault(j["label"], []).append(j)
    for label, js in by_exp.items():
        hyp = js[0]["hyp"]
        diff = {
            k: v
            for k, v in hyp.items()
            if k not in ("description", "env", "resume") and defaults.get(k) != v
        }
        seeds = [j["seed"] for j in js]
        seed_str = (
            f"{seeds[0]}..{seeds[-1]}"
            if seeds == list(range(seeds[0], seeds[-1] + 1))
            else str(seeds)
        )
        resume = [j["resume_step"] for j in js if "resume_step" in j]
        print(
            f"  {label}: seeds={seed_str} ({len(seeds)})  "
            + (f"resume from steps {min(resume)}..{max(resume)}  " if resume else "")
            + "  ".join(f"{k}={v}" for k, v in diff.items())
        )


def main():
    tic = time.time()
    sys.stdout.reconfigure(
        line_buffering=True
    )  # progress lines show up promptly when stdout is a file (sbatch, nohup)
    args = parse_args()
    if args.dry_run:
        print(
            f"{len(args.slots)} core blocks of {args.cores_per_run} available; "
            f"{len(args.jobs)} runs would launch, {args.n_workers} at a time:"
        )
        describe(args.jobs)
        return

    # One folder per run: results/<experiment>-<env>-<seed>-<date>/. All runs in a launch share the same date stamp
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for job in args.jobs:
        if "resume_step" in job:  # continuing an existing folder: keep its files
            continue
        job["run_dir"] = os.path.join(
            args.results_dir, f"{job['name']}-{job['hyp']['env']}-{job['seed']}-{stamp}"
        )
        os.makedirs(job["run_dir"], exist_ok=True)
        with open(os.path.join(job["run_dir"], "multi_experiment.json"), "w") as f:
            json.dump(
                {
                    "experiment": job["name"],
                    "seed": job["seed"],
                    "config": args.config,
                    "n_workers": args.n_workers,
                    "ablation_args": job["argv"],
                    "hyperparams": job["hyp"],
                },
                f,
                indent=2,
                default=str,
            )

    print(
        f"{len(args.slots)} core blocks of {args.cores_per_run} available, {args.reserve_cores} cores reserved"
    )
    print(
        f"Launching {len(args.jobs)} runs ({args.n_workers} at a time) -> "
        f"{os.path.join(args.results_dir, f'<experiment>-<env>-<seed>-{stamp}')}"
    )
    describe(args.jobs)

    pending = list(args.jobs)
    running = {}  # (label, seed) -> (job, proc, log_file, start_time, slot)
    status = {}  # (label, seed) -> return code
    free_slots = list(args.slots)
    try:
        while pending or running:
            while pending and len(running) < args.n_workers:
                job = pending.pop(0)
                slot = free_slots.pop(0) if free_slots else None
                proc, log_f = launch(job, slot)
                running[(job["label"], job["seed"])] = (
                    job,
                    proc,
                    log_f,
                    time.time(),
                    slot,
                )
                print(
                    f"[start] {job['label']} seed {job['seed']} (pid {proc.pid}, "
                    f"cores {slot if pinning_supported() else 'unpinned'})"
                )

            for key, (job, proc, log_f, t0, slot) in list(running.items()):
                if proc.poll() is not None:
                    log_f.close()
                    if slot:
                        free_slots.append(slot)
                    status[key] = proc.returncode
                    del running[key]
                    tag = (
                        "done"
                        if proc.returncode == 0
                        else f"FAILED rc={proc.returncode}"
                    )
                    print(
                        f"[{tag}] {job['label']} seed {job['seed']} in {time.time()-t0:.1f}s "
                        f"({len(status)}/{len(args.jobs)} finished)"
                    )
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted, terminating running workers...")
        for _, proc, log_f, _, _ in running.values():
            proc.terminate()
            proc.wait()
            log_f.close()
        raise

    summarize(args.jobs, status)
    print("All runs finished in {:.1f}s".format(time.time() - tic))


def summarize(jobs, status):
    failed = [k for k, rc in status.items() if rc != 0]
    if failed:
        print(f"Failed runs: {failed} (see run.log in each run folder)")

    # Per-run returns files are written by save_returns() as a (2, n_episodes) array: [ep_lens; rets]
    final_rets = {}
    for job in jobs:
        for fname in os.listdir(job["run_dir"]):
            if fname.endswith("_returns.txt"):
                try:
                    data = np.loadtxt(os.path.join(job["run_dir"], fname)).reshape(
                        2, -1
                    )
                    final_rets.setdefault(job["label"], []).append(
                        float(np.mean(data[1, -10:]))
                    )
                except Exception as e:
                    print(f"Could not parse {fname}: {e}")
    for name, vals in final_rets.items():
        vals = np.array(vals)
        print(
            f"{name}: mean of last-10-episode returns over {len(vals)} runs: {vals.mean():.2f} +- {vals.std():.2f}"
        )


if __name__ == "__main__":
    main()
