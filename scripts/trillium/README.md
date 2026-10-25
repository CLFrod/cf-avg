# Running on Trillium (SciNet)

Trillium CPU jobs always get a **whole node** (192 cores, ~755 GiB), max walltime **24 h**. `$HOME` is
**read-only** on compute nodes and there is **no internet**, so code + venv live in `$HOME/cf-avg`, and every
run writes to `$SCRATCH`. AVG does one tiny update per step, so a run cannot use more than one core: the job
runs **188 runs at once, one pinned core each** (4 cores reserved for the launcher/OS).

| file | purpose |
|---|---|
| `setup_env.sh` | one-time environment build on a login node |
| `smoke_test.sh` | ~1 min login-node check of the whole pipeline (rendering, config, runs, resume) |
| `submit.sh` | submit the job (from anywhere; puts the job directory on scratch) |
| `avg_multi.sh` | the Slurm job script: loads modules, checks rendering, launches the experiment |

## 1. Setup (login node, once)

```bash
cd $HOME && git clone -b optimize_logging https://github.com/CLFrod/cf-avg.git && cd cf-avg
bash scripts/trillium/setup_env.sh
```

Installs `uv` if needed, loads `StdEnv/2023 python/3.13`, and installs the exact versions from `uv.lock` into a
venv on the module's interpreter. The module's Python is required (rather than uv's own standalone build) because
off-screen video rendering loads `libOSMesa` from the Gentoo layer, which only works inside a Python built in that
layer. Rerun after `git pull` whenever `uv.lock` changed.

## 2. Smoke test (login node, ~1 min, 3 cores)

```bash
bash scripts/trillium/smoke_test.sh
```

Checks the interpreter, OSMesa rendering, that the real config resolves (every env/mod validated), runs one tiny
run per env (Hopper / Ant / Humanoid) through a physics + termination switch with videos and checkpoints into
`$SCRATCH/results/smoke_<time>/`, then resumes one of them. Ends with `ALL OK` and per-run steps/s.

## 3. Submit

```bash
bash scripts/trillium/submit.sh        # runs configs/trillium_dynamics_3env.json
```

`submit.sh` is `sbatch --chdir=$SCRATCH/avg_jobs avg_multi.sh`, so Slurm's log lands in
`$SCRATCH/avg_jobs/avg_multi_<jobid>.out` wherever you run it from. Right after, queue the continuation job
behind it (it exits in seconds if nothing needs resuming):

```bash
bash scripts/trillium/submit.sh --dependency=afterany:<jobid> --export=ALL,RESUME_FROM="$SCRATCH/results/*-2026*"
```

Other experiments: `--export=ALL,CONFIG=<file>`; the options are listed in the header of `avg_multi.sh`
(`SHARD=1/2`, `2/2` for a config larger than one node, `TIME_LIMIT`, `RESULTS_DIR`, …).

## 4. Monitor

```bash
squeue -u $USER                                        # PD waiting, R running
head -70 $SCRATCH/avg_jobs/avg_multi_<jobid>.out       # python path, "OSMesa render OK", the plan, [start] lines
jobperf <jobid>                                        # ~186 busy cores
grep -h '"step"' $SCRATCH/results/*/checkpoint.json | sort | uniq -c   # progress of every run
```

Each run folder `$SCRATCH/results/<name>-<env>-<seed>-<date>/` holds `metrics.csv` (one row per episode,
flushed as written), `run.log`, `checkpoint.pt/json`, the learning-curve PNG, and `<run_id>_video_<step>.mp4`.
`step / wall_time` in `checkpoint.json` is the run's real throughput.

## 5. Resume

Runs stop cleanly at `TIME_LIMIT` hours (default 23.5) with a checkpoint (networks, optimizers, TD-error scaler,
observation normaliser, RNG states), and a resumed run is bit-identical to an uninterrupted one. The continuation
job above skips complete runs and continues the rest, appending to the same `metrics.csv`. Arguments given to the
resume job override the stored ones, so a finished run can also be extended (`RUN_ARGS="--N 20000000"`).

## 6. Plot

```bash
# mean +- std over seeds, one panel per metric; `label=glob` groups overlay
python -m incremental_rl.plot_runs "base=$SCRATCH/results/base-Ant-v4-*" "moon=$SCRATCH/results/moon-Ant-v4-*" \
    --vlines 6000000 --out ant_moon.png
```

Works on partial data while a job runs. The `steps_per_sec` panel shows whether per-step speed holds over a run.

## Storage

| | `$HOME` | `$SCRATCH` | `$PROJECT` |
|---|---|---|---|
| use | code + venv | all job output | finished results worth keeping |
| on compute nodes | read-only | read/write | read-only |
| backed up | yes | **no** (purgeable) | yes, shared with the group |

Copy finished run folders (minus `checkpoint.pt` if you like) to `$PROJECT` when a batch is done.

## Notes

- Videos: `n_eval` episodes every `video_every` steps, before each physics switch, and whenever training ceases.
  They render off-screen on the CPU via OSMesa (`MUJOCO_GL=osmesa`); the job log's first lines say
  `OSMesa render OK` or fall back to `MUJOCO_GL=disable` (training unaffected, no videos). ~20–60 s per video.
- wandb: leave `--wandb_mode disabled`, or use `offline` and `wandb sync` from a login node.
- `requirements.txt` (pinned from `uv.lock`) is the pure-pip equivalent of `setup_env.sh` if ever needed.
