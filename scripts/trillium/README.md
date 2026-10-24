# Running on Trillium (SciNet)

Trillium CPU jobs always get a **whole node** (192 cores, ~755 GiB), max walltime **24 h**.
`$HOME` is **read-only** on compute nodes and there's **no internet**, so: code + venv live in
`$HOME`, results go to `$SCRATCH`, and jobs are submitted from `$SCRATCH`.

AVG does one tiny update per env step, so a run can't use more than ~1 core. The default layout is
**188 runs × 1 pinned core** (4 cores left for the launcher/OS). Measure before changing this (see below).

## One-time setup (login node)

```bash
cd $HOME && git clone -b optimize_logging https://github.com/CLFrod/cf-avg.git && cd cf-avg
bash scripts/trillium/setup_env.sh
```

The script installs `uv` if needed, loads `StdEnv/2023 python/3.13`, and runs `uv sync --python "$(which python)"`,
i.e. the exact versions from `uv.lock` installed into a venv on the module's interpreter. Rerun it whenever
`uv.lock` changes.

Why the module's Python rather than uv's own: off-screen video rendering loads `libOSMesa` from the Gentoo
layer (`$EBROOTGENTOO/lib64`), and a library from that layer can only be loaded into a Python built in the same
layer (uv's standalone Python uses the system libc and segfaults). uv still pins every package to `uv.lock`.
Both scripts load `StdEnv/2023 python/3.13` before activating `.venv`, since the venv's interpreter is the
module's. The pure-pip equivalent, if ever needed: `pip install -r requirements.txt && pip install -e .` in a
`python -m venv` made from the same module.

## 1a. Smoke test (login node, ~1 min, 3 cores)

```bash
cd $HOME/cf-avg && bash scripts/trillium/smoke_test.sh
```

Checks OSMesa rendering, resolves the real config, runs one tiny run per env (Hopper / Ant / Humanoid) through a
physics + termination switch with videos and checkpoints into `$SCRATCH/results/smoke_<time>/`, then resumes
one of them. Ends with `ALL OK` and the per-run steps/s. Stays inside the login-node limits.

## 1b. Benchmark (interactive, ~15 min of a 60 min debugjob)

```bash
debugjob
cd $SCRATCH && bash $HOME/cf-avg/scripts/trillium/bench.sh
```

This prints steps/sec for 1/2/4/8 threads on a single run, plus per-run speed with the node full.
Use the full-node number to check `--N` fits in 24 h: `hours ≈ N / steps_per_sec / 3600`.

## 2. Submit

```bash
mkdir -p $SCRATCH/avg_jobs && cd $SCRATCH/avg_jobs
sbatch $HOME/cf-avg/scripts/trillium/avg_multi.sbatch      # runs configs/trillium_dynamics_3env.json
```

Other experiments or a single hand-written run are selected with environment variables
(`sbatch --export=ALL,CONFIG=... avg_multi.sbatch`); see the header of the script.

Results: `$SCRATCH/results/<NAME>-<seed>-<date>/`. Launcher output: `$SCRATCH/avg_jobs/avg_multi_<jobid>.out`.

More seeds than one node holds: submit again with `START_SEED=188`, etc. (each job is a separate node).

Several environments, each with its own hyperparameters, in one job: describe them in a config
(`configs/avg_envs.yaml` has the paper's per-env values) and pass it as `CONFIG`:

```bash
sbatch --export=ALL,CONFIG=$HOME/cf-avg/configs/avg_envs.yaml,RUN_ARGS= $HOME/cf-avg/scripts/trillium/avg_multi.sbatch
```

Runs land in `$SCRATCH/results/<experiment name>-<env>-<seed>-<date>/`. Keep the total seed count across
experiments at or below 188 per job, or the extras queue until a core frees up (and must still finish
inside the 24 h walltime).

## Resuming after the 24 h walltime

Every run writes `checkpoint.pt` + `checkpoint.json` in its folder every `--checkpoint` steps (50k, i.e. every
few minutes) and stops cleanly at `TIME_LIMIT` hours (default 23.5) with a final checkpoint. The checkpoint
holds the networks, both Adam states, the TD-error scaler, the observation normaliser and all RNG states, so a
resumed run is bit-identical to one that was never interrupted (physics schedule included). To continue every
run of a job in a new job:

```bash
sbatch --export=ALL,RESUME_FROM="$SCRATCH/results/gravity-*-20261024_*" $HOME/cf-avg/scripts/trillium/avg_multi.sbatch
# or queue it behind the first job right away:
sbatch --dependency=afterany:<first jobid> --export=ALL,RESUME_FROM="..." avg_multi.sbatch
```

Finished runs are skipped (`checkpoint.json` has `"complete": true`), the rest continue from their last
checkpoint, appending to the same `metrics.csv`. Arguments given to the resume job (e.g. `RUN_ARGS="--N 20000000"`)
override the stored ones, so a finished run can also be extended. Check progress of a batch with
`grep -h '"step"' $SCRATCH/results/gravity-*/checkpoint.json`.

## 3. Plot

Every run writes `<run_id>_metrics.csv`: one row per episode (`step`, `return`, `ep_len`, `wall_time`, episode
means of the losses / TD error / entropy / |action|, sampled grad norms, weight norms). Rows are flushed as
they are written, so a run killed at the walltime limit keeps everything up to that point.

```bash
# mean +- std over all seeds of one experiment, one panel per metric
python -m incremental_rl.plot_runs "$SCRATCH/results/moon_earth-*" --vlines 3000000 6000000 9000000 --out moon_earth.png
# overlay two experiments, chosen metrics only
python -m incremental_rl.plot_runs "moon=$SCRATCH/results/moon_earth-*" "base=$SCRATCH/results/baseline-*" \
    --metrics return critic_loss entropy --out compare.png
```

`wall_time` vs `step` in the CSV gives the real steps/sec of a full node: use it to set `--N` for the next job.

## Monitoring

```bash
squeue -u $USER
jobperf <JOBID>        # CPU usage should be ~188 cores busy
```

## Notes

- Videos (`n_eval` episodes every `video_every` steps, at each physics switch and whenever training ceases, as
  `<run_id>_video_<step>.mp4` in the run folder) render off-screen on the CPU through OSMesa: the job script
  loads `StdEnv/2023` (which provides `libOSMesa`), sets `MUJOCO_GL=osmesa`, and prints `OSMesa render OK` /
  `FAILED` at the top of the Slurm log. On failure it falls back to `MUJOCO_GL=disable`: training is unaffected,
  the runs just log `[video] recording failed ...` and produce no videos.
  Cost: software rendering is ~20–60 s per 2-episode video, a few minutes per 12M-step run.
- wandb: leave `--wandb_mode disabled`, or use `offline` and `wandb sync` from a login node.
- `pyproject.toml` pulls CUDA builds of torch on Linux. They work on CPU nodes, just a larger download.
