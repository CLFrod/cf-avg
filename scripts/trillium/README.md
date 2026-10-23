# Running on Trillium (SciNet)

Trillium CPU jobs always get a **whole node** (192 cores, ~755 GiB), max walltime **24 h**.
`$HOME` is **read-only** on compute nodes and there's **no internet**, so: code + venv live in
`$HOME`, results go to `$SCRATCH`, and jobs are submitted from `$SCRATCH`.

AVG does one tiny update per env step, so a run can't use more than ~1 core. The default layout is
**188 runs × 1 pinned core** (4 cores left for the launcher/OS). Measure before changing this (see below).

## One-time setup (login node)

```bash
cd $HOME && git clone https://github.com/CLFrod/cf-avg.git && cd cf-avg
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv isn't already available
uv sync                                           # builds .venv (Python 3.13 + deps) on the login node
```

## 1. Benchmark (interactive, ~15 min of a 60 min debugjob)

```bash
debugjob
cd $SCRATCH && bash $HOME/cf-avg/scripts/trillium/bench.sh
```

This prints steps/sec for 1/2/4/8 threads on a single run, plus per-run speed with the node full.
Use the full-node number to check `--N` fits in 24 h: `hours ≈ N / steps_per_sec / 3600`.

## 2. Submit

```bash
mkdir -p $SCRATCH/avg_jobs && cd $SCRATCH/avg_jobs
sbatch --export=ALL,NAME=moon_earth,RUN_ARGS="--env Hopper-v4 --N 12000000 --normalize_obs --pnorm --scaled_td" \
    $HOME/cf-avg/scripts/trillium/avg_multi.sbatch
```

Results: `$SCRATCH/results/<NAME>-<seed>-<date>/`. Launcher output: `$SCRATCH/avg_jobs/avg_multi_<jobid>.out`.

More seeds than one node holds: submit again with `START_SEED=188`, etc. (each job is a separate node).

## Monitoring

```bash
squeue -u $USER
jobperf <JOBID>        # CPU usage should be ~188 cores busy
```

## Notes

- `--n_eval > 0` (videos) needs headless OpenGL; the job script sets `MUJOCO_GL=osmesa`, but check OSMesa
  is installed on the nodes first. Otherwise record videos afterwards from saved models (`--save_model`).
- wandb: leave `--wandb_mode disabled`, or use `offline` and `wandb sync` from a login node.
- `pyproject.toml` pulls CUDA builds of torch on Linux. They work on CPU nodes, just a larger download.
