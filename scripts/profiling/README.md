# Compute-utilisation profiling (throwaway branch `profiling-throwaway`)

Purpose: measure, as raw CSV, how much of a Trillium node (2 × AMD EPYC 9655, 192 cores, ~755 GiB) the
`avg_multi_experiment` job actually uses over time, so the efficiency of the "188 runs, one pinned core each"
setup can be shown with graphs made afterwards. Nothing here changes how the runs train.

| file | what it does |
|---|---|
| `node_monitor.py` | background sampler: node-wide CPU / memory, per-CPU busy %, per-run process usage -> CSV. Reads `/proc` and `/sys` only (no extra packages, no venv rebuild). |
| `collect_runs.py` | after the job: each run's steps/s over time from its `metrics.csv`, on the same clock as the monitor -> CSV |
| `../trillium/avg_multi.sh` | (patched on this branch) starts the monitor before the launcher, stops it after, then runs `collect_runs.py` |

## Running it

Everything is on by default in the job script of this branch. On Trillium:

```bash
cd $HOME/cf-avg && git fetch && git checkout profiling-throwaway        # no uv.lock change: no setup_env.sh needed
```

**10-second check on a login node first** (confirms `/proc`, `/sys` and the cgroup files read correctly there):

```bash
source .venv/bin/activate   # or any python3
python scripts/profiling/node_monitor.py --out $SCRATCH/monitor_test --interval 1 --duration 10 && head -3 $SCRATCH/monitor_test/node.csv
```

**Baseline, 3 runs alone on a node** (one per env; shows single-run steps/s with no neighbours, ~30-40 min):

```bash
bash scripts/trillium/submit.sh --time=01:00:00 \
    --export=ALL,CONFIG=$HOME/cf-avg/configs/trillium_jobtest.json,RUN_ARGS="--N 500000"
```

**Full node, the real config** (188 runs at once). Either the real 24 h job as usual (the monitor adds no
measurable load: one niced process on a reserved core, a few ms of `/proc` reads per tick), or a shortened one
so the whole life cycle (start-up, steady state, runs finishing at different rates, tail) fits in a few hours:

```bash
bash scripts/trillium/submit.sh                                           # the normal 24 h job, now profiled
bash scripts/trillium/submit.sh --time=03:00:00 --export=ALL,RUN_ARGS="--N 1500000"   # ~2 h version
```

Knobs (all `--export=ALL,...`): `MONITOR=0` off, `MONITOR_INTERVAL` (default 5 s), `MONITOR_PROC_INTERVAL`
(default 30 s), `MONITOR_DIR` (default `$SCRATCH/results/monitor_<jobid>`).

While the job runs, the CSVs are readable (every row is flushed):

```bash
tail -2 $SCRATCH/results/monitor_<jobid>/node.csv | cut -d, -f1-16        # busy %, cores in use, per-socket
jobperf <jobid>                                                            # Slurm's own view, for comparison
```

`collect_runs.py` runs automatically at the end of the job. Running it by hand (e.g. after a job that hit the
walltime, or for the per-episode file):

```bash
python scripts/profiling/collect_runs.py --monitor_dir $SCRATCH/results/monitor_<jobid> --results_dir $SCRATCH/results
python scripts/profiling/collect_runs.py "$SCRATCH/results/base-*-2026*" --out some_dir   # explicit folders
```

Copy the `monitor_<jobid>` folder off the cluster for plotting (for a 24 h job: `node.csv` ~5 MB, `cpus.csv`
~15 MB, `procs.csv` ~70 MB; the optional per-episode `runs.csv` is larger).

## Output files

All times: `t_unix` (epoch seconds), `t_local` (naive local time; the run folders, run ids and `checkpoint.json`
use the same clock), `elapsed_s` (since the monitor started, i.e. roughly job start). Percentages are of the
sampling interval; "busy" = 100 − idle − iowait.

### `node_info.json` (static)
Host, kernel, Slurm variables, CPU model string, logical CPUs, physical cores, sockets, cores per socket, threads
per core, number and size of L3 groups (CCDs: 8 cores each on the EPYC 9655), NUMA nodes, total memory, the CPUs
Slurm allowed the job, the cgroup used for accounting, the monitor's own settings and the core it is pinned to.
This is the evidence for "2 × EPYC 9655, 192 cores, whole node".

### `cpu_topology.csv` (static, one row per logical CPU)
`cpu, socket, core_id, l3_group, numa_node, smt_siblings, is_smt_sibling, allowed`. Join on `cpu` to group
`cpus.csv` columns by socket / CCD / NUMA node.

### `node.csv` (one row per `MONITOR_INTERVAL`)
| columns | meaning |
|---|---|
| `interval_s` | actual seconds covered by this row |
| `cpu_busy_pct`, `cpu_user_pct`, `cpu_system_pct`, `cpu_iowait_pct`, `cpu_idle_pct`, `cpu_steal_pct` | whole-node CPU time split (`user` includes nice, `system` includes irq/softirq) |
| `cores_in_use` | sum of per-CPU busy fractions = effective number of cores working (max 192) |
| `cores_over_50pct`, `cores_over_90pct` | number of CPUs that were more than 50 % / 90 % busy |
| `allowed_busy_pct` | busy % averaged over the CPUs Slurm gave the job (same as whole-node on a full node) |
| `socket0_busy_pct`, `socket1_busy_pct` | busy % per socket |
| `mem_total_mb`, `mem_used_mb`, `mem_available_mb`, `mem_free_mb`, `mem_cached_mb`, `mem_buffers_mb`, `mem_dirty_mb`, `mem_shmem_mb`, `swap_used_mb` | `/proc/meminfo`; `used` = total − available |
| `load1`, `load5`, `load15`, `procs_running`, `procs_blocked` | load averages, runnable and I/O-blocked task counts at the sample |
| `ctxt_per_s`, `intr_per_s` | node-wide context switches and interrupts per second |
| `cg_mem_mb`, `cg_mem_peak_mb`, `cg_cpu_cores` | the job's Slurm cgroup: memory now, peak memory, CPU-seconds per second (= cores consumed by the job as Slurm accounts it); blank if the cgroup files are not readable |
| `n_runs`, `n_launchers`, `runs_cpu_cores`, `runs_rss_mb`, `runs_hwm_mb`, `runs_threads` | filled on the ticks where processes are sampled: live run processes, their summed CPU (in cores), resident memory, peak resident memory, threads |

### `cpus.csv` (one row per `MONITOR_INTERVAL`)
`t_unix, t_local, elapsed_s, cpu0 … cpu191`: busy % of each logical CPU. Heat-map material (CPU × time); with
`cpu_topology.csv` it shows whether both sockets and every CCD are evenly loaded, and that the 4 reserved cores
are the only idle ones.

### `procs.csv` (one row per tracked process per `MONITOR_PROC_INTERVAL`)
| columns | meaning |
|---|---|
| `pid`, `ppid`, `role` | `run` (an `avg_ablation` process), `launcher` (`avg_multi_experiment`) |
| `run`, `env`, `seed` | the run folder name (`<experiment>-<env>-<seed>-<stamp>`), env id, seed from its command line |
| `age_s` | seconds since the process started |
| `cpu_pct`, `user_pct`, `system_pct` | CPU over the interval (100 = one full core); on `first_sample`=1 rows it is the lifetime average instead |
| `rss_mb`, `hwm_mb`, `threads` | resident memory, peak resident memory, thread count |
| `on_cpu`, `allowed_cpus` | the CPU it last ran on and its affinity mask: proves the pinning holds |
| `vol_ctxt_per_s`, `invol_ctxt_per_s` | voluntary / involuntary context switches per second (involuntary ≈ competition for the core) |
| `minflt_per_s`, `majflt` | minor page faults per second, cumulative major faults |
| `read_mb_per_s`, `write_mb_per_s` | storage I/O (metrics.csv, checkpoints, videos) |

### `run_summary.csv`, `runs_binned.csv`, `runs.csv` (from `collect_runs.py`)
- `run_summary.csv`: per run: `experiment, env, seed, run_id, start_local, end_local, steps, N, complete,
  wall_time_s, mean_steps_per_sec, startup_s` (wall time before the first episode ended: imports + env build),
  `episodes, resumed`.
- `runs_binned.csv`: per run per `--bin` seconds (default 60) of run wall time: `step` reached and `steps_per_sec`
  in that bin, with `t_local`/`t_unix` of the bin end. Overlay on `node.csv` to show throughput holding while
  the node is full, and compare against the baseline job for per-run slowdown from sharing the node.
- `runs.csv` (only when run by hand without `--no_episodes`): every episode of every run with steps/s since the
  previous episode and cumulative.

## Efficiency questions the data answers

- **Core utilisation**: `cores_in_use` vs 188 and `cores_over_90pct` over time; `cpus.csv` for which cores idle.
- **No oversubscription / thread leakage**: every run at `cpu_pct` ≈ 100 with `threads` small, `invol_ctxt_per_s`
  low, `allowed_cpus` a single core, `cpu_system_pct` and `cpu_iowait_pct` near zero.
- **Memory**: `mem_used_mb` and `cg_mem_mb` vs `mem_total_mb` (expected: a few hundred MB per run, i.e. the node's
  memory is mostly spare, which is not waste on a CPU-bound job but worth stating); `swap_used_mb` = 0.
- **Per-run slowdown from sharing**: `runs_binned.csv` steps/s in the full job vs the 3-run baseline job, per env.
  A small gap means memory bandwidth / L3 sharing is not a bottleneck at 188 runs.
- **Tail**: `n_runs` and `cores_in_use` as runs finish (Hopper first, Humanoid last) — the only idle time in the job.
