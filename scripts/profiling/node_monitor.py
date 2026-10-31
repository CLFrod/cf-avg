#!/usr/bin/env python3
"""
Sample whole-node CPU and memory utilisation, plus the per-process usage of the AVG runs, into CSV files.
Linux only (reads /proc and /sys); no third-party packages, so it runs in any Python on the cluster.

    python scripts/profiling/node_monitor.py --out $SCRATCH/results/monitor_$SLURM_JOB_ID \
        --interval 5 --proc_interval 30 --pin auto --watch_pid $$

Every row is flushed as it is written, so the files can be read (and plotted) while the job is still running.
Files in --out:
  node_info.json    static facts: host, Slurm job, CPU model, sockets / cores / threads, L3 groups, NUMA nodes,
                    memory, the CPUs this job may use, the monitor's own settings
  cpu_topology.csv  one row per logical CPU: socket, core id, L3 group, NUMA node, SMT sibling, allowed-by-Slurm
  node.csv          one row per --interval: whole-node CPU split (busy / user / system / iowait / idle / steal %),
                    effective cores in use, cores over 50 % busy, per-socket busy %, memory (used / available /
                    cached / dirty MB), swap, load average, context-switch and interrupt rates, the Slurm cgroup's
                    own accounting (memory now / peak, CPU-cores consumed), and -- on the ticks where processes
                    are sampled -- the number of tracked run / launcher processes and their summed CPU, RSS, threads
  cpus.csv          one row per --interval, one column per logical CPU: busy % of that CPU over the interval
  procs.csv         one row per tracked process per --proc_interval: cpu % (user / system), RSS and peak RSS,
                    threads, the CPU it was last scheduled on, its allowed CPUs, context-switch and page-fault
                    rates, bytes read / written per second, age. Tracked = any process whose command line contains
                    one of the --match strings (default: the avg_ablation runs and the avg_multi_experiment launcher).

Columns are documented once more in scripts/profiling/README.md. All times: `t_unix` (seconds since the epoch,
UTC) and `t_local` (naive local time, the same clock the run folders and run ids are stamped with).
"""
import os, sys, csv, json, time, signal, argparse, socket
from datetime import datetime

PROC = os.environ.get("NODE_MONITOR_PROC_ROOT", "/proc")   # overridable so the parsers can be tested off-cluster
SYS = os.environ.get("NODE_MONITOR_SYS_ROOT", "/sys")
CPU_FIELDS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal", "guest", "guest_nice")
MB = 1024 * 1024


def read(path, default=None):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


def read_int(path, default=None):
    s = read(path)
    if s is None:
        return default
    try:
        return int(s.split()[0])
    except (ValueError, IndexError):
        return default


def parse_cpulist(s):
    """'0-3,8,10-11' -> [0, 1, 2, 3, 8, 10, 11]"""
    out = []
    for part in s.strip().split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


# ----------------------------------------------------------------------------------------------- topology
def topology(cpus):
    """Per logical CPU: socket, core id, L3 group, NUMA node, SMT siblings (from sysfs; -1 where unknown)"""
    numa_of = {}
    node_dir = f"{SYS}/devices/system/node"
    if os.path.isdir(node_dir):
        for d in os.listdir(node_dir):
            if d.startswith("node") and d[4:].isdigit():
                for c in parse_cpulist(read(f"{node_dir}/{d}/cpulist", "")):
                    numa_of[c] = int(d[4:])
    l3_groups, info = {}, {}
    for c in cpus:
        base = f"{SYS}/devices/system/cpu/cpu{c}"
        socket_id = read_int(f"{base}/topology/physical_package_id", -1)
        core_id = read_int(f"{base}/topology/core_id", -1)
        siblings = parse_cpulist(read(f"{base}/topology/thread_siblings_list", str(c)))
        l3 = None
        cache = f"{base}/cache"
        if os.path.isdir(cache):
            for idx in sorted(os.listdir(cache)):
                if read_int(f"{cache}/{idx}/level") == 3:
                    l3 = read(f"{cache}/{idx}/shared_cpu_list", "").strip()
                    break
        l3_id = -1 if not l3 else l3_groups.setdefault(l3, len(l3_groups))
        info[c] = {
            "cpu": c,
            "socket": socket_id,
            "core_id": core_id,
            "l3_group": l3_id,
            "numa_node": numa_of.get(c, -1),
            "smt_siblings": " ".join(str(s) for s in siblings),
            "is_smt_sibling": int(c != min(siblings)),   # 1 for the second hyperthread of a core
        }
    return info


def cpu_model():
    for line in (read(f"{PROC}/cpuinfo", "") or "").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return ""


# ----------------------------------------------------------------------------------------------- /proc/stat
def read_stat():
    """-> (per-cpu {cpu: [ticks...]}, aggregate [ticks...], {ctxt, intr, procs_running, procs_blocked, btime})"""
    per_cpu, agg, extra = {}, None, {}
    for line in (read(f"{PROC}/stat", "") or "").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "cpu":
            agg = [int(x) for x in parts[1:11]]
        elif parts[0].startswith("cpu") and parts[0][3:].isdigit():
            per_cpu[int(parts[0][3:])] = [int(x) for x in parts[1:11]]
        elif parts[0] in ("ctxt", "intr", "procs_running", "procs_blocked", "btime"):
            extra[parts[0]] = int(parts[1])
    return per_cpu, agg, extra


def split_pct(now, before):
    """Two tick vectors -> percentages of the elapsed time per CPU_FIELDS category, plus busy"""
    d = [a - b for a, b in zip(now, before)]
    total = sum(d) or 1
    pct = {k: 100.0 * v / total for k, v in zip(CPU_FIELDS, d)}
    pct["busy"] = 100.0 - pct["idle"] - pct["iowait"]
    return pct


# ----------------------------------------------------------------------------------------------- memory, load
def meminfo():
    out = {}
    for line in (read(f"{PROC}/meminfo", "") or "").splitlines():
        k, _, v = line.partition(":")
        v = v.split()
        if v:
            out[k.strip()] = int(v[0]) * 1024   # kB -> bytes
    return out


def loadavg():
    parts = (read(f"{PROC}/loadavg", "") or "").split()
    if len(parts) < 4:
        return None
    running, _, total = parts[3].partition("/")
    return float(parts[0]), float(parts[1]), float(parts[2]), int(running), int(total)


# ----------------------------------------------------------------------------------------------- cgroup (Slurm job)
def find_cgroup():
    """The cgroup Slurm accounts this job in (the `job_<id>` level if visible, else our own), v2 or v1"""
    lines = (read(f"{PROC}/self/cgroup", "") or "").splitlines()

    def job_level(path):
        p = path
        while p and p != "/":
            if os.path.basename(p).startswith("job_"):
                return p
            p = os.path.dirname(p)
        return path

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":      # cgroup v2 (unified)
            base = f"{SYS}/fs/cgroup" + job_level(parts[2])
            if os.path.exists(f"{base}/memory.current"):
                return {"version": 2, "path": base, "mem": base, "cpu": base}
    mem = cpu = None
    for line in lines:                                                   # cgroup v1 (per controller)
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        ctrls = parts[1].split(",")
        if "memory" in ctrls and mem is None:
            cand = f"{SYS}/fs/cgroup/memory" + job_level(parts[2])
            if os.path.exists(f"{cand}/memory.usage_in_bytes"):
                mem = cand
        if "cpuacct" in ctrls and cpu is None:
            for mount in ("cpu,cpuacct", "cpuacct,cpu", "cpuacct"):
                cand = f"{SYS}/fs/cgroup/{mount}" + job_level(parts[2])
                if os.path.exists(f"{cand}/cpuacct.usage"):
                    cpu = cand
                    break
    if mem or cpu:
        return {"version": 1, "path": mem or cpu, "mem": mem, "cpu": cpu}
    return None


def cgroup_sample(cg):
    """-> (memory bytes, peak memory bytes, cpu seconds used) with None where unavailable"""
    if not cg:
        return None, None, None
    if cg["version"] == 2:
        mem = read_int(f"{cg['mem']}/memory.current")
        peak = read_int(f"{cg['mem']}/memory.peak")
        usage = None
        for line in (read(f"{cg['cpu']}/cpu.stat", "") or "").splitlines():
            if line.startswith("usage_usec"):
                usage = int(line.split()[1]) / 1e6
        return mem, peak, usage
    mem = read_int(f"{cg['mem']}/memory.usage_in_bytes") if cg["mem"] else None
    peak = read_int(f"{cg['mem']}/memory.max_usage_in_bytes") if cg["mem"] else None
    ns = read_int(f"{cg['cpu']}/cpuacct.usage") if cg["cpu"] else None
    return mem, peak, (ns / 1e9 if ns is not None else None)


# ----------------------------------------------------------------------------------------------- processes
def cmdline(pid):
    raw = read(f"{PROC}/{pid}/cmdline", "")
    return [a for a in raw.split("\0") if a] if raw else []


def arg_after(argv, flag, default=""):
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def find_procs(matches, self_pid):
    """{pid: argv} for every process whose command line contains one of the match strings"""
    found = {}
    for d in os.listdir(PROC):
        if not d.isdigit() or int(d) == self_pid:
            continue
        argv = cmdline(d)
        if not argv:
            continue
        joined = " ".join(argv)
        if any(m in joined for m in matches):
            found[int(d)] = argv
    return found


def proc_stat(pid):
    """Fields of /proc/<pid>/stat we use (the comm field may contain spaces, hence the rpartition)"""
    s = read(f"{PROC}/{pid}/stat")
    if not s:
        return None
    _, _, rest = s.rpartition(")")
    f = rest.split()
    # f[0] is field 3 (state) of proc(5); field n of the man page is f[n - 3]
    return {
        "ppid": int(f[1]),
        "minflt": int(f[7]),
        "majflt": int(f[9]),
        "utime": int(f[11]),
        "stime": int(f[12]),
        "threads": int(f[17]),
        "starttime": int(f[19]),      # clock ticks after boot
        "rss_pages": int(f[21]),
        "processor": int(f[36]),
    }


def proc_status(pid):
    out = {}
    for line in (read(f"{PROC}/{pid}/status", "") or "").splitlines():
        k, _, v = line.partition(":")
        v = v.strip()
        if k in ("VmRSS", "VmHWM"):
            out[k] = int(v.split()[0]) * 1024
        elif k == "Threads":
            out[k] = int(v)
        elif k in ("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches"):
            out[k] = int(v)
        elif k == "Cpus_allowed_list":
            out[k] = v
    return out


def proc_io(pid):
    out = {}
    for line in (read(f"{PROC}/{pid}/io", "") or "").splitlines():
        k, _, v = line.partition(":")
        if k in ("read_bytes", "write_bytes"):
            out[k] = int(v)
    return out


def role_of(argv):
    joined = " ".join(argv)
    if "avg_ablation" in joined:
        return "run"
    if "avg_multi_experiment" in joined:
        return "launcher"
    return "other"


# ----------------------------------------------------------------------------------------------- main loop
class Monitor:
    def __init__(self, args):
        self.args = args
        self.pid = os.getpid()
        self.clk = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        self.page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
        self.stop = False
        self.proc_prev = {}     # pid -> previous sample (for rates)
        self.t0 = time.time()
        self.t0_mono = time.monotonic()
        self.cg = find_cgroup()
        os.makedirs(args.out, exist_ok=True)

        per_cpu, agg, extra = read_stat()
        if agg is None:
            sys.exit(f"{PROC}/stat not readable: this monitor needs Linux")
        self.cpus = sorted(per_cpu)
        try:
            self.allowed = sorted(os.sched_getaffinity(0))
        except AttributeError:
            self.allowed = list(self.cpus)
        self.btime = extra.get("btime", 0)
        self.topo = topology(self.cpus)
        self.sockets = sorted({v["socket"] for v in self.topo.values()})
        self.write_static()

        self.f_node = open(os.path.join(args.out, "node.csv"), "w", newline="")
        self.f_cpus = open(os.path.join(args.out, "cpus.csv"), "w", newline="")
        self.f_procs = open(os.path.join(args.out, "procs.csv"), "w", newline="")
        self.w_node = csv.DictWriter(self.f_node, fieldnames=self.node_fields())
        self.w_cpus = csv.writer(self.f_cpus)
        self.w_procs = csv.DictWriter(self.f_procs, fieldnames=self.proc_fields())
        self.w_node.writeheader()
        self.w_cpus.writerow(["t_unix", "t_local", "elapsed_s"] + [f"cpu{c}" for c in self.cpus])
        self.w_procs.writeheader()

        self.prev = {"t": time.monotonic(), "per_cpu": per_cpu, "agg": agg, "extra": extra,
                     "cg": cgroup_sample(self.cg)}

    # -- headers
    def node_fields(self):
        f = ["t_unix", "t_local", "elapsed_s", "interval_s",
             "cpu_busy_pct", "cpu_user_pct", "cpu_system_pct", "cpu_iowait_pct", "cpu_idle_pct", "cpu_steal_pct",
             "cores_in_use", "cores_over_50pct", "cores_over_90pct", "allowed_busy_pct"]
        f += [f"socket{s}_busy_pct" for s in self.sockets]
        f += ["mem_total_mb", "mem_used_mb", "mem_available_mb", "mem_free_mb", "mem_cached_mb", "mem_buffers_mb",
              "mem_dirty_mb", "mem_shmem_mb", "swap_used_mb",
              "load1", "load5", "load15", "procs_running", "procs_blocked", "ctxt_per_s", "intr_per_s",
              "cg_mem_mb", "cg_mem_peak_mb", "cg_cpu_cores",
              "n_runs", "n_launchers", "runs_cpu_cores", "runs_rss_mb", "runs_hwm_mb", "runs_threads"]
        return f

    def proc_fields(self):
        return ["t_unix", "t_local", "elapsed_s", "pid", "ppid", "role", "run", "env", "seed", "age_s",
                "cpu_pct", "user_pct", "system_pct", "rss_mb", "hwm_mb", "threads", "on_cpu", "allowed_cpus",
                "vol_ctxt_per_s", "invol_ctxt_per_s", "minflt_per_s", "majflt", "read_mb_per_s", "write_mb_per_s",
                "first_sample"]

    # -- static files
    def write_static(self):
        topo = self.topo
        cores = {(v["socket"], v["core_id"]) for v in topo.values()}
        per_socket = {s: len({v["core_id"] for v in topo.values() if v["socket"] == s}) for s in self.sockets}
        l3 = {}
        for v in topo.values():
            l3.setdefault(v["l3_group"], []).append(v["cpu"])
        mem = meminfo()
        info = {
            "hostname": socket.gethostname(),
            "kernel": " ".join(os.uname()) if hasattr(os, "uname") else sys.platform,
            "python": sys.version.split()[0],
            "started_unix": self.t0,
            "started_local": datetime.fromtimestamp(self.t0).isoformat(timespec="seconds"),
            "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
            "cpu_model": cpu_model(),
            "logical_cpus": len(self.cpus),
            "physical_cores": len(cores),
            "sockets": len(self.sockets),
            "cores_per_socket": per_socket,
            "threads_per_core": round(len(self.cpus) / max(1, len(cores)), 2),
            "l3_groups": len([k for k in l3 if k >= 0]),
            "cpus_per_l3_group": sorted({len(v) for k, v in l3.items() if k >= 0}),
            "numa_nodes": len({v["numa_node"] for v in topo.values() if v["numa_node"] >= 0}),
            "mem_total_mb": round(mem.get("MemTotal", 0) / MB),
            "allowed_cpus": self.allowed,
            "allowed_cpu_count": len(self.allowed),
            "clk_tck": self.clk,
            "page_size": self.page,
            "cgroup": self.cg,
            "monitor": {"pid": self.pid, "args": vars(self.args),
                        "pinned_to": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None},
        }
        with open(os.path.join(self.args.out, "node_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        with open(os.path.join(self.args.out, "cpu_topology.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["cpu", "socket", "core_id", "l3_group", "numa_node",
                                              "smt_siblings", "is_smt_sibling", "allowed"])
            w.writeheader()
            allowed = set(self.allowed)
            for c in self.cpus:
                w.writerow({**topo[c], "allowed": int(c in allowed)})

    # -- one tick
    def sample(self, with_procs):
        t_mono = time.monotonic()
        t = time.time()
        per_cpu, agg, extra = read_stat()
        dt = t_mono - self.prev["t"]
        if dt <= 0 or agg is None:
            return
        elapsed = t_mono - self.t0_mono
        t_local = datetime.fromtimestamp(t).isoformat(timespec="seconds")

        # CPUs
        busy = {}
        for c in self.cpus:
            if c in per_cpu and c in self.prev["per_cpu"]:
                busy[c] = split_pct(per_cpu[c], self.prev["per_cpu"][c])["busy"]
        whole = split_pct(agg, self.prev["agg"])
        self.w_cpus.writerow([f"{t:.3f}", t_local, f"{elapsed:.1f}"] + [f"{busy.get(c, 0):.1f}" for c in self.cpus])
        self.f_cpus.flush()

        row = {
            "t_unix": f"{t:.3f}", "t_local": t_local, "elapsed_s": f"{elapsed:.1f}", "interval_s": f"{dt:.3f}",
            "cpu_busy_pct": f"{whole['busy']:.2f}",
            "cpu_user_pct": f"{whole['user'] + whole['nice']:.2f}",
            "cpu_system_pct": f"{whole['system'] + whole['irq'] + whole['softirq']:.2f}",
            "cpu_iowait_pct": f"{whole['iowait']:.2f}",
            "cpu_idle_pct": f"{whole['idle']:.2f}",
            "cpu_steal_pct": f"{whole['steal']:.2f}",
            "cores_in_use": f"{sum(busy.values()) / 100:.2f}",
            "cores_over_50pct": sum(1 for b in busy.values() if b > 50),
            "cores_over_90pct": sum(1 for b in busy.values() if b > 90),
        }
        allowed = [busy[c] for c in self.allowed if c in busy]
        row["allowed_busy_pct"] = f"{sum(allowed) / len(allowed):.2f}" if allowed else ""
        for s in self.sockets:
            vals = [busy[c] for c in self.cpus if c in busy and self.topo[c]["socket"] == s]
            row[f"socket{s}_busy_pct"] = f"{sum(vals) / len(vals):.2f}" if vals else ""

        # memory, load, kernel counters
        m = meminfo()
        total, avail = m.get("MemTotal", 0), m.get("MemAvailable", 0)
        row.update({
            "mem_total_mb": round(total / MB), "mem_used_mb": round((total - avail) / MB),
            "mem_available_mb": round(avail / MB), "mem_free_mb": round(m.get("MemFree", 0) / MB),
            "mem_cached_mb": round(m.get("Cached", 0) / MB), "mem_buffers_mb": round(m.get("Buffers", 0) / MB),
            "mem_dirty_mb": round(m.get("Dirty", 0) / MB), "mem_shmem_mb": round(m.get("Shmem", 0) / MB),
            "swap_used_mb": round((m.get("SwapTotal", 0) - m.get("SwapFree", 0)) / MB),
        })
        la = loadavg()
        if la:
            row.update({"load1": la[0], "load5": la[1], "load15": la[2]})
        row["procs_running"] = extra.get("procs_running", "")
        row["procs_blocked"] = extra.get("procs_blocked", "")
        pe = self.prev["extra"]
        if "ctxt" in extra and "ctxt" in pe:
            row["ctxt_per_s"] = f"{(extra['ctxt'] - pe['ctxt']) / dt:.0f}"
        if "intr" in extra and "intr" in pe:
            row["intr_per_s"] = f"{(extra['intr'] - pe['intr']) / dt:.0f}"
        cg = cgroup_sample(self.cg)
        pcg = self.prev["cg"]
        if cg[0] is not None:
            row["cg_mem_mb"] = round(cg[0] / MB)
        if cg[1] is not None:
            row["cg_mem_peak_mb"] = round(cg[1] / MB)
        if cg[2] is not None and pcg[2] is not None:
            row["cg_cpu_cores"] = f"{(cg[2] - pcg[2]) / dt:.2f}"

        if with_procs:
            row.update(self.sample_procs(t, t_local, elapsed))
        self.w_node.writerow(row)
        self.f_node.flush()
        self.prev = {"t": t_mono, "per_cpu": per_cpu, "agg": agg, "extra": extra, "cg": cg}

    def sample_procs(self, t, t_local, elapsed):
        now = time.monotonic()
        procs = find_procs(self.args.match, self.pid)
        agg = {"n_runs": 0, "n_launchers": 0, "cpu": 0.0, "rss": 0, "hwm": 0, "threads": 0}
        seen = set()
        for pid, argv in sorted(procs.items()):
            st = proc_stat(pid)
            if st is None:        # exited between listing and reading
                continue
            status, io = proc_status(pid), proc_io(pid)
            seen.add(pid)
            age = time.time() - (self.btime + st["starttime"] / self.clk) if self.btime else None
            prev = self.proc_prev.get(pid)
            first = prev is None
            if prev is not None:
                dt = now - prev["t"]
                du, ds = st["utime"] - prev["utime"], st["stime"] - prev["stime"]
                dvol = status.get("voluntary_ctxt_switches", 0) - prev["vol"]
                dinvol = status.get("nonvoluntary_ctxt_switches", 0) - prev["invol"]
                dminflt = st["minflt"] - prev["minflt"]
                drd = io.get("read_bytes", 0) - prev["rd"]
                dwr = io.get("write_bytes", 0) - prev["wr"]
            else:                 # first sighting: lifetime averages
                dt = age or 0.0
                du, ds = st["utime"], st["stime"]
                dvol = status.get("voluntary_ctxt_switches", 0)
                dinvol = status.get("nonvoluntary_ctxt_switches", 0)
                dminflt = st["minflt"]
                drd, dwr = io.get("read_bytes", 0), io.get("write_bytes", 0)
            dt = max(dt, 1e-3)
            user_pct, sys_pct = 100.0 * du / self.clk / dt, 100.0 * ds / self.clk / dt
            rss = status.get("VmRSS", st["rss_pages"] * self.page)
            hwm = status.get("VmHWM", rss)
            role = role_of(argv)
            run_dir = arg_after(argv, "--results_dir")
            self.w_procs.writerow({
                "t_unix": f"{t:.3f}", "t_local": t_local, "elapsed_s": f"{elapsed:.1f}",
                "pid": pid, "ppid": st["ppid"], "role": role,
                "run": os.path.basename(run_dir.rstrip("/")) if run_dir else "",
                "env": arg_after(argv, "--env"), "seed": arg_after(argv, "--seed"),
                "age_s": f"{age:.0f}" if age is not None else "",
                "cpu_pct": f"{user_pct + sys_pct:.1f}", "user_pct": f"{user_pct:.1f}", "system_pct": f"{sys_pct:.1f}",
                "rss_mb": f"{rss / MB:.1f}", "hwm_mb": f"{hwm / MB:.1f}",
                "threads": status.get("Threads", st["threads"]), "on_cpu": st["processor"],
                "allowed_cpus": status.get("Cpus_allowed_list", ""),
                "vol_ctxt_per_s": f"{dvol / dt:.1f}", "invol_ctxt_per_s": f"{dinvol / dt:.1f}",
                "minflt_per_s": f"{dminflt / dt:.1f}", "majflt": st["majflt"],
                "read_mb_per_s": f"{drd / MB / dt:.3f}", "write_mb_per_s": f"{dwr / MB / dt:.3f}",
                "first_sample": int(first),
            })
            self.proc_prev[pid] = {"t": now, "utime": st["utime"], "stime": st["stime"],
                                   "vol": status.get("voluntary_ctxt_switches", 0),
                                   "invol": status.get("nonvoluntary_ctxt_switches", 0),
                                   "minflt": st["minflt"], "rd": io.get("read_bytes", 0), "wr": io.get("write_bytes", 0)}
            if role == "run":
                agg["n_runs"] += 1
                agg["cpu"] += (user_pct + sys_pct) / 100
                agg["rss"] += rss
                agg["hwm"] += hwm
                agg["threads"] += status.get("Threads", st["threads"])
            elif role == "launcher":
                agg["n_launchers"] += 1
        self.f_procs.flush()
        for pid in list(self.proc_prev):
            if pid not in seen:
                del self.proc_prev[pid]
        return {"n_runs": agg["n_runs"], "n_launchers": agg["n_launchers"],
                "runs_cpu_cores": f"{agg['cpu']:.2f}", "runs_rss_mb": round(agg["rss"] / MB),
                "runs_hwm_mb": round(agg["hwm"] / MB), "runs_threads": agg["threads"]}

    def run(self):
        a = self.args
        interval = a.interval
        proc_every = max(1, round(a.proc_interval / interval))
        next_t = self.t0_mono + interval
        tick = 0
        watch = f"{PROC}/{a.watch_pid}" if a.watch_pid else None
        while not self.stop:
            time.sleep(max(0.0, next_t - time.monotonic()))
            tick += 1
            self.sample(with_procs=(tick % proc_every == 0))
            next_t += interval
            if a.duration and time.monotonic() - self.t0_mono >= a.duration:
                break
            if watch and not os.path.exists(watch):
                print(f"watched pid {a.watch_pid} is gone; stopping", flush=True)
                break
        self.sample(with_procs=True)      # final tick: catches the end state
        for f in (self.f_node, self.f_cpus, self.f_procs):
            f.close()
        print(f"stopped after {time.monotonic() - self.t0_mono:.0f}s, {tick} ticks", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output directory (created)")
    p.add_argument("--interval", type=float, default=5.0, help="seconds between node/CPU samples")
    p.add_argument("--proc_interval", type=float, default=30.0, help="seconds between per-process samples")
    p.add_argument("--match", nargs="+", default=["incremental_rl.avg_ablation", "incremental_rl.avg_multi_experiment"],
                   help="track processes whose command line contains any of these strings")
    p.add_argument("--duration", type=float, default=0, help="stop after this many seconds (0: run until killed)")
    p.add_argument("--watch_pid", type=int, default=0, help="stop when this process has exited")
    p.add_argument("--pin", default=None,
                   help="CPU to pin the monitor to ('auto': the lowest CPU this job may use, i.e. a reserved core)")
    p.add_argument("--nice", type=int, default=10, help="niceness to run at (so the runs always win the core)")
    args = p.parse_args()
    if args.proc_interval < args.interval:
        args.proc_interval = args.interval

    if hasattr(os, "nice") and args.nice:
        try:
            os.nice(args.nice)
        except OSError:
            pass
    if args.pin is not None and hasattr(os, "sched_setaffinity"):
        cpu = min(os.sched_getaffinity(0)) if args.pin == "auto" else int(args.pin)
        os.sched_setaffinity(0, {cpu})

    mon = Monitor(args)

    def on_signal(signum, _frame):
        print(f"signal {signum}: taking a final sample and stopping", flush=True)
        mon.stop = True
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, on_signal)
    print(f"monitoring {len(mon.cpus)} CPUs ({len(mon.allowed)} allowed), {len(mon.sockets)} socket(s); "
          f"node every {args.interval}s, processes every {args.proc_interval}s -> {args.out}", flush=True)
    mon.run()


if __name__ == "__main__":
    main()
