#!/bin/bash
# Measure AVG throughput on a Trillium compute node, to choose cores/threads per run and check walltime.
# Run inside an interactive debug session (1 full node, 60 min):
#   debugjob
#   cd $SCRATCH && bash $HOME/cf-avg/scripts/trillium/bench.sh
#
# 1) Thread scaling: one run alone, pinned to T cores with T torch threads.
#    Each config runs at two lengths; the difference cancels out startup cost.
# 2) Full node: 188 concurrent 1-core runs, to see how much each run slows when the node is packed
#    (shared memory bandwidth / L3 / lower boost clocks).

set -euo pipefail
REPO=${REPO:-$HOME/cf-avg}
ENV_ID=${ENV_ID:-Hopper-v4}
N_SHORT=${N_SHORT:-2000}
N_LONG=${N_LONG:-22000}
OUT=${OUT:-$SCRATCH/avg_bench}
mkdir -p "$OUT"

cd "$REPO"
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg

time_run() {  # time_run <threads> <N>  -> prints seconds
    local t=$1 n=$2 start end
    start=$(date +%s.%N)
    OMP_NUM_THREADS=$t MKL_NUM_THREADS=$t taskset -c 0-$((t - 1)) \
        python -m incremental_rl.avg_ablation --env "$ENV_ID" --N "$n" --n_threads "$t" \
        --do_not_save --results_dir "$OUT" > /dev/null 2>&1
    end=$(date +%s.%N)
    awk -v a="$start" -v b="$end" 'BEGIN { print b - a }'
}

echo "== Thread scaling ($ENV_ID, single run) =="
printf "%-8s %-12s\n" threads steps/sec
for t in 1 2 4 8; do
    s=$(time_run "$t" "$N_SHORT")
    l=$(time_run "$t" "$N_LONG")
    awk -v t="$t" -v d="$((N_LONG - N_SHORT))" -v s="$s" -v l="$l" 'BEGIN { printf "%-8s %-12.1f\n", t, d / (l - s) }'
done

echo
echo "== Full node: 188 x 1-core runs, N=$N_LONG each =="
python -m incremental_rl.avg_multi_experiment --name bench --results_dir "$OUT/full_node" \
    --n_runs 188 --cores_per_run 1 --reserve_cores 4 --env "$ENV_ID" --N "$N_LONG" --do_not_save \
    | grep -E "^\[(done|FAILED)" \
    | sed -E 's/.* in ([0-9.]+)s.*/\1/' \
    | awk -v n="$N_LONG" '{ sum += $1; if ($1 > max) max = $1 } END {
        printf "runs: %d  mean wall: %.1fs  slowest: %.1fs  ->  ~%.1f steps/sec per run (incl. startup)\n", NR, sum/NR, max, n/(sum/NR) }'

echo
echo "Hours for a 12M-step run ~= 12e6 / (steps/sec) / 3600. Trillium max walltime is 24h."
