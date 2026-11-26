"""Rank the draws of a hyperparameter sweep (scripts/sweep/make_sweep_configs.py) by return.

    python scripts/sweep/sweep_summary.py --root results_sweep [--env Reacher-v4] [--top 5]

Per draw: mean over seeds of the average return over the whole run (what hyp_sweep.py ranked by) and over the
last 10% of steps. Unfinished runs are skipped.
"""
import argparse, glob, json, os
import numpy as np

KEYS = ["actor_lr", "critic_lr", "beta1", "alpha_lr", "gamma"]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results_sweep")
    p.add_argument("--env", default=None)
    p.add_argument("--top", default=None, type=int, help="Only the best N draws per env")
    args = p.parse_args()
    draws = {}
    # run folders sit in <root>/<env>/ (or directly in <root>)
    for d in sorted(glob.glob(os.path.join(args.root, "h*-*")) + glob.glob(os.path.join(args.root, "*", "h*-*"))):
        if not os.path.isfile(os.path.join(d, "multi_experiment.json")):
            continue
        meta = json.load(open(os.path.join(d, "multi_experiment.json")))
        hyp, files = meta["hyperparams"], glob.glob(os.path.join(d, "*_returns.txt"))
        if not files or (args.env and hyp["env"] != args.env):
            continue
        ep_lens, rets = np.loadtxt(files[0]).reshape(2, -1)
        steps = np.cumsum(ep_lens)
        if steps[-1] < 0.99 * hyp["N"]:
            continue
        key = (hyp["env"], hyp["N"], meta["experiment"])  # runs of different length are ranked apart
        draws.setdefault(key, {"hyp": hyp, "all": [], "last": []})
        draws[key]["all"].append(rets.mean())
        draws[key]["last"].append(rets[steps > 0.9 * hyp["N"]].mean())
    print("| env | steps | draw | seeds | return, whole run | return, last 10% | " + " | ".join(KEYS) + " |")
    print("|---|---|---|---|---|---|" + "---|" * len(KEYS))
    shown = {}
    for (env, N, name), r in sorted(draws.items(), key=lambda kv: (kv[0][0], kv[0][1], -np.mean(kv[1]["all"]))):
        shown[(env, N)] = shown.get((env, N), 0) + 1
        if args.top and shown[(env, N)] > args.top:
            continue
        se = lambda v: np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        print(f"| {env} | {N:g} | {name} | {len(r['all'])} | {np.mean(r['all']):.1f} ± {se(r['all']):.1f} | {np.mean(r['last']):.1f} ± {se(r['last']):.1f} | "
              + " | ".join(f"{r['hyp'][k]:g}" for k in KEYS) + " |")
