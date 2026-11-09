"""Privileged targets for the batch-2 levers, from the batch-1 runs: writes configs/privileged_targets.json.

    python scripts/analysis/privileged_targets.py [--cache results_trillium/lever_results/analysis/cache]

Per env/shift: the entropy of the from-scratch policy at the end of its run (the entropy a competent post-shift
policy has), the reward per step before the switch and over 50k..100k steps after it in the fix-only runs, and the
offset that makes the early post-shift reward match the pre-shift one. Needs the cache written by lever_recovery.py.
"""
import argparse, glob, json, os
import numpy as np

SWITCH = {"Hopper-v4": 3_000_000, "Ant-v4": 6_000_000, "Humanoid-v4": 6_000_000}
SHIFTS = {"Hopper-v4": ["gear2x", "heavy"], "Ant-v4": ["gear2x", "limb_heavy"], "Humanoid-v4": ["gear2x", "heavy"]}


def runs(cache, cond, env, shift):
    return [np.load(f) for f in sorted(glob.glob(f"{cache}/{cond}-{shift}-{env}-*.npz")) if "_dense" not in f]


def reward_per_step(zs, lo, hi):
    return float(np.mean([np.nanmean(z["return"][(z["x"] > lo) & (z["x"] <= hi)]) / np.nanmean(z["ep_len"][(z["x"] > lo) & (z["x"] <= hi)]) for z in zs]))


def entropy(zs, lo, hi):
    return float(np.mean([np.nanmean(z["entropy"][(z["x"] > lo) & (z["x"] <= hi)]) for z in zs]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="results_trillium/lever_results/analysis/cache")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "configs", "privileged_targets.json"))
    args = p.parse_args()
    targets = {}
    for env, shifts in SHIFTS.items():
        S = SWITCH[env]
        for shift in shifts:
            fix, scr = runs(args.cache, "fix", env, shift), runs(args.cache, "scratch", env, shift)
            N = int(fix[0]["N"])
            r_pre, r_post = reward_per_step(fix, S - 200_000, S), reward_per_step(fix, S + 50_000, S + 100_000)
            targets.setdefault(env, {})[shift] = {
                "entropy_target": round(entropy(scr, N - 1_000_000, N), 2),
                "entropy_pre": round(entropy(fix, S - 200_000, S), 2),
                "entropy_post_fix": round(entropy(fix, S + 50_000, S + 300_000), 2),
                "r_pre": round(r_pre, 3), "r_post_early": round(r_post, 3), "reward_offset": round(r_pre - r_post, 2),
            }
            print(f"{env:12s} {shift:10s} {targets[env][shift]}")
    with open(args.out, "w") as f:
        json.dump(targets, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
