"""Side-by-side of each method's paired result under the G-truncation fix and under the legacy G reset.

    python scripts/analysis/compare_worlds.py --fix results_trillium/lever_results/analysis --legacy results_trillium/analysis_legacy

Reads recovery_summary.csv from both analysis folders and prints, per method and shift, the early-return change
and the recovery-time change against each world's own base run, with wins/5 and significance stars.
"""
import argparse, csv, os

PAIRS = [("Hopper-v4", "gear2x"), ("Hopper-v4", "heavy"), ("Ant-v4", "gear2x"), ("Ant-v4", "limb_heavy"),
         ("Humanoid-v4", "gear2x"), ("Humanoid-v4", "heavy")]


def load(folder):
    rows = list(csv.DictReader(open(os.path.join(folder, "recovery_summary.csv"))))
    return {(r["env"], r["shift"], r["cond"], r["metric"]): r for r in rows}


def cell(r, metric):
    if r is None:
        return "-"
    m, b = float(r["mean"]), float(r["fix_mean"])
    if metric == "t80_fix":
        return f"{m / 1e6:.2f}/{b / 1e6:.2f} {r['wins']}/5{r['sig']}"
    return f"{100 * (m - b) / abs(b):+.0f}% {r['wins']}/5{r['sig']}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fix", default="results_trillium/lever_results/analysis")
    p.add_argument("--legacy", default="results_trillium/analysis_legacy")
    p.add_argument("--conds", nargs="*", default=None, help="Conditions to show (default: all in the legacy summary)")
    args = p.parse_args()
    F, L = load(args.fix), load(args.legacy)
    conds = args.conds or sorted({k[2] for k in L if k[2] not in ("fix", "legacy", "scratch")}, key=lambda c: (c[0], c))
    for metric, title in [("early_1M", "early return, first 1M steps after the switch: % change vs own base (wins/5)"),
                          ("t80_fix", "recovery time: method/base in M steps to 80% of the base asymptote (wins/5)")]:
        print(f"\n=== {title}")
        print(f"{'method':22s}{'world':8s}" + "".join(f"{e.split('-')[0][:3] + ' ' + s[:6]:>20s}" for e, s in PAIRS))
        for c in conds:
            for world, D in (("fix", F), ("legacy", L)):
                print(f"{c if world == 'fix' else '':22s}{world:8s}" + "".join(f"{cell(D.get((e, s, c, metric)), metric):>20s}" for e, s in PAIRS))


if __name__ == "__main__":
    main()
