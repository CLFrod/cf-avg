# Configs

One folder per study. Every file is an experiment list for `python -m incremental_rl.avg_multi_experiment --config <file>`
(`--dry_run` prints the plan).

| Folder | Objective | Files | Written by |
|---|---|---|---|
| `dynamics_shifts/` | How AVG adapts when the physics or reward changes mid-run: baselines, whole-body shifts, per-limb shifts | `trillium_dynamics_3env*.json`, `trillium_dynamics_shifts_18m.json`, `trillium_limb_shifts_18m.json`; `trillium_jobtest.json` rehearses the job script | by hand |
| `levers/` | Lever study batch 1: single on-shift levers and stacks against the fix-only and legacy G-reset splits | `levers_<env>.json`, `levers_legacy_<env>.json`, smoke configs | `scripts/configs/make_lever_configs.py` |
| `privileged/` | Lever study batch 2: levers driven by privileged targets measured from batch 1 | `priv_<env>.json`, `priv_legacy_<env>.json`, smoke configs, `privileged_targets.json` (the targets) | `scripts/configs/make_privileged_configs.py`, `scripts/analysis/privileged_targets.py` |
| `greedy_avg/` | Greedy AVG study: the Greedy AC actor update inside AVG, on every shift and recovery method | `gavg_<env>.json`, smoke config | `scripts/configs/make_gavg_configs.py` |
| `sweep/` | Hyperparameter search, stage 1: 62 random draws x 3 seeds per env, short runs | `sweep_<env>.json` | `scripts/sweep/make_sweep_configs.py` |
| `sweep_long/` | Hyperparameter search, stage 2: the best draws rerun longer at 10 seeds | `sweep_long_<env>.json` | `scripts/sweep/make_sweep_configs.py --tag long` |

Generated configs are regenerated rather than edited. `*_smoke.json` files are the 3000-step versions that
`scripts/trillium/smoke_levers.sh` runs on a login node.
