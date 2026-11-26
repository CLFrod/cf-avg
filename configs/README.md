# Configs

One folder per study. Every file is an experiment list for `python -m incremental_rl.avg_multi_experiment --config <file>`
(`--dry_run` prints the plan).

| Folder | Objective | Files | Written by |
|---|---|---|---|
| `dynamics_shifts/` | How AVG adapts when the physics or reward changes mid-run: baselines, whole-body shifts, per-limb shifts | `trillium_dynamics_3env*.json`, `trillium_dynamics_shifts_18m.json`, `trillium_limb_shifts_18m.json`; `trillium_jobtest.json` rehearses the job script | by hand |
| `levers/` | Lever study batch 1: single on-shift levers and stacks against the fix-only and legacy G-reset splits | `levers_<env>.json`, `levers_legacy_<env>.json`, smoke configs | `scripts/configs/make_lever_configs.py` |
| `privileged/` | Lever study batch 2: levers driven by privileged targets measured from batch 1 | `priv_<env>.json`, `priv_legacy_<env>.json`, smoke configs, `privileged_targets.json` (the targets) | `scripts/configs/make_privileged_configs.py`, `scripts/analysis/privileged_targets.py` |
| `shift_study/` | Per-environment shift study on tuned hyperparameters: baseline, each shift at the halfway step, each shift from step 0, and the shortlisted levers on each shift. One node per file | `study_<env>_core.json` (baseline and the shifts every env gets), `study_<env>_body.json` (the shifts of that body) | `scripts/configs/make_shift_study_configs.py` |
| `traces/` | Critic eligibility trace (`critic_lambda`): a critic_lambda x critic_lr grid on the shift study's baseline, shifted and from-scratch conditions, no levers. One node per file | `traces_<env>_none.json` (baseline), `traces_<env>_<shift>.json` (one shift, switched at the halfway step and from step 0); `traces_smoke.json` by hand | `scripts/configs/make_trace_configs.py` |
| `sweep/` | Hyperparameter search, stage 1: 62 random draws x 3 seeds per env, short runs | `sweep_<env>.json` | `scripts/sweep/make_sweep_configs.py` |
| `sweep_long/` | Hyperparameter search, stage 2: the best draws rerun longer at 10 seeds | `sweep_long_<env>.json` | `scripts/sweep/make_sweep_configs.py --tag long` |

Generated configs are regenerated rather than edited. `*_smoke.json` files are the 3000-step versions that
`scripts/trillium/smoke_levers.sh` runs on a login node.
