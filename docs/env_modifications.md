# Environment modifications: Ant-v4, Hopper-v4, Humanoid-v4

All names and default values below were read from the models as loaded by `gymnasium 1.3.0` / `mujoco 3.14`,
so they are exactly what `incremental_rl/env_mods.py` sees. Each environment's section is self-contained.

| kind | what it changes | how it is set | can change mid-run? |
|---|---|---|---|
| **Reward / termination** (Section 1) | the Gymnasium env's settings (constructor arguments) | `env_kwargs` at step 0; `env_kwargs` inside a `schedule` entry later | yes, set in place — except `use_contact_forces` and `exclude_current_positions_from_observation`, which change the observation size |
| **Physics / dynamics** (Section 2) | fields of the MuJoCo model (`env.unwrapped.model`) | `env_mods` at step 0; `mods` inside a `schedule` entry later | yes |

Config example (one experiment of a multi-experiment file):

```json
{"name": "moon_mars", "env": "Ant-v4", "n_runs": 30,
 "env_kwargs": {"ctrl_cost_weight": 0.25, "healthy_z_range": [0.2, 1.5], "max_episode_steps": 1000},
 "env_mods":   {"gravity": -1.62, "mass_scale": 0.8},
 "schedule":   [{"step": 3000000, "mods": {"gravity": -3.71}},
                {"step": 6000000, "mods": {"gravity": -9.81, "gear_scale": 1.2}, "env_kwargs": {"healthy_z_range": [0.2, 1.0]}}]}
```

A schedule entry needs `mods`, `env_kwargs`, or both. The Gymnasium MuJoCo envs read their reward/termination
settings from attributes on every step, so a mid-run `env_kwargs` change takes effect immediately without
rebuilding the env (the episode in progress continues under the new settings).

`*_scale` keys are relative to the **original** model values, not the current ones, so `mass_scale: 0.8`
followed later by `mass_scale: 1.0` restores the default. All applied values are printed to `run.log` once and
the full `env_kwargs` / `env_mods` / `schedule` are stored in the run's `args.json`.

---

# Section 1 — Reward and termination (`env_kwargs`)

These are the Gymnasium environment's constructor arguments. Given as `env_kwargs` on the experiment they apply
from step 0; given as `env_kwargs` in a `schedule` entry they are changed in place at that step (all except the
two observation-size arguments). One extra knob is applied at `gym.make` level and is also accepted in
`env_kwargs`: **`max_episode_steps`** (default **1000** for all three environments) — the step count at which an
episode is truncated. Range-valued arguments are given as 2-element lists in JSON.

## 1.1 Ant-v4

**Reward per step** = `healthy_reward` + `forward_reward` − `ctrl_cost` − `contact_cost`

- `forward_reward = Δx_torso / dt` — forward velocity of the torso (weight fixed at 1 in v4; there is no
  `forward_reward_weight` argument)
- `ctrl_cost = ctrl_cost_weight · Σ action²`
- `contact_cost = contact_cost_weight · Σ clip(external contact forces, contact_force_range)²` — only when
  `use_contact_forces` is true

**Healthy** ⇔ torso height z within `healthy_z_range` **and** every state value finite.

| argument | default | effect |
|---|---|---|
| `ctrl_cost_weight` | 0.5 | penalty on action magnitude (energy) |
| `use_contact_forces` | false | adds the 84 contact-force values to the observation **and** enables `contact_cost` |
| `contact_cost_weight` | 5e‑4 | weight of the contact penalty (only if `use_contact_forces`) |
| `contact_force_range` | [−1, 1] | clipping of contact forces before squaring |
| `healthy_reward` | 1.0 | constant bonus every step the ant is healthy |
| `healthy_z_range` | [0.2, 1.0] | torso height band for "healthy" |
| `terminate_when_unhealthy` | true | end the episode when unhealthy; if false the episode continues but without `healthy_reward` |
| `reset_noise_scale` | 0.1 | uniform noise added to the initial joint positions / velocities |
| `exclude_current_positions_from_observation` | true | drop torso x, y from the observation (27 dims instead of 29) |
| `max_episode_steps` | 1000 | truncation limit |

## 1.2 Hopper-v4

**Reward per step** = `healthy_reward` + `forward_reward_weight · Δx / dt` − `ctrl_cost_weight · Σ action²`

**Healthy** ⇔ all of: every state value (excluding x) within `healthy_state_range`, torso height z within
`healthy_z_range`, torso angle within `healthy_angle_range`.

| argument | default | effect |
|---|---|---|
| `forward_reward_weight` | 1.0 | weight of forward velocity |
| `ctrl_cost_weight` | 1e‑3 | penalty on action magnitude |
| `healthy_reward` | 1.0 | constant bonus per healthy step |
| `healthy_state_range` | [−100, 100] | bound on every state element (positions and velocities) |
| `healthy_z_range` | [0.7, inf] | minimum torso height; falling below = unhealthy |
| `healthy_angle_range` | [−0.2, 0.2] rad | torso pitch allowed before "fallen" (≈ ±11.5°) |
| `terminate_when_unhealthy` | true | terminate vs. continue without the healthy bonus |
| `reset_noise_scale` | 5e‑3 | initial-state noise |
| `exclude_current_positions_from_observation` | true | drop x from the observation (11 dims instead of 12) |
| `max_episode_steps` | 1000 | truncation limit |

## 1.3 Humanoid-v4

**Reward per step** = `healthy_reward` + `forward_reward_weight · Δx_com / dt` − `ctrl_cost_weight · Σ action²`

- forward velocity is that of the **centre of mass**, not the torso
- v4 has **no contact cost** argument (the 376‑dim observation does include contact forces)

**Healthy** ⇔ torso height z within `healthy_z_range`.

| argument | default | effect |
|---|---|---|
| `forward_reward_weight` | 1.25 | weight of the centre-of-mass forward velocity |
| `ctrl_cost_weight` | 0.1 | penalty on action magnitude |
| `healthy_reward` | 5.0 | constant bonus per healthy step (dominates the reward early in training) |
| `healthy_z_range` | [1.0, 2.0] | torso height band for "healthy" (fallen or launched = unhealthy) |
| `terminate_when_unhealthy` | true | terminate vs. continue without the healthy bonus |
| `reset_noise_scale` | 1e‑2 | initial-state noise |
| `exclude_current_positions_from_observation` | true | drop x, y from the observation (376 dims instead of 378) |
| `max_episode_steps` | 1000 | truncation limit |

## 1.4 Not available as arguments

- **Reward scaling / shaping, observation or action noise** — not constructor arguments; they would be Gymnasium
  wrappers (`TransformReward`, custom noise wrappers). Not implemented.
- **Geometry** (limb lengths, obstacles, slopes) — needs a modified XML via the `xml_file` argument. Not covered by
  `env_mods`.

---

# Section 2 — Physics and dynamics (`env_mods` / `schedule`)

Each subsection lists **every** modification available for that environment, with that environment's own body,
geom, joint and actuator names and default values. After any mass change `mujoco.mj_setConst` is called so
derived quantities (subtree masses etc.) stay consistent.

What the fields mean for the agent (same for all three):

- **gravity** — changes every jump, fall and ground-reaction force; observation and reward are untouched, so this
  is a pure dynamics shift
- **mass / inertia** — heavier bodies need more torque for the same motion; same gear → slower, lower movement
- **friction** — sliding μ governs push-off; lowering the floor's μ makes the agent slip
- **damping / armature / stiffness** — joint-level resistance: damping scales with joint velocity, armature adds
  rotor inertia, stiffness pulls the joint towards its spring reference
- **gear** — torque per unit of action (`torque = gear × ctrl`): rescales motor strength without changing the
  action bounds
- **timestep** — physics integration step; `agent dt = timestep × frame_skip`. Smaller is more accurate and slower,
  larger can go unstable
- **wind / density / viscosity** — aerodynamic forces; `wind` only acts when `density` or `viscosity` is non-zero

## 2.1 Ant-v4

**Simulation options** (`model.opt`): `timestep` **0.01**, `frame_skip` 5 (agent dt 0.05), `gravity` **[0, 0, −9.81]**,
`wind` [0, 0, 0], `density` 0, `viscosity` 0. Total mass **0.91 kg**.

| key | value | what it sets (Ant defaults) |
|---|---|---|
| `gravity` | `-1.62` or `[gx, gy, gz]` | `opt.gravity`, default [0, 0, −9.81]; a scalar sets the z component (Moon −1.62, Mars −3.71, K2‑18b ≈ −11.95) |
| `wind` | `[wx, wy, wz]` m/s | `opt.wind`, default [0, 0, 0] |
| `density` | kg/m³ | `opt.density`, default 0 (air ≈ 1.2) |
| `viscosity` | Pa·s | `opt.viscosity`, default 0 (air ≈ 1.8e‑5) |
| `timestep` | s | `opt.timestep`, default 0.01 |
| `mass_scale` | factor | mass and inertia of all 13 bodies |
| `mass` | `{body: kg}` | per body; inertia scaled by the same ratio. Bodies: `torso` 0.3272; `front_left_leg`, `aux_1`, `front_right_leg`, `aux_2`, `back_leg`, `aux_3`, `right_back_leg`, `aux_4` 0.0392 each; the four lower-leg (ankle) bodies are **unnamed** (0.0676 each) — reach them with `mass_scale` |
| `friction_scale` | factor or `[slide, spin, roll]` | friction of all 14 geoms |
| `friction` | `{geom: μ}` or `{geom: [slide, spin, roll]}` | per geom. All geoms default **[1.0, 0.5, 0.5]**: `floor`, `torso_geom`, `aux_1_geom`, `left_leg_geom`, `left_ankle_geom`, `aux_2_geom`, `right_leg_geom`, `right_ankle_geom`, `aux_3_geom`, `back_leg_geom`, `third_ankle_geom`, `aux_4_geom`, `rightback_leg_geom`, `fourth_ankle_geom` |
| `damping_scale` | factor | `dof_damping` of all joints (the free `root` joint has 0 and stays 0) |
| `damping` | `{joint: value}` | per joint. `hip_1`…`hip_4`, `ankle_1`…`ankle_4`: damping **1.0** each |
| `armature_scale` | factor | `dof_armature`: hips and ankles **1.0** each |
| `stiffness_scale` | factor | `jnt_stiffness`: all joints **0** in Ant, so this has no effect |
| `gear_scale` | factor | `actuator_gear` of all 8 actuators |
| `gear` | `{actuator: value}` | per actuator. Ant's actuators are **unnamed**: address them by the joint they drive — `hip_1`…`hip_4`, `ankle_1`…`ankle_4`, gear **150** each, ctrl range [−1, 1] |

Joint ranges for reference: hips ±30°, ankles 30°–70° (sign alternates per leg).

## 2.2 Hopper-v4

**Simulation options** (`model.opt`): `timestep` **0.002**, `frame_skip` 4 (agent dt 0.008), `gravity` **[0, 0, −9.81]**,
`wind` [0, 0, 0], `density` 0, `viscosity` 0. Total mass **15.82 kg**.

| key | value | what it sets (Hopper defaults) |
|---|---|---|
| `gravity` | `-1.62` or `[gx, gy, gz]` | `opt.gravity`, default [0, 0, −9.81]; a scalar sets the z component (Moon −1.62, Mars −3.71, K2‑18b ≈ −11.95) |
| `wind` | `[wx, wy, wz]` m/s | `opt.wind`, default [0, 0, 0] |
| `density` | kg/m³ | `opt.density`, default 0 (air ≈ 1.2) |
| `viscosity` | Pa·s | `opt.viscosity`, default 0 (air ≈ 1.8e‑5) |
| `timestep` | s | `opt.timestep`, default 0.002 |
| `mass_scale` | factor | mass and inertia of all 4 bodies |
| `mass` | `{body: kg}` | per body; inertia scaled by the same ratio. Bodies: `torso` 3.6652, `thigh` 4.0579, `leg` 2.7814, `foot` 5.3156 |
| `friction_scale` | factor or `[slide, spin, roll]` | friction of all 5 geoms |
| `friction` | `{geom: μ}` or `{geom: [slide, spin, roll]}` | per geom: `floor` [1.0, 0.005, 0.0001]; `torso_geom`, `thigh_geom`, `leg_geom` [0.9, 0.005, 0.0001]; `foot_geom` **[2.0, 0.005, 0.0001]** |
| `damping_scale` | factor | `dof_damping` of all joints (`rootx`, `rootz`, `rooty` have 0 and stay 0) |
| `damping` | `{joint: value}` | per joint. `thigh_joint`, `leg_joint`, `foot_joint`: damping **1.0** each |
| `armature_scale` | factor | `dof_armature`: `thigh_joint`, `leg_joint`, `foot_joint` **1.0** each |
| `stiffness_scale` | factor | `jnt_stiffness`: all joints **0** in Hopper, so this has no effect |
| `gear_scale` | factor | `actuator_gear` of all 3 actuators |
| `gear` | `{actuator: value}` | per actuator. Hopper's actuators are **unnamed**: address them by the joint they drive — `thigh_joint`, `leg_joint`, `foot_joint`, gear **200** each, ctrl range [−1, 1] |

Joint ranges for reference: `thigh_joint`, `leg_joint` −150°–0°, `foot_joint` ±45°.

## 2.3 Humanoid-v4

**Simulation options** (`model.opt`): `timestep` **0.003**, `frame_skip` 5 (agent dt 0.015), `gravity` **[0, 0, −9.81]**,
`wind` [0, 0, 0], `density` 0, `viscosity` 0. Total mass **42.12 kg**.

| key | value | what it sets (Humanoid defaults) |
|---|---|---|
| `gravity` | `-1.62` or `[gx, gy, gz]` | `opt.gravity`, default [0, 0, −9.81]; a scalar sets the z component (Moon −1.62, Mars −3.71, K2‑18b ≈ −11.95) |
| `wind` | `[wx, wy, wz]` m/s | `opt.wind`, default [0, 0, 0] |
| `density` | kg/m³ | `opt.density`, default 0 (air ≈ 1.2) |
| `viscosity` | Pa·s | `opt.viscosity`, default 0 (air ≈ 1.8e‑5) |
| `timestep` | s | `opt.timestep`, default 0.003 |
| `mass_scale` | factor | mass and inertia of all 13 bodies |
| `mass` | `{body: kg}` | per body; inertia scaled by the same ratio. Bodies: `torso` 8.9075, `lwaist` 2.2619, `pelvis` 6.6162, `right_thigh` / `left_thigh` 4.7518, `right_shin` / `left_shin` 2.7557, `right_foot` / `left_foot` 1.7671, `right_upper_arm` / `left_upper_arm` 1.6611, `right_lower_arm` / `left_lower_arm` 1.2295 |
| `friction_scale` | factor or `[slide, spin, roll]` | friction of all 18 geoms |
| `friction` | `{geom: μ}` or `{geom: [slide, spin, roll]}` | per geom: `floor` **[1.0, 0.1, 0.1]**; all body geoms [1.0, 0.005, 0.0001]: `torso1`, `head`, `uwaist`, `lwaist`, `butt`, `right_thigh1`, `right_shin1`, `right_foot`, `left_thigh1`, `left_shin1`, `left_foot`, `right_uarm1`, `right_larm`, `right_hand`, `left_uarm1`, `left_larm`, `left_hand` |
| `damping_scale` | factor | `dof_damping` of all joints (the free `root` joint has 0 and stays 0) |
| `damping` | `{joint: value}` | per joint: `abdomen_z`, `abdomen_y`, `abdomen_x`, `right_hip_x`, `right_hip_z`, `right_hip_y`, `left_hip_x`, `left_hip_z`, `left_hip_y` **5**; `right_knee`, `left_knee`, `right_shoulder1`, `right_shoulder2`, `right_elbow`, `left_shoulder1`, `left_shoulder2`, `left_elbow` **1** |
| `armature_scale` | factor | `dof_armature`: abdomen 0.02; hips 0.01 (`right_hip_y` 0.008); knees 0.006; shoulders 0.007 / 0.005; elbows 0.003 |
| `stiffness_scale` | factor | `jnt_stiffness` (Humanoid is the only one of the three with joint springs): `abdomen_z` 20, `abdomen_y` / `abdomen_x` 10, hip x/z 10, hip y 20, `right_knee` 0, `left_knee` 1, shoulders 1, elbows 0 |
| `gear_scale` | factor | `actuator_gear` of all 17 actuators |
| `gear` | `{actuator: value}` | per actuator (named after their joint): `abdomen_y`, `abdomen_z`, `abdomen_x` 100; `right_hip_x`, `right_hip_z`, `left_hip_x`, `left_hip_z` 100; `right_hip_y`, `left_hip_y` **300**; `right_knee`, `left_knee` 200; `right_shoulder1`, `right_shoulder2`, `right_elbow`, `left_shoulder1`, `left_shoulder2`, `left_elbow` 25. Ctrl range **[−0.4, 0.4]** for all |

Joint ranges for reference: abdomen ±45° (z), −75°–30° (y), ±35° (x); hips −25°–5° (x), −60°–35° (z),
−110°–20° (y); knees −160°–−2°; shoulders ±60°–85°; elbows −90°–50°.
