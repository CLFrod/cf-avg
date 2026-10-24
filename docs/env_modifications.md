# Environment modifications: Ant-v4, Hopper-v4, Humanoid-v4

Everything below was read from the models as loaded by `gymnasium 1.3.0` / `mujoco 3.14` (not from the docs), so
the names and default values are what `incremental_rl/env_mods.py` actually sees. Two kinds of modification:

| kind | what it changes | how it is set | can change mid-run? |
|---|---|---|---|
| **Physics / dynamics** | fields of the MuJoCo model (`env.unwrapped.model`) | `env_mods` (at step 0) and `schedule` (at a given step) | yes |
| **Reward / termination** | the Gymnasium env's constructor arguments | `env_kwargs` → `gym.make(env, **env_kwargs)` | no (fixed for the run) |

Config example (one experiment of a multi-experiment file):

```json
{"name": "moon_mars", "env": "Ant-v4", "n_runs": 30,
 "env_kwargs": {"ctrl_cost_weight": 0.25, "healthy_z_range": [0.2, 1.5]},
 "env_mods":   {"gravity": -1.62, "mass_scale": 0.8},
 "schedule":   [{"step": 3000000, "mods": {"gravity": -3.71}},
                {"step": 6000000, "mods": {"gravity": -9.81, "gear_scale": 1.2}}]}
```

---

## Part 1 — Physics / dynamics (`env_mods` / `schedule`)

### 1.1 The modification keys (same for every env)

`*_scale` keys are relative to the **original** model values, not to the current ones, so a schedule of
`mass_scale: 0.8` then `mass_scale: 1.0` restores the default. Absolute keys take the model's own names
(tables in 1.2–1.4).

| key | value | sets | notes |
|---|---|---|---|
| `gravity` | `-1.62` or `[gx, gy, gz]` | `opt.gravity` | scalar = z component (default `-9.81`). Moon −1.62, Mars −3.71, K2‑18b ≈ −11.95 |
| `wind` | `[wx, wy, wz]` m/s | `opt.wind` | only has an effect if `density` or `viscosity` > 0 |
| `density` | kg/m³ | `opt.density` | medium density; enables lift/drag (air ≈ 1.2) |
| `viscosity` | Pa·s | `opt.viscosity` | medium viscosity (air ≈ 1.8e‑5, water ≈ 1e‑3) |
| `timestep` | s | `opt.timestep` | physics step; `env.dt = timestep × frame_skip`. Smaller = more accurate & slower; larger can go unstable |
| `mass_scale` | factor | `body_mass`, `body_inertia` | all bodies |
| `mass` | `{body: kg}` | `body_mass[i]`, inertia scaled by the same ratio | per body name |
| `friction_scale` | factor or `[slide, spin, roll]` | `geom_friction` | all geoms (floor included) |
| `friction` | `{geom: μ}` or `{geom: [slide, spin, roll]}` | `geom_friction[i]` | per geom name; scalar sets sliding μ only |
| `damping_scale` | factor | `dof_damping` | all joints (root/free joints have 0 and stay 0) |
| `damping` | `{joint: value}` | `dof_damping` | per joint name |
| `armature_scale` | factor | `dof_armature` | rotor inertia added to each joint |
| `stiffness_scale` | factor | `jnt_stiffness` | joint springs (Humanoid only has non‑zero values) |
| `gear_scale` | factor | `actuator_gear` | all actuators = motor strength |
| `gear` | `{actuator: value}` | `actuator_gear[i, 0]` | per actuator name, or the joint it drives (Ant's actuators are unnamed) |

After any mass/inertia change `mujoco.mj_setConst` is called so derived quantities (subtree masses, etc.) stay
consistent. All applied values are printed once and stored in the run's `args.json`.

What the physics fields mean for the agent:

- **gravity** — changes the dynamics of every jump/fall and the ground reaction forces; the observation and reward
  functions are untouched, so this is a pure dynamics shift (the continual-learning setting you already use).
- **mass / inertia** — heavier bodies need more torque for the same motion; same gear → slower, lower jumps.
- **friction** — sliding μ governs push-off; lowering the floor's μ makes the agent slip (Hopper's foot is 2.0, the
  rest 0.9; Humanoid's floor has spin/roll friction 0.1 vs 0.005 on bodies).
- **damping / armature / stiffness** — joint-level resistance; damping scales with velocity, armature adds inertia,
  stiffness pulls towards the joint's spring reference.
- **gear** — the torque applied per unit of action (`torque = gear × ctrl`); a gear change effectively rescales the
  action space's strength without changing its bounds.
- **timestep** — integration accuracy; changing it also changes how many physics sub-steps each agent step covers.

### 1.2 Ant-v4

`timestep 0.01`, `frame_skip 5` (agent `dt 0.05`), gravity `[0, 0, −9.81]`, total mass **0.91**, wind/density/viscosity 0.

Bodies (mass kg, inertia diag):

| body | mass | inertia |
|---|---|---|
| `torso` | 0.3272 | 0.0082 ×3 |
| `front_left_leg`, `aux_1`, (unnamed ankle) | 0.0392, 0.0392, 0.0676 | 0.0006 / 0.0006 / 0.0027 |
| `front_right_leg`, `aux_2`, (unnamed ankle) | same | same |
| `back_leg`, `aux_3`, (unnamed ankle) | same | same |
| `right_back_leg`, `aux_4`, (unnamed ankle) | same | same |

The four ankle (lower-leg) bodies have no name in the XML: address them through their geoms (`left_ankle_geom`, …)
for friction, or use `mass_scale` for mass.

Geoms (friction `[slide, spin, roll]` is `[1.0, 0.5, 0.5]` for **all** geoms): `floor`, `torso_geom`,
`aux_1_geom`, `left_leg_geom`, `left_ankle_geom`, `aux_2_geom`, `right_leg_geom`, `right_ankle_geom`,
`aux_3_geom`, `back_leg_geom`, `third_ankle_geom`, `aux_4_geom`, `rightback_leg_geom`, `fourth_ankle_geom`.

Joints: `root` (free, no damping); `hip_1..4`, `ankle_1..4` — damping **1.0**, armature **1.0**, stiffness 0,
ranges hip ±30°, ankle 30°–70°.

Actuators (unnamed; address by joint): `hip_1..4`, `ankle_1..4` — gear **150**, ctrl range [−1, 1].

### 1.3 Hopper-v4

`timestep 0.002`, `frame_skip 4` (agent `dt 0.008`), gravity `[0, 0, −9.81]`, total mass **15.82**.

| body | mass | inertia |
|---|---|---|
| `torso` | 3.6652 | 0.0692, 0.0692, 0.0045 |
| `thigh` | 4.0579 | 0.0933, 0.0933, 0.0049 |
| `leg` | 2.7814 | 0.0723, 0.0723, 0.0022 |
| `foot` | 5.3156 | 0.1035, 0.1035, 0.0092 |

Geoms and friction `[slide, spin, roll]`:

| geom | friction |
|---|---|
| `floor` | 1.0, 0.005, 0.0001 |
| `torso_geom`, `thigh_geom`, `leg_geom` | 0.9, 0.005, 0.0001 |
| `foot_geom` | **2.0**, 0.005, 0.0001 |

Joints: `rootx`, `rootz` (slide), `rooty` (hinge) — no damping; `thigh_joint`, `leg_joint` (range −150°–0),
`foot_joint` (±45°) — damping **1.0**, armature **1.0**, stiffness 0.

Actuators (unnamed; address by joint): `thigh_joint`, `leg_joint`, `foot_joint` — gear **200**, ctrl range [−1, 1].

### 1.4 Humanoid-v4

`timestep 0.003`, `frame_skip 5` (agent `dt 0.015`), gravity `[0, 0, −9.81]`, total mass **42.12**.

| body | mass | inertia |
|---|---|---|
| `torso` | 8.9075 | 0.1732, 0.1540, 0.0411 |
| `lwaist` | 2.2619 | 0.0099, 0.0099, 0.0037 |
| `pelvis` | 6.6162 | 0.0523, 0.0523, 0.0243 |
| `right_thigh` / `left_thigh` | 4.7518 | 0.0750, 0.0750, 0.0082 |
| `right_shin` / `left_shin` | 2.7557 | 0.0326, 0.0326, 0.0032 |
| `right_foot` / `left_foot` | 1.7671 | 0.004 ×3 |
| `right_upper_arm` / `left_upper_arm` | 1.6611 | 0.0159, 0.0159, 0.0013 |
| `right_lower_arm` / `left_lower_arm` | 1.2295 | 0.0137, 0.0137, 0.0006 |

Geoms: `floor` friction `[1.0, 0.1, 0.1]`; all body geoms `[1.0, 0.005, 0.0001]`: `torso1`, `head`, `uwaist`,
`lwaist`, `butt`, `right_thigh1`, `right_shin1`, `right_foot`, `left_thigh1`, `left_shin1`, `left_foot`,
`right_uarm1`, `right_larm`, `right_hand`, `left_uarm1`, `left_larm`, `left_hand`.

Joints (damping / armature / stiffness):

| joint(s) | damping | armature | stiffness |
|---|---|---|---|
| `root` (free) | 0 | 0 | 0 |
| `abdomen_z` | 5 | 0.02 | 20 |
| `abdomen_y`, `abdomen_x` | 5 | 0.02 | 10 |
| `right_hip_x`, `right_hip_z`, `left_hip_x`, `left_hip_z` | 5 | 0.01 | 10 |
| `right_hip_y`, `left_hip_y` | 5 | 0.008 / 0.01 | 20 |
| `right_knee` | 1 | 0.006 | 0 |
| `left_knee` | 1 | 0.006 | 1 |
| `right_shoulder1/2`, `left_shoulder1/2` | 1 | 0.007 / 0.005 | 1 |
| `right_elbow`, `left_elbow` | 1 | 0.003 | 0 |

Actuators (named after their joint), gear: `abdomen_y/z/x` 100; `right/left_hip_x`, `right/left_hip_z` 100;
`right/left_hip_y` **300**; `right/left_knee` 200; shoulders and elbows 25. Ctrl range **[−0.4, 0.4]** for all.

---

## Part 2 — Reward / termination (`env_kwargs`)

These are Gymnasium constructor arguments; they are fixed for a run. One more knob is set at `gym.make` level and
is also accepted in `env_kwargs`: `max_episode_steps` (default **1000** for all three) — the truncation limit.

### 2.1 Ant-v4

Reward per step = `healthy_reward` + `forward_reward` − `ctrl_cost` − `contact_cost`, where
`forward_reward = Δx_torso / dt` (weight fixed at 1 in v4),
`ctrl_cost = ctrl_cost_weight · Σ action²`,
`contact_cost = contact_cost_weight · Σ clip(external contact forces, contact_force_range)²`.

| argument | default | effect |
|---|---|---|
| `ctrl_cost_weight` | 0.5 | penalty on action magnitude (energy) |
| `use_contact_forces` | False | adds the 84 contact-force values to the observation **and** enables `contact_cost` |
| `contact_cost_weight` | 5e‑4 | weight of the contact penalty (only if `use_contact_forces`) |
| `contact_force_range` | (−1, 1) | clipping of contact forces before squaring |
| `healthy_reward` | 1.0 | constant bonus every step the ant is healthy |
| `healthy_z_range` | (0.2, 1.0) | torso height for "healthy"; also every state value must be finite |
| `terminate_when_unhealthy` | True | end the episode when unhealthy (otherwise just no `healthy_reward`) |
| `reset_noise_scale` | 0.1 | uniform noise on initial joint positions/velocities |
| `exclude_current_positions_from_observation` | True | drop torso x, y from the observation (27 dims instead of 29) |

### 2.2 Hopper-v4

Reward per step = `healthy_reward` + `forward_reward_weight · Δx / dt` − `ctrl_cost_weight · Σ action²`.
Healthy ⇔ all of: state values (excluding x) within `healthy_state_range`, torso z within `healthy_z_range`,
torso angle within `healthy_angle_range`.

| argument | default | effect |
|---|---|---|
| `forward_reward_weight` | 1.0 | weight of forward velocity |
| `ctrl_cost_weight` | 1e‑3 | penalty on action magnitude |
| `healthy_reward` | 1.0 | constant bonus per healthy step |
| `healthy_state_range` | (−100, 100) | bound on every state element |
| `healthy_z_range` | (0.7, ∞) | minimum torso height (falling = unhealthy) |
| `healthy_angle_range` | (−0.2, 0.2) rad | torso pitch allowed before "fallen" (≈ ±11.5°) |
| `terminate_when_unhealthy` | True | terminate vs. continue without the healthy bonus |
| `reset_noise_scale` | 5e‑3 | initial-state noise |
| `exclude_current_positions_from_observation` | True | drop x from the observation (11 dims instead of 12) |

### 2.3 Humanoid-v4

Reward per step = `healthy_reward` + `forward_reward_weight · Δx_com / dt` − `ctrl_cost_weight · Σ action²`
(v4 has no contact cost; the 376‑dim observation does include contact forces).

| argument | default | effect |
|---|---|---|
| `forward_reward_weight` | 1.25 | weight of the centre-of-mass forward velocity |
| `ctrl_cost_weight` | 0.1 | penalty on action magnitude |
| `healthy_reward` | 5.0 | constant bonus per healthy step (dominates early reward) |
| `healthy_z_range` | (1.0, 2.0) | torso height for "healthy" (fallen or launched = unhealthy) |
| `terminate_when_unhealthy` | True | terminate vs. continue without the healthy bonus |
| `reset_noise_scale` | 1e‑2 | initial-state noise |
| `exclude_current_positions_from_observation` | True | drop x, y from the observation (376 dims instead of 378) |

### 2.4 Things that change the task but are not plain kwargs

- **Episode length**: `max_episode_steps` (above).
- **Reward scaling / shaping** and **observation or action noise**: not constructor arguments; they would be
  Gymnasium wrappers (`TransformReward`, custom noise wrappers). Not implemented yet — say if you want them.
- **Geometry** (limb lengths, extra obstacles, slopes): needs a modified XML (`xml_file` kwarg). Not covered by
  `env_mods`.
