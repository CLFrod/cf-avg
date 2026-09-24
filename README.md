# Action Value Gradient Algorithm

This repo provides an implementation of the following incremental learning algorithms:
- Action Value Gradient (AVG)
- Incremental One-Step Actor-Critic (IAC)
- Incremental Soft Actor Critic (SAC-1)

## Setup Instructions:
This project has been modified to make use of [UV's dependency management tooling](https://docs.astral.sh/uv/). 

1. Ensure you have UV installed and setup correctly by following their **[startup guide](https://docs.astral.sh/uv/getting-started/).**
2. Once UV is installed, install all of the dependencies by using this command in the project folder: 
```bash
uv sync
```
3. Done! All of the dependencies should be installed automatically. 

### Adding new packages/dependencies:
To add new dependencies, you must run this command:
```bash
uv add <package_name>
```

### Activating the virtual environment
After running `uv sync` or adding a dependency, activate the generated `.venv` before running the project.

On Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:
```bat
.venv\Scripts\activate.bat
```

On macOS or Linux:
```bash
source .venv/bin/activate
```

Your terminal prompt should begin with `(cf-avg) or (.venv)`.


## Running the Simulation: 
```python
python avg.py --env "Humanoid-v4" --N 10001000
```

## Learned Behavior in Simulation
<!-- ![AVG](assets/AVG.gif){width=200px height=200px} -->

<img src="assets/AVG.gif" width="320" height="240" alt="Description">

### Hyper-parameters used in the paper

| hyp_seed |                Envs                 | actor_lr | critic_lr | beta1 |    betas     | alpha_lr | gamma |
|----------|-------------------------------------|----------|-----------|-------|--------------|----------|-------|
|   122    |       Hopper-v4, Walker2d-v4        | 1.1e-05  |  7.7e-05  |  0.0  | [0.0, 0.999] |   0.3    | 0.99  |
|   129    | Ant-v4, HalfCheetah-v4, Humanoid-v4 |  0.0063  |  0.0087   |  0.0  | [0.0, 0.999] |   0.07   | 0.99  |
|    12    |            reacher_hard             |  3e-06   |  0.0049   |  0.0  | [0.0, 0.999] |   0.05   | 0.97  |
|    57    |    dog_walk, dog_trot, dog_stand    |  6e-06   |   8e-05   |  0.0  | [0.0, 0.999] |  0.009   | 0.95  |
|   145    |             finger_spin             | 0.00038  |  8.7e-05  |  0.9  | [0.9, 0.999] |  0.006   | 0.95  |
|   223    |               dog_run               | 1.8e-05  |  4.8e-05  |  0.0  | [0.0, 0.999] |  0.007   | 0.97  |


## Robot Tasks

| ![UR-Reacher-2](assets/UR-Reacher-2.gif) <br> UR-Reacher-2 | ![Create-Mover](assets/Create-Mover.gif) <br /> Create-Mover |
| --- | --- |


## Hyper-parameter search
*AVG*
```
cd incremental_rl
python hyp_sweep.py --algo "avg" --hyp_seed 122 --env "Hopper-v4" --N 10001000 --n_seeds 10
python replicate_run.py --algo "avg_norm_obs_scaled_td" --hyp_seed 129 --env "Ant-v4" --N 10001000
```

*Incremental Actor Critic*
```
cd incremental_rl
python hyp_sweep.py --algo "iac" --hyp_seed 122 --env "Hopper-v4" --N 10001000 --n_seeds 10
python replicate_run.py --algo "iac_all" --hyp_seed 294 --env "Hopper-v4" --N 10001000
```

*Incremental Soft Actor Critic*
```
cd incremental_rl
python hyp_sweep.py --algo "isac" --hyp_seed 146 --env "HalfCheetah-v4" --N 10001000 
```

## Cite
```bash
@inproceedings{vasan2024deep,
  title={Deep Policy Gradient Methods Without Batch Updates, Target Networks, or Replay Buffers},
  author={Vasan, Gautham and Elsayed, Mohamed and Azimi, Seyed Alireza and He, Jiamin and Shahriar, Fahim and Bellinger, Colin and White, Martha and Mahmood, A Rupam},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}

```

```
Vasan, G., Elsayed, M., Azimi, S. A., He, J., Shahriar, F., Bellinger, C., White, M., & Mahmood, A. R. (2024). Deep Policy Gradient Methods Without Batch Updates, Target Networks, or Replay Buffers. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.
```
