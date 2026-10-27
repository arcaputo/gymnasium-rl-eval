# Gymnasium RL benchmark eval (Research Harness)

This is an unmodified checkout of [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) with
six files added at the top level to make it a Research Harness eval:

| File | What it is |
|---|---|
| `eval.toml` | How the harness runs it (`python rl_eval.py`) and the metrics it reports |
| `job.toml` | What it runs on: `python:3.12-slim`, 8+ CPUs, 16+ GB, 2 h |
| `requirements.txt` | Installs this checkout of Gymnasium (with the Box2D extra) |
| `benchmarks.toml` | Which environments, training budgets, seeds and targets. Edit this to change the benchmarks |
| `rl_eval.py` | The eval |
| `EVAL.md` | This file |

## What the subject provides

A function the eval calls once per benchmark and seed. By default it is `train` in `rl_agent.py`
in any subject (`subjects/<name>/rl_agent.py` or `subjects/<name>/src/rl_agent.py`); change
`[agent] entry` in `benchmarks.toml` to point elsewhere (`"my_pkg.agents:train_ppo"`).

```python
def train(make_env, env_id, total_timesteps, seed):
    """Train on the benchmark and return the policy."""
```

The function may take any subset of these keyword arguments:

- `make_env(**overrides)`: a new environment for the benchmark (same as `gymnasium.make(env_id)`
  with the benchmark's kwargs). Wrap it in `gymnasium.vector.SyncVectorEnv` /
  `AsyncVectorEnv` for vectorized training.
- `env_id`: the Gymnasium id, for libraries that make their own envs (`PPO("MlpPolicy", env_id)`).
- `total_timesteps`: the training budget in environment steps.
- `seed`: the run's seed. Python, NumPy and (if imported) PyTorch are already seeded with it.

It returns the trained policy, as any of: an object with a Stable-Baselines3-style
`predict(obs, deterministic=True)`, an object with `act(obs)`, or a plain `obs -> action` callable.
A Stable-Baselines3 subject is two lines:

```python
from stable_baselines3 import PPO

def train(env_id, total_timesteps, seed):
    return PPO("MlpPolicy", env_id, seed=seed).learn(total_timesteps)
```

The subject's dependencies are installed before the eval runs (its `requirements.txt`, else its
`pyproject.toml` / `setup.py`; turn off with `[agent] install = false`). Gymnasium itself is
always this checkout, whatever version the subject's dependencies ask for.

## How it scores

- Each (benchmark, seed) trains in its own process, in parallel (`[run] workers`), with the thread
  count per process set so they share the machine.
- Every env step the agent takes is counted, whichever way it makes its envs (the factory,
  `gymnasium.make`, `make_vec`, sync or async vector envs). A run that goes more than 5% past its
  budget is stopped and fails.
- The returned policy is evaluated on `eval_episodes` fresh episodes with fixed seeds the agent
  never trained on.
- **Normalized score** per seed = (mean return − random policy's return) / (target − random),
  floored at 0. The target is the env's `reward_threshold` unless `benchmarks.toml` sets one.
  A failed run (crash, timeout, budget overrun) scores 0.
- `score` (primary) is the mean over benchmarks; `solved` counts benchmarks where every seed ran
  and the mean return reached the target. `results` has one line per benchmark.

## Running it locally

```bash
pip install -r requirements.txt
HARNESS_SUBJECTS_DIR=/path/to/subjects RL_EVAL_SCALE=0.1 python rl_eval.py
```

`RL_EVAL_SCALE` multiplies every budget (for a quick check); `RL_EVAL_WORKERS` overrides `workers`.
