"""Research Harness eval: train the subject's RL agent on Gymnasium benchmarks and score it.

The eval owns the environments (this Gymnasium checkout), the training budget, the seeds and the
scoring; the subject only supplies a training function (benchmarks.toml [agent] entry, default
`rl_agent:train`, see EVAL.md). For each benchmark and seed the function is called in its own
process with a step-counted env factory, and the policy it returns is evaluated on fresh, seeded
episodes. A benchmark's normalized score is (return - random policy's return) / (target - random),
so 0 is a random policy (or worse, or a failed run) and 1 reaches the target; `score` averages it
over the benchmarks.
The last line printed is the result.

Local run: HARNESS_SUBJECTS_DIR=/path/to/subjects python rl_eval.py   (RL_EVAL_SCALE=0.1 for a
quick pass with a tenth of each budget).
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "benchmarks.toml"
EVAL_SEED_BASE = 1_000_003  # evaluation episodes never reuse a training seed
BUDGET_SLACK = 0.05  # rollout-based agents may finish their last batch a little past the budget
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def subjects_dir() -> Path:
    return Path(os.environ.get("HARNESS_SUBJECTS_DIR") or HERE.parent.parent / "subjects")


def eval_seeds(seed: int, episodes: int) -> list[int]:
    return [EVAL_SEED_BASE + 1000 * seed + ep for ep in range(episodes)]


# --- config ----------------------------------------------------------------------------------------

def load_config() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import gymnasium as gym

    cfg = tomllib.loads(CONFIG.read_text())
    run = cfg.get("run", {})
    scale = float(os.environ.get("RL_EVAL_SCALE", "1"))
    benches = []
    for b in cfg.get("benchmark", []):
        spec = gym.spec(b["env"])  # an unknown id fails here, before anything trains
        target = b.get("target", spec.reward_threshold)
        if target is None:
            raise SystemExit(f"{b['env']} has no reward_threshold; give it a `target` in benchmarks.toml")
        benches.append({
            "env": b["env"], "kwargs": b.get("kwargs", {}), "target": float(target),
            "timesteps": max(1, int(int(b["timesteps"]) * scale)),
            "seeds": int(b.get("seeds", run.get("seeds", 3))),
            "eval_episodes": int(b.get("eval_episodes", run.get("eval_episodes", 10))),
        })
    if not benches:
        raise SystemExit("benchmarks.toml lists no [[benchmark]]")
    return cfg, benches


def find_agent(entry: str) -> tuple[Path | None, Path | None]:
    """(subject folder, import root) of the module in `entry`; (None, None) if it's already importable."""
    module = entry.split(":")[0]
    rel = module.replace(".", "/")
    for subject in sorted(p for p in subjects_dir().glob("*") if p.is_dir()):
        for base in (subject, subject / "src"):
            if (base / f"{rel}.py").is_file() or (base / rel / "__init__.py").is_file():
                return subject, base
    if importlib.util.find_spec(module.split(".")[0]) is not None:
        return None, None
    raise SystemExit(f"no subject under {subjects_dir()} has {rel}.py or {rel}/__init__.py "
                     f"(benchmarks.toml [agent] entry = {entry!r})")


def install_subject(subject: Path) -> None:
    if (subject / "requirements.txt").is_file():
        args = ["-r", str(subject / "requirements.txt")]
    elif (subject / "pyproject.toml").is_file() or (subject / "setup.py").is_file():
        args = ["-e", str(subject)]
    else:
        return
    print(f"installing the subject's dependencies: pip install {' '.join(args)}", flush=True)
    # This checkout of Gymnasium stays the one imported (it is first on sys.path) even if the
    # subject's dependencies pull in another version.
    subprocess.run([sys.executable, "-m", "pip", "install", "--progress-bar", "off", *args], check=True)


# --- one training run (child process) ----------------------------------------------------------------

class BudgetExceeded(RuntimeError):
    pass


class Counter:
    def __init__(self, budget: int):
        self.budget, self.steps, self.limit = budget, 0, int(budget * (1 + BUDGET_SLACK))

    def add(self, n: int) -> None:
        self.steps += n
        if self.steps > self.limit:
            raise BudgetExceeded(f"the agent took more than {self.limit} env steps (budget {self.budget})")


def count_steps(counter: Counter) -> None:
    """Count every env step the agent takes in this process: envs from gymnasium.make (and so from
    the factory it is given), make_vec, and AsyncVectorEnv (counted here, where step results land)."""
    import gymnasium as gym
    from gymnasium.envs import registration
    from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorWrapper

    class StepCounter(gym.Wrapper):
        def step(self, action):
            counter.add(1)
            return self.env.step(action)

    class VecStepCounter(VectorWrapper):
        def step(self, actions):
            counter.add(self.num_envs)
            return self.env.step(actions)

    make, make_vec, step_wait = registration.make, registration.make_vec, AsyncVectorEnv.step_wait

    def counted_make(*args, **kwargs):
        return StepCounter(make(*args, **kwargs))

    def counted_make_vec(*args, **kwargs):
        env = make_vec(*args, **kwargs)  # sync sub-envs come from counted_make; async from step_wait
        return env if isinstance(env, (SyncVectorEnv, AsyncVectorEnv)) else VecStepCounter(env)

    def counted_step_wait(self, *args, **kwargs):
        out = step_wait(self, *args, **kwargs)
        counter.add(self.num_envs)
        return out

    gym.make = registration.make = counted_make
    gym.make_vec = registration.make_vec = counted_make_vec
    AsyncVectorEnv.step_wait = counted_step_wait

    def restore() -> None:
        gym.make = registration.make = make
        gym.make_vec = registration.make_vec = make_vec
        AsyncVectorEnv.step_wait = step_wait

    count_steps.restore = restore  # type: ignore[attr-defined]


def act_fn(policy: Any):
    """The policy's obs -> action function: SB3-style predict(), act(), or the policy itself."""
    if hasattr(policy, "predict"):
        det = "deterministic" in inspect.signature(policy.predict).parameters

        def act(obs):
            out = policy.predict(obs, deterministic=True) if det else policy.predict(obs)
            return out[0] if isinstance(out, tuple) else out
        return act
    if hasattr(policy, "act"):
        return policy.act
    if callable(policy):
        return policy
    raise TypeError(f"train() returned {type(policy).__name__}, which has no predict() or act() and isn't callable")


def to_action(a: Any, space: Any) -> Any:
    import numpy as np
    from gymnasium import spaces

    if isinstance(space, spaces.Discrete):
        return int(np.asarray(a).reshape(-1)[0])
    if isinstance(space, spaces.Box):
        return np.asarray(a, dtype=space.dtype).reshape(space.shape)
    return a


def episode_returns(act, env_id: str, kwargs: dict[str, Any], seeds: list[int], random_policy: bool = False) -> list[float]:
    import gymnasium as gym

    env = gym.make(env_id, **kwargs)
    out = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        env.action_space.seed(s)
        total, done = 0.0, False
        while not done:
            a = env.action_space.sample() if random_policy else to_action(act(obs), env.action_space)
            obs, r, term, trunc, _ = env.step(a)
            total += float(r)
            done = term or trunc
        out.append(total)
    env.close()
    return out


def child(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text())
    import numpy as np

    if spec["import_root"]:
        sys.path.insert(1, spec["import_root"])  # after this folder, so Gymnasium is this checkout
    counter = Counter(spec["timesteps"])
    count_steps(counter)
    module, _, name = spec["entry"].partition(":")
    train = getattr(importlib.import_module(module), name or "train")
    import gymnasium as gym

    random.seed(spec["seed"])
    np.random.seed(spec["seed"])
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(spec["seed"])

    def make_env(**overrides):
        return gym.make(spec["env"], **{**spec["kwargs"], **overrides})

    args = {"make_env": make_env, "env_id": spec["env"], "total_timesteps": spec["timesteps"], "seed": spec["seed"]}
    params = inspect.signature(train).parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        args = {k: v for k, v in args.items() if k in params}
    t0 = time.monotonic()
    policy = train(**args)
    train_s = time.monotonic() - t0
    count_steps.restore()  # type: ignore[attr-defined]
    returns = episode_returns(act_fn(policy), spec["env"], spec["kwargs"], eval_seeds(spec["seed"], spec["eval_episodes"]))
    if not all(math.isfinite(r) for r in returns):
        raise FloatingPointError(f"non-finite evaluation return {returns}")
    Path(spec["out"]).write_text(json.dumps({"returns": returns, "train_steps": counter.steps, "train_s": train_s}))


# --- the benchmark (parent process) ------------------------------------------------------------------

def run_one(bench: dict[str, Any], seed: int, entry: str, import_root: Path | None, threads: int,
            timeout: float | None) -> dict[str, Any]:
    label = f"{bench['env']} seed {seed}"
    work = Path(tempfile.mkdtemp(prefix="rl-eval-"))
    spec = {**bench, "seed": seed, "entry": entry, "import_root": str(import_root) if import_root else None,
            "out": str(work / "result.json")}
    (work / "spec.json").write_text(json.dumps(spec))
    env = dict(os.environ)
    for v in THREAD_VARS:
        env.setdefault(v, str(threads))
    log = work / "log.txt"
    t0 = time.monotonic()
    with open(log, "w") as f:
        p = subprocess.Popen([sys.executable, str(HERE / "rl_eval.py"), "--run", str(work / "spec.json")],
                             cwd=work, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            code = None
    result = json.loads((work / "result.json").read_text()) if code == 0 else None
    if result is not None:
        print(f"[{label}] mean return {statistics.fmean(result['returns']):.1f} "
              f"(trained {result['train_s']:.0f}s, {result['train_steps']} steps)", flush=True)
        return {"ok": True, **result}
    lines = log.read_text(errors="replace").strip().splitlines()
    why = f"timed out after {timeout:.0f}s" if code is None else (lines[-1] if lines else f"exit code {code}")
    print(f"[{label}] FAILED after {time.monotonic() - t0:.0f}s: {why}", flush=True)
    for line in lines[-30:]:
        print(f"  | {line}", flush=True)
    return {"ok": False, "error": f"{label}: {why}"[:500], "train_steps": 0}


def main() -> None:
    t0 = time.time()
    import gymnasium as gym

    print(f"gymnasium {gym.__version__} from {Path(gym.__file__).parent}", flush=True)
    cfg, benches = load_config()
    agent, run = cfg.get("agent", {}), cfg.get("run", {})
    entry = agent.get("entry", "rl_agent:train")
    subject, import_root = find_agent(entry)
    print(f"agent: {entry}" + (f" from {import_root}" if import_root else " (installed)"), flush=True)
    if subject and agent.get("install", True):
        install_subject(subject)

    runs = [(b, seed) for b in benches for seed in range(b["seeds"])]
    workers = int(os.environ.get("RL_EVAL_WORKERS") or run.get("workers") or 0) or os.cpu_count() or 1
    workers = max(1, min(workers, len(runs)))
    threads = max(1, (os.cpu_count() or 1) // workers)
    timeout = float(run.get("run_timeout_s") or 0) or None
    print(f"{len(runs)} training runs, {workers} at a time", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        outs = list(pool.map(lambda r: run_one(r[0], r[1], entry, import_root, threads, timeout), runs))

    results, errors, norms, solved, it = [], [], [], 0, iter(outs)
    for b in benches:
        per_seed = [next(it) for _ in range(b["seeds"])]
        rnd = statistics.fmean(episode_returns(None, b["env"], b["kwargs"], [
            s for seed in range(b["seeds"]) for s in eval_seeds(seed, b["eval_episodes"])], random_policy=True))
        span = b["target"] - rnd
        means = [statistics.fmean(o["returns"]) for o in per_seed if o["ok"]]
        # Worse than random counts as random, the same as a failed run, so one diverged seed
        # can't outweigh the other benchmarks.
        seed_norms = [max(0.0, (statistics.fmean(o["returns"]) - rnd) / span) if o["ok"] else 0.0 for o in per_seed]
        norm = statistics.fmean(seed_norms)
        norms.append(norm)
        errors += [o["error"] for o in per_seed if not o["ok"]]
        ok = len(means) == len(per_seed) and statistics.fmean(means) >= b["target"]
        solved += ok
        if means:
            spread = statistics.stdev(means) if len(means) > 1 else 0.0
            ret = f"return {statistics.fmean(means):.1f} ± {spread:.1f} ({len(means)}/{len(per_seed)} seeds)"
        else:
            ret = f"no successful runs (0/{len(per_seed)} seeds)"
        results.append(f"{b['env']}: {ret}, normalized {norm:.3f} (random {rnd:.1f}, target {b['target']:g}, "
                       f"{b['timesteps']} steps){', solved' if ok else ''}")
    for line in results:
        print(line, flush=True)
    result = {"score": statistics.fmean(norms), "solved": solved, "benchmarks": len(benches), "results": results,
              "failed_runs": len(errors), "train_steps": sum(o["train_steps"] for o in outs),
              "wall_s": time.time() - t0}
    if errors:
        result["errors"] = errors
    print(json.dumps(result))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--run":
        child(sys.argv[2])
    else:
        main()
