"""Execution of strategies on instances, with caching and parallelism.

Instances are kept in a FIXED order.  Every strategy always evaluates the
prefix instances[0:c].  That is not cosmetic: it means any two strategies
share their first min(c1, c2) instances, which is what makes the paired
estimator in `estimators.py` possible.  Never shuffle per-strategy.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Sequence

from .core import Strategy, StrategyKey


class RunError(RuntimeError):
    pass


class Runner:
    """Evaluates (strategy, instance) pairs, never twice."""

    def __init__(
        self,
        instances: Sequence[str],
        n_jobs: int = 1,
        timeout: float | None = None,
        parse: Callable[[str], float] | None = None,
    ):
        self.instances = list(instances)
        self.n_jobs = max(1, n_jobs)
        self.timeout = timeout
        self.parse = parse or _parse_last_float
        self.evals: Dict[StrategyKey, List[float]] = {}
        self.seconds: Dict[StrategyKey, float] = {}   # wall time per strategy
        self.total_runs = 0
        self.wall_time = 0.0

    # -- introspection ----------------------------------------------------- #

    def count(self, s: Strategy) -> int:
        return len(self.evals.get(s.key, ()))

    def values(self, s: Strategy) -> List[float]:
        return self.evals.get(s.key, [])

    def mean_seconds(self, s: Strategy) -> float:
        """Average wall time of one execution of `s` (0 if never run)."""
        c = self.count(s)
        return self.seconds.get(s.key, 0.0) / c if c else 0.0

    def is_complete(self, s: Strategy) -> bool:
        return self.count(s) >= len(self.instances)

    # -- execution --------------------------------------------------------- #

    def run(self, s: Strategy, k: int) -> int:
        """Run `s` on the next k unevaluated instances. Returns runs performed."""
        c = self.count(s)
        todo = self.instances[c : c + k]
        if not todo:
            return 0
        t0 = time.time()
        if self.n_jobs == 1 or len(todo) == 1:
            out = [self._evaluate(s, pi) for pi in todo]
        else:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
                out = list(pool.map(lambda pi: self._evaluate(s, pi), todo))
        dt = time.time() - t0
        self.wall_time += dt
        self.seconds[s.key] = self.seconds.get(s.key, 0.0) + dt
        self.evals.setdefault(s.key, []).extend(out)
        self.total_runs += len(out)
        return len(out)

    def _evaluate(self, s: Strategy, instance: str) -> float:
        if callable(s.command):
            return float(s.command(instance, **s.theta))
        cmd = s.command.format(INSTANCE=instance, **s.theta)
        if "{INSTANCE}" not in s.command and instance not in cmd:
            cmd = f"{cmd} {instance}"
        proc = subprocess.run(
            shlex.split(cmd),
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise RunError(f"{cmd!r} exited {proc.returncode}: {proc.stderr[:300]}")
        return self.parse(proc.stdout)


def _parse_last_float(stdout: str) -> float:
    for token in reversed(stdout.split()):
        try:
            return float(token)
        except ValueError:
            continue
    raise RunError(f"no numeric output found in: {stdout[:200]!r}")
