"""The incremental, speculative execution engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import (
    Context,
    SimulationAbort,
    Strategy,
    StrategyKey,
    set_context,
)
from .estimators import ESTIMATORS, Draws
from .runner import Runner
from .utils import order_instances, default_family


@dataclass
class Snapshot:
    """One row of the execution log -- everything a calibration study needs."""

    runs: int
    output: Any
    likelihood: float
    state_depth: int
    selected: Optional[str]
    impact: float


@dataclass
class Result:
    output: Any
    likelihood: float
    runs: int
    wasted_runs: int
    sequential_runs: int
    history: List[Snapshot] = field(default_factory=list)

    @property
    def speedup(self) -> float:
        return self.sequential_runs / max(self.runs, 1)

    def __str__(self) -> str:
        return (
            f"{self.output!r}  (likelihood {self.likelihood:.0%}, "
            f"{self.runs} runs vs {self.sequential_runs} sequential, "
            f"{self.wasted_runs} wasted)"
        )


class Engine:
    def __init__(
        self,
        experiment: Callable[[], Any],
        instances: Sequence[str],
        *,
        estimator: str = "paired",
        nsims: int = 400,
        nsims_impact: int | None = None,
        nruns: int = 10,
        batch: int = 10,
        percentile: float = 0.05,
        tolerance: float = 0.0,
        confidence: float = 1.0,
        min_state_likelihood: float = 0.10,
        n_jobs: int = 1,
        timeout: float | None = None,
        seed: int = 0,
        verbose: bool = False,
        instance_order: str = "stratified",
        family=default_family,
        cost_aware: bool = False,
        parse=None,
    ):
        self.experiment = experiment
        instances = order_instances(instances, instance_order, family, seed)
        self.runner = Runner(instances, n_jobs=n_jobs, timeout=timeout, parse=parse)
        self.cost_aware = cost_aware
        self.n_instances = len(instances)
        self.estimator = ESTIMATORS[estimator]()
        self.nsims = nsims
        self.nsims_impact = nsims_impact or max(40, nsims // 4)
        self.nruns = nruns
        self.batch = batch
        self.percentile = percentile
        self.tolerance = tolerance
        self.confidence = confidence
        self.min_state_likelihood = min_state_likelihood
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.history: List[Snapshot] = []
        self._path_runs = 0

    # ------------------------------------------------------------------ #
    # Hooks called from core.best_strategy
    # ------------------------------------------------------------------ #

    def ensure_minimum_data(self, s: Strategy) -> None:
        missing = self.nruns - self.runner.count(s)
        if missing > 0:
            self.runner.run(s, missing)

    def compare_observed(self, s1: Strategy, s2: Strategy) -> Strategy:
        """Compare on the instances both have actually run (a paired prefix)."""
        v1, v2 = self.runner.values(s1), self.runner.values(s2)
        c = min(len(v1), len(v2))
        if c == 0:
            return s1
        d = float(np.mean(v1[:c]) - np.mean(v2[:c]))
        return s1 if d >= -self.tolerance else s2

    # ------------------------------------------------------------------ #
    # Executions of the user's experiment function
    # ------------------------------------------------------------------ #

    def predictive_execution(self, report: bool = False):
        """Run the experiment against the data gathered so far."""
        ctx = Context(mode="predict", engine=self, report=report)
        set_context(ctx)
        try:
            output = self.experiment()
        finally:
            set_context(None)
        if output is None:
            output = ctx.state()
        return output, ctx

    def _simulate_once(self, sums: Dict[StrategyKey, float]):
        ctx = Context(mode="simulate", engine=self, sums=sums)
        set_context(ctx)
        try:
            output = self.experiment()
        except SimulationAbort:
            return None, ctx
        finally:
            set_context(None)
        if output is None:
            output = ctx.state()
        return output, ctx

    # ------------------------------------------------------------------ #
    # Likelihoods
    # ------------------------------------------------------------------ #

    def _draws(self, keys) -> Draws:
        evals = {k: self.runner.evals.get(k, []) for k in keys}
        evals = {k: v for k, v in evals.items() if v}
        return self.estimator.draws(evals, self.n_instances, self.nsims, self.rng)

    def _run_simulations(self, draws: Draws, override=None, budget=None):
        """Return (states reached, outputs reached) over all simulations."""
        states, outputs = [], []
        for sim in range(min(budget or self.nsims, draws.sums.shape[1])):
            sums = draws.as_dict(sim)
            if override is not None:
                sums[override[0]] = override[1]
            out, ctx = self._simulate_once(sums)
            if out is None:
                continue
            states.append(ctx.state())
            outputs.append(_hashable(out))
        return states, outputs

    @staticmethod
    def _prefix_likelihood(states, target) -> float:
        if not states:
            return 0.0
        d = len(target)
        hits = sum(1 for s in states if s[:d] == target)
        return hits / len(states)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self) -> Result:
        while True:
            output, ctx = self.predictive_execution()
            involved = list(ctx.seen)
            keys = [s.key for s in involved]

            draws = self._draws(keys)
            states, outputs = self._run_simulations(draws)
            target_output = _hashable(output)
            likelihood = (
                sum(1 for o in outputs if o == target_output) / len(outputs)
                if outputs
                else 0.0
            )

            # deepest checkpoint we still believe in (sec. 4.2.4 of the paper,
            # but expressed as a prefix of the decision sequence)
            probable_state = ()
            for cp in ctx.checkpoints:
                if self._prefix_likelihood(states, cp) >= self.min_state_likelihood:
                    probable_state = cp
                else:
                    break
            base = self._prefix_likelihood(states, probable_state) or 1e-9

            # --- impact of each strategy on that state --------------------- #
            # Score = (impact, swing).  Impact saturates at 1.0 for every
            # strategy able to knock the probable state down to zero -- which
            # early on is most of them -- so the swing |L+ - L-| breaks ties by
            # how much the strategy can actually move the likelihood, instead
            # of by the arbitrary order in which strategies appear in the code.
            # With cost_aware=True both are divided by the strategy's mean
            # wall time: impact per second, not per execution.
            best_s, best_score, best_impact = None, (-1.0, -1.0), 0.0
            for s in involved:
                if self.runner.is_complete(s) or s.key not in draws.keys:
                    continue
                lo = draws.quantile(s.key, self.percentile)
                hi = draws.quantile(s.key, 1 - self.percentile)
                l_minus, _ = self._run_simulations(
                    draws, override=(s.key, lo), budget=self.nsims_impact)
                l_plus, _ = self._run_simulations(
                    draws, override=(s.key, hi), budget=self.nsims_impact)
                lm = self._prefix_likelihood(l_minus, probable_state)
                lp = self._prefix_likelihood(l_plus, probable_state)
                impact = max(0.0, min(1.0, 1.0 - min(lm, lp) / base))
                swing = abs(lp - lm)
                cost = self.runner.mean_seconds(s) if self.cost_aware else 0.0
                scale = 1.0 / cost if cost > 0 else 1.0
                score = (round(impact * scale, 6), swing * scale)
                if score > best_score:
                    best_s, best_score, best_impact = s, score, impact

            self.history.append(
                Snapshot(
                    runs=self.runner.total_runs,
                    output=output,
                    likelihood=likelihood,
                    state_depth=len(probable_state),
                    selected=repr(best_s) if best_s else None,
                    impact=best_impact if best_s else 0.0,
                )
            )
            if self.verbose:
                print(
                    f"[{self.runner.total_runs:>6} runs] L={likelihood:5.1%} "
                    f"depth={len(probable_state)} -> {best_s} (I={best_impact:.2f})"
                )

            if best_s is None or likelihood >= self.confidence:
                # either everything on the path is fully evaluated (exact), or
                # the user asked to stop at a given degree of certainty
                needed = self._sequential_runs(involved)
                return Result(
                    output=output,
                    likelihood=1.0 if best_s is None else likelihood,
                    runs=self.runner.total_runs,
                    wasted_runs=self.runner.total_runs - needed,
                    sequential_runs=needed,
                    history=self.history,
                )
            self.runner.run(best_s, self.batch)

    def _sequential_runs(self, involved: List[Strategy]) -> int:
        """How many runs a plain sequential execution would have needed."""
        return len(involved) * self.n_instances


def _hashable(x):
    try:
        hash(x)
        return x
    except TypeError:
        return repr(x)


def experiment_execution(experiment, instances, **kw) -> Result:
    """Run `experiment` incrementally over `instances`.

    `instances` may be a list, or the path to a file with one instance per line.
    """
    if isinstance(instances, str):
        with open(instances) as fh:
            instances = [ln.strip() for ln in fh if ln.strip()]
    return Engine(experiment, instances, **kw).run()
