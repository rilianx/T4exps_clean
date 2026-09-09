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
    alive: int = 0            # simulations that completed (all of them when imputing)


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
        honest_likelihood: bool = True,
        impute_missing: bool = True,
        impute_prior_scale: float = 1.0,
        nsims_confirm: int | None = None,
        impact_on: str = "auto",
        impute_prior: str = "mean",
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
        self.seed = seed
        # Both default on.  honest_likelihood: an aborted simulation counts as
        # 'did not reproduce the output' instead of being dropped -- dropping
        # them conditions on agreement and 1/1 = 100%.  impute_missing: a
        # simulation that reaches a strategy without data draws its total from
        # a prior instead of aborting, so unexplored branches count as
        # uncertainty and the impact heuristic can see them.  Both False +
        # estimator='paired_legacy' reproduces the original engine.
        self.honest_likelihood = honest_likelihood
        self.impute_missing = impute_missing
        # impute_prior_scale multiplica la sd del prior de imputacion (sd de los
        # totales conocidos en esa simulacion): >1 explora mas, <1 menos.
        self.impute_prior_scale = impute_prior_scale
        # nsims_confirm: cuando la likelihood queda a menos de 0.05 del umbral,
        # se re-estima con este numero de simulaciones antes de decidir parar.
        # Con 400 el error MC es +-0.7 pp, del orden de la distancia al 0.98.
        self.nsims_confirm = nsims_confirm
        # impact_on: 'prefix' mide el impacto de una estrategia sobre la
        # likelihood del prefijo de decisiones creido (original); 'output' lo
        # mide sobre la likelihood del OUTPUT, que es lo que la regla de parada
        # certifica -- asi no se gasta en decisiones que no cambian la respuesta.
        # DEFAULT 'auto' desde 0.3.0 (A/B: 10/10 correctas, -12% vs 'prefix', sin
        # corridas a ciegas). 'prefix' reproduce 0.2.
        # 'auto': output cuando la likelihood ya es informativa (>= 0.05, o
        # sea >= ~20 simulaciones reproducen el output), prefijo si no -- con
        # L = 0 el criterio por output degenera (todo impacto satura en 1.0 y
        # gana el orden de codigo), que es exactamente lo que se ve en las
        # corridas con cientos de iteraciones en L = 0.
        if impact_on not in ("prefix", "output", "auto"):
            raise ValueError("impact_on must be 'prefix', 'output' or 'auto'")
        self.impact_on = impact_on
        # impute_prior: centro del prior para una estrategia SIN datos.
        # 'mean' (0.2/0.3): la media de los totales conocidos -- asume que lo no
        # explorado es promedio, y como el titular es la mejor conocida, la rama
        # inexplorada casi siempre pierde en las simulaciones: L se infla sin
        # haberla verificado (candidato a D5).  'best': el mejor total conocido
        # -- un vecino no explorado es tan bueno como el lider hasta que se
        # demuestre lo contrario; fuerza a explorar antes de certificar.
        if impute_prior not in ("mean", "best"):
            raise ValueError("impute_prior must be 'mean' or 'best'")
        self.impute_prior = impute_prior
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

    ABORT = ("__abort__",)          # never equals a real output or checkpoint

    def _run_simulations(self, draws: Draws, override=None, budget=None):
        """Return (states reached, outputs reached) over all simulations."""
        states, outputs = [], []
        for sim in range(min(budget or self.nsims, draws.sums.shape[1])):
            sums = draws.as_dict(sim)
            if override is not None:
                sums[override[0]] = override[1]
            if self.impute_missing:
                col = draws.sums[:, sim]
                mu = float(col.max() if self.impute_prior == "best" else col.mean())
                sd = float(col.std()) if col.size > 1 else 0.0
                sums = _ImputingSums(sums, np.random.default_rng(self.seed * 100003 + sim),
                                     mu, max(sd * self.impute_prior_scale, 1e-9))
            out, ctx = self._simulate_once(sums)
            if out is None:
                if self.honest_likelihood:
                    states.append(self.ABORT)
                    outputs.append(self.ABORT)
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
            if (self.nsims_confirm and self.nsims_confirm > self.nsims
                    and likelihood >= self.confidence - 0.05):
                # cerca del umbral: la decision de parar no debe depender del
                # ruido MC de nsims. Se re-estima con mas simulaciones (mismo
                # cache, otros sorteos) y esa es la likelihood que vale.
                draws_c = self.estimator.draws(
                    {k: self.runner.evals.get(k, []) for k in keys if self.runner.evals.get(k)},
                    self.n_instances, self.nsims_confirm, self.rng)
                _, outputs_c = self._run_simulations(draws_c, budget=self.nsims_confirm)
                likelihood = (sum(1 for o in outputs_c if o == target_output) / len(outputs_c)
                              if outputs_c else 0.0)

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
                l_minus, o_minus = self._run_simulations(
                    draws, override=(s.key, lo), budget=self.nsims_impact)
                l_plus, o_plus = self._run_simulations(
                    draws, override=(s.key, hi), budget=self.nsims_impact)
                use_output = self.impact_on == "output" or (self.impact_on == "auto" and likelihood >= 0.05)
                if use_output:
                    frac = lambda outs: (sum(1 for o in outs if o == target_output) / len(outs)) if outs else 0.0
                    lm, lp = frac(o_minus), frac(o_plus)
                    base_i = likelihood or 1e-9
                else:
                    lm = self._prefix_likelihood(l_minus, probable_state)
                    lp = self._prefix_likelihood(l_plus, probable_state)
                    base_i = base
                impact = max(0.0, min(1.0, 1.0 - min(lm, lp) / base_i))
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
                    alive=sum(1 for o in outputs if o != self.ABORT),
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
                # wasted = evaluations spent on strategies that are NOT on the
                # final decision path (branches opened and abandoned)
                on_path = {s.key for s in involved}
                wasted = sum(len(v) for k, v in self.runner.evals.items() if k not in on_path)
                return Result(
                    output=output,
                    likelihood=1.0 if best_s is None else likelihood,
                    runs=self.runner.total_runs,
                    wasted_runs=wasted,
                    sequential_runs=needed,
                    history=self.history,
                )
            self.runner.run(best_s, self.batch)

    def _sequential_runs(self, involved: List[Strategy]) -> int:
        """How many runs a plain sequential execution would have needed."""
        return len(involved) * self.n_instances


class _ImputingSums(dict):
    """Simulated totals that invent one for a strategy WITHOUT data.

    Drawn from a normal over the known strategies' totals in the same
    simulation, cached so the value is consistent within the simulation, and
    seeded per simulation index so the impact counterfactuals see the same
    imputed values as the base simulation.
    """

    def __init__(self, base, rng, mu, sd):
        super().__init__(base)
        self._rng, self._mu, self._sd = rng, mu, sd

    def __missing__(self, key):
        v = float(self._mu + self._rng.standard_normal() * self._sd)
        self[key] = v
        return v


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
