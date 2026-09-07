"""Core abstractions: strategies, the re-execution context, and best_strategy.

Design note
-----------
The whole tool rests on one trick: the user's experiment function is a *pure
function of the evaluation cache*.  We run it many times -- once to predict the
outcome, and once per Monte Carlo simulation -- feeding it different views of
the data each time.  For that to work the experiment function must be
deterministic and side-effect free with respect to anything but our API.
This is an explicit contract, checked (partially) by `assert_deterministic`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

StrategyKey = Tuple[str, str, Tuple[Tuple[str, Any], ...]]


@dataclass(frozen=True)
class Strategy:
    """A strategy s = (A, theta): an algorithm plus a parameter assignment.

    `command` is either
      * a command-line template with {placeholders}, e.g. './solver -a {a}', or
      * a Python callable f(instance, **theta) -> float  (handy for tests).

    Strategies are *value objects*: two Strategy instances with the same name,
    command and theta are the same strategy.  This matters because the user's
    experiment function is re-executed many times and will construct fresh
    Strategy objects on every pass; identity must survive that.
    """

    name: str
    command: Any = None
    theta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "theta", dict(self.theta))
        if self.command is None:
            object.__setattr__(self, "command", self.name)

    @property
    def key(self) -> StrategyKey:
        cmd = getattr(self.command, "__name__", None) or str(self.command)
        return (self.name, cmd, tuple(sorted(self.theta.items())))

    def with_theta(self, **kw) -> "Strategy":
        theta = dict(self.theta)
        theta.update(kw)
        return Strategy(self.name, self.command, theta)

    @property
    def params(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in sorted(self.theta.items()))

    def __repr__(self) -> str:  # stable across re-executions
        return f"<{self.name} {self.params}>"

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return isinstance(other, Strategy) and self.key == other.key


# --------------------------------------------------------------------------- #
# Execution context
# --------------------------------------------------------------------------- #


class SimulationAbort(Exception):
    """Raised when a simulation needs a strategy that has no simulated data."""


@dataclass
class Context:
    """Per-execution state, installed by the engine around the user function."""

    mode: str                                   # 'predict' | 'simulate'
    engine: Any
    report: bool = False                        # user-visible run?
    # simulate mode: predicted TOTAL sum of evaluations per strategy
    sums: Optional[Dict[StrategyKey, float]] = None
    # decisions taken so far; the tuple of winners *is* the state of the run
    decisions: List[StrategyKey] = field(default_factory=list)
    seen: List[Strategy] = field(default_factory=list)
    checkpoints: List[Tuple[StrategyKey, ...]] = field(default_factory=list)

    def state(self) -> Tuple[StrategyKey, ...]:
        return tuple(self.decisions)


_local = threading.local()


def current_context() -> Optional[Context]:
    return getattr(_local, "ctx", None)


def set_context(ctx: Optional[Context]) -> None:
    _local.ctx = ctx


# --------------------------------------------------------------------------- #
# The one operation the user actually writes
# --------------------------------------------------------------------------- #


def best_strategy(s1: Strategy, s2: Strategy) -> Strategy:
    """Return the better of two strategies (higher mean evaluation wins).

    Outside an experiment execution this raises: strategies are never run
    eagerly, comparisons only mean something inside the engine.
    """
    ctx = current_context()
    if ctx is None:
        raise RuntimeError(
            "best_strategy() called outside an experiment; "
            "wrap your experiment in experiment_execution(...)"
        )
    eng = ctx.engine

    for s in (s1, s2):
        if s not in ctx.seen:
            ctx.seen.append(s)

    if ctx.mode == "predict":
        eng.ensure_minimum_data(s1)
        eng.ensure_minimum_data(s2)
        winner = eng.compare_observed(s1, s2)
    else:  # simulate
        try:
            v1, v2 = ctx.sums[s1.key], ctx.sums[s2.key]
        except KeyError as exc:  # a branch we never predicted -> give up
            raise SimulationAbort(str(exc)) from exc
        # tolerance is applied on the mean scale
        n = eng.n_instances
        winner = s1 if (v1 - v2) / n >= -eng.tolerance else s2

    ctx.decisions.append(winner.key)
    ctx.checkpoints.append(ctx.state())
    return winner


def anytime() -> bool:
    """True when the run is the user-visible one (so printing is wanted)."""
    ctx = current_context()
    return ctx is not None and ctx.report
