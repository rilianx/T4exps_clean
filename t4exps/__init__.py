"""T4exps -- algorithm experiments with incremental speculative execution."""

from .core import Strategy, best_strategy, anytime
from .engine import experiment_execution, Engine, Result, Snapshot
from .estimators import PairedEstimator, IndependentEstimator
from .runner import Runner
from .utils import (cartesian_product, sequential_execution, stratify_instances,
                    shuffle_instances, order_instances, default_family)

__all__ = [
    "Strategy",
    "best_strategy",
    "anytime",
    "experiment_execution",
    "sequential_execution",
    "cartesian_product",
    "stratify_instances",
    "shuffle_instances",
    "order_instances",
    "default_family",
    "Engine",
    "Result",
    "Snapshot",
    "PairedEstimator",
    "IndependentEstimator",
    "Runner",
]
__version__ = "0.1.0"
