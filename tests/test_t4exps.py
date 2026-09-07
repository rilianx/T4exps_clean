import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from t4exps import (
    Strategy,
    best_strategy,
    cartesian_product,
    experiment_execution,
    sequential_execution,
)
from t4exps.estimators import IndependentEstimator, PairedEstimator
from examples.dummy_algo import dummy_algo, reseed

INSTANCES = [f"i{j}" for j in range(30)]


def factorial(sigma=3.0, sigma_inst=0.0):
    def exp():
        S = Strategy("d", dummy_algo,
                     {"a": -1, "b": -1, "sigma": sigma, "sigma_inst": sigma_inst})
        for a, b in cartesian_product([-1, 0, 1], [-1, 0, 1]):
            S = best_strategy(S, S.with_theta(a=a, b=b))
        return (S.theta["a"], S.theta["b"])
    return exp


# --------------------------------------------------------------------- #

@pytest.mark.parametrize("rep", range(6))
@pytest.mark.parametrize("estimator", ["paired", "independent"])
def test_matches_sequential(rep, estimator):
    """The whole promise of the tool: same answer as a full sequential run."""
    reseed(rep)
    exp = factorial()
    truth, _ = sequential_execution(exp, INSTANCES)
    res = experiment_execution(exp, INSTANCES, estimator=estimator,
                               seed=rep, nsims=80)
    assert res.output == truth
    assert res.likelihood == 1.0


def test_no_instance_is_run_twice():
    reseed(1)
    eng_res = experiment_execution(factorial(), INSTANCES, nsims=60, seed=1)
    assert eng_res.runs <= 9 * len(INSTANCES)
    assert 0 <= eng_res.wasted_runs <= eng_res.runs      # ramas abiertas y abandonadas


def test_evaluations_are_deterministic():
    reseed(4)
    a = dummy_algo("i7", a=1, b=0, sigma=3.0)
    b = dummy_algo("i7", a=1, b=0, sigma=3.0)
    assert a == b
    reseed(5)
    assert dummy_algo("i7", a=1, b=0, sigma=3.0) != a


def test_pairing_beats_independence_under_instance_heterogeneity():
    """The core claim: instance effects must cancel in a comparison."""
    reseed(2)
    thetas = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
    n = 60
    evals = {
        t: [dummy_algo(f"i{j}", a=t[0], b=t[1], sigma=2.0, sigma_inst=8.0)
            for j in range(15)]
        for t in thetas
    }
    spread = {}
    for est in (IndependentEstimator(), PairedEstimator()):
        d = est.draws(evals, n, 2000, np.random.default_rng(0))
        i, k = d.keys.index((0, 0)), d.keys.index((1, 0))
        spread[est.name] = float(np.std(d.sums[i] - d.sums[k]))
    assert spread["paired"] < 0.6 * spread["independent"]


def test_tolerance_saves_runs():
    """Declaring near-ties a tie must stop the engine burning time on noise."""
    reseed(3)
    exp = factorial(sigma=6.0)
    strict = experiment_execution(exp, INSTANCES, nsims=60, seed=3)
    loose = experiment_execution(exp, INSTANCES, nsims=60, seed=3, tolerance=1.0)
    assert loose.runs <= strict.runs


def test_best_strategy_outside_experiment_raises():
    s = Strategy("d", dummy_algo)
    with pytest.raises(RuntimeError):
        best_strategy(s, s)


# --------------------------------------------------------------------- #
# Instance ordering
# --------------------------------------------------------------------- #

from collections import Counter

from t4exps import default_family, stratify_instances, order_instances


def test_default_family_parses_benchmark_names():
    assert default_family("BR7_23.txt") == "BR7"
    assert default_family("data/BR15_100.dat") == "BR15"
    assert default_family("/abs/path/thpack1_5") == "thpack1"


def test_stratified_prefixes_are_balanced():
    """Every prefix must contain the families in roughly their proportion."""
    inst = [f"BR{f}_{i}" for f in range(1, 6) for i in range(100)]   # grouped
    ordered = stratify_instances(inst, seed=3)
    assert sorted(ordered) == sorted(inst)                           # permutation
    for c in (10, 50, 100, 250):
        counts = Counter(default_family(x) for x in ordered[:c])
        assert len(counts) == 5, f"prefix {c} misses families: {counts}"
        assert max(counts.values()) - min(counts.values()) <= 1


def test_stratified_handles_unequal_families():
    inst = ["A_%d" % i for i in range(90)] + ["B_%d" % i for i in range(10)]
    ordered = stratify_instances(inst, seed=0)
    first20 = Counter(default_family(x) for x in ordered[:20])
    assert first20["B"] == 2 and first20["A"] == 18


def test_file_order_is_preserved_when_asked():
    inst = ["z", "y", "x"]
    assert order_instances(inst, "file") == inst


@pytest.mark.parametrize("order", ["stratified", "shuffle", "file"])
def test_any_ordering_still_matches_sequential(order):
    reseed(7)
    exp = factorial()
    truth, _ = sequential_execution(exp, INSTANCES)
    res = experiment_execution(exp, INSTANCES, instance_order=order,
                               nsims=60, seed=7)
    assert res.output == truth


def test_cost_aware_runs_and_matches_sequential():
    reseed(8)
    exp = factorial()
    truth, _ = sequential_execution(exp, INSTANCES)
    res = experiment_execution(exp, INSTANCES, cost_aware=True, nsims=60, seed=8)
    assert res.output == truth
