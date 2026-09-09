"""paired_t = K3 con tau2 sorteado de su posterior: nunca mas estrecho que K3."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from t4exps.estimators import ESTIMATORS, PairedEstimatorK3, PairedEstimatorT
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_dominant_strategy as D


def _spread(est, depth_c):
    d = est().draws(D._evals(depth_c), D.N, 800, np.random.default_rng(0))
    return float((d.sums[d.keys.index(D.B)] - d.sums[d.keys.index(D.A)]).std())


def test_paired_t_is_registered():
    assert ESTIMATORS["paired_t"] is PairedEstimatorT and PairedEstimatorT.tau2_posterior


def test_paired_t_not_narrower_than_k3():
    for dc in (15, 300):
        assert _spread(PairedEstimatorT, dc) >= 0.95 * _spread(PairedEstimatorK3, dc)


def test_k3_unchanged_when_tau2_posterior_off():
    """La rama plug-in de K3 no cambia: mismos draws con la misma semilla."""
    a = PairedEstimatorK3().draws(D._evals(60), D.N, 50, np.random.default_rng(3)).sums
    b = PairedEstimatorK3().draws(D._evals(60), D.N, 50, np.random.default_rng(3)).sums
    assert np.array_equal(a, b)
