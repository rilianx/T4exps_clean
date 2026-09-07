"""El estimador pareado cuando UNA estrategia domina la cobertura de instancias.

Es el estado que produce el picoteo del motor: una estrategia con cientos de
evaluaciones y las demas en el minimo.  Casi todas las instancias vistas
tienen k=1 (una sola estrategia las evaluo), asi que inv_k -> 1 y la
descomposicion de varianza divide por (1 - inv_k) -> 0: s_b2 se dispara y
tau2 = max(v_total - s_b2, 1e-12) colapsa a cero.  Con tau2 = 0 los 400 draws
son identicos: el posterior es una masa puntual sobre lo que digan 15 puntos.

Medido en el run real (replay honesto, semilla 2, ejecucion 1410): 1075 de
1140 instancias con k=1, s_b2 = 3.87 (real: 0.72), tau2 = 1e-12,
400/400 simulaciones identicas, P(g=0.3 > g=0.4) = 1.000 con 15 vs 15 evals.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from t4exps.core import Strategy
from t4exps.estimators import PairedEstimator, PairedEstimatorLegacy

N = 400
A = Strategy("s", "cmd {a}", {"a": 1}).key      # nivel 100.00
B = Strategy("s", "cmd {a}", {"a": 2}).key      # nivel 100.05, mejor
C = Strategy("s", "cmd {a}", {"a": 3}).key      # la que el motor machaca


def _evals(depth_c: int, seed: int = 0):
    """Efectos de instancia realistas (sd 0.9 pp, como BR) y residuo sd 0.3."""
    rng = np.random.default_rng(seed)
    b = rng.normal(0, 0.9, N)
    def series(level, depth):
        return list(level + b[:depth] + rng.normal(0, 0.3, depth))
    return {A: series(100.00, 15), B: series(100.05, 15), C: series(99.0, depth_c)}


def _diff_draws(evals, nsims=400, seed=0, est=PairedEstimatorLegacy):
    d = est().draws(evals, N, nsims, np.random.default_rng(seed))
    return d.sums[d.keys.index(B)] - d.sums[d.keys.index(A)]


def test_balanced_posterior_has_spread():
    """Control: con C tambien en 15, el posterior de B-A tiene incertidumbre."""
    assert _diff_draws(_evals(15)).std() > 1.0


@pytest.mark.parametrize("est", [
    pytest.param(PairedEstimatorLegacy, marks=pytest.mark.xfail(
        reason="DEFECTO del legado: con casi todas las instancias en k=1, "
               "inv_k -> 1, la solucion de componentes divide por ~0 y tau2 "
               "colapsa a 1e-12: el posterior es una masa puntual.",
        strict=True), id="legacy"),
    pytest.param(PairedEstimator, id="paired"),
])
def test_dominant_strategy_keeps_posterior_spread(est):
    """Que C tenga 300 evals no deberia volver CIERTA la comparacion A vs B."""
    assert _diff_draws(_evals(300), est=est).std() > 1.0


def test_the_legacy_defect_is_a_point_mass_not_just_narrow():
    """Documenta la magnitud del defecto legado: los 400 draws de B-A son casi iguales."""
    spread = _diff_draws(_evals(300)).std()
    control = _diff_draws(_evals(15)).std()
    assert spread < control / 1000, f"spread={spread} vs control={control}"
