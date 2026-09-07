"""El estimador pareado con estrategias evaluadas a profundidades desiguales.

Esto es lo que la ejecucion especulativa produce todo el tiempo: el motor
evalua mas a fondo las estrategias que le interesan, asi que los conteos son
desparejos por diseno.  El modelo pareado tiene que corregir el efecto de
instancia igual -- si no lo hace, la estrategia mas evaluada queda juzgada
contra las instancias extra que le tocaron, y no contra su rival.

Medido sobre el run real (etapa 3, fase exacta): con beta=1.0 en 40
evaluaciones y beta=2.0 en 90, `compare_observed` daba ganador a beta=2.0 por
+0.047 pp sobre el prefijo pareado, mientras que las 400 simulaciones daban
ganador a beta=1.0 -- porque las 50 instancias extra de beta=2.0 eran 0.18 pp
mas dificiles.  Predecir y simular respondian cosas distintas, y por eso la
likelihood se fue a 0 y todas las simulaciones abortaron.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from t4exps.core import Strategy
from t4exps.estimators import PairedEstimator

N = 100
EASY = 40                    # a partir de aqui las instancias son ademas mas duras
PENALTY = 2.0
GAP = 0.10                   # ventaja real de B sobre A, en pp

A = Strategy("s", "cmd {a}", {"a": 1}).key      # nivel 100.00
B = Strategy("s", "cmd {a}", {"a": 2}).key      # nivel 100.10, mejor


def _evals(depth_a: int, depth_b: int, seed: int = 0):
    """Efectos de instancia realistas (sd 0.9 pp, como BR) mas un tramo extra
    sistematicamente mas duro, y residuo sd 0.2 pp."""
    rng = np.random.default_rng(seed)
    b = rng.normal(0, 0.9, N)
    b[EASY:] -= PENALTY
    def series(level, depth):
        return list(level + b[:depth] + rng.normal(0, 0.2, depth))
    return {A: series(100.00, depth_a), B: series(100.00 + GAP, depth_b)}


def _who_wins(evals, nsims=400, seed=0):
    """Fraccion de draws en que B (el mejor de verdad) supera a A."""
    draws = PairedEstimator().draws(evals, N, nsims, np.random.default_rng(seed))
    ia, ib = draws.keys.index(A), draws.keys.index(B)
    return float((draws.sums[ib] > draws.sums[ia]).mean())


def test_the_truth_is_that_B_is_better():
    """Sobre la matriz completa B gana por ~GAP: no hay ambiguedad."""
    ev = _evals(N, N)
    d = (sum(ev[B]) - sum(ev[A])) / N
    assert 0.5 * GAP < d < 1.5 * GAP


def test_equal_depth_recovers_the_better_strategy():
    """Control: con profundidades iguales el pareado acierta comodo."""
    assert _who_wins(_evals(40, 40)) > 0.90


@pytest.mark.xfail(
    reason="DEFECTO CONOCIDO: con profundidades desiguales el efecto de "
           "instancia queda sin corregir en las instancias que una sola "
           "estrategia evaluo (k=1 -> lam~0), asi que la estrategia mas "
           "evaluada carga con la dificultad de sus instancias extra.",
    strict=True)
def test_unequal_depth_still_recovers_the_better_strategy():
    """B sigue siendo mejor aunque se la haya evaluado en instancias peores.

    B corre 90 instancias (50 de ellas dificiles) y A solo 40 (todas faciles).
    El modelo pareado existe justamente para que eso no importe.
    """
    assert _who_wins(_evals(40, 90)) > 0.90


def test_the_defect_is_a_reversal_not_just_noise():
    """Documenta la magnitud: no es que dude, es que se da vuelta del todo."""
    equal = _who_wins(_evals(40, 40))
    unequal = _who_wins(_evals(40, 90))
    assert equal > 0.90, "control"
    assert unequal < 0.10, (
        f"con profundidades desiguales el estimador prefiere la estrategia "
        f"PEOR en {100 * (1 - unequal):.0f}% de los draws (con profundidades "
        f"iguales acierta en {100 * equal:.0f}%)")
