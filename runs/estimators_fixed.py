"""Shim: los estimadores corregidos viven ahora en t4exps.estimators.

Se mantiene este modulo porque runs/replay.py y los scripts de la ablacion lo
importan.  "paired" es hoy PairedEstimatorK3; el original es "paired_legacy".
"""
from t4exps.estimators import (ESTIMATORS, PairedEstimatorK2, PairedEstimatorK3,  # noqa: F401
                               PairedEstimatorLegacy)
