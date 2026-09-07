"""Estimador pareado con niveles e instancias identificados por separado.

El original estima el nivel de cada estrategia con la media de TODAS sus
observaciones y el efecto de cada instancia como residuo contra ese nivel.
Cuando una estrategia fue evaluada mas a fondo que otra, su media incluye
instancias que nadie mas vio (k=1): el nivel queda contaminado por la
dificultad de esas instancias y el residuo queda mal (tests/test_unequal_depth).
Y cuando casi todas las instancias vistas son k=1, la solucion de componentes
de varianza divide por (1 - mean(1/k)) -> 0 y tau2 colapsa a cero
(tests/test_dominant_strategy).

Aqui, en dos pasos:
  1. NIVELES desde el bloque compartido (instancias con k >= 2), que es el
     unico donde nivel y efecto de instancia son separables.
  2. EFECTOS DE INSTANCIA para todas las vistas como residuo contra esos
     niveles, encogidos con lam(k) -- para k=1 es un encogimiento real, no
     cero, porque el nivel ya no depende de esa observacion.
  Las componentes de varianza se estiman solo en el bloque compartido, donde
  mean(1/k) <= 1/2 y el denominador no puede explotar.

No toca t4exps/: se registra en ESTIMATORS como "paired_k2".
"""
from __future__ import annotations

import warnings

import numpy as np

from t4exps.estimators import ESTIMATORS, Draws, IndependentEstimator, PairedEstimator


class PairedEstimatorK2(PairedEstimator):
    name = "paired_k2"

    def draws(self, evals, n_instances, nsims, rng) -> Draws:
        keys = list(evals)
        n, m = n_instances, len(keys)
        X = np.full((m, n), np.nan)
        for i, k in enumerate(keys):
            v = np.asarray(evals[k], dtype=float)
            X[i, : len(v)] = v
        counts = np.array([np.isfinite(X[i]).sum() for i in range(m)])
        if counts.min() < 2 or m < 2:
            return IndependentEstimator().draws(evals, n, nsims, rng)

        warnings.filterwarnings("ignore", message="Mean of empty slice")
        k = np.isfinite(X).sum(axis=0)
        seen, inform = k > 0, k >= 2
        if inform.sum() < 2:
            return IndependentEstimator().draws(evals, n, nsims, rng)

        # 1. niveles: solo bloque compartido (fallback a la media global si una
        #    estrategia no tiene ninguna instancia compartida)
        with np.errstate(invalid="ignore"):
            level = np.nanmean(X[:, inform], axis=1)
        bad = ~np.isfinite(level)
        if bad.any():
            level[bad] = np.nanmean(X[bad], axis=1)
        centered = X - level[:, None]

        # 2. efectos de instancia como residuo contra esos niveles, en TODAS
        #    las vistas; componentes de varianza solo en el bloque compartido
        with np.errstate(invalid="ignore"):
            raw_b = np.nanmean(centered, axis=0)
        ci = centered[:, inform]
        v_total = float(np.nanvar(ci))
        v_b_raw = float(np.var(raw_b[inform]))
        inv_k = float(np.mean(1.0 / k[inform]))              # <= 0.5
        denom = max(1.0 - inv_k, 1e-9)
        s_b2 = max((v_b_raw - inv_k * v_total) / denom, 0.0)
        s_e2 = max(v_total - s_b2, 1e-12)

        lam = np.zeros(n)
        lam[seen] = s_b2 / (s_b2 + s_e2 / k[seen])            # k=1: encoge, no anula
        b_hat = np.where(seen, np.nan_to_num(raw_b) * lam, np.nan)
        tau2 = max(s_e2, 1e-12)
        known = np.isfinite(b_hat)
        b_sd = float(np.sqrt(s_b2))

        out = np.empty((m, nsims))
        for sim in range(nsims):
            b = np.empty(n)
            b[known] = b_hat[known]
            n_unknown = int((~known).sum())
            if n_unknown:
                b[~known] = rng.standard_normal(n_unknown) * b_sd
            b_cum = np.concatenate(([0.0], np.cumsum(b)))
            for i, key in enumerate(keys):
                c = int(counts[i]); rem = n - c
                obs_sum = float(np.nansum(X[i]))
                if rem <= 0:
                    out[i, sim] = obs_sum
                    continue
                obs_mask = np.isfinite(X[i])
                mu_hat = float(np.mean(X[i][obs_mask] - b[obs_mask]))
                mu = mu_hat + rng.standard_normal() * np.sqrt(tau2 / c)
                noise = rng.standard_normal() * np.sqrt(tau2 * rem)
                out[i, sim] = obs_sum + mu * rem + (b_cum[n] - b_cum[c]) + noise
        return Draws(keys, out)


ESTIMATORS["paired_k2"] = PairedEstimatorK2


class PairedEstimatorK3(PairedEstimatorK2):
    """Como K2, pero el residuo se estima por ANOVA de dos vias en el bloque
    compartido: sum de residuos^2 sobre sum_j (k_j - 1) grados de libertad.
    Insensible a que k varie entre instancias (el despeje de momentos de
    K2/original usa un unico mean(1/k) y se sesga cuando k es heterogeneo:
    medido 0.45x el tau2 real con k entre 2 y 14 en el mismo bloque)."""
    name = "paired_k3"

    def _variance_components(self, centered, raw_b, k, inform):
        ci = centered[:, inform]                       # (m, n_inf), con nan
        resid = ci - raw_b[inform][None, :]
        ss = float(np.nansum(resid ** 2))
        df = float(np.sum(k[inform] - 1))
        s_e2 = max(ss / max(df, 1.0), 1e-12)
        v_b_raw = float(np.var(raw_b[inform]))
        inv_k = float(np.mean(1.0 / k[inform]))
        s_b2 = max(v_b_raw - s_e2 * inv_k, 0.0)
        return s_b2, s_e2

    def draws(self, evals, n_instances, nsims, rng) -> Draws:
        keys = list(evals); n, m = n_instances, len(keys)
        X = np.full((m, n), np.nan)
        for i, kk in enumerate(keys):
            v = np.asarray(evals[kk], dtype=float); X[i, : len(v)] = v
        counts = np.array([np.isfinite(X[i]).sum() for i in range(m)])
        if counts.min() < 2 or m < 2:
            return IndependentEstimator().draws(evals, n, nsims, rng)
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        k = np.isfinite(X).sum(axis=0); seen, inform = k > 0, k >= 2
        if inform.sum() < 2:
            return IndependentEstimator().draws(evals, n, nsims, rng)
        with np.errstate(invalid="ignore"):
            level = np.nanmean(X[:, inform], axis=1)
        bad = ~np.isfinite(level)
        if bad.any(): level[bad] = np.nanmean(X[bad], axis=1)
        centered = X - level[:, None]
        with np.errstate(invalid="ignore"):
            raw_b = np.nanmean(centered, axis=0)
        s_b2, s_e2 = self._variance_components(centered, raw_b, k, inform)
        lam = np.zeros(n); lam[seen] = s_b2 / (s_b2 + s_e2 / k[seen])
        b_hat = np.where(seen, np.nan_to_num(raw_b) * lam, np.nan)
        tau2 = max(s_e2, 1e-12); known = np.isfinite(b_hat); b_sd = float(np.sqrt(s_b2))
        out = np.empty((m, nsims))
        for sim in range(nsims):
            b = np.empty(n); b[known] = b_hat[known]
            nu = int((~known).sum())
            if nu: b[~known] = rng.standard_normal(nu) * b_sd
            b_cum = np.concatenate(([0.0], np.cumsum(b)))
            for i, kk in enumerate(keys):
                c = int(counts[i]); rem = n - c; obs_sum = float(np.nansum(X[i]))
                if rem <= 0: out[i, sim] = obs_sum; continue
                om = np.isfinite(X[i]); mu_hat = float(np.mean(X[i][om] - b[om]))
                mu = mu_hat + rng.standard_normal() * np.sqrt(tau2 / c)
                out[i, sim] = obs_sum + mu * rem + (b_cum[n] - b_cum[c]) + rng.standard_normal() * np.sqrt(tau2 * rem)
        return Draws(keys, out)


ESTIMATORS["paired_k3"] = PairedEstimatorK3
