"""Predicting the total sum of evaluations of each strategy.

Two estimators, deliberately interchangeable so they can be compared in an
ablation.

`independent` -- the model of the original paper.  Each strategy is fitted on
its own data, N(mu, sigma^2), and the remaining n-c evaluations are drawn
independently of every other strategy.  Difference: no MCMC.  For a normal
with both parameters unknown the Normal-Inverse-Gamma posterior is conjugate,
so exact draws cost one call to the RNG instead of a Metropolis-Hastings
chain.  Same model, ~3 orders of magnitude cheaper, no burn-in, no prior on
sigma that can silently exclude the truth.

`paired` -- the recommended one.  Instances are shared between strategies, so
most of the observed spread is instance difficulty, not strategy noise.  Model:

    E_s(pi) = mu_s + b_pi + eps,      eps ~ N(0, tau^2)

The instance effects b_pi are estimated once, pooled across all strategies,
and -- crucially -- a simulation draw uses the SAME b_pi for every strategy.
The instance effect therefore cancels in any comparison, and the variance that
actually drives the decision collapses from Var(E) to tau^2.  In practice this
is the difference between deciding a comparison after 60 instances and after
1500.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import warnings

import numpy as np

from .core import StrategyKey


@dataclass
class Draws:
    """nsims simulated TOTAL sums of evaluations, per strategy."""

    keys: List[StrategyKey]
    sums: np.ndarray  # shape (n_strategies, nsims)

    def as_dict(self, sim: int) -> Dict[StrategyKey, float]:
        return {k: float(self.sums[i, sim]) for i, k in enumerate(self.keys)}

    def quantile(self, key: StrategyKey, q: float) -> float:
        i = self.keys.index(key)
        return float(np.quantile(self.sums[i], q))


def _nig_draws(x: np.ndarray, nsims: int, rng: np.random.Generator):
    """Exact posterior draws of (mu, sigma) for N(mu, sigma^2), Jeffreys prior.

    sigma^2 | data ~ Inv-chi2(c-1, s^2);  mu | sigma^2 ~ N(xbar, sigma^2 / c)
    """
    c = len(x)
    xbar = float(np.mean(x))
    if c < 2:
        # Not enough data to say anything about spread; stay agnostic.
        return np.full(nsims, xbar), np.full(nsims, abs(xbar) + 1.0)
    s2 = float(np.var(x, ddof=1))
    s2 = max(s2, 1e-12)
    df = c - 1
    sigma2 = df * s2 / rng.chisquare(df, size=nsims)
    mu = xbar + rng.standard_normal(nsims) * np.sqrt(sigma2 / c)
    return mu, np.sqrt(sigma2)


class IndependentEstimator:
    """The original paper's model (minus the MCMC)."""

    name = "independent"

    def draws(self, evals, n_instances, nsims, rng) -> Draws:
        keys = list(evals)
        out = np.empty((len(keys), nsims))
        for i, k in enumerate(keys):
            x = np.asarray(evals[k], dtype=float)
            c, rem = len(x), n_instances - len(x)
            mu, sigma = _nig_draws(x, nsims, rng)
            noise = rng.standard_normal(nsims) * sigma * np.sqrt(max(rem, 0))
            out[i] = x.sum() + mu * rem + (noise if rem > 0 else 0.0)
        return Draws(keys, out)


class PairedEstimatorLegacy:
    """Two-way additive model; instance effects shared across strategies.

    LEGACY.  Kept for reproducing the ablation in PLAN.md.  Two measured
    defects: with unequal depths the level of the deeper strategy absorbs the
    difficulty of instances nobody else saw (tests/test_unequal_depth.py), and
    when almost every seen instance has k=1 the variance-component solve
    divides by ~0 and tau2 collapses to a point mass
    (tests/test_dominant_strategy.py).  Use `PairedEstimator` (= K3).
    """

    name = "paired_legacy"

    def __init__(self, shrinkage: float = 1.0):
        # shrinks instance effects estimated from few strategies toward 0
        self.shrinkage = shrinkage

    def draws(self, evals, n_instances, nsims, rng) -> Draws:
        keys = list(evals)
        n = n_instances
        m = len(keys)

        # ---- observed data as a ragged prefix matrix ----------------------- #
        X = np.full((m, n), np.nan)
        for i, k in enumerate(keys):
            v = np.asarray(evals[k], dtype=float)
            X[i, : len(v)] = v
        counts = np.array([np.isfinite(X[i]).sum() for i in range(m)])
        if counts.min() < 2 or m < 2:
            return IndependentEstimator().draws(evals, n, nsims, rng)

        import warnings
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        row_mean = np.nanmean(X, axis=1, keepdims=True)
        centered = X - row_mean                       # remove strategy level
        with np.errstate(invalid="ignore"):
            raw_b = np.nanmean(centered, axis=0)      # noisy instance effect
        k = np.isfinite(X).sum(axis=0)                # strategies per instance

        # --- variance components ------------------------------------------ #
        # Var(raw_b_j) = s_b^2 + s_e^2 / k_j        (noisy instance effects)
        # Var(centered) = s_b^2 + s_e^2             (everything)
        # Solve the 2x2 system; s_b^2 collapses to 0 when instances carry no
        # signal, which is exactly what we want -- no free confidence.
        seen = k > 0
        v_total = float(np.nanvar(centered))
        v_b_raw = float(np.var(raw_b[seen])) if seen.sum() > 1 else 0.0
        inv_k = float(np.mean(1.0 / k[seen])) if seen.sum() else 1.0
        denom = max(1.0 - inv_k, 1e-9)
        s_b2 = max((v_b_raw - inv_k * v_total) / denom, 0.0)
        s_e2 = max(v_total - s_b2, 1e-12)

        # empirical-Bayes shrinkage, per instance
        lam = np.zeros(n)
        lam[seen] = s_b2 / (s_b2 + s_e2 / np.maximum(k[seen], 1))
        b_hat = np.where(seen, np.nan_to_num(raw_b) * lam, np.nan)

        # residual (strategy-level) noise: what pairing CANNOT remove
        tau2 = max(s_e2, 1e-12)
        known = np.isfinite(b_hat)
        b_sd = float(np.sqrt(s_b2))

        out = np.empty((m, nsims))
        for sim in range(nsims):
            # ONE instance-effect vector, shared by every strategy this draw
            b = np.empty(n)
            b[known] = b_hat[known]
            n_unknown = int((~known).sum())
            if n_unknown:
                b[~known] = rng.standard_normal(n_unknown) * b_sd
            b_cum = np.concatenate(([0.0], np.cumsum(b)))  # prefix sums

            for i, k in enumerate(keys):
                c = int(counts[i])
                rem = n - c
                obs_sum = float(np.nansum(X[i]))
                if rem <= 0:
                    out[i, sim] = obs_sum
                    continue
                # posterior of the strategy level, adjusted for instance effects
                obs_mask = np.isfinite(X[i])
                mu_hat = float(np.mean(X[i][obs_mask] - b[obs_mask]))
                mu = mu_hat + rng.standard_normal() * np.sqrt(tau2 / c)
                noise = rng.standard_normal() * np.sqrt(tau2 * rem)
                out[i, sim] = obs_sum + mu * rem + (b_cum[n] - b_cum[c]) + noise
        return Draws(keys, out)


class PairedEstimatorK2(PairedEstimatorLegacy):
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


class PairedEstimatorK3(PairedEstimatorK2):
    tau2_posterior = False                  # True en PairedEstimatorT
    """The paired estimator.  Levels from the shared block (k >= 2), instance
    effects as shrunken residuals against those levels, residual variance by
    two-way ANOVA.  Fixes both defects of the legacy version.

    Como K2, pero el residuo se estima por ANOVA de dos vias en el bloque
    compartido: sum de residuos^2 sobre sum_j (k_j - 1) grados de libertad.
    Insensible a que k varie entre instancias (el despeje de momentos de
    K2/original usa un unico mean(1/k) y se sesga cuando k es heterogeneo:
    medido 0.45x el tau2 real con k entre 2 y 14 en el mismo bloque)."""
    name = "paired"

    def _variance_components(self, centered, raw_b, k, inform):
        ci = centered[:, inform]                       # (m, n_inf), con nan
        resid = ci - raw_b[inform][None, :]
        ss = float(np.nansum(resid ** 2))
        df = float(np.sum(k[inform] - 1))
        s_e2 = max(ss / max(df, 1.0), 1e-12)
        v_b_raw = float(np.var(raw_b[inform]))
        inv_k = float(np.mean(1.0 / k[inform]))
        s_b2 = max(v_b_raw - s_e2 * inv_k, 0.0)
        self._last_df = max(df, 1.0)             # para PairedEstimatorT
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
        df = getattr(self, "_last_df", 1.0)
        out = np.empty((m, nsims))
        for sim in range(nsims):
            if self.tau2_posterior:              # posterior t: tau2 | datos ~ df*s_e2 / chi2(df)
                tau2 = max(df * s_e2 / rng.chisquare(df), 1e-12)
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


class PairedEstimatorT(PairedEstimatorK3):
    """K3 con incertidumbre en tau2: en cada simulacion tau2 se sortea de su
    posterior escalado inv-chi2 con los grados de libertad del bloque
    compartido, asi que el nivel y la extrapolacion siguen una t en vez de una
    normal con tau2 plug-in.  Candidato para D5 (sobreconfianza a umbrales
    intermedios en la calibracion con ordenes aleatorios)."""
    name = "paired_t"
    tau2_posterior = True


PairedEstimator = PairedEstimatorK3          # the default "paired"

ESTIMATORS = {
    "paired": PairedEstimatorK3,
    "paired_k3": PairedEstimatorK3,
    "paired_t": PairedEstimatorT,
    "paired_k2": PairedEstimatorK2,
    "paired_legacy": PairedEstimatorLegacy,
    "independent": IndependentEstimator,
}

