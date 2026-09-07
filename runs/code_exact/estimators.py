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


class PairedEstimator:
    """Two-way additive model; instance effects shared across strategies."""

    name = "paired"

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


ESTIMATORS = {"paired": PairedEstimator, "independent": IndependentEstimator}
