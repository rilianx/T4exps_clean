"""Generate the comparison figures from actual runs (no synthetic curves).

    python examples/make_figures.py --reps 12 --instances 60

Writes PDF (for the paper) and PNG (for looking at) into figures/.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from t4exps import experiment_execution, sequential_execution
from t4exps.estimators import IndependentEstimator, PairedEstimator
from examples.dummy_algo import dummy_algo, reseed
from examples.factorial import make_experiment

OUT = Path(__file__).resolve().parents[1] / "figures"
C = {"independent": "#c0392b", "paired": "#1f6feb"}
M = {"independent": "s", "paired": "o"}

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "legend.frameon": False,
})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


# --------------------------------------------------------------------- #
# Fig 1: predictive spread of the DIFFERENCE vs instance heterogeneity
# --------------------------------------------------------------------- #
def fig_spread(n=60, c=15):
    thetas = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
    sig_inst = [0, 1, 2, 4, 6, 8, 10]
    res = {k: [] for k in C}
    for si in sig_inst:
        reseed(2)
        evals = {t: [dummy_algo(f"i{j}", a=t[0], b=t[1], sigma=2.0, sigma_inst=si)
                     for j in range(c)] for t in thetas}
        for est in (IndependentEstimator(), PairedEstimator()):
            d = est.draws(evals, n, 3000, np.random.default_rng(0))
            i, k = d.keys.index((0, 0)), d.keys.index((1, 0))
            res[est.name].append(float(np.std((d.sums[i] - d.sums[k]) / n)))

    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    for name, ys in res.items():
        ax.plot(sig_inst, ys, marker=M[name], color=C[name], label=name, lw=1.6, ms=4)
    ax.set_xlabel(r"instance heterogeneity  $\sigma_{inst}$")
    ax.set_ylabel("predictive s.d. of the\nstrategy difference")
    ax.set_title("Pairing is immune to instance difficulty", fontsize=9.5)
    ax.legend()
    save(fig, "fig1_spread_vs_heterogeneity")
    return res


# --------------------------------------------------------------------- #
# Fig 2: likelihood trajectories
# --------------------------------------------------------------------- #
def fig_trajectories(reps, n_inst, sigma, sigma_inst, nsims):
    instances = [f"seed{i}" for i in range(n_inst)]
    exp = make_experiment(sigma, sigma_inst)
    curves = defaultdict(list)
    for rep in range(reps):
        for est in C:
            reseed(rep)
            r = experiment_execution(exp, instances, estimator=est,
                                     seed=rep, nsims=nsims)
            curves[est].append([(s.runs, s.likelihood) for s in r.history])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    grid = np.arange(0, 9 * n_inst + 1, 10)
    for ax, est in zip(axes, ("independent", "paired")):
        band = []
        for cur in curves[est]:
            x, y = zip(*cur)
            ax.plot(x, np.array(y) * 100, color=C[est], alpha=0.18, lw=0.9)
            band.append(np.interp(grid, x, y, left=np.nan, right=1.0))
        band = np.array(band)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(band, axis=0) * 100
        ax.plot(grid, med, color=C[est], lw=2.2, label="median")
        ax.axhline(90, ls="--", lw=0.9, color="0.35")
        ax.set_title(est, fontsize=9.5)
        ax.set_xlabel("# executions")
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("likelihood of the\nprobable output (%)")
    axes[0].legend(loc="lower right")
    fig.suptitle(f"$\\sigma={sigma}$, $\\sigma_{{inst}}={sigma_inst}$, "
                 f"{reps} replications", fontsize=9)
    save(fig, "fig2_likelihood_trajectories")
    return curves


# --------------------------------------------------------------------- #
# Fig 3: calibration + runs-to-90%
# --------------------------------------------------------------------- #
def fig_calibration(reps, n_inst, sigma, sigma_inst, nsims):
    instances = [f"seed{i}" for i in range(n_inst)]
    exp = make_experiment(sigma, sigma_inst)
    edges = np.array([0, .2, .4, .55, .7, .8, .9, .96, 1.001])
    acc = {e: [np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)] for e in C}
    r90 = defaultdict(list)

    for rep in range(reps):
        reseed(rep)
        truth, seq = sequential_execution(exp, instances)
        for est in C:
            reseed(rep)
            r = experiment_execution(exp, instances, estimator=est,
                                     seed=rep, nsims=nsims)
            hit = None
            for s in r.history:
                b = int(np.searchsorted(edges, s.likelihood, "right") - 1)
                b = min(max(b, 0), len(edges) - 2)
                acc[est][1][b] += 1
                acc[est][0][b] += int(s.output == truth)
                if hit is None and s.likelihood >= .90 and s.output == truth:
                    hit = s.runs
            r90[est].append(hit if hit is not None else seq)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax1.plot([0, 100], [0, 100], color="0.5", ls=":", lw=1)
    for est in C:
        good, tot = acc[est]
        m = tot >= 5
        centers = (edges[:-1] + edges[1:])[m] / 2 * 100
        ax1.plot(centers, good[m] / tot[m] * 100, marker=M[est], color=C[est],
                 label=est, lw=1.6, ms=5)
    ax1.set_xlabel("reported likelihood (%)")
    ax1.set_ylabel("empirical accuracy (%)")
    ax1.set_title("Calibration", fontsize=9.5)
    ax1.legend(loc="upper left")
    ax1.set_xlim(0, 102); ax1.set_ylim(0, 102)

    data = [r90["independent"], r90["paired"]]
    bp = ax2.boxplot(data, tick_labels=["independent", "paired"], widths=.5,
                     patch_artist=True, medianprops=dict(color="k"))
    for patch, est in zip(bp["boxes"], ("independent", "paired")):
        patch.set_facecolor(C[est]); patch.set_alpha(.45)
    ax2.axhline(9 * n_inst, ls="--", color="0.35", lw=1)
    ax2.text(2.42, 9 * n_inst, " sequential", va="center", fontsize=8, color="0.35")
    ax2.set_ylabel("# executions to reach 90%\nand be right")
    ax2.set_title("Cost of an answer", fontsize=9.5)
    save(fig, "fig3_calibration_and_cost")
    return acc, r90


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=12)
    p.add_argument("--instances", type=int, default=60)
    p.add_argument("--sigma", type=float, default=5.0)
    p.add_argument("--sigma-inst", type=float, default=4.0)
    p.add_argument("--nsims", type=int, default=120)
    a = p.parse_args()

    print("fig 1 ..."); fig_spread()
    print("fig 2 ..."); fig_trajectories(a.reps, a.instances, a.sigma,
                                         a.sigma_inst, a.nsims)
    print("fig 3 ..."); acc, r90 = fig_calibration(a.reps, a.instances, a.sigma,
                                                   a.sigma_inst, a.nsims)
    print("\nmedian executions to 90% and correct:")
    for est, v in r90.items():
        print(f"  {est:<12} {int(np.median(v)):>4}  "
              f"of {9 * a.instances} sequential "
              f"({9 * a.instances / max(np.median(v), 1):.2f}x)")


if __name__ == "__main__":
    main()
