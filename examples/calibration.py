"""Calibration study -- the experiment the paper is missing.

A likelihood is a promise.  This script checks whether the promise is kept:
across many replications of the same experiment design, we bin every reported
likelihood and measure how often the reported output actually matched the
outcome of a full sequential execution.  A well-calibrated tool puts the
points on the diagonal.

Also reports, per estimator, the number of runs needed to first reach 90%
certainty -- the number that actually quantifies the speed-up.

    python examples/calibration.py --reps 30 --sigma 5 --sigma-inst 4
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from t4exps import experiment_execution, sequential_execution
from examples.dummy_algo import reseed
from examples.factorial import make_experiment

BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)]


def study(estimator, reps, sigma, sigma_inst, n_instances, nsims):
    instances = [f"seed{i}" for i in range(n_instances)]
    exp = make_experiment(sigma, sigma_inst)
    hits = defaultdict(lambda: [0, 0])
    runs_to_90, final_ok, total_runs = [], 0, []

    for rep in range(reps):
        reseed(rep)
        truth, seq_runs = sequential_execution(exp, instances)
        res = experiment_execution(
            exp, instances, estimator=estimator, seed=rep, nsims=nsims
        )
        final_ok += int(res.output == truth)
        total_runs.append(res.runs)

        reached = None
        for snap in res.history:
            for lo, hi in BINS:
                if lo <= snap.likelihood < hi:
                    hits[(lo, hi)][0] += int(snap.output == truth)
                    hits[(lo, hi)][1] += 1
                    break
            if reached is None and snap.likelihood >= 0.90 and snap.output == truth:
                reached = snap.runs
        runs_to_90.append(reached if reached is not None else seq_runs)

    return hits, runs_to_90, final_ok, seq_runs, total_runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--sigma", type=float, default=5.0)
    p.add_argument("--sigma-inst", type=float, default=4.0)
    p.add_argument("--instances", type=int, default=100)
    p.add_argument("--nsims", type=int, default=200)
    args = p.parse_args()

    for est in ("independent", "paired"):
        hits, r90, ok, seq, runs = study(
            est, args.reps, args.sigma, args.sigma_inst, args.instances, args.nsims
        )
        print(f"\n=== {est}  (sigma={args.sigma}, sigma_inst={args.sigma_inst}, "
              f"{args.reps} reps) ===")
        print("  reported likelihood -> empirical accuracy")
        for b in BINS:
            good, tot = hits[b]
            if tot:
                print(f"    [{b[0]:.2f},{b[1]:.2f})  n={tot:>4}   "
                      f"empirical = {good / tot:5.1%}")
        print(f"  runs to first reach 90% and be right: "
              f"median {int(np.median(r90))} of {seq} sequential "
              f"({seq / max(np.median(r90), 1):.1f}x)")
        print(f"  final answer identical to sequential: {ok}/{args.reps}")


if __name__ == "__main__":
    main()
