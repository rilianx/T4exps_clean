"""Full factorial design over a, b in {-1, 0, 1} -- the paper's first case.

Run:  python examples/factorial.py --sigma 5 --sigma-inst 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from t4exps import (
    Strategy,
    anytime,
    best_strategy,
    cartesian_product,
    experiment_execution,
    sequential_execution,
)
from examples.dummy_algo import dummy_algo, reseed


def make_experiment(sigma: float, sigma_inst: float):
    def factorial_experiment():
        S = Strategy(
            "dummy",
            dummy_algo,
            theta={"a": -1, "b": -1, "sigma": sigma, "sigma_inst": sigma_inst},
        )
        for a, b in cartesian_product([-1, 0, 1], [-1, 0, 1]):
            S2 = S.with_theta(a=a, b=b)
            S = best_strategy(S, S2)
        if anytime():
            print(f"    best so far: {S.params}")
        return (S.theta["a"], S.theta["b"])

    return factorial_experiment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, default=5.0)
    p.add_argument("--sigma-inst", type=float, default=0.0)
    p.add_argument("--instances", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    instances = [f"seed{i}" for i in range(args.instances)]
    exp = make_experiment(args.sigma, args.sigma_inst)

    reseed(args.seed)
    truth, seq_runs = sequential_execution(exp, instances)
    print(f"\nsequential : {truth}  ({seq_runs} runs)\n")

    for est in ("independent", "paired"):
        reseed(args.seed)
        res = experiment_execution(
            exp, instances, estimator=est, seed=args.seed, nsims=300, verbose=False
        )
        ok = "OK " if res.output == truth else "!! "
        print(
            f"{ok}{est:<12}: {res.output}  {res.runs:>4} runs "
            f"({res.speedup:.2f}x fewer than sequential)"
        )


if __name__ == "__main__":
    main()
