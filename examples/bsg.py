"""Sequential tuning of the four BSG parameters -- this is the script behind
the "Fig. ??" that is missing from section 5.2 of the paper.

BSG solves the Container Loading Problem and prints the reached volume
utilisation, so higher is better and no negation is needed.

    python examples/bsg.py --bin ./BSG_CLP --instances BR.txt

The instance file is stratified by family automatically (BR1..BR15).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from t4exps import Strategy, anytime, best_strategy, experiment_execution

GRID = {
    "a": [0.0, 1.0, 2.0, 4.0, 8.0],
    "b": [0.5, 1.0, 2.0, 4.0],
    "g": [0.1, 0.2, 0.3, 0.4],
    "p": [0.01, 0.02, 0.03, 0.04],
}


def make_experiment(binary: str):
    template = "{INSTANCE} -a {a} -b {b} -g {g} -p {p}"

    def bsg_experiment():
        S = Strategy("bsg", f"{binary} {template}",
                     theta={"a": 0.0, "b": 0.5, "g": 0.1, "p": 0.01})
        for name, values in GRID.items():          # one checkpoint per parameter
            for v in values:
                S = best_strategy(S, S.with_theta(**{name: v}))
        if anytime():
            print("best configuration so far:", S.params)
        return tuple(S.theta[k] for k in ("a", "b", "g", "p"))

    return bsg_experiment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", default="./BSG_CLP")
    p.add_argument("--instances", default="instances.txt")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--confidence", type=float, default=1.0)
    p.add_argument("--tolerance", type=float, default=0.0,
                   help="treat differences below this as ties (in utilisation)")
    args = p.parse_args()

    res = experiment_execution(
        make_experiment(args.bin),
        args.instances,
        estimator="paired",          # BR instances differ wildly; pairing is key
        n_jobs=args.jobs,
        timeout=args.timeout,
        confidence=args.confidence,
        tolerance=args.tolerance,
        cost_aware=True,             # 30 s cap: instance cost varies widely
        nruns=15,
        batch=25,
        verbose=True,
    )
    print("\n", res)
    csv = Path("bsg_history.csv")
    csv.write_text(
        "runs,likelihood,depth,output\n"
        + "\n".join(f"{s.runs},{s.likelihood},{s.state_depth},\"{s.output}\""
                    for s in res.history)
    )
    print(f"execution log written to {csv} (plot likelihood vs runs from this)")


if __name__ == "__main__":
    main()
