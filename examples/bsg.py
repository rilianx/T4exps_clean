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


def make_experiment(binary: str, solver_time: int = 30):
    # Los flags cortos -a/-b/-g NO existen en BSG_CLP: producen un error de
    # parseo, exit 1 y por lo tanto RunError en cada evaluacion.  Y `-t` va al
    # FINAL porque las lineas de instancesCLP-shuf.txt ya traen su propio
    # `-t 30`, y en BSG_CLP gana el ultimo.
    template = ("{INSTANCE} --alpha={a} --beta={b} --gamma={g} -p {p} -t "
                + str(solver_time))

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
    p.add_argument("--solver-time", type=int, default=30,
                   help="presupuesto que se le pasa a BSG por -t")
    p.add_argument("--timeout", type=float, default=75.0,
                   help="red de seguridad del harness; NUNCA el mecanismo de "
                        "corte (ese es --solver-time).  Medido: con -t 30 el "
                        "peor wall observado fue 32.8 s")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--confidence", type=float, default=1.0)
    p.add_argument("--tolerance", type=float, default=0.0,
                   help="treat differences below this as ties (in utilisation)")
    p.add_argument("--cost-aware", action="store_true",
                   help="solo vale la pena si el costo por ejecucion varia "
                        "mucho entre instancias; con un -t vinculante no varia")
    args = p.parse_args()

    res = experiment_execution(
        make_experiment(args.bin, args.solver_time),
        args.instances,
        estimator="paired",          # BR instances differ wildly; pairing is key
        n_jobs=args.jobs,
        timeout=args.timeout,
        confidence=args.confidence,
        tolerance=args.tolerance,
        cost_aware=args.cost_aware,
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
