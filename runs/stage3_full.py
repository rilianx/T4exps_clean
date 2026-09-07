"""Etapa 3 del PLAN: experimento completo, via Engine directo.

Se usa `Engine` en vez de `experiment_execution` por B3 del PLAN: hace falta
`eng.runner.evals` al final para poder re-analizar sin re-ejecutar.  Ademas se
inyecta el LoggingRunner, que persiste cada evaluacion a CSV en el momento --
red de seguridad contra B2 (un TimeoutExpired tira el run entero).
"""
from __future__ import annotations

import argparse, json, pickle, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "runs"))

from t4exps import Engine, Strategy, anytime, best_strategy, order_instances
from monitor import LoggingRunner

BIN  = "/home/iaraya/T4exps/bsg_algo/BSG_CLP"
OUT  = ROOT / "runs"

GRID = {
    "a": [0.0, 1.0, 2.0, 4.0, 8.0],
    "b": [0.5, 1.0, 2.0, 4.0],
    "g": [0.1, 0.2, 0.3, 0.4],
    "p": [0.01, 0.02, 0.03, 0.04],
}
START = {"a": 0.0, "b": 0.5, "g": 0.1, "p": 0.01}


def make_experiment(T: int):
    template = ("{INSTANCE} --alpha={a} --beta={b} --gamma={g} -p {p} -t "
                + str(T))          # -t AL FINAL: pisa el -t 30 del archivo

    def bsg_experiment():
        S = Strategy("bsg", f"{BIN} {template}", theta=dict(START))
        for name, values in GRID.items():
            for v in values:
                S = best_strategy(S, S.with_theta(**{name: v}))
        if anytime():
            print("  mejor configuracion hasta ahora:", S.params, flush=True)
        return tuple(S.theta[k] for k in ("a", "b", "g", "p"))

    return bsg_experiment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instances", default=str(OUT / "instances_abs.txt"))
    p.add_argument("--limit", type=int, default=0, help="0 = todas")
    p.add_argument("--solver-time", type=int, default=30, help="-t del solver")
    p.add_argument("--timeout", type=float, default=75.0, help="red de seguridad")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--confidence", type=float, default=0.98)
    p.add_argument("--tolerance", type=float, default=0.0)
    p.add_argument("--cost-aware", action="store_true")
    p.add_argument("--tag", default="phase1")
    p.add_argument("--resume", action="store_true",
                   help="reanuda desde el CSV de este mismo tag si existe; las "
                        "evaluaciones ya pagadas no se vuelven a correr")
    args = p.parse_args()

    pool = [ln.strip() for ln in open(args.instances) if ln.strip()]
    instances = order_instances(pool, "stratified", seed=0)
    if args.limit:
        instances = instances[: args.limit]

    print("=" * 78)
    print(f"ETAPA 3 [{args.tag}]  {len(instances)} instancias  T={args.solver_time}s  "
          f"timeout={args.timeout}s  jobs={args.jobs}")
    print(f"  confidence={args.confidence}  tolerance={args.tolerance}  "
          f"cost_aware={args.cost_aware}")
    print("=" * 78, flush=True)

    eng = Engine(make_experiment(args.solver_time), instances,
                 estimator="paired", n_jobs=args.jobs, timeout=args.timeout,
                 confidence=args.confidence, tolerance=args.tolerance,
                 cost_aware=args.cost_aware, nruns=15, batch=25,
                 instance_order="file",     # ya viene ordenado arriba
                 verbose=True, seed=0)
    eng.runner = LoggingRunner.adopt(
        eng.runner, str(OUT / f"stage3_{args.tag}_evals.csv"),
        echo_every=100, resume=args.resume)
    if args.resume:
        n = sum(eng.runner._resumed.values())
        print(f"  reanudando: {n} evaluaciones ya en disco, "
              f"{len(eng.runner._resumed)} estrategias\n", flush=True)
    t0 = time.time()
    try:
        res = eng.run()
    finally:
        eng.runner.close()
        with open(OUT / f"stage3_{args.tag}_runner.pkl", "wb") as fh:
            pickle.dump({"evals": dict(eng.runner.evals),
                         "instances": eng.runner.instances,
                         "seconds": dict(eng.runner.seconds),
                         "total_runs": eng.runner.total_runs}, fh)
    wall = time.time() - t0

    print(f"\n{res}")
    print(f"  runs            = {res.runs}")
    print(f"  sequential_runs = {res.sequential_runs}")
    print(f"  speedup         = {res.speedup:.2f}x")
    print(f"  likelihood      = {res.likelihood:.4f}")
    print(f"  wasted_runs     = {res.wasted_runs}")
    print(f"  wall            = {wall/3600:.2f} h")

    csv = OUT / f"stage3_{args.tag}_history.csv"
    csv.write_text("runs,likelihood,depth,output\n" +
                   "\n".join(f'{s.runs},{s.likelihood},{s.state_depth},"{s.output}"'
                             for s in res.history))
    (OUT / f"stage3_{args.tag}_result.json").write_text(json.dumps({
        "output": list(res.output), "likelihood": res.likelihood,
        "runs": res.runs, "sequential_runs": res.sequential_runs,
        "wasted_runs": res.wasted_runs, "speedup": res.speedup,
        "wall_hours": wall / 3600, "n_instances": len(instances),
        "solver_time": args.solver_time, "timeout": args.timeout,
        "confidence": args.confidence, "tolerance": args.tolerance,
        "cost_aware": args.cost_aware,
    }, indent=2))
    print(f"  log -> {csv}")


if __name__ == "__main__":
    main()
