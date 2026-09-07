"""Diagnostico: cuantas simulaciones abortan, con los datos que hay AHORA.

No corre el solver: reconstruye el cache desde el CSV del run en curso y
re-ejecuta solo la parte Monte Carlo del Engine.
"""
from __future__ import annotations
import sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "runs"))

from t4exps import Engine, order_instances
from t4exps.core import Context, SimulationAbort, set_context
from monitor import LoggingRunner
from stage3_full import make_experiment


class ReadOnlyRunner(LoggingRunner):
    """LoggingRunner que no puede ejecutar el solver: solo sirve el cache."""

    def _evaluate(self, s, instance):
        raise RuntimeError(
            f"el probe intento EJECUTAR el solver para {s.params} en "
            f"{instance[-40:]}: el estado truncado no cubre el camino predicho")

tag = sys.argv[1] if len(sys.argv) > 1 else "exact"
pool = [l.strip() for l in open(ROOT / "runs" / "instances_abs.txt") if l.strip()]
inst = order_instances(pool, "stratified", seed=0)

eng = Engine(make_experiment(30), inst, estimator="paired", n_jobs=1,
             tolerance=0.0116, confidence=1.0, nruns=15, batch=25,
             instance_order="file", seed=0)
eng.runner = ReadOnlyRunner.adopt(eng.runner,
    str(ROOT / "runs" / ".probe_scratch.csv"), echo_every=0)
eng.runner.csv_path = str(ROOT / "runs" / f"stage3_{tag}_evals.csv")  # solo lectura
eng.runner.resume_from_csv()
eng.runner.csv_path = str(ROOT / "runs" / ".probe_scratch.csv")
print(f"cache: {sum(eng.runner._resumed.values())} evaluaciones, "
      f"{len(eng.runner._resumed)} estrategias\n")

output, ctx = eng.predictive_execution()
involved = list(ctx.seen)
draws = eng._draws([s.key for s in involved])
print(f"output predicho : {output}")
print(f"estrategias en el camino : {len(involved)}")
print(f"estrategias con datos (draws) : {len(draws.keys)}")

# re-implementa _run_simulations contando abortos y por que
n = min(eng.nsims, draws.sums.shape[1])
ok = 0; aborted = 0; missing = collections.Counter(); outs = collections.Counter()
for sim in range(n):
    sums = draws.as_dict(sim)
    c = Context(mode="simulate", engine=eng, sums=sums)
    set_context(c)
    try:
        o = eng.experiment()
        ok += 1; outs[o] += 1
    except SimulationAbort as e:
        aborted += 1
        k = eval(str(e)) if str(e).startswith("(") else str(e)
        missing[", ".join(f"{a}={b}" for a, b in k[2]) if isinstance(k, tuple) else k] += 1
    finally:
        set_context(None)

print(f"\nsimulaciones: {n}")
print(f"  completadas : {ok}  ({100*ok/n:.1f}%)")
print(f"  abortadas   : {aborted}  ({100*aborted/n:.1f}%)")
if missing:
    print("\n  estrategias que las simulaciones necesitan y no existen en el cache:")
    for k, v in missing.most_common(6):
        print(f"    {v:>4}x  theta = {k}")
if outs:
    print(f"\n  outputs alcanzados ({len(outs)} distintos):")
    for o, v in outs.most_common(6):
        print(f"    {v:>4}x  {o}" + ("   <- el predicho" if o == output else ""))
    print(f"\n  likelihood = {outs.get(output,0)}/{ok} = {outs.get(output,0)/ok:.3f}")
else:
    print("\n  NINGUNA simulacion sobrevivio: likelihood y depth son 0 por falta de\n"
          "  informacion, no porque la prediccion sea improbable.")
eng.runner.close()
