"""Re-juega una corrida especulativa contra la matriz ya evaluada (oraculo).

    .venv/bin/python runs/replay.py --tag replay_p1 --confidence 0.98 --tolerance 0.0116
    .venv/bin/python runs/replay.py --tag replay_fix --confidence 0.98 --nsims 2000 --allow-solver

Sin --allow-solver, un par fuera del cache aborta (seguro con un run vivo).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(line_buffering=True)   # el print del Engine no hace flush
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "runs"))

from t4exps import Engine, order_instances
from t4exps.engine import _hashable
import numpy as np
from monitor import OracleRunner
from stage3_full import make_experiment
import estimators_fixed  # noqa: F401  registra ESTIMATORS['paired_k2']


class HonestEngine(Engine):
    """Arreglo #1: el denominador de la likelihood son las simulaciones
    INTENTADAS, no las que sobrevivieron.

    Una simulacion aborta cuando su camino pide una estrategia sin datos, o
    sea cuando un sorteo del posterior discrepa con la prediccion.  Eso es
    evidencia en contra de la prediccion, no ruido a descartar.  Aqui se
    representa con un centinela que nunca iguala al output predicho ni a
    ningun checkpoint real, asi que `likelihood = aciertos / intentadas` y
    `_prefix_likelihood` se vuelven conservadoras sin tocar el resto del
    motor.  Con esto, 1 sobreviviente que acierta da 1/400, no 1/1.
    """
    ABORT = ("__abort__",)

    def _run_simulations(self, draws, override=None, budget=None):
        states, outputs = [], []
        for sim in range(min(budget or self.nsims, draws.sums.shape[1])):
            sums = draws.as_dict(sim)
            if override is not None:
                sums[override[0]] = override[1]
            out, ctx = self._simulate_once(sums)
            if out is None:
                states.append(self.ABORT); outputs.append(self.ABORT)
                continue
            states.append(ctx.state()); outputs.append(_hashable(out))
        return states, outputs

p = argparse.ArgumentParser()
p.add_argument("--oracle", default=str(ROOT / "runs" / "stage3_exact_evals.csv"))
p.add_argument("--tag", required=True)
p.add_argument("--confidence", type=float, default=0.98)
p.add_argument("--tolerance", type=float, default=0.0116)
p.add_argument("--nsims", type=int, default=400)
p.add_argument("--nruns", type=int, default=15)
p.add_argument("--batch", type=int, default=25)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--limit", type=int, default=0)
p.add_argument("--allow-solver", action="store_true")
p.add_argument("--jobs", type=int, default=1, help="n_jobs del solver real en los cache miss")
p.add_argument("--estimator", default="paired", help="paired | paired_k2")
p.add_argument("--impute", action="store_true",
               help="estrategias sin datos se imputan del prior en vez de abortar (implica honesto)")
p.add_argument("--honest", action="store_true",
               help="denominador de la likelihood = simulaciones intentadas")
a = p.parse_args()

pool = [l.strip() for l in open(ROOT / "runs" / "instances_abs.txt") if l.strip()]
inst = order_instances(pool, "stratified", seed=0)
if a.limit:
    inst = inst[: a.limit]

class _ImputingSums(dict):
    """`sums` que inventa un total para una estrategia SIN datos, coherente
    dentro de la misma simulacion (se cachea) y reproducible entre la
    simulacion base y sus contrafactuales de impacto (rng por indice de sim)."""
    def __init__(self, base, rng, mu, sd):
        super().__init__(base); self._rng, self._mu, self._sd = rng, mu, sd
    def __missing__(self, key):
        v = float(self._mu + self._rng.standard_normal() * self._sd); self[key] = v; return v


class ImputingEngine(HonestEngine):
    """Arreglo #4: una simulacion que llega a una estrategia no evaluada no
    aborta -- le sortea un total del prior (normal sobre los totales de las
    estrategias conocidas en esa misma simulacion) y sigue.  Asi la rama
    inexplorada cuenta como INCERTIDUMBRE en la likelihood, y el impacto ve
    que empujar la estrategia que decide esa rama cambia el resultado: la
    brujula vuelve a apuntar a la comparacion que hay que resolver."""
    _imp_seed = 0

    def _run_simulations(self, draws, override=None, budget=None):
        states, outputs = [], []
        for sim in range(min(budget or self.nsims, draws.sums.shape[1])):
            base = draws.as_dict(sim)
            if override is not None:
                base[override[0]] = override[1]
            col = draws.sums[:, sim]
            mu, sd = float(col.mean()), float(col.std()) if col.size > 1 else 0.0
            sums = _ImputingSums(base, np.random.default_rng(self._imp_seed + sim), mu, max(sd, 1e-9))
            out, ctx = self._simulate_once(sums)
            if out is None:                       # no deberia pasar ya, pero por si acaso
                states.append(self.ABORT); outputs.append(self.ABORT); continue
            states.append(ctx.state()); outputs.append(_hashable(out))
        return states, outputs


EngineCls = ImputingEngine if a.impute else (HonestEngine if a.honest else Engine)
eng = EngineCls(make_experiment(30), inst, estimator=a.estimator, n_jobs=1,
             confidence=a.confidence, tolerance=a.tolerance, nsims=a.nsims,
             nruns=a.nruns, batch=a.batch, instance_order="file",
             seed=a.seed, verbose=True, timeout=75)
eng._imp_seed = a.seed * 100003
eng.runner = OracleRunner(inst, oracle_csv=a.oracle, strict=not a.allow_solver,
                          n_jobs=a.jobs if a.allow_solver else 1, timeout=75,
                          csv_path=str(ROOT / "runs" / f"stage3_{a.tag}_evals.csv"),
                          echo_every=0)
print(f"REPLAY [{a.tag}]  oraculo: {len(eng.runner.oracle)} pares  "
      f"confidence={a.confidence} tolerance={a.tolerance} nsims={a.nsims}  "
      f"engine={'IMPUTE' if a.impute else 'HONEST' if a.honest else 'original'} estimator={a.estimator}", flush=True)
t0 = time.time()
res = eng.run()
eng.runner.close()
r = eng.runner
print(f"\n{res}")
print(f"  runs={res.runs}  sequential={res.sequential_runs}  speedup={res.speedup:.2f}x  "
      f"L={res.likelihood:.4f}  wall={time.time()-t0:.0f}s")
print(f"  cache hits={r.hits}  misses={r.misses}  "
      f"(solver real: {'si' if a.allow_solver else 'no'})")
(ROOT / "runs" / f"stage3_{a.tag}_result.json").write_text(json.dumps({
    "output": list(res.output), "likelihood": res.likelihood, "runs": res.runs,
    "sequential_runs": res.sequential_runs, "speedup": res.speedup,
    "hits": r.hits, "misses": r.misses, "confidence": a.confidence,
    "tolerance": a.tolerance, "nsims": a.nsims, "oracle": a.oracle,
    "honest": a.honest, "impute": a.impute, "estimator": a.estimator,
    "history": [[s.runs, s.likelihood, s.state_depth] for s in res.history]}, indent=1))
