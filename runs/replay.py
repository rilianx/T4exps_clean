"""Re-juega una corrida especulativa contra la matriz ya evaluada (oraculo).

    .venv/bin/python runs/replay.py --tag replay_x --confidence 0.98 --tolerance 0.0116
    .venv/bin/python runs/replay.py --tag replay_y --impute --estimator paired --seed 3

Por defecto reproduce el MOTOR ORIGINAL (estimator paired_legacy, likelihood
sesgada, sin imputacion), para que los replays viejos sigan siendo re-jugables.
--honest y --impute activan los arreglos del Engine 0.2; --estimator paired es
el estimador corregido (K3).  Sin --allow-solver, un par fuera del cache aborta.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "runs"))
sys.stdout.reconfigure(line_buffering=True)   # el print del Engine no hace flush

import t4exps
from t4exps import Engine, order_instances
from monitor import OracleRunner
from stage3_full import make_experiment
import estimators_fixed  # noqa: F401  (shim: los estimadores viven en el paquete)

p = argparse.ArgumentParser()
p.add_argument("--oracle", default=str(ROOT / "runs" / "stage3_exact_evals.csv"))
p.add_argument("--tag", required=True)
p.add_argument("--confidence", type=float, default=0.98)
p.add_argument("--tolerance", type=float, default=0.0116)
p.add_argument("--nsims", type=int, default=400)
p.add_argument("--nruns", type=int, default=15)
p.add_argument("--batch", type=int, default=25)
p.add_argument("--seed", type=int, default=0, help="semilla del Monte Carlo y de la imputacion")
p.add_argument("--order-seed", type=int, default=0,
               help="semilla de la ESTRATIFICACION (orden de instancias). Para calibrar hay que variarla: con un orden fijo todas las replicas ven el mismo prefijo")
p.add_argument("--limit", type=int, default=0)
p.add_argument("--allow-solver", action="store_true")
p.add_argument("--jobs", type=int, default=1, help="n_jobs del solver real en los cache miss")
p.add_argument("--estimator", default="paired_legacy",
               help="paired_legacy (default: reproduce los replays viejos) | paired (=k3) | paired_k2")
p.add_argument("--impute", action="store_true",
               help="estrategias sin datos se imputan del prior en vez de abortar (implica honesto)")
p.add_argument("--impute-scale", type=float, default=1.0, help="ancho del prior de imputacion (x sd)")
p.add_argument("--nsims-confirm", type=int, default=None, help="re-estimar la likelihood con N sims cerca del umbral")
p.add_argument("--impact-on", default="prefix", choices=["prefix","output"])
p.add_argument("--honest", action="store_true",
               help="denominador de la likelihood = simulaciones intentadas")
a = p.parse_args()

pool = [l.strip() for l in open(ROOT / "runs" / "instances_abs.txt") if l.strip()]
inst = order_instances(pool, "stratified", seed=a.order_seed)
if a.limit:
    inst = inst[: a.limit]

eng = Engine(make_experiment(30), inst, estimator=a.estimator, n_jobs=1,
             honest_likelihood=(a.honest or a.impute), impute_missing=a.impute,
             impute_prior_scale=a.impute_scale, nsims_confirm=a.nsims_confirm, impact_on=a.impact_on,
             confidence=a.confidence, tolerance=a.tolerance, nsims=a.nsims,
             nruns=a.nruns, batch=a.batch, instance_order="file",
             seed=a.seed, verbose=True, timeout=75)
eng.runner = OracleRunner(inst, oracle_csv=a.oracle, strict=not a.allow_solver,
                          n_jobs=a.jobs if a.allow_solver else 1, timeout=75,
                          csv_path=str(ROOT / "runs" / f"stage3_{a.tag}_evals.csv"),
                          echo_every=0)
mode = "IMPUTE" if a.impute else "HONEST" if a.honest else "original"
print(f"REPLAY [{a.tag}]  oraculo: {len(eng.runner.oracle)} pares  "
      f"confidence={a.confidence} tolerance={a.tolerance} nsims={a.nsims}  "
      f"engine={mode} estimator={a.estimator} ({type(eng.estimator).__name__})  t4exps {t4exps.__version__}")
t0 = time.time()
res = eng.run()
eng.runner.close()
r = eng.runner
print(f"\n{res}")
print(f"  runs={res.runs}  sequential={res.sequential_runs}  speedup={res.speedup:.2f}x  "
      f"L={res.likelihood:.4f}  wall={time.time()-t0:.0f}s")
print(f"  cache hits={r.hits}  misses={r.misses}  (solver real: {'si' if a.allow_solver else 'no'})")
(ROOT / "runs" / f"stage3_{a.tag}_result.json").write_text(json.dumps({
    "tag": a.tag, "output": list(res.output), "likelihood": res.likelihood, "runs": res.runs,
    "sequential_runs": res.sequential_runs, "speedup": res.speedup, "wasted_runs": res.wasted_runs,
    "hits": r.hits, "misses": r.misses, "confidence": a.confidence, "tolerance": a.tolerance,
    "nsims": a.nsims, "nsims_confirm": a.nsims_confirm, "impact_on": a.impact_on, "order_seed": a.order_seed, "impute_scale": a.impute_scale, "seed": a.seed, "oracle": a.oracle,
    "honest": a.honest or a.impute, "impute": a.impute, "estimator": a.estimator,
    "estimator_class": type(eng.estimator).__name__, "t4exps_version": t4exps.__version__,
    "history": [[s.runs, s.likelihood, s.state_depth, s.alive] for s in res.history]}, indent=1))
