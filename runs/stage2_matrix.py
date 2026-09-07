"""Etapa 2 del PLAN: piloto estadistico.

Las 5 configuraciones de `a` sobre 200 instancias estratificadas, con los 30 s
reales.  El objetivo es el DATASET (1000 evaluaciones), no el speedup: se
corre la matriz completa con un Runner directo, sin especulacion.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "runs"))

import numpy as np
from t4exps import Strategy, order_instances
from monitor import LoggingRunner

BIN  = "/home/iaraya/T4exps/bsg_algo/BSG_CLP"
BASE = {"a": 4.0, "b": 1.0, "g": 0.2, "p": 0.04}
T    = 30
TIMEOUT = float(sys.argv[1]) if len(sys.argv) > 1 else 75.0
NINST   = int(sys.argv[2]) if len(sys.argv) > 2 else 200
NJOBS   = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ALPHAS  = (0.0, 1.0, 2.0, 4.0, 8.0)

TEMPLATE = "{INSTANCE} --alpha={a} --beta={b} --gamma={g} -p {p} -t " + str(T)
OUT = ROOT / "runs"

pool = [ln.strip() for ln in open(OUT / "instances_abs.txt") if ln.strip()]
instances = order_instances(pool, "stratified", seed=0)[:NINST]

strategies = [Strategy("bsg", f"{BIN} {TEMPLATE}", theta={**BASE, "a": a})
              for a in ALPHAS]

print("=" * 78)
print(f"ETAPA 2 -- matriz completa  {len(strategies)} x {NINST} = "
      f"{len(strategies)*NINST} ejecuciones   T={T}s timeout={TIMEOUT}s jobs={NJOBS}")
print(f"estimado: {len(strategies)*NINST*(T*1.15)/NJOBS/60:.0f} min de reloj")
print("=" * 78, flush=True)

runner = LoggingRunner(instances, n_jobs=NJOBS, timeout=TIMEOUT,
                       csv_path=str(OUT / "stage2_evals.csv"), echo_every=0)
t_start = time.time()
for i, s in enumerate(strategies):
    t0 = time.time()
    runner.run(s, len(instances))
    v = runner.values(s)
    print(f"[{i+1}/{len(strategies)}] alpha={s.theta['a']:<4} n={len(v)} "
          f"media={np.mean(v):.4f} sd={np.std(v, ddof=1):.4f} "
          f"min={min(v):.2f} max={max(v):.2f}  "
          f"wall={time.time()-t0:.0f}s  ({time.time()-t_start:.0f}s total)", flush=True)

runner.close()
Y = np.array([runner.values(s) for s in strategies])
np.savez(OUT / "stage2_evals.npz", Y=Y,
         instances=np.array(instances),
         thetas=np.array([s.theta["a"] for s in strategies]),
         seconds=np.array([runner.mean_seconds(s) for s in strategies]))
print(f"\nguardado {OUT/'stage2_evals.npz'}  Y.shape={Y.shape}")
print(f"errores: {len(runner.errors)}")
for e in runner.errors[:10]: print("  " + e)
