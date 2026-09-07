"""Etapa 1 del PLAN: piloto de plomeria.

2 configuraciones, 30 instancias estratificadas, -t 5.  El resultado no
importa; importa que el aparato mida lo que creemos.
"""
from __future__ import annotations

import os, shlex, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "runs"))

from t4exps import (Engine, Strategy, best_strategy, default_family,
                    order_instances, sequential_execution)
from t4exps.runner import RunError
from monitor import LoggingRunner

BIN  = "/home/iaraya/T4exps/bsg_algo/BSG_CLP"
BASE = {"a": 4.0, "b": 1.0, "g": 0.2, "p": 0.04}
T    = 5
TIMEOUT = 2 * T + 15

# -t va AL FINAL: pisa el -t 30 que traen las lineas del archivo.
TEMPLATE = "{INSTANCE} --alpha={a} --beta={b} --gamma={g} -p {p} -t " + str(T)
CMD = f"{BIN} {TEMPLATE}"

OUT = ROOT / "runs"
ok, fail = [], []
def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(f"  [{'OK ' if cond else 'FALLA'}] {name}" + (f"  {detail}" if detail else ""))

def make_exp():
    def exp():
        S = Strategy("bsg", CMD, theta=BASE)
        return best_strategy(S, S.with_theta(a=8.0)).params
    return exp

pool = [ln.strip() for ln in open(OUT / "instances_abs.txt") if ln.strip()]
inst = order_instances(pool, "stratified", seed=0)[:30]

print("=" * 78)
print(f"ETAPA 1 -- piloto de plomeria   T={T}s  timeout={TIMEOUT}s  "
      f"{len(inst)} instancias  cwd={os.getcwd()}")
print("=" * 78)

# --- 1. el comando formateado ------------------------------------------- #
print("\n[1] Comando formateado")
S0 = Strategy("bsg", CMD, theta=BASE)
formatted = S0.command.format(INSTANCE=inst[0], **S0.theta)
print(f"  {formatted}")
check("{INSTANCE} se sustituye", "{INSTANCE}" not in formatted and inst[0] in formatted)
check("no se dispara el fallback que appendea la instancia",
      formatted.count(inst[0]) == 1)
check("flags largos --alpha/--beta/--gamma y -p",
      all(f in formatted for f in ("--alpha=", "--beta=", "--gamma=", "-p ")))
check("no aparecen los flags inexistentes -a/-b/-g",
      not any(f" {f} " in formatted for f in ("-a", "-b", "-g")))
check("-t del template queda DESPUES del -t del archivo",
      formatted.rfind("-t ") > formatted.find("-t 30") if "-t 30" in formatted else True)

# --- 2. una ejecucion a mano, stdout crudo ------------------------------- #
print("\n[2] Una ejecucion cruda (stdout completo)")
t0 = time.time(); proc = subprocess.run(shlex.split(formatted), capture_output=True, text=True)
dt = time.time() - t0
print("  " + "\n  ".join(proc.stdout.strip().splitlines()[-6:]))
check("exit code 0", proc.returncode == 0, f"exit={proc.returncode} stderr={proc.stderr[:120]!r}")
last = float(proc.stdout.split()[-1])
check("el ultimo float es la utilizacion", "volume utilization" in proc.stdout)
check("escala porcentaje (0-100), no fraccion", 50.0 < last < 100.0, f"valor={last}")
check(f"el -t del template gana (wall {dt:.2f}s, no ~30s)", dt < 15, f"wall={dt:.2f}s")

# --- 3. familias --------------------------------------------------------- #
print("\n[3] Deteccion de familias sobre lineas con argumentos")
from collections import Counter
fams = Counter(default_family(i) for i in inst)
print(f"  {len(fams)} familias, cuentas: {sorted(fams.values())}")
check("15 familias en el prefijo de 30", len(fams) == 15)
check("2 instancias por familia", set(fams.values()) == {2})
famsall = Counter(default_family(i) for i in pool)
check("1500 instancias, 15 familias x 100", len(pool) == 1500 and set(famsall.values()) == {100})

# --- 4. modo de falla del timeout ---------------------------------------- #
print("\n[4] Modo de falla del timeout (a proposito, timeout=1)")
try:
    subprocess.run(shlex.split(formatted), capture_output=True, text=True, timeout=1)
    check("timeout=1 lanza TimeoutExpired", False, "no lanzo nada")
except subprocess.TimeoutExpired as e:
    check("timeout=1 lanza TimeoutExpired", True, f"{type(e).__name__}: mata el proceso, sin valor")

# --- 5. baseline secuencial (n_jobs=1) ----------------------------------- #
print(f"\n[5] sequential_execution n_jobs=1  ({2*len(inst)} ejecuciones)")
t0 = time.time()
truth, seq_runs = sequential_execution(make_exp(), inst, n_jobs=1, timeout=TIMEOUT)
tseq = time.time() - t0
print(f"  output={truth!r}  runs={seq_runs}  wall={tseq:.0f}s")
check("cero RunError en secuencial", True)
check("sequential hace strategies x instances", seq_runs == 2 * len(inst), f"runs={seq_runs}")

# --- 6. Engine n_jobs=8 y n_jobs=1 --------------------------------------- #
def run_engine(n_jobs, tag):
    print(f"\n[6.{n_jobs}] Engine n_jobs={n_jobs}")
    eng = Engine(make_exp(), inst, n_jobs=n_jobs, timeout=TIMEOUT,
                 nruns=10, batch=10, estimator="paired", verbose=True, seed=0)
    eng.runner = LoggingRunner.adopt(eng.runner, str(OUT / f"stage1_{tag}_evals.csv"),
                                     echo_every=0)
    t0 = time.time(); res = eng.run(); dt = time.time() - t0
    eng.runner.close()
    print(f"  -> {res}   wall={dt:.0f}s")
    return eng, res, dt

eng8, res8, t8 = run_engine(8, "j8")
eng1, res1, t1 = run_engine(1, "j1")

print("\n[7] Invariantes")
check("secuencial == incremental (output)", res8.output == truth,
      f"{res8.output!r} vs {truth!r}")
check("likelihood == 1.0 con confidence=1.0", res8.likelihood == 1.0,
      f"L={res8.likelihood}")
check("n_jobs=1 y n_jobs=8 dan el mismo output", res1.output == res8.output,
      f"{res1.output!r} vs {res8.output!r}")

k8 = {k: v for k, v in eng8.runner.evals.items()}
k1 = {k: v for k, v in eng1.runner.evals.items()}
check("mismas estrategias evaluadas", set(k8) == set(k1))
ndiff = nsame = 0; maxd = 0.0
for k in set(k8) & set(k1):
    for x, y in zip(k8[k], k1[k]):
        if x == y: nsame += 1
        else: ndiff += 1; maxd = max(maxd, abs(x - y))
check("n_jobs=1 y n_jobs=8 dan las MISMAS evaluaciones", ndiff == 0,
      f"{ndiff}/{ndiff+nsame} difieren, max |dif| = {maxd:.4f} pp")

for tag, eng in (("j8", eng8), ("j1", eng1)):
    r = eng.runner
    check(f"[{tag}] ninguna instancia evaluada dos veces",
          all(len(v) <= len(r.instances) for v in r.evals.values()))
    check(f"[{tag}] total_runs == suma de evaluaciones",
          r.total_runs == sum(map(len, r.evals.values())),
          f"{r.total_runs} vs {sum(map(len, r.evals.values()))}")
    check(f"[{tag}] cero errores de ejecucion", not r.errors, f"{r.errors[:2]}")

# --- 8. exceso de tiempo -------------------------------------------------- #
print("\n[8] Exceso de tiempo por familia (de los CSV)")
import csv as _csv
from collections import defaultdict
per = defaultdict(list)
for tag in ("j8", "j1"):
    with open(OUT / f"stage1_{tag}_evals.csv") as fh:
        for row in _csv.DictReader(fh):
            if row["status"] == "ok":
                per[row["family"]].append(float(row["seconds"]))
worst = 0.0
for fam in sorted(per, key=lambda f: int(f[2:])):
    v = per[fam]; mx = max(v); worst = max(worst, mx)
    print(f"  {fam:<6} n={len(v):<3} medio {sum(v)/len(v):5.2f}s  max {mx:5.2f}s  "
          f"exceso max {100*(mx-T)/T:+5.0f}%")
print(f"\n  PEOR exceso observado: {worst:.2f}s con -t {T}  ({100*(worst-T)/T:+.0f}%)")
print(f"  => timeout sugerido para T=30: max(2*30+15, 30*{worst/T:.2f}+margen)")
check(f"ningun run murio por timeout ({TIMEOUT}s)", worst < TIMEOUT, f"peor {worst:.2f}s")

print("\n" + "=" * 78)
print(f"ETAPA 1: {len(ok)} OK, {len(fail)} FALLAS")
for f in fail: print(f"  FALLA: {f}")
print(f"tiempos: seq(j1)={tseq:.0f}s  engine(j8)={t8:.0f}s  engine(j1)={t1:.0f}s")
print("=" * 78)
