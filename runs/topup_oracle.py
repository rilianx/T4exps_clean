"""Completa el oraculo: evalua DE VERDAD, hasta una profundidad dada, las
estrategias que los replays estrictos pidieron y no estaban.  Escribe a su
propio CSV (lo recoge runs/build_oracle.py); nunca toca el CSV del run exacto."""
import sys, csv, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"runs"))
from t4exps import Strategy, order_instances
from monitor import LoggingRunner
from stage3_full import BIN
DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 65
JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
THETAS = sys.argv[3:] or ["a=8.0;b=1.0;g=0.1;p=0.01", "a=8.0;b=0.5;g=0.2;p=0.01", "a=8.0;b=0.5;g=0.3;p=0.01",
                          "a=8.0;b=0.5;g=0.4;p=0.01", "a=8.0;b=0.5;g=0.3;p=0.02", "a=8.0;b=0.5;g=0.3;p=0.03",
                          "a=8.0;b=0.5;g=0.3;p=0.04"]
TPL = "{INSTANCE} --alpha={a} --beta={b} --gamma={g} -p {p} -t 30"
pool = [l.strip() for l in open(ROOT/"runs"/"instances_abs.txt") if l.strip()]
inst = order_instances(pool, "stratified", seed=0)
oracle = collections.defaultdict(dict)
for r in csv.DictReader(open(ROOT/"runs"/"oracle_extended.csv")):
    oracle[r["theta"]][r["instance"]] = float(r["value"])
r = LoggingRunner(inst, n_jobs=JOBS, timeout=75, csv_path=str(ROOT/"runs"/"stage3_replay_topup_solver_s0_evals.csv"), echo_every=0)
total = 0
for th in THETAS:
    s = Strategy("bsg", f"{BIN} {TPL}", theta={k: float(v) for k, v in (kv.split("=") for kv in th.split(";"))})
    have = []
    for pi in inst:
        if pi in oracle[th]: have.append(oracle[th][pi])
        else: break
    r.evals[s.key] = list(have)                       # prefijo ya conocido: no se repite
    need = max(0, DEPTH - len(have))
    print(f"{th}: tiene {len(have)}, evalua {need} -> {DEPTH}", flush=True)
    if need: r.run(s, need); total += need
r.close(); print(f"evaluaciones reales: {total}   errores: {len(r.errors)}")
