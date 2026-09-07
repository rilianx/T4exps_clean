"""Mide si el valor devuelto por BSG_CLP depende de la carga de la maquina."""
import subprocess, shlex, sys, time
from concurrent.futures import ThreadPoolExecutor

BIN = "/home/iaraya/T4exps/bsg_algo/BSG_CLP"
ARGS = "--min_fr=0.98 --alpha=4.0 --beta=1.0 --gamma=0.2 -p 0.04"
T = int(sys.argv[1]) if len(sys.argv) > 1 else 5
NINST = int(sys.argv[2]) if len(sys.argv) > 2 else 16

pool = [ln.strip() for ln in open("/home/iaraya/sepeculative_experiments/T4exps_clean/runs/instances_abs.txt") if ln.strip()]
sys.path.insert(0, "/home/iaraya/sepeculative_experiments/T4exps_clean")
from t4exps import order_instances, default_family
inst = order_instances(pool, "stratified", seed=0)[:NINST]

def run(pi):
    cmd = f"{BIN} {pi} {ARGS} -t {T}"
    t0 = time.time()
    p = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    dt = time.time() - t0
    assert p.returncode == 0, p.stderr[:200]
    return float(p.stdout.split()[-1]), dt

def sweep(n_jobs):
    if n_jobs == 1:
        return [run(pi) for pi in inst]
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(run, inst))

print(f"T={T}  n_inst={NINST}")
a = sweep(1);  print(f"  n_jobs=1  hecho, wall medio {sum(d for _,d in a)/len(a):.2f}s")
b = sweep(8);  print(f"  n_jobs=8  hecho, wall medio {sum(d for _,d in b)/len(b):.2f}s")
c = sweep(8);  print(f"  n_jobs=8' hecho, wall medio {sum(d for _,d in c)/len(c):.2f}s")

print(f"\n{'instancia':<44} {'j=1':>10} {'j=8':>10} {'j=8 rep':>10}  {'d(1,8)':>8}")
n_diff_18 = n_diff_88 = 0
for pi,(v1,d1),(v8,d8),(v8b,d8b) in zip(inst,a,b,c):
    fam = default_family(pi); idx = pi.split("-i ")[1].split()[0]
    n_diff_18 += v1 != v8; n_diff_88 += v8 != v8b
    print(f"{fam+'-'+idx:<44} {v1:10.4f} {v8:10.4f} {v8b:10.4f}  {v1-v8:+8.4f}")
import statistics as st
d = [x[0]-y[0] for x,y in zip(a,b)]
print(f"\ncambian j=1 vs j=8 : {n_diff_18}/{NINST}   media dif {st.mean(d):+.4f} pp   max |dif| {max(map(abs,d)):.4f} pp")
print(f"cambian j=8 vs j=8': {n_diff_88}/{NINST}  (reproducibilidad a carga constante)")
print(f"wall max j=1 {max(d for _,d in a):.2f}s  j=8 {max(d for _,d in b):.2f}s  (exceso sobre -t {T})")
