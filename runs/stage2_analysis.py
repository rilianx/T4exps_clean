"""Analisis offline de la matriz de la Etapa 2 (re-corrible sin re-ejecutar)."""
from __future__ import annotations
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t4exps.utils import default_family

d = np.load(ROOT / "runs" / "stage2_evals.npz", allow_pickle=True)
Y, instances, thetas = d["Y"], list(d["instances"]), d["thetas"]
S, N = Y.shape
fams = np.array([default_family(i) for i in instances])
print("=" * 78)
print(f"ETAPA 2 -- analisis   Y {Y.shape}   alphas {list(thetas)}")
print("=" * 78)
print("\nmedias por configuracion (puntos porcentuales de utilizacion):")
for a, row in zip(thetas, Y):
    print(f"  alpha={a:<5} media={row.mean():8.4f}  sd={row.std(ddof=1):6.4f}")
print(f"  sd marginal promedio = {Y.std(axis=1, ddof=1).mean():.4f} pp "
      f"(NOTES.md dice ~0.95)")

# --- Analisis 1: descomposicion de varianzas ---------------------------- #
m     = Y.mean()
alpha = Y.mean(axis=1) - m
beta  = Y.mean(axis=0) - m
E     = Y - m - alpha[:, None] - beta[None, :]
v_resid = (E ** 2).sum() / ((S - 1) * (N - 1))
v_inst  = max(beta.var(ddof=1) - v_resid / S, 0.0)
v_strat = max(alpha.var(ddof=1) - v_resid / N, 0.0)
frac_inst = v_inst / (v_inst + v_strat + v_resid)
naive = beta.var(ddof=1) / (beta.var(ddof=1) + alpha.var(ddof=1) + v_resid)
print(f"\n[A1] Descomposicion de varianzas")
print(f"  v_inst  (efecto instancia, corregido) = {v_inst:.5f}")
print(f"  v_strat (efecto configuracion, corr.) = {v_strat:.5f}")
print(f"  v_resid (residuo)                     = {v_resid:.5f}")
print(f"  frac_inst = {frac_inst:.4f}   (naive sin correccion: {naive:.4f})")
veredicto = ("PAREADO ES LA HISTORIA (>0.7)" if frac_inst > 0.7 else
             "NOTA AL PIE (<0.3)" if frac_inst < 0.3 else "ZONA GRIS (0.3-0.7)")
print(f"  => {veredicto}")

# --- Analisis 2: aditividad (Spearman) ---------------------------------- #
def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])

rho = np.array([[spearman(Y[i], Y[j]) for j in range(S)] for i in range(S)])
iu = np.triu_indices(S, k=1)
rho_min = rho[iu].min()
print(f"\n[A2] Aditividad -- Spearman entre configuraciones")
print("     " + "".join(f"{a:>8}" for a in thetas))
for a, row in zip(thetas, rho):
    print(f"  {a:<4}" + "".join(f"{x:8.3f}" for x in row))
zona = ("pareado sin miedo (>0.8)" if rho_min > 0.8 else
        "volver a independent / bloquear por familia (<=0.4)" if rho_min <= 0.4 else
        "ZONA GRIS (0.4-0.8): reportar ambos o bloquear por familia")
print(f"  rho_min (sobre pares) = {rho_min:.3f}  => {zona}")

if rho_min <= 0.8:
    print("\n  rho intra-familia (minimo sobre pares, por familia):")
    for f in sorted(set(fams), key=lambda x: int(x[2:])):
        k = fams == f
        if k.sum() < 5: continue
        r = np.array([[spearman(Y[i][k], Y[j][k]) for j in range(S)] for i in range(S)])
        print(f"    {f:<6} n={k.sum():<4} rho_min={r[iu].min():6.3f}  medio={r[iu].mean():6.3f}")

# --- Analisis 3: tolerance ----------------------------------------------- #
D  = Y[:, None, :] - Y[None, :, :]
se = D.std(axis=2, ddof=1) / np.sqrt(N)
tol = float(np.median(se[iu]))
# Comparacion honesta: SE de la DIFERENCIA de dos medias en ambos casos.  El
# independiente paga sqrt(2)*sd/sqrt(N); usar sd/sqrt(N) a secas subestima su
# error en un factor sqrt(2) y hace ver al pareado peor de lo que es.
sd_marg = float(np.median(Y.std(axis=1, ddof=1)))
se_indep = np.sqrt(2) * sd_marg / np.sqrt(N)
print(f"\n[A3] Calibracion de tolerance")
print(f"  SE de la diferencia de medias, PAREADO   = {tol:.5f} pp   <- tolerance")
print(f"  SE de la diferencia de medias, INDEP.    = {se_indep:.5f} pp")
print(f"  brecha = {se_indep/tol:.2f}x  ->  el independiente necesita "
      f"{(se_indep/tol)**2:.1f}x mas instancias para la misma precision")
print(f"  (sd marginal = {sd_marg:.4f} pp; sd de la diferencia pareada = "
      f"{tol*np.sqrt(N):.4f} pp)")
print(f"  => tolerance = {tol:.4f}")

# separacion entre configuraciones, en unidades de ese SE
print(f"\n  separacion entre configuraciones consecutivas (en SE pareados):")
for i in range(S - 1):
    d = Y[i+1].mean() - Y[i].mean()
    print(f"    alpha {thetas[i]:>4} -> {thetas[i+1]:<4}  {d:+8.4f} pp = "
          f"{d/se[i, i+1]:+7.1f} SE" +
          ("   <- por DEBAJO de tolerance: empate" if abs(d) < tol else ""))

# --- Analisis 4: cost_aware ----------------------------------------------- #
print(f"\n[A4] cost_aware")
try:
    per_fam = defaultdict(list)
    with open(ROOT / "runs" / "stage2_evals.csv") as fh:
        for row in csv.DictReader(fh):
            if row["status"] == "ok":
                per_fam[row["family"]].append(float(row["seconds"]))
    med = {f: float(np.median(v)) for f, v in per_fam.items()}
    lo, hi = min(med.values()), max(med.values())
    allsec = [s for v in per_fam.values() for s in v]
    print(f"  segundos por ejecucion: min {min(allsec):.2f}  mediana "
          f"{np.median(allsec):.2f}  max {max(allsec):.2f}")
    for f in sorted(med, key=lambda x: int(x[2:])):
        print(f"    {f:<6} mediana {med[f]:6.2f}s  max {max(per_fam[f]):6.2f}s")
    print(f"  dispersion entre familias = {hi/lo:.2f}x  "
          f"(>3x justificaria cost_aware=True)")
    print(f"  => cost_aware = {hi/lo > 3.0}")
    print(f"\n  PEOR wall observado = {max(allsec):.2f}s "
          f"(exceso {100*(max(allsec)-30)/30:+.0f}% sobre -t 30)")
except FileNotFoundError:
    print("  (sin stage2_evals.csv)")

print("\n" + "=" * 78)
print(f"RESUMEN PARA LA ETAPA 3:")
print(f"  estimator  = paired  ({veredicto})")
print(f"  tolerance  = {tol:.4f}")
print(f"  cost_aware = decidido arriba")
print("=" * 78)
