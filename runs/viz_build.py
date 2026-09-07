"""Genera runs/dashboard.html desde los artefactos del run.

Re-ejecutable: cada corrida toma el estado actual de los logs y CSVs.  La
trayectoria de la fase 1 se lee del LOG, no del history.csv, porque ese ultimo
recien se escribe cuando el run termina.
"""
from __future__ import annotations

import csv, glob, json, os, re, sys, time
from collections import defaultdict
from pathlib import Path
import pathlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t4exps.utils import default_family                      # noqa: E402

R = ROOT / "runs"
SNAP = re.compile(r"\[\s*(\d+) runs\] L=\s*([\d.]+)% depth=(\d+)")


def traj_from_log(path: Path, tags: dict[str, str] | None = None):
    """Extrae [(runs, likelihood, depth)] del stdout verbose del Engine."""
    if not path.exists():
        return {} if tags else []
    out, cur = defaultdict(list), None
    for ln in path.read_text().splitlines():
        if tags:
            for t, pat in tags.items():
                if pat in ln:
                    cur = t
        m = SNAP.match(ln.strip())
        if m:
            row = [int(m[1]), float(m[2]) / 100, int(m[3])]
            out[cur if tags else "_"].append(row)
    return dict(out) if tags else out["_"]


d = {"stamp": time.strftime("%Y-%m-%d %H:%M")}

# --- trayectorias --------------------------------------------------------- #
t1 = traj_from_log(R / "stage1.log",
                   {"j8": "[6.8] Engine n_jobs=8", "j1": "[6.1] Engine n_jobs=1"})
d["traj"] = {
    "piloto_j8": t1.get("j8", []),
    "piloto_j1": t1.get("j1", []),
    "ensayo": traj_from_log(R / "stage3_rehearsal.log"),
    "fase1": traj_from_log(R / "stage3_phase1.log"),
}

# --- etapa 2 -------------------------------------------------------------- #
z = np.load(R / "stage2_evals.npz", allow_pickle=True)
Y, thetas = z["Y"], z["thetas"]
S, N = Y.shape
D = Y[:, None, :] - Y[None, :, :]
se = D.std(axis=2, ddof=1) / np.sqrt(N)
iu = np.triu_indices(S, k=1)
m = Y.mean()
alpha, beta = Y.mean(axis=1) - m, Y.mean(axis=0) - m
E = Y - m - alpha[:, None] - beta[None, :]
v_resid = (E ** 2).sum() / ((S - 1) * (N - 1))
v_inst = max(beta.var(ddof=1) - v_resid / S, 0.0)
v_strat = max(alpha.var(ddof=1) - v_resid / N, 0.0)
sd_marg = float(np.median(Y.std(axis=1, ddof=1)))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


rho = np.array([[spearman(Y[i], Y[j]) for j in range(S)] for i in range(S)])
d["stage2"] = {
    "alphas": [float(a) for a in thetas],
    "means": [float(x) for x in Y.mean(axis=1)],
    "sds": [float(x) for x in Y.std(axis=1, ddof=1)],
    "se_paired": float(np.median(se[iu])),
    "se_indep": float(np.sqrt(2) * sd_marg / np.sqrt(N)),
    "se_consecutive": [float(se[i, i + 1]) for i in range(S - 1)],
    "var": {"inst": float(v_inst), "strat": float(v_strat), "resid": float(v_resid),
            "frac_inst": float(v_inst / (v_inst + v_strat + v_resid))},
    "rho_min": float(rho[iu].min()),
    "N": int(N),
}

# --- costo por familia ----------------------------------------------------- #
per = defaultdict(list)
with open(R / "stage2_evals.csv") as fh:
    for row in csv.DictReader(fh):
        if row["status"] == "ok":
            per[row["family"]].append(float(row["seconds"]))
d["cost"] = [{"fam": f, "median": float(np.median(per[f])),
              "max": max(per[f]), "n": len(per[f])}
             for f in sorted(per, key=lambda x: int(x[2:]))]

# --- estado en vivo de la fase 1 -------------------------------------------- #
rows = list(csv.DictReader(open(R / "stage3_phase1_evals.csv"))) \
    if (R / "stage3_phase1_evals.csv").exists() else []
d["phase1"] = {
    "n": len(rows),
    "errors": sum(1 for r in rows if r["status"] != "ok"),
    "elapsed": float(rows[-1]["t_wall"]) if rows else 0.0,
    "strategies": len({r["theta"] for r in rows}),
    "mean_sec": float(np.mean([float(r["seconds"]) for r in rows])) if rows else 0.0,
}

# --- trayectoria completa del run exacto (los dos tramos) -------------------- #
ex = []
for f in ("stage3_exact.PARCIAL.log", "stage3_exact.log"):
    ex += traj_from_log(R / f)
seen = {}
for runs, l, dep in ex:
    seen[runs] = (l, dep)                       # el resume repite el reinicio
d["traj_exact"] = [[r, v[0], v[1]] for r, v in sorted(seen.items())]

# --- en que ejecucion se completo cada estrategia ---------------------------- #
comp, seen_c = [], defaultdict(int)
if (R / "stage3_exact_evals.csv").exists():
    with open(R / "stage3_exact_evals.csv", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh), 1):
            seen_c[row["theta"]] += 1
            if seen_c[row["theta"]] == 1500:
                comp.append([i, row["theta"]])
d["completions"] = comp

# --- estado en vivo del run exacto, con la misma logica que runs/status.py --- #
sys.path.insert(0, str(R))
from status import compute as _status                            # noqa: E402
_l = _status("exact")
if "missing" in _l or _l.get("empty"):
    d["live"] = None
else:
    d["live"] = {k: _l[k] for k in (
        "n", "errors", "done", "path_done", "on_path", "goal", "off_path",
        "rate", "seg_t", "elapsed", "attempts", "foreign", "pid", "estado")}
    d["live"]["sweep"] = _l["sweep"]
    d["live"]["path_counts"] = {k: _l["counts"].get(k, 0) for k in _l["sweep"]}
    d["live"]["winner"] = _l["winner"]

# --- ablacion: replays contra el oraculo (2x2 denominador x estimador) ------- #
abp = R / "ablation.json"
d["ablation"] = json.loads(abp.read_text()) if abp.exists() else []

# --- trayectorias de los replays de imputacion (historia guardada en el json) -- #
d["traj_impute"] = {}
for f in sorted(glob.glob(str(R / "stage3_replay_imp_k3_solver_s*_result.json"))):
    j = json.loads(pathlib.Path(f).read_text())
    seed = re.search(r"_s(\d+)_result", f).group(1)
    d["traj_impute"][seed] = {"history": j.get("history", []), "runs": j["runs"],
                              "speedup": j["speedup"], "output": j["output"]}

# --- variabilidad de la convergencia del motor arreglado (muchas semillas) ---- #
vp = R / "variability.json"
d["variability"] = json.loads(vp.read_text()) if vp.exists() else None

# --- calibracion: fraccion correcta vs likelihood reportada, por umbral -------- #
cp_ = R / "calibration.json"
d["calibration"] = json.loads(cp_.read_text()) if cp_.exists() else None
abp_ = R / "ab_impact.json"
d["ab_impact"] = json.loads(abp_.read_text()) if abp_.exists() else None
crp_ = R / "calibration_random.json"
d["calibration_random"] = json.loads(crp_.read_text()) if crp_.exists() else None

# --- anotaciones de la trayectoria (sobrevivientes por punto) ---------------- #
ap = R / "annotations.json"
d["annot"] = json.loads(ap.read_text()) if ap.exists() else {}

# --- resultado final, si el run ya cerro ------------------------------------ #
rp = R / "stage3_phase1_result.json"
d["result"] = json.loads(rp.read_text()) if rp.exists() else None

tpl = (R / "dashboard.tpl.html").read_text()
assert "__DATA__" in tpl, "falta el marcador __DATA__ en el template"
blob = json.dumps(d, separators=(",", ":"), allow_nan=False)
out = tpl.replace("__DATA__", blob, 1)
assert "__DATA__" not in out
(R / "dashboard.html").write_text(out)
print(f"runs/dashboard.html  ({len(out)//1024} KB)")
print(f"  fase1: {d['phase1']['n']} evals, {d['phase1']['errors']} errores, "
      f"{len(d['traj']['fase1'])} puntos de likelihood")
for k, v in d["traj"].items():
    print(f"  traj[{k}]: {len(v)} puntos")
