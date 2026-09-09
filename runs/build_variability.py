#!/usr/bin/env python3
"""Variabilidad de la convergencia por motor, a partir de los replays.

Agrupa los result.json por motor (original vs cuatro arreglos), y para cada uno
guarda: trayectorias (runs, L), puntos de parada, speedups, correccion contra la
referencia, y una banda de cuantiles de L sobre una grilla comun de ejecuciones
(solo entre las corridas que siguen vivas en ese punto).  -> runs/variability.json
"""
import glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from status import compute as _status, ORDER as _ORDER
REF = tuple(float(_status("exact")["winner"][k]) for k in _ORDER)

def engine_of(d):
    if d.get("impute") and estimator_label(d) == "paired_k3": return "arreglado"
    if not d.get("honest") and not d.get("impute") and estimator_label(d) == "paired_legacy": return "original"
    return None                                  # variantes intermedias: fuera de esta figura

def estimator_label(d):
    """Antes de la integracion, "paired" era el estimador legado; desde 0.2.0 los
    result.json llevan estimator_class y "paired" significa K3."""
    cls = d.get("estimator_class")
    if cls: return {"PairedEstimatorK3": "paired_k3", "PairedEstimatorK2": "paired_k2",
                    "PairedEstimatorLegacy": "paired_legacy", "IndependentEstimator": "independent"}.get(cls, cls)
    e = d.get("estimator", "paired")
    return "paired_legacy" if e == "paired" else e

groups = {"original": [], "arreglado": []}
for f in sorted(glob.glob(os.path.join(HERE, "stage3_replay_*_result.json")) + glob.glob(os.path.join(HERE, "stage3_var_*_result.json"))):
    d = json.load(open(f)); e = engine_of(d)
    if not e or not d.get("history"): continue
    tag = re.sub(r"^stage3_(replay_)?|_result\.json$", "", os.path.basename(f))
    if any(tag.startswith(x) for x in ("calr_","calv2_","calv3_","cal_","d5_","ab_out_","ab_auto_","regress","smoke")): continue   # calibracion / A-B: fuera de la figura de variabilidad
    groups[e].append(dict(tag=tag, seed=d.get("seed", None), runs=d["runs"], speedup=d["speedup"],
                          L=d["likelihood"], output=list(d["output"]), correcto=tuple(d["output"]) == REF,
                          history=[[h[0], h[1]] for h in d["history"]]))
out = {"ref": list(REF), "engines": {}}
for e, rs in groups.items():
    if not rs: continue
    xmax = max(r["runs"] for r in rs); grid = list(range(0, xmax + 26, 25))
    M = np.full((len(rs), len(grid)), np.nan)
    for i, r in enumerate(rs):
        h = r["history"]; hx = [p[0] for p in h]; hy = [p[1] for p in h]
        for j, x in enumerate(grid):
            if x > r["runs"]: break                        # ya paro
            k = np.searchsorted(hx, x, side="right") - 1
            M[i, j] = hy[k] if k >= 0 else np.nan
    alive = np.sum(~np.isnan(M), axis=0)
    q = lambda p: [float(np.nanpercentile(M[:, j], p)) if alive[j] else None for j in range(len(grid))]
    out["engines"][e] = dict(
        n=len(rs), correctas=sum(r["correcto"] for r in rs),
        stops=sorted(r["runs"] for r in rs), speedups=sorted(r["speedup"] for r in rs),
        stop_L=[r["L"] for r in rs], correct_flags=[r["correcto"] for r in rs],
        grid=grid, q10=q(10), q50=q(50), q90=q(90), alive=[int(a) for a in alive],
        runs=[dict(tag=r["tag"], runs=r["runs"], speedup=r["speedup"], L=r["L"],
                   output=r["output"], correcto=r["correcto"],
                   history=r["history"][::max(1, len(r["history"]) // 120)]) for r in rs])
json.dump(out, open(os.path.join(HERE, "variability.json"), "w"))
for e, g in out["engines"].items():
    st = g["stops"]; sp = g["speedups"]
    print(f"{e:<10} n={g['n']:>2}  correctas {g['correctas']}/{g['n']}  paradas: min {st[0]} mediana {st[len(st)//2]} max {st[-1]}"
          f"  speedup: min {sp[0]:.2f} mediana {sp[len(sp)//2]:.2f} max {sp[-1]:.2f}")
print("referencia:", REF, "->", os.path.join(HERE, "variability.json"))
