#!/usr/bin/env python3
"""Consolida los replays (runs/stage3_replay_*_result.json) y las corridas reales
en runs/ablation.json, e imprime la tabla del 2x2 denominador x estimador."""
import glob, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
# Referencia = ganador del barrido por medias sobre instancias comunes con
# tolerance, calculado desde el CSV del run exacto (runs/status.py), asi que
# se vuelve definitiva sola cuando el run cierra.  Fallback: el valor visto.
sys.path.insert(0, HERE)
try:
    from status import compute as _status, ORDER as _ORDER
    _w = _status("exact")["winner"]
    REF = tuple(float(_w[k]) for k in _ORDER)
except Exception as _e:                       # pragma: no cover
    REF = (4.0, 2.0, 0.4, 0.01)
    print(f"(referencia fija; no pude derivarla: {_e})", file=sys.stderr)

def cell_of(d):
    return "imputa" if d.get("impute") else ("honesto" if d.get("honest") else "sesgado")

def estimator_label(d):
    """Antes de la integracion, "paired" era el estimador legado; desde 0.2.0 los
    result.json llevan estimator_class y "paired" significa K3."""
    cls = d.get("estimator_class")
    if cls: return {"PairedEstimatorK3": "paired_k3", "PairedEstimatorK2": "paired_k2",
                    "PairedEstimatorLegacy": "paired_legacy", "IndependentEstimator": "independent"}.get(cls, cls)
    e = d.get("estimator", "paired")
    return "paired_legacy" if e == "paired" else e

rows = []
for f in sorted(glob.glob(os.path.join(HERE, "stage3_replay_*_result.json")) + glob.glob(os.path.join(HERE, "stage3_var_*_result.json"))):
    d = json.load(open(f)); tag = re.sub(r"^stage3_(replay_)?|_result\.json$", "", os.path.basename(f))
    if tag.startswith("var_orig_"): continue
    if any(tag.startswith(x) for x in ("calr_","calv2_","calv3_","cal_","ab_out_","ab_auto_","regress","smoke")): continue   # calibracion: otro orden/umbral, no va a la tabla        # el motor original quedo descartado; sus 20 semillas no van a la tabla
    m = re.search(r"_s(\d+)$", tag)
    rows.append(dict(tag=tag, denominador=cell_of(d), estimador=estimator_label(d),
                     seed=int(m.group(1)) if m else 0, runs=d["runs"], speedup=round(d["speedup"], 2),
                     L=round(d["likelihood"], 4), output=list(d["output"]),
                     correcto=tuple(d["output"]) == REF, misses=d.get("misses", 0),
                     solver_real=bool(d.get("misses", 0)) and d.get("honest") is not None))
for tag, f in (("fase1_real", "stage3_phase1_result.json"), ("exacto_intento1", "stage3_exact_result.ABORTADO.json")):
    f = os.path.join(HERE, f)
    if os.path.exists(f):
        d = json.load(open(f))
        rows.append(dict(tag=tag, denominador="sesgado", estimador="paired_legacy", seed=0, runs=d["runs"],
                         speedup=round(d["speedup"], 2), L=round(d["likelihood"], 4), output=list(d["output"]),
                         correcto=tuple(d["output"]) == REF, misses=0, solver_real=True))
json.dump(rows, open(os.path.join(HERE, "ablation.json"), "w"), indent=1)

order = {"sesgado": 0, "honesto": 1, "imputa": 2}
rows.sort(key=lambda r: (order[r["denominador"]], r["estimador"], r["seed"], r["tag"]))
print(f"referencia: {REF}\n")
print(f"{'denominador':<9} {'estimador':<10} {'seed':>4} {'runs':>6} {'speedup':>8} {'L':>7}  {'output':<22} ok  tag")
for r in rows:
    print(f"{r['denominador']:<9} {r['estimador']:<10} {r['seed']:>4} {r['runs']:>6} {r['speedup']:>7.2f}x {r['L']:>7.4f}  "
          f"{str(tuple(r['output'])):<22} {'✓' if r['correcto'] else '✗'}  {r['tag']}")
print("\npor celda:")
cells = {}
for r in rows: cells.setdefault((r["denominador"], r["estimador"]), []).append(r)
for (den, est), v in cells.items():
    ok = [r for r in v if r["correcto"]]
    sp = lambda rs: f"{min(r['speedup'] for r in rs):.2f}–{max(r['speedup'] for r in rs):.2f}x" if rs else "—"
    print(f"  {den:<8} {est:<10} {len(ok)}/{len(v)} correctas · correctas {sp(ok)} · incorrectas {sp([r for r in v if not r['correcto']])}")
