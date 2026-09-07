"""Anota una trayectoria de likelihood con cuantas simulaciones sobrevivieron.

`Snapshot.likelihood` es `aciertos / simulaciones_completadas`, y las que
abortan se descartan (engine.py:161-163).  Como una simulacion aborta cuando
su camino necesita una estrategia sin datos -- o sea cuando DISCREPA con la
prediccion -- el denominador esta correlacionado con el acuerdo y la cifra
queda sesgada hacia arriba.  Sin saber cuantas sobrevivieron, un 99% y un 50%
no se pueden comparar.

Reconstruye el cache del motor truncado a las primeras N filas del CSV (que
esta en orden de finalizacion, asi que N filas = el estado real tras N
ejecuciones) y re-corre solo la parte Monte Carlo.  No usa el solver.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "runs"))

from t4exps import Engine, order_instances
from t4exps.core import Context, SimulationAbort, set_context
from monitor import LoggingRunner
from stage3_full import make_experiment


class ReadOnlyRunner(LoggingRunner):
    """LoggingRunner que no puede ejecutar el solver: solo sirve el cache."""

    def _evaluate(self, s, instance):
        raise RuntimeError(
            f"el probe intento EJECUTAR el solver para {s.params} en "
            f"{instance[-40:]}: el estado truncado no cubre el camino predicho")


def state_at(csv_path: Path, n: int, instances):
    """evals[(name,theta)] -> prefijo ordenado, usando solo las primeras n filas."""
    by = {}
    with open(csv_path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i >= n:
                break
            if row["status"] == "ok" and row["value"]:
                by.setdefault((row["strategy"], row["theta"]), {})[
                    row["instance"]] = float(row["value"])
    out = {}
    for k, vals in by.items():
        c = 0
        for pi in instances:
            if pi not in vals:
                break
            c += 1
        if c:
            out[k] = c
    return by, out


def probe(tag: str, n: int, instances, tolerance: float, nsims: int = 400):
    eng = Engine(make_experiment(30), instances, estimator="paired", n_jobs=1,
                 tolerance=tolerance, confidence=1.0, nruns=15, batch=25,
                 instance_order="file", seed=0, nsims=nsims)
    r = ReadOnlyRunner(instances, n_jobs=1,
                       csv_path=str(ROOT / "runs" / ".probe_scratch.csv"),
                       echo_every=0)
    r._by_key, r._resumed = state_at(
        ROOT / "runs" / f"stage3_{tag}_evals.csv", n, instances)
    eng.runner = r
    try:
        output, ctx = eng.predictive_execution()
    except RuntimeError as e:
        # El camino que el motor predice con este cache truncado pasa por una
        # estrategia sin datos: el motor real habria gastado 15 ejecuciones
        # ahi.  Un probe no debe pagarlas; se reporta el punto como no
        # evaluable en vez de ejecutar el solver sobre el CSV vivo.
        r.close()
        return None, -1, -1
    draws = eng._draws([s.key for s in ctx.seen])
    ok = matched = 0
    for sim in range(min(nsims, draws.sums.shape[1])):
        c = Context(mode="simulate", engine=eng, sums=draws.as_dict(sim))
        set_context(c)
        try:
            o = eng.experiment()
            ok += 1
            matched += (o == output)
        except SimulationAbort:
            pass
        finally:
            set_context(None)
    r.close()
    return output, ok, matched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tag")
    p.add_argument("--at", type=int, nargs="+", required=True,
                   help="numeros de ejecucion donde sondear")
    p.add_argument("--tolerance", type=float, default=0.0116)
    p.add_argument("--nsims", type=int, default=400)
    p.add_argument("--json", help="acumula el resultado en este archivo")
    a = p.parse_args()

    pool = [l.strip() for l in open(ROOT / "runs" / "instances_abs.txt") if l.strip()]
    instances = order_instances(pool, "stratified", seed=0)

    print(f"{'runs':>7} {'vivas':>7} {'aciertos':>9} {'L reportada':>12} "
          f"{'L honesta':>10}  output")
    print("-" * 78)
    import json
    path = Path(a.json) if a.json else None
    blob = json.loads(path.read_text()) if path and path.exists() else {}
    # se conserva lo ya calculado para este tag y se mergea por `runs`, asi un
    # job matado a mitad de camino no pierde los puntos que si termino
    rows = {r["runs"]: r for r in blob.get(a.tag, [])}
    for n in a.at:
        out, ok, m = probe(a.tag, n, instances, a.tolerance, a.nsims)
        if ok < 0:
            # un valor previo para este N se calculo con un runner que SI
            # podia ejecutar: 15 evaluaciones forzadas que el motor real no
            # tenia a ese N.  Se descarta en vez de conservarlo.
            stale = rows.pop(n, None)
            print(f"{n:>7} {'n/a':>7} {'n/a':>9} {'--':>12} {'--':>10}  "
                  f"camino no evaluable con el cache truncado"
                  + ("  [descartado valor previo]" if stale else ""), flush=True)
            if path:
                blob[a.tag] = [rows[k] for k in sorted(rows)]
                path.write_text(json.dumps(blob, indent=1))
            continue
        rep = m / ok if ok else 0.0
        honest = m / a.nsims
        print(f"{n:>7} {ok:>7} {m:>9} {rep:>11.1%} {honest:>10.1%}  {out}", flush=True)
        rows[n] = {"runs": n, "alive": ok, "matched": m, "nsims": a.nsims,
                   "reported": rep, "honest": honest, "output": str(out)}
        if path:
            blob[a.tag] = [rows[k] for k in sorted(rows)]
            path.write_text(json.dumps(blob, indent=1))
    if path:
        print(f"\n-> {path}")


if __name__ == "__main__":
    main()
