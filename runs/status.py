#!/usr/bin/env python3
"""Estado del run en curso.  Sin dependencias: corre con cualquier python3.

    ./runs/status.py              una foto
    ./runs/status.py 60           refresca cada 60 s
    ./runs/status.py 60 phase1    otro tag
"""
import collections
import csv
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOTAL_INSTANCES = 1500

# El barrido secuencial visita, para cada parametro, todos sus valores partiendo
# del ganador de los parametros anteriores.  Asi que el CAMINO son 14
# estrategias, no todas las que el motor llego a tocar: cuando el camino
# predicho se mueve, deja ramas a medio evaluar que nunca se completan.
GRID = {"a": [0.0, 1.0, 2.0, 4.0, 8.0], "b": [0.5, 1.0, 2.0, 4.0],
        "g": [0.1, 0.2, 0.3, 0.4], "p": [0.01, 0.02, 0.03, 0.04]}
START = {"a": 0.0, "b": 0.5, "g": 0.1, "p": 0.01}
ORDER = ["a", "b", "g", "p"]


def key_of(th):
    return ";".join(f"{k}={th[k]}" for k in sorted(th))


def path_for(winner):
    """Las estrategias que el barrido necesita completas, dado el ganador."""
    th, out = dict(START), []
    for name in ORDER:
        for v in GRID[name]:
            out.append(key_of({**th, name: v}))
        th[name] = winner[name]
    seen, uniq = set(), []
    for k in out:
        if k not in seen:
            seen.add(k); uniq.append(k)
    return uniq


def last_output(log):
    """El ultimo output predicho que registro el log, para inferir el camino."""
    if not os.path.exists(log):
        return None
    best = None
    for ln in open(log):
        if ln.startswith("  mejor configuracion") or "-> (" in ln:
            pass
    hist = log.replace(".log", "_history.csv")
    if os.path.exists(hist):
        rows = list(csv.DictReader(open(hist)))
        if rows:
            best = rows[-1]["output"]
    return best


def alive(tag):
    try:
        out = subprocess.run(["pgrep", "-af", "stage3_full.py"],
                             capture_output=True, text=True).stdout
    except FileNotFoundError:
        return None
    # Solo una invocacion real: "<pid> <...>python[3] [-u] <...>stage3_full.py ...".
    # Un `bash -c "..."` que cite el comando en su texto no empieza asi y no cuenta.
    import re
    pat = re.compile(r"^(\d+)\s+\S*python[\d.]*\s+(?:-u\s+)?\S*stage3_full\.py\b.*--tag\s+" + re.escape(tag) + r"\b")
    for ln in out.splitlines():
        m = pat.match(ln)
        if m:
            return int(m.group(1))
    return None


def compute(tag="exact"):
    """Todo lo que status.py sabe del run, como dict (sin imprimir)."""
    path = os.path.join(HERE, f"stage3_{tag}_evals.csv")
    log = os.path.join(HERE, f"stage3_{tag}.log")
    if not os.path.exists(path):
        return {"missing": path}

    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        return {"empty": True}
    n = len(rows)
    errors = sum(1 for r in rows if r["status"] != "ok")
    counts = collections.Counter(r["theta"] for r in rows)   # se filtra abajo
    done = sum(1 for v in counts.values() if v >= TOTAL_INSTANCES)

    # el camino se infiere de la configuracion que el motor cree ganadora ahora
    # El history solo se escribe al terminar, y el `-> <bsg ...>` del log es la
    # PROXIMA estrategia a evaluar, no la ganadora.  Asi que el camino se
    # reconstruye de los datos: se replica el barrido comparando medias sobre
    # las instancias que las dos estrategias tienen en comun, que es lo que
    # hace `Engine.compare_observed`.  El orden no hace falta -- una media
    # sobre un conjunto no depende de como se recorra.
    vals = collections.defaultdict(dict)
    for r in rows:
        if r["status"] == "ok" and r["value"]:
            vals[r["theta"]][r["instance"]] = float(r["value"])

    def better(k1, k2, tol=0.0116):
        """True si k1 le gana a k2 sobre sus instancias comunes."""
        a1, a2 = vals.get(k1), vals.get(k2)
        if not a1 or not a2:
            return True                      # sin datos, gana el titular
        common = a1.keys() & a2.keys()
        if not common:
            return True
        m1 = sum(a1[i] for i in common) / len(common)
        m2 = sum(a2[i] for i in common) / len(common)
        return (m1 - m2) >= -tol

    winner = dict(START)
    for name in ORDER:
        for v in GRID[name]:
            cand = {**winner, name: v}
            if not better(key_of(winner), key_of(cand)):
                winner = cand
    sweep = path_for(winner)
    on_path = sum(min(counts.get(k, 0), TOTAL_INSTANCES) for k in sweep)
    goal = len(sweep) * TOTAL_INSTANCES
    off_path = n - on_path
    path_done = sum(1 for k in sweep if counts.get(k, 0) >= TOTAL_INSTANCES)

    quiet = time.time() - os.path.getmtime(path)
    pid = alive(tag)
    if pid:
        estado = "CORRIENDO"
        if quiet > 1200:
            estado = f"ESTANCADO ({quiet/60:.0f} min sin escribir)"
    else:
        estado = "COMPLETO" if path_done == len(sweep) else "DETENIDO SIN TERMINAR"
    # `t_wall` vuelve a cero en cada relanzamiento con --resume, asi que hay
    # que partir el CSV en tramos y sumarlos.  El ritmo se mide sobre el tramo
    # actual, que es el unico que refleja la velocidad de ahora.
    # `t_wall` es monotono dentro de UNA corrida (se asigna bajo lock).  Una
    # fila con t_wall menor que la cadena es o bien un relanzamiento real
    # (todas las filas siguientes quedan por debajo) o bien una fila AJENA:
    # otro proceso escribiendo al mismo CSV, p.ej. un probe que ejecuto el
    # solver.  Se distinguen por persistencia: un relanzamiento rompe la
    # cadena durante >= 50 filas seguidas; una fila ajena la rompe una vez.
    times = [float(r["t_wall"]) for r in rows]
    attempts, chain, chain_t, pending = [], [], -1.0, []
    for i, tw in enumerate(times):
        if tw >= chain_t:
            chain.append(i); chain_t = tw; pending = []
        else:
            pending.append(i)
            if len(pending) >= 50:
                attempts.append(chain)
                chain, pending = pending, []
                chain_t = times[chain[-1]]
    attempts.append(chain)
    in_attempt = {i for ch in attempts for i in ch}
    foreign = [i for i in range(n) if i not in in_attempt]
    starts = [ch[0] for ch in attempts]
    elapsed = sum(times[ch[-1]] - times[ch[0]] for ch in attempts)
    main = attempts[-1]
    seg_from = main[0]
    seg_n, seg_t = len(main), times[main[-1]] - times[main[0]]
    rate = seg_n / seg_t * 3600 if seg_t > 60 else 0


    return dict(tag=tag, n=n, errors=errors, done=done, sweep=sweep,
                on_path=on_path, goal=goal, off_path=off_path,
                path_done=path_done, counts=counts, winner=winner,
                elapsed=elapsed, seg_t=seg_t, seg_n=seg_n, rate=rate,
                attempts=len(attempts), foreign=len(foreign), quiet=quiet,
                pid=pid, estado=estado, log=log)


def report(tag="exact"):
    c = compute(tag)
    if "missing" in c:
        print(f"no existe {c['missing']}"); return
    if c.get("empty"):
        print("CSV vacio"); return
    globals().update({k: v for k, v in c.items()})  # nombres que usa la impresion
    n, errors, done, sweep, on_path, goal, off_path, path_done = (c[k] for k in
        ("n","errors","done","sweep","on_path","goal","off_path","path_done"))
    counts, winner, elapsed, seg_t, seg_n, rate = (c[k] for k in
        ("counts","winner","elapsed","seg_t","seg_n","rate"))
    quiet, pid, estado, log = c["quiet"], c["pid"], c["estado"], c["log"]
    starts = list(range(c["attempts"])); foreign = list(range(c["foreign"]))
    rows = list(csv.DictReader(open(os.path.join(HERE, f"stage3_{tag}_evals.csv"), newline="")))
    bar_w = 40
    filled = min(bar_w, int(bar_w * on_path / goal))
    print(f"\n  run '{tag}'   {estado}" + (f"   pid {pid}" if pid else ""))
    print(f"  [{'#'*filled}{'.'*(bar_w-filled)}] {on_path}/{goal} del camino "
          f"({100*on_path//goal}%)")
    print(f"  estrategias del camino: {path_done}/{len(sweep)} completas")
    print(f"  fuera del camino      : {off_path} evals en "
          f"{len(counts)-len(sweep)} ramas abandonadas "
          f"({100*off_path//max(n,1)}% del trabajo)")
    print(f"  errores               : {errors}")
    if foreign:
        print(f"  FILAS AJENAS          : {len(foreign)} escritas por otro proceso "
              f"(purgar al terminar)")
    if len(starts) > 1:
        print(f"  esta corrida          : {seg_t/3600:.1f} h   "
              f"({seg_n} evaluaciones nuevas)")
        print(f"  acumulado             : {elapsed/3600:.1f} h de CPU en "
              f"{len(starts)} intentos")
    else:
        print(f"  transcurrido          : {elapsed/3600:.1f} h")
    if rate:
        print(f"  ritmo actual          : {rate:.0f} evaluaciones/h")
        if pid:
            print(f"  faltan                : ~{(goal-on_path)/rate:.1f} h "
                  f"({goal-on_path} evaluaciones del camino)")
    else:
        print(f"  ritmo actual          : midiendo (tramo de {seg_t:.0f} s)")
    print(f"  ultima evaluacion     : hace {quiet/60:.0f} min")

    print(f"\n  camino actual (ganador {key_of(winner)}):")
    for k in sweep:
        c = counts.get(k, 0)
        w = int(24 * c / TOTAL_INSTANCES)
        print(f"    {c:>5}/{TOTAL_INSTANCES} [{'#'*w}{'.'*(24-w)}] {k}")

    if os.path.exists(log):
        tail = [l.rstrip() for l in open(log) if "runs] L=" in l][-3:]
        if tail:
            print("\n  ultimas iteraciones:")
            for l in tail:
                print("   ", l)
    if errors:
        print("\n  ULTIMOS ERRORES:")
        for r in [x for x in rows if x["status"] != "ok"][-3:]:
            print(f"    {r['status']} {r['instance'][:60]} {r['detail'][:60]}")
    if not pid and done != len(counts):
        print("\n  para retomar:")
        print("    .venv/bin/python -u runs/stage3_full.py --confidence 2.0 \\")
        print("      --tolerance 0.0116 --timeout 75 --jobs 8 --tag exact --resume \\")
        print("      >> runs/stage3_exact.log 2>&1 &")
    print()


if __name__ == "__main__":
    every = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tag = sys.argv[2] if len(sys.argv) > 2 else "exact"
    if not every:
        report(tag)
    else:
        try:
            while True:
                os.system("clear")
                print(time.strftime("  %F %T"))
                report(tag)
                time.sleep(every)
        except KeyboardInterrupt:
            pass
