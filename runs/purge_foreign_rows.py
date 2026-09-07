#!/usr/bin/env python3
"""Quita del CSV de un run las filas escritas por OTRO proceso.

Un probe (annotate_trajectory / diag_sims, antes del ReadOnlyRunner) llego a
ejecutar el solver y a escribir en el CSV vivo del run.  Esas filas se
reconocen porque `t_wall` es monotono dentro de una corrida: una fila con
t_wall menor que la cadena, que NO inaugura un relanzamiento (>= 50 filas
seguidas por debajo), es ajena.  Misma regla que runs/status.py.

    ./runs/purge_foreign_rows.py exact          # dry-run: muestra que sacaria
    ./runs/purge_foreign_rows.py exact --apply  # escribe; guarda las ajenas aparte

Se niega a aplicar mientras el run este vivo: tiene el archivo abierto en
append y acortarlo bajo su offset corrompe las proximas escrituras.
"""
import csv, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def attempts_of(times, min_break=50):
    attempts, chain, chain_t, pending = [], [], -1.0, []
    for i, tw in enumerate(times):
        if tw >= chain_t:
            chain.append(i); chain_t = tw; pending = []
        else:
            pending.append(i)
            if len(pending) >= min_break:
                attempts.append(chain); chain, pending = pending, []
                chain_t = times[chain[-1]]
    attempts.append(chain)
    return attempts


def run_alive(tag):
    out = subprocess.run(["pgrep", "-af", "stage3_full.py"],
                         capture_output=True, text=True).stdout
    return any("python" in l and f"--tag {tag}" in l for l in out.splitlines())


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "exact"
    apply = "--apply" in sys.argv
    path = os.path.join(HERE, f"stage3_{tag}_evals.csv")
    with open(path, newline="") as fh:
        rd = csv.reader(fh); header = next(rd); rows = list(rd)
    tcol = header.index("t_wall")
    times = [float(r[tcol]) for r in rows]
    keep = {i for ch in attempts_of(times) for i in ch}
    foreign = [i for i in range(len(rows)) if i not in keep]

    print(f"{len(rows)} filas, {len(attempts_of(times))} corridas, "
          f"{len(foreign)} ajenas")
    for i in foreign[:8]:
        print(f"   fila {i:>6}  t_wall={times[i]:>7.1f}  {rows[i][2]}  "
              f"...{rows[i][3][-28:]}")
    if len(foreign) > 8:
        print(f"   ... y {len(foreign)-8} mas")
    if not foreign:
        return
    if not apply:
        print("\n(dry-run; --apply para escribir)")
        return
    if run_alive(tag):
        sys.exit("!! el run sigue vivo con el CSV abierto: no se aplica")

    bak = path + ".con_ajenas.bak"
    shutil.copy2(path, bak)
    side = os.path.join(HERE, f"stage3_{tag}_evals.FOREIGN.csv")
    with open(side, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header)
        for i in foreign: w.writerow(rows[i])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header)
        for i in range(len(rows)):
            if i in keep: w.writerow(rows[i])
    print(f"\nescrito {path}  ({len(keep)} filas)\n"
          f"ajenas -> {side}\nrespaldo -> {bak}")


if __name__ == "__main__":
    main()
