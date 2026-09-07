#!/usr/bin/env python3
"""Oraculo extendido: la matriz del run exacto + las evaluaciones REALES que
pagaron los replays con --allow-solver (status 'ok' en sus CSV).  Un par
(estrategia, instancia) repetido conserva la primera aparicion."""
import csv, glob, os
HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, "oracle_extended.csv")
seen, rows, src = set(), [], {}
def take(path, label):
    n = 0
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh); hdr = rd.fieldnames
        for r in rd:
            if r["status"] != "ok" or not r["value"]: continue
            k = (r["strategy"], r["theta"], r["instance"])
            if k in seen: continue
            seen.add(k); rows.append(r); n += 1
    src[label] = n; return hdr
hdr = take(os.path.join(HERE, "stage3_exact_evals.csv"), "exacto")
for f in sorted(glob.glob(os.path.join(HERE, "stage3_replay_*_solver_*_evals.csv"))):
    take(f, os.path.basename(f))
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=hdr); w.writeheader(); w.writerows(rows)
strategies = {(r["strategy"], r["theta"]) for r in rows}
print(f"{out}: {len(rows)} pares unicos, {len(strategies)} estrategias")
for k, v in src.items(): print(f"   +{v:>6}  {k}")
