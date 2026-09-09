#!/usr/bin/env python3
"""Tabla de calibracion para un prefijo de tag: fraccion correcta vs L reportada, por umbral.
    aggregate_calib.py <prefijo> [--orders 1-8] [--levels 0.8,0.98] [--json salida.json]
Busca runs/stage3_<prefijo>_o<o>_c<c>_result.json."""
import argparse, json, math, os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from status import compute, ORDER
REF=tuple(float(compute("exact")["winner"][k]) for k in ORDER)
p=argparse.ArgumentParser(); p.add_argument("prefix"); p.add_argument("--orders", default="1-8"); p.add_argument("--levels", default="0.8,0.98"); p.add_argument("--json")
a=p.parse_args(); lo,hi=map(int,a.orders.split("-")); levels=a.levels.split(",")
out={}; print(f"[{a.prefix}]  {'conf':>5} {'n':>3} {'ok':>3} {'frac':>5} {'IC95':>13} {'L media':>8} {'runs':>6} {'reales':>7}  incorrectas")
for c in levels:
    rows=[]
    for o in range(lo,hi+1):
        f=f"{HERE}/stage3_{a.prefix}_o{o}_c{c}_result.json"
        if os.path.exists(f):
            d=json.load(open(f)); rows.append(dict(order=o,runs=d["runs"],L=d["likelihood"],correct=tuple(d["output"])==REF,output=d["output"],misses=d.get("misses",0)))
    n=len(rows); ok=sum(r["correct"] for r in rows)
    if n:
        pr=ok/n; z=1.96; den=1+z*z/n; ctr=(pr+z*z/(2*n))/den; hw=z*math.sqrt(pr*(1-pr)/n+z*z/(4*n*n))/den
        print(f"{'':<{len(a.prefix)+3}}{c:>5} {n:>3} {ok:>3} {pr:>5.2f} [{max(0,ctr-hw):.2f}, {min(1,ctr+hw):.2f}] {sum(r['L'] for r in rows)/n:>8.3f} {sum(r['runs'] for r in rows)/n:>6.0f} {sum(r['misses'] for r in rows)/n:>7.0f}  {'; '.join('o'+str(r['order']) for r in rows if not r['correct'])}")
    out[c]=dict(n=n,correct=ok,rows=rows)
if a.json: json.dump(dict(ref=list(REF),levels=out,prefix=a.prefix),open(a.json,"w"),indent=1)
