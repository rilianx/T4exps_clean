"""Agrega el A/B (impact_on) y la calibracion con ordenes aleatorios."""
import json, os, sys, math, statistics as st
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from status import compute, ORDER
REF=tuple(float(compute("exact")["winner"][k]) for k in ORDER)
def load(f): return json.load(open(f)) if os.path.exists(f) else None
print("=== A/B impact_on: output vs prefijo (semillas 30-39, 0.98) ===")
pr=[];ou=[];rows=[]
for sd in range(30,40):
    a=load(f"{HERE}/stage3_cal_c0.98_s{sd}_result.json"); b=load(f"{HERE}/stage3_ab_out_s{sd}_result.json")
    z=lambda d: sum(1 for h in d["history"] if h[1]==0) if d else None
    print(f"  s{sd}: prefijo {(a['runs'] if a else '—'):>5}  output {(b['runs'] if b else '—'):>5}  ok {('✓' if a and tuple(a['output'])==REF else '—')}/{('✓' if b and tuple(b['output'])==REF else ('✗' if b else '—'))}  L0 {z(a)}/{z(b)}")
    if a and b: pr.append(a["runs"]); ou.append(b["runs"]); rows.append(dict(seed=sd,prefix=a["runs"],output=b["runs"],ok_prefix=tuple(a["output"])==REF,ok_output=tuple(b["output"])==REF,L0_prefix=z(a),L0_output=z(b)))
if pr: print(f"  pares {len(pr)}: prefijo {st.mean(pr):.0f} vs output {st.mean(ou):.0f}  delta {100*(st.mean(ou)/st.mean(pr)-1):+.0f}%  output gana {sum(o<p for p,o in zip(pr,ou))}/{len(pr)}  speedup {21000/st.mean(pr):.2f}x -> {21000/st.mean(ou):.2f}x")
json.dump(dict(ref=list(REF),pairs=rows),open(f"{HERE}/ab_impact.json","w"),indent=1)
print("\n=== calibracion honesta: ordenes aleatorios ===")
out={}
print(f"  {'conf':>5} {'n':>3} {'ok':>3} {'frac':>5} {'IC95':>13} {'L media':>8} {'runs':>6} {'reales':>7}  incorrectas")
for c in ("0.8","0.98"):
    rows=[]
    for o in range(1,9):
        d=load(f"{HERE}/stage3_replay_calr_solver_o{o}_c{c}_result.json")
        if d: rows.append(dict(order=o,runs=d["runs"],L=d["likelihood"],correct=tuple(d["output"])==REF,output=d["output"],misses=d.get("misses",0)))
    n=len(rows); ok=sum(r["correct"] for r in rows)
    if n:
        p=ok/n; zz=1.96; den=1+zz*zz/n; ctr=(p+zz*zz/(2*n))/den; hw=zz*math.sqrt(p*(1-p)/n+zz*zz/(4*n*n))/den
        bad="; ".join(f"o{r['order']}" for r in rows if not r["correct"])
        print(f"  {c:>5} {n:>3} {ok:>3} {p:>5.2f} [{max(0,ctr-hw):.2f}, {min(1,ctr+hw):.2f}] {sum(r['L'] for r in rows)/n:>8.3f} {sum(r['runs'] for r in rows)/n:>6.0f} {sum(r['misses'] for r in rows)/n:>7.0f}  {bad}")
    out[c]=dict(n=n,correct=ok,rows=rows)
json.dump(dict(ref=list(REF),levels=out,design="orden aleatorio por replica (order_seed 1..8), MC seed 100+o, solver para pares fuera del oraculo"),open(f"{HERE}/calibration_random.json","w"),indent=1)
