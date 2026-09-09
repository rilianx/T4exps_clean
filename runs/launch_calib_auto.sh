#!/usr/bin/env bash
# Calibracion con el default 0.3 (impact_on=auto, paired K3, prior mean).
#   launch_calib_auto.sh "<ordenes>" "<umbrales>" [paralelismo]
set -u; cd "$(dirname "$0")/.."; PY=.venv/bin/python
ORD=${1:-1-8}; LEV=${2:-0.98}; PAR=${3:-8}
run_one() { c=$1; o=$2; tag=calauto_o${o}_c${c}
  [ -f runs/stage3_${tag}_result.json ] && return 0
  $PY runs/replay.py --tag $tag --impute --estimator paired --impute-prior mean --impact-on auto \
     --confidence $c --tolerance 0.0116 --seed $((100+o)) --order-seed $o \
     --oracle runs/oracle_extended.csv --allow-solver --jobs 2 > runs/stage3_${tag}.log 2>&1; }
export -f run_one; export PY
lo=${ORD%-*}; hi=${ORD#*-}
echo "[$(date +%T)] calib auto: ordenes $ORD, umbrales $LEV, -P $PAR"
( for o in $(seq $lo $hi); do for c in ${LEV//,/ }; do echo "$c $o"; done; done ) | xargs -P $PAR -n 2 bash -c 'run_one "$0" "$1"'
echo "[$(date +%T)] agregando"
$PY - <<'PY'
import json, glob, re, math, collections
REF=[4.0,2.0,0.4,0.01]; by=collections.defaultdict(list)
for f in sorted(glob.glob("runs/stage3_calauto_o*_c*_result.json")):
    m=re.search(r"_o(\d+)_c([\d.]+)_", f); d=json.load(open(f)); by[m[2]].append((int(m[1]),d))
def pb(k, ps):
    dist=[1.0]
    for p in ps:
        nd=[0.0]*(len(dist)+1)
        for i,v in enumerate(dist): nd[i]+=v*(1-p); nd[i+1]+=v*p
        dist=nd
    return sum(dist[:k+1])
out={}
print(f"  {'conf':>5} {'n':>3} {'ok':>3} {'esperados':>10} {'P(X<=ok)':>9} {'runs':>6} {'reales':>7}  incorrectas")
for c in sorted(by):
    rows=[d for _,d in by[c]]; ok=sum(d["output"]==REF for d in rows); ps=[d["likelihood"] for d in rows]
    bad=", ".join(f"o{o}" for o,d in by[c] if d["output"]!=REF)
    print(f"  {c:>5} {len(rows):>3} {ok:>3} {sum(ps):>10.2f} {pb(ok,ps):>9.3f} {sum(d['runs'] for d in rows)/len(rows):>6.0f} {sum(d.get('misses',0) for d in rows)/len(rows):>7.0f}  {bad}")
    out[c]=dict(n=len(rows),correct=ok,expected=sum(ps),p_value=pb(ok,ps),
                rows=[dict(order=o,runs=d["runs"],L=d["likelihood"],correct=d["output"]==REF,output=d["output"],misses=d.get("misses",0)) for o,d in by[c]])
json.dump(dict(ref=REF,design="default 0.3 (impact_on=auto, paired K3, prior mean); orden aleatorio por replica",levels=out),
          open("runs/calibration_auto.json","w"), indent=1)
PY
./runs/build_oracle.py | head -1; echo "[$(date +%T)] CALIB AUTO COMPLETA"
