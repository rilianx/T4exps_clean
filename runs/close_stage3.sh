#!/usr/bin/env bash
# Cierre mecanico de la Etapa 3, para correr cuando el run exacto haya terminado.
# Idempotente: se puede repetir.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo "== 1. estado del run =="; ./runs/status.py | sed -n '2,7p'
# pgrep -f se auto-matchea con shells que llevan el comando en su linea; usar
# el chequeo de status.py (busca un proceso python con --tag exact)
if ./runs/status.py | grep -qE "CORRIENDO|ESTANCADO"; then
  echo "!! el run sigue vivo; no se purga ni se cierra"; exit 1; fi
echo; echo "== 2. purga de filas ajenas (idempotente) =="; ./runs/purge_foreign_rows.py exact --apply || true
echo; echo "== 3. referencia definitiva y comparaciones clave =="
$PY - <<'P'
import csv, collections, math, sys; sys.path.insert(0,"runs")
from status import compute, ORDER, key_of
c=compute("exact"); w=c["winner"]; print("ganador (medias completas + tolerance 0.0116):", key_of(w))
vals=collections.defaultdict(dict)
for r in csv.DictReader(open("runs/stage3_exact_evals.csv")):
    if r["status"]=="ok": vals[r["theta"]][r["instance"]]=float(r["value"])
def cmp(A,B):
    com=vals[A].keys()&vals[B].keys(); d=[vals[B][i]-vals[A][i] for i in com]; n=len(d); m=sum(d)/n
    se=math.sqrt(sum((x-m)**2 for x in d)/(n-1))/math.sqrt(n); return m,se,n
for A,B,lab in (("a=4.0;b=1.0;g=0.1;p=0.01","a=4.0;b=2.0;g=0.1;p=0.01","beta 2.0 - 1.0"),
                ("a=4.0;b=2.0;g=0.3;p=0.01","a=4.0;b=2.0;g=0.4;p=0.01","gamma 0.4 - 0.3"),
                ("a=2.0;b=0.5;g=0.1;p=0.01","a=4.0;b=0.5;g=0.1;p=0.01","alpha 4.0 - 2.0")):
    m,se,n=cmp(A,B); print(f"  {lab:<16} {m:+.4f} pp  ({m/se:+.1f} SE, n={n})")
print(f"estrategias completas: {c['path_done']}/{len(c['sweep'])}   filas ajenas restantes: {c['foreign']}")
P
echo; echo "== 4. ablacion con la referencia definitiva =="; ./runs/build_ablation.py | tail -9
echo; echo "== 5. dashboard =="; $PY runs/viz_build.py | head -1
echo; echo "== 6. resultado del run =="; tail -8 runs/stage3_exact.log
echo; echo "listo: revisar PLAN.md (seccion 'Resultados de la Etapa 3') y republicar runs/dashboard.html"
