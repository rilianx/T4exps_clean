#!/usr/bin/env bash
# D5, experimento focalizado: prior de imputacion 'best' vs 'mean' (control).
# Ambos brazos con paired k3 e impact_on=auto, a confidence 0.8 -- donde la
# calibracion con ordenes aleatorios dio 0.50 correctas contra 0.824 reportada.
# Unica diferencia entre brazos: el centro del prior de imputacion.
set -u; cd "$(dirname "$0")/.."; PY=.venv/bin/python
run_one() { prior=$1; o=$2; tag=replay_d5_${prior}_o${o}_c0.8
  [ -f runs/stage3_${tag}_result.json ] && return 0
  $PY runs/replay.py --tag $tag --impute --estimator paired --impute-prior $prior \
     --impact-on auto --confidence 0.8 --tolerance 0.0116 --seed $((100+o)) --order-seed $o \
     --oracle runs/oracle_extended.csv --allow-solver --jobs 2 > runs/stage3_${tag}.log 2>&1; }
export -f run_one; export PY
echo "[$(date +%T)] D5 prior: best vs mean (control), 8 ordenes, conf 0.8, -P 8 x 2 jobs"
( for o in $(seq 1 8); do echo "best $o"; echo "mean $o"; done ) | xargs -P 8 -n 2 bash -c 'run_one "$0" "$1"'
echo "[$(date +%T)] agregando"
for pr in best mean; do ./runs/aggregate_calib.py replay_d5_${pr} --orders 1-8 --levels 0.8 --json runs/calibration_d5_${pr}.json; done
./runs/build_oracle.py | head -1; echo "[$(date +%T)] D5 PRIOR COMPLETO"
