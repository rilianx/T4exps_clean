#!/usr/bin/env bash
# Calibracion v2: tres configuraciones candidatas para D5, mismos 8 ordenes x {0.8, 0.98}.
set -u; cd "$(dirname "$0")/.."; PY=.venv/bin/python
cfg_args() { case $1 in t) echo "--estimator paired_t --impute-prior mean";; best) echo "--estimator paired --impute-prior best";; tbest) echo "--estimator paired_t --impute-prior best";; esac; }
export -f cfg_args; export PY
run_one() { cfg=$1; c=$2; o=$3; tag=replay_calv2_${cfg}_solver_o${o}_c${c}
  [ -f runs/stage3_${tag}_result.json ] && return 0
  $PY runs/replay.py --tag $tag --impute $(cfg_args $cfg) --impact-on auto --confidence $c --tolerance 0.0116 --seed $((100+o)) --order-seed $o --oracle runs/oracle_extended.csv --allow-solver --jobs 2 > runs/stage3_${tag}.log 2>&1; }
export -f run_one
echo "[$(date +%T)] calib v2: 3 cfg x 8 ordenes x 2 umbrales"
( for cfg in t best tbest; do for o in $(seq 1 8); do echo "$cfg 0.98 $o"; echo "$cfg 0.8 $o"; done; done ) | xargs -P 6 -n 3 bash -c 'run_one "$0" "$1" "$2"'
echo "[$(date +%T)] agregando"
for cfg in t best tbest; do ./runs/aggregate_calib.py replay_calv2_${cfg}_solver --json runs/calibration_v2_${cfg}.json; done
echo "--- referencia (0.3 default: paired k3, prior mean, impact prefix) ---"; ./runs/aggregate_calib.py replay_calr_solver
./runs/build_oracle.py | head -1
echo "[$(date +%T)] CALIB V2 COMPLETA"
