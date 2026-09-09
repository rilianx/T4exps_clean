#!/usr/bin/env bash
# Paso 2: mas replicas de calibracion con la configuracion ganadora.
#   launch_calib_v3.sh <cfg: t|best|tbest|k3> [ordenes: 9-16] [umbrales: 0.98,0.8]
set -u; cd "$(dirname "$0")/.."; PY=.venv/bin/python
CFG=${1:?cfg}; ORD=${2:-9-16}; LEV=${3:-0.98,0.8}
cfg_args() { case $1 in t) echo "--estimator paired_t --impute-prior mean";; best) echo "--estimator paired --impute-prior best";; tbest) echo "--estimator paired_t --impute-prior best";; k3) echo "--estimator paired --impute-prior mean";; esac; }
export -f cfg_args; export PY
run_one() { cfg=$1; c=$2; o=$3; tag=replay_calv3_${cfg}_solver_o${o}_c${c}
  [ -f runs/stage3_${tag}_result.json ] && return 0
  $PY runs/replay.py --tag $tag --impute $(cfg_args $cfg) --impact-on auto --confidence $c --tolerance 0.0116 --seed $((100+o)) --order-seed $o --oracle runs/oracle_extended.csv --allow-solver --jobs 2 > runs/stage3_${tag}.log 2>&1; }
export -f run_one
lo=${ORD%-*}; hi=${ORD#*-}
echo "[$(date +%T)] calib v3: cfg=$CFG ordenes $ORD umbrales $LEV"
( for o in $(seq $lo $hi); do for c in ${LEV//,/ }; do echo "$CFG $c $o"; done; done ) | xargs -P 6 -n 3 bash -c 'run_one "$0" "$1" "$2"'
echo "[$(date +%T)] agregando"; ./runs/aggregate_calib.py replay_calv3_${CFG}_solver --orders $ORD --levels $LEV --json runs/calibration_v3_${CFG}.json
./runs/build_oracle.py | head -1; echo "[$(date +%T)] CALIB V3 COMPLETA"
