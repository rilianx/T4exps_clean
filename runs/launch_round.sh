#!/usr/bin/env bash
# Ronda: top-up 4 -> A/B (5 semillas) || calibracion con ordenes aleatorios (16) -> agregaciones.
# Lanzar con: setsid nohup bash runs/launch_round.sh > runs/round.log 2>&1 &
set -u; cd "$(dirname "$0")/.."; PY=.venv/bin/python
echo "[$(date +%T)] A. top-up 4 (a=8.0;b=1.0;g=0.4 -> 165, 16 jobs) + oraculo"
$PY -u runs/topup_oracle.py 165 16 "a=8.0;b=1.0;g=0.4;p=0.01" 2>&1 | grep -v "^  \.\."
./runs/build_oracle.py | head -1

( echo "[$(date +%T)] B. A/B: re-jugando semillas sin resultado"
  for sd in $(seq 30 39); do [ -f runs/stage3_ab_out_s${sd}_result.json ] || { rm -f runs/stage3_ab_out_s${sd}_evals.csv; echo $sd; }; done > runs/.ab_todo
  run_ab() { sd=$1; $PY runs/replay.py --tag ab_out_s$sd --impute --estimator paired --impact-on output --confidence 0.98 --tolerance 0.0116 --seed $sd --oracle runs/oracle_extended.csv > runs/stage3_ab_out_s$sd.log 2>&1; }
  export -f run_ab; export PY; xargs -P 5 -n 1 bash -c 'run_ab "$0"' < runs/.ab_todo
  echo "[$(date +%T)] B. A/B terminado"
) &

( echo "[$(date +%T)] C. calibracion con ordenes aleatorios (solver, 2 jobs c/u, 5 en paralelo)"
  for o in $(seq 1 8); do for c in 0.98 0.8; do [ -f runs/stage3_replay_calr_solver_o${o}_c${c}_result.json ] || { rm -f runs/stage3_replay_calr_solver_o${o}_c${c}_evals.csv; echo "$c $o"; }; done; done > runs/.calr_todo
  run_cal() { c=$1; o=$2; $PY runs/replay.py --tag replay_calr_solver_o${o}_c${c} --impute --estimator paired --confidence $c --tolerance 0.0116 --seed $((100+o)) --order-seed $o --oracle runs/oracle_extended.csv --allow-solver --jobs 2 > runs/stage3_replay_calr_solver_o${o}_c${c}.log 2>&1; }
  export -f run_cal; export PY; xargs -P 5 -n 2 bash -c 'run_cal "$0" "$1"' < runs/.calr_todo
  echo "[$(date +%T)] C. calibracion terminada"
) &
wait
echo "[$(date +%T)] D. agregaciones"
$PY runs/aggregate_round.py
./runs/build_oracle.py | head -1; ./runs/build_variability.py | tail -2; $PY runs/viz_build.py | head -1
echo "[$(date +%T)] RONDA COMPLETA"
