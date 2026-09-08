#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."; PY=.venv/bin/python
for sd in $(seq 30 39); do [ -f runs/stage3_ab_auto_s${sd}_result.json ] || { rm -f runs/stage3_ab_auto_s${sd}_evals.csv; echo $sd; }; done > runs/.ab_auto_todo
echo "[$(date +%T)] auto con solver: $(tr '\n' ' ' < runs/.ab_auto_todo)"
run_ab() { sd=$1; $PY runs/replay.py --tag ab_auto_s$sd --impute --estimator paired --impact-on auto --confidence 0.98 --tolerance 0.0116 --seed $sd --oracle runs/oracle_extended.csv --allow-solver --jobs 2 > runs/stage3_ab_auto_s$sd.log 2>&1; }
export -f run_ab; export PY; xargs -P 3 -n 1 bash -c 'run_ab "$0"' < runs/.ab_auto_todo
$PY runs/aggregate_round.py 2>&1 | sed -n '/--- auto/,$p'; ./runs/build_oracle.py | head -1
echo "[$(date +%T)] AB AUTO SOLVER COMPLETO"
