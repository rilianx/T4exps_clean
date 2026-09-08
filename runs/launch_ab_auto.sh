#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."; PY=.venv/bin/python
echo "[$(date +%T)] A/B auto: semillas 30-39 (estricto)"
run_ab() { sd=$1; $PY runs/replay.py --tag ab_auto_s$sd --impute --estimator paired --impact-on auto --confidence 0.98 --tolerance 0.0116 --seed $sd --oracle runs/oracle_extended.csv > runs/stage3_ab_auto_s$sd.log 2>&1; }
export -f run_ab; export PY; seq 30 39 | xargs -P 6 -n 1 bash -c 'run_ab "$0"'
echo "[$(date +%T)] misses: $(grep -l 'cache MISS' runs/stage3_ab_auto_s*.log 2>/dev/null | wc -l)"; grep -h "cache MISS: a=" runs/stage3_ab_auto_s*.log 2>/dev/null | sed -E 's/.*MISS: ([^ ]+ [^ ]+ [^ ]+ [^ ]+) en .*/\1/' | sort | uniq -c
echo "[$(date +%T)] AB AUTO COMPLETO"
