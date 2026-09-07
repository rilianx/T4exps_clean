#!/usr/bin/env bash
# Estado de la ronda (A/B + calibracion con ordenes aleatorios). Sin dependencias.
cd "$(dirname "$0")/.."
alive() { pgrep -af "$1" | grep -c '^[0-9]* \S*python'; }
echo "ronda viva: $(pgrep -af 'launch_round.sh' | grep -vc grep) launcher | replays $(alive 'runs/replay.py') | topup $(alive 'topup_oracle') | BSG_CLP $(pgrep -c BSG_CLP)"
echo "A/B  : $(ls runs/stage3_ab_out_s*_result.json 2>/dev/null | wc -l)/10 resultados"
echo "calib: $(ls runs/stage3_replay_calr_solver_*_result.json 2>/dev/null | wc -l)/16 resultados"
for f in runs/stage3_replay_calr_solver_o*_c*.log; do [ -f "$f" ] || continue; r=${f%.log}_result.json; t=$(basename $f .log | sed 's/stage3_replay_calr_solver_//'); if [ -f "$r" ]; then echo "  $t: OK"; else echo "  $t: $(grep -c 'runs\] L=' $f) it, $(grep -c ',ok,' ${f%.log}_evals.csv 2>/dev/null) reales$(grep -q Traceback $f && echo '  !! Traceback')"; fi; done 2>/dev/null
tail -3 runs/round.log 2>/dev/null
