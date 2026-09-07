"""Cobertura del camino subprocess de `Runner`, que los tests existentes no
ejercitan (PLAN.md, "Deuda de tests").

Todo usa un solver falso: un script de shell que imprime un numero.  El bug de
los flags `-a/-b/-g` en `examples/bsg.py` es exactamente la clase de cosa que
estos tests atrapan.
"""
import os
import stat
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from t4exps import Engine, Strategy, best_strategy, sequential_execution
from t4exps.runner import Runner, RunError, _parse_last_float
from t4exps.utils import default_family


def _script(tmp_path, name, body):
    p = tmp_path / name
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(p)


@pytest.fixture
def solver(tmp_path):
    """Imita a BSG: acepta flags largos, imprime una cabecera y el valor ultimo.

    El valor depende de la instancia y de --alpha, para que las comparaciones
    entre estrategias signifiquen algo.
    """
    return _script(tmp_path, "solver.sh", """
alpha=0
inst=""
for arg in "$@"; do
  case "$arg" in
    --alpha=*) alpha=${arg#--alpha=} ;;
    -*|[0-9]*) ;;
    *) [ -z "$inst" ] && inst="$arg" ;;
  esac
done
n=$(basename "$inst" | tr -dc '0-9')
echo "% volume utilization"
awk -v n="${n:-0}" -v a="$alpha" 'BEGIN{printf "%.6f\\n", 90 + (n % 7) + a}'
""")


# --- template de comando ------------------------------------------------- #

def test_template_substitutes_instance_and_theta(solver):
    s = Strategy("fake", solver + " {INSTANCE} --alpha={a}", theta={"a": 2.0})
    r = Runner(["/bench/BR3_5.txt"])
    assert r._evaluate(s, "/bench/BR3_5.txt") == pytest.approx(90 + (35 % 7) + 2.0)


def test_instance_is_appended_when_template_has_no_placeholder(solver):
    s = Strategy("fake", solver + " --alpha={a}", theta={"a": 1.0})
    r = Runner(["/bench/BR3_5.txt"])
    assert r._evaluate(s, "/bench/BR3_5.txt") == pytest.approx(90 + (35 % 7) + 1.0)


def test_multi_token_instance_survives_shlex_split(solver):
    """Las lineas de instancesCLP-shuf.txt traen sus propios argumentos."""
    inst = "/bench/BR8.txt -i 53 --min_fr=0.98 -t 30"
    s = Strategy("fake", solver + " {INSTANCE} --alpha={a}", theta={"a": 0.0})
    r = Runner([inst])
    assert r._evaluate(s, inst) == pytest.approx(90 + (8 % 7))
    assert default_family(inst) == "BR8"


def test_last_flag_wins_is_what_the_template_relies_on(tmp_path):
    """`-t` al final del template tiene que pisar el `-t` que trae la instancia."""
    sh = _script(tmp_path, "lastwins.sh", """
t=0
for arg in "$@"; do case "$arg" in -t) next=1 ;; *) [ "$next" = 1 ] && t=$arg && next=0 ;; esac; done
echo "$t"
""")
    s = Strategy("f", sh + " {INSTANCE} -t 5")
    assert Runner(["x -t 30"])._evaluate(s, "x -t 30") == 5.0


# --- errores -------------------------------------------------------------- #

def test_nonzero_exit_raises_runerror(tmp_path):
    sh = _script(tmp_path, "boom.sh", "echo 'usage: ...' >&2\nexit 1\n")
    with pytest.raises(RunError, match="exited 1"):
        Runner(["i0"])._evaluate(Strategy("f", sh + " {INSTANCE}"), "i0")


def test_stdout_without_a_number_raises_runerror(tmp_path):
    sh = _script(tmp_path, "mute.sh", "echo 'no numbers here'\n")
    with pytest.raises(RunError, match="no numeric output"):
        Runner(["i0"])._evaluate(Strategy("f", sh + " {INSTANCE}"), "i0")


def test_parse_takes_the_last_float_not_the_first():
    assert _parse_last_float("best 0.94 nodes 26\n% util\n94.656563\n") == 94.656563


def test_a_trailing_debug_line_silently_breaks_the_measurement():
    """Documenta la fragilidad senalada en el PLAN: un cout de debug al final
    cambia lo que se mide, sin error."""
    assert _parse_last_float("% util\n94.65\n[debug] elapsed 3\n") == 3.0


def test_custom_parse_is_reachable_from_the_public_api(tmp_path):
    """B1 del PLAN: `parse` tiene que llegar desde Engine/sequential_execution."""
    sh = _script(tmp_path, "tagged.sh", "echo 'UTIL=93.5'\necho '[debug] done'\n")
    parse = lambda out: float(
        [ln for ln in out.splitlines() if ln.startswith("UTIL=")][0][5:])
    s = Strategy("f", sh + " {INSTANCE}")
    eng = Engine(lambda: best_strategy(s, s).params, ["i0"], parse=parse, nruns=1)
    assert eng.runner._evaluate(s, "i0") == 93.5
    out, runs = sequential_execution(lambda: best_strategy(s, s).params, ["i0"],
                                     parse=parse)
    assert runs == 1


def test_timeout_propagates_and_aborts_the_run(tmp_path):
    """B2 del PLAN: nadie atrapa TimeoutExpired, asi que tira el experimento.

    Comportamiento definido a proposito -- el limite de tiempo se le pasa al
    solver, y el `timeout` del harness es solo red de seguridad.
    """
    sh = _script(tmp_path, "slow.sh", "sleep 5\necho 1.0\n")
    with pytest.raises(subprocess.TimeoutExpired):
        Runner(["i0"], timeout=0.3)._evaluate(Strategy("f", sh + " {INSTANCE}"), "i0")


# --- paralelismo ---------------------------------------------------------- #

def test_n_jobs_8_gives_the_same_evaluations_as_n_jobs_1(solver):
    inst = [f"/bench/BR{i}_{i}.txt" for i in range(24)]
    s = Strategy("fake", solver + " {INSTANCE} --alpha={a}", theta={"a": 3.0})
    r1, r8 = Runner(inst, n_jobs=1), Runner(inst, n_jobs=8)
    r1.run(s, len(inst)); r8.run(s, len(inst))
    assert r1.values(s) == r8.values(s)
    assert r8.total_runs == len(inst)


def test_parallel_preserves_instance_order(solver):
    """El pareado depende de que evals[k][j] corresponda a instances[j]."""
    inst = [f"/bench/BR{i}_{i}.txt" for i in range(16)]
    s = Strategy("fake", solver + " {INSTANCE} --alpha={a}", theta={"a": 0.0})
    r = Runner(inst, n_jobs=8); r.run(s, len(inst))
    assert r.values(s) == [90.0 + (int(f"{i}{i}") % 7) for i in range(16)]


# --- confidence ------------------------------------------------------------ #

def _sweep(solver, alphas):
    def exp():
        S = Strategy("fake", solver + " {INSTANCE} --alpha={a}", theta={"a": alphas[0]})
        for a in alphas[1:]:
            S = best_strategy(S, S.with_theta(a=a))
        return S.theta["a"]
    return exp


def test_confidence_below_one_stops_earlier(solver):
    inst = [f"/bench/BR{i}_{i % 13}.txt" for i in range(60)]
    alphas = [0.0, 1.0, 2.0]
    kw = dict(n_jobs=4, nruns=5, batch=5, seed=0)
    exact = Engine(_sweep(solver, alphas), inst, confidence=1.0, **kw).run()
    early = Engine(_sweep(solver, alphas), inst, confidence=0.8, **kw).run()
    assert early.likelihood >= 0.8
    assert early.runs <= exact.runs
    assert exact.likelihood == 1.0
    assert early.output == exact.output          # el solver falso es limpio
