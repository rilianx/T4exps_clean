"""Instrumentacion para los runs de BSG.

`LoggingRunner` es un `Runner` que ademas escribe una fila por ejecucion a un
CSV, a medida que ocurren.  Sirve para dos cosas:

  1. Monitorear en vivo (`tail -f`, o `progress.py`): valores, tiempos de
     pared, familia, errores.
  2. Sobrevivir un aborto.  `Runner._evaluate` no atrapa `TimeoutExpired` y
     con n_jobs>1 el pool la re-lanza, tirando el experimento entero y
     perdiendo el cache (PLAN.md, B2).  Con el CSV en disco las evaluaciones
     ya pagadas quedan, y el analisis se puede rehacer sin re-ejecutar.

Se inyecta en un `Engine` ya construido, reutilizando el orden de instancias
que el Engine ya estratifico:

    eng = Engine(exp, instances, ...)
    eng.runner = LoggingRunner.adopt(eng.runner, "runs/foo_evals.csv")
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import threading
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from t4exps.core import Strategy
from t4exps.runner import Runner, RunError
from t4exps.utils import default_family

FIELDS = ["t_wall", "strategy", "theta", "instance", "family", "value",
          "seconds", "status", "detail"]


class LoggingRunner(Runner):
    def __init__(self, *a, csv_path: str = "evals.csv", echo_every: int = 25, **kw):
        super().__init__(*a, **kw)
        self.csv_path = csv_path
        self.echo_every = echo_every
        self._lock = threading.Lock()
        self._t0 = time.time()
        self.errors: List[str] = []
        self._resumed: dict = {}
        self._by_key: dict = {}
        self._seeded: set = set()
        self._resumed_sec = 0.0
        new = not os.path.exists(csv_path)
        self._fh = open(csv_path, "a", newline="", buffering=1)
        self._w = csv.writer(self._fh)
        if new:
            self._w.writerow(FIELDS)

    @classmethod
    def adopt(cls, runner: Runner, csv_path: str, echo_every: int = 25,
              resume: bool = False):
        """Crea un LoggingRunner con la config (y el orden) de `runner`."""
        out = cls(runner.instances, n_jobs=runner.n_jobs,
                  timeout=runner.timeout, parse=runner.parse,
                  csv_path=csv_path, echo_every=echo_every)
        if resume:
            out.resume_from_csv()
        return out

    def resume_from_csv(self) -> int:
        """Indexa lo que ya esta en el CSV.  Devuelve cuantas evaluaciones hay.

        El CSV se escribe en orden de FINALIZACION, que con n_jobs>1 no es el
        orden de instancia.  `evals[k]` tiene que ser el prefijo
        instances[0:c] EN ORDEN -- de eso depende el pareado -- asi que se
        reconstruye por clave (estrategia, instancia) y se corta en el primer
        hueco.  Una fila con status != ok no cuenta como evaluada.
        """
        self._resumed = {}
        self._by_key = {}
        if not os.path.exists(self.csv_path):
            return 0
        with open(self.csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") != "ok" or not row.get("value"):
                    continue
                self._by_key.setdefault((row["strategy"], row["theta"]), {})[
                    row["instance"]] = float(row["value"])
        secs = []
        with open(self.csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") == "ok" and row.get("seconds"):
                    secs.append(float(row["seconds"]))
        self._resumed_sec = sum(secs) / len(secs) if secs else 0.0
        total = 0
        for k, vals in self._by_key.items():
            n = 0
            for pi in self.instances:
                if pi not in vals:
                    break
                n += 1
            if n:
                self._resumed[k] = n
                total += n
        return total

    def _maybe_seed(self, s: Strategy) -> None:
        """Rellena `evals[s.key]` desde el CSV la primera vez que se pregunta.

        Es perezoso porque el motor descubre las estrategias sobre la marcha:
        no se pueden enumerar antes de correr el experimento.  `StrategyKey`
        incluye el comando, que el CSV no guarda, asi que la busqueda va por
        (nombre, theta) y la clave se arma desde la Strategy real.
        """
        if not self._resumed or s.key in self._seeded:
            return
        self._seeded.add(s.key)
        theta = ";".join(f"{k}={v}" for k, v in sorted(s.theta.items()))
        key = (s.name, theta)
        n = self._resumed.get(key)
        if not n or s.key in self.evals:
            return
        vals = self._by_key[key]
        self.evals[s.key] = [vals[pi] for pi in self.instances[:n]]
        self.total_runs += n
        self.seconds[s.key] = self.seconds.get(s.key, 0.0) + n * self._resumed_sec
        print(f"  ~~ reanudado: {n} evaluaciones de {s.params}", flush=True)

    # -- todos los accesos del Engine pasan por aca ------------------------ #

    def count(self, s: Strategy) -> int:
        self._maybe_seed(s)
        return super().count(s)

    def values(self, s: Strategy):
        self._maybe_seed(s)
        return super().values(s)

    def is_complete(self, s: Strategy) -> bool:
        self._maybe_seed(s)
        return super().is_complete(s)

    def run(self, s: Strategy, k: int) -> int:
        self._maybe_seed(s)
        return super().run(s, k)

    def _evaluate(self, s: Strategy, instance: str) -> float:
        t0 = time.time()
        status, detail, value = "ok", "", ""
        try:
            value = super()._evaluate(s, instance)
            return value
        except subprocess.TimeoutExpired as e:
            status, detail = "TIMEOUT", f"timeout={self.timeout}s"
            raise
        except RunError as e:
            status, detail = "RUNERROR", str(e)[:200]
            raise
        except Exception as e:                                  # pragma: no cover
            status, detail = type(e).__name__, str(e)[:200]
            raise
        finally:
            dt = time.time() - t0
            with self._lock:
                self._w.writerow([
                    f"{time.time() - self._t0:.1f}", s.name,
                    ";".join(f"{k}={v}" for k, v in sorted(s.theta.items())),
                    instance, default_family(instance),
                    f"{value:.6f}" if value != "" else "",
                    f"{dt:.2f}", status, detail])
                if status != "ok":
                    self.errors.append(f"{status} {instance}: {detail}")
                    print(f"  !! {status} en {instance}: {detail}", flush=True)
                n = self.total_runs + 1
            if self.echo_every and n % self.echo_every == 0:
                print(f"  .. {n} ejecuciones, {time.time()-self._t0:.0f}s de reloj",
                      flush=True)

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass
