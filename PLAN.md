# Plan de experimentos BSG

Tres etapas, en orden, con una compuerta go/no-go entre cada una. Ninguna etapa
empieza hasta que la anterior cierra su checklist: cada una existe para
descubrir un tipo distinto de problema, y descubrirlos mezclados cuesta días.

| etapa | costo estimado | qué decide |
|---|---|---|
| 1. Plomería | ~1 hora (de las cuales ~2 min de CPU) | si el pipeline mide lo que creemos que mide |
| 2. Estadística | ~70 min de CPU + análisis | si el estimador pareado es la historia del paper o una nota al pie |
| 3. Experimento completo | ~26 h de reloj a 8 jobs | los números que van al paper |

Máquina: Xeon E5-2620 v4, 16 hilos. `n_jobs=8` deja margen; subir a 16 sobre un
solver compute-bound con hyperthreading rinde poco.

---

## Hechos verificados del solver

Todo lo de abajo está medido, no supuesto. Binario usado:
`/home/iaraya/T4exps/bsg_algo/BSG_CLP` (marzo 2022).

### Interfaz de línea de comando

Los flags reales, leídos del parseo de argumentos y confirmados corriendo el
binario:

| parámetro | flag | notas |
|---|---|---|
| alpha | `--alpha` | double |
| beta | `--beta` | double |
| gamma | `--gamma` | double |
| p | `-p` | double (short, sin versión larga) |
| límite de tiempo | `-t` / `--timelimit` | **int**, segundos |
| índice de instancia | `-i` | int, elige una instancia dentro del archivo |
| min_fr | `--min_fr` | double, default 0.98 |
| semilla | `--seed` | int, **default 1** |
| formato | `-f` | `BR`, `BRw`, `1C`; default `BR` |

**`-a`, `-b` y `-g` no existen.** El template de `examples/bsg.py:31` es
`"{INSTANCE} -a {a} -b {b} -g {g} -p {p}"`, que produce un error de parseo,
sale con código 1 e imprime el usage por stderr. Verificado:

```
$ ./BSG_CLP bsg_algo/BR8.txt -i 53 -a 4.0 -b 1.0 -g 0.2 -p 0.04 -t 2
  (usage por stderr)                                          EXIT=1
```

Con T4exps eso significa `RunError` en **cada** evaluación (`runner.py:92-93`).
El template correcto es el que ya usaba el T4exps viejo
(`/home/iaraya/T4exps/experiment.py:19`):

```
--alpha={a} --beta={b} --gamma={g} -p {p} -t {T}
```

### Las instancias traen sus propios argumentos

Las líneas de `instancesCLP-shuf.txt` (1500 instancias) no son rutas, son
fragmentos de línea de comando:

```
bsg_algo/BR8.txt -i 53 --min_fr=0.98 -t 30
bsg_algo/BR12.txt -i 57 --min_fr=0.98 -t 30
```

Esto funciona con T4exps porque `_evaluate` hace `shlex.split` sobre el comando
ya formateado (`runner.py:87`), así que la "instancia" se expande a varios
tokens. Y la estratificación sigue andando, verificado sobre el archivo real:
`default_family` toma el basename y le aplica `^([A-Za-z]+\d*)`, así que
`bsg_algo/BR8.txt -i 53 --min_fr=0.98 -t 30` → `BR8`. El archivo tiene 15
familias (BR1..BR15) de 100 instancias cada una, y los prefijos estratificados
quedan exactamente balanceados:

| prefijo | familias | instancias por familia |
|---|---|---|
| 30 | 15 | 2 y 2 |
| 200 | 15 | entre 13 y 14 |

Es decir que `instances[0:c]` es un retrato en miniatura del benchmark completo
desde c=30, que es justo lo que el estimador pareado necesita mientras se
ajusta.

El listado está completo y consistente, verificado: los 15 archivos
referenciados existen, cada familia aporta exactamente 100 instancias con `-i`
de 0 a 99, y eso coincide con el `100` que declara la cabecera de cada
`BR*.txt`. (`bsg_algo/BR0.txt` existe pero el listado no lo usa.)

**Las rutas del listado son relativas**, así que `Runner` —que no pasa `cwd=` a
`subprocess.run` (`runner.py:86-91`)— hereda el directorio de trabajo de
Python. El run tiene que largarse desde `/home/iaraya/T4exps`, mientras que el
paquete vive en `/home/iaraya/sepeculative_experiments/T4exps_clean`. Dos
salidas, en orden de preferencia:

1. **Absolutizar el listado una vez** y olvidarse del cwd:
   ```bash
   sed 's|^bsg_algo/|/home/iaraya/T4exps/bsg_algo/|' \
     /home/iaraya/T4exps/bsg_algo/instancesCLP-shuf.txt > instances_abs.txt
   ```
   `default_family` sigue funcionando (toma el basename), así que la
   estratificación no cambia. Recomendado: un run de 26 horas no debería
   depender de desde dónde se lo lanzó.
2. Lanzar desde `/home/iaraya/T4exps` con
   `PYTHONPATH=/home/iaraya/sepeculative_experiments/T4exps_clean`.

Consecuencia importante: el archivo ya trae `-t 30`. Medido, **el último `-t`
gana**:

```
$ ./BSG_CLP ... -t 30 --alpha=4.0 ... -t 5     ->  real 0m5.622s
```

Así que poner `-t {T}` **al final** del template pisa el límite del archivo.
Es el mecanismo limpio para cambiar el presupuesto entre etapas sin tocar el
archivo de instancias.

### El límite de tiempo es vinculante, y se pasa de largo

El solver usa todo el presupuesto: `-t 30` dio `real 0m30.673s`. La búsqueda
(DoubleEffort sobre BSG) sigue duplicando esfuerzo hasta que se acaba el
tiempo, no termina sola.

Y chequea el reloj solo entre iteraciones, así que **se excede, y el exceso
crece con el tamaño de la instancia**. Medido con `-t 5`:

| familia | wall | exceso |
|---|---|---|
| BR1 | 5.43 s | +9% |
| BR8 | 5.52 s | +10% |
| BR15 | 8.51 s | **+70%** |

Esto decide dos parámetros:

- **El `timeout` de T4exps necesita mucha holgura.** Con `-t 30` y `timeout=30`
  (el default de `examples/bsg.py:50`) el proceso se mata **siempre**: ya el
  caso fácil tardó 30.67 s. Medir el peor exceso sobre BR15 en la Etapa 1 y
  poner el timeout a partir de eso, no del límite nominal.
- **`cost_aware=True` rinde poco acá.** El README lo recomienda para límites de
  tiempo "donde las instancias grandes cuestan mucho más", pero con un límite
  vinculante todas cuestan aproximadamente lo mismo: el rango medido es 1.6x
  (5.43 s a 8.51 s), no órdenes de magnitud. Dejarlo en `False` salvo que la
  Etapa 2 muestre una dispersión de costo mayor con los 30 s reales.

### El parser default funciona

`_parse_last_float` toma el último float del stdout (`runner.py:97-103`). El
stdout del solver termina así:

```
% volume utilization
94.656563
```

El último float **es** la utilización. Así que no hace falta un parser custom
—lo cual es una suerte, porque no se puede pasar uno (ver B1). Dos cuidados:

- **La escala es porcentaje (0-100), no fracción.** La `tolerance` va en puntos
  porcentuales. Es consistente con la desviación de 0.95 de NOTES.md, que
  entonces son 0.95 puntos porcentuales.
- Es frágil: depende de que ese número quede último. Cualquier línea de debug
  agregada después rompe silenciosamente la medición. En la Etapa 1 mirar el
  stdout completo con los ojos, y no volver a asumirlo si se recompila.

### No escribe archivos por instancia

Las únicas escrituras a disco de `main_clp.cpp` están dentro de `if(_plot)`
(`pointsToTxt`, y un `system("firefox ...")`). Sin `--plot`, no hay archivos
temporales, así que **la preocupación por los 8 procesos pisándose no aplica**.
`Runner` corre los subprocesos compartiendo el cwd y sin `cwd=`
(`runner.py:71-72, 86-91`), lo cual sería un problema si el solver escribiera;
no lo hace. Igual conviene el chequeo `n_jobs=1` vs `n_jobs=8` de la Etapa 1,
que cuesta nada y cubre el caso general.

### Es determinista por default

`--seed` tiene default 1 y se aplica con `srand(seed)`. Sin pasar `--seed`,
cada `(instancia, theta)` da siempre el mismo número, que es exactamente lo que
pide la regla 2 del README: `E_s(pi)` bien definido. **No pasar `--seed`
variable**, o "idéntico a secuencial" deja de ser testeable.

### Advertencia: el fuente y el binario no coinciden

`/home/iaraya/Metasolver/problems/clp/main/main_clp.cpp` es una versión
**distinta** de la que está compilada en `bsg_algo/BSG_CLP`. En ese fuente la
línea 201 corre `gr->run(...)` — el greedy — y nunca usa el `DoubleEffort` que
construye tres líneas antes; además imprime un encabezado distinto
(`best_volume best_volume(weight) hypervolume`). El binario de `bsg_algo`
claramente sí corre DoubleEffort: su traza lo dice y consume todo el tiempo.

**Fijar el binario que se usa y no recompilar desde ese fuente sin revisar la
línea 201.** Si se recompila sin mirar, el experimento mide un greedy y el
límite de tiempo deja de ser vinculante — se caen las tres etapas.

---

## Bloqueos de la librería a resolver

### B1. El parser no es configurable desde la API pública

`Runner.__init__` acepta `parse=`, pero ni `Engine.__init__` ni
`experiment_execution` lo reciben ni lo pasan (`engine.py:80`);
`sequential_execution` tampoco (`utils.py:33`). Pasar `parse=` a
`experiment_execution` da `TypeError`.

**Hoy no bloquea**, porque el default acierta (ver arriba). Pero deja el
proyecto a merced de un `cout` de debug. Threadear `parse` por `Engine` y
`sequential_execution` hasta `Runner` son ~6 líneas; hacerlo antes de la
Etapa 3, junto con el test del camino subprocess que hoy no existe.

**RESUELTO.** `parse` es ahora un parámetro de `Engine.__init__`
(`engine.py:77`) y de `sequential_execution` (`utils.py:14`), y llega al
`Runner`. Cubierto por `tests/test_subprocess_path.py`
(`test_custom_parse_is_reachable_from_the_public_api`).

### B2. El timeout aborta el run, no degrada la instancia

`subprocess.run(..., timeout=self.timeout)` lanza `TimeoutExpired` y nadie la
atrapa (`runner.py:86-91`); con `n_jobs>1`, `pool.map` la re-lanza al iterar.
**Una sola instancia que pase el timeout tira todo el experimento y se pierde
el cache completo de evaluaciones.**

**Resuelto, y la decisión es: el límite se le pasa a BSG por `-t`.** El
`timeout` de T4exps queda como red de seguridad contra cuelgues, nunca como el
mecanismo de corte. Es lo correcto por dos razones:

- El solver que administra su propio presupuesto devuelve una solución válida
  (peor, pero válida) y el experimento sigue. El timeout del harness mata el
  proceso y se lleva horas de CPU.
- La alternativa —atrapar `TimeoutExpired` y devolver un valor de castigo—
  cambia lo que se mide e introduce una distribución bimodal que viola el
  supuesto normal de los dos estimadores. Dejar que el solver se detenga solo
  mantiene todas las evaluaciones en la misma escala.

Regla operativa, dado el exceso medido: `timeout` a partir del peor exceso
observado en BR15, con margen. Punto de partida `timeout = 2*T + 15` (para
`T=30`: 75 s), y ajustar con lo que mida la Etapa 1.

### B3. `runner.evals` no es alcanzable desde el resultado

`Result` no guarda referencia al `Runner` (`engine.py:34-52`) y
`experiment_execution` descarta el `Engine` (`engine.py:283`).
`sequential_execution` también crea su `Runner` local y devuelve solo
`(output, total_runs)` (`utils.py:30-56`).

La Etapa 2 necesita la matriz completa, así que hay que bajar un nivel:

```python
from t4exps import Engine
eng = Engine(exp, instances, ...)
res = eng.run()
evals = eng.runner.evals          # dict[StrategyKey, list[float]]
order = eng.runner.instances      # el orden estratificado ya aplicado
```

`evals[k][j]` corresponde a `order[j]`: el prefijo compartido es lo que hace
posible el pareado (`runner.py:3-6`) y lo que permite reconstruir la matriz.

### B4 (menor). Rutas con espacios

`shlex.split` sobre el comando formateado parte por espacios. Ya se está
explotando eso a propósito (las instancias traen argumentos), así que ninguna
ruta de instancia puede contener espacios. Con `instancesCLP-shuf.txt` no los
hay; verificar si se cambia de archivo.

---

## Etapa 1 — Piloto de plomería (~1 hora)

**2 configuraciones, 30 instancias estratificadas, `-t 5`.**

Acá el resultado no importa. Importa que el aparato mida lo que creemos.
Son 60 ejecuciones por pase, ~45 s de reloj a 8 jobs: la hora es para depurar,
no para esperar.

### Preparar el subconjunto

```bash
cd /home/iaraya/T4exps
head -400 bsg_algo/instancesCLP-shuf.txt > /tmp/smoke_pool.txt
```

Y que T4exps estratifique y recorte: `order_instances(pool, "stratified")[:30]`.
No cortar el archivo a 30 líneas a mano —la estratificación es parte de lo que
hay que probar.

### El script

`examples/bsg_smoke.py`: dos estrategias y nada más.

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from t4exps import Engine, Strategy, best_strategy, sequential_execution, order_instances

BIN  = "/home/iaraya/T4exps/bsg_algo/BSG_CLP"
BASE = {"a": 4.0, "b": 1.0, "g": 0.2, "p": 0.04}
T    = 5

# -t va AL FINAL: pisa el -t 30 que traen las lineas del archivo.
TEMPLATE = "{INSTANCE} --alpha={a} --beta={b} --gamma={g} -p {p} -t " + str(T)

def make_exp():
    def exp():
        S = Strategy("bsg", f"{BIN} {TEMPLATE}", theta=BASE)
        return best_strategy(S, S.with_theta(a=8.0)).params
    return exp

pool = [ln.strip() for ln in open("/tmp/smoke_pool.txt") if ln.strip()]
inst = order_instances(pool, "stratified", seed=0)[:30]
```

Correrlo en tres pasos, en este orden:

1. **Una ejecución a mano**, fuera de T4exps, mirando el stdout crudo completo.
2. **`sequential_execution` con `n_jobs=1`** — 60 ejecuciones, el baseline.
3. **`Engine` con `n_jobs=8`** — la misma cosa, especulada y en paralelo.

### Resultados medidos (ejecutado)

`runs/stage1_smoke.py`, log en `runs/stage1.log`. **26 OK, 1 falla.**

Comando formateado, verificado token a token:

```
BSG_CLP .../BR8.txt -i 93 --min_fr=0.98 -t 30 --alpha=4.0 --beta=1.0 --gamma=0.2 -p 0.04 -t 5
```

El `-t 5` del template queda después del `-t 30` del archivo y gana: wall 5.49 s.
Último float del stdout = `94.953339`, precedido de `% volume utilization`:
escala porcentaje, parser default correcto. Cero `RunError` en 180 ejecuciones.
El listado absolutizado hace que el cwd no importe (corrido desde
`T4exps_clean`, no desde `/home/iaraya/T4exps`).

**Los tres caminos coinciden.** `sequential_execution(n_jobs=1)`,
`Engine(n_jobs=8)` y `Engine(n_jobs=1)` dan los tres
`a=4.0, b=1.0, g=0.2, p=0.04`, con `likelihood == 1.0` y sin instancias
evaluadas dos veces. Wall: 338 s secuencial contra 67 s con 8 jobs.

**Exceso de tiempo.** Crece con el tamaño de la familia, como el plan
anticipaba, pero **el número que importa es el de 30 s, no el de 5 s**:

| `-t` | peor wall observado | exceso |
|---|---|---|
| 5 s | 7.75 s (BR15) | +55% |
| 30 s | 32.77 s (BR10) | **+9%** |

El exceso es sobre todo costo fijo de arranque (lectura de instancia, greedy
inicial), así que a 30 s se diluye. `timeout = 2*T + 15 = 75 s` deja 2.3x de
margen sobre el peor caso medido. Ninguna ejecución murió por timeout.

### La falla: el solver NO es determinista

El plan daba por sentado que `--seed` con default 1 alcanza para que `E_s(pi)`
esté bien definido. **No alcanza.** BSG es anytime con corte por reloj: lo que
devuelve depende de cuánto trabajo entra en el presupuesto, y eso depende de la
frecuencia del CPU. La máquina está en governor `schedutil` con turbo activo
(idle a 1200 MHz, turbo a 3000 MHz), así que la frecuencia depende de cuántos
cores estén ocupados.

Medido, la misma línea de comando exacta:

```
sola          -> 94.656563
8 en paralelo -> 94.144363     (0.51 pp de diferencia)
```

Cuantificado sobre muestras estratificadas (`runs/load_probe.py`):

| presupuesto | cambian j=1 vs j=8 | sesgo medio | peor \|dif\| |
|---|---|---|---|
| `-t 5`, 16 inst. | 2/16 | +0.013 pp | 0.11 pp |
| `-t 30`, 16 inst. | 2/16 | +0.016 pp | 0.21 pp |
| `-t 5`, 60 evals (Etapa 1) | 2/60 | — | 0.15 pp |

**No es un efecto del paralelismo.** Dos barridos idénticos de `n_jobs=8`
también difieren en 2/16: es jitter de frecuencia alrededor de los saltos de
nivel de `DoubleEffort`, que es donde el valor reportado da un escalón. Las 8
copias paralelas simultáneas sí coinciden entre sí exactamente, o sea que a
carga constante el solver se comporta.

Consecuencias, en orden de importancia:

1. **El ítem "`n_jobs=1` y `n_jobs=8` dan las mismas evaluaciones" no se puede
   cumplir exacto, y no es un bug de T4exps.** Se reemplaza por medir la
   magnitud: ~3% de las evaluaciones cambian, con desvíos de hasta 0.2 pp.
2. **No invalida el experimento.** Lo que se compara son medias sobre N
   instancias. Con 3% de evaluaciones desviadas ~0.15 pp, la contribución al
   error de la media sobre N=200 es del orden de 0.002 pp, dos órdenes de
   magnitud por debajo del SE pareado. En la Etapa 1 los tres caminos dieron el
   mismo argmax.
3. **Sí entra en `v_resid` del Análisis 1**, o sea que infla ligeramente el
   residuo y por lo tanto *subestima* `frac_inst`. El sesgo va en la dirección
   conservadora: si `frac_inst` sale alta, lo es de verdad.
4. Si en algún momento se quiere reproducibilidad exacta entre corridas,
   la palanca es `cpupower frequency-set -g performance` (necesita root), no
   nada del lado de T4exps. **No se hizo**: hubiera cambiado la máquina bajo
   un experimento ya en marcha.

### Checklist de aceptación

- [x] **El comando formateado es exactamente el esperado.** Imprimirlo antes de
      ejecutar y compararlo con el que se corre a mano. Confirmar que
      `{INSTANCE}` se sustituye y que **no** se dispara el fallback que
      appendea la instancia al final (`runner.py:84-85`) — no debería, porque
      el template contiene `{INSTANCE}`.
- [x] **Cero `RunError`.** Si aparecen, es una de dos: los flags (el template
      tiene que ser `--alpha/--beta/--gamma/-p`, no `-a/-b/-g`) o el cwd (las
      rutas del listado son relativas a `/home/iaraya/T4exps`).
- [x] **El cwd no importa.** Correr el piloto desde un directorio cualquiera y
      confirmar que anda — con el listado absolutizado debería. Es el chequeo
      que evita descubrirlo a las 3 horas de un run de 26.
- [x] **El último float del stdout es la utilización.** Leer el stdout completo
      de una ejecución e identificar qué número queda último. Confirmar la
      escala: porcentaje (0-100), no fracción.
- [x] **La detección de familias funciona sobre las líneas con argumentos.**
      Ya verificado: 15 familias detectadas, y el prefijo estratificado de 30
      da exactamente 2 instancias por familia. No hace falta re-chequearlo
      salvo que se cambie de archivo de instancias.
- [x] **El `-t` del template gana.** Cronometrar: con `T=5` las ejecuciones
      tienen que rondar 5-9 s, no 30. Si tardan 30, el `-t` quedó antes del
      `-t 30` del archivo.
- [x] **Medir el peor exceso de tiempo**, en particular sobre la familia más
      grande presente. De ahí sale el `timeout` de las Etapas 2 y 3. Anotar el
      número acá.
- [x] **Ninguna ejecución muere por timeout.** Poner `timeout=2*T+15` y
      confirmar que no salta. Después provocar uno a propósito (`timeout=1`)
      para ver el modo de falla con los propios ojos.
- [x] **Secuencial e incremental dan idéntico.** El invariante central:
      ```python
      truth, seq_runs = sequential_execution(make_exp(), inst, n_jobs=1)
      eng = Engine(make_exp(), inst, n_jobs=8, timeout=2*T+15, nruns=10, batch=10)
      res = eng.run()
      assert res.output == truth
      assert res.likelihood == 1.0
      ```
- [~] **`n_jobs=1` y `n_jobs=8` dan lo mismo.** Mismo *output*, no las
      mismas evaluaciones: ver "El solver NO es determinista" arriba.** Correr el `Engine` dos veces
      cambiando solo `n_jobs` y comparar `res.output` **y** las listas de
      `eng.runner.evals`. Esto separa "el paralelismo anda" de "el paralelismo
      no rompe nada". Debería pasar (el solver no escribe archivos), pero es el
      chequeo que detecta lo contrario si el binario cambia.
- [x] **Ninguna instancia se evaluó dos veces.**
      `all(len(v) <= len(eng.runner.instances) for v in eng.runner.evals.values())`
      y `eng.runner.total_runs == sum(map(len, eng.runner.evals.values()))`.

**Compuerta:** no pasar a la Etapa 2 con ningún ítem abierto. Un parser que lee
el número equivocado hace que las dos etapas siguientes midan prolijamente la
cosa incorrecta.

---

## Etapa 2 — Piloto estadístico (~70 min de CPU + análisis)

**Las 5 configuraciones de `a` sobre 200 instancias estratificadas, con los
30 segundos reales.**

Esta etapa decide qué versión del paper se escribe. No se saltea.

### Diseño

Solo el barrido de `a` = `[0.0, 1.0, 2.0, 4.0, 8.0]`, el resto en la baseline.
Son 5 estrategias distintas, 5 × 200 = 1000 ejecuciones. A ~33 s cada una
(30 s más el exceso) sobre 8 jobs: ~69 min.

**Correr la matriz completa, no una ejecución especulada.** El objetivo de esta
etapa es el dataset, no el speedup: se necesitan las 1000 evaluaciones para
poder descomponer varianzas. Con un `Runner` directo:

```python
from t4exps import Runner, Strategy, order_instances

T = 30
TEMPLATE = "{INSTANCE} --alpha={a} --beta={b} --gamma={g} -p {p} -t " + str(T)

pool = [ln.strip() for ln in open("bsg_algo/instancesCLP-shuf.txt") if ln.strip()]
instances = order_instances(pool, "stratified", seed=0)[:200]

runner = Runner(instances, n_jobs=8, timeout=2*T+15)
strategies = [Strategy("bsg", f"{BIN} {TEMPLATE}", theta={**BASE, "a": a})
              for a in (0.0, 1.0, 2.0, 4.0, 8.0)]
for s in strategies:
    runner.run(s, len(instances))
```

### Persistir el dataset primero

Guardar la matriz cruda **antes** de cualquier análisis. Es el activo más caro
de la etapa y no se quiere volver a pagar una hora de CPU por un bug en el
análisis.

```python
import numpy as np
Y = np.array([runner.values(s) for s in strategies])     # (5, 200)
np.savez("stage2_evals.npz", Y=Y,
         instances=np.array(instances),
         thetas=np.array([s.theta["a"] for s in strategies]),
         seconds=np.array([runner.mean_seconds(s) for s in strategies]))
```

Con eso, todo el análisis queda offline y re-corrible.

### Análisis 1 — Descomposición de varianzas

La pregunta: **de esa desviación de ~0.95 puntos porcentuales, qué fracción es
efecto de instancia compartido.** Es la fracción que el pareado cancela y el
independiente paga.

Modelo aditivo de dos vías, el mismo que asume `PairedEstimator`
(`estimators.py:17`): `Y[s,i] = m + alpha_s + beta_i + e_si`.

```python
S, N  = Y.shape
m     = Y.mean()
alpha = Y.mean(axis=1) - m          # efecto de configuración
beta  = Y.mean(axis=0) - m          # efecto de instancia (compartido)
E     = Y - m - alpha[:, None] - beta[None, :]

# El residuo tiene (S-1)(N-1) grados de libertad, no S*N-1.
v_resid = (E**2).sum() / ((S - 1) * (N - 1))
# beta_i lleva el ruido promediado sobre S estrategias, y alpha_s sobre N
# instancias: hay que descontarlo o los efectos quedan inflados.
v_inst  = max(beta.var(ddof=1) - v_resid / S, 0.0)
v_strat = max(alpha.var(ddof=1) - v_resid / N, 0.0)
frac_inst = v_inst / (v_inst + v_strat + v_resid)
```

**Las correcciones de sesgo no son cosmética.** `beta.var(ddof=1)` incluye el
ruido residual promediado sobre las S estrategias, así que sobreestima el
efecto de instancia — y por lo tanto **infla `frac_inst`**, exactamente en la
dirección que favorece al pareado. Verificado sobre datos sintéticos con la
respuesta conocida:

| régimen | verdad | naive | corregido |
|---|---|---|---|
| `sigma_inst=0.9, sigma_resid=0.25` | 0.860 | 0.864 | 0.851 |
| `sigma_inst=0.1, sigma_resid=0.9` | 0.011 | **0.182** | 0.002 |

En el régimen bueno da casi igual. En el adverso —el que hay que detectar— el
estimador naive reporta 18% de efecto de instancia donde la verdad es 1%. Es
justo el caso en que uno concluiría "el pareado ayuda un poco" sin que ayude
nada. Usar la versión corregida.

**Interpretación:**

- `frac_inst` alto (> 0.7) → el pareado colapsa la varianza que decide las
  comparaciones de `Var(E)` a `tau^2`, y **el estimador pareado es la historia
  del paper**. La tabla de NOTES.md (0.879 → 0.637 con `sigma_inst=3`) pasa de
  ilustrativa sobre un solver dummy a medida sobre un benchmark real.
- `frac_inst` bajo (< 0.3) → el pareado es una mejora marginal y honesta, y va
  como **nota al pie**. La historia pasa a ser otra: los posteriors conjugados
  en lugar de MCMC, el early stopping, la estratificación.

### Análisis 2 — ¿La aditividad aguanta?

El pareado asume que las instancias difíciles son difíciles para todos **en la
misma magnitud**. Cuando las configuraciones son complementarias eso se rompe,
y se rompe **hacia el exceso de confianza** (README, Limitations) — el peor
modo de falla para una herramienta que reporta likelihoods.

Correlación de rangos de Spearman entre configuraciones a través de las
instancias. Sin scipy (el proyecto solo depende de numpy):

```python
def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])

rho = np.array([[spearman(Y[i], Y[j]) for j in range(S)] for i in range(S)])
iu  = np.triu_indices(S, k=1)
rho_min = rho[iu].min()
```

**Criterio de decisión:**

| `rho` mínimo entre pares | decisión |
|---|---|
| > 0.8 | pareado sin miedo |
| 0.4 – 0.8 | zona gris: reportar ambos estimadores, o bloquear por familia |
| ~0.4 o menos | volver a `estimator="independent"`, o bloquear por familia |

Mirar el **mínimo** sobre los pares, no el promedio: basta un par de
configuraciones complementarias para meter sobre-confianza en esa comparación,
y el promedio lo esconde.

Si cae en la zona gris, calcular `rho` **dentro** de cada familia BR antes de
rendirse. Si la aditividad aguanta intra-familia y se rompe entre familias,
bloquear por familia es la respuesta correcta y es un resultado publicable de
por sí.

### Análisis 3 — Calibrar `tolerance`

Insumo directo de la Etapa 3. La `tolerance` se compara contra diferencias de
medias (`core.py:141`, `engine.py:112-113`), así que la escala correcta es el
error estándar de la **diferencia pareada de medias**, en puntos porcentuales:

```python
D  = Y[:, None, :] - Y[None, :, :]        # (5, 5, 200) diferencias por instancia
se = D.std(axis=2, ddof=1) / np.sqrt(N)
tolerance = float(np.median(se[iu]))
```

Este `se` pareado debería ser **mucho** menor que el `0.95/sqrt(N)` que sale de
las desviaciones marginales, y esa brecha es exactamente el valor del estimador
pareado — el mismo resultado del Análisis 1 por otro camino. Si las dos cifras
son parecidas, ya está la respuesta.

### Análisis 4 — ¿Sirve `cost_aware`?

Con el `seconds` guardado: si la dispersión del costo por ejecución entre
instancias es del orden de 1.6x (lo medido con `-t 5`), `cost_aware=True` no
compra nada y agrega una división por un número ruidoso. Dejarlo en `False`
salvo que acá salga una dispersión sustancialmente mayor.

### Resultados medidos (ejecutado)

`runs/stage2_matrix.py` (1000 ejecuciones, 3863 s de reloj a 8 jobs, **cero
errores**), analizado offline por `runs/stage2_analysis.py`. Dataset en
`runs/stage2_evals.npz` y `runs/stage2_evals.csv`.

| alpha | media (pp) | sd (pp) |
|---|---|---|
| 0.0 | 94.9064 | 1.0381 |
| 1.0 | 95.2749 | 0.8848 |
| 2.0 | 95.4466 | 0.8604 |
| **4.0** | **95.4516** | 0.8440 |
| 8.0 | 95.3037 | 0.9154 |

La sd marginal promedio es **0.9085 pp**, que confirma sobre el benchmark real
el ~0.95 que NOTES.md traía de otro lado.

**El óptimo es plano.** La separación entre configuraciones consecutivas, medida
en errores estándar pareados:

| paso | diferencia | en SE |
|---|---|---|
| 0.0 → 1.0 | +0.3685 pp | +10.8 |
| 1.0 → 2.0 | +0.1717 pp | +6.2 |
| 2.0 → 4.0 | +0.0050 pp | **+0.2** |
| 4.0 → 8.0 | -0.1479 pp | -5.4 |

`alpha=2.0` y `alpha=4.0` son **indistinguibles**: 0.005 pp de diferencia contra
una `tolerance` de 0.0316. Eso no es un problema del método, es el resultado: la
superficie de tuning tiene una meseta y el argmax dentro de la meseta es
arbitrario. Vale reportarlo así en el paper, y es justo el caso donde
`tolerance` deja de ser cosmética — sin ella el motor gasta ejecuciones tratando
de resolver un empate que no existe.

#### Análisis 1 — Descomposición de varianzas

```
v_inst  (efecto instancia, corregido) = 0.72455
v_strat (efecto configuracion, corr.) = 0.04881
v_resid (residuo)                     = 0.10567
frac_inst = 0.8243        (naive sin correccion: 0.8279)
```

**`frac_inst` = 0.82 > 0.7: el estimador pareado es la historia del paper.** El
82% de la varianza que ve una comparación es efecto de instancia compartido, o
sea varianza que el pareado cancela entera y el independiente paga entera. La
tabla de NOTES.md deja de ser una ilustración sobre un solver dummy y pasa a ser
una medida sobre un benchmark real.

La corrección de sesgo casi no movió el número (0.8279 → 0.8243), que es lo
esperado en el régimen bueno según la validación sintética del plan. Pero eso se
sabe *después* de correr: en el régimen adverso la diferencia era 0.182 contra
0.002. Haberla usado no costó nada y era la única forma de distinguir los dos
casos.

Nota: el jitter del solver (ver Etapa 1) cae dentro de `v_resid`, así que si
algo hace es **subestimar** `frac_inst`. El 0.82 es un piso.

#### Análisis 2 — Aditividad

Spearman entre configuraciones a través de las 200 instancias:

```
        0.0     1.0     2.0     4.0     8.0
0.0    1.000   0.859   0.838   0.851   0.840
1.0    0.859   1.000   0.884   0.851   0.857
2.0    0.838   0.884   1.000   0.865   0.842
4.0    0.851   0.851   0.865   1.000   0.882
8.0    0.840   0.857   0.842   0.882   1.000
```

`rho_min = 0.838 > 0.8`: **pareado sin miedo.** Las instancias difíciles son
difíciles para todas las configuraciones, y en magnitudes parecidas. No hace
falta el análisis intra-familia ni bloquear por familia. El modo de falla que
preocupaba —sobre-confianza por configuraciones complementarias— no está.

#### Análisis 3 — Calibración de `tolerance`

```
SE de la diferencia de medias, PAREADO = 0.03165 pp
SE de la diferencia de medias, INDEP.  = 0.08848 pp
brecha = 2.80x  ->  el independiente necesita 7.8x mas instancias
```

**`tolerance = 0.0316`** a N=200.

Cuidado al comparar: las dos cifras tienen que ser el SE de una *diferencia de
medias*. El independiente paga `sqrt(2)*sd/sqrt(N)`, no `sd/sqrt(N)`; usar esto
último subestima su error en un factor `sqrt(2)` y hace ver al pareado peor de
lo que es. Con la comparación correcta, el pareado es **2.8x más preciso**, o
sea que el independiente necesitaría **7.8x más instancias** para la misma
precisión. Es el mismo resultado del Análisis 1 por otro camino y las dos cifras
cierran: `2*v_resid / (2*sd_marginal^2) = (1/2.80)^2`.

**`tolerance` escala con N.** La sd de la diferencia pareada es 0.4476 pp, así
que el SE es `0.4476/sqrt(N)`. A las 1500 instancias de la Etapa 3 eso da
**0.0116**, no 0.0316. Usar el valor de N=200 sobre N=1500 declararía empates a
2.7 SE, que es demasiado. Lo que se transporta entre etapas es la sd de la
diferencia pareada, no la `tolerance` ya dividida.

#### Análisis 4 — `cost_aware`

Con `-t 30` vinculante, todas las instancias cuestan lo mismo: mediana 30.50 s,
rango entre familias de **1.05x** (BR1 30.07 s, BR14 31.62 s). No hay nada que
ganar dividiendo por un costo constante, y sí una división por un número ruidoso
que perder. **`cost_aware = False`**, confirmado con los 30 s reales.

De paso sale el peor wall sobre 1000 ejecuciones: **33.27 s**, +11% sobre el
límite nominal. `timeout = 75 s` deja 2.25x de margen.

### Checklist de aceptación

- [x] `stage2_evals.npz` guardado, con las 1000 evaluaciones completas.
- [x] `frac_inst` = **0.8243** (corregida). **El pareado es la historia**, no
      la nota al pie.
- [x] Matriz `rho` calculada; `rho_min` = **0.838** → "pareado sin miedo".
- [x] No cae en zona gris, así que no hizo falta el `rho` intra-familia.
- [x] `tolerance` = **0.0316** pp a N=200; **0.0116** pp a N=1500 (escala
      con `1/sqrt(N)` desde una sd de la diferencia pareada de 0.4476 pp).
- [x] `cost_aware = False`: dispersión de costo entre familias de 1.05x.
- [x] **Cero** `TimeoutExpired` y cero `RunError` en las 1000 ejecuciones.
      B2 validado con los 30 s reales.

**Compuerta:** de acá salen cuatro parámetros que la Etapa 3 necesita como
entrada (`estimator`, `tolerance`, `cost_aware`, `timeout`) más la decisión
editorial sobre el paper.

---

## Etapa 3 — Experimento completo

Solo después de cerrar las dos compuertas.

### Arreglar `examples/bsg.py` primero

Dos cambios obligatorios:

1. El template: `--alpha={a} --beta={b} --gamma={g} -p {p} -t {T}`, con `-t` al
   final. Hoy es `-a {a} -b {b} -g {g} -p {p}` y falla en toda evaluación.
2. El default de `--timeout` es 30 (`bsg.py:50`), que mata el proceso siempre.
   Cambiarlo al valor medido en la Etapa 1.

**HECHO.** Además de esos dos: `--solver-time` es ahora un flag propio (el
presupuesto es del solver, el `timeout` es red de seguridad), `--timeout`
quedó en 75 s con la medición que lo justifica en el help, y `cost_aware`
dejó de estar cableado en `True` — pasa a ser `--cost-aware`, apagado por
defecto, porque con un `-t` vinculante el costo por ejecución no varía.

### El run

```bash
cd /home/iaraya/T4exps
python examples/bsg.py \
  --bin bsg_algo/BSG_CLP \
  --instances bsg_algo/instancesCLP-shuf.txt \
  --jobs 8 --timeout <TIMEOUT_MEDIDO> \
  --confidence 0.98 --tolerance <TOL> \
  2>&1 | tee bsg_full.log
```

`verbose=True` ya está puesto en `bsg.py:68`, y el script escribe
`bsg_history.csv` con likelihood contra número de ejecuciones — el insumo de la
figura que falta en la sección 5.2.

### La aritmética

El grid completo son 17 valores sobre 4 parámetros; el barrido secuencial
instancia entre 14 y 17 estrategias distintas según el camino. El baseline
secuencial es `len(involved) * n_instances` (`engine.py:264`):

```
~15 estrategias × 1500 instancias = ~22 500 ejecuciones
22 500 × ~33 s / 8 jobs ≈ 26 horas de reloj
```

**Corrección de una estimación anterior:** en una versión previa de este plan
había puesto 18 días. Está mal; son ~26 horas. Cambia la conclusión: la
ejecución exacta completa es **asequible en un día**, así que el
"idéntico a secuencial" sobre el benchmark real está al alcance y no hay que
conformarse con el dummy.

Aun así `confidence=0.98` vale: la fase rápida debería dar el resultado en una
fracción de esas 26 horas, y sin early stopping el run no salva una sola
ejecución porque la exactitud exige evaluar todo (NOTES.md, punto 4).

### Ejecutar en dos fases

1. **Fase rápida, `confidence=0.98`.** Da el resultado del paper, la
   trayectoria de likelihood para la figura, y el speedup reportado.
2. **Fase exacta, `confidence=1.0`.** ~26 h. Es la que da el "idéntico a
   secuencial" sobre el benchmark real en vez de sobre el solver dummy.
   Decidir con el número de la fase 1 en la mano.

### Checklist de aceptación

### Ensayo general (ejecutado): 200 instancias, parámetros finales

`runs/stage3_full.py --limit 200 --confidence 0.98 --tolerance 0.0316`.
19 minutos, 260 evaluaciones, cero errores.

```
(4.0, 2.0, 0.3, 0.01)   likelihood 100%
runs = 260   sequential_runs = 2800   speedup = 10.77x
```

La trayectoria de likelihood (el insumo de la figura de la sección 5.2):

| ejecuciones | likelihood | profundidad |
|---|---|---|
| 210 | 95.2% | 17 |
| 235 | 96.9% | 17 |
| 260 | 100.0% | 17 |

La profundidad llega a 17 —todas las comparaciones del barrido— ya en la
primera iteración: con 15 evaluaciones por estrategia el motor ya cree en el
camino entero, y lo que le falta es solamente confianza en el *output*.

**Extrapolación a 1500 instancias.** El baseline secuencial crece a
14 x 1500 = 21 000 ejecuciones, pero el conteo del motor no escala con N (el
mínimo es `nruns=15` por estrategia y los lotes son de 25). Así que el speedup
debería *crecer*, no bajar. Al costo medido de 3.86 s por ejecución a 8 jobs,
la fase exacta serían ~22.5 h y la fase rápida un par de horas.

### Un defecto de la librería que expone el ensayo

`wasted_runs = -2540`. En `engine.py:266` se calcula como
`total_runs - sequential_runs`, o sea que es **negativo siempre que haya
speedup**, y además es redundante: se obtiene restando dos campos que el
`Result` ya expone. No hay lectura bajo la cual "-2540 ejecuciones
desperdiciadas" signifique algo, y `Result.__str__` lo imprime.

No bloquea (el checklist de abajo no lo pide), pero hay que decidir qué es
"desperdiciado" antes de que el número entre al paper. La noción que tendría
sentido es *ejecuciones gastadas en estrategias que no están en el camino
ganador*, que se puede reconstruir de `ctx.decisions`; la actual no es esa.

### Checklist de aceptación

- [x] `examples/bsg.py` arreglado (template y timeout) y probado con
      `--instances` de 30 líneas antes de largar.
- [x] Antes del run largo: **200 instancias** con los parámetros finales.
      19 minutos, cero errores, speedup 10.77x.
- [x] Detección de familias verificada sobre el archivo real (Etapa 1).
- [x] Log en disco: `runs/stage3_<tag>.log`, más un CSV por evaluación
      escrito en el momento por `runs/monitor.py`.
- [x] `stage3_*_history.csv` guardado.
- [x] `eng.runner.evals` guardado al final (vía `Engine` directo, B3) en
      `runs/stage3_<tag>_runner.pkl`, dentro de un `finally` para que
      sobreviva a un aborto.
- [x] Reportar `res.runs`, `res.sequential_runs`, `res.speedup` y
      `res.likelihood`.
      **Exacto:** runs 21 745 · sequential 21 000 · speedup 0.97× · likelihood 1.0 · 24.74 h.
      **Con los cuatro arreglos (replay, 3 semillas):** 6745/7245/6870 · 3.11×/2.90×/3.06× ·
      likelihood 98.0–98.25% · output = referencia.

---

### Resultados de la Etapa 3 (cerrada)

**Ensayo de 200 instancias** (`runs/stage3_full.py --limit 200`): 260 ejecuciones vs
2800, 10.77×, "likelihood 100%", output `(4.0, 2.0, 0.3, 0.01)`. Pipeline completo
OK de punta a punta. Wall 0.31 h.

**Fase 1** (1500 instancias, `confidence=0.98`, `tolerance=0.0116`): 260 ejecuciones vs
21 000, **80.8×**, "likelihood 99%", output `(4.0, 2.0, 0.3, 0.01)`.
**RETIRADA.** Ver D1: el 99% se calculó sobre 80 simulaciones vivas de 400 (19% con
denominador completo), y el output tiene γ=0.3 cuando la referencia da γ=0.4.

**Fase exacta, intento 1** (`confidence=1.0`): paró al 17% (3570 ejecuciones) con
"likelihood 100%" — una simulación viva que acertó: `1.0 >= 1.0`. El mismo defecto
corta también el modo "exacto". Relanzada con `--confidence 2.0` (inalcanzable) y
`--resume` (reaprovechó las 3570). **Cerró: 21 745 ejecuciones** (21 000 del camino +
745 en 18 ramas que el picoteo abrió y abandonó), 14/14 estrategias completas, cero
errores, 24.74 h de reloj (+4.8 h del intento 1). `speedup = 0.97×`: el modo exacto
*es* el barrido secuencial más el desperdicio del picoteo, como debe ser.

**Referencia definitiva (medias completas con tolerance 0.0116, las 14 estrategias
a 1500):** `(4.0, 2.0, 0.4, 0.01)`. α=4.0 vs α=2.0: +0.0948 pp (9.3 SE) — la
"meseta" de la Etapa 2 (0.005 pp a N=200) no lo era a N=1500. β=2.0 vs β=1.0:
+0.0323 pp (3.4 SE). γ=0.4 vs γ=0.3: +0.0242 pp (2.6 SE). γ se da vuelta a favor de 0.4 sólo desde
N≈126 en el prefijo estratificado; en las primeras 15 instancias 0.3 va +0.20 pp
arriba, y **todas las paradas del motor sesgado cayeron ahí**.

#### Cuatro defectos, cada uno con reproducción

**D1 — Denominador de la likelihood** (`engine.py:189-193`, `:162-163`).
`likelihood = aciertos / simulaciones_vivas`; las que abortan se descartan. Una
simulación aborta cuando su camino pide una estrategia sin datos
(`core.py:139-142`), es decir cuando *discrepa* con la predicción. El denominador
está correlacionado con el acuerdo: sesgo hacia arriba, creciente con la tasa de
aborto, y `1/1 = 100%` satisface cualquier `confidence`. Medido con
`runs/annotate_trajectory.py`: fase 1 en 260 → 80 vivas, 77 aciertos, 96% reportado,
**19% honesto**; exacto en 3570 → 1 viva → 100%. Fix (una línea): denominador =
simulaciones intentadas (`HonestEngine` en `runs/replay.py`). Insuficiente solo: ver D3.

**D2 — Profundidades desiguales** (`tests/test_unequal_depth.py`, xfail estricto).
El nivel de una estrategia se estima con la media de TODAS sus observaciones; si fue
evaluada más a fondo, ese nivel absorbe la dificultad de las instancias que nadie
más vio (k=1) y su efecto de instancia queda mal. Sintético con B mejor por +0.10 pp
en promedio: P(elige B) = 98% a 40/40, **0% a 40/60 y 40/90**. Real: β=1.0 (40) vs
β=2.0 (90) — `compare_observed` daba β=2.0 por +0.047 y las 400 simulaciones daban
β=1.0 → todas abortaban. Fix: niveles desde el bloque compartido (k≥2), efectos de
instancia como residuo contra esos niveles (`paired_k2`, `runs/estimators_fixed.py`):
0% → 90% / 70%.

**D3 — Estrategia dominante** (`tests/test_dominant_strategy.py`, xfail estricto).
Cuando casi todas las instancias vistas tienen k=1, `inv_k → 1`, `denom → 0`,
`s_b2` explota y `tau2 = max(v_total − s_b2, 1e-12)` **colapsa a cero**: los 400
draws son idénticos, el posterior es una masa puntual sobre lo que digan 15 puntos.
Real (honesto+paired, semillas 2 y 3, ejecución 1410): 1075/1140 instancias con k=1,
`s_b2 = 3.87` (real 0.72), `tau2 = 1e-12`, **400/400 en γ=0.3** — parada con
certeza total en la respuesta equivocada. Fix: componentes de varianza sólo en el
bloque compartido (`k2`); residuo por ANOVA de dos vías, insensible a k heterogéneo
(`paired_k3`: tau2 de 0.45× → 0.84× el real).

**D4 — La brújula se apaga.** El impacto se mide contra `probable_state` (prefijo con
≥10% de likelihood). Si las simulaciones que discrepan abortan, supervivencia → 0,
`depth = 0`, `base = 1`, impacto **0 para todas**, empate, gana la primera en orden
de código: el motor machaca `a=0.0` (la inicial, que pierde α por 0.55 pp) hasta
1500. Ocurre **también con k2/k3**: honesto+k3 mandó +1100 evaluaciones a `a=0.0`.
Es estructural: las simulaciones que irían a explorar la rama incierta mueren antes
de contarse. Fix: imputar del prior las estrategias sin datos (`ImputingEngine`,
`runs/replay.py --impute`): depth 8–12 en vez de 0, y la asignación va a la meseta
α (2.0/4.0/8.0) y a β — cero a `a=0.0`. Exige `--allow-solver` cuando la brújula
pide ramas que el oráculo no tiene (en curso).

#### Ablación contra el oráculo (`runs/replay.py`, `runs/ablation.json`)

Con la matriz del run exacto como cache, cualquier variante se re-juega en minutos:
mismos datos, distinta regla. El replay de la fase 1 con semilla 0 reprodujo la
trayectoria del intento exacto 1 **bit a bit** (3570 hits, 0 misses, 97 s vs 4.8 h).

| denominador | estimador | resultado |
|---|---|---|
| sesgado | paired | 0/6 correctas · correctas a — · incorrectas a 5.88–89.36× |
| sesgado | paired_k2 | 0/1 correctas · correctas a — · incorrectas a 73.68–73.68× |
| honesto | paired | 2/4 correctas · correctas a 1.47–1.62× · incorrectas a 14.89–14.89× |
| honesto | paired_k2 | 4/4 correctas · correctas a 1.43–1.95× · incorrectas — |
| honesto | paired_k3 | 1/1 correctas · 1.35× · sólo 2 iteraciones con L=0 (vs 99–151): el tau2 ANOVA mantiene las simulaciones vivas, pero sigue miope al prefijo |
| **imputación** | **paired_k3** | **3/3 correctas · 2.90×–3.11×** · paradas en 6745/7245/6870 · 0–1 iteraciones con L=0 · 115 evaluaciones reales por semilla, siempre las mismas 7 estrategias (rama α=8.0) |

**El speedup grande sólo aparece en las celdas que se equivocan.** Sin imputación
las correctas quedan en 1.35–1.95× con la brújula apagada (o miope al prefijo).
**Con los cuatro arreglos (denominador honesto + k3 + imputación) el motor llega a
la referencia a 3.11×, nunca pierde la brújula, y gasta 115 evaluaciones reales
(1.7%) en descartar la rama α=8.0 que el run exacto jamás exploró** — la
trayectoria de likelihood cae a 7% justo cuando γ pasa a ser la duda, manda las
evaluaciones a γ=0.3/0.4 y remonta a 98%. **Tres semillas: 3.11×, 2.90×, 3.06×, las tres con la referencia, paradas entre 6745 y 7245** — y las tres exploraron exactamente la misma rama con las mismas 115 evaluaciones reales.

#### Variabilidad de la convergencia (motor con los cuatro arreglos)

`runs/build_variability.py` sobre **23 semillas** re-jugadas contra el oráculo
(`runs/oracle_extended.csv`: la matriz exacta más las evaluaciones reales que las
semillas pidieron al explorar la rama α=8.0, 22 550 pares en total):

- **23/23 paran en la referencia** `(4.0, 2.0, 0.4, 0.01)`.
- Parada: mín 6320, mediana 7045, máx 8120 ejecuciones (±13% alrededor de la
  mediana; el original paraba entre 235 y 3570, un factor 15, y siempre mal).
- Speedup: mín 2.59×, mediana 2.98×, máx 3.32×.
- 21 de las 23 sin una sola iteración en L=0; las otras dos, 1 y 2.
- La forma es la misma en todas: L sube al resolverse α y β, cae (hasta ~7%) cuando
  γ=0.3 vs 0.4 pasa a ser la duda y recibe las evaluaciones, y remonta al umbral.
- Exploración: cada semilla gastó 90–290 evaluaciones *reales* fuera del camino
  (la sub-rama α=8.0 → β → γ → p), 5–10% de su total. Distintas semillas la
  recorren a distinta profundidad, por eso el oráculo se completó en dos vueltas
  (`runs/topup_oracle.py`) y cuatro semillas corrieron con `--allow-solver`.

Contra el trabajo mínimo: si cada comparación se certificara por separado al 98%
con su SE pareado harían falta ~9355 evaluaciones; el motor para en ~7000 porque
certifica el *output* (no cada decisión) y el modelo pareado comparte los efectos
de instancia conocidos. La asignación va donde está la información: `β=1.0/2.0`
~1150 cada una, `γ=0.2/0.3/0.4` 870–1140, perdedores claros en 15–65.

#### Calibración: el orden de las instancias es la variable aleatoria

Primer intento (`runs/stage3_cal_*`): 20 semillas Monte Carlo por umbral, orden
estratificado fijo (`seed=0`). Resultado completo (18 réplicas por umbral; 2 murieron
por *cache miss* pidiendo `a=8.0;b=0.5;γ=0.4;p=0.02` más profundo):

| confidence | correctas | fracción | L media al parar | runs medios |
|---:|---:|---:|---:|---:|
| 0.6 | 0/18 | 0.00 | 0.615 | 1800 |
| 0.8 | 2/18 | 0.11 | 0.809 | 2541 |
| 0.9 | 14/18 | 0.78 | 0.913 | 5279 |
| 0.98 | 18/18 | 1.00 | 0.983 | 7078 |

Todas las incorrectas dan γ=0.3.
Parece sobreconfianza brutal, **pero no mide lo que dice medir**: todas las
réplicas comparten el mismo prefijo de instancias, y en las primeras ~15 γ=0.3
va +0.20 pp arriba por azar (el cruce a favor de γ=0.4 llega en N≈126). La semilla
sólo mueve el Monte Carlo, no los datos; 0/18 es *una* realización del orden,
repetida. Un posterior perfectamente calibrado también daría ~97% a γ=0.3 con esas
15 instancias — la verdad es una sorpresa de ~2σ respecto al prefijo.

Consecuencia: **calibrar exige aleatorizar la estratificación** (`--order-seed` en
`runs/replay.py`), no la semilla del Monte Carlo. Con orden aleatorio, las ramas
exploradas (α=8.0 …) están en el oráculo sólo para el orden por defecto, así que
esas corridas necesitan `--allow-solver`. Los niveles 0.9 y 0.98 del primer
intento siguen siendo válidos como lo que son: con este orden, a umbral alto el
motor no para hasta resolver γ y acierta.

#### A/B: impacto medido sobre el output vs sobre el prefijo

`Engine(impact_on="output")`: el impacto de una estrategia se mide sobre la
likelihood del *output* — lo que la regla de parada certifica — en vez de sobre el
prefijo de decisiones creído. Mismas semillas (30–39), `confidence=0.98`, contra el
oráculo; 5 semillas necesitaron solver porque exploran la rama (α=8.0, β=1.0) más
profundo que el oráculo (50–150 evaluaciones reales cada una).

**Resultado: bimodal.** Las 10 corridas por output son correctas. En las 8 parejas
comparables, output gana 6/8 y ahorra de media 9%
(7226 → 6579; speedup 2.91× → 3.19×). Pero cuando funciona
ahorra ~20% (4980–6380) y cuando falla se dispara (7930–12865), y esas corridas pasan
**57–244 iteraciones con L=0**. Mecanismo: con `impact_on="output"`, `base` es la
likelihood del output; cuando ninguna simulación reproduce la tupla, L=0 → `base=1e-9`
→ todo impacto satura en 1.0 → empate → orden de código. Es la brújula apagada (D4)
por otra puerta: el criterio por output es excelente cuando la respuesta ya está a la
vista y ciego cuando no.

**Híbrido (`impact_on="auto"`)**: output cuando L ≥ 0.05 (≥ ~20 simulaciones
reproducen el output), prefijo si no. Mismas 10 semillas: **10/10 correctas**, 6382
ejecuciones de media (5995–7420), **L0 máximo 8** (output puro: hasta 244).
En las 8 parejas comparables: 6362 vs 7226 del prefijo (**-12%**),
gana 7/8; contra output gana 4/8 — output es mejor cuando funciona y peor cuando se
apaga; auto se queda con la estabilidad del prefijo y casi toda la ganancia. Speedup
2.91× → 3.30×. Candidato a default en 0.3 (pendiente hasta cerrar la calibración,
que corre con el default actual). Datos en `runs/ab_impact.json`.

#### Integración al paquete (0.2.0)

Los cuatro arreglos están en `t4exps/`: `estimator="paired"` es ahora el de dos
pasos con ANOVA (`PairedEstimatorK3`; el original queda como `"paired_legacy"`),
y `Engine` tiene `honest_likelihood=True` e `impute_missing=True` por defecto.
`sequential_execution` acepta `tolerance`, `Result.wasted_runs` pasa a ser "evaluaciones
en ramas fuera del camino final" (nunca negativo) y `Snapshot.alive` registra las
simulaciones vivas. Con `estimator="paired_legacy", honest_likelihood=False,
impute_missing=False` se reproduce el motor original bit a bit, que es lo que
`runs/replay.py` hace por defecto para que la ablación siga siendo re-jugable.
Los tests de los defectos D2 y D3 corren sobre ambos: xfail estricto en el legado,
pasan en el nuevo. Suite: 45 passed, 2 xfailed.

#### Hallazgos operativos

- **`batch` debe ser múltiplo de `n_jobs`.** Con 25 y 8 jobs cada lote son olas
  [8, 8, 8, 1]: un solver solo durante 30 s con 7 cores ociosos. Medido: 25% del
  wall de solver perdido; el "overhead Monte Carlo" que creí ver era esto. Con
  `batch=24` el run exacto habría tardado ~19 h en vez de ~26.
- **`tolerance` escala con 1/√N**: 0.0316 a N=200 → 0.0116 a N=1500 (sd de la
  diferencia pareada 0.4476 pp). Se transporta la sd, no la tolerance.
- **La regla de parada es caótica con el defecto D1**: mismo algoritmo, misma
  semilla, dos datasets que difieren sólo por el jitter del solver → paradas en
  260 y 3570.
- **Jitter del solver a escala (cierra el ítem abierto de la Etapa 1).** Los 260
  pares de la fase 1 y los 260 del ensayo, re-evaluados por el run exacto en otro
  momento y otra carga: **5/260 (1.9%) y 8/260 (3.1%) distintos**, |dif| máx 0.25 pp,
  sesgo medio ~0. El mayor desvío de una media por estrategia es 0.015 pp: bajo la
  tolerance y ~10× menor que el SE pareado a n=15. No afecta decisiones; sí impide
  la reproducibilidad bit a bit entre corridas reales — por eso las comparaciones
  entre variantes del motor se hacen por replay contra el oráculo.
- El CSV del run exacto tenía 29 filas ajenas (un probe ejecutó el solver antes de
  `ReadOnlyRunner`); **purgadas** con `runs/purge_foreign_rows.py exact --apply`
  (quedan en `stage3_exact_evals.FOREIGN.csv`, respaldo `.con_ajenas.bak`). Los
  probes ya no pueden ejecutar.
- Herramientas nuevas en `runs/`: `status.py`, `monitor.py` (`LoggingRunner`
  con `--resume`, `ReadOnlyRunner`, `OracleRunner`), `replay.py`,
  `annotate_trajectory.py`, `diag_sims.py`, `estimators_fixed.py`,
  `purge_foreign_rows.py`, `viz_build.py` + dashboard publicado.

## Deuda de tests que este plan expone

Los 25 tests actuales pasan (91% de cobertura), pero **el camino subprocess
está sin cubrir** (`runner.py` al 74%): template de comando, `RunError` en exit
code distinto de cero, `_parse_last_float`, y la rama `n_jobs>1`. Es decir:
justo todo lo que la Etapa 1 va a ejercitar por primera vez, a mano. El bug de
los flags `-a/-b/-g` en `examples/bsg.py` es exactamente la clase de cosa que
un test de esa rama habría atrapado.

`confidence` —la feature de la que depende la Etapa 3— tampoco tiene un solo
test.

**SALDADA.** `tests/test_subprocess_path.py`, 13 tests sobre un solver falso
de shell que imita a BSG (acepta `--alpha=`, imprime cabecera y valor último).
Suite completa: 38 pasan.

- [x] Estrategia con template de comando: sustitución de `{INSTANCE}` y de
      `theta`, y parseo del stdout.
- [x] Instancia multi-token (con argumentos propios) que sobrevive
      `shlex.split`, y `default_family` sobre ella.
- [x] Exit code distinto de cero → `RunError`.
- [x] Stdout sin número → `RunError`.
- [x] `n_jobs=8` da las mismas evaluaciones que `n_jobs=1`.
- [x] `confidence=0.8` corta antes que `confidence=1.0` y devuelve
      `likelihood >= 0.8`.
- [x] `TimeoutExpired`: comportamiento definido y testeado.

Tres más que el plan no pedía y que salieron de lo medido acá:

- [x] **El último `-t` gana** — el mecanismo del que depende todo el diseño de
      presupuesto, y que hasta ahora era un hecho observado sin test.
- [x] **El paralelismo preserva el orden de instancias** — `evals[k][j]` tiene
      que corresponder a `instances[j]` o el pareado compara cosas distintas.
- [x] **Una línea de debug al final rompe la medición en silencio** — el test
      documenta la fragilidad de `_parse_last_float` en vez de dejarla como
      nota en prosa.
