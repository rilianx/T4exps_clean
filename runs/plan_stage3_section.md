### Resultados de la Etapa 3 (en curso: el run exacto cierra en ~4 h)

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
`--resume`. Al escribir esto va al 84% del camino, 10/14 estrategias completas.

**Referencia (medias completas con tolerance, provisional hasta que cierre):**
`(4.0, 2.0, 0.4, 0.01)`. β=2.0 vs β=1.0: +0.0323 pp (3 SE, 1500 inst). γ=0.4 vs
γ=0.3: +0.0226 pp (2.3 SE, 1356 inst). γ se da vuelta a favor de 0.4 sólo desde
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
| honesto | paired_k2 | 3/3 correctas · correctas a 1.46–1.95× · incorrectas a — (+1 cache miss) |
| imputación | paired_k3 | en curso con solver real |

**El speedup grande sólo aparece en las celdas que se equivocan.** Las correctas
están en 1.5–2× con la brújula apagada 100–150 iteraciones; la imputación es la
candidata a subir eso sin perder la respuesta.

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
- El CSV del run exacto tiene 29 filas ajenas (un probe ejecutó el solver antes de
  `ReadOnlyRunner`); `runs/purge_foreign_rows.py exact --apply` al terminar. Los
  probes ya no pueden ejecutar.
- Herramientas nuevas en `runs/`: `status.py`, `monitor.py` (`LoggingRunner`
  con `--resume`, `ReadOnlyRunner`, `OracleRunner`), `replay.py`,
  `annotate_trajectory.py`, `diag_sims.py`, `estimators_fixed.py`,
  `purge_foreign_rows.py`, `viz_build.py` + dashboard publicado.

