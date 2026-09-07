# T4exps

**Algorithm experiments with incremental speculative execution.**

You write your experiment as a Python function made of pairwise comparisons.
T4exps runs it *speculatively*: it evaluates a fraction of the instances,
predicts the winner of each comparison, and reports a provisional answer with a
likelihood attached. As evidence accumulates the answer sharpens. Let it run to
the end and the result is, by construction, exactly the one a full sequential
execution would have produced — you just got a usable answer hours earlier.

```
Reported likelihood 92% after 195 of 450 executions.
Best configuration so far: a=4.0, b=2.0, g=0.4, p=0.01
```

---

## Install

Requires Python 3.10+. The library itself needs only NumPy.

```bash
git clone https://github.com/rilianx/T4exps_clean.git
cd T4exps_clean
pip install -e .
```

For the tests and the figure scripts:

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q          # 43 tests + 2 expected failures, ~23 s
```

---

## Quick start

```python
from t4exps import Strategy, best_strategy, experiment_execution

def experiment():
    A = Strategy('./solverA')
    B = Strategy('./solverB')
    C = Strategy('./solverC')
    best = best_strategy(A, B)      # compare A and B
    best = best_strategy(best, C)   # then the winner against C
    return best.name

result = experiment_execution(experiment, "instances.txt", n_jobs=8)
print(result)
```

`instances.txt` lists one instance per line. Each strategy is invoked once per
instance and must print a numeric score on stdout; **higher is better**, so
negate runtimes if that is your measure.

Run the same experiment without speculation to get the baseline:

```python
from t4exps import sequential_execution
truth, runs = sequential_execution(experiment, "instances.txt")
```

### Parameters and command templates

```python
S = Strategy('bsg', './BSG_CLP {INSTANCE} --alpha={a} --beta={b} -t 30',
             theta={'a': 0.0, 'b': 0.5})
S2 = S.with_theta(a=4.0)            # a copy with one parameter changed
```

`{INSTANCE}` is replaced by the instance, `{name}` by the corresponding entry of
`theta`. A Python callable `f(instance, **theta) -> float` works too, which is
what the dummy example uses.

An instance line may itself carry arguments (`BR8.txt -i 53 --min_fr=0.98 -t 30`):
the formatted command is split with `shlex`, so it expands to several tokens.
Put the solver's own time-limit flag **last** in the template — for BSG_CLP the
last `-t` wins, which is how a per-stage budget overrides the one in the file.
The score is parsed as the last float on stdout by default; pass `parse=` for
anything else.

### Printing preliminary results

```python
from t4exps import anytime

def experiment():
    ...
    if anytime():                   # true only on the user-visible pass
        print("best so far:", S.params)
    return tuple(S.theta[k] for k in ('a', 'b'))
```

Guard every `print` with `anytime()`. The experiment function is re-executed
hundreds of times internally; unguarded output would flood your terminal.

---

## Running the examples

```bash
# Factorial design on a synthetic solver; compares both estimators
python examples/factorial.py --sigma 5 --sigma-inst 3

# Does a reported 90% really mean 90%?  Calibration study
python examples/calibration.py --reps 20 --instances 40

# Regenerate every figure in figures/
python examples/make_figures.py --reps 12 --instances 50

# Four-parameter sweep of the BSG container-loading solver
python examples/bsg.py --bin ./BSG_CLP --instances BR.txt --jobs 8 \
    --solver-time 30 --timeout 75 --confidence 0.98 --tolerance 0.0116
```

`examples/bsg.py` writes `bsg_history.csv` with likelihood against number of
executions, ready to plot.

---

## Options

Passed as keyword arguments to `experiment_execution`.

| option | default | what it does |
|---|---|---|
| `estimator` | `"paired"` | `"paired"` cancels instance difficulty between strategies; `"independent"` reproduces the classic per-strategy model. See [NOTES.md](NOTES.md). |
| `confidence` | `1.0` | Stop as soon as the reported likelihood reaches this. **Without it the run only saves time-to-answer, not executions** — exactness requires evaluating everything. See *Known defects*: with the current likelihood denominator a single surviving simulation reports 100 % and satisfies even `confidence=1.0`; pass `confidence=2.0` to force exhaustion. |
| `tolerance` | `0.0` | Differences below this count as a tie. Keeps the engine from burning thousands of runs separating strategies that differ by less than the standard error. |
| `nruns` | `10` | Executions given to a strategy the first time it appears. |
| `batch` | `10` | Executions added each iteration to the selected strategy. |
| `nsims` | `400` | Monte Carlo simulations per likelihood estimate. |
| `nsims_impact` | `nsims/4` | Cheaper budget for the per-strategy impact estimates. |
| `percentile` | `0.05` | Optimistic/pessimistic quantile used to measure impact. |
| `instance_order` | `"stratified"` | How the instance file is ordered before use. `"stratified"` interleaves families so every prefix is representative; `"shuffle"` is a plain fixed-seed shuffle; `"file"` keeps your order. |
| `family` | `default_family` | Function mapping an instance to its family, used by stratification. Default reads the leading letters+digits of the file name (`BR7_23.txt` → `BR7`). |
| `cost_aware` | `False` | Divide a strategy's impact by its mean wall time per execution: impact per second rather than per run. Only worth it when per-run cost really varies; with a *binding* time limit it does not (BR under `-t 30`: 1.05× across families). |
| `n_jobs` | `1` | Instances evaluated in parallel. Set to your core count, and make `batch` a multiple of it — `batch=25` with 8 jobs runs waves of 8, 8, 8, **1** and wastes 25 % of solver wall time. |
| `timeout` | `None` | Seconds per single execution, as a **safety net only**. Give the solver its own budget through its command line; a `TimeoutExpired` here aborts the whole run and loses the cache. Measured on BSG_CLP with `-t 30`: worst wall 33.3 s, so `timeout=75` is comfortable. |
| `parse` | last float | Function `stdout -> float`. The default takes the last float printed, which is fragile: a debug line appended after the score silently changes what is measured. |
| `seed` | `0` | Reproducibility of the simulations. |
| `verbose` | `False` | One progress line per iteration. |

### What comes back

```python
res = experiment_execution(...)
res.output            # the experiment's return value
res.likelihood        # degree of certainty (1.0 if run to completion)
res.runs              # executions actually performed
res.sequential_runs   # what a sequential run would have needed
res.speedup
res.history           # per-iteration log: runs, output, likelihood, selected...
```

`res.history` is the data behind every figure in `figures/`.

---

## Two rules you must follow

**1. The experiment function must be deterministic and side-effect free.**
It is re-executed once per predictive pass and once per simulation. No global
counters, no clock reads, no appending to outside lists. Only calls to the
T4exps API.

**2. `E_s(π)` must be a well-defined number.** If your solver is stochastic,
fold the random seed into the instance identifier rather than drawing fresh
noise on each call — see `examples/dummy_algo.py`. Otherwise a sequential run
and an incremental run evaluate different things, and "identical to sequential"
stops being testable. Beware that an *anytime* solver with a wall-clock limit is
not deterministic under load even with a fixed seed: on BSG_CLP with `-t 30`,
2–3 % of (strategy, instance) pairs changed by up to 0.25 pp between two runs
(CPU frequency scaling). Negligible for the decisions, fatal for bit-for-bit
reproducibility — compare engine variants by replaying against a stored matrix.

---

## Before a long run

Some practical advice learned the hard way, especially for benchmarks like BR
that ship grouped by family:

- **Instance order is handled for you — check the family detection.**
  Strategies always evaluate the prefix `instances[0:c]`, so the order matters.
  By default T4exps *stratifies*: it interleaves families so any prefix has all
  of them in proportion. Family is read from the file name (`BR7_23.txt` →
  `BR7`); if your naming differs, pass `family=lambda inst: ...`. Pass
  `instance_order="shuffle"` if you have no families, or `"file"` if you have
  already ordered the file yourself.
- **Set a `tolerance`.** Pick something just under the standard error of the
  *paired difference of means* at your instance count — and remember it scales
  with 1/√N: 0.0316 pp at N=200 became 0.0116 pp at N=1500 on BR. Without it
  the engine will spend most of its budget deciding ties.
- **Set a `confidence`** below 1.0 unless you truly need the exact sequential
  answer, or the run will not finish early no matter how certain it gets.
- **Try 200–300 instances first** to check the pipeline before committing to
  the full set.

```python
experiment_execution(
    experiment, "BR.txt",                 # any order; stratified automatically
    estimator="paired", n_jobs=8, timeout=75,   # the solver gets -t 30 itself
    nruns=15, batch=24, nsims=400,              # batch = multiple of n_jobs
    confidence=0.98, tolerance=0.0116,
    cost_aware=False,                           # binding time limit: cost is flat
    verbose=True,
)
```

- **Persist every evaluation as it happens.** `runs/monitor.py` has a
  `LoggingRunner` that appends one CSV row per execution and can `--resume` from
  it; the completed matrix then serves as an *oracle* against which any variant
  of the engine can be replayed in minutes without touching the solver
  (`runs/replay.py`). The whole Stage 3 campaign in [PLAN.md](PLAN.md) — the
  original engine, the fixes, and a 20-seed variability study — was done that way.

---

## Limitations

- Evaluations are assumed normal. Fine for solution quality, poor for runtimes,
  which are right-skewed — take logs first.
- The paired estimator assumes instance difficulty is *additive*: hard instances
  are hard for everyone by the same amount. When strategies are complementary
  (different ones win on different instance families) this degrades, and it
  degrades toward over-confidence. Use `estimator="independent"` if you suspect
  that regime.
- Strategy selection is greedy and one step ahead: impact first, then the
  swing `|L⁺ − L⁻|` as tie-break, optionally per second of CPU. It is a good
  heuristic, not an optimal allocation policy.
- Engine overhead grows with the number of strategies. On very fast solvers the
  machinery can cost more than it saves; measure before trusting the speed-up.

### Known defects (measured on BSG_CLP / BR1–BR15, September 2026)

Found while running the plan in [PLAN.md](PLAN.md); each one is reproducible
and the fixes exist as drop-in variants under `runs/` but are **not yet merged
into the package**.

1. **Likelihood denominator** (`engine.py`, `_run_simulations`/`run`). The
   likelihood is *matches / surviving simulations*; simulations that reach a
   strategy with no data abort and are dropped. A simulation wanders off the
   evaluated path exactly when it *disagrees* with the prediction, so the
   estimate is biased upward, and 1/1 = 100 % satisfies any `confidence`. On BR
   the original engine stopped 7 times out of 7 with ≥ 98 % reported confidence
   and the wrong γ (20 more seeds: 0/20 correct). Fix: count aborted simulations
   as non-matching (`HonestEngine` in `runs/replay.py`).
2. **Unequal depths** (`tests/test_unequal_depth.py`, strict xfail). The paired
   estimator takes a strategy's level from *all* its observations; a strategy
   evaluated deeper absorbs the difficulty of instances nobody else saw, and the
   posterior can invert (0 % for the truly better strategy at 40 vs 90 evals).
   Fix: levels from the shared block only, instance effects as residuals
   (`paired_k2`/`paired_k3` in `runs/estimators_fixed.py`).
3. **Dominant strategy** (`tests/test_dominant_strategy.py`, strict xfail). When
   almost every seen instance has `k=1`, the variance-component solve divides by
   `1 − mean(1/k) → 0`, `tau²` collapses to `1e-12` and the posterior becomes a
   point mass: 400/400 simulations agree on 15 data points. Fix: components from
   the shared block, residual by two-way ANOVA (`paired_k3`).
4. **The compass goes dark.** Impact is measured against the believed prefix of
   decisions; when the disagreeing simulations abort, survival → 0, depth → 0,
   every impact → 0, and the tie falls to code order — the engine hammers the
   first strategy in the sweep to completion. Fix: impute a prior total for
   strategies without data instead of aborting (`ImputingEngine`,
   `runs/replay.py --impute`).

With all four fixes the engine reached the reference configuration in 23/23
seeds at 2.6–3.3× fewer executions than the sequential sweep; the original
engine's 6–89× speed-ups occurred only in runs that stopped on the wrong answer.

---

## Repository layout

```
t4exps/
  core.py         Strategy, the re-execution context, best_strategy
  runner.py       evaluation, caching, parallelism
  estimators.py   paired and independent predictive models
  engine.py       the incremental speculative execution loop
  utils.py        sequential baseline, instance ordering, cartesian_product
examples/         dummy solver, factorial design, calibration, figures, BSG
tests/            43 tests + 2 strict xfails documenting defects 2 and 3
figures/          PDF + PNG, regenerate with examples/make_figures.py
runs/             BSG campaign: per-evaluation CSVs, the 21 745-row matrix used as
                  oracle, replay/ablation/variability tooling, status & close scripts
NOTES.md          design rationale and measured results
PLAN.md           the three-stage BSG plan with every measured result and defect
```

## Citing

If you use T4exps in academic work, please cite:

> I. Araya, A. Marchant, A. Espinoza. *T4exps: A Tool for Algorithm Experiments
> with Incremental Speculative Execution.* (in preparation)

## License

MIT — see [LICENSE](LICENSE).
