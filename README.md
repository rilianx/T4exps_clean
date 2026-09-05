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
python -m pytest tests -q          # 25 tests, ~12 s
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
S = Strategy('bsg', './BSG_CLP {INSTANCE} -a {a} -b {b}', theta={'a': 0.0, 'b': 0.5})
S2 = S.with_theta(a=4.0)            # a copy with one parameter changed
```

`{INSTANCE}` is replaced by the instance, `{name}` by the corresponding entry of
`theta`. A Python callable `f(instance, **theta) -> float` works too, which is
what the dummy example uses.

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
python examples/bsg.py --bin ../Metasolver/BSG_CLP --instances BR.txt --jobs 8
```

`examples/bsg.py` writes `bsg_history.csv` with likelihood against number of
executions, ready to plot.

---

## Options

Passed as keyword arguments to `experiment_execution`.

| option | default | what it does |
|---|---|---|
| `estimator` | `"paired"` | `"paired"` cancels instance difficulty between strategies; `"independent"` reproduces the classic per-strategy model. See [NOTES.md](NOTES.md). |
| `confidence` | `1.0` | Stop as soon as the reported likelihood reaches this. **Without it the run only saves time-to-answer, not executions** — exactness requires evaluating everything. |
| `tolerance` | `0.0` | Differences below this count as a tie. Keeps the engine from burning thousands of runs separating strategies that differ by less than the standard error. |
| `nruns` | `10` | Executions given to a strategy the first time it appears. |
| `batch` | `10` | Executions added each iteration to the selected strategy. |
| `nsims` | `400` | Monte Carlo simulations per likelihood estimate. |
| `nsims_impact` | `nsims/4` | Cheaper budget for the per-strategy impact estimates. |
| `percentile` | `0.05` | Optimistic/pessimistic quantile used to measure impact. |
| `instance_order` | `"stratified"` | How the instance file is ordered before use. `"stratified"` interleaves families so every prefix is representative; `"shuffle"` is a plain fixed-seed shuffle; `"file"` keeps your order. |
| `family` | `default_family` | Function mapping an instance to its family, used by stratification. Default reads the leading letters+digits of the file name (`BR7_23.txt` → `BR7`). |
| `cost_aware` | `False` | Divide a strategy's impact by its mean wall time per execution: impact per second rather than per run. Useful with time limits, where big instances cost far more than small ones. |
| `n_jobs` | `1` | Instances evaluated in parallel. Set to your core count. |
| `timeout` | `None` | Seconds per single execution. |
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
stops being testable.

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
- **Set a `tolerance`.** Pick something just under the standard error of your
  measure. Without it the engine will spend most of its budget deciding ties.
- **Set a `confidence`** below 1.0 unless you truly need the exact sequential
  answer, or the run will not finish early no matter how certain it gets.
- **Try 200–300 instances first** to check the pipeline before committing to
  the full set.

```python
experiment_execution(
    experiment, "BR.txt",                 # any order; stratified automatically
    estimator="paired", n_jobs=8, timeout=30,
    nruns=30, batch=50, nsims=300,
    confidence=0.98, tolerance=0.0005,
    cost_aware=True,                      # 30 s time limit: cost varies a lot
    verbose=True,
)
```

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
tests/            25 tests, including "must equal the sequential result"
figures/          PDF + PNG, regenerate with examples/make_figures.py
NOTES.md          design rationale and measured results
```

## Citing

If you use T4exps in academic work, please cite:

> I. Araya, A. Marchant, A. Espinoza. *T4exps: A Tool for Algorithm Experiments
> with Incremental Speculative Execution.* (in preparation)

## License

MIT — see [LICENSE](LICENSE).
