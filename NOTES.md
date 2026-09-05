# Design notes

Rationale and measured results behind T4exps. For installation and usage see
[README.md](README.md).

## What is different from the FWK4EXP draft

Four changes, in decreasing order of how much they matter.

**1. Instance-paired prediction (`estimator="paired"`, default).**
The draft simulates each strategy's remaining evaluations independently. But
strategies share the instance set, and on a real benchmark most of the spread
is instance difficulty, not strategy quality — an effect common to both sides
of every comparison. T4exps fits

    E_s(pi) = mu_s + b_pi + eps

pools the instance effects `b_pi` across all strategies with empirical-Bayes
shrinkage, and uses **the same** `b_pi` draw for every strategy within a
simulation, so it cancels in the comparison. The uncertainty that actually
drives decisions collapses from `Var(E)` to `Var(eps)`.

Measured on the dummy design (15 instances observed, 60 total, predicted
spread of the difference between two strategies):

| instance heterogeneity | independent | paired |
|---|---|---|
| sigma_inst = 0 | 0.667 | 0.637 |
| sigma_inst = 3 | 0.879 | 0.637 |
| sigma_inst = 8 | 1.829 | 0.637 |

The paired estimator is *immune* to instance heterogeneity. The independent
one degrades linearly with it. Set `estimator="independent"` to reproduce the
draft's behaviour — the two are drop-in interchangeable, which is what makes
this an ablation you can put in the paper.

**2. Conjugate posteriors instead of Metropolis-Hastings.**
For a normal with unknown mean and variance the Normal-Inverse-Gamma posterior
is available in closed form. Exact draws, no chain, no burn-in, and — the real
bug it removes — no hard-coded `Uniform[0.001, 10]` prior on sigma that can
silently exclude the true value (in the draft, three of the four tested sigmas
lay outside its support). Roughly three orders of magnitude cheaper, which is
what lets `nsims` be large enough to matter.

**3. Calibration is measured, not asserted.**
`examples/calibration.py` replicates the design many times, bins every reported
likelihood, and reports how often the answer was actually right. Sample output
(10 reps, 9 strategies, 40 instances, sigma=5, sigma_inst=4):

```
=== paired ===
    [0.70,0.85)  n=  25   empirical = 96.0%
    [0.85,0.95)  n=  40   empirical = 92.5%
    [0.95,1.01)  n=  48   empirical = 95.8%
  runs to first reach 90% and be right: median 190 of 360 sequential (1.9x)
  final answer identical to sequential: 10/10
```

The `independent` estimator comes out *conservative* — it reports 0.70–0.85
where it is empirically right 93–100% of the time, and pays for that timidity
in wasted runs.

**4. Early stopping and tolerance.**
`confidence=0.95` stops as soon as the reported certainty crosses the
threshold — without it, "incremental" saves time-to-answer but not a single
run, since exactness demands evaluating everything. `tolerance=t` declares
differences below `t` a tie, so the engine stops spending thousands of runs
separating two strategies that differ by less than the standard error. (In the
draft's Table 4, `b=2.0` and `b=4.0` differ by 0.002 with sd 0.95.)

**5. Instance ordering and strategy selection.**
Strategies evaluate the prefix `instances[0:c]`, so the order of the file is
part of the model. Benchmarks ship grouped by family (all of BR1, then BR2...),
which makes early prefixes biased and, worse, hides the instance heterogeneity
from the paired estimator exactly when it is being fitted. The engine now
stratifies by default (`utils.stratify_instances`): proportional round-robin
across families, each family shuffled with a fixed seed, so every prefix is a
miniature of the whole set.

The impact indicator `I(s) = 1 − min(L⁻,L⁺)/L` saturates at 1.0 for any
strategy that can push the probable state to zero — early on, most of them —
and the draft breaks the tie by order of appearance, which in a sequential
parameter sweep systematically favours the first parameter. Ties are now broken
by the swing `|L⁺ − L⁻|`, and with `cost_aware=True` both quantities are
divided by the strategy's mean wall time, giving impact per second.

See [README.md](README.md) for the API, usage contract and limitations.
