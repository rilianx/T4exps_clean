from __future__ import annotations

import itertools
from typing import Any, Callable, Sequence

from .core import Context, Strategy, set_context
from .runner import Runner


def cartesian_product(*iterables):
    return itertools.product(*iterables)


def sequential_execution(experiment, instances, *, n_jobs: int = 1, timeout=None,
                         parse=None):
    """Ground truth: every strategy on every instance, no speculation.

    This is the baseline the incremental engine must reproduce exactly.
    Returns (output, total_runs).
    """
    if isinstance(instances, str):
        with open(instances) as fh:
            instances = [ln.strip() for ln in fh if ln.strip()]

    runner = Runner(instances, n_jobs=n_jobs, timeout=timeout, parse=parse)

    class _Seq:
        n_instances = len(instances)
        tolerance = 0.0

        def ensure_minimum_data(self, s: Strategy):
            missing = len(instances) - runner.count(s)
            if missing > 0:
                runner.run(s, missing)

        def compare_observed(self, s1: Strategy, s2: Strategy):
            import numpy as np

            v1, v2 = runner.values(s1), runner.values(s2)
            return s1 if np.mean(v1) >= np.mean(v2) else s2

    ctx = Context(mode="predict", engine=_Seq())
    set_context(ctx)
    try:
        output = experiment()
    finally:
        set_context(None)
    return (output if output is not None else ctx.state()), runner.total_runs


# --------------------------------------------------------------------------- #
# Instance ordering
# --------------------------------------------------------------------------- #

import random
import re
from collections import OrderedDict, defaultdict

_FAMILY_RE = re.compile(r"^([A-Za-z]+\d*)")


def default_family(instance: str) -> str:
    """Family of an instance from its file name: 'BR7_23.txt' -> 'BR7'."""
    name = str(instance).replace("\\", "/").rsplit("/", 1)[-1]
    m = _FAMILY_RE.match(name)
    return m.group(1) if m else name


def stratify_instances(instances, family=default_family, seed: int = 0):
    """Interleave instance families so that EVERY prefix is representative.

    Strategies always evaluate the prefix instances[0:c] in order.  Benchmarks
    usually ship grouped by family (BR1..BR15, all of BR1 first); with the file
    order the first hundred evaluations would all come from the easiest,
    most homogeneous family -- biased means and an instance-effect estimate
    that has seen no diversity.  This shuffles within each family (fixed seed)
    and then deals the families out round-robin, proportionally to their size.
    """
    rng = random.Random(seed)
    groups = OrderedDict()
    for inst in instances:
        groups.setdefault(family(inst), []).append(inst)
    for g in groups.values():
        rng.shuffle(g)
    # proportional round-robin: at each step take from the family that is
    # furthest behind its fair share (largest remainder)
    total = len(instances)
    taken = defaultdict(int)
    out = []
    while len(out) < total:
        best, best_deficit = None, -1.0
        for fam, g in groups.items():
            if taken[fam] >= len(g):
                continue
            deficit = len(g) / total * (len(out) + 1) - taken[fam]
            if deficit > best_deficit:
                best, best_deficit = fam, deficit
        out.append(groups[best][taken[best]])
        taken[best] += 1
    return out


def shuffle_instances(instances, seed: int = 0):
    out = list(instances)
    random.Random(seed).shuffle(out)
    return out


def order_instances(instances, how: str = "stratified", family=default_family,
                    seed: int = 0):
    if how == "file":
        return list(instances)
    if how == "shuffle":
        return shuffle_instances(instances, seed)
    if how == "stratified":
        return stratify_instances(instances, family, seed)
    raise ValueError(f"unknown instance_order {how!r}; "
                     "use 'stratified', 'shuffle' or 'file'")
