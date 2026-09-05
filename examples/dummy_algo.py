"""A dummy solver: A(pi) = -(a^2 + 2b^2) + b_pi + eps.

`sigma`      run-to-run noise (the paper's X ~ N(0, sigma^2))
`sigma_inst` instance difficulty, identical for every strategy on that instance

Two properties worth keeping:

1. It is DETERMINISTIC in (instance, theta, rep).  E_s(pi) must be a
   well-defined number, otherwise a sequential run and an incremental run of
   the same experiment evaluate different things and "identical to sequential"
   becomes untestable.  Noise comes from hashing, not from a global RNG.
2. sigma_inst = 0 reproduces the dummy of the paper exactly (pure run noise,
   nothing for the paired estimator to exploit).  sigma_inst > 0 behaves like
   a real benchmark, where instances differ far more than strategies do.

Expected optimum: a = b = 0.
"""

from __future__ import annotations

import hashlib
import math

_REP = 0


def reseed(rep: int) -> None:
    """Select a replication: same rep -> byte-identical evaluations."""
    global _REP
    _REP = int(rep)


def _gauss(*parts) -> float:
    """A standard normal deterministically derived from its arguments."""
    h = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=16).digest()
    u1 = max(int.from_bytes(h[:8], "big") / 2**64, 1e-12)
    u2 = int.from_bytes(h[8:], "big") / 2**64
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def dummy_algo(instance, a=0.0, b=0.0, sigma=1.0, sigma_inst=0.0):
    base = -(a**2 + 2 * b**2)
    inst = sigma_inst * _gauss("inst", instance, _REP) if sigma_inst else 0.0
    eps = sigma * _gauss("run", instance, a, b, _REP)
    return base + inst + eps
