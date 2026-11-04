r"""Data thinning: split one test's counts into two independent half-tests.

A test that picks its winner and measures the winner's lift on the same data overstates
that lift (the winner's curse). The honest check is a replication, and data thinning makes
one out of the data already collected: send each impression to half A or half B at random,
pick the winner on A, and measure it on B, which played no part in the choice.

:func:`split_binomial` does the split for every arm at once. Of an arm's ``n`` impressions,
``n_A`` go to half A (``frac * n``, rounded up or down at random so that
:math:`E[n_A] = \text{frac}\cdot n` exactly), and the clicks that land in half A are the
clicks in a uniformly random subset of size ``n_A``:

.. math::
    x_A \mid x, n_A \sim \text{Hypergeometric}(N = n,\ K = x,\ \text{draws} = n_A),
    \qquad x_B = x - x_A.

Why the halves are independent
------------------------------
If the arm's clicks are binomial, :math:`x \sim \text{Bin}(n, p)`, the impressions are
exchangeable Bernoulli(p) trials, and a random subset of them is again a set of
Bernoulli(p) trials. So :math:`x_A \sim \text{Bin}(n_A, p)` and
:math:`x_B \sim \text{Bin}(n - n_A, p)`, *independently*, given :math:`n_A`. Each half is
a smaller copy of the same test with the same true click rate.

That rests on the binomial model. If an arm has its own extra-binomial shock (its realized
click rate drifts away from the headline's long-run rate because of the hour it ran, a
front-page slot, a share that went viral), the shock is in every impression and therefore
in both halves. The halves are then positively correlated and half B is not a fresh draw:
selection on that shared shock is invisible to the split. The archive's natural A/A pairs
come out significant about 13% of the time instead of 5%, so some such shock exists;
thinning measures the binomial part of the winner's curse, which makes it a lower bound.

:func:`contrasts` and :func:`replicate_winners` are vectorized versions of
:func:`abtest.bayes.arm_contrasts` and :func:`abtest.bayes.best_arm_contrast` over many
tests at once (arms in flat arrays, tagged with a test index), so a thinning analysis can
be repeated over seeds on tens of thousands of tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

ArrayLike = int | Sequence[int] | np.ndarray


@dataclass(frozen=True)
class Split:
    """Impressions and clicks of half A and half B, same shape as the input."""

    n_a: np.ndarray
    x_a: np.ndarray
    n_b: np.ndarray
    x_b: np.ndarray


def split_binomial(
    n: ArrayLike, x: ArrayLike, frac: float = 0.5, rng: np.random.Generator | int | None = None
) -> Split:
    """Split impressions ``n`` and clicks ``x`` into half A (share ``frac``) and half B.

    ``n`` and ``x`` are integers or equal-shape integer arrays (one entry per arm), with
    ``0 <= x <= n``. ``rng`` is a :class:`numpy.random.Generator` or a seed. Clicks are
    conserved exactly: ``x_a + x_b == x`` and ``n_a + n_b == n``.
    """
    if not 0 < frac < 1:
        raise ValueError("frac must be strictly between 0 and 1")
    n = np.asarray(n)
    x = np.asarray(x)
    if n.shape != x.shape:
        raise ValueError("n and x must have the same shape")
    if not (np.issubdtype(n.dtype, np.integer) and np.issubdtype(x.dtype, np.integer)):
        raise ValueError("n and x must be integer counts")
    if (x < 0).any() or (x > n).any():
        raise ValueError("need 0 <= x <= n")
    rng = np.random.default_rng(rng)
    target = frac * n
    base = np.floor(target)
    n_a = (base + (rng.random(n.shape) < target - base)).astype(np.int64)
    x_a = rng.hypergeometric(x, n - x, n_a) if n.size else np.zeros_like(n, dtype=np.int64)
    x_a = np.asarray(x_a, dtype=np.int64)
    return Split(n_a, x_a, n.astype(np.int64) - n_a, x.astype(np.int64) - x_a)


# --- many tests at once ----------------------------------------------------------------------


def _test_index(test: Sequence[int] | np.ndarray, size: int) -> np.ndarray:
    t = np.asarray(test)
    if t.shape != (size,):
        raise ValueError("test must have one entry per arm")
    if size and (np.diff(t) < 0).any():
        raise ValueError("arms must be sorted by test")
    return t


def contrasts(
    test: Sequence[int] | np.ndarray, x: ArrayLike, n: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Log odds ratio and standard error of every arm against the pooled rest of its test.

    Same estimator as :func:`abtest.bayes.arm_contrasts` (0.5 added to all four cells of
    an arm's 2x2 table when any cell is zero). ``test`` labels each arm's test and must be
    sorted. Arms in a test with only one arm get NaN.
    """
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    t = _test_index(test, x.size)
    _, inv, counts = np.unique(t, return_inverse=True, return_counts=True)
    tx = np.bincount(inv, x)[inv]
    tn = np.bincount(inv, n)[inv]
    cells = np.stack([x, n - x, tx - x, (tn - n) - (tx - x)])
    cells = np.where((cells == 0).any(axis=0), cells + 0.5, cells)
    a, b, c, e = cells
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.log(a / b) - np.log(c / e)
        se = np.sqrt((1 / cells).sum(axis=0))
    alone = counts[inv] < 2
    return np.where(alone, np.nan, d), np.where(alone, np.nan, se)


def best_arms(test: Sequence[int] | np.ndarray, x: ArrayLike, n: ArrayLike) -> np.ndarray:
    """Flat index of the highest-CTR arm of each test (ties to the first; arms with n = 0 skipped).

    One entry per distinct test, in sorted test order; -1 for a test with no arm with
    impressions.
    """
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    t = _test_index(test, x.size)
    rate = np.where(n > 0, x / np.where(n > 0, n, 1), -np.inf)
    order = np.lexsort((np.arange(x.size), -rate, t))
    first = order[np.r_[True, t[order][1:] != t[order][:-1]]] if x.size else order
    return np.where(np.isfinite(rate[first]), first, -1).astype(np.int64)


@dataclass(frozen=True)
class Replication:
    """One thinning split of many tests: the winner picked on half A, measured on half B.

    Per test (sorted test order): ``best`` is the flat index of the arm picked on A,
    ``d_a``/``se_a`` its log OR vs the rest on A (the naive lift), ``d_b``/``se_b`` the same
    arm's log OR on B (the replication), ``still_best`` whether it also has the highest CTR
    on B, ``n_arms`` the arms in the test. Per arm: ``d_a_all``/``se_a_all`` every arm's
    contrast on half A, for fitting a prior without using half B.
    """

    tests: np.ndarray
    best: np.ndarray
    n_arms: np.ndarray
    d_a: np.ndarray
    se_a: np.ndarray
    d_b: np.ndarray
    se_b: np.ndarray
    still_best: np.ndarray
    d_a_all: np.ndarray
    se_a_all: np.ndarray
    split: Split


def replicate_winners(
    test: Sequence[int] | np.ndarray,
    x: ArrayLike,
    n: ArrayLike,
    rng: np.random.Generator | int | None = None,
    frac: float = 0.5,
) -> Replication:
    """Split every arm, pick each test's best arm on half A, and measure it on half B.

    Every test needs 2 or more arms; arms are in flat arrays sorted by ``test``.
    """
    x = np.asarray(x)
    n = np.asarray(n)
    t = _test_index(test, x.size)
    tests, inv, counts = np.unique(t, return_inverse=True, return_counts=True)
    if (counts < 2).any():
        raise ValueError("every test needs 2 or more arms")
    s = split_binomial(n, x, frac, rng)
    d_a, se_a = contrasts(t, s.x_a, s.n_a)
    d_b, se_b = contrasts(t, s.x_b, s.n_b)
    best = best_arms(t, s.x_a, s.n_a)
    best_b = best_arms(t, s.x_b, s.n_b)
    if (best < 0).any():
        raise ValueError("a test has no impressions in half A")
    return Replication(
        tests=tests,
        best=best,
        n_arms=counts,
        d_a=d_a[best],
        se_a=se_a[best],
        d_b=d_b[best],
        se_b=se_b[best],
        still_best=best == best_b,
        d_a_all=d_a,
        se_a_all=se_a,
        split=s,
    )
