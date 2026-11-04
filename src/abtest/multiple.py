"""Multiple-testing corrections, inside one test and across a portfolio of tests.

A test with k headlines makes k - 1 comparisons against its control, and an archive of
thirty thousand tests makes thirty thousand omnibus tests. At ``alpha = 0.05`` both
produce "winners" by chance alone. This module adjusts a vector of p-values for that, at
two levels:

* **within a test**: every arm against the control, adjusted for the k - 1 comparisons
  (:func:`within_test`, Holm by default, which controls the family-wise error rate);
* **across the portfolio**: one omnibus p-value per test, adjusted across all tests
  (:func:`portfolio`, Benjamini-Hochberg by default, which controls the false discovery
  rate).

Everything takes plain sequences or numpy arrays and returns frozen dataclasses. Adjusted
p-values match ``statsmodels.stats.multitest.multipletests`` with ``method`` ``bonferroni``,
``holm``, ``fdr_bh`` and ``fdr_by``.

Notation: ``m`` p-values, sorted ascending as :math:`p_{(1)} \\le \\dots \\le p_{(m)}`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from abtest.proportions import chi2_omnibus, two_proportion_ztest

__all__ = [
    "METHODS",
    "Adjustment",
    "PortfolioResult",
    "WinnerSurvival",
    "WithinTestResult",
    "adjust",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "bonferroni",
    "holm",
    "portfolio",
    "storey_pi0",
    "winner_survival",
    "within_test",
]

Method = Literal["bonferroni", "holm", "bh", "by"]
METHODS: tuple[str, ...] = ("bonferroni", "holm", "bh", "by")


def _pvalues(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be a 1-d sequence")
    if np.isnan(p).any():
        raise ValueError("p_values must not contain NaN")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("p_values must be in [0, 1]")
    return p


def _check_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def bonferroni(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    r"""Bonferroni-adjusted p-values, :math:`\tilde p_i = \min(1, m\,p_i)`.

    Controls the family-wise error rate (the chance of any false rejection) under any
    dependence between the tests. It is the bluntest correction: Holm rejects everything
    Bonferroni rejects, and sometimes more.
    """
    p = _pvalues(p_values)
    return np.minimum(1.0, p.size * p)


def holm(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    r"""Holm (1979) step-down adjusted p-values.

    .. math::
        \tilde p_{(i)} = \max_{j \le i} \min\{1,\ (m - j + 1)\, p_{(j)}\}

    Controls the family-wise error rate under any dependence, like Bonferroni, but is
    uniformly more powerful. Use it when a single false "winner" is costly, such as the
    k - 1 arm-vs-control comparisons inside one test.
    """
    p = _pvalues(p_values)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="mergesort")
    factors = m - np.arange(m)
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(factors * p[order]))
    out = np.empty(m)
    out[order] = adj_sorted
    return out


def _step_up(p: np.ndarray, scale: float) -> np.ndarray:
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="mergesort")
    ranks = np.arange(1, m + 1)
    raw = scale * m * p[order] / ranks
    adj_sorted = np.minimum(1.0, np.minimum.accumulate(raw[::-1])[::-1])
    out = np.empty(m)
    out[order] = adj_sorted
    return out


def benjamini_hochberg(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    r"""Benjamini-Hochberg (1995) step-up adjusted p-values (often called q-values).

    .. math::
        \tilde p_{(i)} = \min_{j \ge i} \min\left\{1,\ \frac{m\, p_{(j)}}{j}\right\}

    Rejecting :math:`\tilde p_i \le \alpha` keeps the false discovery rate (the expected
    share of false winners among the declared ones) at most :math:`\pi_0 \alpha \le \alpha`
    for independent or positively dependent p-values. Use it across a portfolio of tests,
    where some false winners are acceptable as long as most winners are real.
    """
    return _step_up(_pvalues(p_values), 1.0)


def benjamini_yekutieli(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    r"""Benjamini-Yekutieli (2001) adjusted p-values, BH made safe for any dependence.

    .. math::
        c(m) = \sum_{j=1}^{m} \frac{1}{j}, \qquad
        \tilde p_{(i)} = \min_{j \ge i} \min\left\{1,\ \frac{c(m)\, m\, p_{(j)}}{j}\right\}

    Controls the false discovery rate under arbitrary dependence at the cost of a factor of
    about :math:`\ln m + 0.58` (roughly 11 for 30,000 tests). Use it as a conservative
    check on BH when tests may be negatively correlated.
    """
    p = _pvalues(p_values)
    c_m = float(np.sum(1.0 / np.arange(1, p.size + 1))) if p.size else 1.0
    return _step_up(p, c_m)


_ADJUSTERS = {
    "bonferroni": bonferroni,
    "holm": holm,
    "bh": benjamini_hochberg,
    "by": benjamini_yekutieli,
}


@dataclass(frozen=True)
class Adjustment:
    """Raw and adjusted p-values for one family, with rejections at ``alpha``."""

    method: str
    alpha: float
    p_values: tuple[float, ...]
    p_adjusted: tuple[float, ...]
    reject: tuple[bool, ...]

    @property
    def n_rejected(self) -> int:
        return sum(self.reject)


def adjust(
    p_values: Sequence[float] | np.ndarray, method: Method = "bh", alpha: float = 0.05
) -> Adjustment:
    """Adjust a family of p-values and reject those with :math:`\\tilde p_i \\le \\alpha`.

    ``method`` is one of ``"bonferroni"``, ``"holm"`` (family-wise error rate), ``"bh"`` or
    ``"by"`` (false discovery rate). The order of the input is kept.
    """
    _check_alpha(alpha)
    if method not in _ADJUSTERS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    p = _pvalues(p_values)
    adj = _ADJUSTERS[method](p)
    return Adjustment(
        method,
        alpha,
        tuple(float(v) for v in p),
        tuple(float(v) for v in adj),
        tuple(bool(v) for v in adj <= alpha),
    )


def storey_pi0(p_values: Sequence[float] | np.ndarray, lambda_: float = 0.5) -> float:
    r"""Storey's (2002) estimate of :math:`\pi_0`, the share of true nulls.

    .. math::
        \hat\pi_0(\lambda) = \min\left\{1,\
            \frac{\#\{i : p_i > \lambda\}}{m\,(1 - \lambda)}\right\}

    Null p-values are uniform, so the ones above :math:`\lambda` are almost all nulls and
    their density, scaled up to the whole interval, estimates how many nulls there are.
    Real effects pile up near 0, so the estimate is biased slightly upward (conservative).
    Use it to answer "what share of these tests had no real effect at all?"; with
    :math:`\hat\pi_0 \alpha` it also tells you the FDR that BH actually delivers.
    """
    if not 0 <= lambda_ < 1:
        raise ValueError(f"lambda_ must be in [0, 1), got {lambda_}")
    p = _pvalues(p_values)
    if p.size == 0:
        raise ValueError("need at least one p-value")
    return float(min(1.0, np.sum(p > lambda_) / (p.size * (1 - lambda_))))


@dataclass(frozen=True)
class WithinTestResult:
    """Every arm against the control in one test, adjusted for the k - 1 comparisons.

    ``arms``, ``diffs`` and the p-value tuples are aligned and skip the control.
    """

    control: int
    arms: tuple[int, ...]
    diffs: tuple[float, ...]
    adjustment: Adjustment


def within_test(
    clicks: Sequence[int],
    impressions: Sequence[int],
    control: int = 0,
    method: Method = "holm",
    alpha: float = 0.05,
) -> WithinTestResult:
    """Pooled z-test of each arm against ``control``, adjusted across the k - 1 comparisons.

    ``control`` is a 0-based index. Each comparison uses
    :func:`abtest.proportions.two_proportion_ztest`; the family is the k - 1 two-sided
    p-values. Holm is the default because inside one test you usually ship one arm, and a
    single false winner is the error that matters.
    """
    k = len(clicks)
    if len(impressions) != k:
        raise ValueError("clicks and impressions must have the same length")
    if k < 2:
        raise ValueError(f"need at least 2 arms, got {k}")
    if not 0 <= control < k:
        raise ValueError(f"control must index one of the {k} arms, got {control}")
    x_c, n_c = int(clicks[control]), int(impressions[control])
    arms, diffs, ps = [], [], []
    for i in range(k):
        if i == control:
            continue
        z = two_proportion_ztest(x_c, n_c, int(clicks[i]), int(impressions[i]))
        arms.append(i)
        diffs.append(z.diff)
        ps.append(z.p_value)
    return WithinTestResult(control, tuple(arms), tuple(diffs), adjust(ps, method, alpha))


@dataclass(frozen=True)
class PortfolioResult:
    """One omnibus p-value per test, adjusted across the whole portfolio."""

    adjustment: Adjustment
    pi0: float


def portfolio(
    tests: Iterable[tuple[Sequence[int], Sequence[int]]],
    method: Method = "bh",
    alpha: float = 0.05,
    lambda_: float = 0.5,
) -> PortfolioResult:
    """Chi-square omnibus p-value for each ``(clicks, impressions)`` test, adjusted across all.

    The family is one p-value per test from :func:`abtest.proportions.chi2_omnibus` ("do any
    of this test's arms differ?"). BH is the default: across thousands of tests, the useful
    guarantee is that most declared winners are real, not that none is false. ``pi0`` is
    :func:`storey_pi0` on the same p-values. The order of ``tests`` is kept.
    """
    ps = [chi2_omnibus(clicks, imps)[2] for clicks, imps in tests]
    if not ps:
        raise ValueError("need at least one test")
    return PortfolioResult(adjust(ps, method, alpha), storey_pi0(ps, lambda_))


@dataclass(frozen=True)
class WinnerSurvival:
    """How many declared winners survive each correction.

    ``survive_within``: the winner beats every other arm in its test, by a positive
    difference, after Holm across its k - 1 comparisons. ``survive_portfolio``: the test's
    omnibus p-value survives BH across every test passed in. ``survive_both`` needs both.
    """

    n_tests: int
    n_with_winner: int
    survive_within: int
    survive_portfolio: int
    survive_both: int
    pi0: float
    within_method: str
    portfolio_method: str
    alpha: float


def winner_survival(
    tests: Iterable[tuple[Sequence[int], Sequence[int], int | None]],
    alpha: float = 0.05,
    within_method: Method = "holm",
    portfolio_method: Method = "bh",
) -> WinnerSurvival:
    """Count the declared winners that survive multiple-testing corrections.

    Each item is ``(clicks, impressions, winner)``, where ``winner`` is the 0-based index of
    the declared winning arm, or ``None`` when no winner was declared. Tests with fewer than
    two arms are skipped.

    Upworthy has no designated control arm, so inside a test the declared winner is the
    reference: it is compared with each of the other k - 1 arms, those p-values are
    adjusted with ``within_method``, and the winner survives when every comparison is
    rejected and every other arm has a lower click rate. In other words the winner beat
    the whole field, not just the weakest arm. Across the portfolio, every multi-arm test
    contributes its omnibus p-value (tests without a winner too, since they are part of the
    family that was searched), adjusted with ``portfolio_method``.
    """
    _check_alpha(alpha)
    rows = [(list(c), list(n), w) for c, n, w in tests if len(c) >= 2]
    if not rows:
        raise ValueError("need at least one test with 2 or more arms")
    port = portfolio(((c, n) for c, n, _ in rows), portfolio_method, alpha)
    n_with = within = port_ok = both = 0
    for (clicks, imps, winner), port_reject in zip(rows, port.adjustment.reject, strict=True):
        if winner is None:
            continue
        n_with += 1
        res = within_test(clicks, imps, control=winner, method=within_method, alpha=alpha)
        beats_all = all(res.adjustment.reject) and all(d < 0 for d in res.diffs)
        within += beats_all
        port_ok += port_reject
        both += beats_all and port_reject
    return WinnerSurvival(
        len(rows),
        n_with,
        within,
        port_ok,
        both,
        port.pi0,
        within_method,
        portfolio_method,
        alpha,
    )
