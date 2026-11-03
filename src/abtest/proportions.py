"""Tests and intervals for click-through rates, computed from counts.

Every function takes plain integers (successes and trials per arm) and returns a frozen
dataclass. Nothing here touches pandas, so the same code runs on one test or, in a loop, on
thirty thousand.

Conventions: ``control`` is arm A with ``x_c`` clicks out of ``n_c`` impressions and
``treatment`` is arm B with ``x_t`` out of ``n_t``. Differences are treatment minus control,
lifts are treatment over control minus one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "ArmComparison",
    "Interval",
    "MultiArmResult",
    "ZTestResult",
    "LiftResult",
    "chi2_omnibus",
    "compare_arms",
    "multi_arm_test",
    "newcombe_diff_ci",
    "relative_lift",
    "two_proportion_ztest",
    "wilson_ci",
]


def _z(alpha: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return float(stats.norm.ppf(1 - alpha / 2))


def _check_counts(x: int, n: int) -> None:
    if n <= 0:
        raise ValueError(f"trials must be positive, got n={n}")
    if x < 0 or x > n:
        raise ValueError(f"successes must be in [0, n], got x={x}, n={n}")


@dataclass(frozen=True)
class Interval:
    """A point estimate with a two-sided confidence interval."""

    estimate: float
    lower: float
    upper: float
    confidence: float

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


@dataclass(frozen=True)
class ZTestResult:
    """Pooled two-proportion z-test of treatment vs control."""

    p_control: float
    p_treatment: float
    diff: float
    z: float
    p_value: float


@dataclass(frozen=True)
class LiftResult:
    """Relative lift (p_t / p_c - 1) with a delta-method interval."""

    lift: float
    se: float
    lower: float
    upper: float
    confidence: float


@dataclass(frozen=True)
class ArmComparison:
    """One treatment arm against the control arm."""

    arm: int
    ztest: ZTestResult
    diff_ci: Interval
    lift: LiftResult
    p_value_bonferroni: float


@dataclass(frozen=True)
class MultiArmResult:
    """Chi-square omnibus test across all arms plus every arm vs a control."""

    n_arms: int
    control: int
    chi2: float
    df: int
    p_value: float
    arm_rates: tuple[Interval, ...]
    comparisons: tuple[ArmComparison, ...]


def two_proportion_ztest(x_c: int, n_c: int, x_t: int, n_t: int) -> ZTestResult:
    r"""Two-sided pooled two-proportion z-test.

    .. math::
        \hat p = \frac{x_c + x_t}{n_c + n_t}, \qquad
        z = \frac{\hat p_t - \hat p_c}{\sqrt{\hat p (1 - \hat p)(1/n_c + 1/n_t)}}, \qquad
        p = 2\,\Phi(-|z|)

    Use it to ask whether two arms' click rates differ, when both arms have enough clicks
    and non-clicks (roughly 5 or more each) for the normal approximation to hold. The pooled
    variance is the variance under the null of equal rates, which is what makes this the
    classic test (it matches ``statsmodels.stats.proportion.proportions_ztest``). If every
    impression in both arms clicked, or none did, there is no variation: z is 0 and p is 1.
    """
    _check_counts(x_c, n_c)
    _check_counts(x_t, n_t)
    p_c, p_t = x_c / n_c, x_t / n_t
    pooled = (x_c + x_t) / (n_c + n_t)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    if se == 0:
        return ZTestResult(p_c, p_t, p_t - p_c, 0.0, 1.0)
    z = (p_t - p_c) / se
    p = float(min(1.0, 2 * stats.norm.sf(abs(z))))
    return ZTestResult(p_c, p_t, p_t - p_c, z, p)


def wilson_ci(x: int, n: int, alpha: float = 0.05) -> Interval:
    r"""Wilson score interval for one arm's rate.

    .. math::
        \frac{\hat p + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}
             {1 + \frac{z^2}{n}}

    Use it instead of the Wald interval (p +/- z*se) for any single click rate, especially
    small ones: it stays inside [0, 1] and keeps close to nominal coverage near 0, which is
    where headline CTRs live.
    """
    _check_counts(x, n)
    z = _z(alpha)
    p = x / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    lower = 0.0 if x == 0 else max(0.0, centre - half)
    upper = 1.0 if x == n else min(1.0, centre + half)
    return Interval(p, lower, upper, 1 - alpha)


def newcombe_diff_ci(x_c: int, n_c: int, x_t: int, n_t: int, alpha: float = 0.05) -> Interval:
    r"""Newcombe (hybrid Wilson score) interval for ``p_t - p_c``.

    With Wilson limits :math:`[l_c, u_c]` and :math:`[l_t, u_t]` and
    :math:`d = \hat p_t - \hat p_c`:

    .. math::
        L = d - \sqrt{(\hat p_t - l_t)^2 + (u_c - \hat p_c)^2}, \qquad
        U = d + \sqrt{(u_t - \hat p_t)^2 + (\hat p_c - l_c)^2}

    Use it for the absolute difference in click rates. Unlike the Wald interval it behaves
    well with small rates or unequal arms, and it is Newcombe's (1998) method 10, the one
    ``statsmodels`` calls ``method="newcomb"``.
    """
    wc = wilson_ci(x_c, n_c, alpha)
    wt = wilson_ci(x_t, n_t, alpha)
    d = wt.estimate - wc.estimate
    lower = d - math.sqrt((wt.estimate - wt.lower) ** 2 + (wc.upper - wc.estimate) ** 2)
    upper = d + math.sqrt((wt.upper - wt.estimate) ** 2 + (wc.estimate - wc.lower) ** 2)
    return Interval(d, lower, upper, 1 - alpha)


def relative_lift(x_c: int, n_c: int, x_t: int, n_t: int, alpha: float = 0.05) -> LiftResult:
    r"""Relative lift ``p_t / p_c - 1`` with a delta-method interval.

    With :math:`r = \hat p_t / \hat p_c` and independent arms,

    .. math::
        \mathrm{SE}(r) \approx r \sqrt{\frac{1 - \hat p_t}{x_t} + \frac{1 - \hat p_c}{x_c}},
        \qquad \text{lift} \pm z\,\mathrm{SE}

    (using :math:`\mathrm{Var}(\hat p)/\hat p^2 = (1-\hat p)/(n \hat p)`). Use it to report
    "B beat A by 12%" with an honest interval; it needs clicks in both arms and is
    symmetric by construction, so for small counts prefer a log-ratio or bootstrap interval.
    If the control has no clicks the lift is undefined and every field is NaN; if only the
    treatment has none, the lift is -1 with NaN standard error.
    """
    _check_counts(x_c, n_c)
    _check_counts(x_t, n_t)
    conf = 1 - alpha
    z = _z(alpha)
    if x_c == 0:
        nan = float("nan")
        return LiftResult(nan, nan, nan, nan, conf)
    p_c, p_t = x_c / n_c, x_t / n_t
    ratio = p_t / p_c
    if x_t == 0:
        return LiftResult(-1.0, float("nan"), float("nan"), float("nan"), conf)
    se = ratio * math.sqrt((1 - p_t) / x_t + (1 - p_c) / x_c)
    lift = ratio - 1
    return LiftResult(lift, se, lift - z * se, lift + z * se, conf)


def compare_arms(
    x_c: int, n_c: int, x_t: int, n_t: int, alpha: float = 0.05
) -> tuple[ZTestResult, Interval, LiftResult]:
    """z-test, Newcombe interval and relative lift for one pair of arms, in one call.

    Use it when you want the full two-arm readout; it just calls the three functions above.
    """
    return (
        two_proportion_ztest(x_c, n_c, x_t, n_t),
        newcombe_diff_ci(x_c, n_c, x_t, n_t, alpha),
        relative_lift(x_c, n_c, x_t, n_t, alpha),
    )


def chi2_omnibus(clicks: Sequence[int], impressions: Sequence[int]) -> tuple[float, int, float]:
    r"""Pearson chi-square test of independence on the 2 x k table of clicks and non-clicks.

    .. math::
        E_{ij} = \frac{R_i C_j}{N}, \qquad
        \chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \qquad df = k - 1

    Use it as the single "do any of these k arms differ?" test before looking at pairwise
    comparisons. No continuity correction (it matches ``scipy.stats.chi2_contingency`` with
    ``correction=False``). If no impression clicked, or all did, the table has no variation and
    the result is chi2 = 0, p = 1. Returns ``(chi2, df, p_value)``.
    """
    x = np.asarray(clicks, dtype=float)
    n = np.asarray(impressions, dtype=float)
    if x.ndim != 1 or x.shape != n.shape:
        raise ValueError("clicks and impressions must be 1-d sequences of the same length")
    k = x.size
    if k < 2:
        raise ValueError(f"need at least 2 arms, got {k}")
    for xi, ni in zip(x, n, strict=True):
        _check_counts(xi, ni)
    df = k - 1
    total_x, total_n = x.sum(), n.sum()
    if total_x == 0 or total_x == total_n:
        return 0.0, df, 1.0
    observed = np.vstack([x, n - x])
    expected = np.vstack([n * total_x / total_n, n * (total_n - total_x) / total_n])
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    return chi2, df, float(stats.chi2.sf(chi2, df))


def multi_arm_test(
    clicks: Sequence[int],
    impressions: Sequence[int],
    control: int = 0,
    alpha: float = 0.05,
) -> MultiArmResult:
    """Chi-square omnibus test across k arms, then every other arm against ``control``.

    Each comparison carries a pooled z-test, a Newcombe interval and a delta-method lift, all
    at level ``alpha``, plus a Bonferroni-adjusted p-value, ``min(1, (k - 1) * p)``, for the
    k - 1 comparisons. ``control`` is a 0-based index into the arms.

    Use it for a multi-headline test: read the omnibus p first, then the adjusted pairwise
    p-values to see which arms beat the control.
    """
    k = len(clicks)
    if not 0 <= control < k:
        raise ValueError(f"control must index one of the {k} arms, got {control}")
    chi2, df, p = chi2_omnibus(clicks, impressions)
    rates = tuple(
        wilson_ci(int(x), int(n), alpha) for x, n in zip(clicks, impressions, strict=True)
    )
    x_c, n_c = int(clicks[control]), int(impressions[control])
    comps = []
    for i in range(k):
        if i == control:
            continue
        zt, ci, lift = compare_arms(x_c, n_c, int(clicks[i]), int(impressions[i]), alpha)
        comps.append(ArmComparison(i, zt, ci, lift, min(1.0, (k - 1) * zt.p_value)))
    return MultiArmResult(k, control, chi2, df, p, rates, tuple(comps))
