r"""Power and sample size for comparing two click rates.

Everything uses Cohen's arcsine effect size for proportions,

.. math::
    h = 2 \arcsin\sqrt{p_1} - 2 \arcsin\sqrt{p_0},

under which the test statistic is approximately normal with variance 1 per observation
whatever the baseline rate. For arms of size :math:`n_0` and :math:`n_1`, with
:math:`n_\text{eff} = (1/n_0 + 1/n_1)^{-1}` and :math:`z = \Phi^{-1}(1 - \alpha/2)`, the
two-sided power is

.. math::
    1 - \beta = \Phi\left(|h|\sqrt{n_\text{eff}} - z\right)
              + \Phi\left(-|h|\sqrt{n_\text{eff}} - z\right).

This is exactly ``statsmodels.stats.power.NormalIndPower`` fed with
``statsmodels.stats.proportion.proportion_effectsize``, and the functions here solve it
numerically (rather than with the closed form that drops the second term) so the numbers
match statsmodels to the decimal.

With k arms each compared with one control, the family has k - 1 comparisons and every
function takes ``n_arms`` to apply the Bonferroni level :math:`\alpha / (k - 1)`.

The relative MDE is the lift ``p_1 / p_0 - 1``; only increases are solved for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import optimize, stats

__all__ = [
    "PowerResult",
    "SampleSizeResult",
    "MDEResult",
    "achieved_power",
    "bonferroni_alpha",
    "cohens_h",
    "mde_for_n",
    "power_two_sided",
    "sample_size_per_arm",
]


@dataclass(frozen=True)
class SampleSizeResult:
    """Impressions needed per arm to detect a relative lift."""

    baseline: float
    relative_mde: float
    target_rate: float
    effect_size: float
    alpha: float
    alpha_adjusted: float
    power: float
    n_arms: int
    n_per_arm: int
    n_per_arm_exact: float
    n_total: int


@dataclass(frozen=True)
class MDEResult:
    """Smallest relative lift detectable with a given number of impressions per arm."""

    baseline: float
    n_per_arm: float
    alpha: float
    alpha_adjusted: float
    power: float
    n_arms: int
    effect_size: float
    target_rate: float
    relative_mde: float


@dataclass(frozen=True)
class PowerResult:
    """Power of a two-sided test for a given effect and observed arm sizes."""

    baseline: float
    target_rate: float
    effect_size: float
    n_control: float
    n_treatment: float
    alpha: float
    alpha_adjusted: float
    n_arms: int
    power: float


def _check_rate(p: float, name: str) -> None:
    if not 0 < p < 1:
        raise ValueError(f"{name} must be in (0, 1), got {p}")


def _check_level(alpha: float, power: float | None = None) -> None:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if power is not None and not alpha < power < 1:
        raise ValueError(f"power must be in (alpha, 1), got {power}")


def bonferroni_alpha(alpha: float, n_arms: int = 2) -> float:
    """Per-comparison level for ``n_arms - 1`` comparisons against one control.

    :math:`\\alpha_\\text{adj} = \\alpha / (k - 1)`. Use it whenever one test has several
    challengers, so the chance of any false winner stays at ``alpha``.
    """
    if n_arms < 2:
        raise ValueError(f"n_arms must be at least 2, got {n_arms}")
    return alpha / (n_arms - 1)


def cohens_h(p0: float, p1: float) -> float:
    r"""Cohen's h, :math:`2\arcsin\sqrt{p_1} - 2\arcsin\sqrt{p_0}`.

    Use it as the effect size for power calculations on proportions (it is variance
    stabilising, so the same h is equally hard to detect at any baseline).
    """
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p0))


def power_two_sided(effect_size: float, n_eff: float, alpha: float) -> float:
    r"""Two-sided normal power, :math:`\Phi(|h|\sqrt{n} - z) + \Phi(-|h|\sqrt{n} - z)`.

    ``n_eff`` is :math:`(1/n_0 + 1/n_1)^{-1}` (n/2 for equal arms). Use it as the building
    block; the functions below are friendlier.
    """
    z = stats.norm.ppf(1 - alpha / 2)
    shift = abs(effect_size) * math.sqrt(n_eff)
    return float(stats.norm.cdf(shift - z) + stats.norm.cdf(-shift - z))


def sample_size_per_arm(
    baseline: float,
    relative_mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    n_arms: int = 2,
) -> SampleSizeResult:
    r"""Impressions per arm to detect ``baseline * (1 + relative_mde)`` with equal arms.

    Solves :math:`1 - \beta = \Phi(|h|\sqrt{n/2} - z) + \Phi(-|h|\sqrt{n/2} - z)` for n,
    starting from the closed form
    :math:`n \approx 2\left((z_{1-\alpha'/2} + z_{1-\beta})/h\right)^2`
    with :math:`\alpha' = \alpha / (k - 1)`. Use it before a test to decide how much traffic
    each arm needs; ``n_per_arm`` is rounded up.
    """
    _check_rate(baseline, "baseline")
    _check_level(alpha, power)
    if relative_mde <= 0:
        raise ValueError(f"relative_mde must be positive, got {relative_mde}")
    target = baseline * (1 + relative_mde)
    _check_rate(target, "baseline * (1 + relative_mde)")
    a = bonferroni_alpha(alpha, n_arms)
    h = cohens_h(baseline, target)
    guess = 2 * ((stats.norm.ppf(1 - a / 2) + stats.norm.ppf(power)) / h) ** 2
    exact = optimize.brentq(
        lambda n: power_two_sided(h, n / 2, a) - power, guess / 4, guess * 4, xtol=1e-10
    )
    n = math.ceil(exact - 1e-9)
    return SampleSizeResult(
        baseline, relative_mde, target, h, alpha, a, power, n_arms, n, exact, n * n_arms
    )


def mde_for_n(
    baseline: float,
    n_per_arm: float,
    alpha: float = 0.05,
    power: float = 0.8,
    n_arms: int = 2,
) -> MDEResult:
    r"""Smallest relative lift detectable with ``n_per_arm`` impressions in each arm.

    Solves the power equation for h, then inverts h:
    :math:`p_1 = \sin^2\left(\arcsin\sqrt{p_0} + h/2\right)`, MDE :math:`= p_1/p_0 - 1`.
    Use it after the fact ("with the traffic we had, what could we have seen?") or to judge
    whether a test is worth running at all. Raises if no rate below 1 is detectable.
    """
    _check_rate(baseline, "baseline")
    _check_level(alpha, power)
    if n_per_arm <= 0:
        raise ValueError(f"n_per_arm must be positive, got {n_per_arm}")
    a = bonferroni_alpha(alpha, n_arms)
    h_max = math.pi - 2 * math.asin(math.sqrt(baseline))
    if power_two_sided(h_max, n_per_arm / 2, a) < power:
        raise ValueError(f"no lift is detectable at power {power} with n={n_per_arm} per arm")
    h = optimize.brentq(
        lambda e: power_two_sided(e, n_per_arm / 2, a) - power, 1e-12, h_max, xtol=1e-14
    )
    target = math.sin(math.asin(math.sqrt(baseline)) + h / 2) ** 2
    return MDEResult(baseline, n_per_arm, alpha, a, power, n_arms, h, target, target / baseline - 1)


def achieved_power(
    baseline: float,
    relative_mde: float,
    n_control: float,
    n_treatment: float | None = None,
    alpha: float = 0.05,
    n_arms: int = 2,
) -> PowerResult:
    """Power to detect ``relative_mde`` given the impressions each arm actually got.

    Uses :math:`n_\\text{eff} = (1/n_0 + 1/n_1)^{-1}`, so unequal arms are handled
    (``n_treatment`` defaults to ``n_control``). Use it to say how likely an observed test
    was to find a lift of a given size; never plug in the observed lift itself, since
    "observed power" is a function of the p-value and tells you nothing new.
    """
    _check_rate(baseline, "baseline")
    _check_level(alpha)
    n_t = n_control if n_treatment is None else n_treatment
    if n_control <= 0 or n_t <= 0:
        raise ValueError("arm sizes must be positive")
    target = baseline * (1 + relative_mde)
    _check_rate(target, "baseline * (1 + relative_mde)")
    a = bonferroni_alpha(alpha, n_arms)
    h = cohens_h(baseline, target)
    n_eff = 1 / (1 / n_control + 1 / n_t)
    return PowerResult(
        baseline, target, h, n_control, n_t, alpha, a, n_arms, power_two_sided(h, n_eff, a)
    )
