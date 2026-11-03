"""abtest.power against statsmodels NormalIndPower."""

from __future__ import annotations

import math

import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from abtest.power import (
    achieved_power,
    bonferroni_alpha,
    cohens_h,
    mde_for_n,
    sample_size_per_arm,
)

SM = NormalIndPower()
CASES = [
    (0.015, 0.10, 0.05, 0.80, 2),
    (0.015, 0.20, 0.05, 0.90, 2),
    (0.01, 0.05, 0.01, 0.80, 2),
    (0.03, 0.15, 0.05, 0.80, 4),
    (0.10, 0.30, 0.10, 0.95, 6),
]


def test_cohens_h_matches_statsmodels():
    assert cohens_h(0.015, 0.018) == pytest.approx(proportion_effectsize(0.018, 0.015))


@pytest.mark.parametrize(("p0", "mde", "alpha", "power", "k"), CASES)
def test_sample_size_matches_statsmodels(p0, mde, alpha, power, k):
    r = sample_size_per_arm(p0, mde, alpha, power, n_arms=k)
    es = proportion_effectsize(p0 * (1 + mde), p0)
    a = alpha / (k - 1)
    ref = SM.solve_power(effect_size=es, alpha=a, power=power, ratio=1.0)
    assert r.n_per_arm_exact == pytest.approx(ref, rel=1e-6)
    assert r.n_per_arm == math.ceil(ref)
    assert r.n_total == r.n_per_arm * k
    assert r.alpha_adjusted == pytest.approx(a)


@pytest.mark.parametrize(("p0", "mde", "alpha", "power", "k"), CASES)
def test_mde_inverts_sample_size(p0, mde, alpha, power, k):
    n = sample_size_per_arm(p0, mde, alpha, power, n_arms=k).n_per_arm_exact
    r = mde_for_n(p0, n, alpha, power, n_arms=k)
    assert r.relative_mde == pytest.approx(mde, rel=1e-6)
    # statsmodels' own effect-size solver stops at a loose tolerance, so check the other way:
    # its power at our MDE is the target power.
    es = proportion_effectsize(r.target_rate, p0)
    assert r.effect_size == pytest.approx(es, rel=1e-9)
    assert SM.power(effect_size=es, nobs1=n, alpha=alpha / (k - 1), ratio=1.0) == pytest.approx(
        power, rel=1e-9
    )


@pytest.mark.parametrize(("p0", "mde", "alpha", "k"), [(c[0], c[1], c[2], c[4]) for c in CASES])
@pytest.mark.parametrize(("n_c", "n_t"), [(5_000, 5_000), (20_000, 12_000), (800, 2_500)])
def test_achieved_power_matches_statsmodels(p0, mde, alpha, k, n_c, n_t):
    r = achieved_power(p0, mde, n_c, n_t, alpha, n_arms=k)
    es = proportion_effectsize(p0 * (1 + mde), p0)
    ref = SM.power(effect_size=es, nobs1=n_c, alpha=alpha / (k - 1), ratio=n_t / n_c)
    assert r.power == pytest.approx(ref, rel=1e-10)


def test_achieved_power_default_equal_arms():
    assert achieved_power(0.02, 0.1, 30_000).n_treatment == 30_000


def test_bonferroni():
    assert bonferroni_alpha(0.05, 2) == 0.05
    assert bonferroni_alpha(0.05, 6) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        bonferroni_alpha(0.05, 1)


def test_more_arms_need_more_per_arm():
    two = sample_size_per_arm(0.015, 0.1).n_per_arm
    four = sample_size_per_arm(0.015, 0.1, n_arms=4).n_per_arm
    assert four > two


@pytest.mark.parametrize(
    "kwargs",
    [
        {"baseline": 0.0, "relative_mde": 0.1},
        {"baseline": 0.5, "relative_mde": 1.5},
        {"baseline": 0.1, "relative_mde": -0.1},
        {"baseline": 0.1, "relative_mde": 0.1, "power": 0.01},
        {"baseline": 0.1, "relative_mde": 0.1, "alpha": 0.0},
    ],
)
def test_sample_size_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        sample_size_per_arm(**kwargs)


def test_mde_rejects_bad_input():
    with pytest.raises(ValueError):
        mde_for_n(0.1, 0)
    with pytest.raises(ValueError):
        mde_for_n(0.5, 2)


def test_achieved_power_rejects_bad_input():
    with pytest.raises(ValueError):
        achieved_power(0.1, 0.1, 0)
