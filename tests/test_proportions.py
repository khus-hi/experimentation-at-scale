"""Fixed-example checks of abtest.proportions against statsmodels and scipy."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportions_ztest,
)

from abtest.proportions import (
    chi2_omnibus,
    compare_arms,
    multi_arm_test,
    newcombe_diff_ci,
    relative_lift,
    two_proportion_ztest,
    wilson_ci,
)

# (x_c, n_c, x_t, n_t): typical Upworthy CTRs, a small-count case and unequal arms
EXAMPLES = [
    (120, 10_000, 150, 10_000),
    (13, 3_200, 25, 2_900),
    (480, 31_000, 455, 30_100),
    (3, 150, 9, 160),
    (0, 500, 4, 500),
]


@pytest.mark.parametrize(("x_c", "n_c", "x_t", "n_t"), EXAMPLES)
def test_ztest_matches_statsmodels(x_c, n_c, x_t, n_t):
    ours = two_proportion_ztest(x_c, n_c, x_t, n_t)
    z, p = proportions_ztest([x_t, x_c], [n_t, n_c])
    assert ours.z == pytest.approx(z, rel=1e-12)
    assert ours.p_value == pytest.approx(p, rel=1e-10)
    assert ours.diff == pytest.approx(x_t / n_t - x_c / n_c)


@pytest.mark.parametrize(("x_c", "n_c", "x_t", "n_t"), EXAMPLES)
@pytest.mark.parametrize("alpha", [0.05, 0.01, 0.1])
def test_wilson_matches_statsmodels(x_c, n_c, x_t, n_t, alpha):
    for x, n in [(x_c, n_c), (x_t, n_t)]:
        ci = wilson_ci(x, n, alpha)
        lo, hi = proportion_confint(x, n, alpha=alpha, method="wilson")
        assert ci.lower == pytest.approx(lo, abs=1e-12)
        assert ci.upper == pytest.approx(hi, abs=1e-12)
        assert ci.confidence == pytest.approx(1 - alpha)


@pytest.mark.parametrize(("x_c", "n_c", "x_t", "n_t"), EXAMPLES)
@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_newcombe_matches_statsmodels(x_c, n_c, x_t, n_t, alpha):
    ci = newcombe_diff_ci(x_c, n_c, x_t, n_t, alpha)
    lo, hi = confint_proportions_2indep(
        x_t, n_t, x_c, n_c, method="newcomb", compare="diff", alpha=alpha
    )
    assert ci.lower == pytest.approx(lo, abs=1e-12)
    assert ci.upper == pytest.approx(hi, abs=1e-12)


def test_ztest_without_variation():
    r = two_proportion_ztest(0, 100, 0, 200)
    assert (r.z, r.p_value) == (0.0, 1.0)
    r = two_proportion_ztest(50, 50, 80, 80)
    assert (r.z, r.p_value) == (0.0, 1.0)


def test_wilson_edges():
    assert wilson_ci(0, 40).lower == 0.0
    assert wilson_ci(40, 40).upper == 1.0
    assert wilson_ci(5, 40).contains(5 / 40)


@pytest.mark.parametrize(("x", "n"), [(-1, 10), (11, 10), (0, 0)], ids=["negative", "x>n", "n=0"])
def test_bad_counts_raise(x, n):
    with pytest.raises(ValueError):
        wilson_ci(x, n)


def test_bad_alpha_raises():
    with pytest.raises(ValueError):
        wilson_ci(1, 10, alpha=1.5)


def test_relative_lift_delta_method():
    x_c, n_c, x_t, n_t = 120, 10_000, 150, 10_000
    r = relative_lift(x_c, n_c, x_t, n_t)
    p_c, p_t = x_c / n_c, x_t / n_t
    var_ratio = (p_t / p_c) ** 2 * (p_t * (1 - p_t) / n_t / p_t**2 + p_c * (1 - p_c) / n_c / p_c**2)
    assert r.lift == pytest.approx(0.25)
    assert r.se == pytest.approx(math.sqrt(var_ratio), rel=1e-12)
    z = stats.norm.ppf(0.975)
    assert r.lower == pytest.approx(0.25 - z * r.se)
    assert r.upper == pytest.approx(0.25 + z * r.se)


def test_relative_lift_zero_clicks():
    r = relative_lift(0, 100, 5, 100)
    assert math.isnan(r.lift) and math.isnan(r.lower)
    r = relative_lift(5, 100, 0, 100)
    assert r.lift == -1.0 and math.isnan(r.se)


def test_compare_arms_bundles_the_three():
    zt, ci, lift = compare_arms(120, 10_000, 150, 10_000)
    assert zt == two_proportion_ztest(120, 10_000, 150, 10_000)
    assert ci == newcombe_diff_ci(120, 10_000, 150, 10_000)
    assert lift == relative_lift(120, 10_000, 150, 10_000)


def test_omnibus_matches_scipy():
    clicks = [120, 150, 131, 98]
    imps = [10_000, 10_200, 9_800, 9_900]
    chi2, df, p = chi2_omnibus(clicks, imps)
    table = np.array([clicks, np.subtract(imps, clicks)])
    ref = stats.chi2_contingency(table, correction=False)
    assert chi2 == pytest.approx(ref.statistic, rel=1e-12)
    assert p == pytest.approx(ref.pvalue, rel=1e-10)
    assert df == ref.dof == 3


def test_omnibus_two_arms_equals_z_squared():
    chi2, _, p = chi2_omnibus([120, 150], [10_000, 10_000])
    zt = two_proportion_ztest(120, 10_000, 150, 10_000)
    assert chi2 == pytest.approx(zt.z**2)
    assert p == pytest.approx(zt.p_value)


def test_omnibus_edges():
    assert chi2_omnibus([0, 0, 0], [10, 20, 30]) == (0.0, 2, 1.0)
    with pytest.raises(ValueError):
        chi2_omnibus([1], [10])
    with pytest.raises(ValueError):
        chi2_omnibus([1, 2], [10])


def test_multi_arm_test():
    clicks = [120, 150, 131, 98]
    imps = [10_000, 10_200, 9_800, 9_900]
    r = multi_arm_test(clicks, imps, control=2)
    assert r.n_arms == 4 and r.control == 2 and r.df == 3
    assert [c.arm for c in r.comparisons] == [0, 1, 3]
    assert len(r.arm_rates) == 4
    assert r.arm_rates[1] == wilson_ci(150, 10_200)
    for c in r.comparisons:
        zt = two_proportion_ztest(131, 9_800, clicks[c.arm], imps[c.arm])
        assert c.ztest == zt
        assert c.p_value_bonferroni == pytest.approx(min(1.0, 3 * zt.p_value))
    with pytest.raises(ValueError):
        multi_arm_test(clicks, imps, control=4)
