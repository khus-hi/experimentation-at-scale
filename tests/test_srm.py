"""abtest.srm against scipy.stats.chisquare."""

from __future__ import annotations

import pytest
from scipy import stats

from abtest.srm import SRM_THRESHOLD, srm_check


def test_equal_split_matches_scipy():
    imps = [10_000, 10_400, 9_700]
    r = srm_check(imps)
    ref = stats.chisquare(imps)
    assert r.chi2 == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)
    assert r.df == 2
    assert r.expected == pytest.approx((10_033.333333,) * 3)
    assert r.srm == bool(ref.pvalue < 0.001)


def test_custom_ratio_is_normalised():
    r = srm_check([20_000, 10_150], expected_ratio=[2, 1])
    ref = stats.chisquare([20_000, 10_150], [30_150 * 2 / 3, 30_150 / 3])
    assert r.p_value == pytest.approx(ref.pvalue)
    assert not r.srm


def test_flags_clear_mismatch():
    r = srm_check([10_000, 11_000])
    assert r.srm and r.p_value < SRM_THRESHOLD


def test_balanced_is_not_flagged():
    r = srm_check([5_000, 5_000])
    assert r.chi2 == 0 and r.p_value == 1.0 and not r.srm


@pytest.mark.parametrize(
    ("imps", "ratio"),
    [([10], None), ([-1, 5], None), ([0, 0], None), ([5, 5], [1]), ([5, 5], [1, 0])],
)
def test_bad_input_raises(imps, ratio):
    with pytest.raises(ValueError):
        srm_check(imps, ratio)
