"""Hypothesis property tests: invariants that must hold for any counts."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from abtest.proportions import (
    chi2_omnibus,
    newcombe_diff_ci,
    relative_lift,
    two_proportion_ztest,
    wilson_ci,
)
from abtest.srm import srm_check


@st.composite
def arm(draw, max_n=200_000):
    n = draw(st.integers(1, max_n))
    return draw(st.integers(0, n)), n


alphas = st.sampled_from([0.01, 0.05, 0.1, 0.2])
SETTINGS = settings(max_examples=300, deadline=None)


@SETTINGS
@given(arm(), alphas)
def test_wilson_contains_estimate(a, alpha):
    ci = wilson_ci(*a, alpha=alpha)
    assert 0 <= ci.lower <= ci.estimate <= ci.upper <= 1


@SETTINGS
@given(arm(), arm(), alphas)
def test_newcombe_contains_estimate(c, t, alpha):
    ci = newcombe_diff_ci(*c, *t, alpha=alpha)
    assert -1 <= ci.lower <= ci.estimate <= ci.upper <= 1


@SETTINGS
@given(arm(), arm(), alphas)
def test_lift_ci_contains_estimate(c, t, alpha):
    r = relative_lift(*c, *t, alpha=alpha)
    if not math.isnan(r.lower):
        assert r.lower <= r.lift <= r.upper


@SETTINGS
@given(arm(), arm())
def test_p_value_in_unit_interval(c, t):
    assert 0 <= two_proportion_ztest(*c, *t).p_value <= 1
    _, _, p = chi2_omnibus([c[0], t[0]], [c[1], t[1]])
    assert 0 <= p <= 1


@SETTINGS
@given(st.lists(st.integers(0, 1_000_000), min_size=2, max_size=8).filter(lambda v: sum(v) > 0))
def test_srm_p_in_unit_interval(imps):
    assert 0 <= srm_check(imps).p_value <= 1


@SETTINGS
@given(arm(), arm(), alphas)
def test_symmetric_when_arms_swap(c, t, alpha):
    ab, ba = two_proportion_ztest(*c, *t), two_proportion_ztest(*t, *c)
    assert ab.z == pytest.approx(-ba.z, abs=1e-12)
    assert ab.p_value == pytest.approx(ba.p_value, abs=1e-12)
    ci_ab = newcombe_diff_ci(*c, *t, alpha=alpha)
    ci_ba = newcombe_diff_ci(*t, *c, alpha=alpha)
    assert ci_ab.lower == pytest.approx(-ci_ba.upper, abs=1e-12)
    assert ci_ab.upper == pytest.approx(-ci_ba.lower, abs=1e-12)
    om_ab = chi2_omnibus([c[0], t[0]], [c[1], t[1]])
    om_ba = chi2_omnibus([t[0], c[0]], [t[1], c[1]])
    assert om_ab[2] == pytest.approx(om_ba[2], abs=1e-12)
