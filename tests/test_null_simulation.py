"""Under a true null the z-test should reject about alpha of the time."""

from __future__ import annotations

import numpy as np

from abtest.proportions import chi2_omnibus, two_proportion_ztest
from abtest.srm import srm_check

N_SIMS = 2_000
SEED = 20131


def test_ztest_false_positive_rate_is_five_percent():
    rng = np.random.default_rng(SEED)
    n, p = 10_000, 0.015
    a = rng.binomial(n, p, size=N_SIMS)
    b = rng.binomial(n, p, size=N_SIMS)
    rejections = sum(
        two_proportion_ztest(int(x), n, int(y), n).p_value < 0.05 for x, y in zip(a, b, strict=True)
    )
    fpr = rejections / N_SIMS
    assert 0.035 <= fpr <= 0.065, fpr


def test_omnibus_false_positive_rate_is_five_percent():
    rng = np.random.default_rng(SEED + 1)
    n, p, k = 8_000, 0.02, 4
    clicks = rng.binomial(n, p, size=(N_SIMS, k))
    fpr = np.mean([chi2_omnibus(row, [n] * k)[2] < 0.05 for row in clicks])
    assert 0.035 <= fpr <= 0.065, fpr


def test_srm_rarely_fires_on_fair_splits():
    rng = np.random.default_rng(SEED + 2)
    splits = rng.multinomial(20_000, [0.5, 0.5], size=N_SIMS)
    flagged = sum(srm_check(s).srm for s in splits)
    assert flagged <= 8
