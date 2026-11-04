"""abtest.thinning: splitting counts into independent halves, and replicating winners."""

from __future__ import annotations

import numpy as np
import pytest

from abtest.bayes import arm_contrasts, best_arm_contrast
from abtest.thinning import best_arms, contrasts, replicate_winners, split_binomial

REPS = 40_000


def _sim(n: int, p: float, reps: int = REPS, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).binomial(n, p, reps)


# --- split_binomial ----------------------------------------------------------------------


def test_split_conserves_counts_and_bounds():
    rng = np.random.default_rng(1)
    n = rng.integers(0, 5_000, 500)
    x = rng.binomial(n, 0.04)
    s = split_binomial(n, x, 0.5, rng)
    assert (s.n_a + s.n_b == n).all() and (s.x_a + s.x_b == x).all()
    assert (s.x_a >= 0).all() and (s.x_b >= 0).all()
    assert (s.x_a <= s.n_a).all() and (s.x_b <= s.n_b).all()
    assert (np.abs(s.n_a - n / 2) <= 0.5).all()


def test_split_is_reproducible_with_a_seed():
    n, x = np.array([1000, 1001, 7]), np.array([30, 0, 7])
    a, b = split_binomial(n, x, 0.5, 42), split_binomial(n, x, 0.5, 42)
    assert (a.x_a == b.x_a).all() and (a.n_a == b.n_a).all()


def test_split_scalar_and_empty():
    s = split_binomial(10, 4, 0.5, 0)
    assert s.n_a == 5 and s.x_a + s.x_b == 4
    e = split_binomial(np.array([], dtype=int), np.array([], dtype=int), 0.5, 0)
    assert e.n_a.size == 0


@pytest.mark.parametrize(
    "n, x, frac",
    [(10, 11, 0.5), (10, -1, 0.5), (10, 3, 0.0), (10, 3, 1.0), ([10, 5], [3], 0.5)],
)
def test_split_rejects_bad_input(n, x, frac):
    with pytest.raises(ValueError):
        split_binomial(n, x, frac, 0)


def test_split_rejects_non_integer_counts():
    with pytest.raises(ValueError):
        split_binomial(np.array([10.0]), np.array([3.0]), 0.5, 0)


@pytest.mark.parametrize("frac", [0.5, 0.3])
def test_halves_have_the_right_mean_and_variance(frac):
    n, p = 2_000, 0.03
    x = _sim(n, p)
    s = split_binomial(np.full(REPS, n), x, frac, 7)
    n_a = frac * n
    se_mean = np.sqrt(n_a * p * (1 - p) / REPS)
    assert abs(s.x_a.mean() - n_a * p) < 5 * se_mean
    assert abs(s.x_b.mean() - (n - n_a) * p) < 5 * np.sqrt((n - n_a) * p * (1 - p) / REPS)
    # Each half is binomial with its own n, so the variance is binomial too.
    assert s.x_a.var() == pytest.approx(n_a * p * (1 - p), rel=0.05)


def test_halves_are_uncorrelated_under_the_binomial_model():
    n, p = 2_000, 0.03
    s = split_binomial(np.full(REPS, n), _sim(n, p), 0.5, 8)
    r = np.corrcoef(s.x_a, s.x_b)[0, 1]
    assert abs(r) < 4 / np.sqrt(REPS)


def test_half_ctrs_uncorrelated_with_odd_n():
    """Odd n: half sizes vary by one; the click rates stay unbiased and uncorrelated."""
    n, p = 1_001, 0.05
    s = split_binomial(np.full(REPS, n), _sim(n, p, seed=3), 0.5, 9)
    assert s.n_a.mean() == pytest.approx(n / 2, abs=0.02)
    ctr_a, ctr_b = s.x_a / s.n_a, s.x_b / s.n_b
    assert ctr_a.mean() == pytest.approx(p, rel=0.01)
    assert ctr_b.mean() == pytest.approx(p, rel=0.01)
    assert abs(np.corrcoef(ctr_a, ctr_b)[0, 1]) < 4 / np.sqrt(REPS)


def test_overdispersed_arms_give_correlated_halves():
    """The documented limitation: an arm-level shock is shared by both halves."""
    rng = np.random.default_rng(4)
    n = 2_000
    p = rng.beta(30, 970, REPS)  # mean 3%, extra-binomial spread
    s = split_binomial(np.full(REPS, n), rng.binomial(n, p), 0.5, rng)
    assert np.corrcoef(s.x_a, s.x_b)[0, 1] > 0.2


# --- many tests at once ------------------------------------------------------------------


def _random_tests(seed: int, n_tests: int = 60):
    rng = np.random.default_rng(seed)
    t, x, n = [], [], []
    for i in range(n_tests):
        k = 2 + i % 4
        ni = rng.integers(1, 4_000, k)
        if i % 7 == 0:
            ni[0] = 0
        xi = rng.binomial(ni, 0.02 if i % 5 else 0.0)
        t += [i] * k
        x += list(xi)
        n += list(ni)
    return np.array(t), np.array(x), np.array(n)


def test_contrasts_match_bayes_arm_contrasts():
    t, x, n = _random_tests(5)
    d, se = contrasts(t, x, n)
    for i in np.unique(t):
        m = np.flatnonzero(t == i)
        if (n[m] == 0).any():
            continue  # bayes drops empty arms before pooling; the analysis never has them
        ref = arm_contrasts(x[m], n[m])
        assert np.allclose(d[m], [c.d for c in ref])
        assert np.allclose(se[m], [c.se for c in ref])


def test_contrasts_single_arm_test_is_nan():
    d, se = contrasts([0, 1, 1], [3, 5, 9], [100, 200, 210])
    assert np.isnan(d[0]) and np.isnan(se[0]) and np.isfinite(d[1:]).all()


def test_best_arms_match_bayes_including_ties_and_empty_arms():
    t, x, n = _random_tests(6)
    best = best_arms(t, x, n)
    for i, b in zip(np.unique(t), best, strict=True):
        m = np.flatnonzero(t == i)
        ref = best_arm_contrast(x[m], n[m])
        if ref is not None:  # bayes needs 2 arms with impressions
            assert b - m[0] == ref.arm
    assert best_arms([0, 0, 0], [1, 2, 4], [10, 20, 40]).tolist() == [0]
    assert best_arms([0, 0], [0, 0], [0, 0]).tolist() == [-1]


def test_test_labels_must_be_sorted_and_aligned():
    with pytest.raises(ValueError):
        contrasts([1, 0], [1, 1], [10, 10])
    with pytest.raises(ValueError):
        best_arms([0], [1, 1], [10, 10])


def test_replication_under_the_null_shows_the_curse_and_no_real_lift():
    """All arms share one click rate: the half-A winner looks good on A and is average on B."""
    rng = np.random.default_rng(11)
    k, n_tests, n_arm, p = 4, 3_000, 4_000, 0.02
    t = np.repeat(np.arange(n_tests), k)
    n = np.full(t.size, n_arm)
    x = rng.binomial(n, p)
    r = replicate_winners(t, x, n, rng)
    assert r.d_a.mean() > 0.15
    assert abs(r.d_b.mean()) < 4 * r.se_b.mean() / np.sqrt(n_tests)
    assert r.still_best.mean() == pytest.approx(1 / k, abs=0.03)
    assert r.d_a_all.size == t.size and (r.n_arms == k).all()


def test_replication_recovers_a_real_winner():
    rng = np.random.default_rng(12)
    t = np.repeat(np.arange(500), 2)
    n = np.full(t.size, 200_000)
    p = np.tile([0.02, 0.03], 500)
    r = replicate_winners(t, rng.binomial(n, p), n, rng)
    assert r.still_best.mean() > 0.99
    assert r.d_b.mean() == pytest.approx(np.log(0.03 / 0.97) - np.log(0.02 / 0.98), abs=0.01)


def test_replication_rejects_single_arm_tests():
    with pytest.raises(ValueError):
        replicate_winners([0, 1, 1], [1, 2, 3], [10, 20, 30], 0)
    with pytest.raises(ValueError):
        replicate_winners([0, 0], [0, 0], [1, 0], np.random.default_rng(0), frac=0.01)
