"""abtest.multiple: parity with statsmodels, error-rate simulations, and the fixture helper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.multitest import multipletests

from abtest.multiple import (
    METHODS,
    adjust,
    benjamini_hochberg,
    holm,
    portfolio,
    storey_pi0,
    winner_survival,
    within_test,
)
from abtest.proportions import chi2_omnibus, two_proportion_ztest

FIXTURE = Path(__file__).parent / "fixtures" / "packages_sample.csv"
SM_NAME = {"bonferroni": "bonferroni", "holm": "holm", "bh": "fdr_bh", "by": "fdr_by"}

FIXED_VECTORS = [
    [0.01, 0.04, 0.03, 0.005, 0.20, 0.50, 0.012, 0.9],
    [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216],
    [0.02, 0.02, 0.02, 0.02],
    [0.0, 1.0, 0.5, 0.049999, 0.05, 0.050001],
    [0.3],
    list(np.random.default_rng(7).uniform(size=200) ** 3),
]


# --- statsmodels parity -----------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("p", FIXED_VECTORS, ids=lambda p: f"m{len(p)}")
@pytest.mark.parametrize("alpha", [0.05, 0.10])
def test_matches_statsmodels_on_fixed_vectors(p, method, alpha):
    ours = adjust(p, method, alpha)
    reject, p_adj, _, _ = multipletests(p, alpha=alpha, method=SM_NAME[method])
    np.testing.assert_allclose(ours.p_adjusted, p_adj, rtol=1e-12, atol=1e-15)
    assert ours.reject == tuple(bool(r) for r in reject)
    assert ours.p_values == pytest.approx(p)
    assert ours.n_rejected == int(reject.sum())


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.floats(0, 1, allow_nan=False), min_size=1, max_size=60),
    st.sampled_from(METHODS),
)
def test_matches_statsmodels_on_random_vectors(p, method):
    _, p_adj, _, _ = multipletests(p, alpha=0.05, method=SM_NAME[method])
    np.testing.assert_allclose(adjust(p, method).p_adjusted, p_adj, rtol=1e-10, atol=1e-14)


@settings(max_examples=100, deadline=None)
@given(st.lists(st.floats(0, 1, allow_nan=False), min_size=1, max_size=60))
def test_ordering_between_methods(p):
    adj = {m: np.array(adjust(p, m).p_adjusted) for m in METHODS}
    p = np.array(p)
    assert (adj["holm"] <= adj["bonferroni"] + 1e-15).all()
    assert (adj["bh"] <= adj["holm"] + 1e-15).all()
    assert (adj["bh"] <= adj["by"] + 1e-15).all()
    assert all((a >= p - 1e-15).all() and (a <= 1).all() for a in adj.values())


def test_empty_vector_and_bad_input():
    assert holm([]).size == 0 and benjamini_hochberg([]).size == 0
    assert adjust([], "by").n_rejected == 0
    for bad in ([0.1, np.nan], [1.2], [-0.1], [[0.1, 0.2]]):
        with pytest.raises(ValueError):
            adjust(bad)
    with pytest.raises(ValueError):
        adjust([0.1], method="fdr_bh")
    with pytest.raises(ValueError):
        adjust([0.1], alpha=0)


# --- Storey's pi0 -----------------------------------------------------------------------


def test_storey_pi0_formula():
    p = [0.01, 0.2, 0.6, 0.7, 0.8, 0.9, 0.95, 0.4]
    assert storey_pi0(p, lambda_=0.5) == 1.0  # 5 / (8 * 0.5) = 1.25, capped at 1
    assert storey_pi0([0.01, 0.02, 0.6, 0.9], lambda_=0.5) == pytest.approx(2 / 2)
    assert storey_pi0([0.01, 0.02, 0.03, 0.9], lambda_=0.5) == pytest.approx(0.5)
    assert storey_pi0([0.01, 0.3, 0.7, 0.9], lambda_=0.0) == 1.0


def test_storey_pi0_bad_input():
    with pytest.raises(ValueError):
        storey_pi0([0.5], lambda_=1.0)
    with pytest.raises(ValueError):
        storey_pi0([])


def test_storey_pi0_recovers_share_of_nulls():
    rng = np.random.default_rng(11)
    null = rng.uniform(size=8_000)
    alt = stats.norm.sf(rng.normal(3, 1, size=2_000))  # one-sided p-values of real effects
    assert storey_pi0(null) == pytest.approx(1.0, abs=0.03)
    assert storey_pi0(np.concatenate([null, alt])) == pytest.approx(0.8, abs=0.03)


# --- within a test and across the portfolio ---------------------------------------------


def test_within_test_matches_ztests_plus_holm():
    clicks, imps = [150, 190, 155, 230], [10_000, 10_000, 10_000, 10_000]
    r = within_test(clicks, imps, control=0)
    raw = [two_proportion_ztest(150, 10_000, x, 10_000).p_value for x in clicks[1:]]
    _, p_adj, _, _ = multipletests(raw, method="holm")
    assert r.control == 0 and r.arms == (1, 2, 3)
    assert r.adjustment.p_values == pytest.approx(raw)
    assert r.adjustment.p_adjusted == pytest.approx(p_adj)
    assert r.diffs == pytest.approx([0.004, 0.0005, 0.008])
    assert r.adjustment.reject == (False, False, True)


def test_within_test_other_control_and_method():
    r = within_test([150, 190, 230], [10_000] * 3, control=2, method="bonferroni")
    assert r.arms == (0, 1)
    assert all(d < 0 for d in r.diffs)
    assert r.adjustment.method == "bonferroni"


@pytest.mark.parametrize(
    ("clicks", "imps", "control"),
    [([1], [10], 0), ([1, 2], [10], 0), ([1, 2], [10, 10], 2), ([1, 2], [10, 10], -1)],
)
def test_within_test_bad_input(clicks, imps, control):
    with pytest.raises(ValueError):
        within_test(clicks, imps, control)


def test_portfolio_uses_omnibus_pvalues():
    tests = [([150, 230], [10_000, 10_000]), ([150, 152, 149], [10_000] * 3), ([0, 0], [5, 5])]
    r = portfolio(tests)
    raw = [chi2_omnibus(c, n)[2] for c, n in tests]
    assert r.adjustment.p_values == pytest.approx(raw)
    assert r.adjustment.method == "bh"
    assert r.adjustment.reject == (True, False, False)
    assert r.pi0 == storey_pi0(raw)
    with pytest.raises(ValueError):
        portfolio([])


# --- simulation: 1,000 tests, 80% null, 20% with a real lift ----------------------------

N_TESTS, SHARE_NULL, N_REPS = 1_000, 0.8, 1_000
N_PER_ARM, BASE_CTR, LIFT = 10_000, 0.015, 0.30
SIM_SEED = 20240501


def _ztest_pvalues(x_c: np.ndarray, x_t: np.ndarray, n: int) -> np.ndarray:
    """Vectorised pooled two-proportion z-test, equal arm sizes (same formula as abtest)."""
    pooled = (x_c + x_t) / (2 * n)
    se = np.sqrt(pooled * (1 - pooled) * 2 / n)
    z = np.divide(x_t - x_c, n * se, out=np.zeros_like(se), where=se > 0)
    return 2 * stats.norm.sf(np.abs(z))


@pytest.fixture(scope="module")
def simulation():
    rng = np.random.default_rng(SIM_SEED)
    n_null = int(N_TESTS * SHARE_NULL)
    is_null = np.arange(N_TESTS) < n_null
    p_t = np.where(is_null, BASE_CTR, BASE_CTR * (1 + LIFT))
    fdp, fwe, power, pi0 = {}, {}, {}, []
    for m in ("none", "holm", "bh"):
        fdp[m], fwe[m], power[m] = [], [], []
    for _ in range(N_REPS):
        x_c = rng.binomial(N_PER_ARM, BASE_CTR, size=N_TESTS)
        x_t = rng.binomial(N_PER_ARM, p_t)
        p = _ztest_pvalues(x_c, x_t, N_PER_ARM)
        pi0.append(storey_pi0(p))
        for m in ("none", "holm", "bh"):
            rej = p <= 0.05 if m == "none" else np.array(adjust(p, m, 0.05).reject)
            false = int((rej & is_null).sum())
            fdp[m].append(false / max(1, int(rej.sum())))
            fwe[m].append(false > 0)
            power[m].append((rej & ~is_null).mean() / (1 - SHARE_NULL))
    out = {
        m: {
            "FDR": float(np.mean(fdp[m])),
            "FWER": float(np.mean(fwe[m])),
            "power": float(np.mean(power[m])),
        }
        for m in fdp
    }
    out["pi0"] = float(np.mean(pi0))
    print(f"\n{N_REPS} reps of {N_TESTS} tests, {SHARE_NULL:.0%} null, lift {LIFT:.0%}")
    for m in ("none", "holm", "bh"):
        print(f"  {m:>4}: " + "  ".join(f"{k} {v:.4f}" for k, v in out[m].items()))
    print(f"  mean Storey pi0: {out['pi0']:.4f}")
    return out


def _mc_se(rate: float) -> float:
    return float(np.sqrt(rate * (1 - rate) / N_REPS))


def test_bh_controls_fdr_at_five_percent(simulation):
    fdr = simulation["bh"]["FDR"]
    assert fdr <= 0.05
    # BH's actual guarantee is pi0 * alpha = 4%; the average should sit near it.
    assert fdr == pytest.approx(SHARE_NULL * 0.05, abs=0.006)
    assert simulation["none"]["FDR"] > 0.10  # without a correction, far above 5%


def test_holm_controls_fwer_at_five_percent(simulation):
    fwer = simulation["holm"]["FWER"]
    # Monte Carlo average of a 0/1 outcome: allow two standard errors of noise above 5%.
    assert fwer <= 0.05 + 2 * _mc_se(0.05)
    assert simulation["none"]["FWER"] == 1.0  # 800 nulls at 5%: some always slip through


def test_bh_finds_more_real_effects_than_holm(simulation):
    assert simulation["bh"]["power"] > simulation["holm"]["power"]


def test_storey_pi0_in_simulation(simulation):
    assert simulation["pi0"] == pytest.approx(SHARE_NULL, abs=0.03)


# --- declared winners on the committed fixture ------------------------------------------


def _fixture_tests() -> list[tuple[list[int], list[int], int | None]]:
    df = pd.read_csv(FIXTURE, usecols=["clickability_test_id", "created_at", "impressions",
                                       "clicks", "winner"])  # fmt: skip
    out = []
    for _, g in df.sort_values(["clickability_test_id", "created_at"], kind="stable").groupby(
        "clickability_test_id", sort=True
    ):
        winners = np.flatnonzero(g["winner"].to_numpy(dtype=bool))
        out.append(
            (
                g["clicks"].astype(int).tolist(),
                g["impressions"].astype(int).tolist(),
                int(winners[0]) if winners.size else None,
            )
        )
    return out


def test_winner_survival_on_fixture():
    tests = _fixture_tests()
    r = winner_survival(tests)
    print(
        f"\nfixture: {r.n_tests} tests, {r.n_with_winner} with a declared winner; "
        f"survive Holm within test {r.survive_within}, BH across portfolio "
        f"{r.survive_portfolio}, both {r.survive_both}; Storey pi0 {r.pi0:.3f}"
    )
    assert r.n_tests == 200 and r.n_with_winner == 52
    assert r.survive_both <= min(r.survive_within, r.survive_portfolio)
    assert r.survive_portfolio <= r.n_with_winner
    assert 0 <= r.pi0 <= 1
    # Cross-check the portfolio count by hand with statsmodels.
    omni = [chi2_omnibus(c, n)[2] for c, n, _ in tests]
    reject, _, _, _ = multipletests(omni, alpha=0.05, method="fdr_bh")
    assert r.survive_portfolio == sum(
        bool(rej) for rej, (_, _, w) in zip(reject, tests, strict=True) if w is not None
    )


def test_winner_survival_toy_cases():
    tests = [
        ([300, 150, 140], [10_000] * 3, 0),  # clear winner beats both
        ([300, 295, 140], [10_000] * 3, 0),  # beats one arm, ties the other
        ([150, 300], [10_000] * 2, 0),  # declared winner is the loser
        ([150, 152], [10_000] * 2, None),  # no winner, still in the portfolio family
        ([10], [100], 0),  # single arm: skipped
    ]
    r = winner_survival(tests)
    assert (r.n_tests, r.n_with_winner) == (4, 3)
    assert r.survive_within == 1
    assert r.survive_portfolio == 3
    assert r.survive_both == 1
    with pytest.raises(ValueError):
        winner_survival([([1], [10], 0)])
