"""abtest.bayes: beta-binomial decisions and empirical Bayes shrinkage, on simulated data.

Simulated data only checks the math here; every finding comes from the real archive via
analysis/shrinkage.py.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import duckdb
import numpy as np
import pytest
from scipy import stats

from abtest.bayes import (
    NormalPrior,
    SpikeSlabPrior,
    arm_contrasts,
    best_arm_contrast,
    beta_posteriors,
    decide,
    fit_normal_prior,
    fit_spike_slab,
    normal_loglik,
    prob_best,
    shrink,
    shrink_best_arms,
    winners_curse,
)

ROOT = Path(__file__).resolve().parents[1]

# --- beta-binomial -------------------------------------------------------------------------


def test_beta_posteriors_conjugate():
    (p,) = beta_posteriors([3], [10], prior_alpha=2, prior_beta=5)
    assert (p.alpha, p.beta) == (5, 12)
    assert p.mean == pytest.approx(5 / 17)
    assert p.lower == pytest.approx(stats.beta.ppf(0.025, 5, 12))
    assert p.lower < p.mean < p.upper


@pytest.mark.parametrize(
    "clicks,imps",
    [([10, 12, 9], [1000, 1000, 1000]), ([1, 0], [5, 3]), ([150, 160, 140, 175], [9e3] * 4)],
)
def test_prob_best_sums_to_one(clicks, imps):
    res = prob_best(clicks, imps, n_draws=5000, seed=1)
    assert sum(res.prob_best) == pytest.approx(1.0)
    assert all(v >= 0 for v in res.expected_loss)


def test_prob_best_seeded_and_ordered():
    a = prob_best([100, 130], [10_000, 10_000], seed=3)
    b = prob_best([100, 130], [10_000, 10_000], seed=3)
    assert a == b and a.seed == 3
    assert a.prob_best[1] > 0.9 and a.expected_loss[1] < a.expected_loss[0]
    # two arms: analytic P(B > A) from the normal approximation to the Beta posteriors
    ma, mb = a.mean
    sd = np.sqrt(ma * (1 - ma) / 10_002 + mb * (1 - mb) / 10_002)
    assert a.prob_best[1] == pytest.approx(stats.norm.cdf((mb - ma) / sd), abs=0.01)


def test_expected_loss_zero_when_one_arm_dominates():
    res = prob_best([900, 5, 3], [1000, 1000, 1000], n_draws=10_000, seed=0)
    assert res.prob_best[0] == 1.0
    assert res.expected_loss[0] == 0.0
    assert res.expected_loss[1] > 0.8
    d = decide(res, threshold=1e-6)
    assert d.arm == 0 and d.ship and d.loss == 0.0


def test_decide_absolute_and_relative():
    close = prob_best([100, 101], [10_000, 10_000], seed=2)
    assert not decide(close, threshold=1e-5).ship  # coin flip: losing ~0.1pp is not safe
    assert decide(close, threshold=0.5, relative=True).ship
    rel = decide(close, threshold=0.01, relative=True)
    assert rel.relative and rel.loss == pytest.approx(
        close.expected_loss[rel.arm] / close.mean[rel.arm]
    )
    with pytest.raises(ValueError):
        decide(close, threshold=-1)


def test_beta_input_validation():
    with pytest.raises(ValueError):
        beta_posteriors([5], [3])
    with pytest.raises(ValueError):
        beta_posteriors([1, 2], [3])
    with pytest.raises(ValueError):
        beta_posteriors([1], [3], prior_alpha=0)
    with pytest.raises(ValueError):
        prob_best([1], [3], n_draws=0)


# --- contrasts -----------------------------------------------------------------------------


def test_arm_contrasts_log_odds_ratio():
    cs = arm_contrasts([10, 20, 30], [1000, 1000, 1000])
    c = cs[2]
    expected = np.log(30 / 970) - np.log(30 / 1970)
    assert c.arm == 2 and c.d == pytest.approx(expected)
    assert c.se == pytest.approx(np.sqrt(1 / 30 + 1 / 970 + 1 / 30 + 1 / 1970))
    assert best_arm_contrast([10, 20, 30], [1000, 1000, 1000]) == c


def test_contrasts_zero_cells_and_empty_arms():
    cs = arm_contrasts([0, 5, 0], [100, 100, 0])
    assert [c.arm for c in cs] == [0, 1]
    assert np.isfinite(cs[0].d) and cs[0].d == pytest.approx(
        np.log(0.5 / 100.5) - np.log(5.5 / 95.5)
    )
    assert arm_contrasts([1, 0], [10, 0]) == []
    assert best_arm_contrast([1, 0], [10, 0]) is None


# --- empirical Bayes on simulated lifts ----------------------------------------------------


def simulate_lifts(n, tau, mu=0.0, pi0=0.0, seed=0):
    rng = np.random.default_rng(seed)
    se = rng.uniform(0.05, 0.3, n)
    theta = np.where(rng.random(n) < pi0, 0.0, rng.normal(mu, tau, n))
    return theta, theta + rng.normal(0, se), se


def test_normal_prior_recovers_variance_and_shrinks_mse():
    tau = 0.15
    theta, d, se = simulate_lifts(20_000, tau, mu=0.02, seed=1)
    prior = fit_normal_prior(d, se)
    assert abs(prior.tau2 - tau**2) / tau**2 < 0.10
    assert prior.mu == pytest.approx(0.02, abs=0.01) and prior.tau == pytest.approx(tau, rel=0.05)
    post = shrink(prior, d, se)
    assert np.mean((post.mean - theta) ** 2) < 0.6 * np.mean((d - theta) ** 2)
    cover = np.mean((post.lower <= theta) & (theta <= post.upper))
    assert cover == pytest.approx(0.95, abs=0.01)
    assert (post.p_null == 0).all() and np.allclose(
        post.sd**2, prior.tau2 * se**2 / (prior.tau2 + se**2)
    )


def test_spike_slab_recovers_prior_and_shrinks_mse():
    tau, pi0 = 0.2, 0.6
    theta, d, se = simulate_lifts(20_000, tau, mu=0.05, pi0=pi0, seed=2)
    prior = fit_spike_slab(d, se)
    assert abs(prior.tau2 - tau**2) / tau**2 < 0.10
    assert prior.pi0 == pytest.approx(pi0, abs=0.05) and prior.mu == pytest.approx(0.05, abs=0.03)
    assert prior.loglik > normal_loglik(fit_normal_prior(d, se), d, se)
    post = shrink(prior, d, se)
    assert np.mean((post.mean - theta) ** 2) < np.mean((d - theta) ** 2) / 2
    mom = shrink(fit_normal_prior(d, se), d, se)
    assert np.mean((post.mean - theta) ** 2) < np.mean((mom.mean - theta) ** 2)
    cover = np.mean((post.lower <= theta) & (theta <= post.upper))
    assert cover == pytest.approx(0.95, abs=0.015)
    assert post.p_null[theta == 0].mean() > post.p_null[np.abs(theta) > 0.3].mean() + 0.3
    assert (post.lower <= post.mean).all() and (post.mean <= post.upper).all()


def test_normal_prior_truncates_at_zero_and_shrinks_fully():
    rng = np.random.default_rng(3)
    se = np.full(5000, 0.1)
    d = rng.normal(0.0, 0.1, 5000)  # pure noise
    prior = fit_normal_prior(d, se)
    assert prior.tau2 >= 0 and prior.tau2 < 1e-3
    post = shrink(NormalPrior(0.0, 0.0, 5000), d, se)
    assert np.allclose(post.mean, 0) and np.allclose(post.lower, 0)


def test_mixture_interval_includes_spike():
    prior = SpikeSlabPrior(pi0=0.9, mu=0.0, tau2=0.04, loglik=0.0, n=1)
    post = shrink(prior, [0.01, 1.5], [0.1, 0.1])
    assert post.p_null[0] > 0.9 and post.lower[0] <= 1e-9 and post.upper[0] >= 0
    assert post.p_null[1] < 1e-6 and post.lower[1] > 0.5


def test_prior_input_validation():
    with pytest.raises(ValueError):
        fit_normal_prior([1.0], [0.1])
    with pytest.raises(ValueError):
        fit_normal_prior([1.0, 2.0], [0.1, 0.0])
    with pytest.raises(ValueError):
        fit_spike_slab([1.0, np.nan], [0.1, 0.1])


# --- the winner's curse on simulated multi-arm tests ----------------------------------------


def simulate_tests(n_tests, k=4, tau=0.1, pi0=0.0, seed=0):
    """Tests with k arms at a few percent CTR; returns (tests, true log-odds per arm)."""
    rng = np.random.default_rng(seed)
    tests, truth = [], []
    for _ in range(n_tests):
        eta = rng.uniform(-4.3, -3.5) + np.where(rng.random(k) < pi0, 0.0, rng.normal(0, tau, k))
        n = rng.integers(2_000, 10_000, k)
        tests.append((rng.binomial(n, 1 / (1 + np.exp(-eta))), n))
        truth.append((eta, n))
    return tests, truth


def true_contrast(eta, n, arm):
    p = 1 / (1 + np.exp(-eta))
    rest = np.arange(len(eta)) != arm
    pr = (p[rest] * n[rest]).sum() / n[rest].sum()
    return eta[arm] - np.log(pr / (1 - pr))


@pytest.mark.parametrize("prior", ["normal", "spike_slab"])
def test_shrinkage_removes_winners_curse(prior):
    tests, truth = simulate_tests(3000, pi0=0.5, seed=4)
    res = shrink_best_arms(tests, prior=prior)
    assert res.fit_on == "all" and res.n_fit == 4 * 3000
    theta = np.array(
        [true_contrast(*truth[i], a) for i, a in zip(res.test_index, res.arm, strict=True)]
    )
    naive_bias = np.mean(res.shrunk.d - theta)
    shrunk_bias = np.mean(res.shrunk.mean - theta)
    assert naive_bias > 0.08  # picking the max inflates the lift
    assert abs(shrunk_bias) < 0.2 * naive_bias
    assert np.mean((res.shrunk.mean - theta) ** 2) < np.mean((res.shrunk.d - theta) ** 2) / 2
    wc = winners_curse(res)
    assert wc.n_tests == 3000 and wc.mean_shrunk < wc.mean_naive and 0 < wc.retained < 0.6
    assert wc.mean_naive_rel == pytest.approx(np.expm1(wc.mean_naive))
    assert wc.share_interval_excludes_zero_shrunk < wc.share_interval_excludes_zero_naive
    assert winners_curse(res.shrunk) == wc


def test_fitting_on_winners_does_not_fix_the_curse():
    tests, truth = simulate_tests(2000, tau=0.0, seed=5)  # every arm identical
    res = shrink_best_arms(tests, prior="normal", fit_on="best")
    assert res.n_fit == 2000
    assert res.prior.mu > 0.1  # learned the selection bias as if it were real
    assert np.mean(res.shrunk.mean) > 0.1
    honest = shrink_best_arms(tests, prior="normal")
    assert abs(np.mean(honest.shrunk.mean)) < 0.01


def test_shrink_best_arms_skips_and_validates():
    tests = [([5, 9], [1000, 1000]), ([3], [100]), ([4, 6, 0], [900, 900, 0]), ([7, 2], [800, 800])]
    res = shrink_best_arms(tests, prior="normal")
    assert list(res.test_index) == [0, 2, 3] and list(res.arm) == [1, 1, 0]
    with pytest.raises(ValueError):
        shrink_best_arms(tests, fit_on="winners")
    with pytest.raises(ValueError):
        shrink_best_arms(tests, prior="t")
    with pytest.raises(ValueError):
        shrink_best_arms([([1, 2], [10, 10])])


# --- analysis/shrinkage.py end to end on a tiny warehouse ----------------------------------


def _shrinkage():
    spec = importlib.util.spec_from_file_location("shrinkage", ROOT / "analysis" / "shrinkage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_shrinkage_script(tmp_path):
    rng = np.random.default_rng(9)
    arms = []
    for t in range(40):
        k = 2 + t % 3
        n = rng.integers(2_000, 6_000, k)
        x = rng.binomial(n, 0.02 * np.exp(rng.normal(0, 0.15, k)))
        win = int(np.argmax(x / n)) if t % 2 else -1
        risk = t < 5
        arms += [(f"t{t:02d}", j + 1, int(x[j]), int(n[j]), j == win, risk) for j in range(k)]
    arms.append(("lonely", 1, 3, 100, False, False))
    db = tmp_path / "w.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("create schema marts; create schema intermediate")
        con.execute(
            "create table intermediate.int_test_arms(clickability_test_id varchar, "
            "arm_number int, clicks int, impressions int, winner boolean, risk boolean)"
        )
        con.executemany("insert into intermediate.int_test_arms values (?,?,?,?,?,?)", arms)
        con.execute(
            "create table marts.fct_tests as select clickability_test_id, "
            "bool_or(risk) as randomization_risk from intermediate.int_test_arms group by 1"
        )
    mod = _shrinkage()
    with duckdb.connect(str(db), read_only=True) as con:
        assert len(mod.load_tests(con)) == 36
        assert len(mod.load_tests(con, include_risk=True)) == 41

    out = tmp_path / "out"
    assert mod.main(["--db", str(db), "--out", str(out), "--draws", "500"]) == 0
    with (out / "shrinkage_summary.csv").open() as f:
        summary = list(csv.DictReader(f))
    assert [r["fit"] for r in summary] == [label for _, _, label in mod.FITS]
    with (out / "shrinkage_tests.csv").open() as f:
        per_test = list(csv.DictReader(f))
    assert len(per_test) == 35 and all(r["randomization_risk"] == "False" for r in per_test)
    assert mod.main(["--db", str(tmp_path / "missing.duckdb")]) == 1
