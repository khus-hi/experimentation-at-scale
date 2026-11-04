"""abtest.sequential: boundaries, the mSPRT p-value, and operating characteristics by simulation."""

from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path

import duckdb
import numpy as np
import pytest

from abtest.proportions import two_proportion_ztest
from abtest.sequential import (
    TestDesign,
    designs_from_rows,
    draw_designs,
    look_fractions,
    msprt_p_values,
    obf_boundaries,
    obf_spending,
    pooled_z,
    simulate,
    simulate_calibrated,
    simulate_paths,
)

ROOT = Path(__file__).resolve().parents[1]
N_SIMS = 20_000


def within(rate: float, target: float, n: int = N_SIMS, k: float = 3.0) -> bool:
    """rate <= target plus k Monte Carlo standard errors."""
    return rate <= target + k * math.sqrt(target * (1 - target) / n)


# --- boundaries --------------------------------------------------------------------------


def test_obf_spending_endpoints():
    assert obf_spending(1.0) == pytest.approx(0.05)
    assert obf_spending(0.2) < 1e-5
    assert np.all(np.diff(obf_spending(look_fractions(10))) > 0)


def test_obf_boundaries_match_textbook_five_looks():
    # gsDesign sfLDOF / ldbounds, two-sided alpha = 0.05, five equally spaced looks
    expected = [4.8769, 3.3569, 2.6803, 2.2898, 2.0310]
    assert obf_boundaries(look_fractions(5)) == pytest.approx(expected, abs=2e-3)


def test_obf_boundaries_single_look_is_fixed_test():
    assert obf_boundaries([1.0]) == pytest.approx([1.959964], abs=1e-5)


def test_obf_boundaries_unequal_and_unreachable_looks():
    c = obf_boundaries([1e-4, 0.5, 1.0])
    assert math.isinf(c[0])
    assert 2.5 < c[1] < 3.2 and 1.9 < c[2] < 2.1
    assert math.isinf(obf_boundaries([1e-4, 2e-4, 1.0])[1])


@pytest.mark.parametrize(
    "fractions, alpha",
    [([], 0.05), ([0.5, 0.4, 1.0], 0.05), ([0.5, 0.9], 0.05), ([0.0, 1.0], 0.05), ([1.0], 1.5)],
)
def test_obf_boundaries_validate(fractions, alpha):
    with pytest.raises(ValueError):
        obf_boundaries(fractions, alpha)


def test_look_fractions():
    assert look_fractions(4) == pytest.approx([0.25, 0.5, 0.75, 1.0])
    with pytest.raises(ValueError):
        look_fractions(0)


# --- paths and statistics ----------------------------------------------------------------


def test_simulate_paths_shape_and_totals():
    rng = np.random.default_rng(1)
    paths = simulate_paths(0.02, 0.1, 10_001, 7, 50, rng)
    assert paths.n_looks == 7 and paths.n_c.shape == (50, 7)
    assert np.all(paths.n_c[:, -1] == 5000) and np.all(paths.n_total == 10_000)
    assert np.all(np.diff(paths.n_c, axis=1) > 0)
    assert np.all(np.diff(paths.x_t, axis=1) >= 0)
    assert np.all(paths.x_c <= paths.n_c)


def test_simulate_paths_per_sim_arrays():
    rng = np.random.default_rng(2)
    paths = simulate_paths(np.array([0.01, 0.5]), 0.0, np.array([100, 2000]), 2, 2, rng)
    assert list(paths.n_c[:, -1]) == [50, 1000]


@pytest.mark.parametrize(
    "p_c, lift, n, looks, sims",
    [(0.0, 0, 100, 2, 5), (0.6, 1.0, 100, 2, 5), (0.1, 0, 10, 6, 5), (0.1, 0, 100, 2, 0)],
)
def test_simulate_paths_validate(p_c, lift, n, looks, sims):
    with pytest.raises(ValueError):
        simulate_paths(p_c, lift, n, looks, sims, np.random.default_rng(0))


def test_pooled_z_matches_scalar_test():
    x_c, n_c, x_t, n_t = (
        np.array([[30, 0]]),
        np.array([[1000, 50]]),
        np.array([[45, 0]]),
        np.array([[1100, 50]]),
    )
    z = pooled_z(x_c, n_c, x_t, n_t)
    assert z[0, 0] == pytest.approx(two_proportion_ztest(30, 1000, 45, 1100).z)
    assert z[0, 1] == 0.0


def test_msprt_p_value_closed_form_and_monotone():
    rng = np.random.default_rng(3)
    paths = simulate_paths(0.05, 0.3, 20_000, 10, 200, rng)
    tau2 = 1e-4
    p = msprt_p_values(paths, tau2)
    assert np.all(np.diff(p, axis=1) <= 0) and np.all((p > 0) & (p <= 1))
    pc, pt = paths.x_c[0, 0] / paths.n_c[0, 0], paths.x_t[0, 0] / paths.n_t[0, 0]
    v = pc * (1 - pc) / paths.n_c[0, 0] + pt * (1 - pt) / paths.n_t[0, 0]
    lam = math.sqrt(v / (v + tau2)) * math.exp(tau2 * (pt - pc) ** 2 / (2 * v * (v + tau2)))
    assert p[0, 0] == pytest.approx(min(1.0, 1 / lam))
    per_sim = msprt_p_values(paths, np.full(200, tau2))
    assert per_sim == pytest.approx(p)


def test_msprt_no_clicks_and_bad_tau():
    rng = np.random.default_rng(4)
    paths = simulate_paths(1e-9, 0.0, 200, 2, 3, rng)
    assert np.all(msprt_p_values(paths, 1e-4) == 1.0)
    with pytest.raises(ValueError):
        msprt_p_values(paths, 0.0)


# --- operating characteristics -----------------------------------------------------------


@pytest.fixture(scope="module")
def null_10_looks():
    return simulate(0.02, 0.0, 40_000, 10, n_sims=N_SIMS, seed=11)


def test_naive_peeking_inflates_false_positives(null_10_looks):
    assert null_10_looks["naive"].rejection_rate > 0.15
    assert within(null_10_looks["fixed"].rejection_rate, 0.05)


def test_obf_and_msprt_control_false_positives(null_10_looks):
    for rule in ("obf", "msprt"):
        assert within(null_10_looks[rule].rejection_rate, 0.05), rule


def test_power_and_sample_saved_under_a_real_lift():
    # n chosen so the fixed-n test has about 80% power for a 10% lift on a 2% CTR
    res = simulate(0.02, 0.1, 160_000, 10, n_sims=5_000, seed=12)
    fixed, obf, ms = res["fixed"], res["obf"], res["msprt"]
    assert 0.75 < fixed.rejection_rate < 0.85 and fixed.sample_saved == 0
    assert obf.rejection_rate > fixed.rejection_rate - 0.03
    assert obf.sample_saved > 0.15
    assert 0 < ms.rejection_rate < fixed.rejection_rate and ms.sample_saved > 0
    assert res["naive"].rejection_rate > fixed.rejection_rate
    assert obf.mean_n_used < obf.mean_n_total == fixed.mean_n_total


def test_simulate_rejects_unknown_rule():
    with pytest.raises(ValueError, match="unknown rule"):
        simulate(0.02, 0.0, 1000, 2, rules=("bayes",), n_sims=10)


# --- calibration -------------------------------------------------------------------------


def test_designs_from_rows_filters_and_derives():
    rows = [
        ("a", 4, 40_000, 600, 1),
        ("b", 2, 10_000, 100, 3),
        ("single", 1, 5_000, 50, 1),
        ("noclicks", 3, 5_000, 0, 1),
        ("tiny", 2, 4, 1, 5),
        ("zero_days", 2, 1000, 10, 0),
    ]
    d = {x.test_id: x for x in designs_from_rows(rows)}
    assert set(d) == {"a", "b", "zero_days"}
    assert d["a"] == TestDesign("a", 0.015, 20_000, 1)
    assert d["b"].n_looks == 3 and d["zero_days"].n_looks == 1


def test_draw_designs():
    ds = [TestDesign("a", 0.01, 1000, 1), TestDesign("b", 0.02, 2000, 2)]
    drawn = draw_designs(ds, 100, np.random.default_rng(0))
    assert len(drawn) == 100 and {x.test_id for x in drawn} == {"a", "b"}
    with pytest.raises(ValueError):
        draw_designs([], 1, np.random.default_rng(0))


@pytest.fixture(scope="module")
def designs():
    rng = np.random.default_rng(5)
    ctr = rng.uniform(0.005, 0.03, 200)
    n = rng.integers(10_000, 40_000, 200)
    looks = rng.choice([1, 1, 1, 2, 3], 200)
    return [
        TestDesign(str(i), c, int(k), int(m))
        for i, (c, k, m) in enumerate(zip(ctr, n, looks, strict=True))
    ]


def test_calibrated_null_sweep(designs):
    res = simulate_calibrated(designs, 0.0, n_sims=N_SIMS, n_looks=10, seed=6)
    assert res["naive"].rejection_rate > 0.15 and res["naive"].n_looks == 10
    assert within(res["obf"].rejection_rate, 0.05)
    assert within(res["msprt"].rejection_rate, 0.05)


def test_calibrated_empirical_looks_and_big_lift(designs):
    res = simulate_calibrated(designs, 0.5, n_sims=2_000, seed=7)
    assert res["fixed"].n_looks is None and res["fixed"].n_sims == 2_000
    assert res["fixed"].rejection_rate > 0.3
    high = [TestDesign("h", 0.8, 1000, 1)] + designs[:1]
    res = simulate_calibrated(high, 0.5, rules=("fixed",), n_sims=500, seed=8)
    assert res["fixed"].n_sims < 500  # the 0.8 CTR design cannot take a 50% lift


def test_calibrated_too_many_looks():
    with pytest.raises(ValueError, match="large enough"):
        simulate_calibrated([TestDesign("a", 0.1, 10, 1)], 0.0, n_sims=10, n_looks=20)


# --- analysis/peeking.py end to end on a tiny warehouse ------------------------------------


def _peeking():
    spec = importlib.util.spec_from_file_location("peeking", ROOT / "analysis" / "peeking.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_peeking_script(tmp_path):
    db = tmp_path / "w.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("create schema marts; create schema intermediate")
        con.execute(
            "create table marts.fct_tests as select * from (values "
            "('t1', 2, 20000, 300, false), ('t2', 3, 30000, 450, false), "
            "('t3', 2, 20000, 200, true)) "
            "v(clickability_test_id, n_arms, total_impressions, total_clicks, randomization_risk)"
        )
        con.execute(
            "create table intermediate.int_test_arms as select * from (values "
            "('t1', timestamp '2014-01-01 10:00'), ('t1', timestamp '2014-01-03 09:00'), "
            "('t2', timestamp '2014-02-01 10:00'), ('t2', timestamp '2014-02-01 10:05'), "
            "('t2', timestamp '2014-02-01 10:09'), ('t3', timestamp '2013-08-01 10:00'), "
            "('t3', timestamp '2013-08-01 11:00')) v(clickability_test_id, created_at)"
        )
    peeking = _peeking()
    with duckdb.connect(str(db), read_only=True) as con:
        ds = {d.test_id: d for d in peeking.load_designs(con)}
        assert set(ds) == {"t1", "t2"} and ds["t1"].n_looks == 3 and ds["t2"].n_looks == 1
        assert len(peeking.load_designs(con, include_risk=True)) == 3

    out = tmp_path / "out"
    argv = ["--db", str(db), "--out", str(out), "--looks", "1,5", "--lifts", "0,0.2"]
    assert peeking.main([*argv, "--sims", "300"]) == 0
    with (out / "peeking.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * 3 * 4  # lifts x (empirical + 2 sweeps) x rules
    assert {r["scenario"] for r in rows} == {"empirical", "sweep"}
    assert (out / "peeking_fpr.png").stat().st_size > 10_000
    assert peeking.main(["--db", str(tmp_path / "missing.duckdb")]) == 1
