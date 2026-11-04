"""analysis/portfolio.py end to end on a tiny warehouse shaped like the dbt marts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _portfolio():
    spec = importlib.util.spec_from_file_location("portfolio", ROOT / "analysis" / "portfolio.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def warehouse(tmp_path):
    rng = np.random.default_rng(3)
    arms = []
    for t in range(60):
        k = 2 + t % 4
        risk = t < 10
        n = rng.integers(3_000, 8_000, k)
        if risk and t % 2:
            n[0] = n[0] * 3  # sample ratio mismatch
        rates = 0.02 * np.exp(rng.normal(0, 0.2 if t % 3 else 0.0, k))
        x = rng.binomial(n, rates)
        win = int(np.argmax(x / n)) if t % 2 == 0 else -1
        start = f"{2013 + t % 3}-0{1 + t % 9}-15"
        arms += [
            (f"t{t:02d}", j + 1, start, int(x[j]), int(n[j]), j == win, risk, "h", "e")
            for j in range(k)
        ]
    arms.append(("lonely", 1, "2014-02-01", 3, 100, False, False, "h", "e"))
    db = tmp_path / "w.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("create schema marts; create schema intermediate")
        con.execute(
            "create table intermediate.int_test_arms(clickability_test_id varchar, "
            "arm_number int, created_at date, clicks int, impressions int, winner boolean, "
            "randomization_risk boolean, headline varchar, eyecatcher_id varchar)"
        )
        con.executemany("insert into intermediate.int_test_arms values (?,?,?,?,?,?,?,?,?)", arms)
        con.execute(
            """create table marts.fct_tests as
            select clickability_test_id, min(created_at) as start_date,
                bool_or(randomization_risk) as randomization_risk, count(*) as n_arms,
                sum(impressions) as total_impressions, sum(clicks) as total_clicks,
                count(*) filter (where winner) as n_declared_winners,
                min(arm_number) filter (where winner) as winner_arm_number
            from intermediate.int_test_arms group by 1"""
        )
        con.execute(
            """create table marts.fct_aa_pairs as
            select a.clickability_test_id || ':' || a.arm_number || '-' || b.arm_number
                    as aa_pair_id,
                a.clickability_test_id, a.arm_number as arm_a_number,
                b.arm_number as arm_b_number, a.clicks as clicks_a,
                a.impressions as impressions_a, b.clicks as clicks_b,
                b.impressions as impressions_b, a.arm_number = 1 as is_strict_aa,
                a.randomization_risk as randomization_risk
            from intermediate.int_test_arms a join intermediate.int_test_arms b
              on a.clickability_test_id = b.clickability_test_id
             and a.arm_number < b.arm_number
            where a.clickability_test_id in ('t03', 't06', 't09', 't12')"""
        )
    return db


def test_portfolio_script(warehouse, tmp_path):
    mod = _portfolio()
    out = tmp_path / "out"
    assert mod.main(["--db", str(warehouse), "--out", str(out)]) == 0
    for name in (
        "headline",
        "health",
        "power",
        "inference",
        "pvalues",
        "winners",
        "tests",
    ):
        assert (out / f"portfolio_{name}.csv").exists()
    for fig in ("pvalues", "mde", "winners", "health"):
        assert (out / f"portfolio_{fig}.png").stat().st_size > 10_000

    tests = pd.read_csv(out / "portfolio_tests.csv")
    assert len(tests) == 61
    assert (tests["status"] == "single_arm").sum() == 1
    # every risk-window test is flagged and none is clean
    assert tests.loc[tests["randomization_risk"], "flagged"].all()
    assert not tests.loc[tests["randomization_risk"], "clean"].any()
    assert tests.loc[tests["randomization_risk"], "srm"].sum() >= 4
    # BH q-values only on clean tests, survival flags only on clean tests
    assert tests.loc[~tests["clean"], "bh_q"].isna().all()
    assert not tests.loc[~tests["clean"], "survive_both"].any()
    assert (tests["survive_both"] <= tests["survive_within"]).all()

    top = pd.read_csv(out / "portfolio_headline.csv")
    assert list(top["rank"]) == [1, 2, 3, 4, 5]

    win = pd.read_csv(out / "portfolio_winners.csv")
    overall = win[win["dimension"] == "all"].iloc[0]
    clean = tests[tests["clean"]]
    assert overall["n_tests"] == len(clean)
    assert overall["n_with_winner"] == clean["has_winner"].sum()
    for dim in ("year", "size", "arms"):
        d = win[win["dimension"] == dim]
        assert d["n_with_winner"].sum() == overall["n_with_winner"]
        assert d["n_survive_both"].sum() == overall["n_survive_both"]

    inf = pd.read_csv(out / "portfolio_inference.csv")
    v = inf[inf["sample"] == "clean"].set_index("metric")["value"]
    assert v["n_tests"] == len(clean)
    assert v["n_bh_significant"] <= v["n_raw_significant"]
    assert 0 <= v["pi0"] <= 1

    hist = pd.read_csv(out / "portfolio_pvalues.csv")
    assert hist["n_tests"].sum() == len(clean)

    health = pd.read_csv(out / "portfolio_health.csv").set_index(["metric", "window"])
    assert health.loc[("srm_fail", "all"), "n"] == 60
    aa = health.loc[("aa_p05_all_pairs", "all")]
    assert aa["n"] == sum(k * (k - 1) // 2 for k in (5, 4, 3, 2))

    assert mod.main(["--db", str(tmp_path / "missing.duckdb")]) == 1


def test_mde_edge_cases():
    mod = _portfolio()
    assert np.isnan(mod._mde(0.0, 1000, 2))
    assert np.isinf(mod._mde(0.5, 1, 2))
    assert 0 < mod._mde(0.02, 5000, 2) < mod._mde(0.02, 5000, 4)


def test_findings_doc_is_generated_from_the_csvs(warehouse, tmp_path):
    out, doc = tmp_path / "out", tmp_path / "findings.md"
    assert _portfolio().main(["--db", str(warehouse), "--out", str(out), "--no-figures"]) == 0
    spec = importlib.util.spec_from_file_location(
        "portfolio_findings", ROOT / "analysis" / "portfolio_findings.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--out", str(out), "--doc", str(doc)]) == 0
    text = doc.read_text()
    inf = pd.read_csv(out / "portfolio_inference.csv")
    v = inf[inf["sample"] == "clean"].set_index("metric")["value"]
    win = pd.read_csv(out / "portfolio_winners.csv")
    w = win[win["dimension"] == "all"].iloc[0]
    assert f"pi0 = {v['pi0']:.2f}" in text
    assert f"{int(v['n_bh_significant']):,} after Benjamini-Hochberg" in text
    assert f"{100 * w['share_survive_both']:.1f}% of {int(w['n_with_winner']):,} declared" in text
    assert "## 4. Upworthy's declared winners" in text
    assert mod.main(["--out", str(tmp_path / "nothing"), "--doc", str(doc)]) == 1
