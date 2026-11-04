"""analysis/winners_curse.py and its findings doc, end to end on a synthetic warehouse."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
N_TESTS = 240


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod


HEADLINES = [
    "This Man Did Something Amazing For His Neighbors",
    "Why Would Anyone Do This To A Puppy?",
    "You Will Not Believe What 3 Kids Built",
    "A Quiet Story About A Terrible Loss",
    "Here Is What Your Senator Did Not Tell You About The Budget This Year",
    "Is This The Best 10 Minutes On The Internet?",
    "A Wonderful Gift",
    "Can You Guess Which Of These 5 Cities Is Happiest?",
    "The Awful Truth About 2 Hospitals",
    "We Lost Something Here",
]


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    """Questions cost 30% CTR; otherwise arm lifts are mostly zero with a few real ones."""
    rng = np.random.default_rng(5)
    arms = []
    for t in range(N_TESTS):
        k = 2 + t % 4
        risk = t < 20
        part = rng.choice(["exploratory", "confirmatory", "confirmatory", "holdout"])
        n = rng.multinomial(6_000 * k, [1 / k] * k)
        if risk and t % 2:
            n[0] *= 3
        if t == 25:
            n[0] *= 3  # SRM outside the risk window
        heads = list(rng.choice(HEADLINES, k))
        lift = np.where(rng.random(k) < 0.3, rng.normal(0, 0.25, k), 0.0)
        q = np.array([h.endswith("?") for h in heads])
        p = 0.02 * np.exp(lift) * np.where(q, 0.7, 1.0)
        x = rng.binomial(n, p)
        arms += [
            (f"t{t:03d}", j + 1, str(part), int(x[j]), int(n[j]), risk, str(heads[j]))
            for j in range(k)
        ]
    arms.append(("lonely", 1, "confirmatory", 3, 100, False, "Alone"))
    db = tmp_path_factory.mktemp("wc") / "w.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("create schema marts; create schema intermediate")
        con.execute(
            "create table intermediate.int_test_arms(clickability_test_id varchar, "
            "arm_number int, source_part varchar, clicks int, impressions int, "
            "randomization_risk boolean, headline varchar)"
        )
        con.executemany("insert into intermediate.int_test_arms values (?,?,?,?,?,?,?)", arms)
        con.execute(
            """create table marts.fct_tests as
            select clickability_test_id, min(source_part) as source_part,
                bool_or(randomization_risk) as randomization_risk
            from intermediate.int_test_arms group by 1"""
        )
    return db


@pytest.fixture(scope="module")
def outputs(warehouse, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    assert (
        _load("winners_curse").main(["--db", str(warehouse), "--out", str(out), "--seeds", "3"])
        == 0
    )
    return out


def test_outputs_and_sample(outputs):
    for name in ("headline", "seeds", "by_arms", "calibration", "oos", "features", "tests"):
        assert (outputs / f"winners_curse_{name}.csv").exists()
    for fig in ("lift", "calibration", "by_arms", "features"):
        assert (outputs / f"winners_curse_{fig}.png").stat().st_size > 10_000
    top = pd.read_csv(outputs / "winners_curse_headline.csv").set_index("metric")["mean"]
    assert top["tests_tests"] == N_TESTS + 1
    assert top["tests_risk"] == 20
    assert top["tests_lt2_arms"] == 1
    assert top["tests_srm"] >= 1
    clean = N_TESTS - 20 - top["tests_srm"]
    assert top["tests_clean"] == clean == top["n_tests"]
    tests = pd.read_csv(outputs / "winners_curse_tests.csv")
    assert len(tests) == clean and "t025" not in set(tests["clickability_test_id"])


def test_curse_and_shrinkage(outputs):
    seeds = pd.read_csv(outputs / "winners_curse_seeds.csv")
    assert list(seeds["seed"]) == [0, 1, 2]
    assert (seeds["mean_naive"] > seeds["mean_replicated"]).all()
    assert (seeds["kept"] < 1).all()
    assert (seeds["still_best"] > seeds["still_best_chance"]).all()
    # shrinkage beats the naive lift against the held-out half
    assert (seeds["mse_spike_slab"] < seeds["mse_naive"]).all()
    assert (seeds["mse_normal"] < seeds["mse_naive"]).all()
    assert np.allclose(seeds["mse_naive"] - seeds["noise_b"], seeds["mse_true_naive"])
    by = pd.read_csv(outputs / "winners_curse_by_arms.csv", dtype={"n_arms": str})
    assert list(by["n_arms"]) == ["2", "3", "4", "5"]
    assert by["n_tests"].sum() == seeds["n_tests"].iloc[0]
    cal = pd.read_csv(outputs / "winners_curse_calibration.csv")
    assert set(cal["method"]) == {"naive", "normal", "spike_slab"}
    assert (cal.groupby("method")["n"].sum() == 3 * seeds["n_tests"].iloc[0]).all()


def test_out_of_sample(outputs):
    oos = pd.read_csv(outputs / "winners_curse_oos.csv")
    assert set(oos["target"]) == {"confirmatory", "holdout"}
    pr = oos[oos["check"] == "predictive"]
    assert len(pr) == 8 and pr["coverage95"].between(0.5, 1).all()
    th = oos[(oos["check"] == "thinning") & (oos["target"] == "confirmatory")]
    assert set(th["fitted_on"]) == {"none", "exploratory", "confirmatory half A (in-sample)"}


def test_planted_question_effect_is_found(outputs):
    f = pd.read_csv(outputs / "winners_curse_features.csv")
    q = f[(f["feature"] == "is_question") & (f["model"] == "joint")].iloc[0]
    assert q["bh_reject"] and q["ctr_change_low"] < -0.3 < q["ctr_change_high"]
    assert (f.groupby("model").size() == 5).all()
    assert f["bh_q"].between(0, 1).all()


def test_findings_doc(outputs, tmp_path):
    doc = tmp_path / "wc.md"
    assert _load("winners_curse_findings").main(["--out", str(outputs), "--doc", str(doc)]) == 0
    text = doc.read_text()
    assert text.startswith("# The winner's curse")
    assert "## Headline numbers" in text and "1. **The half-A winner's lift" in text
    assert "is a question (?)" in text and "lower bound" in text
    assert _load("winners_curse_findings").main(["--out", str(tmp_path), "--doc", str(doc)]) == 1


def test_missing_db(tmp_path):
    assert _load("winners_curse").main(["--db", str(tmp_path / "nope.duckdb")]) == 1


def test_headline_features():
    f = _load("winners_curse").headline_features(
        pd.Series(["Why You Can't Miss These 7 Facts?", "A sad, terrible day", None])
    )
    assert f["n_words"].tolist() == [7, 4, 0]
    assert f["has_number"].tolist() == [1, 0, 0]
    assert f["is_question"].tolist() == [1, 0, 0]
    assert f["uses_you"].tolist() == [1, 0, 0]
    assert f["sentiment"].iloc[1] < 0
