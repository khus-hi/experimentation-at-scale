"""Build the dbt project against a small synthetic archive and check the results.

The fixture CSVs use the real archive's header (a leading unnamed index column, which DuckDB
names column00) and go through ingest.load, so the dbt models see the same raw.packages shape
as on real data.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

from ingest import load

pytest.importorskip("dbt.cli.main")

ROOT = Path(__file__).resolve().parents[1]
DBT_DIR = ROOT / "dbt"
HEADER = (
    ",created_at,updated_at,clickability_test_id,excerpt,headline,lede,slug,eyecatcher_id,"
    "impressions,clicks,significance,first_place,winner,share_text,square,test_week"
)


def row(i, test, created, headline, eye, imp, clk, winner=False, excerpt="ex", lede="le"):
    return (
        f'{i},{created},{created},{test},{excerpt},"{headline}",{lede},slug,{eye},'
        f"{imp},{clk},50.0,{str(winner).upper()},{str(winner).upper()},share,,201301"
    )


EXPLORATORY = [
    # t1: three distinct arms, one declared winner, outside the risk window
    row(0, "t1", "2013-02-01 10:00:00", "Alpha", "e1", 1000, 10),
    row(1, "t1", "2013-02-01 10:01:00", "Beta", "e1", 1000, 20, winner=True),
    row(2, "t1", "2013-02-01 10:02:00", "Gamma", "e2", 1000, 30),
    # t4: a single arm (should warn, not fail)
    row(3, "t4", "2015-03-01 10:00:00", "Lonely", "e9", 500, 5),
]
CONFIRMATORY = [
    # t2: inside the risk window; arms 1 and 2 identical once whitespace is trimmed;
    # one exact duplicate row (index differs, content identical) to be deduped
    row(0, "t2", "2013-07-01 09:00:00", "Same", "e3", 800, 8),
    row(1, "t2", "2013-07-01 09:01:00", "  Same  ", "e3", 900, 9),
    row(2, "t2", "2013-07-01 09:02:00", "Other", "e3", 700, 14, winner=True),
    row(3, "t2", "2013-07-01 09:03:00", "Other 2", "e4", 600, 6),
    row(4, "t2", "2013-07-01 09:03:00", "Other 2", "e4", 600, 6),
    # t5: same headline but missing image ids: not treated as identical
    row(5, "t5", "2014-05-01 09:00:00", "NoImg", "", 400, 4),
    row(6, "t5", "2014-05-01 09:01:00", "NoImg", "", 400, 8),
]
HOLDOUT = [
    # t3: three identical arms -> 3 A/A pairs; arm 3 has a different lede, so only 1-2 is strict
    row(0, "t3", "2014-06-01 12:00:00", "Triple", "e5", 300, 3),
    row(1, "t3", "2014-06-01 12:01:00", "Triple", "e5", 300, 6),
    row(2, "t3", "2014-06-01 12:02:00", "Triple", "e5", 300, 9, lede="other lede"),
]


def write_fixture(raw: Path, risk_column: bool = False) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for part, rows in [
        ("exploratory", EXPLORATORY),
        ("confirmatory", CONFIRMATORY),
        ("holdout", HOLDOUT),
    ]:
        header, body = HEADER, rows
        if risk_column:
            header += ",problem"
            # flag only t5 (outside the date window) via the archive column
            body = [r + (",TRUE" if ",t5," in r else ",FALSE") for r in rows]
        (raw / f"fixture-{part}-packages.csv").write_text("\n".join([header, *body]) + "\n")


def dbt(db: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "UPWORTHY_DB": str(db), "DBT_PROFILES_DIR": str(DBT_DIR)}
    target = db.parent / "dbt_target"
    return subprocess.run(
        [sys.executable, "-m", "dbt.cli.main", *args, "--target-path", str(target)],
        cwd=DBT_DIR,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module", params=[False, True], ids=["no_risk_column", "risk_column"])
def built(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dbt")
    write_fixture(tmp / "raw", risk_column=request.param)
    db = tmp / "fixture.duckdb"
    assert load.main(["--db", str(db), "--raw", str(tmp / "raw")]) == 0
    result = dbt(db, "build")
    assert result.returncode == 0, result.stdout + result.stderr
    with duckdb.connect(str(db), read_only=True) as con:
        yield request.param, con, result.stdout


def q(con, sql):
    return con.execute(sql).fetchall()


def test_staging_keeps_every_row(built):
    _, con, _ = built
    assert q(con, "select count(*) from staging.stg_packages") == [(14,)]
    assert q(con, "select count(*) from staging.stg_packages where headline = 'Same'") == [(2,)]


def test_dedupe_and_arm_numbers(built):
    _, con, _ = built
    assert q(con, "select count(*) from intermediate.int_test_arms") == [(13,)]
    assert q(
        con,
        "select arm_number from intermediate.int_test_arms "
        "where clickability_test_id = 't2' order by 1",
    ) == [(1,), (2,), (3,), (4,)]


def test_fct_tests(built):
    _, con, _ = built
    rows = q(
        con,
        "select clickability_test_id, n_arms, total_impressions, best_ctr, worst_ctr, "
        "winner_arm_number, has_aa_pair, n_aa_pairs, n_duplicates_dropped "
        "from marts.fct_tests order by 1",
    )
    assert rows == [
        ("t1", 3, 3000, 0.03, 0.01, 2, False, 0, 0),
        ("t2", 4, 3000, 0.02, 0.01, 3, True, 1, 1),
        ("t3", 3, 900, 0.03, 0.01, None, True, 3, 0),
        ("t4", 1, 500, 0.01, 0.01, None, False, 0, 0),
        ("t5", 2, 800, 0.02, 0.01, None, False, 0, 0),
    ]


def test_randomization_risk(built):
    has_column, con, _ = built
    flagged = q(con, "select clickability_test_id from marts.fct_tests where randomization_risk")
    expected = {"t2", "t5"} if has_column else {"t2"}
    assert {r[0] for r in flagged} == expected


def test_aa_pairs(built):
    _, con, _ = built
    rows = q(
        con,
        "select aa_pair_id, impressions_a, impressions_b, clicks_a, clicks_b, is_strict_aa "
        "from marts.fct_aa_pairs order by 1",
    )
    assert rows == [
        ("t2:1-2", 800, 900, 8, 9, True),
        ("t3:1-2", 300, 300, 3, 6, True),
        ("t3:1-3", 300, 300, 3, 9, False),
        ("t3:2-3", 300, 300, 6, 9, False),
    ]
    assert q(
        con,
        "select clickability_test_id, n_strict_aa_pairs from marts.fct_tests "
        "where has_strict_aa_pair order by 1",
    ) == [("t2", 1), ("t3", 1)]


def test_single_arm_test_is_a_warning(built):
    _, _, stdout = built
    assert "WARN 1 fct_tests_has_at_least_two_arms" in stdout
