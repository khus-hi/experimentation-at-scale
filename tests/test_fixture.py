"""The CI fixture: sampler and loader on a synthetic archive, plus checks on the committed CSV."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import duckdb
import pytest
from test_dbt import write_fixture

from ingest import load

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE = FIXTURES / "packages_sample.csv"


def _module(name):
    spec = importlib.util.spec_from_file_location(name, FIXTURES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


make_fixture = _module("make_fixture")
load_fixture = _module("load_fixture")


@pytest.fixture
def archive(tmp_path):
    write_fixture(tmp_path / "raw")
    db = tmp_path / "archive.duckdb"
    assert load.main(["--db", str(db), "--raw", str(tmp_path / "raw")]) == 0
    return db


def arms_by_test(db):
    with duckdb.connect(str(db), read_only=True) as con:
        return dict(
            con.execute(
                "SELECT clickability_test_id, COUNT(*) FROM raw.packages GROUP BY 1"
            ).fetchall()
        )


def test_sample_keeps_every_arm_and_roundtrips(archive, tmp_path):
    out = tmp_path / "sample.csv"
    n, rows, _, _ = make_fixture.make(archive, out, n=3, seed=1)
    assert n == 3
    assert load_fixture.load(tmp_path / "ci.duckdb", out) == (rows, 3)
    full, sample = arms_by_test(archive), arms_by_test(tmp_path / "ci.duckdb")
    assert all(full[t] == k for t, k in sample.items())
    with (
        duckdb.connect(str(archive), read_only=True) as a,
        duckdb.connect(str(tmp_path / "ci.duckdb"), read_only=True) as b,
    ):
        cols = "SELECT column_name FROM information_schema.columns WHERE table_name='packages'"
        assert a.execute(cols).fetchall() == b.execute(cols).fetchall()


def test_sample_is_deterministic(archive, tmp_path):
    make_fixture.make(archive, tmp_path / "a.csv", n=3, seed=7)
    make_fixture.make(archive, tmp_path / "b.csv", n=3, seed=7)
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


def test_sample_shrinks_to_size_limit(archive, tmp_path):
    out = tmp_path / "small.csv"
    n, _, _, size = make_fixture.make(archive, out, n=5, seed=1, max_bytes=1200)
    assert n < 5
    assert size == out.stat().st_size < 1200 or n == 1


@pytest.mark.skipif(not SAMPLE.exists(), reason="packages_sample.csv not generated yet")
def test_committed_sample(tmp_path):
    assert SAMPLE.stat().st_size < make_fixture.MAX_BYTES
    rows, tests = load_fixture.load(tmp_path / "ci.duckdb", SAMPLE)
    assert 150 <= tests <= make_fixture.N_TESTS
    with duckdb.connect(str(tmp_path / "ci.duckdb"), read_only=True) as con:
        parts = {
            r[0] for r in con.execute("SELECT DISTINCT source_part FROM raw.packages").fetchall()
        }
        dupes = con.execute(
            "SELECT COUNT(*) - COUNT(DISTINCT (source_part, column00)) FROM raw.packages"
        ).fetchone()[0]
    assert parts == {"exploratory", "confirmatory", "holdout"}
    assert dupes == 0
