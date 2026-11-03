"""Sample a small, real slice of the Upworthy archive for CI.

Picks about 200 tests at random (fixed seed) from raw.packages and writes every arm of those
tests, with all raw columns (column00, source_part, source_file included), to
tests/fixtures/packages_sample.csv. If the file would be 1 MB or more, tests are dropped from
the end of the shuffled order until it fits, so the output stays deterministic.

Needs the real data, so run it where `make ingest` has been run:
    .venv/bin/python tests/fixtures/make_fixture.py [--db data/upworthy.duckdb]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent
OUT = HERE / "packages_sample.csv"
SEED = 20210195
N_TESTS = 200
MAX_BYTES = 1_000_000
TEST_ID = "clickability_test_id"


def pick_tests(con: duckdb.DuckDBPyConnection, n: int, seed: int) -> list[str]:
    ids = [
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT {TEST_ID} FROM raw.packages WHERE {TEST_ID} IS NOT NULL ORDER BY 1"
        ).fetchall()
    ]
    random.Random(seed).shuffle(ids)
    return ids[:n]


def write_sample(con: duckdb.DuckDBPyConnection, ids: list[str], out: Path) -> int:
    con.execute("CREATE OR REPLACE TEMP TABLE fixture_ids (id VARCHAR)")
    con.executemany("INSERT INTO fixture_ids VALUES (?)", [(i,) for i in ids])
    con.execute(
        f"COPY (SELECT p.* FROM raw.packages p "
        f"JOIN fixture_ids f ON p.{TEST_ID} = f.id "
        f"ORDER BY p.source_part, p.{TEST_ID}, p.column00) "
        f"TO '{out.as_posix()}' (HEADER, DELIMITER ',')"
    )
    return out.stat().st_size


def make(db: Path, out: Path, n: int = N_TESTS, seed: int = SEED, max_bytes: int = MAX_BYTES):
    with duckdb.connect(str(db), read_only=True) as con:
        ids = pick_tests(con, n, seed)
        size = write_sample(con, ids, out)
        while size >= max_bytes and len(ids) > 1:
            ids = ids[: max(1, int(len(ids) * 0.95))]
            size = write_sample(con, ids, out)
        rows, parts = con.execute(
            f"SELECT COUNT(*), string_agg(DISTINCT source_part, ', ' ORDER BY source_part) "
            f"FROM raw.packages p JOIN fixture_ids f ON p.{TEST_ID} = f.id"
        ).fetchone()
    return len(ids), rows, parts, size


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=Path("data/upworthy.duckdb"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tests", type=int, default=N_TESTS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    if not args.db.exists():
        print(f"{args.db} not found; run `make ingest` first", file=sys.stderr)
        return 1
    n, rows, parts, size = make(args.db, args.out, args.tests, args.seed)
    print(f"wrote {args.out}: {n} tests, {rows} arms, parts: {parts}, {size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
