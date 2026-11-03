"""Load tests/fixtures/packages_sample.csv into a DuckDB file as raw.packages.

CI uses this to build the dbt project on real (sampled) rows without touching OSF:
    python tests/fixtures/load_fixture.py --db "$RUNNER_TEMP/fixture.duckdb"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent
CSV = HERE / "packages_sample.csv"


def load(db: Path, csv: Path = CSV) -> tuple[int, int]:
    with duckdb.connect(str(db)) as con:
        con.execute("CREATE SCHEMA IF NOT EXISTS raw")
        con.execute(
            "CREATE OR REPLACE TABLE raw.packages AS SELECT * FROM "
            f"read_csv('{csv.as_posix()}', header=true, sample_size=-1)"
        )
        return con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT clickability_test_id) FROM raw.packages"
        ).fetchone()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--csv", type=Path, default=CSV)
    args = ap.parse_args(argv)
    if not args.csv.exists():
        print(f"{args.csv} not found; run tests/fixtures/make_fixture.py", file=sys.stderr)
        return 1
    rows, tests = load(args.db, args.csv)
    print(f"loaded {rows} arms from {tests} tests into {args.db} (raw.packages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
