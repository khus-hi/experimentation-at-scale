"""Load every downloaded Upworthy CSV into DuckDB as raw.packages.

Each CSV in data/raw/ whose name mentions a released part (exploratory,
confirmatory, holdout) is read with read_csv_auto and stacked with
UNION ALL BY NAME, tagged with a source_part column. The script then prints
a summary and compares it with the paper (Matias et al. 2021).

Usage:
    .venv/bin/python -m ingest.load [--db data/upworthy.duckdb] [--raw data/raw]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

PARTS = ("exploratory", "confirmatory", "holdout")
PAPER_TESTS = 32_487
PAPER_ARMS = 150_817
RISK_HINTS = ("problem", "randomiz", "randomis", "imbalance")
TEST_ID = "clickability_test_id"


def find_csvs(raw: Path) -> list[tuple[str, Path]]:
    found = []
    for path in sorted(raw.iterdir()):
        name = path.name.lower()
        if not (name.endswith(".csv") or name.endswith(".csv.gz")):
            continue
        part = next((p for p in PARTS if p in name), None)
        if part:
            found.append((part, path))
    return found


def quote(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def build(con: duckdb.DuckDBPyConnection, csvs: list[tuple[str, Path]]) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    selects = [
        f"SELECT *, '{part}' AS source_part, '{path.name}' AS source_file "
        f"FROM read_csv_auto('{path.as_posix()}', sample_size=-1)"
        for part, path in csvs
    ]
    con.execute("CREATE OR REPLACE TABLE raw.packages AS " + " UNION ALL BY NAME ".join(selects))


def q(con: duckdb.DuckDBPyConnection, sql: str) -> list[tuple]:
    return con.execute(sql).fetchall()


def summarize(con: duckdb.DuckDBPyConnection) -> None:
    cols = q(
        con,
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema='raw' AND table_name='packages' ORDER BY ordinal_position",
    )
    names = [c for c, _ in cols]
    has_test = TEST_ID in names
    has_created = "created_at" in names
    test_expr = f"COUNT(DISTINCT {TEST_ID})" if has_test else "NULL"
    created = "TRY_CAST(created_at AS TIMESTAMP)" if has_created else "NULL"

    print("\n== raw.packages by source_part")
    print(f"{'part':<14}{'rows':>10}{'tests':>10}  {'created_at min':<20}  created_at max")
    rows = q(
        con,
        f"SELECT source_part, COUNT(*), {test_expr}, MIN({created}), MAX({created}) "
        "FROM raw.packages GROUP BY 1 ORDER BY 1",
    )
    for part, n, t, lo, hi in rows:
        print(f"{part:<14}{n:>10,}{(t or 0):>10,}  {str(lo):<20}  {hi}")
    total_rows, total_tests, lo, hi = q(
        con, f"SELECT COUNT(*), {test_expr}, MIN({created}), MAX({created}) FROM raw.packages"
    )[0]
    print(f"{'TOTAL':<14}{total_rows:>10,}{(total_tests or 0):>10,}  {str(lo):<20}  {hi}")
    missing = [p for p in PARTS if p not in {r[0] for r in rows}]
    if missing:
        print(f"parts not loaded: {', '.join(missing)}")

    print("\n== vs paper (Matias et al. 2021): 32,487 tests, 150,817 arms")
    print(f"tests: {total_tests or 0:,}  (diff {(total_tests or 0) - PAPER_TESTS:+,})")
    print(f"arms (rows): {total_rows:,}  (diff {total_rows - PAPER_ARMS:+,})")
    data_cols = [quote(c) for c in names if c not in ("source_part", "source_file")]
    dupes = q(
        con,
        f"SELECT COUNT(*) - COUNT(DISTINCT ({', '.join(data_cols)})) FROM raw.packages",
    )[0][0]
    print(f"exact duplicate rows: {dupes:,}")
    if has_test:
        overlap = q(
            con,
            f"SELECT COUNT(*) FROM (SELECT {TEST_ID} FROM raw.packages "
            f"GROUP BY 1 HAVING COUNT(DISTINCT source_part) > 1)",
        )[0][0]
        print(f"tests appearing in more than one part: {overlap:,}")
        nulls = q(con, f"SELECT COUNT(*) FROM raw.packages WHERE {TEST_ID} IS NULL")[0][0]
        print(f"rows with NULL {TEST_ID}: {nulls:,}")

    print("\n== randomization-risk column (June 2024 update)")
    risk_cols = [c for c in names if any(h in c.lower() for h in RISK_HINTS)]
    if not risk_cols:
        print(f"none found (looked for column names containing {', '.join(RISK_HINTS)})")
    for c in risk_cols:
        qc = quote(c)
        print(f"column {c!r}; value counts:")
        for val, n in q(
            con, f"SELECT {qc}, COUNT(*) FROM raw.packages GROUP BY 1 ORDER BY 2 DESC LIMIT 20"
        ):
            print(f"  {val!s:<30}{n:>10,} rows")
        flag = (
            f"COALESCE(TRY_CAST({qc} AS BOOLEAN), "
            f"lower(trim(CAST({qc} AS VARCHAR))) NOT IN ('', '0', 'false', 'no', 'none', 'na'))"
        )
        if has_test:
            for part, t, n in q(
                con,
                f"SELECT source_part, COUNT(DISTINCT {TEST_ID}), COUNT(*) FROM raw.packages "
                f"WHERE {qc} IS NOT NULL AND {flag} GROUP BY 1 ORDER BY 1",
            ):
                print(f"  flagged in {part}: {t:,} tests ({n:,} rows)")
            t = q(
                con,
                f"SELECT COUNT(DISTINCT {TEST_ID}) FROM raw.packages "
                f"WHERE {qc} IS NOT NULL AND {flag}",
            )[0][0]
            print(f"  flagged total: {t:,} tests")

    print("\n== columns (raw.packages)")
    for c, t in cols:
        print(f"  {c:<32}{t}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=Path("data/upworthy.duckdb"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    args = ap.parse_args(argv)

    csvs = find_csvs(args.raw) if args.raw.exists() else []
    if not csvs:
        msg = f"no part CSVs found in {args.raw}; run `make ingest` (download) first"
        print(msg, file=sys.stderr)
        return 1
    for part, path in csvs:
        print(f"loading {path.name} as {part}")
    with duckdb.connect(str(args.db)) as con:
        build(con, csvs)
        summarize(con)
    print(f"\nwrote raw.packages to {args.db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
