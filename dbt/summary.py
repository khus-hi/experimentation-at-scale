"""Print what `dbt build` produced: tests and arms by year and by n_arms, plus data checks.

Runs after `dbt build` in `make build`, also when the build failed, so the diagnostics show why.

Usage:
    .venv/bin/python dbt/summary.py [--db data/upworthy.duckdb]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb


def table(con: duckdb.DuckDBPyConnection, title: str, sql: str) -> None:
    rel = con.sql(sql)
    names, rows = rel.columns, rel.fetchall()
    cells = [
        [f"{v:,}" if isinstance(v, int) else ("" if v is None else str(v)) for v in r] for r in rows
    ]
    widths = [
        max(len(n), *(len(c[i]) for c in cells)) if cells else len(n) for i, n in enumerate(names)
    ]
    print(f"\n== {title}")
    print("  ".join(n.rjust(w) for n, w in zip(names, widths, strict=True)))
    for c in cells:
        print("  ".join(v.rjust(w) for v, w in zip(c, widths, strict=True)))


def exists(con: duckdb.DuckDBPyConnection, schema: str, name: str) -> bool:
    return bool(
        con.execute(
            "select count(*) from information_schema.tables "
            "where table_schema = ? and table_name = ?",
            [schema, name],
        ).fetchall()[0][0]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=Path("data/upworthy.duckdb"))
    args = ap.parse_args(argv)
    if not args.db.exists():
        print(f"{args.db} not found; run `make ingest` first", file=sys.stderr)
        return 1

    with duckdb.connect(str(args.db), read_only=True) as con:
        if exists(con, "staging", "stg_packages"):
            table(
                con,
                "staging checks (rows that would fail the stg_packages tests)",
                """
                select
                    count(*)                                                 as rows,
                    count(*) filter (where impressions is null)              as null_impressions,
                    count(*) filter (where impressions = 0)                  as zero_impressions,
                    count(*) filter (where impressions < 0)                  as neg_impressions,
                    count(*) filter (where clicks > impressions)             as clicks_gt_imps,
                    count(*) filter (where created_at is null)               as null_created_at,
                    count(*) filter (where clickability_test_id is null)     as null_test_id,
                    count(*) - count(distinct package_id)                    as dup_package_ids,
                    count(*) filter (where randomization_risk)               as risk_rows,
                    coalesce(max(randomization_risk_source_column), 'none')   as risk_column_used
                from staging.stg_packages
            """,
            )

        if not exists(con, "marts", "fct_tests"):
            print("\nmarts.fct_tests not built; see the dbt errors above")
            return 1

        table(
            con,
            "totals",
            """
            select
                count(*)                                       as tests,
                sum(n_arms)                                    as arms,
                sum(n_duplicates_dropped)                      as duplicates_dropped,
                count(*) filter (where n_arms < 2)             as single_arm_tests,
                count(*) filter (where n_declared_winners = 0) as tests_no_winner,
                count(*) filter (where n_declared_winners > 1) as tests_multi_winner,
                count(*) filter (where randomization_risk)     as risk_tests,
                count(*) filter (where has_aa_pair)            as tests_with_aa,
                sum(n_aa_pairs)                                as aa_pairs,
                count(*) filter (where has_strict_aa_pair)     as tests_strict_aa,
                sum(n_strict_aa_pairs)                         as strict_aa_pairs
            from marts.fct_tests
        """,
        )

        table(
            con,
            "tests and arms by year (of start_date)",
            """
            select
                cast(year(start_date) as varchar)            as year,
                count(*)                                       as tests,
                sum(n_arms)                                    as arms,
                round(avg(n_arms), 2)                          as arms_per_test,
                count(*) filter (where randomization_risk)     as risk_tests,
                count(*) filter (where has_aa_pair)            as tests_with_aa,
                sum(n_aa_pairs)                                as aa_pairs,
                sum(n_strict_aa_pairs)                         as strict_aa_pairs,
                sum(total_impressions)                         as impressions
            from marts.fct_tests
            group by all
            order by 1
        """,
        )

        table(
            con,
            "tests and arms by n_arms",
            """
            select
                n_arms,
                count(*)                                       as tests,
                sum(n_arms)                                    as arms,
                round(100.0 * count(*) / sum(count(*)) over (), 1) as pct_tests,
                count(*) filter (where randomization_risk)     as risk_tests,
                count(*) filter (where has_aa_pair)            as tests_with_aa,
                sum(n_aa_pairs)                                as aa_pairs,
                sum(n_strict_aa_pairs)                         as strict_aa_pairs,
                round(median(impressions_per_arm))::bigint     as median_imps_per_arm
            from marts.fct_tests
            group by all
            order by 1
        """,
        )

        table(
            con,
            "tests by source_part",
            """
            select source_part, count(*) as tests, sum(n_arms) as arms,
                   count(*) filter (where has_aa_pair) as tests_with_aa,
                   sum(n_aa_pairs) as aa_pairs,
                   sum(n_strict_aa_pairs) as strict_aa_pairs
            from marts.fct_tests group by all order by 1
        """,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
