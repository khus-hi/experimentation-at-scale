"""Run the SRM check and the chi-square omnibus test on every test in marts.fct_tests.

Prints counts only: how many tests fail SRM (p < 0.001, equal split expected) and how many
have an omnibus p < 0.05, overall and inside vs outside the randomization-risk window.
Single-arm tests have nothing to compare, so they are counted and skipped.

Usage (after `make build`):
    .venv/bin/python analysis/srm_scan.py --db data/upworthy.duckdb
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from abtest.proportions import chi2_omnibus
from abtest.srm import SRM_THRESHOLD, srm_check

OMNIBUS_ALPHA = 0.05

ARMS_SQL = """
select
    t.clickability_test_id,
    coalesce(t.randomization_risk, false)               as randomization_risk,
    list(a.impressions order by a.arm_number)           as impressions,
    list(a.clicks order by a.arm_number)                as clicks
from marts.fct_tests as t
join intermediate.int_test_arms as a using (clickability_test_id)
group by 1, 2
"""


@dataclass
class Counts:
    tests: int = 0
    single_arm: int = 0
    no_impressions: int = 0
    checked: int = 0
    srm: int = 0
    omnibus: int = 0
    omnibus_without_srm: int = 0


@dataclass
class Scan:
    groups: dict[str, Counts] = field(
        default_factory=lambda: {"all": Counts(), "inside": Counts(), "outside": Counts()}
    )


def scan(rows: Iterable[tuple[str, bool, list[int], list[int]]]) -> Scan:
    """Tally SRM and omnibus results per risk group from (test, risk, impressions, clicks)."""
    out = Scan()
    for _, risk, impressions, clicks in rows:
        targets = [out.groups["all"], out.groups["inside" if risk else "outside"]]
        for c in targets:
            c.tests += 1
        if len(impressions) < 2:
            for c in targets:
                c.single_arm += 1
            continue
        if sum(impressions) == 0:
            for c in targets:
                c.no_impressions += 1
            continue
        srm = srm_check(impressions).srm
        _, _, p = chi2_omnibus(clicks, impressions)
        for c in targets:
            c.checked += 1
            c.srm += srm
            c.omnibus += p < OMNIBUS_ALPHA
            c.omnibus_without_srm += (p < OMNIBUS_ALPHA) and not srm
    return out


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.1f}%" if n else "-"


def report(result: Scan) -> str:
    labels = {"all": "all tests", "inside": "inside risk window", "outside": "outside risk window"}
    rows = [("", "tests", "single-arm", "checked", "SRM fail", "%", "omnibus p<.05", "%")]
    for key, label in labels.items():
        c = result.groups[key]
        rows.append(
            (
                label,
                f"{c.tests:,}",
                f"{c.single_arm:,}",
                f"{c.checked:,}",
                f"{c.srm:,}",
                pct(c.srm, c.checked),
                f"{c.omnibus:,}",
                pct(c.omnibus, c.checked),
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = [
        f"SRM: chi-square goodness of fit vs equal split, flag at p < {SRM_THRESHOLD}",
        f"omnibus: chi-square on clicks x arms, p < {OMNIBUS_ALPHA}",
        "",
    ]
    for r in rows:
        lines.append(
            "  ".join(
                [r[0].ljust(widths[0])]
                + [v.rjust(w) for v, w in zip(r[1:], widths[1:], strict=True)]
            )
        )
    a = result.groups["all"]
    lines += [
        "",
        f"omnibus p<.05 among tests that pass SRM: {a.omnibus_without_srm:,}",
    ]
    if a.no_impressions:
        lines.append(f"skipped with zero impressions: {a.no_impressions:,}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("data/upworthy.duckdb"))
    args = ap.parse_args(argv)
    if not args.db.exists():
        print(f"{args.db} not found; run `make ingest` and `make build` first", file=sys.stderr)
        return 1
    with duckdb.connect(str(args.db), read_only=True) as con:
        rows = con.execute(ARMS_SQL).fetchall()
    print(report(scan(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
