"""How much does peeking inflate false positives, and do sequential rules fix it?

A calibrated simulation: baseline CTR, test size and number of looks are drawn jointly from
real tests in marts.fct_tests, and the daily stream of impressions is simulated, because the
Upworthy archive has only final counts per arm (no daily counts). See abtest.sequential.

Writes:
    <out>/peeking.csv       one row per (scenario, rule, n_looks, lift)
    <out>/peeking_fpr.png   false positive rate vs number of looks for naive, OBF and mSPRT

Scenarios: "sweep" forces every simulated test to the same number of looks (1..20) with CTR
and n drawn from the data; "empirical" also draws each test's own number of looks, derived
as the calendar days spanned by its arms' created_at (a proxy; the archive has no end date).

Usage (after `make build`):
    .venv/bin/python analysis/peeking.py --db data/upworthy.duckdb
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import duckdb

from abtest.sequential import SimulationResult, TestDesign, designs_from_rows, simulate_calibrated

RULES = ("fixed", "naive", "obf", "msprt")
PLOTTED = {"naive": "Naive peeking", "obf": "OBF alpha spending", "msprt": "mSPRT (always valid)"}
COLORS = {"naive": "#eb6834", "obf": "#2a78d6", "msprt": "#1baf7a"}
MARKERS = {"naive": "o", "obf": "s", "msprt": "D"}

DESIGNS_SQL = """
select
    t.clickability_test_id,
    t.n_arms,
    t.total_impressions,
    t.total_clicks,
    date_diff('day', min(a.created_at)::date, max(a.created_at)::date) + 1 as days_spanned
from marts.fct_tests as t
join intermediate.int_test_arms as a using (clickability_test_id)
where {where}
group by 1, 2, 3, 4
order by 1
"""

FIELDS = ["scenario", *SimulationResult.__dataclass_fields__]


def load_designs(con: duckdb.DuckDBPyConnection, include_risk: bool = False) -> list[TestDesign]:
    """Read one calibration record per test (see designs_from_rows for the look proxy)."""
    where = "true" if include_risk else "not coalesce(t.randomization_risk, false)"
    return designs_from_rows(con.execute(DESIGNS_SQL.format(where=where)).fetchall())


def run(
    designs: list[TestDesign],
    looks: list[int],
    lifts: list[float],
    n_sims: int,
    seed: int,
    tau_rel: float,
) -> list[dict]:
    rows = []
    for lift in lifts:
        scenarios = [("empirical", None)] + [("sweep", k) for k in looks]
        for scenario, k in scenarios:
            res = simulate_calibrated(
                designs, lift, RULES, n_sims=n_sims, n_looks=k, tau_rel=tau_rel, seed=seed
            )
            rows += [{"scenario": scenario, **asdict(r)} for r in res.values()]
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def plot(rows: list[dict], path: Path, alpha: float = 0.05) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    null = [r for r in rows if r["scenario"] == "sweep" and r["lift"] == 0]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    ax.axhline(alpha, color="#52514e", lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(
        0.06,
        alpha - 0.004,
        "nominal 5%",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="top",
        color="#52514e",
        fontsize=9,
    )
    for rule, label in PLOTTED.items():
        pts = sorted(
            (r["n_looks"], r["rejection_rate"], r["se"]) for r in null if r["rule"] == rule
        )
        if not pts:
            continue
        x, y, se = zip(*pts, strict=True)
        ax.fill_between(
            x,
            [a - 1.96 * s for a, s in zip(y, se, strict=True)],
            [a + 1.96 * s for a, s in zip(y, se, strict=True)],
            color=COLORS[rule],
            alpha=0.15,
            lw=0,
        )
        ax.plot(
            x,
            y,
            color=COLORS[rule],
            lw=2,
            marker=MARKERS[rule],
            ms=6,
            label=label,
            markeredgecolor="#fcfcfb",
            markeredgewidth=1.5,
        )
        ax.annotate(
            f"{label}  {y[-1]:.1%}",
            (x[-1], y[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color="#0b0b0b",
        )
    ax.set_xlabel("Number of looks (daily batches)")
    ax.set_ylabel("False positive rate (no true lift)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    xs = sorted({r["n_looks"] for r in null})
    ax.set_xticks(xs)
    ax.set_ylim(bottom=-0.005)
    ax.set_xlim(0, max(xs) * 1.45)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ae")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title(
        "Peeking inflates false positives; OBF and mSPRT hold the line\n"
        "Calibrated simulation: CTR and n drawn from Upworthy tests, daily stream simulated",
        loc="left",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)


def report(designs: list[TestDesign], rows: list[dict]) -> str:
    looks = Counter(d.n_looks for d in designs)
    lines = [
        f"calibration tests: {len(designs):,}",
        "derived looks (days spanned by arms' created_at): "
        + ", ".join(f"{k}: {v:,}" for k, v in sorted(looks.items())[:8])
        + (" ..." if len(looks) > 8 else ""),
        "",
        f"{'scenario':<10}{'lift':>6}{'looks':>7}  {'rule':<6}{'reject':>8}{'se':>8}{'saved':>8}",
    ]
    for r in rows:
        k = "emp" if r["n_looks"] is None else str(r["n_looks"])
        lines.append(
            f"{r['scenario']:<10}{r['lift']:>6.2f}{k:>7}  {r['rule']:<6}"
            f"{r['rejection_rate']:>8.3f}{r['se']:>8.3f}{r['sample_saved']:>8.1%}"
        )
    return "\n".join(lines)


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("data/upworthy.duckdb"))
    ap.add_argument("--out", type=Path, default=Path("analysis/output"))
    ap.add_argument("--looks", default="1,2,3,5,7,10,15,20")
    ap.add_argument("--lifts", default="0,0.1,0.25,0.5")
    ap.add_argument("--sims", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=20170101)
    ap.add_argument("--tau-rel", type=float, default=0.1, help="mSPRT tau = tau_rel * CTR")
    ap.add_argument(
        "--include-risk",
        action="store_true",
        help="also use tests in the randomization-risk window",
    )
    args = ap.parse_args(argv)
    if not args.db.exists():
        print(f"{args.db} not found; run `make ingest` and `make build` first", file=sys.stderr)
        return 1
    with duckdb.connect(str(args.db), read_only=True) as con:
        designs = load_designs(con, args.include_risk)
    if not designs:
        print("no usable tests in marts.fct_tests", file=sys.stderr)
        return 1
    looks = [int(x) for x in _floats(args.looks)]
    rows = run(designs, looks, _floats(args.lifts), args.sims, args.seed, args.tau_rel)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / "peeking.csv")
    plot(rows, args.out / "peeking_fpr.png")
    print(report(designs, rows))
    print(f"\nwrote {args.out / 'peeking.csv'} and {args.out / 'peeking_fpr.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
