"""How much of a winning headline's lift is real? Bayesian decisions and shrinkage per test.

Two analyses on marts.fct_tests and intermediate.int_test_arms (see abtest.bayes):

1. Beta-binomial decision rule, test by test: P(best) and expected loss for each arm under a
   uniform prior, then "ship the arm with the smallest expected loss if that loss is below
   --threshold of its click rate". Compared with the winners Upworthy declared.
2. Empirical Bayes shrinkage across tests: each test's best observed arm against the rest on
   the log-odds scale, a prior fitted on every arm against the rest (normal by method of
   moments, then spike at zero plus normal), and the shrunk lift of the picked arm vs its
   naive lift (the winner's curse). A prior fitted on the winners only is shown as the
   wrong way to do it.

Tests in the 2013-06-25..2014-01-10 randomization-risk window are excluded by default
(about 70% fail SRM); pass --include-risk to keep them.

Writes:
    <out>/shrinkage_summary.csv  one row per prior: winner's curse summary
    <out>/shrinkage_tests.csv    one row per test: naive and shrunk lifts, decision

Usage (after `make build`):
    .venv/bin/python analysis/shrinkage.py --db data/upworthy.duckdb
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

import duckdb
import numpy as np

from abtest.bayes import (
    BestArmShrinkage,
    arm_contrasts,
    decide,
    normal_loglik,
    prob_best,
    shrink_best_arms,
    winners_curse,
)

ARMS_SQL = """
select
    t.clickability_test_id,
    coalesce(t.randomization_risk, false)                    as randomization_risk,
    list(a.clicks order by a.arm_number)                     as clicks,
    list(a.impressions order by a.arm_number)                as impressions,
    list(coalesce(a.winner, false) order by a.arm_number)    as winner
from marts.fct_tests as t
join intermediate.int_test_arms as a using (clickability_test_id)
where {where}
group by 1, 2
order by 1
"""

FITS = (
    ("normal", "all", "normal prior, fit on all arms"),
    ("spike_slab", "all", "spike + normal, fit on all arms"),
    ("normal", "best", "normal prior, fit on winners (wrong)"),
)


def load_tests(con: duckdb.DuckDBPyConnection, include_risk: bool = False) -> list[tuple]:
    """(test_id, risk, clicks, impressions, winner flags) per test, arms in arm_number order."""
    where = "true" if include_risk else "not coalesce(t.randomization_risk, false)"
    return con.execute(ARMS_SQL.format(where=where)).fetchall()


def decisions(rows: list[tuple], threshold: float, n_draws: int, seed: int) -> list[dict]:
    """Beta-binomial decision for every test with 2 or more arms with impressions."""
    out = []
    for i, (test_id, _, clicks, imps, winner) in enumerate(rows):
        if sum(n > 0 for n in imps) < 2:
            continue
        res = prob_best(clicks, imps, n_draws=n_draws, seed=seed + i)
        d = decide(res, threshold, relative=True)
        declared = [j for j, w in enumerate(winner) if w]
        out.append(
            {
                "test_id": test_id,
                "chosen_arm": d.arm,
                "prob_best": d.prob_best,
                "rel_loss": d.loss,
                "ship": d.ship,
                "declared_arm": declared[0] if declared else None,
                "declared_prob_best": res.prob_best[declared[0]] if declared else None,
            }
        )
    return out


def fit_all(rows: list[tuple]) -> dict[str, BestArmShrinkage]:
    tests = [(c, n) for _, _, c, n, _ in rows]
    return {label: shrink_best_arms(tests, prior, fit_on) for prior, fit_on, label in FITS}


def summary_rows(fits: dict[str, BestArmShrinkage]) -> list[dict]:
    out = []
    for label, res in fits.items():
        p = res.prior
        out.append(
            {
                "fit": label,
                "n_fit": res.n_fit,
                "prior_pi0": getattr(p, "pi0", 0.0),
                "prior_mu": p.mu,
                "prior_tau": p.tau,
                **asdict(winners_curse(res)),
            }
        )
    return out


def test_rows(rows: list[tuple], fits: dict[str, BestArmShrinkage], dec: list[dict]) -> list[dict]:
    normal, spike = fits[FITS[0][2]], fits[FITS[1][2]]
    by_test = {d["test_id"]: d for d in dec}
    out = []
    for k, i in enumerate(normal.test_index):
        test_id, risk, clicks, _, _ = rows[i]
        s, n = spike.shrunk, normal.shrunk
        d = by_test.get(test_id, {})
        out.append(
            {
                "test_id": test_id,
                "randomization_risk": risk,
                "n_arms": len(clicks),
                "best_arm": int(normal.arm[k]),
                "naive_log_or": s.d[k],
                "se": s.se[k],
                "normal_mean": n.mean[k],
                "normal_lower": n.lower[k],
                "normal_upper": n.upper[k],
                "spike_mean": s.mean[k],
                "spike_lower": s.lower[k],
                "spike_upper": s.upper[k],
                "spike_p_null": s.p_null[k],
                "bayes_chosen_arm": d.get("chosen_arm"),
                "bayes_prob_best": d.get("prob_best"),
                "bayes_rel_loss": d.get("rel_loss"),
                "bayes_ship": d.get("ship"),
                "declared_arm": d.get("declared_arm"),
            }
        )
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _pct(k: float, n: float) -> str:
    return f"{100 * k / n:.1f}%" if n else "-"


def report(
    rows: list[tuple], dec: list[dict], fits: dict[str, BestArmShrinkage], threshold: float
) -> str:
    n_ship = sum(d["ship"] for d in dec)
    declared = [d for d in dec if d["declared_arm"] is not None]
    agree = sum(d["chosen_arm"] == d["declared_arm"] for d in declared)
    pb = np.array([d["declared_prob_best"] for d in declared], dtype=float)
    lines = [
        f"tests: {len(rows):,} loaded, {len(dec):,} with 2+ arms",
        "",
        f"1. Beta-binomial decision rule (uniform prior, ship if expected loss < "
        f"{threshold:.1%} of the arm's CTR)",
        f"   ships: {n_ship:,} of {len(dec):,} ({_pct(n_ship, len(dec))}); "
        f"keeps testing: {len(dec) - n_ship:,}",
        f"   tests with a declared winner: {len(declared):,}; rule picks the same arm in "
        f"{agree:,} ({_pct(agree, len(declared))})",
    ]
    if pb.size:
        lines.append(
            f"   P(best) of declared winners: median {np.median(pb):.2f}, "
            f">= 0.95 in {_pct((pb >= 0.95).sum(), pb.size)}, "
            f"< 0.5 in {_pct((pb < 0.5).sum(), pb.size)}"
        )
    lines += [
        "",
        "2-3. Empirical Bayes shrinkage of the best observed arm vs the rest (log odds ratio)",
        f"   {'prior':<38}{'pi0':>6}{'mu':>8}{'tau':>7}{'naive':>8}{'shrunk':>8}"
        f"{'kept':>7}{'naive%':>8}{'shrunk%':>8}{'excl0 raw':>11}{'excl0 EB':>10}",
    ]
    for r in summary_rows(fits):
        lines.append(
            f"   {r['fit']:<38}{r['prior_pi0']:>6.2f}{r['prior_mu']:>8.3f}{r['prior_tau']:>7.3f}"
            f"{r['mean_naive']:>8.3f}{r['mean_shrunk']:>8.3f}{r['retained']:>7.0%}"
            f"{r['mean_naive_rel']:>8.1%}{r['mean_shrunk_rel']:>8.1%}"
            f"{r['share_interval_excludes_zero_naive']:>11.1%}"
            f"{r['share_interval_excludes_zero_shrunk']:>10.1%}"
        )
    spike = fits[FITS[1][2]]
    normal = fits[FITS[0][2]]
    contrasts = [(c, n) for _, _, c, n, _ in rows]
    d, se = zip(*((c.d, c.se) for cl, im in contrasts for c in arm_contrasts(cl, im)), strict=True)
    gain = spike.prior.loglik - normal_loglik(normal.prior, d, se)
    lines += [
        "",
        f"   spike+normal vs normal log likelihood gain: {gain:,.1f} (on {len(d):,} arm contrasts)",
        "   naive/shrunk: mean log OR of the picked arm; kept = shrunk / naive;",
        "   %: exp(mean) - 1 as relative lift; excl0: share of picked arms whose 95% interval",
        "   excludes zero (raw Wald vs posterior).",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("data/upworthy.duckdb"))
    ap.add_argument("--out", type=Path, default=Path("analysis/output"))
    ap.add_argument("--threshold", type=float, default=0.01, help="relative expected loss")
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20140101)
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
        rows = load_tests(con, args.include_risk)
    dec = decisions(rows, args.threshold, args.draws, args.seed)
    fits = fit_all(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(summary_rows(fits), args.out / "shrinkage_summary.csv")
    write_csv(test_rows(rows, fits, dec), args.out / "shrinkage_tests.csv")
    print(report(rows, dec, fits, args.threshold))
    return 0


if __name__ == "__main__":
    sys.exit(main())
