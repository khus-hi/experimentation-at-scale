"""Run the abtest library on every test in marts.fct_tests and summarise the portfolio.

No statistics are defined here; every test statistic comes from abtest. Four questions:

1. Data health. SRM (abtest.srm.srm_check, p < 0.001 vs an equal split) on every test, and
   a pooled z-test (abtest.proportions.two_proportion_ztest) on every A/A pair in
   marts.fct_aa_pairs, whose share at p < 0.05 should be about 5% if randomization works.
   Both inside vs outside the 2013-06-25..2014-01-10 randomization-risk window. A test is
   *flagged* when it is in the window or fails SRM; flagged tests are dropped from 2-4.
2. Power. For each test, the smallest relative lift detectable at 80% power
   (abtest.power.mde_for_n) at the test's pooled CTR and mean impressions per arm, with
   alpha split over the k - 1 comparisons (the library's convention) and, as a check,
   without the split. Then the share of tests powered for a 5%, 10% and 20% lift.
3. Portfolio inference. One chi-square omnibus p-value per test
   (abtest.proportions.chi2_omnibus), Storey's pi0 (abtest.multiple.storey_pi0), and the
   number significant at 0.05 raw vs after Benjamini-Hochberg (abtest.multiple.adjust).
4. Declared winners. The declared winner against every other arm with Holm inside the test
   (abtest.multiple.within_test with the winner as reference; it survives when it has the
   higher CTR in every comparison and all are rejected), and BH across tests on the omnibus
   p-values. The totals are checked against abtest.multiple.winner_survival. Broken down by
   year, by test size (quartiles of total impressions) and by number of arms.

Writes to --out (default analysis/output):
    portfolio_headline.csv     the five headline numbers
    portfolio_health.csv       SRM and A/A rates by risk window, with Wilson 95% intervals
    portfolio_power.csv        share of clean tests powered for each lift, MDE quantiles
    portfolio_inference.csv    pi0, raw and BH significant counts (clean and all tests)
    portfolio_pvalues.csv      histogram of omnibus p-values (20 bins)
    portfolio_winners.csv      winner survival overall, by year, size and arms
    portfolio_tests.csv        one row per test (gitignored; large)
    portfolio_pvalues.png, portfolio_mde.png, portfolio_winners.png, portfolio_health.png

Usage (after `make build`):
    .venv/bin/python analysis/portfolio.py --db data/upworthy.duckdb
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from abtest.multiple import adjust, storey_pi0, winner_survival, within_test
from abtest.power import mde_for_n
from abtest.proportions import chi2_omnibus, two_proportion_ztest, wilson_ci
from abtest.srm import SRM_THRESHOLD, srm_check

ALPHA = 0.05
POWER = 0.8
LIFTS = (0.05, 0.10, 0.20)
PI0_LAMBDAS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
N_BINS = 20

TESTS_SQL = """
select
    t.clickability_test_id,
    t.start_date,
    coalesce(t.randomization_risk, false)                   as randomization_risk,
    t.n_arms,
    t.total_impressions,
    t.total_clicks,
    t.n_declared_winners,
    t.winner_arm_number,
    list(a.arm_number order by a.arm_number)                as arm_numbers,
    list(a.clicks order by a.arm_number)                    as clicks,
    list(a.impressions order by a.arm_number)               as impressions
from marts.fct_tests as t
join intermediate.int_test_arms as a using (clickability_test_id)
group by all
order by 1
"""

AA_SQL = """
select
    aa_pair_id,
    clickability_test_id,
    arm_a_number,
    arm_b_number,
    clicks_a,
    impressions_a,
    clicks_b,
    impressions_b,
    is_strict_aa,
    coalesce(randomization_risk, false)                     as randomization_risk
from marts.fct_aa_pairs
order by clickability_test_id, arm_a_number, arm_b_number
"""

WINDOWS = {"all": None, "inside": True, "outside": False}


# --- per test ------------------------------------------------------------------------------


def _mde(baseline: float, n_per_arm: float, n_arms: int) -> float:
    """MDE at 80% power, inf when no lift is detectable, NaN when the CTR is 0 or 1."""
    if not 0 < baseline < 1:
        return math.nan
    try:
        return mde_for_n(baseline, n_per_arm, ALPHA, POWER, n_arms).relative_mde
    except ValueError:
        return math.inf


def analyze_tests(rows: list[tuple]) -> pd.DataFrame:
    """One row per test: SRM, omnibus p, MDE and the winner's within-test Holm result."""
    out = []
    for (
        test_id,
        start,
        risk,
        k,
        imps_total,
        clicks_total,
        n_winners,
        winner_arm,
        arm_numbers,
        clicks,
        imps,
    ) in rows:
        rec = {
            "clickability_test_id": test_id,
            "start_date": start,
            "year": pd.Timestamp(start).year if start is not None else None,
            "randomization_risk": bool(risk),
            "n_arms": k,
            "total_impressions": imps_total,
            "total_clicks": clicks_total,
            "n_declared_winners": n_winners,
            "status": "analyzed",
            "srm_p": math.nan,
            "srm": False,
            "omnibus_p": math.nan,
            "mde80": math.nan,
            "mde80_unadjusted": math.nan,
            "winner_index": None,
            "winner_holm_max_p": math.nan,
            "winner_beats_all": False,
            "survive_within": False,
        }
        if k < 2:
            rec["status"] = "single_arm"
        elif min(imps) <= 0:
            rec["status"] = "arm_without_impressions"
        if rec["status"] != "analyzed":
            rec["flagged"] = bool(risk)
            out.append(rec)
            continue
        s = srm_check(imps)
        rec["srm_p"], rec["srm"] = s.p_value, s.srm
        rec["omnibus_p"] = chi2_omnibus(clicks, imps)[2]
        baseline = clicks_total / imps_total
        rec["mde80"] = _mde(baseline, imps_total / k, k)
        rec["mde80_unadjusted"] = _mde(baseline, imps_total / k, 2)
        if winner_arm is not None:
            w = list(arm_numbers).index(winner_arm)
            res = within_test(clicks, imps, control=w, method="holm", alpha=ALPHA)
            beats = all(d < 0 for d in res.diffs)
            rec["winner_index"] = w
            rec["winner_holm_max_p"] = max(res.adjustment.p_adjusted)
            rec["winner_beats_all"] = beats
            rec["survive_within"] = beats and all(res.adjustment.reject)
        rec["flagged"] = bool(risk) or s.srm
        out.append(rec)
    df = pd.DataFrame(out)
    df["clean"] = (df["status"] == "analyzed") & ~df["flagged"]
    df["has_winner"] = df["winner_index"].notna()
    df["bh_q"] = math.nan
    df["bh_reject"] = False
    clean = df["clean"]
    if clean.any():
        a = adjust(df.loc[clean, "omnibus_p"].to_numpy(), "bh", ALPHA)
        df.loc[clean, "bh_q"] = a.p_adjusted
        df.loc[clean, "bh_reject"] = a.reject
    df["survive_portfolio"] = df["has_winner"] & df["bh_reject"]
    df["survive_both"] = df["survive_within"] & df["survive_portfolio"]
    df.loc[~clean, ["survive_within", "survive_portfolio", "survive_both"]] = False
    return df


def check_against_library(df: pd.DataFrame, rows: list[tuple]) -> None:
    """The per-test winner flags must add up to abtest.multiple.winner_survival's totals."""
    clean_ids = set(df.loc[df["clean"], "clickability_test_id"])
    items = []
    for r in rows:
        if r[0] not in clean_ids:
            continue
        arm_numbers, clicks, imps, winner_arm = r[8], r[9], r[10], r[7]
        w = None if winner_arm is None else list(arm_numbers).index(winner_arm)
        items.append((clicks, imps, w))
    if not items:
        return
    ws = winner_survival(items, alpha=ALPHA)
    c = df[df["clean"]]
    mine = (
        int(c["has_winner"].sum()),
        int(c["survive_within"].sum()),
        int(c["survive_portfolio"].sum()),
        int(c["survive_both"].sum()),
    )
    lib = (ws.n_with_winner, ws.survive_within, ws.survive_portfolio, ws.survive_both)
    if mine != lib:
        raise AssertionError(f"winner counts {mine} differ from winner_survival {lib}")


def analyze_aa(rows: list[tuple]) -> pd.DataFrame:
    """Pooled z-test on every A/A pair with impressions in both arms."""
    out = []
    for pid, test_id, a, b, xa, na, xb, nb, strict, risk in rows:
        if not na or not nb:
            continue
        out.append(
            {
                "aa_pair_id": pid,
                "clickability_test_id": test_id,
                "arm_a_number": a,
                "arm_b_number": b,
                "is_strict_aa": bool(strict),
                "randomization_risk": bool(risk),
                "p_value": two_proportion_ztest(xa, na, xb, nb).p_value,
            }
        )
    df = pd.DataFrame(
        out,
        columns=[
            "aa_pair_id",
            "clickability_test_id",
            "arm_a_number",
            "arm_b_number",
            "is_strict_aa",
            "randomization_risk",
            "p_value",
        ],
    )
    # The first pair of each test (lowest arm numbers), so no arm counts twice in a test.
    df["first_in_test"] = ~df.duplicated("clickability_test_id")
    return df


# --- summaries -----------------------------------------------------------------------------


def _rate(metric: str, window: str, k: int, n: int, expected: float | None) -> dict:
    ci = wilson_ci(k, n) if n else None
    return {
        "metric": metric,
        "window": window,
        "n": n,
        "k": k,
        "share": k / n if n else math.nan,
        "ci_low": ci.lower if ci else math.nan,
        "ci_high": ci.upper if ci else math.nan,
        "expected_share": expected,
    }


def _window(df: pd.DataFrame, window: str) -> pd.DataFrame:
    flag = WINDOWS[window]
    return df if flag is None else df[df["randomization_risk"] == flag]


def health(tests: pd.DataFrame, aa: pd.DataFrame) -> pd.DataFrame:
    rows = []
    checked = tests[tests["status"] == "analyzed"]
    for w in WINDOWS:
        t = _window(checked, w)
        rows.append(_rate("srm_fail", w, int(t["srm"].sum()), len(t), SRM_THRESHOLD))
    for w in WINDOWS:
        t = _window(tests, w)
        rows.append(_rate("flagged", w, int(t["flagged"].sum()), len(t), None))
    families = {
        "aa_p05_all_pairs": aa,
        "aa_p05_first_pair_per_test": aa[aa["first_in_test"]],
        "aa_p05_strict_pairs": aa[aa["is_strict_aa"]],
    }
    for metric, d in families.items():
        for w in WINDOWS:
            p = _window(d, w)["p_value"]
            rows.append(_rate(metric, w, int((p < ALPHA).sum()), len(p), ALPHA))
    clean_aa = aa[
        aa["clickability_test_id"].isin(tests.loc[tests["clean"], "clickability_test_id"])
    ]
    p = clean_aa["p_value"]
    rows.append(
        _rate("aa_p05_all_pairs_clean_tests", "outside", int((p < ALPHA).sum()), len(p), ALPHA)
    )
    return pd.DataFrame(rows)


def power_table(tests: pd.DataFrame) -> pd.DataFrame:
    c = tests[tests["clean"]]
    rows = []
    for col, correction in (("mde80", "bonferroni_k_minus_1"), ("mde80_unadjusted", "none")):
        m = c[col].dropna()
        q = m.replace(math.inf, np.nan).quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        for lift in LIFTS:
            rows.append(
                {
                    "correction": correction,
                    "target_lift": lift,
                    "n_tests": len(m),
                    "n_powered": int((m <= lift).sum()),
                    "share_powered": float((m <= lift).mean()) if len(m) else math.nan,
                    "mde_p10": q.loc[0.1],
                    "mde_p25": q.loc[0.25],
                    "mde_median": float(m.median()) if len(m) else math.nan,
                    "mde_p75": q.loc[0.75],
                    "mde_p90": q.loc[0.9],
                    "n_undetectable": int(np.isinf(m).sum()),
                }
            )
    return pd.DataFrame(rows)


def inference(tests: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    samples = {
        "clean": tests[tests["clean"]],
        "all_analyzed": tests[tests["status"] == "analyzed"],
    }
    for name, t in samples.items():
        p = t["omnibus_p"].to_numpy()
        if not len(p):
            continue
        bh = adjust(p, "bh", ALPHA)
        pi0 = storey_pi0(p, 0.5)
        n_raw = int((p < ALPHA).sum())
        vals = {
            "n_tests": len(p),
            "n_raw_significant": n_raw,
            "share_raw_significant": n_raw / len(p),
            "n_bh_significant": bh.n_rejected,
            "share_bh_significant": bh.n_rejected / len(p),
            "pi0": pi0,
            "share_with_real_difference": 1 - pi0,
            "expected_false_raw": pi0 * len(p) * ALPHA,
            "expected_false_share_of_raw": pi0 * len(p) * ALPHA / n_raw if n_raw else math.nan,
            "bh_fdr_bound": pi0 * ALPHA,
        }
        for lam in PI0_LAMBDAS:
            vals[f"pi0_lambda_{lam:.1f}"] = storey_pi0(p, lam)
        rows += [{"sample": name, "metric": k, "value": v} for k, v in vals.items()]
    c = samples["clean"]["omnibus_p"].to_numpy()
    edges = np.linspace(0, 1, N_BINS + 1)
    counts, _ = np.histogram(c, bins=edges)
    hist = pd.DataFrame({"bin_low": edges[:-1], "bin_high": edges[1:], "n_tests": counts})
    return pd.DataFrame(rows), hist


def _k(x: float) -> str:
    return f"{x / 1000:.0f}k" if x >= 10_000 else f"{x / 1000:.1f}k" if x >= 1000 else f"{x:.0f}"


def _size_labels(t: pd.DataFrame) -> pd.Series:
    q = t["total_impressions"].quantile([0, 0.25, 0.5, 0.75, 1]).to_numpy()
    edges = np.unique(q)
    if len(edges) < 2:
        return pd.Series("Q1", index=t.index)
    bins = pd.cut(t["total_impressions"], edges, include_lowest=True, labels=False)
    names = [f"Q{i + 1} ({_k(edges[i])}-{_k(edges[i + 1])})" for i in range(len(edges) - 1)]
    return bins.map(lambda b: names[int(b)])


def winners(tests: pd.DataFrame) -> pd.DataFrame:
    c = tests[tests["clean"]].copy()
    c["size"] = _size_labels(c) if len(c) else []
    c["arms"] = c["n_arms"].clip(upper=6).map(lambda k: "6+" if k >= 6 else str(k))
    c["all"] = "all"
    rows = []
    for dim in ("all", "year", "size", "arms"):
        for level, g in c.groupby(dim, sort=True):
            w = g[g["has_winner"]]
            n = len(w)
            rec = {
                "dimension": dim,
                "level": str(level),
                "n_tests": len(g),
                "n_with_winner": n,
                "share_with_winner": n / len(g) if len(g) else math.nan,
            }
            for col in ("survive_within", "survive_portfolio", "survive_both"):
                k = int(w[col].sum())
                rec[f"n_{col}"] = k
                rec[f"share_{col}"] = k / n if n else math.nan
            rec["n_winner_not_top_ctr"] = int((~w["winner_beats_all"]).sum())
            rows.append(rec)
    return pd.DataFrame(rows)


def headline(
    hl: pd.DataFrame, pw: pd.DataFrame, inf: pd.DataFrame, win: pd.DataFrame
) -> pd.DataFrame:
    def h(metric, window):
        return hl[(hl["metric"] == metric) & (hl["window"] == window)].iloc[0]

    def iv(metric):
        return inf[(inf["sample"] == "clean") & (inf["metric"] == metric)]["value"].iloc[0]

    srm_in, srm_out = h("srm_fail", "inside"), h("srm_fail", "outside")
    aa_in, aa_out = h("aa_p05_all_pairs", "inside"), h("aa_p05_all_pairs", "outside")
    p10 = pw[(pw["correction"] == "bonferroni_k_minus_1") & (pw["target_lift"] == 0.10)].iloc[0]
    w = win[win["dimension"] == "all"].iloc[0]
    rows = [
        (
            "srm_fail_inside_vs_outside",
            srm_in["share"],
            f"SRM fails in {srm_in['share']:.1%} of tests inside the risk window "
            f"({srm_in['k']:,}/{srm_in['n']:,}) vs {srm_out['share']:.1%} outside "
            f"({srm_out['k']:,}/{srm_out['n']:,}); A/A pairs at p < 0.05: "
            f"{aa_in['share']:.1%} inside vs {aa_out['share']:.1%} outside (5% expected)",
        ),
        (
            "share_powered_for_10pct_lift",
            p10["share_powered"],
            f"{p10['share_powered']:.1%} of {p10['n_tests']:,} clean tests could detect a 10% "
            f"lift at 80% power (median MDE {p10['mde_median']:.1%})",
        ),
        (
            "storey_pi0",
            iv("pi0"),
            f"Storey's pi0 = {iv('pi0'):.2f}: about {1 - iv('pi0'):.0%} of clean tests have "
            "some real difference between arms",
        ),
        (
            "bh_significant",
            iv("n_bh_significant"),
            f"{int(iv('n_raw_significant')):,} of {int(iv('n_tests')):,} clean tests "
            f"significant at p < 0.05 raw, {int(iv('n_bh_significant')):,} after BH at 5%",
        ),
        (
            "winners_surviving_both",
            w["share_survive_both"],
            f"{w['share_survive_both']:.1%} of {w['n_with_winner']:,} declared winners survive "
            f"Holm within the test and BH across tests ({w['share_survive_within']:.1%} Holm "
            f"only, {w['share_survive_portfolio']:.1%} BH only)",
        ),
    ]
    return pd.DataFrame(
        [{"rank": i + 1, "name": n, "value": v, "text": t} for i, (n, v, t) in enumerate(rows)]
    )


# --- figures -------------------------------------------------------------------------------

INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, GREEN = "#2a6fdb", "#e8833a", "#3a9d6a"


def _style(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ae")


def plot_all(
    out: Path,
    tests: pd.DataFrame,
    hl: pd.DataFrame,
    hist: pd.DataFrame,
    inf: pd.DataFrame,
    win: pd.DataFrame,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    paths = []

    # p-value histogram with the pi0 floor
    pi0 = inf[(inf["sample"] == "clean") & (inf["metric"] == "pi0")]["value"].iloc[0]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    width = hist["bin_high"] - hist["bin_low"]
    ax.bar(
        hist["bin_low"], hist["n_tests"], width=width, align="edge", color=BLUE, edgecolor=BG, lw=1
    )
    floor = pi0 * hist["n_tests"].sum() / len(hist)
    ax.axhline(floor, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.text(
        0.98,
        floor,
        f"tests with no real difference (pi0 = {pi0:.2f})",
        ha="right",
        va="bottom",
        color=INK,
        fontsize=9,
        transform=ax.get_yaxis_transform(),
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Omnibus p-value (chi-square across arms)")
    ax.set_ylabel("Tests")
    ax.set_title(
        "Distribution of p-values across clean tests\n"
        "Flat part = nulls; the spike near 0 = tests with real differences",
        loc="left",
        fontsize=10,
    )
    _style(ax)
    fig.tight_layout()
    paths.append(out / "portfolio_pvalues.png")
    fig.savefig(paths[-1], facecolor=BG)
    plt.close(fig)

    # MDE ECDF
    m = tests.loc[tests["clean"], "mde80"].replace(math.inf, np.nan).dropna().sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    if len(m):
        y = np.arange(1, len(m) + 1) / len(tests[tests["clean"]])
        ax.step(m.to_numpy(), y, where="post", color=BLUE, lw=2)
        for lift in LIFTS:
            share = float((tests.loc[tests["clean"], "mde80"] <= lift).mean())
            ax.axvline(lift, color=MUTED, lw=0.8, ls=(0, (4, 3)))
            ax.text(
                lift * 1.04,
                0.97,
                f"{lift:.0%} lift: {share:.1%} powered",
                rotation=90,
                ha="left",
                va="top",
                fontsize=8.5,
                color=INK,
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Minimum detectable relative lift at 80% power (log scale)")
    ax.set_ylabel("Share of clean tests")
    ax.set_title(
        "Most tests could only detect large lifts\n"
        "Pooled CTR, mean impressions per arm, alpha split over k - 1 comparisons",
        loc="left",
        fontsize=10,
    )
    _style(ax)
    fig.tight_layout()
    paths.append(out / "portfolio_mde.png")
    fig.savefig(paths[-1], facecolor=BG)
    plt.close(fig)

    # winner survival by year and by size
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=150, sharey=True)
    series = (
        ("share_survive_within", "Holm within test", BLUE),
        ("share_survive_portfolio", "BH across tests", ORANGE),
        ("share_survive_both", "both", GREEN),
    )
    for ax, dim, label in ((axes[0], "year", "Year"), (axes[1], "size", "Total impressions")):
        d = win[(win["dimension"] == dim) & (win["n_with_winner"] > 0)]
        x = np.arange(len(d))
        bw = 0.26
        for i, (col, name, color) in enumerate(series):
            ax.bar(x + (i - 1) * bw, d[col], width=bw, color=color, label=name, edgecolor=BG, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{lv}\nn={n:,}" for lv, n in zip(d["level"], d["n_with_winner"], strict=True)],
            fontsize=8,
        )
        ax.set_xlabel(label)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        _style(ax)
    axes[0].set_ylabel("Share of declared winners surviving")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(
        "How many of Upworthy's declared winners survive multiple-testing corrections",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    paths.append(out / "portfolio_winners.png")
    fig.savefig(paths[-1], facecolor=BG)
    plt.close(fig)

    # health: SRM and A/A inside vs outside
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    metrics = (("srm_fail", "SRM fail (p < 0.001)"), ("aa_p05_all_pairs", "A/A pairs p < 0.05"))
    x = np.arange(len(metrics))
    for i, (w, color) in enumerate((("inside", ORANGE), ("outside", BLUE))):
        vals, lo, hi = [], [], []
        for metric, _ in metrics:
            r = hl[(hl["metric"] == metric) & (hl["window"] == w)].iloc[0]
            vals.append(r["share"])
            lo.append(r["share"] - r["ci_low"])
            hi.append(r["ci_high"] - r["share"])
        bars = ax.bar(
            x + (i - 0.5) * 0.38,
            vals,
            width=0.38,
            color=color,
            edgecolor=BG,
            label=f"{w} risk window",
            yerr=[lo, hi],
            ecolor=MUTED,
            capsize=3,
        )
        for b, v in zip(bars, vals, strict=True):
            if not math.isnan(v):
                ax.annotate(
                    f"{v:.1%}",
                    (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color=INK,
                )
    ax.axhline(ALPHA, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(
        "Randomization broke inside the 2013-06-25..2014-01-10 window\n"
        "Dashed line: 5%, the A/A rate expected if randomization works",
        loc="left",
        fontsize=10,
    )
    _style(ax)
    fig.tight_layout()
    paths.append(out / "portfolio_health.png")
    fig.savefig(paths[-1], facecolor=BG)
    plt.close(fig)
    return paths


# --- main ----------------------------------------------------------------------------------


def run(con: duckdb.DuckDBPyConnection, out: Path, figures: bool = True) -> pd.DataFrame:
    rows = con.execute(TESTS_SQL).fetchall()
    tests = analyze_tests(rows)
    check_against_library(tests, rows)
    aa = analyze_aa(con.execute(AA_SQL).fetchall())
    hl = health(tests, aa)
    pw = power_table(tests)
    inf, hist = inference(tests)
    win = winners(tests)
    top = headline(hl, pw, inf, win)

    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "portfolio_headline.csv": top,
        "portfolio_health.csv": hl,
        "portfolio_power.csv": pw,
        "portfolio_inference.csv": inf,
        "portfolio_pvalues.csv": hist,
        "portfolio_winners.csv": win,
        "portfolio_tests.csv": tests,
    }
    for name, df in tables.items():
        df.to_csv(out / name, index=False, float_format="%.6g")
    if figures:
        plot_all(out, tests, hl, hist, inf, win)
    return top


def summary(tests_csv: Path) -> str:
    t = pd.read_csv(tests_csv)
    status = t["status"].value_counts().to_dict()
    return (
        f"tests: {len(t):,}; analyzed {status.get('analyzed', 0):,}, single-arm "
        f"{status.get('single_arm', 0):,}, arm without impressions "
        f"{status.get('arm_without_impressions', 0):,}; flagged {int(t['flagged'].sum()):,}; "
        f"clean {int(t['clean'].sum()):,}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("data/upworthy.duckdb"))
    ap.add_argument("--out", type=Path, default=Path("analysis/output"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    if not args.db.exists():
        print(f"{args.db} not found; run `make ingest` and `make build` first", file=sys.stderr)
        return 1
    with duckdb.connect(str(args.db), read_only=True) as con:
        top = run(con, args.out, figures=not args.no_figures)
    print(summary(args.out / "portfolio_tests.csv"))
    print()
    for r in top.itertuples():
        print(f"{r.rank}. {r.text}")
    print(f"\nwrote {args.out}/portfolio_*.csv and figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
