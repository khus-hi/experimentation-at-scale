"""How much does a winning headline's lift shrink on fresh data? Data thinning on every test.

No test statistic is defined here; the split, the contrasts and the priors come from
abtest.thinning and abtest.bayes. Sample: *clean* tests, as in analysis/portfolio.py
(outside the 2013-06-25..2014-01-10 randomization-risk window, no SRM at p < 0.001,
2 or more arms, every arm with impressions).

1. Thinning replication. For each of --seeds seeds, every arm's impressions are split 50/50
   and its clicks drawn hypergeometrically (abtest.thinning.split_binomial). The arm with
   the highest CTR on half A is the "winner"; its log odds ratio against the rest of its
   test on half A is the naive lift, the same contrast on half B is the honest replication.
2. Winner's curse. Average naive vs replicated lift (kept = replicated / naive, and the
   overstatement naive / replicated - 1), and how often the half-A winner is still the best
   arm on half B (vs 1 / k by chance), overall and by number of arms.
3. Shrinkage. On each seed an empirical Bayes prior (normal, and spike at zero + normal) is
   fitted on every arm's half-A contrast and used to shrink the winner's half-A lift
   (abtest.bayes.shrink). Naive and shrunk predictions are scored against half B: MSE,
   bias (mean prediction minus mean half B), calibration slope (regression of half B on
   the prediction; 1 is calibrated; median over seeds, NaN when the predictions barely
   vary) and a
   binned calibration table. Half B is noisy, but its noise is independent of anything
   computed from half A, so it adds the same constant, mean(se_B^2), to every method's
   MSE; MSE minus that constant estimates the MSE against the true lift.
4. Out-of-sample prior. A prior fitted on all arms of the exploratory tests (full data) is
   checked on the confirmatory tests (and on the holdout tests): predictive coverage and
   log score of their contrasts, and the thinning score of step 3 with the exploratory
   prior in place of the in-sample one.
5. Headline features, descriptive. Every arm's empirical logit CTR is regressed on features
   of its headline (words, has a digit, is a question, uses "you", VADER compound
   sentiment) with test fixed effects (everything demeaned within test, so only arms of the
   same test are compared) and standard errors clustered by test; BH across features.

Caveat: thinning assumes binomial clicks. Natural A/A pairs in the archive are significant
about 13% of the time instead of 5%, so arms carry extra-binomial shocks that both halves
share. Selection on those shocks survives into half B, so the measured curse is a lower
bound.

Writes to --out (default analysis/output):
    winners_curse_headline.csv     headline numbers (mean over seeds, with seed min/max)
    winners_curse_seeds.csv        one row per seed
    winners_curse_by_arms.csv      curse and still-best rate by number of arms
    winners_curse_calibration.csv  binned predicted vs half-B lift, per method
    winners_curse_oos.csv          exploratory prior on confirmatory/holdout tests
    winners_curse_features.csv     within-test headline feature associations
    winners_curse_tests.csv        one row per test, averaged over seeds (gitignored)
    winners_curse_lift.png, winners_curse_calibration.png, winners_curse_by_arms.png,
    winners_curse_features.png

Usage (after `make build`):
    .venv/bin/python analysis/winners_curse.py --db data/upworthy.duckdb
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from abtest.bayes import (
    NormalPrior,
    SpikeSlabPrior,
    fit_normal_prior,
    fit_spike_slab,
    shrink,
)
from abtest.multiple import adjust
from abtest.srm import srm_check
from abtest.thinning import Replication, contrasts, replicate_winners

BASE_SEED = 20131124
N_BINS = 10
ALPHA = 0.05
METHODS = ("naive", "normal", "spike_slab")
METHOD_LABELS = {
    "naive": "naive (half-A lift)",
    "normal": "EB shrunk, normal prior",
    "spike_slab": "EB shrunk, spike + normal",
}

ARMS_SQL = """
select
    t.clickability_test_id,
    t.source_part,
    coalesce(t.randomization_risk, false)                   as randomization_risk,
    list(a.arm_number order by a.arm_number)                as arm_numbers,
    list(a.clicks order by a.arm_number)                    as clicks,
    list(a.impressions order by a.arm_number)               as impressions,
    list(a.headline order by a.arm_number)                  as headlines
from marts.fct_tests as t
join intermediate.int_test_arms as a using (clickability_test_id)
group by all
order by 1
"""


# --- data ----------------------------------------------------------------------------------


def load_arms(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict[str, int]]:
    """Arms of clean tests, one row each, sorted by test; plus counts of what was dropped."""
    rows = con.execute(ARMS_SQL).fetchall()
    out, drops = [], {"tests": len(rows), "risk": 0, "srm": 0, "lt2_arms": 0, "empty_arm": 0}
    for test_id, part, risk, arm_numbers, clicks, imps, heads in rows:
        if risk:
            drops["risk"] += 1
            continue
        if len(imps) < 2:
            drops["lt2_arms"] += 1
            continue
        if min(imps) <= 0:
            drops["empty_arm"] += 1
            continue
        if srm_check(imps).srm:
            drops["srm"] += 1
            continue
        out += [
            (test_id, part, a, int(c), int(n), h)
            for a, c, n, h in zip(arm_numbers, clicks, imps, heads, strict=True)
        ]
    arms = pd.DataFrame(
        out,
        columns=[
            "clickability_test_id",
            "source_part",
            "arm_number",
            "clicks",
            "impressions",
            "headline",
        ],
    )
    arms["test"] = pd.factorize(arms["clickability_test_id"], sort=True)[0]
    arms = arms.sort_values(["test", "arm_number"], kind="stable").reset_index(drop=True)
    drops["clean"] = int(arms["test"].nunique())
    return arms, drops


def subset(arms: pd.DataFrame, parts: tuple[str, ...] | None) -> pd.DataFrame:
    a = arms if parts is None else arms[arms["source_part"].isin(parts)]
    a = a.copy()
    a["test"] = pd.factorize(a["clickability_test_id"], sort=True)[0]
    return a.reset_index(drop=True)


# --- priors --------------------------------------------------------------------------------


def _finite(d: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(d) & np.isfinite(se) & (se > 0)
    return d[ok], se[ok]


def fit_priors(d: np.ndarray, se: np.ndarray) -> dict[str, NormalPrior | SpikeSlabPrior]:
    d, se = _finite(d, se)
    return {"normal": fit_normal_prior(d, se), "spike_slab": fit_spike_slab(d, se)}


def predictive_logpdf(prior: NormalPrior | SpikeSlabPrior, d: np.ndarray, se: np.ndarray):
    """Log density of each observed contrast under the prior plus its sampling noise."""
    slab = stats.norm.logpdf(d, prior.mu, np.sqrt(se**2 + prior.tau2))
    if isinstance(prior, SpikeSlabPrior):
        spike = stats.norm.logpdf(d, 0.0, se)
        return np.logaddexp(np.log(prior.pi0) + spike, np.log1p(-prior.pi0) + slab)
    return slab


def predictive_coverage(prior: NormalPrior | SpikeSlabPrior, d: np.ndarray, se: np.ndarray):
    """Share of contrasts inside the prior's central 95% predictive interval."""
    sd_slab = np.sqrt(se**2 + prior.tau2)
    if isinstance(prior, NormalPrior):
        return float((np.abs(d - prior.mu) <= 1.959964 * sd_slab).mean())
    cdf = prior.pi0 * stats.norm.cdf(d, 0.0, se) + (1 - prior.pi0) * stats.norm.cdf(
        d, prior.mu, sd_slab
    )
    return float(((cdf >= 0.025) & (cdf <= 0.975)).mean())


def _prior_cols(p: NormalPrior | SpikeSlabPrior) -> dict:
    return {"prior_pi0": getattr(p, "pi0", 0.0), "prior_mu": p.mu, "prior_tau": p.tau}


# --- thinning replication ------------------------------------------------------------------


@dataclass
class SeedResult:
    seed: int
    rep: Replication
    priors: dict[str, NormalPrior | SpikeSlabPrior]
    pred: dict[str, np.ndarray]


def run_seed(
    arms: pd.DataFrame,
    seed: int,
    priors: dict[str, NormalPrior | SpikeSlabPrior] | None = None,
    frac: float = 0.5,
) -> SeedResult:
    """One split. Priors are fitted on half A unless given (out-of-sample check)."""
    rng = np.random.default_rng([BASE_SEED, seed])
    rep = replicate_winners(
        arms["test"].to_numpy(),
        arms["clicks"].to_numpy(),
        arms["impressions"].to_numpy(),
        rng,
        frac,
    )
    if priors is None:
        priors = fit_priors(rep.d_a_all, rep.se_a_all)
    pred = {"naive": rep.d_a}
    for name, p in priors.items():
        pred[name] = shrink(p, rep.d_a, rep.se_a).mean
    return SeedResult(seed, rep, priors, pred)


MIN_PRED_SD = 0.01  # below this the predictions are ~constant and a slope means nothing


def _slope(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    if pred.std() < MIN_PRED_SD:
        return float("nan"), float("nan")
    b, a = np.polyfit(pred, target, 1)
    return float(b), float(a)


def score(r: SeedResult) -> dict:
    """Curse, still-best rate and prediction scores of one split."""
    rep = r.rep
    noise = float(np.mean(rep.se_b**2))
    naive, replicated = float(rep.d_a.mean()), float(rep.d_b.mean())
    kept = replicated / naive if naive else float("nan")
    row = {
        "seed": r.seed,
        "n_tests": int(rep.tests.size),
        "mean_naive": naive,
        "mean_replicated": replicated,
        "naive_rel": float(np.expm1(naive)),
        "replicated_rel": float(np.expm1(replicated)),
        "kept": kept,
        "overstatement": 1 / kept - 1 if kept > 0 else float("nan"),
        "still_best": float(rep.still_best.mean()),
        "still_best_chance": float(np.mean(1 / rep.n_arms)),
        "noise_b": noise,
    }
    for m, p in r.pred.items():
        mse = float(np.mean((p - rep.d_b) ** 2))
        b, a = _slope(p, rep.d_b)
        row |= {
            f"mean_{m}": float(p.mean()),
            f"bias_{m}": float(p.mean() - replicated),
            f"mse_{m}": mse,
            f"mse_true_{m}": mse - noise,
            f"slope_{m}": b,
            f"intercept_{m}": a,
        }
    for name, p in r.priors.items():
        row |= {f"{name}_{k}": v for k, v in _prior_cols(p).items()}
    return row


def calibration(results: list[SeedResult], n_bins: int = N_BINS) -> pd.DataFrame:
    """Pooled over seeds: half-B lift by decile of each method's prediction."""
    d_b = np.concatenate([r.rep.d_b for r in results])
    out = []
    for m in METHODS:
        p = np.concatenate([r.pred[m] for r in results])
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, edges.size - 2)
        for k in range(edges.size - 1):
            sel = b == k
            if not sel.any():
                continue
            y = d_b[sel]
            out.append(
                {
                    "method": m,
                    "bin": k + 1,
                    "n": int(sel.sum()),
                    "mean_predicted": float(p[sel].mean()),
                    "mean_half_b": float(y.mean()),
                    "se_half_b": float(y.std(ddof=1) / np.sqrt(y.size)) if y.size > 1 else 0.0,
                }
            )
    return pd.DataFrame(out)


def by_arms(results: list[SeedResult]) -> pd.DataFrame:
    k = np.concatenate([r.rep.n_arms for r in results])
    d_a = np.concatenate([r.rep.d_a for r in results])
    d_b = np.concatenate([r.rep.d_b for r in results])
    still = np.concatenate([r.rep.still_best for r in results])
    group = np.where(k >= 6, "6+", k.astype(str))
    out = []
    for g in ["2", "3", "4", "5", "6+"]:
        sel = group == g
        if not sel.any():
            continue
        naive, rep = d_a[sel].mean(), d_b[sel].mean()
        out.append(
            {
                "n_arms": g,
                "n_tests": int(sel.sum() // len(results)),
                "mean_naive": naive,
                "mean_replicated": rep,
                "naive_rel": np.expm1(naive),
                "replicated_rel": np.expm1(rep),
                "kept": rep / naive if naive else np.nan,
                "still_best": still[sel].mean(),
                "still_best_chance": np.mean(1 / k[sel]),
            }
        )
    return pd.DataFrame(out)


def per_test(arms: pd.DataFrame, results: list[SeedResult]) -> pd.DataFrame:
    ids = arms.groupby("test")[["clickability_test_id", "source_part"]].first()
    df = pd.DataFrame(
        {
            "clickability_test_id": ids["clickability_test_id"].to_numpy(),
            "source_part": ids["source_part"].to_numpy(),
            "n_arms": results[0].rep.n_arms,
        }
    )
    for name, f in {
        "naive": lambda r: r.rep.d_a,
        "replicated": lambda r: r.rep.d_b,
        "normal": lambda r: r.pred["normal"],
        "spike_slab": lambda r: r.pred["spike_slab"],
        "still_best": lambda r: r.rep.still_best.astype(float),
    }.items():
        df[f"mean_{name}"] = np.mean([f(r) for r in results], axis=0)
    return df


def headline(seeds: pd.DataFrame, drops: dict[str, int]) -> pd.DataFrame:
    rows = []

    def add(metric: str, col: str | None = None, value: float | None = None):
        v = seeds[col] if col else pd.Series([value])
        rows.append({"metric": metric, "mean": v.mean(), "min": v.min(), "max": v.max()})

    for k in ("tests", "risk", "srm", "lt2_arms", "empty_arm", "clean"):
        add(f"tests_{k}", value=drops[k])
    add("n_seeds", value=len(seeds))
    for col in (
        "n_tests",
        "naive_rel",
        "replicated_rel",
        "kept",
        "overstatement",
        "still_best",
        "still_best_chance",
    ):
        add(col, col)
    for m in METHODS:
        for s in ("mean", "bias", "mse", "mse_true"):
            add(f"{s}_{m}", f"{s}_{m}")
        add(f"slope_{m}", value=float(seeds[f"slope_{m}"].median()))
        if m != "naive":
            add(f"mean_{m}_rel", value=float(np.expm1(seeds[f"mean_{m}"]).mean()))
    add("noise_b", "noise_b")
    for c in ("normal_prior_tau", "spike_slab_prior_pi0", "spike_slab_prior_tau"):
        add(c, c)
    return pd.DataFrame(rows)


# --- out of sample prior -------------------------------------------------------------------


def _full_contrasts(arms: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    d, se = contrasts(
        arms["test"].to_numpy(), arms["clicks"].to_numpy(), arms["impressions"].to_numpy()
    )
    return _finite(d, se)


def out_of_sample(
    arms: pd.DataFrame,
    seeds: list[int],
    fit_part: str = "exploratory",
    targets: tuple[str, ...] = ("confirmatory", "holdout"),
) -> pd.DataFrame:
    """Prior fitted on one archive part, scored on the others (and vs in-sample priors)."""
    fit_arms = subset(arms, (fit_part,))
    if fit_arms["test"].nunique() < 2:
        return pd.DataFrame()
    fit_priors_ = fit_priors(*_full_contrasts(fit_arms))
    out = []
    for target in targets:
        t_arms = subset(arms, (target,))
        if t_arms["test"].nunique() < 2:
            continue
        d, se = _full_contrasts(t_arms)
        own = fit_priors(d, se)
        for name in ("normal", "spike_slab"):
            for source, prior in (
                (fit_part, fit_priors_[name]),
                (f"{target} (in-sample)", own[name]),
            ):
                out.append(
                    {
                        "target": target,
                        "check": "predictive",
                        "prior": name,
                        "fitted_on": source,
                        "n_tests": int(t_arms["test"].nunique()),
                        "n": int(d.size),
                        "coverage95": predictive_coverage(prior, d, se),
                        "mean_log_score": float(predictive_logpdf(prior, d, se).mean()),
                        **_prior_cols(prior),
                    }
                )
        # Thinning on the target: in-sample half-A prior vs the fit part's prior.
        ins = [run_seed(t_arms, s) for s in seeds]
        oos = [run_seed(t_arms, s, priors=fit_priors_) for s in seeds]
        for source, res in ((f"{target} half A (in-sample)", ins), (fit_part, oos)):
            sc = pd.DataFrame([score(r) for r in res])
            for m in METHODS:
                if source == fit_part and m == "naive":
                    continue
                out.append(
                    {
                        "target": target,
                        "check": "thinning",
                        "prior": m,
                        "fitted_on": "none" if m == "naive" else source,
                        "n_tests": int(sc["n_tests"].iloc[0]),
                        "n": len(seeds),
                        "mean_pred": sc[f"mean_{m}"].mean(),
                        "mean_half_b": sc["mean_replicated"].mean(),
                        "mse": sc[f"mse_{m}"].mean(),
                        "mse_true": sc[f"mse_true_{m}"].mean(),
                        "bias": sc[f"bias_{m}"].mean(),
                        "slope": sc[f"slope_{m}"].median(),
                    }
                )
    return pd.DataFrame(out)


# --- headline features ---------------------------------------------------------------------

_YOU = re.compile(
    r"\b(you|your|yours|yourself|yourselves|you're|you'll|you've|you'd|youre)\b", re.IGNORECASE
)
FEATURES = {
    "words_per_5": "length (+5 words)",
    "has_number": "has a digit",
    "is_question": "is a question (?)",
    "uses_you": 'uses "you"/"your"',
    "sentiment": "VADER sentiment (+0.5)",
}


def headline_features(headlines: pd.Series) -> pd.DataFrame:
    """Features of each headline. Sentiment is VADER compound, in units of 0.5 (-2..2)."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    vader = SentimentIntensityAnalyzer()
    h = headlines.fillna("").str.replace("’", "'", regex=False)
    return pd.DataFrame(
        {
            "n_words": h.str.split().str.len(),
            "words_per_5": h.str.split().str.len() / 5,
            "has_number": h.str.contains(r"\d", regex=True).astype(float),
            "is_question": h.str.contains("?", regex=False).astype(float),
            "uses_you": h.map(lambda s: bool(_YOU.search(s))).astype(float),
            "sentiment": h.map(lambda s: vader.polarity_scores(s)["compound"]) / 0.5,
        },
        index=headlines.index,
    )


def feature_associations(arms: pd.DataFrame) -> pd.DataFrame:
    """Within-test regression of empirical-logit CTR on headline features.

    Test fixed effects by demeaning within test; OLS (arms weighted equally) with
    cluster-robust SEs by test. Joint model (all features together) and one model per
    feature alone; BH across the five features within each.
    """
    a = arms[arms["headline"].notna() & (arms["headline"].str.strip() != "")].copy()
    a = a[a.groupby("test")["test"].transform("size") >= 2]
    f = headline_features(a["headline"])
    x, n = a["clicks"].to_numpy(float), a["impressions"].to_numpy(float)
    y = pd.Series(np.log((x + 0.5) / (n - x + 0.5)), index=a.index)
    cols = list(FEATURES)
    g = a["test"]
    y_dm = y - y.groupby(g).transform("mean")
    f_dm = f[cols] - f[cols].groupby(g).transform("mean")
    varies = (f[cols].groupby(g).transform("nunique") > 1).groupby(g).first()

    def fit(features: list[str]):
        return sm.OLS(y_dm.to_numpy(), f_dm[features].to_numpy()).fit(
            cov_type="cluster", cov_kwds={"groups": g.to_numpy()}
        )

    out = []
    joint = fit(cols)
    alone = {c: fit([c]) for c in cols}
    for model, get in (
        ("joint", lambda c: (joint, cols.index(c))),
        ("alone", lambda c: (alone[c], 0)),
    ):
        rows = []
        for c in cols:
            res, i = get(c)
            b, se, p = float(res.params[i]), float(res.bse[i]), float(res.pvalues[i])
            rows.append(
                {
                    "model": model,
                    "feature": c,
                    "label": FEATURES[c],
                    "coef_log_odds": b,
                    "se": se,
                    "ci_low": b - 1.959964 * se,
                    "ci_high": b + 1.959964 * se,
                    "ctr_change": float(np.expm1(b)),
                    "ctr_change_low": float(np.expm1(b - 1.959964 * se)),
                    "ctr_change_high": float(np.expm1(b + 1.959964 * se)),
                    "p": p,
                    "n_arms": int(len(a)),
                    "n_tests": int(g.nunique()),
                    "n_tests_varying": int(varies[c].sum()),
                    "mean_feature": float(f[c].mean()),
                }
            )
        bh = adjust([r["p"] for r in rows], "bh", ALPHA)
        for r, q, rej in zip(rows, bh.p_adjusted, bh.reject, strict=True):
            r |= {"bh_q": q, "bh_reject": rej}
        out += rows
    return pd.DataFrame(out)


# --- figures -------------------------------------------------------------------------------

INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLORS = {"naive": BLUE, "normal": ORANGE, "spike_slab": AQUA}


def _style(ax, grid: str = "y") -> None:
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ae")


def plot_all(
    out: Path, seeds: pd.DataFrame, cal: pd.DataFrame, arms_tbl: pd.DataFrame, feats: pd.DataFrame
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update(
        {
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.facecolor": BG,
        }
    )
    paths = []

    # 1. naive vs replicated vs shrunk (relative lift of the winner vs the rest)
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=BG)
    bars = [
        ("Naive: winner on half A", np.expm1(seeds["mean_naive"]), BLUE),
        ("Shrunk, normal prior", np.expm1(seeds["mean_normal"]), ORANGE),
        ("Shrunk, spike + normal", np.expm1(seeds["mean_spike_slab"]), AQUA),
        ("Replicated: same arm on half B", np.expm1(seeds["mean_replicated"]), MUTED),
    ]
    for i, (_label, v, color) in enumerate(bars):
        ax.barh(i, v.mean(), color=color, height=0.6)
        ax.errorbar(
            v.mean(),
            i,
            xerr=[[v.mean() - v.min()], [v.max() - v.mean()]],
            color=INK,
            lw=1,
            capsize=3,
        )
        ax.text(v.max() + 0.005, i, f"{v.mean():.1%}", va="center", color=INK)
    ax.set_yticks(range(len(bars)), [b[0] for b in bars])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("mean lift of the half-A winner vs the rest of its test (exp(log OR) - 1)")
    ax.set_title(
        "The winner's lift on fresh data (bars: mean over seeds, whiskers: range)",
        loc="left",
        fontsize=10,
    )
    _style(ax, "x")
    fig.tight_layout()
    paths.append(out / "winners_curse_lift.png")
    fig.savefig(paths[-1], facecolor=BG, dpi=150)
    plt.close(fig)

    # 2. calibration
    fig, ax = plt.subplots(figsize=(5.5, 5), facecolor=BG)
    lo = min(cal["mean_predicted"].min(), cal["mean_half_b"].min())
    hi = max(cal["mean_predicted"].max(), cal["mean_half_b"].max())
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=(0, (4, 3)), label="perfect calibration")
    for m in METHODS:
        c = cal[cal["method"] == m]
        ax.errorbar(
            c["mean_predicted"],
            c["mean_half_b"],
            yerr=1.96 * c["se_half_b"],
            color=COLORS[m],
            lw=2,
            marker="o",
            ms=5,
            capsize=0,
            label=METHOD_LABELS[m],
        )
    ax.set_xlabel("predicted lift of the half-A winner (log OR, decile mean)")
    ax.set_ylabel("observed lift on half B (log OR)")
    ax.set_title("Calibration against the held-out half", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax, "both")
    fig.tight_layout()
    paths.append(out / "winners_curse_calibration.png")
    fig.savefig(paths[-1], facecolor=BG, dpi=150)
    plt.close(fig)

    # 3. still best by number of arms
    fig, ax = plt.subplots(figsize=(6, 3.4), facecolor=BG)
    xs = np.arange(len(arms_tbl))
    ax.bar(xs - 0.2, arms_tbl["still_best"], width=0.38, color=BLUE, label="still best on half B")
    ax.bar(
        xs + 0.2,
        arms_tbl["still_best_chance"],
        width=0.38,
        color=GRID,
        edgecolor=MUTED,
        lw=0.8,
        label="by chance (1 / arms)",
    )
    for x_, v in zip(xs, arms_tbl["still_best"], strict=True):
        ax.text(x_ - 0.2, v + 0.01, f"{v:.0%}", ha="center", color=INK, fontsize=9)
    ax.set_xticks(xs, arms_tbl["n_arms"])
    ax.set_xlabel("arms in the test")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1)
    ax.set_title("How often the half-A winner is also best on half B", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    paths.append(out / "winners_curse_by_arms.png")
    fig.savefig(paths[-1], facecolor=BG, dpi=150)
    plt.close(fig)

    # 4. features (joint model)
    j = feats[feats["model"] == "joint"].reset_index(drop=True)
    if not j.empty:
        fig, ax = plt.subplots(figsize=(6.5, 3.2), facecolor=BG)
        for i, r in j.iterrows():
            ax.plot([r["ctr_change_low"], r["ctr_change_high"]], [i, i], color=BLUE, lw=2)
            ax.plot(
                r["ctr_change"], i, "o", ms=8, color=BLUE, mfc=BLUE if r["bh_reject"] else BG, mew=2
            )
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(range(len(j)), j["label"])
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("CTR change vs other arms of the same test (95% CI, clustered by test)")
        ax.set_title(
            "Headline features, within-test associations (filled: BH q < 0.05)",
            loc="left",
            fontsize=10,
        )
        _style(ax, "x")
        fig.tight_layout()
        paths.append(out / "winners_curse_features.png")
        fig.savefig(paths[-1], facecolor=BG, dpi=150)
        plt.close(fig)
    return paths


# --- run -----------------------------------------------------------------------------------


def run(
    con: duckdb.DuckDBPyConnection, out: Path, n_seeds: int = 20, figures: bool = True
) -> pd.DataFrame:
    arms, drops = load_arms(con)
    if drops["clean"] < 2:
        raise ValueError("fewer than two clean tests")
    seed_ids = list(range(n_seeds))
    results = [run_seed(arms, s) for s in seed_ids]
    seeds = pd.DataFrame([score(r) for r in results])
    cal = calibration(results)
    arms_tbl = by_arms(results)
    oos = out_of_sample(arms, seed_ids)
    feats = feature_associations(arms)
    top = headline(seeds, drops)

    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "winners_curse_headline.csv": top,
        "winners_curse_seeds.csv": seeds,
        "winners_curse_by_arms.csv": arms_tbl,
        "winners_curse_calibration.csv": cal,
        "winners_curse_oos.csv": oos,
        "winners_curse_features.csv": feats,
        "winners_curse_tests.csv": per_test(arms, results),
    }
    for name, df in tables.items():
        df.to_csv(out / name, index=False, float_format="%.6g")
    if figures:
        plot_all(out, seeds, cal, arms_tbl, feats)
    return top


def summary(top: pd.DataFrame) -> str:
    v = top.set_index("metric")["mean"]
    return "\n".join(
        [
            f"clean tests: {int(v['tests_clean']):,} of {int(v['tests_tests']):,} "
            f"(risk window {int(v['tests_risk']):,}, SRM {int(v['tests_srm']):,})",
            f"seeds: {int(v['n_seeds'])}",
            f"half-A winner lift: naive {v['naive_rel']:.1%}, replicated on half B "
            f"{v['replicated_rel']:.1%} (kept {v['kept']:.0%}, overstated by "
            f"{v['overstatement']:.0%})",
            f"still best on half B: {v['still_best']:.1%} (chance {v['still_best_chance']:.1%})",
            f"MSE vs half B: naive {v['mse_naive']:.4f}, normal {v['mse_normal']:.4f}, "
            f"spike+normal {v['mse_spike_slab']:.4f} (noise floor {v['noise_b']:.4f})",
            f"calibration slope: naive {v['slope_naive']:.2f}, normal {v['slope_normal']:.2f}, "
            f"spike+normal {v['slope_spike_slab']:.2f}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("data/upworthy.duckdb"))
    ap.add_argument("--out", type=Path, default=Path("analysis/output"))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    if not args.db.exists():
        print(f"{args.db} not found; run `make ingest` and `make build` first", file=sys.stderr)
        return 1
    with duckdb.connect(str(args.db), read_only=True) as con:
        top = run(con, args.out, args.seeds, figures=not args.no_figures)
    print(summary(top))
    print(f"\nwrote {args.out}/winners_curse_*.csv and figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
