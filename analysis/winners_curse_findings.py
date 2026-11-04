"""Render docs/winners_curse.md from the CSVs written by analysis/winners_curse.py.

Every number in the document is read from analysis/output/winners_curse_*.csv; the prose
only picks words (better/worse, above/below) by comparing those numbers. Re-run after
`make winners-curse` and the document follows the data.

Usage:
    .venv/bin/python analysis/winners_curse_findings.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = "../analysis/output"
METHOD_NAMES = {
    "naive": "Naive (half-A lift)",
    "normal": "EB shrunk, normal prior",
    "spike_slab": "EB shrunk, spike + normal prior",
}


PROSE = {
    "normal": "shrunk estimate with the normal prior",
    "spike_slab": "shrunk estimate with the spike + normal prior",
}


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%" if np.isfinite(x) else "n/a"


def num(x: float) -> str:
    return f"{int(round(x)):,}"


def f3(x: float) -> str:
    return f"{x:.3f}" if np.isfinite(x) else "n/a"


def f2(x: float) -> str:
    return f"{x:.2f}" if np.isfinite(x) else "n/a"


class Outputs:
    def __init__(self, out: Path):
        self.top = pd.read_csv(out / "winners_curse_headline.csv").set_index("metric")
        self.seeds = pd.read_csv(out / "winners_curse_seeds.csv")
        self.arms = pd.read_csv(out / "winners_curse_by_arms.csv", dtype={"n_arms": str})
        self.oos = pd.read_csv(out / "winners_curse_oos.csv")
        self.feats = pd.read_csv(out / "winners_curse_features.csv")

    def v(self, metric: str, col: str = "mean") -> float:
        return float(self.top.loc[metric, col])


def _range(o: Outputs, metric: str, fmt=pct) -> str:
    return f"{fmt(o.v(metric, 'min'))} to {fmt(o.v(metric, 'max'))}"


def _best(o: Outputs, prefix: str) -> str:
    vals = {m: o.v(f"{prefix}_{m}") for m in ("normal", "spike_slab")}
    return min(vals, key=vals.get)


def headline(o: Outputs) -> list[str]:
    best = _best(o, "mse")
    cut = 1 - o.v(f"mse_true_{best}") / o.v("mse_true_naive")
    oos = o.oos[(o.oos["target"] == "confirmatory") & (o.oos["check"] == "thinning")]
    items = [
        f"**The half-A winner's lift is overstated by {pct(o.v('overstatement'), 0)}.** "
        f"Measured on the half it was picked on, the winner beats the rest of its test by "
        f"{pct(o.v('naive_rel'))}; on the other half, the same arm beats it by "
        f"{pct(o.v('replicated_rel'))}, so {pct(o.v('kept'), 0)} of the naive lift replicates "
        f"(mean over {num(o.v('n_seeds'))} splits of {num(o.v('n_tests'))} clean tests).",
        f"**The half-A winner is still the best arm on half B in "
        f"{pct(o.v('still_best'))} of tests**, against {pct(o.v('still_best_chance'))} "
        "expected if every arm were equally good.",
        f"**Shrinkage fixes {'most' if cut >= 0.5 else 'some'} of it.** The {PROSE[best]} "
        "predicts the "
        f"held-out lift with {pct(cut, 0)} less error than the naive lift (MSE against the "
        f"true lift, estimated by removing half B's own noise), and its calibration slope is "
        f"{f2(o.v(f'slope_{best}'))} against {f2(o.v('slope_naive'))} for the naive lift "
        "(1 is perfectly calibrated).",
    ]
    if not oos.empty:
        ex = oos[oos["fitted_on"] == "exploratory"].set_index("prior")
        ins = oos[oos["fitted_on"].str.contains("in-sample")].set_index("prior")
        if best in ex.index and best in ins.index:
            a, b = ex.loc[best, "mse_true"], ins.loc[best, "mse_true"]
            word = (
                "about as well as"
                if abs(a - b) <= 0.1 * b
                else "better than"
                if a < b
                else "nearly as well as"
                if a <= 1.25 * b
                else "worse than"
            )
            naive = oos.set_index("prior").loc["naive", "mse_true"]
            holds = a <= 1.25 * b and a < naive
            items.append(
                f"**The prior {'holds' if holds else 'only partly holds'} out of sample.** "
                f"Fitted on the exploratory tests only, the {PROSE[best]} predicts the "
                "confirmatory tests' held-out lifts "
                f"{word} a prior fitted on the confirmatory tests themselves (MSE {f3(a)} vs "
                f"{f3(b)}; naive {f3(naive)})."
            )
    j = o.feats[o.feats["model"] == "joint"]
    sig = j[j["bh_reject"]]
    if sig.empty:
        items.append(
            "**No headline feature is associated with CTR after BH correction** in the "
            "within-test comparison."
        )
    else:
        parts = [f"{r.label} {pct(r.ctr_change, 1)}" for r in sig.itertuples()]
        items.append(
            f"**{len(sig)} of {len(j)} headline features are associated with CTR within tests "
            f"after BH correction:** {', '.join(parts)} (CTR vs other arms of the same test, "
            "holding the other features fixed). These are associations, not rules."
        )
    return [f"{i}. {t}" for i, t in enumerate(items, 1)]


def render(o: Outputs) -> str:
    v = o.v
    s = o.seeds
    lines = [
        "# The winner's curse, measured by data thinning",
        "",
        "_Generated by `analysis/winners_curse_findings.py` from "
        "`analysis/output/winners_curse_*.csv` (`make winners-curse`, then "
        "`make winners-curse-doc`). Do not edit by hand._",
        "",
        "## Headline numbers",
        "",
        *headline(o),
        "",
        "## Method",
        "",
        "Every arm's impressions are split 50/50 at random and its clicks drawn "
        "hypergeometrically (`abtest.thinning.split_binomial`), which turns each test into two "
        "independent half-tests with the same true click rates, provided clicks are binomial. "
        "The arm with the highest CTR on half A is the winner; its log odds ratio against the "
        "pooled rest of its test on half A is the naive lift, and the same contrast on half B is "
        "an honest replication, because half B played no part in picking it. This is repeated "
        f"with {num(v('n_seeds'))} fixed seeds and averaged.",
        "",
        f"Sample: {num(v('tests_clean'))} clean tests of {num(v('tests_tests'))}, dropping "
        f"{num(v('tests_risk'))} in the 2013-06-25 to 2014-01-10 randomization-risk window, "
        f"{num(v('tests_srm'))} failing SRM (p < 0.001), {num(v('tests_lt2_arms'))} with one "
        f"arm and {num(v('tests_empty_arm'))} with an arm without impressions (the same rule as "
        "`analysis/portfolio.py`).",
        "",
        "Lifts are relative lifts `exp(mean log OR) - 1` of the winner against the rest of its "
        "test. Each half has half the data, so a half-test is noisier than the full test and "
        "its winner's curse is bigger than the full test's; the comparison between methods is "
        "fair because every method sees only half A.",
        "",
        "## 1. Naive vs replicated lift",
        "",
        "| | mean relative lift | range over seeds |",
        "|---|---:|---:|",
        f"| Naive: winner on half A | {pct(v('naive_rel'))} | {_range(o, 'naive_rel')} |",
        f"| Replicated: same arm on half B | {pct(v('replicated_rel'))} | "
        f"{_range(o, 'replicated_rel')} |",
        f"| EB shrunk, normal prior (half A only) | {pct(v('mean_normal_rel'))} | |",
        f"| EB shrunk, spike + normal (half A only) | {pct(v('mean_spike_slab_rel'))} | |",
        "",
        f"Kept: {pct(v('kept'), 0)} of the naive log-odds lift replicates (range "
        f"{_range(o, 'kept', lambda x: pct(x, 0))}); the naive lift overstates the replicated "
        f"one by {pct(v('overstatement'), 0)}.",
        "",
        f"![Naive, shrunk and replicated lift]({FIG}/winners_curse_lift.png)",
        "",
        "### Does the winner stay the winner?",
        "",
        "| arms | tests | naive lift | replicated lift | kept | still best on half B | by chance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in o.arms.itertuples():
        lines.append(
            f"| {r.n_arms} | {num(r.n_tests)} | {pct(r.naive_rel)} | {pct(r.replicated_rel)} | "
            f"{pct(r.kept, 0)} | {pct(r.still_best)} | {pct(r.still_best_chance)} |"
        )
    lines += [
        f"| all | {num(v('n_tests'))} | {pct(v('naive_rel'))} | {pct(v('replicated_rel'))} | "
        f"{pct(v('kept'), 0)} | {pct(v('still_best'))} | {pct(v('still_best_chance'))} |",
        "",
        f"![Still best by number of arms]({FIG}/winners_curse_by_arms.png)",
        "",
        "## 2. Do shrunk estimates predict the held-out half?",
        "",
        "On every split an empirical Bayes prior (`abtest.bayes`, from PR #8) is fitted on every "
        "arm's half-A contrast, before any picking, and used to shrink the winner's half-A lift. "
        "Each prediction is scored against the winner's half-B lift.",
        "",
        "| method | mean prediction | bias vs half B | MSE vs half B | MSE vs true lift | "
        "calibration slope |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m, name in METHOD_NAMES.items():
        lines.append(
            f"| {name} | {f3(v(f'mean_{m}'))} | {f3(v(f'bias_{m}'))} | {f3(v(f'mse_{m}'))} | "
            f"{f3(v(f'mse_true_{m}'))} | {f2(v(f'slope_{m}'))} |"
        )
    lines += [
        "",
        "Log-odds scale, mean over seeds (slope: median). Half B is noisy, but its noise is "
        "independent of anything computed on half A, so it adds the same "
        f"{f3(v('noise_b'))} (mean squared standard error of the half-B lift) to every "
        "method's MSE; subtracting it estimates the error against the true lift. Bias is the "
        "mean prediction minus the mean half-B lift. The calibration slope regresses half B on "
        "the prediction: below 1 means the predictions are too spread out (overconfident), "
        "which is what the winner's curse looks like.",
        "",
        f"Fitted priors on half A (mean over seeds): normal tau {f3(v('normal_prior_tau'))}; "
        f"spike + normal pi0 {f2(v('spike_slab_prior_pi0'))}, tau "
        f"{f3(v('spike_slab_prior_tau'))}.",
        "",
        f"![Calibration plot]({FIG}/winners_curse_calibration.png)",
        "",
        "## 3. Out-of-sample prior: exploratory to confirmatory",
        "",
        "The archive comes in three random parts (exploratory, confirmatory, holdout). The "
        "prior is fitted on every arm of the clean exploratory tests (full data) and checked "
        "on the other parts two ways: the share of their arm contrasts inside the prior's 95% "
        "predictive interval (should be about 95%) with the mean log predictive density, and "
        "the thinning score of section 2 with the exploratory prior in place of the in-sample "
        "half-A prior.",
        "",
    ]
    if o.oos.empty:
        lines += ["_Not enough exploratory tests to fit a prior._", ""]
    else:
        pr = o.oos[o.oos["check"] == "predictive"]
        lines += [
            "| target | prior | fitted on | contrasts | 95% coverage | mean log score | pi0 | "
            "tau |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for r in pr.itertuples():
            lines.append(
                f"| {r.target} | {r.prior} | {r.fitted_on} | {num(r.n)} | {pct(r.coverage95)} | "
                f"{f3(r.mean_log_score)} | {f2(r.prior_pi0)} | {f3(r.prior_tau)} |"
            )
        th = o.oos[o.oos["check"] == "thinning"]
        lines += [
            "",
            "| target | method | prior fitted on | tests | bias vs half B | MSE vs true lift | "
            "calibration slope |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for r in th.itertuples():
            lines.append(
                f"| {r.target} | {METHOD_NAMES[r.prior]} | {r.fitted_on} | {num(r.n_tests)} | "
                f"{f3(r.bias)} | {f3(r.mse_true)} | {f2(r.slope)} |"
            )
        lines += [
            "",
            "The in-sample rows fit the prior on the same tests they score, which flatters "
            "them; a prior that transfers should come close to them.",
            "",
        ]
    j = o.feats[o.feats["model"] == "joint"]
    a = o.feats[o.feats["model"] == "alone"].set_index("feature")
    lines += [
        "## 4. Headline features (descriptive)",
        "",
        "Each arm's empirical-logit CTR is regressed on features of its headline, demeaned "
        "within test (test fixed effects), so an arm is only ever compared with other arms of "
        "the same test, which saw the same audience at the same time. Standard errors are "
        "clustered by test, and the five p-values are BH-corrected. Sentiment is VADER's "
        "compound score (a free lexicon tuned to social media text), in steps of 0.5 on its "
        "-1 to 1 scale.",
        "",
        f"{num(j['n_arms'].iloc[0]) if not j.empty else 0} arms in "
        f"{num(j['n_tests'].iloc[0]) if not j.empty else 0} clean tests.",
        "",
        "| feature | share / mean | tests where it varies | CTR change (joint) | 95% CI | "
        "BH q | CTR change (alone) | BH q (alone) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in j.itertuples():
        al = a.loc[r.feature]
        mean = (
            f2(r.mean_feature * 5) + " words"
            if r.feature == "words_per_5"
            else (f2(r.mean_feature * 0.5) if r.feature == "sentiment" else pct(r.mean_feature))
        )
        lines.append(
            f"| {r.label} | {mean} | {num(r.n_tests_varying)} | {pct(r.ctr_change)} | "
            f"{pct(r.ctr_change_low)} to {pct(r.ctr_change_high)} | {f3(r.bh_q)} | "
            f"{pct(al['ctr_change'])} | {f3(al['bh_q'])} |"
        )
    lines += [
        "",
        f"![Headline feature associations]({FIG}/winners_curse_features.png)",
        "",
        "How to read this. Randomization inside a test makes each within-test contrast causal "
        "for that pair of packages: the difference in CTR between two arms of one test was "
        "caused by showing one package instead of the other. It does not make a feature "
        "causal. Two headlines that differ in length also differ in wording, topic emphasis "
        "and, often, image, and editors chose which variants to write; the coefficients are "
        "average associations across those pairs, not rules for writing headlines.",
        "",
        "## Caveats",
        "",
        "- **Overdispersion makes this a lower bound.** Thinning gives independent halves only "
        "if clicks are binomial. The archive's natural A/A pairs are significant at p < 0.05 "
        "about 13% of the time instead of 5% (`docs/portfolio_findings.md`), so arms carry "
        "extra-binomial shocks (time of day, placement, a share that took off) that land in "
        "both halves. Picking on half A also picks on those shocks, and half B inherits them, "
        "so the replicated lift still contains some luck: the true overstatement is at least "
        "as large as measured here.",
        "- **Half-tests are noisier than full tests**, so the size of the curse here is for a "
        "test half as large as the one Upworthy ran. The shrinkage results carry over because "
        "the prior is over true lifts, which do not depend on sample size.",
        "- **Lift is against the pooled rest of the test**, as in `abtest.bayes`, not against a "
        "control arm, since the archive has no control.",
        "- Feature associations are averaged over tests where the feature varies; tests that "
        "only changed the image add nothing to them.",
        "",
        f"Seeds: {', '.join(str(x) for x in s['seed'])}.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "analysis" / "output")
    ap.add_argument("--doc", type=Path, default=ROOT / "docs" / "winners_curse.md")
    args = ap.parse_args(argv)
    if not (args.out / "winners_curse_headline.csv").exists():
        print("run `make winners-curse` first", file=sys.stderr)
        return 1
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(render(Outputs(args.out)))
    print(f"wrote {args.doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
