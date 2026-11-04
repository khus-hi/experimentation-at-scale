# experimentation-at-scale

[![CI](https://github.com/khus-hi/experimentation-at-scale/actions/workflows/ci.yml/badge.svg)](https://github.com/khus-hi/experimentation-at-scale/actions/workflows/ci.yml)

**Question:** When you run thousands of A/B tests, how many winners are real and how inflated are their lifts?

## Data

[The Upworthy Research Archive](https://upworthy.natematias.com/): about 32,000 real headline A/B tests run by Upworthy between 2013 and 2015.

> Matias, J. N., Munger, K., Le Quere, M. A., & Ebersole, C. (2021). The Upworthy Research Archive, a time series of 32,487 experiments in U.S. media. *Scientific Data*, 8, 195. https://doi.org/10.1038/s41597-021-00934-7

## Status

Work in progress.

## Pipeline

```
make setup PYTHON=/opt/anaconda3/bin/python3
make ingest
make build
make docs
make test
```

- `make ingest` downloads the archive from OSF and loads it into `data/upworthy.duckdb` as `raw.packages`.
- `make build` runs `dbt build` in `dbt/` (models plus data tests), then prints tests and arms by year and by number of arms (`dbt/summary.py`).
- `make docs` writes the dbt docs site to `dbt/target/static_index.html`.

dbt models (DuckDB schemas `staging`, `intermediate`, `marts`):

| model | grain |
| --- | --- |
| `stg_packages` | one row per package (arm), types cleaned, `ctr`, `randomization_risk` |
| `int_test_arms` | exact duplicates removed, arms numbered within a test, identical arms flagged |
| `fct_tests` | one row per test: arms, impressions, best/worst ctr, Upworthy's winner, A/A flag |
| `fct_aa_pairs` | every pair of identical arms in a test (natural A/A tests) |

`randomization_risk` is true when the archive's own risk column says so (if the loaded files have one; the 2020/2021 OSF files do not) or when the package was created between 2013-06-25 and 2014-01-10.

## Statistics (`src/abtest`)

Pure functions on counts that return frozen dataclasses; no pandas in the math.

| module | what |
| --- | --- |
| `abtest.proportions` | pooled two-proportion z-test, Wilson CI per arm, Newcombe CI for the difference, relative lift with a delta-method CI, chi-square omnibus plus every arm vs a control (`multi_arm_test`) |
| `abtest.power` | sample size per arm for a baseline and relative MDE, MDE for a given n, achieved power for observed n; k arms with Bonferroni `alpha / (k - 1)` (matches statsmodels `NormalIndPower`) |
| `abtest.srm` | sample ratio mismatch: chi-square goodness of fit vs the expected split, flagged at p < 0.001 |
| `abtest.sequential` | peeking simulator: a two-arm test streamed in daily batches, scored by naive peeking (stop at first p < 0.05), a Lan-DeMets O'Brien-Fleming group-sequential design, and the always-valid mSPRT p-value (normal mixture, Johari et al. 2017); calibrated to `fct_tests` |
| `abtest.multiple` | Bonferroni, Holm, Benjamini-Hochberg and Benjamini-Yekutieli adjusted p-values (match statsmodels `multipletests`) and Storey's pi0; `within_test` (k - 1 arms vs a control, Holm), `portfolio` (one omnibus p per test, BH), `winner_survival` (declared winners that survive Holm within the test and BH across tests) |
| `abtest.bayes` | Beta-binomial posteriors per arm, Monte Carlo P(best) and expected loss (seeded), and a "ship if expected loss < threshold" rule; empirical Bayes shrinkage of each test's best arm vs the rest on the log-odds scale, with a normal prior (method of moments) or a spike-at-zero plus normal prior (maximum likelihood), fitted on every arm before picking; `winners_curse` compares naive and shrunk lifts |

`make srm` runs the SRM check and the omnibus test on every test in `fct_tests` and prints
counts overall and inside vs outside the randomization-risk window (`analysis/srm_scan.py`).

`make peeking` (`analysis/peeking.py`) writes `analysis/output/peeking.csv` and
`analysis/output/peeking_fpr.png`: false positive rate, power and sample saved vs a fixed-n
test for each rule. It is a **calibrated simulation**: the archive has only final counts per arm,
no daily counts, so baseline CTR, n and number of looks are drawn jointly from real tests (outside
the randomization-risk window) and the daily stream is simulated. Number of looks has no column;
it is the calendar days spanned by a test's arms' `created_at` (a lower bound on duration, 1 for
most tests), and the sweep over 1 to 20 looks does not depend on it.

`make shrinkage` (`analysis/shrinkage.py`) runs the Bayesian decision rule on every test and the
empirical Bayes shrinkage of each test's best arm, prints the winner's curse summary and writes
`analysis/output/shrinkage_summary.csv` (committed) and `shrinkage_tests.csv` (per test,
gitignored). Tests in the randomization-risk window are excluded unless `--include-risk`.

Results on the real archive (25,468 tests outside the randomization-risk window, 2026-11-04):

- The beta-binomial rule (ship when expected loss < 1% of the arm's CTR) ships 21.9% of tests
  and keeps testing the rest. Of the 5,958 tests with a declared winner, it picks the same arm
  in 72.4%. The declared winners' median P(best) is 0.61; 37.9% are below 0.5.
- The best observed arm beats the rest of its test by 48.4% (mean log OR 0.395, as relative
  lift). After shrinkage that is 31.3% (normal prior) or 28.9% (spike + normal), so about a
  third of the reported lift is the winner's curse. The spike + normal prior fits far better
  (log likelihood +2,884 on 122,377 arm contrasts) and puts 33% of arms at exactly no
  difference from the rest. Fitted on winners only, the prior keeps 98% of the naive lift,
  which is why it is fitted on every arm.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on every push and pull request: ruff (lint and
format), pytest with a coverage gate (the build fails if `src/abtest` drops below 90%), and
`dbt build` against `tests/fixtures/packages_sample.csv`, a committed 200-test sample of the real
archive loaded into a temporary DuckDB, so CI never downloads from OSF. `make fixture`
regenerates the sample from `data/upworthy.duckdb`; see `tests/fixtures/README.md` for how it is
drawn and for the data license (CC BY 4.0).
