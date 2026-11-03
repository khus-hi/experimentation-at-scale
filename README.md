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

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on every push and pull request: ruff (lint and
format), pytest with a coverage gate (the build fails if `src/abtest` drops below 90%), and
`dbt build` against `tests/fixtures/packages_sample.csv`, a committed 200-test sample of the real
archive loaded into a temporary DuckDB, so CI never downloads from OSF. `make fixture`
regenerates the sample from `data/upworthy.duckdb`; see `tests/fixtures/README.md` for how it is
drawn and for the data license (CC BY 4.0).
