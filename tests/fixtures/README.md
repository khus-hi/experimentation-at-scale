# CI fixture

`packages_sample.csv` is a small slice of the real Upworthy Research Archive, used by CI to run
`dbt build` without downloading anything from OSF.

- **What:** about 200 tests drawn at random (seed `20210195`) from all three released parts
  (exploratory, confirmatory, holdout), with **every arm** of each sampled test and every raw
  column exactly as `ingest/load.py` loads it (`column00`, `source_part` and `source_file`
  included). The file is kept under 1 MB; if it would be larger, tests are dropped from the
  end of the shuffled order until it fits.
- **Made by:** `make fixture` (runs `tests/fixtures/make_fixture.py` against
  `data/upworthy.duckdb`). It needs the real data, so it runs only where `make ingest` has run.
- **Loaded by:** `tests/fixtures/load_fixture.py --db <path>`, which creates `raw.packages`
  in a fresh DuckDB file. CI does this in a temp directory.

Do not use this sample for analysis. It is a random 0.6% of tests and exists only to test the
pipeline.

## Source and license

The rows come unchanged (apart from sampling) from:

> Matias, J. N., Munger, K., Le Quere, M. A., & Ebersole, C. (2021). The Upworthy Research
> Archive, a time series of 32,487 experiments in U.S. media. *Scientific Data*, 8, 195.
> https://doi.org/10.1038/s41597-021-00934-7

Data: OSF project [jd64p](https://osf.io/jd64p/), files
`exploratory-packages-03.12.2020.csv`, `confirmatory-packages-03.12.2020.csv` and
`holdout-packages-03.12.2020.csv`. Project site: https://upworthy.natematias.com/.

The archive is published by Cornell University under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
through an agreement between Good/Upworthy and Cornell. CC BY 4.0 allows sharing and adapting
the data with attribution; this sample is a subset (a modification) of the original files. The
license for this repository's code does not apply to this file.

The authors also make non-binding requests, including that analysis code be published openly.
They discourage confirmatory analyses of tests run between 2013-06-25 and 2014-01-10 (a
randomization problem); the dbt models flag these as `randomization_risk`.
