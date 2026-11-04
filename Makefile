PYTHON ?= python3.11
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: setup ingest build docs summary fixture test srm peeking shrinkage portfolio findings winners-curse winners-curse-doc analysis report all

DB := $(CURDIR)/data/upworthy.duckdb
DBT := cd dbt && UPWORTHY_DB=$(DB) DBT_PROFILES_DIR=. ../$(PY) -W ignore::RuntimeWarning -m dbt.cli.main

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	$(PY) -c "import sys; print(sys.executable)"

ingest:
	$(PY) -m ingest.download
	$(PY) -m ingest.load

build:
	($(DBT) build); status=$$?; $(PY) dbt/summary.py --db $(DB); exit $$status

docs:
	($(DBT) docs generate --static)
	@echo "docs written to dbt/target/static_index.html (open it in a browser)"

summary:
	$(PY) dbt/summary.py --db $(DB)

fixture:
	$(PY) tests/fixtures/make_fixture.py --db $(DB)

test:
	$(PY) -m pytest
	$(PY) -m ruff check .

srm:
	$(PY) analysis/srm_scan.py --db $(DB)

shrinkage:
	$(PY) analysis/shrinkage.py --db $(DB)

peeking:
	$(PY) analysis/peeking.py --db $(DB)

portfolio:
	$(PY) analysis/portfolio.py --db $(DB)

findings:
	$(PY) analysis/portfolio_findings.py

winners-curse:
	$(PY) analysis/winners_curse.py --db $(DB)

winners-curse-doc:
	$(PY) analysis/winners_curse_findings.py

analysis:
	$(PY) -c "print('analysis: not implemented yet')"

report:
	$(PY) -c "print('report (Quarto): not implemented yet')"

all: ingest build test analysis report
