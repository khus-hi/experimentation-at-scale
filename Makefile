PYTHON ?= python3.11
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: setup ingest build test analysis report all

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	$(PY) -c "import sys; print(sys.executable)"

ingest:
	$(PY) -m ingest.download
	$(PY) -m ingest.load

build:
	$(PY) -c "print('build (dbt): not implemented yet')"

test:
	$(PY) -m pytest
	$(PY) -m ruff check .

analysis:
	$(PY) -c "print('analysis: not implemented yet')"

report:
	$(PY) -c "print('report (Quarto): not implemented yet')"

all: ingest build test analysis report
