# Virtualenv lives outside iCloud; override with IMAGEGEN_VENV.
VENV ?= $(HOME)/.cache/imagegen-venv
PY := $(VENV)/bin/python

.PHONY: setup test lint platform

setup:
	./scripts/setup.sh

test:
	$(PY) -m pytest -m "not integration"

lint:
	$(PY) -m ruff check src tests

platform:
	$(PY) -m imagegen.cli platform
