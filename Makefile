# Project venv lives OUTSIDE iCloud so uv hardlinks from its global cache (no
# duplicated bytes) and iCloud never syncs it. Override with IMGEN_VENV.
IMGEN_VENV ?= $(HOME)/.venvs/imgen
PY := $(IMGEN_VENV)/bin/python

.PHONY: install test lint fmt style platform clean

# Create the venv (outside iCloud) and install imgen (editable) + the platform
# backend extra. Also installs the `imgen` CLI entry point into the venv.
install:
	./scripts/setup.sh

test:
	$(PY) -m pytest -m "not integration"

# Format the code in place (ruff format).
fmt:
	$(PY) -m ruff format src tests

# Format in place, then lint (developer convenience).
style: fmt
	$(PY) -m ruff check src tests

# Lint + verify formatting + type-check (no changes made). The merge gate.
lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests
	$(PY) -m mypy src tests

platform:
	$(PY) -m imgen.cli platform

clean:
	rm -rf "$(IMGEN_VENV)"
