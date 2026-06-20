# Project venv lives OUTSIDE iCloud so uv hardlinks from its global cache (no
# duplicated bytes) and iCloud never syncs it. Override with IMAGEGEN_VENV.
IMAGEGEN_VENV ?= $(HOME)/.venvs/imagegen
PY := $(IMAGEGEN_VENV)/bin/python

.PHONY: install test lint fmt style platform clean

# Create the venv (outside iCloud) and install imagegen (editable) + the platform
# backend extra. Also installs the `imagegen` CLI entry point into the venv.
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

# Lint + verify formatting (no changes made).
lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

platform:
	$(PY) -m imagegen.cli platform

clean:
	rm -rf "$(IMAGEGEN_VENV)"
