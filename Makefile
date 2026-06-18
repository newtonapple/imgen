# Project venv lives OUTSIDE iCloud so uv hardlinks from its global cache (no
# duplicated bytes) and iCloud never syncs it. Override with IMAGEGEN_VENV.
IMAGEGEN_VENV ?= $(HOME)/.venvs/imagegen
PY := $(IMAGEGEN_VENV)/bin/python

.PHONY: install test lint platform clean

# Create the venv (outside iCloud) and install imagegen (editable) + the platform
# backend extra. Also installs the `imagegen` CLI entry point into the venv.
install:
	./scripts/setup.sh

test:
	$(PY) -m pytest -m "not integration"

lint:
	$(PY) -m ruff check src tests

platform:
	$(PY) -m imagegen.cli platform

clean:
	rm -rf "$(IMAGEGEN_VENV)"
