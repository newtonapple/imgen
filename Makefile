# In-project virtualenv. Created by `make install`.
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: install test lint platform clean

# Create .venv and install imagegen (editable) + the platform backend extra.
# This also installs the `imagegen` CLI entry point into the venv.
install:
	./scripts/setup.sh

test:
	$(PY) -m pytest -m "not integration"

lint:
	$(PY) -m ruff check src tests

platform:
	$(PY) -m imagegen.cli platform

clean:
	rm -rf $(VENV)
