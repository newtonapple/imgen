#!/usr/bin/env bash
# Detect the platform, create the in-project virtualenv (.venv), and install
# imagegen (editable) + the matching backend extra. This also installs the
# `imagegen` CLI entry point into .venv.
#
#   Apple Silicon (Darwin/arm64) -> MLX backend  (extra: mlx)
#   Linux                        -> PyTorch/CUDA (extra: cuda)
set -euo pipefail
cd "$(dirname "$0")/.."

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS/$ARCH" in
  Darwin/arm64 | Darwin/aarch64)
    EXTRA="mlx" ;;
  Linux/*)
    EXTRA="cuda" ;;
  *)
    echo "Unsupported platform: $OS/$ARCH" >&2
    exit 1 ;;
esac

echo "Detected $OS/$ARCH -> installing extra: [$EXTRA]"

# Non-editable install: copies imagegen into the venv so the `imagegen` CLI works
# reliably. (Hatchling's editable .pth is not honored when the venv lives under a
# path like iCloud's; tests read from src/ via pyproject's pytest pythonpath.)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python ".[$EXTRA,dev]"

echo
echo "Done. The 'imagegen' CLI is installed in .venv:"
echo "  source .venv/bin/activate && imagegen --help"
echo "  # or without activating:  .venv/bin/imagegen run ..."
echo "Run tests with:  make test"
