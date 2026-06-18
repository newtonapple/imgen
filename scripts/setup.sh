#!/usr/bin/env bash
# Create the project virtualenv OUTSIDE iCloud and install imagegen (editable) +
# the matching backend extra. Living outside iCloud lets uv hardlink packages
# from its global cache (no duplicated bytes) and keeps the venv off iCloud sync.
# The install also provides the `imagegen` CLI entry point.
#
#   Apple Silicon (Darwin/arm64) -> MLX backend  (extra: mlx)
#   Linux                        -> PyTorch/CUDA (extra: cuda)
#
# Override the venv location with IMAGEGEN_VENV.
set -euo pipefail
cd "$(dirname "$0")/.."

OS="$(uname -s)"
ARCH="$(uname -m)"
VENV="${IMAGEGEN_VENV:-$HOME/.venvs/imagegen}"

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
echo "venv: $VENV  (outside iCloud; override with IMAGEGEN_VENV)"

uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" -e ".[$EXTRA,dev]"

echo
echo "Done. The 'imagegen' CLI is installed in $VENV:"
echo "  source $VENV/bin/activate && imagegen --help"
echo "Run tests with:  make test"
