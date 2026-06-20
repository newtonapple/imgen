#!/usr/bin/env bash
# Create the project virtualenv OUTSIDE iCloud and install imgen (editable) +
# the matching backend extra. Living outside iCloud lets uv hardlink packages
# from its global cache (no duplicated bytes) and keeps the venv off iCloud sync.
# The install also provides the `ig` CLI entry point.
#
#   Apple Silicon (Darwin/arm64) -> MLX backend  (extra: mlx)
#   Linux                        -> PyTorch/CUDA (extra: cuda)
#
# Override the venv location with IMGEN_VENV.
set -euo pipefail
cd "$(dirname "$0")/.."

OS="$(uname -s)"
ARCH="$(uname -m)"
VENV="${IMGEN_VENV:-$HOME/.venvs/imgen}"

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
echo "venv: $VENV  (outside iCloud; override with IMGEN_VENV)"

uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" -e ".[$EXTRA,dev]"

echo
echo "Done. The 'ig' CLI is installed in $VENV:"
echo "  source $VENV/bin/activate && ig --help"
echo "Run tests with:  make test"
