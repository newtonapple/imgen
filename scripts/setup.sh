#!/usr/bin/env bash
# Detect the platform and install the matching dependency set with uv.
#
#   Apple Silicon (Darwin/arm64) -> MLX backend  (extra: mlx)
#   Linux                        -> PyTorch/CUDA (extra: cuda)
#
# The virtualenv is created OUTSIDE the repo (the repo lives in iCloud) so iCloud
# does not churn on venv files. Override with IMAGEGEN_VENV.
set -euo pipefail
cd "$(dirname "$0")/.."

OS="$(uname -s)"
ARCH="$(uname -m)"
VENV="${IMAGEGEN_VENV:-$HOME/.cache/imagegen-venv}"

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
echo "venv: $VENV  (override with IMAGEGEN_VENV)"

uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" -e ".[$EXTRA,dev]"

echo
echo "Done. Run tests with:  IMAGEGEN_VENV=$VENV make test"
echo "Or activate with:      source $VENV/bin/activate"
