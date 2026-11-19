#!/bin/bash
# Installation script for Great Lakes cluster.
#
# Run once to set up the conda environment and install this package.
#
# Usage:
#   bash scripts/install_greatlakes.sh
#
# What this does:
#   1. Creates a conda env 'hcd_env' with Python 3.11
#   2. Installs fake_spectra from GitHub
#   3. Installs this package in editable mode, with the pinned versions of the supported environment
#      (tools/ci/constraints-devenv.txt) and the tutorials extra

set -euo pipefail

ENV_NAME="hcd_env"
PYTHON_VERSION="3.11"

echo "=== hcd-priya (hcd_analysis) installation on Great Lakes ==="

# Load modules if needed
module load python/3.11 2>/dev/null || true

# Create environment
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda env '${ENV_NAME}' already exists. Activating."
else
    echo "Creating conda env '${ENV_NAME}'..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "Python: $(python --version)"

pip install --upgrade pip

# ---------------------------------------------------------------------------
# fake_spectra from Simeon Bird's GitHub
# ---------------------------------------------------------------------------
echo "Installing fake_spectra..."

# fake_spectra requires a C extension (spectra_priv.so) which is compiled
# during pip install. Make sure gcc is available.
# On Great Lakes: module load gcc/10.3.0 may be needed.
module load gcc 2>/dev/null || true

pip install git+https://github.com/sbird/fake_spectra.git

# Verify
python -c "import fake_spectra; print('fake_spectra OK:', fake_spectra.__file__)"

# ---------------------------------------------------------------------------
# Install this package
# ---------------------------------------------------------------------------
echo "Installing hcd-priya and its dependencies..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"
# hcd-priya requires the model package hcd-priya-models, which is not on the package index yet: build a stand-in
# with a synthetic bundle (untrained networks; tests and development only) and install both, then make the install
# editable. With a real model wheel in hand, install that file in place of the stand-in.
pip install build
python -m build --wheel --outdir dist .
tools/ci/install_with_synthetic_models.sh python "$(ls -t dist/hcd_priya-*.whl | head -n 1)[tutorials]" \
    dist/models -c tools/ci/constraints-devenv.txt
pip install pytest  # the synthetic stand-in is written with the test suite's builder
pip install --no-deps -e .

# Verify
python -c "import hcd_analysis; print('hcd_analysis OK:', hcd_analysis.__version__)"
python -c "from hcd_analysis.voigt_utils import tau_voigt; print('voigt_utils OK')"

echo ""
echo "=== Installation complete ==="
echo "Activate environment with: conda activate ${ENV_NAME}"
echo "Run pipeline with: hcd --help"
echo ""
echo "To test with one (sim, snap):"
echo "  hcd run-one --config config/default.yaml --sim ns0.803 --snap 17 --debug"
echo ""
echo "To benchmark:"
echo "  hcd benchmark --config config/default.yaml --n-sims 1 --n-snaps 2"
