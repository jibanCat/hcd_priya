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
#   2. Installs all requirements
#   3. Installs fake_spectra from GitHub
#   4. Installs this package in editable mode

set -euo pipefail

ENV_NAME="hcd_env"
PYTHON_VERSION="3.11"

echo "=== hcd_analysis installation on Great Lakes ==="

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

# ---------------------------------------------------------------------------
# Core dependencies
# ---------------------------------------------------------------------------
echo "Installing core dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

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
echo "Installing hcd_analysis..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"
pip install -e .

# Verify
python -c "import hcd_analysis; print('hcd_analysis OK:', hcd_analysis.__version__)"
python -c "from hcd_analysis.voigt_utils import tau_voigt; print('voigt_utils OK')"

echo ""
echo "=== Installation complete ==="
echo "Activate environment with: conda activate ${ENV_NAME}"
echo "Run pipeline with: hcd --help"
echo ""
echo "To test with one (sim, snap):"
echo "  hcd run-one --sim ns0.803 --snap 17 --debug"
echo ""
echo "To benchmark:"
echo "  hcd benchmark --n-sims 1 --n-snaps 2"
