#!/bin/bash
# Patch existing snap dirs with p1d_per_class.h5 (no other output touched).
#
# Usage:  sbatch scripts/batch_patch_per_class.sh

#SBATCH --job-name=hcd_patch_per_class
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=21
#SBATCH --mem-per-cpu=4g
#SBATCH --time=02:00:00
#SBATCH --output=logs/patch_per_class_%j.out
#SBATCH --error=logs/patch_per_class_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"   # this checkout: submit from its root (emulator-debug 2026-10)
cd "$REPO" || exit 2
[ -f "$REPO/hcd_analysis/paths.py" ] || { echo "submit from the repository root (got $REPO)" >&2; exit 2; }
set -euo pipefail
cd $REPO
PYTHON="/sw/pkgs/arc/mamba/py3.11/bin/python3"

echo "=== patch_per_class_p1d  start $(date) ==="
"$PYTHON" scripts/patch_per_class_p1d.py --n-workers "${SLURM_CPUS_PER_TASK:-4}"
echo "=== patch_per_class_p1d  end   $(date) ==="
