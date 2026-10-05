#!/bin/bash
# Patch existing LF snap dirs only with p1d_per_class.h5.  No dependency.
# Run in parallel with the HiRes job that is still in flight.

#SBATCH --job-name=hcd_patch_LF
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=21
#SBATCH --mem-per-cpu=4g
#SBATCH --time=01:30:00
#SBATCH --output=logs/patch_lf_%j.out
#SBATCH --error=logs/patch_lf_%j.err

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"   # this checkout: submit from its root (emulator-debug 2026-10)
cd "$REPO" || exit 2
[ -f "$REPO/hcd_analysis/paths.py" ] || { echo "submit from the repository root (got $REPO)" >&2; exit 2; }
set -euo pipefail
cd $REPO
PYTHON="/sw/pkgs/arc/mamba/py3.11/bin/python3"

echo "=== patch LF per_class  start $(date) ==="
"$PYTHON" scripts/patch_per_class_p1d.py --n-workers "${SLURM_CPUS_PER_TASK:-4}" --lf-only
echo "=== patch LF per_class  end   $(date) ==="
