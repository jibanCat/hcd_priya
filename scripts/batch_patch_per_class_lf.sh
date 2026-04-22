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
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/patch_lf_%j.out
#SBATCH --error=/home/mfho/hcd_priya/logs/patch_lf_%j.err

set -euo pipefail
cd /home/mfho/hcd_priya
PYTHON="/sw/pkgs/arc/mamba/py3.11/bin/python3"

echo "=== patch LF per_class  start $(date) ==="
"$PYTHON" scripts/patch_per_class_p1d.py --n-workers "${SLURM_CPUS_PER_TASK:-4}" --lf-only
echo "=== patch LF per_class  end   $(date) ==="
