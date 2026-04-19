#!/bin/bash
# Compute HiRes/LowRes convergence ratios after both campaigns complete.
#
# Usage:
#   sbatch --dependency=afterok:<LF_JOB>,<HIRES_JOB> scripts/batch_convergence.sh

#SBATCH --job-name=hcd_convergence
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8g
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/hcd_convergence_%j.out
#SBATCH --error=/home/mfho/hcd_priya/logs/hcd_convergence_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu

set -euo pipefail

HCD_ROOT="/home/mfho/hcd_priya"
PYTHON="/sw/pkgs/arc/mamba/py3.11/bin/python3"
HCD_CONFIG="${HCD_CONFIG:-${HCD_ROOT}/config/default.yaml}"
OUTPUT_ROOT="/scratch/cavestru_root/cavestru0/mfho/hcd_outputs"

mkdir -p "${HCD_ROOT}/logs"
cd "${HCD_ROOT}"

echo "=== hcd_analysis convergence ratios ==="
echo "Date:     $(date)"
echo "Output:   ${OUTPUT_ROOT}/hires/"

"$PYTHON" -m cli.run convergence \
  --config "$HCD_CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --verbose

echo "=== Done: $(date) ==="
