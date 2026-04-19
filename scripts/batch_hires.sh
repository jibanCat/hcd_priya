#!/bin/bash
# Great Lakes SLURM batch script for HiRes simulations (3 sims, 2x npart).
#
# Usage:
#   sbatch scripts/batch_hires.sh
#
# Processes 3 HiRes sims from /nfs/turbo/lsa-cavestru/mfho/priya/emu_full_hires/
# Outputs go to /scratch/cavestru_root/cavestru0/mfho/hcd_outputs_hires/

#SBATCH --job-name=hcd_hires
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8g
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/hcd_hires_%j.out
#SBATCH --error=/home/mfho/hcd_priya/logs/hcd_hires_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu

set -euo pipefail

HCD_ROOT="/home/mfho/hcd_priya"
PYTHON="/sw/pkgs/arc/mamba/py3.11/bin/python3"
HCD_CONFIG="${HCD_CONFIG:-${HCD_ROOT}/config/default.yaml}"
OUTPUT_ROOT="/scratch/cavestru_root/cavestru0/mfho/hcd_outputs"   # hires/ subdir created by pipeline

mkdir -p "${HCD_ROOT}/logs"
cd "${HCD_ROOT}"

echo "=== hcd_analysis pipeline (HiRes) ==="
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "Output:   ${OUTPUT_ROOT}/hires/"
echo "CPUs:     ${SLURM_CPUS_PER_TASK:-4}"

"$PYTHON" -m cli.run run-hires \
  --config "$HCD_CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --n-workers "${SLURM_CPUS_PER_TASK:-4}" \
  --verbose

echo "=== HiRes done: $(date) ==="
