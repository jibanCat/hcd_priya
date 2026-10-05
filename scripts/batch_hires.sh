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
#SBATCH --cpus-per-task=21
#SBATCH --mem-per-cpu=8g
#SBATCH --time=24:00:00
#SBATCH --output=logs/hcd_hires_%j.out
#SBATCH --error=logs/hcd_hires_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"   # this checkout: submit from its root (emulator-debug 2026-10)
cd "$REPO" || exit 2
[ -f "$REPO/hcd_analysis/paths.py" ] || { echo "submit from the repository root (got $REPO)" >&2; exit 2; }
set -euo pipefail

HCD_ROOT="$REPO"
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

# n_workers=3   : 3 HiRes sims in parallel
# n_workers_skewer=7 : 7 CPUs per sim for intra-snap skewer parallelism (21/3)
"$PYTHON" -m cli.run run-hires \
  --config "$HCD_CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --n-workers 3 \
  --set "n_workers_skewer=7" \
  --verbose

echo "=== HiRes done: $(date) ==="
