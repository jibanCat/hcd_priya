#!/bin/bash
# Great Lakes SLURM batch script for HiRes simulations (3 sims, 2x npart).
#
# Usage:
#   sbatch scripts/batch_hires.sh
#
# Runs the 3 HiRes sims from /nfs/turbo/lsa-cavestru/mfho/priya/emu_full_hires/
# Outputs to ./outputs/hires/<sim>/...
# After completion, run: sbatch scripts/batch_greatlakes.sh convergence

#SBATCH --job-name=hcd_hires
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4              # 3 HiRes sims x 1 worker each
#SBATCH --mem-per-cpu=8g              # 32 GB total (smaller than LF run)
#SBATCH --time=02:00:00               # HiRes: fewer sims, ~same per-snap time
#SBATCH --output=logs/hcd_hires_%j.out
#SBATCH --error=logs/hcd_hires_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu

set -euo pipefail

module load python/3.11 2>/dev/null || true
source activate hcd_env 2>/dev/null || conda activate hcd_env 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HCD_ROOT="${HCD_ROOT:-$(dirname "$SCRIPT_DIR")}"
cd "$HCD_ROOT"

HCD_CONFIG="${HCD_CONFIG:-config/default.yaml}"
mkdir -p logs

echo "=== hcd_analysis pipeline (HiRes) ==="
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "Root:     $HCD_ROOT"
echo "CPUs:     ${SLURM_CPUS_PER_TASK:-4}"

python -m hcd_analysis.cli.run run-hires \
  --config "$HCD_CONFIG" \
  --n-workers "${SLURM_CPUS_PER_TASK:-4}" \
  --verbose

echo "=== HiRes done: $(date) ==="
echo ""
echo "Next step: compute convergence ratios with:"
echo "  sbatch scripts/batch_convergence.sh"
