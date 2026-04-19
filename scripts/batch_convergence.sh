#!/bin/bash
# Compute HiRes/LowRes convergence ratios after both campaigns are complete.
#
# Usage:
#   sbatch scripts/batch_convergence.sh
#
# Requires: LF outputs in ./outputs/ and HiRes outputs in ./outputs/hires/

#SBATCH --job-name=hcd_convergence
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8g
#SBATCH --time=00:30:00
#SBATCH --output=logs/hcd_convergence_%j.out
#SBATCH --error=logs/hcd_convergence_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu

set -euo pipefail

module load python/3.11 2>/dev/null || true
source activate hcd_env 2>/dev/null || conda activate hcd_env 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HCD_ROOT="${HCD_ROOT:-$(dirname "$SCRIPT_DIR")}"
cd "$HCD_ROOT"

echo "=== hcd_analysis convergence ratios ==="
echo "Date: $(date)"

python -m hcd_analysis.cli.run convergence \
  --config "${HCD_CONFIG:-config/default.yaml}" \
  --verbose

echo "=== Done: $(date) ==="
