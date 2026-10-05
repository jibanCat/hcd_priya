#!/bin/bash
# Great Lakes — 1-PAIR TIMING for the v3.3 (uniform) production tau0 build.
# Measures per-pair wall, MaxRSS, TotalCPU, and AllocTRES(billing) so we can
# size + cost the production array under the cavestru0 budget (<~4000 CPU-h).
# Submit:  sbatch scripts/batch_tau0_timing.sh
#SBATCH --job-name=tau0_timing
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48g
#SBATCH --time=04:00:00
#SBATCH --output=logs/tau0_timing_%j.out
#SBATCH --error=logs/tau0_timing_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"   # this checkout: submit from its root (emulator-debug 2026-10)
cd "$REPO" || exit 2
[ -f "$REPO/hcd_analysis/paths.py" ] || { echo "submit from the repository root (got $REPO)" >&2; exit 2; }
set -euo pipefail
export LD_LIBRARY_PATH=/sw/pkgs/arc/stacks/gcc/10.3.0/gsl/2.7/lib:/home/mfho/.conda/envs/emu-3.9/lib:${LD_LIBRARY_PATH:-}
PY=/home/mfho/.conda/envs/emu-3.9/bin/python3
echo "=== tau0 1-pair timing start: $(date) ==="
/usr/bin/time -v "$PY" scripts/build_emulator_cache_tau0.py \
    --fidelity lf --offset 0 --limit 1 \
    --output /scratch/cavestru_root/cavestru0/mfho/timing_one_pair_v33.h5 \
    --spot-check
echo "=== tau0 1-pair timing done: $(date) ==="
