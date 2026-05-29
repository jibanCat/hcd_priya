#!/bin/bash
# Great Lakes SLURM batch script — Phase-1 (re)processing of ALL 6 HiRes sims
# from the emu_full_hires_2 6-sim reference.
#
# 3 of the 6 are fresh re-runs (not copies), so we regenerate every catalog from
# the canonical raw tau (config has resume:false). Outputs land under
#   /scratch/cavestru_root/cavestru0/mfho/hcd_outputs/hires/<sim>/snap_NNN/
# so the tau0 cache builder can discover all 6 consistently.
#
# Usage:
#   sbatch scripts/batch_hires2.sh
#
# NOTE: resume:false forces a from-scratch reprocess. If this job times out,
# re-submit AFTER flipping config/hires2.yaml resume:true so completed snaps are
# skipped (their catalogs are then valid).

#SBATCH --job-name=hcd_hires_all6
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=6g
#SBATCH --time=1-12:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/hcd_hires_all6_%j.out
#SBATCH --error=/home/mfho/hcd_priya/logs/hcd_hires_all6_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu

set -euo pipefail

HCD_ROOT="/home/mfho/hcd_priya"
PYTHON="/sw/pkgs/arc/mamba/py3.11/bin/python3"
HCD_CONFIG="${HCD_ROOT}/config/hires2.yaml"
OUTPUT_ROOT="/scratch/cavestru_root/cavestru0/mfho/hcd_outputs"   # hires/ subdir created by pipeline

mkdir -p "${HCD_ROOT}/logs"
cd "${HCD_ROOT}"

echo "=== hcd_analysis pipeline (HiRes: all 6 sims, reprocess) ==="
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "Config:   ${HCD_CONFIG}"
echo "Output:   ${OUTPUT_ROOT}/hires/"
echo "CPUs:     ${SLURM_CPUS_PER_TASK:-30}"

# n_workers=6          : the 6 HiRes sims in parallel
# n_workers_skewer=5   : 5 CPUs per sim for intra-snap skewer parallelism (30/6)
"$PYTHON" -m cli.run run-hires \
  --config "$HCD_CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --n-workers 6 \
  --set "n_workers_skewer=5" \
  --verbose

# The current pipeline writes a CORRECT cddf.npz (the (1+z)*h dX bug #7 was
# fixed in-code, commit c210990 — verified: ns0.972 dX_per_sightline matches the
# fixed (1+z)^2*L*H0/c formula). As of 2026-05-22 the legacy cddf_corrected.npz
# dual naming is RETIRED: discover_sim_snap_pairs + the tau0 builder now read
# cddf.npz directly, so no copy step is needed. Do NOT run patch_cddf_dx.py here
# (it would wrongly divide by (1+z)*h a second time on the already-correct cddf).

echo "=== HiRes all-6 done: $(date) ==="
