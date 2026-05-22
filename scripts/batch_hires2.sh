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
# fixed (1+z)^2*L*H0/c formula). But discover_sim_snap_pairs + the tau0 builder
# require cddf_corrected.npz / cddf_stacked_corrected.npz (the legacy patched
# name). Since the fresh cddf is ALREADY correct, the "corrected" file is just a
# COPY — do NOT run patch_cddf_dx.py here (it would wrongly divide by (1+z)*h
# a second time). This makes all 6 reprocessed sims discoverable.
echo "=== materialising cddf_corrected.npz copies (fresh cddf is already correct) ==="
for snap in "${OUTPUT_ROOT}"/hires/ns*/snap_*/; do
  [ -f "${snap}/cddf.npz" ] && cp -f "${snap}/cddf.npz" "${snap}/cddf_corrected.npz"
done
for sim in "${OUTPUT_ROOT}"/hires/ns*/; do
  [ -f "${sim}/cddf_stacked.npz" ] && cp -f "${sim}/cddf_stacked.npz" "${sim}/cddf_stacked_corrected.npz"
done

echo "=== HiRes all-6 done: $(date) ==="
