#!/bin/bash
# Great Lakes SLURM batch — re-run Phase-1 for the single LF sim ns0.907.
#
# Why: ns0.907's emu_full was re-dumped to the corrected snapshot ladder (z=3.0
# at SPECTRA_017), but its Phase-1 outputs were never re-run, so hcd_outputs was
# stale (snap_017 meta z=2.8 vs raw z=3.0; no on-grid z=3.0 row). resume:false
# forces a from-scratch reprocess against the current emu_full.
#
# SPECTRA_015 is a spurious 32k-skewer duplicate of z=3.2 (no grid_480); Phase-1
# may produce a junk snap_015, but the tau0 cache builder skips it (grid_480
# required — see discover_tau0_pairs hardening).
#
# Usage:  sbatch scripts/batch_lf_ns0907.sh

#SBATCH --job-name=hcd_lf_ns0907
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=8g
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/hcd_lf_ns0907_%j.out
#SBATCH --error=/home/mfho/hcd_priya/logs/hcd_lf_ns0907_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu

set -euo pipefail
HCD_ROOT="/home/mfho/hcd_priya"
PYTHON="/sw/pkgs/arc/mamba/py3.11/bin/python3"
OUTPUT_ROOT="/scratch/cavestru_root/cavestru0/mfho/hcd_outputs"
mkdir -p "${HCD_ROOT}/logs"; cd "${HCD_ROOT}"

echo "=== Phase-1 re-run: LF ns0.907 (resume:false) === $(date)"
"$PYTHON" -m cli.run run-sim \
  --sim ns0.907Ap1.5e-09 \
  --config config/default.yaml \
  --output-root "$OUTPUT_ROOT" \
  --set resume=false \
  --set n_workers_skewer=20 \
  --verbose

# SUPERSEDED by the all-60 LF re-run (batch_lf_rerun_all.sh, job 50696155).
# Kept for reference. As of 2026-05-22 the cddf_corrected.npz dual naming is
# retired: discover_sim_snap_pairs + the tau0 builder read cddf.npz directly, so
# no copy step is needed. The (1+z)*h dX bug is fixed in current code, so the
# fresh cddf is already correct (do NOT run patch_cddf_dx.py — it would
# double-divide).
echo "=== ns0.907 Phase-1 re-run done: $(date) ==="
