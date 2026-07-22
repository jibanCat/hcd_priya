#!/bin/bash
# === ATTIC (2026-07-22 freeze triage execution; dispositions PI-approved 2026-07-20, hcd_priya_notes docs/superpowers/2026-07-20-freeze-script-triage.md). NOT IN THE FROZEN FORWARD. ===
# Step-A closure / slope-model era diagnostic batch.
# cavestru1 SLURM — HCD-SLOPE-MODEL SELECTION: the SAME 4 per-survey closure mocks as the slfix
# batch, but with the 2D AMPLITUDE×TILT HCD model (hcd_2d_tilt=True + hierarchical_hcd=True) instead
# of the fixed/marginalized 1D re-centered power-law. The 2D model lets the DATA float the global HCD
# z-tilt B_HCD (centered on HCD_INCIDENCE_SLOPE[0]=2.465 with FIXED per-class δs_c). Same corrected
# forward (slope fix 3603522), same mocks/fold/sim/seed → byte-identical mock data + noise, so the
# ONLY change vs checkpoints/stepA_slfix/ is the HCD-slope MODEL. Purpose: pick the production HCD-slope
# model — diff the n_s/A_p posterior of 2D-tilt vs 1D-recenter vs truth_vec per survey.
#
# HCD-MODEL FLAGS are threaded via the env var STEPA_2DTILT_MOCKS (run_stepA.build_config flips
# hcd_2d_tilt=True + hierarchical_hcd=True on EXACTLY those mock_ids; byte-identical when unset). The
# 2D-tilt model changes the HCD sample-site structure (adds B_HCD after A_hcd, replaces the per-class
# z-slope sampling with s_c = B_HCD + δs_c) — watch the smoke for site-order / divergence issues.
#
# SEPARATE OUTPUT DIR: STEPA_CKPT_DIR=checkpoints/stepA_2dtilt => run_stepA writes <id>.npz THERE
# (same mock id, different dir) so the 1D-recenter checkpoints/stepA_slfix/<id>.npz baseline is
# UNTOUCHED and the health.json single-writer race is avoided (per-batch dir). Restartable: skip-if-exists.
#
# SAME 4 representative mocks as slfix (4 chains each, --array=0-15):
#   DESI     D_f3_c{0..3}       (~1.7 CPU-h/chain)   compare: checkpoints/stepA_slfix/D_f3_c*.npz
#   KS       K_f4_c{0..3}       (~0.07 CPU-h/chain)  compare: checkpoints/stepA_slfix/K_f4_c*.npz
#   DESI+KS  XS_f6_s0_c{0..3}   (~1.4 CPU-h/chain)   compare: checkpoints/stepA_slfix/XS_f6_s0_c*.npz
#   eBOSS    E_f5_c{0..3}       (~1.0 CPU-h/chain)   compare: checkpoints/stepA_slfix/E_f5_c*.npz
# Total ~= 17 CPU-h (the 2D-tilt adds 1 sampled site → modest overhead; budget ~20-30 CPU-h).
#
# Usage:
#   sbatch --array=0-15 scripts/batch_stepA_cosmo_2dtilt.sh
#
#SBATCH --job-name=stepA_cosmo_2dtilt
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/stepA_cosmo_2dtilt_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/stepA_cosmo_2dtilt_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

REPO=/home/mfho/hcd_priya
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
export STEPA_CKPT_DIR=$REPO/checkpoints/stepA_2dtilt
mkdir -p "$STEPA_CKPT_DIR" "$REPO/logs"

# HCD-SLOPE MODEL = 2D amplitude×tilt + hierarchical, on EXACTLY the 4 per-survey closure mocks.
export STEPA_2DTILT_MOCKS=default

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"
export PYTHONNOUSERSITE=1 PYTHONPATH=$REPO JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""

# 16 chain ids = 4 representative mocks (one per survey) x 4 chains. Order matches --array=0-15.
CHAINS=(
  D_f3_c0 D_f3_c1 D_f3_c2 D_f3_c3                 # DESI
  K_f4_c0 K_f4_c1 K_f4_c2 K_f4_c3                 # KS
  XS_f6_s0_c0 XS_f6_s0_c1 XS_f6_s0_c2 XS_f6_s0_c3 # DESI+KS
  E_f5_c0 E_f5_c1 E_f5_c2 E_f5_c3                 # eBOSS
)
TID=${SLURM_ARRAY_TASK_ID:-0}
if [ "$TID" -ge "${#CHAINS[@]}" ]; then
  echo "ERROR: task $TID >= ${#CHAINS[@]}" >&2; exit 1
fi
CHAIN=${CHAINS[$TID]}

if [[ -f "$STEPA_CKPT_DIR/${CHAIN}.npz" ]]; then
  echo "=== ${CHAIN} (2dtilt) checkpoint exists — SKIP (task ${TID}) ==="
  exit 0
fi

echo "=== cosmo-impact 2DTILT (2D amplitude×tilt HCD model, full forward): ${CHAIN} -> ${STEPA_CKPT_DIR} (task ${TID}) start: $(date) ==="
"$PY" -u scripts/run_stepA.py --run-one "$CHAIN" \
    --n-warmup 250 --n-samples 400 --max-tree-depth 10 --target-accept 0.9
echo "=== done ${CHAIN}: $(date) ==="
