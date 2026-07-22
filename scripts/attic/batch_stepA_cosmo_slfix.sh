#!/bin/bash
# === ATTIC (2026-07-22 freeze triage execution; dispositions PI-approved 2026-07-20, hcd_priya_notes docs/superpowers/2026-07-20-freeze-script-triage.md). NOT IN THE FROZEN FORWARD. ===
# Step-A closure / slope-model era diagnostic batch.
# cavestru1 SLURM — COSMOLOGY-IMPACT per-survey closures on the CORRECTED-z-slope forward
# (slope fix 3603522, 2026-06-16). One representative closure mock per survey x 4 chains, run through
# the FULL production forward (NOT the NORC cert path — these are the standard closure mocks, which is
# what the real fit uses). Purpose: diff the n_s / A_p posterior vs the OLD-slope baseline checkpoints
# in checkpoints/stepA/<id>.npz to measure the cosmology SHIFT the slope fix induces per survey.
#
# SEPARATE OUTPUT DIR: STEPA_CKPT_DIR=checkpoints/stepA_slfix => run_stepA writes <id>.npz THERE
# (same mock id, different dir) so the OLD-slope checkpoints/stepA/<id>.npz baseline is UNTOUCHED and
# the health.json single-writer race is avoided (per-batch dir). Restartable: skip-if-exists.
#
# REPRESENTATIVE mocks (4 chains each, --array=0-15):
#   DESI     D_f3_c{0..3}       (~1.7 CPU-h/chain)   baseline: checkpoints/stepA/D_f3_c*.npz
#   KS       K_f4_c{0..3}       (~0.07 CPU-h/chain)  baseline: checkpoints/stepA/K_f4_c*.npz
#   DESI+KS  XS_f6_s0_c{0..3}   (~1.4 CPU-h/chain)   baseline: checkpoints/stepA/XS_f6_s0_c*.npz
#   eBOSS    E_f5_c{0..3}       (~1.0 CPU-h/chain)   baseline: checkpoints/stepA/E_f5_c*.npz
# Total ~= 17 CPU-h. (eBOSS has no HR cache so it is ABSENT from the Gate-A NORC cert — this batch is
# the ONLY corrected-forward eBOSS re-run.)
#
# Usage:
#   sbatch --array=0-15 scripts/batch_stepA_cosmo_slfix.sh
#
#SBATCH --job-name=stepA_cosmo_slfix
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/stepA_cosmo_slfix_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/stepA_cosmo_slfix_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

REPO=/home/mfho/hcd_priya
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
export STEPA_CKPT_DIR=$REPO/checkpoints/stepA_slfix
mkdir -p "$STEPA_CKPT_DIR" "$REPO/logs"

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
  echo "=== ${CHAIN} (slfix) checkpoint exists — SKIP (task ${TID}) ==="
  exit 0
fi

echo "=== cosmo-impact SLFIX (corrected slope, full forward): ${CHAIN} -> ${STEPA_CKPT_DIR} (task ${TID}) start: $(date) ==="
"$PY" -u scripts/run_stepA.py --run-one "$CHAIN" \
    --n-warmup 250 --n-samples 400 --max-tree-depth 10 --target-accept 0.9
echo "=== done ${CHAIN}: $(date) ==="
