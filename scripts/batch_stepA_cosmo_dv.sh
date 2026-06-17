#!/bin/bash
# cavestru1 SLURM — FINALIZED-PRIOR VALIDATING CLOSURE SWEEP (DV arms, PI 2026-06-17).
# Re-validates the CORRECTED + FINALIZED HCD prior (a04d501 litWLS γ_LLS=2.127 + 25a51bc z=3
# pivot-construction fix) on the real-fit-direction LLS center, through the FULL production forward.
# ALL arms: prior_center="lit" (per-survey effective-LLS pin) + zslope_realfit=True (forward LLS
# z-slope centered on the litWLS γ_LLS=2.127 the REAL fit uses). σ_LLS=0.15 = production 1× width.
#
# SEPARATE OUTPUT DIR: STEPA_CKPT_DIR=checkpoints/stepA_dv => run_stepA writes <id>.npz THERE, so the
# existing checkpoints/stepA*/<id>.npz baselines are UNTOUCHED and the health.json single-writer race
# is avoided (per-batch dir). Restartable: skip-if-exists.
#
# 5 mocks x 4 chains = 20 chains (--array=0-19):
#   DV_f3_c{0..3}        DESI fold3 (n_s≈0.90), lit-boosted — the DESI n_s-pull arm (→ <0.5σ?)
#   DV_XS_f6_s0_c{0..3}  DESI+KS fold6 Planck, sim_mean subDLA (shift 0) — the joint-leg closure
#   DV_sig15_c{0..3}     σ-isolator 1× (σ_LLS=0.15) on the median-w_LLS sim — LOAD-BEARING
#   DV_sig30_c{0..3}     σ-isolator 2× (σ_LLS=0.30) on the SAME sim/noise — does +1.01σ A_p reappear?
#   DV_littruth_c{0..3}  real-fit-direction LEAK arm (lit-boosted, σ0.15) — leak gate (≈0.00σ)
# DESI chains ~1.0-2.0 CPU-h, DESI+KS ~1.2-1.5 CPU-h, median-w sim ~0.9-1.1 CPU-h.
# Total budget ~20-30 CPU-h. (cavestru1 cap 5000 CPU-h — far under.)
#
# Usage:
#   sbatch --array=0-19 scripts/batch_stepA_cosmo_dv.sh
#   # then compare (see the manifest): pool each DV_*_c* via the battery bias_z convention,
#   #   σ-decision = DV_sig15 vs DV_sig30 bias_Ap; DV_f3 bias_ns; DV_littruth bias_ns (leak ≈0).
#
#SBATCH --job-name=stepA_cosmo_dv
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/stepA_cosmo_dv_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/stepA_cosmo_dv_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

REPO=/home/mfho/hcd_priya
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
export STEPA_CKPT_DIR=$REPO/checkpoints/stepA_dv
mkdir -p "$STEPA_CKPT_DIR" "$REPO/logs"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"
export PYTHONNOUSERSITE=1 PYTHONPATH=$REPO JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""

# 20 chain ids = 5 mocks x 4 chains. Order matches --array=0-19.
CHAINS=(
  DV_f3_c0 DV_f3_c1 DV_f3_c2 DV_f3_c3                         # DESI n_s-pull arm
  DV_XS_f6_s0_c0 DV_XS_f6_s0_c1 DV_XS_f6_s0_c2 DV_XS_f6_s0_c3 # DESI+KS joint closure
  DV_sig15_c0 DV_sig15_c1 DV_sig15_c2 DV_sig15_c3             # σ-isolator 1× (LOAD-BEARING)
  DV_sig30_c0 DV_sig30_c1 DV_sig30_c2 DV_sig30_c3             # σ-isolator 2× (same sim/noise)
  DV_littruth_c0 DV_littruth_c1 DV_littruth_c2 DV_littruth_c3 # real-fit-direction leak arm
)
TID=${SLURM_ARRAY_TASK_ID:-0}
if [ "$TID" -ge "${#CHAINS[@]}" ]; then
  echo "ERROR: task $TID >= ${#CHAINS[@]}" >&2; exit 1
fi
CHAIN=${CHAINS[$TID]}

if [[ -f "$STEPA_CKPT_DIR/${CHAIN}.npz" ]]; then
  echo "=== ${CHAIN} (dv) checkpoint exists — SKIP (task ${TID}) ==="
  exit 0
fi

echo "=== finalized-prior closure DV (lit center + zslope_realfit, full forward): ${CHAIN} -> ${STEPA_CKPT_DIR} (task ${TID}) start: $(date) ==="
"$PY" -u scripts/run_stepA.py --run-one "$CHAIN" \
    --n-warmup 250 --n-samples 400 --max-tree-depth 10 --target-accept 0.9
echo "=== done ${CHAIN}: $(date) ==="
