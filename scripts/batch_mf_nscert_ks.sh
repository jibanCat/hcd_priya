#!/bin/bash
# cavestru1 SLURM — Phase-5a MF n_s HIGH-K CERTIFICATION on KS-ONLY (PI-approved 2026-06-14).
# PER-SURVEY headline cert: the real fits are SEPARATE, so the KS leg gets its own genuine HF-LOSO
# Test B THROUGH the production MF forward (the HFLOSO_KS* fiducials in run_stepA.build_config).
# Per HR sim: truth = its REAL measured P1D, forward = LF emu × MF correction fit EXCLUDING it,
# KS-ONLY leg (survey="KS"; DESI leg OFF, with_eboss=False; KS carries its loader-default
# mf_floor_on=True). KS reaches higher k than DESI, so this is the decisive high-k tilt test for
# the KS real fit. VERDICT GATE: per-fold n_s |bias_z| < 1 (ideally <0.2σ). 5 HR sims × 4 chains =
# 20 chains; each is one ARRAY TASK, one core, single-thread (the validated run_stepA --run-one
# path). Restartable: a task whose checkpoint exists is SKIPPED. Read the verdict (per-survey
# DESI-only vs KS-only) with analyze_mf_nscert_dk.py.
#
# Usage:
#   sbatch --array=0-19 scripts/batch_mf_nscert_ks.sh
#
#SBATCH --job-name=mf_nscert_ks
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/mf_nscert_ks_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/mf_nscert_ks_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

REPO=/home/mfho/hcd_priya
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
CKPT=$REPO/checkpoints/stepA
mkdir -p "$CKPT" "$REPO/logs"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"
export PYTHONNOUSERSITE=1 PYTHONPATH=$REPO JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""

# The 20 KS-only Test-B chain ids (5 HR sims × 4 chains), in a STABLE order matching --array=0-19.
SIMS=(859 885 909 972 979)
TID=${SLURM_ARRAY_TASK_ID:-0}
SIDX=$(( TID / 4 ))
CIDX=$(( TID % 4 ))
CHAIN="HFLOSO_KS${SIMS[$SIDX]}_c${CIDX}"

if [[ -f "$CKPT/${CHAIN}.npz" ]]; then
  echo "=== ${CHAIN} checkpoint exists — SKIP (task ${TID}) ==="
  exit 0
fi

echo "=== MF n_s cert KS-only: ${CHAIN} (task ${TID}) start: $(date) ==="
"$PY" -u scripts/run_stepA.py --run-one "$CHAIN" \
    --n-warmup 250 --n-samples 400 --max-tree-depth 10 --target-accept 0.9
echo "=== done ${CHAIN}: $(date) ==="
