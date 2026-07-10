#!/bin/bash
# cavestru1 SLURM -- DESI HCD-bias 3-rung MCMC (NEW standalone runner; SBC-safe, edits nothing).
# Fit the SAME matched DESI closure mock under three HCD configurations that differ ONLY by which
# per-class alpha the fit floats, and measure the (A_p, n_s) leverage + recovered dN/dX(z) / tau0:
#   task 0 -> clean : pin ALL THREE alpha ~ 0          (pure clean forest, P_model ~ P_clean)
#   task 1 -> tierp : DELTA-PIN alpha_LLS + alpha_subDLA at the (WRONG) deployed dN/dX pin centers,
#                     NOT marginalized (sigma->~0); pin alpha_DLA ~ 0  (fixed mis-correction)
#   task 2 -> marg  : deployed prior on all three      (the prior-sensitivity "on" arm)
# The override is a host-side ctx field swap (alpha_hcd_mu/alpha_hcd_sigma) inside
# scripts/desi_hcd_3rung_bias.py (the prior-sensitivity idiom). MATCHED 3-way A/B/C: held-out-sim
# mock (leg_a=False) => the truth is the sim's measured power thru the production MF, HCD-prior-
# INDEPENDENT, so all three rungs over the SAME (seed, fold, mock) share IDENTICAL data.
#
# Mock 0 -> held_out_sims(fold0)[0] (production ensemble final_prod_seed0..4, DESI leg, DR1 metals).
# ISOLATED job-name / logs / OUTDIR from the live Gate-B SBC arms. One rung per array task.
# Usage (the full fits -- do NOT run until smoke is green):
#   sbatch --array=0-2 scripts/batch_desi_hcd_3rung.sh
# Smoke (fast, all 3 rungs):
#   SMOKE=1 sbatch --array=0-2 scripts/batch_desi_hcd_3rung.sh        # (or run the 3 --smoke calls directly)
#
#SBATCH --job-name=desi_hcd_3rung
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/desi_hcd_3rung_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/desi_hcd_3rung_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

# array task -> rung
RUNGS=(clean tierp marg)
TID=${SLURM_ARRAY_TASK_ID:-0}
RUNG=${RUNGS[$TID]:?array index $TID out of range (use --array=0-2)}

MOCK=${MOCK:-0}
FOLD=${FOLD:-0}
N_MOCKS=${N:-8}                                       # for fold_in(seed,m) reproducibility
SEED=${SEED:-20260621}
# ISOLATED outdir per rung (separate from desi_hcd_prior_{on,off} and the SBC arms).
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru1/mfho/desi_hcd_3rung_${RUNG}}
NCPU=${SLURM_CPUS_PER_TASK:-4}
N_SAMPLES=${N_SAMPLES:-2000}
N_WARMUP=${N_WARMUP:-250}
SMOKE_FLAG=""
[[ "${SMOKE:-0}" == "1" ]] && SMOKE_FLAG="--smoke"

export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

if [[ -f "$OUTDIR/mock_$(printf %04d "$MOCK").pkl" ]]; then
  echo "=== rung ${RUNG} mock ${MOCK} pkl exists -- SKIP ==="; exit 0
fi

echo "=== desi_hcd_3rung RUNG=${RUNG} mock ${MOCK} (fold=${FOLD}, N_MOCKS=${N_MOCKS}, seed=${SEED}, warmup=${N_WARMUP} samples=${N_SAMPLES}, ${NCPU} cpu) start: $(date) ==="
"$PY" -u scripts/desi_hcd_3rung_bias.py \
    --rung "$RUNG" --mock "$MOCK" --fold "$FOLD" --n-mocks "$N_MOCKS" --seed "$SEED" \
    --n-warmup "$N_WARMUP" --n-samples "$N_SAMPLES" \
    --out-dir "$OUTDIR" $SMOKE_FLAG
echo "=== done: $(date) ==="
