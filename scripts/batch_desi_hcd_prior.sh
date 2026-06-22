#!/bin/bash
# cavestru1 SLURM -- DESI HCD-prior sensitivity A/B (NEW standalone runner; SBC-safe, edits nothing).
# Fit a DESI closure mock WITH vs WITHOUT the HCD incidence prior and measure the (A_p, n_s) shift +
# the recovered per-class dN/dX(z). PRIOR=on (deployed 0.15/0.40/0.50) or PRIOR=off (~flat 5/5/5),
# applied by a runtime monkeypatch inside scripts/desi_hcd_prior_sensitivity.py (the SBC_SUBDLA_AMP_SIGMA
# idiom). MATCHED A/B: held-out-sim mock (leg_a=False) => truth is the sim's measured power thru the
# production MF, PRIOR-INDEPENDENT, so on/off over the SAME (seed, array-index) share IDENTICAL data.
#
# Mock m -> held_out_sims(fold0)[m % 8] (production ensemble final_prod_seed0..4, DESI leg, DR1 metals).
# Separate job-name / logs / OUTDIR from the live SBC arms. One mock per array task.
# Usage (run BOTH arms, same array => matched mocks):
#   PRIOR=on  OUTDIR=/scratch/cavestru_root/cavestru1/mfho/desi_hcd_prior_on  sbatch --array=0-5 scripts/batch_desi_hcd_prior.sh
#   PRIOR=off OUTDIR=/scratch/cavestru_root/cavestru1/mfho/desi_hcd_prior_off sbatch --array=0-5 scripts/batch_desi_hcd_prior.sh
# Analyze (when pkls land): scripts/plot_desi_hcd_prior_sensitivity.py (do NOT run until done).
#
#SBATCH --job-name=desi_hcd_prior
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/desi_hcd_prior_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/desi_hcd_prior_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

PRIOR=${PRIOR:?set PRIOR=on or PRIOR=off}            # which HCD-prior arm
N_MOCKS=${N:-8}                                       # for fold_in(seed,m) reproducibility
SEED=${SEED:-20260621}
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru1/mfho/desi_hcd_prior_${PRIOR}}
TID=${SLURM_ARRAY_TASK_ID:-0}
NCPU=${SLURM_CPUS_PER_TASK:-4}
# Sample depth (env-overridable): default = the deployed SBC depth (600/250 -> ~200 stored draws).
# For PUBLICATION-grade contours bump N_SAMPLES (thin_to_ess keeps ~ESS draws, ESS scales with samples;
# 3000 -> ~1000 independent stored draws). N_WARMUP can rise too (better adaptation -> higher ESS/sample).
N_SAMPLES=${N_SAMPLES:-600}
N_WARMUP=${N_WARMUP:-250}
# FOLD (env): LOSO fold for the held-out sim (n_s-sorted; 0=low edge default, 4=mid-box demo, 7=high).
FOLD=${FOLD:-0}

export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

if [[ -f "$OUTDIR/mock_$(printf %04d "$TID").pkl" ]]; then
  echo "=== mock ${TID} pkl exists -- SKIP ==="; exit 0
fi

echo "=== desi_hcd_prior PRIOR=${PRIOR} mock ${TID} (N_MOCKS=${N_MOCKS}, seed=${SEED}, warmup=${N_WARMUP} samples=${N_SAMPLES}, ${NCPU} cpu) start: $(date) ==="
"$PY" -u scripts/desi_hcd_prior_sensitivity.py \
    --prior "$PRIOR" --mock "$TID" --n-mocks "$N_MOCKS" --seed "$SEED" \
    --n-warmup "$N_WARMUP" --n-samples "$N_SAMPLES" --fold "$FOLD" \
    --out-dir "$OUTDIR"
echo "=== done: $(date) ==="
