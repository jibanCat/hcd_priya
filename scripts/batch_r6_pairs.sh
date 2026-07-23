#!/bin/bash
# R6 matched old-vs-new KS parameterization comparison (disposition row 6; PI-approved
# 2026-07-23, truth source mapped-selfdraw). 8 pairs = 16 fits: array ids 0-7 = LEGACY arm
# (mocks 0-7), 8-15 = MAPPED arm (mocks 0-7). Both arms share truth + noise keys (run_legb
# truth_fn from the mapped ctx), so each pair fits IDENTICAL data under the two priors.
# Launch from the freeze-2026-07-23-gate-b tree (cert-campaign-2026-07 branch). ~53 CPU-h.
#   sbatch --array=0-15 scripts/batch_r6_pairs.sh    |    SMOKE=1 sbatch --array=0,8 ...
# Readout: scripts/analyze_r6_pairs.py --shard-dir $OUTDIR (refuses unpaired/drifted pkls).
# Promotion per the 2026-07-23 runbook: R6 has NO gate; it is supporting evidence, promoted
# with its readout memo only if the pair-identity + stamp refusals all held.
#
#SBATCH --job-name=r6_pairs
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=10:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/r6_pairs_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/r6_pairs_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail
TID=${SLURM_ARRAY_TASK_ID:-0}
SEED=${SEED:-20260615}
N_WARMUP=${N_WARMUP:-250}; N_SAMPLES=${N_SAMPLES:-300}
N_MOCKS=${N_MOCKS:-8}
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru1/mfho/cert_2026-07/r6_pairs}
NCPU=${SLURM_CPUS_PER_TASK:-4}; SMOKE_FLAG=""; [[ "${SMOKE:-0}" == "1" ]] && SMOKE_FLAG="--smoke"
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

if (( TID < 8 )); then ARM=legacy; MOCK=$TID; else ARM=mapped; MOCK=$((TID - 8)); fi
echo "[r6] task $TID -> arm=$ARM mock=$MOCK n_mocks=$N_MOCKS outdir=$OUTDIR smoke=${SMOKE:-0}"

# One fit per task: --shard MOCK --n-shards N_MOCKS selects exactly mock MOCK.
"$PY" scripts/run_ks_selboost_shard.py \
    --arm-id K0_clean --r6-arm "$ARM" --r6-truth-source mapped-selfdraw \
    --shard "$MOCK" --n-shards "$N_MOCKS" --n-mocks "$N_MOCKS" \
    --seed "$SEED" --n-warmup "$N_WARMUP" --n-samples "$N_SAMPLES" \
    --expect-lls-boost 2.5 --expect-lls-frac-sigma 0.40 \
    --out-dir "$OUTDIR" $SMOKE_FLAG
