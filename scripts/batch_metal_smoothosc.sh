#!/bin/bash
# cavestru1 -- DECISIVE smooth-vs-osc metal injection NUTS test (the Gate B n_s mechanism arbiter).
# SBC-SAFE: run_metal_smoothosc_shard.py is a NEW file that monkeypatches metal_inject IN-PROCESS
# (on-disk closure_legb unchanged). Same seed (20260615) as the Gate B metal_misspec:desi cell.
#
# array 0-7 -> (arm, shard):  0-3 = metal_smooth shards 0-3 ; 4-7 = metal_osc shards 0-3.
# N_MOCKS=4, N_SHARDS=4 => 1 mock per shard, paired (2 fits/task). 8 tasks x 2 fits = 16 fits.
# PREDICTION: metal_smooth n_s ~ -0.96 (the full metal_misspec leak), metal_osc n_s ~ 0 (inert zigzag).
#
# Usage:  sbatch --array=0-7 scripts/batch_metal_smoothosc.sh
# Smoke:  SMOKE=1 sbatch --array=0 scripts/batch_metal_smoothosc.sh
#
#SBATCH --job-name=metal_smoothosc
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=18:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/metal_smoothosc_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/metal_smoothosc_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

TID=${SLURM_ARRAY_TASK_ID:-0}
N_SHARDS=${N_SHARDS:-4}
if (( TID < N_SHARDS )); then ARM=metal_smooth; SHARD=$TID; else ARM=metal_osc; SHARD=$((TID-N_SHARDS)); fi

N_MOCKS=${N_MOCKS:-4}
SEED=${SEED:-20260615}
N_WARMUP=${N_WARMUP:-250}
N_SAMPLES=${N_SAMPLES:-300}
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru1/mfho/metal_smoothosc}
NCPU=${SLURM_CPUS_PER_TASK:-4}
SMOKE_FLAG=""
[[ "${SMOKE:-0}" == "1" ]] && SMOKE_FLAG="--smoke"

export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

if [[ -f "$OUTDIR/${ARM}_desi_shard_$(printf %03d "$SHARD").pkl" ]]; then
  echo "=== ${ARM} shard ${SHARD} pkl exists -- SKIP ==="; exit 0
fi

echo "=== metal_smoothosc ARM=${ARM} shard ${SHARD}/${N_SHARDS} (n_mocks=${N_MOCKS} seed=${SEED} ${NCPU} cpu) start: $(date) ==="
"$PY" -u scripts/run_metal_smoothosc_shard.py \
    --arm "$ARM" --shard "$SHARD" --n-shards "$N_SHARDS" --n-mocks "$N_MOCKS" --seed "$SEED" \
    --n-warmup "$N_WARMUP" --n-samples "$N_SAMPLES" --out-dir "$OUTDIR" $SMOKE_FLAG
echo "=== done: $(date) ==="
