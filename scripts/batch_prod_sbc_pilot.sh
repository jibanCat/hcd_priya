#!/bin/bash
# Great Lakes ARRAY — PRODUCTION-ensemble Leg-A SBC PILOT (N=24 mocks).
# The §6 inference-calibration gate at pilot scale: de-risks the path, MEASURES the true
# ensemble-forward overhead, and confirms L_eff>=99 at n_samples=600 BEFORE the full N=128 run.
# Each array task runs a SHARD of mocks (m % N_SHARDS == task_id); per-mock RNG is fold_in
# so shards are disjoint + reproducible. Merge with scripts/merge_prod_sbc_shards.py.
#
# Usage (8 tasks x 3 mocks = 24):
#   OUTDIR=<scratch>/prod_sbc_pilot sbatch --array=0-7 scripts/batch_prod_sbc_pilot.sh
#   # then: python scripts/merge_prod_sbc_shards.py --shard-dir $OUTDIR
#SBATCH --job-name=prod_sbc_pilot
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/prod_sbc_pilot_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/prod_sbc_pilot_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

N_SHARDS=${N_SHARDS:-8}
N_MOCKS=${N_MOCKS:-24}
# CONFIRM the cavestru1 scratch path before launch.
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru1/mfho/prod_sbc_pilot}
TID=${SLURM_ARRAY_TASK_ID:-0}
NCPU=${SLURM_CPUS_PER_TASK:-8}

# cap threads to the allocation (Great Lakes bills max(cores, mem/7)*wall).
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

echo "=== prod-sbc pilot shard ${TID}/${N_SHARDS} (N_MOCKS=${N_MOCKS}, ${NCPU} cpu) start: $(date) ==="
"$PY" -u scripts/run_prod_sbc_shard.py \
    --shard "$TID" --n-shards "$N_SHARDS" --n-mocks "$N_MOCKS" \
    --out-dir "$OUTDIR"
echo "=== done: $(date) ==="
