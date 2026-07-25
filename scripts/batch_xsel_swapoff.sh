#!/bin/bash
# X-battery launch gate 2: X1 swap-off byte-identity certificates (batch_xsel.sh header).
# One task per shard: sbatch --array=0-2 scripts/batch_xsel_swapoff.sh
#SBATCH --job-name=xsel_swapoff
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=2:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/xsel_swapoff_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/xsel_swapoff_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail
TID=${SLURM_ARRAY_TASK_ID:-0}
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru1/mfho/cert_2026-07/xsel}
NCPU=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs
/home/mfho/.conda/envs/emu-jax/bin/python3 -u scripts/run_xsel_shard.py \
    --arm-id X1_dla100 --swap-off-check --shard "$TID" --n-shards 16 \
    --expect-lls-boost 2.5 --expect-lls-frac-sigma 0.40 --out-dir "$OUTDIR"
