#!/bin/bash
# === ATTIC (2026-07-22 freeze triage execution; dispositions PI-approved 2026-07-20, hcd_priya_notes docs/superpowers/2026-07-20-freeze-script-triage.md). NOT IN THE FROZEN FORWARD. ===
# Leg-B pilot batch (pilot/diag era).
# Great Lakes ARRAY — Leg-B coverage PILOT (N=50 mocks, de-circularised folds-1-7 rho).
# Each array task runs a SHARD of mocks (m % N_SHARDS == task_id); per-mock RNG is fold_in
# so shards are disjoint + reproducible. Merge with scripts/merge_legb_shards.py.
#
# Usage (10 tasks x 5 mocks = 50):
#   sbatch --array=0-9 scripts/batch_legb_pilot.sh
#   # then: python scripts/merge_legb_shards.py --shard-dir $OUTDIR
#SBATCH --job-name=legb_pilot
#SBATCH --account=yueyingn0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/legb_pilot_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/legb_pilot_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

N_SHARDS=${N_SHARDS:-10}
N_MOCKS=${N_MOCKS:-50}
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru0/mfho/legb_pilot}
TID=${SLURM_ARRAY_TASK_ID:-0}
NCPU=${SLURM_CPUS_PER_TASK:-8}

# cap threads to the allocation (Great Lakes bills max(cores, mem/7)*wall).
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

echo "=== legb pilot shard ${TID}/${N_SHARDS} (N_MOCKS=${N_MOCKS}, ${NCPU} cpu) start: $(date) ==="
"$PY" -u scripts/run_legb_shard.py \
    --shard "$TID" --n-shards "$N_SHARDS" --n-mocks "$N_MOCKS" \
    --out-dir "$OUTDIR"
echo "=== done: $(date) ==="
