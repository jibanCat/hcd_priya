#!/bin/bash
# cavestru1 -- de-double-counted metal_misspec:desi re-measure (the HONEST metal n_s bias).
# Monkeypatches draw_leg_a_leg_truth -> truth a_siiii=0 (process-local; on-disk unchanged), so
# metal_inject's SiIII is the only copy. Same seed (20260615) + N=8 as the original -0.96 cell.
# array 0-3 = 4 shards x 2 mocks (n_mocks=8, n_shards=4), paired (4 fits/task).
#   sbatch --array=0-3 scripts/batch_metal_dedoublecount.sh
#   SMOKE=1 sbatch --array=0 scripts/batch_metal_dedoublecount.sh
#
#SBATCH --job-name=metal_dedbl
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=18:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/metal_dedbl_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/metal_dedbl_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

TID=${SLURM_ARRAY_TASK_ID:-0}
N_SHARDS=${N_SHARDS:-4}
N_MOCKS=${N_MOCKS:-8}
SEED=${SEED:-20260615}
N_WARMUP=${N_WARMUP:-250}
N_SAMPLES=${N_SAMPLES:-300}
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru1/mfho/metal_dedoublecount}
NCPU=${SLURM_CPUS_PER_TASK:-4}
SMOKE_FLAG=""
[[ "${SMOKE:-0}" == "1" ]] && SMOKE_FLAG="--smoke"

export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

if [[ -f "$OUTDIR/metal_dedbl_desi_shard_$(printf %03d "$TID").pkl" ]]; then
  echo "=== shard ${TID} pkl exists -- SKIP ==="; exit 0
fi

echo "=== metal_dedbl shard ${TID}/${N_SHARDS} (n_mocks=${N_MOCKS} seed=${SEED} ${NCPU} cpu) start: $(date) ==="
"$PY" -u scripts/run_dnuis_dedoublecount_shard.py \
    --shard "$TID" --n-shards "$N_SHARDS" --n-mocks "$N_MOCKS" --seed "$SEED" \
    --n-warmup "$N_WARMUP" --n-samples "$N_SAMPLES" --out-dir "$OUTDIR" $SMOKE_FLAG
echo "=== done: $(date) ==="
