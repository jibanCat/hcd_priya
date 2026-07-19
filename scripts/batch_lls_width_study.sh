#!/bin/bash
# cavestru1 -- LLS prior-width HEDGE-FORM cosmology-sensitivity study (PI 2026-07-18).
# CLEAN-ARM-ONLY selfdraw fits at overridden DESI-family LLS widths {0.416, 0.574} on the SAME
# mocks (seed 20260615, shards {0,8}) as the running spot-check baseline (width 0.287, job
# 53937355 -> dla_selfdraw_spotchk; NEVER written to here). Runtime prior override only; deployed
# constants untouched. 4 tasks x 1 clean fit (~15 ks/fit on 4 CPUs) => ~70 CPU-h total.
#   sbatch --array=0-3 scripts/batch_lls_width_study.sh   |   SMOKE=1 sbatch --array=0 ...
# array map: 0=(0.416,shard0) 1=(0.416,shard8) 2=(0.574,shard0) 3=(0.574,shard8)
#
#SBATCH --job-name=lls_width_study
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=10:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/lls_width_study_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/lls_width_study_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail
TID=${SLURM_ARRAY_TASK_ID:-0}
WIDTHS=(0.416 0.416 0.574 0.574)
SHARDS=(0 8 0 8)
WIDTH=${WIDTHS[$TID]}; SHARD=${SHARDS[$TID]}
N_SHARDS=${N_SHARDS:-16}; N_MOCKS=${N_MOCKS:-16}; SEED=${SEED:-20260615}
N_WARMUP=${N_WARMUP:-250}; N_SAMPLES=${N_SAMPLES:-300}
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru1/mfho/lls_width_study}
NCPU=${SLURM_CPUS_PER_TASK:-4}; SMOKE_FLAG=""; [[ "${SMOKE:-0}" == "1" ]] && SMOKE_FLAG="--smoke"
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs
TAG=$(echo "$WIDTH" | tr '.' 'p')
[[ -f "$OUTDIR/lls_width_${TAG}_shard_$(printf %03d "$SHARD").pkl" ]] && { echo "width ${WIDTH} shard ${SHARD} exists -- SKIP"; exit 0; }
echo "=== lls_width_study width=${WIDTH} shard=${SHARD}/${N_SHARDS} (seed=${SEED}) start: $(date) ==="
"$PY" -u scripts/run_lls_width_study_shard.py --width "$WIDTH" --shard "$SHARD" \
    --n-shards "$N_SHARDS" --n-mocks "$N_MOCKS" --seed "$SEED" \
    --n-warmup "$N_WARMUP" --n-samples "$N_SAMPLES" --out-dir "$OUTDIR" $SMOKE_FLAG
echo "=== done: $(date) ==="
