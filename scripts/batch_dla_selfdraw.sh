#!/bin/bash
# cavestru1 -- displaced-truth PRIYA DLA-completeness closure arm (PI disposition 2026-07-17).
# Paired clean-vs-boosted (boost=1.5 => truth median at the 15% observed-incidence residual) on the
# deployed NORC forward + the REDUCED DESI covariance (DESI_DLA_COV_REDUCE). NO e_dla template.
# array 0-15 = 16 shards x 1 mock (N=16, seed matches the e_dla campaign). Paired (2 fits/task,
# ~8.5 h wall at the measured ~15 ks/fit on 4 CPUs; ~530 CPU-h total for the 32 fits).
#   sbatch --array=0-15 scripts/batch_dla_selfdraw.sh   |   SMOKE=1 sbatch --array=0 ...
#
#SBATCH --job-name=dla_selfdraw
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=18:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/dla_selfdraw_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/dla_selfdraw_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail
TID=${SLURM_ARRAY_TASK_ID:-0}
N_SHARDS=${N_SHARDS:-16}; N_MOCKS=${N_MOCKS:-16}; SEED=${SEED:-20260615}
N_WARMUP=${N_WARMUP:-250}; N_SAMPLES=${N_SAMPLES:-300}; BOOST=${BOOST:-1.5}
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru1/mfho/dla_selfdraw}
NCPU=${SLURM_CPUS_PER_TASK:-4}; SMOKE_FLAG=""; [[ "${SMOKE:-0}" == "1" ]] && SMOKE_FLAG="--smoke"
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs
[[ -f "$OUTDIR/dla_selfdraw_desi_shard_$(printf %03d "$TID").pkl" ]] && { echo "shard ${TID} exists -- SKIP"; exit 0; }
echo "=== dla_selfdraw shard ${TID}/${N_SHARDS} (N=${N_MOCKS} seed=${SEED} boost=${BOOST}) start: $(date) ==="
"$PY" -u scripts/run_dla_selfdraw_shard.py --shard "$TID" --n-shards "$N_SHARDS" --n-mocks "$N_MOCKS" \
    --seed "$SEED" --n-warmup "$N_WARMUP" --n-samples "$N_SAMPLES" --boost "$BOOST" --out-dir "$OUTDIR" $SMOKE_FLAG
echo "=== done: $(date) ==="
