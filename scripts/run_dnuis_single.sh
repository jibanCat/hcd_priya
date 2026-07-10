#!/bin/bash
# Single-cell dnuis runner (Phase 0 / profiling helper — NOT the array).
# Runs ONE arm/survey paired clean-vs-injected for N mocks. Env-parameterized so the
# same script serves smoke (SMOKE=1) and the full single-mock Phase-0 profile (SMOKE=0).
# Submit, e.g.:
#   sbatch --time=1:00:00 --job-name=dnuis_smoke_mm \
#     --output=/home/mfho/hcd_priya/logs/dnuis_smoke_mm.out \
#     --export=ALL,ARM=metal_matched,SURVEY=desi,SMOKE=1,OUTDIR=$SCR/dnuis_smoke \
#     scripts/run_dnuis_single.sh
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

ARM=${ARM:?set ARM}
SURVEY=${SURVEY:?set SURVEY}
OUTDIR=${OUTDIR:?set OUTDIR}
SMOKE=${SMOKE:-0}
NMOCKS=${NMOCKS:-1}
NSHARDS=${NSHARDS:-1}
SHARD=${SHARD:-0}
NCPU=${SLURM_CPUS_PER_TASK:-8}

export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

SMOKEFLAG=""
[ "$SMOKE" = "1" ] && SMOKEFLAG="--smoke"

echo "=== dnuis-single ${ARM}/${SURVEY} smoke=${SMOKE} nmocks=${NMOCKS} (${NCPU} cpu) start: $(date) ==="
"$PY" -u scripts/run_dnuis_bias_shard.py \
    --arm "$ARM" --survey "$SURVEY" \
    --shard "$SHARD" --n-shards "$NSHARDS" --n-mocks "$NMOCKS" \
    --out-dir "$OUTDIR" $SMOKEFLAG
echo "=== done: $(date) ==="
