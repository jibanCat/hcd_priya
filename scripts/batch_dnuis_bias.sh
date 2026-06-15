#!/bin/bash
# Great Lakes ARRAY — DATA-NUISANCE injection-recovery BIAS gate (PAIRED clean-vs-injected).
# The data-side bias gate on the PRODUCTION forward (N=5 ensemble, per-survey LLS pin). Each shard
# runs run_dnuis_bias_shard.py PAIRED: TWICE per mock at the same (seed, mock index) — CLEAN
# (inject_spec=None) and INJECTED — so the truth θ + cosmic noise ε cancel in the per-mock Δbias.
#
# 8 CELLS (arm × survey), each sharded N_SHARDS ways. The array index encodes (cell, shard):
#   task = cell_idx * N_SHARDS + shard   →  N_CELLS(8) * N_SHARDS tasks total.
# With N_SHARDS=4, N_MOCKS=8 (paired → 16 fits/cell): 8 cells × 4 shards = 32 array tasks; each
# shard does 2 mocks × 2 (clean+inj) = 4 fits ≈ 4×13 ≈ 52 CPU-h. Total ≈ 8×16 = 128 fits.
# metal_misspec is DESI/eBOSS only (KS is a no-op no-test); resolution is DESI/KS; lls_excess is
# DESI/KS; metal_matched is DESI/eBOSS.
#
# Usage (the array span must be 0 .. N_CELLS*N_SHARDS-1):
#   N_SHARDS=4 OUTDIR=<scratch>/dnuis_bias sbatch --array=0-31 scripts/batch_dnuis_bias.sh
#   # then, per cell:
#   python scripts/analyze_dnuis_bias.py --shard-dir $OUTDIR --arm <arm> --survey <survey>
#   # or pooled across all cells:
#   python scripts/analyze_dnuis_bias.py --shard-dir $OUTDIR
#SBATCH --job-name=dnuis_bias
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/dnuis_bias_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/dnuis_bias_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

# The 8 cells (arm:survey). metal_misspec KS is intentionally ABSENT (no-op no-test).
CELLS=(
  "metal_misspec:desi"
  "metal_misspec:eboss"
  "resolution:desi"
  "resolution:ks"
  "lls_excess:desi"
  "lls_excess:ks"
  "metal_matched:desi"
  "metal_matched:eboss"
)
N_CELLS=${#CELLS[@]}

N_SHARDS=${N_SHARDS:-4}
N_MOCKS=${N_MOCKS:-8}                 # paired → 16 fits/cell at this default
# CONFIRM the cavestru1 scratch path before launch.
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru1/mfho/dnuis_bias}
TID=${SLURM_ARRAY_TASK_ID:-0}
NCPU=${SLURM_CPUS_PER_TASK:-8}

# decode the array index into (cell, shard).
CELL_IDX=$(( TID / N_SHARDS ))
SHARD=$(( TID % N_SHARDS ))
if [ "$CELL_IDX" -ge "$N_CELLS" ]; then
    echo "ERROR: task $TID -> cell $CELL_IDX >= N_CELLS=$N_CELLS (array span too large; expect 0..$((N_CELLS*N_SHARDS-1)))" >&2
    exit 1
fi
ARM=${CELLS[$CELL_IDX]%%:*}
SURVEY=${CELLS[$CELL_IDX]##*:}

# cap threads to the allocation (Great Lakes bills max(cores, mem/7)*wall).
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

echo "=== dnuis-bias PAIRED ${ARM}/${SURVEY} shard ${SHARD}/${N_SHARDS} (cell ${CELL_IDX}/${N_CELLS}, N_MOCKS=${N_MOCKS}, ${NCPU} cpu) start: $(date) ==="
"$PY" -u scripts/run_dnuis_bias_shard.py \
    --arm "$ARM" --survey "$SURVEY" \
    --shard "$SHARD" --n-shards "$N_SHARDS" --n-mocks "$N_MOCKS" \
    --out-dir "$OUTDIR" ${EXTRA_ARGS:-}
echo "=== done: $(date) ==="
