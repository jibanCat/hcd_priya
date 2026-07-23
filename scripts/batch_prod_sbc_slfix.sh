#!/bin/bash
# cavestru1 SLURM — PRODUCTION-ensemble Leg-A SBC RE-EVAL on the CORRECTED-z-slope forward
# (slope fix 3603522, 2026-06-16). Re-tests the n_s + α_lls rank-uniformity FAIL that was a likely
# prior-truth-mismatch artifact (the old forward z-slope center sat at 0.95 vs the truth's ~2.4; the
# fix re-centers it on HCD_INCIDENCE_SLOPE). run_prod_sbc_shard.py is the path-B leg-grid runner with
# the PRODUCTION baseline (2-param τ₀, lit-pinned HCD, emucoh + cross-class C_emu, MF P1D+dN/dX
# off-diag cov, eBOSS metals; N=5 ensemble) — it inherits the corrected slope automatically.
#
# 24h-WALL SAFETY: ONE mock per shard (m % N_SHARDS == shard, N_SHARDS == N_MOCKS) so each task
# carries exactly one NUTS fit — no shard can stall behind a slow mock and hit the wall with nothing
# written. The per-mock pkl IS the unit of progress; merge whatever finished.
#
# OUTPUT to a NEW scratch dir (prod_sbc_slfix) — never the old-slope pilot dir. Restartable:
# a shard whose pkl exists is implicitly re-done (cheap to just let it overwrite; or skip below).
#
# Usage (N mocks, one per shard):
#   N=32 OUTDIR=/scratch/cavestru_root/cavestru1/mfho/prod_sbc_slfix \
#     sbatch --array=0-31 scripts/batch_prod_sbc_slfix.sh
#   # then: scripts/merge_prod_sbc_shards.py --shard-dir $OUTDIR
#
#SBATCH --job-name=prod_sbc_slfix
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/prod_sbc_slfix_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/prod_sbc_slfix_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

N_MOCKS=${N:-${N_MOCKS:-32}}
N_SHARDS=${N_MOCKS}                 # ONE mock per shard (24h-wall safety)
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru1/mfho/prod_sbc_slfix}
TID=${SLURM_ARRAY_TASK_ID:-0}
NCPU=${SLURM_CPUS_PER_TASK:-4}

export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONHASHSEED=0            # P0: reproducibility belt-and-braces (seed fold is crc32; this pins any residual hash-order effect)
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

if [[ -f "$OUTDIR/shard_$(printf %03d "$TID").pkl" ]]; then
  echo "=== shard ${TID} pkl exists — SKIP ==="
  exit 0
fi

echo "=== prod-sbc SLFIX shard ${TID}/${N_SHARDS} (N_MOCKS=${N_MOCKS}, ${NCPU} cpu, corrected slope) start: $(date) ==="
"$PY" -u scripts/run_prod_sbc_shard.py \
    --shard "$TID" --n-shards "$N_SHARDS" --n-mocks "$N_MOCKS" \
    --out-dir "$OUTDIR" ${EXTRA_ARGS:-}
echo "=== done: $(date) ==="
