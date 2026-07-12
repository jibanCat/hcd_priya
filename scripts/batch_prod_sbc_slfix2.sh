#!/bin/bash
# cavestru1 SLURM — PRODUCTION-ensemble Leg-A SBC RE-EVAL on the CORRECTED-z-slope forward,
# with the PER-MOCK CHECKPOINT (OOM / 24h-wall fix, 2026-06-17).
#
# WHY THIS SUPERSEDES batch_prod_sbc_slfix.sh: the previous re-eval (job 51884006) ran every
# mock of a shard inside ONE run_legb call that accumulates all draws in memory and writes only
# at the very end → it hit OUT_OF_MEMORY (MaxRSS≈16.77 GB on --mem=16g, after 3–6.8 h) and lost
# the WHOLE shard's work. run_prod_sbc_shard.py now runs ONE mock at a time, writing
# mock_{m:04d}.pkl atomically after each and SKIPPING any mock whose pkl already exists — so an
# OOM/wall now loses ≤1 mock and a resubmit resumes. --mem raised to 24g (≈7 GB over the observed
# per-mock peak). merge_prod_sbc_shards.py globs mock_*.pkl (index from the FILENAME).
#
# FORWARD: the CORRECTED HCD z-slope (re-centered on HCD_INCIDENCE_SLOPE ~2.4, commits
# 3603522/6358742/bad8f15, forward-exponent guard ACTIVE) + 1D power-law incidence (NOT 2D-tilt)
# + the production MF correction, N=5 ensemble, emucoh + cross-class C_emu, eBOSS metals.
# HISTORICAL NOTE (stale when written, corrected 2026-07-12): NORC has been WIRED since 2026-07-04
# (Gate-A) and its single authority is now closure_legb.PROD_RES_CORR_ON (NORC-refactor panel). The
# explicit --no-res-corr-on below is REDUNDANT with the authority-derived default (same value today);
# it is kept only as a historical pin of this campaign's forward. New drivers should OMIT the flag
# and inherit the authority. C_mock ≡ C_like regardless (same forward for mock truth + likelihood).
#
# 24h-WALL SAFETY: ONE mock per shard (m % N_SHARDS == shard, N_SHARDS == N_MOCKS) so each task
# carries exactly one NUTS fit (the OOM job's per-mock wall was 3–6.8 h ≪ 24 h). The per-mock pkl
# IS the unit of progress; merge whatever finished. RESUBMIT-safe: a shard whose per-mock pkl
# exists is SKIPPED inside run_prod_sbc_shard (and the wrapper skips below as a fast-path).
#
# OUTPUT to a NEW scratch dir (prod_sbc_slfix2) — never the old-slope / old-OOM dirs.
#
# Usage (N mocks, one per shard):
#   N=48 OUTDIR=/scratch/cavestru_root/cavestru1/mfho/prod_sbc_slfix2 \
#     sbatch --array=0-47 scripts/batch_prod_sbc_slfix2.sh
#   # then: scripts/merge_prod_sbc_shards.py --shard-dir $OUTDIR
#
#SBATCH --job-name=prod_sbc_slfix2
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/prod_sbc_slfix2_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/prod_sbc_slfix2_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

N_MOCKS=${N:-${N_MOCKS:-48}}
N_SHARDS=${N_MOCKS}                 # ONE mock per shard (24h-wall safety)
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru1/mfho/prod_sbc_slfix2}
TID=${SLURM_ARRAY_TASK_ID:-0}
NCPU=${SLURM_CPUS_PER_TASK:-4}

# Single-thread BLAS (avoid the per-thread BLAS workspaces that inflate RSS); let XLA's eigen
# backend use the allocated cores for the leapfrog/cov linear algebra.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

# This shard's single mock is m == TID (since N_SHARDS == N_MOCKS). Fast-path skip if its
# per-mock pkl already exists (run_prod_sbc_shard also skips internally — this saves the import).
MPKL="$OUTDIR/mock_$(printf %04d "$TID").pkl"
if [[ -f "$MPKL" ]]; then
  echo "=== mock ${TID} pkl exists ($MPKL) — SKIP ==="
  exit 0
fi

echo "=== prod-sbc SLFIX2 shard ${TID}/${N_SHARDS} (N_MOCKS=${N_MOCKS}, ${NCPU} cpu, mem=24g, "
echo "    corrected slope, per-mock checkpoint) start: $(date) ==="
"$PY" -u scripts/run_prod_sbc_shard.py \
    --shard "$TID" --n-shards "$N_SHARDS" --n-mocks "$N_MOCKS" \
    --no-res-corr-on --no-shard-pkl \
    --out-dir "$OUTDIR" ${EXTRA_ARGS:-}
echo "=== done: $(date) ==="
