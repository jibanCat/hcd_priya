#!/bin/bash
# === ATTIC (2026-07-22 freeze triage execution; dispositions PI-approved 2026-07-20, hcd_priya_notes docs/superpowers/2026-07-20-freeze-script-triage.md). NOT IN THE FROZEN FORWARD. ===
# SBC amp/width-check batch (superseded by the Wave-2 re-SBC design).
# cavestru1 SLURM — referee pre-check #1 (over-dispersion DIRECTION). Run the production-ensemble
# Leg-A SBC on mocks 0..7 with the subDLA AMPLITUDE prior tightened sigma/mu 0.40 -> 0.20
# (SBC_SUBDLA_AMP_SIGMA env -> run_prod_sbc_shard, verified-propagate + assert, no silent no-op).
# Matched to the existing 0.40 baseline: N_MOCKS=48 so fold_in(seed,m) gives the SAME mock m as the
# full 48-mock SBC (prod_sbc_corrected) — theta/tau0/noise-base shared; only the subDLA TRUTH
# re-draws from the narrower prior (the correct per-arm SBC construction, per the Bayesian referee).
# Compare post_sd(n_s) + pull_std at 0.20 (here) vs 0.40 (prod_sbc_corrected mocks 0-7):
#   pass the std gate iff bias_scatter shrinks at least as fast as post_sd (b/r <= 0.83).
#
# Usage:  sbatch --array=0-7 scripts/batch_sbc_amp020_widthcheck.sh
# Merge/analyze: read prod_sbc_amp020/mock_000{0..7}.pkl vs prod_sbc_corrected/mock_000{0..7}.pkl.
#
#SBATCH --job-name=sbc_amp020
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/sbc_amp020_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/sbc_amp020_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

N_MOCKS=48                          # MATCH the full SBC so per-mock RNG aligns (mocks 0..7 used)
N_SHARDS=48                         # one mock per shard
OUTDIR=/scratch/cavestru_root/cavestru1/mfho/prod_sbc_amp020
TID=${SLURM_ARRAY_TASK_ID:-0}
NCPU=${SLURM_CPUS_PER_TASK:-4}

export SBC_SUBDLA_AMP_SIGMA=0.20    # <-- the pre-check lever (subDLA amplitude width)
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

if [[ -f "$OUTDIR/mock_$(printf %04d "$TID").pkl" ]]; then
  echo "=== mock ${TID} pkl exists — SKIP ==="; exit 0
fi

echo "=== sbc_amp020 (subDLA amp sigma/mu=0.20) mock ${TID} (N_MOCKS=${N_MOCKS}, ${NCPU} cpu) start: $(date) ==="
"$PY" -u scripts/run_prod_sbc_shard.py \
    --shard "$TID" --n-shards "$N_SHARDS" --n-mocks "$N_MOCKS" \
    --out-dir "$OUTDIR" --no-shard-pkl
echo "=== done: $(date) ==="
