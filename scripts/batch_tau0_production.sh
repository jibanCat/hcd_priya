#!/bin/bash
# Great Lakes ARRAY — production tau0 cache build (v3.3 uniform, 20 alpha).
#
# 20 alpha (--alpha-refine 2): PRIYA-exact 10 at even indices + 10 midpoints,
# enabling the held-out-alpha validation (train on PRIYA-10, predict midpoints).
# Memory-OPTIMIZED build (Tier P = sum of filtered pieces, no flux_power) ->
# peak ~the 2 tau arrays, NOT ~34 GB. Great Lakes bills max(cores, mem/7)*wall,
# so --mem floors the cost. *** SHARD_SIZE, --mem and --time below are placeholders
# pending the 1-pair RE-TIMING on the optimized build; set --mem ~= peak*1.2 and
# SHARD_SIZE so per-task wall < ~20 h, then submit. ***
#
# Usage (resource flags finalized post-re-timing; override --mem/--time/SHARD_SIZE):
#   LF: sbatch --array=0-N --mem=<M>g --time=<T> --export=ALL,FIDELITY=lf,SHARD_SIZE=<S> scripts/batch_tau0_production.sh
#   HR: sbatch --array=0-M --mem=<M>g --time=<T> --export=ALL,FIDELITY=hr,SHARD_SIZE=<S> scripts/batch_tau0_production.sh
#   (nbins-tiered: submit low/mid/high-nbins offset ranges as separate arrays with their own --mem.)
#SBATCH --job-name=tau0_prod
#SBATCH --account=yueyingn0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=44g
#SBATCH --time=24:00:00
#SBATCH --output=logs/tau0_prod_%A_%a.out
#SBATCH --error=logs/tau0_prod_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"   # this checkout: submit from its root (emulator-debug 2026-10)
cd "$REPO" || exit 2
[ -f "$REPO/hcd_analysis/paths.py" ] || { echo "submit from the repository root (got $REPO)" >&2; exit 2; }
set -euo pipefail

export LD_LIBRARY_PATH=/sw/pkgs/arc/stacks/gcc/10.3.0/gsl/2.7/lib:/home/mfho/.conda/envs/emu-3.9/lib:${LD_LIBRARY_PATH:-}
PY=/home/mfho/.conda/envs/emu-3.9/bin/python3

FIDELITY=${FIDELITY:-lf}
SHARD_SIZE=${SHARD_SIZE:-12}
TID=${SLURM_ARRAY_TASK_ID:-0}
OFFSET=$(( TID * SHARD_SIZE ))
OUTDIR=/scratch/cavestru_root/cavestru0/mfho/tau0_shards
mkdir -p "$OUTDIR" "$(dirname $REPO/logs/x)"
OUT="$OUTDIR/observables_tau0_${FIDELITY}.shard$(printf '%03d' "$TID").h5"

echo "=== tau0 prod ${FIDELITY} shard ${TID} (offset=${OFFSET} limit=${SHARD_SIZE}, 20 alpha) start: $(date) ==="
"$PY" scripts/build_emulator_cache_tau0.py \
    --fidelity "$FIDELITY" \
    --offset "$OFFSET" --limit "$SHARD_SIZE" \
    --alpha-refine 2 \
    --output "$OUT" --spot-check
echo "=== done: $(date) ==="
