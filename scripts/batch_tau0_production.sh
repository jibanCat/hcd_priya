#!/bin/bash
# Great Lakes ARRAY — production tau0 cache build (v3.3 uniform, 10 alpha).
#
# Cost (profiled job 51140681 + diag_alpha_density.py): 10 alpha is sufficient
# (<=0.35% interp error). ~1 h/pair wall, ~31-39 GB peak. Great Lakes bills
# max(cores, mem/7)*walltime, so memory floors the cost; we request 48 GB (safe
# for the max-nbins pairs) + 2 cores. Projected ~6-8k CPU-h total (LF 1072 + HR
# 103 pairs) — under the 20k cavestru/yueyingn0 ceiling.
#
# Usage:
#   LF: sbatch --array=0-71 --export=ALL,FIDELITY=lf,SHARD_SIZE=15 scripts/batch_tau0_production.sh
#   HR: sbatch --array=0-6  --export=ALL,FIDELITY=hr,SHARD_SIZE=15 scripts/batch_tau0_production.sh
#   smoke: sbatch --array=0-0 --time=04:00:00 --export=ALL,FIDELITY=lf,SHARD_SIZE=2 scripts/batch_tau0_production.sh
#SBATCH --job-name=tau0_prod
#SBATCH --account=yueyingn0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48g
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/tau0_prod_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/tau0_prod_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

export LD_LIBRARY_PATH=/sw/pkgs/arc/stacks/gcc/10.3.0/gsl/2.7/lib:/home/mfho/.conda/envs/emu-3.9/lib:${LD_LIBRARY_PATH:-}
PY=/home/mfho/.conda/envs/emu-3.9/bin/python3

FIDELITY=${FIDELITY:-lf}
SHARD_SIZE=${SHARD_SIZE:-15}
TID=${SLURM_ARRAY_TASK_ID:-0}
OFFSET=$(( TID * SHARD_SIZE ))
OUTDIR=/scratch/cavestru_root/cavestru0/mfho/tau0_shards
mkdir -p "$OUTDIR" "$(dirname /home/mfho/hcd_priya/logs/x)"
OUT="$OUTDIR/observables_tau0_${FIDELITY}.shard$(printf '%03d' "$TID").h5"

echo "=== tau0 prod ${FIDELITY} shard ${TID} (offset=${OFFSET} limit=${SHARD_SIZE}, 10 alpha) start: $(date) ==="
"$PY" scripts/build_emulator_cache_tau0.py \
    --fidelity "$FIDELITY" \
    --offset "$OFFSET" --limit "$SHARD_SIZE" \
    --alpha-refine 1 \
    --output "$OUT" --spot-check
echo "=== done: $(date) ==="
