#!/bin/bash
# Great Lakes ARRAY -- Model C+ per-leg metal-injection BIAS gate (PAIRED clean-vs-injected).
# Injects a METAL_ZEVO arm as the sole metal truth and fits with the Model C+ forward
# (flatlog2node: floated f-nodes + per-z k_SiIII/k_SiII decorrelation + the undamped SiIII-SiII
# cross). Per-leg: DESI (SiIII+SiII+cross) and eBOSS (SiIII-only). KS is metals_off -> no metal cell.
#
# CELLS = arm x survey. The array index encodes (cell, shard): task = cell_idx*N_SHARDS + shard.
# Production read (4-lens condition 3): warmup 500 / samples 1000 (NOT the 300 budget cut) so the
# A_p/n_s bias is read in ABSOLUTE units with stable widths/ESS. At N_SHARDS=8, N_MOCKS=8 -> 1 mock
# /task = 2 fits (clean+inj) ~ 2x36 ~ 72 CPU-h, ~9 h wall at 8 cpu (fits the 12h limit + the 7/1
# deadline if submitted today). Cell order: arm2 (realistic in-class) FIRST, then arm1 (flat,
# k=0.05 -> the floated-k x old-k CONTROL crossing), arm4 (gauss ooc), arm3 (ma2025 ooc).
#
# ON yueyingn0 (annual, use-it-or-lose-it, expires 7/1) -- front-load before the deadline:
#   # SMOKE one cell end-to-end first (validates NUTS 0-div on the corrected arm2):
#   SMOKE=1 N_SHARDS=8 OUTDIR=/scratch/cavestru_root/cavestru1/mfho/metal_zevo \
#     sbatch --account=yueyingn0 --array=0 scripts/batch_metal_zevo.sh
#   # 6-cell run (arm2+arm1+arm4 x {desi,eboss}) = cells 0-5 = array 0-47 (~3500 CPU-h):
#   N_SHARDS=8 OUTDIR=/scratch/cavestru_root/cavestru1/mfho/metal_zevo \
#     sbatch --array=0-47 scripts/batch_metal_zevo.sh
#   # essential 4-cell (arm2+arm4) = swap CELLS or run --array=0-15,32-47; full 8-cell = --array=0-63.
#   # analyze per cell:  python scripts/analyze_dnuis_bias.py --shard-dir $OUTDIR --arm <arm> --survey <survey>
#SBATCH --job-name=metal_zevo
#SBATCH --account=yueyingn0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/metal_zevo_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/metal_zevo_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

# arm x survey, in run order. arm2_increasing (generic in-prior increasing arm, the realistic
# in-class gate) FIRST; arm1 (flat in-class, k=0.05 -> the floated-k x old-k CONTROL crossing);
# arm4 (gauss out-of-class); arm3 (ma2025 out-of-class).
CELLS=(
  "arm2_increasing:desi"        # 0,1 -- realistic in-class (headline gate)
  "arm2_increasing:eboss"
  "arm1_decreasing:desi"        # 2,3 -- flat in-class, k=0.05 (floated-k x old-k CONTROL crossing)
  "arm1_decreasing:eboss"
  "arm4_decreasing_ooc:desi"    # 4,5 -- out-of-class gauss (robustness)
  "arm4_decreasing_ooc:eboss"
  "arm3_ma2025:desi"            # 6,7 -- out-of-class ma2025 (robustness)
  "arm3_ma2025:eboss"
)
N_CELLS=${#CELLS[@]}

N_SHARDS=${N_SHARDS:-8}                # 1 mock/task at N_MOCKS=8 -> ~9h wall (deadline-safe)
N_MOCKS=${N_MOCKS:-8}
SEED=${SEED:-20260615}
N_WARMUP=${N_WARMUP:-500}              # production read (4-lens condition 3), not the 250/300 cut
N_SAMPLES=${N_SAMPLES:-1000}
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru1/mfho/metal_zevo}
TID=${SLURM_ARRAY_TASK_ID:-0}
NCPU=${SLURM_CPUS_PER_TASK:-8}
SMOKE_FLAG=""
[[ "${SMOKE:-0}" == "1" ]] && SMOKE_FLAG="--smoke"

CELL_IDX=$(( TID / N_SHARDS ))
SHARD=$(( TID % N_SHARDS ))
if [ "$CELL_IDX" -ge "$N_CELLS" ]; then
    echo "ERROR: task $TID -> cell $CELL_IDX >= N_CELLS=$N_CELLS (array span too large; expect 0..$((N_CELLS*N_SHARDS-1)))" >&2
    exit 1
fi
ARM=${CELLS[$CELL_IDX]%%:*}
SURVEY=${CELLS[$CELL_IDX]##*:}

export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

if [[ -f "$OUTDIR/metal_zevo_${ARM}_${SURVEY}_shard_$(printf %03d "$SHARD").pkl" ]]; then
  echo "=== cell ${ARM}/${SURVEY} shard ${SHARD} pkl exists -- SKIP ==="; exit 0
fi

echo "=== metal_zevo PAIRED ${ARM}/${SURVEY} shard ${SHARD}/${N_SHARDS} (cell ${CELL_IDX}/${N_CELLS}, N_MOCKS=${N_MOCKS} seed=${SEED} ${NCPU} cpu) start: $(date) ==="
"$PY" -u scripts/run_metal_zevo_shard.py \
    --survey "$SURVEY" --arm "$ARM" \
    --shard "$SHARD" --n-shards "$N_SHARDS" --n-mocks "$N_MOCKS" --seed "$SEED" \
    --n-warmup "$N_WARMUP" --n-samples "$N_SAMPLES" --out-dir "$OUTDIR" $SMOKE_FLAG
echo "=== done: $(date) ==="
