#!/bin/bash
# PHASE-2 FILL: re-run the DESI (+ 1 eBOSS) OOS-bstar shards that TIMED OUT in the 53163081 array (DESI
# fits ran ~2.8h each; 4-fit tasks exceeded the 14h wall under cavestru0 contention). Here 1 mock/task
# (n_shards=8 -> 2 fits/task, ~5-6h) so nothing risks the wall. Mocks are seed-deterministic
# (fold_in(seed,m)), so these fill the EXACT missing mock indices and pool cleanly with the completed
# n_shards=4 pkls already in each cell (no mock double-count):
#   DESI s-1 done {0,4}  -> fill mocks 1,2,3,5,6,7   (n_shards=8 shards 1,2,3,5,6,7)
#   DESI s+1 done {1,5}  -> fill mocks 0,2,3,4,6,7   (shards 0,2,3,4,6,7)
#   eBOSS s-1 done {0,1,3,4,5,7} -> fill mocks 2,6   (shards 2,6)
#   sbatch --array=0-13 scripts/batch_res_oos_fill.sh
#SBATCH --job-name=res_oos_fill
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36g
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/res_oos_fill_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/res_oos_fill_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -uo pipefail

NCPU=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
OUT=${OUT:-/scratch/cavestru_root/cavestru0/mfho/res_oos}
TID=${SLURM_ARRAY_TASK_ID:-0}

# "leg treatment c_prior strength n_shards shard"   (n_shards=8 -> 1 mock/task)
SPECS=(
  "desi  b 0.02 -1.0 8 1"
  "desi  b 0.02 -1.0 8 2"
  "desi  b 0.02 -1.0 8 3"
  "desi  b 0.02 -1.0 8 5"
  "desi  b 0.02 -1.0 8 6"
  "desi  b 0.02 -1.0 8 7"
  "desi  b 0.02  1.0 8 0"
  "desi  b 0.02  1.0 8 2"
  "desi  b 0.02  1.0 8 3"
  "desi  b 0.02  1.0 8 4"
  "desi  b 0.02  1.0 8 6"
  "desi  b 0.02  1.0 8 7"
  "eboss c 0.05 -1.0 8 2"
  "eboss c 0.05 -1.0 8 6"
)
read -r LEG TREAT CPRIOR STRENGTH NSHARDS SHARD <<< "${SPECS[$TID]}"
CELL="$OUT/oos_${LEG}_bstar_s${STRENGTH}"    # SAME cell dir as the array -> pools with the completed shards
mkdir -p "$CELL"

echo "=== res_oos_fill task ${TID}: leg=$LEG treat=$TREAT strength=$STRENGTH shard=$SHARD/$NSHARDS (1 mock) ==="
echo "    out=$CELL  start: $(date)"
"$PY" -u scripts/run_dnuis_bias_shard.py \
    --arm resolution --survey "$LEG" --treatment "$TREAT" --c-prior-sigma "$CPRIOR" \
    --b-res-oos-member bstar --b-res-oos-strength "$STRENGTH" \
    --shard "$SHARD" --n-shards "$NSHARDS" --n-mocks 8 \
    --n-warmup 250 --n-samples 300 --max-tree-depth 10 \
    --out-dir "$CELL"
echo "=== res_oos_fill task ${TID} done: $(date) ==="
