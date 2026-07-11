#!/bin/bash
# cavestru1 -- NUTS-confirm the Stage-B DLA-completeness bias (Fisher: n_s -0.63, A_p -0.47).
# Inject +1sigma syst_e_dla_completeness (additive, DESI leg) into the mock truth; deployed forward.
# array 0-3 = 4 shards x 2 mocks (N=8, same seed as the metal cells). Paired (4 fits/task).
#   sbatch --array=0-3 scripts/batch_dla_completeness.sh   |   SMOKE=1 sbatch --array=0 ...
#
#SBATCH --job-name=dla_compl
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=18:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/dla_compl_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/dla_compl_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail
TID=${SLURM_ARRAY_TASK_ID:-0}
N_SHARDS=${N_SHARDS:-4}; N_MOCKS=${N_MOCKS:-8}; SEED=${SEED:-20260615}
N_WARMUP=${N_WARMUP:-250}; N_SAMPLES=${N_SAMPLES:-300}; STRENGTH=${STRENGTH:-1.0}   # STRENGTH=-1.0 => -1sigma arm
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru1/mfho/dla_completeness}
NCPU=${SLURM_CPUS_PER_TASK:-4}; SMOKE_FLAG=""; [[ "${SMOKE:-0}" == "1" ]] && SMOKE_FLAG="--smoke"
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs
[[ -f "$OUTDIR/dla_completeness_desi_shard_$(printf %03d "$TID").pkl" ]] && { echo "shard ${TID} exists -- SKIP"; exit 0; }
echo "=== dla_compl shard ${TID}/${N_SHARDS} (N=${N_MOCKS} seed=${SEED}) start: $(date) ==="
"$PY" -u scripts/run_dla_completeness_shard.py --shard "$TID" --n-shards "$N_SHARDS" --n-mocks "$N_MOCKS" \
    --seed "$SEED" --n-warmup "$N_WARMUP" --n-samples "$N_SAMPLES" --strength "$STRENGTH" --out-dir "$OUTDIR" $SMOKE_FLAG
echo "=== done: $(date) ==="
