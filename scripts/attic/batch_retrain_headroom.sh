#!/bin/bash
# === ATTIC (2026-07-22 freeze triage execution; dispositions PI-approved 2026-07-20, hcd_priya_notes docs/superpowers/2026-07-20-freeze-script-triage.md). NOT IN THE FROZEN FORWARD. ===
# emulator headroom experiment; not the deployed emulator.
# cavestru1 SLURM — emulator retraining HEADROOM pilot. ONE array task == one (variant, fold).
# Each task: ~5-7 min wall, ~2.2 GB RSS, single LF train_fold + held-out per-class eval.
# The task list is a flat array of "variant fold" pairs (see TASKS below); SLURM_ARRAY_TASK_ID
# indexes it. Writes checkpoints/retrain/<tag>_<variant>_fold<f>_seed0.json per task.
#
# Usage:
#   TAG=pilot sbatch --array=0-$((NTASKS-1)) scripts/batch_retrain_headroom.sh
# Then aggregate:
#   PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3 \
#     scripts/retrain_headroom_aggregate.py --tag $TAG
#
#SBATCH --job-name=retrain_headroom
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6g
#SBATCH --time=1:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/retrain_headroom_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/retrain_headroom_%A_%a.err

set -e
mkdir -p /home/mfho/hcd_priya/logs /home/mfho/hcd_priya/checkpoints/retrain
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NCPU=${SLURM_CPUS_PER_TASK:-4}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
TAG=${TAG:-pilot}
PY=/home/mfho/.conda/envs/emu-jax/bin/python3

# flat task list: "variant fold". Folds 0,1,2 for the pilot (3 LOSO held-out groups).
TASKS=(
  "baseline 0"  "baseline 1"  "baseline 2"
  "nbasis48 0"  "nbasis48 1"  "nbasis48 2"
  "nbasis12 0"  "nbasis12 1"  "nbasis12 2"
  "wider_enc 0" "wider_enc 1" "wider_enc 2"
  "deeper_enc 0" "deeper_enc 1" "deeper_enc 2"
  "longer 0"    "longer 1"    "longer 2"
  "presid_hi 0" "presid_hi 1" "presid_hi 2"
  "wcoh_hi 0"   "wcoh_hi 1"   "wcoh_hi 2"
  "no_datarange 0" "no_datarange 1" "no_datarange 2"
  "no_edge 0"   "no_edge 1"   "no_edge 2"
)
TID=${SLURM_ARRAY_TASK_ID:-0}
ITEM="${TASKS[$TID]}"
V=$(echo "$ITEM" | awk '{print $1}')
F=$(echo "$ITEM" | awk '{print $2}')
echo "[batch] task $TID -> variant=$V fold=$F tag=$TAG  $(date)"
$PY scripts/retrain_headroom_pilot.py --fold "$F" --variant "$V" --tag "$TAG" --seed 0
echo "[batch] task $TID DONE  $(date)"
