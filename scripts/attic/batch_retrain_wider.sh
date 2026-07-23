#!/bin/bash
# === ATTIC (2026-07-22 freeze triage execution; dispositions PI-approved 2026-07-20, hcd_priya_notes docs/superpowers/2026-07-20-freeze-script-triage.md). NOT IN THE FROZEN FORWARD. ===
# wider-architecture retrain experiment; not the deployed emulator.
# Resubmit ONLY the 3 wider_enc tasks (the capacity variant whose final-width bug is now fixed).
#SBATCH --job-name=retrain_wider
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6g
#SBATCH --time=1:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/retrain_wider_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/retrain_wider_%A_%a.err
set -e
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK:-4}"
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
F=${SLURM_ARRAY_TASK_ID:-0}
echo "[wider] variant=wider_enc fold=$F  $(date)"
$PY scripts/retrain_headroom_pilot.py --fold "$F" --variant wider_enc --tag pilot --seed 0
echo "[wider] fold=$F DONE  $(date)"
