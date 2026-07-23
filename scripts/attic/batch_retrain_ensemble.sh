#!/bin/bash
# === ATTIC (2026-07-22 freeze triage execution; dispositions PI-approved 2026-07-20, hcd_priya_notes docs/superpowers/2026-07-20-freeze-script-triage.md). NOT IN THE FROZEN FORWARD. ===
# emulator retrain experiment (provenance: 'we tried retrain waves'); not the deployed emulator.
# Ensemble headroom probe: train baseline recipe at seeds 1 & 2 on folds 0,1,2 so we can
# measure whether AVERAGING members (the one lever that helps a finite-sim-limited emulator)
# reduces the HELD-OUT per-class error vs a single member. seed0 already exists from the pilot.
# Task list: "seed fold" pairs.
#SBATCH --job-name=retrain_ens
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6g
#SBATCH --time=1:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/retrain_ens_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/retrain_ens_%A_%a.err
set -e
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK:-4}"
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
TASKS=( "0 0" "0 1" "0 2" "1 0" "1 1" "1 2" "2 0" "2 1" "2 2" )
ITEM="${TASKS[${SLURM_ARRAY_TASK_ID:-0}]}"
S=$(echo "$ITEM" | awk '{print $1}'); F=$(echo "$ITEM" | awk '{print $2}')
echo "[ens] baseline seed=$S fold=$F  $(date)"
# --save-model so we can ensemble-average the members in eval
$PY scripts/retrain_headroom_pilot.py --fold "$F" --variant baseline --tag ens --seed "$S" --save-model
echo "[ens] seed=$S fold=$F DONE  $(date)"
