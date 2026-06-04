#!/bin/bash
# Phase-2b emulator training — cavestru0 GPU PROFILING run.
#
# This is a PROFILING run: ONE LOSO fold, 30 epochs, to size the per-fold wall +
# memory before launching the full k-fold sweep. DO NOT launch all folds until the
# timing is known — the cavestru0 budget is tight (~4000 CPU-h equiv).
# Submit:  sbatch scripts/batch_train_emulator.sh
#SBATCH --account=cavestru0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=emu_train
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/%x_%j.out
#SBATCH --error=/home/mfho/hcd_priya/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

# emu-jax bundles its own CUDA wheels (self-contained); set LD_LIBRARY_PATH only
# if a runtime CUDA-lib load failure appears in the logs.
# export LD_LIBRARY_PATH=/home/mfho/.conda/envs/emu-jax/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/mfho/hcd_priya
PY=/home/mfho/.conda/envs/emu-jax/bin/python3

# Confirm the GPU is visible to JAX (first line of the body, as required).
"$PY" -c "import jax; print('jax.devices():', jax.devices())"

echo "=== emu_train PROFILE start: $(date) ==="
"$PY" scripts/train_emulator.py \
    --profile --fold 0 --n-basis 12 --epochs 30 --batch 512 \
    --out checkpoints/emu_fold0_profile
echo "=== emu_train PROFILE done: $(date) ==="
