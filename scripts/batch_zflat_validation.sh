#!/bin/bash
# Full-suite regression for the z-flat-alpha hardening (comparison-core safe-by-default + assert->raise
# + the 6 diag fixes). Runs the whole tests/ dir (minus slow-marked) so nothing downstream regressed.
# Submitted via SLURM because the interactive session is 8 CPU / 32 GB and cache-loading pytest gets
# SIGTERM'd inline (see feedback-interactive-node-slurm-compute).
#   sbatch scripts/batch_zflat_validation.sh
#SBATCH --job-name=zflat_val
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=03:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/zflat_val_%j.out
#SBATCH --error=/home/mfho/hcd_priya/logs/zflat_val_%j.err

NCPU=8
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3

echo "=== node $(hostname) | $(date) | full tests/ (not slow) ==="
$PY -m pytest tests/ -q -m "not slow" -p no:cacheprovider -rf
echo "PYTEST EXIT: $?"
