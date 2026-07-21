#!/bin/bash
# cavestru1 SLURM — KS z_lo=2.8 DIAGNOSTIC blind fit (one-off wrapper).
# The PI wants KS z2.4 as the BASELINE (run via batch_real_fit.sh, survey=ks) and z2.8 as a
# DIAGNOSTIC comparison (the KS-author published conservative cut; see load_ks_leg docstring).
# Same blind.lock, same baseline likelihood — ONLY the KS leg's low-z cut moves (z_lo 2.4 -> 2.8,
# drops the z=2.4/2.6 bins). run_real_fit.py routes a non-baseline z_lo to a DISTINCT root
# (real_ks_z28) so it never clobbers the z2.4 baseline. A_p/n_s remain BLINDED; sampler health
# (R-hat/divergences/ESS) is visible. Output committable (results/real_fit/).
#
# Usage:
#   sbatch scripts/batch_real_fit_ks_z28.sh
#
#SBATCH --job-name=real_ks_z28
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/real_ks_z28_%j.out
#SBATCH --error=/home/mfho/hcd_priya/logs/real_ks_z28_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

NCPU=${SLURM_CPUS_PER_TASK:-8}
N_CHAINS=${N_CHAINS:-4}
N_WARMUP=${N_WARMUP:-250}
N_SAMPLES=${N_SAMPLES:-600}
MAX_TREE_DEPTH=${MAX_TREE_DEPTH:-10}
SEED=${SEED:-20260614}
KS_ZLO=${KS_ZLO:-2.8}
BLIND_LOCK=${BLIND_LOCK:-/home/mfho/hcd_priya/blind.lock}

export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONHASHSEED=0            # P0: reproducibility belt-and-braces (seed fold is crc32; this pins any residual hash-order effect)
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p /home/mfho/hcd_priya/logs

echo "=== KS z_lo=${KS_ZLO} DIAGNOSTIC blind fit  ${NCPU} cpu  start: $(date) ==="
echo "    chains=${N_CHAINS} warmup=${N_WARMUP} samples=${N_SAMPLES} mtd=${MAX_TREE_DEPTH} blind_lock=${BLIND_LOCK}"
"$PY" -u scripts/run_real_fit.py \
    --survey ks --ks-zlo "$KS_ZLO" \
    --n-chains "$N_CHAINS" --n-warmup "$N_WARMUP" --n-samples "$N_SAMPLES" \
    --max-tree-depth "$MAX_TREE_DEPTH" --seed "$SEED" \
    --blind-lock "$BLIND_LOCK"
echo "=== done KS z_lo=${KS_ZLO}: $(date) ==="
