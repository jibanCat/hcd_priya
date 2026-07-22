#!/bin/bash
# === ATTIC (2026-07-22 freeze triage execution; dispositions PI-approved 2026-07-20, hcd_priya_notes docs/superpowers/2026-07-20-freeze-script-triage.md). NOT IN THE FROZEN FORWARD. ===
# norm-vs-tilt diagnostic batch (pilot/diag era).
# cavestru1 SLURM — referee pre-check #2: 3-leg norm-vs-tilt validation (does the data pull subDLA via
# NORMALIZATION not TILT, on the REAL production NUTS path — not the DESI-only Laplace). 3 held-out-sim
# mocks x 2 chains = 6 tasks, ONE (mock,chain) per task (24h-wall safety), per-task npz + skip-if-exists.
# array index t -> mock = t/2, chain = t%2.
#
# Usage:  sbatch --array=0-5 scripts/batch_diag_normtilt.sh
# Analyze: scripts/analyze_subdla_norm_vs_tilt.py over checkpoints/diag_normtilt/diag_m*_c*.npz
#
#SBATCH --job-name=diag_normtilt
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/diag_normtilt_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/diag_normtilt_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

TID=${SLURM_ARRAY_TASK_ID:-0}
NCPU=${SLURM_CPUS_PER_TASK:-4}
export DIAG_MOCK=$(( TID / 2 ))
export DIAG_CHAIN=$(( TID % 2 ))
export DIAG_OUTDIR=/home/mfho/hcd_priya/checkpoints/diag_normtilt
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$DIAG_OUTDIR" /home/mfho/hcd_priya/logs

echo "=== diag_normtilt task ${TID} -> mock ${DIAG_MOCK} chain ${DIAG_CHAIN} (${NCPU} cpu) start: $(date) ==="
"$PY" -u scripts/diag_subdla_norm_vs_tilt.py
echo "=== done: $(date) ==="
