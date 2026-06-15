#!/bin/bash
# cavestru1 SLURM — REAL-DATA BLIND production fit for the three surveys (eBOSS, KS, DESI).
# THIS IS THE ACTUAL COSMOLOGY MEASUREMENT. The A_p / n_s VALUES in the exported chains are
# BLINDED (hidden additive offset from blind.lock); sampler health (R-hat/divergences/ESS) is
# visible. DESI outputs route to results_local/ (gitignored, PRIVATE); eBOSS/KS to results/real_fit
# (committable). Run as a 3-task array (one survey per task).
#
# PRE-FLIGHT (do ONCE, BEFORE submit):
#   1. Freeze the blind seed:
#        PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3 \
#          -c "from hcd_analysis.emulator import blinding as B; \
#              print(B.write_blind_lock('/home/mfho/hcd_priya/blind.lock','hcd_priya_real_fit_v1'))"
#   2. Commit blind.lock + analysis.lock (the SEED + the frozen analysis choices) so the blind is
#      pinned to a git commit BEFORE any fit is run.
#   3. CONFIRM the cavestru1 scratch path exists and you have quota.
#
# Usage (3 tasks = eboss, ks, desi):
#   sbatch --array=0-2 scripts/batch_real_fit.sh
# or one survey:
#   SURVEY=desi sbatch --array=0 scripts/batch_real_fit.sh
#
#SBATCH --job-name=real_fit
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/real_fit_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/real_fit_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

# survey selection: array index → survey, or an explicit SURVEY=... override.
SURVEYS=(eboss ks desi)
TID=${SLURM_ARRAY_TASK_ID:-0}
SURVEY=${SURVEY:-${SURVEYS[$TID]}}

NCPU=${SLURM_CPUS_PER_TASK:-8}
N_CHAINS=${N_CHAINS:-4}
N_WARMUP=${N_WARMUP:-250}
N_SAMPLES=${N_SAMPLES:-600}
MAX_TREE_DEPTH=${MAX_TREE_DEPTH:-10}
SEED=${SEED:-20260614}
BLIND_LOCK=${BLIND_LOCK:-/home/mfho/hcd_priya/blind.lock}
# scratch is used only for SLURM scratch staging; the chains write to the repo results dirs (the
# privacy routing lives in run_real_fit.py). Kept here for parity with batch_prod_sbc_pilot.sh.
SCRATCH=${SCRATCH:-/scratch/cavestru_root/cavestru1/mfho/real_fit}

# thread-pinning (Great Lakes bills max(cores, mem/7)*wall; cap threads to the allocation).
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$SCRATCH" /home/mfho/hcd_priya/logs

echo "=== REAL-DATA BLIND fit: survey=${SURVEY} (task ${TID}) ${NCPU} cpu  start: $(date) ==="
echo "    chains=${N_CHAINS} warmup=${N_WARMUP} samples=${N_SAMPLES} mtd=${MAX_TREE_DEPTH} blind_lock=${BLIND_LOCK}"
"$PY" -u scripts/run_real_fit.py \
    --survey "$SURVEY" \
    --n-chains "$N_CHAINS" --n-warmup "$N_WARMUP" --n-samples "$N_SAMPLES" \
    --max-tree-depth "$MAX_TREE_DEPTH" --seed "$SEED" \
    --blind-lock "$BLIND_LOCK"
echo "=== done survey=${SURVEY}: $(date) ==="
