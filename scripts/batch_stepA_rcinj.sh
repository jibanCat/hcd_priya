#!/bin/bash
# cavestru0 -- NORC res_corr TRUTH-injection gate (Gate-A, panel-required before freeze), PER-LEG.
# Runs one survey's RCINJ arm (clean + injected paired self-draw mocks) via run_stepA's worker pool.
# TAG selects the leg (D=DESI, K=KS, E=eBOSS); the legs run as SEPARATE jobs (never joint).
# Conservative config: NORC forward (res_corr off / alpha_res pinned / KS 0.045) + deployed C_emu
# (emucoh DESI/KS) + a_SiIII (uniform); f_res NOT floated -> the measured bias is an UPPER BOUND.
#   TAG=D sbatch scripts/batch_stepA_rcinj.sh   # DESI  (RCINJD, 64 chains)
#   TAG=K sbatch scripts/batch_stepA_rcinj.sh   # KS    (RCINJK, 64 chains)
#   TAG=E sbatch scripts/batch_stepA_rcinj.sh   # eBOSS (RCINJE, 64 chains)
# Analyze on completion:
#   python scripts/analyze_res_corr_injection.py --ckpt-dir checkpoints/stepA_norc_rcinj
#SBATCH --job-name=rcinj
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/rcinj_%A.out
#SBATCH --error=/home/mfho/hcd_priya/logs/rcinj_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

REPO=/home/mfho/hcd_priya
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
TAG=${TAG:-D}
export STEPA_CKPT_DIR=$REPO/checkpoints/stepA_norc_rcinj
mkdir -p "$STEPA_CKPT_DIR" "$REPO/logs"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"
export PYTHONNOUSERSITE=1 PYTHONPATH=$REPO JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
NCPU=${SLURM_CPUS_PER_TASK:-16}
WORKERS=$(( NCPU > 2 ? NCPU - 2 : 1 ))     # 2 cores headroom (Cholesky leaks ~1.5 during warmup)

echo "=== RCINJ leg TAG=$TAG  workers=$WORKERS  ckptdir=$STEPA_CKPT_DIR  start: $(date) ==="
"$PY" scripts/run_stepA.py --run --only "RCINJ${TAG}" --workers "$WORKERS"
echo "=== RCINJ leg TAG=$TAG done: $(date) ==="
