#!/bin/bash
# cavestru1 SLURM — HELD-OUT-SIM SBC (leg_a=False), the HONEST n_s instrument that CONTAINS emulator
# error (the self-draw cancels it). Production-ensemble forward, deployed C_emu (cemu-variant=current),
# the CORRECTED z-resolved alpha (require_zresolved guard active), the closure-honesty fix
# (make_legb_mock builds alpha_hcd_z). mock m -> held_out_sims(fold0)[m % 8] with noise fold_in(seed,m).
# config-key skip-cache (run_cfg stamped + asserted) protects this dir from a self-draw mix-up.
#
# PILOT: --array=0-7 (one per held-out sim). Scale to 0-23 (3 noise reals) after the pilot is clean.
# Usage:  sbatch --array=0-7 scripts/batch_sbc_heldout.sh
# Analyze: read prod_sbc_heldout/mock_*.pkl — n_s pull mean (+-0.3sig) + over-dispersion (<=1.1) at low-z,
#          + tau0/dtau0 per the standing rule; alpha_HCD coverage reported separately.
#
#SBATCH --job-name=sbc_heldout
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/sbc_heldout_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/sbc_heldout_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

N_MOCKS=${N:-24}                    # mock m -> held_out sim[m%8]; pilot array runs 0-7
N_SHARDS=${N_MOCKS}                 # one mock per shard
# OUTDIR env-overridable so the subDLA arms go to their own dir, e.g.
#   SBC_SUBDLA_AMP_SIGMA=0.20 OUTDIR=/scratch/.../prod_sbc_heldout_amp020 sbatch --array=0-7 scripts/batch_sbc_heldout.sh
# (SBC_SUBDLA_AMP_SIGMA is read by run_prod_sbc_shard, verified-propagate; the config-key skip-cache also guards.)
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru1/mfho/prod_sbc_heldout}
TID=${SLURM_ARRAY_TASK_ID:-0}
NCPU=${SLURM_CPUS_PER_TASK:-4}

export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

if [[ -f "$OUTDIR/mock_$(printf %04d "$TID").pkl" ]]; then
  echo "=== mock ${TID} pkl exists — SKIP ==="; exit 0
fi

echo "=== sbc_heldout (leg_a=False, deployed forward) mock ${TID} (N_MOCKS=${N_MOCKS}, ${NCPU} cpu) start: $(date) ==="
"$PY" -u scripts/run_prod_sbc_shard.py \
    --shard "$TID" --n-shards "$N_SHARDS" --n-mocks "$N_MOCKS" \
    --held-out --out-dir "$OUTDIR" --no-shard-pkl
echo "=== done: $(date) ==="
