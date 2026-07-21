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
# PER-LEG SBC: LEG=DESI|KS|eBOSS restricts the likelihood to one survey leg (the deployed real
# analysis is per-leg — overlapping QSOs preclude a joint fit). LEG=all (default) = the joint fit.
LEG=${LEG:-all}
# TRUE-LOSO held-out (2026-06-21): FOLD=k uses the SINGLE per-fold net final_fold{k} (trained
# EXCLUDING fold k) on held_out_sims(fold=k) — a genuine held-out + n_s-UNCONFOUNDED cert (the
# default OUTDIR path uses the production ensemble on fold-0's all-low-n_s sims). Sweep k=0..7
# (the folds are n_s-sorted) to cover the box. FOLD empty (default) = the back-compat ensemble/
# fold-0 path UNCHANGED (--fold is passed ONLY when FOLD is non-empty). Composes with LEG.
#   FOLD=7 LEG=DESI OUTDIR=.../prod_sbc_loso_fold7_desi sbatch --array=0-6 scripts/batch_sbc_heldout.sh
FOLD=${FOLD:-}
# HELDOUT=1 (default) = held-out-sim SBC (leg_a=False, the honest forward+emulator cert). HELDOUT=0 =
# SELF-DRAW (leg_a=True, truths drawn from the PRIOR): the calibration CONTROL. A calibrated posterior
# gives ZERO pull-vs-truth tilt, proving the held-out -2.6 n_s tilt is benign posterior shrinkage (weak
# n_s signal under the k-coherent C_emu + ~10% data cov), not a forward/emulator bias. Self-draw ignores
# FOLD (truth is a prior draw, not a sim) — run with the production ensemble (FOLD empty). Use a SEPARATE
# OUTDIR: run_cfg.leg_a differs, so the skip-cache would (correctly) refuse to mix self-draw + held-out.
HELDOUT=${HELDOUT:-1}
TID=${SLURM_ARRAY_TASK_ID:-0}
NCPU=${SLURM_CPUS_PER_TASK:-4}

export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONHASHSEED=0            # P0: reproducibility belt-and-braces (seed fold is crc32; this pins any residual hash-order effect)
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

if [[ -f "$OUTDIR/mock_$(printf %04d "$TID").pkl" ]]; then
  echo "=== mock ${TID} pkl exists — SKIP ==="; exit 0
fi

echo "=== sbc ($([[ "$HELDOUT" == "1" ]] && echo 'held-out leg_a=False' || echo 'SELF-DRAW leg_a=True')) mock ${TID} leg=${LEG} fold=${FOLD:-ensemble/fold0} (N_MOCKS=${N_MOCKS}, ${NCPU} cpu) start: $(date) ==="
# --fold is passed ONLY when FOLD is non-empty (preserves the default ensemble/fold-0 path byte-for-byte).
FOLD_ARGS=()
if [[ -n "$FOLD" ]]; then
  FOLD_ARGS=(--fold "$FOLD")
fi
# PROD_EMU=1 (with FOLD set): the LOSO-tilt A/B — production ensemble on fold-k sims (only the emulator
# differs from the LOSO run). Default empty = LOSO single-net path unchanged.
if [[ -n "${PROD_EMU:-}" ]]; then
  FOLD_ARGS+=(--prod-emu)
fi
# --held-out passed ONLY when HELDOUT=1 (default). HELDOUT=0 = self-draw control (leg_a=True default).
HELDOUT_ARG=()
if [[ "$HELDOUT" == "1" ]]; then HELDOUT_ARG=(--held-out); fi
"$PY" -u scripts/run_prod_sbc_shard.py \
    --shard "$TID" --n-shards "$N_SHARDS" --n-mocks "$N_MOCKS" \
    --leg "$LEG" "${FOLD_ARGS[@]}" \
    "${HELDOUT_ARG[@]}" --out-dir "$OUTDIR" --no-shard-pkl
echo "=== done: $(date) ==="
