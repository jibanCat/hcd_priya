#!/bin/bash
# Cross-leg r6x paired prior-sensitivity campaign (PI decisions #7 execution annex,
# 2026-07-24; design 2, fresh seed 20260724, n=8 pairs/leg). 16 fits per leg: array ids
# 0-7 = DEPLOYED arm (mocks 0-7), 8-15 = DISPPRIOR arm (mocks 0-7). Both arms share the
# deployed-selfdraw truth + noise keys (run_legb truth_fn over the DEPLOYED ctx), so each
# pair fits IDENTICAL data under the two priors. Tilt arms DROPPED (no freeze-safe knob;
# scripts/crossleg_r6_common.py R6X_TILT_VERDICT). Launch from freeze-2026-07-23-gate-b /
# branch cert-campaign-2026-07.
#
# Per-leg submission (LEG env selects the leg; walltime per measured anchors -- eBOSS
# ~4.2 CPU-h/fit, KS ~3.3, DESI ~17-18 at 4 CPU):
#   LEG=eBOSS sbatch --time=8:00:00  --array=0-15 scripts/batch_crossleg_r6.sh
#   LEG=KS    sbatch --time=8:00:00  --array=0-15 scripts/batch_crossleg_r6.sh
#   LEG=DESI  sbatch --time=30:00:00 --array=0-15 scripts/batch_crossleg_r6.sh
# SMOKE:  SMOKE=1 LEG=eBOSS sbatch --time=1:00:00 --array=0,8 scripts/batch_crossleg_r6.sh
# Readout: scripts/analyze_crossleg_r6.py --shard-dir $OUTDIR --leg $LEG (refuses
# unpaired/drifted/wrong-signature pkls; pre-registered signature PAIR per leg).
#
#SBATCH --job-name=r6x_crossleg
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=30:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/r6x_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/r6x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail
LEG=${LEG:?set LEG=eBOSS|DESI|KS (per-leg submission)}
TID=${SLURM_ARRAY_TASK_ID:-0}
SEED=${SEED:-20260724}
N_WARMUP=${N_WARMUP:-250}; N_SAMPLES=${N_SAMPLES:-300}
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru1/mfho/cert_2026-07/r6x_crossleg}
NCPU=${SLURM_CPUS_PER_TASK:-4}
SMOKE_FLAG=""; SUFFIX=""
[[ "${SMOKE:-0}" == "1" ]] && { SMOKE_FLAG="--smoke"; SUFFIX=".smoke"; }
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

# Per-leg DEPLOYED prior-state expectations (the --expect tripwire; the driver aborts on any
# drift and re-verifies against its own pre-registered pins).
case "$LEG" in
  eBOSS|DESI) EXPECT_BOOST=1.0; EXPECT_FRAC=0.287 ;;
  KS)         EXPECT_BOOST=2.5; EXPECT_FRAC=0.40 ;;
  *) echo "[r6x] unknown LEG=$LEG" >&2; exit 1 ;;
esac

if (( TID < 8 )); then ARM=deployed; MOCK=$TID; else ARM=dispprior; MOCK=$((TID - 8)); fi
LEG_LC=$(echo "$LEG" | tr '[:upper:]' '[:lower:]')
PKL=$(printf "r6x_%s_%s_shard_%03d%s.pkl" "$LEG_LC" "$ARM" "$MOCK" "$SUFFIX")

# SKIP-GUARD: a finished fit is never re-run (resubmits pick up stragglers only).
if [[ -f "$OUTDIR/$PKL" ]]; then
  echo "[r6x] task $TID -> $PKL already exists in $OUTDIR; SKIP"; exit 0
fi
echo "[r6x] task $TID -> leg=$LEG arm=$ARM mock=$MOCK outdir=$OUTDIR smoke=${SMOKE:-0}"

"$PY" scripts/run_crossleg_r6_shard.py \
    --leg "$LEG" --arm "$ARM" --mock "$MOCK" \
    --seed "$SEED" --n-warmup "$N_WARMUP" --n-samples "$N_SAMPLES" \
    --expect-lls-boost "$EXPECT_BOOST" --expect-lls-frac-sigma "$EXPECT_FRAC" \
    --out-dir "$OUTDIR" $SMOKE_FLAG
