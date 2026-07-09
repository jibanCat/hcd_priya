#!/bin/bash
# Great Lakes ARRAY -- the 4-ARM RESOLUTION COMPARISON BRACKET (PAIRED clean-vs-injected), per leg.
# Runs all FOUR resolution treatments side-by-side on the SAME injected mocks, then the analyzer picks the
# winner by the panel M1 rule (smallest ABSOLUTE honest sigma(n_s) among gate-passers):
#   a = option-a : resolution stays IN the covariance, no float (the deployed baseline; FAILED DESI n_s -1.83)
#   b = option-b : float f_res, TIGHT prior N(0, 0.02) (ours)
#   c = option-b WIDE (cup1d-faithful / eBOSS leg-match): float f_res, prior N(0, C_PRIOR)
#   d = arm-D    : coherent cross-z covariance mode, no float (marginalize resolution in the cov)
# See notes: 2026-07-02-coherent-cov-vs-float-resolution.md + 2026-07-02-resolution-findings.md (§13).
#
# The 4 treatments write DISTINCT tagged pkls (resolution_{a,b,c,d}_{survey}_shard_NNN.pkl) into the SAME
# OUTDIR, so a single analyze reads the whole bracket:
#   python scripts/analyze_dnuis_bias.py --shard-dir $OUTDIR      # -> the 4-ARM BRACKET table + winner
#
# M2 TUPLE GUARDRAIL (per leg): the injection level, R_z frame, and prior are ONE tuple. This driver injects
# at the LEG's OWN level in the PROXY frame the code uses (eBOSS 0.044, DESI ~0.022) and sets treatment-c's
# prior to a leg-matched C_PRIOR (eBOSS 0.05). Do NOT inject the DESI level on eBOSS or mix frames.
#
# PER-LEG ORDER (weak-first): eBOSS -> KS -> DESI. KS is now LIVE (echelle R_z=3.2 km/s landed 2026-07-07,
# commit 104aeb5 / resolution_ready flag): run it with SURVEY=ks BRES=0.15 C_PRIOR=0.15 (the deployed KS
# f_res tuple). eBOSS + DESI already certified.
#
# Usage (NOT launched by default -- budget-gated: cavestru0 tight ~4000 + cavestru1 ~5000 CPU-h; SE-pilot
# to size N before the full run):
#   SMOKE one treatment first (validate the arm-D/bracket pipeline, 0 div):
#     SMOKE=1 SURVEY=eboss sbatch --account=cavestru0 --array=6 scripts/batch_res_bracket.sh   # arm-D shard 0
#   SE-pilot the 4 arms at small N (size N so mean+-2SE doesn't straddle 0.30):
#     N_MOCKS=4 N_SHARDS=4 SURVEY=eboss sbatch --account=cavestru0 --array=0-15 scripts/batch_res_bracket.sh
#   Full eBOSS bracket (4 treatments x N_SHARDS):
#     SURVEY=eboss sbatch --account=cavestru0 --array=0-31 scripts/batch_res_bracket.sh
#   Then DESI (its own level + bracket via BRES): SURVEY=desi BRES=0.022 ...
#
# OOS (Task 2C, out-of-span instrument-resolution ADVERSARIAL arm; Phase-2 will run this on cavestru0):
# set OOS_MEMBER=(bres1|bres2|bres_real|bstar) to inject the Task-2A per-z basis member instead of the
# scalar BRES -- e.g. OOS_MEMBER=bstar OOS_STRENGTH=1.0 SURVEY=desi sbatch ... . OUTDIR gets an
# _oos_${OOS_MEMBER}_s${OOS_STRENGTH} suffix (member AND strength, so the +/-1sigma 3-point bracket
# never pools) and the pkls are tagged resolution_oos_ (never collides with the scalar bracket's
# resolution_ pkls). Default (OOS_MEMBER unset) is byte-identical to today.
#SBATCH --job-name=res_bracket
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/res_bracket_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/res_bracket_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

TREAT=(a b c d)                        # the 4 bracket arms
N_TREAT=${#TREAT[@]}
SURVEY=${SURVEY:-eboss}                 # eBOSS first (weak-first order); NOT ks (guarded)
# leg's OWN injection level in the proxy frame the code uses (M2 tuple): eBOSS 0.044, DESI ~0.022
if [[ -z "${BRES:-}" ]]; then
  case "$SURVEY" in
    eboss) BRES=0.044 ;;
    desi)  BRES=0.022 ;;
    ks)    BRES=0.15  ;;   # echelle R_z=3.2 landed 2026-07-07 (104aeb5); pass C_PRIOR=0.15 for the wide arm
    *)     echo "ERROR: set BRES for SURVEY=$SURVEY" >&2; exit 1 ;;
  esac
fi
C_PRIOR=${C_PRIOR:-0.05}                # treatment-c (wide) prior width (eBOSS leg-matched)
N_SHARDS=${N_SHARDS:-8}
N_MOCKS=${N_MOCKS:-8}
N_WARMUP=${N_WARMUP:-500}
N_SAMPLES=${N_SAMPLES:-1000}
OUTBASE=${OUTBASE:-/scratch/cavestru_root/cavestru1/mfho/res_bracket}
OUTDIR="$OUTBASE/${SURVEY}_bres_$(echo "$BRES" | tr -d .)"
TID=${SLURM_ARRAY_TASK_ID:-0}
NCPU=${SLURM_CPUS_PER_TASK:-8}
SMOKE_FLAG=""; [[ "${SMOKE:-0}" == "1" ]] && SMOKE_FLAG="--smoke"

# OOS (Task 2C): OOS_MEMBER set -> inject the out-of-span basis member instead of the scalar BRES.
# Default (unset) is byte-identical to today: OOS_FLAG empty, OUTDIR unsuffixed, ARM_TAG=resolution.
OOS_FLAG=""; ARM_TAG="resolution"
if [[ -n "${OOS_MEMBER:-}" ]]; then
  OOS_STRENGTH=${OOS_STRENGTH:-1.0}
  OOS_FLAG="--b-res-oos-member $OOS_MEMBER --b-res-oos-strength $OOS_STRENGTH"
  OUTDIR="${OUTDIR}_oos_${OOS_MEMBER}_s${OOS_STRENGTH}"
  ARM_TAG="resolution_oos"
fi

TREAT_IDX=$(( TID / N_SHARDS ))
SHARD=$(( TID % N_SHARDS ))
if [ "$TREAT_IDX" -ge "$N_TREAT" ]; then
    echo "ERROR: task $TID -> treat $TREAT_IDX >= N_TREAT=$N_TREAT (array span too large; expect 0..$((N_TREAT*N_SHARDS-1)))" >&2
    exit 1
fi
T=${TREAT[$TREAT_IDX]}
C_FLAG=""; [[ "$T" == "c" ]] && C_FLAG="--c-prior-sigma $C_PRIOR"
# DIAGNOSTIC: PIN_HUB=1 pins hub (mechanism ablation for the eBOSS resolution leak) -> separate out-dir.
PIN_FLAG=""; if [[ "${PIN_HUB:-0}" == "1" ]]; then PIN_FLAG="--pin-hub"; OUTDIR="${OUTDIR}_pinhub"; fi

export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

if [[ -f "$OUTDIR/${ARM_TAG}_${T}_${SURVEY}_shard_$(printf %03d "$SHARD").pkl" ]]; then
  echo "=== treat=$T $SURVEY shard $SHARD pkl exists -- SKIP ==="; exit 0
fi
echo "=== res_bracket treat=$T ($SURVEY) b_res=$BRES ${C_FLAG} ${PIN_FLAG} ${OOS_FLAG} shard ${SHARD}/${N_SHARDS} (${NCPU} cpu) start: $(date) ==="
"$PY" -u scripts/run_dnuis_bias_shard.py \
    --arm resolution --survey "$SURVEY" --treatment "$T" $C_FLAG $PIN_FLAG $OOS_FLAG \
    --shard "$SHARD" --n-shards "$N_SHARDS" --n-mocks "$N_MOCKS" \
    --n-warmup "$N_WARMUP" --n-samples "$N_SAMPLES" --b-res "$BRES" --out-dir "$OUTDIR" $SMOKE_FLAG
echo "=== done: $(date) ==="
