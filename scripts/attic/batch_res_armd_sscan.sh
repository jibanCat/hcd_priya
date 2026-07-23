#!/bin/bash
# === ATTIC (2026-07-22 freeze triage execution; dispositions PI-approved 2026-07-20, hcd_priya_notes docs/superpowers/2026-07-20-freeze-script-triage.md). NOT IN THE FROZEN FORWARD. ===
# res_corr arm-D amplitude scan (Gate-A era); batch consts documented (BRES=0.044 eBOSS proxy); superseded by NORC.
# Great Lakes ARRAY -- ARM-D AMPLITUDE (s) SCAN on eBOSS (PAIRED clean-vs-injected). The s=1 pilot
# (job 52774195) FAILED: n_s -0.61 / A_p -0.62, a MEAN-bias floor (eBOSS resolution<->n_s degenerate,
# cos 0.71). arm-D at s=1 adds the 1-sigma coherent mode, which only PARTIALLY down-weights a coherent
# offset injected at the 1-sigma level. The injection is ~along the arm-D mode, so a LARGER s should drive
# the bias -> 0 (fully projecting out the resolution direction) at the cost of sigma(n_s) inflation. This
# scan traces bias(s) + sigma(n_s)(s) to find the smallest s that clears 0.30 (the panel-M1 trade), or to
# show it plateaus (an out-of-span residual). See 2026-07-02-coherent-cov-vs-float-resolution.md +
# 2026-07-02-arm-d-eboss-se-pilot.md.  s=1 already have -> scan {2, 3, 5}.
#
# Each s writes a SEPARATE out-dir (treatment tag is 'd' for all s, so same-dir would collide). Analyze each:
#   for S in 2 3 5; do python scripts/analyze_dnuis_bias.py --shard-dir $OUTBASE/eboss_armd_s$S; done
#
#SBATCH --job-name=res_sscan
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/res_sscan_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/res_sscan_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

COH_AMP=(2 3 5)                        # arm-D amplitudes to scan (s=1 already done in the pilot)
N_AMP=${#COH_AMP[@]}
SURVEY=eboss
BRES=${BRES:-0.044}                    # eBOSS own level, proxy frame (M2 tuple)
N_SHARDS=${N_SHARDS:-6}
N_MOCKS=${N_MOCKS:-6}
N_WARMUP=${N_WARMUP:-500}
N_SAMPLES=${N_SAMPLES:-1000}
OUTBASE=${OUTBASE:-/scratch/cavestru_root/cavestru0/mfho/res_bracket}
TID=${SLURM_ARRAY_TASK_ID:-0}
NCPU=${SLURM_CPUS_PER_TASK:-8}
SMOKE_FLAG=""; [[ "${SMOKE:-0}" == "1" ]] && SMOKE_FLAG="--smoke"

AMP_IDX=$(( TID / N_SHARDS ))
SHARD=$(( TID % N_SHARDS ))
if [ "$AMP_IDX" -ge "$N_AMP" ]; then
    echo "ERROR: task $TID -> amp $AMP_IDX >= N_AMP=$N_AMP (array span too large; expect 0..$((N_AMP*N_SHARDS-1)))" >&2
    exit 1
fi
S=${COH_AMP[$AMP_IDX]}
OUTDIR="$OUTBASE/${SURVEY}_armd_s${S}"

export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

if [[ -f "$OUTDIR/resolution_d_${SURVEY}_shard_$(printf %03d "$SHARD").pkl" ]]; then
  echo "=== arm-D s=$S $SURVEY shard $SHARD pkl exists -- SKIP ==="; exit 0
fi
echo "=== res_sscan arm-D s=$S ($SURVEY) b_res=$BRES shard ${SHARD}/${N_SHARDS} (${NCPU} cpu) start: $(date) ==="
"$PY" -u scripts/run_dnuis_bias_shard.py \
    --arm resolution --survey "$SURVEY" --treatment d --coh-amp "$S" \
    --shard "$SHARD" --n-shards "$N_SHARDS" --n-mocks "$N_MOCKS" \
    --n-warmup "$N_WARMUP" --n-samples "$N_SAMPLES" --b-res "$BRES" --out-dir "$OUTDIR" $SMOKE_FLAG
echo "=== done: $(date) ==="
