#!/bin/bash
# Great Lakes ARRAY -- OPTION-A resolution certification (PAIRED clean-vs-injected).
# Inject the spectral-resolution distortion P*exp(2 b_res k^2 R_z^2) over a +/-1 sigma bracket
# {0.015, 0.02, 0.03} on DESI (b_res=0.02 = the realistic ~1 sigma level DERIVED from the DESI data's
# own syst_e_resolution, dP/P ~3.4%; confirmed by Karacayli 2025 "a few per cent"), and fit with the
# DEPLOYED production forward (resolution_on=False -> the resolution mode stays in the covariance =
# OPTION-A). No new nuisance. Gate: |mean Dbias_z| + 2 SE < 0.30 on A_p AND n_s. If option-a passes,
# resolution needs NO forward change; if it fails, escalate to option-b (float the 2-param f_res).
#
# 3 CELLS (b_res) x N_SHARDS. task = cell_idx*N_SHARDS + shard. Each b_res writes a SEPARATE out-dir
# (resolution_desi_shard_NNN.pkl would otherwise collide across b_res). Analyze per b_res:
#   python scripts/analyze_dnuis_bias.py --shard-dir $OUTBASE/bres_020   (etc.)
#
# Usage (DESI per-fit ~25-30 CPU-h at warmup 500/samples 1000; 3 x 8 x 2 fits ~ 1200 CPU-h full):
#   SMOKE one task first (validates the resolution-injection pipeline, 0 div):
#     SMOKE=1 sbatch --account=cavestru1 --array=8 scripts/batch_res_cert.sh   # b_res=0.02 shard 0
#   Fiducial-first (cheapest informative, ~450 CPU-h): only the b_res=0.02 cell:
#     sbatch --account=cavestru1 --array=8-15 scripts/batch_res_cert.sh
#   Full +/-1 sigma sweep (~1200 CPU-h): all 3 cells:
#     sbatch --account=cavestru1 --array=0-23 scripts/batch_res_cert.sh
#SBATCH --job-name=res_cert
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/res_cert_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/res_cert_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

BRES=(0.015 0.02 0.03)                 # cell 0 = -1 sigma, cell 1 = fiducial, cell 2 = +1 sigma
BTAG=(015 020 030)
N_CELLS=${#BRES[@]}
N_SHARDS=${N_SHARDS:-8}
N_MOCKS=${N_MOCKS:-8}                   # 1 mock/shard at N_SHARDS=8 -> paired 2 fits/task
N_WARMUP=${N_WARMUP:-500}               # production read (stable widths for the abs-units gate)
N_SAMPLES=${N_SAMPLES:-1000}
OUTBASE=${OUTBASE:-/scratch/cavestru_root/cavestru1/mfho/res_cert}
TID=${SLURM_ARRAY_TASK_ID:-0}
NCPU=${SLURM_CPUS_PER_TASK:-8}
SMOKE_FLAG=""; [[ "${SMOKE:-0}" == "1" ]] && SMOKE_FLAG="--smoke"

CELL_IDX=$(( TID / N_SHARDS ))
SHARD=$(( TID % N_SHARDS ))
if [ "$CELL_IDX" -ge "$N_CELLS" ]; then
    echo "ERROR: task $TID -> cell $CELL_IDX >= N_CELLS=$N_CELLS (array span too large; expect 0..$((N_CELLS*N_SHARDS-1)))" >&2
    exit 1
fi
B=${BRES[$CELL_IDX]}; TAG=${BTAG[$CELL_IDX]}
OUTDIR="$OUTBASE/bres_$TAG"

export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

if [[ -f "$OUTDIR/resolution_desi_shard_$(printf %03d "$SHARD").pkl" ]]; then
  echo "=== b_res=$B desi shard $SHARD pkl exists -- SKIP ==="; exit 0
fi
echo "=== res_cert OPTION-A b_res=$B desi shard ${SHARD}/${N_SHARDS} (${NCPU} cpu) start: $(date) ==="
"$PY" -u scripts/run_dnuis_bias_shard.py \
    --arm resolution --survey desi \
    --shard "$SHARD" --n-shards "$N_SHARDS" --n-mocks "$N_MOCKS" \
    --n-warmup "$N_WARMUP" --n-samples "$N_SAMPLES" --b-res "$B" --out-dir "$OUTDIR" $SMOKE_FLAG
echo "=== done: $(date) ==="
