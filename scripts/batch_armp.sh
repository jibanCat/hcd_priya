#!/bin/bash
# Great Lakes ARRAY -- ARM-P: the DEPLOYED-PRIOR per-leg SBC certificate (PI decision #7 stage A;
# disposition row 5 "THIS IS the certificate"). ONE mock per array task (wall safety + per-mock
# checkpoint); a task whose mock_%04d.pkl exists is SKIPPED, so resubmits are free.
#
# GATE (PI amendment #5 decision 3): ARM-P launches only after the cross-leg r6x paired test is
# DEFINED + COMPLETED + REVIEWED on every leg. Satisfied 2026-07-26 (eBOSS/KS/DESI n=12 FINAL,
# review memo 2026-07-26-r6x-crossleg-review-and-armp-gate.md).
#
# Geometry: --deployed-prior --leg <LEG> => survey=<leg> deployed LLS pin, NORC from the single
# authority (res_corr_on=False + fix_alpha_res, KS cap 0.045), reduced DESI cov, per-leg metals,
# production N=5 ensemble, self-draw truths (C_mock == C_like => exact rank null). The runner
# REFUSES --leg all / --held-out / --fold / --prod-emu / --single-member and the prior-mutating
# env arms, so a wrong-geometry certificate cannot be produced silently.
#
# Spend order of record (ledger A1-A4): eBOSS N=48 (~200 CPU-h) -> DESI pilot N=4 (~72) ->
# KS N=48 (~165) -> DESI full N=48 (~865, pilot-gated).
#
# Usage:
#   LEG=eBOSS N=48 sbatch --array=0-47 scripts/batch_armp.sh
#   LEG=DESI  N=48 sbatch --array=0-3  scripts/batch_armp.sh      # pilot = first 4 mocks of the 48
#   # readout: scripts/analyze_sbc_perleg.py --shard-dir $OUTDIR   (gate json = certificate input)
#
#SBATCH --job-name=armp
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=16:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/armp_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/armp_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

LEG=${LEG:?set LEG=eBOSS|DESI|KS (ARM-P is per-leg by decision)}
N_MOCKS=${N:-${N_MOCKS:-48}}
N_SHARDS=${N_MOCKS}                 # ONE mock per shard: the per-mock pkl is the unit of progress
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru1/mfho/cert_2026-07/armp_${LEG}}
TID=${SLURM_ARRAY_TASK_ID:-0}
NCPU=${SLURM_CPUS_PER_TASK:-4}

# Single-thread BLAS (per-thread BLAS workspaces inflate RSS); XLA eigen uses the allocated cores.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

MPKL="$OUTDIR/mock_$(printf %04d "$TID").pkl"
if [[ -f "$MPKL" ]]; then
  echo "=== mock ${TID} pkl exists ($MPKL) -- SKIP ==="
  exit 0
fi

echo "=== ARM-P ${LEG} shard ${TID}/${N_SHARDS} (N_MOCKS=${N_MOCKS}, ${NCPU} cpu, mem=24g) start: $(date) ==="
"$PY" -u scripts/run_prod_sbc_shard.py \
    --shard "$TID" --n-shards "$N_SHARDS" --n-mocks "$N_MOCKS" \
    --deployed-prior --leg "$LEG" --no-shard-pkl \
    --out-dir "$OUTDIR" ${EXTRA_ARGS:-}
echo "=== done: $(date) ==="
