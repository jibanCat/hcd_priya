#!/bin/bash
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=priya_pcfix
#SBATCH --output=scripts/consistency_checks/sbatch_logs/%x_%A_%a.out
#SBATCH --error=scripts/consistency_checks/sbatch_logs/%x_%A_%a.err
#SBATCH --array=0-3

# Resubmit the 4 (sim, snap) pairs that failed in 50565913 because the
# snap-to-z mapping was sim-44-specific. Each sim has slightly different snap
# numbering for z=3.0 and z=2.4.

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"   # this checkout: submit from its root (emulator-debug 2026-10)
cd "$REPO" || exit 2
[ -f "$REPO/hcd_analysis/paths.py" ] || { echo "submit from the repository root (got $REPO)" >&2; exit 2; }
SIM_IDX=(    0 0   29 29 )
SIM_FOLDER=(
    "ns0.842Ap1.36e-09herei3.51heref2.85alphaq2hub0.658omegamh20.14hireionz6.72bhfeedback0.0453"
    "ns0.842Ap1.36e-09herei3.51heref2.85alphaq2hub0.658omegamh20.14hireionz6.72bhfeedback0.0453"
    "ns0.901Ap1.22e-09herei3.93heref2.87alphaq1.68hub0.712omegamh20.146hireionz6.97bhfeedback0.068"
    "ns0.901Ap1.22e-09herei3.93heref2.87alphaq1.68hub0.712omegamh20.146hireionz6.97bhfeedback0.068"
)
SNAP=(      16 20    16 19 )
ZIDX=(       8 11     8 11 )

i=$SLURM_ARRAY_TASK_ID
echo "Job array task $i: sim_idx=${SIM_IDX[$i]} snap=${SNAP[$i]} z_idx=${ZIDX[$i]}"

set -euo pipefail
export LD_LIBRARY_PATH=/sw/pkgs/arc/stacks/gcc/10.3.0/gsl/2.7/lib:/home/mfho/.conda/envs/emu-3.9/lib:${LD_LIBRARY_PATH:-}
PY=/home/mfho/.conda/envs/emu-3.9/bin/python3

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

OUT_DIR=$REPO/docs/superpowers/figs/multipoint
mkdir -p "$OUT_DIR"

$PY $REPO/scripts/consistency_checks/priya_p1d_consistency_one_snap.py \
    --sim-idx ${SIM_IDX[$i]} \
    --sim-folder "${SIM_FOLDER[$i]}" \
    --snap ${SNAP[$i]} \
    --z-idx ${ZIDX[$i]} \
    --out "$OUT_DIR/sim${SIM_IDX[$i]}_snap${SNAP[$i]}.npz"
