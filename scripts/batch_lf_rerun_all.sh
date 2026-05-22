#!/bin/bash
# Great Lakes SLURM ARRAY — re-run ALL 60 LF sims' Phase-1 from emu_full with the
# FIXED (1+z)*h dX code, resume:false. One array task per sim (all its z).
#
# Goal: produce a NATIVE-correct cddf.npz for every LF sim, so the dual
# cddf / cddf_corrected naming can be retired (single source of truth). The
# fix is in-code (commit c210990), so this writes correct cddf.npz directly —
# no patch_cddf_dx.py and NO cddf_corrected copy step. After this lands, flip
# discover_sim_snap_pairs + read_cddf to read cddf.npz (Task 12).
#
# Catalogs/P1D come out identical to the existing ones (same raw tau, same NHI
# code); only cddf.npz changes from buggy -> correct.
#
# Submit:  sbatch --array=0-59 scripts/batch_lf_rerun_all.sh

#SBATCH --job-name=hcd_lf_rerun
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=21
#SBATCH --mem-per-cpu=8g
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/hcd_lf_rerun_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/hcd_lf_rerun_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu

set -euo pipefail
HCD_ROOT="/home/mfho/hcd_priya"
PYTHON="/sw/pkgs/arc/mamba/py3.11/bin/python3"
HCD_CONFIG="${HCD_ROOT}/config/default.yaml"
OUTPUT_ROOT="/scratch/cavestru_root/cavestru0/mfho/hcd_outputs"
SIM_LIST="${HCD_ROOT}/sim_list.txt"
mkdir -p "${HCD_ROOT}/logs"; cd "${HCD_ROOT}"

SIM_NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SIM_LIST")
if [ -z "$SIM_NAME" ]; then echo "ERROR: no sim at array index ${SLURM_ARRAY_TASK_ID}"; exit 1; fi
echo "=== LF re-run: array ${SLURM_ARRAY_TASK_ID} -> ${SIM_NAME}  $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-21}"

# n_workers=1 (one sim/task); n_workers_skewer = all CPUs (intra-snap); resume:false
"$PYTHON" -m cli.run run-sim \
  --config "$HCD_CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --set "n_workers=1" \
  --set "n_workers_skewer=${SLURM_CPUS_PER_TASK:-21}" \
  --set "resume=false" \
  --sim "$SIM_NAME" \
  --verbose

echo "=== done ${SIM_NAME}: $(date) ==="
