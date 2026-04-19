#!/bin/bash
# Great Lakes SLURM batch script for hcd_analysis – LF campaign (60 sims).
#
# Usage:
#   sbatch --array=0-59 scripts/batch_greatlakes.sh run-one-array
#   sbatch scripts/batch_greatlakes.sh run-all
#   sbatch scripts/batch_greatlakes.sh run-sim --sim ns0.803...
#
# Outputs go to /scratch/cavestru_root/cavestru0/mfho/hcd_outputs/

#SBATCH --job-name=hcd_pipeline
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=21
#SBATCH --mem-per-cpu=8g
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/hcd_%j.out
#SBATCH --error=/home/mfho/hcd_priya/logs/hcd_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
HCD_ROOT="/home/mfho/hcd_priya"
PYTHON="/sw/pkgs/arc/mamba/py3.11/bin/python3"
HCD_CONFIG="${HCD_CONFIG:-${HCD_ROOT}/config/default.yaml}"
OUTPUT_ROOT="/scratch/cavestru_root/cavestru0/mfho/hcd_outputs"

mkdir -p "${HCD_ROOT}/logs"
cd "${HCD_ROOT}"

echo "=== hcd_analysis pipeline (LF) ==="
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "Root:     ${HCD_ROOT}"
echo "Output:   ${OUTPUT_ROOT}"
echo "Mode:     ${1:-run-one-array}"
echo "CPUs:     ${SLURM_CPUS_PER_TASK:-1}"
echo "Job:      ${SLURM_JOB_ID:-local}  Array: ${SLURM_ARRAY_TASK_ID:-none}"

# ---------------------------------------------------------------------------
# Subcommand dispatch
# ---------------------------------------------------------------------------

MODE="${1:-run-one-array}"
shift || true

case "$MODE" in

  run-one-array)
    # SLURM array mode: each task handles one LF simulation (all its redshifts).
    # Submit: sbatch --array=0-59 scripts/batch_greatlakes.sh run-one-array
    #
    # n_workers=1      : only one sim per task, no need for sim-level parallelism
    # n_workers_skewer : use ALL allocated CPUs for intra-snap skewer parallelism
    #   (reduces catalog build from ~4.5h → ~13min per snap at 21 CPUs)
    SIM_LIST="${HCD_ROOT}/sim_list.txt"
    SIM_NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SIM_LIST")
    if [ -z "$SIM_NAME" ]; then
      echo "ERROR: No sim at array index ${SLURM_ARRAY_TASK_ID}"
      exit 1
    fi
    echo "Array task ${SLURM_ARRAY_TASK_ID} → ${SIM_NAME}"
    "$PYTHON" -m cli.run run-sim \
      --config "$HCD_CONFIG" \
      --output-root "$OUTPUT_ROOT" \
      --set "n_workers=1" \
      --set "n_workers_skewer=${SLURM_CPUS_PER_TASK:-4}" \
      --sim "$SIM_NAME" \
      --verbose
    ;;

  run-all)
    "$PYTHON" -m cli.run run-all \
      --config "$HCD_CONFIG" \
      --output-root "$OUTPUT_ROOT" \
      --n-workers "${SLURM_CPUS_PER_TASK:-4}" \
      --verbose \
      "$@"
    ;;

  run-sim)
    "$PYTHON" -m cli.run run-sim \
      --config "$HCD_CONFIG" \
      --output-root "$OUTPUT_ROOT" \
      --verbose \
      "$@"
    ;;

  run-one)
    "$PYTHON" -m cli.run run-one \
      --config "$HCD_CONFIG" \
      --output-root "$OUTPUT_ROOT" \
      --verbose \
      "$@"
    ;;

  convergence)
    "$PYTHON" -m cli.run convergence \
      --config "$HCD_CONFIG" \
      --output-root "$OUTPUT_ROOT" \
      --verbose \
      "$@"
    ;;

  benchmark)
    "$PYTHON" -m cli.run benchmark \
      --config "$HCD_CONFIG" \
      --output-root "$OUTPUT_ROOT" \
      --n-sims 2 --n-snaps 3 \
      --out "${OUTPUT_ROOT}/benchmark.json" \
      --verbose
    ;;

  report)
    "$PYTHON" -m cli.run report \
      --config "$HCD_CONFIG" \
      --output-root "$OUTPUT_ROOT" \
      --figures-dir "${HCD_ROOT}/figures" \
      --docs-dir "${HCD_ROOT}/docs" \
      --verbose
    ;;

  *)
    echo "ERROR: Unknown mode: $MODE"
    echo "Available: run-one-array, run-all, run-sim, run-one, convergence, benchmark, report"
    exit 1
    ;;

esac

echo "=== Done: $(date) ==="
