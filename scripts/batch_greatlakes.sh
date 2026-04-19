#!/bin/bash
# Great Lakes SLURM batch script for hcd_analysis.
#
# Usage:
#   sbatch scripts/batch_lf.sh            # all 60 LF sims
#   sbatch scripts/batch_lf.sh run-sim --sim ns0.803...
#   sbatch --array=0-59 scripts/batch_lf.sh run-one-array
#   sbatch scripts/batch_hires.sh         # 3 HiRes sims
#   sbatch scripts/batch_greatlakes.sh convergence  # after both above
#
# Environment variables:
#   HCD_CONFIG   path to YAML config (default: config/default.yaml)
#   HCD_ROOT     repo root (default: auto-detected from script location)

#SBATCH --job-name=hcd_pipeline
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=21              # one standard node (21 CPUs)
#SBATCH --mem-per-cpu=8g               # 8 GB/CPU = 168 GB total
#SBATCH --time=08:00:00                # ~4.4h measured + safety margin
#SBATCH --output=logs/hcd_%j.out
#SBATCH --error=logs/hcd_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

module load python/3.11 2>/dev/null || true
source activate hcd_env 2>/dev/null || conda activate hcd_env 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HCD_ROOT="${HCD_ROOT:-$(dirname "$SCRIPT_DIR")}"
cd "$HCD_ROOT"

HCD_CONFIG="${HCD_CONFIG:-config/default.yaml}"
mkdir -p logs

echo "=== hcd_analysis pipeline (LF) ==="
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "Root:     $HCD_ROOT"
echo "Config:   $HCD_CONFIG"
echo "Mode:     ${1:-run-all}"
echo "CPUs:     ${SLURM_CPUS_PER_TASK:-1}"

# ---------------------------------------------------------------------------
# Subcommand dispatch
# ---------------------------------------------------------------------------

MODE="${1:-run-all}"
shift || true

case "$MODE" in

  run-all)
    python -m hcd_analysis.cli.run run-all \
      --config "$HCD_CONFIG" \
      --n-workers "${SLURM_CPUS_PER_TASK:-4}" \
      --verbose \
      "$@"
    ;;

  run-hires)
    python -m hcd_analysis.cli.run run-hires \
      --config "$HCD_CONFIG" \
      --n-workers "${SLURM_CPUS_PER_TASK:-4}" \
      --verbose \
      "$@"
    ;;

  convergence)
    python -m hcd_analysis.cli.run convergence \
      --config "$HCD_CONFIG" \
      --verbose \
      "$@"
    ;;

  run-sim)
    python -m hcd_analysis.cli.run run-sim \
      --config "$HCD_CONFIG" \
      --verbose \
      "$@"
    ;;

  run-one)
    python -m hcd_analysis.cli.run run-one \
      --config "$HCD_CONFIG" \
      --verbose \
      "$@"
    ;;

  run-one-array)
    # SLURM array mode: each task handles one LF simulation.
    # Submit with: sbatch --array=0-59 scripts/batch_greatlakes.sh run-one-array
    SIM_LIST_FILE="${HCD_ROOT}/sim_list.txt"
    if [ ! -f "$SIM_LIST_FILE" ]; then
      ls /nfs/turbo/umor-yueyingn/mfho/emu_full/ | grep '^ns' > "$SIM_LIST_FILE"
      echo "Generated $SIM_LIST_FILE with $(wc -l < "$SIM_LIST_FILE") entries"
    fi
    SIM_NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SIM_LIST_FILE")
    if [ -z "$SIM_NAME" ]; then
      echo "ERROR: No sim at array index $SLURM_ARRAY_TASK_ID"
      exit 1
    fi
    echo "Array task $SLURM_ARRAY_TASK_ID → $SIM_NAME"
    python -m hcd_analysis.cli.run run-sim \
      --config "$HCD_CONFIG" \
      --sim "$SIM_NAME" \
      --verbose
    ;;

  benchmark)
    python -m hcd_analysis.cli.run benchmark \
      --config "$HCD_CONFIG" \
      --n-sims 2 \
      --n-snaps 3 \
      --out "outputs/benchmark.json" \
      --verbose
    ;;

  report)
    python -m hcd_analysis.cli.run report \
      --config "$HCD_CONFIG" \
      --figures-dir figures \
      --docs-dir docs \
      --verbose
    ;;

  *)
    echo "ERROR: Unknown mode: $MODE"
    echo "Available modes: run-all, run-hires, convergence, run-sim, run-one, run-one-array, benchmark, report"
    exit 1
    ;;

esac

echo "=== Done: $(date) ==="
