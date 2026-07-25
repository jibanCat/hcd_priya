#!/bin/bash
# STAGE-V conditional-P1D validation pass (runbook v2.1 step 1; PI decisions #7).
# Array over sims: one task = one sim, all its z snaps, full deployed-convention pass
# with the per-(sim,z) cache byte-anchor (build_xsel_truth_tables.py fails loud on any
# cache mismatch). Launch tooling only; frozen code untouched.
#
#   LF (60 sims):  sbatch --array=0-59 scripts/batch_xsel_truth_tables.sh
#   HR (6 sims):   sbatch --array=0-5 --export=ALL,FIDELITY=hr scripts/batch_xsel_truth_tables.sh
#   (array sizing check: $PY scripts/build_xsel_truth_tables.py --fidelity lf --list)
#
# After both arrays land, pool + figures (login node, minutes):
#   $PY scripts/build_xsel_truth_tables.py --pool --outdir $OUTDIR \
#       --fig-dir cert_campaign_2026-07_report/figures/xsel_truth
#
# Resources: streaming design peaks ~2-3 GB RAM (subsets in float32); 1 CPU (numpy rfft
# is single-threaded here). LF ~18 z x ~3-5 min + IO => walltime 6 h is conservative;
# HR nbins ~1522 => 8 h. Skip-guard: --skip-existing (reruns only unverified outputs),
# so a resubmit after a node failure is cheap.
#
#SBATCH --job-name=xsel_truth
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10g
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/xsel_truth_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/xsel_truth_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

# emu-3.9 + gsl, exactly like the deployed cache builder (batch_tau0_production.sh)
export LD_LIBRARY_PATH=/sw/pkgs/arc/stacks/gcc/10.3.0/gsl/2.7/lib:/home/mfho/.conda/envs/emu-3.9/lib:${LD_LIBRARY_PATH:-}
PY=/home/mfho/.conda/envs/emu-3.9/bin/python3
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

FIDELITY=${FIDELITY:-lf}
TID=${SLURM_ARRAY_TASK_ID:-0}
SEED=${SEED:-20260724}
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru0/mfho/cert_2026-07/xsel_truth}
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

echo "=== xsel_truth ${FIDELITY} sim-index ${TID} -> ${OUTDIR} start: $(date) ==="
"$PY" scripts/build_xsel_truth_tables.py \
    --fidelity "$FIDELITY" --sim-index "$TID" \
    --outdir "$OUTDIR" --seed "$SEED" --skip-existing
echo "=== done: $(date) ==="
