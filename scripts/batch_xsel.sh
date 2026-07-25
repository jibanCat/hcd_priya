#!/bin/bash
# X-battery-2 (PROPOSAL-extreme-battery-v2 as adopted by PI record #7): X1/X2/X3 binding
# corners + X4 profiled sub-arm + K8 in-manifold calibration, all paired against the REUSED
# R4/R5 K0 shards (annex OQ5; PAIR_SEED 20260615). Array id -> (arm, mock) via
# scripts/ks_xsel_arms.batch_cells (registry order, 88 cells):
#   0-15 X1_dla100 | 16-31 X2_sub100 | 32-47 X3_lls100 | 48-63 X4_prof |
#   64-71 K8a_eps_hi | 72-79 K8b_eps_lo | 80-87 K8c_kap_hi        (1 fit/task)
# Cost: 440 CPU-h nominal at the 5 CPU-h anchor (X1-X3 240, X4 80, K8 120); 660 worst case.
# LAUNCH GATES (do NOT submit before ALL of):
#   1. stage-V truth tables emitted + reviewed + sha PINNED in scripts/ks_xsel_arms.py
#      (every task fails loud otherwise, by design);
#   2. the X1 swap-off byte-identity gate PASSED:
#      python scripts/run_xsel_shard.py --arm-id X1_dla100 --swap-off-check \
#          --shard 0 --n-shards 16 --expect-lls-boost 2.5 --expect-lls-frac-sigma 0.40 \
#          --out-dir $OUTDIR    (repeat for 2-3 shards; certificates in $OUTDIR).
#   sbatch --array=0-87 scripts/batch_xsel.sh   |   SMOKE=1 sbatch --array=0,16,32,48,64 ...
# Readout: scripts/analyze_xsel.py --shard-dir $OUTDIR --k0-dir <ks_rerun dir>.
#
#SBATCH --job-name=ks_xsel
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=8:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/ks_xsel_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/ks_xsel_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail
TID=${SLURM_ARRAY_TASK_ID:-0}
SEED=${SEED:-20260615}                       # PAIR_SEED: K0-reuse pairing (annex OQ5)
N_WARMUP=${N_WARMUP:-250}; N_SAMPLES=${N_SAMPLES:-300}
EXPECT_LLS_BOOST=${EXPECT_LLS_BOOST:-2.5}
EXPECT_LLS_FRAC_SIGMA=${EXPECT_LLS_FRAC_SIGMA:-0.40}
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru1/mfho/cert_2026-07/xsel}
TRUTH_TABLE=${TRUTH_TABLE:-}                 # optional override; sha pin enforced either way
NCPU=${SLURM_CPUS_PER_TASK:-4}; SMOKE_FLAG=""; [[ "${SMOKE:-0}" == "1" ]] && SMOKE_FLAG="--smoke"
TT_FLAG=(); [[ -n "$TRUTH_TABLE" ]] && TT_FLAG=(--truth-table "$TRUTH_TABLE")
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

# resolve (arm, mock, n_arm, pkl) through the numpy-light registry (single source with the
# runner; no arm/offset table duplicated in bash; batch_cells needs NO truth table).
read -r ARM MOCK N_ARM PKL < <("$PY" - "$TID" "${SMOKE:-0}" <<'EOF'
import sys
from scripts.ks_xsel_arms import ARMS, batch_cells, shard_pkl_name
cells = batch_cells()
tid = int(sys.argv[1])
assert 0 <= tid < len(cells), f"array id {tid} outside 0..{len(cells)-1}"
arm, mock = cells[tid]
print(arm, mock, ARMS[arm]["n_mocks"], shard_pkl_name(arm, mock, smoke=(sys.argv[2] == "1")))
EOF
)
[[ -f "$OUTDIR/$PKL" ]] && { echo "shard $PKL exists -- SKIP"; exit 0; }
echo "=== ks_xsel task ${TID} -> arm=${ARM} mock=${MOCK}/${N_ARM} (seed=${SEED}) start: $(date) ==="
"$PY" -u scripts/run_xsel_shard.py --arm-id "$ARM" --shard "$MOCK" --n-shards "$N_ARM" \
    --seed "$SEED" --n-warmup "$N_WARMUP" --n-samples "$N_SAMPLES" \
    --expect-lls-boost "$EXPECT_LLS_BOOST" --expect-lls-frac-sigma "$EXPECT_LLS_FRAC_SIGMA" \
    --out-dir "$OUTDIR" "${TT_FLAG[@]}" $SMOKE_FLAG
echo "=== done: $(date) ==="
