#!/bin/bash
# cavestru1 -- KS selection-function mock challenge (spec 2026-07-18 + Amendment 2, PI-signed
# 2026-07-19; second-ask matrix). Array id -> (arm, mock) via scripts/ks_selboost_arms.batch_cells:
#   0-15 K0_clean | 16-27 K1_flat_lo | 28-43 K2_flat_hi | 44-59 K3_rising | 60-75 K4_u_paper |
#   76-87 K6_inv_u | 88-99 K7_falling | 100-107 K5_joint_meas   (108 fits, 1 fit/task).
# Cost: ~540 CPU-h nominal at the 5 CPU-h/fit planning anchor; 850 CPU-h worst-case envelope
# (approved second ask). LAUNCH IS GATED to the eBOSS-to-KS unblind window AND the mandatory
# 2-mock x 2-setting pilot (spec Sec 7) -- do NOT submit the full array before both.
#   sbatch --array=0-107 scripts/batch_ks_selboost.sh   |   SMOKE=1 sbatch --array=44 ...
# The runner's prior-state tripwire aborts on a drifted prior (EXPECT_* below = the K1a-design
# values); skip-if-exists makes resubmission idempotent.
#
#SBATCH --job-name=ks_selboost
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=10:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/ks_selboost_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/ks_selboost_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail
TID=${SLURM_ARRAY_TASK_ID:-0}
SEED=${SEED:-20260615}
N_WARMUP=${N_WARMUP:-250}; N_SAMPLES=${N_SAMPLES:-300}
EXPECT_LLS_BOOST=${EXPECT_LLS_BOOST:-2.5}
EXPECT_LLS_FRAC_SIGMA=${EXPECT_LLS_FRAC_SIGMA:-0.40}
OUTDIR=${OUTDIR:-/scratch/cavestru_root/cavestru1/mfho/ks_selboost}
NCPU=${SLURM_CPUS_PER_TASK:-4}; SMOKE_FLAG=""; [[ "${SMOKE:-0}" == "1" ]] && SMOKE_FLAG="--smoke"
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
mkdir -p "$OUTDIR" /home/mfho/hcd_priya/logs

# resolve (arm, mock, n_arm, pkl) from the array id through the numpy-light registry module
# (single source with the runner; no arm/offset table duplicated in bash).
read -r ARM MOCK N_ARM PKL < <("$PY" - "$TID" "${SMOKE:-0}" <<'EOF'
import sys
from scripts.ks_selboost_arms import ARMS, batch_cells
cells = batch_cells()
tid = int(sys.argv[1])
assert 0 <= tid < len(cells), f"array id {tid} outside 0..{len(cells)-1}"
arm, mock = cells[tid]
tag = "clean" if arm == "K0_clean" else arm
suffix = ".smoke" if sys.argv[2] == "1" else ""
print(arm, mock, ARMS[arm]["n_mocks"], f"ks_selboost_{tag}_shard_{mock:03d}{suffix}.pkl")
EOF
)
[[ -f "$OUTDIR/$PKL" ]] && { echo "shard $PKL exists -- SKIP"; exit 0; }
echo "=== ks_selboost task ${TID} -> arm=${ARM} mock=${MOCK}/${N_ARM} (seed=${SEED}) start: $(date) ==="
"$PY" -u scripts/run_ks_selboost_shard.py --arm-id "$ARM" --shard "$MOCK" --n-shards "$N_ARM" \
    --seed "$SEED" --n-warmup "$N_WARMUP" --n-samples "$N_SAMPLES" \
    --expect-lls-boost "$EXPECT_LLS_BOOST" --expect-lls-frac-sigma "$EXPECT_LLS_FRAC_SIGMA" \
    --out-dir "$OUTDIR" $SMOKE_FLAG
echo "=== done: $(date) ==="
