#!/bin/bash
# PHASE 2 (Task-2 OOS instrument-resolution arm): NUTS CONFIRMATION of the analytic n_s-safety.
# Injects the analytic worst-n_s out-of-span member bstar (res_instr_injection_basis.npz) into the mock
# truth under each leg's DEPLOYED float-f_res forward, at the +/-1sigma 3-point strength bracket, paired
# clean-vs-injected. The arm is analytically n_s-SAFE (f_perp = cos_M(b*,r_ns) = DESI 0.078 / KS 0.029 /
# eBOSS 0.025, all <0.1); this run CONFIRMS it via NUTS (metals precedent: verify out-of-span with NUTS).
# Disposition: adversarial Tier-A -> FLAG in [0.30,0.50], not auto-fail (analyzer keys on member bstar).
#
# 16-task array: DESI (slow) 2 strengths x 4 shards; eBOSS + KS 2 strengths x 2 shards. Each cell's
# +/-1sigma strengths land in SEPARATE out-dirs so they never pool. Per-leg treatment = the DEPLOYED
# f_res width (DESI tight 0.02 = treatment b; eBOSS 0.05 / KS 0.15 = treatment c --c-prior-sigma).
#   sbatch --array=0-15 scripts/batch_res_oos.sh
# Analyze each cell on completion:
#   for d in oos_desi_bstar_s-1.0 oos_desi_bstar_s1.0 oos_eboss_* oos_ks_*; do \
#     python scripts/analyze_dnuis_bias.py --shard-dir $OUT/$d ; done
#SBATCH --job-name=res_oos
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36g
#SBATCH --time=14:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/res_oos_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/res_oos_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -uo pipefail

NCPU=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
OUT=${OUT:-/scratch/cavestru_root/cavestru0/mfho/res_oos}
TID=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$OUT" /home/mfho/hcd_priya/logs

# cell spec: "leg treatment c_prior strength shard n_shards"   (N_MOCKS=8 per cell, sharded)
SPECS=(
  "desi  b 0.02 -1.0 0 4"
  "desi  b 0.02 -1.0 1 4"
  "desi  b 0.02 -1.0 2 4"
  "desi  b 0.02 -1.0 3 4"
  "desi  b 0.02  1.0 0 4"
  "desi  b 0.02  1.0 1 4"
  "desi  b 0.02  1.0 2 4"
  "desi  b 0.02  1.0 3 4"
  "eboss c 0.05 -1.0 0 2"
  "eboss c 0.05 -1.0 1 2"
  "eboss c 0.05  1.0 0 2"
  "eboss c 0.05  1.0 1 2"
  "ks    c 0.15 -1.0 0 2"
  "ks    c 0.15 -1.0 1 2"
  "ks    c 0.15  1.0 0 2"
  "ks    c 0.15  1.0 1 2"
)
read -r LEG TREAT CPRIOR STRENGTH SHARD NSHARDS <<< "${SPECS[$TID]}"
CELL="$OUT/oos_${LEG}_bstar_s${STRENGTH}"    # +/-1sigma strengths in SEPARATE dirs -> never pool
mkdir -p "$CELL"

echo "=== res_oos task ${TID}: leg=$LEG treat=$TREAT cprior=$CPRIOR strength=$STRENGTH shard=$SHARD/$NSHARDS ==="
echo "    out=$CELL  start: $(date)"
"$PY" -u scripts/run_dnuis_bias_shard.py \
    --arm resolution --survey "$LEG" --treatment "$TREAT" --c-prior-sigma "$CPRIOR" \
    --b-res-oos-member bstar --b-res-oos-strength "$STRENGTH" \
    --shard "$SHARD" --n-shards "$NSHARDS" --n-mocks 8 \
    --n-warmup 250 --n-samples 300 --max-tree-depth 10 \
    --out-dir "$CELL"
echo "=== res_oos task ${TID} done: $(date) ==="
