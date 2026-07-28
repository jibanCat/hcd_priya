#!/bin/bash
#SBATCH --job-name=a1c_tests
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=6:00:00
#SBATCH --output=/home/mfho/hcd_priya/logs/a1c_tests_%A.out
#SBATCH --error=/home/mfho/hcd_priya/logs/a1c_tests_%A.err
#
# The REAL-CONTEXT test suites, run on a compute node.
#
# WHY THIS EXISTS (2026-07-28). The A1c pre-launch panel could not complete
# test_legb_metal_modelcplus, test_legb_resolution or test_prod_sbc_checkpoint on the
# interactive node: it sits at load average ~39 on 8 CPUs, so every suite that builds a real
# ensemble context thrashes and times out. They were honestly recorded as NOT-RUN rather than
# as passing. Per the standing rule that heavy JAX work goes through sbatch, they run here.
set -euo pipefail

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK:-4}"
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3

cd /home/mfho/hcd_priya
echo "=== A1c heavy test suites: start $(date) on $(hostname) ==="
echo "=== HEAD: $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD)) ==="

# NOTE on OMP_NUM_THREADS: test_legb_per_leg_alpha carries two KNOWN bitwise baselines that FAIL
# under OMP_NUM_THREADS=1 and pass under default threads (documented env-sensitivity, not a
# regression -- see the 2026-07-23 session record). It is deliberately NOT in this list.
rc=0
for suite in tests/test_legb_metal_modelcplus.py \
             tests/test_legb_resolution.py \
             tests/test_prod_sbc_checkpoint.py \
             tests/test_prod_sbc.py \
             tests/test_legb_leg_a.py \
             tests/test_metal_selfdraw.py \
             tests/test_cert_prereqs.py \
             tests/test_a1c_prelaunch_fixes.py; do
  echo ""
  echo "########## $suite ##########"
  if timeout 3600 "$PY" -m pytest "$suite" -q --no-header -p no:cacheprovider; then
    echo "##### $suite: PASS"
  else
    echo "##### $suite: FAIL (rc=$?)"
    rc=1
  fi
done

echo ""
echo "=== OVERALL: $([ $rc -eq 0 ] && echo ALL-PASS || echo SOME-FAILED) ==="
echo "=== done $(date) ==="
exit $rc
