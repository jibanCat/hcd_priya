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

# PRE-SUITE BASELINE for the clean-tree check below. The check compares the END state against
# THIS, so it fires only on mutations the SUITE caused. Needed because hcd_priya_notes is a live
# working-document repo that is legitimately dirty while records are being written -- without a
# baseline the check would fail the job on edits no test made.
BASE_CODE="$(git -C /home/mfho/hcd_priya status --porcelain --untracked-files=no || true)"
BASE_NOTES="$(git -C /home/mfho/hcd_priya_notes status --porcelain --untracked-files=no || true)"
[[ -n "$BASE_NOTES" ]] && echo "=== NOTE: notes repo already dirty at start ($(echo "$BASE_NOTES" | wc -l) files); the clean-tree check compares against this baseline ==="

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
             tests/test_a1c_prelaunch_fixes.py \
             tests/test_a1c_paired.py \
             tests/test_a1c_disposition.py \
             tests/test_finite_l_null.py \
             tests/test_ksfd_paired.py; do
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
echo "########## POST-SUITE CLEAN-TREE CHECK ##########"
# FAIL-LOUD (2026-07-28, PI-directed). A test suite must not mutate committed artifacts. This
# repo has been bitten three times: prod_sbc_pilot.png silently rewritten by
# test_prod_sbc_checkpoint (fixed by passing --figdir), analyze_sbc_perleg's default prefix
# overwriting the committed gate JSON, and emu_bias_allfolds_mf.txt truncated to empty TWICE.
# Manual restore-after-the-fact is not a control: it depends on someone noticing. This turns a
# dirty tree into a test failure, so it cannot be missed.
#
# Scope: tracked files only. Untracked scratch is not a mutation of the record. Deliberate source
# edits are not in scope either -- the suite is run on a committed tree, so ANY tracked-file diff
# here was produced by the tests themselves.
#
# BOTH REPOS (2026-07-29, round 3). The check originally covered only hcd_priya, but
# analyze_sbc_perleg writes its artifacts into the NOTES repo -- and writing the round-3
# regression test fired exactly that: a default OUT_PREFIX overwrote the committed
# sbc_perleg_gate.json with synthetic data. A control that watches the wrong repo is not a
# control. OUT_PREFIX is now mandatory and SBC_PERLEG_OUTDIR redirects the artifact dir, but the
# check is what proves it.
for REPO in /home/mfho/hcd_priya /home/mfho/hcd_priya_notes; do
  if [[ "$REPO" == *_notes ]]; then BASE="$BASE_NOTES"; else BASE="$BASE_CODE"; fi
  NOW="$(git -C "$REPO" status --porcelain --untracked-files=no || true)"
  # Only paths dirty NOW that were not dirty BEFORE are attributable to the suite.
  NEW="$(comm -13 <(printf '%s\n' "$BASE" | sort) <(printf '%s\n' "$NOW" | sort) | sed '/^$/d')"
  if [[ -n "$NEW" ]]; then
    echo "CLEAN-TREE CHECK: FAIL -- the test suite MUTATED tracked files in $REPO:" >&2
    echo "$NEW" >&2
    echo "" >&2
    echo "Per-file diffstat (whole tree, baseline included for context):" >&2
    git -C "$REPO" diff --stat >&2
    echo "" >&2
    echo "Fix the test that writes into the repo (point it at tmp_path / SBC_PERLEG_OUTDIR)," >&2
    echo "do NOT just restore." >&2
    rc=1
  else
    echo "CLEAN-TREE CHECK: PASS ($REPO -- no tracked file modified BY THE SUITE)"
  fi
done

echo ""
echo "=== OVERALL: $([ $rc -eq 0 ] && echo ALL-PASS || echo SOME-FAILED) ==="
echo "=== done $(date) ==="
exit $rc
