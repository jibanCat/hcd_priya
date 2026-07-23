#!/bin/bash
# === ATTIC (2026-07-22 freeze triage execution; dispositions PI-approved 2026-07-20, hcd_priya_notes docs/superpowers/2026-07-20-freeze-script-triage.md). NOT IN THE FROZEN FORWARD. ===
# res_corr injection batch (Gate-A era, NORC-superseded).
# cavestru1 SLURM — res_corr INJECTION-RECOVERY gate (Task 2.1, 2026-06-16).
# Runs the committed RCINJ injection arms from run_stepA.build_config (HEAD e1ed5db) DIRECTLY via
# the validated run_stepA.py --run-one path (NO wrapper, NO config edit — the RCINJ block is
# config-only). Per-survey (DESI + KS; eBOSS excluded by design) PAIRED clean/injected mocks:
#   - 2 worst-tilt HF-LOSO sims (ns0.972, ns0.979)
#   - 8 independent mock-noise SEEDS each (_RCINJ_SEEDS = range(8))
#   - clean & injected arms SHARE each (sim,seed) -> byte-identical base truth + cosmic noise,
#     differing ONLY by exp(b1) on the z>=2.8 truth (the out-of-span, worst-n_s-projecting
#     log-res_corr basis member injected into TRUTH ONLY -> the misspecification alpha_res must
#     absorb). Paired Delta_i = post_mean(inj) - post_mean(clean) cancels the shared noise.
#   - REALISTIC mock: sample_metals + a_SiIII ripple (0.045) + tau0_extreme (dtau0 z-slope) +
#     the HR truth's own HCD excess (prior_center="truth"); ALL nuisances free in the fit
#     (alpha_res/_slope, a_SiIII, tau0/dtau0, the 3 alpha_HCD) -> the gate tests whether the
#     misspecification LEAKS into n_s via the high-k nuisance couplings, not a bare arm.
#   - _RCINJ_NCHAINS = 2 chains/arm (post_mean pooled; gives R-hat at minimal cost).
#
# That is per survey: 2 sims x 8 seeds x 2 arms (clean/inj) x 2 chains = 64 fits.
# Both surveys (DESI + KS) => 128 fits total = 128 ARRAY TASKS (--array=0-127).
# Paired-mock count: 16 DESI pairs (sim x seed) + 16 KS pairs -> meets >=8 DESI / >=16 KS.
# One task = one core, single-thread; restartable (a chain whose .npz exists is SKIPPED).
#
# CPU-H ESTIMATE: DESI ~1.2 CPU-h/fit x 64 ~= 77 CPU-h; KS ~0.14 CPU-h/fit x 64 ~= 9 CPU-h.
# Total ~= 86 CPU-h of compute. With ARRAY 1 (MF cert full, ~26 CPU-h) the combined budget is
# ~112 CPU-h -- WELL under the ~300 CPU-h cap (even at ~1.5x DESI headroom, < ~190 CPU-h).
# (The 06:00:00 wall is a SAFETY LIMIT, not the billed time.)
#
# Usage:
#   sbatch --array=0-127 scripts/batch_res_corr_injection.sh
#   # then (compute-gated, run LATER): scripts/analyze_res_corr_injection.py
#
#SBATCH --job-name=rcinj
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/rcinj_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/logs/rcinj_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -euo pipefail

REPO=/home/mfho/hcd_priya
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
CKPT=$REPO/checkpoints/stepA
mkdir -p "$CKPT" "$REPO/logs"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"
export PYTHONNOUSERSITE=1 PYTHONPATH=$REPO JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""

# Build the EXACT 128 RCINJ chain ids that run_stepA.build_config emits (validated against
# build_config: 128 ids, 32 per (survey,arm)). Naming: RCINJ{D|K}_{clean|inj}{sim}s{seed}_c{chain}
# Outer-to-inner loop order matches build_config (survey -> sim -> seed -> arm), with _c{chain}
# appended by add_fiducial; we expand chains last so the list is deterministic and self-checking.
CHAINS=()
for SV in D K; do                      # D=DESI, K=KS   (eBOSS excluded by design)
  for SIM in 972 979; do               # the two worst-tilt HF-LOSO sims
    for SD in 0 1 2 3 4 5 6 7; do       # _RCINJ_SEEDS = range(8): independent mock-noise seeds
      for ARM in clean inj; do          # PAIRED arms (share sim+seed)
        for C in 0 1; do                # _RCINJ_NCHAINS = 2
          CHAINS+=( "RCINJ${SV}_${ARM}${SIM}s${SD}_c${C}" )
        done
      done
    done
  done
done

# Self-check: must be exactly 128 (matches build_config; mismatch => stop before wasting compute).
if [ "${#CHAINS[@]}" -ne 128 ]; then
  echo "ERROR: built ${#CHAINS[@]} chain ids, expected 128 (RCINJ enumeration drift)" >&2
  exit 1
fi

TID=${SLURM_ARRAY_TASK_ID:-0}
if [ "$TID" -ge "${#CHAINS[@]}" ]; then
  echo "ERROR: task $TID >= ${#CHAINS[@]} (array span too large; expect 0..127)" >&2
  exit 1
fi
CHAIN=${CHAINS[$TID]}

if [[ -f "$CKPT/${CHAIN}.npz" ]]; then
  echo "=== ${CHAIN} checkpoint exists — SKIP (task ${TID}) ==="
  exit 0
fi

echo "=== res_corr injection: ${CHAIN} (task ${TID}) start: $(date) ==="
"$PY" -u scripts/run_stepA.py --run-one "${CHAIN}" \
    --n-warmup 250 --n-samples 400 --max-tree-depth 10 --target-accept 0.9
echo "=== done ${CHAIN}: $(date) ==="
