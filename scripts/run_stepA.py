#!/usr/bin/env python3
"""STEP-A closure-mock launcher — ROBUST, RESTARTABLE, single-node 16-core background runner.

Runs the STEP-A closure battery (23 mocks / 44 chains) as a POOL of single-thread worker
SUBPROCESSES (1 chain ≈ 1 core; ``OMP_NUM_THREADS=1`` + JAX intra/inter-op=1 +
``JAX_PLATFORMS=cpu`` — NO numpyro ``num_chains>1``, which serializes on CPU). Each chain is an
isolated subprocess (``--run-one <chain_id>``) so a crash takes down only that chain. Per-chain
checkpoints (``checkpoints/stepA/<chain_id>.npz``) make the run RESTARTABLE — a re-launch skips
any chain whose checkpoint already exists. A machine-readable health log
(``checkpoints/stepA/health.json`` + a human ``health.txt``) is updated atomically (temp+rename)
per chain transition so a watcher can poll it without ever reading a half-written file.

THE RUN (resolve exact sim names via ``held_out_sims``, closest to each target n_s):
  TIER 1 (LF, mf=False):
    L1a (convergence, 4 dispersed chains): fold0 n_s≈0.81; fold4 n_s≈0.92; fold7 n_s≈1.0  (12)
    L1b (bias, 1 chain): 2 held-out sims/fold × 8 folds spanning n_s                         (16)
  TIER 2 (MF, mf=True; matching fold's emulator + with_mf=True, mf_with_floor=True),
    4 dispersed chains each:
    M1 fold7 n_s≈1.019; M2 fold7 n_s≈1.040 (τ₀ anchor = ladder EXTREME); M3 = M1 truth with
    the HCD per-class z-slope MARGINALIZED; M4 HR-resolution truth (HR sim n_s≈0.979)         (16)

Usage:
  # print the resolved 44-chain table (no run)
  ... run_stepA.py --print-table
  # SMOKE (1 L1b single-chain + 1 fiducial's 2 chains, ~40/40): proves the launcher end-to-end
  ... run_stepA.py --smoke
  # FULL run, 14 workers, in the background (the orchestrator launches this; see bg wrapper)
  ... run_stepA.py --run --workers 14
  # one chain (used internally by the pool; also runnable by hand to debug a single chain)
  ... run_stepA.py --run-one <chain_id>

Env (MANDATORY for every process):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=1 /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/run_stepA.py ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

REPO = "/home/mfho/hcd_priya"
PY = "/home/mfho/.conda/envs/emu-jax/bin/python3"
# CKPT_DIR is env-overridable (STEPA_CKPT_DIR) so a re-run on a CORRECTED forward can write to a
# SEPARATE dir (e.g. checkpoints/stepA_slfix/) without overwriting the existing baseline checkpoints
# and without colliding on the single-writer health.json. Default = the canonical stepA dir.
CKPT_DIR = os.environ.get("STEPA_CKPT_DIR", f"{REPO}/checkpoints/stepA")
HEALTH_JSON = f"{CKPT_DIR}/health.json"
HEALTH_TXT = f"{CKPT_DIR}/health.txt"

# Production NUTS knobs (STEP-A fiducials): dense mass + mtd=10 + warmup 250 (the
# run_legb_convergence defaults). ESS target ~400 (pooled for the ≥4-chain fiducials; per-chain
# for the L1b single chains).
PROD = dict(n_warmup=250, n_samples=400, dense_mass=True, max_tree_depth=10, target_accept=0.9)
ESS_TARGET = 400
N_FOLDS = 8

# ----------------------------------------------------------------------------- #
#  Single-thread environment for every worker (1 chain ≈ 1 core).
# ----------------------------------------------------------------------------- #
def _force_single_thread_env():
    """Set the thread caps BEFORE importing jax/numpy-with-BLAS (must precede the heavy import).
    Returns the env dict (also exported into os.environ for the current process)."""
    # XLA CPU single-thread: disable the Eigen thread pool (the real intra-op knob for JAX-CPU)
    # AND force a single host device. With OMP/BLAS=1 this pins one chain ≈ one core (verified by
    # ~90% — not >100% — per-worker CPU; XLA still allocates an idle thread pool but does not run
    # it). xla_cpu_enable_fast_math left default (numerics unchanged).
    env = dict(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
        XLA_FLAGS=("--xla_cpu_multi_thread_eigen=false "
                   "--xla_force_host_platform_device_count=1"),
        JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
        PYTHONNOUSERSITE="1", PYTHONPATH=REPO)
    os.environ.update(env)
    return env


# ----------------------------------------------------------------------------- #
#  CONFIG TABLE — resolve the 44 chains (sim names at build time).
# ----------------------------------------------------------------------------- #
def _ns_of_sim(d, sim, PARAM_LIMITS):
    names = np.asarray(d["sim_name"])
    rows = np.where(names == sim)[0]
    u = d["params_unit"][rows[0], 0]
    lo, hi = PARAM_LIMITS[0]
    return float(lo + u * (hi - lo))


def _closest_sim(sims_ns, target):
    """(ns, sim) pair whose ns is closest to target, from a list of (ns, sim)."""
    return min(sims_ns, key=lambda t: abs(t[0] - target))


def build_config(verbose=False):
    """Resolve the full 44-chain config (a list of dicts). Each chain dict:
      id, tier, fold, ckpt, sim, n_s, mf, z_slope_marginalized, hr_truth, tau0_extreme,
      chain_id, seed, n_chains (the fiducial pool size for the battery merge).
    Sim names are resolved here from the cache (closest held-out sim to each target n_s)."""
    import hcd_analysis.emulator  # noqa: F401  x64 BEFORE jax
    from hcd_analysis.emulator.data import PARAM_LIMITS
    from hcd_analysis.emulator.closure_legb import held_out_sims, CACHE_PATH, load_cache

    d = load_cache(CACHE_PATH)
    # per-fold (ns, sim) sorted by ns.
    fold_sims = {}
    for f in range(N_FOLDS):
        sims, _ = held_out_sims(d, fold=f)
        fold_sims[f] = sorted([(_ns_of_sim(d, s, PARAM_LIMITS), s) for s in sims])

    def ckpt_of(fold):
        return f"{REPO}/checkpoints/final_fold{fold}"

    cfg = []

    def add_fiducial(mock_id, fold, target_ns=None, *, survey, prior_center="sim_mean",
                     sigma_lls=None, sigma_subdla=None, tau0_extreme=False, n_chains=4, sim=None,
                     lls_truth_boost=1.0, mf=False, hr_truth=False, mf_shape=0.0, desi_floor=False,
                     mf_emucoh=0.0, mf_emucoh_offdiag_only=False, sample_metals=False,
                     inject_a_siiii=0.0, inject_res_corr=None, subdla_center_shift=0.0,
                     hierarchical_hcd=False, hcd_ratio_infl=1.0, hcd_center_shift=0.0,
                     hcd_2d_tilt=False, z_slope_marginalized=False, zslope_realfit=False, seed=0):
        # ``seed`` (default 0 = back-compat for every legacy battery) sets the PRNGKey root, hence the
        # MOCK-NOISE key k_mock = split(fold_in(PRNGKey(seed), mock_index)) in run_one_chain. Distinct
        # seeds at the SAME (survey, sim, fold) therefore give INDEPENDENT cosmic-noise realizations —
        # the RCINJ injection gate uses this to build N>=8 independent paired mocks (clean & injected
        # SHARE each seed so the shared noise cancels in the paired Δ). All chains of one fiducial keep
        # the same seed (they differ only in chain_id → the NUTS key), as before.
        if sim is None:
            ns, sim = _closest_sim(fold_sims[fold], target_ns)
        else:
            ns = _ns_of_sim(d, sim, PARAM_LIMITS)
        for c in range(n_chains):
            cfg.append(dict(
                id=f"{mock_id}_c{c}", mock_id=mock_id, tier="P4", fold=fold, ckpt=ckpt_of(fold),
                sim=sim, n_s=round(ns, 4), survey=survey, prior_center=prior_center,
                sigma_lls=sigma_lls, sigma_subdla=sigma_subdla, tau0_extreme=tau0_extreme,
                lls_truth_boost=lls_truth_boost,
                mf=bool(mf), z_slope_marginalized=bool(z_slope_marginalized), hr_truth=bool(hr_truth),
                mf_shape=float(mf_shape), desi_floor=bool(desi_floor), mf_emucoh=float(mf_emucoh),
                mf_emucoh_offdiag_only=bool(mf_emucoh_offdiag_only),
                sample_metals=bool(sample_metals), inject_a_siiii=float(inject_a_siiii),
                inject_res_corr=(dict(inject_res_corr) if inject_res_corr else None),
                subdla_center_shift=float(subdla_center_shift),
                hierarchical_hcd=bool(hierarchical_hcd), hcd_ratio_infl=float(hcd_ratio_infl),
                hcd_center_shift=float(hcd_center_shift), hcd_2d_tilt=bool(hcd_2d_tilt),
                zslope_realfit=bool(zslope_realfit),
                chain_id=c, n_chains=n_chains, seed=int(seed)))

    # === Phase-4 SEPARATE-inference closure: PRIYA τ₀ + physical HCD slope, NON-circular center ===
    # Supersedes the old joint+13-rung STEP-A list. DESI-only + KS-only, sim-mean (NON-circular) HCD
    # prior center, spanning INTERIOR n_s (avoid the fold0 n_s wall). 4 dispersed chains each.
    add_fiducial("D_f3", 3, 0.90, survey="DESI")     # n_s≈0.90
    add_fiducial("D_f4", 4, 0.92, survey="DESI")     # n_s≈0.92
    add_fiducial("D_f6", 6, 0.966, survey="DESI")    # n_s≈0.966 (Planck) — reference point
    add_fiducial("D_f7", 7, 1.00, survey="DESI")     # n_s≈1.0 (eBOSS / extrapolation ridge)
    add_fiducial("K_f4", 4, 0.92, survey="KS")       # KS leg, mid n_s
    add_fiducial("K_f6", 6, 0.966, survey="KS")      # KS leg, Planck n_s
    # SENSITIVITY arms at the fold6 (Planck) DESI point:
    add_fiducial("D_f6_lit", 6, 0.966, survey="DESI", prior_center="lit")     # real-fit prior center
    add_fiducial("D_f6_sig40", 6, 0.966, survey="DESI", sigma_lls=0.40)       # looser LLS width
    add_fiducial("D_f6_tau0x", 6, 0.966, survey="DESI", tau0_extreme=True)    # τ₀-funnel check

    # === Phase-4b (2026-06-11): σ_LLS + σ_subDLA width scans + IGM-parameter stress fiducials ===
    # Width scans at the D_f6 Planck mock (same sim → same noise → clean bias-vs-width):
    for sl in (0.08, 0.25, 0.80):
        add_fiducial(f"D_f6_sigL{int(round(sl*100)):02d}", 6, 0.966, survey="DESI", sigma_lls=sl, n_chains=3)
    for ss in (0.20, 0.80, 1.50):
        add_fiducial(f"D_f6_sigS{int(round(ss*100)):03d}", 6, 0.966, survey="DESI", sigma_subdla=ss, n_chains=3)
    # IGM-parameter stress fiducials (DESI, sim-mean center): the held-out sim at each IGM extreme.
    import numpy as _np
    from hcd_analysis.emulator.inference import PARAM_NAMES as _PN
    _pidx = {n: i for i, n in enumerate(_PN)}; _sn = _np.asarray(d["sim_name"]); _pu = d["params_unit"]
    _all = sorted(set(s for lst in fold_sims.values() for _, s in lst))
    def _punit(s, p): return float(_pu[_np.where(_sn == s)[0][0], _pidx[p]])
    def _fold_of(s):
        for f, lst in fold_sims.items():
            if s in [ss for _, ss in lst]: return f
    def _igm_pick(p, hi): return (max if hi else min)((_punit(s, p), s) for s in _all)[1]
    for nm, p, hi in [("IGM_herei_hi", "herei", True), ("IGM_heref_lo", "heref", False),
                      ("IGM_heref_hi", "heref", True), ("IGM_alphaq_lo", "alphaq", False),
                      ("IGM_alphaq_hi", "alphaq", True), ("IGM_bhfb_lo", "bhfeedback", False)]:
        s = _igm_pick(p, hi); add_fiducial(nm, _fold_of(s), survey="DESI", sim=s, n_chains=4)

    # === Phase-4c (2026-06-11): per-survey LLS pin VALIDATION (real-fit "lit" prior) ===
    # Each survey's MOCK carries that survey's effective LLS level, fit with that survey's pin:
    #   D_lls — DESI leg, DESI pin (1.06× cosmic-avg, σ0.15); mock at the sim level (≈cosmic, the
    #           DESI-pin target to ~6%). Validates the real-fit DESI closure + α_LLS→dN/dX recovery.
    #   K_lls — KS leg, KS pin (2.5×1.06=2.65× cosmic, σ0.40 broad); mock LLS BOOSTED ×2.65 to the
    #           KS selection level (arXiv:2509.18271). Validates that the boosted pin recovers
    #           cosmology + the boosted α_LLS when the data genuinely carries the excess.
    # NOTE: D_lls/D_lls_m pinned to the ORIGINAL σ_LLS=0.15 (the DESI default was later moved to 0.30,
    # b5cc088) so the σ0.15-vs-σ0.30 contrast (D_lls_m vs D_lls_m30) stays reproducible from config.
    add_fiducial("D_lls", 6, 0.966, survey="DESI", prior_center="lit", lls_truth_boost=1.0, sigma_lls=0.15)
    add_fiducial("K_lls", 6, 0.966, survey="KS",   prior_center="lit", lls_truth_boost=2.65)
    # MATCHED-center arms (mock LLS == the survey pin center): isolates pin self-consistency from
    # the center-OFFSET sensitivity. D_lls (boost 1.0) is 6% below the DESI 1.06× pin → the
    # center-sensitivity arm; D_lls_m (boost 1.06) sits ON the DESI pin center → the clean DESI case.
    add_fiducial("D_lls_m", 6, 0.966, survey="DESI", prior_center="lit", lls_truth_boost=1.06, sigma_lls=0.15)
    # D_lls_m30: matched DESI at the MODERATE σ_LLS=0.30 (the new HCD_LLS_SURVEY_FRAC_SIGMA[DESI];
    # PI 2026-06-11) — re-validates that DESI recovers A_p in-gate at the moderate width (D_lls_m at
    # the old σ0.15 gave +1.01σ). Uses the survey pin center (1.06×) with the explicit 0.30 width.
    add_fiducial("D_lls_m30", 6, 0.966, survey="DESI", prior_center="lit", lls_truth_boost=1.06,
                 sigma_lls=0.30)
    # UN-CONFOUND width-vs-center (Bayesian referee 2026-06-11): the D_lls_m fold6-Planck sim is
    # LLS-POOR (w_LLS≈77.5% of the population median the DESI prior centers on) → its "matched" arm
    # was actually +0.8σ HIGH. Re-run on a sim whose w_LLS ≈ the population median (ratio 0.992) so
    # the lit prior center genuinely EQUALS the mock truth — then σ0.15 vs σ0.30 isolates the WIDTH
    # effect alone. If σ0.15 STILL biases A_p here, it is the width; if it recovers, it was the center.
    _SIM_MED = "ns0.972Ap1.69e-09herei3.87heref2.65alphaq2.12hub0.722omegamh20.144hireionz7.53bhfeedback0.0507"
    add_fiducial("D_llsmed",   6, survey="DESI", prior_center="lit", lls_truth_boost=1.06,
                 sigma_lls=0.15, sim=_SIM_MED)
    add_fiducial("D_llsmed30", 6, survey="DESI", prior_center="lit", lls_truth_boost=1.06,
                 sigma_lls=0.30, sim=_SIM_MED)
    # MULTI-FOLD width check (PI 2026-06-11): repeat the matched-center σ0.15 vs σ0.30 contrast across
    # folds spanning n_s, to test whether the n_s −0.94σ (D_llsmed σ0.15) is the LLS WIDTH or this
    # sim's LOSO scatter (sign-flips across folds ⇒ scatter), and whether σ0.15-matched recovers A_p
    # generally. All median-w_LLS (ratio≈1.01), box-interior. With fold6 above: 4 folds, n_s 0.907–0.982.
    _MED_FOLDS = {
        3: "ns0.907Ap1.5e-09herei3.75heref2.77alphaq2.04hub0.662omegamh20.144hireionz7.47bhfeedback0.0347",
        5: "ns0.953Ap1.74e-09herei4.07heref2.93alphaq2.31hub0.692omegamh20.142hireionz7.38bhfeedback0.0573",
        7: "ns0.982Ap1.81e-09herei3.62heref2.78alphaq1.78hub0.72omegamh20.143hireionz7.55bhfeedback0.0421",
    }
    for _f, _s in _MED_FOLDS.items():
        add_fiducial(f"D_lmed{_f}_15", _f, survey="DESI", prior_center="lit", lls_truth_boost=1.06,
                     sigma_lls=0.15, sim=_s)
        add_fiducial(f"D_lmed{_f}_30", _f, survey="DESI", prior_center="lit", lls_truth_boost=1.06,
                     sigma_lls=0.30, sim=_s)

    # === Phase-5a EMUCOH closure validation (2026-06-12): the BLOCKING referee gate for the
    # 60-sim LF-emulator k-coherent C_emu term ("emucoh"). The term is wired (run_one_chain
    # mf_emucoh knob) + unit-tested + 4-referee-reviewed, but has NO inference-level validation;
    # both the Bayesian and cosmology referees call that blocking before production.
    #
    # DESIGN: mirror the two multi-fold families that exhibit the per-fold A_p/n_s LOSO scatter the
    # emucoh term is meant to absorb. For each baseline we run a MATCHED PAIR with CURRENT code —
    # "_EC0" (mf_emucoh=0, the OFF control) and "_EC1" (mf_emucoh=1.0, infl=1, ON) — IDENTICAL in
    # every other field (same fold+sim+seed ⇒ byte-identical mock data + noise; mock generation does
    # not read mf_emucoh). We run the OFF arm FRESH rather than reuse the old D_f*/D_lmed* checkpoints
    # so that code drift since those ran (e.g. the 2026-06-11 LLS-prior hardening) cannot confound
    # emucoh; the old checkpoints remain a free back-compat cross-check. These mocks carry NO MF floor
    # (mf=False, no mf_shape/desi_floor) so the emucoh term is isolated cleanly.
    #   Family A (sim-mean center)          — D_f3/4/6/7      (folds 3,4,6,7; OFF A_p z-RMS ~0.75σ).
    #   Family B (matched-lit center σ0.15) — D_lmed{3,5,7}_15 + D_llsmed (folds 3,5,7,6; OFF n_s
    #                                         sign-flips ±~0.9σ; A_p fold7 +2.44σ ridge outlier).
    # Triad to read EC1 vs EC0 (scripts/analyze_emucoh_validation.py): per-fold point-estimate scatter
    # SHRINKS, σ_Ap/σ_ns WIDEN (PSD monotonicity — a SHRINK is the θ-dependent-covariance bug
    # signature), |bias z| coverage → nominal.
    _EMUCOH_BASE = [
        ("D_f3",       3, dict(target_ns=0.90)),
        ("D_f4",       4, dict(target_ns=0.92)),
        ("D_f6",       6, dict(target_ns=0.966)),
        ("D_f7",       7, dict(target_ns=1.00)),
        ("D_lmed3_15", 3, dict(prior_center="lit", lls_truth_boost=1.06, sigma_lls=0.15, sim=_MED_FOLDS[3])),
        ("D_lmed5_15", 5, dict(prior_center="lit", lls_truth_boost=1.06, sigma_lls=0.15, sim=_MED_FOLDS[5])),
        ("D_lmed7_15", 7, dict(prior_center="lit", lls_truth_boost=1.06, sigma_lls=0.15, sim=_MED_FOLDS[7])),
        ("D_llsmed",   6, dict(prior_center="lit", lls_truth_boost=1.06, sigma_lls=0.15, sim=_SIM_MED)),
    ]
    for _base, _fld, _kw in _EMUCOH_BASE:
        add_fiducial(f"{_base}_EC0", _fld, survey="DESI", mf_emucoh=0.0, **_kw)   # OFF control
        add_fiducial(f"{_base}_EC1", _fld, survey="DESI", mf_emucoh=1.0, **_kw)   # ON (infl=1)

    # EC2: per-term DIAGONAL ALLOCATION re-validation (cosmology referee follow-on, 2026-06-12) —
    # same as EC1 (emucoh ON, infl=1) but mf_emucoh_offdiag_only=True: absorb emucoh's diagonal into
    # emu_var (via max) and add ONLY its off-diagonal, dropping the conservative ×1.3 on-top DESI
    # diagonal. Representative subset (the fold7 outlier emucoh helped most + a worsened + a typical):
    # does the off-diagonal still de-bias fold7 while the tighter diagonal keeps coverage ≥ nominal?
    _EMUCOH_ODA = {"D_f3", "D_f6", "D_lmed7_15", "D_llsmed"}
    for _base, _fld, _kw in _EMUCOH_BASE:
        if _base in _EMUCOH_ODA:
            add_fiducial(f"{_base}_EC2", _fld, survey="DESI", mf_emucoh=1.0,
                         mf_emucoh_offdiag_only=True, **_kw)

    # === eBOSS DR14 closure cert (2026-06-13): the low-k PRODUCTION SHAKEDOWN before KS. Held-out-sim
    # recovery on the eBOSS leg (survey="eBOSS" → run_one_chain builds build_legb_ctx with_eboss and
    # filters the ctx to the eBOSS-only leg). Fiducials span n_s incl. Planck 0.966. sim-mean
    # (non-circular) center, LF path. NOTE: a_SiIII is NOT yet sampled (Phase-4d test-3 follow-on) →
    # this is the BASIC RECOVERY cert (the emulator is most accurate at eBOSS low-k → expect clean);
    # the decisive SiIII-injection arm follows once a_SiIII is wired. Gate: R-1<0.01 / ≥4 dispersed
    # chains / |bias z|<0.2σ / χ²~1.
    for _enm, _ef, _ens in [("E_f5", 5, 0.95), ("E_f6", 6, 0.966), ("E_f7", 7, 1.00)]:
        add_fiducial(_enm, _ef, _ens, survey="eBOSS")
    # eBOSS WITH MF (PI 2026-06-13): Fernandez+2024 ran the resolution correction, and the MF
    # correction is a TILT (~+4% low-k → −6% high-k) whose low-k end reaches the eBOSS band — n_s
    # reads the tilt. mf=True applies the MF-corrected forward AND truth (the gate-invariant Test-A
    # style → should recover ≈ the LF result IF MF doesn't alias cosmology on eBOSS; same sims as the
    # LF E_f* so the LF-vs-MF comparison is at matched cosmology/noise). The forward MF imprint on the
    # eBOSS low-k tilt is quantified separately (scripts/diag_eboss_mf_imprint.py).
    for _enm, _ef, _ens in [("E_f5_mf", 5, 0.95), ("E_f6_mf", 6, 0.966), ("E_f7_mf", 7, 1.00)]:
        add_fiducial(_enm, _ef, _ens, survey="eBOSS", mf=True)
    # eBOSS SiIII-INJECTION cert (PI 2026-06-13): inject SiIII (a_SiIII≈0.045 = f_SiIII/(1−⟨F⟩), a
    # ±9% ripple, ~6.7 periods in-band) into the mock truth AND sample a_SiIII in the forward. The
    # decisive test (Bayesian lens): does a_SiIII absorb the in-band ripple with NO n_s/A_p leakage?
    # Compare to the LF E_f* (no metals): if cosmology recovery matches, SiIII is cleanly marginalized.
    for _enm, _ef, _ens in [("E_f5_si", 5, 0.95), ("E_f6_si", 6, 0.966), ("E_f7_si", 7, 1.00)]:
        add_fiducial(_enm, _ef, _ens, survey="eBOSS", sample_metals=True, inject_a_siiii=0.045)

    # === Cosmology-safety arm (#9 referee-panel must-do, 2026-06-13): subDLA prior-CENTER shift ===
    # The panel upheld DO-NOT-BUILD the HCD-class C_emu term but flagged the subDLA pull is NOT
    # orthogonal to cosmology (closure corr(subDLA, n_s)=+0.82). The one residual risk: on REAL data a
    # MIS-CENTERED subDLA prior could drag n_s along that degeneracy. This arm tests it on the JOINT
    # DESI+KS legs (the production combination, where KS could re-activate the subDLA<->A_p channel):
    # for 2 mocks, run the subDLA prior center at 0 / +1σ / −1σ and verify n_s/A_p MEANS stay <0.3σ.
    for _xnm, _xf, _xns in [("XS_f6", 6, 0.966), ("XS_f4", 4, 0.92)]:
        for _sh, _tag in [(0.0, "_s0"), (1.0, "_sp"), (-1.0, "_sm")]:
            add_fiducial(_xnm + _tag, _xf, _xns, survey="DESI+KS",
                         prior_center="sim_mean", subdla_center_shift=_sh)

    # === HIERARCHICAL HCD-prior closure ("Option B", 2026-06-13): PAIRED ON-vs-OFF validation ===
    # The A_hcd × {r_subdla, r_dla} reparam collapses the flat subDLA↔DLA exchange direction (the
    # n_s-leak source flagged by the XS cosmology-safety arm: closure corr(subDLA pull, n_s)=+0.82)
    # onto ONE fixed-shape additive amplitude + prior-pinned shape ratios. Run on the JOINT DESI+KS
    # legs (the production combination where KS could re-activate the subDLA↔A_p channel), sim-mean
    # (NON-circular) HCD center. Each baseline runs a MATCHED PAIR with the SAME fold/sim/seed (the
    # mock data + noise are byte-identical; mock generation does NOT read hierarchical_hcd):
    #   "_HB0" — hierarchical_hcd=False (the legacy 3-independent-α control);
    #   "_HB1" — hierarchical_hcd=True  (the reparam, hcd_ratio_infl=1 = the closure 10%/12% widths).
    # The validation gate (design doc §"Validation gate"): bias_z(ON)==bias_z(OFF) within MC + the
    # subDLA mean bias collapses + the ±1σ subDLA-center n_s coupling drops <0.3σ + 0 divergences.
    # NOT launched here — config only. (run_stepA.run_one_chain threads hierarchical_hcd through
    # build_legb_ctx; the ratio centers/widths auto-derive from the raw sim w_c, must-fix #1.)
    for _hnm, _hf, _hns in [("HB_f6", 6, 0.966), ("HB_f4", 4, 0.92)]:
        add_fiducial(_hnm + "_HB0", _hf, _hns, survey="DESI+KS", prior_center="sim_mean",
                     hierarchical_hcd=False)
        add_fiducial(_hnm + "_HB1", _hf, _hns, survey="DESI+KS", prior_center="sim_mean",
                     hierarchical_hcd=True, hcd_ratio_infl=1.0)
        # A_HCD-center ±1σ cosmology-safety arm (design must-fix #3, gate-side): under Option B the
        # n_s coupling relocates onto the overall HCD amplitude, so mis-specify the A_HCD center ±1σ
        # and verify n_s/A_p MEANS stay <0.3σ (the hierarchical analog of the XS subDLA-center arm).
        add_fiducial(_hnm + "_HBp", _hf, _hns, survey="DESI+KS", prior_center="sim_mean",
                     hierarchical_hcd=True, hcd_ratio_infl=1.0, hcd_center_shift=1.0)
        add_fiducial(_hnm + "_HBm", _hf, _hns, survey="DESI+KS", prior_center="sim_mean",
                     hierarchical_hcd=True, hcd_ratio_infl=1.0, hcd_center_shift=-1.0)

    # === v2 SUBMANIFOLD test (PI insight 2026-06-14): hierarchical-HCD + z-slope MARGINALIZED ===
    # HB_f6 showed Option B (1D A_HCD, FIXED slopes) RELOCATES the n_s coupling onto the A_HCD center
    # (+0.51σ per ±1σ, corr +0.52). The PI's physics: the HCD classes trace a correlated 2D submanifold
    # (CDDF amplitude × DIFFERENTIAL z-evolution — the N_HI-rich gas evolves differently with z, measured
    # in PRIYA's HR/LF dN/dX: LLS swings 1.29→0.96 while subDLA/DLA stay ~flat). Collapsing to 1D threw
    # away the z-evolution dimension, whose z-tilt signature is ORTHOGONAL to the n_s k-tilt and could
    # DECORRELATE A_HCD from n_s if the data constrains it. Test: marginalize the per-class z-slope
    # (s_lls/s_subdla/s_dla sampled) ON TOP of Option B and re-measure the A_HCD-center→n_s coupling.
    # If it drops below the 0.51σ fixed-slope value → the z-dimension breaks the coupling → refine to a
    # tight 1–2 param submanifold (one CDDF z-tilt + class-differential slopes fixed from the HR dN/dX).
    # NOT launched here — config only; launch after HB_f4 lands (run_hz_pool.py).
    for _zfn, _zff, _zfns in [("HZ_f6", 6, 0.966), ("HZ_f4", 4, 0.92)]:
        add_fiducial(_zfn + "_z0", _zff, _zfns, survey="DESI+KS", prior_center="sim_mean",
                     hierarchical_hcd=True, hcd_ratio_infl=1.0, z_slope_marginalized=True)
        add_fiducial(_zfn + "_zp", _zff, _zfns, survey="DESI+KS", prior_center="sim_mean",
                     hierarchical_hcd=True, hcd_ratio_infl=1.0, z_slope_marginalized=True, hcd_center_shift=1.0)
        add_fiducial(_zfn + "_zm", _zff, _zfns, survey="DESI+KS", prior_center="sim_mean",
                     hierarchical_hcd=True, hcd_ratio_infl=1.0, z_slope_marginalized=True, hcd_center_shift=-1.0)

    # === 2D AMPLITUDE×TILT submanifold closure (HT arms, PI refinement 2026-06-14) ===
    # The PI's 2D submanifold: a GENUINE 2-dof HCD sector = pivot AMPLITUDE A_HCD × a GLOBAL z-TILT
    # B_HCD, with the class-differential z-evolution FIXED (δs_c = HCD_INCIDENCE_SLOPE − slope[0],
    # the SIM incidence-weight slope; NOT the lit/sim ratio — see hcd-dndx-zslope-bug):
    #   α_c(z) = A_HCD · r_c · ((1+z)/(1+z_p))^(B_HCD + δs_c).
    # B_HCD is the DATA-constrained 2nd submanifold dimension whose z-tilt signature is ORTHOGONAL
    # to the n_s k-tilt — the hypothesis (HZ diagnosis) is that this z-evolution dof DECORRELATES
    # A_HCD from n_s, fixing the relocation that Option B-1D (A_HCD × FIXED ratios) suffered (the
    # A_HCD-center→n_s coupling +0.51σ). Differs from HZ (free per-class s_*): HT ties the slopes to
    # ONE B_HCD + FIXED δs_c (one z-tilt dof, not three) — the tight submanifold the HZ note proposed.
    # Joint DESI+KS (the production combination), sim_mean (NON-circular) center, hcd_2d_tilt=True.
    # A_HCD-center 0/+1σ/−1σ via hcd_center_shift (the cosmology-safety arm: does the 2D tilt keep
    # the A_HCD-center→n_s coupling <0.3σ?). 2 folds × 3 shifts × 4 chains = 24 chains. NOT launched
    # here — config only. (run_one_chain threads hcd_2d_tilt → build_legb_ctx; the closure δs_c /
    # B_HCD center/width auto-derive from HCD_INCIDENCE_SLOPE, self-consistent with the forward.)
    for _tfn, _tff, _tfns in [("HT_f6", 6, 0.966), ("HT_f4", 4, 0.92)]:
        add_fiducial(_tfn + "_t0", _tff, _tfns, survey="DESI+KS", prior_center="sim_mean",
                     hierarchical_hcd=True, hcd_2d_tilt=True, hcd_ratio_infl=1.0)
        add_fiducial(_tfn + "_tp", _tff, _tfns, survey="DESI+KS", prior_center="sim_mean",
                     hierarchical_hcd=True, hcd_2d_tilt=True, hcd_ratio_infl=1.0, hcd_center_shift=1.0)
        add_fiducial(_tfn + "_tm", _tff, _tfns, survey="DESI+KS", prior_center="sim_mean",
                     hierarchical_hcd=True, hcd_2d_tilt=True, hcd_ratio_infl=1.0, hcd_center_shift=-1.0)

    # === Phase-5a Test A (2026-06-11): MF gate-invariant M-tier re-run at HR cosmologies ===
    # with_mf=True → truth = MF-corrected LF AND forward = MF-corrected LF (the GATE INVARIANT: the
    # correction cancels in ΔP). Confirms MF does not ALIAS cosmology and that A_p/n_s recover at HR
    # cosmologies with the CURRENT baseline (per-survey pin, box, τ₀ 2-param, subDLA 1.0, α≥0). The old
    # STEP-A M4 had A_p +1.63σ (pre-fix HCD→A_p) — this re-run tests whether the fixes carried it in-gate.
    # sim-mean (non-circular cert) center, DESI leg. (The genuine resolution test = Phase-5a Test B, the
    # real-HR-truth HF-LOSO, which needs the make_hr_truth_from_cache builder.)
    _hr_seen = set()
    for _tns in (0.90, 0.95, 0.979):
        _hr = _resolve_hr_sim(d, fold_sims, PARAM_LIMITS, target_ns=_tns)
        if _hr["sim"] in _hr_seen:
            continue
        _hr_seen.add(_hr["sim"])
        add_fiducial(f"M_hr{int(round(_hr['n_s'] * 1000))}", _hr["fold"], survey="DESI",
                     sim=_hr["sim"], mf=True)

    # === Phase-5a Test B (2026-06-11): GENUINE HF-LOSO — real HR truth + MF correction EXCLUDING it ===
    # For each of the 6 HR sims: truth = its REAL measured P1D (make_hr_truth_from_cache, no MF);
    # forward = LF emu × MF correction fit WITHOUT it (mf_exclude_held via hr_truth=True). Tests the
    # MF-resolution GENERALIZATION in the inference (the integrated MF-LOSO). prior_center="truth"
    # centers the HCD α on the HR sim's own w_c → isolates the resolution from the LLS-center lever.
    from hcd_analysis.emulator import multifidelity as _MF
    _hrn = sorted(set(s.decode() if isinstance(s, bytes) else s for s in _MF.load_cache(_MF.HR_CACHE)["sim_name"]))
    for _hs in _hrn:
        _f = _fold_of(_hs)
        if _f is None:
            continue
        _ns = _ns_of_sim(d, _hs, PARAM_LIMITS)
        add_fiducial(f"HFLOSO{int(round(_ns * 1000))}", _f, survey="DESI", sim=_hs,
                     mf=True, hr_truth=True, prior_center="truth")

    # === Phase-5a MF n_s HIGH-K CERTIFICATION on DESI+KS (PI-approved 2026-06-14) ===
    # The open item (notes 2026-06-12-phase5a-mf-hf-closure §"Next steps": "re-run Test B on BOTH
    # DESI and KS, require the n_s bias |z|<1"; 2026-06-14-decisions §7): the DESI-only Test B gave a
    # coherent n_s tilt up to +2.80σ at ns0.972 that the σ-only shape-floor can't de-bias. This is the
    # PRODUCTION cert: re-run the GENUINE HF-LOSO Test B THROUGH the production MF forward on the JOINT
    # DESI+KS legs (survey="DESI+KS" → run_one_chain keeps BOTH legs; KS carries its mf_floor_on=True,
    # DESI no floor — the production baseline). Per HR sim: truth = its REAL measured P1D
    # (make_hr_truth_from_cache, no MF), forward = LF emu × MF correction fit EXCLUDING it
    # (mf_exclude_held via hr_truth=True), prior_center="truth" (isolates resolution from the LLS
    # center). VERDICT GATE: per-fold n_s |bias_z| < 1 (ideally <0.2σ) on the DESI+KS joint. If it
    # holds, the production MF controls the high-k n_s tilt; if not, a θ-resolved res_corr / wider σ
    # is needed. 6 HR sims × 4 chains = 24 chains; CPU forward, no SLURM. id = HFLOSO_DK{ns}.
    for _hs in _hrn:
        _f = _fold_of(_hs)
        if _f is None:
            continue
        _ns = _ns_of_sim(d, _hs, PARAM_LIMITS)
        add_fiducial(f"HFLOSO_DK{int(round(_ns * 1000))}", _f, survey="DESI+KS", sim=_hs,
                     mf=True, hr_truth=True, prior_center="truth")

    # === Phase-5a MF n_s HIGH-K CERTIFICATION on KS-ONLY (PI-approved 2026-06-14) — PER-SURVEY ===
    # The real fits are SEPARATE per survey, so the cert must be per-survey too: the DESI-only
    # (HFLOSO*) and KS-only (HFLOSO_KS*) Test-B are the HEADLINE; the joint DESI+KS (HFLOSO_DK*)
    # above is context/secondary. Identical GENUINE HF-LOSO Test B as HFLOSO_DK, but on the KS-ONLY
    # leg (survey="KS" → run_one_chain keeps ONLY the KS leg; with_eboss=False and the DESI leg is
    # filtered OUT). KS carries its loader default mf_floor_on=True (the small-scale leg), and the
    # production MF forward (with_mf=True, mf_with_floor=True) is on; the HF-LOSO MF correction is
    # fit EXCLUDING this HR sim (mf_exclude_held via hr_truth=True). Per HR sim: truth = its REAL
    # measured P1D (make_hr_truth_from_cache, no MF), prior_center="truth" (isolates resolution from
    # the LLS center). VERDICT GATE: per-fold n_s |bias_z| < 1 (ideally <0.2σ) on KS-only. KS reaches
    # higher k than DESI, so this is the decisive high-k tilt test for the KS real fit. 5 HR sims ×
    # 4 chains = 20 chains; CPU forward, no SLURM. id = HFLOSO_KS{ns}.
    for _hs in _hrn:
        _f = _fold_of(_hs)
        if _f is None:
            continue
        _ns = _ns_of_sim(d, _hs, PARAM_LIMITS)
        add_fiducial(f"HFLOSO_KS{int(round(_ns * 1000))}", _f, survey="KS", sim=_hs,
                     mf=True, hr_truth=True, prior_center="truth")

    # === Phase-2 res_corr INJECTION-RECOVERY gate (TASK-2.1, spec §4.2) — the DECISIVE n_s-safety
    # gate for the anchor+marginalize design. The marginalized res_corr amplitude alpha_res(/_slope)
    # is ALWAYS sampled in _legb_model (Task 1.3, no flag), so an HFLOSO-style cert with alpha free
    # is the forward. This arm injects the worst-n_s-projecting, OUT-OF-SPAN, z>=2.8-localized
    # log-res_corr basis member b1 (built by scripts/build_res_corr_injection_basis.py, provably
    # C_data^-1-orthogonal to the alpha(z) span, cos<0.8) into the mock TRUTH ONLY (never the
    # forward → it cannot cancel; it is the misspecification alpha_res must absorb). PAIRED design:
    # a CLEAN control (no injection) + an INJECTED arm at the SAME (survey, sim, fold, SEED) so the
    # two mocks share byte-identical base truth + cosmic noise and differ ONLY by exp(b1) on the
    # z>=2.8 truth → the shared noise cancels in Δ_i = post_mean(inj) − post_mean(clean). The Phase-2
    # gate (scripts/analyze_res_corr_injection.py, run-time, compute-gated) asserts the PAIRED
    # |mean Δ|+2·SE < 0.3·σ_ref on A_p AND n_s, per survey, in FIXED-reference (α-fixed) units.
    # Config-only here (no NUTS): the b1 member is loaded inside make_legb_mock from the spec dict.
    #
    # 4-LENS PANEL FIXES (2026-06-15) baked in below:
    #  (1) REALISTIC mock — sample_metals=True + inject_a_siiii (SiIII ripple) + tau0_extreme
    #      (nonzero dτ₀≈0.20 z-slope) + the HR truth's own HCD excess (prior_center="truth"); ALL
    #      nuisances free in the fit (alpha_res/_slope always sampled, a_SiIII via sample_metals,
    #      tau0/dτ₀ + the 3 α_HCD always sampled). So the gate tests whether the misspecification
    #      LEAKS into n_s VIA the high-k nuisance couplings (alpha_res↔a_SiIII↔n_s, alpha_res↔dτ₀),
    #      not a bare arm. (The draft used sample_metals=False / no SiIII / dτ₀=0 — fixed.)
    #  (2) N>=8 INDEPENDENT paired mocks per survey (>=16 for KS): 2 worst-tilt HF-LOSO sims
    #      (ns0.972, ns0.979) × _RCINJ_SEEDS distinct mock-noise SEEDS = 2×len(seeds) pairs/survey;
    #      the clean & injected arms SHARE each seed so Δ_i cancels the shared noise; distinct seeds
    #      give independent realizations (the draft hardcoded seed=0 → only ~2 independent mocks).
    #  (3) eBOSS EXCLUDED from the injection gate (documented): eBOSS k_max 0.0195 s/km sits
    #      essentially inside the 5×k_box(z=3)≈0.019 res_corr anchor → res_corr has minimal high-k
    #      leverage on the eBOSS band (and eBOSS has no HR cache → mf=False → no alpha leverage). The
    #      eBOSS res_corr safety is covered by the SEPARATE eBOSS MF-anchored re-cert (spec §4.2
    #      gate 4 / plan Task 2.3). So the gate surveys are DESI + KS only.
    _RC_BASIS = f"{REPO}/hcd_analysis/_emulator_data/res_corr_injection_basis.npz"
    _rc_spec = dict(path=_RC_BASIS, member="b1", strength=1.0)   # the pre-selected gate member
    _RCINJ_SEEDS = tuple(range(8))          # 8 independent mock-noise seeds → 2 sims × 8 = 16 pairs/survey
    _RCINJ_A_SIIII = 0.045                   # representative SiIII (a ±9% in-band ripple; the eBOSS-cert level)
    _RCINJ_NCHAINS = 2                       # per-arm chains (post_mean pooled; 2 → R-hat with minimal cost)
    for _survey, _tag in (("DESI", "D"), ("KS", "K")):   # eBOSS EXCLUDED by design (see fix #3 above)
        for _hs in _hrn:
            _f = _fold_of(_hs)
            if _f is None:
                continue
            _ns = _ns_of_sim(d, _hs, PARAM_LIMITS)
            if int(round(_ns * 1000)) not in (972, 979):     # the two worst-tilt HF-LOSO sims
                continue
            _t = int(round(_ns * 1000))
            # REALISTIC mock + all-nuisances-free fit (fix #1). hr_truth=True → truth = the HR sim's
            # REAL measured P1D (carries its own HCD excess); mf=True → production MF forward; the MF
            # correction is fit EXCLUDING this HR sim (mf_exclude_held via hr_truth). prior_center
            # ="truth" isolates resolution from the LLS-center lever.
            _base = dict(survey=_survey, sim=_hs, mf=True, hr_truth=True, prior_center="truth",
                         sample_metals=True, inject_a_siiii=_RCINJ_A_SIIII, tau0_extreme=True,
                         n_chains=_RCINJ_NCHAINS)
            for _sd in _RCINJ_SEEDS:                          # N independent paired mocks (fix #2)
                # clean & injected SHARE _sd (→ shared base truth + noise; differ only by exp(b1)).
                add_fiducial(f"RCINJ{_tag}_clean{_t}s{_sd}", _f, seed=_sd, **_base)
                add_fiducial(f"RCINJ{_tag}_inj{_t}s{_sd}",   _f, seed=_sd,
                             inject_res_corr=_rc_spec, **_base)

    # === Phase-5a SHAPE-FLOOR validation (2026-06-12): the genuine HF-LOSO worst cases re-run
    # with the shape-aware MF floor (fires on the DESI leg). Compares: the existing DIAGONAL floor
    # (the simpler fix) vs the shape floor at infl∈{1.0,1.5,2.0}. Worst sims = ns0.972 (+2.80σ)
    # and ns0.979 (+1.08σ). HFLOSOSF{ns}_{tag} so the battery analysis groups them. ===
    for _hs in _hrn:
        _f = _fold_of(_hs)
        if _f is None:
            continue
        _ns = _ns_of_sim(d, _hs, PARAM_LIMITS)
        if int(round(_ns * 1000)) not in (972, 979):
            continue
        _base = dict(survey="DESI", sim=_hs, mf=True, hr_truth=True, prior_center="truth")
        _tag = int(round(_ns * 1000))
        add_fiducial(f"HFSFdiag{_tag}", _f, desi_floor=True, **_base)              # diagonal floor on DESI
        add_fiducial(f"HFSF10_{_tag}", _f, mf_shape=1.0, **_base)                  # shape floor infl 1.0
        add_fiducial(f"HFSF15_{_tag}", _f, mf_shape=1.5, **_base)                  # shape floor infl 1.5
        add_fiducial(f"HFSF20_{_tag}", _f, mf_shape=2.0, **_base)                  # shape floor infl 2.0

    # === 2D-TILT HCD-MODEL OVERRIDE (env-gated, 2026-06-17) ============================== #
    # HCD-slope-model SELECTION re-run: the 1D re-centered power-law (the slfix batch) ran the 4
    # per-survey closure mocks D_f3 / K_f4 / XS_f6_s0 / E_f5 with the FIXED/marginalized 1D HCD
    # z-slope. This override re-runs the SAME 4 mocks (same fold/sim/seed/noise → byte-identical
    # mock data) but with the 2D AMPLITUDE×TILT HCD model (hcd_2d_tilt=True + hierarchical_hcd=True),
    # which lets the DATA float the global z-tilt B_HCD (centered on HCD_INCIDENCE_SLOPE[0]=2.465 with
    # FIXED per-class δs_c). The ONLY change vs slfix is the HCD-slope MODEL — so the n_s/A_p recovery
    # diff isolates which slope model recovers truth best. Activated ONLY when STEPA_2DTILT_MOCKS is set
    # (a comma-list of mock_ids, or "default" = the 4 per-survey closures); BYTE-IDENTICAL when unset, so
    # every other battery/golden is untouched. Output dir is selected SEPARATELY via STEPA_CKPT_DIR
    # (=checkpoints/stepA_2dtilt) so the slfix checkpoints are never overwritten.
    _2d_env = os.environ.get("STEPA_2DTILT_MOCKS", "").strip()
    if _2d_env:
        _2d_mocks = ({"D_f3", "K_f4", "XS_f6_s0", "E_f5"} if _2d_env.lower() == "default"
                     else set(m.strip() for m in _2d_env.split(",") if m.strip()))
        _n_flipped = 0
        for c in cfg:
            if c["mock_id"] in _2d_mocks:
                c["hcd_2d_tilt"] = True
                c["hierarchical_hcd"] = True
                c["hcd_ratio_infl"] = 1.0
                _n_flipped += 1
        if verbose:
            print(f"[config] STEPA_2DTILT_MOCKS={_2d_env!r} -> 2D-tilt+hierarchical on "
                  f"{sorted(_2d_mocks)} ({_n_flipped} chains flipped)")

    if verbose:
        print(f"[config] resolved {len(cfg)} chains")
    return cfg


def _resolve_hr_sim(d, fold_sims, PARAM_LIMITS, target_ns=0.979):
    """The HR sim closest to target_ns (from the HR cache) + the LF fold whose held-out pool
    contains the SAME sim name (so make_truth_from_sim can read its LF cache rows for the
    MF-resolution truth). Returns dict(sim, fold, n_s)."""
    from hcd_analysis.emulator import multifidelity as MF
    hr = MF.load_cache(MF.HR_CACHE)
    hrs = np.array([s.decode() if isinstance(s, bytes) else s for s in hr["sim_name"]])
    if "params_unit" in hr:
        pu = hr["params_unit"]
    else:
        from hcd_analysis.emulator.data import normalize_params
        pu = normalize_params(hr["params"])
    lo, hi = PARAM_LIMITS[0]
    cand = []
    for s in sorted(set(hrs)):
        rows = np.where(hrs == s)[0]
        cand.append((float(lo + pu[rows[0], 0] * (hi - lo)), s))
    ns, sim = min(cand, key=lambda t: abs(t[0] - target_ns))
    # which LF fold holds this sim out?
    fold = None
    for f, lst in fold_sims.items():
        if sim in [ss for _, ss in lst]:
            fold = f
            break
    if fold is None:
        # fall back to the fold whose held-out ns is closest (the sim should be in some fold).
        raise ValueError(f"HR sim {sim!r} not found in any LF held-out fold")
    return dict(sim=sim, fold=fold, n_s=ns)


def print_table(cfg):
    print(f"\n=== STEP-A config: {len(cfg)} chains "
          f"({len(set(c['mock_id'] for c in cfg))} mocks) ===")
    hdr = (f"{'id':22s} {'tier':6s} {'fold':>4s} {'sim n_s':>8s} {'mf':>3s} "
           f"{'zslope':>6s} {'hr':>3s} {'τ0ext':>6s} {'cid':>3s}  sim")
    print(hdr)
    print("-" * len(hdr))
    for c in cfg:
        print(f"{c['id']:22s} {c['tier']:6s} {c['fold']:>4d} {c['n_s']:>8.4f} "
              f"{'Y' if c['mf'] else '.':>3s} {'Y' if c['z_slope_marginalized'] else '.':>6s} "
              f"{'Y' if c['hr_truth'] else '.':>3s} "
              f"{'Y' if c['tau0_extreme'] else '.':>6s} {c['chain_id']:>3d}  {c['sim'][:46]}")
    # per-tier counts
    from collections import Counter
    tc = Counter(c['tier'] for c in cfg)
    print(f"\nchain counts by tier: {dict(tc)}  TOTAL={len(cfg)}")


# ----------------------------------------------------------------------------- #
#  Health log — atomic (temp+rename) writes.
# ----------------------------------------------------------------------------- #
def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path, text):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)        # atomic on POSIX same-fs rename


def _load_health():
    if os.path.exists(HEALTH_JSON):
        try:
            with open(HEALTH_JSON) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _render_txt(health):
    lines = [f"# STEP-A health  ({_now()})", ""]
    order = {"FAILED": 0, "running": 1, "queued": 2, "done": 3}
    chains = health.get("chains", {})
    for cid in sorted(chains, key=lambda k: (order.get(chains[k].get("status"), 9), k)):
        c = chains[cid]
        extra = ""
        if c.get("status") == "done":
            extra = (f" n={c.get('n_samples')} div={c.get('divergences')} "
                     f"mtdsat={c.get('treedepth_sat_frac')}")
        if c.get("status") == "FAILED":
            extra = f"  ERR: {str(c.get('error',''))[:120]}"
        lines.append(f"  {c.get('status','?'):8s} {cid:24s}{extra}")
    # fiducial batteries
    fids = health.get("fiducials", {})
    if fids:
        lines += ["", "# fiducial batteries (all-chains-in):"]
        for mid, b in sorted(fids.items()):
            lines.append(f"  {mid:12s} Rhat_max={b.get('rhat_max')} "
                         f"ESSbulk_min={b.get('ess_bulk_min')} "
                         f"ESStail_min={b.get('ess_tail_min')} "
                         f"ns_bias={b.get('bias_ns')} Ap_bias={b.get('bias_Ap')}")
    counts = health.get("counts", {})
    lines += ["", f"# counts: {counts}"]
    return "\n".join(lines) + "\n"


def update_health(updates_per_chain=None, fiducial=None, counts_recompute=True):
    """Merge per-chain status updates (and/or a fiducial battery) into health.json + .txt,
    ATOMICALLY. ``updates_per_chain`` = {chain_id: {field: val, ...}}. NOTE: this is called by
    the PARENT pool process serially (workers communicate via their checkpoint files + a tiny
    status sentinel) so there is a single writer — no cross-process race on health.json."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    h = _load_health()
    h.setdefault("chains", {})
    h.setdefault("fiducials", {})
    h["updated"] = _now()
    if updates_per_chain:
        for cid, fields in updates_per_chain.items():
            h["chains"].setdefault(cid, {})
            h["chains"][cid].update(fields)
    if fiducial:
        mid, battery = fiducial
        h["fiducials"][mid] = battery
    if counts_recompute:
        from collections import Counter
        cc = Counter(c.get("status", "?") for c in h["chains"].values())
        h["counts"] = dict(cc)
    _atomic_write(HEALTH_JSON, json.dumps(h, indent=2, default=_json_default))
    _atomic_write(HEALTH_TXT, _render_txt(h))
    return h


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ----------------------------------------------------------------------------- #
#  RUN ONE CHAIN (the worker entrypoint; isolated subprocess, single-thread).
# ----------------------------------------------------------------------------- #
def run_one_chain(chain, *, n_warmup, n_samples, dense_mass, max_tree_depth, target_accept):
    """Build the chain's fold-matched ctx + run ONE NUTS chain reproducing the exact draw a
    monolithic run_legb_convergence would (same seed derivation), and WRITE the checkpoint.

    Reproduces run_legb_convergence's per-chain seed: key0=PRNGKey(seed);
    k_mock,k_nuts=split(fold_in(key0, mock_index)); chain_key=fold_in(k_nuts, chain_id). The mock
    NOISE key (k_mock) is shared across a fiducial's chains (same mock dataset; chains differ
    only in the NUTS seed) — exactly run_legb_convergence's contract."""
    import jax
    import jax.numpy as jnp
    from numpyro.infer import init_to_sample
    import hcd_analysis.emulator  # noqa: F401
    from hcd_analysis.emulator import closure_legb as C
    from hcd_analysis.emulator.closure_legb import (
        build_legb_ctx, held_out_sims, make_truth_from_sim, make_hr_truth_from_cache, make_legb_mock,
        _mock_core_per_leg, _run_nuts_legb, _draws_matrix, _packed_names_for, _hcd_latent_truths,
        _hcd_latent_truths_2d, CACHE_PATH, ZSLOPE_PRIOR_SIGMA, HCD_INCIDENCE_SLOPE,
        hcd_pivot_wc_and_xbar)
    # NB: HCD_LIT_OVER_SIM_SLOPE is deliberately NOT imported — it is the lit/sim RATIO slope
    # (prior-center at the z=3 PIVOT only), NEVER the forward z-exponent. The forward z-slope is
    # closure_legb.HCD_INCIDENCE_SLOPE (imported above). See hcd-dndx-zslope-bug.
    from hcd_analysis.emulator.inference import (PARAM_NAMES, hcd_incidence_prior,
        hcd_lls_realfit_alpha_center, assert_hcd_pivot_z3, HCD_LLS_SURVEY_BOOST,
        HCD_LLS_SURVEY_FRAC_SIGMA, HCD_PRIOR_FRAC_SIGMA, HCD_Z_PIVOT)

    fold = chain["fold"]
    # MOCK INDEX: run_legb_convergence selects the sim by mock_index OR an explicit sim. We pass
    # sim explicitly, but the SEED stream uses mock_index — so derive a STABLE mock_index from the
    # sim's position in the fold's held-out list (so all of a mock's chains share one noise draw,
    # and distinct mocks get distinct noise). This makes the seed reproducible + per-mock unique.
    sims, _ = held_out_sims(C.load_cache(CACHE_PATH), fold=fold)
    try:
        mock_index = sims.index(chain["sim"])
    except ValueError:
        mock_index = 0

    # fold-matched emulator backbone; the error vector is the production (fold0) C_emu (the ONE
    # matched xclass pair; there is no per-fold error vector — see SESSION_HANDOVER §248).
    # Phase-5a shape-floor validation knobs: mf_shape>0 fires the shape-aware MF floor on the
    # tested leg (the survey of this chain) at that inflation; desi_floor turns the EXISTING
    # diagonal MF floor ON for DESI (the simpler-fix comparison arm).
    _infl = float(chain.get("mf_shape", 0.0) or 0.0)
    _einfl = float(chain.get("mf_emucoh", 0.0) or 0.0)   # 60-sim LF-emulator-coherence C_emu term
    _eoda = bool(chain.get("mf_emucoh_offdiag_only", False))   # per-term diagonal allocation
    _survey = chain.get("survey", "DESI")
    _desi_kw = {"mf_floor_on": True} if chain.get("desi_floor") else None
    ctx, d = build_legb_ctx(
        ckpt=chain["ckpt"], with_mf=bool(chain["mf"]),
        mf_fold=fold, mf_with_floor=bool(chain["mf"]),
        mf_exclude_held=bool(chain.get("hr_truth", False)),    # HF-LOSO: MF fit EXCLUDING this HR sim
        # TRUE leave-ONE-out (Task 1.5): drop EXACTLY this HR sim from the MF head fit, not the
        # whole LF fold group (two HR sims can share a group → leave-TWO-out → an inflated bias).
        # Only the hr_truth (HF-LOSO) arms opt in; None for every other caller (back-compat).
        mf_target_hr_sim=(chain["sim"] if chain.get("hr_truth", False) else None),
        with_eboss=(_survey == "eBOSS"),                       # eBOSS DR14 leg (low-k shakedown)
        sample_metals=bool(chain.get("sample_metals", False)), # shared a_SiIII nuisance (eBOSS/DESI)
        mf_shape=(_infl > 0), mf_shape_infl=(_infl if _infl > 0 else 1.0),
        mf_shape_legs=(_survey,),
        mf_emucoh=(_einfl > 0), mf_emucoh_infl=(_einfl if _einfl > 0 else 1.0),
        mf_emucoh_legs=(_survey,), mf_emucoh_offdiag_only=_eoda, desi_kwargs=_desi_kw,
        # DIAGNOSTIC (decomp): res_corr low-k anchor multiple. Default 5.0 = production anchor
        # (byte-identical for every existing chain); 0.0 = NO anchor (raw clamped res_corr table).
        # The decomp diagnostic sets chain["mf_anchor_mult"]=0.0 for the "minus-anchor" variant.
        mf_anchor_mult=float(chain.get("mf_anchor_mult", 5.0)),
        # HIERARCHICAL HCD prior ("Option B"): A_hcd × {r_subdla, r_dla} reparam (default OFF →
        # byte-identical). The ratio centers are auto-derived from the RAW sim w_c pool (must-fix #1).
        hierarchical_hcd=bool(chain.get("hierarchical_hcd", False)),
        hcd_ratio_infl=float(chain.get("hcd_ratio_infl", 1.0) or 1.0),
        # 2D AMPLITUDE×TILT submanifold (HT arms): A_hcd × global z-tilt B_hcd × FIXED ratios; the
        # per-class slope s_c = B_hcd + δs_c REPLACES the marginalize_zslope sampling. Requires
        # hierarchical_hcd; the closure δs_c / B_hcd center+width auto-derive from the forward slopes.
        hcd_2d_tilt=bool(chain.get("hcd_2d_tilt", False)))

    # DIAGNOSTIC (decomp): FIX the res_corr-amplitude nuisance alpha_res to the no-op (alpha0=1,
    # s=0, NOT sampled) instead of marginalizing it. Default False = production (alpha SAMPLED,
    # byte-identical). The "minus-alpha" decomp variant sets chain["fix_alpha_res"]=True.
    if bool(chain.get("fix_alpha_res", False)):
        ctx = ctx._replace(fix_alpha_res=True)

    # HYPOTHESIS #3 (res_corr COVARIANCE form, PI 2026-06-16): inject a rank-1 FRACTIONAL
    # res_corr-shape C_emu term sigma_res^2·dhat dhat^T (dhat = anchored log res_corr) into the
    # mf_shape_per_leg slot, INSTEAD of marginalizing the res_corr amplitude alpha(z). Used with
    # fix_alpha_res=True so alpha is NOT floated. Default None → byte-identical no-op (the npz is
    # only read when this chain key is set, so every existing arm/golden is untouched). The cov is
    # on the leg's flat (z,k) grid; we assert k/z-alignment to the live leg before binding.
    _rccov = chain.get("res_corr_shape_cov_npz")
    if _rccov:
        _cd = np.load(_rccov, allow_pickle=True)
        _per_leg = {}
        for _leg in ctx.legs:
            _key = f"{_leg.name}_cov"
            if _key not in _cd.files:
                continue                                   # leg not covered by this cov → skip
            _cov = np.asarray(_cd[_key], float)
            _n = np.asarray(_leg.k).shape[0]
            assert _cov.shape == (_n, _n), (
                f"res_corr_shape_cov[{_leg.name}] {_cov.shape} != leg ({_n},{_n})")
            assert np.allclose(np.asarray(_leg.k), np.asarray(_cd[f"{_leg.name}_k"])), \
                f"res_corr_shape_cov[{_leg.name}]: leg k grid mismatch (cov built on a different grid)"
            _per_leg[_leg.name] = jnp.asarray(_cov)
        ctx = ctx._replace(mf_shape_per_leg=_per_leg,
                           mf_shape_infl=float(chain.get("res_corr_shape_cov_infl", 1.0)))

    # CLOSURE/SBC sim-truth z-slope center (PI re-determination 2026-06-17): run_stepA is the
    # CLOSURE driver — its held-out-sim mocks carry the SIM incidence slope HCD_INCIDENCE_SLOPE
    # (2.465,…), NOT the litWLS γ_LLS=2.127 the REAL fit uses. build_legb_ctx(survey=…) now plumbs
    # the litWLS LLS zslope_mu for the REAL fit, so the CLOSURE must RESET zslope_mu back to the
    # sim-truth center here (regardless of z_slope_marginalized) so the forward LLS slope tracks the
    # mock truth. (The litWLS-direction closure arm — a lit-truth mock — is a SEPARATE fiducial; it
    # sets zslope_realfit=True to opt INTO the litWLS center against a lit-drawn truth.)
    _zslope_realfit = bool(chain.get("zslope_realfit", False))
    if not bool(chain.get("hcd_2d_tilt", False)):
        if _zslope_realfit:
            # REAL-FIT-DIRECTION arm (zslope_realfit=True): center the LLS forward z-slope on the
            # litWLS γ_LLS=2.127 (sim subDLA/DLA), the slope the REAL fit uses — to validate the
            # real-fit-direction LLS→n_s leak against this mock. subDLA/DLA stay sim incidence slope.
            from hcd_analysis.emulator.inference import HCD_LLS_REALFIT_ZSLOPE
            _inc = np.asarray(HCD_INCIDENCE_SLOPE, float)
            ctx = ctx._replace(marginalize_zslope=True,
                               zslope_mu=jnp.asarray([HCD_LLS_REALFIT_ZSLOPE, _inc[1], _inc[2]]),
                               zslope_sigma=jnp.asarray(ZSLOPE_PRIOR_SIGMA))
        else:
            # CLOSURE default: CENTER on the SIM incidence-weight slope HCD_INCIDENCE_SLOPE (~2.4, the
            # slope the held-out-sim mock truth's w_c(z) carries — dN/dX(z) RISES with z), NOT the
            # lit/sim RATIO slope HCD_LIT_OVER_SIM_SLOPE (~0.95) NOR the litWLS real-fit slope 2.127.
            # Matches the _zslope_sites None-default + the 2D-tilt anchor. Pinned marginalized or fixed.
            ctx = ctx._replace(zslope_mu=jnp.asarray(HCD_INCIDENCE_SLOPE))
            if chain["z_slope_marginalized"]:
                ctx = ctx._replace(marginalize_zslope=True,
                                   zslope_sigma=jnp.asarray(ZSLOPE_PRIOR_SIGMA))

    # SEPARATE per-survey inference (2026-06-10): keep only this chain's leg(s). A "+"-joined survey
    # (e.g. "DESI+KS", the cosmology-safety arm) keeps BOTH legs for a genuine joint fit.
    _keep = chain.get("survey", "DESI").split("+")
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name in _keep])

    # TRUTH: MF-resolution (gate invariant) when mf is set; LF otherwise. τ₀ anchor = a PRIYA
    # curve (the new 2-param model — closure self-consistency needs it; "becker13" was only
    # 2-param-representable to ~6.6% and is RETIRED). For M2 (tau0_extreme) use the upper PRIYA
    # prior corner (high amplitude+slope) to stress the τ₀×cosmology interaction; else the
    # central PRIYA curve (τ₀=1, dτ₀=0 = Kim), the regime the data visits.
    tau0_anchor = (1.20, 0.20) if chain["tau0_extreme"] else "priya"
    if chain.get("hr_truth"):
        # GENUINE HF-LOSO: truth = the held-out HR sim's REAL measured P1D (no MF correction);
        # the ctx MF correction was fit EXCLUDING this sim (mf_exclude_held above), so the forward
        # (LF emu × held-out MF) vs this real HR truth tests the MF-resolution GENERALIZATION.
        truth_sim = make_hr_truth_from_cache(chain["sim"], ctx.cache_k, tau0_anchor=tau0_anchor)
    else:
        truth_sim = make_truth_from_sim(d, chain["sim"], fold=fold, tau0_anchor=tau0_anchor,
                                        mf=ctx.mf, lls_truth_boost=float(chain.get("lls_truth_boost", 1.0)))

    # PRIOR CENTER (non-circular closure, 2026-06-10): re-center the HCD α prior.
    #   "lit"      = build_legb_ctx default (lit/sim·w_c_med) — the real-fit prior.
    #   "sim_mean" = the sim-population median w_c (lit_over_sim=1) — the NON-circular closure center.
    #   "truth"    = this sim's own w_c (circular reference only).
    pc = chain.get("prior_center", "lit")
    survey = chain.get("survey", "DESI")
    # CENTER-CONSTRUCTION FIX (PI 2026-06-17): build the pivot from the z=3 STRUCTURAL w_c, NOT the
    # all-z median nanmedian(w_c_cache[:,1:]) (=z≈3.6 — the dN/dX low-z overshoot bug). The same
    # cache, restricted to the z=3 pivot rows. (truth_sim["w_c"] is already this sim's z=3 w_c.)
    wc_z3, Xbar_z3 = hcd_pivot_wc_and_xbar(d, z_pivot=HCD_Z_PIVOT)
    if pc == "lit":
        # REAL-FIT prior: per-survey effective-LLS pin (DESI cosmic-avg/tight; KS boosted ~2.5×/broad,
        # arXiv:2509.18271 §4.3.3). subDLA/DLA survey-agnostic. The LLS center is built from the lit
        # dN/dX law DIRECTLY (alt-(b), hcd_lls_realfit_alpha_center ≈0.194×boost) — the same construction
        # build_legb_ctx(survey=…) uses — NOT the sim z=3 w_c·(lit/sim). subDLA/DLA from the z=3 w_c.
        amu, asd = hcd_incidence_prior(jnp.asarray(wc_z3), z=HCD_Z_PIVOT, survey=survey)
        _boost = HCD_LLS_SURVEY_BOOST.get(survey, 1.0)
        amu = amu.at[0].set(hcd_lls_realfit_alpha_center(Xbar_z3, z=HCD_Z_PIVOT, boost=_boost))
        _fl = HCD_LLS_SURVEY_FRAC_SIGMA.get(survey, float(HCD_PRIOR_FRAC_SIGMA[0]))
        asd = asd.at[0].set(_fl * amu[0])
        assert_hcd_pivot_z3(float(np.asarray(amu)[0]), z=HCD_Z_PIVOT,
                            where=f"run_stepA pc=lit survey={survey}", boost=_boost)
        ctx = ctx._replace(alpha_hcd_mu=amu, alpha_hcd_sigma=asd)
    else:
        # NON-circular closure cert: center on the sim population z=3 w_c (sim_mean) or this sim's z=3
        # w_c (truth); lit_over_sim=1 → no literature/survey offset (the cert tests recovery, not the
        # real prior). Uses the z=3 STRUCTURAL w_c (not the all-z median) — the CENTER-construction fix.
        wc_c = (wc_z3 if pc == "sim_mean" else np.asarray(truth_sim["w_c"]))
        amu, asd = hcd_incidence_prior(jnp.asarray(wc_c), z=HCD_Z_PIVOT, lit_over_sim=jnp.ones(3))
        ctx = ctx._replace(alpha_hcd_mu=amu, alpha_hcd_sigma=asd)
    if chain.get("sigma_lls"):     # σ_LLS width-sensitivity arm
        sig = ctx.alpha_hcd_sigma.at[0].set(float(chain["sigma_lls"]) * float(ctx.alpha_hcd_mu[0]))
        ctx = ctx._replace(alpha_hcd_sigma=sig)
    if chain.get("sigma_subdla"):  # σ_subDLA width-sensitivity arm
        sig = ctx.alpha_hcd_sigma.at[1].set(float(chain["sigma_subdla"]) * float(ctx.alpha_hcd_mu[1]))
        ctx = ctx._replace(alpha_hcd_sigma=sig)
    # subDLA prior-CENTER shift (cosmology-safety arm, #9 panel must-do 2026-06-13): deliberately
    # MIS-SPECIFY the subDLA incidence prior center by shift·σ_subDLA and ask whether it drags the
    # cosmology MEAN — the residual risk the panel flagged (closure corr(subDLA pull, n_s bias)=+0.82,
    # harmless only because the prior isn't binding; a wrong prior CENTER on real data could become so).
    _subshift = float(chain.get("subdla_center_shift", 0.0) or 0.0)
    if _subshift != 0.0:
        mu = ctx.alpha_hcd_mu.at[1].add(_subshift * float(ctx.alpha_hcd_sigma[1]))
        ctx = ctx._replace(alpha_hcd_mu=mu)
    # A_HCD prior-CENTER shift (hierarchical-HCD cosmology-safety arm, design must-fix #3 2026-06-13):
    # the ON-branch ANALOG of subdla_center_shift (which is a no-op under hierarchical_hcd, since the
    # subDLA center is DERIVED = A_hcd·r). Under Option B the n_s coupling RELOCATES onto the overall
    # HCD amplitude A_HCD, so THIS is the knob that certifies net cosmology safety: _hcd_sites reads
    # ctx.alpha_hcd_mu[0] for A_hcd, so shifting it by shift·σ_A mis-specifies the whole HCD amplitude
    # (all three α centers scale together via α_c=A_hcd·r_c) and we ask whether n_s/A_p means move.
    _hcdshift = float(chain.get("hcd_center_shift", 0.0) or 0.0)
    if _hcdshift != 0.0:
        mu = ctx.alpha_hcd_mu.at[0].add(_hcdshift * float(ctx.alpha_hcd_sigma[0]))
        ctx = ctx._replace(alpha_hcd_mu=mu)

    key0 = jax.random.PRNGKey(int(chain["seed"]))
    k_mock, k_nuts = jax.random.split(jax.random.fold_in(key0, int(mock_index)), 2)
    # res_corr injection arm (TASK-1.6): inject an OUT-OF-SPAN log-res_corr misspecification into
    # the leg-binned mock TRUTH ONLY (never the forward) so the paired clean-vs-injected gate can
    # test that marginalizing alpha_res protects n_s. The spec is the (path, member, strength) dict
    # the injection-arm config builds; None ⇒ no-op (the default for every non-injection chain).
    mock_legs, truth_pack, info = make_legb_mock(
        ctx, truth_sim, k_mock, inject_a_siiii=float(chain.get("inject_a_siiii", 0.0) or 0.0),
        inject_res_corr=chain.get("inject_res_corr", None))
    core_per_leg = _mock_core_per_leg(ctx, truth_sim)
    kept_global = truth_pack["kept_global_z"]

    chain_key = jax.random.fold_in(k_nuts, int(chain["chain_id"]))
    t0 = time.time()
    samples, n_div, extra = _run_nuts_legb(
        ctx, mock_legs, core_per_leg, n_warmup=n_warmup, n_samples=n_samples,
        seed=chain_key, target_accept=target_accept, dense_mass=dense_mass,
        max_tree_depth=max_tree_depth, init_strategy=init_to_sample, return_extra=True)
    wall = time.time() - t0

    draws = _draws_matrix(samples, kept_global)            # (N, P) — appends A_hcd/r_…/a_SiIII if present
    # the packed column NAMES mirror _draws_matrix EXACTLY: θ9, τ₀(kept), α3, [A_hcd/r_subdla/r_dla
    # when hierarchical], [a_SiIII when sampled] (must-fix #5: index α by name downstream).
    packed_names = list(_packed_names_for(samples, kept_global))
    truth_vec = np.concatenate([
        truth_pack["theta9"], truth_pack["tau0_global"][kept_global], truth_pack["alpha_hcd"]])
    if bool(chain.get("hcd_2d_tilt", False)):              # match the A_hcd/B_hcd/r_subdla/r_dla columns
        truth_vec = np.concatenate([truth_vec, _hcd_latent_truths_2d(truth_pack["alpha_hcd"], ctx)])
    elif bool(chain.get("hierarchical_hcd", False)):       # match the A_hcd/r_subdla/r_dla draw columns
        truth_vec = np.concatenate([truth_vec, _hcd_latent_truths(truth_pack["alpha_hcd"])])
    if bool(chain.get("sample_metals", False)):            # match the a_SiIII column _draws_matrix added
        truth_vec = np.concatenate([truth_vec, [float(chain.get("inject_a_siiii", 0.0) or 0.0)]])

    num_steps = np.asarray(extra["num_steps"])
    mtd_sat = float(np.mean(num_steps >= (2 ** int(max_tree_depth) - 1))) if num_steps.size else float("nan")

    out = dict(
        chain_id=chain["id"], mock_id=chain["mock_id"], tier=chain["tier"], fold=fold,
        sim=chain["sim"], n_s=chain["n_s"], mf=bool(chain["mf"]),
        survey=chain.get("survey", "DESI"), prior_center=chain.get("prior_center", "lit"),
        sigma_lls=(float(chain["sigma_lls"]) if chain.get("sigma_lls") else 0.0),
        sigma_subdla=(float(chain["sigma_subdla"]) if chain.get("sigma_subdla") else 0.0),
        lls_truth_boost=float(chain.get("lls_truth_boost", 1.0)),
        z_slope_marginalized=bool(chain["z_slope_marginalized"]),
        hr_truth=bool(chain["hr_truth"]), tau0_extreme=bool(chain["tau0_extreme"]),
        mf_shape=float(chain.get("mf_shape", 0.0) or 0.0), desi_floor=bool(chain.get("desi_floor", False)),
        mf_emucoh=float(chain.get("mf_emucoh", 0.0) or 0.0),
        mf_emucoh_offdiag_only=bool(chain.get("mf_emucoh_offdiag_only", False)),
        sample_metals=bool(chain.get("sample_metals", False)),
        inject_a_siiii=float(chain.get("inject_a_siiii", 0.0) or 0.0),
        # res_corr injection provenance (TASK-1.6 paired gate): the (path, member, strength) spec
        # this chain injected into the mock TRUTH, or "" for a clean (non-injected) chain. Lets the
        # paired clean-vs-injected analysis identify the two arms straight from the checkpoint.
        inject_res_corr=json.dumps(chain.get("inject_res_corr", None), default=str),
        hierarchical_hcd=bool(chain.get("hierarchical_hcd", False)),
        hcd_ratio_infl=float(chain.get("hcd_ratio_infl", 1.0) or 1.0),
        hcd_2d_tilt=bool(chain.get("hcd_2d_tilt", False)),
        # litWLS real-fit-direction LLS z-slope arm (PI re-determination 2026-06-17): True → the
        # forward LLS z-slope is centered on γ_LLS=2.127 (the real-fit center) vs this mock; the
        # closure default (False) keeps the sim-truth slope 2.465. Lets the analysis flag the arm.
        zslope_realfit=bool(chain.get("zslope_realfit", False)),
        # mock-noise SEED (the PRNGKey root → k_mock). The RCINJ injection gate pairs the clean &
        # injected arms by (survey, sim, fold, seed); distinct seeds = independent noise draws.
        seed=int(chain.get("seed", 0)),
        chain_index=int(chain["chain_id"]), n_chains_target=int(chain["n_chains"]),
        # battery inputs: the per-chain packed draws + the extra fields (energy/num_steps/diverg).
        packed=draws.astype(np.float64), names=np.array(packed_names),
        truth_vec=truth_vec.astype(np.float64), kept_global=kept_global,
        energy=np.asarray(extra["energy"]), num_steps=num_steps,
        diverging=np.asarray(extra["diverging"]),
        n_samples=int(draws.shape[0]), divergences=int(n_div),
        treedepth_sat_frac=mtd_sat, wall_s=float(wall),
        dropped=json.dumps(info["dropped"], default=str))
    ckpt_path = f"{CKPT_DIR}/{chain['id']}.npz"
    os.makedirs(CKPT_DIR, exist_ok=True)
    # ATOMIC write: np.savez_compressed APPENDS ".npz" if the name lacks it, so the temp name
    # MUST already end in ".npz" (else os.replace can't find what numpy actually wrote). Use a
    # ".tmp<pid>.npz" suffix → numpy writes exactly that → rename onto the final path.
    tmp = f"{CKPT_DIR}/{chain['id']}.tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, ckpt_path)
    return dict(n_samples=int(draws.shape[0]), divergences=int(n_div),
                treedepth_sat_frac=mtd_sat, wall_s=float(wall))


# ----------------------------------------------------------------------------- #
#  AGGREGATE a fiducial's battery once all its chains are in.
# ----------------------------------------------------------------------------- #
def aggregate_fiducial(mock_id, cfg):
    """If ALL chains of ``mock_id`` have checkpoints, load them, stack the per-chain draws +
    energy/num_steps, and compute the convergence battery + the n_s/A_p bias-z. Returns a JSON-
    serialisable battery summary dict, or None if not all chains are present yet."""
    import hcd_analysis.emulator  # noqa: F401
    from hcd_analysis.emulator.closure_legb import convergence_battery
    from hcd_analysis.emulator.inference import PARAM_NAMES

    chain_ids = [c["id"] for c in cfg if c["mock_id"] == mock_id]
    paths = [f"{CKPT_DIR}/{cid}.npz" for cid in chain_ids]
    if not all(os.path.exists(p) for p in paths):
        return None
    return _battery_over_chains(chain_ids)


def _battery_over_chains(chain_ids):
    """Compute the convergence battery + n_s/A_p bias-z over an explicit list of chain ids whose
    checkpoints all exist. Shared by aggregate_fiducial (all 4 chains) + the smoke proof (2)."""
    import hcd_analysis.emulator  # noqa: F401
    from hcd_analysis.emulator.closure_legb import convergence_battery
    paths = [f"{CKPT_DIR}/{cid}.npz" for cid in chain_ids]
    if not all(os.path.exists(p) for p in paths):
        return None
    packs, energies, num_steps_all, divs, names, truth_vec, kept = [], [], [], [], None, None, None
    for p in paths:
        z = np.load(p, allow_pickle=True)
        packs.append(np.asarray(z["packed"]))
        energies.append(np.asarray(z["energy"]))
        num_steps_all.append(np.asarray(z["num_steps"]))
        divs.append(int(z["divergences"]))
        names = list(z["names"]) if names is None else names
        truth_vec = np.asarray(z["truth_vec"]) if truth_vec is None else truth_vec
        kept = z["kept_global"] if kept is None else kept
    # truncate to a common draw count (chains should match; guard anyway).
    nmin = min(p.shape[0] for p in packs)
    packed = np.stack([p[:nmin] for p in packs], axis=0)         # (C, N, P)
    mtd = int(PROD["max_tree_depth"])
    battery = convergence_battery(
        packed, names,
        energy=np.stack([e[:nmin] for e in energies]) if all(e.size for e in energies) else None,
        num_steps=np.stack([n[:nmin] for n in num_steps_all]) if all(n.size for n in num_steps_all) else None,
        max_tree_depth=mtd, n_div=int(sum(divs)))
    # n_s / A_p bias-z (truth − pooled-mean)/pooled-sd.
    pooled = packed.reshape(-1, packed.shape[-1])
    def bias_z(pname):
        if pname not in names:
            return None
        j = names.index(pname)
        sd = float(pooled[:, j].std())
        return float((truth_vec[j] - pooled[:, j].mean()) / sd) if sd > 0 else None
    summary = dict(
        n_chains=int(packed.shape[0]), n_draws=int(packed.shape[1]),
        rhat_max=_r4(battery["rhat_max"]), ess_bulk_min=_r1(battery["ess_bulk_min"]),
        ess_tail_min=_r1(battery["ess_tail_min"]), ebfmi_min=_r3(battery["ebfmi_min"]),
        treedepth_sat_frac=_r4(battery["treedepth_sat_frac"]), n_divergent=int(sum(divs)),
        bias_ns=_r3(bias_z("ns")), bias_Ap=_r3(bias_z("Ap")),
        ess_target=ESS_TARGET, completed=_now())
    return summary


def _r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)
def _r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)
def _r1(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 1)


# ----------------------------------------------------------------------------- #
#  POOL dispatcher (parent process; subprocess per chain; semaphore = workers).
# ----------------------------------------------------------------------------- #
def dispatch(cfg, *, workers, nuts_kwargs, smoke=False):
    """Run the chains as a pool of single-thread SUBPROCESSES (one per chain). RESTARTABLE: a
    chain whose checkpoint already exists is SKIPPED. Crash-isolated: a worker that exits non-zero
    is logged FAILED (with its stderr tail) and the pool continues. The PARENT is the SOLE writer
    of health.json (workers only write their own .npz + a stderr capture file)."""
    import subprocess

    os.makedirs(CKPT_DIR, exist_ok=True)
    # seed the health log: queued / done(skip).
    init = {}
    todo = []
    for c in cfg:
        ckpt = f"{CKPT_DIR}/{c['id']}.npz"
        if os.path.exists(ckpt):
            rec = dict(status="done", note="checkpoint exists (skipped)",
                       tier=c["tier"], mock_id=c["mock_id"])
            try:                                  # populate the human log from the existing ckpt
                z = np.load(ckpt, allow_pickle=True)
                rec.update(n_samples=int(z["n_samples"]), divergences=int(z["divergences"]),
                           treedepth_sat_frac=_r4(float(z["treedepth_sat_frac"])),
                           wall_s=round(float(z["wall_s"]), 1))
            except Exception:
                pass
            init[c["id"]] = rec
        else:
            init[c["id"]] = dict(status="queued", tier=c["tier"], mock_id=c["mock_id"])
            todo.append(c)
    update_health(updates_per_chain=init)
    print(f"[dispatch] {len(cfg)} chains total; {len(cfg)-len(todo)} already done (skipped); "
          f"{len(todo)} to run on {workers} workers", flush=True)

    base_env = dict(os.environ)
    base_env.update(_force_single_thread_env())
    nk = nuts_kwargs

    # CPU-AFFINITY pinning: env thread caps alone don't fully tame XLA/LAPACK's internal pool
    # (Cholesky leaks ~1.5 cores during dense-mass warmup), so PIN each worker to ONE core via
    # ``taskset`` — a hard guarantee of 1 chain ≈ 1 core, no oversubscription. We pin to cores
    # [0, workers) and leave the remaining cores as headroom. A small free-slot pool recycles a
    # core when its chain finishes. (If taskset is absent we fall back to env caps only.)
    have_taskset = _which("taskset")
    # PIN to the ALLOWED cpuset: a SLURM cgroup gives a NON-CONTIGUOUS set (e.g.
    # [1,2,5,6,9,10,14,18,...]); range(workers) would taskset out-of-set ids -> "Invalid
    # argument" and FAIL those chains. Use os.sched_getaffinity. (fixed 2026-06-09)
    _allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
        else list(range(os.cpu_count() or workers))
    free_cores = _allowed[:workers]

    def launch(chain, core):
        cmd = []
        if have_taskset and core is not None:
            cmd += ["taskset", "-c", str(core)]
        cmd += [PY, os.path.abspath(__file__), "--run-one", chain["id"],
                "--n-warmup", str(nk["n_warmup"]), "--n-samples", str(nk["n_samples"]),
                "--max-tree-depth", str(nk["max_tree_depth"]),
                "--target-accept", str(nk["target_accept"])]
        if not nk["dense_mass"]:
            cmd.append("--diag-mass")
        if smoke:
            cmd.append("--smoke-cfg")
        logf = open(f"{CKPT_DIR}/{chain['id']}.log", "wb")
        p = subprocess.Popen(cmd, env=base_env, stdout=logf, stderr=subprocess.STDOUT)
        p._logf = logf
        p._chain = chain
        p._core = core
        p._t0 = time.time()
        return p

    running = {}        # pid -> Popen
    queue = list(todo)
    completed_mocks = set()

    def on_finish(p):
        chain = p._chain
        p._logf.close()
        rc = p.returncode
        ckpt = f"{CKPT_DIR}/{chain['id']}.npz"
        if rc == 0 and os.path.exists(ckpt):
            z = np.load(ckpt, allow_pickle=True)
            update_health(updates_per_chain={chain["id"]: dict(
                status="done", tier=chain["tier"], mock_id=chain["mock_id"],
                end=_now(), wall_s=round(float(z["wall_s"]), 1),
                n_samples=int(z["n_samples"]), divergences=int(z["divergences"]),
                treedepth_sat_frac=_r4(float(z["treedepth_sat_frac"])))})
            print(f"[done] {chain['id']} ({float(z['wall_s']):.0f}s "
                  f"n={int(z['n_samples'])} div={int(z['divergences'])})", flush=True)
        else:
            tail = _tail_file(f"{CKPT_DIR}/{chain['id']}.log", 1500)
            update_health(updates_per_chain={chain["id"]: dict(
                status="FAILED", tier=chain["tier"], mock_id=chain["mock_id"],
                end=_now(), returncode=rc, error=tail)})
            print(f"[FAILED] {chain['id']} rc={rc}\n  {tail[-300:]}", flush=True)
        # if this chain's fiducial is now complete, compute + log the battery.
        mid = chain["mock_id"]
        if mid not in completed_mocks:
            summ = aggregate_fiducial(mid, cfg)
            if summ is not None:
                completed_mocks.add(mid)
                update_health(fiducial=(mid, summ))
                print(f"[battery] {mid}: Rhat_max={summ['rhat_max']} "
                      f"ESSbulk_min={summ['ess_bulk_min']} ESStail_min={summ['ess_tail_min']} "
                      f"ns_bias={summ['bias_ns']} Ap_bias={summ['bias_Ap']}", flush=True)

    while queue or running:
        while queue and len(running) < workers:
            c = queue.pop(0)
            core = free_cores.pop(0) if free_cores else None
            p = launch(c, core)
            running[p.pid] = p
            update_health(updates_per_chain={c["id"]: dict(
                status="running", tier=c["tier"], mock_id=c["mock_id"], start=_now(),
                core=core)})
            print(f"[launch] {c['id']} (pid {p.pid}, core {core}); "
                  f"running={len(running)} queued={len(queue)}", flush=True)
        # poll for any finished worker.
        done_pids = []
        for pid, p in list(running.items()):
            if p.poll() is not None:
                done_pids.append(pid)
        for pid in done_pids:
            p = running.pop(pid)
            if getattr(p, "_core", None) is not None:
                free_cores.append(p._core)          # recycle the core slot
            on_finish(p)
        if not done_pids:
            time.sleep(2.0)

    # final battery pass for any fiducial not yet aggregated (e.g. all chains were pre-existing).
    for mid in sorted(set(c["mock_id"] for c in cfg)):
        if mid in completed_mocks:
            continue
        summ = aggregate_fiducial(mid, cfg)
        if summ is not None:
            update_health(fiducial=(mid, summ))
    print("[dispatch] ALL CHAINS DONE.", flush=True)


def _which(name):
    from shutil import which
    return which(name) is not None


def _tail_file(path, n_bytes):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            sz = f.tell()
            f.seek(max(0, sz - n_bytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return "(no log)"


# ----------------------------------------------------------------------------- #
#  CLI
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="STEP-A closure-mock launcher")
    ap.add_argument("--print-table", action="store_true", help="resolve + print the 44-chain table")
    ap.add_argument("--run", action="store_true", help="run the FULL battery (pool of workers)")
    ap.add_argument("--smoke", action="store_true",
                    help="SMOKE: 1 L1b single-chain + 1 fiducial's 2 chains @ ~40/40")
    ap.add_argument("--run-one", type=str, default=None,
                    help="(internal) run ONE chain by id + write its checkpoint")
    ap.add_argument("--workers", type=int, default=14, help="pool size (default 14; 2 cores headroom)")
    # NUTS knobs (the worker reads these; the pool forwards them).
    ap.add_argument("--n-warmup", type=int, default=PROD["n_warmup"])
    ap.add_argument("--n-samples", type=int, default=PROD["n_samples"])
    ap.add_argument("--max-tree-depth", type=int, default=PROD["max_tree_depth"])
    ap.add_argument("--target-accept", type=float, default=PROD["target_accept"])
    ap.add_argument("--diag-mass", action="store_true", help="diagonal NUTS mass (smoke/debug)")
    ap.add_argument("--smoke-cfg", action="store_true",
                    help="(internal) the worker is part of a SMOKE pool (uses the smoke subset)")
    args = ap.parse_args()

    if args.run_one is not None:
        # WORKER: single-thread env MUST be set before the heavy import (done at module import
        # via the wrapper env; re-assert here for direct invocation).
        _force_single_thread_env()
        cfg = _smoke_subset(build_config()) if args.smoke_cfg else build_config()
        chain = next((c for c in cfg if c["id"] == args.run_one), None)
        if chain is None:
            # the smoke subset may not contain it; fall back to the full config.
            chain = next((c for c in build_config() if c["id"] == args.run_one), None)
        if chain is None:
            print(f"[run-one] unknown chain id {args.run_one!r}", file=sys.stderr)
            sys.exit(2)
        try:
            r = run_one_chain(
                chain, n_warmup=args.n_warmup, n_samples=args.n_samples,
                dense_mass=not args.diag_mass, max_tree_depth=args.max_tree_depth,
                target_accept=args.target_accept)
            print(f"[run-one] {args.run_one} OK: {r}")
        except Exception:
            traceback.print_exc()
            sys.exit(1)
        return

    _force_single_thread_env()
    cfg = build_config(verbose=True)

    if args.print_table:
        print_table(cfg)
        return

    nuts_kwargs = dict(n_warmup=args.n_warmup, n_samples=args.n_samples,
                       dense_mass=not args.diag_mass, max_tree_depth=args.max_tree_depth,
                       target_accept=args.target_accept)

    if args.smoke:
        smoke_cfg = _smoke_subset(cfg)
        # smoke NUTS: tiny warmup/samples, diagonal mass + capped tree depth (path proof, fast).
        nuts_kwargs.update(n_warmup=40, n_samples=40, dense_mass=False, max_tree_depth=7)
        print("\n=== STEP-A SMOKE ===")
        print_table(smoke_cfg)
        print(f"\n[smoke] NUTS knobs: {nuts_kwargs}; workers={min(args.workers, len(smoke_cfg))}")
        dispatch(smoke_cfg, workers=min(args.workers, len(smoke_cfg)),
                 nuts_kwargs=nuts_kwargs, smoke=True)
        _print_smoke_report(smoke_cfg)
        return

    if args.run:
        print_table(cfg)
        dispatch(cfg, workers=args.workers, nuts_kwargs=nuts_kwargs)
        return

    ap.print_help()


def _smoke_subset(cfg):
    """1 L1b single-chain + 1 fiducial's first 2 chains (the cheapest end-to-end proof)."""
    l1b = next(c for c in cfg if c["tier"] == "L1b")
    fid = [c for c in cfg if c["mock_id"] == "L1a_fold0"][:2]   # 2 of the 4 chains
    return [l1b] + fid


def _print_smoke_report(smoke_cfg):
    print("\n========== SMOKE RESULT ==========")
    h = _load_health()
    ok_ckpt = []
    for c in smoke_cfg:
        p = f"{CKPT_DIR}/{c['id']}.npz"
        ok_ckpt.append((c["id"], os.path.exists(p)))
    print("checkpoints written:")
    for cid, ok in ok_ckpt:
        if ok:
            z = np.load(f"{CKPT_DIR}/{cid}.npz", allow_pickle=True)
            print(f"  {cid:24s} OK  (n={int(z['n_samples'])} div={int(z['divergences'])} "
                  f"wall={float(z['wall_s']):.0f}s packed={np.asarray(z['packed']).shape})")
        else:
            print(f"  {cid:24s} MISSING")
    print(f"\nhealth.json counts: {h.get('counts')}")
    print(f"health.json valid JSON: {os.path.exists(HEALTH_JSON)}  "
          f"(fiducials logged: {list(h.get('fiducials', {}).keys())})")

    # PROVE the battery path on the 2 smoke L1a chains (the full gate needs all 4; the smoke runs
    # 2, so aggregate them directly here as a path proof — same convergence_battery code).
    l1a = [c for c in smoke_cfg if c["mock_id"] == "L1a_fold0"]
    if len(l1a) >= 2 and all(os.path.exists(f"{CKPT_DIR}/{c['id']}.npz") for c in l1a):
        b = _battery_over_chains([c["id"] for c in l1a])
        if b is not None:
            print(f"\n[battery PROOF on {len(l1a)} smoke chains of L1a_fold0]:")
            print(f"  Rhat_max={b['rhat_max']}  ESSbulk_min={b['ess_bulk_min']}  "
                  f"ESStail_min={b['ess_tail_min']}  ns_bias={b['bias_ns']}σ  "
                  f"Ap_bias={b['bias_Ap']}σ  (smoke depth → values indicative only)")
    print("\nTo launch the FULL run (14 workers) in the BACKGROUND:")
    print(f"  bash {REPO}/scripts/run_stepA_bg.sh")
    print(f"  (or: nohup env PYTHONNOUSERSITE=1 PYTHONPATH={REPO} JAX_PLATFORMS=cpu "
          f"CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 {PY} "
          f"{REPO}/scripts/run_stepA.py --run --workers 14 "
          f"> {CKPT_DIR}/run_stepA.out 2>&1 &)")


if __name__ == "__main__":
    main()
