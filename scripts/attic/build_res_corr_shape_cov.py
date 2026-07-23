"""BUILD: the res_corr-shape COVARIANCE npz for Hypothesis #3 (PI, 2026-06-16).

=== ATTIC (2026-07-21 freeze triage, PI-approved). NOT IN THE FROZEN FORWARD. ===
NORC retires the physics this term existed for: PROD_RES_CORR_ON=False
(closure_legb.py:566) drops res_corr entirely and prod_norc_forward() pins
fix_alpha_res=True (closure_legb.py:569-577), so there is no floated alpha_res left whose
uncertainty this covariance could stand in for.
Verified NOT reachable: the output npz has exactly ONE reader, run_stepA.py:904, gated on
the chain key "res_corr_shape_cov_npz" -- and that key is SET NOWHERE in the repo. The real
fit never passes mf_shape at all (run_real_fit.py). Independently, the shipped npz is now
physically incompatible with the deployed config: its KS grid runs to k=0.0627 while NORC
auto-caps KS at k<=0.045 (closure_legb.py:838), so it would trip the run_stepA shape assert
rather than enter a run silently.
Attic'd rather than deleted because run_stepA.py:898-920 keeps a COMMITTED reader that
documents this artifact's schema; deleting the builder would orphan it. DELETE is defensible
only if that reader is tombstoned at the same time.

CONSTANTS AUDIT (2026-07-21, for the freeze package): no un-provenanced science choice.
  sigma_res=1.0  <- matches SIGMA_A0=1.0, the alpha_res amplitude prior width
                    (closure_legb.py:95).
  LEGS=("DESI","KS")  <- matches the build_legb_ctx(mf_shape_legs=...) default
                         (closure_legb.py:771). NOTE: eBOSS is excluded even though the
                         injection basis carries eBOSS_v1_span. Undeclared scope omission,
                         recorded here.
  dhat = {leg}_v1_span <- committed builder scripts/build_res_corr_injection_basis.py:379.
  SCOPE OMISSION: only the AMPLITUDE direction is covered. The alpha_res z-slope
  (s ~ Normal(0, SIGMA_S=0.5), closure_legb.py:96) is absent from this rank-1 cov.

WHY
---
Hypothesis #3 tests whether the MF n_s cert bias comes from the res_corr AMPLITUDE
nuisance alpha(z) being FLOATED (sampled, free to drag n_s via the ~0.65 alignment
between the anchored log-res_corr shape dhat and the n_s k-response) rather than from
the res_corr table mismatch itself.

The test: instead of MARGINALIZING alpha(z), FIX alpha=1 (fix_alpha_res=True, the
no-op forward) and fold res_corr's uncertainty into the COVARIANCE as a rank-1
FRACTIONAL C_emu term

    C_frac_leg = sigma_res^2 * dhat_leg (dhat_leg)^T          (sigma_res = 1.0)

where dhat_leg = the ANCHORED log res_corr on the leg's flat (z,k) grid = exactly the
amplitude direction the marginalized alpha rescales (v1 = deltahat in the injection
basis). With alpha ~ N(1, 1) marginalized, the linearized forward perturbation is
(alpha-1)*dhat with second moment 1.0 * dhat dhat^T, so sigma_res=1.0 is a CONSERVATIVE
UPPER BOUND on the marginalization's implied covariance. (Corrected 2026-07-21: this
previously read "MATCHES ... EXACTLY". The deployed prior is TruncatedNormal(loc=1.0,
scale=1.0, low=0.0) (closure_legb.py:90), truncated at 1 sigma below the mean, so its true
implied variance is < 1.0. sigma_res=1.0 over-sizes the term; the direction is safe.)
The ONLY other difference from marginalization
is that alpha no longer FLOATS (it is integrated analytically in the Gaussian
likelihood, which cannot drift the MAP, vs. sampled, which can).

This is a FRACTIONAL second moment: dhat is log-res_corr (~fractional), and the
data_likelihood shape-term assembly scales each mf_shape cov by the FIXED leg data
power P_data (x) P_data (the theta-independent fiducial amplitude, the same fractional
.P (x) P scaling as the diagonal C_emu floor). So C_frac_leg binds straight into the
mf_shape_per_leg slot.

REUSE: dhat_leg = {leg}_v1_span from the injection basis (golden-verified, the same
anchored interp_res_corr the marginalized alpha multiplies). Row order is the leg's
flat z-major grid (verified k/z-aligned to a live build_legb_ctx leg, both DESI & KS).

OUT: hcd_analysis/_emulator_data/res_corr_shape_cov.npz
  {leg}_cov   (N,N) fractional rank-1 cov sigma_res^2 dhat dhat^T  (PSD by construction)
  {leg}_dhat  (N,)  the dhat used (= v1_span)
  {leg}_k, {leg}_z_row  the leg grid (for the run-time alignment assertion)
  _meta_sigma_res, _meta_legs, _meta_doc

Env:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
    CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3 \
    scripts/build_res_corr_shape_cov.py
"""
from __future__ import annotations
import functools
print = functools.partial(print, flush=True)

import numpy as np

BASIS = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/res_corr_injection_basis.npz"
OUT = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/res_corr_shape_cov.npz"
SIGMA_RES = 1.0          # MATCH the marginalization's implied cov (alpha ~ N(1,1) => 1.0 dhat dhat^T)
LEGS = ("DESI", "KS")


def main():
    basis = np.load(BASIS, allow_pickle=True)
    out = {}
    for name in LEGS:
        dhat = np.asarray(basis[f"{name}_v1_span"], float)      # anchored log res_corr (= v1)
        N = dhat.shape[0]
        cov = (SIGMA_RES ** 2) * np.outer(dhat, dhat)           # (N,N) rank-1 fractional, PSD
        # sanity: rank-1 PSD, symmetric, leading eigenvalue = sigma_res^2 |dhat|^2
        assert cov.shape == (N, N)
        assert np.allclose(cov, cov.T)
        ev = np.linalg.eigvalsh(cov)
        assert ev[0] > -1e-12 * max(ev[-1], 1e-30), f"{name}: cov not PSD (min eig {ev[0]:.2e})"
        lead = (SIGMA_RES ** 2) * float(dhat @ dhat)
        out[f"{name}_cov"] = cov
        out[f"{name}_dhat"] = dhat
        out[f"{name}_k"] = np.asarray(basis[f"{name}_k"], float)
        out[f"{name}_z_row"] = np.asarray(basis[f"{name}_z_row"], float)
        print(f"[{name}] N={N}  |dhat|_max={np.max(np.abs(dhat)):.4f}  "
              f"lead-eig={lead:.4e}  (frac trace = {np.trace(cov):.4e})  rank-1 PSD OK")

    out["_meta_sigma_res"] = float(SIGMA_RES)
    out["_meta_legs"] = np.array(list(LEGS))
    out["_meta_doc"] = np.array(
        "res_corr-shape COVARIANCE (Hypothesis #3). {leg}_cov = sigma_res^2 dhat dhat^T, a "
        "rank-1 FRACTIONAL PSD C_emu term on the leg flat (z,k) grid, dhat = anchored log "
        "res_corr (= injection-basis v1_span = the direction marginalized alpha rescales). "
        "sigma_res=1.0 matches alpha~N(1,1)'s implied cov. Binds via mf_shape_per_leg "
        "(scaled by P_data (x) P_data). Use with fix_alpha_res=True (alpha NOT floated)."
    )
    np.savez(OUT, **out)
    print(f"\nsaved -> {OUT}  (sigma_res={SIGMA_RES})")


if __name__ == "__main__":
    main()
