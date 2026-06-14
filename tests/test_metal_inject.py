"""Phase-4d test-3: the mock metal-injection helper (closure_legb.metal_inject).

Pins the two injection forms against the survey formulae: eBOSS/McDonald SiIIIcorr (exact) and the
DESI-full model carrying an ADDITIVE SiII-SiII term the multiplicative _metal_factor cannot match.
"""
import numpy as np
import hcd_analysis.emulator  # x64 BEFORE jax
from hcd_analysis.emulator import closure_legb as C
from hcd_analysis.emulator import data_likelihood as DL

K = np.array([1e-3, 5e-3, 0.01, 0.02, 0.04])
P = np.ones_like(K) * 10.0          # toy flat P1D
FBAR = 0.7                           # mean flux -> 1-<F>=0.3


def test_zero_amplitude_is_identity():
    for form in ("desi_full", "eboss"):
        out = C.metal_inject(P, K, FBAR, form=form, f_SiIII=0.0, f_SiII=0.0, f_SiII_SiII=0.0)
        np.testing.assert_allclose(out, P, rtol=0, atol=0, err_msg=f"{form} a=0 not identity")


def test_eboss_matches_mcdonald():
    f = 0.009
    out = C.metal_inject(P, K, FBAR, form="eboss", f_SiIII=f)
    aa = f / (1.0 - FBAR)
    ref = P * (1.0 + aa ** 2 + 2.0 * aa * np.cos(2271.0 * K))
    np.testing.assert_allclose(out, ref, rtol=1e-12, err_msg="eboss != McDonald SiIIIcorr")


def test_desi_full_has_additive_siII_term():
    # With ONLY the additive SiII-SiII amplitude on (SiIII=SiII cross = 0), the multiplicative
    # _metal_factor (which has no SiII-SiII term) would give P exactly; metal_inject must NOT.
    out = C.metal_inject(P, K, FBAR, form="desi_full", f_SiIII=0.0, f_SiII=0.0, f_SiII_SiII=0.003)
    assert not np.allclose(out, P), "additive SiII-SiII term absent (the un-fittable piece)"
    # it is a small (sub-10%), Gaussian-damped (decreasing with k) perturbation
    frac = out / P - 1.0
    assert np.all(np.abs(frac) < 0.1), "additive term implausibly large"
    assert abs(frac[0]) > abs(frac[-1]), "additive term should be Gaussian-damped at high k"


def test_desi_full_amplitude_scales_with_mean_flux():
    # A_SiIII = f/(1-<F>): lower mean flux (higher 1-<F>) -> SMALLER effective amplitude.
    hi = C.metal_inject(P, K, 0.5, form="desi_full", f_SiIII=0.01, f_SiII=0.0, f_SiII_SiII=0.0)  # 1-F=0.5
    lo = C.metal_inject(P, K, 0.8, form="desi_full", f_SiIII=0.01, f_SiII=0.0, f_SiII_SiII=0.0)  # 1-F=0.2
    # at the SiIII oscillation peak the lower-<F> (1-F=0.2 -> A larger) case deviates MORE
    assert np.max(np.abs(lo / P - 1.0)) > np.max(np.abs(hi / P - 1.0))


if __name__ == "__main__":
    test_zero_amplitude_is_identity(); test_eboss_matches_mcdonald()
    test_desi_full_has_additive_siII_term(); test_desi_full_amplitude_scales_with_mean_flux()
    print("[metal_inject] all forms OK (eBOSS=McDonald exact; DESI-full additive term present).")
