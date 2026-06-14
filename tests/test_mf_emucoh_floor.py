"""60-sim LF-emulator-coherence C_emu term (Phase-5a, 2026-06-12) + coexistence with the
6-HR resolution shape floor.

Covers: load_mf_emucoh PSD; the z-support guard (leg rows above the table's z_max zero out, and
the shape floor stays byte-identical because its z grid covers all leg z); the on-leg term adds
a PD off-diagonal; θ-independence (∂C/∂θ ≡ 0); back-compat byte-exact when off; and the
SHAPE+EMUCOH COEXISTENCE (both PSD terms summed, conservative top-up over both, PD).
"""
import os
import hcd_analysis.emulator  # noqa: F401  x64
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator.model import Emulator
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator.likelihood import gaussian_loglik

EMUCOH_NPZ = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/mf_cemu_emucoh.npz"
SHAPE_NPZ = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/mf_cemu_shape.npz"
DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"
KS_BASE = "/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/"
_have_e = os.path.exists(EMUCOH_NPZ)
_have_s = os.path.exists(SHAPE_NPZ)
_have_desi = os.path.exists(DESI_NPZ)
_have_ks = os.path.exists(KS_BASE + "final-conservative-p1d-karacayli_etal2021.txt")


def _emu_ctx(n_k=172, seed=0):
    rng = np.random.default_rng(seed)
    model = Emulator(in_dim=10, n_k=n_k, n_basis=12, key=jax.random.PRNGKey(seed))
    pf = {"mu_marg": jnp.asarray(rng.normal(-2, 0.5, (4, n_k))),
          "sig_marg": jnp.asarray(rng.uniform(0.5, 1.5, (4, n_k))),
          "sig_cosmo": jnp.asarray(rng.uniform(0.02, 0.1, (4, n_k)))}
    return dict(model=model, pf=pf, dla_core=jnp.asarray(rng.uniform(0, 0.5, n_k)),
                cache_k=jnp.asarray(np.linspace(4.04e-4, 0.0694, n_k)),
                alpha_hcd=jnp.asarray([0.06, 0.02, 0.003]), theta9=jnp.full(9, 0.5))


@pytest.mark.skipif(not _have_e, reason="mf_cemu_emucoh.npz not built")
def test_load_mf_emucoh_symmetric_psd():
    e = DL.load_mf_emucoh(EMUCOH_NPZ)
    F = e.f_shape
    assert F.shape == (len(e.z) * len(e.k),) * 2
    assert np.allclose(F, F.T, atol=1e-12)
    assert np.linalg.eigvalsh(F).min() > -1e-10
    assert e.n_sim >= 50
    assert float(e.z.max()) <= 4.61   # full leg-z table (z up to ~4.4; built over all z)


@pytest.mark.skipif(not (_have_e and _have_ks), reason="emucoh npz / KS data missing")
def test_emucoh_binder_zeros_outside_z_support_and_psd():
    # KS spans z up to 4.6; the emucoh table reaches z≈4.4, so KS's z=4.6 bin exercises the
    # z-support guard (the table now covers all DESI z, so DESI no longer has out-of-support rows).
    e = DL.load_mf_emucoh(EMUCOH_NPZ)
    leg = DL.load_ks_leg()
    C = DL.mf_shape_cov_for_leg(e, leg)
    N = leg.k.shape[0]
    assert C.shape == (N, N) and np.allclose(C, C.T, atol=1e-12)
    assert np.linalg.eigvalsh(C).min() > -1e-10
    # rows whose leg-z is ABOVE the table's z_max (~4.4) must be exactly zero (no extrapolation)
    z_row = np.asarray(leg.z)[np.asarray(leg.z_idx)]
    hi = z_row > float(e.z.max()) + 0.1
    assert hi.any(), "KS should have z>4.4 rows (the 4.6 bin) to exercise the guard"
    assert np.allclose(C[hi], 0.0), "emucoh leaked onto z>z_max rows (z-support guard failed)"
    # in-support rows DO get a nonzero diagonal
    lo = z_row <= float(e.z.max()) + 1e-6
    assert np.any(np.diag(C)[lo] > 0)


@pytest.mark.skipif(not (_have_s and _have_desi), reason="shape npz / DESI data missing")
def test_shape_floor_binder_unchanged_by_z_guard():
    # the resolution shape floor (z≤4.6) covers all DESI leg z (≤4.2) → the z_tol=0.1 guard
    # must be a NO-OP for it (byte-identical to no guard). (Out-of-k-band rows legitimately have
    # a zero diagonal — that's the pre-existing k-band behavior, unrelated to the z-guard.)
    s = DL.load_mf_shape(SHAPE_NPZ)
    leg = DL.load_desi_leg()
    z_row = np.asarray(leg.z)[np.asarray(leg.z_idx)]
    assert np.all(z_row <= float(s.z.max()) + 0.1)
    C_guard = DL.mf_shape_cov_for_leg(s, leg, z_tol=0.1)
    C_noguard = DL.mf_shape_cov_for_leg(s, leg, z_tol=1e9)
    assert np.array_equal(C_guard, C_noguard)   # z-guard changes nothing for the shape floor
    assert np.all(np.isfinite(C_guard))


@pytest.mark.skipif(not (_have_e and _have_desi), reason="emucoh npz / DESI data missing")
def test_emucoh_offdiag_pd_and_backcompat():
    c = _emu_ctx()
    e = DL.load_mf_emucoh(EMUCOH_NPZ)
    leg = DL.load_desi_leg()
    Cf = DL.mf_shape_cov_for_leg(e, leg)
    tau0 = jnp.full(leg.n_z, 0.9)
    kw = dict(pf_stats=c["pf"], dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg)
    P0, C0 = DL.predict_P_obs_on_leg(c["model"], c["theta9"], tau0, c["alpha_hcd"], **kw)
    P1, C1 = DL.predict_P_obs_on_leg(c["model"], c["theta9"], tau0, c["alpha_hcd"],
                                     mf_emucoh_cov=Cf, mf_emucoh_infl=1.0, **kw)
    assert np.allclose(np.asarray(P0), np.asarray(P1))                 # covariance-only change
    D = np.asarray(C1) - np.asarray(C0)
    assert np.max(np.abs(D - np.diag(np.diag(D)))) > 0                 # nonzero off-diagonal added
    assert np.linalg.eigvalsh(np.asarray(C1)).min() > 0               # PD
    assert np.all(np.diag(np.asarray(C1)) >= np.diag(np.asarray(C0)) - 1e-12)


@pytest.mark.skipif(not (_have_e and _have_desi), reason="emucoh npz / DESI data missing")
def test_emucoh_offdiag_only_per_term_diagonal_allocation():
    """Per-term diagonal allocation (cosmology referee 2026-06-12): mf_emucoh_offdiag_only=True
    absorbs emucoh's diagonal into emu_var (via max) and adds ONLY its off-diagonal — the OFF-diagonal
    coherent structure is preserved, the diagonal is NOT added on top (less conservative), and PD holds.
    Default (False) is byte-identical to the on-top behavior."""
    c = _emu_ctx()
    e = DL.load_mf_emucoh(EMUCOH_NPZ)
    leg = DL.load_desi_leg()
    Cf = DL.mf_shape_cov_for_leg(e, leg)
    tau0 = jnp.full(leg.n_z, 0.9)
    # a nonzero diagonal C_emu (emu_var>0) is required for the per-term allocation to have any effect
    # (with emu_var=0, max(0, emucoh_diag)=emucoh_diag = the on-top value — correctly no reduction).
    n_k = np.asarray(c["cache_k"]).shape[0]
    rng = np.random.default_rng(1)
    sig = jnp.asarray(rng.uniform(0.02, 0.06, (leg.n_z, 4, n_k, 4)))   # (n_z, n_class, n_k, n_tb)
    ac = jnp.asarray([0.66, 0.83, 1.15, 1.33])
    kw = dict(pf_stats=c["pf"], dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg,
              sigma_zb=sig, alpha_centres=ac, mf_emucoh_cov=Cf, mf_emucoh_infl=1.0)
    _, C_full = DL.predict_P_obs_on_leg(c["model"], c["theta9"], tau0, c["alpha_hcd"], **kw)
    _, C_oda = DL.predict_P_obs_on_leg(c["model"], c["theta9"], tau0, c["alpha_hcd"],
                                       mf_emucoh_offdiag_only=True, **kw)
    _, C_def = DL.predict_P_obs_on_leg(c["model"], c["theta9"], tau0, c["alpha_hcd"],
                                       mf_emucoh_offdiag_only=False, **kw)
    Cfull, Coda, Cdef = np.asarray(C_full), np.asarray(C_oda), np.asarray(C_def)
    # explicit False == default (byte-identical)
    assert np.array_equal(Cfull, Cdef)
    # off-diagonal coherent structure IDENTICAL (only the diagonal allocation differs)
    offdiff = (Coda - np.diag(np.diag(Coda))) - (Cfull - np.diag(np.diag(Cfull)))
    assert np.max(np.abs(offdiff)) < 1e-10
    # diagonal NOT added on top → reduced (never larger), and strictly smaller somewhere on DESI
    # (emucoh diag absorbed into emu_var rather than summed)
    assert np.all(np.diag(Coda) <= np.diag(Cfull) + 1e-12)
    assert np.min(np.diag(Coda) - np.diag(Cfull)) < -1e-12
    # still PD
    assert np.linalg.eigvalsh(Coda).min() > 0


@pytest.mark.skipif(not (_have_e and _have_desi), reason="emucoh npz / DESI data missing")
def test_emucoh_cov_theta_independent():
    c = _emu_ctx()
    e = DL.load_mf_emucoh(EMUCOH_NPZ)
    leg = DL.load_desi_leg()
    Cf = jnp.asarray(DL.mf_shape_cov_for_leg(e, leg))
    tau0 = jnp.full(leg.n_z, 0.9)

    def trace_emucoh(th):
        _, C = DL.predict_P_obs_on_leg(c["model"], th, tau0, c["alpha_hcd"], pf_stats=c["pf"],
                                       dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg,
                                       mf_emucoh_cov=Cf, mf_emucoh_infl=1.5)
        _, C0 = DL.predict_P_obs_on_leg(c["model"], th, tau0, c["alpha_hcd"], pf_stats=c["pf"],
                                        dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg)
        return jnp.trace(C - C0)
    g = jax.grad(trace_emucoh)(c["theta9"])
    assert np.max(np.abs(np.asarray(g))) == 0.0, "C_emucoh must be θ-independent (fixed P_data)"


@pytest.mark.skipif(not (_have_e and _have_s and _have_desi), reason="data missing")
def test_shape_and_emucoh_coexist_pd_and_topup():
    """The key coexistence test (closes the shape-floor §6 open gap): both PSD terms summed,
    conservative top-up over both, PD, both off-diagonals present, grad finite."""
    c = _emu_ctx()
    s = DL.load_mf_shape(SHAPE_NPZ); e = DL.load_mf_emucoh(EMUCOH_NPZ)
    leg = DL.load_desi_leg()
    Cs = jnp.asarray(DL.mf_shape_cov_for_leg(s, leg))
    Ce = jnp.asarray(DL.mf_shape_cov_for_leg(e, leg))
    tau0 = jnp.full(leg.n_z, 0.9)
    P_data = jnp.asarray(leg.P_data)
    base = dict(pf_stats=c["pf"], dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg)
    _, C_no = DL.predict_P_obs_on_leg(c["model"], c["theta9"], tau0, c["alpha_hcd"], **base)
    _, C_s = DL.predict_P_obs_on_leg(c["model"], c["theta9"], tau0, c["alpha_hcd"],
                                     mf_shape_cov=Cs, **base)
    _, C_e = DL.predict_P_obs_on_leg(c["model"], c["theta9"], tau0, c["alpha_hcd"],
                                     mf_emucoh_cov=Ce, **base)
    _, C_both = DL.predict_P_obs_on_leg(c["model"], c["theta9"], tau0, c["alpha_hcd"],
                                        mf_shape_cov=Cs, mf_emucoh_cov=Ce, **base)
    Cb = np.asarray(C_both)
    assert np.linalg.eigvalsh(Cb).min() > 0                            # PD with both on
    # both off-diagonals present: the combined off-diagonal ≈ shape + emucoh contributions
    off = lambda C: np.asarray(C) - np.diag(np.diag(np.asarray(C)))
    d_both, d_s, d_e = off(C_both) - off(C_no), off(C_s) - off(C_no), off(C_e) - off(C_no)
    assert np.allclose(d_both, d_s + d_e, atol=1e-8), "combined off-diagonal ≠ shape + emucoh"
    # diagonal never below any single term (conservative top-up over both)
    diag_both = np.diag(Cb)
    assert np.all(diag_both >= np.diag(np.asarray(C_s)) - 1e-9)
    assert np.all(diag_both >= np.diag(np.asarray(C_e)) - 1e-9)
    # loglik finite + differentiable through both terms
    def f(th):
        P, C = DL.predict_P_obs_on_leg(c["model"], th, tau0, c["alpha_hcd"],
                                       mf_shape_cov=Cs, mf_emucoh_cov=Ce, **base)
        return gaussian_loglik(P_data - P, C)
    assert np.isfinite(float(f(c["theta9"])))
    assert np.all(np.isfinite(np.asarray(jax.grad(f)(c["theta9"]))))


@pytest.mark.skipif(not (_have_e and _have_s and _have_desi), reason="data missing")
def test_coexist_with_populated_cemu_diagonal():
    """Production-path regression lock (CS step-review gap): exercise the assembly with a NONZERO
    diagonal C_emu (sigma_zb populated) AND both shape+emucoh terms on — so emu_var_flat and the
    top-up are both live, not the C_emu=0 path the other coexistence tests use."""
    c = _emu_ctx()
    rng = np.random.default_rng(1)
    s = DL.load_mf_shape(SHAPE_NPZ); e = DL.load_mf_emucoh(EMUCOH_NPZ)
    leg = DL.load_desi_leg()
    Cs = jnp.asarray(DL.mf_shape_cov_for_leg(s, leg))
    Ce = jnp.asarray(DL.mf_shape_cov_for_leg(e, leg))
    tau0 = jnp.full(leg.n_z, 0.9)
    sigma_zb = jnp.asarray(rng.uniform(0.01, 0.05, (leg.n_z, 4, c["cache_k"].shape[0], 4)))
    alpha_centres = jnp.asarray([0.66, 0.83, 1.15, 1.33])
    base = dict(pf_stats=c["pf"], dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg,
                sigma_zb=sigma_zb, alpha_centres=alpha_centres)
    _, C_no = DL.predict_P_obs_on_leg(c["model"], c["theta9"], tau0, c["alpha_hcd"], **base)
    _, C_both = DL.predict_P_obs_on_leg(c["model"], c["theta9"], tau0, c["alpha_hcd"],
                                        mf_shape_cov=Cs, mf_emucoh_cov=Ce, **base)
    Cb = np.asarray(C_both)
    assert np.linalg.eigvalsh(Cb).min() > 0                       # PD with nonzero C_emu + both terms
    # the diagonal with both terms is never below the C_emu-only diagonal (conservative, no shrink)
    assert np.all(np.diag(Cb) >= np.diag(np.asarray(C_no)) - 1e-12)
    # finite + differentiable through the full production assembly
    def f(th):
        P, C = DL.predict_P_obs_on_leg(c["model"], th, tau0, c["alpha_hcd"],
                                       mf_shape_cov=Cs, mf_emucoh_cov=Ce, **base)
        return gaussian_loglik(jnp.asarray(leg.P_data) - P, C)
    assert np.isfinite(float(f(c["theta9"])))
    assert np.all(np.isfinite(np.asarray(jax.grad(f)(c["theta9"]))))


@pytest.mark.skipif(not (_have_e and _have_desi), reason="emucoh npz / DESI data missing")
def test_emucoh_off_is_byte_identical():
    c = _emu_ctx()
    leg = DL.load_desi_leg()
    tau0 = jnp.full(leg.n_z, 0.9)
    base = dict(pf_stats=c["pf"], dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg)
    _, C0 = DL.predict_P_obs_on_leg(c["model"], c["theta9"], tau0, c["alpha_hcd"], **base)
    _, C1 = DL.predict_P_obs_on_leg(c["model"], c["theta9"], tau0, c["alpha_hcd"],
                                    mf_emucoh_cov=None, mf_emucoh_infl=1.0, **base)
    assert np.array_equal(np.asarray(C0), np.asarray(C1))   # byte-identical when off
