"""Shape-aware MF C_emu floor (Phase-5a, 2026-06-12).

Covers:
  * load_mf_shape: symmetric, PSD fractional second-moment.
  * _logk_interp_weights: partition-of-unity in-band, zero out-of-band.
  * mf_shape_cov_for_leg: (N,N) symmetric PSD; diagonal == interpolated f_shape diagonal.
  * predict_P_obs_on_leg with mf_shape_cov: adds a nonzero OFF-diagonal; C_total stays PD;
    mf_shape_cov=None is byte-identical (back-compat); C_shape scales as infl²; the loglik
    is finite + differentiable in θ through the shape covariance (the JAX-safety check).
"""
import os
import hcd_analysis.emulator  # noqa: F401  enables jax_enable_x64
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator.model import Emulator
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator.likelihood import gaussian_loglik

SHAPE_NPZ = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/mf_cemu_shape.npz"
DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"
_have_shape = os.path.exists(SHAPE_NPZ)
_have_desi = os.path.exists(DESI_NPZ)


def _emu_ctx(n_k=172, n_basis=12, seed=0):
    rng = np.random.default_rng(seed)
    model = Emulator(in_dim=10, n_k=n_k, n_basis=n_basis, key=jax.random.PRNGKey(seed))
    pf = {"mu_marg": jnp.asarray(rng.normal(-2, 0.5, (4, n_k))),
          "sig_marg": jnp.asarray(rng.uniform(0.5, 1.5, (4, n_k))),
          "sig_cosmo": jnp.asarray(rng.uniform(0.02, 0.1, (4, n_k)))}
    dla_core = jnp.asarray(rng.uniform(0.0, 0.5, n_k))
    cache_k = jnp.asarray(np.linspace(4.04e-4, 0.0694, n_k))
    return dict(model=model, pf=pf, dla_core=dla_core, cache_k=cache_k,
                alpha_hcd=jnp.asarray([0.06, 0.02, 0.003]), theta9=jnp.full(9, 0.5))


# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have_shape, reason="mf_cemu_shape.npz not built")
def test_load_mf_shape_symmetric_psd():
    sh = DL.load_mf_shape(SHAPE_NPZ)
    F = sh.f_shape
    assert F.shape == (len(sh.z) * len(sh.k),) * 2
    assert np.allclose(F, F.T, atol=1e-12)
    w = np.linalg.eigvalsh(F)
    assert w.min() > -1e-12, f"f_shape not PSD: min eig {w.min():.2e}"
    assert sh.n_sim >= 5


def test_logk_interp_weights_partition_and_support():
    kc = np.power(10.0, np.linspace(np.log10(0.01), np.log10(0.069), 40))
    # in-band midpoint: weights sum to 1, two nonzero entries that bracket it
    w = DL._logk_interp_weights(0.03, kc)
    assert abs(w.sum() - 1.0) < 1e-12
    assert np.count_nonzero(w) == 2
    recon = 10 ** (w @ np.log10(kc))
    assert abs(recon - 0.03) / 0.03 < 1e-9
    # out of band → all zero (no extrapolation)
    assert np.all(DL._logk_interp_weights(0.20, kc) == 0)
    assert np.all(DL._logk_interp_weights(0.001, kc) == 0)


@pytest.mark.skipif(not (_have_shape and _have_desi), reason="shape npz / DESI data missing")
def test_mf_shape_cov_for_leg_psd_and_diagonal():
    sh = DL.load_mf_shape(SHAPE_NPZ)
    leg = DL.load_desi_leg()
    C = DL.mf_shape_cov_for_leg(sh, leg)
    N = leg.k.shape[0]
    assert C.shape == (N, N)
    assert np.allclose(C, C.T, atol=1e-12)
    assert np.linalg.eigvalsh(C).min() > -1e-10
    # diagonal of C == interpolated f_shape diagonal at each row's (nearest z, k)
    Nk = len(sh.k)
    Fdiag = np.diag(sh.f_shape).reshape(len(sh.z), Nk)
    z_row = np.asarray(leg.z)[np.asarray(leg.z_idx)]
    for i in (0, N // 2, N - 1):
        zi = int(np.argmin(np.abs(sh.z - z_row[i])))
        w = DL._logk_interp_weights(leg.k[i], sh.k)
        # diag(S F Sᵀ)_ii = wᵀ F[zi,zi-block] w  (the within-z k-block)
        blk = sh.f_shape[zi * Nk:(zi + 1) * Nk, zi * Nk:(zi + 1) * Nk]
        assert abs(C[i, i] - w @ blk @ w) < 1e-12


@pytest.mark.skipif(not (_have_shape and _have_desi), reason="shape npz / DESI data missing")
def test_predict_with_shape_offdiag_pd_and_backcompat():
    c = _emu_ctx()
    sh = DL.load_mf_shape(SHAPE_NPZ)
    leg = DL.load_desi_leg()
    Cfrac = DL.mf_shape_cov_for_leg(sh, leg)
    tau0 = jnp.full(leg.n_z, 0.9)
    kw = dict(pf_stats=c["pf"], dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg)
    P0, C0 = DL.predict_P_obs_on_leg(c["model"], c["theta9"], tau0, c["alpha_hcd"], **kw)
    P1, C1 = DL.predict_P_obs_on_leg(c["model"], c["theta9"], tau0, c["alpha_hcd"],
                                     mf_shape_cov=Cfrac, mf_shape_infl=1.0, **kw)
    # back-compat: no shape cov → identical
    assert np.allclose(np.asarray(P0), np.asarray(P1))
    # the shape term adds a nonzero OFF-diagonal that the baseline (diagonal-only) lacks
    D = np.asarray(C1) - np.asarray(C0)
    off = D - np.diag(np.diag(D))
    assert np.max(np.abs(off)) > 0, "shape floor added no off-diagonal"
    # C_total stays PD
    assert np.linalg.eigvalsh(np.asarray(C1)).min() > 0
    # diagonal never decreased (no diagonal floor here → topup=0; shape adds ≥0 to diagonal)
    assert np.all(np.diag(np.asarray(C1)) >= np.diag(np.asarray(C0)) - 1e-12)


@pytest.mark.skipif(not (_have_shape and _have_desi), reason="shape npz / DESI data missing")
def test_shape_offdiag_scales_as_infl_squared():
    c = _emu_ctx()
    sh = DL.load_mf_shape(SHAPE_NPZ)
    leg = DL.load_desi_leg()
    Cfrac = DL.mf_shape_cov_for_leg(sh, leg)
    tau0 = jnp.full(leg.n_z, 0.9)
    base_kw = dict(pf_stats=c["pf"], dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg)
    _, C_no = DL.predict_P_obs_on_leg(c["model"], c["theta9"], tau0, c["alpha_hcd"], **base_kw)
    _, Ca = DL.predict_P_obs_on_leg(c["model"], c["theta9"], tau0, c["alpha_hcd"],
                                    mf_shape_cov=Cfrac, mf_shape_infl=1.0, **base_kw)
    _, Cb = DL.predict_P_obs_on_leg(c["model"], c["theta9"], tau0, c["alpha_hcd"],
                                    mf_shape_cov=Cfrac, mf_shape_infl=2.0, **base_kw)
    # the SHAPE CONTRIBUTION (C_total − baseline) scales as infl² everywhere (diag + off-diag).
    da = np.asarray(Ca) - np.asarray(C_no)
    db = np.asarray(Cb) - np.asarray(C_no)
    m = np.abs(da) > 1e-30
    assert np.allclose(db[m] / da[m], 4.0, rtol=1e-6)


@pytest.mark.skipif(not (_have_shape and _have_desi), reason="shape npz / DESI data missing")
def test_loglik_finite_and_differentiable_through_shape():
    c = _emu_ctx()
    sh = DL.load_mf_shape(SHAPE_NPZ)
    leg = DL.load_desi_leg()
    Cfrac = jnp.asarray(DL.mf_shape_cov_for_leg(sh, leg))
    tau0 = jnp.full(leg.n_z, 0.9)
    P_data = jnp.asarray(leg.P_data)

    def f(theta9):
        P, C = DL.predict_P_obs_on_leg(
            c["model"], theta9, tau0, c["alpha_hcd"], pf_stats=c["pf"],
            dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg,
            mf_shape_cov=Cfrac, mf_shape_infl=1.5)
        return gaussian_loglik(P_data - P, C)

    val = f(c["theta9"])
    assert np.isfinite(float(val))
    g = jax.grad(f)(c["theta9"])
    assert np.all(np.isfinite(np.asarray(g))), "non-finite grad through the shape covariance"
