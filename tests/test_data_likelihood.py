"""Phase-C data-binding layer tests — REAL DESI DR1 + KODIAQ-SQUAD P1D legs.

Pins (per the task spec):
  Loaders:  DESI 12z×85k → post-cut shape; KS first-4-bins dropped + k≤cache_kmax; C_data
            symmetric SPD; the npz/file reads match the on-disk values.
  Binding:  predict_P_obs_on_leg returns finite (P_model, C_total) on each leg; SPD C_total;
            jnp.interp round-trips the cache grid to itself (identity at cache k).
  Metals:   a_SiIII=0 → P unchanged; a_SiIII>0 → oscillation present; differentiable.
  Likelihood: data_loglik finite + finite grads in (θ9,τ₀,α,a_SiIII,b_res) on the REAL
            DESI+KS covariance.
  Becker13: meanflux_tau0_prior(center="becker13") matches the closed form + is differentiable.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_data_likelihood.py -v
"""
import os

import hcd_analysis.emulator  # noqa: F401  enables jax_enable_x64
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator.model import Emulator
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator import meanflux_prior as MF

DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"
KS_BASE = "/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/"

_have_desi = os.path.exists(DESI_NPZ)
_have_ks = os.path.exists(KS_BASE + "final-conservative-p1d-karacayli_etal2021.txt")


# ----------------------------------------------------------------------------- #
#  synthetic emulator + cache fixtures (mirror test_likelihood_driver._ctx)
# ----------------------------------------------------------------------------- #
def _emu_ctx(n_k=172, n_basis=12, n_tb=4, seed=0):
    """A small trained-shaped Emulator + pf_stats + dla_core + a cache k-grid covering the
    real legs' k-range, so the binding is exercised end-to-end without a real checkpoint."""
    rng = np.random.default_rng(seed)
    model = Emulator(in_dim=10, n_k=n_k, n_basis=n_basis, key=jax.random.PRNGKey(seed))
    pf = {"mu_marg": jnp.asarray(rng.normal(-2, 0.5, (4, n_k))),
          "sig_marg": jnp.asarray(rng.uniform(0.5, 1.5, (4, n_k))),
          "sig_cosmo": jnp.asarray(rng.uniform(0.02, 0.1, (4, n_k)))}
    dla_core = jnp.asarray(rng.uniform(0.0, 0.5, n_k))
    # cache k-grid: the real v3.3 footprint [4.04e-4, 0.0694], 172 angular-k bins.
    cache_k = jnp.asarray(np.linspace(4.04e-4, 0.0694, n_k))
    alpha_hcd = jnp.asarray([0.06, 0.02, 0.003])
    theta9 = jnp.full(9, 0.5)
    alpha_centres = jnp.asarray([0.66, 0.83, 1.15, 1.33])
    return dict(model=model, pf=pf, dla_core=dla_core, cache_k=cache_k,
                alpha_hcd=alpha_hcd, theta9=theta9, alpha_centres=alpha_centres,
                n_k=n_k, n_tb=n_tb, rng=rng)


def _sigma_zb_for_leg(leg, n_k, n_tb, rng):
    """A (n_z_leg, 4, n_k, n_tb) fractional error vector for one leg's z-bins."""
    return jnp.asarray(rng.uniform(0.01, 0.05, (leg.n_z, 4, n_k, n_tb)))


# ============================================================================ #
#  Loaders
# ============================================================================ #
@pytest.mark.skipif(not _have_desi, reason="DESI npz not present")
def test_desi_loader_shape_and_cuts():
    leg = DL.load_desi_leg()
    # raw is 12z × 85k = 1020; default cut keeps z 2.2–4.2 (11 z) and 1e-3<k<0.5π/R_z.
    assert leg.name == "DESI"
    assert leg.n_z == 11, f"expected 11 z-bins after dropping z=4.4, got {leg.n_z}"
    assert np.all(leg.z >= 2.2 - 1e-6) and np.all(leg.z <= 4.2 + 1e-6)
    assert 4.4 not in np.round(leg.z, 2)
    # k cut applied: every kept k inside (1e-3, 0.5π/R_z(zrow))
    R_row = DL.desi_resolution_R(leg.z_row)
    assert np.all(leg.k > DL.DESI_KMIN)
    assert np.all(leg.k < 0.5 * np.pi / R_row + 1e-12)
    # shapes consistent
    N = leg.k.shape[0]
    assert leg.P_data.shape == (N,)
    assert leg.C_data.shape == (N, N)
    assert leg.z_row.shape == (N,) and leg.z_idx.shape == (N,)
    assert int(leg.n_per_z.sum()) == N


@pytest.mark.skipif(not _have_desi, reason="DESI npz not present")
def test_desi_cov_diag_inflation_applied():
    d = np.load(DESI_NPZ, allow_pickle=True)
    cov = np.asarray(d["cov"], float)
    infl = np.asarray(d["cov_diag_inflation"], float)
    leg = DL.load_desi_leg(add_cov_diag_inflation=True)
    leg_no = DL.load_desi_leg(add_cov_diag_inflation=False)
    # the inflated diagonal exceeds the bare one by exactly the kept cov_diag_inflation rows
    dd = np.diag(leg.C_data) - np.diag(leg_no.C_data)
    assert np.all(dd >= -1e-12)
    assert np.any(dd > 0), "inflation must raise some diagonal entries"
    # the inflation added equals the kept rows' cov_diag_inflation (ordering invariant)
    R_row = DL.desi_resolution_R(np.asarray(d["z"], float))
    keep = ((np.asarray(d["z"]) >= 2.2 - 1e-6) & (np.asarray(d["z"]) <= 4.2 + 1e-6)
            & (np.asarray(d["k"]) > DL.DESI_KMIN)
            & (np.asarray(d["k"]) < 0.5 * np.pi / R_row))
    assert np.allclose(dd, np.asarray(d["cov_diag_inflation"], float)[keep])


@pytest.mark.skipif(not _have_desi, reason="DESI npz not present")
def test_desi_cdata_symmetric_spd():
    leg = DL.load_desi_leg()
    C = leg.C_data
    assert np.allclose(C, C.T, atol=1e-12), "C_data must be symmetric"
    w = np.linalg.eigvalsh(C)
    assert w.min() > 0, f"C_data must be SPD; min eig {w.min():.3e}"


@pytest.mark.skipif(not _have_ks, reason="KS data not present")
def test_ks_loader_drops_first4_and_caps_kmax():
    leg = DL.load_ks_leg()
    assert leg.name == "KS"
    # first 4 k-bins are k ≤ 0.0157527 → dropped; lowest kept k is the 5th bin (~0.0198)
    assert leg.k.min() > DL.KS_DROP_KMAX, f"first-4 not dropped: kmin={leg.k.min()}"
    assert np.isclose(leg.k.min(), 0.0198315, atol=1e-4)
    # cap at the cache Nyquist 0.069
    assert leg.k.max() <= DL.CACHE_KMAX + 1e-9
    # z range 2.0–4.6
    assert leg.z.min() >= 2.0 - 1e-6 and leg.z.max() <= 4.6 + 1e-6


@pytest.mark.skipif(not _have_ks, reason="KS data not present")
def test_ks_cdata_symmetric_spd_and_matches_file():
    leg = DL.load_ks_leg()
    C = leg.C_data
    assert np.allclose(C, C.T, atol=1e-10)
    w = np.linalg.eigvalsh(C)
    assert w.min() > 0, f"KS C_data not SPD; min eig {w.min():.3e}"
    # cross-check the loaded P/k against a fresh raw parse + the same cut
    z, k, P = DL._read_ks_p1d(KS_BASE + "final-conservative-p1d-karacayli_etal2021.txt")
    keep = (z >= 2.0 - 1e-6) & (z <= 4.6 + 1e-6) & (k <= DL.CACHE_KMAX + 1e-9) & (k > DL.KS_DROP_KMAX)
    assert np.allclose(leg.P_data, P[keep])
    assert np.allclose(leg.k, k[keep])


@pytest.mark.skipif(not _have_ks, reason="KS data not present")
def test_ks_metals_resolution_off_by_default():
    leg = DL.load_ks_leg()
    assert leg.metals_on is False and leg.resolution_on is False


# ============================================================================ #
#  Binding
# ============================================================================ #
@pytest.mark.skipif(not (_have_desi and _have_ks), reason="data not present")
def test_binding_finite_and_spd_each_leg():
    c = _emu_ctx()
    desi = DL.load_desi_leg()
    ks = DL.load_ks_leg()
    for leg in (desi, ks):
        szb = _sigma_zb_for_leg(leg, c["n_k"], c["n_tb"], c["rng"])
        tau0_vec = MF.becker13_tau0(jnp.asarray(leg.z))
        P_model, C_total = DL.predict_P_obs_on_leg(
            c["model"], c["theta9"], tau0_vec, c["alpha_hcd"], pf_stats=c["pf"],
            dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg, sigma_zb=szb,
            alpha_centres=c["alpha_centres"])
        Pm = np.asarray(P_model); Ct = np.asarray(C_total)
        assert np.isfinite(Pm).all(), f"{leg.name} P_model has NaN/inf"
        assert np.isfinite(Ct).all(), f"{leg.name} C_total has NaN/inf"
        assert Pm.shape == (leg.k.shape[0],)
        assert Ct.shape == (leg.k.shape[0], leg.k.shape[0])
        assert np.allclose(Ct, Ct.T, atol=1e-8)
        w = np.linalg.eigvalsh(Ct)
        assert w.min() > 0, f"{leg.name} C_total not SPD; min eig {w.min():.3e}"


def test_interp_roundtrips_cache_grid_to_itself():
    """jnp.interp identity at the cache k: interpolating P_cache from cache_k onto cache_k
    must return P_cache exactly (the binding's interp is a no-op at the native grid)."""
    c = _emu_ctx()
    P_cache = jnp.asarray(c["rng"].normal(size=c["n_k"]))
    got = jnp.interp(c["cache_k"], c["cache_k"], P_cache)
    assert np.allclose(np.asarray(got), np.asarray(P_cache), atol=1e-12)


# ============================================================================ #
#  Metals + resolution nuisances
# ============================================================================ #
def test_metal_factor_identity_at_zero():
    k = jnp.asarray(np.linspace(1e-3, 0.06, 50))
    f = DL._metal_factor(k, a_SiIII=0.0, a_SiII=0.0)
    assert np.allclose(np.asarray(f), 1.0, atol=1e-12)


def test_metal_factor_oscillates_when_on_and_is_differentiable():
    k = jnp.asarray(np.linspace(1e-3, 0.06, 200))
    f = np.asarray(DL._metal_factor(k, a_SiIII=0.05, a_SiII=0.0))
    assert np.std(f - 1.0) > 1e-3, "SiIII term must imprint an oscillation"
    # the oscillation crosses 1 (cos changes sign) → both >1 and <1 present
    assert f.max() > 1.0 and f.min() < 1.0
    g = jax.grad(lambda a: jnp.sum(DL._metal_factor(k, a_SiIII=a)))(0.05)
    assert np.isfinite(float(g))


def test_resolution_factor_identity_and_grad():
    k = jnp.asarray(np.linspace(1e-3, 0.06, 50))
    R_z = 5.0
    assert np.allclose(np.asarray(DL._resolution_factor(k, R_z, b_res=0.0)), 1.0)
    g = jax.grad(lambda b: jnp.sum(DL._resolution_factor(k, R_z, b_res=b)))(0.01)
    assert np.isfinite(float(g))


# ============================================================================ #
#  Multi-leg likelihood on the REAL covariance
# ============================================================================ #
@pytest.mark.skipif(not (_have_desi and _have_ks), reason="data not present")
def test_data_loglik_finite_and_grads_on_real_cov():
    c = _emu_ctx()
    desi = DL.load_desi_leg(metals_on=True, resolution_on=True)
    ks = DL.load_ks_leg()
    legs = [desi, ks]
    # global z ladder = the union of leg z (ascending)
    z_global = np.unique(np.round(np.concatenate([desi.z, ks.z]), 6))
    tau0_global = jnp.asarray(MF.becker13_tau0(jnp.asarray(z_global)))
    szb = {leg.name: _sigma_zb_for_leg(leg, c["n_k"], c["n_tb"], c["rng"]) for leg in legs}

    def f(theta9, tau0, alpha, a_SiIII, b_res):
        return DL.data_loglik(
            c["model"], theta9, tau0, alpha, legs, pf_stats=c["pf"], dla_core=c["dla_core"],
            cache_k=c["cache_k"], z_global=z_global, sigma_zb_per_leg=szb,
            alpha_centres=c["alpha_centres"], a_SiIII=a_SiIII, b_res=b_res)

    val = float(f(c["theta9"], tau0_global, c["alpha_hcd"], 0.04, 0.005))
    assert np.isfinite(val), "data_loglik must be finite on the real covariance"
    gth, gt, ga, gm, gr = jax.grad(f, argnums=(0, 1, 2, 3, 4))(
        c["theta9"], tau0_global, c["alpha_hcd"], 0.04, 0.005)
    assert np.isfinite(np.asarray(gth)).all(), "∂/∂θ9 NaN"
    assert np.isfinite(np.asarray(gt)).all(), "∂/∂τ₀ NaN"
    assert np.isfinite(np.asarray(ga)).all(), "∂/∂α NaN"
    assert np.isfinite(float(gm)), "∂/∂a_SiIII NaN"
    assert np.isfinite(float(gr)), "∂/∂b_res NaN"


@pytest.mark.skipif(not (_have_desi and _have_ks), reason="data not present")
def test_data_loglik_block_diagonal_equals_sum_of_legs():
    """The joint logL must equal the sum of the per-leg gaussian_logliks (block-diagonal)."""
    from hcd_analysis.emulator.likelihood import gaussian_loglik
    c = _emu_ctx()
    legs = [DL.load_desi_leg(), DL.load_ks_leg()]
    z_global = np.unique(np.round(np.concatenate([legs[0].z, legs[1].z]), 6))
    tau0_global = jnp.asarray(MF.becker13_tau0(jnp.asarray(z_global)))
    szb = {leg.name: _sigma_zb_for_leg(leg, c["n_k"], c["n_tb"], c["rng"]) for leg in legs}
    total, parts = DL.data_loglik(
        c["model"], c["theta9"], tau0_global, c["alpha_hcd"], legs, pf_stats=c["pf"],
        dla_core=c["dla_core"], cache_k=c["cache_k"], z_global=z_global,
        sigma_zb_per_leg=szb, alpha_centres=c["alpha_centres"], return_parts=True)
    # rebuild per-leg logL independently and sum
    man = 0.0
    for leg in legs:
        sel = np.array([int(np.argmin(np.abs(z_global - zz))) for zz in leg.z])
        tau0_vec = tau0_global[jnp.asarray(sel)]
        Pm, Ct = DL.predict_P_obs_on_leg(
            c["model"], c["theta9"], tau0_vec, c["alpha_hcd"], pf_stats=c["pf"],
            dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg,
            sigma_zb=szb[leg.name], alpha_centres=c["alpha_centres"])
        man += float(gaussian_loglik(jnp.asarray(leg.P_data) - Pm, Ct))
    assert np.isclose(float(total), man, rtol=1e-10), f"joint {total} vs sum {man}"
    assert set(parts) == {"DESI", "KS"}


# ============================================================================ #
#  Becker+2013 mean-flux center
# ============================================================================ #
def test_becker13_matches_closed_form_and_differentiable():
    z = jnp.asarray([2.2, 3.0, 4.2])
    got = np.asarray(MF.becker13_tau0(z))
    ref = 0.751 * ((1.0 + np.asarray(z)) / 4.5) ** 2.90 - 0.132
    assert np.allclose(got, ref, rtol=1e-10)
    # prior wrapper exposes it via center="becker13"
    mu, sig = MF.meanflux_tau0_prior(z, frac_sigma=0.05, center="becker13")
    assert np.allclose(np.asarray(mu), ref, rtol=1e-10)
    assert np.allclose(np.asarray(sig), 0.05 * ref, rtol=1e-10)
    # kim center still the default and differs from becker13
    mu_kim, _ = MF.meanflux_tau0_prior(z, center="kim")
    assert not np.allclose(np.asarray(mu_kim), ref)
    g = jax.grad(lambda zz: jnp.sum(MF.becker13_tau0(zz)))(z)
    assert np.isfinite(np.asarray(g)).all()


def test_meanflux_per_z_frac_sigma_array():
    """frac_sigma may be a per-z array (Becker+2013 reports ~3–8% z-dependent)."""
    z = jnp.asarray([2.2, 3.0, 4.2])
    fs = jnp.asarray([0.03, 0.05, 0.08])
    mu, sig = MF.meanflux_tau0_prior(z, frac_sigma=fs, center="becker13")
    assert np.allclose(np.asarray(sig), np.asarray(fs) * np.asarray(mu), rtol=1e-10)
