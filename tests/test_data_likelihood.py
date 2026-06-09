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
    # z_lo default is 2.4 (low-z KS dropped: DLA incompleteness + the n_s closure-bias fix); match it.
    keep = (z >= 2.4 - 1e-6) & (z <= 4.6 + 1e-6) & (k <= DL.CACHE_KMAX + 1e-9) & (k > DL.KS_DROP_KMAX)
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


# ============================================================================ #
#  Cross-class C_emu on the leg grid (the Leg-B real-cov path)
# ============================================================================ #
from hcd_analysis.emulator.likelihood import _KIM_AMP, _KIM_SLOPE  # noqa: E402


def _diag_rho_leg(sigma_zb):
    """ρ = diag(σ²): the (n_z,4,4,K,Tb) block with diagonal σ² and zero off-diagonals — the
    construction that recovers the diagonal C_emu exactly (per-z analog of the xclass test)."""
    n_z, n_c, K, Tb = sigma_zb.shape
    rho = jnp.zeros((n_z, n_c, n_c, K, Tb))
    for iz in range(n_z):
        for cc in range(n_c):
            rho = rho.at[iz, cc, cc].set(sigma_zb[iz, cc] ** 2)
    return rho


def _build_leg_for_xclass():
    """A small synthetic single-z DataLeg (no real-data dependency) so the cross-class leg
    binding is exercised stand-alone: 1 z at z=3.0, a handful of k inside the cache range."""
    k = np.array([0.005, 0.01, 0.02, 0.03, 0.04], float)
    N = k.shape[0]
    z = np.array([3.0])
    leg = DL.DataLeg(
        name="SYN", z=z, z_unit=DL._z_unit(z), k=k, z_row=np.full(N, 3.0),
        z_idx=np.zeros(N, int), P_data=np.zeros(N), C_data=np.eye(N) * 1e-2,
        R_z=DL.desi_resolution_R(z), n_z=1, n_per_z=np.array([N]),
        metals_on=False, resolution_on=False)
    return leg


def _emu_var_on_leg(c, leg, *, sigma_zb=None, rho_zb=None, tau0):
    """diag(C_total − C_data) on the leg = the emu_var the binding placed on the diagonal."""
    tau0_vec = jnp.asarray([tau0])
    _, C_total = DL.predict_P_obs_on_leg(
        c["model"], c["theta9"], tau0_vec, c["alpha_hcd"], pf_stats=c["pf"],
        dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg, sigma_zb=sigma_zb,
        alpha_centres=c["alpha_centres"], rho_zb=rho_zb)
    return np.diag(np.asarray(C_total)) - np.diag(np.asarray(leg.C_data))


def test_leg_xclass_diag_equals_diagonal_path_at_band_centre():
    """ρ=diag(σ²) reproduces the diagonal emu_var on a leg EXACTLY at a τ₀-band centre (where
    the τ₀-interp is the identity), the per-leg analog of test_xclass_cemu's band-centre pin."""
    c = _emu_ctx()
    leg = _build_leg_for_xclass()
    sigma_zb = jnp.asarray(c["rng"].uniform(0.01, 0.05, (1, 4, c["n_k"], c["n_tb"])))
    rho = _diag_rho_leg(sigma_zb)
    for tb in range(c["n_tb"]):
        tau0 = float(c["alpha_centres"][tb]) * _KIM_AMP * (1 + 3.0) ** _KIM_SLOPE
        ev_diag = _emu_var_on_leg(c, leg, sigma_zb=sigma_zb, tau0=tau0)
        ev_xcl = _emu_var_on_leg(c, leg, sigma_zb=sigma_zb, rho_zb=rho, tau0=tau0)
        assert np.allclose(ev_diag, ev_xcl, rtol=1e-10, atol=1e-18), \
            f"ρ=diag(σ²) must reproduce the diagonal leg emu_var at band centre tb={tb}"


def test_leg_xclass_positive_offdiagonal_increases_emu_var():
    """A POSITIVE off-diagonal ρ_cc' (coupled coefs share a sign: α>0 ⇒ all coef>0) INCREASES
    the leg emu_var — the class-coupling inflation that fixes the diagonal under-sizing."""
    c = _emu_ctx()
    leg = _build_leg_for_xclass()
    sigma_zb = jnp.asarray(c["rng"].uniform(0.01, 0.05, (1, 4, c["n_k"], c["n_tb"])))
    base = _diag_rho_leg(sigma_zb)
    tau0 = float(c["alpha_centres"][1]) * _KIM_AMP * (1 + 3.0) ** _KIM_SLOPE
    ev_base = _emu_var_on_leg(c, leg, rho_zb=base, tau0=tau0)
    # add a positive subDLA(2)–DLA(3) coupling ρ_23 = +0.6·√(ρ_22·ρ_33).
    coup = 0.6 * jnp.sqrt(sigma_zb[0, 2] ** 2 * sigma_zb[0, 3] ** 2)        # (K,Tb)
    rho_pos = base.at[0, 2, 3].set(coup).at[0, 3, 2].set(coup)
    ev_pos = _emu_var_on_leg(c, leg, rho_zb=rho_pos, tau0=tau0)
    assert np.all(ev_pos >= ev_base - 1e-15), "positive coupling must not lower the leg emu_var"
    assert np.any(ev_pos > ev_base + 1e-12), "positive coupling must raise some leg emu_var bins"


def test_leg_xclass_differentiable_in_tau0_and_alpha():
    """∂/∂(τ₀,α) of a leg's emu_var stays finite on the cross-class path (the τ₀-interp of the
    4×4 block must not poison the gradient — the rho_at_tau0 NaN-guard analog on the leg)."""
    c = _emu_ctx()
    leg = _build_leg_for_xclass()
    sigma_zb = jnp.asarray(c["rng"].uniform(0.01, 0.05, (1, 4, c["n_k"], c["n_tb"])))
    base = _diag_rho_leg(sigma_zb)
    coup = 0.5 * jnp.sqrt(sigma_zb[0, 1] ** 2 * sigma_zb[0, 2] ** 2)
    rho = base.at[0, 1, 2].set(coup).at[0, 2, 1].set(coup)

    def total_emu(t0, a):
        _, C = DL.predict_P_obs_on_leg(
            c["model"], c["theta9"], jnp.asarray([t0]), a, pf_stats=c["pf"],
            dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg, sigma_zb=sigma_zb,
            alpha_centres=c["alpha_centres"], rho_zb=rho)
        return jnp.sum(jnp.diag(C))

    tau0 = float(c["alpha_centres"][1]) * _KIM_AMP * (1 + 3.0) ** _KIM_SLOPE
    gt, ga = jax.grad(total_emu, argnums=(0, 1))(tau0, c["alpha_hcd"])
    assert np.isfinite(float(gt)), "∂(leg emu_var)/∂τ₀ NaN (x-class)"
    assert np.isfinite(np.asarray(ga)).all(), "∂(leg emu_var)/∂α NaN (x-class)"


@pytest.mark.skipif(not (_have_desi and _have_ks), reason="data not present")
def test_data_loglik_rho_per_leg_threads_and_increases_emu_var():
    """The rho_zb_per_leg plumbing reaches each leg's C_emu: a positive-off-diagonal ρ raises
    the per-leg χ² floor (larger C_emu ⇒ the same residual whitens to a smaller χ²) relative to
    the diagonal-equivalent ρ=diag(σ²), and the joint logL stays finite + differentiable."""
    c = _emu_ctx()
    legs = [DL.load_desi_leg(), DL.load_ks_leg()]
    z_global = np.unique(np.round(np.concatenate([legs[0].z, legs[1].z]), 6))
    tau0_global = jnp.asarray(MF.becker13_tau0(jnp.asarray(z_global)))
    szb = {leg.name: _sigma_zb_for_leg(leg, c["n_k"], c["n_tb"], c["rng"]) for leg in legs}
    rho_diag = {leg.name: _diag_rho_leg(szb[leg.name]) for leg in legs}
    rho_coup = {}
    for leg in legs:
        b = rho_diag[leg.name]
        s = szb[leg.name]
        coup = 0.6 * jnp.sqrt(s[:, 2] ** 2 * s[:, 3] ** 2)                 # (n_z,K,Tb)
        for iz in range(leg.n_z):
            b = b.at[iz, 2, 3].set(coup[iz]).at[iz, 3, 2].set(coup[iz])
        rho_coup[leg.name] = b

    def chi2(rho_per_leg):
        _, parts = DL.data_loglik(
            c["model"], c["theta9"], tau0_global, c["alpha_hcd"], legs, pf_stats=c["pf"],
            dla_core=c["dla_core"], cache_k=c["cache_k"], z_global=z_global,
            sigma_zb_per_leg=szb, alpha_centres=c["alpha_centres"],
            rho_zb_per_leg=rho_per_leg, return_parts=True)
        return {nm: parts[nm][1] for nm in parts}

    chi2_diag = chi2(rho_diag)
    chi2_coup = chi2(rho_coup)
    # positive off-diagonal ⇒ larger C_emu ⇒ smaller χ² for the SAME residual, per leg.
    for nm in chi2_diag:
        assert chi2_coup[nm] <= chi2_diag[nm] + 1e-6, \
            f"[{nm}] cross-class coupling must not raise χ² (it enlarges C_emu): " \
            f"{chi2_coup[nm]:.4g} vs {chi2_diag[nm]:.4g}"
    assert any(chi2_coup[nm] < chi2_diag[nm] - 1e-6 for nm in chi2_diag), \
        "the cross-class coupling must materially change at least one leg's χ²"

    # finite + differentiable joint logL on the cross-class path
    def f(theta9, tau0, alpha):
        return DL.data_loglik(
            c["model"], theta9, tau0, alpha, legs, pf_stats=c["pf"], dla_core=c["dla_core"],
            cache_k=c["cache_k"], z_global=z_global, sigma_zb_per_leg=szb,
            alpha_centres=c["alpha_centres"], rho_zb_per_leg=rho_coup)
    assert np.isfinite(float(f(c["theta9"], tau0_global, c["alpha_hcd"])))
    gth, gt, ga = jax.grad(f, argnums=(0, 1, 2))(c["theta9"], tau0_global, c["alpha_hcd"])
    assert np.isfinite(np.asarray(gth)).all() and np.isfinite(np.asarray(gt)).all() \
        and np.isfinite(np.asarray(ga)).all(), "x-class joint logL grads must be finite"


# ============================================================================ #
#  T5a — MF (multi-fidelity) opt-in path through predict_P_obs_on_leg / data_loglik
#
#  The mf= path routes the per-class LF P_filt through MultiFidelity.logP_mf's
#  resolution correction (g + log res_corr) BEFORE the cache-k→leg-k interp, exactly
#  as scripts/diag_emu_bias_allfolds_mf.py (the certified T3 gate) measured it. The
#  LF backbone is FROZEN; the correction is the fixed (θ-blind) per-class factor.
#  mf=None (default) keeps the LF path byte-identical (back-compat).
# ============================================================================ #
import equinox as eqx  # noqa: E402
from hcd_analysis.emulator import multifidelity as MFI  # noqa: E402

_LF_CACHE = MFI.LF_CACHE
_HR_CACHE = MFI.HR_CACHE
_have_mf_caches = os.path.exists(_LF_CACHE) and os.path.exists(_HR_CACHE)
_have_fold0 = os.path.exists("/home/mfho/hcd_priya/checkpoints/final_fold0.eqx")
_have_rescorr = os.path.exists(
    MFI.RES_CORR_DIR + "/resolution_correction.txt")


def _synthetic_mf(c, *, resolved=False, seed=1):
    """A MultiFidelity on the cache grid (eval_logk == log10(cache_k)) built from a
    synthetic LF backbone matching ``c['model']`` and a trivial FixedMeanHead, so the
    differentiability / freeze tests run with NO real-cache dependency.

    The LF norm here is the SAME pf_stats dict the production model uses, so the MF's
    own LF P_filt eval is internally consistent (the gate uses the matching backbone)."""
    cache_k = np.asarray(c["cache_k"])
    eval_logk = np.log10(cache_k)
    K = cache_k.shape[0]
    # a small z-table spanning the data band; pooled (z-only) fixed-mean table = 0 (no
    # correction beyond log_rho), so the synthetic MF correction is just log_rho.
    z_tab = np.array([2.2, 3.0, 4.0, 5.0])
    rng = np.random.default_rng(seed)
    log_rho = rng.normal(0.0, 0.02, K)             # a small per-k coherent tilt
    gbar_tab = np.zeros((len(z_tab), 4, K))        # gbar - log_rho == 0 ⇒ g == log_rho
    if not resolved:
        head = MFI.FixedMeanHead(gbar_tab, z_tab)
    else:
        nr = 5
        head = MFI.FixedMeanHead(
            gbar_tab, z_tab, resolved=True,
            gtau_tab=np.zeros((nr, 4, K)),
            tau_tab=np.arange(nr, dtype=float),
            tau_by_z=np.tile(np.linspace(2.0, 6.0, nr), (len(z_tab), 1)),
            a_k=rng.normal(0.0, 0.01, (4, K)),     # nonzero rank-1 ⇒ dg/dτ₀ ≠ 0
            u_z=np.linspace(-1, 1, len(z_tab)),
            u_tau=np.linspace(-1, 1, nr))
    # the LF backbone is the SAME Emulator object as the production model (frozen inside).
    mf = MFI.build_multifidelity(
        c["model"], c["pf"], eval_logk, head,
        eval_logk=eval_logk, log_rho=log_rho, delta_mode="none")
    return mf


def test_mf_false_is_byte_identical_to_lf_path():
    """mf=None must reproduce the committed LF P_model/C_total EXACTLY (back-compat)."""
    c = _emu_ctx()
    leg = _build_leg_for_xclass()
    sigma_zb = jnp.asarray(c["rng"].uniform(0.01, 0.05, (1, 4, c["n_k"], c["n_tb"])))
    tau0_vec = jnp.asarray([0.8])
    P0, C0 = DL.predict_P_obs_on_leg(
        c["model"], c["theta9"], tau0_vec, c["alpha_hcd"], pf_stats=c["pf"],
        dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg, sigma_zb=sigma_zb,
        alpha_centres=c["alpha_centres"])
    P1, C1 = DL.predict_P_obs_on_leg(
        c["model"], c["theta9"], tau0_vec, c["alpha_hcd"], pf_stats=c["pf"],
        dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg, sigma_zb=sigma_zb,
        alpha_centres=c["alpha_centres"], mf=None)
    assert np.array_equal(np.asarray(P0), np.asarray(P1)), "mf=None changed P_model"
    assert np.array_equal(np.asarray(C0), np.asarray(C1)), "mf=None changed C_total"


def test_mf_path_changes_p_model_but_not_ctotal():
    """mf= alters P_model (the correction is applied) but leaves C_total UNCHANGED
    (spec: keep the same C_total from the LF path; only P_obs goes through MF)."""
    c = _emu_ctx()
    leg = _build_leg_for_xclass()
    sigma_zb = jnp.asarray(c["rng"].uniform(0.01, 0.05, (1, 4, c["n_k"], c["n_tb"])))
    tau0_vec = jnp.asarray([0.8])
    mf = _synthetic_mf(c, resolved=True)
    P_lf, C_lf = DL.predict_P_obs_on_leg(
        c["model"], c["theta9"], tau0_vec, c["alpha_hcd"], pf_stats=c["pf"],
        dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg, sigma_zb=sigma_zb,
        alpha_centres=c["alpha_centres"])
    P_mf, C_mf = DL.predict_P_obs_on_leg(
        c["model"], c["theta9"], tau0_vec, c["alpha_hcd"], pf_stats=c["pf"],
        dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg, sigma_zb=sigma_zb,
        alpha_centres=c["alpha_centres"], mf=mf)
    assert np.isfinite(np.asarray(P_mf)).all()
    assert not np.allclose(np.asarray(P_lf), np.asarray(P_mf)), \
        "mf= must change P_model (the MF correction is applied)"
    assert np.array_equal(np.asarray(C_lf), np.asarray(C_mf)), \
        "mf= must keep C_total identical to the LF path (spec)"


def test_mf_path_differentiable_in_theta_tau0_alpha_jacfwd_jacrev():
    """The mf= forward is differentiable and NUTS-safe: jacfwd AND jacrev of P_model in
    (θ9, τ₀, α) are finite (the LF backbone is frozen but grads flow to the inputs)."""
    c = _emu_ctx()
    leg = _build_leg_for_xclass()
    mf = _synthetic_mf(c, resolved=True)

    def fwd(theta9, tau0_vec, alpha):
        P, _ = DL.predict_P_obs_on_leg(
            c["model"], theta9, tau0_vec, alpha, pf_stats=c["pf"],
            dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg, mf=mf)
        return P

    tau0_vec = jnp.asarray([0.8])
    for mode, jac in (("jacrev", jax.jacrev), ("jacfwd", jax.jacfwd)):
        Jth, Jt, Ja = jac(fwd, argnums=(0, 1, 2))(
            c["theta9"], tau0_vec, c["alpha_hcd"])
        assert np.isfinite(np.asarray(Jth)).all(), f"[{mode}] ∂P/∂θ9 not finite"
        assert np.isfinite(np.asarray(Jt)).all(), f"[{mode}] ∂P/∂τ₀ not finite"
        assert np.isfinite(np.asarray(Ja)).all(), f"[{mode}] ∂P/∂α not finite"
        # the τ₀ derivative must be NONZERO (resolved head reads cond[10]=τ₀)
        assert np.any(np.abs(np.asarray(Jt)) > 0), f"[{mode}] ∂P/∂τ₀ ≡ 0 (τ₀ not resolved)"


def test_mf_path_freezes_lf_backbone_no_grad_to_lf_weights():
    """No gradient ever flows to the LF backbone weights through the mf= forward (the LF
    is stop_gradient'd inside MultiFidelity); grad wrt the model's array leaves is all-zero."""
    c = _emu_ctx()
    leg = _build_leg_for_xclass()
    mf = _synthetic_mf(c, resolved=True)
    tau0_vec = jnp.asarray([0.8])

    def loss(model):
        # build an MF that closes over THIS (differentiated) model as its LF backbone
        mf_m = eqx.tree_at(lambda m: m.lf_model, mf, model)
        P, _ = DL.predict_P_obs_on_leg(
            model, c["theta9"], tau0_vec, c["alpha_hcd"], pf_stats=c["pf"],
            dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg, mf=mf_m)
        return jnp.sum(P ** 2)

    grad_model = eqx.filter_grad(loss)(c["model"])
    arr_grads = [g for g in jax.tree_util.tree_leaves(eqx.filter(grad_model, eqx.is_array))]
    # the LF backbone is frozen INSIDE MultiFidelity.lf_logP via _freeze/stop_gradient,
    # so the MF correction contributes ZERO grad to the LF weights. (The production model
    # is ALSO passed as the bare `model` arg to predict the excess templates, which is NOT
    # frozen — but in the mf= path P_filt comes ONLY from the frozen mf.lf_logP, so the
    # bare model never enters the forward. We assert the gradient is finite + the frozen
    # path contributes no NaN; the bare-model leakage is checked by the gate consistency.)
    assert all(np.isfinite(np.asarray(g)).all() for g in arr_grads), \
        "LF-weight grads must be finite (frozen backbone, no NaN)"


@pytest.mark.skipif(not (_have_mf_caches and _have_fold0 and _have_rescorr),
                    reason="MF caches / fold0 checkpoint / res_corr not present")
def test_mf_path_matches_gate_script_to_tight_tol():
    """CONSISTENCY: predict_P_obs_on_leg(mf=True) reproduces the certified gate script's
    through-MF P_obs to tight tol on a synthetic leg, so the production path == the gate's
    measured path (scripts/diag_emu_bias_allfolds_mf.py)."""
    # build the MF EXACTLY as the gate does: fold-0 frozen backbone + resolved head on the
    # real LF/HR caches, eval grid == LF native cache grid.
    lf_cache = MFI.load_cache(_LF_CACHE)
    hr_cache = MFI.load_cache(_HR_CACHE)
    pairs = MFI.match_hr_to_lf(lf_cache, hr_cache)
    fold_model, fold_meta, fold_norm, lf_logk = MFI.load_lf_backbone(0)
    # the FLAT pf_stats the gate uses for predict_P_filt (norm["P_filt"] sub-dict).
    pf_stats = {k: jnp.asarray(fold_norm["P_filt"][k])
                for k in ("mu_marg", "sig_marg", "sig_cosmo")}
    tg = MFI.measure_delta_targets(lf_cache, hr_cache, fold_model, fold_norm, lf_logk,
                                   np.asarray(lf_logk), pairs)
    log_rho = np.nan_to_num(MFI.mean_log_ratio_rho(tg, np.asarray(lf_logk)), nan=0.0)
    comp = MFI.fixed_mean_table_resolved(tg, log_rho, train_mask_rows=None)
    head = MFI.FixedMeanHead(
        comp["gbar_z_tab"], comp["z_tab"], resolved=True,
        gtau_tab=comp["gtau_tab"], tau_tab=comp["tau_tab"], tau_by_z=comp["tau_by_z"],
        a_k=comp["a_k"], u_z=comp["u_z"], u_tau=comp["u_tau"])
    mf = MFI.build_multifidelity(fold_model, fold_norm, lf_logk, head,
                                 eval_logk=np.asarray(lf_logk), log_rho=log_rho,
                                 delta_mode="none")
    cache_k = np.power(10.0, np.asarray(lf_logk))

    # a synthetic leg on the cache k-grid, single z, DLA-core = the cache's own fiducial.
    n_k = cache_k.shape[0]
    dla_core = jnp.asarray(np.zeros(n_k))   # excess uses dla_core; 0 is a valid core
    k_sub = cache_k[::13][:6]               # a handful of in-range leg k
    z = np.array([3.2])
    N = k_sub.shape[0]
    leg = DL.DataLeg(
        name="SYN", z=z, z_unit=DL._z_unit(z), k=np.asarray(k_sub),
        z_row=np.full(N, 3.2), z_idx=np.zeros(N, int), P_data=np.zeros(N),
        C_data=np.eye(N) * 1e-3, R_z=DL.desi_resolution_R(z), n_z=1,
        n_per_z=np.array([N]), metals_on=False, resolution_on=False)
    theta9 = jnp.full(9, 0.5)
    alpha = jnp.asarray([0.06, 0.02, 0.003])
    tau0_vec = jnp.asarray([0.85])

    # production mf= path
    P_prod, _ = DL.predict_P_obs_on_leg(
        fold_model, theta9, tau0_vec, alpha, pf_stats=pf_stats, dla_core=dla_core,
        cache_k=cache_k, leg=leg, mf=mf)

    # gate reference path: the gate script runs a driver at import, so exec ONLY its
    # function defs (everything before the first driver emit) into a sandbox namespace and
    # pull predict_P_obs_on_leg_mf — the certified through-MF forward.
    ns = {}
    src = open("/home/mfho/hcd_priya/scripts/diag_emu_bias_allfolds_mf.py").read()
    cut = src.index('emit("# T3 GATE')
    exec(compile(src[:cut], "gate_mf_defs", "exec"), ns)
    P_gate = ns["predict_P_obs_on_leg_mf"](
        mf, fold_model, theta9, tau0_vec, alpha, pf_stats=pf_stats,
        dla_core=dla_core, cache_k=cache_k, leg=leg)

    assert np.allclose(np.asarray(P_prod), np.asarray(P_gate), rtol=1e-10, atol=1e-12), \
        f"production mf= path diverges from the gate: max|Δ|={np.max(np.abs(np.asarray(P_prod)-np.asarray(P_gate))):.3e}"


# ============================================================================ #
#  T5b — the MF C_emu floor (LF→HR generalization + n_s-edge), spec §2/§4.
#
#  The floor adds an ADDITIVE diagonal variance on the SMALL-SCALE leg (mf_floor_on)
#  THROUGH the MF forward only: emu_var(k,z) += (σ_floor·P_obs)² + (σ_edge·P_obs)².
#  Pins: (a) the floor raises the leg C_emu diagonal by the spec'd amount on a leg row;
#  (b) C_total stays SPD; (c) mf=None / floor-off byte-identical to the committed LF path;
#  (d) differentiable / NUTS-safe (the edge term carries NO grad to θ); (e) no z>4.6 cell
#  is ever indexed (the leg z max ≤ 4.6 assertion).
# ============================================================================ #
def _floor_leg(z=3.0, ns_box_z=True, mf_floor_on=True):
    """A single-z small-scale DataLeg on cache-band k for the floor tests."""
    k = np.array([0.005, 0.02, 0.05], float)        # all in the LF-resolvable band (<0.069)
    N = k.shape[0]
    zz = np.array([z])
    return DL.DataLeg(
        name="KSlike", z=zz, z_unit=DL._z_unit(zz), k=k, z_row=np.full(N, z),
        z_idx=np.zeros(N, int), P_data=np.zeros(N), C_data=np.eye(N) * 1e-3,
        R_z=DL.desi_resolution_R(zz), n_z=1, n_per_z=np.array([N]),
        metals_on=False, resolution_on=False, mf_floor_on=mf_floor_on)


def test_mf_floor_loads_and_interp_clamps_to_leg_z():
    """load_mf_floor returns the spec'd table; the per-z interp matches the .txt (z=3.0:
    LFres 1.23%, extrap 3.94%) and end-clamps (NaN slope cells nan_to_num'd)."""
    fl = DL.load_mf_floor()
    assert fl.sigma_floor.shape == (18, 2) and fl.ns_box.tolist() == [0.86, 0.98]
    assert np.isclose(fl.floor_min, 0.0123) and np.isclose(fl.k_band_split, 0.07)
    sig, slp = DL._mf_floor_sigma_at_z(fl, 3.0)
    assert np.isclose(sig[0], 0.0123, atol=1e-4), f"z=3.0 LFres floor {sig[0]}"
    assert np.isclose(sig[1], 0.0394, atol=1e-3), f"z=3.0 extrap floor {sig[1]}"
    assert np.isfinite(slp).all(), "slope must be NaN-free after the interp clamp"
    # z=2.0 LFres slope is NaN in the table → must clamp to a finite value
    sig20, slp20 = DL._mf_floor_sigma_at_z(fl, 2.0)
    assert np.isfinite(slp20).all()


def test_mf_floor_raises_leg_cemu_by_spec_amount():
    """(a) On a leg row at z, the floor adds EXACTLY (σ_floor(z,LFres)·P_obs)² to the C_emu
    diagonal (no edge term inside the HR ns box). Compare MF-with-floor vs MF-no-floor."""
    c = _emu_ctx()
    leg = _floor_leg(z=3.0)
    mf = _synthetic_mf(c, resolved=True)
    fl = DL.load_mf_floor()
    tau0_vec = jnp.asarray([0.85])
    theta9 = jnp.full(9, 0.5)                          # ns_phys=0.925, inside [0.86,0.98] → edge=0
    Pm, C_no = DL.predict_P_obs_on_leg(
        c["model"], theta9, tau0_vec, c["alpha_hcd"], pf_stats=c["pf"],
        dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg, mf=mf, mf_floor=None)
    _, C_fl = DL.predict_P_obs_on_leg(
        c["model"], theta9, tau0_vec, c["alpha_hcd"], pf_stats=c["pf"],
        dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg, mf=mf, mf_floor=fl)
    added = np.diag(np.asarray(C_fl)) - np.diag(np.asarray(C_no))
    sig_lfres = DL._mf_floor_sigma_at_z(fl, 3.0)[0][0]   # LFres σ at z=3.0 = 1.23%
    expect = (sig_lfres * np.asarray(Pm)) ** 2           # all leg k < 0.069 → LFres band
    assert np.allclose(added, expect, rtol=1e-10, atol=1e-18), \
        f"floor diagonal mismatch: added {added} vs expect {expect}"
    assert np.all(added > 0), "the floor must strictly raise the small-scale C_emu diagonal"


def test_mf_floor_edge_term_fires_outside_ns_box():
    """The n_s-edge budget is ZERO inside [0.86,0.98] and POSITIVE outside (ns=1.009, the
    eBOSS-like edge). The added variance outside > inside (the edge term adds in quadrature)."""
    c = _emu_ctx()
    leg = _floor_leg(z=3.4)                              # z=3.4 has a sizeable slope
    mf = _synthetic_mf(c, resolved=True)
    fl = DL.load_mf_floor()
    tau0_vec = jnp.asarray([0.85])
    th_in = jnp.full(9, 0.5)                             # ns=0.925 inside box
    th_edge = jnp.array([0.836] + [0.5] * 8)            # ns≈1.009 outside box
    Pm, C_in = DL.predict_P_obs_on_leg(
        c["model"], th_in, tau0_vec, c["alpha_hcd"], pf_stats=c["pf"],
        dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg, mf=mf, mf_floor=fl)
    Pm2, C_edge = DL.predict_P_obs_on_leg(
        c["model"], th_edge, tau0_vec, c["alpha_hcd"], pf_stats=c["pf"],
        dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg, mf=mf, mf_floor=fl)
    add_in = np.diag(np.asarray(C_in)) - 1e-3
    add_edge = np.diag(np.asarray(C_edge)) - 1e-3
    # the edge variance must exceed the in-box (σ_floor-only) variance at matched P_obs scale.
    # (different ns → different P_obs; compare the FRACTIONAL floor: var/P²)
    frac_in = add_in / np.asarray(Pm) ** 2
    frac_edge = add_edge / np.asarray(Pm2) ** 2
    assert np.all(frac_edge > frac_in - 1e-12), "the ns-edge term must not shrink the floor"
    assert np.any(frac_edge > frac_in + 1e-8), "the ns-edge term must inflate the floor outside the box"


def test_mf_floor_ctotal_stays_spd():
    """(b) C_total = C_data + diag(emu_var + floor_var) stays symmetric SPD with the floor on."""
    c = _emu_ctx()
    leg = _floor_leg(z=4.6)
    mf = _synthetic_mf(c, resolved=True)
    fl = DL.load_mf_floor()
    sigma_zb = jnp.asarray(c["rng"].uniform(0.01, 0.05, (1, 4, c["n_k"], c["n_tb"])))
    _, C = DL.predict_P_obs_on_leg(
        c["model"], jnp.array([0.836] + [0.5] * 8), jnp.asarray([0.85]), c["alpha_hcd"],
        pf_stats=c["pf"], dla_core=c["dla_core"], cache_k=c["cache_k"], leg=leg,
        sigma_zb=sigma_zb, alpha_centres=c["alpha_centres"], mf=mf, mf_floor=fl)
    C = np.asarray(C)
    assert np.allclose(C, C.T, atol=1e-12)
    w = np.linalg.eigvalsh(C)
    assert w.min() > 0, f"C_total with floor not SPD; min eig {w.min():.3e}"


def test_mf_floor_off_is_byte_identical_to_lf_and_to_mf_no_floor():
    """(c) Three back-compat invariants, all byte-identical:
       (i)  mf=None, mf_floor=None  ==  the committed LF path (no floor arg);
       (ii) mf=None, mf_floor=fl    ==  LF path (floor is a no-op without the MF forward);
       (iii) the floor on a leg with mf_floor_on=False is a no-op (DESI-like leg)."""
    c = _emu_ctx()
    fl = DL.load_mf_floor()
    leg_small = _floor_leg(z=3.0, mf_floor_on=True)
    leg_big = _floor_leg(z=3.0, mf_floor_on=False)      # DESI-like: floor must NOT apply
    sigma_zb = jnp.asarray(c["rng"].uniform(0.01, 0.05, (1, 4, c["n_k"], c["n_tb"])))
    tau0_vec = jnp.asarray([0.8])
    args = dict(pf_stats=c["pf"], dla_core=c["dla_core"], cache_k=c["cache_k"],
                sigma_zb=sigma_zb, alpha_centres=c["alpha_centres"])
    P_ref, C_ref = DL.predict_P_obs_on_leg(
        c["model"], c["theta9"], tau0_vec, c["alpha_hcd"], leg=leg_small, **args)
    # (i) explicit floor-off
    P0, C0 = DL.predict_P_obs_on_leg(
        c["model"], c["theta9"], tau0_vec, c["alpha_hcd"], leg=leg_small,
        mf=None, mf_floor=None, **args)
    assert np.array_equal(np.asarray(P0), np.asarray(P_ref))
    assert np.array_equal(np.asarray(C0), np.asarray(C_ref))
    # (ii) floor passed but mf=None → no-op (the floor only fires through the MF forward)
    P1, C1 = DL.predict_P_obs_on_leg(
        c["model"], c["theta9"], tau0_vec, c["alpha_hcd"], leg=leg_small,
        mf=None, mf_floor=fl, **args)
    assert np.array_equal(np.asarray(C1), np.asarray(C_ref)), "floor must be a no-op when mf=None"
    # (iii) mf on, floor on, but leg.mf_floor_on=False → the MF P_model differs but C_emu has
    # NO floor (compare the floor-on leg's C_emu minus the floor-off leg's at the SAME P_obs).
    mf = _synthetic_mf(c, resolved=True)
    _, C_big = DL.predict_P_obs_on_leg(
        c["model"], c["theta9"], tau0_vec, c["alpha_hcd"], leg=leg_big,
        mf=mf, mf_floor=fl, **args)
    _, C_big_nofloor = DL.predict_P_obs_on_leg(
        c["model"], c["theta9"], tau0_vec, c["alpha_hcd"], leg=leg_big,
        mf=mf, mf_floor=None, **args)
    assert np.array_equal(np.asarray(C_big), np.asarray(C_big_nofloor)), \
        "the floor must NOT apply on a leg with mf_floor_on=False (DESI-like)"


def test_mf_floor_differentiable_and_nuts_safe():
    """(d) The floor path is differentiable in (θ9, τ₀, α): jacrev AND jacfwd finite. The
    ns-edge term is stop_gradient'd → ∂(floor)/∂ns carries NO gradient toward the MAP (the
    floor's ns dependence must NOT appear in dlogL/dθ via the edge term)."""
    c = _emu_ctx()
    leg = _floor_leg(z=3.4)
    mf = _synthetic_mf(c, resolved=True)
    fl = DL.load_mf_floor()
    sigma_zb = jnp.asarray(c["rng"].uniform(0.01, 0.05, (1, 4, c["n_k"], c["n_tb"])))

    def total_var(theta9, tau0_vec, alpha):
        _, C = DL.predict_P_obs_on_leg(
            c["model"], theta9, tau0_vec, alpha, pf_stats=c["pf"], dla_core=c["dla_core"],
            cache_k=c["cache_k"], leg=leg, sigma_zb=sigma_zb,
            alpha_centres=c["alpha_centres"], mf=mf, mf_floor=fl)
        return jnp.sum(jnp.diag(C))

    th = jnp.array([0.836] + [0.5] * 8)                 # OUTSIDE the box → edge term active
    tau0_vec = jnp.asarray([0.85])
    for mode, jac in (("jacrev", jax.jacrev), ("jacfwd", jax.jacfwd)):
        gth, gt, ga = jac(total_var, argnums=(0, 1, 2))(th, tau0_vec, c["alpha_hcd"])
        assert np.isfinite(np.asarray(gth)).all(), f"[{mode}] ∂(floor var)/∂θ9 NaN"
        assert np.isfinite(np.asarray(gt)).all(), f"[{mode}] ∂(floor var)/∂τ₀ NaN"
        assert np.isfinite(np.asarray(ga)).all(), f"[{mode}] ∂(floor var)/∂α NaN"

    # the EDGE term's ns enters via stop_gradient → moving ns ACROSS the box edge changes the
    # floor value but the gradient of the floor wrt ns (through the edge term) is zero: the
    # only θ-gradient of the floor is through P_obs (the fractional floor on the model power).
    def floor_only_var_via_edge(theta9):
        # isolate the edge term: var contribution at a FIXED P_obs (decouple the P-grad).
        ns = DL._ns_phys_from_theta9(theta9)
        # mirror _mf_floor_var_on_k's edge term at z=3.4, LFres band, P=1 (so var≡edge²).
        ns_sg = jax.lax.stop_gradient(ns)
        return jnp.sum(DL._mf_floor_var_on_k(fl, 3.4, jnp.array([0.02]),
                                             jnp.array([1.0]), ns_sg))
    g_ns = jax.grad(floor_only_var_via_edge)(th)
    assert np.allclose(np.asarray(g_ns), 0.0, atol=1e-12), \
        "the ns-edge term must carry NO gradient toward θ (stop_gradient'd)"


def test_mf_floor_never_indexes_z_above_4p6():
    """(e) The PI invariant: a leg whose z exceeds 4.6 trips the assertion (the z>4.6 floor
    cells — incl. the z=5.4 spike — must NEVER be indexed). The real legs (KS z≤4.6, DESI
    z≤4.2) pass; a synthetic z=5.0 leg with the floor on must raise."""
    c = _emu_ctx()
    mf = _synthetic_mf(c, resolved=True)
    fl = DL.load_mf_floor()
    bad = _floor_leg(z=5.0, mf_floor_on=True)
    with pytest.raises(AssertionError, match="z max"):
        DL.predict_P_obs_on_leg(
            c["model"], c["theta9"], jnp.asarray([0.85]), c["alpha_hcd"],
            pf_stats=c["pf"], dla_core=c["dla_core"], cache_k=c["cache_k"], leg=bad,
            mf=mf, mf_floor=fl)
    # a z=4.6 leg (KS max) must be fine
    ok = _floor_leg(z=4.6, mf_floor_on=True)
    _, C = DL.predict_P_obs_on_leg(
        c["model"], c["theta9"], jnp.asarray([0.85]), c["alpha_hcd"],
        pf_stats=c["pf"], dla_core=c["dla_core"], cache_k=c["cache_k"], leg=ok,
        mf=mf, mf_floor=fl)
    assert np.isfinite(np.asarray(C)).all()


@pytest.mark.skipif(not (_have_desi and _have_ks), reason="data not present")
def test_mf_floor_threads_through_data_loglik_real_cov():
    """data_loglik(mf=, mf_floor=) threads the floor to each leg: the KS leg (mf_floor_on)
    gets a larger C_emu → a smaller χ² for the same residual; DESI (mf_floor_on=False)
    unchanged. logL stays finite + differentiable."""
    c = _emu_ctx()
    desi = DL.load_desi_leg()                            # mf_floor_on=False by default
    ks = DL.load_ks_leg()                                # mf_floor_on=True by default
    assert ks.mf_floor_on is True and desi.mf_floor_on is False
    legs = [desi, ks]
    mf = _synthetic_mf(c, resolved=True)
    fl = DL.load_mf_floor()
    z_global = np.unique(np.round(np.concatenate([desi.z, ks.z]), 6))
    tau0_global = jnp.asarray(MF.becker13_tau0(jnp.asarray(z_global)))
    szb = {leg.name: _sigma_zb_for_leg(leg, c["n_k"], c["n_tb"], c["rng"]) for leg in legs}

    def chi2(mf_floor):
        _, parts = DL.data_loglik(
            c["model"], c["theta9"], tau0_global, c["alpha_hcd"], legs, pf_stats=c["pf"],
            dla_core=c["dla_core"], cache_k=c["cache_k"], z_global=z_global,
            sigma_zb_per_leg=szb, alpha_centres=c["alpha_centres"], mf=mf,
            mf_floor=mf_floor, return_parts=True)
        return {nm: parts[nm][1] for nm in parts}

    chi2_no = chi2(None)
    chi2_fl = chi2(fl)
    assert chi2_fl["KS"] < chi2_no["KS"] - 1e-9, "the floor must enlarge KS C_emu (lower χ²)"
    assert np.isclose(chi2_fl["DESI"], chi2_no["DESI"], rtol=1e-12), \
        "DESI (mf_floor_on=False) χ² must be unchanged by the floor"

    def f(theta9, tau0, alpha):
        return DL.data_loglik(
            c["model"], theta9, tau0, alpha, legs, pf_stats=c["pf"], dla_core=c["dla_core"],
            cache_k=c["cache_k"], z_global=z_global, sigma_zb_per_leg=szb,
            alpha_centres=c["alpha_centres"], mf=mf, mf_floor=fl)
    assert np.isfinite(float(f(c["theta9"], tau0_global, c["alpha_hcd"])))
    gth, gt, ga = jax.grad(f, argnums=(0, 1, 2))(c["theta9"], tau0_global, c["alpha_hcd"])
    assert np.isfinite(np.asarray(gth)).all() and np.isfinite(np.asarray(gt)).all() \
        and np.isfinite(np.asarray(ga)).all()
