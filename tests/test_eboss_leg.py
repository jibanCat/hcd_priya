"""eBOSS DR14 (Chabanier+2019) P1D leg — converter + (later) loader tests.

The converter (scripts/convert_eboss_dr14_p1d.py) reconstructs the BLOCK-DIAGONAL covariance from
the 3 raw .dat files. The load-bearing risk is the Pk1D_cor.dat block-parse: a mis-paired z-block
would still be symmetric/PD and still pass diag(cov)==σ² (since diag(corr)≡1) — so the ONLY guard is
a known OFF-DIAGONAL golden, cov[0,1] = σ0·σ1·corr[0,1] with corr[0,1]=0.198391 (z=2.2 block, the
raw Pk1D_cor.dat line-4 col-2). This file pins exactly that, plus the structure/PD/cross-z checks.
The loader (load_eboss_leg) tests are added once the PI confirms the science flags.
"""
import os

import numpy as np
import pytest

EBOSS_NPZ = "/home/mfho/data/eboss_dr14_p1d/eboss_dr14_p1d.npz"
_have = os.path.exists(EBOSS_NPZ)
NZ, NK = 13, 35


@pytest.mark.skipif(not _have, reason="eBOSS npz not built (run scripts/convert_eboss_dr14_p1d.py)")
def test_eboss_npz_grid_and_zmajor():
    d = np.load(EBOSS_NPZ)
    assert int(d["nz"]) == NZ and int(d["nk"]) == NK
    assert d["z"].shape == (NZ * NK,) and d["k"].shape == (NZ * NK,)
    assert bool(d["row_is_zmajor"]) is True
    assert np.allclose(d["z_unique"], np.arange(2.2, 4.6 + 1e-9, 0.2))
    # k angular [s/km], native eBOSS band, NOT divided by 2π
    assert abs(float(d["k"].min()) - 0.001084) < 1e-6
    assert abs(float(d["k"].max()) - 0.019512) < 1e-6
    assert abs(float(d["plya"][0]) - 19.2561) < 1e-3   # z=2.2, k=0.001084 row


@pytest.mark.skipif(not _have, reason="eBOSS npz not built")
def test_eboss_cov_block_diagonal_pd_and_diag():
    d = np.load(EBOSS_NPZ)
    cov, sigma, z = np.asarray(d["cov"]), np.asarray(d["sigma"]), np.asarray(d["z"])
    assert cov.shape == (NZ * NK, NK * NZ)
    assert np.allclose(cov, cov.T)                          # symmetric
    assert np.allclose(np.diag(cov), sigma ** 2)            # diag == σ²
    # block-diagonal: ZERO cross-z covariance
    assert np.all(cov[np.ix_(z != z[0], z == z[0])] == 0.0)
    # each per-z 35×35 diagonal block PD
    for iz in range(NZ):
        b = cov[iz * NK:(iz + 1) * NK, iz * NK:(iz + 1) * NK]
        assert np.linalg.eigvalsh(b).min() > 0


@pytest.mark.skipif(not _have, reason="eBOSS npz not built")
def test_eboss_offdiagonal_golden_proves_block_pairing():
    # the ONLY check that proves the cor-block landed on the right z with the right σ pairing:
    # cov[0,1] = σ0·σ1·corr[0,1], corr[0,1]=0.198391 (raw Pk1D_cor.dat). A mis-paired block would
    # still pass the diag/PD/symmetry checks but FAIL this.
    d = np.load(EBOSS_NPZ)
    s = np.asarray(d["sigma"])
    assert abs(float(d["cov"][0, 1]) - s[0] * s[1] * 0.198391) < 1e-12
    assert abs(float(d["corr"][0, 1]) - 0.198391) < 1e-9    # raw correlation preserved


# ----------------------------------------------------------------------------- #
#  load_eboss_leg (the DataLeg builder)
# ----------------------------------------------------------------------------- #
import hcd_analysis.emulator  # noqa: E402  x64
from hcd_analysis.emulator import data_likelihood as DL  # noqa: E402


@pytest.mark.skipif(not _have, reason="eBOSS npz not built")
def test_eboss_loader_shape_grid_and_flags():
    leg = DL.load_eboss_leg()
    assert leg.name == "eBOSS"
    assert leg.n_z == NZ
    N = leg.P_data.shape[0]
    assert N == NZ * NK                                     # no cut by default
    assert leg.k.shape == (N,) and leg.z_row.shape == (N,) and leg.z_idx.shape == (N,)
    assert leg.C_data.shape == (N, N)
    assert int(leg.n_per_z.sum()) == N and np.all(leg.n_per_z == NK)
    # PI-confirmed flags
    assert leg.metals_on is True
    assert leg.resolution_on is False
    assert leg.mf_floor_on is False
    assert float(leg.dla_forward_frac) == 0.0


@pytest.mark.skipif(not _have, reason="eBOSS npz not built")
def test_eboss_loader_cov_spd_and_block_diagonal_preserved():
    leg = DL.load_eboss_leg()
    C = np.asarray(leg.C_data)
    assert np.allclose(C, C.T)
    assert np.linalg.eigvalsh(C).min() > 0                  # SPD
    # the kept covariance is still block-diagonal (no cut → full block structure)
    zr = np.asarray(leg.z_row)
    assert np.all(C[np.ix_(zr != zr[0], zr == zr[0])] == 0.0)


@pytest.mark.skipif(not _have, reason="eBOSS npz not built")
def test_eboss_loader_z_unit_and_angular_k():
    leg = DL.load_eboss_leg()
    assert np.allclose(leg.z_unit, (leg.z - 2.0) / 3.4)     # the cache encoder convention
    # angular k [s/km], NOT /2π: matches the raw Pk1D_data.dat k column on the kept rows
    raw = np.loadtxt("/home/mfho/lya_emulator_full/lyaemu/data/boss_dr14_data/Pk1D_data.dat")
    assert np.allclose(np.sort(np.unique(leg.k)), np.sort(np.unique(raw[:, 1])))
    assert leg.k.min() > 1e-3 and leg.k.max() < 0.02


@pytest.mark.skipif(not _have, reason="eBOSS npz not built")
def test_eboss_loader_cuts():
    # z-cut
    leg = DL.load_eboss_leg(z_lo=3.0)
    assert leg.z.min() >= 3.0 - 1e-6
    assert leg.P_data.shape[0] == int((np.arange(2.2, 4.6 + 1e-9, 0.2) >= 3.0).sum()) * NK
    # k-cut (default k_max is a no-op; a tight cut drops high-k bins)
    leg2 = DL.load_eboss_leg(k_max=0.01)
    assert leg2.k.max() <= 0.01
    assert DL.load_eboss_leg().k.max() > 0.019                # default keeps the full band


@pytest.mark.skipif(not _have, reason="eBOSS npz not built")
def test_eboss_binding_finite_and_spd():
    # the emulator forward binds to the eBOSS leg → finite (P_model, C_total), SPD, finite grad.
    import jax
    import jax.numpy as jnp
    from hcd_analysis.emulator.model import Emulator
    rng = np.random.default_rng(0)
    n_k = 172
    model = Emulator(in_dim=10, n_k=n_k, n_basis=12, key=jax.random.PRNGKey(0))
    pf = {"mu_marg": jnp.asarray(rng.normal(-2, 0.5, (4, n_k))),
          "sig_marg": jnp.asarray(rng.uniform(0.5, 1.5, (4, n_k))),
          "sig_cosmo": jnp.asarray(rng.uniform(0.02, 0.1, (4, n_k)))}
    cache_k = jnp.asarray(np.linspace(4.04e-4, 0.0694, n_k))   # spans eBOSS [0.0011,0.0195]
    dla_core = jnp.asarray(rng.uniform(0, 0.5, n_k))
    leg = DL.load_eboss_leg()
    tau0 = jnp.full(leg.n_z, 0.9)
    theta9 = jnp.full(9, 0.5)
    alpha = jnp.asarray([0.06, 0.02, 0.003])

    def ll(th):
        P, C = DL.predict_P_obs_on_leg(model, th, tau0, alpha, pf_stats=pf, dla_core=dla_core,
                                       cache_k=cache_k, leg=leg)
        return jnp.sum(P) + jnp.trace(C)
    P, C = DL.predict_P_obs_on_leg(model, theta9, tau0, alpha, pf_stats=pf, dla_core=dla_core,
                                   cache_k=cache_k, leg=leg)
    Pn, Cn = np.asarray(P), np.asarray(C)
    assert Pn.shape == (NZ * NK,) and Cn.shape == (NZ * NK, NZ * NK)
    assert np.all(np.isfinite(Pn)) and np.all(np.isfinite(Cn))
    assert np.allclose(Cn, Cn.T) and np.linalg.eigvalsh(Cn).min() > 0   # SPD
    g = jax.grad(ll)(theta9)
    assert np.all(np.isfinite(np.asarray(g)))


# ----------------------------------------------------------------------------- #
#  a_SiIII metal nuisance wiring (opt-in ctx.sample_metals; golden-guarded off)
# ----------------------------------------------------------------------------- #
def test_draws_matrix_appends_a_siiii_only_when_present():
    from hcd_analysis.emulator import closure_legb as C
    L, nz = 5, 3
    base = {"theta_unit": np.zeros((L, 9)), "tau0_vec": np.zeros((L, nz)),
            "alpha_lls": np.zeros(L), "alpha_subdla": np.zeros(L), "alpha_dla": np.zeros(L)}
    keep = np.ones(nz, bool)
    d0 = C._draws_matrix(dict(base), keep)
    d1 = C._draws_matrix(dict(base, a_SiIII=np.full(L, 0.04)), keep)
    assert d0.shape[1] == 9 + nz + 3                 # no a_SiIII column when absent
    assert d1.shape[1] == 9 + nz + 3 + 1             # appended LAST when present
    assert np.allclose(d1[:, -1], 0.04)


@pytest.mark.skipif(not _have, reason="eBOSS npz not built")
@pytest.mark.parametrize("marg_zslope", [False, True])   # both real-fit configs: metals × zslope
def test_legb_model_priors_only_site_order_match(marg_zslope):
    # the fast-postprocess constrain_fn relies on _legb_priors_only having the SAME sample sites IN
    # THE SAME ORDER as _legb_model. With sample_metals=True, a_SiIII must appear in both, last —
    # and the zslope block (when marginalized) must not reorder relative to it.
    import jax
    import numpyro
    from numpyro import handlers
    from hcd_analysis.emulator import closure_legb as C
    ctx, _ = C.build_legb_ctx(ckpt="/home/mfho/hcd_priya/checkpoints/final_fold6",
                              with_eboss=True, sample_metals=True)
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "eBOSS"],
                       marginalize_zslope=marg_zslope)
    truth = C.make_truth_from_sim(C.load_cache(C.CACHE_PATH),
                                  C.held_out_sims(C.load_cache(C.CACHE_PATH), fold=6)[0][0], fold=6)
    mock_legs, _, _ = C.make_legb_mock(ctx, truth, jax.random.PRNGKey(0))
    core = C._mock_core_per_leg(ctx, truth)

    def sites(model, *a):
        tr = handlers.trace(handlers.seed(model, jax.random.PRNGKey(1))).get_trace(*a)
        return [k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")]
    s_model = sites(C._legb_model, ctx, mock_legs, core)
    s_prior = sites(C._legb_priors_only, ctx)
    assert "a_SiIII" in s_model and s_model[-1] == "a_SiIII"
    assert s_model == s_prior, f"site-order mismatch: model {s_model} vs priors {s_prior}"


def test_eboss_si_cert_arms_inject_and_sample():
    import sys as _sys
    _sys.path.insert(0, "/home/mfho/hcd_priya/scripts")
    import run_stepA  # noqa
    cfg = run_stepA.build_config()
    for base in ("E_f5_si", "E_f6_si", "E_f7_si"):
        chs = [c for c in cfg if c["mock_id"] == base]
        assert len(chs) == 4
        for c in chs:
            assert c["sample_metals"] is True
            assert abs(float(c["inject_a_siiii"]) - 0.045) < 1e-9
            assert c["survey"] == "eBOSS"
