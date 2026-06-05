"""Phase-C — cross-class (4×4) C_emu upgrade tests.

The diagonal C_emu (``emu_var = Σ_c coef_c²·σ_c²·P_c²``) under-sizes the held-out P_obs
residual ~2.5× because the 4 per-class residuals come from ONE network and are coherently
CORRELATED. The fix upgrades C_emu to a cross-class 4×4 block:
  ``emu_var = Σ_cc' coef_c·coef_c'·ρ_cc'·P_c·P_c'``,  ρ a sample covariance (SPD).
This pins the contract:
  (a) ρ = diag(σ²) (zero off-diagonals) EXACTLY equals the diagonal Σ coef²σ²P² path;
  (b) a POSITIVE off-diagonal ρ_cc' INCREASES emu_var when the coupled coefs share a sign;
  (c) emu_var ≥ 0 ALWAYS (the SPD guarantee: coefᵀ(P∘ρ∘P)coef ≥ 0);
  (d) ∂emu_var/∂α is finite (the α-derivative now carries the cross terms).

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_xclass_cemu.py -q
"""
import hcd_analysis.emulator  # noqa: F401  enables x64
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator.model import Emulator
from hcd_analysis.emulator import inference as I
from hcd_analysis.emulator.likelihood import (
    sigma_at_tau0, rho_at_tau0, _KIM_AMP, _KIM_SLOPE,
)


def _ctx(n_k=10, n_basis=6, n_tb=4, seed=0):
    """Synthetic single-z forward-model fixture (mirrors test_likelihood_driver._ctx)."""
    rng = np.random.default_rng(seed)
    model = Emulator(in_dim=10, n_k=n_k, n_basis=n_basis, key=jax.random.PRNGKey(seed))
    pf = {"mu_marg": jnp.asarray(rng.normal(-2, 0.5, (4, n_k))),
          "sig_marg": jnp.asarray(rng.uniform(0.5, 1.5, (4, n_k))),
          "sig_cosmo": jnp.asarray(rng.uniform(0.02, 0.1, (4, n_k)))}
    return dict(
        model=model, pf=pf, n_k=n_k, n_tb=n_tb,
        dla_core=jnp.asarray(rng.uniform(0.0, 0.5, n_k)),
        alpha_hcd=jnp.asarray(rng.uniform(0.05, 0.4, 3)),
        sigma_zb=jnp.asarray(rng.uniform(0.01, 0.05, (4, n_k, n_tb))),
        cosmic=jnp.asarray(rng.uniform(1.0, 4.0, n_k)),
        alpha_centres=jnp.asarray([0.66, 0.83, 1.15, 1.33]),
        flag=jnp.zeros(n_k, bool), theta9=jnp.full(9, 0.5), z=3.0, z_unit=0.5)


def _diag_rho(sigma_zb):
    """ρ = diag(σ²): the (4,4,K,Tb) block with diagonal σ² and zero off-diagonals — the
    construction that recovers the diagonal C_emu exactly."""
    n_c, K, Tb = sigma_zb.shape
    rho = jnp.zeros((n_c, n_c, K, Tb))
    for c in range(n_c):
        rho = rho.at[c, c].set(sigma_zb[c] ** 2)
    return rho


def _emu_var(c, sigma_zb=None, rho_zb=None, tau0=0.40):
    """The diagonal of (C − cosmic) = the emu_var the assembly built (diagonal OR x-class)."""
    sig = c["sigma_zb"] if sigma_zb is None else sigma_zb
    _, C = I.predict_P_obs_and_cov_single_z(
        c["model"], c["theta9"], c["z_unit"], c["z"], tau0, c["alpha_hcd"],
        pf_stats=c["pf"], sigma_zb=sig, alpha_centres=c["alpha_centres"],
        cosmic_cov=c["cosmic"], dla_core=c["dla_core"], dla_shot_flag=c["flag"],
        rho_zb=rho_zb)
    return np.diag(np.asarray(C)) - np.asarray(c["cosmic"])


# ---------------------------------------------------------------------------
# (a) ρ = diag(σ²) EXACTLY equals the diagonal Σ coef²σ²P² path.
# ---------------------------------------------------------------------------
def test_xclass_diag_equals_diagonal_path_at_band_centre():
    """At a τ₀ band CENTRE the τ₀-interp is the identity, so ρ=diag(σ²) and the diagonal
    σ path see the SAME per-class variance → emu_var (and the full C) match to machine ε.
    (Off-centre they differ only by interp-then-square vs interp(σ²); see the next test.)"""
    c = _ctx()
    rho = _diag_rho(c["sigma_zb"])
    for tb in range(c["n_tb"]):
        tau0 = float(c["alpha_centres"][tb]) * _KIM_AMP * (1 + c["z"]) ** _KIM_SLOPE
        _, C_diag = I.predict_P_obs_and_cov_single_z(
            c["model"], c["theta9"], c["z_unit"], c["z"], tau0, c["alpha_hcd"],
            pf_stats=c["pf"], sigma_zb=c["sigma_zb"], alpha_centres=c["alpha_centres"],
            cosmic_cov=c["cosmic"], dla_core=c["dla_core"], dla_shot_flag=c["flag"])
        _, C_x = I.predict_P_obs_and_cov_single_z(
            c["model"], c["theta9"], c["z_unit"], c["z"], tau0, c["alpha_hcd"],
            pf_stats=c["pf"], sigma_zb=c["sigma_zb"], alpha_centres=c["alpha_centres"],
            cosmic_cov=c["cosmic"], dla_core=c["dla_core"], dla_shot_flag=c["flag"],
            rho_zb=rho)
        assert np.array_equal(np.asarray(C_diag), np.asarray(C_x)), \
            f"x-class with ρ=diag(σ²) must EXACTLY equal the diagonal path at centre tb={tb}"


def test_xclass_diag_equals_diagonal_path_tau0_flat():
    """With a τ₀-FLAT σ (constant over Tb), interp(σ)²==interp(σ²) for ANY τ₀, so ρ=diag(σ²)
    equals the diagonal path at an arbitrary (off-centre) τ₀ as well — isolating that the
    ONLY off-centre difference is the interp-order, not the cross-class assembly."""
    c = _ctx()
    sig_flat = jnp.broadcast_to(c["sigma_zb"][:, :, :1], c["sigma_zb"].shape)  # flat over Tb
    rho = _diag_rho(sig_flat)
    tau0 = 0.37  # arbitrary, between centres
    diag = _emu_var(c, sigma_zb=sig_flat, tau0=tau0)
    xcl = _emu_var(c, sigma_zb=sig_flat, rho_zb=rho, tau0=tau0)
    assert np.allclose(diag, xcl, rtol=1e-12, atol=1e-18), \
        "ρ=diag(σ²) must equal the diagonal path at any τ₀ when σ is τ₀-flat"


# ---------------------------------------------------------------------------
# (b) a POSITIVE off-diagonal ρ_cc' INCREASES emu_var when coupled coefs share a sign.
# ---------------------------------------------------------------------------
def test_positive_offdiagonal_increases_emu_var():
    """coef = [1−Σα, α_LLS, α_subDLA, α_DLA] with α>0 ⇒ ALL coefs > 0 (same sign). Adding a
    POSITIVE off-diagonal ρ_cc' adds 2·coef_c·coef_c'·ρ_cc'·P_c·P_c' > 0 to emu_var — the
    class-coupling INFLATION that fixes the under-sizing."""
    c = _ctx()
    base = _diag_rho(c["sigma_zb"])
    var_base = _emu_var(c, rho_zb=base)
    # add a positive subDLA(2)–DLA(3) coupling: ρ_23 = +0.6·√(ρ_22·ρ_33) (a valid correlation)
    coup = 0.6 * jnp.sqrt(c["sigma_zb"][2] ** 2 * c["sigma_zb"][3] ** 2)        # (K,Tb)
    rho_pos = base.at[2, 3].set(coup).at[3, 2].set(coup)
    var_pos = _emu_var(c, rho_zb=rho_pos)
    dvar = var_pos - var_base
    assert np.all(dvar >= -1e-15), "positive coupling must not decrease emu_var (coefs same sign)"
    assert np.any(dvar > 1e-12), "positive coupling must strictly increase some emu_var bins"


def test_negative_offdiagonal_decreases_emu_var():
    """The converse / sign check: a NEGATIVE off-diagonal (anti-correlated classes) DECREASES
    emu_var when the coupled coefs share a sign — the coupling is signed, not an |inflation|."""
    c = _ctx()
    base = _diag_rho(c["sigma_zb"])
    var_base = _emu_var(c, rho_zb=base)
    coup = -0.6 * jnp.sqrt(c["sigma_zb"][1] ** 2 * c["sigma_zb"][2] ** 2)       # (K,Tb)
    rho_neg = base.at[1, 2].set(coup).at[2, 1].set(coup)
    var_neg = _emu_var(c, rho_zb=rho_neg)
    assert np.any(var_neg < var_base - 1e-12), "negative coupling must lower some emu_var bins"


# ---------------------------------------------------------------------------
# (c) emu_var ≥ 0 ALWAYS (SPD guarantee).
# ---------------------------------------------------------------------------
def test_emu_var_nonneg_for_spd_rho():
    """ρ a sample covariance ⇒ SPD ⇒ emu_var = coefᵀ(P∘ρ∘P)coef ≥ 0 for ANY coef. Build a
    random SPD ρ per (k,Tb) (AAᵀ) and check emu_var ≥ 0 across many random α (hence coef)."""
    c = _ctx(seed=3)
    rng = np.random.default_rng(7)
    K, Tb = c["n_k"], c["n_tb"]
    # random SPD 4×4 per (k,Tb): rho[:,:,k,tb] = A Aᵀ / scale (positive eigenvalues)
    rho = np.zeros((4, 4, K, Tb))
    for k in range(K):
        for tb in range(Tb):
            A = rng.normal(0, 0.03, (4, 4))
            rho[:, :, k, tb] = A @ A.T
    rho = jnp.asarray(rho)
    for s in range(8):
        a = jnp.asarray(rng.uniform(-0.3, 0.6, 3))    # incl. coefs of mixed sign
        cc = dict(c, alpha_hcd=a)
        var = _emu_var(cc, rho_zb=rho)
        assert np.all(var >= -1e-12), f"emu_var must be ≥0 for SPD ρ (seed {s}); min {var.min():.3e}"


def test_emu_var_nonneg_at_strong_correlation():
    """Even the EXTREME ρ (correlation ≈ ±1, the rank-deficient edge of SPD) keeps emu_var ≥ 0
    — the boundary case the production pooled ρ approaches (clean–LLS corr ≈ +0.93)."""
    c = _ctx(seed=5)
    base = _diag_rho(c["sigma_zb"])
    rho = base
    for (i, j) in [(0, 1), (0, 2), (1, 2), (2, 3)]:
        coup = 0.999 * jnp.sqrt(c["sigma_zb"][i] ** 2 * c["sigma_zb"][j] ** 2)
        rho = rho.at[i, j].set(coup).at[j, i].set(coup)
    var = _emu_var(c, rho_zb=rho)
    assert np.all(var >= -1e-12), f"emu_var must stay ≥0 at near-±1 correlation; min {var.min():.3e}"


# ---------------------------------------------------------------------------
# (d) ∂emu_var/∂α finite (the α-derivative now carries the cross terms).
# ---------------------------------------------------------------------------
def test_demu_var_dalpha_finite_and_carries_cross_terms():
    c = _ctx(seed=2)
    base = _diag_rho(c["sigma_zb"])
    coup = 0.5 * jnp.sqrt(c["sigma_zb"][1] ** 2 * c["sigma_zb"][2] ** 2)
    rho = base.at[1, 2].set(coup).at[2, 1].set(coup)

    def total_emu(a):
        _, C = I.predict_P_obs_and_cov_single_z(
            c["model"], c["theta9"], c["z_unit"], c["z"], 0.40, a,
            pf_stats=c["pf"], sigma_zb=c["sigma_zb"], alpha_centres=c["alpha_centres"],
            cosmic_cov=c["cosmic"], dla_core=c["dla_core"], dla_shot_flag=c["flag"],
            rho_zb=rho)
        return jnp.sum(jnp.diag(C))

    g = jax.grad(total_emu)(c["alpha_hcd"])
    assert np.all(np.isfinite(np.asarray(g))), "∂emu_var/∂α must be finite (x-class path)"
    assert np.asarray(g).dtype == np.float64, "x64 must be asserted"
    # the cross term materially changes the α-grad vs the diagonal-only ρ (off-diagonals matter)
    g_diag = jax.grad(lambda a: jnp.sum(jnp.diag(
        I.predict_P_obs_and_cov_single_z(
            c["model"], c["theta9"], c["z_unit"], c["z"], 0.40, a,
            pf_stats=c["pf"], sigma_zb=c["sigma_zb"], alpha_centres=c["alpha_centres"],
            cosmic_cov=c["cosmic"], dla_core=c["dla_core"], dla_shot_flag=c["flag"],
            rho_zb=base)[1])))(c["alpha_hcd"])
    assert not np.allclose(np.asarray(g), np.asarray(g_diag)), \
        "the off-diagonal ρ must change ∂emu_var/∂α (cross terms are live)"


def test_dlogL_dtau0_finite_xclass_path():
    """The full driver's ∂logL/∂τ₀ stays finite on the cross-class path (the τ₀-interp of
    the 4×4 block must not poison the gradient — the rho_at_tau0 NaN-guard analog)."""
    c = _ctx(seed=4)
    base = _diag_rho(c["sigma_zb"])
    coup = 0.4 * jnp.sqrt(c["sigma_zb"][0] ** 2 * c["sigma_zb"][1] ** 2)
    rho = base.at[0, 1].set(coup).at[1, 0].set(coup)
    from hcd_analysis.emulator.predict import predict_P_obs
    P_obs = np.asarray(predict_P_obs(c["model"], c["theta9"], c["z_unit"], 0.40,
                                     c["alpha_hcd"], c["pf"], c["dla_core"]))
    P_data = jnp.asarray(P_obs + np.random.default_rng(0).normal(0, 0.3, c["n_k"]))
    valid_k = jnp.ones(c["n_k"], bool)

    def f(t0):
        return I.log_lik_single_z(
            c["model"], c["theta9"], c["z_unit"], c["z"], t0, c["alpha_hcd"],
            pf_stats=c["pf"], sigma_zb=c["sigma_zb"], alpha_centres=c["alpha_centres"],
            cosmic_cov=c["cosmic"], P_data=P_data, dla_core=c["dla_core"],
            dla_shot_flag=c["flag"], valid_k=valid_k, rho_zb=rho)

    for t0 in (0.30, 0.40, 0.55):
        assert np.isfinite(float(f(t0)))
        assert np.isfinite(float(jax.grad(f)(t0))), f"∂logL/∂τ₀ NaN at τ₀={t0} (x-class)"


def test_rho_at_tau0_recovers_band_value_at_centre():
    """rho_at_tau0 reduces to the band slice at a centre (the 4×4 analog of the
    sigma_at_tau0 band-recovery pin)."""
    c = _ctx()
    rho_zb = _diag_rho(c["sigma_zb"])
    # add some off-diagonal structure so the test is not trivially diagonal
    rho_zb = rho_zb.at[1, 2].set(0.01 * c["sigma_zb"][1]).at[2, 1].set(0.01 * c["sigma_zb"][1])
    for tb, a in enumerate(np.asarray(c["alpha_centres"])):
        tau0 = float(a * _KIM_AMP * (1 + c["z"]) ** _KIM_SLOPE)
        got = np.asarray(rho_at_tau0(rho_zb, c["alpha_centres"], c["z"], tau0))
        assert np.allclose(got, np.asarray(rho_zb)[:, :, :, tb], rtol=1e-6)
