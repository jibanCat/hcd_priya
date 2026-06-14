"""Phase-C Task 1 — ∂P/∂θ gradient-correctness gate (the HARD pre-gate to SBC).

Proves the assembled, JAX-pure P_obs(θ,τ₀,α) forward (hcd_analysis.emulator.predict)
is smooth and its autodiff Jacobian matches central finite differences end-to-end
(encoder -> SVD-basis HeadB -> exp reconstruction -> structural Tier-P -> HCD add-back),
i.e. HMC/NUTS will not hit a non-differentiable op or a NaN gradient.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_grad_fidelity.py -v
"""
import hcd_analysis.emulator  # noqa: F401  enables jax_enable_x64
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator.model import Emulator
from hcd_analysis.emulator.predict import predict_P_obs, predict_P_filt, predict_excess


def _tiny_model_and_stats(n_k=20, n_basis=8, seed=0):
    """A small randomly-initialised production-shaped Emulator + a P_filt norm dict."""
    model = Emulator(in_dim=10, n_k=n_k, n_basis=n_basis, key=jax.random.PRNGKey(seed))
    rng = np.random.default_rng(seed)
    pf_stats = {
        "mu_marg": jnp.asarray(rng.normal(-2.0, 0.5, (4, n_k))),
        "sig_marg": jnp.asarray(rng.uniform(0.5, 1.5, (4, n_k))),
        "sig_cosmo": jnp.asarray(rng.uniform(0.02, 0.10, (4, n_k))),
    }
    dla_core = jnp.asarray(rng.uniform(0.0, 0.5, n_k))   # DLA-core add-back (P_DLA^unf−P_DLA^filt)
    alpha_hcd = jnp.asarray(rng.uniform(0.0, 0.4, 3))    # effective per-class incidence
    return model, pf_stats, dla_core, alpha_hcd, n_k


def _central_fd(f, x, h=1e-3):
    """Central finite-diff Jacobian of f: R^n -> R^m, returns (m, n)."""
    x = np.asarray(x, float)
    cols = []
    for i in range(x.size):
        xp = x.copy(); xm = x.copy()
        xp[i] += h; xm[i] -= h
        cols.append((np.asarray(f(xp)) - np.asarray(f(xm))) / (2 * h))
    return np.stack(cols, axis=-1)  # (m, n)


def test_dPobs_dtheta_autodiff_matches_finite_diff():
    model, pf, dla_core, alpha, n_k = _tiny_model_and_stats()
    z_unit, tau0 = 0.5, 0.4

    def f(theta9):
        return predict_P_obs(model, theta9, z_unit, tau0, alpha, pf, dla_core)

    theta0 = jnp.full(9, 0.5)
    J_ad = np.asarray(jax.jacfwd(f)(theta0))          # (K, 9)
    J_fd = _central_fd(lambda t: f(jnp.asarray(t)), theta0, h=1e-3)  # (K, 9)
    # error relative to the PEAK gradient (floored denom) so derivative zero-crossings
    # — a pure finite-diff artifact, |J_fd|->0 — don't dominate (standard grad-check).
    denom = np.abs(J_fd) + 1e-3 * np.max(np.abs(J_fd))
    rel = np.abs(J_ad - J_fd) / denom
    assert np.isfinite(J_ad).all()
    assert np.median(rel) < 1e-4, f"median rel-err {np.median(rel):.2e}"
    assert np.max(rel) < 1e-2, f"max rel-err {np.max(rel):.2e}"


def test_dPobs_dtau0_and_dalpha_autodiff_matches_finite_diff():
    model, pf, dla_core, alpha, n_k = _tiny_model_and_stats(seed=1)
    z_unit = 0.5
    theta0 = jnp.full(9, 0.5)

    # d/dtau0
    def g(tau0):
        return predict_P_obs(model, theta0, z_unit, tau0, alpha, pf, dla_core)
    Jt_ad = np.asarray(jax.jacfwd(g)(0.4))            # (K,)
    Jt_fd = (np.asarray(g(0.4 + 1e-3)) - np.asarray(g(0.4 - 1e-3))) / 2e-3
    rel_t = np.abs(Jt_ad - Jt_fd) / (np.abs(Jt_fd) + 1e-8 * np.max(np.abs(Jt_fd)))
    assert np.isfinite(Jt_ad).all() and np.median(rel_t) < 1e-4

    # d/dalpha_c == the EXCESS template (P_c − P_clean), the corrected HCD object.
    def h(a):
        return predict_P_obs(model, theta0, z_unit, 0.4, a, pf, dla_core)
    Ja_ad = np.asarray(jax.jacfwd(h)(alpha))          # (K, 3)
    excess = np.asarray(predict_excess(model, theta0, z_unit, 0.4, pf, dla_core))  # (3,K)
    assert np.allclose(Ja_ad, excess.T, rtol=1e-9, atol=1e-12), \
        "∂P_obs/∂α_c must equal the corrected excess template (P_c − P_clean)"
    # KEY: the LLS excess is NON-ZERO now (the old Δ_LLS≡0 bug is fixed)
    assert np.sqrt(np.mean(excess[0] ** 2)) > 1e-6, "LLS excess must be non-zero"


def test_P_obs_finite_across_unit_cube_sweep():
    """No NaN/Inf in P_obs OR its θ-gradient anywhere in [0,1]^9 (HMC won't blow up)."""
    model, pf, dla_core, alpha, n_k = _tiny_model_and_stats(seed=2)
    rng = np.random.default_rng(7)

    def f(theta9, z_unit, tau0):
        return predict_P_obs(model, theta9, z_unit, tau0, alpha, pf, dla_core)

    grad_norm = jax.jit(lambda th, z, t: jnp.linalg.norm(jax.jacfwd(f)(th, z, t)))
    for _ in range(50):
        th = jnp.asarray(rng.uniform(0, 1, 9))
        z = float(rng.uniform(0, 1)); t = float(rng.uniform(0.1, 1.6))
        val = f(th, z, t)
        assert np.isfinite(np.asarray(val)).all()
        assert np.isfinite(float(grad_norm(th, z, t)))


def test_predict_P_filt_matches_numpy_reconstruct():
    """The jnp reconstruct mirror must equal data.reconstruct_P_filt bit-close."""
    from hcd_analysis.emulator.data import reconstruct_P_filt
    model, pf, dla_core, alpha, n_k = _tiny_model_and_stats(seed=3)
    x = jnp.concatenate([jnp.full(9, 0.3), jnp.asarray([0.6])])
    pred = model(x, 0.5)
    P_jax = np.asarray(predict_P_filt(model, jnp.full(9, 0.3), 0.6, 0.5, pf))
    pf_np = {k: np.asarray(v) for k, v in pf.items()}
    P_np = reconstruct_P_filt(np.asarray(pred["P_filt_base"]),
                              np.asarray(pred["P_filt_resid"]), pf_np)
    assert np.allclose(P_jax, P_np, rtol=1e-10, atol=1e-12)
