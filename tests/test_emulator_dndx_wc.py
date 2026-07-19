import jax
jax.config.update("jax_enable_x64", True)  # exact sum-to-1 (1e-12); emulator path is float64

import numpy as np, jax.numpy as jnp
from hcd_analysis.emulator.dndx_wc import w_c_from_mu


def test_wc_sums_to_one_and_matches_paper_point():
    mu = jnp.array([0.231, 0.064, 0.032])           # (LLS, subDLA, DLA)
    w = w_c_from_mu(mu)                              # (clean, LLS, subDLA, DLA)
    assert abs(float(w.sum()) - 1.0) < 1e-12
    assert np.allclose(np.array(w), [0.72121, 0.18715, 0.06027, 0.03136], atol=5e-4)


def test_wc_batched_shape_and_sum():
    mu = jnp.array([[0.231, 0.064, 0.032],
                    [0.10, 0.05, 0.01],
                    [0.0, 0.0, 0.0]])               # (3, 3)
    w = w_c_from_mu(mu)                              # (3, 4)
    assert w.shape == (3, 4)
    assert np.allclose(np.array(w.sum(axis=-1)), 1.0, atol=1e-12)
    # all-zero mu -> fully clean
    assert np.allclose(np.array(w[2]), [1.0, 0.0, 0.0, 0.0], atol=1e-12)


from hcd_analysis.emulator.dndx_wc import w_c_corrected, delta_c


def test_delta_c_correction_keeps_sum_to_one_and_is_differentiable():
    dndx = jnp.array([0.36, 0.10, 0.05]); Xbar = jnp.array(0.642); z = jnp.array(3.0)
    w = w_c_corrected(dndx, Xbar, z)
    assert w.shape == (4,)
    assert abs(float(w.sum()) - 1.0) < 1e-10            # renormalised after correction
    g = jax.grad(lambda d: w_c_corrected(d, Xbar, z)[3])(dndx)   # d w_DLA / d dndx
    assert jnp.all(jnp.isfinite(g))


def test_delta_c_polynomial_values():
    # delta_c(z) is deg-2 per class (clean,LLS,subDLA,DLA), np.polyval order coeffs.
    import numpy as np
    z = 3.0
    out = np.array(delta_c(jnp.array(z)))
    assert out.shape == (4,)
    # clean coeff [-0.00446, 0.02941, -0.04148] at z=3 -> small (~ -0.0% to +1%)
    assert np.all(np.abs(out) < 0.05)


def test_wc_corrected_batched_z_shape_and_sum_at_extremes():
    # batched z (the likelihood/vmap path): (N,) z -> (N,4), sum-to-1 at z-extremes
    # and at tiny dndx (where the correction is largest relative to w0).
    z = jnp.array([2.0, 3.0, 5.4])                      # full PRIYA z-range
    dndx = jnp.array([[1e-9, 1e-9, 1e-9],               # ~all-clean
                      [0.36, 0.10, 0.05],               # coupling-note point
                      [0.0, 0.0, 0.0]])                 # exact all-zero
    Xbar = jnp.full((3,), 0.642)
    d = delta_c(z)
    assert d.shape == (3, 4)
    w = w_c_corrected(dndx, Xbar, z)
    assert w.shape == (3, 4)
    assert np.allclose(np.array(w.sum(axis=-1)), 1.0, atol=1e-10)
    # all-zero dndx (mu=0 -> w_clean=1) renorms [1*(1+d_clean),0,0,0] -> [1,0,0,0]
    assert np.allclose(np.array(w[2]), [1.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_wc_corrected_grad_through_continuous_z_is_finite():
    # the likelihood differentiates the polynomial z-path; pin it (and the all-zero
    # dndx edge, where renorm is 1/(1+d_clean) -> a real 0/0 risk if a class vanished).
    dndx = jnp.array([0.36, 0.10, 0.05]); Xbar = jnp.array(0.642); z = jnp.array(3.0)
    gz = jax.grad(lambda zz: w_c_corrected(dndx, Xbar, zz)[3])(z)      # d w_DLA / d z
    assert jnp.isfinite(gz)
    gz0 = jax.grad(lambda zz: w_c_corrected(jnp.zeros(3), Xbar, zz)[0])(z)  # all-zero edge
    assert jnp.isfinite(gz0)


# --- Task 3: alpha_c(z) PW14 forward + analytic M0-inverse -----------------------
from hcd_analysis.emulator.dndx_wc import (
    dndx_powerlaw, alpha_from_dndx_law, alpha_to_dndx,
    Z_FIT_LO, Z_FIT_HI, DELTA_C_RESID_STD,
)


def test_alpha_to_dndx_exact_inverse_of_wc():
    # Telescoping inverse must exactly invert w_c_from_mu on the HCD classes.
    rng = np.random.default_rng(0)
    mu = jnp.array(rng.uniform(0.0, 1.5, size=(64, 3)))      # mu>0, (N,3)
    w = w_c_from_mu(mu)                                       # (N,4)
    Xbar = jnp.full((64,), 0.642); z = jnp.full((64,), 3.0)
    mu_rec = alpha_to_dndx(w[..., 1:], Xbar, z, apply_delta=False) * Xbar[..., None]
    assert np.allclose(np.array(mu_rec), np.array(mu), atol=1e-10, rtol=0)


def test_forward_inverse_roundtrip_with_delta():
    # forward law -> alpha_c(z) -> inverse recovers dN/dX up to the renorm caveat.
    A = jnp.array([0.5, 0.12, 0.04]); gamma = jnp.array([1.0, 0.8, 0.5])
    Xbar = jnp.array(0.642); z = jnp.array(3.0)
    dndx0 = dndx_powerlaw(z, A, gamma)                       # (3,)
    alpha = alpha_from_dndx_law(A, gamma, Xbar, z)           # (3,)
    dndx_rec = alpha_to_dndx(alpha, Xbar, z, apply_delta=True)
    # Exact only up to the 4-class renorm factor (small-HCD regime ~ sub-percent here).
    assert np.allclose(np.array(dndx_rec), np.array(dndx0), rtol=0.03, atol=0)


def test_delta_c_clamped_outside_fit_range():
    # deg-2 fit clamped to [Z_FIT_LO, Z_FIT_HI]: boundary value, not divergent quadratic.
    assert np.allclose(np.array(delta_c(jnp.array(1.0))), np.array(delta_c(jnp.array(Z_FIT_LO))))
    assert np.allclose(np.array(delta_c(jnp.array(7.0))), np.array(delta_c(jnp.array(Z_FIT_HI))))
    assert np.all(np.isfinite(np.array(delta_c(jnp.array([1.0, 7.0, 100.0])))))


def test_alpha_to_dndx_jax_pure():
    # jit + vmap over batched (alpha, Xbar, z); grad finite; float64.
    alpha = jnp.array([[0.18, 0.06, 0.03], [0.10, 0.04, 0.01]])
    Xbar = jnp.array([0.642, 0.55]); z = jnp.array([3.0, 4.0])
    f = jax.jit(jax.vmap(lambda a, x, zz: alpha_to_dndx(a, x, zz, apply_delta=True)))
    out = f(alpha, Xbar, z)
    assert out.shape == (2, 3)
    assert out.dtype == jnp.float64
    assert np.all(np.isfinite(np.array(out)))
    g = jax.grad(lambda a: alpha_to_dndx(a, jnp.array(0.642), jnp.array(3.0))[2])(alpha[0])
    assert jnp.all(jnp.isfinite(g))


def test_delta_c_resid_std_shape():
    assert DELTA_C_RESID_STD.shape == (4,)
    assert np.all(np.array(DELTA_C_RESID_STD) > 0)


def test_alpha_to_dndx_edge_domain_and_clamp_grads():
    # Edge alpha_c that imply infeasible w (DLA fraction >= 1, or w_sub > 1-w_DLA):
    # the _LOG_FLOOR clip must keep value AND grad finite (not NaN) so HMC proposals
    # in the infeasible region degrade gracefully instead of poisoning the trajectory.
    Xbar = jnp.array(0.642); z = jnp.array(3.0)
    edges = [jnp.array([0.0, 0.0, 1.5]),    # w_DLA > 1
             jnp.array([0.1, 0.8, 0.5]),    # w_sub > 1 - w_DLA  (negative telescoping arg)
             jnp.array([0.0, 0.0, 1.0])]    # exactly w_DLA == 1 (clip boundary)
    for w in edges:
        out = alpha_to_dndx(w, Xbar, z, apply_delta=False)
        assert jnp.all(jnp.isfinite(out)), f"value not finite at edge {w}"
        g = jax.grad(lambda a: jnp.sum(alpha_to_dndx(a, Xbar, z, apply_delta=False)))(w)
        assert jnp.all(jnp.isfinite(g)), f"grad not finite at edge {w}"

    # delta_c clamp-grad: 0 outside the fit range (plateau), finite & nonzero inside.
    g_lo = jax.grad(lambda zz: jnp.sum(delta_c(zz)))(jnp.array(1.0))   # below Z_FIT_LO
    g_hi = jax.grad(lambda zz: jnp.sum(delta_c(zz)))(jnp.array(7.0))   # above Z_FIT_HI
    g_in = jax.grad(lambda zz: jnp.sum(delta_c(zz)))(jnp.array(3.5))   # inside
    assert float(g_lo) == 0.0 and float(g_hi) == 0.0
    assert jnp.isfinite(g_in) and abs(float(g_in)) > 0.0


# --- Corrected-law round-trip (spec sec 7, test-file addition a+b; PI 2026-07-18) ---
def test_alpha_roundtrip_at_corrected_laws_remeasured_tolerance():
    """alpha_from_dndx_law -> alpha_to_dndx round-trips the DEPLOYED corrected laws
    (K1a-constrained LLS + Poisson-GLM subDLA + PW09 DLA) to the RE-MEASURED tolerance:
    <0.2% at the z=3 pivot, <0.7% over z in [2.4, 4.2] (worst at z=4.2 where the renorm
    caveat grows; measured 2026-07-18: 0.11% @z3, 0.58% worst). Guards BOTH that the
    corrected laws are in place (subDLA gamma > 2 — the wrong-object 0.937 fails here)
    and that the deployed w_c map still inverts them."""
    from hcd_analysis.emulator.inference import HCD_LIT_DNDX_LAW
    # the corrected laws are in place (RED on the old wrong-object constants)
    assert HCD_LIT_DNDX_LAW["subDLA"][1] > 2.0, \
        "subDLA slope must be the corrected Poisson-GLM law (~2.44), not the DLA-block 0.937"
    assert HCD_LIT_DNDX_LAW["LLS"][0] < 0.020, \
        "LLS amplitude must be the kernel-corrected law (<0.0201, the cumulative-bug value)"
    A = jnp.asarray([HCD_LIT_DNDX_LAW[c][0] for c in ("LLS", "subDLA", "DLA")])
    g = jnp.asarray([HCD_LIT_DNDX_LAW[c][1] for c in ("LLS", "subDLA", "DLA")])
    for z, xb, tol in [(2.4, 0.454, 0.007), (3.0, 0.632, 0.002),
                       (3.6, 0.839, 0.007), (4.2, 1.076, 0.007)]:
        al = alpha_from_dndx_law(A, g, jnp.asarray(xb), jnp.asarray(z))
        rt = np.asarray(alpha_to_dndx(al, jnp.asarray(xb), jnp.asarray(z)))
        law = np.asarray(A) * (1.0 + z) ** np.asarray(g)
        assert np.all(np.abs(rt / law - 1.0) < tol), (z, rt / law - 1.0)


def test_wc_from_mu_inputs_are_disjoint_binned_classes():
    """DOCSTRING-LEVEL PIN (spec sec 7b): ``w_c_from_mu``'s mu inputs are the DISJOINT
    binned classes LLS [17.2,19.0), subDLA [19.0,20.3), DLA >=20.3 — NEVER cumulative
    rates. The telescoping construction double-counts a cumulative input: feeding the
    cumulative mu (LLS+sub+DLA in the LLS slot) produces a STRICTLY larger total covered
    fraction than the disjoint input, which is the numerical signature this test pins
    (the 2026-07-18 corrected-law re-derivation exists because the old deployed 'LLS' law
    was the cumulative tau>=2 compilation mislabeled as the binned class)."""
    mu_disjoint = jnp.array([0.12, 0.05, 0.03])          # (LLS, subDLA, DLA), disjoint
    mu_cumulative_bug = jnp.array([0.12 + 0.05 + 0.03, 0.05, 0.03])   # cumulative in slot 0
    w_ok = np.asarray(w_c_from_mu(mu_disjoint))
    w_bug = np.asarray(w_c_from_mu(mu_cumulative_bug))
    assert w_bug[1] > w_ok[1]                            # LLS share over-counted
    assert w_bug[0] < w_ok[0]                            # clean fraction under-counted
    # exact telescoping identity on disjoint classes: 1 - w_clean = P(any absorber)
    total = 1.0 - np.exp(-(0.12 + 0.05 + 0.03))
    assert abs((1.0 - w_ok[0]) - total) < 1e-12


def test_alpha_from_dndx_law_grad_through_A_gamma():
    # The forward law feeds the likelihood; grad of alpha_c w.r.t. the free (A, gamma)
    # power-law params must be finite (and the end-to-end fwd->inverse chain too).
    A = jnp.array([0.5, 0.12, 0.04]); gamma = jnp.array([1.0, 0.8, 0.5])
    Xbar = jnp.array(0.642); z = jnp.array(3.0)
    gA = jax.grad(lambda a: jnp.sum(alpha_from_dndx_law(a, gamma, Xbar, z)))(A)
    gg = jax.grad(lambda g: jnp.sum(alpha_from_dndx_law(A, g, Xbar, z)))(gamma)
    assert jnp.all(jnp.isfinite(gA)) and jnp.all(jnp.isfinite(gg))
    # end-to-end: A -> alpha_c -> dN/dX readout
    def chain(a):
        al = alpha_from_dndx_law(a, gamma, Xbar, z)
        return jnp.sum(alpha_to_dndx(al, Xbar, z, apply_delta=True))
    assert jnp.all(jnp.isfinite(jax.grad(chain)(A)))
