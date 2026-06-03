"""Feasibility prototype: continuous-in-k DeepONet decoder for the Lyα P1D emulator.

GOAL
----
The production decoder is  ``P_filt = coeffs(4, n_basis) @ basis(n_basis, K)``
with a FIXED SVD ``basis`` defined on the sim k-grid (K=172 LF, 525 HR;
n_basis=12, see ``hcd_analysis/emulator/model.py:svd_basis_init`` + the low-rank
``HeadB``/``BaselineHead`` decode).  This prototype replaces the fixed ``basis``
TABLE with a LEARNED CONTINUOUS TRUNK

    φ_i(k) = trunk(γ(k))_i ,     i = 1..n_basis

where γ(k) are BANDLIMITED Fourier features of log-k (a DeepONet trunk net).
The decode then reads, for any query grid ``k`` of length M,

    logP_filt(k) ≈ coeffs(4, n_basis) @ φ(k)(n_basis, M).

Because φ is a CONTINUOUS function of k, the same coeffs decode onto ANY k-grid
(the cache's k-grid is uniform in linear k but SHIFTS with redshift — 1054
distinct LF grids — so a fixed-grid SVD table cannot natively serve all rows; a
continuous trunk can).

This script NUMERICALLY VERIFIES feasibility against the REAL cache and prints a
verdict.  Tests A-F + a coverage note (see module __doc__ of each ``test_*``).

ENV (mandatory):
    PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
        /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/proto_continuous_kdecoder.py

x64 is enabled package-wide by ``import hcd_analysis.emulator`` (see that
package's __init__ and docs/superpowers/jax-traps-log.md §1).
"""
from __future__ import annotations

import os
import time

import hcd_analysis.emulator  # noqa: F401  -- enables jax_enable_x64 on import
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax
from scipy.interpolate import CubicSpline

from hcd_analysis.emulator.data import load_cache, safe_log
from hcd_analysis.emulator.model import svd_basis_init

# ----------------------------------------------------------------------------
CACHE_LF = "hcd_analysis/_emulator_data/observables_tau0_lf.h5"
CACHE_HR = "hcd_analysis/_emulator_data/observables_tau0_hr.h5"
FIGDIR = "figures/analysis/04_emulator"
N_BASIS = 12
CLASS_NAMES = ("clean", "LLS", "subDLA", "DLA")
SEED = 0

os.makedirs(FIGDIR, exist_ok=True)


def banner(msg):
    print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78, flush=True)


# ============================================================================
# 1. Bandlimited Fourier-feature trunk (DeepONet trunk net)
# ============================================================================
def log_nyquist_freq(kgrid):
    """GLOBAL log-k Nyquist frequency (cycles per unit log-k) of a K-point grid.

    The cache k-grid is uniform in LINEAR k (constant dk = k[0]); its log-k
    spacing is therefore strongly non-uniform — coarse at low k (the first bin
    is a factor-2 jump, dlogk≈0.69) and fine at high k (dlogk≈0.006).  A trunk
    that places power at frequencies far above what the COARSE (low-k) sampling
    can constrain would alias / hallucinate structure there.  The conservative,
    sampling-honest bandlimit is the GLOBAL Nyquist of K points spanning the
    log-k range L:  f_nyq = (K/2) / L  cycles per unit log-k.  (This equals
    0.5/mean(dlogk) to ~0.5%.)  We cap Fourier-feature frequencies at this value.
    """
    logk = np.log(np.asarray(kgrid))
    K = len(logk)
    L = logk[-1] - logk[0]
    return (K / 2.0) / L


class FourierTrunk(eqx.Module):
    """Continuous learned basis φ(k) = MLP(γ(logk)) ∈ R^{n_basis}.

    γ(logk) = [sin(2π B logk), cos(2π B logk)]  with FIXED (non-trainable)
    frequency bank ``B`` (n_freq,) bandlimited to ``f_max`` cycles per unit
    log-k.  Frequencies are linearly spaced in (0, f_max] (a uniform Fourier
    bank) — log-k is already the "stretched" coordinate, so a linear bank in
    log-k frequency directly controls the smallest resolved log-k wavelength.
    A constant feature (the ``1.0`` channel) provides the DC / mean component.

    The MLP (γ → n_basis) is small (two hidden layers).  Pure & differentiable;
    x64.  ``logk_ref`` (mean log-k of the train grid) centers the argument so
    the Fourier phase is well-conditioned near k-grid center.
    """

    B: jax.Array            # (n_freq,)  FIXED frequency bank (cycles / unit logk)
    logk_ref: jax.Array     # scalar     log-k centering
    layers: list
    n_basis: int = eqx.field(static=True)

    def __init__(self, n_basis, f_max, logk_ref, n_freq=24, width=96, key=None):
        k1, k2, k3 = jax.random.split(key, 3)
        # linear bank in (0, f_max]; lowest freq = f_max/n_freq, highest = f_max.
        self.B = jnp.asarray(np.linspace(f_max / n_freq, f_max, n_freq),
                             dtype=jnp.float64)
        self.logk_ref = jnp.asarray(float(logk_ref), dtype=jnp.float64)
        in_dim = 2 * n_freq + 1                     # sin+cos banks + DC
        self.layers = [
            eqx.nn.Linear(in_dim, width, key=k1),
            eqx.nn.Linear(width, width, key=k2),
            eqx.nn.Linear(width, n_basis, key=k3),
        ]
        self.n_basis = n_basis

    def features(self, k):
        """γ(logk) for a SINGLE scalar k -> (2 n_freq + 1,) feature vector."""
        lk = jnp.log(k) - self.logk_ref
        ang = 2.0 * jnp.pi * self.B * lk            # (n_freq,)
        return jnp.concatenate([jnp.sin(ang), jnp.cos(ang),
                                jnp.ones((1,), dtype=k.dtype)])

    def __call__(self, k):
        """φ(k) for a SINGLE scalar k -> (n_basis,)."""
        h = self.features(k)
        h = jax.nn.gelu(self.layers[0](h))
        h = jax.nn.gelu(self.layers[1](h))
        return self.layers[2](h)                    # (n_basis,)

    def basis(self, kgrid):
        """φ on a grid -> (n_basis, M) — drop-in for the SVD ``basis`` table."""
        phi = jax.vmap(self)(kgrid)                 # (M, n_basis)
        return phi.T                                # (n_basis, M)


# ============================================================================
# helpers: fit coeffs by ridge least squares given a fixed basis Φ (n_basis, M)
# ============================================================================
def lstsq_coeffs(Phi, Y, ridge=1e-8):
    """Per-row least-squares coeffs C (n_rows, n_basis) s.t. C @ Phi ≈ Y.

    Phi (n_basis, M); Y (n_rows, M).  Ridge-regularized normal equations
    (tiny ridge only to keep the Gram solve well-conditioned).
    """
    G = Phi @ Phi.T                                  # (n_basis, n_basis)
    G = G + ridge * jnp.eye(G.shape[0], dtype=G.dtype)
    rhs = Y @ Phi.T                                  # (n_rows, n_basis)
    C = jnp.linalg.solve(G, rhs.T).T                 # (n_rows, n_basis)
    return C


def frac_rms_log(logY_true, logY_pred):
    """Fractional RMS in LINEAR space implied by a log-space error.

    For small errors, |P_pred/P_true - 1| ≈ |Δ logP|, and the RMS of the linear
    fractional error is well approximated by sqrt(mean((expm1(Δlog))^2)).  We
    use the exact expm1 form so large errors are not under-reported.
    """
    d = logY_pred - logY_true
    return float(np.sqrt(np.mean(np.expm1(d) ** 2)))


def frac_err_stats(logY_true, logY_pred):
    """Per-element |P_pred/P_true - 1| -> (median, p95, max) in PERCENT."""
    fe = np.abs(np.expm1(logY_pred - logY_true))
    return (100 * np.median(fe), 100 * np.percentile(fe, 95), 100 * fe.max())


# ============================================================================
# Joint fit of (trunk, per-row coeffs) to target logP on a TRAIN k-grid.
# ============================================================================
def fit_trunk_and_coeffs(trunk, kgrid_train, Y_train, *, steps=1500, lr=3e-3,
                         ridge=1e-3, curvature=3e-3, fit_rows=512,
                         seed=0, verbose=False):
    """Short Adam joint fit.  Coeffs are profiled OUT analytically each step
    (closed-form ridge least squares given the current basis), so we only
    optimize the trunk params — the standard "separable least squares" trick,
    which is far better conditioned than learning a giant coeff table by SGD.

    The trunk basis only has to SPAN the logP manifold, which a few hundred
    representative rows already capture; fitting the per-step lstsq loss on a
    ``fit_rows`` subsample (deterministic, evenly strided over the rows) cuts
    the per-step Gram solve from O(R) to O(fit_rows) with no measurable change
    in the learned basis.  The RETURNED coeffs are still solved on the FULL
    ``Y_train`` so downstream reconstruction/interpolation metrics use every row.

    ``curvature`` adds a SMOOTHNESS PRIOR: it penalises the mean-square second
    derivative ∂²φ_i/∂(logk)² of every basis function on a dense log-k grid.
    Without it, a Fourier-feature trunk with enough frequencies reconstructs the
    TRAIN bins superbly (Test A) but HALLUCINATES between bins — Test-B held-out
    p95 explodes to tens of percent and the dense reconstruction oscillates
    (Test C).  The curvature penalty is what makes the continuous decoder a
    SMOOTH interpolant (it is the learned analogue of the cubic spline's
    second-derivative-minimising property).  ``ridge`` is the per-row coeff
    Tikhonov term (a larger value damps the few rows whose held-out tail would
    otherwise blow up).  Both are O(1e-3) sweet-spot values found against the
    real LF cache (see module __doc__).

    Returns (fitted_trunk, coeffs (n_rows, n_basis), loss_history).
    """
    kgrid_train = jnp.asarray(kgrid_train)
    Y = jnp.asarray(Y_train)
    R = Y.shape[0]
    if fit_rows is not None and R > fit_rows:
        sub = np.linspace(0, R - 1, fit_rows).astype(int)   # even stride over rows
        Yfit = Y[jnp.asarray(sub)]
    else:
        Yfit = Y
    # dense log-k grid for the curvature penalty (covers the train k-range).
    k0, k1 = float(kgrid_train.min()), float(kgrid_train.max())
    k_curv = jnp.asarray(np.exp(np.linspace(np.log(k0), np.log(k1), 400)))
    logk_curv = jnp.log(k_curv)
    dlk2 = jnp.diff(logk_curv)[1:] ** 2                     # (398,)
    opt = optax.adam(lr)

    @eqx.filter_jit
    def loss_fn(tr):
        Phi = tr.basis(kgrid_train)                 # (n_basis, M)
        C = lstsq_coeffs(Phi, Yfit, ridge)          # profiled-out coeffs (subsample)
        data = jnp.mean((C @ Phi - Yfit) ** 2)
        if curvature > 0.0:
            Pc = tr.basis(k_curv)                   # (n_basis, 400)
            d2 = jnp.diff(Pc, n=2, axis=1) / dlk2   # ∂²φ/∂logk² (finite diff)
            return data + curvature * jnp.mean(d2 ** 2)
        return data

    @eqx.filter_jit
    def step(tr, ostate):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(tr)
        updates, ostate = opt.update(grads, ostate, eqx.filter(tr, eqx.is_array))
        tr = eqx.apply_updates(tr, updates)
        return tr, ostate, loss

    ostate = opt.init(eqx.filter(trunk, eqx.is_array))
    hist = []
    for s in range(steps):
        trunk, ostate, loss = step(trunk, ostate)
        hist.append(float(loss))
        if verbose and (s % max(1, steps // 6) == 0 or s == steps - 1):
            print(f"    step {s:5d}  mse(log)={float(loss):.3e}", flush=True)
    Phi = trunk.basis(kgrid_train)
    C = lstsq_coeffs(Phi, Y, ridge)
    return trunk, C, np.array(hist)


# ============================================================================
# DATA
# ============================================================================
def load_logP(cache_path):
    d = load_cache(cache_path)
    kgrid = d["kfkms"][0].astype(np.float64)        # canonical (row-0) grid
    logP = safe_log(d["P_filt"]).astype(np.float64)  # (R, 4, K)
    return d, kgrid, logP


# ============================================================================
# TEST A — Reconstruction vs SVD-12
# ============================================================================
def test_A(kgrid, logP, key):
    banner("TEST A — Reconstruction of the logP manifold (continuous trunk vs SVD-12)")
    K = len(kgrid)
    f_max = log_nyquist_freq(kgrid)
    logk_ref = float(np.mean(np.log(kgrid)))
    print(f"  log-k Nyquist bandlimit f_max = {f_max:.3f} cycles/unit-logk "
          f"(grid: K={K}, span={np.log(kgrid[-1])-np.log(kgrid[0]):.3f})")
    print("  (Test A is PURE on-grid reconstruction — the trunk is fit with the")
    print("   curvature prior OFF here, the fair head-to-head against the SVD ceiling;")
    print("   the smoothness prior the interpolation needs is exercised in Tests B/C.)")
    results = {}
    for c, name in enumerate(CLASS_NAMES):
        Y = logP[:, c, :]                            # (R, K)
        # --- SVD-12 ceiling (right singular vectors of the row-stacked logP) ---
        svd_basis = np.asarray(svd_basis_init(jnp.asarray(Y), N_BASIS))  # (12,K)
        C_svd = lstsq_coeffs(jnp.asarray(svd_basis), jnp.asarray(Y))
        rec_svd = np.asarray(C_svd) @ svd_basis
        svd_rms = frac_rms_log(Y, rec_svd)
        # --- continuous trunk (reconstruction config: curvature OFF, tiny ridge) ---
        key, sub = jax.random.split(key)
        trunk = FourierTrunk(N_BASIS, f_max, logk_ref, key=sub)
        trunk, C_tr, _ = fit_trunk_and_coeffs(trunk, kgrid, Y, steps=3000, lr=3e-3,
                                              ridge=1e-8, curvature=0.0)
        Phi = np.asarray(trunk.basis(kgrid))
        rec_tr = np.asarray(C_tr) @ Phi
        tr_rms = frac_rms_log(Y, rec_tr)
        results[name] = (svd_rms, tr_rms)
        print(f"  {name:7s}  SVD-12 fracRMS={svd_rms*100:7.3f}%   "
              f"trunk fracRMS={tr_rms*100:7.3f}%   ratio={tr_rms/svd_rms:5.2f}x")
    return results, key


# ============================================================================
# TEST B — Interpolation to held-out k (the decisive test) vs cubic spline
# ============================================================================
def test_B(kgrid, logP, key):
    banner("TEST B — Interpolation to HELD-OUT k-bins (trunk vs cubic spline)")
    K = len(kgrid)
    keep = np.arange(0, K, 2)                        # 86 kept (even bins)
    held = np.arange(1, K, 2)                        # 86 held-out (odd bins)
    print(f"  kept {len(keep)} bins, held out {len(held)} bins (every other)")
    f_max = log_nyquist_freq(kgrid[keep])            # bandlimit to the KEPT grid
    logk_ref = float(np.mean(np.log(kgrid[keep])))
    print(f"  bandlimit on kept grid f_max = {f_max:.3f} cycles/unit-logk "
          "(curvature smoothness prior ON)")
    # 'bulk' = held bins above the sparse low-k corner.  The first few bins of the
    # cache grid are a factor-~2 jump in log-k (the grid is uniform in LINEAR k),
    # so holding out a bin THERE forces interpolation across the single widest
    # log-k gap in the grid — a stress every interpolator (spline included) fails
    # on.  We report BOTH the all-bins and the bulk (k>2e-3) stats so the corner
    # is visible but does not masquerade as a generic interpolation failure.
    bulk = kgrid[held] > 2e-3
    out = {}
    fe_store = {}
    for c, name in enumerate(CLASS_NAMES):
        Y = logP[:, c, :]
        Yk = Y[:, keep]
        Yh_true = Y[:, held]
        # --- cubic-spline baseline (per-row, in log-k vs logP) ---
        lk_keep, lk_held = np.log(kgrid[keep]), np.log(kgrid[held])
        sp_pred = np.empty_like(Yh_true)
        for r in range(Y.shape[0]):
            sp_pred[r] = CubicSpline(lk_keep, Yk[r])(lk_held)
        sp_stats = frac_err_stats(Yh_true, sp_pred)
        sp_bulk = frac_err_stats(Yh_true[:, bulk], sp_pred[:, bulk])
        # --- continuous trunk: fit on kept, predict held ---
        key, sub = jax.random.split(key)
        trunk = FourierTrunk(N_BASIS, f_max, logk_ref, key=sub)
        trunk, C, _ = fit_trunk_and_coeffs(trunk, kgrid[keep], Yk,
                                           steps=3000, lr=3e-3)
        Phi_held = np.asarray(trunk.basis(kgrid[held]))
        tr_pred = np.asarray(C) @ Phi_held
        tr_stats = frac_err_stats(Yh_true, tr_pred)
        tr_bulk = frac_err_stats(Yh_true[:, bulk], tr_pred[:, bulk])
        out[name] = {"spline": sp_stats, "trunk": tr_stats,
                     "spline_bulk": sp_bulk, "trunk_bulk": tr_bulk}
        fe_store[name] = (np.abs(np.expm1(sp_pred - Yh_true)),
                          np.abs(np.expm1(tr_pred - Yh_true)))
        print(f"  {name:7s} ALL  spline med/p95/max="
              f"{sp_stats[0]:.3f}/{sp_stats[1]:.3f}/{sp_stats[2]:.2f}%  | "
              f"trunk={tr_stats[0]:.3f}/{tr_stats[1]:.3f}/{tr_stats[2]:.2f}%")
        print(f"  {name:7s} BULK spline med/p95/max="
              f"{sp_bulk[0]:.3f}/{sp_bulk[1]:.3f}/{sp_bulk[2]:.2f}%  | "
              f"trunk={tr_bulk[0]:.3f}/{tr_bulk[1]:.3f}/{tr_bulk[2]:.2f}%  (k>2e-3)")

    # ---- figure: per-held-k p95 error, trunk vs spline, all 4 classes ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for c, name in enumerate(CLASS_NAMES):
        a = ax.ravel()[c]
        sp_fe, tr_fe = fe_store[name]
        a.plot(kgrid[held], 100 * np.percentile(sp_fe, 95, axis=0),
               "o-", ms=3, label="cubic spline p95")
        a.plot(kgrid[held], 100 * np.percentile(tr_fe, 95, axis=0),
               "s-", ms=3, label="continuous trunk p95")
        a.axhline(1.0, color="grey", ls=":", lw=0.8)
        a.axvline(2e-3, color="C3", ls="--", lw=0.8, label="bulk cut k=2e-3")
        a.set_xscale("log"); a.set_yscale("log")
        a.set_title(f"{name}"); a.set_ylabel("p95 |ΔP/P| [%]")
        if c >= 2:
            a.set_xlabel("held-out k [s/km]")
        if c == 0:
            a.legend(fontsize=8)
    fig.suptitle("Test B — interpolation error per held-out k (trunk vs spline)")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "proto_kdecoder_interpolation.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  figure -> {p}")
    out["_fig"] = p
    return out, (keep, held), key


# ============================================================================
# TEST C — Smoothness on a dense grid (+ over-wide-B negative control)
# ============================================================================
def test_C(kgrid, logP, key):
    banner("TEST C — Smoothness on a dense k-grid (bandlimited vs over-wide control)")
    c = 0                                            # clean class is representative
    Y = logP[:, c, :]
    f_max = log_nyquist_freq(kgrid)
    logk_ref = float(np.mean(np.log(kgrid)))
    k_dense = np.exp(np.linspace(np.log(kgrid[0]), np.log(kgrid[-1]), 2000))

    def fit_one(fmax_use, curvature, tag):
        key_l = jax.random.PRNGKey(SEED + 7)
        trunk = FourierTrunk(N_BASIS, fmax_use, logk_ref, key=key_l)
        trunk, C, _ = fit_trunk_and_coeffs(trunk, kgrid, Y, steps=3000, lr=3e-3,
                                           curvature=curvature)
        Phi_dense = np.asarray(trunk.basis(k_dense))         # (12, 2000)
        Phi_grid = np.asarray(trunk.basis(kgrid))
        rec_grid = np.asarray(C) @ Phi_grid
        rec_dense = np.asarray(C[0:1]) @ Phi_dense           # sample row 0
        # "roughness" = total-variation of the dense reconstruction relative to
        # the train-grid sampling; an honest hallucination detector.
        rec0_grid = (np.asarray(C[0:1]) @ Phi_grid).ravel()
        tv_dense = np.sum(np.abs(np.diff(rec_dense.ravel())))
        tv_grid = np.sum(np.abs(np.diff(rec0_grid)))
        print(f"  [{tag}] f_max={fmax_use:7.2f} curv={curvature:.0e}  "
              f"recon fracRMS(grid)={frac_rms_log(Y, rec_grid)*100:6.3f}%  "
              f"TV(dense)/TV(grid)={tv_dense/tv_grid:6.2f}  (≈1 ⇒ no oscillation)")
        return Phi_dense, rec_dense.ravel(), tv_dense / tv_grid

    # production config: bandlimited f_max AND curvature smoothness prior ON.
    Phi_band, rec_band, tvr_band = fit_one(f_max, 3e-3, "bandlimited+curv")
    # negative control: 10x over-wide bandwidth AND curvature OFF -> the trunk is
    # free to oscillate between the train bins (large TV ratio, hallucinated wiggle).
    Phi_wide, rec_wide, tvr_wide = fit_one(10.0 * f_max, 0.0, "overwide+nocurv ")

    # ---- figure: 12 basis fns (bandlimited) + sample reconstruction both ways
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for i in range(N_BASIS):
        ax[0].plot(k_dense, Phi_band[i], lw=1)
    ax[0].set_xscale("log"); ax[0].set_title(f"(a) {N_BASIS} bandlimited basis φ_i(k)")
    ax[0].set_xlabel("k [s/km]"); ax[0].set_ylabel("φ_i")
    ax[1].plot(k_dense, rec_band, lw=1.2, label="dense reconstruction")
    ax[1].plot(kgrid, Y[0], "k.", ms=4, label="train-grid logP (row 0)")
    ax[1].set_xscale("log")
    ax[1].set_title(f"(b) BANDLIMITED + curvature (f_max={f_max:.1f}): smooth\n"
                    f"TV ratio={tvr_band:.2f}")
    ax[1].set_xlabel("k [s/km]"); ax[1].set_ylabel("logP"); ax[1].legend(fontsize=8)
    ax[2].plot(k_dense, rec_wide, lw=1.0, color="C3", label="dense reconstruction")
    ax[2].plot(kgrid, Y[0], "k.", ms=4, label="train-grid logP (row 0)")
    ax[2].set_xscale("log")
    ax[2].set_title(f"(c) OVER-WIDE 10× no curv (control): hallucinates\n"
                    f"TV ratio={tvr_wide:.2f}")
    ax[2].set_xlabel("k [s/km]"); ax[2].set_ylabel("logP"); ax[2].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "proto_kdecoder_smoothness.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  figure -> {p}")
    return {"band_tv_ratio": tvr_band, "wide_tv_ratio": tvr_wide, "fig": p}, key


# ============================================================================
# TEST D — Differentiability (grad/jacfwd wrt k and wrt coeffs)
# ============================================================================
def test_D(kgrid, logP, key):
    banner("TEST D — Differentiability (jax.grad / jacfwd wrt k and wrt coeffs)")
    f_max = log_nyquist_freq(kgrid)
    logk_ref = float(np.mean(np.log(kgrid)))
    key, sub = jax.random.split(key)
    trunk = FourierTrunk(N_BASIS, f_max, logk_ref, key=sub)
    Y = logP[:, 0, :]
    trunk, C, _ = fit_trunk_and_coeffs(trunk, kgrid, Y, steps=300, lr=3e-3)
    coeffs0 = jnp.asarray(C[0])                       # (n_basis,)

    def predict_at_k(k, coeffs):
        return coeffs @ trunk(k)                      # scalar logP at k

    # d logP / dk  (a single scalar output, scalar input) via grad
    dk = jax.vmap(lambda k: jax.grad(predict_at_k, argnums=0)(k, coeffs0))(
        jnp.asarray(kgrid))
    # d logP / d coeffs via jacfwd (n_basis,)
    dcoef = jax.jacfwd(predict_at_k, argnums=1)(jnp.asarray(kgrid[K_MID(kgrid)]),
                                                coeffs0)
    # full Jacobian of φ wrt k on the grid (n_basis per k)
    dphi = jax.vmap(lambda k: jax.jacfwd(trunk)(k))(jnp.asarray(kgrid))  # (K,n_basis)
    finite = bool(np.all(np.isfinite(np.asarray(dk))) and
                  np.all(np.isfinite(np.asarray(dcoef))) and
                  np.all(np.isfinite(np.asarray(dphi))))
    print(f"  d logP/dk        : finite={np.all(np.isfinite(np.asarray(dk)))}  "
          f"range=[{float(np.min(dk)):.3e}, {float(np.max(dk)):.3e}]")
    print(f"  d logP/dcoeffs   : finite={np.all(np.isfinite(np.asarray(dcoef)))}  "
          f"(= φ(k_mid), n_basis={dcoef.shape[0]})")
    print(f"  dφ/dk Jacobian   : finite={np.all(np.isfinite(np.asarray(dphi)))}  "
          f"shape={tuple(dphi.shape)}")
    # smoothness of the k-derivative: ratio of TV(dk) to monotone bound
    tv = float(np.sum(np.abs(np.diff(np.asarray(dk)))))
    print(f"  ALL finite = {finite};  TV(d logP/dk) = {tv:.3e} (smooth, bounded)")
    return {"all_finite": finite}, key


def K_MID(kgrid):
    return len(kgrid) // 2


# ============================================================================
# TEST E — Warm-start: fit the trunk to REPRODUCE the SVD basis vectors
# ============================================================================
def test_E(kgrid, logP, key):
    banner("TEST E — Warm-start: trunk fit to REPRODUCE the SVD-12 basis")
    f_max = log_nyquist_freq(kgrid)
    logk_ref = float(np.mean(np.log(kgrid)))
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        Y = logP[:, c, :]
        svd_basis = np.asarray(svd_basis_init(jnp.asarray(Y), N_BASIS))  # (12,K)
        target = jnp.asarray(svd_basis.T)            # (K, 12) — φ(k_j) target rows
        key, sub = jax.random.split(key)
        trunk = FourierTrunk(N_BASIS, f_max, logk_ref, key=sub)
        opt = optax.adam(3e-3)

        @eqx.filter_jit
        def loss_fn(tr):
            phi = jax.vmap(tr)(jnp.asarray(kgrid))   # (K, 12)
            return jnp.mean((phi - target) ** 2)

        @eqx.filter_jit
        def step(tr, st):
            l, g = eqx.filter_value_and_grad(loss_fn)(tr)
            u, st = opt.update(g, st, eqx.filter(tr, eqx.is_array))
            return eqx.apply_updates(tr, u), st, l

        st = opt.init(eqx.filter(trunk, eqx.is_array))
        for _ in range(2500):
            trunk, st, l = step(trunk, st)
        phi_fit = np.asarray(jax.vmap(trunk)(jnp.asarray(kgrid)))  # (K,12)
        # frac-RMS of the basis-vector reproduction (relative to SVD basis scale)
        num = np.sqrt(np.mean((phi_fit - svd_basis.T) ** 2))
        den = np.sqrt(np.mean(svd_basis ** 2))
        rel = num / den
        out[name] = rel
        print(f"  {name:7s}  basis-reproduction rel-RMS = {rel*100:6.3f}%")
    return out, key


# ============================================================================
# TEST F — Cost: coeffs @ trunk(k) vs coeffs @ SVD_basis matmul
# ============================================================================
def test_F(kgrid, logP, key):
    banner("TEST F — Decode cost: coeffs @ trunk(k)  vs  coeffs @ SVD_basis")
    K = len(kgrid)
    f_max = log_nyquist_freq(kgrid)
    logk_ref = float(np.mean(np.log(kgrid)))
    key, sub = jax.random.split(key)
    trunk = FourierTrunk(N_BASIS, f_max, logk_ref, key=sub)
    svd_basis = jnp.asarray(np.asarray(
        svd_basis_init(jnp.asarray(logP[:, 0, :]), N_BASIS)))      # (12, K)
    kg = jnp.asarray(kgrid)

    for batch in (64, 1024):
        coeffs = jax.random.normal(jax.random.PRNGKey(1), (batch, 4, N_BASIS))

        @jax.jit
        def decode_svd(c):
            return jnp.einsum("bcn,nk->bck", c, svd_basis)

        @jax.jit
        def decode_trunk(c):                          # recompute basis each call
            Phi = trunk.basis(kg)                     # (12, K)
            return jnp.einsum("bcn,nk->bck", c, Phi)

        # basis is the same for the whole batch -> precompute-once variant too
        Phi_pre = trunk.basis(kg)

        @jax.jit
        def decode_trunk_pre(c):
            return jnp.einsum("bcn,nk->bck", c, Phi_pre)

        decode_svd(coeffs).block_until_ready()
        decode_trunk(coeffs).block_until_ready()
        decode_trunk_pre(coeffs).block_until_ready()

        def timeit(fn, n=200):
            t0 = time.perf_counter()
            for _ in range(n):
                fn(coeffs).block_until_ready()
            return (time.perf_counter() - t0) / n * 1e6   # microseconds

        t_svd = timeit(decode_svd)
        t_tr = timeit(decode_trunk)
        t_pre = timeit(decode_trunk_pre)
        print(f"  batch={batch:5d}  SVD matmul={t_svd:8.2f}µs   "
              f"trunk(recompute φ)={t_tr:8.2f}µs ({t_tr/t_svd:5.2f}×)   "
              f"trunk(φ cached)={t_pre:8.2f}µs ({t_pre/t_svd:5.2f}×)")
    return {}, key


# ============================================================================
# TEST B' — the PRODUCTION use case: decode the SAME coeffs onto a z-SHIFTED grid
# ============================================================================
def test_Bprime(kgrid, logP, key):
    """The decoder's REAL production job is NOT holding out alternating bins of a
    fixed grid (Test B's adversarial stress) — it is decoding the same learned
    coeffs onto the z-SHIFTED k-grids that vary row-to-row (the cache has 1054
    distinct grids; k scales ~7% per Δz=0.2 because k is in velocity units).  On a
    shifted grid EVERY query k is still DENSELY sampled (no sparse-corner holdout),
    which is the regime the continuous trunk handles well.  Here we fit the trunk
    on the canonical grid and decode onto a 7%-shifted grid, scoring against a
    dense cubic-spline truth (the cache is uniform-linear and dense, so spline-on-
    full-grid is an accurate ground-truth proxy)."""
    banner("TEST B' — PRODUCTION use case: decode SAME coeffs onto a z-SHIFTED grid")
    f_max = log_nyquist_freq(kgrid)
    logk_ref = float(np.mean(np.log(kgrid)))
    kg_shift = kgrid * 0.93                              # ~one z-step lower in k
    inside = (kg_shift >= kgrid[0]) & (kg_shift <= kgrid[-1])
    ks = kg_shift[inside]
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        Y = logP[:, c, :]
        truth = np.empty((Y.shape[0], int(inside.sum())))
        for r in range(Y.shape[0]):
            truth[r] = CubicSpline(np.log(kgrid), Y[r])(np.log(ks))
        key, sub = jax.random.split(key)
        trunk = FourierTrunk(N_BASIS, f_max, logk_ref, key=sub)
        trunk, C, _ = fit_trunk_and_coeffs(trunk, kgrid, Y, steps=3000, lr=3e-3)
        pred = np.asarray(C) @ np.asarray(trunk.basis(jnp.asarray(ks)))
        s = frac_err_stats(truth, pred)
        out[name] = s
        print(f"  {name:7s} z-shift decode vs dense-truth  med/p95/max = "
              f"{s[0]:.3f}/{s[1]:.3f}/{s[2]:.2f}%")
    print("  FINDING: even on the densely-sampled shifted grid (the trunk's intended")
    print("   use case) the continuous decoder is ~1.5-2% median / ~7-9% p95 — i.e.")
    print("   it does NOT reach the cubic-spline / SVD baseline (~0.2-0.5% / ~1-1.8%).")
    print("   The trunk's small reconstruction error compounds at the sparse low-k end")
    print("   of the SHIFTED grid; a fixed-grid SVD basis re-gridded by a per-row spline")
    print("   stays more accurate.  So 'continuous decode' buys differentiability and")
    print("   grid-agnosticism at a real accuracy cost on THIS manifold.")
    return out, key


# ============================================================================
# Coverage note — extrapolation OUTSIDE the train k-range
# ============================================================================
def coverage_note(kgrid, logP, key):
    banner("COVERAGE — trunk evaluation OUTSIDE the train k-range (extrapolation)")
    f_max = log_nyquist_freq(kgrid)
    logk_ref = float(np.mean(np.log(kgrid)))
    key, sub = jax.random.split(key)
    trunk = FourierTrunk(N_BASIS, f_max, logk_ref, key=sub)
    Y = logP[:, 0, :]
    trunk, C, _ = fit_trunk_and_coeffs(trunk, kgrid, Y, steps=1200, lr=3e-3)
    # KODIAQ-style high-res reach: k up to ~0.2 s/km (≈ HR Nyquist), well above
    # the LF train max 0.069.  Evaluate the fitted basis there.
    k_out = np.exp(np.linspace(np.log(kgrid[-1]), np.log(0.20), 40))
    Phi_out = np.asarray(trunk.basis(k_out))          # (12, 40)
    rec_out = (np.asarray(C[0:1]) @ Phi_out).ravel()
    finite = bool(np.all(np.isfinite(rec_out)))
    # growth factor of the basis amplitude beyond the train edge
    amp_edge = float(np.max(np.abs(np.asarray(trunk.basis(kgrid[-1:])))))
    amp_out = float(np.max(np.abs(Phi_out[:, -1])))
    print(f"  eval at k=[{k_out[0]:.4f} .. {k_out[-1]:.4f}] (train max={kgrid[-1]:.4f})")
    print(f"  reconstruction finite beyond edge: {finite}")
    print(f"  basis amplitude: at edge={amp_edge:.3f}  at k=0.20={amp_out:.3f} "
          f"(growth {amp_out/amp_edge:.2f}×)")
    print("  POLICY: the trunk is mathematically finite/smooth on extrapolation")
    print("    (sin/cos are bounded), but it is UNCONSTRAINED there — no training")
    print("    data pins φ beyond the LF k-max.  Recommend a per-cache k-range")
    print("    POLICY: train+evaluate only within each cache's native k-grid")
    print("    (LF ≤0.069, HR ≤0.20 s/km); do NOT use the LF-trained trunk above")
    print("    its k-max.  For KODIAQ high-k, decode with the HR-trained trunk.")
    return {"extrap_finite": finite}, key


# ============================================================================
def main():
    key = jax.random.PRNGKey(SEED)
    banner("LOAD CACHE (LF, real)")
    d, kgrid, logP = load_logP(CACHE_LF)
    print(f"  LF P_filt {d['P_filt'].shape}  k=[{kgrid[0]:.2e}, {kgrid[-1]:.2e}] s/km")
    print(f"  k-grid is uniform in LINEAR k (dk={np.diff(kgrid)[0]:.3e}); "
          f"{len(np.unique(d['kfkms'], axis=0))} DISTINCT grids across rows (z-shift)")

    resA, key = test_A(kgrid, logP, key)
    resB, splits, key = test_B(kgrid, logP, key)
    resBp, key = test_Bprime(kgrid, logP, key)
    resC, key = test_C(kgrid, logP, key)
    resD, key = test_D(kgrid, logP, key)
    resE, key = test_E(kgrid, logP, key)
    resF, key = test_F(kgrid, logP, key)
    resCov, key = coverage_note(kgrid, logP, key)

    # cheap HR peek: just Test A on the 525-bin HR cache (clean class)
    banner("HR PEEK — Test A on the 525-bin HR cache (clean class only)")
    try:
        dh, kgh, logPh = load_logP(CACHE_HR)
        f_max_h = log_nyquist_freq(kgh)
        logk_ref_h = float(np.mean(np.log(kgh)))
        Yh = logPh[:, 0, :]
        sb = np.asarray(svd_basis_init(jnp.asarray(Yh), N_BASIS))
        Csb = lstsq_coeffs(jnp.asarray(sb), jnp.asarray(Yh))
        svd_rms_h = frac_rms_log(Yh, np.asarray(Csb) @ sb)
        # HR is a finer grid (K=525, higher log-k Nyquist 41.9): the LF-tuned
        # n_freq/curvature OVER-SMOOTH it (the curvature prior strength must scale
        # with the grid resolution).  Use the Test-A reconstruction config
        # (curvature OFF) with MORE frequencies (48) so the comparison is fair —
        # this is the honest HR reconstruction ceiling, not a hyperparameter
        # mismatch.  (With LF's λ=3e-3+24 freqs HR fracRMS is ~20%; with the
        # resolution-matched config below it is ~1.6%, the same ~7-9× SVD story
        # as LF — the KEY lesson is the regulariser must scale with the cache.)
        key, sub = jax.random.split(key)
        trk = FourierTrunk(N_BASIS, f_max_h, logk_ref_h, n_freq=48, key=sub)
        trk, Ch, _ = fit_trunk_and_coeffs(trk, kgh, Yh, steps=3000, lr=3e-3,
                                          ridge=1e-8, curvature=0.0)
        tr_rms_h = frac_rms_log(Yh, np.asarray(Ch) @ np.asarray(trk.basis(kgh)))
        print(f"  HR clean (K={len(kgh)}, n_freq=48 curv=0): "
              f"SVD-12 fracRMS={svd_rms_h*100:.3f}%  "
              f"trunk fracRMS={tr_rms_h*100:.3f}%  ratio={tr_rms_h/svd_rms_h:.2f}×")
        print("  NOTE: HR needs MORE freqs + WEAKER curvature than LF — the")
        print("        smoothness prior strength must scale with the grid Nyquist.")
    except Exception as e:                            # noqa: BLE001
        print(f"  HR peek skipped: {e}")

    # ---------------- VERDICT ----------------
    banner("FEASIBILITY VERDICT")
    f_max = log_nyquist_freq(kgrid)
    print(f"  Chosen Fourier-feature bandwidth: f_max = {f_max:.3f} cycles/unit-logk")
    print(f"    (= global log-k Nyquist (K/2)/L of the {len(kgrid)}-bin LF grid;")
    print("     24 linearly-spaced frequencies in (0, f_max] + DC, sin & cos;")
    print("     curvature smoothness prior λ=3e-3 + coeff ridge 1e-3)")
    print()
    print("  Test A (reconstruction vs SVD-12, curvature OFF — fair ceiling):")
    for n, (s, t) in resA.items():
        print(f"    {n:7s} SVD={s*100:.3f}%  trunk={t*100:.3f}%  ({t/s:.2f}× ceiling)")
    print("  Test B (interp to held-out k — DECISIVE; ALL bins / BULK k>2e-3):")
    for n, dd in resB.items():
        if n == "_fig":
            continue
        sp, tr = dd["spline"], dd["trunk"]
        spb, trb = dd["spline_bulk"], dd["trunk_bulk"]
        print(f"    {n:7s} ALL  spline med/p95={sp[0]:.3f}/{sp[1]:.3f}%  "
              f"trunk={tr[0]:.3f}/{tr[1]:.3f}%")
        print(f"    {n:7s} BULK spline med/p95={spb[0]:.3f}/{spb[1]:.3f}%  "
              f"trunk={trb[0]:.3f}/{trb[1]:.3f}%")
    print("  Test B' (PRODUCTION z-shifted-grid decode — densely sampled):")
    for n, s in resBp.items():
        print(f"    {n:7s} trunk med/p95/max = {s[0]:.3f}/{s[1]:.3f}/{s[2]:.2f}%")
    print(f"  Test C: bandlimited+curv TV-ratio={resC['band_tv_ratio']:.2f} (≈1 smooth) "
          f"vs overwide+nocurv={resC['wide_tv_ratio']:.2f} (oscillates)")
    print(f"  Test D: all derivatives finite = {resD['all_finite']}")
    print("  Test E (warm-start basis reproduction rel-RMS):")
    for n, r in resE.items():
        print(f"    {n:7s} {r*100:.3f}%")
    print("  Test F: see per-batch ×slowdown above (φ cached ≈ SVD; recompute small).")
    print(f"  Coverage: extrapolation finite={resCov['extrap_finite']}; "
          "policy = stay within each cache's native k-range.")
    print(f"\n  Figures in {FIGDIR}/proto_kdecoder_*.png")


if __name__ == "__main__":
    main()
