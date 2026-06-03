"""Equinox emulator: encoder + Head A (tau0-invariant) + Head B (tau0-dependent).

Head A consumes ONLY the latent (no tau0) so its outputs (f_nhi, dN/dX) are
tau0-invariant by construction (spec sec.3). Head B consumes concat(latent, tau0).
P_tier_p is the structural sum Sum_c w_c * P_c^filt (spec sec.3/sec.6).
"""
from __future__ import annotations
import jax, jax.numpy as jnp, equinox as eqx


def svd_basis_init(P_filt_transformed, n_basis):
    """SVD warm-start for HeadB's learned P_filt basis (review finding I1/I2).

    Given the cache's standardized-log P_filt stacked over (rows*classes, n_k) --
    i.e. the SAME standardized-log space HeadB's P_filt output lives in -- return
    the top-``n_basis`` right singular vectors as a ``(n_basis, n_k)`` warm-start
    basis. Pass the result as ``p_filt_basis_init`` to ``Emulator``/``HeadB``.

    The right singular vectors are orthonormal, so the returned basis is full rank
    (rank-collapse guard) whenever ``n_basis <= rank(P_filt_transformed)``.
    """
    M = jnp.asarray(P_filt_transformed)
    if M.ndim != 2:
        raise ValueError(f"P_filt_transformed must be 2D (rows*classes, n_k); got {M.shape}")
    # full_matrices=False -> Vh is (min(m,n), n_k); rows are the right sing. vectors.
    _, _, Vh = jnp.linalg.svd(M, full_matrices=False)
    return Vh[:n_basis]


class Encoder(eqx.Module):
    layers: list

    def __init__(self, in_dim=10, widths=(256, 128, 64), key=None):
        ks = jax.random.split(key, len(widths))
        dims = [in_dim, *widths]
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=ks[i]) for i in range(len(widths))]

    def __call__(self, x):
        for lin in self.layers[:-1]:
            x = jax.nn.gelu(lin(x))
        return self.layers[-1](x)


class HeadA(eqx.Module):
    trunk: eqx.nn.Linear
    cddf: eqx.nn.Linear
    dndx: eqx.nn.Linear

    def __init__(self, latent=64, n_k_cddf=30, key=None):
        k1, k2, k3 = jax.random.split(key, 3)
        self.trunk = eqx.nn.Linear(latent, 64, key=k1)
        self.cddf = eqx.nn.Linear(64, n_k_cddf, key=k2)
        self.dndx = eqx.nn.Linear(64, 3, key=k3)

    def __call__(self, latent):
        h = jax.nn.gelu(self.trunk(latent))
        return {"f_nhi": self.cddf(h), "dndx": self.dndx(h)}


class BaselineHead(eqx.Module):
    """θ-BLIND P_filt baseline head (normalization REDESIGN).

    Input is ONLY ``[z, τ₀]`` (2-dim) -- it is blind to the 9 cosmology/IGM params
    BY CONSTRUCTION (it never sees the shared latent, which encodes θ). Output is
    the σ_marg-standardized (z,τ₀)-CONDITIONAL-MEAN spectrum m̂ for the 4 P_filt
    classes (the dominant ~99.5% predictable variation = the Kennedy-O'Hagan
    structured mean). Because θ never enters, ∂m̂/∂θ = 0 exactly.

    Mirrors HeadB's P_filt path: low-rank when ``n_basis`` is set (4 coeffs ×
    n_basis through its OWN trainable basis ``p_filt_basis``), dense (4*n_k)
    otherwise. Lives in the same standardized-log target space as the baseline
    target t_p_base = (cell_mean − mu_marg)/sig_marg.
    """
    trunk: eqx.nn.Linear
    out: eqx.nn.Linear
    p_filt_basis: jax.Array | None
    n_k: int = eqx.field(static=True)
    n_basis: int | None = eqx.field(static=True)

    def __init__(self, n_k=172, width=64, n_basis=None, p_filt_basis_init=None, key=None):
        k1, k2, k3 = jax.random.split(key, 3)
        self.n_k = n_k
        self.n_basis = n_basis
        self.trunk = eqx.nn.Linear(2, width, key=k1)     # input is ONLY [z, tau0]
        if n_basis is None:
            self.out = eqx.nn.Linear(width, 4 * n_k, key=k2)
            self.p_filt_basis = None
        else:
            self.out = eqx.nn.Linear(width, 4 * n_basis, key=k2)
            if p_filt_basis_init is not None:
                basis = jnp.asarray(p_filt_basis_init)
                if basis.shape != (n_basis, n_k):
                    raise ValueError(
                        f"p_filt_basis_init shape {basis.shape} != (n_basis, n_k) "
                        f"= ({n_basis}, {n_k})")
                self.p_filt_basis = basis
            else:
                self.p_filt_basis = jax.nn.initializers.orthogonal()(k3, (n_basis, n_k))

    def __call__(self, z, tau0):
        # θ-BLIND: only (z, τ₀) enter; the shared latent (which encodes θ) is NEVER
        # passed in. This is what makes the baseline structurally identifiable.
        inp = jnp.stack([jnp.atleast_1d(z)[0], jnp.atleast_1d(tau0)[0]])
        h = jax.nn.gelu(self.trunk(inp))
        y = self.out(h)
        if self.n_basis is None:
            return y.reshape(4, self.n_k)
        coeffs = y.reshape(4, self.n_basis)
        return coeffs @ self.p_filt_basis                # (4, n_k)


class HeadB(eqx.Module):
    """tau0-dependent head: 4 filtered class P1D RESIDUAL (log-space) + 3 HCD deltas.

    Reads concat(latent, tau0).

    Normalization REDESIGN: the P_filt output is now the θ-dependent RESIDUAL r̂
    (4,K) in σ_cosmo-whitened units (the cosmology signal = Δ-learning / GraphCast
    increment), NOT the absolute spectrum. The θ-blind baseline m̂ is supplied
    separately by ``BaselineHead``; full logP̂ = baseline + sig_cosmo·r̂ (see
    data.reconstruct_P_filt). The returned key is ``P_filt_resid``.

    Dense default (``n_basis=None``): the ``7*n_k`` output is reshaped to (7, n_k)
    and split into 4 P_filt residual (clean,LLS,subDLA,DLA) and 3 delta (LLS,subDLA,
    DLA). This is the original ~370k-param head's shape, byte-for-byte unchanged
    (only the semantic of the first 4 rows is now "residual").

    Low-rank P_filt (``n_basis=int``, review finding I1/I2): the output layer emits
    ``4*n_basis + 3*n_k``. The first ``4*n_basis`` reshape to (4, n_basis) P_filt
    *coefficients* that decode through a TRAINABLE basis ``p_filt_basis (n_basis,
    n_k)`` as ``coeffs @ basis -> P_filt residual (4, n_k)``. The remaining ``3*n_k``
    reshape to (3, n_k) delta and stay DENSE/arcsinh -- Delta_c's low-k sign flip
    cannot go through a shared positive/log basis, so it is NEVER routed through the
    basis. The structural_tier_p path still operates on the untransformed LINEAR
    P_filt (reconstructed from baseline+residual) downstream.

    n_basis is a STATIC field (it sets array shapes / output-layer width and must be
    known at trace time for jit). The optional SVD warm-start basis is supplied at
    construction via ``p_filt_basis_init`` (see ``svd_basis_init``); if omitted, the
    basis is orthogonal-initialised from the key (full-rank, no rank-collapse).
    """
    trunk: eqx.nn.Linear
    out: eqx.nn.Linear
    p_filt_basis: jax.Array | None
    n_k: int = eqx.field(static=True)
    n_basis: int | None = eqx.field(static=True)

    def __init__(self, latent=64, n_k=172, n_basis=None, p_filt_basis_init=None, key=None):
        k1, k2, k3 = jax.random.split(key, 3)
        self.n_k = n_k
        self.n_basis = n_basis
        self.trunk = eqx.nn.Linear(latent + 1, 256, key=k1)
        if n_basis is None:
            # dense path: byte-for-byte the original head (single out Linear).
            self.out = eqx.nn.Linear(256, 7 * n_k, key=k2)
            self.p_filt_basis = None
        else:
            # low-rank P_filt coeffs (4*n_basis) + dense delta (3*n_k).
            self.out = eqx.nn.Linear(256, 4 * n_basis + 3 * n_k, key=k2)
            if p_filt_basis_init is not None:
                basis = jnp.asarray(p_filt_basis_init)
                if basis.shape != (n_basis, n_k):
                    raise ValueError(
                        f"p_filt_basis_init shape {basis.shape} != (n_basis, n_k) "
                        f"= ({n_basis}, {n_k})")
                self.p_filt_basis = basis
            else:
                # orthogonal init -> rows are orthonormal => basis is full rank
                # (rank-collapse guard) and (n_basis, n_k) oriented.
                self.p_filt_basis = jax.nn.initializers.orthogonal()(k3, (n_basis, n_k))

    def __call__(self, latent, tau0):
        h = jax.nn.gelu(self.trunk(jnp.concatenate([latent, jnp.atleast_1d(tau0)])))
        y = self.out(h)
        if self.n_basis is None:
            y = y.reshape(7, self.n_k)
            return {"P_filt_resid": y[:4], "delta": y[4:]}
        nb = self.n_basis
        coeffs = y[: 4 * nb].reshape(4, nb)          # (4, n_basis)
        delta = y[4 * nb:].reshape(3, self.n_k)      # (3, n_k) dense
        P_filt_resid = coeffs @ self.p_filt_basis    # (4, n_basis) @ (n_basis, n_k)
        return {"P_filt_resid": P_filt_resid, "delta": delta}


class Emulator(eqx.Module):
    """Full emulator: Encoder -> {HeadA (tau0-invariant), BaselineHead (θ-blind
    P_filt baseline), HeadB (tau0-dependent residual + delta)}.

    P_filt is split (normalization REDESIGN): ``P_filt_base`` from the θ-BLIND
    BaselineHead(z, τ₀) and ``P_filt_resid`` (the σ_cosmo-whitened cosmology signal)
    from HeadB(latent, τ₀). Full LINEAR P_filt = exp((P_filt_base·sig_marg+mu_marg)
    + sig_cosmo·P_filt_resid) (see data.reconstruct_P_filt); structural_tier_p still
    consumes LINEAR P_filt downstream. ∂logP̂/∂θ = sig_cosmo·∂P_filt_resid/∂θ
    (the baseline is θ-blind).
    """
    enc: Encoder
    head_a: HeadA
    head_base: BaselineHead
    head_b: HeadB

    def __init__(self, in_dim=10, n_k=172, n_basis=None, p_filt_basis_init=None,
                 baseline_basis_init=None, key=None):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.enc = Encoder(in_dim=in_dim, key=k1)
        self.head_a = HeadA(latent=64, key=k2)
        self.head_base = BaselineHead(n_k=n_k, n_basis=n_basis,
                                      p_filt_basis_init=baseline_basis_init, key=k3)
        self.head_b = HeadB(latent=64, n_k=n_k, n_basis=n_basis,
                            p_filt_basis_init=p_filt_basis_init, key=k4)

    def __call__(self, x, tau0):
        lat = self.enc(x)
        # x layout = [params_unit (9), z_unit (1)] -> z = x[9]; the BaselineHead takes
        # ONLY (z, τ₀) and is θ-blind (never sees the latent / the 9 params).
        z = x[..., 9]
        base = {"P_filt_base": self.head_base(z, tau0)}
        return {**self.head_a(lat), **base, **self.head_b(lat, tau0)}


def structural_tier_p(w_c, P_filt_lin):
    """Structural Tier-P total = Sum_c w_c * P_c^filt (LINEAR space).

    w_c: (...,4); P_filt_lin: (...,4,K) LINEAR space. Returns (...,K).
    Batches over leading dims; differentiable.
    """
    return jnp.einsum("...c,...ck->...k", w_c, P_filt_lin)


def masked_mse(pred, target, mask, weight=None):
    """NaN-safe masked MSE. Sanitise target to finite BEFORE the masked diff so the
    jnp.where double-NaN-gradient trap never fires; guard the denominator with max(n,1)."""
    target_safe = jnp.nan_to_num(target, nan=0.0)
    diff = jnp.where(mask, pred - target_safe, 0.0)
    sq = diff ** 2
    if weight is not None:
        sq = sq * weight
    denom = jnp.maximum(jnp.sum(mask.astype(sq.dtype)), 1.0)
    return jnp.sum(sq) / denom


def joint_loss(model, batch, term_w=None):
    """Single joint scalar (spec sec.4). Per-element means balance the 172 vs 3
    channel counts; term_w optionally rescales the named terms.

    Normalization REDESIGN: the single P_filt term is split into TWO —
      - ``p_base``  : MSE of the θ-blind BASELINE head vs t_p_base (the
        σ_marg-standardized (z,τ₀)-conditional cell-mean);
      - ``p_resid`` : MSE of the θ-dependent RESIDUAL head vs t_p_resid (the
        σ_cosmo-whitened COSMOLOGY signal, ~unit variance) — this is the term
        that actually resolves cosmology, weighted by inv_nc per class.
    Both use the per-row Nyquist mask. The now-FIVE terms (f_nhi, dndx, p_base,
    p_resid, delta) are all O(1) in standardized space, so the default term_w is
    UNIFORM.

    A4: with per-channel standardization every term's per-element-mean MSE is O(1)
    in transformed space, so the default term_w is uniform (no hand-sweep). It
    stays an optional override.

    A3: the old meanF term was removed (zero model gradient; structural mean-flux).
    """
    term_w = term_w or {"f_nhi": 1.0, "dndx": 1.0,
                        "p_base": 1.0, "p_resid": 1.0, "delta": 1.0}
    preds = jax.vmap(model)(batch["x"], batch["tau0"])
    # A4b: Head-A masks exclude the safe_log-floored structural-zero CDDF/dN/dX bins.
    la_cddf = masked_mse(preds["f_nhi"], batch["t_f_nhi"],
                         batch["t_f_nhi_mask"],
                         weight=batch["inv_nalpha"][:, None])
    la_dndx = masked_mse(preds["dndx"], batch["t_dndx"],
                         batch["t_dndx_mask"],
                         weight=batch["inv_nalpha"][:, None])
    m3 = batch["mask"][:, None, :]
    wcls4 = batch["inv_nc"][:, :, None]
    wcls3 = batch["inv_nc"][:, 1:, None]
    mask4 = m3 & jnp.ones_like(batch["t_p_resid"], bool)
    # BASELINE term: θ-blind cell-mean fit (σ_marg-standardized). inv_nc-weighted.
    lb_base = masked_mse(preds["P_filt_base"], batch["t_p_base"], mask4, weight=wcls4)
    # COSMOLOGY term: σ_cosmo-whitened residual (the signal inference needs).
    lb_resid = masked_mse(preds["P_filt_resid"], batch["t_p_resid"], mask4, weight=wcls4)
    lb_dl = masked_mse(preds["delta"], batch["t_delta"],
                       m3 & jnp.ones_like(batch["t_delta"], bool), weight=wcls3)
    return (term_w["f_nhi"]*la_cddf + term_w["dndx"]*la_dndx
            + term_w["p_base"]*lb_base + term_w["p_resid"]*lb_resid
            + term_w["delta"]*lb_dl)


def p_resid_loss(model, batch):
    """The COSMOLOGY (σ_cosmo-whitened residual) term alone — the early-stop /
    validation metric for the redesign (the quantity inference cares about).

    inv_nc-weighted masked MSE of the residual head vs t_p_resid, NaN-safe over the
    per-row Nyquist mask. Lower == better cosmology resolution."""
    preds = jax.vmap(model)(batch["x"], batch["tau0"])
    mask4 = batch["mask"][:, None, :] & jnp.ones_like(batch["t_p_resid"], bool)
    return masked_mse(preds["P_filt_resid"], batch["t_p_resid"], mask4,
                      weight=batch["inv_nc"][:, :, None])
