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
    """θ-BLIND P_filt baseline head (normalization REDESIGN; DEEP recipe).

    Input is ONLY ``[z, τ₀]`` (2-dim) -- it is blind to the 9 cosmology/IGM params
    BY CONSTRUCTION (it never sees the shared latent, which encodes θ). Output is
    the σ_marg-standardized (z,τ₀)-CONDITIONAL-MEAN spectrum m̂ for the 4 P_filt
    classes (the dominant ~99.5% predictable variation = the Kennedy-O'Hagan
    structured mean). Because θ never enters, ∂m̂/∂θ = 0 exactly.

    Depth/width are configurable (``n_layers`` hidden layers of ``width``; default
    n_layers=3, width=256). The validated sub-percent recipe
    (scripts/feasibility_subpercent.py Exp 2 / scripts/diag_tilt_bias_lf_hr.py)
    PROVED a 1-hidden-layer head plateaus at ~0.10–0.18·σ_cosmo, while a ≥3-layer
    w256 head drops the baseline misfit (term (b)) to ~0.03–0.05·σ_cosmo — so the
    production default is deep.

    Mirrors HeadB's P_filt path: low-rank when ``n_basis`` is set (4 coeffs ×
    n_basis through its OWN trainable basis ``p_filt_basis``), dense (4*n_k)
    otherwise. Lives in the same standardized-log target space as the baseline
    target t_p_base = (cell_mean − mu_marg)/sig_marg.

    ``n_layers``/``width``/``n_k``/``n_basis`` are STATIC fields (they set array
    shapes / layer counts and must be known at trace time for jit).
    """
    layers: list
    p_filt_basis: jax.Array | None
    n_k: int = eqx.field(static=True)
    n_basis: int | None = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    width: int = eqx.field(static=True)

    def __init__(self, n_k=172, width=256, n_layers=3, n_basis=None,
                 p_filt_basis_init=None, key=None):
        if n_layers < 1:
            raise ValueError(f"BaselineHead n_layers must be >= 1, got {n_layers}")
        self.n_k = n_k
        self.n_basis = n_basis
        self.n_layers = n_layers
        self.width = width
        out_dim = 4 * n_k if n_basis is None else 4 * n_basis
        # n_layers hidden Linears (input ONLY [z, tau0]) + a final output Linear.
        ks = jax.random.split(key, n_layers + 2)
        dims = [2] + [width] * n_layers          # [2, w, w, ..., w]
        hidden = [eqx.nn.Linear(dims[i], dims[i + 1], key=ks[i]) for i in range(n_layers)]
        out = eqx.nn.Linear(width, out_dim, key=ks[n_layers])
        self.layers = hidden + [out]
        if n_basis is None:
            self.p_filt_basis = None
        else:
            if p_filt_basis_init is not None:
                basis = jnp.asarray(p_filt_basis_init)
                if basis.shape != (n_basis, n_k):
                    raise ValueError(
                        f"p_filt_basis_init shape {basis.shape} != (n_basis, n_k) "
                        f"= ({n_basis}, {n_k})")
                self.p_filt_basis = basis
            else:
                self.p_filt_basis = jax.nn.initializers.orthogonal()(
                    ks[n_layers + 1], (n_basis, n_k))

    def __call__(self, z, tau0):
        # θ-BLIND: only (z, τ₀) enter; the shared latent (which encodes θ) is NEVER
        # passed in. This is what makes the baseline structurally identifiable.
        x = jnp.stack([jnp.atleast_1d(z)[0], jnp.atleast_1d(tau0)[0]])
        for lin in self.layers[:-1]:
            x = jax.nn.gelu(lin(x))
        y = self.layers[-1](x)
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
                 baseline_basis_init=None, baseline_n_layers=3, baseline_width=256,
                 key=None):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.enc = Encoder(in_dim=in_dim, key=k1)
        self.head_a = HeadA(latent=64, key=k2)
        # DEEP θ-blind baseline (validated recipe): default 3 hidden layers × w256
        # so term (b) collapses from ~0.84·σ_cosmo to ~0.03–0.05·σ_cosmo.
        self.head_base = BaselineHead(n_k=n_k, n_basis=n_basis,
                                      n_layers=baseline_n_layers, width=baseline_width,
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
    """NaN-safe masked WEIGHTED-MEAN squared error.

    Sanitise target to finite BEFORE the masked diff so the jnp.where double-NaN-
    gradient trap never fires; guard the denominator with max(·,1). The ``weight`` is
    also NaN-safe at masked bins: it is SELECTED to 0 there via jnp.where (not
    weight·mask), so a non-finite weight at a masked position can never form 0·inf=NaN
    (jax-traps #7 — weights now covered, not just the target).

    GRADIENT-SCALE FIX (validated recipe, scripts/feasibility_subpercent.py): when a
    ``weight`` is supplied this is a TRUE weighted mean — the denominator is
    ``Σ(weight·mask)``, NOT the raw masked count ``Σ(mask)``. The old form divided
    the weighted numerator by the *unweighted* count, so a per-class inv_nc weight
    (≈1/n_c, e.g. 6e-3 for a populous class) silently SHRANK the term's absolute
    gradient by that factor — the baseline head could not train (term (b) stuck at
    ~0.84·σ_cosmo). Normalizing by Σ(weight·mask) restores the absolute scale while
    PRESERVING inv_nc's relative intent: each masked element's contribution is its
    weight ÷ the total weight, so an alpha-sibling block down-weighted by 1/n_a (or a
    class by 1/n_c) still gets proportionally less say — it is just no longer globally
    scaled down. With no weight this is the plain masked mean (Σ(mask) denominator),
    unchanged."""
    target_safe = jnp.nan_to_num(target, nan=0.0)
    diff = jnp.where(mask, pred - target_safe, 0.0)
    sq = diff ** 2
    if weight is not None:
        # effective per-element weight = weight at UNMASKED bins, exactly 0 at masked.
        # Use jnp.where (NOT weight*mask): a non-finite weight at a masked bin would
        # make weight*0 = ±inf*0 = NaN, poisoning BOTH the Σ(w) denominator and the
        # sq*w numerator (the "NaN-safe" claim). jnp.where SELECTS 0.0 there instead of
        # multiplying, so a ±inf masked weight is fully inert. At unmasked bins this is
        # exactly ``weight`` (mask broadcast against weight), so every current finite-
        # weight value — and gradient scale — is preserved bit-for-bit.
        w = jnp.where(mask.astype(bool), weight, 0.0)
        w = jnp.broadcast_to(w, sq.shape)
        denom = jnp.maximum(jnp.sum(w), 1.0)
        return jnp.sum(sq * w) / denom
    denom = jnp.maximum(jnp.sum(mask.astype(sq.dtype)), 1.0)
    return jnp.sum(sq) / denom


def _k_weight_from_batch(batch):
    """Per-k residual loss weight -> broadcastable (1,1,K), or None.

    The RESIDUAL-HEAD k-emphasis knob (A_p low-k bias fix). When the batch carries a
    ``k_weight`` array (shape (B,K) — a per-row tile of one (K,) profile, so it pads/
    batches like every other array; all rows identical), the COSMOLOGY (``p_resid``)
    term's per-element weight becomes ``inv_nc · k_weight`` so the optimizer attends
    MORE to the up-weighted band edges (low-k where the Lyα amplitude Δ²_* — hence
    A_p — pivots, and high-k) where the deployed residual was biased. Absent the key,
    the residual term is the plain inv_nc-weighted MSE (back-compat / uniform).
    Edge-emphasis fixes COHERENT bias; the CV/sampling SCATTER at low-k is left to
    early-stop + C_emu (not chased).

    Reads row 0 (the profile is row-invariant) and returns (1,1,K) so it broadcasts
    against the (B,4,K) residual MSE without depending on the (padded) batch size.
    """
    kw = batch.get("k_weight")
    if kw is None:
        return None
    kw = jnp.atleast_2d(kw)          # (B,K) (or (K,) -> (1,K))
    return kw[0][None, None, :]      # (1,1,K), row-invariant profile


def _datarange_weight_from_batch(batch):
    """Per-(z,k) DATA-RANGE soft down-weight -> broadcastable (B,1,K), or None.

    Data-range SCOPING (the in-range capacity-focus knob): out-of-range bins
    (z∉[z_lo,z_hi] OR k<k_min) are SOFT down-weighted by a factor ~0.2 (NOT
    hard-zeroed — z>4.6's signal still regularizes the encoder, just gets less
    say). The weight is built host-side per ROW (so it depends on each row's z) ×
    per-k (the k-cut is row-invariant) and carried in the batch under
    ``datarange_weight`` (B,K); see ``data.datarange_loss_weight``. Absent the key,
    the P_filt-channel terms are un-down-weighted (back-compat / full footprint).

    Returns (B,1,K) so it broadcasts against the (B,4,K) / (B,3,K) P_filt-channel
    MSE. Padded rows are masked out anyway, so their weight value is irrelevant."""
    dw = batch.get("datarange_weight")
    if dw is None:
        return None
    dw = jnp.atleast_2d(dw)          # (B,K)
    return dw[:, None, :]            # (B,1,K)


def coherent_debias_term(preds, batch):
    """FLAT coherent de-bias: Σ_cell ⟨ r̂ − t_p_resid ⟩_θ², per-(z,τ₀)-CELL mean.

    The de-bias REGULARIZER (productionized from scripts/push_residual_refine.py,
    the winning ``flat_w80`` recipe; jax-traps #24). For each global (z,τ₀)-cell it
    forms the per-cell MEAN OVER SIMS (cosmologies) of the residual fit error
    ``(r̂ − t_p_resid)``, then averages its square over the populated cells / valid
    (class,k). Because the training target t_p_resid = (logP − cell_mean)/σ_cosmo has
    ⟨t_p_resid⟩_θ = 0 per cell BY CONSTRUCTION, this drives the systematic θ-mean of
    the residual head toward zero — i.e. it FLATTENS the coherent k-tilt the per-sim
    MSE leaves an unconstrained coherent d.o.f. (the per-sim MSE is dominated by the
    CV scatter and does NOT directly penalize the per-cell θ-mean offset).

    UNIFORM in k DELIBERATELY: the FLAT (no edge-weight) k-shape was the winner — it
    de-biases all bands evenly. Edge-weighting this term OVER-corrects low-k and
    trades it for a mid-band regression (jax-traps #24); inverse-CV makes low-k
    worse. So this term carries NO ``k_weight`` (unlike the per-sim p_resid term).

    Requires ``batch["cell"]`` (global (z,τ₀)-cell id, int (B,)) and the STATIC
    scalar ``batch["n_cells"]`` (number of distinct global cells = n_z·n_alpha; the
    segment count). NaN-safe: above-Nyquist bins are zeroed via the per-row Nyquist
    mask BEFORE the segment-sum and excluded from the per-cell count. Padded rows
    (mask all-False) add 0 to both numerator and count, so they are inert.

    Returns a scalar. ``num_segments`` is read as a python int from batch["n_cells"]
    (it sets the segment-sum output shape and so must be static at trace time — it is
    a fold-level constant, traced once)."""
    n_cells = int(batch["n_cells"])
    rhat = preds["P_filt_resid"]                            # (B,4,K)
    t = jnp.nan_to_num(batch["t_p_resid"], nan=0.0)         # (B,4,K)
    m = batch["mask"][:, None, :]                           # (B,1,K) Nyquist mask
    err = jnp.where(m, rhat - t, 0.0)                       # (B,4,K) per-row resid error
    seg = batch["cell"]                                     # (B,) global cell id
    # per-cell sum of err and of the mask (# finite sims contributing per (c,k))
    num = jax.ops.segment_sum(err, seg, num_segments=n_cells)             # (N,4,K)
    cnt = jax.ops.segment_sum(
        jnp.broadcast_to(m.astype(err.dtype), err.shape), seg,
        num_segments=n_cells)                                            # (N,4,K)
    cell_mean_err = jnp.where(cnt > 0, num / jnp.maximum(cnt, 1.0), 0.0)  # ⟨r̂−t⟩_θ
    sq = cell_mean_err ** 2
    valid = (cnt > 0).astype(sq.dtype)
    denom = jnp.maximum(jnp.sum(valid), 1.0)
    return jnp.sum(sq) / denom


def joint_loss(model, batch, term_w=None, w_coh=0.0):
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

    ``term_w`` rescales the named terms. ``p_resid`` is the inference-relevant
    cosmology term — UP-WEIGHTING it (e.g. term_w["p_resid"]>1) concentrates the
    optimizer on the signal the likelihood needs (the joint default uniform dilutes
    it 1:5 vs f_nhi/dndx/p_base/delta). The optional batch key ``k_weight`` (K,)
    additionally re-weights the p_resid term PER-k (low/high-k edge emphasis); see
    ``_k_weight_from_batch``.

    ``w_coh`` adds the FLAT coherent de-bias REGULARIZER ``w_coh·coherent_debias_term``
    (the productionized flat_w80 winner; default 80 in train_fold). It requires the
    batch carry ``cell``/``n_cells``; with w_coh<=0 the term (and the requirement) is
    skipped. The de-bias term is UNIFORM in k (do NOT edge-weight it — flat is the
    winner; see ``coherent_debias_term`` / jax-traps #24).

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
    # data-range SOFT down-weight (factor ~0.2) for out-of-range (z,k) bins — keeps
    # z>4.6/low-k regularizing signal but focuses capacity in the DESI data range.
    # Applied to the residual + baseline + delta terms (the P_filt channels) so the
    # standardized-space MSE attends in-range; None when the batch omits it.
    rng_w = _datarange_weight_from_batch(batch)              # (B,1,K) or None
    # BASELINE term: θ-blind cell-mean fit (σ_marg-standardized). inv_nc-weighted.
    w_base = wcls4 if rng_w is None else wcls4 * rng_w
    lb_base = masked_mse(preds["P_filt_base"], batch["t_p_base"], mask4, weight=w_base)
    # COSMOLOGY term: σ_cosmo-whitened residual (the signal inference needs). The
    # per-k k_weight (if present) emphasizes the band edges (the A_p low-k fix); the
    # data-range weight (if present) softens out-of-range bins.
    kw = _k_weight_from_batch(batch)
    w_resid = wcls4
    if kw is not None:
        w_resid = w_resid * kw
    if rng_w is not None:
        w_resid = w_resid * rng_w
    lb_resid = masked_mse(preds["P_filt_resid"], batch["t_p_resid"], mask4, weight=w_resid)
    w_dl = wcls3 if rng_w is None else wcls3 * rng_w
    lb_dl = masked_mse(preds["delta"], batch["t_delta"],
                       m3 & jnp.ones_like(batch["t_delta"], bool), weight=w_dl)
    total = (term_w["f_nhi"]*la_cddf + term_w["dndx"]*la_dndx
             + term_w["p_base"]*lb_base + term_w["p_resid"]*lb_resid
             + term_w["delta"]*lb_dl)
    if w_coh > 0.0:
        total = total + w_coh * coherent_debias_term(preds, batch)
    return total


def p_resid_loss(model, batch, use_k_weight=False, use_datarange=False):
    """The COSMOLOGY (σ_cosmo-whitened residual) term alone — the early-stop /
    validation metric for the redesign (the quantity inference cares about).

    inv_nc-weighted masked MSE of the residual head vs t_p_resid, NaN-safe over the
    per-row Nyquist mask. Lower == better cosmology resolution. When
    ``use_k_weight`` and the batch carries ``k_weight``, the SAME per-k emphasis the
    training loss uses is applied here too, so the early-stop metric tracks the
    quantity the loss optimizes (and the band-edge attention is reflected in it).
    When ``use_datarange`` and the batch carries ``datarange_weight``, the data-range
    soft down-weight is applied too (so the stop tracks the in-range-focused loss)."""
    preds = jax.vmap(model)(batch["x"], batch["tau0"])
    mask4 = batch["mask"][:, None, :] & jnp.ones_like(batch["t_p_resid"], bool)
    wcls4 = batch["inv_nc"][:, :, None]
    if use_k_weight:
        kw = _k_weight_from_batch(batch)
        if kw is not None:
            wcls4 = wcls4 * kw
    if use_datarange:
        rng_w = _datarange_weight_from_batch(batch)
        if rng_w is not None:
            wcls4 = wcls4 * rng_w
    return masked_mse(preds["P_filt_resid"], batch["t_p_resid"], mask4, weight=wcls4)


def coherent_resid_loss(model, batch):
    """The FLAT coherent de-bias term value alone (val-side monitor / early-stop).

    Σ_cell ⟨ r̂ − t_p_resid ⟩_θ² over the populated (z,τ₀)-cells (see
    ``coherent_debias_term``). Lower == less systematic per-cell θ-mean residual (a
    flatter coherent k-tilt). Requires ``cell``/``n_cells`` in the batch."""
    preds = jax.vmap(model)(batch["x"], batch["tau0"])
    return coherent_debias_term(preds, batch)
