"""Phase-C T4a — the canonical closure CONTEXT container + the ONE pack/unpack.

The ``Ctx`` is a frozen ``eqx.Module``: array leaves (the model, the per-z error
slices, the data) are PYTREE LEAVES; static metadata (``n_z``, ``K``, flags) is
``eqx.field(static=True)``. The point is that ``eqx.filter_jit`` traces ONLY over the
sampled ``params`` and closes over the ctx, so the same shapes never recompile across
mocks (CS contract §5: "jit only over params, ctx closed over").

This module is SAMPLER-AGNOSTIC: ``sampler_numpyro`` (now) and ``sampler_cobaya``
(later) both share the SINGLE ``pack/unpack/to_dict/from_dict/param_names`` here and the
``log_lik_from_ctx`` likelihood entrypoint. The packed vector is
``[θ9 (PARAM_NAMES order), τ₀ (per z), α_lls, α_subdla, α_dla]`` — a fixed, asserted
layout so a dict↔vector round-trip can never silently re-order.

x64 is HARD-ASSERTED on import (the emulator's structural identities are bit-level; a
float32 NUTS run would silently mis-rank in the SBC).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from .data import KIM_AMP, KIM_SLOPE
from .inference import PARAM_NAMES, log_lik_multiz

# The closure machinery is bit-level (predict/likelihood structural identities); x64 is
# enabled by ``import hcd_analysis.emulator``. Fail LOUD if a caller imported jax first.
assert jax.config.read("jax_enable_x64"), \
    "x64 must be on (import hcd_analysis.emulator before jax, or set jax_enable_x64)"


# ----------------------------------------------------------------------------
# The ctx container (frozen eqx.Module) — array leaves + static metadata.
# ----------------------------------------------------------------------------
class Ctx(eqx.Module):
    """Frozen context shared by both samplers and the mock generator.

    Dynamic (pytree) leaves are arrays the likelihood reads; static fields are the
    shapes/flags the trace specializes on. Per-z arrays carry a LEADING z-axis (the
    axis ``log_lik_multiz`` vmaps over). All array leaves are jnp/float64.
    """
    # --- dynamic (array / module) leaves: jit traces over params, ctx closed over ---
    model: eqx.Module                 # the trained Emulator (final_fold0)
    pf_stats: dict                    # {'mu_marg','sig_marg','sig_cosmo'} each (4,K)
    z: jnp.ndarray                    # (n_z,) physical redshift
    z_unit: jnp.ndarray               # (n_z,) z mapped to the cache [0,1] coord
    sigma_zb: jnp.ndarray             # (n_z,4,K,Tb) per-z fractional error slice
    alpha_centres: jnp.ndarray        # (Tb,) τ₀-ladder band centres (α units)
    cosmic_cov: jnp.ndarray           # (n_z,K) variance or (n_z,K,K) cov
    P_data: jnp.ndarray               # (n_z,K) the (mock) data spectrum
    dla_core: jnp.ndarray             # (n_z,K) FIXED DLA-core add-back template
    dla_shot_flag: jnp.ndarray        # (n_z,K) bool high-k DLA shot bins
    valid_k: jnp.ndarray              # (n_z,K) bool in-data-range mask
    w_c_fid: jnp.ndarray              # (4,) structural weights (HCD-prior centring)
    tau0_mu: jnp.ndarray              # (n_z,) mean-flux prior centre, τ₀ units
    tau0_sigma: jnp.ndarray           # (n_z,) mean-flux prior width, τ₀ units
    alpha_hcd_mu: jnp.ndarray         # (3,) HCD incidence prior centre
    alpha_hcd_sigma: jnp.ndarray      # (3,) HCD incidence prior width
    # --- static metadata (trace specializes; never an array leaf) ---
    n_z: int = eqx.field(static=True)
    K: int = eqx.field(static=True)
    Tb: int = eqx.field(static=True)
    shot_inflate: float = eqx.field(static=True)
    cemu_inflate: float = eqx.field(static=True)
    include_logdet: bool = eqx.field(static=True)
    # OPT-IN cross-class C_emu block (n_z,4,4,K,Tb); None → the diagonal σ path (DEFAULT).
    # A None leaf is an empty pytree node (no trace cost); the diagonal closure is unchanged.
    # Defaulted ⇒ must come last (dataclass field-ordering); back-compat constructors omit it.
    rho_zb: jnp.ndarray = None


def ctx_kim(z):
    """Kim2013 central mean-flux τ₀(z) = KIM_AMP·(1+z)^KIM_SLOPE — the curve the cache's
    τ₀ ladder is anchored to, so α = τ₀/Kim(z) is the z-isotropic ladder coordinate."""
    return KIM_AMP * (1.0 + jnp.asarray(z)) ** KIM_SLOPE


def log_lik_from_ctx(theta9, tau0_vec, alpha_hcd, ctx: Ctx):
    """The SINGLE likelihood entrypoint both samplers call (closes over ctx).

    LIKELIHOOD-ONLY (vmap over z inside ``log_lik_multiz``); priors live in the sampler.
    Trace/jit over (theta9, tau0_vec, alpha_hcd); ctx is closed over (static shapes →
    no recompiles across mocks)."""
    return log_lik_multiz(
        ctx.model, theta9, tau0_vec, alpha_hcd,
        pf_stats=ctx.pf_stats, z=ctx.z, z_unit=ctx.z_unit, sigma_zb=ctx.sigma_zb,
        alpha_centres=ctx.alpha_centres, cosmic_cov=ctx.cosmic_cov, P_data=ctx.P_data,
        dla_core=ctx.dla_core, dla_shot_flag=ctx.dla_shot_flag, valid_k=ctx.valid_k,
        shot_inflate=ctx.shot_inflate, cemu_inflate=ctx.cemu_inflate,
        include_logdet=ctx.include_logdet, rho_zb=ctx.rho_zb)


# ----------------------------------------------------------------------------
# The ONE canonical pack / unpack (dict ↔ vector). Shared by BOTH samplers.
# ----------------------------------------------------------------------------
def param_names(n_z):
    """The packed-vector names: 9 cosmo (PARAM_NAMES) + per-z τ₀ + 3 HCD α.

    The FIXED layout the dict↔vector maps assert against; ``tau0_z{i}`` for i in
    range(n_z), then ``alpha_lls/alpha_subdla/alpha_dla``."""
    return (list(PARAM_NAMES)
            + [f"tau0_z{i}" for i in range(n_z)]
            + ["alpha_lls", "alpha_subdla", "alpha_dla"])


def unpack(vec, n_z):
    """Vector → (theta9 (9,), tau0_vec (n_z,), alpha_hcd (3,)). The inverse of ``pack``."""
    vec = jnp.asarray(vec)
    theta9 = vec[:9]
    tau0_vec = vec[9:9 + n_z]
    alpha_hcd = vec[9 + n_z:9 + n_z + 3]
    return theta9, tau0_vec, alpha_hcd


def pack(theta9, tau0_vec, alpha_hcd):
    """(theta9, tau0_vec, alpha_hcd) → the single packed vector (9 + n_z + 3,)."""
    return jnp.concatenate([jnp.asarray(theta9), jnp.asarray(tau0_vec),
                            jnp.asarray(alpha_hcd)])


def to_dict(vec, n_z):
    """Packed vector → ``{name: float}``. Asserts the first 9 names == PARAM_NAMES
    so the cosmo block can never silently re-order vs the emulator's input contract."""
    names = param_names(n_z)
    assert names[:9] == list(PARAM_NAMES), "cosmo block must match PARAM_NAMES order"
    return {nm: float(v) for nm, v in zip(names, np.asarray(vec))}


def from_dict(d, n_z):
    """``{name: value}`` → packed vector (the inverse of ``to_dict``). Asserts the
    first 9 names == PARAM_NAMES before pulling them in PARAM_NAMES order."""
    names = param_names(n_z)
    assert names[:9] == list(PARAM_NAMES), "cosmo block must match PARAM_NAMES order"
    return jnp.asarray([d[nm] for nm in names])
