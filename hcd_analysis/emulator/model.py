"""Equinox emulator: encoder + Head A (tau0-invariant) + Head B (tau0-dependent).

Head A consumes ONLY the latent (no tau0) so its outputs (f_nhi, dN/dX) are
tau0-invariant by construction (spec sec.3). Head B consumes concat(latent, tau0).
P_tier_p is the structural sum Sum_c w_c * P_c^filt (spec sec.3/sec.6).
"""
from __future__ import annotations
import jax, jax.numpy as jnp, equinox as eqx


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


class HeadB(eqx.Module):
    """tau0-dependent head: 4 filtered class P1D (log-space) + 3 HCD deltas.

    Reads concat(latent, tau0). The 7*n_k output is reshaped to (7, n_k) and
    split into 4 P_filt (clean,LLS,subDLA,DLA) and 3 delta (LLS,subDLA,DLA).
    """
    trunk: eqx.nn.Linear
    out: eqx.nn.Linear
    n_k: int = eqx.field(static=True)

    def __init__(self, latent=64, n_k=172, key=None):
        k1, k2 = jax.random.split(key, 2)
        self.n_k = n_k
        self.trunk = eqx.nn.Linear(latent + 1, 256, key=k1)
        self.out = eqx.nn.Linear(256, 7 * n_k, key=k2)

    def __call__(self, latent, tau0):
        h = jax.nn.gelu(self.trunk(jnp.concatenate([latent, jnp.atleast_1d(tau0)])))
        y = self.out(h).reshape(7, self.n_k)
        return {"P_filt": y[:4], "delta": y[4:]}


class Emulator(eqx.Module):
    """Full emulator: Encoder -> {HeadA (tau0-invariant), HeadB (tau0-dependent)}."""
    enc: Encoder
    head_a: HeadA
    head_b: HeadB

    def __init__(self, in_dim=10, n_k=172, key=None):
        k1, k2, k3 = jax.random.split(key, 3)
        self.enc = Encoder(in_dim=in_dim, key=k1)
        self.head_a = HeadA(latent=64, key=k2)
        self.head_b = HeadB(latent=64, n_k=n_k, key=k3)

    def __call__(self, x, tau0):
        lat = self.enc(x)
        return {**self.head_a(lat), **self.head_b(lat, tau0)}


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
    channel counts; term_w optionally rescales the named terms."""
    term_w = term_w or {"f_nhi": 1.0, "dndx": 1.0, "P_filt": 1.0, "delta": 1.0, "meanF": 0.1}
    preds = jax.vmap(model)(batch["x"], batch["tau0"])
    la_cddf = masked_mse(preds["f_nhi"], batch["t_f_nhi"],
                         jnp.ones_like(batch["t_f_nhi"], bool),
                         weight=batch["inv_nalpha"][:, None])
    la_dndx = masked_mse(preds["dndx"], batch["t_dndx"],
                         jnp.ones_like(batch["t_dndx"], bool),
                         weight=batch["inv_nalpha"][:, None])
    m3 = batch["mask"][:, None, :]
    wcls4 = batch["inv_nc"][:, :, None]
    wcls3 = batch["inv_nc"][:, 1:, None]
    lb_pf = masked_mse(preds["P_filt"], batch["t_P_filt"],
                       m3 & jnp.ones_like(batch["t_P_filt"], bool), weight=wcls4)
    lb_dl = masked_mse(preds["delta"], batch["t_delta"],
                       m3 & jnp.ones_like(batch["t_delta"], bool), weight=wcls3)
    l_meanF = jnp.mean((batch["mean_F_clean"] - jnp.exp(-batch["tau0"])) ** 2)
    return (term_w["f_nhi"]*la_cddf + term_w["dndx"]*la_dndx
            + term_w["P_filt"]*lb_pf + term_w["delta"]*lb_dl
            + term_w["meanF"]*l_meanF)
