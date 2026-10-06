"""LF -> HF multi-fidelity correction measured MODE BY MODE at equal physical k (gate D, emulator-debug campaign 2026-10;
spec GATE_D_SPEC.md in the notes repository; incident note 2026-10-05-INCIDENT-kgrid-representation-regression).

LF and HR runs of one design point share box, cosmology and snapshot redshifts, so the n-th comoving mode is the same
physical k in both. The correction g = log P_HR - log P_LF is measured per mode (no interpolation of either side) and
placed on k_skm(z, theta) only at prediction. Model family unchanged from the certified historical form
(``multifidelity.fixed_mean_table_resolved``: log_rho + gbar_z + gbar_tau + rank-1, theta-independent), with one
correctness fix: log_rho is fitted on the TRAINING rows only (historically it used every matched row, so a held-out HR
simulation leaked into its own correction)."""
from __future__ import annotations

import json

import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp

from .multifidelity import FixedMeanHead, fixed_mean_table_resolved, make_cond, match_hr_to_lf
from .schema import CHECKPOINT_SCHEMA_VERSION, L_BOX_HMPC

TABLE_KEYS = ("log_rho", "gbar_z_tab", "gtau_tab", "a_k", "u_z", "u_tau", "z_tab", "tau_tab", "tau_by_z")
_VELOCITY_KEYS = ("kfkms", "k_skm", "k_kms", "kgrid")


def match_pairs(lf, hr):
    """[(hr_row, lf_row)] matched exactly on (theta, z_grid, rung)."""
    return match_hr_to_lf(lf, hr)


def mode_mapping_error(lf, hr, pairs, n_modes):
    """max over matched pairs and modes 1..n_modes of |k_HR,n / k_LF,n - 1| (stored per-row grids)."""
    worst = 0.0
    for h, l in pairs:
        a = np.asarray(hr["kfkms"][h][:n_modes], float)
        b = np.asarray(lf["kfkms"][l][:n_modes], float)
        m = np.isfinite(a) & np.isfinite(b)
        worst = max(worst, float(np.max(np.abs(a[m] / b[m] - 1.0))))
    return worst


def canonical_mapping_error(hr, pairs, k_com_hmpc):
    """max over matched HR rows of |k_skm(z, theta)_n / k_HR,n - 1| with k_skm from the canonical coordinate."""
    from .kcoord import k_skm_from_kcom
    k_com = np.asarray(k_com_hmpc, float)
    worst = 0.0
    for h, _ in pairs:
        hub, omh2 = float(hr["params"][h][5]), float(hr["params"][h][6])
        k = np.asarray(k_skm_from_kcom(k_com, float(hr["z_grid"][h]), hub, omh2))
        kh = np.asarray(hr["kfkms"][h][:k_com.size], float)
        m = np.isfinite(kh)
        worst = max(worst, float(np.max(np.abs(k[m] / kh[m] - 1.0))))
    return worst


def measure_mode_targets(lf, hr, pairs, P_lf, *, x, n_modes):
    """g = log P_HR - log P_LF per matched pair, class and mode 1..n_modes. ``P_lf`` (M, 4, n_modes) is the LF
    prediction for the matched LF rows in ``pairs`` order; ``x`` the LF encoder inputs indexed by LF row. NaN where
    either side is non-positive or non-finite (e.g. an empty HCD class)."""
    h = np.array([p[0] for p in pairs]); l_ = np.array([p[1] for p in pairs])
    P_hr = np.asarray(hr["P_filt"])[h][:, :, :n_modes]
    P_lf = np.asarray(P_lf, float)
    ok = np.isfinite(P_hr) & np.isfinite(P_lf) & (P_hr > 0) & (P_lf > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(ok, np.log(np.where(ok, P_hr, 1.0)) - np.log(np.where(ok, P_lf, 1.0)), np.nan)
    return dict(x=np.asarray(x)[l_], tau0=np.asarray(lf["tau0"])[l_], alpha_idx=np.asarray(lf["alpha_idx"])[l_],
                sim=np.asarray(lf["sim_name"])[l_].astype(str), hr_row=h, lf_row=l_, g=g)


def fit_mode_mf(targets, train_rows=None):
    """Fit the correction tables on ``train_rows`` only (log_rho included)."""
    rows = np.arange(targets["g"].shape[0]) if train_rows is None else np.asarray(train_rows)
    with np.errstate(invalid="ignore"):
        log_rho = np.nanmean(targets["g"][rows].reshape(-1, targets["g"].shape[-1]), axis=0)
    log_rho = np.nan_to_num(log_rho, nan=0.0)
    comp = fixed_mean_table_resolved(targets, log_rho, train_mask_rows=rows)
    return dict(log_rho=log_rho, **{k: comp[k] for k in TABLE_KEYS if k != "log_rho"})


def _head(tables):
    return FixedMeanHead(tables["gbar_z_tab"], tables["z_tab"], resolved=True, gtau_tab=tables["gtau_tab"],
                         tau_tab=tables["tau_tab"], tau_by_z=tables["tau_by_z"], a_k=tables["a_k"],
                         u_z=tables["u_z"], u_tau=tables["u_tau"])


class ModeMF(eqx.Module):
    """The MF correction g (4, n_modes) on the mode axis, JAX-pure: log_rho + the resolved FixedMeanHead. It reads
    only z_unit (x[9]) and tau0, so dg / d theta_i == 0 exactly for every cosmology parameter."""
    log_rho: jax.Array
    head: FixedMeanHead

    @classmethod
    def from_tables(cls, tables):
        bad = [k for k in TABLE_KEYS if not np.all(np.isfinite(np.asarray(tables[k], float)))]
        if bad:
            raise ValueError(f"MF tables must be finite: {bad}")
        return cls(jnp.asarray(tables["log_rho"]), _head(tables))

    def __call__(self, x, tau0):
        return self.log_rho[None, :] + self.head(make_cond(jnp.asarray(x), jnp.asarray(tau0)), None)

    def rank1_off(self):
        """The same correction with the rank-1 (z x tau0) interaction removed (a_k = 0; aliasing probe E6)."""
        return eqx.tree_at(lambda m: m.head.a_k, self, jnp.zeros_like(self.head.a_k))


def apply_mode_mf(tables, x, tau0):
    """The correction g (4, n_modes) at one LF encoder input ``x`` (10,) and mean-flux input ``tau0`` (numpy)."""
    return np.asarray(ModeMF.from_tables(tables)(x, tau0))


def save_mode_mf(path, tables, *, k_com_hmpc, provenance):
    """Write the correction tables labelled by the comoving modes (schema 2.0). Refuses velocity-labelled keys and any
    k_com that is not 2 pi n / L."""
    bad = [k for k in tables if any(v in k for v in _VELOCITY_KEYS)]
    if bad:
        raise ValueError(f"velocity-labelled keys refused in an MF product: {bad}")
    k_com = np.asarray(k_com_hmpc, float)
    expect = 2 * np.pi * np.arange(1, k_com.size + 1) / L_BOX_HMPC
    if not np.allclose(k_com, expect, rtol=1e-9, atol=0):
        raise ValueError("k_com_hmpc is not 2 pi n / L (n = 1..K)")
    for k in TABLE_KEYS:
        if k not in tables:
            raise ValueError(f"missing table {k}")
    np.savez(path, schema_version=CHECKPOINT_SCHEMA_VERSION, k_com_hmpc=k_com,
             provenance=json.dumps(provenance, sort_keys=True), **{k: np.asarray(tables[k]) for k in TABLE_KEYS})


def load_mode_mf(path):
    with np.load(path, allow_pickle=False) as f:
        if str(f["schema_version"]) != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"{path}: MF product schema {f['schema_version']} != {CHECKPOINT_SCHEMA_VERSION}")
        tables = {k: np.asarray(f[k]) for k in TABLE_KEYS}
        return tables, np.asarray(f["k_com_hmpc"]), json.loads(str(f["provenance"]))
