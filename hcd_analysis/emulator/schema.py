"""Cache and checkpoint SCHEMA: every key carries a class; collapsing a per-row key is an error.
Boundary of the 2026 k-grid representation regression; see hcd_priya_notes
docs/superpowers/emulator-paper-history/2026-10-05-INCIDENT-kgrid-representation-regression.md."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

L_BOX_HMPC = 120.0                     # PRIYA box, comoving Mpc/h (LF and HR)
CHECKPOINT_SCHEMA_VERSION = "2.0"
K_CONVENTION = ("angular s/km; k_com = 2 pi n / L_box [h/Mpc], n = 1..n_k; "
                "k_skm = k_com (1+z) / (100 E(z)), E = sqrt(Om (1+z)^3 + 1 - Om), "
                "Om = omegamh2/hub^2, flat LCDM, no radiation")
KCOM_RTOL = 1e-9                       # k_com constancy across rows
VBOX_RTOL = 2e-4                       # stored v_box vs analytic flat-LCDM v_box


class KeyClass(Enum):
    GLOBAL_STATIC = "GLOBAL_STATIC"          # identical for the whole product (validated, never read from row 0)
    PER_ROW = "PER_ROW"                      # varies by (simulation, snapshot, rung)
    PER_SNAPSHOT = "PER_SNAPSHOT"            # one entry per (simulation, snapshot), indexed by snap_group_idx
    STATIC_TABLE = "STATIC_TABLE"            # product-wide constant table, not row-indexed
    PER_QUERY_DERIVED = "PER_QUERY_DERIVED"  # recomputed from the prediction request (z, theta)


class SchemaCollapseError(ValueError):
    """A per-row quantity was collapsed, a global invariant does not hold, or a key is unregistered."""


@dataclass(frozen=True)
class KeySpec:
    cls: KeyClass
    tol: float | None
    doc: str


_PR = KeyClass.PER_ROW
_PS = KeyClass.PER_SNAPSHOT
_ST = KeyClass.STATIC_TABLE
CACHE_SCHEMA_V33: dict[str, KeySpec] = {
    # per-row physical coordinates and targets (stored)
    "kfkms": KeySpec(_PR, None, "this row's own native FFT grid, angular s/km, = 2 pi n / v_box(row)"),
    "nbins_native": KeySpec(_PR, None, "pixels per sightline of this snapshot"),
    "dv_kms": KeySpec(_PR, None, "pixel width km/s; nbins_native * dv_kms = v_box(row)"),
    "params": KeySpec(_PR, None, "physical cosmology/IGM parameters (varies by simulation)"),
    "z_grid": KeySpec(_PR, None, "PRIYA grid redshift of the snapshot"),
    "z_meta": KeySpec(_PR, None, "snapshot header redshift"),
    "alpha_idx": KeySpec(_PR, None, "mean-flux ladder rung index"),
    "alpha_slope": KeySpec(_PR, None, "ladder alpha = tau0 / Kim(z)"),
    "target_F": KeySpec(_PR, None, "imposed mean flux"),
    "scale": KeySpec(_PR, None, "tau rescale reaching target_F"),
    "P_tier_p": KeySpec(_PR, None, "filtered total P1D on this row's grid"),
    "P_tier_c": KeySpec(_PR, None, "per-fine-class unfiltered P1D on this row's grid"),
    "P_tier_c_filtered": KeySpec(_PR, None, "per-fine-class filtered P1D on this row's grid"),
    "tier_c_counts": KeySpec(_PR, None, "sightline counts per fine class"),
    "mean_F_by_bin": KeySpec(_PR, None, "mean flux per fine class"),
    "sim_name": KeySpec(_PR, None, "simulation name"),
    "snap": KeySpec(_PR, None, "snapshot number"),
    "snap_group_idx": KeySpec(_PR, None, "index into the PER_SNAPSHOT tables"),
    # per-row products of load_cache (derived)
    "P_filt": KeySpec(_PR, None, "4-class filtered P1D (derived)"),
    "delta": KeySpec(_PR, None, "HCD add-back (derived)"),
    "coarse_counts": KeySpec(_PR, None, "4-class counts (derived)"),
    "tau0": KeySpec(_PR, None, "-ln target_F (derived)"),
    "mask": KeySpec(_PR, None, "finite P_tier_p bins (derived)"),
    "w_c_cache": KeySpec(_PR, None, "class fractions (derived)"),
    "inv_nc": KeySpec(_PR, None, "inverse counts (derived)"),
    "params_unit": KeySpec(_PR, None, "unit-cube parameters (derived)"),
    "x": KeySpec(_PR, None, "encoder input (derived)"),
    "in_domain": KeySpec(_PR, None, "in-box flag (derived)"),
    # per-snapshot tables
    "snap_dNdX": KeySpec(_PS, None, "dN/dX per class"),
    "snap_f_nhi": KeySpec(_PS, None, "column-density distribution"),
    "snap_total_path_dX": KeySpec(_PS, None, "absorption path"),
    "snap_n_absorbers": KeySpec(_PS, None, "absorber counts"),
    "snap_dNdX_LLS": KeySpec(_PS, None, "dN/dX LLS"),
    "snap_dNdX_subDLA": KeySpec(_PS, None, "dN/dX subDLA"),
    "snap_dNdX_DLA": KeySpec(_PS, None, "dN/dX DLA"),
    "snap_sim_name": KeySpec(_PS, None, "simulation per snapshot"),
    "snap_snap": KeySpec(_PS, None, "snapshot number per snapshot"),
    # static tables
    "log_nhi_centres": KeySpec(_ST, None, "column-density bin centres"),
    "log_nhi_edges": KeySpec(_ST, None, "column-density bin edges"),
    "param_names": KeySpec(_ST, None, "parameter names"),
    "tier_c_labels": KeySpec(_ST, None, "fine-class labels"),
    "tier_c_nhi_edges": KeySpec(_ST, None, "fine-class edges"),
    "schema_report": KeySpec(_ST, None, "validator report attached by load_cache"),
    # global invariants DERIVED from per-row keys and validated across ALL rows
    "k_com_hmpc": KeySpec(KeyClass.GLOBAL_STATIC, KCOM_RTOL, "2 pi n / L_box; = kfkms * v_box / L_box for every row"),
    "L_box_hmpc": KeySpec(KeyClass.GLOBAL_STATIC, VBOX_RTOL, "120; checked via v_box = L 100 E(z)/(1+z)"),
}


def _vbox_analytic(z, params):
    hub = params[:, 5]
    om = params[:, 6] / hub ** 2
    E = np.sqrt(om * (1 + z) ** 3 + 1 - om)
    return L_BOX_HMPC * 100.0 * E / (1 + z)


def k_com_hmpc_from_cache(d):
    """GLOBAL_STATIC comoving modes, validated across ALL rows (never row 0 alone)."""
    kf = np.asarray(d["kfkms"], float)
    vbox = np.asarray(d["nbins_native"], float) * np.asarray(d["dv_kms"], float)
    kcom_rows = kf * vbox[:, None] / L_BOX_HMPC                      # (R, K)
    fin = np.isfinite(kcom_rows)
    ref = np.nanmedian(np.where(fin, kcom_rows, np.nan), axis=0)       # (K,)
    dev = np.nanmax(np.abs(np.where(fin, kcom_rows, ref) / ref - 1.0))
    if not np.isfinite(dev) or dev > KCOM_RTOL:
        raise SchemaCollapseError(
            f"k_com_hmpc is declared GLOBAL_STATIC but varies across rows by {dev:.3e} > {KCOM_RTOL}")
    return ref


def validate_cache_schema(d, *, n_rows=None):
    """Check every key of a loaded cache dict against CACHE_SCHEMA_V33 and prove the GLOBAL_STATIC invariants.

    Returns {"n_rows", "checked", "derived": {"k_com_hmpc", "vbox_rtol_measured"}}; raises SchemaCollapseError."""
    R = int(n_rows if n_rows is not None else np.asarray(d["z_grid"]).shape[0])
    checked = []
    for k, v in d.items():
        if k not in CACHE_SCHEMA_V33:
            arr = np.asarray(v) if not isinstance(v, dict) else None
            if arr is not None and arr.ndim >= 1 and arr.shape[0] == R:
                raise SchemaCollapseError(
                    f"unregistered row-indexed key {k!r} (shape {arr.shape}); register it with a KeyClass")
            continue
        spec = CACHE_SCHEMA_V33[k]
        if spec.cls is KeyClass.PER_ROW:
            arr = np.asarray(v)
            if arr.ndim == 0 or arr.shape[0] != R:
                raise SchemaCollapseError(
                    f"PER_ROW key {k!r} has shape {arr.shape}, expected leading dimension {R}: it was collapsed")
            if R > 1 and arr.dtype.kind in "fiub":
                flat = arr.reshape(R, -1)
                if np.all(np.isclose(flat, flat[0:1], rtol=1e-12, atol=0, equal_nan=True)):
                    raise SchemaCollapseError(
                        f"PER_ROW key {k!r} is identical on every row: a collapsed copy of row 0 was stored")
            checked.append(k)
    derived = {"k_com_hmpc": k_com_hmpc_from_cache(d)}
    vb = np.asarray(d["nbins_native"], float) * np.asarray(d["dv_kms"], float)
    ratio = vb / _vbox_analytic(np.asarray(d["z_grid"], float), np.asarray(d["params"], float))
    dev = float(np.max(np.abs(ratio - 1.0)))
    if dev > VBOX_RTOL:
        raise SchemaCollapseError(
            f"L_box_hmpc invariant fails: v_box / analytic deviates by {dev:.3e} > {VBOX_RTOL}")
    derived["vbox_rtol_measured"] = dev
    return {"n_rows": R, "checked": checked, "derived": derived}
