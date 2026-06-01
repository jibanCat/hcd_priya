"""v3.3 cache loader for the Phase-2b emulator. Returns numpy arrays (host side).

15->4 coarse-class collapse, tau0 coordinate, native-Nyquist NaN masks, and the
analytic 1/n_c sample-variance weights. See spec sec.4, sec.7, sec.9.
"""
from __future__ import annotations
import h5py, numpy as np

COARSE_SLICES = (slice(0, 1), slice(1, 8), slice(8, 13), slice(13, 15))
COARSE_NAMES = ("clean", "LLS", "subDLA", "DLA")

def _collapse_counts(counts15):
    return np.stack([counts15[:, s].sum(1) for s in COARSE_SLICES], axis=1)

def _collapse_p1d(p15, counts15):
    """Count-weighted collapse of fine-bin P1D to coarse classes (R,15,K)->(R,4,K).
    Empty coarse classes -> NaN row. NaN-safe over fine bins."""
    R, _, K = p15.shape
    out = np.full((R, 4, K), np.nan)
    for ci, s in enumerate(COARSE_SLICES):
        c = counts15[:, s].astype(float)
        w = c / np.where(c.sum(1, keepdims=True) == 0, 1.0, c.sum(1, keepdims=True))
        seg = p15[:, s, :]
        contrib = np.where(np.isfinite(seg), seg, 0.0) * w[:, :, None]
        summ = contrib.sum(1)
        allnan = (~np.isfinite(seg)).all(1)
        summ[allnan] = np.nan
        out[:, ci, :] = summ
    return out

def load_cache(path):
    with h5py.File(path, "r") as h:
        assert h.attrs["cache_version"] == "3.3", h.attrs.get("cache_version")
        g = lambda k: h[k][:]
        counts15 = g("tier_c_counts")
        Pf15, Pu15 = g("P_tier_c_filtered"), g("P_tier_c")
        out = dict(
            params=g("params").astype(np.float64),
            kfkms=g("kfkms").astype(np.float64),
            P_tier_p=g("P_tier_p").astype(np.float64),
            target_F=g("target_F").astype(np.float64),
            scale=g("scale").astype(np.float64),
            z_grid=g("z_grid").astype(np.float64),
            z_meta=g("z_meta").astype(np.float64),
            alpha_idx=g("alpha_idx").astype(np.int32),
            snap_group_idx=g("snap_group_idx").astype(np.int32),
            sim_name=np.array([s.decode() if isinstance(s, bytes) else s
                               for s in h["sim_name"][:]]),
            snap_dNdX=np.stack([g("snap_dNdX_LLS"), g("snap_dNdX_subDLA"),
                                g("snap_dNdX_DLA")], axis=1),
            snap_f_nhi=g("snap_f_nhi").astype(np.float64),
            snap_total_path_dX=g("snap_total_path_dX").astype(np.float64),
            snap_n_absorbers=g("snap_n_absorbers").astype(np.float64),
        )
    coarse_counts = _collapse_counts(counts15)
    Pf4 = _collapse_p1d(Pf15, counts15)
    Pu4 = _collapse_p1d(Pu15, counts15)
    out["P_filt"] = Pf4
    out["delta"] = (Pu4 - Pf4)[:, 1:, :]
    out["coarse_counts"] = coarse_counts
    out["tau0"] = -np.log(out["target_F"])
    out["mask"] = np.isfinite(out["P_tier_p"])
    N = coarse_counts.sum(1, keepdims=True)
    out["w_c_cache"] = coarse_counts / N
    out["inv_nc"] = np.where(coarse_counts > 0, 1.0 / np.maximum(coarse_counts, 1), 0.0)
    return out
