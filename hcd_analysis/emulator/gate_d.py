"""Gate D read-outs added by the blind review (PU-0063; review notes reviews/gate_D in the notes repository).

D3 (registered, PU-0053) pools the held-out log residual over the band z 2.8-3.4, k >= 0.0442 s/km. For a
leave-one-simulation-out MEAN correction that pooled value is fixed by construction: over complete (z, rung) blocks the
held-out prediction is the mean of the other simulations, so the pooled residual cancels, and on a band of common mode
indices it is zero; the registered band's small residual comes only from per-simulation band membership (k >= 0.0442
maps to different modes for different v_box). The accuracy of the correction at high k is therefore read per held-out
simulation."""
from __future__ import annotations

import numpy as np

CLS = ("clean", "LLS", "subDLA", "DLA")
D3_Z = (2.8, 3.4)
D3_KMIN = 0.0442


def _zsel(z):
    z = np.asarray(z, float)
    return (z >= D3_Z[0] - 1e-9) & (z <= D3_Z[1] + 1e-9)


def _keep3(keep, n_cls):
    keep = np.asarray(keep, bool)
    return np.broadcast_to(keep[:, None, :], (keep.shape[0], n_cls, keep.shape[1])) if keep.ndim == 2 else keep


def registered_band(z, k_hr, keep, n_cls=4):
    """The registered D3 band (M, C, K): kept entries with z in [2.8, 3.4] and k >= 0.0442 on each row's own grid."""
    k = np.asarray(k_hr, float)
    return _keep3(keep, n_cls) & _zsel(z)[:, None, None] & (k >= D3_KMIN)[:, None, :]


def common_mode_band(z, k_hr, keep):
    """(M, K): rows with z in [2.8, 3.4], modes n >= n0 where n0 is the largest, over those rows, first mode with
    k >= 0.0442 (the same mode indices for every simulation)."""
    k = np.asarray(k_hr, float)
    rows = _zsel(z)
    first = np.argmax(np.where(np.isfinite(k[rows]), k[rows], -np.inf) >= D3_KMIN, axis=1)
    n0 = int(first.max())
    band = np.zeros(k.shape, bool)
    band[rows, n0:] = True
    keep = np.asarray(keep, bool)
    k2 = keep if keep.ndim == 2 else keep.all(axis=1)
    return band & k2


def _mean(l, mask):
    return float(np.nanmean(np.where(mask, l, np.nan)))


def d3_per_simulation(l_mf, l_lf, sim, z, k_hr, keep):
    """D3 read per held-out simulation: signed band means of log(P/P_HR) for MF and for LF alone, their RMS over
    simulations, the pooled registered-band value (the registered D3) and the pooled value on the common mode-index
    band (zero for a leave-one-out mean correction)."""
    l_mf, l_lf = np.asarray(l_mf, float), np.asarray(l_lf, float)
    sim = np.asarray(sim).astype(str)
    n_cls = l_mf.shape[1]
    band = registered_band(z, k_hr, keep, n_cls)
    common = common_mode_band(z, k_hr, keep)[:, None, :] & _keep3(keep, n_cls)
    out = {"pooled_registered_band": {}, "pooled_common_mode_band": {}, "per_simulation": {},
           "rms_over_simulations": {},
           "common_mode_band_first_mode": int(np.argmax(common_mode_band(z, k_hr, keep).any(axis=0))) + 1}
    for c, name in enumerate(CLS[:n_cls]):
        out["pooled_registered_band"][name] = dict(mf=_mean(l_mf[:, c], band[:, c]), lf=_mean(l_lf[:, c], band[:, c]))
        out["pooled_common_mode_band"][name] = dict(mf=_mean(l_mf[:, c], common[:, c]),
                                                    lf=_mean(l_lf[:, c], common[:, c]))
    sims = sorted(set(sim))
    for s in sims:
        r = sim == s
        out["per_simulation"][s] = {name: dict(mf=_mean(l_mf[r, c], band[r, c]), lf=_mean(l_lf[r, c], band[r, c]))
                                    for c, name in enumerate(CLS[:n_cls])}
    for name in CLS[:n_cls]:
        out["rms_over_simulations"][name] = {
            arm: float(np.sqrt(np.mean([out["per_simulation"][s][name][arm] ** 2 for s in sims]))) for arm in ("mf", "lf")}
    return out
