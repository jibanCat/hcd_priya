"""hcd_priya adapter (emu-jax): observable O1 from the manifest-verified LF production ensemble.

O1 at the emulator level is P_tier_p = sum_c w_c P_filt_c, with P_filt the deployed ensemble mean of the members'
reconstructed per-class P_filt (``predict.predict_P_filt``). The class weights are the case's: ``"cache"`` takes the
matched cache row's 4-class count fractions (the simulation's own HCD content, design points only); ``"head_a"``
takes the members' mean physical Head-A dN/dX through ``dndx_wc.w_c_corrected`` with the cache Xbar(z) fit."""
from __future__ import annotations

import os
import platform
import subprocess

import numpy as np

from .. import conventions as C
from ..schema import SCHEMA, validate_record

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_CACHE_KEYS = ("params", "sim_name", "z_grid", "alpha_idx", "alpha_slope", "tier_c_counts", "snap_group_idx",
               "snap_total_path_dX", "snap_dNdX_LLS", "snap_dNdX_subDLA", "snap_dNdX_DLA")


def read_cache_weights(path):
    """The small cache arrays the weights need (never the per-class spectra)."""
    import h5py
    from hcd_analysis.emulator.data import _collapse_counts
    with h5py.File(path, "r") as h:
        d = {k: h[k][...] for k in _CACHE_KEYS}
    d["sim_name"] = np.array([s.decode() if isinstance(s, bytes) else s for s in d["sim_name"]])
    cc = _collapse_counts(d["tier_c_counts"])
    N = cc.sum(1, keepdims=True)
    d["w_c_cache"] = np.where(N > 0, cc / np.maximum(N, 1), 0.0)                 # as data.load_cache
    d["snap_dNdX"] = np.stack([d["snap_dNdX_LLS"], d["snap_dNdX_subDLA"], d["snap_dNdX_DLA"]], axis=1)
    return d


def _cache_row(d, params_vec, z, rung_index, rung):
    sim = np.all(np.abs(d["params"] - params_vec) <= 1e-6 * np.abs(params_vec), axis=1)
    rows = np.where(sim & (np.abs(d["z_grid"] - z) < 1e-6) & (d["alpha_idx"] == 2 * rung_index))[0]
    if rows.size != 1 or abs(d["alpha_slope"][rows[0]] - rung) > 1e-6:
        raise RuntimeError(f"expected one cache row for the design point at z={z}, rung {rung_index}; found {rows.size}")
    return int(rows[0])


def predict(cases, *, cache_path, manifest_path=None, commit=None):
    """A validated record for ``cases`` (each carrying ``support`` from the runner's domain check)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from hcd_analysis.emulator import prod_ensemble as PE
    from hcd_analysis.emulator.closure_legb import hcd_xbar_polyfit
    from hcd_analysis.emulator.data import PARAM_LIMITS, Z_LIMITS, normalize_params, untransform_prediction
    from hcd_analysis.emulator.dndx_wc import w_c_corrected
    from hcd_analysis.emulator.kcoord import k_skm_from_kcom
    from hcd_analysis.emulator.meanflux_prior import kim_tau0, tau0_alpha_priya
    from hcd_analysis.emulator.model import structural_tier_p
    from hcd_analysis.emulator.predict import predict_P_filt
    from hcd_analysis.emulator.safe_io import sha256_file

    ens, meta0, norm0, manifest = PE.load_production_ensemble(manifest_path=manifest_path)
    if sha256_file(cache_path) != meta0["cache_sha256"]:
        raise RuntimeError(f"{cache_path} is not the cache the production members were trained on")
    k_com = np.asarray(meta0["k_com_hmpc"], float)
    modes = list(range(1, k_com.size + 1))
    if not np.allclose(k_com, C.k_com_modes(np.asarray(modes), meta0["L_box_hmpc"]), rtol=1e-12, atol=0):
        raise RuntimeError("checkpoint k_com_hmpc are not the box modes 2 pi n / L_box")
    d = read_cache_weights(cache_path)
    xcf = hcd_xbar_polyfit(d)
    pf = norm0["P_filt"]
    out_cases = []
    for case in cases:
        st, sup = case["state"], case["support"]
        rec = {k: case[k] for k in ("case_id", "family", "z", "state")}
        rec["support"] = dict(sup)
        if sup["status"] != "COMPARABLE":
            out_cases.append(rec)
            continue
        pv = np.array([st["params"][n] for n in C.PARAM_NAMES])
        th = np.asarray(normalize_params(pv), float)
        if np.any(th < 0) or np.any(th > 1):
            raise RuntimeError(f"{case['case_id']}: outside the unit cube after the domain check")
        mf = st["mean_flux"]
        P, K, PF, W, T0 = [], [], [], [], []
        for z in case["z"]:
            zu = (z - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])
            t0 = float(tau0_alpha_priya(z, mf["tau0_amp"], mf["dtau0"], z_pivot=mf["z_pivot"]) * kim_tau0(z))
            Pf = predict_P_filt(ens, jnp.asarray(th), zu, t0, pf)
            if case["family"] == "D":
                row = _cache_row(d, pv, z, case["inputs"]["rung_index"], case["inputs"]["rung"])
                w = jnp.asarray(d["w_c_cache"][row])
            else:
                x = jnp.concatenate([jnp.asarray(th), jnp.asarray([zu])])
                dn = np.mean([untransform_prediction({"dndx": np.asarray(m(x, t0)["dndx"])[None]}, norm0)["dndx"][0]
                              for m in ens.members], axis=0)
                w = w_c_corrected(jnp.asarray(dn), jnp.asarray(np.polyval(xcf, z)), jnp.asarray(z))
            P.append(np.asarray(structural_tier_p(w, Pf)))
            K.append(np.asarray(k_skm_from_kcom(k_com, z, st["params"]["hub"], st["params"]["omegamh2"])))
            PF.append(np.asarray(Pf)); W.append(np.asarray(w)); T0.append(t0)
        rec.update(P=np.asarray(P).tolist(), k_skm=np.asarray(K).tolist(),
                   native={"P_filt_by_class": np.asarray(PF).tolist(), "w_c": np.asarray(W).tolist(),
                           "tau0_input": T0, "theta9_unit": th.tolist(),
                           "weights_source": "cache" if case["family"] == "D" else "head_a",
                           "class_order": ["clean", "LLS", "subDLA", "DLA"]})
        out_cases.append(rec)
    members = manifest["members"]
    record = {
        "schema": SCHEMA, "observable": "O1", "L_box_hmpc": float(meta0["L_box_hmpc"]), "modes": modes,
        "comparator": {
            "id": "C1", "name": "hcd_priya LF production ensemble", "repo": "https://github.com/jibanCat/hcd_priya",
            "commit": commit or subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                                               text=True).stdout.strip() or "unknown",
            "implementation": {"forward": "predict.predict_P_filt (ensemble mean) + model.structural_tier_p",
                               "param_limits": np.asarray(PARAM_LIMITS).tolist()},
            "artifacts": ([{"path": m[k], "sha256": m[k + "_sha256"]} for m in members for k in ("eqx", "norm", "meta")]
                          + [{"path": os.path.basename(cache_path), "sha256": meta0["cache_sha256"]}]),
            "fidelity": "LF",
            "training": "60 PRIYA LF sims, 20-rung mean-flux ladder, z 2.0-5.4 (shares the PRIYA LF spectra of R1)",
            "env": {"python": platform.python_version(), "jax": jax.__version__, "numpy": np.__version__}},
        "cases": out_cases}
    return validate_record(record)
