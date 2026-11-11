"""The bounded case matrix of a comparison config (families D and I of phase0_v1) and its mechanical domain check."""

from __future__ import annotations

import json
import os

import numpy as np

from . import conventions as C

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(name="phase0_v1"):
    with open(os.path.join(CONFIG_DIR, f"{name}.json")) as f:
        return json.load(f)


def limits_by_name(param_limits):
    """{name: (lo, hi)} from a (9, 2) limits array in PARAM_NAMES order."""
    lim = np.asarray(param_limits, float)
    if lim.shape != (len(C.PARAM_NAMES), 2):
        raise C.ConventionError(f"limits shape {lim.shape}")
    return {n: (float(lo), float(hi)) for n, (lo, hi) in zip(C.PARAM_NAMES, lim)}


def unit_to_params(u, limits):
    u = np.asarray(u, float)
    return {
        n: float(limits[n][0] + u[i] * (limits[n][1] - limits[n][0]))
        for i, n in enumerate(C.PARAM_NAMES)
    }


def _state(params, tau0_amp, dtau0, z_pivot):
    return {
        "params": params,
        "mean_flux": {
            "tau0_amp": float(tau0_amp),
            "dtau0": float(dtau0),
            "z_pivot": float(z_pivot),
        },
    }


def family_D(cfg, ks_design_params, ks_param_index, ks_rungs):
    """``ks_design_params`` (n_sims, 9) in the reference's own order ``ks_param_index`` ({name: column})."""
    f = cfg["families"]["D"]
    names = sorted(ks_param_index, key=ks_param_index.get)
    out = []
    for i in f["ks_lf_sims"]:
        prm = C.params_from_vector(ks_design_params[i], names)
        for j in f["rung_indices"]:
            out.append(
                {
                    "case_id": f"D-s{i:02d}-r{j}",
                    "family": "D",
                    "z": list(cfg["z_grid"]),
                    "state": _state(prm, ks_rungs[j], f["dtau0"], cfg["z_pivot"]),
                    "inputs": {
                        "ks_lf_index": int(i),
                        "rung_index": int(j),
                        "rung": float(ks_rungs[j]),
                    },
                }
            )
    return out


def family_I(cfg, limits):
    from scipy.stats import qmc

    f = cfg["families"]["I"]
    pts = {
        "I1": dict(f["fiducial"]),
        "I2": unit_to_params(np.full(9, f["box_centre_unit"]), limits),
    }
    s = f["sobol"]
    u = qmc.Sobol(d=9, scramble=s["scramble"], seed=s["seed"]).random_base2(m=s["m"])[: s["take"]]
    for k, row in enumerate(u):
        pts[f"I{3 + k}"] = unit_to_params(s["unit_lo"] + s["unit_span"] * row, limits)
    out = []
    for pid, prm in pts.items():
        for mid, (amp, dt) in f["mean_flux_states"].items():
            out.append(
                {
                    "case_id": f"{pid}-{mid}",
                    "family": "I",
                    "z": list(cfg["z_grid"]),
                    "state": _state(prm, amp, dt, cfg["z_pivot"]),
                    "inputs": {},
                }
            )
    return out


def domain_support(case, cfg, limits):
    """COMPARABLE, or NOT COMPARABLE with the reason (predeclared rule of the spec, section 4)."""
    st = case["state"]
    if not C.in_limits(st["params"], limits):
        return {"status": "NOT COMPARABLE", "reason": "parameter outside the emulator design box"}
    a = C.alpha_priya(np.asarray(case["z"]), st["mean_flux"])
    d = cfg["domain"]
    if np.any(a < d["alpha_min"]) or np.any(a > d["alpha_max"]):
        return {
            "status": "NOT COMPARABLE",
            "reason": f"alpha(z) outside the reference rungs [{d['alpha_min']}, "
            f"{d['alpha_max']}]",
        }
    if any(min(abs(z - g) for g in cfg["z_grid"]) > 1e-9 for z in case["z"]):
        return {"status": "NOT COMPARABLE", "reason": "z not on the reference grid"}
    return {"status": "COMPARABLE", "reason": ""}
