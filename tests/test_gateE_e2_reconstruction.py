"""Gate E criterion E2 (GATE_E_SPEC v1 section 4, PU-0061): the production forward's model P on every kept bin of every
leg equals an independent reconstruction (tests/gate_e_reconstruction.py, import-firewalled) to <= 1e-10 relative, at
the E1 theta set, with stated nuisance values: per-z tau0 = Becker13 at the leg z, alpha_c = the HCD prior centres,
Model C+ metal nodes and the resolution amplitude at the values below. The covariance part is registered with the
amendment (S10-S12)."""
import ast
from pathlib import Path

import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401
import jax.numpy as jnp

from hcd_analysis.emulator import forward as FW
from hcd_analysis.emulator.data import PARAM_LIMITS
from tests.test_gateE_e1_e4 import GATEC, MF, _inputs, _need, _theta_set, prod_ctx  # noqa: F401

RECON = Path(__file__).with_name("gate_e_reconstruction.py")
ALLOWED = {"json", "pickle", "numpy", "jax", "equinox", "h5py", "hcd_analysis.emulator.model"}
NUIS = dict(f_SiIII_nodes=jnp.asarray([0.006, 0.012]), f_SiII_nodes=jnp.asarray([0.002, 0.003]),
            k_SiIII_nodes=jnp.asarray([0.01, 0.006]), k_SiII_nodes=jnp.asarray([0.005, 0.004]),
            metal_node_z=(2.2, 4.2), b_res=0.03)


def test_reconstruction_import_firewall():
    tree = ast.parse(RECON.read_text())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            assert n.level == 0, "no relative imports in the reconstruction"
            mods.add(n.module)
    assert mods <= ALLOWED, mods - ALLOWED


def test_e2_forward_equals_independent_reconstruction(prod_ctx):
    import tests.gate_e_reconstruction as R
    from hcd_analysis.emulator.mf_modes import load_mode_mf
    members = [R.load_member(f"{GATEC}/prod_repaired_seed{i}") for i in range(5)]
    tables, _, _ = load_mode_mf(MF)          # the product's arrays only; evaluated by R.mf_g
    tau0, alpha, cores = _inputs(prod_ctx)
    zg = np.asarray(prod_ctx.z_global)
    worst = 0.0
    for th in _theta_set():
        for leg in prod_ctx.legs:
            sel = np.array([int(np.argmin(np.abs(zg - z))) for z in leg.z])
            P_fw = np.asarray(FW.predict_leg(prod_ctx.model, jnp.asarray(th), tau0[sel], alpha[sel], leg=leg,
                                             k_com=prod_ctx.k_com_hmpc, pf_stats=prod_ctx.pf_stats,
                                             dla_core=cores[leg.name], mf=prod_ctx.mf, nuis=NUIS).P_model)
            legd = dict(z=np.asarray(leg.z), k=np.asarray(leg.k), z_idx=np.asarray(leg.z_idx), R_z=np.asarray(leg.R_z),
                        metals_on=bool(leg.metals_on), resolution_on=bool(leg.resolution_on),
                        dla_forward_frac=float(getattr(leg, "dla_forward_frac", 1.0)))
            nuis = {k: (np.asarray(v) if not isinstance(v, (tuple, float)) else v) for k, v in NUIS.items()}
            P_rc = R.reconstruct_leg(members, tables, legd, np.asarray(th), np.asarray(tau0)[sel],
                                     np.asarray(alpha)[sel], np.asarray(cores[leg.name]), PARAM_LIMITS, nuis)
            keep = np.isfinite(np.asarray(leg.P_data))
            worst = max(worst, float(np.max(np.abs(P_fw[keep] / P_rc[keep] - 1))))
    assert worst <= 1e-10, worst
