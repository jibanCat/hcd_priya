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
from tests.test_gateE_e1_e4 import DLA_CORE, GATEC, MF, _inputs, _need, _theta_set, prod_ctx  # noqa: F401

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
    with np.load(DLA_CORE) as f:              # the product's arrays only; read by R.core_at
        core_grid = {l.name: (np.asarray(f["k_grid"]), np.asarray(f[f"core_{l.name}"])) for l in prod_ctx.legs}
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
                                     np.asarray(alpha)[sel], core_grid[leg.name], PARAM_LIMITS, nuis)
            keep = np.isfinite(np.asarray(leg.P_data))
            worst = max(worst, float(np.max(np.abs(P_fw[keep] / P_rc[keep] - 1))))
    assert worst <= 1e-10, worst


# ------------------------------------------------------------------------------------------------ E2 (covariance part)
T1_PRODUCT = "/nfs/turbo/umor-yueyingn/mfho/hcd/emulator_v2/gateE/products/t1_gateE.npz"
T3_PRODUCT = "/nfs/turbo/umor-yueyingn/mfho/hcd/emulator_v2/gateE/products/t3_gateE.npz"


@pytest.fixture(scope="module")
def prod_ctx_cemu():
    from hcd_analysis.emulator import closure_legb as CL
    _need(f"{GATEC}/prod_repaired_seed0.eqx", MF, DLA_CORE, T1_PRODUCT, T3_PRODUCT)
    ctx, _ = CL.build_legb_ctx(ensemble_ckpts=[f"{GATEC}/prod_repaired_seed{i}" for i in range(5)], mf_product=MF,
                               dla_core_product=DLA_CORE, t1_product=T1_PRODUCT, t3_product=T3_PRODUCT,
                               ks_kwargs=dict(resolution_float=True, k_max=0.065), with_eboss=True, metals_on=True,
                               sample_res=True)
    return ctx


def test_e2_covariance_equals_independent_reconstruction(prod_ctx_cemu):
    """Amendment A1 rev 1 section 6, E2 (covariance part): C_emu on every kept bin pair of every leg (T1 with the
    data-bin algebra and the physical-k DLA core, T3, the production assembly) equals the reconstruction to <= 1e-10
    relative (|dC_ij| / sqrt(C_ii C_jj)) at the E1 theta set. T2 joins with amendment A2."""
    import tests.gate_e_reconstruction as R
    ctx = prod_ctx_cemu
    members = [R.load_member(f"{GATEC}/prod_repaired_seed{i}") for i in range(5)]
    tau0, alpha, cores = _inputs(ctx)
    with np.load(DLA_CORE) as f:
        core_grid = {l.name: (np.asarray(f["k_grid"]), np.asarray(f[f"core_{l.name}"])) for l in ctx.legs}
    with np.load(T1_PRODUCT) as f:
        rho_all, zc, centres = np.asarray(f["rho"]), np.asarray(f["z_cells"]), np.asarray(f["alpha_centres"])
    with np.load(T3_PRODUCT) as f:
        t3 = {l.name: (np.asarray(f[f"U_{l.name}"]), np.asarray(f[f"w_{l.name}"])) for l in ctx.legs
              if f"U_{l.name}" in f.files}
    zg = np.asarray(ctx.z_global)
    worst = 0.0
    for th in _theta_set():
        for leg in ctx.legs:
            sel = np.array([int(np.argmin(np.abs(zg - z))) for z in leg.z])
            out = FW.predict_leg(ctx.model, jnp.asarray(th), tau0[sel], alpha[sel], leg=leg, k_com=ctx.k_com_hmpc,
                                 pf_stats=ctx.pf_stats, dla_core=cores[leg.name], mf=ctx.mf, nuis=NUIS,
                                 t1=(ctx.rho_zb_per_leg[leg.name], ctx.alpha_centres), t3=ctx.t3_per_leg.get(leg.name))
            rho_leg = np.moveaxis(rho_all[[int(np.argmin(np.abs(zc - z))) for z in leg.z]], 1, -1)
            legd = dict(z=np.asarray(leg.z), k=np.asarray(leg.k), z_idx=np.asarray(leg.z_idx), R_z=np.asarray(leg.R_z),
                        metals_on=bool(leg.metals_on), resolution_on=bool(leg.resolution_on),
                        dla_forward_frac=float(getattr(leg, "dla_forward_frac", 1.0)),
                        C_data=np.asarray(leg.C_data), P_data=np.asarray(leg.P_data))
            nuis = {k: (np.asarray(v) if not isinstance(v, (tuple, float)) else v) for k, v in NUIS.items()}
            C_rc = R.reconstruct_cov_leg(members, legd, np.asarray(th), np.asarray(tau0)[sel], np.asarray(alpha)[sel],
                                         core_grid[leg.name], rho_leg, centres, t3.get(leg.name), PARAM_LIMITS, nuis)
            keep = np.where(np.isfinite(np.asarray(leg.P_data)))[0]
            Cf = np.asarray(out.C_total)[np.ix_(keep, keep)]
            Cr = C_rc[np.ix_(keep, keep)]
            sd = np.sqrt(np.diag(Cr))
            worst = max(worst, float(np.max(np.abs(Cf - Cr) / np.outer(sd, sd))))
    assert worst <= 1e-10, worst
