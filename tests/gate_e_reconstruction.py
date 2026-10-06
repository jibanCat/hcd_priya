"""Gate E criterion E2 (GATE_E_SPEC v1 section 4): an INDEPENDENT reconstruction of the production forward's model P1D on
a leg's data bins. Import firewall (asserted by tests/test_gateE_e2_reconstruction.py): numpy, jax, equinox, h5py,
json, pickle and hcd_analysis.emulator.model (the network architecture only). Nothing from kcoord, predict, mf_modes,
mf_family, multifidelity, data_likelihood, closure_legb, forward, likelihood, ensemble or train.

Everything is re-derived here: checkpoint deserialisation, the normalisation inversion, the ensemble mean, the MF table
evaluation, the class combination with the DLA core, the physical k of each mode from first principles, the linear
interpolation onto the data k, and the metal and resolution factors from their formulas."""
import json
import pickle

import equinox as eqx
import jax
import numpy as np

from hcd_analysis.emulator.model import Emulator

L_BOX = 120.0                    # Mpc/h
Z_LO, Z_HI = 2.0, 5.4            # the encoder's z_unit map
C_KMS = 299792.458
LAM = dict(lya=1215.67, siiii=1206.50, siiia=1190.42, siiib=1193.28)
R_DOUBLET = 0.5


def load_member(prefix):
    meta = json.load(open(prefix + ".meta.json"))
    skeleton = Emulator(**meta["arch_cfg"], key=jax.random.PRNGKey(meta["seed"]))
    model = eqx.tree_deserialise_leaves(prefix + ".eqx", skeleton)
    norm = pickle.load(open(prefix + ".norm.pkl", "rb"))
    return model, meta, {k: np.asarray(norm["P_filt"][k]) for k in ("mu_marg", "sig_marg", "sig_cosmo")}


def ensemble_P_filt(members, theta_unit, z, tau0):
    x = np.concatenate([np.asarray(theta_unit, float), [(z - Z_LO) / (Z_HI - Z_LO)]])
    out = []
    for model, _, st in members:
        pr = model(jax.numpy.asarray(x), jax.numpy.asarray(tau0))
        base, resid = np.asarray(pr["P_filt_base"]), np.asarray(pr["P_filt_resid"])
        out.append(np.exp(base * st["sig_marg"] + st["mu_marg"] + st["sig_cosmo"] * resid))
    return np.mean(out, axis=0)                                                    # (4, K)


def mf_g(tables, z, tau0):
    """The MF correction (4, K): log_rho + gbar_z(z) + gbar_tau(rung) + a_k u_z(z) u_tau(rung), all marginals
    clamped-linear; rung = the fractional rung of tau0 on the z-blended per-z tau0 ladder."""
    zt, tt = np.asarray(tables["z_tab"], float), np.asarray(tables["tau_tab"], float)
    zq = ((z - Z_LO) / (Z_HI - Z_LO)) * (Z_HI - Z_LO) + Z_LO
    iz = int(np.clip(np.searchsorted(zt, zq, side="right") - 1, 0, zt.size - 2))
    wz = float(np.clip((zq - zt[iz]) / (zt[iz + 1] - zt[iz]), 0.0, 1.0))
    ladder = np.asarray(tables["tau_by_z"])[iz] * (1 - wz) + np.asarray(tables["tau_by_z"])[iz + 1] * wz
    rung = float(np.interp(tau0, ladder, tt))
    gbz = np.asarray(tables["gbar_z_tab"]); gtt = np.asarray(tables["gtau_tab"])
    C, K = gbz.shape[1], gbz.shape[2]
    gz = np.array([[np.interp(zq, zt, gbz[:, c, k]) for k in range(K)] for c in range(C)])
    gt = np.array([[np.interp(rung, tt, gtt[:, c, k]) for k in range(K)] for c in range(C)])
    uz = np.interp(zq, zt, np.asarray(tables["u_z"]))
    ut = np.interp(rung, tt, np.asarray(tables["u_tau"]))
    return np.asarray(tables["log_rho"])[None, :] + gz + gt + np.asarray(tables["a_k"]) * uz * ut


def k_modes(z, theta_unit, param_limits, K):
    lim = np.asarray(param_limits, float)
    hub = lim[5, 0] + theta_unit[5] * (lim[5, 1] - lim[5, 0])
    omh2 = lim[6, 0] + theta_unit[6] * (lim[6, 1] - lim[6, 0])
    om = omh2 / hub ** 2
    return 2 * np.pi * np.arange(1, K + 1) / L_BOX * (1 + z) / (100.0 * np.sqrt(om * (1 + z) ** 3 + 1 - om))


def metal(k, a3, a2, k3, k2):
    dv3 = C_KMS * np.log(LAM["lya"] / LAM["siiii"])
    dva = C_KMS * np.log(LAM["lya"] / LAM["siiia"])
    dvb = C_KMS * np.log(LAM["lya"] / LAM["siiib"])
    D3 = 2.0 - 2.0 / (1.0 + np.exp(-k / k3))
    D2 = 2.0 - 2.0 / (1.0 + np.exp(-k / k2))
    f = a3 ** 2 + 2 * a3 * np.cos(k * dv3) * D3 + a2 ** 2 * (1 + R_DOUBLET ** 2) \
        + 2 * a2 * (np.cos(k * dvb) + R_DOUBLET * np.cos(k * dva)) * D2
    cb = C_KMS * np.log(LAM["siiib"] / LAM["siiii"])
    ca = C_KMS * np.log(LAM["siiia"] / LAM["siiii"])
    return 1.0 + f + 2 * a3 * a2 * (np.cos(k * cb) + R_DOUBLET * np.cos(k * ca))


def metal_model_cplus(k, z, tau0, nodes_z, f3, f2, kk3, kk2):
    """a(z) = f(z) / (1 - exp(-tau0)), log10 f and log10 k-scale linear in log10(1+z) between the nodes (clamped)."""
    lz, xp = np.log10(1 + z), np.log10(1 + np.asarray(nodes_z, float))
    li = lambda v: 10 ** np.interp(lz, xp, np.log10(np.asarray(v, float)))
    omF = 1.0 - np.exp(-tau0)
    return metal(k, li(f3) / omF, li(f2) / omF, li(kk3), li(kk2))


def core_at(kd, grid, core):
    """The DLA core (a fixed function of physical k on its grid) at k: linear in ln k over the finite grid points."""
    fin = np.isfinite(core)
    return np.interp(np.log(kd), np.log(np.asarray(grid)[fin]), np.asarray(core)[fin])


def reconstruct_leg(members, tables, leg, theta_unit, tau0_leg, alpha_leg, core, param_limits, nuis):
    """Model P1D on every bin of ``leg`` (dict of arrays: z, k, z_idx, R_z, metals_on, resolution_on, dla_forward_frac);
    ``core`` = (k_grid, core on the grid), the physical-k DLA core product's arrays."""
    out = np.zeros(len(leg["k"]))
    alpha_leg = np.asarray(alpha_leg, float) * np.array([1.0, 1.0, float(leg["dla_forward_frac"])])[None, :]
    for iz, z in enumerate(leg["z"]):
        rows = np.where(np.asarray(leg["z_idx"]) == iz)[0]
        if rows.size == 0:
            continue
        kd = np.asarray(leg["k"])[rows]
        P = ensemble_P_filt(members, theta_unit, float(z), float(tau0_leg[iz]))
        if tables is not None:
            P = P * np.exp(mf_g(tables, float(z), float(tau0_leg[iz])))
        kn = k_modes(float(z), theta_unit, param_limits, P.shape[-1])
        Pc = np.stack([np.interp(kd, kn, P[c]) for c in range(4)])
        cr = core_at(kd, core[0], core[1])
        a = alpha_leg[iz]
        Pz = Pc[0] + a[0] * (Pc[1] - Pc[0]) + a[1] * (Pc[2] - Pc[0]) + a[2] * (Pc[3] + cr - Pc[0])
        if leg["metals_on"]:
            Pz = Pz * metal_model_cplus(kd, float(z), float(tau0_leg[iz]), nuis["metal_node_z"], nuis["f_SiIII_nodes"],
                                        nuis["f_SiII_nodes"], nuis["k_SiIII_nodes"], nuis["k_SiII_nodes"])
        if leg["resolution_on"]:
            Pz = Pz * np.exp(2.0 * nuis["b_res"] * kd ** 2 * float(leg["R_z"][iz]) ** 2)
        out[rows] = Pz
    return out
