"""Gate B: historical parity against lya_emulator_full (PI ruling 2026-10-05 section 4). One test per parity-table
row (hcd_priya_notes docs/superpowers/emulator-debug-2026-10/gateB/PARITY_TABLE.md). Where upstream code can run, the
test runs it (tests/upstream_helpers.py); every number compared is computed, not transcribed."""
import json
import os

import h5py
import numpy as np
import pytest

from hcd_analysis.emulator import data as D
from hcd_analysis.emulator import kcoord as KC
from hcd_analysis.emulator import schema as S
from tests.gate_helpers import real_cache_path, require_real_cache
from tests.upstream_helpers import UPSTREAM, UPSTREAM_PRODUCT, run_upstream

LF = real_cache_path("lf")
HR = real_cache_path("hr")
UP_LF = f"{UPSTREAM_PRODUCT}/mf_emulator_flux_vectors_tau1000000.hdf5"
UP_HR = f"{UPSTREAM_PRODUCT}/hires/mf_emulator_flux_vectors_tau1000000.hdf5"


# ------------------------------------------------------------------------------------- row 1: k conversion
def test_row01_velocity_conversion_matches_upstream_rebin():
    """Upstream rebin_power_to_kms with P(k_mpc) = k_mpc returns velfac(z) x k_skm; recover velfac and compare with
    kcoord's 100 E(z)/(1+z) on a (z, hub, omegamh2) grid."""
    zs = [2.2, 2.6, 3.0, 3.8, 4.6]
    cosmos = [(0.65, 0.140), (0.70, 0.143), (0.75, 0.146)]
    code = f"""
import json, numpy as np
from lyaemu.flux_power import rebin_power_to_kms
kfmpc = 2*np.pi*np.arange(1, 173)/120.0
out = []
for hub, omh2 in {cosmos!r}:
    zb = np.array({zs!r})
    kfkms = np.array([1e-3, 5e-3, 1e-2])
    P = np.tile(kfmpc, (len(zb), 1))
    _, r = rebin_power_to_kms(kfkms, kfmpc, P, zb, omh2/hub**2)
    out.append((r / kfkms[None, :]).tolist())
print(json.dumps(out))
"""
    up = np.array(run_upstream(code))                       # (cosmo, z, 3) = velfac(z) km/s per Mpc/h
    for c, (hub, omh2) in enumerate(cosmos):
        for i, z in enumerate(zs):
            ours = 100.0 * float(KC.E_of_z(z, hub, omh2)) / (1.0 + z)
            assert np.allclose(up[c, i], ours, rtol=1e-12), (hub, omh2, z)


# ------------------------------------------------------------------------------------- row 2: modes, DC, 2 pi
def test_row02_comoving_modes_equal_upstream_kfmpc():
    require_real_cache(LF)
    require_real_cache(HR)
    with h5py.File(UP_LF, "r") as f:
        up_lf = f["kfmpc"][...]
    with h5py.File(UP_HR, "r") as f:
        up_hr = f["kfmpc"][...]
    lf = D.load_cache(LF)
    hr = D.load_cache(HR)
    assert up_lf.shape == (172,) and np.allclose(lf["k_com_hmpc"], up_lf, rtol=1e-12, atol=0)
    assert np.allclose(hr["k_com_hmpc"][:172], up_hr, rtol=1e-12, atol=0)   # LF and HR share the comoving modes


# ------------------------------------------------------------------------------------- shared row matching
def _match_rows(ours, up_params, up_zout, max_rows=None):
    """Map (upstream row, z index) -> our cache row with the same simulation parameters, grid z and mean-flux rung."""
    pairs = []
    for ur in range(up_params.shape[0]):
        t0, p9 = up_params[ur, 0], up_params[ur, 1:]
        same_sim = np.all(np.isclose(ours["params"], p9[None, :], rtol=1e-6, atol=0), axis=1)
        same_rung = np.isclose(ours["alpha_slope"], t0, rtol=0, atol=2e-6)
        for zi, z in enumerate(up_zout):
            m = np.where(same_sim & same_rung & np.isclose(ours["z_grid"], z, atol=1e-6))[0]
            if m.size == 1:
                pairs.append((ur, zi, int(m[0])))
        if max_rows and len(pairs) >= max_rows:
            break
    return pairs


@pytest.fixture(scope="module")
def lf_matched():
    require_real_cache(LF)
    ours = D.load_cache(LF)
    with h5py.File(LF, "r") as f:
        ours["alpha_slope"] = f["alpha_slope"][...]          # rung value (load_cache exposes only alpha_idx)
    with h5py.File(UP_LF, "r") as f:
        up = {k: f[k][...] for k in ("params", "zout", "kfkms", "flux_vectors")}
    pairs = _match_rows(ours, up["params"], up["zout"])
    return ours, up, pairs


# Discrepancies established in the gate B audit (2026-10-06) from the raw spectra headers; every OTHER entry must match.
#  - SIM_UPSTREAM_DEFECT at z = 2.2: upstream's entry carries the z = 2.3936 snapshot (SPECTRA_021, a second
#    snapshot within 0.01 of the z = 2.4 grid point); ours carries the z = 2.2 snapshot (SPECTRA_022).
#  - SIM_OUR_GAP at z = 3.0: no cache row (no per-snapshot absorber catalog exists for that snapshot); upstream has it.
#  - SIM_OUR_GAP at z = 2.8: our stored velocity width differs from the spectra header by 9.2e-5 (k and P scale with it).
SIM_UPSTREAM_DEFECT = "ns0.959Ap2.34e-09herei3.81heref2.99alphaq1.77hub0.725omegamh20.144hireionz6.83bhfeedback0.0467"
SIM_OUR_GAP = "ns0.907Ap1.5e-09herei3.75heref2.77alphaq2.04hub0.662omegamh20.144hireionz7.47bhfeedback0.0347"
RAW = "/nfs/turbo/umor-yueyingn/mfho/emu_full"


def _is(ours, r, sim, z, up_zout):
    return str(ours["sim_name"][r]) == sim and abs(float(up_zout) - z) < 1e-6


def test_row03_parameters_and_rungs_match_upstream_training_set(lf_matched):
    """Every upstream (t0, simulation, z) entry exists in our cache with identical parameters and rung (our Ap from
    SimulationICs with the 5 pi pivot ratio, upstream's from its own ICs path), except the 10 rungs of SIM_OUR_GAP at
    z = 3.0, which our cache lacks."""
    ours, up, pairs = lf_matched
    matched = {(ur, zi) for ur, zi, _ in pairs}
    missing = [(ur, zi) for ur in range(up["params"].shape[0]) for zi in range(len(up["zout"])) if (ur, zi) not in matched]
    assert len(missing) == 10 and {float(up["zout"][zi]) for _, zi in missing} == {3.0}
    ns_gap = float(SIM_OUR_GAP[2:7])
    assert all(abs(up["params"][ur, 1] - ns_gap) < 5e-4 for ur, _ in missing)


def test_row04_per_row_velocity_grids_equal_upstream(lf_matched):
    ours, up, pairs = lf_matched
    worst = 0.0
    for ur, zi, r in pairs:
        dk = float(np.max(np.abs(ours["kfkms"][r] / up["kfkms"][ur, zi] - 1.0)))
        if _is(ours, r, SIM_UPSTREAM_DEFECT, 2.2, up["zout"][zi]):
            continue                                     # established upstream defect, checked below
        worst = max(worst, dk)
    assert worst < 2e-4, worst


def test_row04b_upstream_defect_entry_is_the_z2p39_snapshot():
    """Raw headers: upstream's z = 2.2 grid for SIM_UPSTREAM_DEFECT equals SPECTRA_021 (z = 2.3936); our row equals
    SPECTRA_022 (z = 2.2)."""
    require_real_cache(LF)
    p021 = f"{RAW}/{SIM_UPSTREAM_DEFECT}/output/SPECTRA_021/lya_forest_spectra_grid_480.hdf5"
    p022 = f"{RAW}/{SIM_UPSTREAM_DEFECT}/output/SPECTRA_022/lya_forest_spectra_grid_480.hdf5"
    for p in (p021, p022):
        if not os.path.exists(p):
            if os.environ.get("HCD_GATE_RUN") == "1":
                pytest.fail(f"raw spectra absent: {p}")
            pytest.skip(f"raw spectra absent: {p}")

    def k1(path):
        with h5py.File(path, "r") as f:
            h = f["Header"].attrs
            z, box, om = float(h["redshift"]), float(h["box"]), float(h["omegam"])
        return z, 2 * np.pi / ((box / 1000.0) * 100.0 * np.sqrt(om * (1 + z) ** 3 + 1 - om) / (1 + z))

    z21, k21 = k1(p021)
    z22, k22 = k1(p022)
    with h5py.File(UP_LF, "r") as f:
        params, zout, kfkms = f["params"][...], f["zout"][...], f["kfkms"][...]
    ur = int(np.where(np.isclose(params[:, 1], float(SIM_UPSTREAM_DEFECT[2:7]), atol=5e-4))[0][0])
    zi = int(np.where(np.isclose(zout, 2.2))[0][0])
    assert abs(z21 - 2.3936) < 1e-3 and abs(z22 - 2.2) < 1e-4
    assert abs(kfkms[ur, zi, 0] / k21 - 1) < 1e-6                     # upstream z = 2.2 entry = the z = 2.39 snapshot
    d = D.load_cache(LF)
    r = int(np.where((d["sim_name"] == SIM_UPSTREAM_DEFECT) & np.isclose(d["z_grid"], 2.2))[0][0])
    assert abs(d["kfkms"][r, 0] / k22 - 1) < 2e-4                     # ours = the z = 2.2 snapshot


def test_row05_training_p1d_equals_upstream_flux_vectors(lf_matched):
    """The cache's filtered total P1D (Tier P) equals upstream's flux_vectors at the same (simulation, z, rung) to 1e-5,
    except the established entries: the upstream defect (different snapshot) and SIM_OUR_GAP z = 2.8 (2e-4 level)."""
    ours, up, pairs = lf_matched
    K = 172
    rel, small = [], []
    for ur, zi, r in pairs:
        p_up = up["flux_vectors"][ur, zi * K:(zi + 1) * K]
        p_ours = ours["P_tier_p"][r]
        ok = np.isfinite(p_ours) & np.isfinite(p_up)
        d = float(np.max(np.abs(p_ours[ok] / p_up[ok] - 1.0)))
        if _is(ours, r, SIM_UPSTREAM_DEFECT, 2.2, up["zout"][zi]):
            continue
        (small if _is(ours, r, SIM_OUR_GAP, 2.8, up["zout"][zi]) else rel).append(d)
    assert max(rel) < 1e-5, f"max relative P1D difference {max(rel):.3e} (median {np.median(rel):.3e})"
    assert len(small) == 10 and max(small) < 2e-4, small


def test_row05b_hr_training_p1d_equals_upstream_hires_flux_vectors():
    """High-resolution cache vs upstream's hires flux_vectors on the shared 172 modes, every matched entry."""
    require_real_cache(HR)
    ours = D.load_cache(HR)
    with h5py.File(HR, "r") as f:
        ours["alpha_slope"] = f["alpha_slope"][...]
    with h5py.File(UP_HR, "r") as f:
        up = {k: f[k][...] for k in ("params", "zout", "kfkms", "flux_vectors")}
    pairs = _match_rows(ours, up["params"], up["zout"])
    assert len(pairs) == up["params"].shape[0] * len(up["zout"]), len(pairs)
    K = 172
    worst_p = max(float(np.nanmax(np.abs(ours["P_tier_p"][r][:K] / up["flux_vectors"][ur, zi * K:(zi + 1) * K] - 1.0)))
                  for ur, zi, r in pairs)
    worst_k = max(float(np.max(np.abs(ours["kfkms"][r][:K] / up["kfkms"][ur, zi] - 1.0))) for ur, zi, r in pairs)
    assert worst_p < 1e-5 and worst_k < 2e-4, (worst_p, worst_k)


# ------------------------------------------------------------------------------------- row 6/7: mean flux
def test_row06_mean_flux_targets_use_grid_redshift_and_upstream_tau():
    require_real_cache(LF)
    with h5py.File(LF, "r") as f:
        z = f["z_grid"][...]
        a = f["alpha_slope"][...]
        tF = f["target_F"][...]
    zs = sorted(set(np.round(z, 4).tolist()))
    up = run_upstream(f"import json; from lyaemu.mean_flux import obs_mean_tau; "
                      f"print(json.dumps([float(obs_mean_tau(z)) for z in {zs!r}]))")
    tau_up = dict(zip(zs, up))
    expect = np.exp(-a * np.array([tau_up[round(float(zz), 4)] for zz in z]))
    assert np.max(np.abs(tF / expect - 1.0)) < 1e-10


def test_row07_likelihood_mean_flux_slope_factor_equals_upstream():
    from hcd_analysis.emulator.meanflux_prior import tau0_alpha_priya
    zout = [4.6, 4.4, 4.2, 4.0, 3.8, 3.6, 3.4, 3.2, 3.0, 2.8, 2.6, 2.4, 2.2]
    slopes = [-0.4, -0.1, 0.0, 0.25]
    up = np.array(run_upstream(f"import json, numpy as np; from lyaemu.mean_flux import mean_flux_slope_to_factor; "
                               f"print(json.dumps([mean_flux_slope_to_factor(np.array({zout!r}), s).tolist() for s in {slopes!r}]))"))
    for i, s in enumerate(slopes):
        ours = np.array([float(tau0_alpha_priya(z, 1.0, s)) for z in zout])
        assert np.allclose(ours, up[i], rtol=1e-12), s


# ------------------------------------------------------------------------------------- row 8: parameters
def test_row08_parameter_order_limits_and_sampling_tightening_match_upstream():
    up = json.load(open(f"{UPSTREAM_PRODUCT}/emulator_params.json"))
    order = sorted(up["param_names"], key=up["param_names"].get)
    with h5py.File(real_cache_path("lf"), "r") as f:
        ours_names = [s.decode() if isinstance(s, bytes) else s for s in f["param_names"][...]]
    assert ours_names == order
    assert np.allclose(np.asarray(D.PARAM_LIMITS), np.asarray(up["param_limits"]), rtol=0, atol=0)
    src = open(f"{UPSTREAM}/lyaemu/likelihood.py").read()
    assert "self.param_limits[herei][1] = 4.1" in src and "self.param_limits[heref][0] = 2.6" in src
    sl = np.asarray(D.SAMPLING_LIMITS)
    assert sl[2, 1] == 4.1 and sl[3, 0] == 2.6


# ------------------------------------------------------------------------------------- row 9: A_p pivot
def test_row09_ap_pivot_ratio_matches_upstream_definition():
    """Upstream coarse_grid.py:360: A_s = (0.05/(2 pi/8))^(ns-1) Ap, i.e. Ap = A_s (k_p/0.05)^(ns-1) with
    k_p = 2 pi/8 Mpc^-1; ours uses _AP_PIVOT_RATIO = 5 pi. Row 3 checks the resulting Ap values on every simulation."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("bt0", os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                                      "scripts", "build_emulator_cache_tau0.py"))
    src = open(spec.origin).read()
    assert "_AP_PIVOT_RATIO = 5.0 * np.pi" in src
    assert np.isclose((2 * np.pi / 8.0) / 0.05, 5.0 * np.pi, rtol=1e-15)
    up_src = open(f"{UPSTREAM}/lyaemu/coarse_grid.py").read()
    assert "(0.05 / (2 * math.pi / 8.0)) ** (ns - 1.0)" in up_src


# ------------------------------------------------------------------------------------- row 10: target
def test_row10_class_decomposition_reconstructs_the_upstream_total():
    """Our emulator target is the per-class filtered P1D; its count-weighted sum is the Tier P total that row 5 shows
    equal to upstream's flux_vectors (structural identity on every real row)."""
    require_real_cache(LF)
    d = D.load_cache(LF)
    tot = np.einsum("rc,rck->rk", d["w_c_cache"], d["P_filt"])
    ok = np.isfinite(d["P_tier_p"])
    assert np.max(np.abs(tot[ok] / d["P_tier_p"][ok] - 1.0)) < 1e-10


# ------------------------------------------------------------------------------------- row 11: coverage
def test_row11_production_legs_inside_the_simulated_modes_for_every_theta_in_the_box():
    """Upstream raises outside the simulated k range; the old forward clamped. Every production data bin lies inside
    [k_1, k_172] for every (hub, omegamh2) corner of the box at its redshift, so no data bin needs extrapolation."""
    from hcd_analysis.emulator import data_likelihood as DL
    k1, kN = 2 * np.pi / S.L_BOX_HMPC, 2 * np.pi * 172 / S.L_BOX_HMPC
    lim = np.asarray(D.PARAM_LIMITS)
    corners = [(h, w) for h in lim[5] for w in lim[6]]
    for leg in (DL.load_eboss_leg(), DL.load_desi_leg(), DL.load_ks_leg()):
        k, zr = np.asarray(leg.k), np.asarray(leg.z_row)
        for z in np.unique(zr):
            kk = k[np.isclose(zr, z)]
            lo = max(float(KC.k_skm_from_kcom(k1, z, h, w)) for h, w in corners)
            hi = min(float(KC.k_skm_from_kcom(kN, z, h, w)) for h, w in corners)
            assert kk.min() >= lo and kk.max() <= hi, (leg.name, z, kk.min(), kk.max(), lo, hi)


# ------------------------------------------------------------------------------------- row 13: LF/HF alignment
def test_row13_upstream_aligns_fidelities_by_comoving_mode():
    """Upstream combines LF and HF on identical comoving grids (2 pi n / 120, both 172 modes); the rebuilt
    multi-fidelity layer (gate D) must align by mode index, never by a velocity label."""
    with h5py.File(UP_LF, "r") as f:
        a = f["kfmpc"][...]
    with h5py.File(UP_HR, "r") as f:
        b = f["kfmpc"][...]
    assert a.shape == b.shape == (172,) and np.allclose(a, b, rtol=1e-12, atol=0)   # equal to float precision (3e-16)


# ------------------------------------------------------------------------------------- row 14: res_corr
def test_row14_resolution_correction_is_off_in_our_production_and_upstream_applies_above_002():
    from hcd_analysis.emulator import closure_legb as CL
    assert CL.prod_norc_forward()["res_corr_on"] is False
    src = open(f"{UPSTREAM}/lyaemu/likelihood.py").read()
    assert "ind = kf > kfmin" in src


# ------------------------------------------------------------------------------------- row 15: metals
def test_row15_siiii_velocity_separation_matches_upstream_constant():
    """Ours c ln(1215.67/1206.50) = 2269.96 km/s; upstream hard-codes the rounded literature value 2271 km/s
    (0.046 percent apart: a phase difference k dv = 0.02 rad at k = 0.02 s/km, negligible)."""
    from hcd_analysis.emulator import data_likelihood as DL
    dv = DL.C_KMS * np.log(DL.LAMBDA_LYA / DL.LAMBDA_SiIII)
    src = open(f"{UPSTREAM}/lyaemu/likelihood.py").read()
    assert "2271" in src
    assert abs(dv / 2271.0 - 1.0) < 1e-3, dv


# ------------------------------------------------------------------------------------- row 16: data loading
def test_row16_eboss_dr14_data_equal_upstream_loader():
    from hcd_analysis.emulator import data_likelihood as DL
    leg = DL.load_eboss_leg(k_max=1.0)
    up = run_upstream("""
import json, numpy as np
from lyaemu.lyman_data import BOSSData
b = BOSSData()
zs = b.get_redshifts(); kf = b.get_kf()
out = {"z": zs.tolist(), "k": kf.tolist(), "pf": [b.get_pf(zbin=z).tolist() for z in zs],
       "diag": [np.diag(b.get_covar(zbin=z)).tolist() for z in zs]}
print(json.dumps(out))
""")
    zs, kf = np.array(up["z"]), np.array(up["k"])
    for i, z in enumerate(zs):
        m = np.isclose(leg.z_row, z)
        if not m.any():
            continue
        ks = np.asarray(leg.k)[m]
        sel = [int(np.argmin(np.abs(kf - kk))) for kk in ks]
        assert np.allclose(ks, kf[sel], rtol=1e-6)
        assert np.allclose(np.asarray(leg.P_data)[m], np.asarray(up["pf"][i])[sel], rtol=1e-6)


def test_row16_ks_conservative_data_equal_upstream_loader():
    from hcd_analysis.emulator import data_likelihood as DL
    leg = DL.load_ks_leg(z_lo=0.0, k_max=1.0)
    up = run_upstream("""
import json, numpy as np
from lyaemu.lyman_data import KSData
d = KSData(conservative=True)
zs = d.get_redshifts(); kf = d.get_kf()
print(json.dumps({"z": zs.tolist(), "k": kf.tolist(), "pf": [d.get_pf(zbin=z).tolist() for z in zs]}))
""")
    zs, kf = np.array(up["z"]), np.array(up["k"])
    n_cmp = 0
    for i, z in enumerate(zs):
        m = np.isclose(leg.z_row, z)
        if not m.any():
            continue
        ks = np.asarray(leg.k)[m]
        sel = [int(np.argmin(np.abs(kf - kk))) for kk in ks]
        assert np.allclose(ks, kf[sel], rtol=1e-6)
        assert np.allclose(np.asarray(leg.P_data)[m], np.asarray(up["pf"][i])[sel], rtol=1e-6)
        n_cmp += int(m.sum())
    assert n_cmp > 0
