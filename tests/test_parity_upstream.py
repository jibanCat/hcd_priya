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
#  - SIM_UPSTREAM_DEFECT at z = 2.2 (F1): upstream's entry carries the z = 2.3936 snapshot (SPECTRA_021, a second
#    snapshot within 0.01 of the z = 2.4 grid point); ours carries the z = 2.2 snapshot (SPECTRA_022).
#  - SIM_OUR_GAP (ns0.907), F8: in the HISTORICAL cache the z = 2.8 rows carried the z = 3.0 file's catalogue and pixel
#    width (9.2e-5 in k and P) and the z = 3.0 rows were absent (F2, reinterpreted by F8). The production cache
#    (observables_tau0_lf.h5) is the S4-repaired one; the historical facts are pinned by the *_hist tests.
SIM_UPSTREAM_DEFECT = "ns0.959Ap2.34e-09herei3.81heref2.99alphaq1.77hub0.725omegamh20.144hireionz6.83bhfeedback0.0467"
SIM_OUR_GAP = "ns0.907Ap1.5e-09herei3.75heref2.77alphaq2.04hub0.662omegamh20.144hireionz7.47bhfeedback0.0347"
RAW = "/nfs/turbo/umor-yueyingn/mfho/emu_full"
LF_HIST = real_cache_path("lf_hist")


def _is(ours, r, sim, z, up_zout):
    return str(ours["sim_name"][r]) == sim and abs(float(up_zout) - z) < 1e-6


def _missing(ours, up):
    matched = {(ur, zi) for ur, zi, _ in _match_rows(ours, up["params"], up["zout"])}
    return [(ur, zi) for ur in range(up["params"].shape[0]) for zi in range(len(up["zout"])) if (ur, zi) not in matched]


def test_row03_parameters_and_rungs_match_upstream_training_set(lf_matched):
    """Every upstream (t0, simulation, z) entry exists in our (repaired) cache with identical parameters and rung (our
    Ap from SimulationICs with the 5 pi pivot ratio, upstream's from its own ICs path): all 7800."""
    ours, up, pairs = lf_matched
    assert len(pairs) == 7800 and _missing(ours, up) == []


def test_row03_hist_historical_cache_lacked_the_ns0907_z3_rows():
    """Provenance (F2, reinterpreted by F8): the historical cache lacked exactly the 10 upstream rungs of ns0.907 at
    z = 3.0."""
    require_real_cache(LF_HIST)
    ours = D.load_cache(LF_HIST)
    with h5py.File(LF_HIST, "r") as f:
        ours["alpha_slope"] = f["alpha_slope"][...]
    with h5py.File(UP_LF, "r") as f:
        up = {k: f[k][...] for k in ("params", "zout")}
    missing = _missing(ours, up)
    assert len(missing) == 10 and {float(up["zout"][zi]) for _, zi in missing} == {3.0}
    assert all(abs(up["params"][ur, 1] - float(SIM_OUR_GAP[2:7])) < 5e-4 for ur, _ in missing)


def test_row04_per_row_velocity_grids_equal_upstream(lf_matched):
    ours, up, pairs = lf_matched
    worst = 0.0
    for ur, zi, r in pairs:
        dk = float(np.max(np.abs(ours["kfkms"][r] / up["kfkms"][ur, zi] - 1.0)))
        if _is(ours, r, SIM_UPSTREAM_DEFECT, 2.2, up["zout"][zi]):
            continue                                     # established upstream defect F1, checked below
        worst = max(worst, dk)
    assert worst < 1e-12, worst


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


def test_row04c_upstream_snapshot_selection_rule_fills_the_z2p2_slot_with_the_z2p39_snapshot():
    """Mechanism of the upstream defect: MySpectra.get_snapshot_list (flux_power.py) walks SPECTRA_000..029 in order,
    keeps every snapshot whose redshift is within 0.01 of ANY zout (_check_redshift), assigns them to the zout slots by
    POSITION and stops once it has len(zout) of them. SIM_UPSTREAM_DEFECT has two snapshots within 0.01 of z = 2.4, so
    the 13th accepted snapshot (the z = 2.2 slot) is SPECTRA_021, z = 2.3936. Upstream's own predicate is applied to the
    raw header redshifts."""
    import glob
    require_real_cache(LF)
    paths = sorted(glob.glob(f"{RAW}/{SIM_UPSTREAM_DEFECT}/output/SPECTRA_0[0-2][0-9]/lya_forest_spectra_grid_480.hdf5"))
    if not paths:
        if os.environ.get("HCD_GATE_RUN") == "1":
            pytest.fail("raw spectra absent")
        pytest.skip("raw spectra absent")
    snaps = []
    for p in paths:
        with h5py.File(p, "r") as f:
            snaps.append((int(p.split("SPECTRA_")[1][:3]), float(f["Header"].attrs["redshift"])))
    up = run_upstream(f"""
import json
from lyaemu.flux_power import MySpectra
ms = MySpectra(max_z=4.6, min_z=2.2)
kept = []
for snap, red in {snaps!r}:
    if len(kept) == ms.zout.size:
        break
    if ms._check_redshift(red):
        kept.append(snap)
print(json.dumps({{"kept": kept, "zout": ms.zout.tolist()}}))
""")
    kept, zout = up["kept"], up["zout"]
    assert len(kept) == len(zout) == 13 and abs(zout[-1] - 2.2) < 1e-9
    zmap = dict(snaps)
    assert kept[-1] == 21 and abs(zmap[21] - 2.3936) < 1e-3          # z = 2.2 slot holds the z = 2.39 snapshot
    assert sum(abs(zmap[s] - 2.4) < 0.01 for s in kept) == 2          # two accepted snapshots near z = 2.4
    assert 22 in zmap and abs(zmap[22] - 2.2) < 1e-4 and 22 not in kept


def test_row05_training_p1d_equals_upstream_flux_vectors(lf_matched):
    """The (repaired) cache's filtered total P1D (Tier P) equals upstream's flux_vectors at the same (simulation, z,
    rung) to 1e-5 on every entry except the upstream defect F1 (different snapshot); this includes the rebuilt ns0.907
    z = 2.8 and the restored z = 3.0 entries (independent check of the S4 repair)."""
    ours, up, pairs = lf_matched
    K = 172
    rel, f8 = [], []
    for ur, zi, r in pairs:
        p_up = up["flux_vectors"][ur, zi * K:(zi + 1) * K]
        p_ours = ours["P_tier_p"][r]
        ok = np.isfinite(p_ours) & np.isfinite(p_up)
        d = float(np.max(np.abs(p_ours[ok] / p_up[ok] - 1.0)))
        if _is(ours, r, SIM_UPSTREAM_DEFECT, 2.2, up["zout"][zi]):
            continue
        rel.append(d)
        if str(ours["sim_name"][r]) == SIM_OUR_GAP and abs(float(up["zout"][zi]) - 2.9) < 0.15:
            f8.append(d)
    assert len(rel) == 7790 and max(rel) < 1e-5, f"max relative P1D difference {max(rel):.3e} (median {np.median(rel):.3e})"
    assert len(f8) == 20 and max(f8) < 1e-5, f8


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


def test_row08b_unit_cube_map_equals_upstream_and_round_trips_through_kcoord():
    """Input normalisation: upstream map_to_unit_cube_list(params, param_limits) (latin_hypercube.py) on the 60 PRIYA-LF
    simulations equals our data.normalize_params; kcoord's inverse recovers the physical hub and omegamh2."""
    with h5py.File(UP_LF, "r") as f:
        p9 = np.unique(f["params"][:, 1:], axis=0)                     # the 60 simulations (each has 10 rungs)
    up = np.array(run_upstream(f"""
import json, numpy as np
from lyaemu.latin_hypercube import map_to_unit_cube_list
lim = np.array(json.load(open("{UPSTREAM_PRODUCT}/emulator_params.json"))["param_limits"])
print(json.dumps(map_to_unit_cube_list(np.array({p9.tolist()!r}), lim).tolist()))
"""))
    ours = D.normalize_params(p9)
    assert up.shape == ours.shape == (60, 9) and np.allclose(ours, up, rtol=0, atol=1e-12)
    for i in range(60):
        hub, omh2 = KC.hub_omegamh2_from_theta9(ours[i])
        assert abs(float(hub) / p9[i, 5] - 1) < 1e-12 and abs(float(omh2) / p9[i, 6] - 1) < 1e-12


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
       "cov": [b.get_covar(zbin=z).tolist() for z in zs]}
print(json.dumps(out))
""")
    zs, kf = np.array(up["z"]), np.array(up["k"])
    C = np.asarray(leg.C_data)
    n_cmp = 0
    for i, z in enumerate(zs):
        m = np.isclose(leg.z_row, z)
        assert m.any(), f"upstream eBOSS z = {z} has no rows in our leg"
        ks = np.asarray(leg.k)[m]
        sel = [int(np.argmin(np.abs(kf - kk))) for kk in ks]
        assert np.allclose(ks, kf[sel], rtol=1e-6)
        assert np.allclose(np.asarray(leg.P_data)[m], np.asarray(up["pf"][i])[sel], rtol=1e-6)
        idx = np.where(m)[0]
        assert np.allclose(C[np.ix_(idx, idx)], np.asarray(up["cov"][i])[np.ix_(sel, sel)], rtol=1e-10, atol=0)
        assert not np.any(C[np.ix_(idx, np.where(~m)[0])]), "eBOSS covariance has cross-z terms (upstream has none)"
        n_cmp += int(m.sum())
    assert n_cmp == 13 * 35, n_cmp


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
    assert n_cmp == 182, n_cmp                         # the full conservative table (13 z x 14 k)


def test_row16b_ks_differs_from_upstream_production_path_by_documented_choice():
    """Upstream PRODUCTION (sdss_name "kodiaq_squad_only") loads KSData(conservative=False): the detailed Karacayli+21
    table with upstream's own P_metal subtraction and covariance assembly. Ours uses the published CONSERVATIVE product
    (decision record 2026-06-14 knobs-and-reasons: it already subtracts metals). Pin both facts so the parity table
    cannot go stale silently: upstream production really is the detailed path, and our data vector really differs
    from it (order 1 sigma scatter)."""
    from hcd_analysis.emulator import data_likelihood as DL
    src = open(f"{UPSTREAM}/lyaemu/likelihood.py").read()
    blk = src[src.index('elif sdss_name == "kodiaq_squad_only"'):]
    blk = blk[:blk.index("self._kf_old")]
    assert "lyman_data.KSData(" in blk and "conservative=False" in blk
    leg = DL.load_ks_leg()
    up = run_upstream("""
import json
from lyaemu.lyman_data import KSData
d = KSData(conservative=False)
zs = d.get_redshifts(); kf = d.get_kf()
print(json.dumps({"z": zs.tolist(), "k": kf.tolist(), "pf": [d.get_pf(zbin=z).tolist() for z in zs]}))
""")
    zs, kf = np.array(up["z"]), np.array(up["k"])
    sig = np.sqrt(np.diag(np.asarray(leg.C_data)))
    dP = []
    for i, z in enumerate(zs):
        m = np.isclose(leg.z_row, z)
        if not m.any():
            continue
        ks = np.asarray(leg.k)[m]
        sel = [int(np.argmin(np.abs(kf - kk))) for kk in ks]
        assert np.allclose(ks, kf[sel], rtol=1e-6)
        dP.append((np.asarray(leg.P_data)[m] - np.asarray(up["pf"][i])[sel]) / sig[m])
    dP = np.concatenate(dP)
    assert dP.size == np.asarray(leg.k).size
    assert 0.1 < np.sqrt(np.mean(dP ** 2)) < 2.0, np.sqrt(np.mean(dP ** 2))


# ===================================================================================== gate B review blocking tests
# (reviews/gate_B/gate_B_review.md section 10: BT-B1 .. BT-B5)
HR_RAW = "/scratch/yueyingn_root/yueyingn0/mfho/priya/emu_full_hires_2"
HCD_OUT = "/scratch/cavestru_root/cavestru0/mfho/hcd_outputs"
UP_HR6 = ("/nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48_20260414_newhires/hires/"
          "mf_emulator_flux_vectors_tau1000000.hdf5")
# F8 (found at gate B while root-causing the 9.2e-5 offset): our LF rows for SIM_OUR_GAP at z = 2.8 (Phase-1 snap 17)
# take tau from SPECTRA_018 (header z = 2.8) but the snap_017 absorber catalogue and its dv_kms were built on SPECTRA_017
# (header z = 3.0). The only group of either cache whose catalogue and tau come from different spectra files.
F8_GROUP = (SIM_OUR_GAP, 17)


def _require_paths(*paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        if os.environ.get("HCD_GATE_RUN") == "1":
            pytest.fail(f"inputs absent: {missing}")
        pytest.skip(f"inputs absent: {missing}")


def _raw_headers(raw_root, sim):
    """Header facts of every raw grid_480 savefile of one simulation; vmax exactly as fake_spectra (spectra.py:208-215)."""
    import glob
    out = []
    for d in sorted(glob.glob(f"{raw_root}/{sim}/output/SPECTRA_*")):
        p = f"{d}/lya_forest_spectra_grid_480.hdf5"
        if not os.path.exists(p):
            continue
        with h5py.File(p, "r") as f:
            a = f["Header"].attrs
            z, h = float(a["redshift"]), float(a["hubble"])
            at = 1.0 / (1.0 + z)
            hz = float(a["Hz"]) if "Hz" in a else 100.0 * h * np.sqrt(float(a["omegam"]) / at ** 3 + float(a["omegal"]))
            vmax = float(a["box"]) * (3.085678e21 * at / h) * hz / 3.085678e24
            out.append(dict(spec=int(d.rsplit("_", 1)[1]), z=z, vmax=vmax, nbins=int(a["nbins"])))
    return out


@pytest.mark.parametrize("fid", ["lf", "hr", "lf_hist"])
def test_row04d_every_cache_row_matches_its_raw_header(fid):
    """BT-B4 and the S4 hard invariant. For every (simulation, snapshot) group of our caches: the snapshot used is THE
    raw file at the grid z (the nearest header z is within 1e-9 of z_grid; no other file within 1e-3), every row's k
    equals 2 pi n / vmax_header to 1e-12, nbins_native equals the header nbins, z_meta is within 2e-5 of the header z,
    and the Phase-1 absorber catalogue directory used was built on that same spectra file (its meta nbins and dv_kms
    equal the header's). No exception in the production caches; in the historical LF cache exactly the F8 group, whose
    defect is pinned exactly (provenance)."""
    from collections import defaultdict
    path = {"lf": LF, "hr": HR, "lf_hist": LF_HIST}[fid]
    raw_root = HR_RAW if fid == "hr" else RAW
    hcd = f"{HCD_OUT}/hires" if fid == "hr" else HCD_OUT
    require_real_cache(path)
    _require_paths(raw_root, hcd)
    with h5py.File(path, "r") as f:
        sim = f["sim_name"].asstr()[...]
        snap = f["snap"][...]
        zg, zm = f["z_grid"][...], f["z_meta"][...]
        kf, nb = f["kfkms"][...], f["nbins_native"][...]
    groups = defaultdict(list)
    for r in range(sim.size):
        groups[(str(sim[r]), int(snap[r]))].append(r)
    hdr, seen_f8 = {}, False
    worst = dict(k=0.0, dz_grid=0.0, dz_meta=0.0)
    for (s, sn), rows in groups.items():
        r0 = rows[0]
        h = hdr.setdefault(s, _raw_headers(raw_root, s))
        assert h, f"no raw headers for {s}"
        zs = np.array([e["z"] for e in h])
        order = np.argsort(np.abs(zs - zg[r0]))
        e = h[int(order[0])]
        assert all(zg[r] == zg[r0] and zm[r] == zm[r0] and nb[r] == nb[r0] for r in rows), (s, sn)
        worst["dz_grid"] = max(worst["dz_grid"], abs(e["z"] - zg[r0]))
        assert abs(e["z"] - zg[r0]) < 1e-9, (s, sn, e["z"], zg[r0])
        assert order.size == 1 or abs(zs[order[1]] - zg[r0]) > 1e-3, (s, sn, "second raw file at the grid z")
        assert int(nb[r0]) == e["nbins"], (s, sn, int(nb[r0]), e["nbins"])
        worst["dz_meta"] = max(worst["dz_meta"], abs(zm[r0] - e["z"]))
        assert abs(zm[r0] - e["z"]) <= 2e-5, (s, sn, zm[r0], e["z"])
        n = np.arange(1, kf.shape[1] + 1)
        fin = np.isfinite(kf[r0])
        meta = json.load(open(f"{hcd}/{s}/snap_{sn:03d}/meta.json"))
        if fid == "lf_hist" and (s, sn) == F8_GROUP:
            seen_f8 = True
            e30 = h[int(np.argmin(np.abs(zs - 3.0)))]                              # SPECTRA_017, header z = 3.0
            assert e["spec"] == 18 and e30["spec"] == 17 and abs(e30["z"] - 3.0) < 1e-6
            assert int(meta["nbins"]) == e30["nbins"] == 1397                        # catalogue built on z = 3.0 spectra
            assert abs(float(meta["dv_kms"]) / (e30["vmax"] / e30["nbins"]) - 1) < 1e-12
            for r in rows:                                                          # k from the z = 3.0 file's pixel width
                assert np.allclose(kf[r][fin], 2 * np.pi * n[fin] / (nb[r] * float(meta["dv_kms"])), rtol=1e-12, atol=0)
                off = float(np.max(np.abs(kf[r][fin] / (2 * np.pi * n[fin] / e["vmax"]) - 1)))
                assert 9.0e-5 < off < 9.3e-5, off
            m18 = json.load(open(f"{hcd}/{s}/snap_018/meta.json"))                  # the correct catalogue exists,
            assert int(m18["nbins"]) == e["nbins"] == 1365                          # filed under snap_018 (meta z 2.67)
            assert abs(float(m18["dv_kms"]) / (e["vmax"] / e["nbins"]) - 1) < 1e-12
            continue
        assert int(meta["nbins"]) == e["nbins"], (s, sn, "catalogue built on another spectra file")
        assert abs(float(meta["dv_kms"]) / (e["vmax"] / e["nbins"]) - 1) < 1e-12, (s, sn)
        kexp = 2 * np.pi * n / e["vmax"]
        for r in rows:
            fr = np.isfinite(kf[r])
            rel = float(np.max(np.abs(kf[r][fr] / kexp[fr] - 1)))
            worst["k"] = max(worst["k"], rel)
            assert rel < 1e-12, (s, sn, r, rel)
    assert seen_f8 == (fid == "lf_hist")
    assert len(groups) == {"lf": 1073, "hr": 103, "lf_hist": 1072}[fid]
    print(f"{fid}: {len(groups)} groups; worst k rel {worst['k']:.2e}; worst |z_header - z_grid| {worst['dz_grid']:.2e}; "
          f"worst |z_meta - z_header| {worst['dz_meta']:.2e}")


def _implied_z(velfac, om):
    """Redshift at which 100 E(z) / (1+z) equals velfac (monotone for z > 1)."""
    from scipy.optimize import brentq
    return brentq(lambda z: 100.0 * np.sqrt(om * (1 + z) ** 3 + 1 - om) / (1 + z) - velfac, 1.0, 7.0, xtol=1e-12)


def test_row23b_priya_hr6_slot_labels_audited():
    """BT-B1 (upstream DEFECT, finding F7). In PRIYA-HR6 exactly 120 of the 1020 (row, z-slot) entries hold a velocity
    grid that is not the slot label's: all slots z <= 3.8 of sim ns0.885 and z <= 2.6 of sim ns0.972 (10 rungs each).
    Each holds the neighbouring snapshot, the same positional mechanism as F1 (row 4c): the implied redshift is the
    next-higher grid z, except the first shifted slot of each simulation, which holds an off-grid snapshot (z = 3.9936
    and 2.7936). Our HR rows for these simulations carry the label-z snapshot."""
    require_real_cache(HR)
    _require_paths(UP_HR6)
    with h5py.File(UP_HR6, "r") as f:
        params, zout, kfkms, kfmpc = f["params"][...], f["zout"][...], f["kfkms"][...], f["kfmpc"][...]
    bad = {}
    for r in range(params.shape[0]):
        hub, omh2 = float(params[r, 6]), float(params[r, 7])
        for zi, z in enumerate(zout):
            kexp = np.asarray(KC.k_skm_from_kcom(kfmpc, float(z), hub, omh2))
            if float(np.max(np.abs(kfkms[r, zi] / kexp - 1))) > 1e-6:
                bad[(r, round(float(z), 1))] = _implied_z(kfmpc[0] / kfkms[r, zi, 0], omh2 / hub ** 2)
    ns = np.round(params[:, 1], 3)
    expect = {(r, z) for r in np.where(ns == 0.885)[0] for z in (3.8, 3.6, 3.4, 3.2, 3.0, 2.8, 2.6, 2.4, 2.2)}
    expect |= {(r, z) for r in np.where(ns == 0.972)[0] for z in (2.6, 2.4, 2.2)}
    assert len(expect) == 120 and set(bad) == expect, sorted(set(bad) ^ expect)[:10]
    for (r, z), zi in bad.items():
        first = (ns[r] == 0.885 and z == 3.8) or (ns[r] == 0.972 and z == 2.6)
        assert abs(zi - (3.9936 if ns[r] == 0.885 else 2.7936)) < 2e-3 if first else abs(zi - (z + 0.2)) < 1e-4, (r, z, zi)
    ours = D.load_cache(HR)
    for s in ("ns0.885", "ns0.972"):
        rows = np.where(np.char.startswith(ours["sim_name"].astype(str), s))[0]
        assert rows.size > 0
        for r in rows:
            fin = np.isfinite(ours["kfkms"][r])
            kexp = np.asarray(KC.k_skm_from_kcom(ours["k_com_hmpc"], float(ours["z_grid"][r]),
                                                 float(ours["params"][r, 5]), float(ours["params"][r, 6])))
            assert float(np.max(np.abs(ours["kfkms"][r][fin] / kexp[fin] - 1))) < 2e-5, (s, float(ours["z_grid"][r]))


def test_row19b_eboss_production_covariance_vs_upstream():
    """BT-B2. Production eBOSS (prod_forward_config: sample_res) removes the resolution systematic from sigma and floats
    f_res in the forward: per z, C_prod = C_up * outer(r, r) with r = sqrt(1 - e_res^2 / diag C_up), C_up from upstream
    BOSSData.get_covar(z) and e_res the resolution column of the same Pk1D_syst.dat upstream sums; no cross-z terms.
    Pins the size of the change (largest variance reduction 0.80) that the parity table records as DIFFERENT."""
    from hcd_analysis.emulator import closure_legb as CL
    from hcd_analysis.emulator import data_likelihood as DL
    cfg = CL.prod_forward_config("eBOSS")
    assert cfg["sample_res"] is True
    leg = DL.load_eboss_leg(resolution_float=cfg["sample_res"])          # the call build_legb_ctx makes for eBOSS
    assert leg.resolution_on is True                                     # f_res is sampled in the forward
    up = run_upstream("""
import json, os, numpy as np
from lyaemu import lyman_data
b = lyman_data.BOSSData()
zs = b.get_redshifts(); kf = b.get_kf()
fn = os.path.join(os.path.dirname(lyman_data.__file__), "data/boss_dr14_data/Pk1D_syst.dat")
syst = np.loadtxt(fn)
ires = open(fn).readline().lstrip("#").split().index("resolution")
rows = [np.where(np.abs(b.redshifts - z) < 0.01)[0] for z in zs]          # the data file's rows of each z
print(json.dumps({"z": zs.tolist(), "k": kf.tolist(), "cov": [b.get_covar(zbin=z).tolist() for z in zs],
                  "e_res": [syst[r, ires].tolist() for r in rows], "k_rows": [b.kf[r].tolist() for r in rows]}))
""")
    zs, kf = np.array(up["z"]), np.array(up["k"])
    C = np.asarray(leg.C_data)
    red = []
    for i, z in enumerate(zs):
        m = np.isclose(leg.z_row, z)
        idx = np.where(m)[0]
        sel = [int(np.argmin(np.abs(kf - kk))) for kk in np.asarray(leg.k)[m]]
        Cu = np.asarray(up["cov"][i])[np.ix_(sel, sel)]
        assert np.allclose(np.asarray(up["k_rows"][i])[sel], np.asarray(leg.k)[m], rtol=1e-9)
        e = np.asarray(up["e_res"][i])[sel]
        r = np.sqrt(1.0 - e ** 2 / np.diag(Cu))
        assert np.allclose(C[np.ix_(idx, idx)], Cu * np.outer(r, r), rtol=1e-10, atol=0)
        assert not np.any(C[np.ix_(idx, np.where(~m)[0])])
        red.append(1.0 - r ** 2)
    red = np.concatenate(red)
    assert red.size == 455 and 0.80 < red.max() < 0.81, red.max()
    print(f"eBOSS production covariance: largest variance reduction {red.max():.4f}, median {np.median(red):.4f}")


def test_row20b_ks_production_covariance_and_cross_z():
    """BT-B3. Production KS (prod_forward_config ks_kwargs: resolution_float, k_max 0.065) = the published conservative
    covariance (FULL, cross-z blocks included) restricted to the kept bins, minus esyst_res_ks^2 on the diagonal only
    ("diag" mode), with f_res floated. Upstream's own likelihood evaluates chi2 per z block (cross-z dropped). Pins the
    retained cross-z correlation (largest |rho| about 0.61)."""
    from hcd_analysis.emulator import closure_legb as CL
    from hcd_analysis.emulator import data_likelihood as DL
    cfg = CL.prod_forward_config("KS")
    leg = DL.load_ks_leg(**cfg["ks_kwargs"])
    assert cfg["ks_kwargs"]["resolution_float"] is True and leg.resolution_on is True
    up = run_upstream("""
import json, os, numpy as np, pandas
from lyaemu import lyman_data
c = lyman_data.KSData(conservative=True)
d = os.path.join(os.path.dirname(lyman_data.__file__), "data/kodiaq_squad/detailed-p1d-results-karacayli_etal2021.txt")
a = pandas.read_csv(d, sep="|", header=0)
cols = {c.strip(): c for c in a.columns}
print(json.dumps({"z": c.redshifts.tolist(), "k": c.kf.tolist(), "cov": c.covar.tolist(),
                  "det_z": a[cols["z"]].values.tolist(), "det_k": a[cols["k"]].values.tolist(),
                  "det_eres": a[cols["esyst_res_ks"]].values.tolist(),
                  "blk_shape": list(np.shape(c.get_covar(zbin=c.get_redshifts()[3])))}))
""")
    uz, uk, Cu = np.array(up["z"]), np.array(up["k"]), np.array(up["cov"])
    eres = {(round(z, 3), round(k, 8)): e for z, k, e in zip(up["det_z"], up["det_k"], up["det_eres"])}
    zr, kk = np.asarray(leg.z_row), np.asarray(leg.k)
    idx = np.array([int(np.where(np.isclose(uz, z) & np.isclose(uk, k, rtol=1e-9))[0][0]) for z, k in zip(zr, kk)])
    e = np.array([eres[(round(float(z), 3), round(float(k), 8))] for z, k in zip(zr, kk)])
    expect = Cu[np.ix_(idx, idx)].copy()
    expect[np.diag_indices_from(expect)] -= e ** 2
    C = np.asarray(leg.C_data)
    assert np.allclose(C, expect, rtol=1e-10, atol=0)
    sd = np.sqrt(np.diag(C))
    rho = C / np.outer(sd, sd)
    cross = ~np.isclose(zr[:, None], zr[None, :])
    assert 0.55 < np.max(np.abs(rho[cross])) < 0.65, np.max(np.abs(rho[cross]))
    assert up["blk_shape"][0] == up["blk_shape"][1] < len(uz)                     # upstream hands out per-z blocks
    src = open(f"{UPSTREAM}/lyaemu/likelihood.py").read()
    assert "chi2 += self.chi2_zbin(" in src and "covar_bin = self.sdss.get_covar(sdssz[zbin])" in src
    red = e ** 2 / np.diag(Cu[np.ix_(idx, idx)])
    print(f"KS production covariance: {kk.size} bins, cross-z max |rho| {np.max(np.abs(rho[cross])):.3f}, "
          f"largest diagonal variance removed {red.max():.3f}")


def test_row10c_parameter_and_tau0_out_of_box_handling():
    """BT-B5 (gate E hand-off). Upstream refuses out-of-box parameters (map_to_unit_cube asserts; the likelihood
    returns -inf at or beyond the box); our unit-cube map extrapolates silently. The production mean-flux box
    (tau0 x dtau0) reaches alpha(z) above the top training rung at low z: pinned here, handed to gates C/E."""
    from hcd_analysis.emulator import meanflux_prior as MP
    src = open(f"{UPSTREAM}/lyaemu/likelihood.py").read()
    assert "if np.any(params >= self.param_limits[:, 1]) or np.any(" in src
    lim = np.asarray(D.PARAM_LIMITS, float)
    x = lim[:, 0] + 0.5 * (lim[:, 1] - lim[:, 0])
    x[0] = lim[0, 1] + 0.1 * (lim[0, 1] - lim[0, 0])                      # ns 10 percent of its range above the box
    up = run_upstream(f"""
import json, numpy as np
from lyaemu.latin_hypercube import map_to_unit_cube
try:
    map_to_unit_cube(np.array({x.tolist()!r}), np.array({lim.tolist()!r})); out = "accepted"
except AssertionError:
    out = "refused"
print(json.dumps(out))
""")
    assert up == "refused"
    u = D.normalize_params(x)
    assert abs(u[0] - 1.1) < 1e-12                                            # ours: 1.1, no refusal
    with h5py.File(LF, "r") as f:
        rungs = np.unique(f["alpha_slope"][...])
    zs = np.round(np.arange(2.2, 4.61, 0.2), 1)
    corners = [(a, d) for a in MP.TAU0_AMP_RANGE for d in MP.DTAU0_RANGE]
    alpha = np.array([[float(MP.tau0_alpha_priya(z, a, d)) for a, d in corners] for z in zs])
    top, bot = float(rungs.max()), float(rungs.min())
    assert abs(top - 1.3312) < 1e-3 and abs(bot - 0.6556) < 1e-3
    assert abs(alpha.max() - 1.3667) < 1e-3 and zs[np.argmax(alpha.max(1))] == 2.2
    print(f"alpha box range {alpha.min():.4f}..{alpha.max():.4f} vs training rungs {bot:.4f}..{top:.4f}; "
          f"excess above the top rung {alpha.max() / top - 1:.3%} at z = 2.2")
