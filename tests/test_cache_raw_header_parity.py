"""Cache and raw-header parity rows extracted from 863df63 tests/test_parity_upstream.py (gate B; test code verbatim):
the rows that execute no upstream code and read no upstream emulator product (row 10: the class decomposition
reconstructs the total; row 11: the production legs lie inside the simulated modes; row 04d: every cache row matches
its raw snapshot header, the S4 invariant; row 23b: the PRIYA-HR6 slot-label audit, F7). Row 11 loads the data legs
through the carried data_likelihood loaders, whose KS default reads the KODIAQ-SQUAD measurement files from the local
upstream checkout's data directory, as every carried KS-leg test does. The other rows of that module run
lya_emulator_full or read its emulator products from the local upstream checkout and stay excluded.
The module constants these rows use are copied verbatim; docstrings are reworded; _raw_headers comes from
tests/raw_headers.py (itself verbatim)."""
import json
import os

import h5py
import numpy as np
import pytest

from hcd_analysis.emulator import data as D
from hcd_analysis.emulator import kcoord as KC
from hcd_analysis.emulator import schema as S
from tests.gate_helpers import real_cache_path, require_real_cache
from tests.raw_headers import _raw_headers

LF = real_cache_path("lf")
HR = real_cache_path("hr")
SIM_OUR_GAP = "ns0.907Ap1.5e-09herei3.75heref2.77alphaq2.04hub0.662omegamh20.144hireionz7.47bhfeedback0.0347"
RAW = "/nfs/turbo/umor-yueyingn/mfho/emu_full"
LF_HIST = real_cache_path("lf_hist")
HR_RAW = "/scratch/yueyingn_root/yueyingn0/mfho/priya/emu_full_hires_2"
HCD_OUT = "/scratch/cavestru_root/cavestru0/mfho/hcd_outputs"
UP_HR6 = ("/nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48_20260414_newhires/hires/"
          "mf_emulator_flux_vectors_tau1000000.hdf5")
F8_GROUP = (SIM_OUR_GAP, 17)


def _require_paths(*paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        if os.environ.get("HCD_GATE_RUN") == "1":
            pytest.fail(f"inputs absent: {missing}")
        pytest.skip(f"inputs absent: {missing}")


def _implied_z(velfac, om):
    """Redshift at which 100 E(z) / (1+z) equals velfac (monotone for z > 1)."""
    from scipy.optimize import brentq
    return brentq(lambda z: 100.0 * np.sqrt(om * (1 + z) ** 3 + 1 - om) / (1 + z) - velfac, 1.0, 7.0, xtol=1e-12)


def test_row10_class_decomposition_reconstructs_the_upstream_total():
    """Our emulator target is the per-class filtered P1D; its count-weighted sum is the Tier P total that row 5 shows
    equal to upstream's flux_vectors (structural identity on every real row)."""
    require_real_cache(LF)
    d = D.load_cache(LF)
    tot = np.einsum("rc,rck->rk", d["w_c_cache"], d["P_filt"])
    ok = np.isfinite(d["P_tier_p"])
    assert np.max(np.abs(tot[ok] / d["P_tier_p"][ok] - 1.0)) < 1e-10


def test_row11_production_legs_inside_the_simulated_modes_for_every_theta_in_the_box():
    """Upstream raises outside the simulated k range; the old forward clamped. Every production data bin lies inside
    [k_1, k_172] for every (hub, omegamh2) corner of the box at its redshift, so no data bin needs extrapolation."""
    from hcd_analysis.emulator import data_likelihood as DL
    k1, kN = 2 * np.pi / S.L_BOX_HMPC, 2 * np.pi * 172 / S.L_BOX_HMPC
    lim = np.asarray(D.PARAM_LIMITS)
    corners = [(h, w) for h in lim[5] for w in lim[6]]
    for leg in (DL.load_eboss_leg(), DL.load_desi_leg(), DL.load_ks_leg(k_max=0.069)):
        k, zr = np.asarray(leg.k), np.asarray(leg.z_row)
        for z in np.unique(zr):
            kk = k[np.isclose(zr, z)]
            lo = max(float(KC.k_skm_from_kcom(k1, z, h, w)) for h, w in corners)
            hi = min(float(KC.k_skm_from_kcom(kN, z, h, w)) for h, w in corners)
            assert kk.min() >= lo and kk.max() <= hi, (leg.name, z, kk.min(), kk.max(), lo, hi)


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


def test_row23b_priya_hr6_slot_labels_audited():
    """BT-B1 (F7): the slot-label audit of the PRIYA-HR6 product. The (row, z-slot) entries whose velocity grid does not
    follow from the slot label are pinned exactly (an enumerated set of 120 of the 1020, slots z <= 3.8 of sim ns0.885
    and z <= 2.6 of sim ns0.972, 10 rungs each) together with the redshift each grid implies; our HR rows for these
    simulations are checked to carry the label-z snapshot."""
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

