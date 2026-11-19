"""Gate D review blocking tests (PU-0063/PU-0064; gate D review records).

BT-D1 the accepted D1 failure is pinned to its cause (one design point's HR/LF parameter precision), so a later genuine
mapping error cannot hide behind "the known 4.4e-10". BT-D2 D3 is reported per held-out simulation and on a common
mode-index band (the pooled value of a leave-one-out mean correction is fixed by construction). BT-D3 the MF prediction
does not depend on log_rho or on the held-out rows (the historical "log_rho leak" was a misstatement). BT-D4 the R-D1
products (the R-D1 summary and its k-resolved table) are complete and consistent."""

import json
import os

import numpy as np
import pytest

from hcd_analysis.emulator import gate_d as GD
from hcd_analysis.emulator import mf_modes as MM
from hcd_analysis.emulator.data import load_cache_historical
from tests.gate_helpers import external_path, real_cache_path

LF, HR = real_cache_path("lf"), real_cache_path("hr")
GATED = external_path("HCD_EMULATOR_V2_DIR", "mf", "gateD")
RAW_LF = external_path("HCD_LF_RAW_DIR")
RAW_HR = external_path("HCD_HR_RAW_DIR")
NS0914 = "ns0.914Ap1.32e-09"


def _need(*paths):
    miss = [p for p in paths if not os.path.exists(p)]
    if miss:
        if os.environ.get("HCD_GATE_RUN") == "1":
            pytest.fail(f"inputs absent: {miss}")
        pytest.skip(f"inputs absent: {miss}")


# ---------------------------------------------------------------------------------------------------------- BT-D1
def test_gate_d_d1_failure_pinned_to_parameter_precision():
    from tests.raw_headers import _raw_headers
    from hcd_analysis.emulator.kcoord import k_skm_from_kcom

    _need(LF, HR, RAW_LF, RAW_HR)
    lf, hr = load_cache_historical(
        LF, reason="test: a cache in the pre-lineage 3.3 format"
    ), load_cache_historical(HR, reason="test: a cache in the pre-lineage 3.3 format")
    pairs = MM.match_pairs(lf, hr)
    K = 172
    hdr = {}
    n_pinned = 0
    for h, l in pairs:
        a, b = np.asarray(hr["kfkms"][h][:K], float), np.asarray(lf["kfkms"][l][:K], float)
        m = np.isfinite(a) & np.isfinite(b)
        dev = float(np.max(np.abs(a[m] / b[m] - 1)))
        if dev <= 1e-15:
            continue
        s_lf, s_hr = str(lf["sim_name"][l]), str(hr["sim_name"][h])
        assert NS0914 in s_lf and NS0914 in s_hr, (s_lf, dev)
        z = float(hr["z_grid"][h])
        vh = min(
            hdr.setdefault(("hr", s_hr), _raw_headers(RAW_HR, s_hr)), key=lambda e: abs(e["z"] - z)
        )["vmax"]
        vl = min(
            hdr.setdefault(("lf", s_lf), _raw_headers(RAW_LF, s_lf)), key=lambda e: abs(e["z"] - z)
        )["vmax"]
        assert abs(dev - abs(vl / vh - 1)) <= 1e-15 and dev < 5e-10, (dev, vl / vh - 1)
        n_pinned += 1
    assert n_pinned == 340
    for cache in (lf, hr):  # each grid is the canonical coordinate of its OWN parameters
        p = np.asarray(cache["params"], float)
        k = np.asarray(cache["kfkms"], float)[:, :K]
        kc = np.stack(
            [
                np.asarray(k_skm_from_kcom(np.asarray(cache["k_com_hmpc"])[:K], float(zz), hub, om))
                for zz, hub, om in zip(cache["z_grid"], p[:, 5], p[:, 6])
            ]
        )
        fin = np.isfinite(k)
        assert np.max(np.abs(kc[fin] / k[fin] - 1)) <= 2e-15  # round-off (C1 measured 1.1e-15)


# ---------------------------------------------------------------------------------------------------------- BT-D2
def _toy(n_sims=6, n_rows=4, K=10, seed=0):
    rng = np.random.default_rng(seed)
    sims = np.repeat([f"s{i}" for i in range(n_sims)], n_rows)
    z = np.tile(np.array([2.8, 3.0, 3.2, 3.4])[:n_rows], n_sims)
    k = np.geomspace(0.01, 0.08, K)[None, :] * (1 + 0.05 * rng.random((sims.size, 1)))
    return sims, z, k, np.ones((sims.size, K), bool)


def test_d3_per_simulation_and_common_mode_band_helpers():
    sims, z, k, keep = _toy()
    l_mf = np.zeros((sims.size, 4, k.shape[1]))
    l_lf = np.full_like(l_mf, -0.01)
    for i, s in enumerate(np.unique(sims)):
        l_mf[sims == s] = 0.01 * (i - 2.5)
    out = GD.d3_per_simulation(l_mf, l_lf, sims, z, k, keep)
    assert set(out["per_simulation"]) == set(np.unique(sims))
    np.testing.assert_allclose(
        out["rms_over_simulations"]["clean"]["mf"],
        np.sqrt(np.mean((0.01 * (np.arange(6) - 2.5)) ** 2)),
        rtol=1e-12,
    )
    np.testing.assert_allclose(out["rms_over_simulations"]["clean"]["lf"], 0.01, rtol=1e-12)
    band = GD.common_mode_band(z, k, keep)
    n0 = int(np.argmax(band[0]))
    assert np.all(band[:, n0:]) and not np.any(band[:, :n0])


def test_d3_on_the_frozen_product_is_reported_per_simulation():
    _need(f"{GATED}/gate_d_arrays.npz", f"{GATED}/gate_d_results.json")
    a = np.load(f"{GATED}/gate_d_arrays.npz")
    R = json.load(open(f"{GATED}/gate_d_results.json"))
    l_mf, l_lf = np.log1p(a["r_mf"]), np.log1p(a["r_lf"])
    keep = a["in_range"][:, None, :] & np.isfinite(a["r_mf"]) & np.isfinite(a["r_lf"])
    out = GD.d3_per_simulation(l_mf, l_lf, a["sim"], a["z"], a["k_hr"], keep)
    for c in GD.CLS:  # the registered pooled values are reproduced
        assert abs(out["pooled_registered_band"][c]["mf"] - R["D3"]["mean_log_mf"][c]) < 1e-12
        assert abs(out["pooled_registered_band"][c]["lf"] - R["D3"]["mean_log_lf"][c]) < 1e-12
        assert (
            abs(out["pooled_common_mode_band"][c]["mf"]) < 1e-12
        )  # fixed by leave-one-out averaging
    assert len(out["per_simulation"]) == 6
    p = f"{GATED}/gate_d_d3_per_simulation.json"
    _need(p)
    assert json.load(open(p)) == json.loads(json.dumps(out))


# ---------------------------------------------------------------------------------------------------------- BT-D3
def _real_targets():
    _need(LF, HR, f"{GATED}/gate_d_arrays.npz")
    lf, hr = load_cache_historical(
        LF, reason="test: a cache in the pre-lineage 3.3 format"
    ), load_cache_historical(HR, reason="test: a cache in the pre-lineage 3.3 format")
    pairs = MM.match_pairs(lf, hr)
    a = np.load(f"{GATED}/gate_d_arrays.npz")
    l_rows = np.array([p[1] for p in pairs])
    assert np.allclose(np.asarray(lf["z_grid"])[l_rows], a["z"])  # the product's pair order
    return dict(
        x=np.asarray(lf["x"])[l_rows],
        tau0=np.asarray(lf["tau0"])[l_rows],
        alpha_idx=np.asarray(lf["alpha_idx"])[l_rows],
        sim=a["sim"].astype(str),
        g=a["g_measured"].copy(),
    )


def test_mode_mf_prediction_invariant_to_log_rho_and_heldout_rows_real():
    t = _real_targets()
    rng = np.random.default_rng(3)
    for s in np.unique(t["sim"]):
        held = np.where(t["sim"] == s)[0]
        train = np.where(t["sim"] != s)[0]
        tab = MM.fit_mode_mf(t, train_rows=train)
        c = rng.normal(0, 0.05, tab["log_rho"].shape)
        tab2 = dict(
            tab, log_rho=tab["log_rho"] + c, gbar_z_tab=tab["gbar_z_tab"] - c[None, None, :]
        )
        t7 = dict(t, g=t["g"].copy())
        t7["g"][held] += 7.0
        tab7 = MM.fit_mode_mf(t7, train_rows=train)
        for i in held[:: max(1, held.size // 12)]:
            g0 = MM.apply_mode_mf(tab, t["x"][i], t["tau0"][i])
            np.testing.assert_allclose(
                MM.apply_mode_mf(tab2, t["x"][i], t["tau0"][i]), g0, rtol=0, atol=1e-15
            )
            assert np.array_equal(MM.apply_mode_mf(tab7, t["x"][i], t["tau0"][i]), g0)


# ---------------------------------------------------------------------------------------------------------- BT-D4
def test_gate_d_rd1_report_matches_json():
    """The JSON products only (the human-readable report is not part of the repository): R_D1 has one value per class
    at every z, all finite on one contiguous data band and NaN outside it, and the k-resolved table covers every class
    at the representative z with one finite value per k, inside the spread over simulations."""
    _need(f"{GATED}/gate_d_results.json", f"{GATED}/gate_d_rd1_k_resolved.json")
    R = json.load(open(f"{GATED}/gate_d_results.json"))["R_D1"]
    zs = sorted(R, key=float)
    assert all(len(R[zz]) == len(GD.CLS) for zz in zs)
    band = [zz for zz in zs if np.isfinite(R[zz][0])]
    assert (
        band and band == zs[zs.index(band[0]) : zs.index(band[-1]) + 1]
    )  # one contiguous data band
    assert all(np.all(np.isfinite(R[zz])) for zz in band)
    assert all(np.all(np.isnan(R[zz])) for zz in zs if zz not in band)  # NaN outside the band
    kres = json.load(open(f"{GATED}/gate_d_rd1_k_resolved.json"))
    nk = len(kres["k_skm"])
    for zz in ("2.2", "3.0", "4.6"):
        assert zz in band
        for c in GD.CLS:
            row = np.asarray(kres["representative"]["table"][c][zz], dtype=float)
            assert row.shape == (nk,) and np.all(np.isfinite(row))
            lo = np.asarray(kres["spread_over_sims"][c][zz]["min"], dtype=float)
            hi = np.asarray(kres["spread_over_sims"][c][zz]["max"], dtype=float)
            assert lo.shape == hi.shape == (nk,) and np.all(lo <= row) and np.all(row <= hi), (
                c,
                zz,
            )
