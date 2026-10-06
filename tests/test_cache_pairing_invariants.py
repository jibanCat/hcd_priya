"""Cache-builder pairing invariants (S4 / F8 repair, emulator-debug campaign 2026-10; spec
hcd_priya_notes docs/superpowers/emulator-debug-2026-10/gateB/S4_REPAIR_SPEC.md).

Every Phase-1 catalogue directory snap_NNN was built on raw SPECTRA_NNN (same index). The builder pairs them by that
identity, takes the redshift label from the catalogue meta except for an enumerated table of mislabelled directories
(each entry verified against the raw header), and refuses at build time any catalogue not built on the paired raw file.
Runs in the suite environment (no fake_spectra)."""
import json
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from hcd_analysis.paths import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_emulator_cache_tau0 as bt0  # noqa: E402

NS0907 = "ns0.907Ap1.5e-09herei3.75heref2.77alphaq2.04hub0.662omegamh20.144hireionz7.47bhfeedback0.0347"
LF_RAW = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")
HR_RAW = Path("/scratch/yueyingn_root/yueyingn0/mfho/priya/emu_full_hires_2")
HCD = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")


def _require(*paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        if os.environ.get("HCD_GATE_RUN") == "1":
            pytest.fail(f"inputs absent: {missing}")
        pytest.skip(f"inputs absent: {missing}")


def _hdr(nbins=1365, z=2.8, box=120000.0, hubble=0.662, omegam=0.3287):
    from types import SimpleNamespace
    a = 1.0 / (1.0 + z)
    hz = 100.0 * hubble * np.sqrt(omegam / a ** 3 + 1.0 - omegam)
    return SimpleNamespace(nbins=nbins, redshift=z, box=box, hubble=hubble, Hz=hz, omegam=omegam, omegal=1.0 - omegam)


# ------------------------------------------------------------------------------- pairing is by file identity
def test_no_tau_repointing_table_remains():
    assert not hasattr(bt0, "_RAW_SPECTRA_DIR_OVERRIDE")
    assert not hasattr(bt0, "_raw_spectra_index")


def test_locate_pairs_the_same_index_raw_file():
    with tempfile.TemporaryDirectory() as tmp:
        for s in (17, 18):
            d = Path(tmp) / NS0907 / "output" / f"SPECTRA_{s:03d}"
            d.mkdir(parents=True)
            (d / "lya_forest_spectra_grid_480.hdf5").write_bytes(b"")
        for s in (17, 18):
            got = bt0.locate_raw_tau_file(Path(tmp), NS0907, s)
            assert got == Path(tmp) / NS0907 / "output" / f"SPECTRA_{s:03d}" / "lya_forest_spectra_grid_480.hdf5"


# ------------------------------------------------------------------------------- redshift label
def test_label_override_table_is_the_two_known_mislabels():
    assert bt0._PHASE1_Z_OVERRIDE == {("ns0.907", 17): 3.0, ("ns0.907", 18): 2.8}


def test_label_is_meta_z_when_not_overridden():
    assert bt0.catalogue_z_label("ns0.803Ap2.2e-09", 17, 3.000004, 3.0) == 3.000004


def test_label_override_is_verified_against_the_raw_header():
    assert bt0.catalogue_z_label(NS0907, 17, 2.799998, 2.9999999999999982) == 3.0
    with pytest.raises(ValueError, match="header"):
        bt0.catalogue_z_label(NS0907, 17, 2.799998, 2.8)          # override disagrees with the file


def test_label_override_refuses_a_stale_entry():
    with pytest.raises(ValueError, match="stale"):
        bt0.catalogue_z_label(NS0907, 17, 3.0, 3.0)                # meta already right: entry must be removed


def test_label_must_match_header_within_2e5():
    bt0.assert_label_matches_header(2.799998, 2.8)
    with pytest.raises(ValueError, match="redshift"):
        bt0.assert_label_matches_header(2.8, 3.0)


# ------------------------------------------------------------------------------- catalogue built on this file
def test_catalogue_identity_accepts_the_file_it_was_built_on():
    h = _hdr()
    vmax = bt0.vmax_from_header(h)
    bt0.assert_catalogue_built_on({"nbins": 1365, "dv_kms": vmax / 1365}, 1364, h)


@pytest.mark.parametrize("meta,pix,what", [
    ({"nbins": 1397, "dv_kms": None}, 1000, "nbins"),
    ({"nbins": 1365, "dv_kms": "off"}, 1000, "dv_kms"),
    ({"nbins": 1365, "dv_kms": None}, 1396, "pixel"),
])
def test_catalogue_identity_refuses_another_file(meta, pix, what):
    h = _hdr()
    vmax = bt0.vmax_from_header(h)
    m = dict(meta)
    m["dv_kms"] = vmax / 1365 * (1 + 9.2e-5) if m["dv_kms"] == "off" else vmax / 1365
    with pytest.raises(ValueError, match=what):
        bt0.assert_catalogue_built_on(m, pix, h)


def test_vmax_from_header_matches_fake_spectra_formula():
    h = _hdr(z=3.0, hubble=0.7, omegam=0.3)
    a = 1.0 / 4.0
    expect = 120000.0 * (3.085678e21 * a / 0.7) * h.Hz / 3.085678e24
    assert abs(bt0.vmax_from_header(h) / expect - 1) < 1e-15


# ------------------------------------------------------------------------------- real data
def test_override_entries_match_the_raw_headers_and_meta():
    for (tag, snap), z in bt0._PHASE1_Z_OVERRIDE.items():
        assert tag in NS0907
        raw = LF_RAW / NS0907 / "output" / f"SPECTRA_{snap:03d}" / "lya_forest_spectra_grid_480.hdf5"
        meta = HCD / NS0907 / f"snap_{snap:03d}" / "meta.json"
        _require(raw, meta)
        with h5py.File(raw, "r") as f:
            zr = float(f["Header"].attrs["redshift"])
        assert abs(zr - z) < 1e-6
        assert abs(float(json.load(open(meta))["z"]) - zr) > 2e-5


def test_lf_discovery_pairs_ns0907_by_identity():
    _require(HCD, LF_RAW)
    pairs = bt0.discover_tau0_pairs(HCD, LF_RAW, fidelity="lf")
    assert len(pairs) == 1073
    got = {snap: (raw, bt0._snap_z_to_priya_grid(bt0.catalogue_z_label(
        sim, snap, float(json.load(open(sd / "meta.json"))["z"]), bt0.raw_header_z(raw))))
        for sim, snap, sd, raw in pairs if sim == NS0907}
    assert {s: round(v[1], 1) for s, v in got.items() if s in (16, 17, 18, 19)} == {16: 3.2, 17: 3.0, 18: 2.8, 19: 2.6}
    for s, (raw, _) in got.items():
        assert Path(raw).parent.name == f"SPECTRA_{s:03d}"


@pytest.mark.parametrize("fid", ["lf", "hr"])
def test_every_discovered_pair_satisfies_the_build_invariants(fid):
    """The hard pairing invariant at discovery level, for every group the builder would build: the catalogue was built
    on the paired raw file and its (corrected) label equals the file's header redshift."""
    raw_root = LF_RAW if fid == "lf" else HR_RAW
    _require(HCD, raw_root)
    from hcd_analysis.io import read_header
    pairs = bt0.discover_tau0_pairs(HCD, raw_root, fidelity=fid)
    assert len(pairs) == (1073 if fid == "lf" else 103)
    for sim, snap, sd, raw in pairs:
        meta = json.load(open(sd / "meta.json"))
        with np.load(sd / "catalog.npz") as c:
            pix = int(c["pix_start"].max()) if c["pix_start"].size else -1
        hdr = read_header(raw)
        bt0.assert_catalogue_built_on(meta, pix, hdr)
        bt0.assert_label_matches_header(bt0.catalogue_z_label(sim, snap, float(meta["z"]), hdr.redshift), hdr.redshift)
