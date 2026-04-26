"""
Tests for ``scripts/patch_cddf_dx.py``.

Locks the per-snap and per-stacked correction so future regressions of
the dX-bug fix are caught.

Run::

    python3 tests/test_cddf_dx_patch.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import patch_cddf_dx as P  # noqa: E402  (path manipulation above)


def _fake_per_snap(z: float, n_sightlines: int = 1000) -> dict:
    """Synthesise a 'buggy' per-snap cddf.npz dict."""
    log_nhi_edges = np.linspace(17.0, 23.0, 31)
    log_nhi_centres = 0.5 * (log_nhi_edges[:-1] + log_nhi_edges[1:])
    n_absorbers = (1e6 * 10 ** (-1.5 * (log_nhi_centres - 17.0))).astype(np.int64)
    dN = 10.0 ** log_nhi_edges[1:] - 10.0 ** log_nhi_edges[:-1]
    dX_buggy = 0.5
    total_path_buggy = dX_buggy * n_sightlines
    f_buggy = np.where(dN > 0, n_absorbers / (dN * total_path_buggy), 0.0)
    return {
        "log_nhi_centres": log_nhi_centres,
        "log_nhi_edges": log_nhi_edges,
        "f_nhi": f_buggy,
        "n_absorbers": n_absorbers,
        "dX_per_sightline": np.array(dX_buggy),
        "total_path": np.array(total_path_buggy),
        "n_sightlines": np.array(n_sightlines),
        "z": np.array(z),
    }


class TestPerSnapPatch(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cddf_patch_"))
        self.snap_dir = self.tmp / "sim_X" / "snap_010"
        self.snap_dir.mkdir(parents=True)
        self.h = 0.7
        with open(self.snap_dir / "meta.json", "w") as f:
            json.dump({"hubble": self.h, "z": 3.0}, f)
        self.cddf_path = self.snap_dir / "cddf.npz"
        np.savez(self.cddf_path, **_fake_per_snap(z=3.0))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_f_nhi_ratio_equals_factor(self):
        """f_nhi_buggy / f_nhi_patched must equal (1+z)·h."""
        orig = np.load(self.cddf_path)
        patched = P.patch_snap_cddf(self.cddf_path, self.h)
        z = float(orig["z"])
        factor = (1 + z) * self.h
        nz = orig["f_nhi"] > 0
        ratio = orig["f_nhi"][nz] / patched["f_nhi"][nz]
        np.testing.assert_allclose(ratio, factor, rtol=1e-12)

    def test_dX_and_total_path_scale_correctly(self):
        orig = np.load(self.cddf_path)
        patched = P.patch_snap_cddf(self.cddf_path, self.h)
        factor = (1 + float(orig["z"])) * self.h
        np.testing.assert_allclose(
            float(patched["dX_per_sightline"]),
            float(orig["dX_per_sightline"]) * factor, rtol=1e-12,
        )
        np.testing.assert_allclose(
            float(patched["total_path"]),
            float(orig["total_path"]) * factor, rtol=1e-12,
        )

    def test_invariants_preserved(self):
        orig = np.load(self.cddf_path)
        patched = P.patch_snap_cddf(self.cddf_path, self.h)
        np.testing.assert_array_equal(orig["n_absorbers"], patched["n_absorbers"])
        np.testing.assert_array_equal(orig["log_nhi_edges"], patched["log_nhi_edges"])
        self.assertTrue(bool(patched["dx_bug_patched"]))

    def test_idempotent(self):
        patched = P.patch_snap_cddf(self.cddf_path, self.h)
        # Save and re-load to round-trip; then re-patch.
        np.savez(self.cddf_path, **patched)
        again = P.patch_snap_cddf(self.cddf_path, self.h)
        np.testing.assert_array_equal(patched["f_nhi"], again["f_nhi"])

    def test_post_patch_self_consistency(self):
        """Recomputing f_nhi from patched values reproduces patched f_nhi."""
        patched = P.patch_snap_cddf(self.cddf_path, self.h)
        edges = patched["log_nhi_edges"]
        dN = 10.0 ** edges[1:] - 10.0 ** edges[:-1]
        n = patched["n_absorbers"].astype(float)
        path = float(patched["total_path"])
        with np.errstate(divide="ignore", invalid="ignore"):
            f_recompute = np.where(dN > 0, n / (dN * path), 0.0)
        np.testing.assert_allclose(f_recompute, patched["f_nhi"], rtol=1e-12)


class TestStackedPatch(unittest.TestCase):
    """Per-sim stacked patch: build a fake sim with two snaps, stack, patch, verify."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cddf_patch_stk_"))
        self.h = 0.7
        self.sim_dir = self.tmp / "sim_X"
        self.sim_dir.mkdir(parents=True)

        # Two snaps at z=3.0 and z=4.0
        self.snaps: list = []
        for snap_idx, z in [(10, 3.0), (12, 4.0)]:
            sd = self.sim_dir / f"snap_{snap_idx:03d}"
            sd.mkdir()
            with open(sd / "meta.json", "w") as f:
                json.dump({"hubble": self.h, "z": z}, f)
            d = _fake_per_snap(z=z, n_sightlines=1000)
            np.savez(sd / "cddf.npz", **d)
            self.snaps.append(d)

        # Build a stacked file mimicking the production layout
        d0, d1 = self.snaps
        bin_label = "z3p0to5p0"  # both snaps fall in this synthesised bin
        log_nhi_edges = d0["log_nhi_edges"]
        log_nhi_centres = d0["log_nhi_centres"]
        n_absorbers = d0["n_absorbers"] + d1["n_absorbers"]
        path_total_buggy = float(d0["total_path"]) + float(d1["total_path"])
        dN = 10.0 ** log_nhi_edges[1:] - 10.0 ** log_nhi_edges[:-1]
        f_stacked_buggy = np.where(
            dN > 0, n_absorbers / (dN * path_total_buggy), 0.0,
        )
        save_dict = {
            f"{bin_label}__f_nhi": f_stacked_buggy,
            f"{bin_label}__f_nhi_per_snap": np.stack([d0["f_nhi"], d1["f_nhi"]]),
            f"{bin_label}__n_absorbers": n_absorbers.astype(np.int64),
            f"{bin_label}__log_nhi_centres": log_nhi_centres,
            f"{bin_label}__log_nhi_edges": log_nhi_edges,
            f"{bin_label}__z_snapshots": np.array([3.0, 4.0]),
            f"{bin_label}__z_min": np.array([3.0]),
            f"{bin_label}__z_max": np.array([5.0]),
            f"{bin_label}__total_path": np.array([path_total_buggy]),
        }
        self.stk_path = self.sim_dir / "cddf_stacked.npz"
        np.savez(self.stk_path, **save_dict)
        self.bin_label = bin_label

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_per_snap_rows_corrected(self):
        patched = P.patch_stacked_cddf(self.stk_path, self.h)
        orig = np.load(self.stk_path)
        z_snaps = np.array([3.0, 4.0])
        factors = (1 + z_snaps) * self.h
        for i, factor in enumerate(factors):
            nz = orig[f"{self.bin_label}__f_nhi_per_snap"][i] > 0
            ratio = (
                orig[f"{self.bin_label}__f_nhi_per_snap"][i][nz]
                / patched[f"{self.bin_label}__f_nhi_per_snap"][i][nz]
            )
            np.testing.assert_allclose(ratio, factor, rtol=1e-12)

    def test_stacked_total_path_recomputed(self):
        patched = P.patch_stacked_cddf(self.stk_path, self.h)
        # Expected: Σ_i (1+z_i)·h · buggy_path_i
        d0, d1 = self.snaps
        expected = (
            (1 + 3.0) * self.h * float(d0["total_path"])
            + (1 + 4.0) * self.h * float(d1["total_path"])
        )
        got = float(patched[f"{self.bin_label}__total_path"].ravel()[0])
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_stacked_f_nhi_self_consistent(self):
        patched = P.patch_stacked_cddf(self.stk_path, self.h)
        edges = patched[f"{self.bin_label}__log_nhi_edges"]
        dN = 10.0 ** edges[1:] - 10.0 ** edges[:-1]
        n = patched[f"{self.bin_label}__n_absorbers"].astype(float)
        path = float(patched[f"{self.bin_label}__total_path"].ravel()[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            f_recompute = np.where(dN > 0, n / (dN * path), 0.0)
        np.testing.assert_allclose(
            f_recompute, patched[f"{self.bin_label}__f_nhi"], rtol=1e-12,
        )

    def test_idempotent(self):
        patched = P.patch_stacked_cddf(self.stk_path, self.h)
        np.savez(self.stk_path, **patched)
        again = P.patch_stacked_cddf(self.stk_path, self.h)
        np.testing.assert_allclose(
            patched[f"{self.bin_label}__f_nhi"],
            again[f"{self.bin_label}__f_nhi"],
            rtol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
