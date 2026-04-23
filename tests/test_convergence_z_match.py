"""
Unit tests for hcd_analysis.pipeline.compute_convergence_ratios —
z-tolerance matching between LF and HiRes snapshot pairs.

Background
----------
LF and HiRes simulations save snapshots at slightly different scale
factors: the same snap number (e.g. snap_010) is ~0.2 higher in z on
HiRes than on LF, because the two campaigns use different
`Snapshots.txt` target-a tables.  Matching by snap folder name
therefore produced a T(k) = P_HR / P_LF ratio that mixed resolution
with 0.2-z-step z-evolution — a bug.  The fix is to match LF↔HR
snapshots by redshift within a tolerance (`|z_HR - z_LF| < z_tol`).

These tests lock in the corrected behaviour before any science claim
is drawn from the convergence figure.  Per project convention
(test-before-claim, see test_absorption_path.py for the template).

Run with:
    python3 tests/test_convergence_z_match.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hcd_analysis.config import PipelineConfig
from hcd_analysis.pipeline import compute_convergence_ratios


# ---------------------------------------------------------------------------
# Test fixture helpers
# ---------------------------------------------------------------------------

def _write_snap(snap_dir: Path, z: float, p1d_value: float, n_k: int = 8) -> None:
    """
    Create a snap directory with meta.json, 'done' sentinel, and a p1d.npz
    whose `p1d_all` and `p1d_no_DLA_priya` arrays are constant = p1d_value.

    The k array is the same for every snap (so LF/HR k match by construction).
    """
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "meta.json").write_text(json.dumps({"z": float(z)}))
    (snap_dir / "done").write_text("")

    k = np.linspace(1e-3, 1e-1, n_k, dtype=float)
    p = np.full(n_k, float(p1d_value), dtype=float)
    np.savez(
        snap_dir / "p1d.npz",
        k_all=k,
        p1d_all=p,
        k_no_DLA_priya=k,
        p1d_no_DLA_priya=p,
    )


def _build_fake_tree(root: Path, sim: str, lf_snaps: dict, hr_snaps: dict) -> None:
    """
    Populate a fake <output_root> layout:

        root/<sim>/snap_NNN/...            (LF)
        root/hires/<sim>/snap_NNN/...      (HR)

    `lf_snaps` / `hr_snaps` map snap_name -> (z, p1d_value).
    """
    for snap_name, (z, pval) in lf_snaps.items():
        _write_snap(root / sim / snap_name, z=z, p1d_value=pval)
    for snap_name, (z, pval) in hr_snaps.items():
        _write_snap(root / "hires" / sim / snap_name, z=z, p1d_value=pval)


# ---------------------------------------------------------------------------
# The actual tests
# ---------------------------------------------------------------------------

def test_exact_z_offset_matches_by_z_not_by_snap_name():
    """
    Real-data scenario: HR snap_N is at the same z as LF snap_(N-1)
    (uniform 0.2-z offset between the two suites).  Matching by snap
    name would pair snap_N with snap_N across both (wrong — mixes z);
    matching by z must pair HR snap_N with LF snap_(N-1).

    We encode each snap's p1d as its snap number so the resulting
    ratio T = p_HR / p_LF reveals which pair was picked.
    """
    sim = "test_sim"
    lf_snaps = {f"snap_{i:03d}": (5.4 - 0.2 * (i - 4), float(i))
                for i in range(4, 20)}
    # HR shifted by one snap: HR snap_N should land at LF snap_(N-1)'s z
    hr_snaps = {f"snap_{i:03d}": (5.4 - 0.2 * (i - 5), float(i))
                for i in range(5, 21)}
    # Sanity: HR snap_010 should be at z=5.4-0.2*5 = 4.4,
    # LF snap_009 should be at z=5.4-0.2*5 = 4.4 → |Δz|=0.0
    assert hr_snaps["snap_010"][0] == lf_snaps["snap_009"][0] == 4.4
    # LF snap_010 is at z=4.2 → same-snap match would have |Δz|=0.2 (wrong)
    assert lf_snaps["snap_010"][0] == 4.2

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_fake_tree(root, sim, lf_snaps, hr_snaps)

        cfg = PipelineConfig(output_root=str(root))
        ratios = compute_convergence_ratios(cfg, z_tol=0.05)

    assert sim in ratios, "expected one matched sim"

    # For every HR snap that has an exact LF z-match, assert that the
    # T_k value equals p_HR / p_LF_matched (= HR_snap_num / (HR_snap_num - 1)).
    for z_key, vdata in ratios[sim].items():
        # HR snap numbers 5..20 correspond to HR z values 5.4..2.4.
        hr_snap_num = 5 + int(round((5.4 - z_key) / 0.2))
        expected_lf_snap_num = hr_snap_num - 1
        expected_T = hr_snap_num / expected_lf_snap_num
        T_all = vdata["all"]["T_k"]
        assert np.allclose(T_all, expected_T, rtol=1e-10), (
            f"z={z_key}: T(k)={T_all[0]:.4f} but expected "
            f"HR snap_{hr_snap_num}/LF snap_{expected_lf_snap_num} = {expected_T:.4f}"
        )
    print("  ✓ HR snap_N matches LF snap_(N-1) by z (not by snap name)")


def test_hr_with_no_lf_within_tolerance_is_skipped():
    """
    HR has a snap at z=5.6 but LF's earliest is z=5.4 → |Δz|=0.2 > z_tol=0.05.
    That HR snap must NOT appear in the output ratio dict.
    """
    sim = "test_sim"
    lf_snaps = {"snap_004": (5.4, 10.0), "snap_005": (5.2, 11.0)}
    hr_snaps = {
        "snap_004": (5.6, 20.0),  # no LF within tol of 5.6 (closest is 5.4)
        "snap_005": (5.4, 21.0),  # matches LF snap_004 exactly
    }

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_fake_tree(root, sim, lf_snaps, hr_snaps)
        cfg = PipelineConfig(output_root=str(root))
        ratios = compute_convergence_ratios(cfg, z_tol=0.05)

    # Only one pair should match: HR snap_005 (z=5.4) ↔ LF snap_004 (z=5.4).
    # The resulting ratio should be keyed by that matched z = 5.4.
    assert sim in ratios
    assert len(ratios[sim]) == 1, (
        f"expected 1 matched z, got {len(ratios[sim])}: "
        f"{list(ratios[sim].keys())}"
    )
    (z_key,) = ratios[sim].keys()
    assert abs(z_key - 5.4) < 1e-6
    # Matched pair should be HR snap_005 / LF snap_004 = 21.0 / 10.0 = 2.1
    T = ratios[sim][z_key]["all"]["T_k"]
    assert np.allclose(T, 2.1, rtol=1e-10), (
        f"T(k)={T[0]:.4f}, expected 2.1 (HR snap_005 / LF snap_004)"
    )
    print("  ✓ HR snap with no LF within z_tol is skipped")


def test_z_tol_controls_match_window():
    """
    With z_tol=0.01, an LF snap that is 0.05 away in z should NOT match.
    With z_tol=0.1, the same pair should match.
    """
    sim = "test_sim"
    lf_snaps = {"snap_010": (4.25, 5.0)}    # z slightly off from HR
    hr_snaps = {"snap_010": (4.20, 7.0)}    # |Δz| = 0.05

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_fake_tree(root, sim, lf_snaps, hr_snaps)
        cfg = PipelineConfig(output_root=str(root))

        # Tight tol: should skip (|Δz|=0.05 > 0.01)
        ratios_tight = compute_convergence_ratios(cfg, z_tol=0.01)
        assert sim not in ratios_tight or len(ratios_tight.get(sim, {})) == 0, (
            "tight z_tol=0.01 must skip |Δz|=0.05 pair"
        )
        # Loose tol: should match
        ratios_loose = compute_convergence_ratios(cfg, z_tol=0.1)
        assert sim in ratios_loose and len(ratios_loose[sim]) == 1, (
            "loose z_tol=0.1 must match |Δz|=0.05 pair"
        )
    print("  ✓ z_tol boundary controls inclusion (tight→skip, loose→match)")


def test_hr_without_done_sentinel_is_skipped():
    """
    A snap directory with meta.json but no `done` sentinel is incomplete
    and must be skipped, even if a z-match exists.
    """
    sim = "test_sim"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Build normal pair
        _build_fake_tree(
            root, sim,
            lf_snaps={"snap_009": (4.4, 1.0), "snap_010": (4.2, 2.0)},
            hr_snaps={"snap_010": (4.4, 3.0), "snap_011": (4.2, 4.0)},
        )
        # Remove HR snap_010's 'done' file to simulate an incomplete snap
        (root / "hires" / sim / "snap_010" / "done").unlink()

        cfg = PipelineConfig(output_root=str(root))
        ratios = compute_convergence_ratios(cfg, z_tol=0.05)

    # Only HR snap_011 (z=4.2) should match → with LF snap_010 (z=4.2)
    assert sim in ratios
    assert len(ratios[sim]) == 1
    (z_key,) = ratios[sim].keys()
    assert abs(z_key - 4.2) < 1e-6
    T = ratios[sim][z_key]["all"]["T_k"]
    assert np.allclose(T, 4.0 / 2.0, rtol=1e-10)
    print("  ✓ snap without 'done' sentinel is skipped")


def test_tie_break_picks_closest_then_lowest_snap():
    """
    If two LF snaps are equidistant in z from an HR snap, picking
    must be deterministic.  We pick the snap with the smaller z-gap;
    if gaps are exactly equal, lower snap number wins.
    """
    sim = "test_sim"
    lf_snaps = {
        "snap_009": (4.5, 10.0),   # Δz = +0.1 from HR z=4.4
        "snap_010": (4.3, 20.0),   # Δz = -0.1 from HR z=4.4  (equidistant)
    }
    hr_snaps = {"snap_010": (4.4, 99.0)}

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_fake_tree(root, sim, lf_snaps, hr_snaps)
        cfg = PipelineConfig(output_root=str(root))
        ratios = compute_convergence_ratios(cfg, z_tol=0.2)

    assert sim in ratios and len(ratios[sim]) == 1
    (z_key,) = ratios[sim].keys()
    T = ratios[sim][z_key]["all"]["T_k"]
    # Lower snap number wins (snap_009): T = 99 / 10 = 9.9
    assert np.allclose(T, 99.0 / 10.0, rtol=1e-10), (
        f"tie-break failed: T(k)={T[0]:.4f}, expected 9.9 (HR/snap_009)"
    )
    print("  ✓ equidistant tie → lower-snap-number wins (deterministic)")


def test_saved_npz_roundtrips_to_matched_z():
    """
    The per-sim convergence_ratios.npz must use z-labels that correspond
    to the *matched* z (i.e. the HR z, since HR is the anchor), not the
    LF z of the same-snap-name partner.
    """
    sim = "test_sim"
    lf_snaps = {f"snap_{i:03d}": (5.4 - 0.2 * (i - 4), float(i))
                for i in range(4, 10)}
    hr_snaps = {f"snap_{i:03d}": (5.4 - 0.2 * (i - 5), float(i))
                for i in range(5, 11)}

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_fake_tree(root, sim, lf_snaps, hr_snaps)
        cfg = PipelineConfig(output_root=str(root))
        compute_convergence_ratios(cfg, z_tol=0.05)

        npz_path = root / "hires" / sim / "convergence_ratios.npz"
        assert npz_path.exists(), "convergence_ratios.npz should be written"
        d = np.load(npz_path, allow_pickle=True)
        z_labels = sorted({k.split("__")[0] for k in d.files})
        z_values = sorted(float(lbl.replace("z", "").replace("p", ".")) for lbl in z_labels)

        # Expected matched z's are the HR z's that fall inside the LF z-range.
        # HR z spans {5.4, 5.2, ..., 4.4}; LF z spans {5.4, 5.2, ..., 4.4, 4.2}.
        # Every HR z is inside the LF range, so all 6 HR snaps match.
        expected = sorted(round(v[0], 3) for v in hr_snaps.values())
        assert z_values == expected, (
            f"saved z labels {z_values} != expected matched z {expected}"
        )
    print("  ✓ saved npz keys match the HR (anchor) z values")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("Running compute_convergence_ratios z-match tests:")
    tests = [
        test_exact_z_offset_matches_by_z_not_by_snap_name,
        test_hr_with_no_lf_within_tolerance_is_skipped,
        test_z_tol_controls_match_window,
        test_hr_without_done_sentinel_is_skipped,
        test_tie_break_picks_closest_then_lowest_snap,
        test_saved_npz_roundtrips_to_matched_z,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
