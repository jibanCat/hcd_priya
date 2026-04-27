"""
Hypothesis test gate for the τ-peak DLA finder vs particle-based truth.

This is a one-off validation script (NOT part of the auto-run unit-test
loop — it walks the HiRes tree and is too heavy / network-bound for that
purpose).  Follow the pattern of `tests/sanity_run_one_snap.py`.

Reads the aggregated summary written by `scripts/validate_dla_truth.py`
(`figures/analysis/data/dla_truth_summary.h5`) and tests four hypotheses:

  H1 — no systematic bias:       |⟨Δlog NHI⟩| ≤ 0.05 dex (95% CI on the mean)
  H2 — random scatter bounded:   σ(Δlog NHI) < 0.15 dex
  H3 — completeness:             ≥ 80 % truth DLAs matched by an LLS-or-stronger system
  H4 — purity:                   ≥ 70 % recovered DLAs match a truth DLA within tol

Each hypothesis prints PASS/FAIL with the recovered numbers; failure does
NOT crash the test (the script is for review). A markdown-formatted summary
is printed at the end.

Usage
-----
    python3 tests/validate_dla_truth_hypothesis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SUMMARY_H5 = REPO_ROOT / "figures" / "analysis" / "data" / "dla_truth_summary.h5"


# ---------------------------------------------------------------------------
# Thresholds — bake these into the gate. Move to a config later if needed.
# ---------------------------------------------------------------------------

H1_TOL_DEX = 0.05      # |mean Δlog NHI| ≤ this within 95% CI
H2_SIGMA_DEX = 0.15    # σ Δlog NHI strictly less than this
H3_COMPLETENESS = 0.80
H4_PURITY = 0.70


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_delta(h5_path: Path) -> tuple[np.ndarray, dict]:
    """Read the validation summary HDF5 and concatenate per-(sim, snap)
    Δlog NHI arrays.

    Returns
    -------
    all_dlog : np.ndarray
        1-D concatenation of every matched-pair Δlog NHI across all files.
    per_file : dict[str, list]
        Per-(sim, snap) metadata used for the markdown report — keys are
        ``sim``, ``snap``, ``z``, ``n_truth``, ``n_recovered_dla``,
        ``n_recovered_lls_or_stronger``, ``n_matched_dla``, ``n_matched_loose``.
        Each is a list aligned with the file index.
    """
    all_dlog: list[np.ndarray] = []
    per_file = {
        "sim": [], "snap": [], "z": [],
        "n_truth": [], "n_recovered_dla": [], "n_recovered_lls_or_stronger": [],
        "n_matched_dla": [], "n_matched_loose": [],
    }
    with h5py.File(h5_path, "r") as f:
        s = f["summary"]
        n = len(s["sim_name"])
        sim_names = [bn.decode() for bn in s["sim_name"][:]]
        snaps = s["snap"][:]
        zs = s["z"][:]
        per_file["sim"] = sim_names
        per_file["snap"] = list(snaps)
        per_file["z"] = list(zs)
        per_file["n_truth"] = list(s["n_truth"][:])
        per_file["n_recovered_dla"] = list(s["n_recovered_dla"][:])
        per_file["n_recovered_lls_or_stronger"] = list(s["n_recovered_lls_or_stronger"][:])
        per_file["n_matched_dla"] = list(s["n_matched_dla"][:])
        per_file["n_matched_loose"] = list(s["n_matched_loose"][:])

        for sim, snap in zip(sim_names, snaps):
            key = f"{sim}__snap_{int(snap):03d}"
            grp = f["scatter"][key]
            all_dlog.append(grp["delta_log_nhi"][:])

    return np.concatenate(all_dlog) if all_dlog else np.array([]), per_file


def _ci95_of_mean(x: np.ndarray) -> tuple[float, float]:
    """(mean, half-width of 95% CI on the mean assuming normal-approx)."""
    if x.size == 0:
        return float("nan"), float("nan")
    m = float(np.mean(x))
    sem = float(np.std(x, ddof=1)) / np.sqrt(x.size) if x.size > 1 else 0.0
    return m, 1.96 * sem


# ---------------------------------------------------------------------------
# Hypothesis tests — each returns (passed: bool, line: str)
# ---------------------------------------------------------------------------

def hypothesis_1(dlog: np.ndarray) -> tuple[bool, str]:
    m, hw = _ci95_of_mean(dlog)
    # Pass if |m| + hw ≤ tol — i.e. the 95% CI on the mean is fully inside [-tol, tol].
    passed = abs(m) + hw <= H1_TOL_DEX
    label = "PASS" if passed else "FAIL"
    return passed, (
        f"H1 (bias): ⟨Δlog NHI⟩ = {m:+.4f} ± {hw:.4f} dex (95% CI), "
        f"tolerance ±{H1_TOL_DEX} dex → {label}"
    )


def hypothesis_2(dlog: np.ndarray) -> tuple[bool, str]:
    sigma = float(np.std(dlog, ddof=1)) if dlog.size > 1 else 0.0
    passed = sigma < H2_SIGMA_DEX
    label = "PASS" if passed else "FAIL"
    return passed, (
        f"H2 (scatter): σ(Δlog NHI) = {sigma:.4f} dex, threshold < {H2_SIGMA_DEX} dex → {label}"
    )


def hypothesis_3(per_file: dict) -> tuple[bool, str]:
    n_truth = sum(per_file["n_truth"])
    n_matched_loose = sum(per_file["n_matched_loose"])
    comp = n_matched_loose / max(n_truth, 1)
    passed = comp >= H3_COMPLETENESS
    label = "PASS" if passed else "FAIL"
    return passed, (
        f"H3 (completeness): {n_matched_loose}/{n_truth} = {comp:.3f} truth DLAs matched "
        f"by an LLS-or-stronger recovered system, threshold ≥ {H3_COMPLETENESS:.2f} → {label}"
    )


def hypothesis_4(per_file: dict) -> tuple[bool, str]:
    """
    Purity = N_matched_DLA / N_recovered_DLA. We compare DLA-class recovered
    against truth DLAs (the canonical "of all my DLAs, how many are real DLAs?").
    """
    n_rec = sum(per_file["n_recovered_dla"])
    n_match = sum(per_file["n_matched_dla"])
    purity = n_match / max(n_rec, 1)
    passed = purity >= H4_PURITY
    label = "PASS" if passed else "FAIL"
    return passed, (
        f"H4 (purity): {n_match}/{n_rec} = {purity:.3f} recovered DLAs match a truth "
        f"DLA, threshold ≥ {H4_PURITY:.2f} → {label}"
    )


# ---------------------------------------------------------------------------
# Pretty output
# ---------------------------------------------------------------------------

def _per_z_breakdown(per_file: dict, dlog_per_file: list[np.ndarray] | None = None):
    """
    Group by z (rounded to 0.1) and report aggregate completeness, purity,
    and Δlog NHI mean / σ. Returns a list of rows for the markdown summary.
    """
    z_arr = np.array(per_file["z"])
    # round-to-nearest 0.1 so z=2.99999 and z=3.0 collapse together
    z_keys = np.round(z_arr, 1)
    rows = []
    for zk in sorted(set(z_keys)):
        sel = (z_keys == zk)
        n_truth = int(np.sum(np.array(per_file["n_truth"])[sel]))
        n_rec = int(np.sum(np.array(per_file["n_recovered_dla"])[sel]))
        n_match_dla = int(np.sum(np.array(per_file["n_matched_dla"])[sel]))
        n_match_loose = int(np.sum(np.array(per_file["n_matched_loose"])[sel]))
        comp = n_match_loose / max(n_truth, 1)
        purity = n_match_dla / max(n_rec, 1)
        if dlog_per_file is not None:
            dlogs = np.concatenate([
                d for d, s in zip(dlog_per_file, sel) if s and d.size
            ]) if any(sel) else np.array([])
        else:
            dlogs = np.array([])
        if dlogs.size:
            m = float(np.mean(dlogs)); s = float(np.std(dlogs, ddof=1))
        else:
            m, s = float("nan"), float("nan")
        rows.append({
            "z": float(zk),
            "n_truth": n_truth,
            "n_rec_dla": n_rec,
            "n_matched_dla": n_match_dla,
            "completeness": comp,
            "purity": purity,
            "mean_dlog": m,
            "sigma_dlog": s,
        })
    return rows


def _load_per_file_dlog(h5_path: Path, per_file: dict) -> list[np.ndarray]:
    out = []
    with h5py.File(h5_path, "r") as f:
        for sim, snap in zip(per_file["sim"], per_file["snap"]):
            key = f"{sim}__snap_{int(snap):03d}"
            out.append(f["scatter"][key]["delta_log_nhi"][:])
    return out


def main():
    if not SUMMARY_H5.exists():
        print(f"Summary HDF5 not found at {SUMMARY_H5}.")
        print("Run scripts/validate_dla_truth.py first.")
        return 1

    dlog, per_file = _aggregate_delta(SUMMARY_H5)
    if dlog.size == 0:
        print("No matched-pair Δlog NHI values found in summary.h5; nothing to test.")
        return 1

    print(f"Loaded {dlog.size} matched (truth, recovered) DLA pairs from "
          f"{len(per_file['sim'])} (sim, snap) files\n")

    p1, l1 = hypothesis_1(dlog)
    p2, l2 = hypothesis_2(dlog)
    p3, l3 = hypothesis_3(per_file)
    p4, l4 = hypothesis_4(per_file)

    print(l1)
    print(l2)
    print(l3)
    print(l4)

    # Markdown summary
    print()
    print("---")
    print()
    print("# DLA truth-validation hypothesis test summary")
    print()
    print(f"Source: `{SUMMARY_H5.relative_to(REPO_ROOT)}`")
    print(f"Files aggregated: {len(per_file['sim'])} "
          f"(sim, snap) pairs across {len(set(per_file['sim']))} sims.")
    print(f"Total truth DLAs: {sum(per_file['n_truth'])}; "
          f"recovered DLA-class systems: {sum(per_file['n_recovered_dla'])}; "
          f"matched DLA-vs-DLA pairs: {dlog.size}.")
    print()
    print("| Hypothesis | Statistic | Threshold | Recovered | Verdict |")
    print("|---|---|---|---|---|")
    m, hw = _ci95_of_mean(dlog)
    print(f"| H1 (no bias) | mean Δlog NHI | ≤ {H1_TOL_DEX} dex (95 % CI) | "
          f"{m:+.4f} ± {hw:.4f} | {'PASS' if p1 else 'FAIL'} |")
    sigma = float(np.std(dlog, ddof=1))
    print(f"| H2 (scatter) | σ Δlog NHI    | < {H2_SIGMA_DEX} dex | {sigma:.4f} | "
          f"{'PASS' if p2 else 'FAIL'} |")
    n_truth = sum(per_file["n_truth"])
    n_match_loose = sum(per_file["n_matched_loose"])
    print(f"| H3 (complete) | matched / truth | ≥ {H3_COMPLETENESS:.2f} | "
          f"{n_match_loose}/{n_truth} = {n_match_loose / max(n_truth, 1):.3f} | "
          f"{'PASS' if p3 else 'FAIL'} |")
    n_rec = sum(per_file["n_recovered_dla"])
    n_match_dla = sum(per_file["n_matched_dla"])
    print(f"| H4 (purity)   | matched / recovered DLA | ≥ {H4_PURITY:.2f} | "
          f"{n_match_dla}/{n_rec} = {n_match_dla / max(n_rec, 1):.3f} | "
          f"{'PASS' if p4 else 'FAIL'} |")
    print()
    print("## Per-z breakdown")
    print()
    print("| z | N_truth | N_rec(DLA) | matched | completeness | purity | "
          "⟨Δlog NHI⟩ | σ |")
    print("|---|---|---|---|---|---|---|---|")
    dlog_pf = _load_per_file_dlog(SUMMARY_H5, per_file)
    for r in _per_z_breakdown(per_file, dlog_pf):
        print(f"| {r['z']:.1f} | {r['n_truth']} | {r['n_rec_dla']} | "
              f"{r['n_matched_dla']} | {r['completeness']:.3f} | "
              f"{r['purity']:.3f} | {r['mean_dlog']:+.4f} | "
              f"{r['sigma_dlog']:.4f} |")
    print()
    if all([p1, p2, p3, p4]):
        print("**All four hypotheses pass.** The τ-peak DLA finder reproduces "
              "particle-based truth NHI within the published tolerance and "
              "matches ≥ 80 % truth completeness with ≥ 70 % purity.")
    else:
        failing = [name for name, p in
                   zip(("H1", "H2", "H3", "H4"), (p1, p2, p3, p4)) if not p]
        print(f"**Failing hypotheses: {', '.join(failing)}.** Review per-z "
              "breakdown above; diagnose either the τ → NHI conversion "
              "(H1/H2), the τ-peak finder coverage (H3), or false-positive "
              "rate (H4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
