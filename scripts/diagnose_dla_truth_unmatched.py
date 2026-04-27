"""
Categorise the unmatched cases in the τ-peak vs colden-truth validation.

The aggregate validation (PR #6) reports completeness=0.865, purity=0.842
across 8340 truth DLAs.  This script walks the same 10 (sim, snap) pairs
and labels every unmatched truth and unmatched recovered DLA with a
specific failure mode, so we can see whether the misses are easy to fix.

Failure modes labelled
----------------------

For each unmatched **truth** DLA T at (skewer, pix_peak):
  T1  no τ system on the same sightline at all
  T2  nearest τ system exists; truth peak is outside its (span ± tol_pixels)
  T3  nearest τ system was already claimed by another truth (1-to-1 collision)

For each unmatched **recovered** DLA R at (skewer, span):
  R1  integrated colden over R's span is below 2 × 10^20  → a strong
      sub-DLA misclassified as a DLA by the τ-peak finder
  R2  integrated colden over R's span ≥ 2 × 10^20, but no truth DLA peak
      lies inside R's (span ± tol_pixels) — the truth-finder split a
      real DLA differently than the τ-finder
  R3  recovered system was the second match for a single truth (1-to-1
      collision: only one of multiple recovered candidates matches)
  R4  none of the above — true τ artefact (no colden DLA anywhere near)

Output
------
- prints per-(sim, snap) and aggregate counts per failure-mode category;
- writes a sample of representative cases (truth/recovered pairs with
  their τ + colden profiles) to figures/analysis/05_truth_validation/
  unmatched_examples.png;
- writes a markdown analysis to docs/dla_truth_unmatched_analysis.md.

Run::

    python3 scripts/diagnose_dla_truth_unmatched.py
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hcd_analysis.catalog import find_systems_in_skewer, nhi_from_tau_fast
from hcd_analysis.dla_truth import (
    RecoveredDLA,
    TruthDLA,
    find_truth_dlas_from_colden,
    match_dla_lists,
)

EMU_HIRES = Path("/nfs/turbo/lsa-cavestru/mfho/priya/emu_full_hires")
OUT_DIR = ROOT / "figures" / "analysis" / "05_truth_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOC_PATH = ROOT / "docs" / "dla_truth_unmatched_analysis.md"

DLA_THRESHOLD = 2.0e20            # cm^-2
LLS_THRESHOLD = 10 ** 17.2

# Production τ-finder params per config/default.yaml
TAU_THRESHOLD = 100.0
MERGE_DV_KMS = 100.0
MIN_PIXELS = 2
MIN_LOG_NHI = 17.2


# dv_kms is computed inline inside ``process_one`` from the H5 Header's
# ``Hz`` attribute and box geometry; no helper is needed here.


def classify_unmatched(
    truth, recovered, matched_result, tol_pixels: int,
    colden_row_view,
):
    """Return per-mode counts for one (sim, snap)."""
    t_modes = {"T1": 0, "T2": 0, "T3": 0}
    r_modes = {"R1": 0, "R2": 0, "R3": 0, "R4": 0}

    # Index recovered by skewer
    rec_by_row = {}
    for r in recovered:
        rec_by_row.setdefault(r.skewer_idx, []).append(r)

    # 1-to-1 collisions: detect whether each unmatched truth had a
    # candidate that was later claimed by another truth.
    used_centres = set()
    for mp in matched_result.matched:
        used_centres.add((mp.recovered.skewer_idx,
                          mp.recovered.pix_start,
                          mp.recovered.pix_end))

    truth_examples = []   # diagnostic samples per category
    rec_examples = []

    for t in matched_result.unmatched_truth:
        cands = rec_by_row.get(t.skewer_idx, [])
        if not cands:
            t_modes["T1"] += 1
            if len(truth_examples) < 4:
                truth_examples.append(("T1", t, None))
            continue
        # find nearest candidate by edge distance
        nearest = None
        nearest_d = 10**9
        for r in cands:
            if t.pix_peak < r.pix_start:
                d = r.pix_start - t.pix_peak
            elif t.pix_peak > r.pix_end:
                d = t.pix_peak - r.pix_end
            else:
                d = 0
            if d < nearest_d:
                nearest_d = d
                nearest = r
        # was nearest claimed by another truth?
        nearest_key = (nearest.skewer_idx, nearest.pix_start, nearest.pix_end)
        if nearest_key in used_centres and nearest_d <= tol_pixels:
            t_modes["T3"] += 1
            if len(truth_examples) < 12:
                truth_examples.append(("T3", t, nearest))
        else:
            t_modes["T2"] += 1
            if len(truth_examples) < 12:
                truth_examples.append(("T2", t, nearest))

    # For unmatched recovered systems: classify by integrated colden
    truth_by_row = {}
    for tt in truth:
        truth_by_row.setdefault(tt.skewer_idx, []).append(tt)

    matched_recovered_keys = used_centres

    for r in matched_result.unmatched_recovered:
        # Integrate colden over the recovered span
        row = colden_row_view(r.skewer_idx)
        nhi_in_span = float(row[r.pix_start : r.pix_end + 1].sum())
        if nhi_in_span < DLA_THRESHOLD:
            r_modes["R1"] += 1
            if len(rec_examples) < 8:
                rec_examples.append(("R1", r, nhi_in_span, None))
            continue
        # nhi_in_span ≥ DLA threshold; is there a truth peak in (span ± tol)?
        tt_candidates = truth_by_row.get(r.skewer_idx, [])
        truth_inside = [
            tt for tt in tt_candidates
            if r.pix_start - tol_pixels <= tt.pix_peak <= r.pix_end + tol_pixels
        ]
        if not truth_inside:
            r_modes["R2"] += 1
            if len(rec_examples) < 16:
                rec_examples.append(("R2", r, nhi_in_span, None))
            continue
        # truth peak(s) inside span — but we're unmatched; must be a 1-to-1 collision
        nearest_t = min(
            tt_candidates,
            key=lambda tt: abs((r.pix_start + r.pix_end) // 2 - tt.pix_peak),
        )
        r_modes["R3"] += 1
        if len(rec_examples) < 20:
            rec_examples.append(("R3", r, nhi_in_span, nearest_t))

    # R4: any unmatched recovered that isn't already in R1/R2/R3 — none
    # by construction above.

    return t_modes, r_modes, truth_examples, rec_examples


def process_one(h5path: Path):
    with h5py.File(h5path, "r") as f:
        box_kpch = float(f["Header"].attrs["box"])
        hubble = float(f["Header"].attrs["hubble"])
        z = float(f["Header"].attrs["redshift"])
        n_pix = int(f["Header"].attrs["nbins"])
        Hz = float(f["Header"].attrs["Hz"])      # km/s/Mpc
        # dv per pixel, comoving km/s
        dx_pix_mpc = (box_kpch / 1000.0) / n_pix / hubble * (1.0 / (1.0 + z))
        dv_kms = dx_pix_mpc * Hz
        tau = f["tau/H/1/1215"][:]
        colden = f["colden/H/1"][:]

    # Truth DLAs
    truth = find_truth_dlas_from_colden(
        colden, dla_threshold=DLA_THRESHOLD, pixel_floor=1.0e17,
        merge_gap_pixels=0, min_pixels=1,
    )
    # Recovered DLAs
    recovered: List[RecoveredDLA] = []
    n_skewers = tau.shape[0]
    merge_gap_pix = max(int(round(MERGE_DV_KMS / dv_kms)), 1)
    for sk in range(n_skewers):
        systems = find_systems_in_skewer(
            tau[sk],
            tau_threshold=TAU_THRESHOLD,
            merge_gap_pixels=merge_gap_pix,
            min_pixels=MIN_PIXELS,
        )
        for s, e in systems:
            nhi = nhi_from_tau_fast(tau[sk, s:e+1], dv_kms)
            if not np.isfinite(nhi) or nhi < 10 ** MIN_LOG_NHI:
                continue
            log_nhi = float(np.log10(nhi))
            if log_nhi < 17.2:
                continue
            elif log_nhi < 19.0:
                cls = "LLS"
            elif log_nhi < 20.3:
                cls = "subDLA"
            else:
                cls = "DLA"
            recovered.append(RecoveredDLA(
                skewer_idx=sk, pix_start=s, pix_end=e,
                NHI_recovered=float(nhi), log_NHI=log_nhi, absorber_class=cls,
            ))

    rec_dla_only = [r for r in recovered if r.absorber_class == "DLA"]
    rec_lls_or_stronger = [r for r in recovered if r.absorber_class in ("LLS", "subDLA", "DLA")]

    tol_pixels = max(int(round(MERGE_DV_KMS / dv_kms)), 5)

    # Match truth → recovered (LLS-or-stronger), like the H3 (loose) result
    match_loose = match_dla_lists(truth, rec_lls_or_stronger, tol_pixels=tol_pixels)
    # Also match truth → DLA-only (the strict completeness)
    match_strict = match_dla_lists(truth, rec_dla_only, tol_pixels=tol_pixels)
    # Purity match (recovered DLA → truth)
    # Purity is derived from match_strict.unmatched_recovered

    def colden_row(sk):
        return colden[sk]

    # Strict (DLA→DLA only) – this is what "purity_dla" uses
    t_modes_strict, r_modes_strict, t_ex_s, r_ex_s = classify_unmatched(
        truth, rec_dla_only, match_strict, tol_pixels, colden_row
    )
    # Loose (truth→LLS-or-stronger) – this is what "completeness_loose" uses
    t_modes_loose, _r_unused, t_ex_l, _r_unused2 = classify_unmatched(
        truth, rec_lls_or_stronger, match_loose, tol_pixels, colden_row
    )

    return {
        "n_truth": len(truth),
        "n_rec_dla": len(rec_dla_only),
        "n_rec_loose": len(rec_lls_or_stronger),
        "n_matched_strict": len(match_strict.matched),
        "n_matched_loose": len(match_loose.matched),
        "tol_pixels": tol_pixels,
        "merge_gap_pix": merge_gap_pix,
        "z": z,
        "hubble": hubble,
        "dv_kms": dv_kms,
        "t_modes_strict": t_modes_strict,
        "t_modes_loose": t_modes_loose,
        "r_modes": r_modes_strict,
        "examples_truth": t_ex_l,           # use loose for truth examples (smaller)
        "examples_rec": r_ex_s,
        "truth": truth,
        "recovered_dla": rec_dla_only,
        "tau": None,                        # don't keep around (large)
        "colden": None,
    }


def main():
    files = sorted(EMU_HIRES.glob("*/output/SPECTRA_*/rand_spectra_DLA.hdf5"))
    print(f"Found {len(files)} rand_spectra_DLA files")
    results = []
    for h5 in files:
        sim = h5.parent.parent.parent.name
        snap = h5.parent.name
        print(f"  processing {sim}/{snap}...", flush=True)
        r = process_one(h5)
        r["sim"] = sim
        r["snap"] = snap
        results.append(r)

    # Aggregate
    from collections import Counter
    agg_truth_strict = Counter()
    agg_truth_loose = Counter()
    agg_rec = Counter()
    n_truth_total = 0
    n_rec_total = 0
    n_matched_loose = 0
    n_matched_strict = 0
    n_rec_loose_total = 0
    for r in results:
        for k, v in r["t_modes_strict"].items(): agg_truth_strict[k] += v
        for k, v in r["t_modes_loose"].items(): agg_truth_loose[k] += v
        for k, v in r["r_modes"].items(): agg_rec[k] += v
        n_truth_total += r["n_truth"]
        n_rec_total += r["n_rec_dla"]
        n_rec_loose_total += r["n_rec_loose"]
        n_matched_loose += r["n_matched_loose"]
        n_matched_strict += r["n_matched_strict"]

    completeness_loose = n_matched_loose / max(n_truth_total, 1)
    purity_strict = n_matched_strict / max(n_rec_total, 1)

    summary = {
        "n_truth_total": n_truth_total,
        "n_rec_dla_total": n_rec_total,
        "n_rec_loose_total": n_rec_loose_total,
        "n_matched_loose": n_matched_loose,
        "n_matched_strict": n_matched_strict,
        "completeness_loose": completeness_loose,
        "purity_strict": purity_strict,
        "agg_truth_strict": dict(agg_truth_strict),
        "agg_truth_loose": dict(agg_truth_loose),
        "agg_rec": dict(agg_rec),
    }
    print(json.dumps(summary, indent=2))

    # Save per-(sim,snap) JSON for the doc to consume
    per_sim = []
    for r in results:
        per_sim.append({
            "sim": r["sim"], "snap": r["snap"], "z": r["z"],
            "n_truth": r["n_truth"], "n_rec_dla": r["n_rec_dla"],
            "n_rec_loose": r["n_rec_loose"],
            "n_matched_strict": r["n_matched_strict"],
            "n_matched_loose": r["n_matched_loose"],
            "tol_pixels": r["tol_pixels"], "dv_kms": r["dv_kms"],
            "t_modes_strict": r["t_modes_strict"],
            "t_modes_loose": r["t_modes_loose"],
            "r_modes": r["r_modes"],
        })

    out_json = OUT_DIR / "unmatched_diagnosis.json"
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "per_sim_snap": per_sim}, f, indent=2)
    print(f"\nWrote diagnostic JSON: {out_json}")
    return results, summary, per_sim


if __name__ == "__main__":
    main()
