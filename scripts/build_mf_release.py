#!/usr/bin/env python3
"""Build one MF record (format ``hcd_priya.mf_record/1``; hcd_analysis/emulator/mf_release.py) from explicit, digest-
pinned inputs, into a NEW directory.

  1. refuses an existing --out-dir, and a release build from a dirty tree, before any input is read;
  2. hashes the LF and HR caches and refuses any that differs from its pin (--lf-cache-sha256, --hr-cache-sha256);
  3. loads the LF backbone bundle anchored on --backbone-manifest-sha256 (a production bundle; with
     --backbone-purpose a research bundle, for a fold build only) and refuses one not trained on the pinned LF cache;
  4. reads the LF cache (strict, lineage-checked) and only the per-class P1D rows and row keys of the HR cache
     (mf_release.hr_view: the catalogue-derived HR columns are never read);
  5. matches HR and LF rows exactly, checks the mode mapping (``d1_check``: canonical <= 1e-12 on every pair; stored
     grids <= 1e-12 except the one recorded design point, mf_release.D1_EXCEPTION, below its bound; recorded in the
     record's ``mode_mapping``), forms the targets g = log P_HR - log P_LF,backbone per matched pair, class and mode;
  6. fits the tables on the composition's training pairs (minus a fold's held-out simulations) and writes the record,
     with the lower edge of its validation domain (--validated-z-min, mf_release.VALIDATED_Z_MIN = 2.2 by default;
     never lower; the domain edge, not a passed validation);
  7. reloads it through mf_release.load_mf_record and prints the record's sha256 (the anchor to publish).

Release:
  python scripts/build_mf_release.py --kind release --mf-version V --out-dir NEW \\
      --lf-cache LF.h5 --lf-cache-sha256 H --hr-cache HR.h5 --hr-cache-sha256 H \\
      --backbone BUNDLE --backbone-manifest-sha256 H \\
      --composition-id ID --exclude-sim NAME [--exclude-sim ...] --ruling-id R [--ruling-id ...]
Fold (held-out fit for grouped cross-validation): add --kind fold --fold-id F --holdout-sim NAME [...], and
--backbone-purpose TEXT when the backbone is a research bundle (for example one that excludes the held-out LF partner).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from hcd_analysis.emulator.mf_release import (
    D1_TOL,
    VALIDATED_Z_MIN,
)  # noqa: E402  (shared with the record check)

HR_HISTORICAL_REASON = (
    "the HR cache predates cache lineage; the MF reads only its per-class P1D rows and row keys "
    "(PI ruling 2026-11-06b A1)"
)


def code_identity(repo=_REPO_ROOT):
    """{commit, dirty} of the checkout this script runs from."""
    commit = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    st = subprocess.run(
        ["git", "-C", repo, "status", "--porcelain"], capture_output=True, text=True
    )
    if not commit or st.returncode != 0:
        raise SystemExit("refusing: the code identity (git commit and status) cannot be read")
    return {"commit": commit, "dirty": bool(st.stdout.strip())}


def parse(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--kind", required=True, choices=("release", "fold"))
    ap.add_argument("--mf-version", required=True)
    ap.add_argument("--out-dir", required=True, help="a NEW directory (an existing one is refused)")
    ap.add_argument("--lf-cache", required=True)
    ap.add_argument("--lf-cache-sha256", required=True)
    ap.add_argument("--hr-cache", required=True)
    ap.add_argument("--hr-cache-sha256", required=True)
    ap.add_argument("--backbone", required=True, help="the LF backbone bundle directory")
    ap.add_argument("--backbone-manifest-sha256", required=True)
    ap.add_argument(
        "--backbone-purpose",
        default=None,
        help="the backbone is a research bundle (fold builds only)",
    )
    ap.add_argument("--composition-id", required=True)
    ap.add_argument("--exclude-sim", action="append", default=[])
    ap.add_argument("--ruling-id", action="append", default=[], required=True)
    ap.add_argument("--fold-id", default=None)
    ap.add_argument("--holdout-sim", action="append", default=[])
    ap.add_argument(
        "--validated-z-min",
        type=float,
        default=VALIDATED_Z_MIN,
        help=f"the lower edge of the redshift range over which the MF is validated (the frozen "
        f"validation domain's edge, not a passed validation; default {VALIDATED_Z_MIN}; it may be "
        "raised, never lowered); the predictor refuses LF+MF below it",
    )
    a = ap.parse_args(argv)
    if a.kind == "release" and (a.fold_id or a.holdout_sim or a.backbone_purpose):
        ap.error("a release build takes no --fold-id, --holdout-sim or --backbone-purpose")
    if a.kind == "fold" and not (a.fold_id and a.holdout_sim):
        ap.error("a fold build needs --fold-id and at least one --holdout-sim")
    return a


def d1_check(lf, hrv, pairs, k_com):
    """The HR/LF mode mapping (D1) over ALL matched pairs, before any composition: the canonical mapping must hold to
    D1_TOL on every pair, and the stored grids too, except the pairs of the one recorded design point
    (mf_release.D1_EXCEPTION: its LF and HR run names both, each pair below its bound). Returns the record's
    ``mode_mapping`` block; refuses (SystemExit) anything else."""
    from hcd_analysis.emulator import mf_modes as MM
    from hcd_analysis.emulator import mf_release as MR

    K = len(k_com)
    canonical = float(MM.canonical_mapping_error(hrv, pairs, k_com))
    if not canonical <= D1_TOL:
        raise SystemExit(
            f"refusing: HR and LF modes do not map one to one (D1 canonical {canonical!r} > {D1_TOL!r})"
        )
    exc = MR.D1_EXCEPTION
    stored, excepted, bad = 0.0, [], []
    for h, l in pairs:
        e = float(MM.mode_mapping_error(lf, hrv, [(h, l)], K))
        if e <= D1_TOL:
            stored = max(stored, e)
        elif (
            str(lf["sim_name"][l]) == exc["lf_sim"]
            and str(hrv["sim_name"][h]) == exc["hr_sim"]
            and e < exc["bound"]
        ):
            excepted.append(e)
        else:
            bad.append((str(lf["sim_name"][l]), e))
    if bad:
        sims = sorted({s for s, _ in bad})
        raise SystemExit(
            f"refusing: HR and LF modes do not map one to one (D1 stored grids: {len(bad)} pairs of "
            f"{sims} above {D1_TOL!r} outside the recorded exception; largest {max(e for _, e in bad)!r})"
        )
    out = {"tol": D1_TOL, "canonical_max": canonical, "stored_max": stored, "exception": None}
    if excepted:
        out["exception"] = dict(MR.copy_json(exc), n_pairs=len(excepted), max_error=max(excepted))
    return MR.check_mode_mapping(out)


def build_from_inputs(
    out_dir,
    *,
    lf,
    hr,
    backbone,
    kind,
    mf_version,
    caches,
    composition,
    fold,
    fit_code,
    validated_z_min=VALIDATED_Z_MIN,
):
    """Steps 5 and 6 on loaded inputs: ``lf`` the LF cache dict, ``hr`` the HR cache dict (seen through
    ``hr_view`` here), ``backbone`` a loaded predictor (``predictor.Bundle`` or ``ResearchPredictor``). Returns
    (record path, record sha256, summary)."""
    import numpy as np
    from hcd_analysis.emulator import mf_modes as MM
    from hcd_analysis.emulator import mf_release as MR

    hrv = MR.hr_view(hr)
    import hcd_analysis.predictor as hp

    if not isinstance(backbone, hp._PredictorBase):
        raise SystemExit(
            "refusing: the backbone must be a loaded predictor (one object forms the targets)"
        )
    k_com = np.asarray(hp.native_modes(backbone), dtype=np.float64)
    K = k_com.size
    if not np.allclose(np.asarray(lf["k_com_hmpc"], float)[:K], k_com, rtol=1e-12, atol=0):
        raise SystemExit("refusing: the LF cache's box modes are not the backbone's")
    pairs = MM.match_pairs(lf, hrv)
    if not pairs:
        raise SystemExit("refusing: no HR row matches an LF row")
    d1 = d1_check(lf, hrv, pairs, k_com)
    l_rows = np.array([p[1] for p in pairs])
    P_lf = hp.lf_P_filt_at(backbone, np.asarray(lf["x"])[l_rows], np.asarray(lf["tau0"])[l_rows])
    t = MM.measure_mode_targets(lf, hrv, pairs, P_lf, x=lf["x"], n_modes=K)
    rows, train = MR.select_train_rows(
        t,
        exclude_sims=composition["excluded_sims"],
        holdout_sims=(fold or {}).get("holdout_sims", ()),
    )
    tables = MM.fit_mode_mf(t, train_rows=rows)
    bb = hp.backbone_identity(backbone)  # derived from the member metas, never a caller flag
    path, sha = MR.write_mf_record(
        out_dir,
        tables,
        k_com_hmpc=k_com,
        kind=kind,
        mf_version=mf_version,
        backbone=bb,
        caches=caches,
        composition=composition,
        fold=fold,
        fit_code=fit_code,
        train_sims=train,
        n_pairs_train=int(rows.size),
        mode_mapping=d1,
        validated_z_min=float(validated_z_min),
    )
    summary = {
        "n_pairs": len(pairs),
        "n_pairs_train": int(rows.size),
        "train_sims": train,
        "D1": d1,
        "validated_z_min": float(validated_z_min),
        "sims": sorted(set(t["sim"])),
    }
    return path, sha, summary


def main(argv=None, *, _code=None):
    a = parse(argv)
    out = os.path.abspath(a.out_dir)
    if os.path.lexists(out):  # write once: before any input is read
        sys.exit(
            f"refusing: --out-dir {out} exists (an MF record is written once into a new directory)"
        )
    code = (_code or code_identity)()
    if a.kind == "release" and code["dirty"]:
        sys.exit("refusing: a release MF is built from a clean tree (git status is not empty)")
    from hcd_analysis import lineage as L

    pins = {"LF": (a.lf_cache, a.lf_cache_sha256), "HR": (a.hr_cache, a.hr_cache_sha256)}
    for side, (p, want) in pins.items():
        got = L.sha256_file(p)
        if got != want:
            sys.exit(f"refusing: the {side} cache {p} has sha256 {got}, not the pinned {want}")
    import hcd_analysis.emulator  # noqa: F401  (float64)
    import hcd_analysis.predictor as hp
    from hcd_analysis.emulator import data as D
    from hcd_analysis.emulator import safe_h5 as H

    if a.backbone_purpose:
        backbone = hp.ResearchPredictor(
            H.load_research(
                a.backbone,
                expected_manifest_sha256=a.backbone_manifest_sha256,
                purpose=a.backbone_purpose,
            )
        )
    else:
        backbone = hp.Bundle(
            H.load_release(a.backbone, expected_manifest_sha256=a.backbone_manifest_sha256)
        )
    if (backbone.lineage.get("cache") or {}).get("sha256") != a.lf_cache_sha256:
        sys.exit("refusing: the backbone was not trained on the pinned LF cache")
    lf = D.load_cache(a.lf_cache)
    if lf["cache_sha256"] != a.lf_cache_sha256:
        sys.exit("refusing: the LF cache changed after it was hashed")
    hr = D.load_cache_historical(a.hr_cache, reason=HR_HISTORICAL_REASON)
    if L.sha256_file(a.hr_cache) != a.hr_cache_sha256:
        sys.exit("refusing: the HR cache changed while it was read")
    caches = {
        "lf": {"file": os.path.basename(a.lf_cache), "sha256": a.lf_cache_sha256},
        "hr": {"file": os.path.basename(a.hr_cache), "sha256": a.hr_cache_sha256},
    }
    composition = {
        "id": a.composition_id,
        "excluded_sims": sorted(set(a.exclude_sim)),
        "ruling_ids": a.ruling_id,
    }
    fold = (
        {"fold_id": a.fold_id, "holdout_sims": sorted(set(a.holdout_sim))}
        if a.kind == "fold"
        else None
    )
    path, sha, summary = build_from_inputs(
        out,
        lf=lf,
        hr=hr,
        backbone=backbone,
        kind=a.kind,
        mf_version=a.mf_version,
        caches=caches,
        composition=composition,
        fold=fold,
        fit_code=code,
        validated_z_min=a.validated_z_min,
    )
    from hcd_analysis.emulator import mf_release as MR

    rec = MR.load_mf_record(
        out,
        expected_record_sha256=sha,
        backbone_manifest_sha256=a.backbone_manifest_sha256,
        kind=a.kind,
        composition=composition,
        lf_cache_sha256=a.lf_cache_sha256,
        hr_cache_sha256=a.hr_cache_sha256,
    )
    print(json.dumps(dict(summary, record=path, record_sha256=sha, kind=rec.kind), indent=1))


if __name__ == "__main__":
    main()
