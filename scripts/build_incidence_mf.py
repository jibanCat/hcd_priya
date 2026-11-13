#!/usr/bin/env python3
"""Build one incidence MF record (format ``hcd_priya.incidence_mf_record/1``; hcd_analysis/emulator/incidence_mf.py)
from explicit, digest-pinned inputs, into a NEW directory.

The record holds the fixed table R_c(z) = mean over the matched HR simulations of (dN/dX)_HR / (dN/dX)_LF per HCD
class and redshift, and the same kind of table per column-density bin for the CDDF. The tables are computed by the
functions of hcd_analysis/emulator/multifidelity.py (build_dndx_res_corr, build_cddf_res_corr); this script adds no
estimator.

  1. refuses an existing --out-dir, and a dirty tree, before any input is read;
  2. hashes the LF cache and the HR incidence product and refuses any that differs from its pin;
  3. reads the LF cache's snapshot columns (multifidelity.load_snap_cache) and the HR incidence product
     (incidence_mf.hr_snap_from_product: an .npz of plain arrays derived from the rebuilt HR catalogues; the HR
     cache is never read);
  4. leaves out the simulations the composition excludes (--exclude-sim), pairs every remaining HR snapshot with its
     LF partner on parameters and redshift (multifidelity.match_snap_hr_to_lf) and refuses an HR snapshot without a
     partner;
  5. fits the two tables, writes the record, reloads it and prints its sha256 (the anchor to publish).

Nothing about the composition has a default. Which HR simulations make the table is stated on the command line:
either --exclude-sim NAME (repeatable) or --exclude-none. So is what the HR incidence product does not hold: either
--group-not-in-product TEXT (repeatable; recorded as given, not compared with the product) or --no-group-missing.

  python scripts/build_incidence_mf.py --version V --out-dir NEW \\
      --lf-cache LF.h5 --lf-cache-sha256 H --hr-incidence HR.npz --hr-incidence-sha256 H \\
      --composition-id ID (--exclude-sim NAME ... | --exclude-none) --ruling-id R [--ruling-id ...] \\
      (--group-not-in-product TEXT ... | --no-group-missing) \\
      (--heldout-error TABLE.json SHA256 [--heldout-error-other-composition] | --no-heldout-error)

The held-out error of the dN/dX and the CDDF is stated in the same way: either a table to copy into the record
(--heldout-error FILE SHA256; the numbers are copied, this script computes none of them; a table measured for
another number of HR simulations than the record uses is refused unless --heldout-error-other-composition says the
difference is intended), or --no-heldout-error, after which predictions of the record warn that they have none.
"""

from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

# the code identity ({commit, dirty} of this checkout) is the P1D MF builder's function, not a second one
_spec = importlib.util.spec_from_file_location(
    "build_mf_release", os.path.join(_REPO_ROOT, "scripts", "build_mf_release.py")
)
_BMR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_BMR)
code_identity = _BMR.code_identity


def parse(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--version", required=True, help="the record's version text")
    ap.add_argument("--out-dir", required=True, help="a NEW directory (an existing one is refused)")
    ap.add_argument("--lf-cache", required=True, help="the LF training cache (.h5)")
    ap.add_argument("--lf-cache-sha256", required=True)
    ap.add_argument("--hr-incidence", required=True, help="the HR incidence product (.npz)")
    ap.add_argument("--hr-incidence-sha256", required=True)
    ap.add_argument("--composition-id", required=True)
    ap.add_argument("--exclude-sim", action="append", default=[], help="an HR simulation left out")
    ap.add_argument(
        "--exclude-none", action="store_true", help="say that every HR simulation is used"
    )
    ap.add_argument("--ruling-id", action="append", default=[], required=True)
    ap.add_argument(
        "--group-not-in-product",
        action="append",
        default=[],
        help="an HR snapshot group that the HR incidence product does not hold (recorded as given)",
    )
    ap.add_argument(
        "--no-group-missing",
        action="store_true",
        help="say that the HR incidence product holds every HR snapshot group",
    )
    ap.add_argument(
        "--heldout-error",
        nargs=2,
        metavar=("FILE", "SHA256"),
        help="a held-out error table (JSON) to copy into the record, with its sha256",
    )
    ap.add_argument(
        "--no-heldout-error",
        action="store_true",
        help="say that the record carries no held-out error (its predictions then warn that they have none)",
    )
    ap.add_argument(
        "--heldout-error-other-composition",
        action="store_true",
        help="accept a held-out error measured for another number of HR simulations than this record uses",
    )
    a = ap.parse_args(argv)
    if (a.heldout_error is not None) == a.no_heldout_error:
        ap.error(
            "state the held-out error: give --heldout-error FILE SHA256, or --no-heldout-error"
        )
    if a.heldout_error_other_composition and a.heldout_error is None:
        ap.error("--heldout-error-other-composition needs --heldout-error")
    if bool(a.exclude_sim) == a.exclude_none:
        ap.error("state the composition: give --exclude-sim NAME (repeatable), or --exclude-none")
    if bool(a.group_not_in_product) == a.no_group_missing:
        ap.error(
            "state what the HR product lacks: give --group-not-in-product TEXT (repeatable), or "
            "--no-group-missing"
        )
    if len(set(a.group_not_in_product)) != len(a.group_not_in_product):
        ap.error("--group-not-in-product names a group twice")
    return a


def _no_constant(name):
    raise ValueError(f"{name} is not a number of strict JSON")


def main(argv=None, *, _code=None):
    a = parse(argv)
    out = os.path.abspath(a.out_dir)
    if os.path.lexists(out):  # write once: before any input is read
        sys.exit(
            f"refusing: --out-dir {out} exists (a record is written once into a new directory)"
        )
    code = (_code or code_identity)()
    if code["dirty"]:
        sys.exit(
            "refusing: an incidence MF record is built from a clean tree (git status is not empty)"
        )
    from hcd_analysis import lineage as L

    pins = {
        "LF cache": (a.lf_cache, a.lf_cache_sha256),
        "HR incidence product": (a.hr_incidence, a.hr_incidence_sha256),
    }
    if a.heldout_error is not None:
        pins["held-out error table"] = tuple(a.heldout_error)
    for what, (p, want) in pins.items():
        got = L.sha256_file(p)
        if got != want:
            sys.exit(f"refusing: the {what} {p} has sha256 {got}, not the pinned {want}")
    import hcd_analysis.emulator  # noqa: F401  (float64)
    from hcd_analysis.emulator import incidence_mf as IM
    from hcd_analysis.emulator import multifidelity as MF

    composition = {
        "id": a.composition_id,
        "excluded_sims": sorted(set(a.exclude_sim)),
        "ruling_ids": a.ruling_id,
    }
    try:
        lf = MF.load_snap_cache(a.lf_cache)
        hr = IM.hr_snap_from_product(a.hr_incidence)
        for what, (p, want) in pins.items():
            if L.sha256_file(p) != want:
                sys.exit(f"refusing: the {what} changed while it was read")
        tables, train, n_pairs = IM.fit_tables(lf, hr, exclude_sims=composition["excluded_sims"])
        heldout = None
        if a.heldout_error is not None:  # copied, never computed here
            with open(a.heldout_error[0], "rb") as fh:
                raw = fh.read()
            if hashlib.sha256(raw).hexdigest() != a.heldout_error[1]:
                sys.exit("refusing: the held-out error table changed while it was read")
            try:
                doc = json.loads(raw.decode("utf-8"), parse_constant=_no_constant)
            except ValueError as e:
                sys.exit(f"refusing: the held-out error table is not strict JSON ({e})")
            heldout = IM.heldout_block_from_table(
                doc,
                log_nhi_centres=tables["log_nhi_centres"],
                n_train_sims=len(train),
                other_composition=a.heldout_error_other_composition,
            )
        path, sha = IM.write_record(
            out,
            tables,
            version=a.version,
            lf_cache={"file": os.path.basename(a.lf_cache), "sha256": a.lf_cache_sha256},
            hr_incidence={
                "file": os.path.basename(a.hr_incidence),
                "sha256": a.hr_incidence_sha256,
            },
            groups_not_in_product=a.group_not_in_product,
            composition=composition,
            train_sims=train,
            n_pairs=n_pairs,
            fit_code=code,
            heldout_error=heldout,
        )
    except (
        IM.MFRecordError,
        OSError,
        KeyError,
    ) as e:  # a refusal of the fit, or an input that is not readable
        sys.exit(f"refusing: {e}")
    inc = IM.load_record(out, expected_record_sha256=sha, lf_cache_sha256=a.lf_cache_sha256)
    summary = {
        "record": path,
        "record_sha256": sha,
        "composition": composition,
        "train_sims": train,
        "n_pairs": n_pairs,
        "z": [float(v) for v in inc.z],
        "n_sims": [int(v) for v in inc.n_sims],
        "heldout_error": (
            None
            if inc.heldout is None
            else {"kind": inc.record["heldout_error"]["kind"], "z": inc.heldout["z"].tolist()}
        ),
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
