"""Check a model release from a fresh install: load it, predict, and compare with reference predictions.

Run it in an environment that has only the installed package (``-I`` keeps the current directory and the user's
site-packages out of the import path, so a source tree lying around cannot stand in for the installation)::

    python -I -m hcd_analysis.check_release RELEASE_DIR RELEASE_SHA256
    python -I -m hcd_analysis.check_release RELEASE_DIR RELEASE_SHA256 --reference reference_predictions.json
    python -I -m hcd_analysis.check_release --published                 # the LF+MF model of the installed model package

What it does, in order: reports where ``hcd_analysis`` was imported from; loads the release (``load_release``, or
``load_published`` with ``--published``, which takes the files from the installed ``hcd-priya-models``), which
verifies every file against the published digests; predicts the
per-class P1D at the centre of the parameter box for each requested redshift, at each fidelity the model serves
there; mixes in the HCD classes; and, with ``--reference``, compares the numbers with reference predictions published beside the release.
It prints one line per step. It exits 0 when everything passes (the last line is PASSED; with a synthetic test
build of the model package it is "PASSED (synthetic build: ...)", so that such a run is never read as the check of a
model), 1 when a step fails
(a line starting with FAILED says why), and 2 for a wrong command line.

The reference file is made once by whoever publishes the release, with ``--write-reference``, on the supported
environment. It holds, for each case, the wavenumbers, the four class spectra, the HCD fractions and dN/dX
(format ``hcd_priya.reference_predictions/1``). A comparison passes when every number agrees to ``--rtol``
(default 1e-6: different processors and library versions change the last digits, not more).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence

import numpy as np

FORMAT = "hcd_priya.reference_predictions/1"
DEFAULT_Z = (2.2, 3.0, 4.6)
ARRAYS = ("k", "P_class", "w_c", "dndx")


def predictions(emu: Any, zs: Sequence[float]) -> list[dict[str, Any]]:
    """The cases compared: the box centre at each redshift, at every fidelity the model serves there."""
    import hcd_analysis.predictor as hp

    params = hp.example_params(verbose=False)
    print(
        "parameters: the midpoint of the training range of each parameter (not a fiducial cosmology): "
        + ", ".join(f"{n} = {v:g}" for n, v in params.items())
    )
    cases = []
    for z in zs:
        for fidelity in hp.FIDELITIES:
            try:
                p = emu.predict(params, z=z, tau0_amp=1.0, dtau0=0.0, fidelity=fidelity)
            except hp.MFUnavailableError:
                continue  # a model without an MF record serves LF only
            except hp.MFRangeError:
                print(f"not served: z = {z}, {fidelity} (below the lower edge of the MF record)")
                continue  # a reference that lists this case then reports it as missing
            total = hp.add_hcd(p, p.w_c)
            for name, values in [(n, getattr(p, n)) for n in ARRAYS] + [
                ("the mixed spectrum", total)
            ]:
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"a prediction at z = {z}, {fidelity}: {name} is not finite")
            case = {"z": float(z), "fidelity": fidelity, "product": p.labels["product"]}
            case.update({name: np.asarray(getattr(p, name)).tolist() for name in ARRAYS})
            cases.append(case)
    return cases


def compare(cases: list[dict[str, Any]], reference: dict[str, Any], rtol: float) -> list[str]:
    """Problems found when comparing the cases with a reference document; an empty list means they agree."""
    if reference.get("format") != FORMAT:
        return [f"the reference has format {reference.get('format')!r}, expected {FORMAT!r}"]
    keys = [(c.get("z"), c.get("fidelity")) for c in reference.get("cases", [])]
    if len(set(keys)) != len(keys):
        return ["the reference lists a case (z, fidelity) more than once"]
    ref = {(c.get("z"), c.get("fidelity")): c for c in reference.get("cases", [])}
    got = {(c["z"], c["fidelity"]): c for c in cases}
    problems = [
        f"case z = {z}, {f}: in the reference, not predicted"
        for z, f in sorted(set(ref) - set(got))
    ]
    problems += [
        f"case z = {z}, {f}: predicted, not in the reference"
        for z, f in sorted(set(got) - set(ref))
    ]
    for key in sorted(set(ref) & set(got)):
        for name in ARRAYS:
            a, b = np.asarray(got[key][name], dtype=float), np.asarray(
                ref[key].get(name), dtype=float
            )
            if a.shape != b.shape:
                problems.append(
                    f"case z = {key[0]}, {key[1]}: {name} has shape {a.shape}, the reference {b.shape}"
                )
            elif not np.allclose(a, b, rtol=rtol, atol=0.0):
                worst = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), np.finfo(float).tiny)))
                problems.append(
                    f"case z = {key[0]}, {key[1]}: {name} differs from the reference by up to {worst:.3e} "
                    f"(relative; allowed {rtol:g})"
                )
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -I -m hcd_analysis.check_release",
        description="Load a model release from a fresh install, predict, and compare with reference predictions.",
    )
    ap.add_argument("release_dir", nargs="?", help="the directory that holds release.json")
    ap.add_argument("release_sha256", nargs="?", help="the published sha256 of release.json")
    ap.add_argument(
        "--published",
        metavar="NAME",
        nargs="?",
        const="LF+MF",
        help="load a model from the installed model package (hcd-priya-models) instead; NAME is "
        "optional, and the LF+MF model is loaded without it",
    )
    ap.add_argument(
        "--z",
        type=float,
        nargs="+",
        default=list(DEFAULT_Z),
        help="snapshot redshifts to predict at",
    )
    ap.add_argument("--reference", help="reference predictions to compare with")
    ap.add_argument(
        "--write-reference",
        metavar="PATH",
        help="write the predictions as a reference file (new file)",
    )
    ap.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="relative tolerance of the comparison (default 1e-6)",
    )
    args = ap.parse_args(argv)
    if (args.published is None) == (args.release_dir is None) or (
        args.published is None and args.release_sha256 is None
    ):
        ap.error(
            "give RELEASE_DIR and RELEASE_SHA256, or --published (a NAME after it is optional)"
        )

    try:
        import hcd_analysis
        import hcd_analysis.predictor as hp

        where = os.path.dirname(os.path.abspath(hcd_analysis.__file__))
        print(f"hcd_analysis {hcd_analysis.__version__} imported from {where}")
        print(f"python {sys.version.split()[0]}; isolated mode (-I): {bool(sys.flags.isolated)}")
        if args.published is not None:
            emu = hp.load_published(args.published)
        else:
            emu = hp.load_release(args.release_dir, args.release_sha256)
        release = emu.provenance.get("release", {})
        synthetic = bool(release.get("synthetic_build"))
        if synthetic:
            print("SYNTHETIC BUILD: untrained networks. This checks the installation, not a model.")
        print(
            f"loaded: release {release.get('release_version')} ({release.get('format')}), "
            f"release.json sha256 {release.get('release_sha256')}, {emu.n_members} members"
        )
        cases = predictions(emu, args.z)
    except Exception as exc:  # any failure to load or predict is the result of this check
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    for c in cases:
        P = np.asarray(c["P_class"])
        print(
            f"predicted: z = {c['z']}, {c['fidelity']}: P_class {P.shape}, finite, product {c['product']}"
        )
    if not cases:
        print("FAILED: no case was predicted")
        return 1
    doc = {"format": FORMAT, "release_sha256": release.get("release_sha256"), "cases": cases}
    if args.write_reference:
        try:
            with open(args.write_reference, "x", encoding="utf-8") as fh:
                json.dump(doc, fh)
        except OSError as exc:  # an existing file is never replaced
            print(f"FAILED: the reference was not written: {type(exc).__name__}: {exc}")
            return 1
        print(f"wrote the reference predictions: {args.write_reference} ({len(cases)} cases)")
    if args.reference:
        try:
            with open(args.reference, encoding="utf-8") as fh:
                reference = json.load(fh)
            if not isinstance(reference, dict):
                raise ValueError("the reference is not a JSON object")
        except (OSError, ValueError) as exc:
            print(f"FAILED: the reference could not be read: {type(exc).__name__}: {exc}")
            return 1
        if reference.get("release_sha256") != doc["release_sha256"]:
            print(
                f"FAILED: the reference is for release.json sha256 {reference.get('release_sha256')}, "
                f"the loaded release has {doc['release_sha256']}"
            )
            return 1
        problems = compare(cases, reference, args.rtol)
        for p in problems:
            print(f"FAILED: {p}")
        if problems:
            return 1
        print(f"reference: {len(cases)} cases agree with {args.reference} to {args.rtol:g}")
    print(
        "what was checked: every file of the release verifies against its digest, the model loads, and its "
        "predictions are finite"
        + (
            " and equal to the reference"
            if args.reference
            else "; no reference was given, so no number was compared"
        )
        + ". This is a check of the installation, not a scientific validation of the model."
    )
    print(
        "PASSED (synthetic build: the installation was checked, not a model)"
        if synthetic
        else "PASSED"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
