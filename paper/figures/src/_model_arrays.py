"""The model arrays of the CDDF figure of the paper (Figure 4), computed from the published release of hcd_priya (v1.0.0).

The arrays are reproduced from public inputs alone:

  - the trained model: the PyPI packages hcd-priya 1.0.0 and hcd-priya-models 1.0.0, loaded with
    `hcd_analysis.predictor.load_published()`;
  - the figure functions of the code repository at the tag v1.0.0 (commit 82ef21f): `scripts/figures/figures_common.py`
    and `scripts/figures/fig_dndx_vs_ho21.py`, fetched at that commit and checked by sha256;
  - the 60 design points of the PRIYA low-fidelity suite, `docs/figures/data/priya_lf_design_60.csv`, same commit.

What is computed, as the package's own figures do it: at the midpoint parameters of those figures
(`figures_common.MIDPOINT`), with tau0_amp 1 and dtau0 0, low fidelity with the multi-fidelity correction, at the 13
snapshots z 2.2 to 4.6: the CDDF (30 bins of log N_HI 17 to 23), dN/dX of the three classes, their held-out ranges,
the range of each over the 60 design points, and the incidence of all absorbers with N_HI >= 10^17.5
(`fig_dndx_vs_ho21.cumulative_dndx`). The CDDF figure draws the CDDF and its band; the incidence arrays are
computed with them, as the package's figure functions do, and are not drawn in the paper.

Self-check, every time the arrays are computed: the model that was loaded must say it is release 1.0.0 with the
published release.json (sha256 below). The claim "reproduced from the published v1.0.0 release" is tested there.
The check is on the model and on the arrays it gives, never on the sha256 of an arrays file.

Where the computation runs. It needs an environment with the two packages (`pip install hcd-priya==1.0.0`):
  - this interpreter, if it has them; or
  - the interpreter named by the variable HCD_PRIYA_PYTHON (the path of a python that has them).
The result is kept in `.build/cache/model_arrays/` (not in git), with the identity of the model inside it, and is
used again as long as that identity is the one below. Delete the file to compute again. If neither interpreter is
there and there is no cache, `load()` returns None and the figure scripts keep their images.

Held-out range rule: drawn on the quantity, from `_lower` to `_upper` as given; never symmetric.
Bands (`_band_min`, `_band_max`): the range over the 60 design points of the training set; not an uncertainty.

To compute the arrays by hand:
    /path/to/python paper/figures/src/_model_arrays.py --write OUT.npz
"""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PYTHON_VAR = "HCD_PRIYA_PYTHON"

# ---- the identity of the published model and of the public files the computation uses
RELEASE_VERSION = "1.0.0"
RELEASE_SHA256 = "c81c33ce65455098620a6e7e54e95bc87f10b06ac10970082ffbc9c6fe5470ad"  # of the published release.json
PACKAGES = {"hcd-priya": "1.0.0", "hcd-priya-models": "1.0.0"}
COMMIT = "82ef21f698fc5cbdb21fcd971a7d2795da1dd984"  # the tag v1.0.0 of the code repository
FILES = {
    "scripts/figures/figures_common.py": "aa1f2bc4320c3fb78ea820a355193b832b1f3454bdae51428cec8ad867a2200e",
    "scripts/figures/fig_dndx_vs_ho21.py": "66301416141d6340cb20b45e58a83f089e84e2e6dd7a14c87511396eb59001e0",
    "docs/figures/data/priya_lf_design_60.csv": "a47f23bfce73b3d765206cef5b116cf42b3705c9a0df85a5e665cda185255d42",
}
IDENTITY = {"release_version": RELEASE_VERSION, "release_sha256": RELEASE_SHA256, "code_commit": COMMIT,
            "files": FILES}
CACHE = ROOT / ".build" / "cache" / "model_arrays" / f"release_{RELEASE_SHA256[:16]}_code_{COMMIT[:7]}.npz"
KEYS = ("z", "log_nhi", "cddf", "cddf_lower", "cddf_upper", "cddf_band_min", "cddf_band_max", "dndx_classes", "dndx",
        "dndx_lower", "dndx_upper", "dndx_band_min", "dndx_band_max", "cumulative_ge_17p5",
        "cumulative_ge_17p5_band_min", "cumulative_ge_17p5_band_max")

WHY = ""  # after load() returns None: the reason, for the message of the figure script


def has_packages():
    try:
        return all(importlib.metadata.version(name) == version for name, version in PACKAGES.items())
    except importlib.metadata.PackageNotFoundError:
        return False


def public_files(folder):
    """Fetch the three public files at the tag into a folder, each checked by sha256. Returns False if one cannot
    be read."""
    sys.path.insert(0, str(HERE))
    from _public import fetch

    for path, want in FILES.items():
        data = fetch(COMMIT, path)
        if data is None:
            return False
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            raise SystemExit(f"STOP: {path} at {COMMIT[:7]} has sha256 {got}, expected {want}")
        (Path(folder) / Path(path).name).write_bytes(data)
    return True


def compute():
    """The arrays, from the published model, in this interpreter. Asserts the identity of the model."""
    import warnings

    import hcd_analysis.predictor as hp

    with tempfile.TemporaryDirectory() as tmp:
        if not public_files(tmp):
            raise SystemExit(f"STOP: cannot read the figure functions of the code at {COMMIT[:7]}")
        sys.path.insert(0, tmp)
        import fig_dndx_vs_ho21 as FD
        import figures_common as C

        emu = hp.load_published()
        release = emu.provenance.get("release", {})
        # the self-check: this is the published release, not another model that happens to be installed
        if release.get("release_version") != RELEASE_VERSION or release.get("release_sha256") != RELEASE_SHA256:
            raise SystemExit(f"STOP: the installed model is release {release.get('release_version')} with release.json "
                             f"sha256 {release.get('release_sha256')}; these figures are of release {RELEASE_VERSION} "
                             f"({RELEASE_SHA256})")
        print(f"model: release {release['release_version']}, release.json sha256 {release['release_sha256']} "
              f"(the published one: yes); packages " + ", ".join(f"{n} {importlib.metadata.version(n)}" for n in PACKAGES))
        design = C.read_design(os.path.join(tmp, "priya_lf_design_60.csv"), hp.PARAM_NAMES)
        if len(design) != 60:
            raise SystemExit(f"STOP: {len(design)} design points, expected 60")
        zs = [float(z) for z in FD.REDSHIFTS]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stack = emu.predict_many(C.MIDPOINT, z=zs, tau0_amp=1.0, dtau0=0.0)
            preds = [emu.predict(C.MIDPOINT, z=z, tau0_amp=1.0, dtau0=0.0) for z in zs]
        out = {"z": np.array(zs), "log_nhi": np.asarray(stack.log_nhi, dtype=float),
               # the CDDF of each redshift from its own prediction, as the package's CDDF figure draws it; the stacked
               # prediction agrees with it to rounding and feeds the cumulative incidence, as in the dN/dX figure of the package
               "cddf": np.array([np.asarray(p.cddf, dtype=float) for p in preds]),
               "cddf_lower": np.array([np.asarray(p.cddf_lower, dtype=float) for p in preds]),
               "cddf_upper": np.array([np.asarray(p.cddf_upper, dtype=float) for p in preds]),
               "dndx_classes": np.array(hp.HCD_CLASSES),
               "dndx": np.asarray(stack.dndx, dtype=float), "dndx_lower": np.asarray(stack.dndx_lower, dtype=float),
               "dndx_upper": np.asarray(stack.dndx_upper, dtype=float),
               "cumulative_ge_17p5": np.asarray(FD.cumulative_dndx(stack.cddf, stack.log_nhi), dtype=float)}
        bands = {k: [] for k in ("cddf_band_min", "cddf_band_max", "dndx_band_min", "dndx_band_max",
                                 "cumulative_ge_17p5_band_min", "cumulative_ge_17p5_band_max")}
        refused = 0
        for z, pred in zip(zs, preds):
            drawn = C.flags_set(pred)
            lo, hi, _, r1 = C.envelope(emu, design, z, "cddf", drawn)
            bands["cddf_band_min"].append(lo)
            bands["cddf_band_max"].append(hi)
            lo, hi, _, r2 = C.envelope(emu, design, z, FD.dndx_and_cumulative, drawn)
            bands["dndx_band_min"].append(lo[:3])
            bands["dndx_band_max"].append(hi[:3])
            bands["cumulative_ge_17p5_band_min"].append(lo[FD.CUMULATIVE_COLUMN])
            bands["cumulative_ge_17p5_band_max"].append(hi[FD.CUMULATIVE_COLUMN])
            refused += len(r1) + len(r2)
        if refused:
            raise SystemExit(f"STOP: the predictor refused {refused} design points; the bands would not be over all 60")
        out.update({k: np.array(v) for k, v in bands.items()})
        sys.path.remove(tmp)
    return out


def save(arrays, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, identity=np.array(json.dumps(IDENTITY, sort_keys=True)), **arrays)


def read(path):
    """The arrays of a file this module wrote for the same model and code, or None."""
    try:
        with np.load(path, allow_pickle=False) as f:
            if json.loads(str(f["identity"])) != json.loads(json.dumps(IDENTITY, sort_keys=True)):
                return None
            return {k: f[k] for k in KEYS}
    except (OSError, KeyError, ValueError):
        return None


def load():
    """The arrays, or None (with the reason in WHY) when they can be neither read from the cache nor computed."""
    global WHY
    arrays = read(CACHE) if CACHE.exists() else None
    if arrays is not None:
        print(f"model arrays: from the cache {CACHE.relative_to(ROOT)} (computed from the published release "
              f"{RELEASE_VERSION})")
        return arrays
    if has_packages():
        arrays = compute()
        save(arrays, CACHE)
        print(f"model arrays: computed in this interpreter; kept in {CACHE.relative_to(ROOT)}")
        return arrays
    python = os.environ.get(PYTHON_VAR)
    if python:
        # -I: the other interpreter must use its installed packages only, not a development tree on PYTHONPATH;
        # -B: it writes no bytecode beside the scripts
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        r = subprocess.run([python, "-I", "-B", str(Path(__file__).resolve()), "--write", str(CACHE)], env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"STOP: {PYTHON_VAR} could not compute the model arrays:\n{r.stderr.strip()[-1500:]}")
        print("\n".join(ln for ln in r.stdout.splitlines() if ln.startswith("model")))
        arrays = read(CACHE)
        if arrays is None:
            raise SystemExit(f"STOP: {PYTHON_VAR} wrote no usable arrays to {CACHE}")
        print(f"model arrays: computed with {PYTHON_VAR}; kept in {CACHE.relative_to(ROOT)}")
        return arrays
    WHY = (f"hcd-priya {RELEASE_VERSION} is not installed in this python, {PYTHON_VAR} is not set, "
           f"and there are no cached arrays")
    return None


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--write":
        sys.exit("usage: python _model_arrays.py --write OUT.npz   (in an environment with hcd-priya 1.0.0)")
    if not has_packages():
        sys.exit("STOP: this python does not have " + " and ".join(f"{n} {v}" for n, v in PACKAGES.items()))
    save(compute(), sys.argv[2])
    print(f"model arrays: written to {sys.argv[2]}")
