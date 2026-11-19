"""Write the golden file of the science test of a trained model (tests/test_science_model.py) for one model.

    python tools/ci/make_science_golden.py CONFIG_JSON OUT_JSON

CONFIG_JSON names the model as that test's HCD_SCIENCE_MODEL_CONFIG does: {"release_dir": ..., "release_sha256": ...}
for a release directory, or {"published": NAME} for a model of the installed model package. OUT_JSON must not exist:
a golden file is never overwritten. The points, redshifts, quantities and the call are imported from the test file,
so that the golden file and the test cannot drift apart. Values are stored as Python repr strings (exact; NaN as
'nan'). Run it from a checkout of the repository, with hcd-priya and its test extra installed.

The file holds predictions of the model: whether it may be published is decided with the model, not here.
"""

import datetime
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import platform
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)  # the test file imports its helper as tests.science_cddf_sums


def _test_module():
    path = os.path.join(REPO, "tests", "test_science_model.py")
    spec = importlib.util.spec_from_file_location("hcd_science_model_tests", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _cpu():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _package():
    """The installed hcd-priya: its version, the sha256 of its RECORD and of the files that make a prediction."""
    dist = md.distribution("hcd-priya")
    files = {str(f): f for f in dist.files or []}
    code = {}
    for rel in ("hcd_analysis/predictor.py", "hcd_analysis/emulator/kcoord.py"):
        code[rel] = _sha256(dist.locate_file(files[rel]))
    record = [f for f in dist.files if str(f).endswith(".dist-info/RECORD")][0]
    return {
        "name": dist.metadata["Name"],
        "version": dist.version,
        "record_sha256": _sha256(dist.locate_file(record)),
        "code_sha256": code,
    }


def main(config_path, out_path):
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists: a golden file is never overwritten")
    T = _test_module()
    with open(config_path) as fh:
        config = json.load(fh)
    emu = T.load_configured_model(config)
    cases = []
    for name, params, a, d in T.science_cases():
        pred = T.predict_case(emu, params, a, d)
        vals, flags = T.case_values(pred)
        case = {"point": name, "tau0_amp": a, "dtau0": d}
        case.update({q: T.encode(vals[q]) for q in T.QUANTITIES})
        case["flags_true"] = flags
        cases.append(case)
    doc = {
        "format": T.GOLDEN_FORMAT,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": T.model_identity(config),
        "package": _package(),
        "stack": {
            "python": platform.python_version(),
            **{n: md.version(n) for n in ("numpy", "jax", "jaxlib", "h5py", "equinox", "scipy")},
        },
        "machine": {"machine": platform.machine(), "cpu": _cpu(), "system": platform.system()},
        "settings": {
            "z": list(T.Z),
            "k_indices": list(T.K_INDICES),
            "cddf_log_nhi": list(T.CDDF_LOG_NHI),
            "fidelity": "the model's default (LF+MF when the release carries an MF record)",
            "call": "emu.predict_many(params, z=[2.2, 3.4, 4.6], tau0_amp=..., dtau0=...)",
            "seed": T.SEED,
            "points_rule": T.science_points.__doc__.strip(),
            "quantities": list(T.QUANTITIES),
            "encoding": "every value is repr(float(x)) as a string; NaN is 'nan'",
        },
        "points": [{"name": n, "params": p} for n, p in T.science_points()],
        "cases": cases,
    }
    with open(out_path, "x") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    print(f"wrote {out_path}: {len(cases)} cases, sha256 {_sha256(out_path)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python tools/ci/make_science_golden.py CONFIG_JSON OUT_JSON")
    main(sys.argv[1], sys.argv[2])
