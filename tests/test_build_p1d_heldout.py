"""scripts/build_p1d_heldout.py: one build of a P1D held-out record from a residual file and a spec, each pinned by its
sha256; the refusal of a file that differs from its pin and of an existing directory; and the record written alone
with --without-evidence. Synthetic values only. A source-tier test: the script is not part of an installation.
"""

import hashlib
import json
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

from hcd_analysis.emulator import p1d_heldout as PH
from hcd_analysis.emulator import safe_io as S
from tests.synthetic_release import make_release

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@pytest.fixture(scope="module")
def rel(tmp_path_factory):
    return make_release(
        tmp_path_factory.mktemp("p1dbuild") / "release",
        incidence=True,
        heldout=True,
        p1d_heldout=True,
    )


@pytest.fixture(scope="module")
def held(rel):
    return PH.load_record(
        str(rel["root"] / "p1d_heldout"), expected_record_sha256=rel["p1d_heldout_sha"]
    )


def test_the_builder_makes_the_record_from_pinned_inputs_and_refuses_another_file(
    rel, held, tmp_path
):
    """scripts/build_p1d_heldout.py: a residual file and a spec, each pinned by its sha256."""
    script = os.path.join(REPO, "scripts", "build_p1d_heldout.py")
    r = held.record
    arrays, _ = S.read_table_h5(
        str(rel["root"] / "p1d_heldout" / PH.TABLES_FILE), PH.table_datasets(r)
    )
    res = tmp_path / "residuals.h5"
    with h5py.File(res, "w") as f:
        for fid in PH.FIDELITIES:
            g = f.create_group(fid)
            g.create_dataset("d", data=arrays[f"{fid}_d"])
            g.create_dataset("z", data=np.array(r[fid]["z"]))
            g.create_dataset("sims", data=np.array(r[fid]["sims"], dtype="S"))
            g.create_dataset("target_F", data=arrays[f"{fid}_target_F"])
            g.create_dataset("counts", data=arrays[f"{fid}_counts"])
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "inputs": {k: r["inputs"][k] for k in ("lf_cache", "hr_cache", "folds")},
                "code": r["code"],
                "sensitivity": {k: r["sensitivity"][k] for k in ("without_sim", "statement")},
                "what_it_does_not_cover": r["what_it_does_not_cover"],
                "notes": r["notes"],
            }
        )
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    base = [sys.executable, "-B", script, "--version", r["version"]]
    pins = [
        "--residuals",
        str(res),
        "--residuals-sha256",
        _sha(res),
        "--spec",
        str(spec),
        "--spec-sha256",
        _sha(spec),
    ]
    out = subprocess.run(
        base + ["--out-dir", str(tmp_path / "built")] + pins,
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "tables recomputed from" in out.stdout and ": equal" in out.stdout
    built = json.loads((tmp_path / "built" / PH.RECORD_FILE).read_text())
    for key in ("lf", "lfmf", "sensitivity", "summary", "notes", "what_it_does_not_cover", "code"):
        assert built[key] == r[key], key
    assert built["inputs"]["residuals"] == {"file": "residuals.h5", "sha256": _sha(res)}
    wrong = list(pins)
    wrong[3] = "0" * 64
    out = subprocess.run(
        base + ["--out-dir", str(tmp_path / "refused")] + wrong,
        capture_output=True,
        text=True,
        env=env,
    )
    assert (
        out.returncode != 0
        and "not the pinned" in out.stderr
        and not (tmp_path / "refused").exists()
    )
    out = subprocess.run(
        base + ["--out-dir", str(tmp_path / "built")] + pins,
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode != 0 and "exists" in out.stderr
    # the record alone: no evidence file is written, and the tables are the same
    out = subprocess.run(
        base + ["--out-dir", str(tmp_path / "alone")] + pins + ["--without-evidence"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "no evidence file was written" in out.stdout
    assert os.listdir(tmp_path / "alone") == [PH.RECORD_FILE]
    alone = json.loads((tmp_path / "alone" / PH.RECORD_FILE).read_text())
    assert alone["tables"] is None and alone["lf"] == r["lf"] and alone["lfmf"] == r["lfmf"]
