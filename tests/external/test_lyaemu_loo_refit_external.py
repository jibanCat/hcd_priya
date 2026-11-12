"""OPTIONAL external test of the GP LOO refit adapter end to end (real GPy fits) on a READ-ONLY copy of the two
reference files it reads. Needs the reference's own interpreter and two paths:
  HCD_KS_BASEDIR=<basedir>  HCD_LYAEMU_PARENT=<directory holding lyaemu/>
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
  <ref-venv>/bin/python -B -m pytest --noconftest -p no:cacheprovider tests/external/test_lyaemu_loo_refit_external.py
Skipped otherwise (a skip is not a pass). One redshift and a subset of simulations keep the fits short; the masters
are never touched."""

import argparse
import os
import shutil
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True

BASEDIR = os.environ.get("HCD_KS_BASEDIR")
LYA = os.environ.get("HCD_LYAEMU_PARENT")
pytest.importorskip("GPy", reason="reference interpreter (GPy) not active")
if not (BASEDIR and LYA and os.path.isdir(BASEDIR) and os.path.isdir(LYA)):
    pytest.skip("HCD_KS_BASEDIR / HCD_LYAEMU_PARENT not set", allow_module_level=True)

from hcd_analysis.comparison.adapters import lyaemu_loo_refit as R  # noqa: E402
from hcd_analysis.comparison.adapters import lyaemu_priya as P  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILES = ("emulator_params.json", "mf_emulator_flux_vectors_tau1000000.hdf5")
ENV = {v: "1" for v in R.THREAD_VARS}
NAMES = ["ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback"]


@pytest.fixture
def ro_base(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    for f in FILES:
        shutil.copyfile(os.path.join(BASEDIR, f), base / f)
        os.chmod(base / f, 0o444)
    os.chmod(base, 0o555)
    yield str(base)
    os.chmod(base, 0o755)


def _inputs(base, out, z=3.0, n_train=20):
    """A one-redshift fold from upstream's own training rows: a held-out simulation and n_train others chosen so that
    every parameter spans its range (the rows are upstream's, so the parity is exact)."""
    import h5py

    with h5py.File(os.path.join(base, FILES[1]), "r") as f:
        params, fv, zout = f["params"][...], f["flux_vectors"][...], f["zout"][...]
    nk = fv.shape[1] // zout.size
    j = int(np.flatnonzero(np.isclose(zout, z))[0])
    keys, sim_of = np.unique(np.round(params[:, 1:], 12), axis=0, return_inverse=True)
    sim_of = sim_of.ravel()
    pick = set()
    for c in range(keys.shape[1]):
        pick |= {int(np.argmin(keys[:, c])), int(np.argmax(keys[:, c]))}
    rest = [s for s in range(keys.shape[0]) if s not in pick]
    held = rest[0]
    train = sorted(pick | set(rest[1 : 1 + n_train - len(pick)]))
    tr = np.flatnonzero(np.isin(sim_of, train))
    he = np.flatnonzero(sim_of == held)
    inp = {
        "X_train": params[tr][None],
        "Y_train": fv[tr, j * nk : (j + 1) * nk][None],
        "X_held": params[he][None],
        "Y_held": fv[he, j * nk : (j + 1) * nk][None],
        "rows_held": he[None],
        "rows_train": tr[None],
        "z": np.array([z]),
        "param_names": np.array(NAMES),
        "holdout_sim": np.asarray(f"upstream_sim_{held}"),
        "train_sims": sim_of[tr].astype(str),
        "train_alpha_idx": np.zeros(tr.size, int),
    }
    with open(out, "xb") as fh:
        np.savez(fh, **inp)
    return str(out)


def _request(base, inputs, seed=11):
    return {
        "seed": seed,
        "inputs": inputs,
        "inputs_sha256": P.sha256_file(inputs),
        "pins": {
            "artifacts_sha256": {f: P.sha256_file(os.path.join(base, f)) for f in FILES},
            "lyaemu_code_md5": P.lyaemu_md5(os.path.join(LYA, "lyaemu")),
            "emulator_json": FILES[0],
            "training_file": FILES[1],
        },
    }


def _args(tmp_path, base, req, name):
    import json

    rp = tmp_path / f"{name}.request.json"
    rp.write_text(json.dumps(req))
    return argparse.Namespace(
        repo=REPO,
        lyaemu_parent=LYA,
        basedir=base,
        request=str(rp),
        out=str(tmp_path / f"{name}.json"),
    )


def test_a_seeded_refit_runs_reproduces_and_writes_only_its_record(tmp_path, ro_base):
    req = _request(ro_base, _inputs(ro_base, tmp_path / "fold.npz"))
    before = P.tree_sha256(ro_base)
    a = R.run(_args(tmp_path, ro_base, req, "a"), environ=ENV)
    b = R.run(_args(tmp_path, ro_base, req, "b"), environ=ENV)
    assert (
        a["status"] == "OK" and np.all(np.isfinite(a["mean"])) and np.all(np.asarray(a["std"]) > 0)
    )
    assert (
        a["mean"] == b["mean"]
        and a["per_z"][0]["hyperparameters"] == b["per_z"][0]["hyperparameters"]
    )
    assert a["training_parity"]["train"][0]["max_rel"] == 0.0 and a["per_z"][0]["statuses"]
    assert P.tree_sha256(ro_base) == before
    assert sorted(os.listdir(tmp_path)) == sorted(
        ["a.json", "a.request.json", "b.json", "b.request.json", "base", "fold.npz"]
    )
    with pytest.raises(FileExistsError):
        R.run(_args(tmp_path, ro_base, req, "a"), environ=ENV)


def test_saving_a_gp_is_refused_and_the_reference_copy_cannot_be_written(tmp_path, ro_base):
    import GPy

    R.install_save_refusal()
    m = GPy.models.GPRegression(
        np.linspace(0, 1, 5)[:, None], np.sin(np.linspace(0, 1, 5))[:, None]
    )
    with pytest.raises(R.RefitRefused, match="save"):
        m.save(str(tmp_path / "m"))
    assert not any(p.name.startswith("m") for p in tmp_path.iterdir())
    with pytest.raises(PermissionError):
        open(os.path.join(ro_base, "zbin3.0.json"), "x").close()


def test_a_writable_copy_unpinned_inputs_and_threads_are_refused(tmp_path, ro_base):
    req = _request(ro_base, _inputs(ro_base, tmp_path / "fold.npz"))
    with pytest.raises(R.RefitRefused, match="single-threaded"):
        R.run(_args(tmp_path, ro_base, req, "t"), environ={})
    with pytest.raises(R.RefitRefused, match="do not match their pin"):
        R.run(_args(tmp_path, ro_base, dict(req, inputs_sha256="0" * 64), "p"), environ=ENV)
    unpinned = dict(
        req,
        pins=dict(
            req["pins"], artifacts_sha256={FILES[0]: req["pins"]["artifacts_sha256"][FILES[0]]}
        ),
    )
    with pytest.raises(R.RefitRefused, match="without a sha256 pin"):
        R.run(_args(tmp_path, ro_base, unpinned, "u"), environ=ENV)
    os.chmod(ro_base, 0o755)
    with pytest.raises(R.RefitRefused, match="writable"):
        R.run(_args(tmp_path, ro_base, req, "w"), environ=ENV)
