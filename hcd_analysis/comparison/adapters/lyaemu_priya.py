"""PRIYA lyaemu adapter: observable O1 from a trained lyaemu multi-fidelity GP (SingleBinAR1), in the reference's OWN
interpreter (it needs GPy/emukit), as a subprocess:

  <ref-venv>/bin/python -I -B hcd_analysis/comparison/adapters/lyaemu_priya.py --repo <hcd_priya> --lyaemu-parent <dir>
      --basedir <read-only copy> --request <request.json> --out <new record.json>

Guards (a reference is never refit or modified):
  * artifact sha256 and the lyaemu code md5 must equal the request's pins;
  * every ``class`` string in the trained GPy JSONs must be on an allowlist (GPy ``from_dict`` evals them);
  * paramz ``Model.optimize`` / ``optimize_restarts`` and GPy ``Model.save`` are replaced by a refusal, because lyaemu
    falls back to fitting and saving into ``traindir`` when a trained file fails to load;
  * exactly 13 'Loading pre-trained GP' and no 'Optimizing GP' lines while loading;
  * the basedir file tree hashes the same before and after.
The AR1 output at fidelity ``res`` is assembled per redshift exactly as ``MultiBinGP.predict`` does (column 0 times
the mean-flux slope factor), and ``--checks`` compares that assembly with the reference's own ``MultiBinGP.predict``
(res 1) and its own k conversion (``utilities.rebin_power_to_kms``, what ``get_predicted`` calls) at hcd_priya-computed
nodes of modes 2..171."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import sys

import numpy as np

GPY_CLASS_ALLOWLIST = {
    "GPy.models.GPRegression", "GPy.kern.Add", "GPy.kern.Linear", "GPy.kern.RBF", "GPy.likelihoods.Gaussian",
    "GPy.likelihoods.link_functions.Identity",
    "GPy.inference.latent_function_inference.exact_gaussian_inference.ExactGaussianInference"}


class ReferenceRefused(RuntimeError):
    """The reference would be refit, modified, or does not match its pins."""


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree_sha256(root):
    """sha256 of the sorted 'relative path TAB sha256' lines of every file under ``root``."""
    lines = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            lines.append(f"{os.path.relpath(p, root)}\t{sha256_file(p)}\n")
    return hashlib.sha256("".join(sorted(lines)).encode()).hexdigest()


def refuse_symlinks(root):
    """A reference tree read through a symlink could point anywhere: refuse any symlink under ``root``."""
    for dp, ds, fs in os.walk(root):
        for x in ds + fs:
            if os.path.islink(os.path.join(dp, x)):
                raise ReferenceRefused(f"symlink in a reference tree: {os.path.join(dp, x)}")


def package_sha256(root):
    """sha256 over (relative path, length, content) of every file of a package tree: unlike the md5 convention below,
    code moved between files or added in subpackages changes it."""
    h = hashlib.sha256()
    for dp, ds, fs in sorted(os.walk(root)):
        ds.sort()
        for f in sorted(fs):
            p = os.path.join(dp, f)
            if "__pycache__" in p.split(os.sep):
                continue
            with open(p, "rb") as fh:
                data = fh.read()
            rel = os.path.relpath(p, root).encode()
            h.update(len(rel).to_bytes(8, "little") + rel + len(data).to_bytes(8, "little") + data)
    return h.hexdigest()


def z_order(zout, z):
    """Positions in the reference's zout of the requested redshifts; refuses one that is not on the grid."""
    out = []
    for zz in z:
        j = int(np.argmin(np.abs(np.asarray(zout) - zz)))
        if abs(zout[j] - zz) > 1e-9:
            raise ReferenceRefused(f"z = {zz} is not on the reference grid")
        out.append(j)
    return out


def lyaemu_md5(pkgdir):
    """The kodiaq-squad-gp verify_basedir convention: md5 over the sorted top-level .py files."""
    h = hashlib.md5()
    for f in sorted(x for x in os.listdir(pkgdir) if x.endswith(".py")):
        with open(os.path.join(pkgdir, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def _classes(o, out):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "class":
                out.add(v)
            _classes(v, out)
    elif isinstance(o, list):
        for v in o:
            _classes(v, out)
    return out


def check_pins(basedir, lyaemu_pkg, pins):
    for rel, want in pins["artifacts_sha256"].items():
        got = sha256_file(os.path.join(basedir, rel))
        if got != want:
            raise ReferenceRefused(f"{rel}: sha256 {got} != pinned {want}")
    if tree_sha256(os.path.join(basedir, "trained_mf")) != pins["trained_mf_tree_sha256"]:
        raise ReferenceRefused("trained_mf tree does not match its pin")
    if lyaemu_md5(lyaemu_pkg) != pins["implementation"]["lyaemu_code_md5"]:
        raise ReferenceRefused("lyaemu code md5 does not match its pin")
    check_class_strings(os.path.join(basedir, "trained_mf"))
    refuse_symlinks(basedir)
    refuse_symlinks(lyaemu_pkg)
    refuse_compiled(lyaemu_pkg)
    parent = os.path.dirname(os.path.abspath(lyaemu_pkg))
    if sorted(os.listdir(parent)) != ["lyaemu"]:
        raise ReferenceRefused(f"{parent} must hold only lyaemu/ (anything else there could shadow imports)")


def refuse_compiled(pkg):
    """Python would execute a cached .pyc that matches its source's size and mtime; the pinned copy must carry none."""
    for dp, ds, fs in os.walk(pkg):
        if "__pycache__" in ds or any(f.endswith((".pyc", ".pyo")) for f in fs):
            raise ReferenceRefused(f"compiled bytecode in the reference package: {dp}")


def check_class_strings(trained_dir):
    """Every ``class`` string GPy's from_dict would eval must be on the allowlist."""
    seen = set()
    for f in sorted(os.listdir(trained_dir)):
        if f.endswith(".json"):
            with open(os.path.join(trained_dir, f)) as fh:
                _classes(json.load(fh), seen)
    if not seen <= GPY_CLASS_ALLOWLIST:
        raise ReferenceRefused(f"trained JSON names classes outside the allowlist: {sorted(seen - GPY_CLASS_ALLOWLIST)}")


def install_refit_refusal():
    import GPy
    import paramz

    def refuse(*a, **k):
        raise ReferenceRefused("the reference loader tried to fit or save a GP")
    paramz.Model.optimize = refuse
    paramz.Model.optimize_restarts = refuse
    GPy.core.Model.save = refuse


def load_reference(basedir, n_z):
    from lyaemu.gp_wrap import GPWrap
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gp = GPWrap(basedir=basedir, emulator_json_file="emulator_params.json",
                    kf=np.logspace(-3, np.log10(0.04), 8), tau_thresh=1e6, use_res_corr=False)
        gp.set_emulator(HRbasedir=os.path.join(basedir, "hires"), max_z=4.6, min_z=2.2,
                        traindir=os.path.join(basedir, "trained_mf"))
        gp.set_mf_param_limits(basedir=basedir)
    log = buf.getvalue()
    n_load, n_opt = log.count("Loading pre-trained GP"), log.count("Optimizing GP")
    if n_load != n_z or n_opt:
        raise ReferenceRefused(f"loading printed {n_load} 'Loading' and {n_opt} 'Optimizing' lines")
    return gp


def assemble(gp, nparams, tau0_fac, res):
    """Per-z AR1 output at fidelity ``res``, exactly as MultiBinGP.predict assembles res 1 (zout order)."""
    means, stds = [], []
    for i, sgp in enumerate(gp.gpemu.gps):
        zp = np.array(nparams, float).reshape(1, -1)
        zp[0][0] *= tau0_fac[i]
        m, s = sgp.predict(zp, res=res)
        means.append(np.asarray(m, float).ravel())
        stds.append(np.asarray(s, float).ravel())
    return np.array(means), np.array(stds)


def run(args):
    sys.path.insert(0, args.repo)
    sys.path.insert(0, args.lyaemu_parent)
    from hcd_analysis.comparison import conventions as C
    from hcd_analysis.comparison.schema import SCHEMA, validate_record

    with open(args.request) as f:
        req = json.load(f)
    pins, res, cfg = req["pins"], int(req["res"]), req["config"]
    lyaemu_pkg = os.path.join(args.lyaemu_parent, "lyaemu")
    check_pins(args.basedir, lyaemu_pkg, pins)
    before = tree_sha256(args.basedir)
    install_refit_refusal()
    import lyaemu
    if os.path.realpath(os.path.dirname(lyaemu.__file__)) != os.path.realpath(lyaemu_pkg):
        raise ReferenceRefused(f"imported lyaemu {lyaemu.__file__} is not the pinned copy {lyaemu_pkg}")
    import lyaemu.mean_flux as mflux
    from lyaemu.utilities import rebin_power_to_kms
    import GPy
    import hcd_analysis
    import paramz
    for mod in (GPy, paramz):
        if os.path.realpath(mod.__file__).startswith(os.path.realpath(args.lyaemu_parent) + os.sep):
            raise ReferenceRefused(f"{mod.__name__} was imported from the reference directory")
    if not os.path.realpath(hcd_analysis.__file__).startswith(os.path.realpath(args.repo) + os.sep):
        raise ReferenceRefused("hcd_analysis was not imported from the repository under test")
    gp = load_reference(args.basedir, len(cfg["z_grid"]))
    zout = np.asarray(gp.zout, float)
    if sorted(np.round(zout, 9).tolist()) != sorted(cfg["z_grid"]):
        raise ReferenceRefused(f"reference zout {zout.tolist()} differs from the config grid")
    ii = int(np.argmin(np.abs(zout - 3.0)))
    probe = ((1.0 + zout) / (1.0 + cfg["z_pivot"])) ** 0.3
    if zout[ii] != cfg["z_pivot"] or not np.allclose(mflux.mean_flux_slope_to_factor(zout, 0.3), probe,
                                                     rtol=1e-13, atol=0):
        raise ReferenceRefused("the reference mean-flux slope factor is not ((1+z)/(1+z_pivot))^dtau0 at z_pivot 3")
    if [float(g.zbin) for g in gp.gpemu.gps] != [round(float(z), 1) for z in zout]:
        raise ReferenceRefused("the reference GPs are not in zout order")
    k_com = np.asarray(gp.gpemu.kf, float)
    modes = list(range(1, k_com.size + 1))
    if not np.allclose(k_com, C.k_com_modes(np.asarray(modes), cfg["L_box_hmpc"]), rtol=1e-12, atol=0):
        raise ReferenceRefused("reference native k are not the box modes 2 pi n / L_box")
    ndense = len(gp.mf.dense_param_names)
    if ndense != 1:
        raise ReferenceRefused(f"expected one dense (mean-flux) parameter, found {ndense}")
    checks = {"glue_res1_max_rel": 0.0, "kconv_max_rel": 0.0, "n_checked_cases": 0}
    cases = []
    for case in req["cases"]:
        rec = {k: case[k] for k in ("case_id", "family", "z", "state")}
        rec["support"] = dict(case["support"])
        if case["support"]["status"] != "COMPARABLE":
            cases.append(rec)
            continue
        st = case["state"]
        mf = st["mean_flux"]
        if mf["z_pivot"] != zout[ii]:
            raise ReferenceRefused(f"{case['case_id']}: z_pivot {mf['z_pivot']} differs from the reference's {zout[ii]}")
        nparams = np.concatenate([[mf["tau0_amp"]], C.vector_from_names(st["params"], gp.param_names)])
        tau0_fac = np.asarray(mflux.mean_flux_slope_to_factor(zout, mf["dtau0"]), float)
        if not np.allclose(tau0_fac, ((1.0 + zout) / (1.0 + mf["z_pivot"])) ** mf["dtau0"], rtol=1e-13, atol=0):
            raise ReferenceRefused(f"{case['case_id']}: the reference slope factor is not the stated convention")
        mean, std = assemble(gp, nparams, tau0_fac, res)
        if args.checks:
            m1, _ = (mean, None) if res == 1 else assemble(gp, nparams, tau0_fac, 1)
            ref, _ = gp.gpemu.predict(nparams.reshape(1, -1), tau0_factors=tau0_fac)
            ref = np.asarray(ref, float).reshape(zout.size, -1)
            checks["glue_res1_max_rel"] = max(checks["glue_res1_max_rel"], float(np.max(np.abs(m1 / ref - 1))))
            om = nparams[ndense + gp.param_names["omegamh2"]] / nparams[ndense + gp.param_names["hub"]] ** 2
            for i, z in enumerate(zout):                       # the reference's own k conversion (get_predicted's)
                kf = C.k_skm_from_kcom(k_com[1:-1], z, st["params"]["hub"], st["params"]["omegamh2"])
                _, pk = rebin_power_to_kms(kfkms=kf, kfmpc=k_com, flux_powers=m1[i], zbins=np.array([z]), omega_m=om)
                checks["kconv_max_rel"] = max(checks["kconv_max_rel"],
                                              float(np.max(np.abs(np.asarray(pk)[0] / m1[i][1:-1] - 1))))
            checks["n_checked_cases"] += 1
        order = z_order(zout, case["z"])
        rec["P"] = mean[order].tolist()
        rec["k_skm"] = [C.k_skm_from_kcom(k_com, zout[j], st["params"]["hub"], st["params"]["omegamh2"]).tolist()
                        for j in order]
        rec["native"] = {"res": res, "std": std[order].tolist(), "gp_input_vector": nparams.tolist(),
                         "tau0_factor": tau0_fac[order].tolist(), "param_index": dict(gp.param_names),
                         "z_ref": zout[order].tolist()}
        cases.append(rec)
    if tree_sha256(args.basedir) != before:
        raise ReferenceRefused("the reference basedir changed during the run")
    record = {
        "schema": SCHEMA, "observable": "O1", "L_box_hmpc": float(cfg["L_box_hmpc"]), "modes": modes,
        "comparator": {"id": req["comparator_id"], "name": pins["name"], "repo": pins["repo"], "commit": pins["commit"],
                       "implementation": dict(pins["implementation"], output=f"SingleBinAR1.predict res={res}",
                                              lyaemu_package_sha256=package_sha256(lyaemu_pkg)),
                       "artifacts": [{"path": k, "sha256": v} for k, v in pins["artifacts_sha256"].items()],
                       "fidelity": "LF" if res == 0 else "HF",
                       "training": "60 PRIYA LF sims x 10 rungs (600) + 3 HF sims x 10 rungs (30), 13 z 2.2-4.6",
                       "env": {"python": platform.python_version(), "numpy": np.__version__, "GPy": GPy.__version__}},
        "cases": cases}
    validate_record(record)
    with open(args.out, "x") as f:
        json.dump({"record": record, "checks": checks, "basedir_tree_sha256": before}, f)


def main():
    ap = argparse.ArgumentParser()
    for k in ("--repo", "--lyaemu-parent", "--basedir", "--request", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--checks", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
