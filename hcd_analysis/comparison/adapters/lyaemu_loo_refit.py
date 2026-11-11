"""lyaemu LF-only GP leave-one-simulation-out REFIT for the matched leave-one-out comparison, in the
reference's OWN interpreter (GPy, emukit, lyaemu), as a subprocess, one fold per call:

  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
  <ref-venv>/bin/python -I -B hcd_analysis/comparison/adapters/lyaemu_loo_refit.py --repo <hcd_priya>
      --lyaemu-parent <dir> --basedir <read-only reference copy> --request <request.json> --out <new record.json>

The algorithm is upstream's LF-only emulator (``MultiBinGP`` without HF data): one ``SkLearnGP`` per redshift on the
fold's training rows, inputs [rung, 9 parameters] mapped to the unit cube with the reference's own training limits
(``GPWrap.get_param_limits(include_dense=True)`` from the basedir's ``emulator_params.json``), outputs normalized by
the median row, ``optimize()`` then ``optimize_restarts(10)`` when not converged. The training values and the truth
are hcd_priya's matched cache rows (``loo_matched.gp_fold_inputs``), identical for both methods; every input row must
be one of upstream's training rows (its pinned training file), and the flux-vector parity against that file is
recorded per redshift. The unit-cube limits are upstream's (``GPWrap.mf`` from ``set_mf_param_limits``); their
mean-flux range reaches above the top training rung, so ``mf.get_params()`` is not the training design.

Changes from upstream, each recorded: NumPy's global seed is set before each redshift's fit (upstream sets none;
``seed_z = seed * 100 + z index``); every optimizer run's objective and status are recorded, with the run the model
keeps, and a redshift whose kept run did not converge is flagged (``converged_final``); a unit-cube limit that a
TRAINING input misses by at most 4 ulp is moved onto that input (``ulp_adjust_limits``), because upstream's
``map_to_unit_cube`` allows only a 1e-16 margin and the computed limit can round past an input (held-out inputs never
move a limit; one outside the limits makes the fold NOT_RUN). Nothing is saved. The guards that prevent a write are
``traindir`` None (``SkLearnGP`` then always fits and never writes its JSON) and the read-only reference copy, whose
tree hashes the same before and after; GPy ``Model.save`` is also replaced by a refusal, and a 'Saving GP' line in
the fit log is refused after the fact. Fold failures: an input column that does not span the unit
cube (upstream's assertion) or a ``LinAlgError`` makes the fold NOT_RUN, recorded with the reason."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import sys
import time

import numpy as np

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
N_RESTARTS = 10  # upstream: optimize_restarts(num_restarts=10) when optimize() did not converge


def adopted_run(runs, restarted):
    """Index of the optimizer run whose parameters the model keeps: the first run, or, after restarts, the one with
    the lowest objective among the last ``N_RESTARTS`` runs (paramz ``optimize_restarts``)."""
    if not runs:
        return None
    if not restarted:
        return 0
    tail = runs[-N_RESTARTS:]
    return len(runs) - len(tail) + int(np.argmin([f for f, _ in tail]))


class RefitRefused(RuntimeError):
    """The refit would write, run on unpinned inputs, or leave the stated conventions."""


def refuse_writable(root):
    """The reference copy must be read-only: no directory or file under ``root`` may be writable by this process."""
    for dp, ds, fs in os.walk(root):
        for p in [dp] + [os.path.join(dp, f) for f in fs]:
            if os.access(p, os.W_OK):
                raise RefitRefused(
                    f"the reference copy is writable ({p}); make it read-only (chmod -R a-w)"
                )


def install_save_refusal():
    """Refits are allowed, saving is not: GPy ``Model.save`` raises."""
    import GPy

    def refuse(*a, **k):
        raise RefitRefused("the refit tried to save a GP")

    GPy.core.Model.save = refuse


def install_status_recorder():
    """Wrap paramz ``Model.optimize`` so that every call's optimizer status is appended to the returned list."""
    import paramz

    statuses = []
    orig = paramz.Model.optimize

    def optimize(self, *a, **k):
        r = orig(self, *a, **k)
        statuses.append(str(getattr(r, "status", r)))
        return r

    paramz.Model.optimize = optimize
    return statuses


def check_threads(environ):
    bad = {v: environ.get(v) for v in THREAD_VARS if environ.get(v) != "1"}
    if bad:
        raise RefitRefused(f"single-threaded fits required (one thread per BLAS library): {bad}")


def span_ok(cube):
    """Upstream's assertion, checked before fitting: every input column spans below 0.2 and above 0.8."""
    return bool(np.all(cube.max(axis=0) > 0.8) and np.all(cube.min(axis=0) < 0.2))


def ulp_adjust_limits(plimits, X_all, max_ulp=4):
    """Upstream's ``map_to_unit_cube`` asserts every input inside the limits with a 1e-16 margin. An input outside a
    limit by at most ``max_ulp`` units in the last place (floating-point rounding of the computed limit) moves that
    limit onto the input; anything farther is refused. Returns (limits, the recorded adjustments).
    """
    lim = np.array(plimits, dtype=np.float64, copy=True)
    X_all = np.asarray(X_all, dtype=np.float64)
    adj = []
    for c in range(lim.shape[0]):
        for side, v in ((0, X_all[:, c].min()), (1, X_all[:, c].max())):
            if (v < lim[c, 0]) if side == 0 else (v > lim[c, 1]):
                n_ulp = abs(v - lim[c, side]) / np.spacing(abs(lim[c, side]))
                if n_ulp > max_ulp:
                    raise RefitRefused(
                        f"input column {c} lies {n_ulp:.3g} ulp outside upstream's limit "
                        f"{lim[c, side]!r}"
                    )
                adj.append(
                    {
                        "column": c,
                        "side": ("low", "high")[side],
                        "limit": float(lim[c, side]),
                        "set_to": float(v),
                        "ulp": float(n_ulp),
                    }
                )
                lim[c, side] = v
    return lim, adj


def match_reference_design(X, Y, z, ref_params, ref_fv, ref_zout):
    """Every input row (rung, 9 parameters) must equal exactly one row of upstream's training file (relative 1e-9),
    so the GP is refit on upstream's own design points. Returns, per redshift, the largest |Y / Y_upstream - 1| and
    the number of rows above 1e-4 (the shared-truth parity: rows that differ are reported, not refused).
    """
    ref_params, ref_fv, ref_zout = (np.asarray(a, float) for a in (ref_params, ref_fv, ref_zout))
    nk = ref_fv.shape[1] // ref_zout.size
    if Y.shape[2] != nk:
        raise RefitRefused(f"{Y.shape[2]} modes in the inputs, upstream's training file has {nk}")
    out = []
    for i, zz in enumerate(z):
        j = np.flatnonzero(np.abs(ref_zout - zz) < 1e-9)
        if j.size != 1:
            raise RefitRefused(f"z={zz} is not on upstream's training grid")
        hit = np.all(
            np.isclose(X[i][:, None, :], ref_params[None, :, :], rtol=1e-9, atol=0), axis=2
        )
        if not np.all(hit.sum(axis=1) == 1):
            raise RefitRefused(
                f"z={zz}: {int(np.sum(hit.sum(axis=1) != 1))} input rows are not exactly one of "
                f"upstream's training rows"
            )
        ref = ref_fv[hit.argmax(axis=1), int(j[0]) * nk : (int(j[0]) + 1) * nk]
        rel = np.max(np.abs(Y[i] / ref - 1.0), axis=1)
        out.append(
            {
                "z": float(zz),
                "max_rel": float(rel.max()),
                "n_rows_above_1e-4": int(np.sum(rel > 1e-4)),
            }
        )
    return out


def fit_fold(X, Y, Xh, z, plimits, seed, *, statuses, sklearn_gp, to_cube):
    """Refit one fold: per redshift i, ``sklearn_gp`` (lyaemu ``SkLearnGP``) on X[i], Y[i], then predict Xh[i].
    Returns (status, reason, per_z, mean, std); status NOT_RUN on a span failure or a LinAlgError.
    """
    nz, m, nk = Xh.shape[0], Xh.shape[1], Y.shape[2]
    for i in range(nz):
        if not span_ok(np.asarray(to_cube(X[i], plimits))):
            return "NOT_RUN", f"input span assertion fails at z={z[i]}", [], None, None
    mean, std, per_z = np.full((nz, m, nk), np.nan), np.full((nz, m, nk), np.nan), []
    for i in range(nz):
        seed_z = int(seed) * 100 + i
        np.random.seed(seed_z)
        n0 = len(statuses)
        buf = io.StringIO()
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(buf):
                gp = sklearn_gp(
                    params=X[i], powers=Y[i], param_limits=plimits, zbin=z[i], traindir=None
                )
            mu, sd = gp.predict(Xh[i])
        except np.linalg.LinAlgError as e:
            return "NOT_RUN", f"LinAlgError at z={z[i]}: {e}", per_z, None, None
        log = buf.getvalue()
        if log.count("Optimizing GP") != 1 or "Saving GP" in log or "Loading pre-trained" in log:
            raise RefitRefused(f"z={z[i]}: unexpected fit log {log!r}")
        st = statuses[n0:]
        restarted = "Restarting optimization" in log
        runs = [
            (float(getattr(o, "f_opt", np.nan)), str(getattr(o, "status", "")))
            for o in getattr(gp.gp, "optimization_runs", [])
        ]
        k = adopted_run(runs, restarted)
        per_z.append(
            {
                "z": float(z[i]),
                "seed_z": seed_z,
                "statuses": st,
                "restarted": restarted,
                "runs": [{"f_opt": f, "status": t} for f, t in runs],
                "adopted_run": k,
                "converged_final": k is not None and runs[k][1] == "Converged",
                "converged_first": bool(st) and st[0] == "Converged",
                "converged_any": any(x == "Converged" for x in st),
                "hyperparameters": dict(
                    zip(
                        gp.gp.parameter_names_flat().tolist(),
                        np.asarray(gp.gp.param_array, float).tolist(),
                    )
                ),
                "log_likelihood": float(gp.gp.log_likelihood()),
                "fit_seconds": time.perf_counter() - t0,
            }
        )
        mean[i], std[i] = np.asarray(mu, float), np.asarray(sd, float)
    return "OK", "", per_z, mean, std


def run(args, environ=os.environ):
    if os.path.lexists(args.out):
        raise FileExistsError(f"{args.out} exists; a fold record is written once")
    with open(args.request) as f:
        req = json.load(f)
    check_threads(environ)
    lyaemu_pkg = os.path.join(args.lyaemu_parent, "lyaemu")
    sys.path.insert(0, args.repo)
    from hcd_analysis.comparison.adapters import lyaemu_priya as P

    pins = req["pins"]
    missing = {pins["emulator_json"], pins["training_file"]} - set(pins["artifacts_sha256"])
    if missing:
        raise RefitRefused(f"reference files read without a sha256 pin: {sorted(missing)}")
    for rel, want in pins["artifacts_sha256"].items():
        if P.sha256_file(os.path.join(args.basedir, rel)) != want:
            raise RefitRefused(f"{rel} does not match its pin")
    if P.lyaemu_md5(lyaemu_pkg) != pins["lyaemu_code_md5"]:
        raise RefitRefused("lyaemu code md5 does not match its pin")
    P.refuse_symlinks(args.basedir)
    P.refuse_symlinks(lyaemu_pkg)
    P.refuse_compiled(lyaemu_pkg)
    if sorted(os.listdir(args.lyaemu_parent)) != ["lyaemu"]:
        raise RefitRefused(f"{args.lyaemu_parent} must hold only lyaemu/")
    refuse_writable(args.basedir)
    if P.sha256_file(req["inputs"]) != req["inputs_sha256"]:
        raise RefitRefused("the fold inputs do not match their pin")
    with np.load(req["inputs"], allow_pickle=False) as f:
        inp = {k: f[k] for k in f.files}
    before = P.tree_sha256(args.basedir)
    sys.path.insert(0, args.lyaemu_parent)
    install_save_refusal()
    statuses = install_status_recorder()
    import lyaemu

    if os.path.realpath(os.path.dirname(lyaemu.__file__)) != os.path.realpath(lyaemu_pkg):
        raise RefitRefused(f"imported lyaemu {lyaemu.__file__} is not the pinned copy")
    from lyaemu.gp_wrap import GPWrap
    from lyaemu.gpemulator import SkLearnGP
    from lyaemu.latin_hypercube import map_to_unit_cube_list
    import GPy

    with contextlib.redirect_stdout(io.StringIO()):
        wrap = GPWrap(
            basedir=args.basedir,
            emulator_json_file=pins["emulator_json"],
            kf=np.logspace(-3, np.log10(0.04), 8),
            tau_thresh=1e6,
            use_res_corr=False,
        )
    plimits = np.asarray(wrap.get_param_limits(include_dense=True), float)
    names = [str(x) for x in inp["param_names"]]
    if sorted(wrap.param_names, key=wrap.param_names.get) != names:
        raise RefitRefused(
            f"input parameter order {names} is not the reference's {dict(wrap.param_names)}"
        )
    import h5py

    with h5py.File(os.path.join(args.basedir, pins["training_file"]), "r") as f:
        ref = (f["params"][...], f["flux_vectors"][...], f["zout"][...])
    parity = {
        "train": match_reference_design(inp["X_train"], inp["Y_train"], inp["z"], *ref),
        "truth": match_reference_design(inp["X_held"], inp["Y_held"], inp["z"], *ref),
    }
    plimits, adjusted = ulp_adjust_limits(plimits, inp["X_train"].reshape(-1, plimits.shape[0]))
    held = inp["X_held"].reshape(-1, plimits.shape[0])
    if np.any(held < plimits[:, 0]) or np.any(held > plimits[:, 1]):
        status, reason, per_z, mean, std = (
            "NOT_RUN",
            "held-out inputs outside the unit-cube limits",
            [],
            None,
            None,
        )
    else:
        status, reason, per_z, mean, std = fit_fold(
            inp["X_train"],
            inp["Y_train"],
            inp["X_held"],
            inp["z"],
            plimits,
            int(req["seed"]),
            statuses=statuses,
            sklearn_gp=SkLearnGP,
            to_cube=map_to_unit_cube_list,
        )
    if P.tree_sha256(args.basedir) != before:
        raise RefitRefused("the reference basedir changed during the refit")
    out = {
        "fold_sim": str(inp["holdout_sim"]),
        "seed": int(req["seed"]),
        "status": status,
        "reason": reason,
        "z": inp["z"].tolist(),
        "rows_held": inp["rows_held"].tolist(),
        "rows_train": inp["rows_train"].tolist(),
        "per_z": per_z,
        "training_parity": parity,
        "mean": None if mean is None else mean.tolist(),
        "std": None if std is None else std.tolist(),
        "param_limits": plimits.tolist(),
        "inputs_sha256": req["inputs_sha256"],
        "pins": pins,
        "basedir_tree_sha256": before,
        "limit_adjustments": adjusted,
        "changes_from_upstream": [
            "np.random.seed(seed*100 + z index) per fit",
            "optimizer statuses recorded",
        ]
        + (
            ["unit-cube limits moved onto training inputs within 4 ulp (limit_adjustments)"]
            if adjusted
            else []
        ),
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "GPy": GPy.__version__,
            **{v: environ.get(v) for v in THREAD_VARS},
        },
    }
    with open(args.out, "x") as f:
        json.dump(out, f)
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("--repo", "--lyaemu-parent", "--basedir", "--request", "--out"):
        ap.add_argument(k, required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
