#!/usr/bin/env python3
"""How much the detection threshold ``absorber.tau_threshold`` changes the HCD catalogue and the per-class templates.

The production value is 100 (``config/default.yaml``). For each raw fake_spectra file given, the catalogue is built
at every requested threshold on the SAME skewers, with every other catalogue setting read from the config file; the
per-class P1D are then computed by the production template path, unchanged:
``hcd_analysis.priya_p1d.compute_tier_c_p1d`` on the PRIYA tau = 1e6-filtered optical depths, and on the unfiltered
ones with the same scale and target flux, as ``scripts/build_emulator_cache_tau0.py`` does, at one mean-flux scaling
``--alpha``. The 15 fine N_HI bins are collapsed to clean / LLS / subDLA / DLA with the count weights of the emulator
cache loader (``hcd_analysis.emulator.data._collapse_p1d``), and the templates are taken as the forward model uses them:

    T_LLS = P_LLS / P_clean - 1,  T_subDLA = P_subDLA / P_clean - 1   (filtered)
    T_DLA = P_DLA,unfiltered / P_clean - 1

Each threshold is compared with the reference threshold (``--reference``, default 100) by: the absorber count ratio
per class, the template amplitude ratio ``a = sum T T_ref / sum T_ref^2``, the shape residual
``s = rms(T - a T_ref) / rms(T_ref)`` and the largest change of ``P_clean``, over the first ``--n-k`` native k bins
(k = 0 excluded). An amplitude change is what a free class amplitude in the likelihood absorbs; a shape change is not.

Outputs, in ``--out``: ``run_settings.json`` (every setting and input, written once; a rerun into the same directory
with any different setting is refused, hcd_analysis/run_settings.py), one ``result_<tag>_T<threshold>.npz`` per input
and threshold (written once; a rerun with identical settings reuses them), and ``summary.json``. With
``--anchor-cache``, the reference threshold's counts and P1D are also compared with the cache rows of the same
simulation, redshift and ``--alpha`` (an implementation check of the path).

Needs fake_spectra (run it in the environment that builds the emulator cache). Example:

  python scripts/tau_threshold_sensitivity.py --out NEW_DIR --sim SIM_NAME \\
      --input z2.2=/path/SPECTRA_021/lya_forest_spectra_grid_480.hdf5 --input z3.0=... \\
      --thresholds 30 50 100 200 300 --alpha 1.011182770662847
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hcd_analysis.run_settings import record_or_verify  # noqa: E402

CLASSES = ("clean", "LLS", "subDLA", "DLA")
# The four classes as half-open ranges of the 15 Tier-C fine bins: hcd_analysis.emulator.data.COARSE_SLICES (pinned
# equal by tests/test_tau_threshold_sensitivity.py; that module needs jax, which this environment may lack).
COARSE_RANGES = ((0, 1), (1, 8), (8, 13), (13, 15))
TEMPLATE_CLASSES = ("LLS", "subDLA", "DLA")


# ------------------------------------------------------------------------------------------------ pure helpers
def collapse(p15, counts15):
    """(15, K) fine-bin P1D and (15,) counts -> (4, K) class P1D, count-weighted within each class; an empty class is
    0 (as hcd_analysis.emulator.data._collapse_p1d, for one row)."""
    p15, counts15 = np.asarray(p15, dtype=np.float64), np.asarray(counts15, dtype=np.float64)
    out = np.zeros((4, p15.shape[1]))
    for i, (lo, hi) in enumerate(COARSE_RANGES):
        c = counts15[lo:hi]
        tot = c.sum()
        if tot > 0:
            out[i] = (c[:, None] / tot * p15[lo:hi]).sum(axis=0)
    return out


def templates(pf4, pu4):
    """Class templates as the forward model uses them: filtered LLS and subDLA, unfiltered DLA, over filtered clean."""
    clean = pf4[0]
    return {"LLS": pf4[1] / clean - 1.0, "subDLA": pf4[2] / clean - 1.0, "DLA": pu4[3] / clean - 1.0}


def compare(t, t_ref):
    """Amplitude ratio a (least squares of t on t_ref) and shape residual s = rms(t - a t_ref) / rms(t_ref)."""
    t, t_ref = np.asarray(t, dtype=np.float64), np.asarray(t_ref, dtype=np.float64)
    ref2 = float(np.sum(t_ref * t_ref))
    if not ref2 > 0:
        return float("nan"), float("nan")
    a = float(np.sum(t * t_ref)) / ref2
    s = float(np.sqrt(np.mean((t - a * t_ref) ** 2)) / np.sqrt(np.mean(t_ref ** 2)))
    return a, s


def class_counts(log_nhi):
    """Absorber counts per HCD class from catalogue log N_HI values (edges 17.2, 19.0, 20.3)."""
    x = np.asarray(log_nhi, dtype=np.float64)
    return {"LLS": int(((x >= 17.2) & (x < 19.0)).sum()), "subDLA": int(((x >= 19.0) & (x < 20.3)).sum()),
            "DLA": int((x >= 20.3).sum())}


def summarize(results, reference, n_k):
    """Per (tag, threshold, class): count ratio, a, s; per (tag, threshold): the largest clean change. ``results`` maps
    (tag, threshold) to dicts with counts15, pf15, pu15 (fine-bin arrays) and abs_counts (class -> count)."""
    out = {}
    for (tag, thr) in sorted(results):
        ref = results.get((tag, reference))
        if ref is None:
            raise KeyError(f"no reference threshold {reference} for {tag}")
        cur = results[(tag, thr)]
        sl = slice(1, n_k)
        pf, pu = collapse(cur["pf15"], cur["counts15"]), collapse(cur["pu15"], cur["counts15"])
        pf_r, pu_r = collapse(ref["pf15"], ref["counts15"]), collapse(ref["pu15"], ref["counts15"])
        t, t_r = templates(pf[:, sl], pu[:, sl]), templates(pf_r[:, sl], pu_r[:, sl])
        row = {"clean_max_frac_change": float(np.max(np.abs(pf[0, sl] / pf_r[0, sl] - 1.0))),
               "sightlines": {c: int(n) for c, n in zip(CLASSES, (cur["counts15"][lo:hi].sum()
                                                                  for lo, hi in COARSE_RANGES))}}
        for c in TEMPLATE_CLASSES:
            a, s = compare(t[c], t_r[c])
            n, n_r = cur["abs_counts"][c], ref["abs_counts"][c]
            row[c] = {"count_ratio": (n / n_r) if n_r else float("nan"), "amplitude_ratio": a, "shape_residual": s}
        out[f"{tag}|T{thr:g}"] = row
    return out


def verdict(summary, headline, reference):
    """The pre-registered reading rule (A3 design): over the headline thresholds, all classes and inputs."""
    rows = [(k, v) for k, v in summary.items() if float(k.split("|T")[1]) in headline]
    worst_a = max(abs(v[c]["amplitude_ratio"] - 1.0) for _, v in rows for c in TEMPLATE_CLASSES)
    worst_s = max(v[c]["shape_residual"] for _, v in rows for c in TEMPLATE_CLASSES)
    worst_clean = max(v["clean_max_frac_change"] for _, v in rows)
    if worst_a < 0.05 and worst_s < 0.05 and worst_clean < 0.01:
        word = "negligible"
    elif worst_a <= 0.29 and worst_s <= 0.10:
        word = "modest"
    else:
        word = "important"
    per_class = {}
    for c in TEMPLATE_CLASSES:
        wa = max(abs(v[c]["amplitude_ratio"] - 1.0) for _, v in rows)
        ws = max(v[c]["shape_residual"] for _, v in rows)
        per_class[c] = {"max_abs_a_minus_1": wa, "max_s": ws,
                        "reading": "negligible" if (wa < 0.05 and ws < 0.05) else
                                   "modest" if (wa <= 0.29 and ws <= 0.10) else "important"}
    return {"headline_thresholds": sorted(headline), "reference": reference, "reading": word,
            "max_abs_a_minus_1": worst_a, "max_s": worst_s, "max_clean_frac_change": worst_clean,
            "per_class": per_class}


# ------------------------------------------------------------------------------------------------ the run
def _file_identity(path):
    """Identity of a raw input without hashing gigabytes: size, mtime and the header fields the path depends on."""
    import h5py
    st = os.stat(path)
    with h5py.File(path, "r") as f:
        h = f["Header"].attrs
        shape = f["tau/H/1/1215"].shape
        return {"path": str(path), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
                "redshift": float(h["redshift"]), "nbins": int(h["nbins"]), "n_skewers": int(shape[0])}


def _save_once(path, **arrays):
    """Write an npz to a temporary name and link it into place; an existing result is never replaced."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    try:
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def run_input(tag, path, thresholds, alpha, absorber, sim, out_dir, log, n_skewers=None):
    from hcd_analysis.catalog import build_catalog
    from hcd_analysis.io import pixel_dv_kms, read_header
    from hcd_analysis.priya_p1d import _apply_priya_filter, compute_tier_c_p1d
    import build_emulator_cache_tau0 as bt                          # scripts/: the cache builder's own helpers

    todo = [t for t in thresholds if not (out_dir / f"result_{tag}_T{t:g}.npz").exists()]
    if not todo:
        return
    hdr = read_header(path)
    dv = pixel_dv_kms(hdr)
    vmax = bt.vmax_from_header(hdr)
    z_grid = bt._snap_z_to_priya_grid(float(hdr.redshift))
    t0 = time.time()
    tau = bt._read_tau(path, n_skewers=n_skewers)
    tau_filt = tau.copy()
    _apply_priya_filter(tau_filt, tau_thresh=1.0e6)                  # label-independent: once per input
    log(f"{tag}: read and filtered {tau.shape} in {time.time() - t0:.0f} s")
    for thr in todo:
        t1 = time.time()
        cat = build_catalog(Path(path), sim_name=sim, snap=-1, z=float(hdr.redshift), dv_kms=dv,
                            tau_threshold=float(thr), merge_dv_kms=absorber["merge_dv_kms"],
                            min_pixels=absorber["min_pixels"], b_init=absorber["b_init_kms"],
                            b_bounds=tuple(absorber["b_bounds"]), tau_fit_cap=absorber["tau_fit_cap"],
                            voigt_max_iter=absorber["voigt_max_iter"], fast_mode=absorber["fast_mode"],
                            n_workers=1, min_log_nhi=absorber["min_log_nhi"], n_skewers=n_skewers)
        t2 = time.time()
        kf, pf15, n15, target_f, scale = compute_tier_c_p1d(tau_filt, vmax, alpha_slope=alpha, z=z_grid, catalog=cat)
        _, pu15, n15u, _, _ = compute_tier_c_p1d(tau, vmax, alpha_slope=alpha, z=z_grid, catalog=cat,
                                                 external_scale=scale, external_target_F=target_f)
        assert np.array_equal(n15, n15u)
        log_nhi = np.array([a.log_NHI for a in cat.absorbers], dtype=np.float64)
        _save_once(out_dir / f"result_{tag}_T{thr:g}.npz", kf=kf, pf15=pf15, pu15=pu15, counts15=n15,
                   log_nhi=log_nhi, skewer_idx=np.array([a.skewer_idx for a in cat.absorbers], dtype=np.int64),
                   target_F=np.array(target_f), scale=np.array(scale), z_grid=np.array(z_grid))
        log(f"{tag} T={thr:g}: {len(cat.absorbers)} absorbers; catalogue {t2 - t1:.0f} s, P1D {time.time() - t2:.0f} s")


def load_results(out_dir, tags, thresholds):
    res = {}
    for tag in tags:
        for thr in thresholds:
            with np.load(out_dir / f"result_{tag}_T{thr:g}.npz") as d:
                res[(tag, thr)] = {k: d[k] for k in d.files}
            res[(tag, thr)]["abs_counts"] = class_counts(res[(tag, thr)]["log_nhi"])
    return res


def anchor(results, cache, sim, alpha, reference, n_k):
    """The reference threshold against the cache rows of the same simulation, z grid and alpha."""
    import h5py
    out = {}
    with h5py.File(cache, "r") as h:
        names = np.array([s.decode() if isinstance(s, bytes) else s for s in h["sim_name"][:]])
        zg, al = h["z_grid"][:], h["alpha_slope"][:]
        for (tag, thr), r in results.items():
            if thr != reference:
                continue
            rows = np.where((names == sim) & np.isclose(zg, float(r["z_grid"]), atol=1e-6, rtol=0)
                            & np.isclose(al, alpha, atol=1e-12, rtol=0))[0]
            if len(rows) != 1:
                out[tag] = {"error": f"{len(rows)} matching cache rows"}
                continue
            i = int(rows[0])
            c_counts = h["tier_c_counts"][i]
            m = min(n_k, h["P_tier_c_filtered"].shape[2])
            ok = (c_counts > 0) & (r["counts15"] > 0)

            def rel(cache_key, mine):
                a, b = h[cache_key][i][ok, 1:m], mine[ok, 1:m]
                return float(np.max(np.abs(b / a - 1.0)))
            out[tag] = {"cache_row": i, "count_diff_by_bin": (r["counts15"] - c_counts).astype(int).tolist(),
                        "max_rel_diff_filtered": rel("P_tier_c_filtered", r["pf15"]),
                        "max_rel_diff_unfiltered": rel("P_tier_c", r["pu15"]),
                        "scale_rel_diff": float(r["scale"] / h["scale"][i] - 1.0)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sim", required=True, help="simulation name (catalogue label; cache lookup for --anchor-cache)")
    ap.add_argument("--input", action="append", required=True, metavar="TAG=PATH",
                    help="a raw fake_spectra grid file, tagged (repeatable)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[30.0, 50.0, 100.0, 200.0, 300.0])
    ap.add_argument("--reference", type=float, default=100.0)
    ap.add_argument("--headline", type=float, nargs="+", default=[50.0, 200.0],
                    help="thresholds the reading rule is applied to")
    ap.add_argument("--alpha", type=float, required=True, help="mean-flux scaling alpha_slope")
    ap.add_argument("--n-k", type=int, default=172, help="native k bins compared (172: the LF emulator grid)")
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "default.yaml",
                    help="every catalogue setting except tau_threshold is read from here")
    ap.add_argument("--anchor-cache", type=Path, default=None)
    ap.add_argument("--n-skewers", type=int, default=None,
                    help="only the first N skewers of each input (a quick look; the anchor then does not apply)")
    a = ap.parse_args(argv)

    import yaml
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    if a.reference not in a.thresholds or not set(a.headline) <= set(a.thresholds):
        ap.error("--reference and every --headline threshold must be among --thresholds")
    if a.n_skewers is not None and a.anchor_cache is not None:
        ap.error("--anchor-cache compares whole files; it cannot be combined with --n-skewers")
    inputs = {}
    for item in a.input:
        tag, _, p = item.partition("=")
        if not tag or not p or tag in inputs:
            ap.error(f"--input {item!r}: expected a unique TAG=PATH")
        inputs[tag] = Path(p).resolve()
    cfg_bytes = a.config.read_bytes()
    absorber = dict(yaml.safe_load(cfg_bytes)["absorber"])
    absorber.pop("tau_threshold", None)
    thresholds = sorted(set(a.thresholds))
    settings = {"tool": "tau_threshold_sensitivity/1", "sim": a.sim,
                "inputs": {t: _file_identity(p) for t, p in sorted(inputs.items())},
                "thresholds": thresholds, "reference": a.reference, "headline": sorted(a.headline),
                "alpha_slope": a.alpha, "n_k": a.n_k, "absorber_except_threshold": absorber,
                "config": {"path": str(a.config.resolve()), "sha256": hashlib.sha256(cfg_bytes).hexdigest()},
                "anchor_cache": str(a.anchor_cache.resolve()) if a.anchor_cache else None,
                "n_skewers": a.n_skewers}

    def info():
        import subprocess
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=str(REPO_ROOT), text=True))
        except Exception:
            commit, dirty = "unknown", None
        import h5py
        try:
            import fake_spectra
            fs = getattr(fake_spectra, "__version__", fake_spectra.__file__)
        except Exception:
            fs = None
        return {"code_commit": commit, "code_dirty": dirty, "python": platform.python_version(),
                "numpy": np.__version__, "h5py": h5py.__version__, "fake_spectra": fs}

    record_or_verify(a.out, settings, info=info)
    logf = open(a.out / "log.txt", "a")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    for tag, p in sorted(inputs.items()):
        run_input(tag, p, thresholds, a.alpha, absorber, a.sim, a.out, log, n_skewers=a.n_skewers)
    results = load_results(a.out, sorted(inputs), thresholds)
    summary = summarize(results, a.reference, a.n_k)
    doc = {"rows": summary, "verdict": verdict(summary, set(a.headline), a.reference)}
    if a.anchor_cache:
        doc["anchor"] = anchor(results, a.anchor_cache, a.sim, a.alpha, a.reference, a.n_k)
    tmp = a.out / f".summary.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, a.out / "summary.json")                         # derived from the result files; recomputable
    log(f"verdict: {doc['verdict']['reading']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
