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

The run reads each input in blocks of ``--chunk`` skewers (twice: once for the mean flux the PRIYA filter uses), so its
memory stays near three blocks; the per-bin P1D is a mean over sightlines, so the blocks combine exactly by their
counts, and ``--self-check N`` compares the blocked path with the whole-array production path on N skewers. The
mean-flux scale is the cache row's (production) value with ``--anchor-cache``, else solved over the file.

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


def _read_rows(path, s, e):
    import h5py
    with h5py.File(path, "r") as f:
        return f["tau/H/1/1215"][s:e].astype(np.float64)


def _global_tau_eff(path, n_total, chunk):
    """-log <exp(-tau)> over the first n_total skewers: the tau_eff _apply_priya_filter computes on the whole array."""
    acc, npix = 0.0, 0
    for s in range(0, n_total, chunk):
        t = _read_rows(path, s, min(n_total, s + chunk))
        acc += float(np.sum(np.exp(-t)))
        npix += t.size
    return float(-np.log(acc / npix))


def _filter_rows(tau, tau_eff):
    """In place: hcd_analysis.priya_p1d._apply_priya_filter on a block of rows, with the WHOLE file's tau_eff (the
    function itself would take the block's)."""
    from types import SimpleNamespace
    from hcd_analysis.priya_p1d import _filter_single_tau_complex
    stub = SimpleNamespace(nbins=tau.shape[1])
    for i in np.where(np.max(tau, axis=1) > 1.0e6)[0]:
        tau[i], _ = _filter_single_tau_complex(stub, tau[i], tau_eff, tau_thresh=1.0e6, thresh2=0.25)
    return tau


def _solve_scale(path, n_total, chunk, tau_eff, target_f):
    """The uniform scale s with <exp(-s tau_filtered)> = target_F over the file (Newton on chunked sums); used only
    when no cache row supplies the production scale (fake_spectra's mean_flux, tolerance 1e-5, needs the whole array)."""
    s_val = 1.0
    for _ in range(50):
        f = df = 0.0
        n = 0
        for s in range(0, n_total, chunk):
            t = _filter_rows(_read_rows(path, s, min(n_total, s + chunk)), tau_eff)
            ex = np.exp(-s_val * t)
            f += float(np.sum(ex))
            df += float(-np.sum(t * ex))
            n += t.size
        step = (f / n - target_f) / (df / n)
        s_val -= step
        if abs(step) < 1e-12 * abs(s_val):
            return s_val
    raise RuntimeError("mean-flux scale did not converge")


def chunked_tier_c(path, n_total, thresholds, absorber, dv, vmax, alpha, z_grid, tau_eff, scale, target_f, chunk,
                   batch_size=4096):
    """For every threshold: the catalogue's absorbers (production process_skewer_batch, batches of batch_size with
    global indices, as build_catalog) and the filtered and unfiltered Tier-C P1D (production compute_tier_c_p1d at
    the given scale and target flux), over the first n_total skewers in blocks of ``chunk`` rows, each read once.
    The per-bin P1D is a mean over sightlines, so the blocks combine exactly by their counts. Returns
    {threshold: dict(absorbers, kf, pf15, pu15, counts15)}."""
    from types import SimpleNamespace
    from hcd_analysis.catalog import process_skewer_batch
    from hcd_analysis.priya_p1d import compute_tier_c_p1d
    acc = {t: {"absorbers": [], "sf": None, "su": None, "n": None, "kf": None} for t in thresholds}
    for s in range(0, n_total, chunk):
        e = min(n_total, s + chunk)
        tau = _read_rows(path, s, e)
        tau_f = _filter_rows(tau.copy(), tau_eff)
        for thr in thresholds:
            new = []
            for b0 in range(0, e - s, batch_size):
                new.extend(process_skewer_batch(
                    tau[b0:b0 + batch_size], s + b0, dv, float(thr), absorber["merge_dv_kms"], absorber["min_pixels"],
                    absorber["b_init_kms"], tuple(absorber["b_bounds"]), absorber["tau_fit_cap"],
                    absorber["voigt_max_iter"], absorber["fast_mode"], min_log_nhi=absorber["min_log_nhi"]))
            a = acc[thr]
            a["absorbers"].extend(new)
            local = SimpleNamespace(absorbers=[SimpleNamespace(skewer_idx=x.skewer_idx - s, log_NHI=x.log_NHI)
                                               for x in new])
            kf, pf, nf, _, _ = compute_tier_c_p1d(tau_f, vmax, alpha_slope=alpha, z=z_grid, catalog=local,
                                                  external_scale=scale, external_target_F=target_f)
            _, pu, nu, _, _ = compute_tier_c_p1d(tau, vmax, alpha_slope=alpha, z=z_grid, catalog=local,
                                                 external_scale=scale, external_target_F=target_f)
            assert np.array_equal(nf, nu)
            if a["n"] is None:
                a["sf"], a["su"], a["n"], a["kf"] = np.zeros_like(pf), np.zeros_like(pu), np.zeros_like(nf), kf
            a["sf"] += nf[:, None] * pf
            a["su"] += nu[:, None] * pu
            a["n"] += nf
    out = {}
    for thr, a in acc.items():
        w = np.where(a["n"] > 0, a["n"], 1)[:, None]
        out[thr] = {"absorbers": a["absorbers"], "kf": a["kf"], "pf15": a["sf"] / w, "pu15": a["su"] / w,
                    "counts15": a["n"]}
    return out


def _check_catalogue(absorbers, nbins, sim, z, dv):
    """build_catalog's own checks on a finished catalogue: no overlapping entries, finite values."""
    from hcd_analysis.catalog import AbsorberCatalog, check_catalog_values, validate_catalog_entries
    if absorbers:
        validate_catalog_entries([a.skewer_idx for a in absorbers], [a.pix_start for a in absorbers],
                                 [a.pix_end for a in absorbers], nbins)
    cat = AbsorberCatalog(sim_name=sim, snap=-1, z=z, dv_kms=dv)
    cat.absorbers.extend(absorbers)
    check_catalog_values(cat)


def _scale_and_target(path, n_total, chunk, tau_eff, alpha, z_grid, cache_row):
    if cache_row is not None:
        return float(cache_row["scale"]), float(cache_row["target_F"]), "cache"
    from hcd_analysis.priya_p1d import obs_mean_tau
    target_f = float(np.exp(-alpha * obs_mean_tau(z_grid)))
    return _solve_scale(path, n_total, chunk, tau_eff, target_f), target_f, "solved"


def cache_row_for(cache, sim, z_grid, alpha):
    """The production cache row of (sim, z grid, alpha_slope): its scale and target flux, or None."""
    if cache is None:
        return None
    import h5py
    with h5py.File(cache, "r") as h:
        names = np.array([x.decode() if isinstance(x, bytes) else x for x in h["sim_name"][:]])
        rows = np.where((names == sim) & np.isclose(h["z_grid"][:], z_grid, atol=1e-6, rtol=0)
                        & np.isclose(h["alpha_slope"][:], alpha, atol=1e-12, rtol=0))[0]
        if len(rows) != 1:
            raise ValueError(f"{len(rows)} cache rows for {sim} z_grid {z_grid} alpha {alpha}")
        i = int(rows[0])
        return {"row": i, "scale": float(h["scale"][i]), "target_F": float(h["target_F"][i])}


def run_input(tag, path, thresholds, alpha, absorber, sim, out_dir, log, n_skewers=None, chunk=40000, cache=None):
    from hcd_analysis.io import pixel_dv_kms, read_header
    import build_emulator_cache_tau0 as bt                          # scripts/: the cache builder's own helpers

    todo = [t for t in thresholds if not (out_dir / f"result_{tag}_T{t:g}.npz").exists()]
    if not todo:
        return
    hdr = read_header(path)
    dv, vmax = pixel_dv_kms(hdr), bt.vmax_from_header(hdr)
    z_grid = bt._snap_z_to_priya_grid(float(hdr.redshift))
    n_total = int(hdr.n_skewers) if n_skewers is None else min(int(n_skewers), int(hdr.n_skewers))
    t0 = time.time()
    tau_eff = _global_tau_eff(path, n_total, chunk)
    scale, target_f, src = _scale_and_target(path, n_total, chunk, tau_eff, alpha, z_grid,
                                             cache_row_for(cache, sim, z_grid, alpha) if n_skewers is None else None)
    log(f"{tag}: {n_total} skewers, tau_eff {tau_eff:.6g}, scale {scale:.10g} ({src}), target_F {target_f:.10g}; "
        f"{time.time() - t0:.0f} s")
    t1 = time.time()
    res = chunked_tier_c(path, n_total, todo, absorber, dv, vmax, alpha, z_grid, tau_eff, scale, target_f, chunk)
    log(f"{tag}: {len(todo)} thresholds in {time.time() - t1:.0f} s")
    for thr in todo:
        r = res[thr]
        _check_catalogue(r["absorbers"], int(hdr.nbins), sim, float(hdr.redshift), dv)
        _save_once(out_dir / f"result_{tag}_T{thr:g}.npz", kf=r["kf"], pf15=r["pf15"], pu15=r["pu15"],
                   counts15=r["counts15"], log_nhi=np.array([a.log_NHI for a in r["absorbers"]], dtype=np.float64),
                   skewer_idx=np.array([a.skewer_idx for a in r["absorbers"]], dtype=np.int64),
                   target_F=np.array(target_f), scale=np.array(scale), z_grid=np.array(z_grid),
                   tau_eff=np.array(tau_eff), scale_source=np.array(src))
        log(f"{tag} T={thr:g}: {len(r['absorbers'])} absorbers")


def self_check(tag, path, n, reference, alpha, absorber, sim, out_dir, log):
    """The chunked path against the whole-array production path (build_catalog, _apply_priya_filter,
    compute_tier_c_p1d) on the first n skewers at the reference threshold; written once to self_check_<tag>.json."""
    dest = out_dir / f"self_check_{tag}.json"
    if dest.exists():
        return
    from hcd_analysis.catalog import build_catalog
    from hcd_analysis.io import pixel_dv_kms, read_header
    from hcd_analysis.priya_p1d import _apply_priya_filter, compute_tier_c_p1d
    import build_emulator_cache_tau0 as bt
    hdr = read_header(path)
    dv, vmax = pixel_dv_kms(hdr), bt.vmax_from_header(hdr)
    z_grid = bt._snap_z_to_priya_grid(float(hdr.redshift))
    tau = bt._read_tau(path, n_skewers=n)
    tau_f = _apply_priya_filter(tau.copy(), tau_thresh=1.0e6)
    cat = build_catalog(Path(path), sim_name=sim, snap=-1, z=float(hdr.redshift), dv_kms=dv,
                        tau_threshold=float(reference), merge_dv_kms=absorber["merge_dv_kms"],
                        min_pixels=absorber["min_pixels"], b_init=absorber["b_init_kms"],
                        b_bounds=tuple(absorber["b_bounds"]), tau_fit_cap=absorber["tau_fit_cap"],
                        voigt_max_iter=absorber["voigt_max_iter"], fast_mode=absorber["fast_mode"], n_workers=1,
                        min_log_nhi=absorber["min_log_nhi"], n_skewers=n)
    kf, pf, nf, target_f, scale = compute_tier_c_p1d(tau_f, vmax, alpha_slope=alpha, z=z_grid, catalog=cat)
    _, pu, _, _, _ = compute_tier_c_p1d(tau, vmax, alpha_slope=alpha, z=z_grid, catalog=cat,
                                        external_scale=scale, external_target_F=target_f)
    tau_eff = float(-np.log(np.mean(np.exp(-tau))))
    del tau, tau_f
    ch = chunked_tier_c(path, n, [reference], absorber, dv, vmax, alpha, z_grid, tau_eff, scale, target_f,
                        chunk=max(1, n // 4))[reference]
    key = lambda a: (a.skewer_idx, a.pix_start, a.pix_end, a.log_NHI)   # noqa: E731
    same_cat = sorted(map(key, cat.absorbers)) == sorted(map(key, ch["absorbers"]))
    ok = nf > 0

    def rel(x, y):
        return float(np.max(np.abs(x[ok] / y[ok] - 1.0)))
    doc = {"n_skewers": n, "threshold": reference, "catalogue_identical": bool(same_cat),
           "n_absorbers": len(cat.absorbers), "counts_identical": bool(np.array_equal(nf, ch["counts15"])),
           "max_rel_diff_filtered": rel(ch["pf15"], pf), "max_rel_diff_unfiltered": rel(ch["pu15"], pu)}
    dest.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    log(f"self-check {tag}: {doc}")


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
    ap.add_argument("--chunk", type=int, default=40000, help="skewers per block (memory: about 3 x chunk x nbins x 8 B)")
    ap.add_argument("--self-check", type=int, default=0, metavar="N",
                    help="compare the blocked path with the whole-array production path on the first N skewers")
    ap.add_argument("--only", action="append", default=None, metavar="TAG",
                    help="process only these inputs (parallel workers; the summary is written once all results exist)")
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
                "n_skewers": a.n_skewers, "chunk": a.chunk, "self_check": a.self_check,
                "scale": "the cache row's (production) scale and target flux" if a.anchor_cache and a.n_skewers is None
                         else "solved over the file (Newton, 1e-12)"}

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

    if a.only is not None and not set(a.only) <= set(inputs):
        ap.error(f"--only {a.only}: not among the inputs {sorted(inputs)}")
    for tag, p in sorted(inputs.items()):
        if a.only is not None and tag not in a.only:
            continue
        if a.self_check:
            self_check(tag, p, a.self_check, a.reference, a.alpha, absorber, a.sim, a.out, log)
        run_input(tag, p, thresholds, a.alpha, absorber, a.sim, a.out, log, n_skewers=a.n_skewers, chunk=a.chunk,
                  cache=a.anchor_cache)
    missing = [f"result_{t}_T{x:g}.npz" for t in sorted(inputs) for x in thresholds
               if not (a.out / f"result_{t}_T{x:g}.npz").exists()]
    if missing:
        log(f"summary deferred: {len(missing)} result files not yet written")
        return 0
    results = load_results(a.out, sorted(inputs), thresholds)
    summary = summarize(results, a.reference, a.n_k)
    doc = {"rows": summary, "verdict": verdict(summary, set(a.headline), a.reference)}
    if a.anchor_cache:
        doc["anchor"] = anchor(results, a.anchor_cache, a.sim, a.alpha, a.reference, a.n_k)
    checks = sorted(a.out.glob("self_check_*.json"))
    if checks:
        doc["self_check"] = {c.stem[len("self_check_"):]: json.loads(c.read_text()) for c in checks}
    tmp = a.out / f".summary.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, a.out / "summary.json")                         # derived from the result files; recomputable
    log(f"verdict: {doc['verdict']['reading']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
