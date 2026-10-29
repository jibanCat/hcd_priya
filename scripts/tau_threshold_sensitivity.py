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

Each threshold is compared with the reference threshold (``--reference``, default 100) over the first ``--n-k`` stored
native k bins (172: the LF emulator grid; k = 0 is never stored) by: the absorber count ratio per class, the template
amplitude ratio ``a = sum T T_ref / sum T_ref^2``, the shape residual ``s = rms(T - a T_ref) / rms(T_ref)``, the
largest fractional change of ``P_clean``, and, at low k within the data range (``--lowk-kmin`` 1e-3 to ``--lowk-kmax``
0.01 s/km), the largest template change relative to the template's largest absolute value there. An amplitude change
is what a free class amplitude in the likelihood absorbs; a shape change is not.

The reading of the ``--headline`` thresholds (default 50 and 200), over every class and input:

    negligible  every |a - 1| < 0.05, every s < 0.05 and every clean change < 1%
    modest      every |a - 1| <= 0.29 and every s <= 0.10   (0.29: the DESI/eBOSS LLS fractional prior width, 0.287)
    important   otherwise

The per-class readings apply the same thresholds to a and s only (the clean change belongs to no class). a weights the
k bins by T_ref^2, so a template that is steep at the lowest k (DLA at low redshift) is dominated by the first stored
bins, which can lie below the data range.

The run reads each input in blocks of ``--chunk`` skewers (twice: once for the mean flux the PRIYA filter uses), so its
memory stays near three blocks. The per-bin P1D is a mean over sightlines, so the blocks combine exactly by their
counts. ``--self-check N`` compares the blocked path with the whole-array production path on the first N skewers (the
catalogue, the per-bin counts and P1D, the filter's mean optical depth and the mean-flux scale), in blocks whose last
one is shorter (for N > 4), as in a full-file run. Once any input has a self-check, every input must have a passed
one: a failed or missing self-check makes the reading "invalid" and the exit status 2. The mean-flux scale is the
cache row's (production) value with ``--anchor-cache``, else solved over the skewers used (the whole file unless
``--n-skewers``).

Outputs, in ``--out``: ``run_settings.json`` (the inputs and every setting the results depend on, written once; a rerun
into the same directory with any of them different is refused, hcd_analysis/run_settings.py), one
``result_<tag>_T<threshold>.npz`` per input and threshold and one ``self_check_<tag>.json`` per input (each written
once; a rerun reuses them), and ``summary.json`` (recomputed from the result files on every complete run; it records
its own parameters and code commit). With ``--anchor-cache``, the reference threshold's counts and P1D are also
compared with the cache row of the same simulation, redshift and ``--alpha`` (an implementation check of the path).

Needs fake_spectra (run it in the environment that builds the emulator cache). Example:

  python scripts/tau_threshold_sensitivity.py --out NEW_DIR --sim SIM_NAME \\
      --input z2.2=/path/SPECTRA_021/lya_forest_spectra_grid_480.hdf5 --input z3.0=... \\
      --thresholds 30 50 100 200 300 --alpha ALPHA --anchor-cache CACHE.h5 --self-check 40000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
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
# The reading rule (module docstring).
NEGLIGIBLE_A, NEGLIGIBLE_S, NEGLIGIBLE_CLEAN = 0.05, 0.05, 0.01
MODEST_A, MODEST_S = 0.29, 0.10
# The self-check's tolerances: the blocked path must equal the whole-array path to rounding; the solved scale is
# compared with fake_spectra's mean_flux, which stops at a mean-flux tolerance of 1e-5.
SELF_CHECK_REL_TOL, SELF_CHECK_SCALE_TOL = 1e-10, 1e-4


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


def lowk_change(t, t_ref, kf, kmax, kmin=0.0):
    """Largest |t - t_ref| at kmin <= k < kmax, relative to the largest |t_ref| there (a pointwise ratio would blow up
    where a template crosses zero)."""
    kf = np.asarray(kf)
    m = (kf >= kmin) & (kf < kmax)
    den = float(np.max(np.abs(np.asarray(t_ref)[m]))) if m.any() else 0.0
    return float(np.max(np.abs(np.asarray(t)[m] - np.asarray(t_ref)[m]))) / den if den > 0 else float("nan")


def summarize(results, reference, n_k, lowk_kmax=0.01, lowk_kmin=1e-3):
    """Per (tag, threshold): the threshold, the sightline counts per class, the largest fractional clean change, and per
    class the absorber count ratio, a, s and the low-k change, all over the first n_k stored k bins (k = 0 is never
    stored). ``results`` maps (tag, threshold) to dicts with kf, counts15, pf15, pu15 and abs_counts
    (class -> count)."""
    out = {}
    for (tag, thr) in sorted(results):
        ref = results.get((tag, reference))
        if ref is None:
            raise KeyError(f"no reference threshold {reference} for {tag}")
        cur = results[(tag, thr)]
        sl = slice(0, n_k)
        pf, pu = collapse(cur["pf15"], cur["counts15"]), collapse(cur["pu15"], cur["counts15"])
        pf_r, pu_r = collapse(ref["pf15"], ref["counts15"]), collapse(ref["pu15"], ref["counts15"])
        t, t_r = templates(pf[:, sl], pu[:, sl]), templates(pf_r[:, sl], pu_r[:, sl])
        kf = np.asarray(cur["kf"])[sl]
        row = {"tag": tag, "threshold": float(thr), "n_k": int(kf.size),
               "clean_max_frac_change": float(np.max(np.abs(pf[0, sl] / pf_r[0, sl] - 1.0))),
               "sightlines": {c: int(n) for c, n in zip(CLASSES, (cur["counts15"][lo:hi].sum()
                                                                  for lo, hi in COARSE_RANGES))}}
        for c in TEMPLATE_CLASSES:
            a, s = compare(t[c], t_r[c])
            n, n_r = cur["abs_counts"][c], ref["abs_counts"][c]
            row[c] = {"count_ratio": (n / n_r) if n_r else float("nan"), "amplitude_ratio": a, "shape_residual": s,
                      "lowk_change": lowk_change(t[c], t_r[c], kf, lowk_kmax, lowk_kmin)}
        out[f"{tag}|T{thr:g}"] = row
    return out


def _reading(a, s, clean=0.0):
    if a < NEGLIGIBLE_A and s < NEGLIGIBLE_S and clean < NEGLIGIBLE_CLEAN:
        return "negligible"
    return "modest" if (a <= MODEST_A and s <= MODEST_S) else "important"


def verdict(summary, headline, reference):
    """The reading rule (module docstring) over the headline thresholds, all classes and inputs; per class too."""
    rows = [v for v in summary.values() if v["threshold"] in headline]
    if not rows:
        raise ValueError(f"no summary rows for the headline thresholds {sorted(headline)}")
    values = [v["clean_max_frac_change"] for v in rows] + [v[c][k] for v in rows for c in TEMPLATE_CLASSES
                                                           for k in ("amplitude_ratio", "shape_residual")]
    if not all(np.isfinite(x) for x in values):
        bad = sorted(f"{v['tag']}|T{v['threshold']:g}" for v in rows
                     if not all(np.isfinite(x) for x in [v["clean_max_frac_change"]] + [
                         v[c][k] for c in TEMPLATE_CLASSES for k in ("amplitude_ratio", "shape_residual")]))
        return {"headline_thresholds": sorted(headline), "reference": reference,
                "reading": f"invalid: undefined comparison (an empty class or clean set) in {bad}",
                "per_class": {c: {"reading": "invalid"} for c in TEMPLATE_CLASSES}}
    worst_a = max(abs(v[c]["amplitude_ratio"] - 1.0) for v in rows for c in TEMPLATE_CLASSES)
    worst_s = max(v[c]["shape_residual"] for v in rows for c in TEMPLATE_CLASSES)
    worst_clean = max(v["clean_max_frac_change"] for v in rows)
    per_class = {}
    for c in TEMPLATE_CLASSES:
        wa = max(abs(v[c]["amplitude_ratio"] - 1.0) for v in rows)
        ws = max(v[c]["shape_residual"] for v in rows)
        per_class[c] = {"max_abs_a_minus_1": wa, "max_s": ws, "max_lowk_change": max(v[c]["lowk_change"] for v in rows),
                        "reading": _reading(wa, ws)}
    return {"headline_thresholds": sorted(headline), "reference": reference,
            "reading": _reading(worst_a, worst_s, worst_clean),
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


def _link_once(path, write):
    """Write through ``write(fileobj)`` into a fresh temporary file beside ``path`` and link it into place; an existing
    file is never replaced (FileExistsError)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def _save_once(path, **arrays):
    _link_once(path, lambda fh: np.savez(fh, **arrays))


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
    when no cache row supplies the production scale (fake_spectra's mean_flux, tolerance 1e-5, needs the whole
    array)."""
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
    """For every threshold: the catalogue's absorbers (production process_skewer_batch with global skewer indices; it
    treats every skewer on its own, so where the batches start does not matter) and the filtered and unfiltered
    Tier-C P1D (production compute_tier_c_p1d at the given scale and target flux), over the first n_total skewers in
    blocks of ``chunk`` rows, each read once.
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


def _self_check_block(n):
    """The self-check's block size: about n / 4, chosen so that the last block is shorter (as in a full-file run)
    whenever n > 4."""
    if n <= 4:
        return 1
    block = n // 4 + 1
    while n % block == 0:
        block += 1
    return block


def self_check_passed(doc):
    """The self-check's criterion: identical catalogue and counts, P1D and tau_eff within SELF_CHECK_REL_TOL, the solved
    scale within SELF_CHECK_SCALE_TOL. A missing or NaN entry fails."""
    try:
        return bool(doc["catalogue_identical"] is True and doc["counts_identical"] is True
                    and all(float(doc[k]) <= SELF_CHECK_REL_TOL
                            for k in ("max_rel_diff_filtered", "max_rel_diff_unfiltered", "tau_eff_rel_diff"))
                    and float(doc["scale_solved_rel_diff"]) <= SELF_CHECK_SCALE_TOL)
    except (KeyError, TypeError, ValueError):
        return False


def self_check(tag, path, n, reference, alpha, absorber, sim, out_dir, log):
    """The blocked path against the whole-array production path (build_catalog, _apply_priya_filter,
    compute_tier_c_p1d with fake_spectra's mean_flux) on the first n skewers at the reference threshold: the
    catalogue, the per-bin counts and P1D, the filter's mean optical depth and the solved mean-flux scale. Written once
    to self_check_<tag>.json, with ``passed``."""
    dest = out_dir / f"self_check_{tag}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    from hcd_analysis.catalog import build_catalog
    from hcd_analysis.io import pixel_dv_kms, read_header
    from hcd_analysis.priya_p1d import _apply_priya_filter, compute_tier_c_p1d
    import build_emulator_cache_tau0 as bt
    hdr = read_header(path)
    dv, vmax = pixel_dv_kms(hdr), bt.vmax_from_header(hdr)
    z_grid = bt._snap_z_to_priya_grid(float(hdr.redshift))
    tau = bt._read_tau(path, n_skewers=n)
    tau_eff = float(-np.log(np.mean(np.exp(-tau))))                  # what _apply_priya_filter computes
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
    del tau, tau_f
    block = _self_check_block(n)
    tau_eff_blocked = _global_tau_eff(path, n, block)
    scale_solved = _solve_scale(path, n, block, tau_eff_blocked, target_f)
    ch = chunked_tier_c(path, n, [reference], absorber, dv, vmax, alpha, z_grid, tau_eff_blocked, scale, target_f,
                        chunk=block)[reference]
    key = lambda a: (a.skewer_idx, a.pix_start, a.pix_end, a.log_NHI)   # noqa: E731
    ok = nf > 0

    def rel(x, y):
        return float(np.max(np.abs(x[ok] / y[ok] - 1.0)))
    doc = {"n_skewers": n, "threshold": reference, "block": block, "n_absorbers": len(cat.absorbers),
           "catalogue_identical": sorted(map(key, cat.absorbers)) == sorted(map(key, ch["absorbers"])),
           "counts_identical": bool(np.array_equal(nf, ch["counts15"])),
           "max_rel_diff_filtered": rel(ch["pf15"], pf), "max_rel_diff_unfiltered": rel(ch["pu15"], pu),
           "tau_eff_rel_diff": abs(tau_eff_blocked / tau_eff - 1.0),
           "scale_solved_rel_diff": abs(scale_solved / scale - 1.0),
           "tolerances": {"p1d_and_tau_eff": SELF_CHECK_REL_TOL, "scale": SELF_CHECK_SCALE_TOL}}
    doc["passed"] = self_check_passed(doc)
    _link_once(dest, lambda fh: fh.write(
        (json.dumps(_finite_or_none(doc), indent=1, sort_keys=True, allow_nan=False) + "\n").encode()))
    log(f"self-check {tag}: {doc}")
    return doc


def load_results(out_dir, tags, thresholds):
    """The result files, refusing an input whose thresholds were computed at different scale, target flux, filter mean
    optical depth, z grid or k grid (results of one input must come from one pass)."""
    res = {}
    for tag in tags:
        for thr in thresholds:
            with np.load(out_dir / f"result_{tag}_T{thr:g}.npz") as d:
                res[(tag, thr)] = {k: d[k] for k in d.files}
            res[(tag, thr)]["abs_counts"] = class_counts(res[(tag, thr)]["log_nhi"])
        first = res[(tag, thresholds[0])]
        for k in ("scale", "target_F", "tau_eff", "z_grid", "kf"):
            same = (lambda x, y: np.array_equal(x, y)) if k in ("z_grid", "kf") else \
                (lambda x, y: x is not None and np.allclose(x, y, rtol=1e-12, atol=0))   # block size: last bits only
            if k in first and not all(same(res[(tag, t)].get(k), first[k]) for t in thresholds):
                raise ValueError(f"{tag}: the result files disagree on {k}; they are not from one pass")
    return res


def anchor(results, cache, sim, alpha, reference, n_k):
    """The reference threshold against the cache row of the same simulation, z grid and alpha, over the first n_k
    stored k bins."""
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
                a, b = h[cache_key][i][ok, :m], mine[ok, :m]
                return float(np.max(np.abs(b / a - 1.0)))
            out[tag] = {"cache_row": i, "count_diff_by_bin": (r["counts15"] - c_counts).astype(int).tolist(),
                        "max_rel_diff_filtered": rel("P_tier_c_filtered", r["pf15"]),
                        "max_rel_diff_unfiltered": rel("P_tier_c", r["pu15"]),
                        "scale_rel_diff": float(r["scale"] / h["scale"][i] - 1.0)}
    return out


def _cache_identity(path):
    if path is None:
        return None
    st = os.stat(path)
    return {"path": str(Path(path).resolve()), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def _dirty():
    d = _git("--no-optional-locks", "status", "--porcelain", "--untracked-files=no")
    return None if d is None else bool(d)


def _finite_or_none(obj):
    """NaN and infinities as null, so summary.json is strict JSON."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite_or_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_or_none(v) for v in obj]
    return obj


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sim", required=True, help="simulation name (catalogue label; cache lookup for --anchor-cache)")
    ap.add_argument("--input", action="append", required=True, metavar="TAG=PATH",
                    help="a raw fake_spectra grid file, tagged (repeatable)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[30.0, 50.0, 100.0, 200.0, 300.0],
                    help="at most 6 significant digits each (they name the result files)")
    ap.add_argument("--reference", type=float, default=100.0)
    ap.add_argument("--headline", type=float, nargs="+", default=[50.0, 200.0],
                    help="thresholds the reading rule is applied to")
    ap.add_argument("--alpha", type=float, required=True, help="mean-flux scaling alpha_slope")
    ap.add_argument("--n-k", type=int, default=172, help="stored native k bins compared (172: the LF emulator grid)")
    ap.add_argument("--lowk-kmax", type=float, default=0.01, help="upper k (s/km) of the low-k change")
    ap.add_argument("--lowk-kmin", type=float, default=1e-3,
                    help="lower k (s/km) of the low-k change (1e-3: the lower edge of the data range)")
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "default.yaml",
                    help="every catalogue setting except tau_threshold is read from here")
    ap.add_argument("--anchor-cache", type=Path, default=None)
    ap.add_argument("--n-skewers", type=int, default=None,
                    help="only the first N skewers of each input (a quick look; the anchor then does not apply)")
    ap.add_argument("--chunk", type=int, default=40000,
                    help="skewers per block (memory: about 3 x chunk x nbins x 8 B)")
    ap.add_argument("--self-check", type=int, default=0, metavar="N",
                    help="compare the blocked path with the whole-array production path on the first N skewers")
    ap.add_argument("--only", action="append", default=None, metavar="TAG",
                    help="process only these inputs (parallel workers; the summary is written once all results exist)")
    a = ap.parse_args(argv)

    thresholds = sorted(set(a.thresholds))
    for t in thresholds + [a.reference] + list(a.headline):
        if not (np.isfinite(t) and t > 0 and float(f"{t:g}") == t):
            ap.error(f"threshold {t!r}: give a positive value with at most 6 significant digits")
    if a.reference not in thresholds or not set(a.headline) <= set(thresholds):
        ap.error("--reference and every --headline threshold must be among --thresholds")
    for name in ("n_k", "chunk"):
        if getattr(a, name) < 1:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    if a.self_check < 0 or (a.n_skewers is not None and a.n_skewers < 1):
        ap.error("--self-check must be 0 (off) or positive, and --n-skewers positive")
    if a.n_skewers is not None and a.anchor_cache is not None:
        ap.error("--anchor-cache compares whole files; it cannot be combined with --n-skewers")
    inputs = {}
    for item in a.input:
        tag, _, p = item.partition("=")
        if not tag or not p or tag in inputs:
            ap.error(f"--input {item!r}: expected a unique TAG=PATH")
        inputs[tag] = Path(p).resolve()
    if a.only is not None and not set(a.only) <= set(inputs):
        ap.error(f"--only {a.only}: not among the inputs {sorted(inputs)}")

    import yaml
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    cfg_bytes = a.config.read_bytes()
    absorber = dict(yaml.safe_load(cfg_bytes)["absorber"])
    absorber.pop("tau_threshold", None)
    # Compared settings: the inputs and everything the result files depend on (the anchor cache by size and mtime, as
    # it supplies the scale; the config file by its hash, deliberately conservative). The block size (rounding only),
    # the self-check size and the summary's parameters (reference, headline, n_k, low-k cut) go to summary.json.
    settings = {"tool": "tau_threshold_sensitivity/3", "sim": a.sim,
                "inputs": {t: _file_identity(p) for t, p in sorted(inputs.items())},
                "thresholds": thresholds, "alpha_slope": a.alpha, "absorber_except_threshold": absorber,
                "config": {"path": str(a.config.resolve()), "sha256": hashlib.sha256(cfg_bytes).hexdigest()},
                "anchor_cache": _cache_identity(a.anchor_cache),
                "n_skewers": a.n_skewers,
                "scale": "the cache row's (production) scale and target flux" if a.anchor_cache and a.n_skewers is None
                         else "solved over the skewers used (Newton, 1e-12)"}

    def info():
        import h5py
        try:
            import fake_spectra
            fs = getattr(fake_spectra, "__version__", fake_spectra.__file__)
        except Exception:
            fs = None
        return {"code_commit": _git("rev-parse", "HEAD") or "unknown", "code_dirty": _dirty(),
                "python": platform.python_version(), "numpy": np.__version__, "h5py": h5py.__version__,
                "fake_spectra": fs}

    too_big = [t for t, ident in settings["inputs"].items() if a.self_check > ident["n_skewers"]]
    if too_big:
        ap.error(f"--self-check {a.self_check} exceeds the skewers of {too_big}")
    record_or_verify(a.out, settings, info=info)
    with open(a.out / "log.txt", "a") as logf:
        return _run(a, inputs, thresholds, absorber, logf)


def _run(a, inputs, thresholds, absorber, logf):

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

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
    summary = summarize(results, a.reference, a.n_k, a.lowk_kmax, a.lowk_kmin)
    doc = {"rows": summary, "verdict": verdict(summary, set(a.headline), a.reference),
           "summary_settings": {"reference": a.reference, "headline": sorted(a.headline), "n_k": a.n_k,
                                "lowk_kmin": a.lowk_kmin, "lowk_kmax": a.lowk_kmax,
                                "code_commit": _git("rev-parse", "HEAD") or "unknown", "code_dirty": _dirty()}}
    if a.anchor_cache:
        doc["anchor"] = anchor(results, a.anchor_cache, a.sim, a.alpha, a.reference, a.n_k)
    checks = {c.stem[len("self_check_"):]: json.loads(c.read_text()) for c in sorted(a.out.glob("self_check_*.json"))}
    if checks:
        doc["self_check"] = checks
    failed = sorted(t for t, c in checks.items()
                    if not self_check_passed(c) or c.get("threshold") != a.reference)
    if a.self_check or checks:                                      # once any input is checked, every input must be
        failed += sorted(set(inputs) - set(checks))
    doc["verdict"]["valid"] = not failed
    if failed:
        doc["verdict"]["reading"] = f"invalid: self-check failed or missing for {sorted(failed)}"
        for c in doc["verdict"]["per_class"].values():
            c["reading"] = "invalid"
    fd, tmp = tempfile.mkstemp(dir=a.out, prefix=".summary.json.", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(_finite_or_none(doc), indent=1, sort_keys=True, allow_nan=False) + "\n")
    os.chmod(tmp, 0o644)
    os.replace(tmp, a.out / "summary.json")                         # derived from the result files; recomputable
    log(f"verdict: {doc['verdict']['reading']}")
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
