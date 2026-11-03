"""The dN/dX-to-w_c map as a stamped product, format ``hcd_priya.wc_map/1``.

The class fractions w_c that the public predictor returns come from Head A's dN/dX through
``dndx_wc.w_c_corrected(dndx, Xbar(z), z, delta_map)``, which needs two count-fitted objects:

  * ``delta_c(z)``: the per-class correction of the telescoping-Poisson map, a degree-2 polynomial in z per class, fitted
    to ``w_counted / w_M0 - 1`` per snapshot block with the geometric path ``Xbar = X_tot / N_sl`` (N_sl from the
    counts); the estimator of ``scripts/calibrate_delta_c.py`` (``collect`` and ``fit``);
  * ``Xbar(z)``: the mean absorption path per sightline, a degree-2 polynomial in z fitted to the snapshot groups with
    2.1 <= z <= 4.7, N_sl from the telescoping clean fraction; the estimator of ``closure_legb.hcd_xbar_polyfit``.

Both are refitted here from a release training cache (format 3.4, read through the strict ``data.load_cache``) and
written to a small JSON file that embeds a ``hcd_priya.lineage/1`` record (kind ``wc_map``) naming the cache by its
sha256, with the cache's finder ids and release eligibility. A model bundle (``safe_h5``, format ``hcd_priya.h5/2``)
lists the file, and its manifest's lineage block pins its sha256.

File layout (UTF-8 JSON, finite numbers only)::

    {"format": "hcd_priya.wc_map/1",
     "class_order": ["clean", "LLS", "subDLA", "DLA"],
     "delta_c": {"coeffs": [[a2, a1, a0] x 4], "deg": 2, "z_clamp": [z_lo, z_hi], "resid_std": [4], "resid_max": [4],
                 "mean": [4], "n_blocks": N, "n_sims": S, "z_min": .., "z_max": ..},
     "xbar": {"coeffs": [b2, b1, b0], "deg": 2, "z_fit": [2.1, 4.7], "n_groups": G},
     "lineage": {the record}}

The record's ``product.payload_sha256`` is the sha256 of the canonical JSON of the document without ``lineage``, so an
edit of any number is refused. The predictor reads the file only through ``read_wc_map`` / ``parse_wc_map``."""
from __future__ import annotations

import hashlib
import json
import os
from typing import NamedTuple

import numpy as np

from .. import lineage as L

FORMAT = "hcd_priya.wc_map/1"
KIND = "wc_map"
KIND_VERSION = 1
DEG = 2
CLASS_ORDER = ("clean", "LLS", "subDLA", "DLA")
DELTA_C_Z_CLAMP = (2.0, 5.4)          # the PRIYA snapshot range: delta_c is never extrapolated beyond it
XBAR_Z_FIT = (2.1, 4.7)               # the snapshot groups the Xbar(z) fit uses (closure_legb.hcd_xbar_polyfit)
MAX_BYTES = 1 << 20
_TOP = {"format", "class_order", "delta_c", "xbar", "lineage"}
_DC_KEYS = {"coeffs", "deg", "z_clamp", "resid_std", "resid_max", "mean", "n_blocks", "n_sims", "z_min", "z_max"}
_XB_KEYS = {"coeffs", "deg", "z_fit", "n_groups"}
ESTIMATORS = {
    "delta_c": "w_counted / w_M0(dN/dX, X_tot / N_sl) - 1 per snapshot block (one representative row; N_sl from the "
               "coarse counts), one unweighted degree-2 np.polyfit in z per class over every block; the residual std "
               "and max per class (scripts/calibrate_delta_c.py collect + fit)",
    "xbar": "Xbar = X_tot / N_sl per snapshot group, N_sl = sum(dN/dX) X_tot / mu_sum with mu_sum = -log(median "
            "clean fraction clipped at 1e-6), groups with 2.1 <= z <= 4.7, one degree-2 np.polyfit in z "
            "(closure_legb.hcd_xbar_polyfit)",
}


class WcMapError(ValueError):
    """A w_c map file or its record is malformed, edited, not release-eligible, or not the one expected."""


# ------------------------------------------------------------------------------------------------- the estimators
def delta_c_blocks(d):
    """Per snapshot block of a loaded cache ``d`` (``data.load_cache``): ``(z, w_counted (N,4), w_M0 (N,4), sims)``,
    exactly as ``scripts/calibrate_delta_c.py`` ``collect`` builds them from one shard."""
    import jax.numpy as jnp
    from .dndx_wc import mu_from_dndx, w_c_from_mu
    z, wc, wm, sims = [], [], [], []
    g = d["snap_group_idx"]
    for gi in np.unique(g):
        r = int(np.where(g == gi)[0][0])          # counts are tau0-invariant -> one rep row
        cc = d["coarse_counts"][r].astype(float)  # (4,) clean,LLS,subDLA,DLA
        n_sl = cc.sum()
        if n_sl <= 0:
            continue
        w_counted = cc / n_sl
        dndx = d["snap_dNdX"][gi]                  # (3,) LLS,subDLA,DLA
        xbar = d["snap_total_path_dX"][gi] / n_sl
        w_m0 = np.asarray(w_c_from_mu(mu_from_dndx(jnp.asarray(dndx), jnp.asarray(xbar))))
        z.append(float(d["z_grid"][r])); wc.append(w_counted); wm.append(w_m0)
        sims.append(d["sim_name"][r])
    return np.array(z), np.array(wc), np.array(wm), np.array(sims, dtype=object)


def fit_delta_c(z, wc, wm, deg=DEG):
    """``(coeffs (4, deg+1), delta (N,4), resid_std (4,), resid_max (4,))``: ``scripts/calibrate_delta_c.py`` ``fit``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(wm > 0, wc / wm - 1.0, 0.0)   # (N,4)
    coeffs = np.zeros((4, deg + 1))
    resid_std = np.zeros(4)
    resid_max = np.zeros(4)
    for ci in range(4):
        c = np.polyfit(z, delta[:, ci], deg)
        coeffs[ci] = c
        resid = delta[:, ci] - np.polyval(c, z)
        resid_std[ci] = resid.std()
        resid_max[ci] = np.abs(resid).max()
    return coeffs, delta, resid_std, resid_max


def fit_xbar(d):
    """``(coeffs (3,), n_groups)``: the Xbar(z) fit of ``closure_legb.hcd_xbar_polyfit`` (the same numpy operations in
    the same order), plus the number of groups it used."""
    wc = np.asarray(d["w_c_cache"])                      # (R,4) clean,LLS,subDLA,DLA
    zrow = np.asarray(d["z_grid"])
    gid = np.asarray(d["snap_group_idx"]); Xtot = np.asarray(d["snap_total_path_dX"])
    dndx = np.asarray(d["snap_dNdX"]); Ng = dndx.shape[0]
    zg = np.array([zrow[gid == g][0] for g in range(Ng)])
    wc_g = np.array([np.nanmedian(wc[gid == g], axis=0) for g in range(Ng)])
    mu_sum = -np.log(np.clip(wc_g[:, 0], 1e-6, None))
    Nsl = np.where(mu_sum > 0, dndx.sum(axis=1) * Xtot / mu_sum, np.nan)
    Xbar = Xtot / Nsl
    ok = np.isfinite(Xbar) & (zg >= XBAR_Z_FIT[0]) & (zg <= XBAR_Z_FIT[1])
    return np.polyfit(zg[ok], Xbar[ok], DEG), int(ok.sum())


def refit(d):
    """The map's payload (everything but the record) from a loaded cache ``d``."""
    z, wc, wm, sims = delta_c_blocks(d)
    if z.size <= DEG:
        raise WcMapError(f"{z.size} snapshot blocks: too few for a degree-{DEG} delta_c fit")
    coeffs, delta, resid_std, resid_max = fit_delta_c(z, wc, wm, DEG)
    xcf, n_groups = fit_xbar(d)
    if n_groups <= DEG:
        raise WcMapError(f"{n_groups} snapshot groups in {list(XBAR_Z_FIT)}: too few for a degree-{DEG} Xbar fit")
    doc = {"format": FORMAT, "class_order": list(CLASS_ORDER),
           "delta_c": {"coeffs": coeffs.tolist(), "deg": DEG, "z_clamp": list(DELTA_C_Z_CLAMP),
                       "resid_std": resid_std.tolist(), "resid_max": resid_max.tolist(),
                       "mean": delta.mean(axis=0).tolist(), "n_blocks": int(z.size),
                       "n_sims": len(set(map(str, sims))), "z_min": float(z.min()), "z_max": float(z.max())},
           "xbar": {"coeffs": np.asarray(xcf, dtype=np.float64).tolist(), "deg": DEG, "z_fit": list(XBAR_Z_FIT),
                    "n_groups": n_groups}}
    _check_payload(doc, "refit")
    return doc


_VIEW_ROW_KEYS = ("sim_name", "z_grid", "params", "x", "coarse_counts", "w_c_cache", "alpha_idx", "tau0")
_VIEW_GROUP_KEYS = ("snap_dNdX", "snap_total_path_dX", "snap_f_nhi", "snap_n_absorbers")


def training_view(d, train_sims):
    """The rows and snapshot groups of the simulations ``train_sims`` only, as a cache-like dict with the groups
    re-indexed 0..G'-1: the keys the w_c estimators read and nothing of any other simulation, so a group-level fit
    given the view cannot reach a held-out group (``fit_xbar`` loops over every group of the dict it is given).
    ``source_groups`` and ``source_rows`` map back to ``d``."""
    sims = sorted({str(s) for s in train_sims})
    sim = np.asarray(d["sim_name"]).astype(str)
    have = set(sim.tolist())
    if not sims or not set(sims) <= have:
        raise WcMapError(f"the training simulations must be a non-empty subset of the cache's: "
                         f"{sorted(set(sims) - have)[:5]} are not in it")
    rows = np.flatnonzero(np.isin(sim, sims))
    old_g = np.asarray(d["snap_group_idx"])[rows]
    groups = np.unique(old_g)
    view = {k: np.asarray(d[k])[rows] for k in _VIEW_ROW_KEYS if k in d}
    view["sim_name"] = sim[rows]
    view["snap_group_idx"] = np.searchsorted(groups, old_g).astype(np.int32)
    for k in _VIEW_GROUP_KEYS:
        if k in d:
            view[k] = np.asarray(d[k])[groups]
    view["source_groups"], view["source_rows"] = groups, rows
    return view


def refit_subset(d, train_sims):
    """The map's payload refitted on the simulations ``train_sims`` only (rule ``wcmap_refit_subset/1``: ``refit`` of
    ``training_view``), so a cross-validation fold's w_c map never sees its held-out simulations. With every
    simulation of the cache it is ``refit(d)`` byte for byte. The map file format is unchanged: the payload records
    ``delta_c.n_sims``; which simulations were used is recorded by the research bundle
    (``provenance.research.wc_map``), not in the map file."""
    return refit(training_view(d, train_sims))


# ---------------------------------------------------------------------------------------------- record and writing
def payload_sha256(doc):
    """sha256 of the canonical JSON of the document without its ``lineage``."""
    body = {k: v for k, v in doc.items() if k != "lineage"}
    return hashlib.sha256(L.canonical_json(body).encode("ascii")).hexdigest()


def settings():
    return L.canonicalise({"deg": DEG, "delta_c_z_clamp": list(DELTA_C_Z_CLAMP), "xbar_z_fit": list(XBAR_Z_FIT),
                           "estimators": ESTIMATORS})


def make_record(doc, cache_record, cache_identity, *, code, allowed_finder_ids=None, cache_name=None):
    """The ``wc_map`` lineage record of the payload ``doc``: its one input is the training cache (``cache_identity``
    {sha256, size}; ``cache_record`` its verified ``tau0_cache`` record, whose finder ids and eligibility are copied;
    ``cache_name`` a plain file name recorded as information, never a path). Eligibility follows lineage.eligibility."""
    from ..catalog import RELEASE_FINDER_IDS
    allowed = RELEASE_FINDER_IDS if allowed_finder_ids is None else allowed_finder_ids
    if cache_record.get("kind") != "tau0_cache":
        raise WcMapError(f"the w_c map is fitted to a tau0_cache, not a {cache_record.get('kind')!r}")
    info = {"file": os.path.basename(str(cache_name))} if cache_name else None
    entry = L.input_entry("cache", "tau0_cache", {"sha256": cache_identity["sha256"], "size": cache_identity["size"]},
                          record=cache_record, info=info)
    entry["info"] = dict(entry.get("info") or {}, lineage_record_sha256=L.record_sha256(cache_record))
    product = {"payload_sha256": payload_sha256(doc), "n_blocks": doc["delta_c"]["n_blocks"],
               "n_sims": doc["delta_c"]["n_sims"], "n_xbar_groups": doc["xbar"]["n_groups"]}
    return L.build_record(KIND, KIND_VERSION, inputs=[entry], finder_ids=None, settings=settings(), code=code,
                          allowed_finder_ids=allowed, product=product)


def to_bytes(doc, record):
    """The file's bytes: canonical JSON of the payload with the record (sorted keys, ASCII)."""
    full = dict(doc, lineage=record)
    _check_payload(full, "w_c map")
    return L.canonical_json(full).encode("ascii")


def write_wc_map(path, doc, record):
    """Write the map to the NEW file ``path`` (never overwritten; published only when complete). Returns its sha256."""
    from . import safe_io as S
    if os.path.lexists(path):
        raise WcMapError(f"{path} exists: a w_c map is never overwritten")
    raw = to_bytes(doc, record)
    parse_wc_map(raw, str(path), require_release_eligible=False)        # what is written must read back
    return S._write_exclusive(str(path), [raw])


def build(cache_path, out_path, *, expected_cache_sha256=None, require_release_eligible=True):
    """Refit the map from the release cache at ``cache_path`` (strict ``data.load_cache``: format 3.4, lineage record,
    payload digest, release eligibility, schema; the file is hashed at load and must hash to
    ``expected_cache_sha256`` when given) and write it to ``out_path`` with its record. The code identity is the
    current checkout's; with ``require_release_eligible`` an ineligible record (a dirty tree, an unknown commit) is
    refused before anything is written. Returns ``(record, sha256 of the written file)``."""
    from .data import load_cache
    if os.path.lexists(out_path):
        raise WcMapError(f"{out_path} exists: a w_c map is never overwritten")
    d = load_cache(cache_path)
    if expected_cache_sha256 is not None and d["cache_sha256"] != expected_cache_sha256:
        raise WcMapError(f"{cache_path}: sha256 {d['cache_sha256']} is not the expected {expected_cache_sha256}")
    doc = refit(d)
    ident = {"sha256": d["cache_sha256"], "size": int(os.stat(cache_path).st_size)}
    rec = make_record(doc, d["lineage"], ident, code=L.code_identity(), cache_name=os.path.basename(str(cache_path)))
    if require_release_eligible and not rec["release_eligible"]:
        raise WcMapError("the w_c map would not be release-eligible (nothing written): "
                         + "; ".join(rec["ineligible_reasons"]))
    return rec, write_wc_map(out_path, doc, rec)


# ------------------------------------------------------------------------------------------------------- reading
class WcMap(NamedTuple):
    """A parsed, checked w_c map. Arrays are host numpy float64."""
    delta_c_coeffs: np.ndarray         # (4, 3) np.polyval order, class order CLASS_ORDER
    delta_c_z_clamp: tuple
    delta_c_resid_std: np.ndarray      # (4,) fit residual std (information; the predictor applies no prior)
    xbar_coeffs: np.ndarray            # (3,) np.polyval order
    xbar_z_fit: tuple
    record: dict
    sha256: str
    doc: dict

    @property
    def cache_sha256(self):
        return L.find_input(self.record, "cache")["sha256"]

    def delta_c_map(self):
        from .dndx_wc import DeltaCMap
        return DeltaCMap(coeffs=np.array(self.delta_c_coeffs, dtype=np.float64), z_lo=float(self.delta_c_z_clamp[0]),
                         z_hi=float(self.delta_c_z_clamp[1]), label=f"wc_map {self.sha256}")

    def xbar(self, z):
        """Xbar(z) = np.polyval(xbar_coeffs, z) (not clamped: outside ``xbar_z_fit`` it is an extrapolation)."""
        return np.polyval(self.xbar_coeffs, np.asarray(z, dtype=np.float64))

    def w_c(self, dndx, z):
        """(..., 4) class fractions (clean, LLS, subDLA, DLA) from dN/dX (..., 3) at z: ``dndx_wc.w_c_corrected``
        with this map's Xbar(z) and delta_c. Host numpy in, host numpy out."""
        import jax.numpy as jnp
        from .dndx_wc import w_c_corrected
        z = np.asarray(z, dtype=np.float64)
        out = w_c_corrected(jnp.asarray(np.asarray(dndx, dtype=np.float64)), jnp.asarray(self.xbar(z)),
                            jnp.asarray(z), self.delta_c_map())
        return np.asarray(out, dtype=np.float64)


def _num_list(x, shape, where, what):
    a = np.asarray(x, dtype=object)
    if a.shape != shape or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in a.ravel()):
        raise WcMapError(f"{where}: {what} must be numbers of shape {shape}")
    a = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise WcMapError(f"{where}: {what} is not finite")
    return a


def _count(x, where, what):
    if not (isinstance(x, int) and not isinstance(x, bool) and x >= 0):
        raise WcMapError(f"{where}: {what} must be a non-negative integer")
    return x


def _check_payload(doc, where):
    if not isinstance(doc, dict):
        raise WcMapError(f"{where}: not a JSON object")
    keys = set(doc)
    if keys - _TOP or not (_TOP - {"lineage"}) <= keys:
        raise WcMapError(f"{where}: keys {sorted(keys)}, expected {sorted(_TOP)}")
    if doc["format"] != FORMAT:
        raise WcMapError(f"{where}: format {doc['format']!r}, this reader reads {FORMAT}")
    if doc["class_order"] != list(CLASS_ORDER):
        raise WcMapError(f"{where}: class_order {doc['class_order']!r}, expected {list(CLASS_ORDER)}")
    dc, xb = doc["delta_c"], doc["xbar"]
    if not isinstance(dc, dict) or set(dc) != _DC_KEYS:
        raise WcMapError(f"{where}: delta_c keys {sorted(dc) if isinstance(dc, dict) else dc!r}, expected "
                         f"{sorted(_DC_KEYS)}")
    if not isinstance(xb, dict) or set(xb) != _XB_KEYS:
        raise WcMapError(f"{where}: xbar keys {sorted(xb) if isinstance(xb, dict) else xb!r}, expected "
                         f"{sorted(_XB_KEYS)}")
    if dc["deg"] != DEG or xb["deg"] != DEG:
        raise WcMapError(f"{where}: polynomial degree must be {DEG}")
    _num_list(dc["coeffs"], (4, DEG + 1), where, "delta_c coeffs")
    for k in ("resid_std", "resid_max", "mean"):
        _num_list(dc[k], (4,), where, f"delta_c {k}")
    zc = _num_list(dc["z_clamp"], (2,), where, "delta_c z_clamp")
    zf = _num_list(xb["z_fit"], (2,), where, "xbar z_fit")
    if not (zc[0] < zc[1] and zf[0] < zf[1]):
        raise WcMapError(f"{where}: a redshift range is empty")
    _num_list([dc["z_min"], dc["z_max"]], (2,), where, "delta_c z_min/z_max")
    _num_list(xb["coeffs"], (DEG + 1,), where, "xbar coeffs")
    for k in ("n_blocks", "n_sims"):
        _count(dc[k], where, f"delta_c {k}")
    _count(xb["n_groups"], where, "xbar n_groups")


def parse_wc_map(raw, where="<w_c map>", *, expected_sha256=None, require_release_eligible=True):
    """A ``WcMap`` from the file's bytes ``raw``: strict JSON (no duplicate keys, no NaN or Infinity), the exact
    layout, finite numbers, a well-formed ``wc_map`` record with exactly one input (the cache), the payload digest,
    and (``require_release_eligible``) release eligibility under the release finder allow-list. With
    ``expected_sha256`` the bytes must hash to it first."""
    from . import safe_io as S
    from ..catalog import RELEASE_FINDER_IDS
    raw = bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and sha != expected_sha256:
        raise WcMapError(f"{where}: sha256 {sha} is not the expected {expected_sha256}")
    if len(raw) > MAX_BYTES:
        raise WcMapError(f"{where}: {len(raw)} bytes exceed the bound {MAX_BYTES}")
    try:
        doc = S._json_loads(raw, where)
    except S.SafeFormatError as e:
        raise WcMapError(str(e)) from None
    _check_payload(doc, where)
    if "lineage" not in doc:
        raise WcMapError(f"{where}: no lineage record")
    try:
        rec = L.check_record(doc["lineage"], RELEASE_FINDER_IDS)
    except L.LineageError as e:
        raise WcMapError(f"{where}: {e}") from None
    if rec["kind"] != KIND or rec["kind_version"] != KIND_VERSION:
        raise WcMapError(f"{where}: a {rec['kind']} record of version {rec['kind_version']}, not {KIND} {KIND_VERSION}")
    if rec["product"].get("payload_sha256") != payload_sha256(doc):
        raise WcMapError(f"{where}: the map differs from its record's payload digest (edited after it was made)")
    if [i["role"] for i in rec["inputs"]] != ["cache"] or rec["inputs"][0]["kind"] != "tau0_cache":
        raise WcMapError(f"{where}: the record's inputs must be exactly the training cache")
    if require_release_eligible and not rec["release_eligible"]:
        raise WcMapError(f"{where}: not release-eligible: " + "; ".join(rec["ineligible_reasons"]))
    dc, xb = doc["delta_c"], doc["xbar"]
    return WcMap(delta_c_coeffs=np.asarray(dc["coeffs"], dtype=np.float64),
                 delta_c_z_clamp=tuple(float(v) for v in dc["z_clamp"]),
                 delta_c_resid_std=np.asarray(dc["resid_std"], dtype=np.float64),
                 xbar_coeffs=np.asarray(xb["coeffs"], dtype=np.float64),
                 xbar_z_fit=tuple(float(v) for v in xb["z_fit"]), record=rec, sha256=sha, doc=doc)


def read_wc_map(path, *, expected_sha256=None, require_release_eligible=True):
    """``parse_wc_map`` of the file at ``path`` (a regular file, not a symlink, at most ``MAX_BYTES``)."""
    from . import safe_io as S
    try:
        raw = S.read_regular(str(path), MAX_BYTES)
    except S.SafeFormatError as e:
        raise WcMapError(str(e)) from None
    return parse_wc_map(raw, str(path), expected_sha256=expected_sha256,
                        require_release_eligible=require_release_eligible)


def dumps_summary(m):
    """A short human-readable summary (for the refit command's output)."""
    return json.dumps({"sha256": m.sha256, "cache_sha256": m.cache_sha256,
                       "release_eligible": m.record["release_eligible"],
                       "ineligible_reasons": m.record["ineligible_reasons"],
                       "delta_c_coeffs": m.delta_c_coeffs.tolist(), "xbar_coeffs": m.xbar_coeffs.tolist(),
                       "delta_c_resid_std": m.delta_c_resid_std.tolist()}, indent=1)
