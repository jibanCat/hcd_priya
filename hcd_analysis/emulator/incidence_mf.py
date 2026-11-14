"""The multi-fidelity (MF) correction of the incidence: format ``hcd_priya.incidence_mf_record/1`` (contract in
docs/release_format_h5.md, section "The incidence MF record").

**Physical meaning.** The LF emulator predicts the incidence dN/dX of the three HCD classes and the column-density
distribution f(N_HI) (the CDDF) of the low-resolution simulations. The high-resolution simulations hold a different
number of absorbers. The correction is a fixed table, measured on matched HR/LF simulation pairs::

    R_c(z)      = mean over the matched HR simulations of (dN/dX)_HR / (dN/dX)_LF      per class c, per redshift
    R(z, N_HI)  = the same mean of f_HR / f_LF                                         per column-density bin

and ``dndx_MF = dndx_LF * R_c(z)``. It depends on the class (or bin) and on redshift only: not on cosmology and not
on the mean flux. Between the tabulated redshifts it is interpolated linearly, and held constant beyond them.

**This module adds no estimator.** The tables are built and evaluated by the functions that have been in
``multifidelity.py`` since June 2026: ``load_snap_cache``, ``match_snap_hr_to_lf``, ``build_dndx_res_corr``,
``build_cddf_res_corr``, ``dndx_res_factor``, ``apply_dndx_res_corr``, ``cddf_res_factor``. What is here is the
container: the record that carries a fitted table with its inputs, and its strict reader. The predictor applies the
dN/dX table before the map to sightline fractions, and the CDDF table to the column-density distribution it returns
(``pred.cddf``), never returning a number in a bin where no HR/LF pair has absorbers at any tabulated redshift,
nor, under LF+MF, where none has them at the redshift used (``cddf_n_pairs``).

**Inputs.** The LF side is the LF training cache (its snapshot-level columns). The HR side is NOT the HR cache: the HR
cache's catalogue-derived columns predate a catalogue fix and stay refused by ``mf_release``. The HR side is an HR incidence product derived from the rebuilt HR catalogues with the functions that made the
LF cache columns (``hr_snap_from_product``). The record names both inputs by sha256. The code pins those digests and
verifies no lineage of the HR product: how it was derived is the statement of whoever builds the record
(``HR_SOURCE`` says exactly this).

**Layout.** One record directory holds exactly one file, ``incidence_mf_record.json``: strict UTF-8 JSON with a closed
key set (``RECORD_KEYS``, and the optional ``heldout_error``) and finite numbers (a cell of the held-out error that
has no estimate is null), the numbers of the tables included (they are a few hundred). A published
record is anchored by the sha256 of that file. Nothing is pickled or evaluated.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import types
import warnings
from typing import NamedTuple

import numpy as np

from . import mf_release as _MR
from . import safe_io as S
from . import wc_map as _WM
from .mf_release import MFRecordError

RECORD_FORMAT = "hcd_priya.incidence_mf_record/1"
RECORD_FILE = "incidence_mf_record.json"
MAX_BYTES = 1 << 20
CLASS_ORDER = ("LLS", "subDLA", "DLA")
MODEL_FORM = (
    "multifidelity.build_dndx_res_corr and build_cddf_res_corr: at each redshift the mean over the matched HR "
    "simulations of HR / LF. dN/dX: per HCD class, over every matched pair (the fit refuses a pair whose dN/dX "
    "is not finite and positive, so no cell is filled in). f(N_HI): per column-density bin, over the pairs with "
    "both values positive, and 1 where there is none. Theta-independent; linear in z between the tabulated "
    "redshifts and constant beyond them (dndx_res_factor, cddf_res_factor); dndx_MF = dndx_LF * R_c(z)"
)
# What the code can and cannot say about each input: both are named by their sha256, nothing more is verified here.
HR_SOURCE = (
    "an HR incidence product (plain arrays of dN/dX and f(N_HI) per HR snapshot), digest-anchored only: this code "
    "pins its sha256 and checks its shapes and values, and verifies no lineage. That it was derived from the "
    "rebuilt HR absorber catalogues, with the functions that made the LF cache's snapshot columns, is the "
    "statement of whoever built the record. It is not the HR cache, whose catalogue-derived columns mf_release "
    "refuses"
)
LF_SOURCE = (
    "the LF cache's snapshot columns (snap_dNdX_*, snap_f_nhi), read with multifidelity.load_snap_cache from a "
    "file pinned by its sha256; the predictor applies the record only when that sha256 is the training cache of "
    "the backbone"
)
# The statement for a record that carries no held-out error (the optional key heldout_error is absent): how such a
# number is computed and what it covers. It is shown with every prediction of such a record
# (labels["incidence_mf_error"], through heldout_error_label). A record that carries the error states these things
# itself, in its heldout_error block.
HELDOUT_ERROR_STATEMENT = (
    "No held-out error of the dN/dX or the CDDF is shipped with this record. How such an error is computed: "
    "for each HR simulation s the table is rebuilt without s; an LF model trained without the LF partner of s (a "
    "fold model) predicts dN/dX at the parameters of s; that prediction times the table without s is compared with "
    "the HR value of s, as ln(prediction / truth), per class and redshift. What it covers: the transfer to a "
    "parameter point that neither the LF model nor the table has seen. What it does not cover: only the LF "
    "simulations that have an HR partner have a fold model (six of sixty), so the LF part is tested at those "
    "points only; every simulation shares one set of initial phases, so the part that depends on the realisation "
    "is left out; the form of the table, one ratio per class and redshift, was chosen with every HR simulation in "
    "view; a redshift with a single pair has no held-out value."
)


def heldout_error_label(n_table_sims: int, heldout: dict | None = None) -> str:
    """The text of ``labels["incidence_mf_error"]``.

    With a held-out error in the record (``heldout``: the record's ``heldout_error`` block): what the numbers of
    ``dndx_err`` and ``cddf_err`` are, in the record's own words, word for word.

    Without one: ``HELDOUT_ERROR_STATEMENT``, the sentences that link it to this record (the statement speaks of the
    six fold models, and a record's table is built from the HR simulations that the record lists, which need not be
    those six), and the statement that the two error fields are NaN."""
    if heldout is not None:
        return (
            f"The held-out error of this prediction's dndx and cddf is in dndx_err and cddf_err ({heldout['kind']})."
            f" {heldout['definition']} LF: {heldout['lf']} LF+MF: {heldout['lfmf']} What it does not cover: "
            f"{heldout['what_it_does_not_cover']} The correction table of this record is built from the HR "
            f"simulations listed in provenance incidence_mf train_sims, {int(n_table_sims)} in number; the held-out "
            f"error was measured for a correction of {int(heldout['n_hr_simulations'])} HR simulations."
        )
    return (
        HELDOUT_ERROR_STATEMENT
        + " The six fold models are those of the six LF simulations that have an HR partner. The correction table "
        "of this record is built from the HR simulations listed in provenance incidence_mf train_sims, "
        f"{int(n_table_sims)} in number (a redshift row or a CDDF bin may rest on fewer: n_sims, cddf_n_pairs); "
        "they need not be all of those six. No held-out error is available for this record: dndx_err and cddf_err "
        "are NaN."
    )


RECORD_KEYS = {
    "format",
    "version",
    "model_form",
    "class_order",
    "tables",
    "lf_cache",
    "hr_incidence",
    "composition",
    "train_sims",
    "n_pairs",
    "fit_code",
}
_TABLE_KEYS = {"z", "n_sims", "dndx_ratio", "log_nhi_centres", "cddf_ratio", "cddf_n_pairs"}
# One optional key of the record: the held-out error of the dN/dX and the CDDF. A record without it is valid, and its
# bytes are what they were before the key existed.
OPTIONAL_RECORD_KEYS = {"heldout_error"}
# What the numbers of a held-out error are. One kind is known; a record that states another is refused, so that a
# number is never read as something it is not.
HELDOUT_KIND = "rms_log_error_heldout"
_HELDOUT_KEYS = {
    "kind",
    "definition",
    "lf",
    "lfmf",
    "what_it_does_not_cover",
    "n_hr_simulations",
    "z",
    "dndx",
    "cddf",
    "source",
    "copied_from",
}
_HELDOUT_TABLE_KEYS = {"lf", "lf_n", "lfmf", "lfmf_n"}
_HELDOUT_SOURCE_KEYS = {"file", "sha256", "how"}
# What the builder itself states about the block: the table file it read, by the digest it was pinned to. The
# key "source" beside it is the table's own statement of where its numbers came from, which nothing here verifies.
_HELDOUT_COPIED_KEYS = {"file", "sha256"}
_HELDOUT_Z_TOL = (
    1e-6  # an error is looked up at a tabulated redshift only; it is never interpolated
)
_LF_KEYS = {"file", "sha256", "source"}
_HR_KEYS = {"file", "sha256", "source", "groups_not_in_product"}
_HR_ARRAYS = ("sim_name", "z", "params", "dndx", "cddf", "log_nhi_centres", "class_names")


# --------------------------------------------------------------------------------------------- the HR product
def hr_snap_from_product(path):
    """The HR incidence product (an ``.npz`` of plain arrays, read with ``allow_pickle=False``) as the snapshot dict
    that ``multifidelity.match_snap_hr_to_lf`` and the two table builders take: the keys of
    ``multifidelity.load_snap_cache``. Required arrays: ``sim_name`` (S,), ``z`` (S,), ``params`` (S, 9), ``dndx``
    (S, 3) in the order LLS, subDLA, DLA (``class_names`` must say so), ``cddf`` (S, Nbin), ``log_nhi_centres``
    (Nbin,). Every number must be finite, and each simulation name must go with exactly one parameter point (a
    name column that is out of step with the parameters is refused)."""
    import zipfile

    from .data import normalize_params

    try:
        with np.load(os.fspath(path), allow_pickle=False) as f:
            a = {k: f[k] for k in _HR_ARRAYS if k in f.files}
    except (
        OSError,
        ValueError,
        zipfile.BadZipFile,
    ) as e:  # not an .npz, or an array that needs pickle
        raise MFRecordError(f"{path}: unreadable HR incidence product ({e})") from None
    missing = [k for k in _HR_ARRAYS if k not in a]
    if missing:
        raise MFRecordError(f"{path}: the HR incidence product lacks the arrays {missing}")
    if a["class_names"].ndim != 1 or [str(c) for c in a["class_names"]] != list(CLASS_ORDER):
        raise MFRecordError(f"{path}: class_names must be {list(CLASS_ORDER)}")
    if a["sim_name"].ndim != 1 or a["log_nhi_centres"].ndim != 1 or a["sim_name"].size < 1:
        raise MFRecordError(f"{path}: HR incidence arrays with inconsistent shapes: sim_name, bins")
    n, nb = a["sim_name"].shape[0], a["log_nhi_centres"].shape[0]
    want = {"z": (n,), "params": (n, 9), "dndx": (n, 3), "cddf": (n, nb)}
    bad = [(k, a[k].shape, v) for k, v in want.items() if a[k].shape != v]
    if bad:
        raise MFRecordError(f"{path}: HR incidence arrays with inconsistent shapes: {bad}")
    try:
        num = {k: np.asarray(a[k], dtype=np.float64) for k in (*want, "log_nhi_centres")}
    except (TypeError, ValueError) as e:
        raise MFRecordError(f"{path}: an HR incidence array is not numeric ({e})") from None
    if not all(np.all(np.isfinite(v)) for v in num.values()):
        raise MFRecordError(f"{path}: the HR incidence arrays must be finite")
    sims = np.asarray(a["sim_name"]).astype(str)
    pu = np.round(normalize_params(num["params"]), 6)
    point = {}
    for s, q in zip(sims, map(tuple, pu)):
        if point.setdefault(s, q) != q:
            raise MFRecordError(f"{path}: the simulation {s} has more than one parameter point")
    if len(set(point.values())) != len(point):
        raise MFRecordError(f"{path}: two simulation names share one parameter point")
    return dict(
        dndx=num["dndx"],
        f_nhi=num["cddf"],
        log_nhi=num["log_nhi_centres"],
        z=np.round(num["z"], 4),
        params=num["params"],
        pu=pu,
        sim=sims,
    )


# --------------------------------------------------------------------------------------------- the fit
def _snap_keys(d):
    """The key ``multifidelity.match_snap_hr_to_lf`` pairs on: the unit-box parameters and the redshift."""
    return [(tuple(d["pu"][i]), round(float(d["z"][i]), 4)) for i in range(len(d["z"]))]


def fit_tables(lf_snap, hr_snap, *, exclude_sims):
    """Fit the two tables on the matched HR/LF snapshot pairs, with the historical functions.

    ``lf_snap`` and ``hr_snap`` are snapshot dicts (``multifidelity.load_snap_cache`` for the LF cache,
    ``hr_snap_from_product`` for the HR product). ``exclude_sims`` (required; an empty list says "none") are the HR
    simulation names the composition leaves out (``mf_release.select_train_rows``).

    This function adds input checks, never an estimator. Refused: an HR snapshot listed twice; an HR snapshot
    without its LF partner (``match_snap_hr_to_lf``: same parameters and redshift), which is never dropped; an LF
    partner key that the LF cache holds twice; a matched pair whose dN/dX is not finite and positive in every class
    (so the dN/dX table is a plain mean over every pair, and no cell is filled in from a neighbour); a matched pair
    whose f(N_HI) is not finite and non-negative; differing column-density bins.

    Returns ``(tables, train_sims, n_pairs)``: ``tables`` has ``z`` (Nz,), ``n_sims`` (Nz,) the number of pairs
    behind each redshift's dN/dX mean, ``dndx_ratio`` (Nz, 3), ``log_nhi_centres`` (Nbin,), ``cddf_ratio``
    (Nz, Nbin) and ``cddf_n_pairs`` (Nz, Nbin), the number of pairs behind each CDDF cell (those with both values
    positive). A cell with no such pair holds the ratio 1, which means "no correction measured", not "measured to
    be 1": ``cddf_n_pairs`` is what tells the two apart.
    """
    from collections import Counter

    from . import multifidelity as MF

    sims = np.asarray(hr_snap["sim"]).astype(str)
    keep, train = _MR.select_train_rows({"sim": sims}, exclude_sims=exclude_sims)
    hr = {
        k: (v if k == "log_nhi" else np.asarray(v)[keep]) for k, v in hr_snap.items()
    }  # log_nhi is per bin
    name = lambda i: f"{hr['sim'][i]} z={hr['z'][i]:g}"
    hr_keys, lf_count = _snap_keys(hr), Counter(_snap_keys(lf_snap))
    twice = sorted({name(i) for i, c in enumerate(hr_keys) if hr_keys.count(c) > 1})
    if twice:
        raise MFRecordError(f"HR snapshots listed more than once: {twice}")
    pairs = MF.match_snap_hr_to_lf(lf_snap, hr)
    if len(pairs) != keep.size:
        got = {h for h, _ in pairs}
        lost = sorted({name(i) for i in range(keep.size) if i not in got})
        raise MFRecordError(
            f"HR snapshots without an LF partner (same parameters and redshift): {lost}"
        )
    ambiguous = sorted({name(i) for i, c in enumerate(hr_keys) if lf_count[c] != 1})
    if ambiguous:
        raise MFRecordError(
            f"HR snapshots whose LF partner is not unique in the LF cache: {ambiguous}"
        )
    lf_bins, hr_bins = np.asarray(lf_snap["log_nhi"]), np.asarray(hr["log_nhi"])
    if lf_bins.shape != hr_bins.shape or not np.allclose(lf_bins, hr_bins, rtol=1e-12, atol=0):
        raise MFRecordError("the LF and HR column-density bins differ")
    h = np.array([p[0] for p in pairs])
    l = np.array([p[1] for p in pairs])
    z = np.asarray(hr["z"])[h]
    lf_d, hr_d = np.asarray(lf_snap["dndx"], float)[l], np.asarray(hr["dndx"], float)[h]
    lf_f, hr_f = np.asarray(lf_snap["f_nhi"], float)[l], np.asarray(hr["f_nhi"], float)[h]
    ok = np.isfinite(lf_d) & np.isfinite(hr_d) & (lf_d > 0) & (hr_d > 0)
    if not ok.all():
        bad = sorted({name(h[i]) for i in np.where(~ok.all(axis=1))[0]})
        raise MFRecordError(f"matched snapshots whose dN/dX is not finite and positive: {bad}")
    okf = np.isfinite(lf_f) & np.isfinite(hr_f) & (lf_f >= 0) & (hr_f >= 0)
    if not okf.all():
        bad = sorted({name(h[i]) for i in np.where(~okf.all(axis=1))[0]})
        raise MFRecordError(
            f"matched snapshots whose f(N_HI) is not finite and non-negative: {bad}"
        )
    dn = MF.build_dndx_res_corr(lf_d, lf_snap["z"][l], hr_d, z)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", RuntimeWarning
        )  # a bin empty in every pair: its ratio is 1 by design
        cd = MF.build_cddf_res_corr(lf_f, lf_snap["z"][l], hr_f, z, hr_bins)
    if not np.array_equal(dn["z"], cd["z"]):
        raise MFRecordError("the dN/dX and CDDF tables came out on different redshifts")
    zr = np.round(z, 4)
    tables = {
        "z": np.asarray(dn["z"], dtype=np.float64),
        "n_sims": np.array([int(np.sum(zr == zz)) for zz in dn["z"]]),
        "dndx_ratio": np.asarray(dn["ratio"], dtype=np.float64),
        "log_nhi_centres": np.asarray(cd["log_nhi"], dtype=np.float64),
        "cddf_ratio": np.asarray(cd["ratio"], dtype=np.float64),
        # a count, not an estimate: the pairs with both values positive, the cells build_cddf_res_corr averages
        "cddf_n_pairs": np.array(
            [np.sum((lf_f > 0) & (hr_f > 0) & (zr == zz)[:, None], axis=0) for zz in dn["z"]]
        ),
    }
    return tables, train, len(pairs)


# --------------------------------------------------------------------------------------------- record checks
def _numbers(x, shape, where, what):
    try:
        return _WM._num_list(x, shape, where, what)
    except (_WM.WcMapError, ValueError, OverflowError) as e:  # a ragged list; an integer too large
        raise MFRecordError(f"{where}: {what}: {e}") from None


def _check_tables(t, where):
    """The record's ``tables`` block: closed keys, plain finite numbers, consistent shapes, ascending redshifts and
    bins, positive ratios, at least one pair per redshift, the CDDF pair counts within the pairs of the redshift,
    and the ratio 1 wherever a CDDF cell has no pair. Returns the arrays (float64; the two counts int).
    """
    _MR._keys(t, _TABLE_KEYS, f"{where}: tables")
    if not isinstance(t["z"], list) or not isinstance(t["log_nhi_centres"], list):
        raise MFRecordError(f"{where}: tables.z and tables.log_nhi_centres must be lists")
    nz, nb = len(t["z"]), len(t["log_nhi_centres"])
    if nz < 2 or nb < 1:
        raise MFRecordError(f"{where}: the tables need at least two redshifts and one bin")
    a = {
        "z": _numbers(t["z"], (nz,), where, "tables.z"),
        "dndx_ratio": _numbers(t["dndx_ratio"], (nz, 3), where, "tables.dndx_ratio"),
        "log_nhi_centres": _numbers(t["log_nhi_centres"], (nb,), where, "tables.log_nhi_centres"),
        "cddf_ratio": _numbers(t["cddf_ratio"], (nz, nb), where, "tables.cddf_ratio"),
    }
    n = t["n_sims"]
    if not isinstance(n, list) or len(n) != nz or not all(type(v) is int and v >= 1 for v in n):
        raise MFRecordError(f"{where}: tables.n_sims must be {nz} positive integers")
    if np.any(np.diff(a["z"]) <= 0) or np.any(np.diff(a["log_nhi_centres"]) <= 0):
        raise MFRecordError(f"{where}: tables.z and tables.log_nhi_centres must ascend")
    if np.any(a["dndx_ratio"] <= 0) or np.any(a["cddf_ratio"] <= 0):
        raise MFRecordError(f"{where}: the ratios must be positive")
    a["n_sims"] = np.asarray(n, dtype=np.int64)
    c = t["cddf_n_pairs"]
    if (
        not isinstance(c, list)
        or len(c) != nz
        or not all(
            isinstance(row, list) and len(row) == nb and all(type(v) is int and v >= 0 for v in row)
            for row in c
        )
    ):
        raise MFRecordError(
            f"{where}: tables.cddf_n_pairs must be {nz} rows of {nb} non-negative integers"
        )
    a["cddf_n_pairs"] = np.asarray(c, dtype=np.int64).reshape(nz, nb)
    if np.any(a["cddf_n_pairs"] > a["n_sims"][:, None]):
        raise MFRecordError(f"{where}: a CDDF cell has more pairs than its redshift has")
    if np.any(a["cddf_ratio"][a["cddf_n_pairs"] == 0] != 1.0):
        raise MFRecordError(
            f"{where}: a CDDF cell with no pair must hold the ratio 1 (no correction measured)"
        )
    return a


def _finite(x):
    """True for an int or float that is a finite float64 (an integer too large for one is not)."""
    try:
        return math.isfinite(float(x))
    except OverflowError:
        return False


def _heldout_table(block, shape, where, what):
    """One block of a held-out error (``dndx`` or ``cddf``): four tables of ``shape``. A value is a finite number
    >= 0, or null exactly where its count is 0. Returns {name: array}; a null becomes NaN."""
    _MR._keys(block, _HELDOUT_TABLE_KEYS, f"{where}: heldout_error.{what}")
    out = {}
    for fid in ("lf", "lfmf"):
        vals, counts = block[fid], block[fid + "_n"]
        ok = (
            isinstance(vals, list)
            and isinstance(counts, list)
            and len(vals) == len(counts) == shape[0]
            and all(isinstance(r, list) and len(r) == shape[1] for r in vals)
            and all(isinstance(r, list) and len(r) == shape[1] for r in counts)
        )
        if not ok:
            raise MFRecordError(
                f"{where}: heldout_error.{what}.{fid} and its counts must have shape {shape}"
            )
        v = np.full(shape, np.nan)
        n = np.zeros(shape, dtype=int)
        for i in range(shape[0]):
            for j in range(shape[1]):
                x, c = vals[i][j], counts[i][j]
                if type(c) is not int or c < 0:
                    raise MFRecordError(
                        f"{where}: heldout_error.{what}.{fid}_n holds a count that is not an integer >= 0"
                    )
                if x is None:
                    if c != 0:
                        raise MFRecordError(
                            f"{where}: heldout_error.{what}.{fid} is null where its count is {c}"
                        )
                    continue
                if type(x) not in (int, float) or not _finite(x) or x < 0 or c == 0:
                    raise MFRecordError(
                        f"{where}: heldout_error.{what}.{fid} must be a finite number >= 0 with a count >= 1, or "
                        "null with a count of 0"
                    )
                v[i, j], n[i, j] = float(x), c
        out[fid], out[fid + "_n"] = v, n
    return out


def _check_heldout(h, where, n_bins):
    """The record's optional ``heldout_error`` block: closed keys, the one known kind, the texts, ascending
    redshifts, and the tables of dN/dX (per redshift and class) and of the CDDF (per redshift and bin). Returns the
    arrays; a cell without an estimate is NaN."""
    _MR._keys(h, _HELDOUT_KEYS, f"{where}: heldout_error")
    if h["kind"] != HELDOUT_KIND:
        raise MFRecordError(
            f"{where}: heldout_error.kind is {h['kind']!r}; this code reads {HELDOUT_KIND!r} only"
        )
    for key in ("definition", "lf", "lfmf", "what_it_does_not_cover"):
        if not isinstance(h[key], str) or not h[key].strip():
            raise MFRecordError(f"{where}: heldout_error.{key} must be a non-empty text")
    if type(h["n_hr_simulations"]) is not int or h["n_hr_simulations"] < 1:
        raise MFRecordError(f"{where}: heldout_error.n_hr_simulations must be an integer >= 1")
    if not isinstance(h["z"], list) or not h["z"]:
        raise MFRecordError(f"{where}: heldout_error.z must be a non-empty list")
    z = _numbers(h["z"], (len(h["z"]),), where, "heldout_error.z")
    if np.any(np.diff(z) <= 0):
        raise MFRecordError(f"{where}: heldout_error.z must be strictly ascending")
    out = {"z": z}
    for what, ncol in (("dndx", len(CLASS_ORDER)), ("cddf", n_bins)):
        for name, arr in _heldout_table(h[what], (z.size, ncol), where, what).items():
            out[f"{what}_{name}"] = arr
    src = h["source"]
    _MR._keys(src, _HELDOUT_SOURCE_KEYS, f"{where}: heldout_error.source")
    if not isinstance(src["file"], str) or not _MR._PLAIN.match(src["file"]):
        raise MFRecordError(f"{where}: heldout_error.source.file must be a plain file name")
    _MR._hex(src["sha256"], 64, f"{where}: heldout_error.source.sha256")
    if not isinstance(src["how"], str) or not src["how"].strip():
        raise MFRecordError(f"{where}: heldout_error.source.how must be a non-empty text")
    cp = h["copied_from"]
    _MR._keys(cp, _HELDOUT_COPIED_KEYS, f"{where}: heldout_error.copied_from")
    if not isinstance(cp["file"], str) or not _MR._PLAIN.match(cp["file"]):
        raise MFRecordError(f"{where}: heldout_error.copied_from.file must be a plain file name")
    _MR._hex(cp["sha256"], 64, f"{where}: heldout_error.copied_from.sha256")
    return out


# Keys of a held-out error table FILE beyond those it shares with the record's block: the two it is checked against,
# and one that is for documentation and is not copied. The block's "copied_from" is not a key of the file: the builder
# states it.
_HELDOUT_FILE_CHECKED = {"class_names", "log_nhi_centres"}
_HELDOUT_FILE_IGNORED = {"pooled_dndx_z2p2_to_4p6"}


def heldout_block_from_table(
    doc, *, log_nhi_centres, n_train_sims, table_file, table_sha256, other_composition=False
):
    """The record's ``heldout_error`` block from a held-out error table (the parsed JSON file that whoever measured
    the error supplies). Nothing is computed: the numbers and the texts are copied. Checked first: the keys; that
    the table's classes are this code's, in order; that its column-density bins are the record's; and that the
    number of HR simulations the error was measured for (``n_hr_simulations``) is the number the record's table is
    built from (``n_train_sims``), unless ``other_composition`` says that a difference is intended. ``table_file``
    and ``table_sha256`` are the name and the sha256 of the file ``doc`` was parsed from, as whoever calls this
    verified them; they become the block's ``copied_from``.
    """
    where = "the held-out error table"
    want = (_HELDOUT_KEYS - {"copied_from"}) | _HELDOUT_FILE_CHECKED
    got = set(doc) if isinstance(doc, dict) else set()
    if not want <= got or got - want - _HELDOUT_FILE_IGNORED:
        raise MFRecordError(
            f"{where}: keys must be {sorted(want)} (missing {sorted(want - got)}, unknown "
            f"{sorted(got - want - _HELDOUT_FILE_IGNORED)})"
        )
    if doc["class_names"] != list(CLASS_ORDER):
        raise MFRecordError(f"{where}: class_names must be {list(CLASS_ORDER)}")
    centres = np.asarray(log_nhi_centres, dtype=np.float64)
    theirs = _numbers(doc["log_nhi_centres"], (centres.size,), where, "log_nhi_centres")
    if not np.allclose(theirs, centres, rtol=0.0, atol=1e-9):
        raise MFRecordError(f"{where}: its column-density bins are not the record's")
    block = {k: doc[k] for k in sorted(_HELDOUT_KEYS - {"copied_from"})}
    block["copied_from"] = {"file": table_file, "sha256": table_sha256}
    _check_heldout(block, where, centres.size)
    if block["n_hr_simulations"] != int(n_train_sims) and not other_composition:
        raise MFRecordError(
            f"{where}: it was measured for a correction of {block['n_hr_simulations']} HR simulations, and this "
            f"record's table is built from {int(n_train_sims)}; say that the difference is intended, or supply the "
            "error of this composition"
        )
    return block


def check_record(rec, where="incidence MF record"):
    """Validate a record dict: the closed schema, the constant statements, the tables, and the optional held-out
    error. Raises MFRecordError. Returns the tables as arrays (and ``"heldout"``: the arrays of the held-out error,
    or None)."""
    got = set(rec) if isinstance(rec, dict) else set()
    if not RECORD_KEYS <= got or got - RECORD_KEYS - OPTIONAL_RECORD_KEYS:
        raise MFRecordError(
            f"{where}: keys must be {sorted(RECORD_KEYS)}, with the optional {sorted(OPTIONAL_RECORD_KEYS)} "
            f"(missing {sorted(RECORD_KEYS - got)}, unknown {sorted(got - RECORD_KEYS - OPTIONAL_RECORD_KEYS)})"
        )
    if rec["format"] != RECORD_FORMAT:
        raise MFRecordError(f"{where}: format {rec['format']!r}, expected {RECORD_FORMAT}")
    if not isinstance(rec["version"], str) or not rec["version"].strip():
        raise MFRecordError(f"{where}: version must be a non-empty string")
    if rec["model_form"] != MODEL_FORM:
        raise MFRecordError(f"{where}: model_form is not the form this code evaluates")
    if rec["class_order"] != list(CLASS_ORDER):
        raise MFRecordError(f"{where}: class_order must be {list(CLASS_ORDER)}")
    arrays = _check_tables(rec["tables"], where)
    lf, hr = rec["lf_cache"], rec["hr_incidence"]
    _MR._keys(lf, _LF_KEYS, f"{where}: lf_cache")
    _MR._keys(hr, _HR_KEYS, f"{where}: hr_incidence")
    for side, d, src in (("lf_cache", lf, LF_SOURCE), ("hr_incidence", hr, HR_SOURCE)):
        if not isinstance(d["file"], str) or not _MR._PLAIN.match(d["file"]):
            raise MFRecordError(f"{where}: {side}.file must be a plain file name, never a path")
        _MR._hex(d["sha256"], 64, f"{where}: {side}.sha256")
        if d["source"] != src:
            raise MFRecordError(f"{where}: {side}.source is not the code's statement of that input")
    g = hr["groups_not_in_product"]
    if not isinstance(g, list) or not all(isinstance(s, str) and s.strip() for s in g):
        raise MFRecordError(
            f"{where}: hr_incidence.groups_not_in_product must be a list of names (it may be empty)"
        )
    comp = rec["composition"]
    _MR._keys(comp, _MR._COMPOSITION_KEYS, f"{where}: composition")
    if not isinstance(comp["id"], str) or not comp["id"].strip():
        raise MFRecordError(f"{where}: composition.id must be a non-empty string")
    _MR._sorted_names(comp["excluded_sims"], f"{where}: composition.excluded_sims")
    if (
        not isinstance(comp["ruling_ids"], list)
        or not comp["ruling_ids"]
        or not all(isinstance(r, str) and r.strip() for r in comp["ruling_ids"])
    ):
        raise MFRecordError(
            f"{where}: composition.ruling_ids must name at least one ruling (the authority for the composition)"
        )
    train = _MR._sorted_names(rec["train_sims"], f"{where}: train_sims")
    if not train or set(train) & set(comp["excluded_sims"]):
        raise MFRecordError(
            f"{where}: train_sims must be non-empty and hold no excluded simulation"
        )
    n = rec["n_pairs"]
    if type(n) is not int or n != int(arrays["n_sims"].sum()):
        raise MFRecordError(f"{where}: n_pairs must be the sum of tables.n_sims")
    if int(arrays["n_sims"].max()) > len(train):
        raise MFRecordError(
            f"{where}: a redshift has more pairs than there are training simulations"
        )
    fc = rec["fit_code"]
    _MR._keys(fc, {"commit", "dirty"}, f"{where}: fit_code")
    _MR._hex(fc["commit"], 40, f"{where}: fit_code.commit")
    if fc["dirty"] is not False:
        raise MFRecordError(f"{where}: an incidence MF record is fitted from a clean tree")
    arrays["heldout"] = (
        _check_heldout(rec["heldout_error"], where, arrays["log_nhi_centres"].size)
        if "heldout_error" in rec
        else None
    )
    if arrays["heldout"] is not None:
        # an error belongs to a row of the correction table: at any other redshift the ratio is interpolated or
        # held at an edge row, and no error was measured for that
        gap = np.abs(arrays["heldout"]["z"][:, None] - arrays["z"][None, :]).min(axis=1)
        if np.any(gap > _HELDOUT_Z_TOL):
            raise MFRecordError(
                f"{where}: heldout_error.z holds a redshift that is not a row of the correction table"
            )
    return arrays


# --------------------------------------------------------------------------------------------- write
def write_record(
    out_dir,
    tables,
    *,
    version,
    lf_cache,
    hr_incidence,
    groups_not_in_product,
    composition,
    train_sims,
    n_pairs,
    fit_code,
    heldout_error=None,
):
    """Write a record directory ONCE: ``out_dir`` must not exist. ``tables`` as from ``fit_tables``; ``lf_cache`` and
    ``hr_incidence`` are ``{file, sha256}`` (the writer adds the code's statement of each source);
    ``groups_not_in_product`` names the HR snapshot groups the HR product does not hold (it may be empty).
    ``heldout_error``, when given, is the block of that name as the record holds it (``_check_heldout``); it is
    copied, not computed. Without it the record's bytes are what they were before the key existed.
    Everything is validated before anything is written, and what is written is read back. Returns (the record's
    path, its sha256: the anchor to publish)."""
    out_dir = os.fspath(out_dir)
    for side, d in (("lf_cache", lf_cache), ("hr_incidence", hr_incidence)):
        if not isinstance(d, dict) or set(d) != {"file", "sha256"}:
            raise MFRecordError(f"{side} must be {{file, sha256}}")
    if os.path.lexists(out_dir):
        raise FileExistsError(
            f"{out_dir} exists: an incidence MF record is written once into a new directory"
        )
    rec = {
        "format": RECORD_FORMAT,
        "version": version,
        "model_form": MODEL_FORM,
        "class_order": list(CLASS_ORDER),
        "tables": {
            k: (
                np.asarray(tables[k]).astype(int).tolist()
                if k in ("n_sims", "cddf_n_pairs")
                else np.asarray(tables[k], dtype=np.float64).tolist()
            )
            for k in sorted(_TABLE_KEYS)
        },
        "lf_cache": dict(_MR.copy_json(lf_cache), source=LF_SOURCE),
        "hr_incidence": dict(
            _MR.copy_json(hr_incidence),
            source=HR_SOURCE,
            groups_not_in_product=list(groups_not_in_product),
        ),
        "composition": dict(
            composition, excluded_sims=sorted(set(composition.get("excluded_sims") or []))
        ),
        "train_sims": sorted(set(train_sims)),
        "n_pairs": int(n_pairs),
        "fit_code": _MR.copy_json(fit_code),
    }
    if heldout_error is not None:
        rec["heldout_error"] = _MR.copy_json(heldout_error)
    try:
        raw = json.dumps(rec, indent=1, sort_keys=True, allow_nan=False).encode("utf-8")
    except ValueError as e:  # a NaN or an infinity in a table
        raise MFRecordError(f"the tables must be finite ({e})") from None
    parse_record(raw, "the record to write")  # refused before anything is written
    os.makedirs(out_dir)  # no exist_ok: a race is refused too
    path = os.path.join(out_dir, RECORD_FILE)
    return path, S._write_exclusive(path, [raw])


# --------------------------------------------------------------------------------------------- load
class IncidenceMF(NamedTuple):
    """A loaded incidence MF record, anchored on its published sha256. Arrays are read-only host numpy float64.

    It is sealed like ``mf_release.MFRecord``: ``record_bytes`` are the anchored bytes it was parsed from, and the
    predictor applies a record only through ``reverify``."""

    record: dict
    record_sha256: str
    z: np.ndarray  # (Nz,)
    n_sims: np.ndarray  # (Nz,) pairs behind each redshift's mean
    dndx_ratio: np.ndarray  # (Nz, 3)   LLS, subDLA, DLA
    log_nhi_centres: np.ndarray  # (Nbin,)
    cddf_ratio: np.ndarray  # (Nz, Nbin)
    cddf_n_pairs: (
        np.ndarray
    )  # (Nz, Nbin) pairs behind each CDDF cell; 0: no correction measured (ratio 1)
    anchored: bool
    record_bytes: bytes = None
    heldout: dict = (
        None  # the arrays of the held-out error (read-only), or None when the record has none
    )

    def dndx_table(self):
        """The table in the form ``multifidelity.dndx_res_factor`` and ``apply_dndx_res_corr`` take."""
        return {"z": self.z, "ratio": self.dndx_ratio}

    def cddf_table(self):
        """The table in the form ``multifidelity.cddf_res_factor`` takes."""
        return {"z": self.z, "log_nhi": self.log_nhi_centres, "ratio": self.cddf_ratio}

    def dndx_factor(self, z):
        """R_c(z), shape (3,): ``multifidelity.dndx_res_factor`` of this record's table, as host numpy."""
        import jax.numpy as jnp
        from .multifidelity import dndx_res_factor

        return np.asarray(
            dndx_res_factor(self.dndx_table(), jnp.asarray(float(z))), dtype=np.float64
        )

    def cddf_factor(self, z):
        """R(z, N_HI), shape (Nbin,): ``multifidelity.cddf_res_factor`` of this record's table, as host numpy."""
        import jax.numpy as jnp
        from .multifidelity import cddf_res_factor

        return np.asarray(
            cddf_res_factor(self.cddf_table(), jnp.asarray(float(z))), dtype=np.float64
        )

    def rows(self, z, tol=1e-9):
        """The table rows a lookup at ``z`` rests on: one row at a tabulated redshift, the two neighbours between
        two of them, the edge row beyond the table (where the historical lookups hold the edge value).
        """
        z = float(z)
        hit = np.where(np.abs(self.z - z) <= tol)[0]
        if hit.size:
            return [int(hit[0])]
        if z < self.z[0]:
            return [0]
        if z > self.z[-1]:
            return [len(self.z) - 1]
        i = int(np.searchsorted(self.z, z))
        return [i - 1, i]

    def single_pair(self, z):
        """True when a row that the lookup at ``z`` rests on is the ratio of one simulation pair, not a mean."""
        return bool(np.any(self.n_sims[self.rows(z)] == 1))

    def cddf_never_supported(self):
        """(Nbin,) bool: the bins with no pair at any tabulated redshift (for example a bin below the catalogue's
        column-density threshold). Neither the LF model nor the table has anything there."""
        return np.all(self.cddf_n_pairs == 0, axis=0)

    def cddf_uncorrected(self, z):
        """(Nbin,) bool: the bins in which a row that the lookup at ``z`` rests on has no pair, so that no
        correction was measured there."""
        return np.any(self.cddf_n_pairs[self.rows(z)] == 0, axis=0)

    def _heldout_row(self, what, z, lfmf):
        """One row of a held-out error table at a TABULATED redshift, else NaN: no interpolation of an error."""
        ncol = len(CLASS_ORDER) if what == "dndx" else self.log_nhi_centres.size
        if self.heldout is None:
            return np.full(ncol, np.nan)
        hit = np.where(np.abs(self.heldout["z"] - float(z)) <= _HELDOUT_Z_TOL)[0]
        if hit.size != 1:
            return np.full(ncol, np.nan)
        return np.array(
            self.heldout[f"{what}_{'lfmf' if lfmf else 'lf'}"][hit[0]], dtype=np.float64
        )

    def heldout_dndx(self, z, lfmf):
        """(3,) the held-out rms of ln(prediction / truth) of dN/dX per class at ``z``, for the corrected (``lfmf``)
        or the uncorrected prediction; NaN where the record has no estimate."""
        return self._heldout_row("dndx", z, lfmf)

    def heldout_cddf(self, z, lfmf):
        """(Nbin,) the same for the CDDF, per column-density bin."""
        return self._heldout_row("cddf", z, lfmf)

    def provenance(self):
        r = self.record
        return {
            "record_sha256": self.record_sha256,
            "version": r["version"],
            "model_form": r["model_form"],
            "composition": _MR.copy_json(r["composition"]),
            "train_sims": list(r["train_sims"]),
            "n_pairs": r["n_pairs"],
            "z": [float(v) for v in self.z],
            "n_sims": [int(v) for v in self.n_sims],
            "cddf_n_pairs": self.cddf_n_pairs.tolist(),
            "lf_cache": _MR.copy_json(r["lf_cache"]),
            "hr_incidence": _MR.copy_json(r["hr_incidence"]),
            "fit_code": _MR.copy_json(r["fit_code"]),
            # the held-out error without its tables (they are on the prediction: dndx_err, cddf_err), or None
            "heldout_error": (
                None
                if self.heldout is None
                else {
                    **{
                        k: _MR.copy_json(r["heldout_error"][k])
                        for k in (
                            "kind",
                            "definition",
                            "lf",
                            "lfmf",
                            "what_it_does_not_cover",
                            "n_hr_simulations",
                            "z",
                            "source",
                            "copied_from",
                        )
                    },
                    "dndx_lf_n": self.heldout["dndx_lf_n"].tolist(),
                    "dndx_lfmf_n": self.heldout["dndx_lfmf_n"].tolist(),
                }
            ),
        }


def parse_record(raw, where="<incidence MF record>", *, expected_sha256=None):
    """An ``IncidenceMF`` from the file's bytes: with ``expected_sha256`` the bytes must hash to it BEFORE they are
    parsed; then strict JSON (no duplicate keys, no NaN or Infinity) and ``check_record``."""
    if type(raw) is not bytes:
        raise MFRecordError(f"{where}: the record is read from bytes")
    if len(raw) > MAX_BYTES:
        raise MFRecordError(f"{where}: {len(raw)} bytes exceed the bound {MAX_BYTES}")
    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        if type(expected_sha256) is not str or not _MR._HEX64.match(expected_sha256):
            raise MFRecordError(
                f"{where}: the published sha256 must be 64 lowercase hex digits, an exact str"
            )
        if sha != expected_sha256:
            raise MFRecordError(f"{where}: sha256 does not match the published record sha256")
    try:
        rec = S._json_loads(raw, where)
    except S.SafeFormatError as e:
        raise MFRecordError(str(e)) from None
    a = check_record(rec, where)
    heldout = a.pop("heldout")
    for v in list(a.values()) + list((heldout or {}).values()):
        v.setflags(write=False)
    return IncidenceMF(
        rec,
        sha,
        a["z"],
        a["n_sims"],
        a["dndx_ratio"],
        a["log_nhi_centres"],
        a["cddf_ratio"],
        a["cddf_n_pairs"],
        expected_sha256 is not None,
        raw,
        None if heldout is None else types.MappingProxyType(heldout),
    )


def load_record(record_dir, *, expected_record_sha256, lf_cache_sha256=None):
    """Load the record in ``record_dir`` anchored on its published sha256 (``None`` is refused). The directory must
    hold exactly ``incidence_mf_record.json``. ``lf_cache_sha256``, when given, must equal the record's LF cache.
    Returns an ``IncidenceMF``."""
    if expected_record_sha256 is None:
        raise MFRecordError("load_record needs the record's published sha256")
    d = os.path.abspath(os.fspath(record_dir))
    if not os.path.isdir(d) or os.path.islink(d):
        raise MFRecordError(f"{record_dir}: not an incidence MF record directory")
    p = os.path.join(d, RECORD_FILE)
    try:
        raw = S.read_regular(p, MAX_BYTES)
    except (S.SafeFormatError, OSError) as e:
        raise MFRecordError(f"{record_dir}: no readable {RECORD_FILE} ({e})") from None
    inc = parse_record(raw, p, expected_sha256=expected_record_sha256)
    extra = sorted(set(os.listdir(d)) - {RECORD_FILE})
    if extra:
        raise MFRecordError(f"{record_dir}: unexpected files {extra}")
    if lf_cache_sha256 is not None:
        if type(lf_cache_sha256) is not str:
            raise MFRecordError("load_record: lf_cache_sha256 must be an exact str")
        if inc.record["lf_cache"]["sha256"] != lf_cache_sha256:
            raise MFRecordError(
                f"{record_dir}: built from the LF cache {inc.record['lf_cache']['sha256'][:12]}, expected "
                f"{lf_cache_sha256[:12]}"
            )
    return inc


def _same_heldout(a, b):
    """Whether two parsed held-out errors are the same arrays (both absent counts as the same)."""
    if a is None or b is None:
        return a is None and b is None
    return (
        isinstance(a, (dict, types.MappingProxyType))
        and set(a) == set(b)
        and all(np.array_equal(np.asarray(a[k]), b[k], equal_nan=True) for k in b)
    )


def reverify(inc):
    """The record to APPLY, re-derived from a loaded record's own sealed bytes, as ``mf_release.reverify`` does for
    the P1D record: anything that is not an anchored ``IncidenceMF`` with exact ``bytes`` and ``str`` is refused, and
    so is an object whose record or tables differ from what its bytes give."""
    if (
        not isinstance(inc, IncidenceMF)
        or inc.anchored is not True
        or type(inc.record_bytes) is not bytes
        or type(inc.record_sha256) is not str
    ):
        raise MFRecordError(
            "the predictor takes an incidence MF record loaded by incidence_mf.load_record (sealed with its "
            "anchored bytes)"
        )
    new = parse_record(
        inc.record_bytes,
        f"the incidence MF record {inc.record_sha256[:12]}",
        expected_sha256=inc.record_sha256,
    )
    same = (
        isinstance(inc.record, dict)
        and inc.record == new.record
        and all(
            np.array_equal(np.asarray(getattr(inc, k)), getattr(new, k))
            for k in ("z", "n_sims", "dndx_ratio", "log_nhi_centres", "cddf_ratio", "cddf_n_pairs")
        )
        and _same_heldout(inc.heldout, new.heldout)
    )
    if not same:
        raise MFRecordError(
            f"the incidence MF record {inc.record_sha256[:12]}: its record or tables were changed in memory"
        )
    return new
