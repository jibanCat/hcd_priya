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
container: the record that carries a fitted table with its inputs, and its strict reader.

**Inputs.** The LF side is the LF training cache (its snapshot-level columns). The HR side is NOT the HR cache: the HR
cache's catalogue-derived columns predate a catalogue fix and stay refused by ``mf_release`` (PI ruling 2026-11-06b
A1). The HR side is an HR incidence product derived from the rebuilt HR catalogues with the functions that made the
LF cache columns (``hr_snap_from_product``). The record names both inputs by sha256 (``HR_SOURCE``).

**Layout.** One record directory holds exactly one file, ``incidence_mf_record.json``: strict UTF-8 JSON with a closed
key set (``RECORD_KEYS``) and finite numbers, the numbers of the tables included (they are a few hundred). A published
record is anchored by the sha256 of that file. Nothing is pickled or evaluated.
"""

from __future__ import annotations

import hashlib
import json
import os
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
    "simulations of HR / LF, per HCD class for dN/dX and per column-density bin for f(N_HI) (1 where no pair has "
    "both values positive); theta-independent; linear in z between the tabulated redshifts and constant beyond "
    "them (dndx_res_factor, cddf_res_factor); dndx_MF = dndx_LF * R_c(z)"
)
HR_SOURCE = (
    "HR dN/dX and f(N_HI) derived from the rebuilt HR absorber catalogues with the functions that made the LF "
    "cache's snapshot columns; not the HR cache's catalogue-derived columns, which mf_release refuses"
)
LF_SOURCE = "the LF training cache's snapshot columns (snap_dNdX_*, snap_f_nhi), read with multifidelity.load_snap_cache"
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
_TABLE_KEYS = {"z", "n_sims", "dndx_ratio", "log_nhi_centres", "cddf_ratio"}
_LF_KEYS = {"file", "sha256", "source"}
_HR_KEYS = {"file", "sha256", "source", "groups_not_in_product"}
_HR_ARRAYS = ("sim_name", "z", "params", "dndx", "cddf", "log_nhi_centres", "class_names")


# --------------------------------------------------------------------------------------------- the HR product
def hr_snap_from_product(path):
    """The HR incidence product (an ``.npz`` of plain arrays, read with ``allow_pickle=False``) as the snapshot dict
    that ``multifidelity.match_snap_hr_to_lf`` and the two table builders take: the keys of
    ``multifidelity.load_snap_cache``. Required arrays: ``sim_name`` (S,), ``z`` (S,), ``params`` (S, 9), ``dndx``
    (S, 3) in the order LLS, subDLA, DLA (``class_names`` must say so), ``cddf`` (S, Nbin), ``log_nhi_centres``
    (Nbin,)."""
    from .data import normalize_params

    try:
        with np.load(os.fspath(path), allow_pickle=False) as f:
            missing = [k for k in _HR_ARRAYS if k not in f.files]
            if missing:
                raise MFRecordError(f"{path}: the HR incidence product lacks the arrays {missing}")
            a = {k: f[k] for k in _HR_ARRAYS}
    except (OSError, ValueError) as e:  # not an .npz, or an array that would need pickle
        raise MFRecordError(f"{path}: unreadable HR incidence product ({e})") from None
    if [str(c) for c in a["class_names"]] != list(CLASS_ORDER):
        raise MFRecordError(f"{path}: class_names must be {list(CLASS_ORDER)}")
    S_ = a["sim_name"].shape[0]
    nb = a["log_nhi_centres"].shape[0]
    want = {"z": (S_,), "params": (S_, 9), "dndx": (S_, 3), "cddf": (S_, nb)}
    bad = [(k, a[k].shape, v) for k, v in want.items() if a[k].shape != v]
    if a["sim_name"].ndim != 1 or S_ < 1 or bad:
        raise MFRecordError(f"{path}: HR incidence arrays with inconsistent shapes: {bad}")
    params = np.asarray(a["params"], dtype=np.float64)
    return dict(
        dndx=np.asarray(a["dndx"], dtype=np.float64),
        f_nhi=np.asarray(a["cddf"], dtype=np.float64),
        log_nhi=np.asarray(a["log_nhi_centres"], dtype=np.float64),
        z=np.round(np.asarray(a["z"], dtype=np.float64), 4),
        params=params,
        pu=np.round(normalize_params(params), 6),
        sim=np.asarray(a["sim_name"]).astype(str),
    )


# --------------------------------------------------------------------------------------------- the fit
def fit_tables(lf_snap, hr_snap, *, exclude_sims=()):
    """Fit the two tables on the matched HR/LF snapshot pairs, with the historical functions.

    ``lf_snap`` and ``hr_snap`` are snapshot dicts (``multifidelity.load_snap_cache`` for the LF cache,
    ``hr_snap_from_product`` for the HR product). ``exclude_sims`` are HR simulation names the composition leaves
    out. Every remaining HR snapshot must have its LF partner (``match_snap_hr_to_lf``: same parameters and
    redshift); one without a partner is refused, never dropped.

    Returns ``(tables, train_sims, n_pairs)``: ``tables`` has ``z`` (Nz,), ``n_sims`` (Nz,) the number of pairs at
    each redshift, ``dndx_ratio`` (Nz, 3), ``log_nhi_centres`` (Nbin,) and ``cddf_ratio`` (Nz, Nbin).
    """
    from . import multifidelity as MF

    sims = np.asarray(hr_snap["sim"]).astype(str)
    ex = _MR._names(exclude_sims, "exclude_sims")
    unknown = sorted(set(ex) - set(sims))
    if unknown:
        raise MFRecordError(f"unknown simulations {unknown}: not among the HR simulations")
    keep = np.where(~np.isin(sims, ex))[0]
    if keep.size == 0:
        raise MFRecordError("no HR simulation left after the exclusions")
    hr = {
        k: (v if k == "log_nhi" else np.asarray(v)[keep]) for k, v in hr_snap.items()
    }  # log_nhi is per bin
    pairs = MF.match_snap_hr_to_lf(lf_snap, hr)
    if len(pairs) != keep.size:
        got = {h for h, _ in pairs}
        lost = sorted(
            {f"{hr['sim'][i]} z={hr['z'][i]:g}" for i in range(keep.size) if i not in got}
        )
        raise MFRecordError(
            f"HR snapshots without an LF partner (same parameters and redshift): {lost}"
        )
    if not np.allclose(lf_snap["log_nhi"], hr["log_nhi"], rtol=1e-12, atol=0):
        raise MFRecordError("the LF and HR column-density bins differ")
    h = np.array([p[0] for p in pairs])
    l = np.array([p[1] for p in pairs])
    z = np.asarray(hr["z"])[h]
    dn = MF.build_dndx_res_corr(lf_snap["dndx"][l], lf_snap["z"][l], hr["dndx"][h], z)
    cd = MF.build_cddf_res_corr(
        lf_snap["f_nhi"][l], lf_snap["z"][l], hr["f_nhi"][h], z, hr["log_nhi"]
    )
    if not np.array_equal(dn["z"], cd["z"]):
        raise MFRecordError("the dN/dX and CDDF tables came out on different redshifts")
    zr = np.round(z, 4)
    tables = {
        "z": np.asarray(dn["z"], dtype=np.float64),
        "n_sims": np.array([int(np.sum(zr == zz)) for zz in dn["z"]]),
        "dndx_ratio": np.asarray(dn["ratio"], dtype=np.float64),
        "log_nhi_centres": np.asarray(cd["log_nhi"], dtype=np.float64),
        "cddf_ratio": np.asarray(cd["ratio"], dtype=np.float64),
    }
    return tables, sorted(set(np.asarray(hr["sim"])[h])), len(pairs)


# --------------------------------------------------------------------------------------------- record checks
def _numbers(x, shape, where, what):
    try:
        return _WM._num_list(x, shape, where, what)
    except (_WM.WcMapError, ValueError) as e:  # ValueError: a ragged list
        raise MFRecordError(f"{where}: {what}: {e}") from None


def _check_tables(t, where):
    """The record's ``tables`` block: closed keys, plain finite numbers, consistent shapes, ascending redshifts and
    bins, positive ratios, at least one pair per redshift. Returns the arrays (float64; ``n_sims`` int).
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
    return a


def check_record(rec, where="incidence MF record"):
    """Validate a record dict: the closed schema, the constant statements, the tables. Raises MFRecordError. Returns
    the tables as arrays."""
    _MR._keys(rec, RECORD_KEYS, where)
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
):
    """Write a record directory ONCE: ``out_dir`` must not exist. ``tables`` as from ``fit_tables``; ``lf_cache`` and
    ``hr_incidence`` are ``{file, sha256}`` (the writer adds the code's statement of each source);
    ``groups_not_in_product`` names the HR snapshot groups the HR product does not hold (it may be empty).
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
                [int(v) for v in np.asarray(tables[k]).ravel()]
                if k == "n_sims"
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
    anchored: bool
    record_bytes: bytes = None

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
            "lf_cache": _MR.copy_json(r["lf_cache"]),
            "hr_incidence": _MR.copy_json(r["hr_incidence"]),
            "fit_code": _MR.copy_json(r["fit_code"]),
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
    for v in a.values():
        v.setflags(write=False)
    return IncidenceMF(
        rec,
        sha,
        a["z"],
        a["n_sims"],
        a["dndx_ratio"],
        a["log_nhi_centres"],
        a["cddf_ratio"],
        expected_sha256 is not None,
        raw,
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
            for k in ("z", "n_sims", "dndx_ratio", "log_nhi_centres", "cddf_ratio")
        )
    )
    if not same:
        raise MFRecordError(
            f"the incidence MF record {inc.record_sha256[:12]}: its record or tables were changed in memory"
        )
    return new
