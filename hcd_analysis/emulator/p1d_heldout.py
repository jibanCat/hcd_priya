"""The held-out prediction residual of the class P1D: an optional part of a release (format
``hcd_priya.p1d_heldout_record/1``; docs/release_format_h5.md, section "The held-out prediction residual of the P1D").

    <part dir>/
      p1d_heldout_record.json   the record: what the numbers are, the tables a prediction uses, the inputs by digest
      p1d_heldout_tables.h5     the evidence: every residual vector the tables were reduced from (hcd_priya.h5 table)

**What it is.** For a simulation that a model was made without, the residual is ``d = ln(prediction / truth)`` of a
class spectrum at one redshift, one mean flux and one mode. The record reduces the residuals of the held-out
simulations to ``e = sqrt(mean(d^2))`` per redshift, class and mode (``DEFINITION`` is the record's own statement). A
prediction carries ``e`` as ``P_class_validation_rms_log``, and the range ``P_class * exp(-e)`` to
``P_class * exp(+e)``. It is an
empirical validation diagnostic, not a calibrated predictive uncertainty: it rests on five or six simulations, and a
large part of it is present for simulations the model was trained on (the record's notes say what was measured).

**Two fidelities, each with the residual of its own prediction.** ``lf``: a fold backbone against the low-fidelity
simulation it was trained without. ``lfmf``: the same backbone with a multi-fidelity table fitted without the
high-resolution run of that design point, against that run. A prediction made with ``fidelity="LF"`` carries the
first, one made with LF+MF the second.

**Five rows.** The four classes in ``CLASS_ORDER``, then the mixture: the class spectra mixed with each held-out
simulation's own counted class fractions, prediction and truth alike. The four class residuals cannot be combined
without their correlations; the mixture row holds the result for those fractions.

**What a load checks.** The record's bytes against its published sha256 before they are parsed; the closed schema;
the constant statements; finite tables; the evidence file's size and sha256. ``verify_tables`` goes further and
recomputes every table of the record from the residual vectors of the evidence file (``check_release`` runs it).

**A record without its evidence file.** A record may be shipped alone: its ``tables`` key is then null, the
directory holds the record only, and ``verify_tables`` refuses it (nothing can be recomputed). The tables a
prediction uses are in the record either way.

The record is written by ``write_record`` (scripts/build_p1d_heldout.py makes one from a residual file). Nothing here
evaluates a model.
"""

from __future__ import annotations

import hashlib
import json
import os
import types
import warnings
from typing import NamedTuple

import numpy as np

from . import mf_release as _MR
from . import safe_io as S
from .mf_release import MFRecordError

RECORD_FORMAT = "hcd_priya.p1d_heldout_record/1"
RECORD_FILE = "p1d_heldout_record.json"
TABLES_FILE = "p1d_heldout_tables.h5"
KIND = "rms_log_heldout_prediction_residual"
CLASS_ORDER = ("clean", "LLS", "subDLA", "DLA")
ROWS = CLASS_ORDER + ("mixture",)  # the rows of every table, in this order
FIDELITIES = ("lf", "lfmf")
MAX_BYTES = S.MAX_JSON_BYTES
# The regimes the record's summary and a prediction's label keep apart. The largest modes are the first
# N_LARGEST_MODES modes of the box (below about 1e-3 s/km, the modes that training gave less weight); the typical
# regime is the redshift range of the grouped validation, above those modes.
N_LARGEST_MODES = 2
Z_TYPICAL = (2.2, 4.6)
_Z_TOL = 1e-6
REGIMES = ("typical", "largest_modes", "above_z_4p6", "above_z_4p6_largest_modes", "below_z_2p2")

# The record's statement of what the numbers are. A record whose statement differs was written by other code and is
# refused, so the words on a prediction are the ones reviewed here.
DEFINITION = (
    "A held-out prediction residual. For a simulation the model was made without, d = ln(prediction / truth) of a "
    "class spectrum at one redshift, one mean flux and one mode. e = sqrt(mean(d^2)) over the held-out simulations "
    "and over the mean-flux rungs of the training ladder, per redshift, class and mode; it is taken about zero, so "
    "a bias counts. n is the number of held-out simulations: the rungs of one snapshot are the same spectra with "
    "the optical depth rescaled, not independent points. e is a number on the logarithm, not on the spectrum: on "
    "the spectrum itself the held-out range of a value v is v * exp(-e) to v * exp(+e), each end computed on its "
    "own. The fifth row, the mixture, is the same quantity for the class spectra mixed with each held-out "
    "simulation's own counted class fractions. It is an empirical validation diagnostic, not a calibrated predictive "
    "uncertainty. No interpolation between redshifts."
)
FIDELITY_STATEMENT = {
    "lf": "a fold backbone against the low-fidelity simulation it was trained without",
    "lfmf": "a fold backbone with a multi-fidelity table fitted without the high-resolution run of the same design "
    "point, against that run",
}

RECORD_KEYS = {
    "format",
    "version",
    "kind",
    "definition",
    "class_order",
    "n_modes",
    "n_rung",
    "lf",
    "lfmf",
    "sensitivity",
    "summary",
    "what_it_does_not_cover",
    "notes",
    "inputs",
    "tables",
    "code",
}
_FID_KEYS = {"what", "z", "sims", "e", "n"}
_SENS_KEYS = {"statement", "fidelity", "without_sim", "z", "e", "n"}
_INPUT_KEYS = {"lf_cache", "hr_cache", "folds", "residuals"}
_FOLD_KEYS = {
    "fold",
    "held_out_lf_sim",
    "backbone_manifest_sha256",
    "mf_record_sha256",
    "hr_sim",
}
_TABLES_KEYS = {"file", "sha256", "bytes"}
_NOTE_KEYS = {
    "training_comparison",
    "anomalous_rows",
    "design_point_trend",
    "truth_sampling_noise",
}
_STAT_KEYS = {"median", "max"}


def design_point(name):
    """The design point of a simulation name: its leading ``ns...Ap...`` part. One high-resolution run is spelled
    with other last digits of two parameters than its low-fidelity partner; the design point is the same.
    """
    return str(name).split("herei")[0]


# --------------------------------------------------------------------------------------------- reduction
def reduce_residuals(d):
    """The tables of one fidelity from its residual vectors ``d`` (simulation, redshift, rung, 5 rows, mode; NaN:
    no value). The stored values are used as they are (float32), in float64 arithmetic. Returns a dict:
    ``e`` and ``b`` (redshift, 5, mode): sqrt(mean(d^2)) and mean(d) over simulations and rungs; ``n``
    (redshift, 5): the simulations with a value; ``e_rung`` (redshift, rung, 5, mode, float32): sqrt(mean(d^2))
    over simulations, per rung."""
    x = np.asarray(d).astype(np.float64)
    if x.ndim != 5 or x.shape[3] != len(ROWS):
        raise MFRecordError(
            f"residuals must be (simulation, redshift, rung, {len(ROWS)}, mode), got {x.shape}"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a cell without any value gives NaN
        e = np.sqrt(np.nanmean(x**2, axis=(0, 2)))
        b = np.nanmean(x, axis=(0, 2))
        e_rung = np.sqrt(np.nanmean(x**2, axis=0)).astype(np.float32)
    n = np.isfinite(x).any(axis=(2, 4)).sum(axis=0).astype(np.int64)
    return {"e": e, "b": b, "n": n, "e_rung": e_rung}


def summarise(e, z):
    """The median and the largest value of ``e`` (redshift, 5, mode) in each regime of ``REGIMES``, over the five
    rows and the modes. A regime without any redshift is absent."""
    e, z = np.asarray(e, dtype=np.float64), np.asarray(z, dtype=np.float64)
    lo, hi = Z_TYPICAL
    zr = {
        "typical": (z >= lo - _Z_TOL) & (z <= hi + _Z_TOL),
        "above_z_4p6": z > hi + _Z_TOL,
        "below_z_2p2": z < lo - _Z_TOL,
    }
    cells = {
        "typical": e[zr["typical"]][:, :, N_LARGEST_MODES:],
        "largest_modes": e[zr["typical"]][:, :, :N_LARGEST_MODES],
        "above_z_4p6": e[zr["above_z_4p6"]][:, :, N_LARGEST_MODES:],
        "above_z_4p6_largest_modes": e[zr["above_z_4p6"]][:, :, :N_LARGEST_MODES],
        "below_z_2p2": e[zr["below_z_2p2"]],
    }
    return {
        k: {"median": float(np.median(v)), "max": float(np.max(v))}
        for k, v in cells.items()
        if v.size
    }


def _size(x):
    """How a label writes one value of e: the number on the natural logarithm and, in brackets, the range it gives
    on the spectrum, the value times exp(-e) to the value times exp(+e). e is never written as a percentage: for a
    large e the two ends of the range are far from equal distances."""
    return f"{x:.2g} (times {np.exp(-x):.3f} to times {np.exp(x):.3f})"


def summary_sentences(summary):
    """The three statements a label keeps apart, from one fidelity's summary: the typical residual, the largest
    modes, and the redshifts above the validated range (and, for LF, the one below it)."""
    s = summary
    out = [
        f"Typical (z {Z_TYPICAL[0]} to {Z_TYPICAL[1]}, above the {N_LARGEST_MODES} largest modes of the box): e has "
        f"a median of {_size(s['typical']['median'])} and reaches {_size(s['typical']['max'])}.",
        f"The {N_LARGEST_MODES} largest modes, that is the first {N_LARGEST_MODES} modes of the box by index (for "
        "example near 4.8e-4 and 9.5e-4 s/km at z = 3 for the 120 Mpc/h box; training gave less weight below "
        "1e-3 s/km), same "
        f"redshifts: a median of {_size(s['largest_modes']['median'])}, up to "
        f"{_size(s['largest_modes']['max'])}.",
        f"Above z {Z_TYPICAL[1]} (outside the validated redshift range): a median of "
        f"{_size(s['above_z_4p6']['median'])}, up to {_size(s['above_z_4p6']['max'])}, and at the "
        f"{N_LARGEST_MODES} largest modes a median of {_size(s['above_z_4p6_largest_modes']['median'])}, "
        f"up to {_size(s['above_z_4p6_largest_modes']['max'])}.",
    ]
    if "below_z_2p2" in s:
        out.append(
            f"Below z {Z_TYPICAL[0]}: a median of {_size(s['below_z_2p2']['median'])}, up to "
            f"{_size(s['below_z_2p2']['max'])}."
        )
    return out


# --------------------------------------------------------------------------------------------- validation
def _text(v, where):
    if not isinstance(v, str) or not v.strip():
        raise MFRecordError(f"{where} must be a non-empty text")


def _table(v, shape, where):
    """A finite table of numbers >= 0 with the given shape (a bool or a nested oddity is refused)."""
    try:
        a = np.array(v, dtype=np.float64)
    except (TypeError, ValueError):
        raise MFRecordError(f"{where} must be a table of numbers") from None
    if a.shape != shape or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise MFRecordError(f"{where} must have shape {shape} and hold finite numbers >= 0")
    return a


def _counts(v, shape, where, largest):
    a = np.asarray(v)
    if (
        a.shape != shape
        or a.dtype.kind not in "iu"
        or np.any(a < 1)
        or np.any(a > largest)
        or any(type(x) is not int for x in np.asarray(v, dtype=object).ravel())
    ):
        raise MFRecordError(
            f"{where} must have shape {shape} and hold integers from 1 to the number of simulations ({largest})"
        )
    return a.astype(np.int64)


def _zs(v, where):
    if not isinstance(v, list) or not v:
        raise MFRecordError(f"{where} must be a non-empty list")
    z = _table(v, (len(v),), where)
    if np.any(np.diff(z) <= 0):
        raise MFRecordError(f"{where} must be strictly ascending")
    return z


def _stats(v, where, need):
    if not isinstance(v, dict) or not need <= set(v) or set(v) - set(REGIMES):
        raise MFRecordError(
            f"{where}: regimes must be among {list(REGIMES)} and hold {sorted(need)}"
        )
    for k, s in v.items():
        _MR._keys(s, _STAT_KEYS, f"{where}.{k}")
        for q in _STAT_KEYS:
            if type(s[q]) is not float or not np.isfinite(s[q]) or s[q] < 0:
                raise MFRecordError(f"{where}.{k}.{q} must be a finite number >= 0")


def check_record(rec, where="P1D held-out record"):
    """Validate a record dict: the closed schema, the constant statements, the tables, and the summary against the
    tables. Raises MFRecordError. Returns the arrays: ``z_<fid>``, ``e_<fid>``, ``n_<fid>`` for the two fidelities,
    and ``z_sens``, ``e_sens``, ``n_sens`` for the sensitivity table."""
    _MR._keys(rec, RECORD_KEYS, where)
    if rec["format"] != RECORD_FORMAT:
        raise MFRecordError(f"{where}: format {rec['format']!r}, expected {RECORD_FORMAT}")
    _text(rec["version"], f"{where}: version")
    if rec["kind"] != KIND:
        raise MFRecordError(f"{where}: kind is {rec['kind']!r}; this code reads {KIND!r} only")
    if rec["definition"] != DEFINITION:
        raise MFRecordError(f"{where}: definition is not the statement this code carries")
    if rec["class_order"] != list(CLASS_ORDER):
        raise MFRecordError(f"{where}: class_order must be {list(CLASS_ORDER)}")
    for k in ("n_modes", "n_rung"):
        if type(rec[k]) is not int or rec[k] < 1:
            raise MFRecordError(f"{where}: {k} must be an integer >= 1")
    K = rec["n_modes"]
    if K <= N_LARGEST_MODES:
        raise MFRecordError(f"{where}: n_modes must exceed {N_LARGEST_MODES}")
    out = {}
    for fid in FIDELITIES:
        t = rec[fid]
        _MR._keys(t, _FID_KEYS, f"{where}: {fid}")
        if t["what"] != FIDELITY_STATEMENT[fid]:
            raise MFRecordError(
                f"{where}: {fid}.what is not the statement this code carries for that fidelity"
            )
        z = _zs(t["z"], f"{where}: {fid}.z")
        sims = _MR._sorted_names(t["sims"], f"{where}: {fid}.sims")
        if not sims:
            raise MFRecordError(f"{where}: {fid}.sims must name the held-out simulations")
        out[f"z_{fid}"] = z
        out[f"e_{fid}"] = _table(t["e"], (z.size, len(ROWS), K), f"{where}: {fid}.e")
        out[f"n_{fid}"] = _counts(t["n"], (z.size, len(ROWS)), f"{where}: {fid}.n", len(sims))
    sens = rec["sensitivity"]
    _MR._keys(sens, _SENS_KEYS, f"{where}: sensitivity")
    _text(sens["statement"], f"{where}: sensitivity.statement")
    if sens["fidelity"] not in FIDELITIES:
        raise MFRecordError(f"{where}: sensitivity.fidelity must be one of {list(FIDELITIES)}")
    if sens["without_sim"] not in rec[sens["fidelity"]]["sims"]:
        raise MFRecordError(
            f"{where}: sensitivity.without_sim must be one of the held-out simulations of that fidelity"
        )
    zs = _zs(sens["z"], f"{where}: sensitivity.z")
    if not np.array_equal(zs, out[f"z_{sens['fidelity']}"]):
        raise MFRecordError(f"{where}: sensitivity.z must be the redshifts of its fidelity")
    out["z_sens"] = zs
    out["e_sens"] = _table(sens["e"], (zs.size, len(ROWS), K), f"{where}: sensitivity.e")
    out["n_sens"] = _counts(
        sens["n"],
        (zs.size, len(ROWS)),
        f"{where}: sensitivity.n",
        len(rec[sens["fidelity"]]["sims"]) - 1,
    )
    # the summary is a reduction of the tables: a record whose summary says anything else is refused
    summ = rec["summary"]
    _MR._keys(summ, set(FIDELITIES) | {"sensitivity"}, f"{where}: summary")
    need = {"typical", "largest_modes", "above_z_4p6", "above_z_4p6_largest_modes"}
    for key, e, z in (
        ("lf", out["e_lf"], out["z_lf"]),
        ("lfmf", out["e_lfmf"], out["z_lfmf"]),
        ("sensitivity", out["e_sens"], out["z_sens"]),
    ):
        _stats(summ[key], f"{where}: summary.{key}", need)
        if summ[key] != summarise(e, z):
            raise MFRecordError(f"{where}: summary.{key} is not the reduction of its table")
    _text(rec["what_it_does_not_cover"], f"{where}: what_it_does_not_cover")
    _MR._keys(rec["notes"], _NOTE_KEYS, f"{where}: notes")
    for k in _NOTE_KEYS:
        _text(rec["notes"][k], f"{where}: notes.{k}")
    inp = rec["inputs"]
    _MR._keys(inp, _INPUT_KEYS, f"{where}: inputs")
    for side in ("lf_cache", "hr_cache", "residuals"):
        _MR._keys(inp[side], {"file", "sha256"}, f"{where}: inputs.{side}")
        if not isinstance(inp[side]["file"], str) or not _MR._PLAIN.match(inp[side]["file"]):
            raise MFRecordError(
                f"{where}: inputs.{side}.file must be a plain file name, never a path"
            )
        _MR._hex(inp[side]["sha256"], 64, f"{where}: inputs.{side}.sha256")
    folds = inp["folds"]
    if not isinstance(folds, list) or not folds:
        raise MFRecordError(f"{where}: inputs.folds must be a non-empty list")
    for i, f in enumerate(folds):
        w = f"{where}: inputs.folds[{i}]"
        _MR._keys(f, _FOLD_KEYS, w)
        _text(f["fold"], f"{w}.fold")
        _text(f["held_out_lf_sim"], f"{w}.held_out_lf_sim")
        _MR._hex(f["backbone_manifest_sha256"], 64, f"{w}.backbone_manifest_sha256")
        if (f["mf_record_sha256"] is None) != (f["hr_sim"] is None):
            raise MFRecordError(f"{w}: mf_record_sha256 and hr_sim are both given, or both null")
        if f["mf_record_sha256"] is not None:
            _MR._hex(f["mf_record_sha256"], 64, f"{w}.mf_record_sha256")
            _text(f["hr_sim"], f"{w}.hr_sim")
    if sorted(f["held_out_lf_sim"] for f in folds) != rec["lf"]["sims"]:
        raise MFRecordError(f"{where}: lf.sims must be the held-out simulations of inputs.folds")
    if sorted(f["hr_sim"] for f in folds if f["hr_sim"] is not None) != rec["lfmf"]["sims"]:
        raise MFRecordError(f"{where}: lfmf.sims must be the high-resolution runs of inputs.folds")
    tb = rec["tables"]
    if tb is None:  # shipped without its evidence file
        tb = {"file": TABLES_FILE, "sha256": "0" * 64, "bytes": 1}
    _MR._keys(tb, _TABLES_KEYS, f"{where}: tables")
    if tb["file"] != TABLES_FILE:
        raise MFRecordError(f"{where}: tables.file must be {TABLES_FILE!r}")
    _MR._hex(tb["sha256"], 64, f"{where}: tables.sha256")
    if type(tb["bytes"]) is not int or tb["bytes"] < 1:
        raise MFRecordError(f"{where}: tables.bytes must be an integer >= 1")
    code = rec["code"]
    _MR._keys(code, {"commit", "dirty"}, f"{where}: code")
    _MR._hex(code["commit"], 40, f"{where}: code.commit")
    if code["dirty"] is not False:
        raise MFRecordError(f"{where}: the residuals are computed from a clean tree")
    return out


# --------------------------------------------------------------------------------------------- the evidence file
def table_datasets(rec):
    """{dataset name: (dtype, shape)} of the evidence file that belongs to a record."""
    K, R = rec["n_modes"], rec["n_rung"]
    out = {"k_com_hmpc": ("float64", (K,))}
    for fid in FIDELITIES:
        S_, Z = len(rec[fid]["sims"]), len(rec[fid]["z"])
        out.update(
            {
                f"{fid}_d": ("float32", (S_, Z, R, len(ROWS), K)),
                f"{fid}_target_F": ("float64", (S_, Z, R)),
                f"{fid}_counts": ("int64", (S_, Z, R, len(CLASS_ORDER))),
                f"{fid}_b": ("float64", (Z, len(ROWS), K)),
                f"{fid}_e_rung": ("float32", (Z, R, len(ROWS), K)),
            }
        )
    return out


def _sens_d(rec, d):
    """The residual vectors of the sensitivity table: those of its fidelity without the named simulation (the
    simulation axis follows the record's sorted names)."""
    sims = rec[rec["sensitivity"]["fidelity"]]["sims"]
    keep = [i for i, s in enumerate(sims) if s != rec["sensitivity"]["without_sim"]]
    return np.asarray(d)[keep]


def verify_tables(record_dir, heldout, rtol=1e-12):
    """Recompute every table of a loaded record from the residual vectors of its evidence file, and compare: the
    tables a prediction uses, the counts, the sensitivity table, the bias and the per-rung tables of the file.
    Raises MFRecordError on the first difference. Returns the number of residual values read."""
    rec = heldout.record
    if rec["tables"] is None:
        raise MFRecordError(
            f"{record_dir}: this record was shipped without its evidence file: its tables cannot be recomputed"
        )
    path = os.path.join(os.fspath(record_dir), TABLES_FILE)
    try:
        arrays, _attrs = S.read_table_h5(path, table_datasets(rec))
    except (S.SafeFormatError, OSError) as e:
        raise MFRecordError(
            f"{record_dir}: the evidence file is not as the record describes ({e})"
        ) from None
    total = 0
    for fid in FIDELITIES:
        d = arrays[f"{fid}_d"]
        total += int(np.isfinite(d).sum())
        red = reduce_residuals(d)
        checks = [
            (f"{fid}.e", red["e"], heldout.e[fid]),
            (f"{fid}_b", red["b"], arrays[f"{fid}_b"]),
            (f"{fid}_e_rung", red["e_rung"], arrays[f"{fid}_e_rung"]),
        ]
        if rec["sensitivity"]["fidelity"] == fid:
            sred = reduce_residuals(_sens_d(rec, d))
            checks.append(("sensitivity.e", sred["e"], heldout.e_sens))
            if not np.array_equal(sred["n"], heldout.n_sens):
                raise MFRecordError(f"{record_dir}: sensitivity.n is not what the residuals give")
        for name, got, want in checks:
            if not np.allclose(got, want, rtol=rtol, atol=0, equal_nan=True):
                raise MFRecordError(f"{record_dir}: {name} is not what the residuals give")
        if not np.array_equal(red["n"], heldout.n[fid]):
            raise MFRecordError(f"{record_dir}: {fid}.n is not what the residuals give")
    return total


# --------------------------------------------------------------------------------------------- write
def write_record(
    out_dir,
    *,
    version,
    k_com_hmpc,
    residuals,
    sensitivity_without,
    sensitivity_statement,
    what_it_does_not_cover,
    notes,
    inputs,
    code,
    with_evidence=True,
):
    """Write a record directory ONCE: ``out_dir`` must not exist.

    ``residuals`` is ``{"lf": {...}, "lfmf": {...}}``; each holds ``sims`` (names, in the order of the first axis
    of ``d``), ``z`` (ascending), ``d`` (simulation, redshift, rung, 5, mode), ``target_F`` (simulation, redshift,
    rung) and ``counts`` (simulation, redshift, rung, 4). The simulations are stored in the sorted order of their
    names. ``sensitivity_without`` names one low-fidelity simulation: the LF table is also reduced without it and
    stored as the sensitivity table, with ``sensitivity_statement``. ``inputs`` and ``code`` are the record's blocks
    of those names without the evidence file. With ``with_evidence=False`` the evidence file is not written and the
    record's ``tables`` is null: the residual vectors stay with whoever measured them. Everything is validated before
    anything is written, and what is written is read back. Returns (the record's path, its sha256: the digest to
    publish)."""
    out_dir = os.fspath(out_dir)
    if os.path.lexists(out_dir):
        raise FileExistsError(
            f"{out_dir} exists: a P1D held-out record is written once into a new directory"
        )
    k_com = np.asarray(k_com_hmpc, dtype=np.float64)
    arrays = {"k_com_hmpc": k_com}
    rec = {
        "format": RECORD_FORMAT,
        "version": version,
        "kind": KIND,
        "definition": DEFINITION,
        "class_order": list(CLASS_ORDER),
        "n_modes": int(k_com.size),
        "what_it_does_not_cover": what_it_does_not_cover,
        "notes": _MR.copy_json(notes),
        "inputs": _MR.copy_json(inputs),
        "code": _MR.copy_json(code),
        "summary": {},
    }
    n_rung = None
    for fid in FIDELITIES:
        r = residuals[fid]
        order = np.argsort(np.array([str(s) for s in r["sims"]]))
        sims = [str(r["sims"][i]) for i in order]
        d = np.asarray(r["d"], dtype=np.float32)[order]
        z = np.asarray(r["z"], dtype=np.float64)
        n_rung = d.shape[2] if n_rung is None else n_rung
        if d.shape[:2] != (len(sims), z.size) or d.shape[2] != n_rung or d.shape[4] != k_com.size:
            raise MFRecordError(f"{fid}: the residuals do not have the shape of their keys")
        red = reduce_residuals(d)
        rec[fid] = {
            "what": FIDELITY_STATEMENT[fid],
            "z": z.tolist(),
            "sims": sims,
            "e": red["e"].tolist(),
            "n": red["n"].tolist(),
        }
        rec["summary"][fid] = summarise(red["e"], z)
        arrays[f"{fid}_d"] = d
        arrays[f"{fid}_target_F"] = np.asarray(r["target_F"], dtype=np.float64)[order]
        arrays[f"{fid}_counts"] = np.asarray(r["counts"], dtype=np.int64)[order]
        arrays[f"{fid}_b"] = red["b"]
        arrays[f"{fid}_e_rung"] = red["e_rung"]
    rec["n_rung"] = int(n_rung)
    if sensitivity_without not in rec["lf"]["sims"]:
        raise MFRecordError("sensitivity_without must be one of the held-out LF simulations")
    rec["sensitivity"] = {
        "statement": sensitivity_statement,
        "fidelity": "lf",
        "without_sim": sensitivity_without,
        "z": rec["lf"]["z"],
    }
    sred = reduce_residuals(_sens_d(rec, arrays["lf_d"]))
    rec["sensitivity"].update(e=sred["e"].tolist(), n=sred["n"].tolist())
    rec["summary"]["sensitivity"] = summarise(sred["e"], np.asarray(rec["lf"]["z"]))
    rec["tables"] = {"file": TABLES_FILE, "sha256": "0" * 64, "bytes": 1}  # filled below
    try:
        raw = json.dumps(rec, indent=1, sort_keys=True, allow_nan=False).encode("utf-8")
    except ValueError as e:  # a NaN or an infinity: a cell without any held-out value
        raise MFRecordError(f"the tables must be finite ({e})") from None
    parse_record(raw, "the record to write")  # refused before anything is written
    want = table_datasets(rec)
    for name, (dt, shape) in want.items():
        a = arrays[name]
        if a.dtype != np.dtype(dt) or a.shape != shape:
            raise MFRecordError(f"{name} is {a.dtype}{a.shape}, the record needs {dt}{shape}")
    os.makedirs(out_dir)  # no exist_ok: a race is refused too
    if with_evidence:
        tpath = os.path.join(out_dir, TABLES_FILE)
        S.write_table_h5(tpath, {k: arrays[k] for k in want}, {"record_format": RECORD_FORMAT})
        sha, size = S._sha256_regular(tpath)
        rec["tables"] = {"file": TABLES_FILE, "sha256": sha, "bytes": size}
    else:
        rec["tables"] = None
    raw = json.dumps(rec, indent=1, sort_keys=True, allow_nan=False).encode("utf-8")
    path = os.path.join(out_dir, RECORD_FILE)
    digest = S._write_exclusive(path, [raw])
    held = load_record(out_dir, expected_record_sha256=digest)
    if with_evidence:
        verify_tables(out_dir, held)
    return path, digest


# --------------------------------------------------------------------------------------------- load
class P1DHeldout(NamedTuple):
    """A loaded P1D held-out record, anchored on its published sha256. The arrays are read-only float64 (counts:
    int64). It is sealed like the other records: ``record_bytes`` are the anchored bytes it was parsed from, and the
    predictor applies a record only through ``reverify``."""

    record: dict
    record_sha256: str
    z: types.MappingProxyType  # {"lf": (Nz,), "lfmf": (Nz',)}
    e: types.MappingProxyType  # {"lf": (Nz, 5, K), "lfmf": (Nz', 5, K)}
    n: types.MappingProxyType  # {"lf": (Nz, 5), "lfmf": (Nz', 5)}
    e_sens: np.ndarray  # the sensitivity table, (Nz, 5, K) of its fidelity
    n_sens: np.ndarray
    anchored: bool
    record_bytes: bytes = None

    @property
    def n_modes(self):
        return int(self.record["n_modes"])

    def err(self, z, lfmf):
        """(5, K): e of the four classes and of the mixture at a TABULATED redshift, for the LF+MF prediction
        (``lfmf``) or the LF one; NaN at any other redshift (a residual is never interpolated between redshifts).
        """
        fid = "lfmf" if lfmf else "lf"
        hit = np.where(np.abs(self.z[fid] - float(z)) <= _Z_TOL)[0]
        if hit.size != 1:
            return np.full((len(ROWS), self.n_modes), np.nan)
        return np.array(self.e[fid][hit[0]], dtype=np.float64)

    def label(self, lfmf):
        """The words a prediction carries about its ``P_class_validation_rms_log``: what it is, the three regimes kept apart, the
        record's notes and what the numbers do not cover."""
        fid = "lfmf" if lfmf else "lf"
        r = self.record
        nmax = int(self.n[fid].max())
        parts = [
            "P_class_validation_rms_log and P_mix_validation_rms_log are the held-out prediction residual of this "
            f"prediction's fidelity ({'LF+MF' if lfmf else 'LF'}: {FIDELITY_STATEMENT[fid]}; {nmax} held-out "
            "simulations). They are empirical validation diagnostics, not calibrated predictive uncertainties. ",
            r["definition"],
            " SIZES: each is e, a number on the natural logarithm, and in brackets the range it gives on a class "
            "spectrum, the value times exp(-e) to the value times exp(+e). ",
            " ".join(summary_sentences(r["summary"][fid])),
        ]
        if r["sensitivity"]["fidelity"] == fid:
            s = r["summary"]["sensitivity"]
            parts.append(
                f" SENSITIVITY TEST, not the table of a prediction: {r['sensitivity']['statement']} Without that "
                f"simulation, above z {Z_TYPICAL[1]}: a median of {_size(s['above_z_4p6']['median'])}, up "
                f"to {_size(s['above_z_4p6']['max'])}, and at the {N_LARGEST_MODES} largest modes up to "
                f"{_size(s['above_z_4p6_largest_modes']['max'])}."
            )
        parts += [
            " NOTES. ",
            " ".join(r["notes"][k] for k in sorted(_NOTE_KEYS)),
            " WHAT IT DOES NOT COVER. ",
            r["what_it_does_not_cover"],
            (
                " The residual vectors behind these tables are not shipped with this record."
                if r["tables"] is None
                else ""
            ),
            " P_mix_validation_rms_log is the residual of the mixture at the held-out simulations' own class fractions; for other "
            "fractions it is an indication only. With a requested k, e is carried to the requested wavenumbers by "
            "the operation that carries P_class there.",
        ]
        return "".join(parts)

    def provenance(self):
        r = self.record
        return {
            "record_sha256": self.record_sha256,
            "version": r["version"],
            "kind": r["kind"],
            "definition": r["definition"],
            "what_it_does_not_cover": r["what_it_does_not_cover"],
            "notes": _MR.copy_json(r["notes"]),
            "summary": _MR.copy_json(r["summary"]),
            "sensitivity": {
                k: _MR.copy_json(r["sensitivity"][k])
                for k in ("statement", "fidelity", "without_sim")
            },
            "lf": {
                "z": list(r["lf"]["z"]),
                "sims": list(r["lf"]["sims"]),
                "n": self.n["lf"].tolist(),
            },
            "lfmf": {
                "z": list(r["lfmf"]["z"]),
                "sims": list(r["lfmf"]["sims"]),
                "n": self.n["lfmf"].tolist(),
            },
            "inputs": _MR.copy_json(r["inputs"]),
            "tables": _MR.copy_json(r["tables"]),
            "code": _MR.copy_json(r["code"]),
        }


def parse_record(raw, where="<P1D held-out record>", *, expected_sha256=None):
    """A ``P1DHeldout`` from the file's bytes: with ``expected_sha256`` the bytes must hash to it BEFORE they are
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
    return P1DHeldout(
        rec,
        sha,
        types.MappingProxyType({f: a[f"z_{f}"] for f in FIDELITIES}),
        types.MappingProxyType({f: a[f"e_{f}"] for f in FIDELITIES}),
        types.MappingProxyType({f: a[f"n_{f}"] for f in FIDELITIES}),
        a["e_sens"],
        a["n_sens"],
        expected_sha256 is not None,
        raw,
    )


def load_record(record_dir, *, expected_record_sha256):
    """Load the record in ``record_dir`` anchored on its published sha256 (``None`` is refused). The directory must
    hold exactly the record and its evidence file, and the evidence file must have the size and the sha256 the
    record states (its content is compared with the record's tables by ``verify_tables``, not here). Returns a
    ``P1DHeldout``."""
    if expected_record_sha256 is None:
        raise MFRecordError("load_record needs the record's published sha256")
    d = os.path.abspath(os.fspath(record_dir))
    if not os.path.isdir(d) or os.path.islink(d):
        raise MFRecordError(f"{record_dir}: not a P1D held-out record directory")
    p = os.path.join(d, RECORD_FILE)
    try:
        raw = S.read_regular(p, MAX_BYTES)
    except (S.SafeFormatError, OSError) as e:
        raise MFRecordError(f"{record_dir}: no readable {RECORD_FILE} ({e})") from None
    held = parse_record(raw, p, expected_sha256=expected_record_sha256)
    tb = held.record["tables"]
    extra = sorted(
        set(os.listdir(d)) - ({RECORD_FILE} if tb is None else {RECORD_FILE, TABLES_FILE})
    )
    if extra:
        raise MFRecordError(f"{record_dir}: unexpected files {extra}")
    if tb is None:  # shipped without its evidence file: the record alone
        return held
    try:
        sha, _ = S._sha256_regular(os.path.join(d, TABLES_FILE), expected_bytes=tb["bytes"])
    except (S.SafeFormatError, OSError) as e:
        raise MFRecordError(f"{record_dir}: the evidence file {TABLES_FILE} ({e})") from None
    if sha != tb["sha256"]:
        raise MFRecordError(
            f"{record_dir}: {TABLES_FILE} does not have the sha256 the record states"
        )
    return held


def reverify(held):
    """The record to APPLY, re-derived from a loaded record's own sealed bytes: anything that is not an anchored
    ``P1DHeldout`` with exact ``bytes`` and ``str`` is refused, and so is an object whose record or tables differ
    from what its bytes give."""
    if (
        not isinstance(held, P1DHeldout)
        or held.anchored is not True
        or type(held.record_bytes) is not bytes
        or type(held.record_sha256) is not str
    ):
        raise MFRecordError(
            "the predictor takes a P1D held-out record loaded by p1d_heldout.load_record (sealed with its "
            "anchored bytes)"
        )
    new = parse_record(
        held.record_bytes,
        f"the P1D held-out record {held.record_sha256[:12]}",
        expected_sha256=held.record_sha256,
    )

    def same_map(a, b):
        return (
            isinstance(a, (dict, types.MappingProxyType))
            and set(a) == set(b)
            and all(np.array_equal(np.asarray(a[k]), b[k]) for k in b)
        )

    same = (
        isinstance(held.record, dict)
        and held.record == new.record
        and same_map(held.z, new.z)
        and same_map(held.e, new.e)
        and same_map(held.n, new.n)
        and np.array_equal(np.asarray(held.e_sens), new.e_sens)
        and np.array_equal(np.asarray(held.n_sens), new.n_sens)
    )
    if not same:
        raise MFRecordError(
            f"the P1D held-out record {held.record_sha256[:12]}: its record or tables were changed in memory"
        )
    return new
