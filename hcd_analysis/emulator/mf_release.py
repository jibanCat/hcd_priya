"""The release multi-fidelity (MF) record: format ``hcd_priya.mf_record/1`` (contract in docs/release_format_h5.md,
section "The multi-fidelity record").

**Physical meaning.** The LF emulator predicts the per-class PRIYA-filtered P1D of the low-resolution simulations. The
MF correction g(z, tau0, k) = log P_HR - log P_LF, measured mode by mode on matched HR/LF pairs and fitted as the
theta-independent table model of ``mf_modes.fit_mode_mf``, turns an LF prediction into an HR-calibrated one:
P_MF = P_LF exp(g). g depends on the LF backbone that formed its targets, so a record is bound to exactly one backbone
bundle (its manifest sha256) and is refused with any other.

**Layout.** One record directory holds exactly two files:

  ``mf_tables.h5``    root attribute ``format = "hcd_priya.mf/1"`` and the group ``/mf`` holding one contiguous,
                      unfiltered, attribute-free float64 dataset per table (``mf_modes.TABLE_KEYS``) plus ``k_com_hmpc``
                      (the box modes 2 pi n / L the tables are labelled by); nothing else.
  ``mf_record.json``  the record (closed key set, ``RECORD_KEYS``): the MF version, the kind (``release``, ``fold`` or
                      ``historical``), the tables file's sha256 and shapes, the backbone's manifest sha256 and kind, the
                      LF and HR caches' file names, sha256 and provenance (``CACHE_PROVENANCE``: the LF cache is
                      lineage-verified through the backbone, the HR cache digest-anchored only), the composition
                      with its ruling ids, the fold (for a held-out fit), the training simulations, the HR inputs used (``HR_INPUTS``), the mean-flux
                      convention of the targets (``MEAN_FLUX_CONVENTION``, PROPOSED) and the fit code's commit and
                      clean-tree flag.

A published record is anchored by the sha256 of ``mf_record.json``; the record anchors the tables. Nothing is pickled
or evaluated: the tables are read from anchored bytes with the strict HDF5 walk of ``safe_h5``.

**Kinds.** ``release``: fitted on every simulation of its composition with a production (release) backbone, from a
clean tree; the only kind the public predictor applies to a production bundle. ``fold``: a held-out fit (one group of
simulations held out, ``select_train_rows``) for grouped cross-validation; its backbone may be a research bundle that
excludes the held-out group's LF partner, and the record states whether it does (``backbone_excludes_holdout``).
``historical``: loadable only when asked for by kind; never applied as a release MF.

**HR inputs (PI ruling 2026-11-06b A1).** The MF uses only the HR per-class P1D rows and their row keys
(``HR_INPUTS["used"]``). The HR cache's catalogue-derived columns (dN/dX, f(N_HI), absorber counts, path length) are
pre-fix; ``hr_view`` refuses them and a record that names them, or claims any status for them other than the one in
``HR_INPUTS``, is refused."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np

from . import safe_io as S
from .schema import L_BOX_HMPC

RECORD_FORMAT = "hcd_priya.mf_record/1"
TABLES_FORMAT = "hcd_priya.mf/1"
RECORD_FILE = "mf_record.json"
TABLES_FILE = "mf_tables.h5"
KINDS = ("release", "fold", "historical")
BACKBONE_KINDS = ("release", "research")
MODEL_FORM = ("mf_modes.fit_mode_mf: log_rho + the resolved FixedMeanHead (gbar_z + gbar_tau + rank-1), "
              "theta-independent; P_MF = P_LF exp(g)")
TABLE_KEYS = ("log_rho", "gbar_z_tab", "gtau_tab", "a_k", "u_z", "u_tau", "z_tab", "tau_tab", "tau_by_z")
_TABLE_DATASETS = TABLE_KEYS + ("k_com_hmpc",)
N_CLASSES = 4
_MAX_TABLE_BYTES = 1 << 26                     # 64 MiB: the tables are a few MB at most

# the HR cache fields the MF reads (data.load_cache keys) and the catalogue-derived ones it must never read
HR_VIEW_KEYS = ("params", "z_grid", "alpha_idx", "sim_name", "kfkms", "P_filt", "k_com_hmpc")
HR_CATALOGUE_DERIVED = ("snap_dNdX", "snap_f_nhi", "snap_n_absorbers", "snap_total_path_dX")
HR_INPUTS = {
    "used": ["P_tier_c_filtered", "tier_c_counts", "params", "z_grid", "alpha_idx", "sim_name", "kfkms"],
    "used_as": "the per-class P1D rows (P_filt, the count-weighted collapse of P_tier_c_filtered over the fine "
               "N_HI bins) and the row keys that match them to LF rows",
    "excluded_catalogue_derived": ["snap_dNdX_LLS", "snap_dNdX_subDLA", "snap_dNdX_DLA", "snap_f_nhi",
                                   "snap_n_absorbers", "snap_total_path_dX"],
    "catalogue_derived_status": "PRE-FIX: not read by the MF; no verified or corrected provenance is claimed for them",
}

# What each cache's digest establishes (PI ruling 2026-11-06b A1: no incompatible products under one verified claim).
# Constants, written by write_mf_record and closed-checked by check_record: a record cannot claim more for either side.
CACHE_PROVENANCE = {
    "lf": "lineage-verified through the bound backbone: the backbone bundle's training cache (caches.lf.sha256 equals "
          "backbone.cache_sha256), which the bundle's lineage block and every member meta name; the builder reads it "
          "with data.load_cache, which checks its lineage record",
    "hr": "digest-anchored only, never lineage-verified: the builder pins its sha256 and reads it with "
          "data.load_cache_historical, which checks no lineage (a pre-lineage format 3.3 cache has no lineage record); "
          "only its per-class P1D rows and row keys are read (hr_inputs)",
}

# The mean-flux normalisation of the per-class P1D targets, as the code computes it today (read at be675c6).
# Under audit (PI ruling 2026-11-06b D2): recorded, PROPOSED, and not changed here.
MEAN_FLUX_CONVENTION = {
    "status": "PROPOSED",
    "name": "global",
    "statement": "each class's flux contrast is normalised by the mean flux of the whole sample, shared by the four "
                 "classes: delta_F,c = exp(-s tau) / F_target - 1, with (s, F_target) of Tier P (all sightlines); "
                 "not the class's own mean flux <F>_c",
    "function": "hcd_analysis.priya_p1d._per_class_p1d_at_scale",
    "source": "hcd_analysis/priya_p1d.py:146 (dflux = np.exp(-tau_eff) / target_F - 1.0), called per class by "
              "compute_tier_c_p1d with Tier P's (scale, target_F)",
    "read_at_commit": "be675c66b7d6d84126eb7a8501d15e874197d96a",
    "audit": "under audit (PI ruling 2026-11-06b D2); no normalisation is changed until the convention is settled",
}

RECORD_KEYS = {"format", "mf_version", "kind", "model_form", "tables", "backbone", "caches", "composition", "fold",
               "train_sims", "n_pairs_train", "hr_inputs", "mean_flux_convention", "fit_code"}
_TABLES_KEYS = {"format", "file", "sha256", "bytes", "n_modes", "n_z", "n_rung"}
_BACKBONE_KEYS = {"manifest_sha256", "kind", "cache_sha256", "held_out_sims"}
_CACHE_KEYS = {"file", "sha256", "provenance"}
_COMPOSITION_KEYS = {"id", "excluded_sims", "ruling_ids"}
_FOLD_KEYS = {"fold_id", "holdout_sims", "backbone_excludes_holdout"}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_PLAIN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*\Z")


class MFRecordError(ValueError):
    """An MF record (or input) that may not be written or loaded as asked."""


# --------------------------------------------------------------------------------------------- HR inputs (06b A1)
class _HRView(Mapping):
    """The HR cache restricted to the per-class P1D rows and their row keys. Reading a catalogue-derived column (or
    anything else outside ``HR_VIEW_KEYS``) raises ``MFRecordError``."""

    def __init__(self, hr):
        self._d = {k: hr[k] for k in HR_VIEW_KEYS if k in hr}

    def __getitem__(self, k):
        if k in self._d:
            return self._d[k]
        if any(str(k).startswith(c) for c in HR_CATALOGUE_DERIVED):
            raise MFRecordError(f"HR field {k!r} is catalogue-derived and pre-fix (PI ruling 2026-11-06b A1): the MF "
                                "uses only the HR per-class P1D rows")
        raise MFRecordError(f"HR field {k!r} is not an MF input (the MF reads only {HR_VIEW_KEYS})")

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)


def hr_view(hr):
    """The HR cache dict (``data.load_cache*``) seen through the MF's allow-list ``HR_VIEW_KEYS``."""
    return _HRView(hr)


# --------------------------------------------------------------------------------------------- training rows
def _names(x, what):
    if isinstance(x, (str, bytes)) or not all(isinstance(s, str) and s for s in x):
        raise MFRecordError(f"{what} must be a list of simulation names")
    return sorted(set(x))


def select_train_rows(targets, *, exclude_sims=(), holdout_sims=()):
    """The training rows of an MF fit: every matched pair whose simulation is neither excluded by the composition
    (``exclude_sims``) nor held out (``holdout_sims``, a fold's group). Returns (rows, sorted training simulations).
    Refuses an unknown name, a name in both lists and an empty training set."""
    sims = np.asarray(targets["sim"]).astype(str)
    known = set(sims)
    ex, ho = _names(exclude_sims, "exclude_sims"), _names(holdout_sims, "holdout_sims")
    unknown = sorted((set(ex) | set(ho)) - known)
    if unknown:
        raise MFRecordError(f"unknown simulations {unknown}: not among the matched pairs")
    both = sorted(set(ex) & set(ho))
    if both:
        raise MFRecordError(f"simulations {both} are both excluded by the composition and held out")
    drop = set(ex) | set(ho)
    rows = np.where(~np.isin(sims, sorted(drop)))[0]
    if rows.size == 0:
        raise MFRecordError("no training pairs left after the exclusions and the held-out group")
    return rows, sorted(known - drop)


def heldout_rows(targets, holdout_sims):
    """The rows of the held-out group (for genuine held-out predictions of a fold fit)."""
    sims = np.asarray(targets["sim"]).astype(str)
    ho = _names(holdout_sims, "holdout_sims")
    if not set(ho) <= set(sims):
        raise MFRecordError(f"unknown simulations {sorted(set(ho) - set(sims))}")
    return np.where(np.isin(sims, ho))[0]


# --------------------------------------------------------------------------------------------- record checks
def _sorted_names(v, where):
    if not isinstance(v, list) or not all(isinstance(s, str) and s for s in v) or v != sorted(set(v)):
        raise MFRecordError(f"{where} must be a sorted list of distinct simulation names")
    return v


def _keys(d, want, where):
    if not isinstance(d, dict) or set(d) != set(want):
        got = set(d) if isinstance(d, dict) else set()
        raise MFRecordError(f"{where}: keys must be exactly {sorted(want)} (missing {sorted(set(want) - got)}, "
                            f"unknown {sorted(got - set(want))})")


def _hex(v, n, where):
    if not isinstance(v, str) or not (_HEX64 if n == 64 else _HEX40).match(v):
        raise MFRecordError(f"{where} must be {n} lowercase hex digits")


def check_record(rec):
    """Validate a record dict (closed schema, kind rules, the HR-input and mean-flux fields). Raises MFRecordError."""
    _keys(rec, RECORD_KEYS, "MF record")
    if rec["format"] != RECORD_FORMAT:
        raise MFRecordError(f"MF record format {rec['format']!r}, expected {RECORD_FORMAT}")
    if not isinstance(rec["mf_version"], str) or not rec["mf_version"].strip():
        raise MFRecordError("mf_version must be a non-empty string")
    if rec["kind"] not in KINDS:
        raise MFRecordError(f"kind {rec['kind']!r} is not one of {KINDS}")
    if rec["model_form"] != MODEL_FORM:
        raise MFRecordError("model_form is not the form this code evaluates")
    t = rec["tables"]
    _keys(t, _TABLES_KEYS, "tables")
    if t["format"] != TABLES_FORMAT or t["file"] != TABLES_FILE:
        raise MFRecordError(f"tables must be {TABLES_FILE} in format {TABLES_FORMAT}")
    _hex(t["sha256"], 64, "tables.sha256")
    for k in ("bytes", "n_modes", "n_z", "n_rung"):
        if not isinstance(t[k], int) or isinstance(t[k], bool) or t[k] < (2 if k in ("n_z", "n_rung") else 1):
            raise MFRecordError(f"tables.{k} must be a positive integer (n_z, n_rung at least 2)")
    b = rec["backbone"]
    _keys(b, _BACKBONE_KEYS, "backbone")
    _hex(b["manifest_sha256"], 64, "backbone.manifest_sha256")
    _hex(b["cache_sha256"], 64, "backbone.cache_sha256")
    if b["kind"] not in BACKBONE_KINDS:
        raise MFRecordError(f"backbone.kind {b['kind']!r} is not one of {BACKBONE_KINDS}")
    _sorted_names(b["held_out_sims"], "backbone.held_out_sims")
    if b["kind"] == "release" and b["held_out_sims"]:
        raise MFRecordError("a release backbone holds out no simulation")
    c = rec["caches"]
    _keys(c, {"lf", "hr"}, "caches")
    for side in ("lf", "hr"):
        _keys(c[side], _CACHE_KEYS, f"caches.{side}")
        if not isinstance(c[side]["file"], str) or not _PLAIN.match(c[side]["file"]):
            raise MFRecordError(f"caches.{side}.file must be a plain file name, never a path")
        _hex(c[side]["sha256"], 64, f"caches.{side}.sha256")
        if c[side]["provenance"] != CACHE_PROVENANCE[side]:
            raise MFRecordError(f"caches.{side}.provenance is not the code's statement for the {side.upper()} cache "
                                "(the HR cache is digest-anchored only, never lineage-verified: PI ruling 2026-11-06b "
                                "A1)")
    if c["lf"]["sha256"] != b["cache_sha256"]:                     # every kind: the targets used the backbone's cache
        raise MFRecordError(f"a {rec['kind']} MF's LF cache must be its backbone's training cache")
    comp = rec["composition"]
    _keys(comp, _COMPOSITION_KEYS, "composition")
    if not isinstance(comp["id"], str) or not comp["id"].strip():
        raise MFRecordError("composition.id must be a non-empty string")
    _sorted_names(comp["excluded_sims"], "composition.excluded_sims")
    if not isinstance(comp["ruling_ids"], list) or not comp["ruling_ids"] or \
            not all(isinstance(r, str) and r.strip() for r in comp["ruling_ids"]):
        raise MFRecordError("composition.ruling_ids must name at least one ruling (the authority for the composition)")
    train = _sorted_names(rec["train_sims"], "train_sims")
    if not train:
        raise MFRecordError("train_sims is empty")
    if set(train) & set(comp["excluded_sims"]):
        both = sorted(set(train) & set(comp["excluded_sims"]))
        raise MFRecordError(f"train_sims include excluded simulations {both}")
    if not isinstance(rec["n_pairs_train"], int) or isinstance(rec["n_pairs_train"], bool) or rec["n_pairs_train"] < 1:
        raise MFRecordError("n_pairs_train must be a positive integer")
    h = rec["hr_inputs"]
    if h != HR_INPUTS:
        bad = sorted(set((h or {}).get("used", [])) & set(HR_INPUTS["excluded_catalogue_derived"])) \
            if isinstance(h, dict) and isinstance(h.get("used"), list) else []
        raise MFRecordError("hr_inputs differ from the MF's HR inputs: the HR catalogue-derived columns are pre-fix "
                            "and are neither used nor given another status (PI ruling 2026-11-06b A1)"
                            + (f"; it names {bad}" if bad else ""))
    if rec["mean_flux_convention"] != MEAN_FLUX_CONVENTION:
        raise MFRecordError("mean-flux convention differs from the code's PROPOSED record (it is under audit, PI "
                            "ruling 2026-11-06b D2; it changes only with the code)")
    fc = rec["fit_code"]
    _keys(fc, {"commit", "dirty"}, "fit_code")
    _hex(fc["commit"], 40, "fit_code.commit")
    if not isinstance(fc["dirty"], bool):
        raise MFRecordError("fit_code.dirty must be a boolean")
    fold = rec["fold"]
    if rec["kind"] == "fold":
        if fold is None:
            raise MFRecordError("a fold record needs its fold (fold_id, holdout_sims)")
        _keys(fold, _FOLD_KEYS, "fold")
        if not isinstance(fold["fold_id"], str) or not fold["fold_id"].strip():
            raise MFRecordError("fold.fold_id must be a non-empty string")
        ho = _sorted_names(fold["holdout_sims"], "fold.holdout_sims")
        if not ho or set(ho) & set(train) or set(ho) & set(comp["excluded_sims"]):
            raise MFRecordError("fold.holdout_sims must be non-empty and outside the training and excluded sets")
        if fold["backbone_excludes_holdout"] is not (set(ho) <= set(b["held_out_sims"])):
            raise MFRecordError("fold.backbone_excludes_holdout disagrees with the backbone's held-out simulations")
    else:
        if fold is not None:
            raise MFRecordError(f"a {rec['kind']} record carries no fold")
        if rec["kind"] == "release":
            if b["kind"] != "release":
                raise MFRecordError("a release MF needs a release backbone, not a research one")
            if fc["dirty"]:
                raise MFRecordError("a release MF is fitted from a clean tree (fit_code.dirty is true)")
    return rec


# --------------------------------------------------------------------------------------------- tables I/O
def _check_tables(tables, k_com):
    k = np.asarray(k_com, dtype=np.float64)
    if k.ndim != 1 or not np.allclose(k, 2 * np.pi * np.arange(1, k.size + 1) / L_BOX_HMPC, rtol=1e-12, atol=0):
        raise MFRecordError("k_com_hmpc must be the box modes 2 pi n / L, n = 1..K")
    a = {n: np.asarray(tables[n], dtype=np.float64) for n in TABLE_KEYS}
    K, nz, nr = k.size, a["z_tab"].size, a["tau_tab"].size
    want = {"log_rho": (K,), "gbar_z_tab": (nz, N_CLASSES, K), "gtau_tab": (nr, N_CLASSES, K), "a_k": (N_CLASSES, K),
            "u_z": (nz,), "u_tau": (nr,), "z_tab": (nz,), "tau_tab": (nr,), "tau_by_z": (nz, nr)}
    bad = [n for n in TABLE_KEYS if a[n].shape != want[n]]
    if bad:
        raise MFRecordError(f"MF tables with inconsistent shapes: {[(n, a[n].shape, want[n]) for n in bad]}")
    if nz < 2 or nr < 2:
        raise MFRecordError("MF tables need at least two redshifts and two mean-flux rungs")
    if not all(np.all(np.isfinite(v)) for v in a.values()):
        raise MFRecordError("MF tables must be finite")
    if np.any(np.diff(a["z_tab"]) <= 0) or not np.array_equal(a["tau_tab"], np.arange(nr, dtype=np.float64)) \
            or np.any(np.diff(a["tau_by_z"], axis=1) < 0):
        raise MFRecordError("MF tables: z_tab must ascend, tau_tab be the rung index 0..N-1, tau_by_z ascend in rung")
    return a, k


def _write_tables(path, arrays):
    import h5py

    def writer(fh):
        with h5py.File(fh, "w", libver=("earliest", "v110")) as f:
            f.attrs["format"] = TABLES_FORMAT
            g = f.create_group("mf")
            for n in sorted(arrays):
                g.create_dataset(n, data=np.ascontiguousarray(arrays[n], dtype="<f8"))
    tmp, held, sha = S._write_tmp(path, ".h5", writer)
    S._publish(tmp, path, held)
    return sha


def _read_tables(raw, where):
    """Strictly parse anchored table bytes: the root ``format`` attribute only, the group ``/mf`` only, flat, holding
    exactly ``_TABLE_DATASETS`` as contiguous unfiltered attribute-free float64 (safe_h5's walk)."""
    import h5py
    from . import safe_h5 as H
    try:
        with h5py.File(io.BytesIO(raw), "r") as f:
            if f.userblock_size:
                raise MFRecordError(f"{where}: a user block is not allowed")
            got = None
            if len(f.attrs) == 1 and list(f.attrs.keys()) == ["format"]:
                aid = f.attrs.get_id("format")
                if aid.shape == () and aid.get_type().get_class() == h5py.h5t.STRING:
                    got = f.attrs["format"]
            if not isinstance(got, str) or got != TABLES_FORMAT:
                raise MFRecordError(f"{where}: root attributes must be exactly a scalar string format = "
                                    f"{TABLES_FORMAT}")
            if len(f) != 1 or list(f.keys()) != ["mf"]:
                raise MFRecordError(f"{where}: the top level must hold exactly /mf")
            link = f.get("mf", getlink=True)
            if not isinstance(link, h5py.HardLink) or not isinstance(f["mf"], h5py.Group) or len(f["mf"].attrs):
                raise MFRecordError(f"{where}: /mf must be a plain hard-linked group")
            info = h5py.h5o.get_info(f["mf"].id)
            found, total, groups = H._walk(f["mf"], "/mf", where, len(raw), {(info.fileno, info.addr)})
            if groups:
                raise MFRecordError(f"{where}: /mf must be flat (it holds the group {groups[0]})")
            names = {p[len("/mf/"):] for p in found}
            if names != set(_TABLE_DATASETS):
                raise MFRecordError(f"{where}: datasets {sorted(names)} are not exactly {sorted(_TABLE_DATASETS)}")
            if total > _MAX_TABLE_BYTES:
                raise MFRecordError(f"{where}: the tables exceed the size bound")
            out = {p[len("/mf/"):]: d[...] for p, d in found.items()}
    except MFRecordError:
        raise
    except S.SafeFormatError as e:
        raise MFRecordError(str(e)) from None
    except Exception as e:                                          # anything libhdf5/h5py raises on crafted bytes
        raise MFRecordError(f"{where}: unreadable HDF5 ({type(e).__name__}: {e})"[:300]) from None
    return out


# --------------------------------------------------------------------------------------------- write
def write_mf_record(out_dir, tables, *, k_com_hmpc, kind, mf_version, backbone, caches, composition, fold, fit_code,
                    train_sims, n_pairs_train):
    """Write a record directory ONCE: ``out_dir`` must not exist (an existing directory, file or symlink is refused);
    it is created, the tables are written, then the record (last). ``caches`` = {lf, hr: {file, sha256}}; the writer
    adds each side's ``provenance`` (``CACHE_PROVENANCE``). ``fold`` = None, or {fold_id, holdout_sims} (the
    writer adds ``backbone_excludes_holdout`` from the backbone's held-out simulations). Every field is validated
    (``check_record``) before anything is written. Returns (the record's path, its sha256: the anchor to publish)."""
    out_dir = os.fspath(out_dir)
    if not isinstance(caches, dict) or set(caches) != {"lf", "hr"} or \
            not all(isinstance(caches[s], dict) and set(caches[s]) == {"file", "sha256"} for s in ("lf", "hr")):
        raise MFRecordError("caches must be {lf, hr: {file, sha256}} (the provenance is the code's, CACHE_PROVENANCE)")
    if os.path.lexists(out_dir):
        raise FileExistsError(f"{out_dir} exists: an MF record is written once into a new directory, never overwritten")
    arrays, k = _check_tables(tables, k_com_hmpc)
    nz, nr = arrays["z_tab"].size, arrays["tau_tab"].size
    fold_rec = None
    if fold is not None:
        if not isinstance(fold, dict) or set(fold) != {"fold_id", "holdout_sims"}:
            raise MFRecordError("fold must be {fold_id, holdout_sims}")
        ho = sorted(set(fold["holdout_sims"]))
        fold_rec = {"fold_id": fold["fold_id"], "holdout_sims": ho,
                    "backbone_excludes_holdout": bool(set(ho) <= set(backbone.get("held_out_sims") or []))}
    rec = {"format": RECORD_FORMAT, "mf_version": mf_version, "kind": kind, "model_form": MODEL_FORM,
           "tables": {"format": TABLES_FORMAT, "file": TABLES_FILE, "sha256": "0" * 64, "bytes": 1, "n_modes": k.size,
                      "n_z": nz, "n_rung": nr},
           "backbone": dict(backbone, held_out_sims=sorted(set(backbone.get("held_out_sims") or []))),
           "caches": {s: dict(copy_json(caches[s]), provenance=CACHE_PROVENANCE[s]) for s in ("lf", "hr")},
           "composition": dict(composition, excluded_sims=sorted(set(composition.get("excluded_sims") or []))),
           "fold": fold_rec, "train_sims": sorted(set(train_sims)), "n_pairs_train": int(n_pairs_train),
           "hr_inputs": copy_json(HR_INPUTS), "mean_flux_convention": copy_json(MEAN_FLUX_CONVENTION),
           "fit_code": copy_json(fit_code)}
    check_record(rec)                                               # everything refused before anything is written
    os.makedirs(out_dir)                                            # no exist_ok: a race is refused too
    tpath = os.path.join(out_dir, TABLES_FILE)
    sha = _write_tables(tpath, dict(arrays, k_com_hmpc=k))
    rec["tables"]["sha256"], rec["tables"]["bytes"] = sha, os.path.getsize(tpath)
    check_record(rec)
    raw = json.dumps(rec, indent=1, sort_keys=True, allow_nan=False).encode("utf-8")
    rpath = os.path.join(out_dir, RECORD_FILE)
    digest = S._write_exclusive(rpath, [raw])
    return rpath, digest


def copy_json(x):
    return json.loads(json.dumps(x, allow_nan=False))


# --------------------------------------------------------------------------------------------- load
class MFRecord(NamedTuple):
    """A loaded MF record, anchored on its published sha256 (``anchored``: the record bytes matched the published
    digest and the tables matched the record's; it says nothing about the HR cache's lineage, see ``caches`` in the
    record). ``g(x, tau0)`` evaluates the correction (4, K) at one LF encoder input.

    It is sealed: ``record_bytes`` and ``tables_bytes`` are the anchored bytes it was parsed from. The predictor applies
    a record only through ``reverify``, which re-derives the record and the tables from those bytes, so an object
    built or modified in memory is refused rather than applied under another record's provenance."""
    record: dict
    record_sha256: str
    tables: dict
    k_com_hmpc: np.ndarray
    kind: str
    mf_dir: str
    anchored: bool
    record_bytes: bytes = None
    tables_bytes: bytes = None

    def module(self):
        from .mf_modes import ModeMF
        return ModeMF.from_tables(self.tables)

    def g(self, x, tau0):
        from .mf_modes import apply_mode_mf
        return apply_mode_mf(self.tables, x, tau0)

    def provenance(self):
        r = self.record
        return {"record_sha256": self.record_sha256, "mf_version": r["mf_version"], "kind": r["kind"],
                "backbone_manifest_sha256": r["backbone"]["manifest_sha256"],
                "composition": copy_json(r["composition"]),
                "fold": copy_json(r["fold"]), "train_sims": list(r["train_sims"]), "caches": copy_json(r["caches"]),
                "tables_sha256": r["tables"]["sha256"], "fit_code": copy_json(r["fit_code"]),
                "mean_flux_convention": copy_json(r["mean_flux_convention"]),
                "hr_inputs": copy_json(r["hr_inputs"])}


def load_mf_record(mf_dir, *, expected_record_sha256, backbone_manifest_sha256, kind="release", composition=None,
                   lf_cache_sha256=None, hr_cache_sha256=None):
    """Load the record in ``mf_dir`` anchored on its published sha256 (``None`` is refused) and bound to the backbone
    ``backbone_manifest_sha256`` (required). ``kind`` is the kind the caller expects (a ``historical`` or ``fold``
    record is refused unless asked for). ``composition`` = {id, excluded_sims[, ruling_ids]} and the cache digests,
    when given, must equal the record's. Returns an ``MFRecord``."""
    # Every digest is taken as an exact str (delta review R1-2): a str subclass can override == and != and pass the
    # comparisons below for any record, so it is refused, not compared.
    if expected_record_sha256 is None or type(expected_record_sha256) is not str \
            or not _HEX64.match(expected_record_sha256):
        raise MFRecordError("load_mf_record needs the record's published sha256 (64 lowercase hex digits, an exact "
                            "str)")
    if backbone_manifest_sha256 is None:
        raise MFRecordError("load_mf_record needs the backbone manifest sha256 the MF must be bound to")
    for name, v in (("backbone_manifest_sha256", backbone_manifest_sha256), ("lf_cache_sha256", lf_cache_sha256),
                    ("hr_cache_sha256", hr_cache_sha256)):
        if v is not None and type(v) is not str:
            raise MFRecordError(f"load_mf_record: {name} must be an exact str (a str subclass is refused)")
    if kind not in KINDS:
        raise MFRecordError(f"kind {kind!r} is not one of {KINDS}")
    d = os.path.abspath(os.fspath(mf_dir))
    if not os.path.isdir(d) or os.path.islink(d):
        raise MFRecordError(f"{mf_dir}: not an MF record directory (a legacy .npz product is not a release record)")
    rp = os.path.join(d, RECORD_FILE)
    try:
        raw = S.read_regular(rp, S.MAX_JSON_BYTES)
    except (S.SafeFormatError, OSError) as e:
        raise MFRecordError(f"{mf_dir}: no readable {RECORD_FILE} ({e})") from None
    if hashlib.sha256(raw).hexdigest() != expected_record_sha256:
        raise MFRecordError(f"{rp}: sha256 does not match the published record sha256")
    rec = _parse_record(raw, rp)
    extra = sorted(set(os.listdir(d)) - {RECORD_FILE, TABLES_FILE})
    if extra:
        raise MFRecordError(f"{mf_dir}: unexpected files {extra}")
    if rec["kind"] != kind:
        raise MFRecordError(f"{mf_dir}: a {rec['kind']} MF record, expected {kind}"
                            + (" (historical only: never applied as the release MF)" if rec["kind"] == "historical"
                               else "") + (" (a held-out fold fit is not the release MF)" if rec["kind"] == "fold"
                                           else ""))
    if rec["backbone"]["manifest_sha256"] != backbone_manifest_sha256:
        raise MFRecordError(f"{mf_dir}: fitted against the backbone {rec['backbone']['manifest_sha256'][:12]}, not "
                            f"{str(backbone_manifest_sha256)[:12]} (an MF is bound to exactly one backbone)")
    if lf_cache_sha256 is not None and rec["caches"]["lf"]["sha256"] != lf_cache_sha256:
        raise MFRecordError(f"{mf_dir}: built from LF cache {rec['caches']['lf']['sha256'][:12]}, expected "
                            f"{lf_cache_sha256[:12]}")
    if hr_cache_sha256 is not None and rec["caches"]["hr"]["sha256"] != hr_cache_sha256:
        raise MFRecordError(f"{mf_dir}: built from HR cache {rec['caches']['hr']['sha256'][:12]}, expected "
                            f"{hr_cache_sha256[:12]}")
    if composition is not None:
        got = rec["composition"]
        want_ex = sorted(set(composition.get("excluded_sims") or []))
        if got["id"] != composition.get("id") or got["excluded_sims"] != want_ex or \
                ("ruling_ids" in composition and got["ruling_ids"] != list(composition["ruling_ids"])):
            raise MFRecordError(f"{mf_dir}: composition {got['id']!r} excluding {got['excluded_sims']} is not the "
                                f"expected {composition.get('id')!r} excluding {want_ex}")
    tp = os.path.join(d, TABLES_FILE)
    try:
        traw = S.read_regular(tp, _MAX_TABLE_BYTES + (1 << 20))
    except (S.SafeFormatError, OSError) as e:
        raise MFRecordError(f"{mf_dir}: tables unreadable ({e})") from None
    tables, k = _parse_tables(rec, traw, tp)
    return MFRecord(rec, expected_record_sha256, tables, k, rec["kind"], d, True, raw, traw)


def _parse_record(raw, where):
    try:
        rec = S._json_loads(raw, where)
    except S.SafeFormatError as e:
        raise MFRecordError(str(e)) from None
    return check_record(rec)


def _parse_tables(rec, traw, where):
    """The tables from their bytes, checked against the record: size and sha256 before libhdf5 parses them, then the
    strict walk, the table checks and the shapes. Returns read-only (tables, k_com_hmpc)."""
    if len(traw) != rec["tables"]["bytes"] or hashlib.sha256(traw).hexdigest() != rec["tables"]["sha256"]:
        raise MFRecordError(f"{where}: the tables' sha256 or size does not match the record")
    arrays = _read_tables(traw, where)
    tables, k = _check_tables(arrays, arrays["k_com_hmpc"])
    t = rec["tables"]
    if (k.size, tables["z_tab"].size, tables["tau_tab"].size) != (t["n_modes"], t["n_z"], t["n_rung"]):
        raise MFRecordError(f"{where}: table shapes differ from the record")
    for v in tables.values():
        v.setflags(write=False)
    k.setflags(write=False)
    return tables, k


def reverify(mf):
    """The record to APPLY, re-derived from a loaded record's own sealed bytes (``MFRecord.record_bytes`` and
    ``tables_bytes``): the record bytes must hash to ``record_sha256``, and the record, its kind, the tables and the
    modes are parsed again from the bytes, exactly as ``load_mf_record`` does. Refused: anything that is not an
    ``MFRecord`` from ``load_mf_record`` (no sealed bytes, not anchored), bytes that do not match the digest, and an
    object whose record, kind, tables or modes were changed in memory (they differ from what its bytes give), and a
    digest or sealed bytes that are not exactly ``str`` and ``bytes`` (a subclass can override ``!=`` or
    ``__bytes__`` and make the bytes hashed differ from the bytes parsed: delta review R1-2). Returns
    a fresh ``MFRecord`` built from the bytes only, so the provenance it reports describes the tables applied. What it
    cannot catch: a caller who replaces the bytes and the digest together is presenting another anchored record; the
    trust root is the published digest the caller passes to ``load_mf_record``."""
    if not isinstance(mf, MFRecord) or mf.anchored is not True or type(mf.record_bytes) is not bytes \
            or type(mf.tables_bytes) is not bytes or type(mf.record_sha256) is not str:
        raise MFRecordError("the predictor takes an MF record loaded by mf_release.load_mf_record (sealed with its "
                            "anchored bytes)")
    where = f"the MF record {mf.record_sha256[:12]}"
    if hashlib.sha256(mf.record_bytes).hexdigest() != mf.record_sha256:
        raise MFRecordError(f"{where}: its sealed bytes do not match its record sha256")
    rec = _parse_record(mf.record_bytes, where)
    tables, k = _parse_tables(rec, mf.tables_bytes, where)
    same = (isinstance(mf.record, dict) and mf.record == rec and mf.kind == rec["kind"]
            and isinstance(mf.tables, Mapping) and set(mf.tables) == set(tables)
            and all(np.array_equal(np.asarray(mf.tables[n]), tables[n]) for n in tables)
            and np.array_equal(np.asarray(mf.k_com_hmpc), k))
    if not same:
        raise MFRecordError(f"{where}: the record, kind, tables or modes were changed in memory (they differ from its "
                            "anchored bytes); load the record with mf_release.load_mf_record")
    return MFRecord(rec, mf.record_sha256, tables, k, rec["kind"], mf.mf_dir, True, mf.record_bytes, mf.tables_bytes)
