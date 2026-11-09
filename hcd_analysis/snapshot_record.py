"""The verifier of a completed snapshot directory (``<output_root>/<sim>/snap_NNN``) written by the pipeline: its
meta.json, the products meta.json lists by sha256, their lineage records, and the copies of the inputs' finder ids and
release eligibility that each record holds. Used by pipeline resume and by every later consumer (the cache builder,
the carry tool, the catalogue acceptance). numpy only (no PyYAML, no fake_spectra)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

from . import lineage as L
from .catalog import RELEASE_FINDER_IDS, AbsorberCatalog, CatalogIntegrityError, load_validated_catalog

DONE_FULL, DONE_CATALOGUE = "done", "done_catalogue"     # the completion markers of the two modes
MODE_MARKERS = {"full": DONE_FULL, "catalogue": DONE_CATALOGUE}
# the files a snapshot directory may list in meta.json["products"], with their lineage kinds
PRODUCT_KINDS = {"catalog.npz": "catalog", "cddf.npz": "cddf", "p1d.npz": "p1d", "p1d_ratios.npz": "p1d_ratios",
                 "p1d_excl.npz": "p1d_excl", "p1d_perturbed.npz": "p1d_perturbed", "p1d_per_class.h5": "p1d_per_class"}


class SnapshotIntegrityError(CatalogIntegrityError):
    """A completed snapshot's products do not match the sha256 its meta.json records, or its records disagree."""


@dataclasses.dataclass
class SnapshotProducts:
    """A verified snapshot directory."""
    meta: dict                     # meta.json
    record: dict                   # the snapshot record (meta.json["lineage"])
    products: Dict[str, dict]      # meta.json["products"]: name -> {sha256, size}
    meta_identity: dict            # sha256 and size of the meta.json bytes that were parsed
    catalog: AbsorberCatalog       # through load_validated_catalog
    cddf: dict                     # the cddf.npz arrays (without the record)
    cddf_record: dict


def _flags(entry):
    return {"finder_ids": entry.get("finder_ids"), "release_eligible": entry.get("release_eligible")}


def verify_snapshot_dir(snap_dir, *, mode: Optional[str] = None, settings: Optional[dict] = None, raw_path=None,
                        n_skewers_built: Optional[int] = None, rehash: bool = False,
                        require_release_eligible: bool = False, sim: Optional[str] = None,
                        snap: Optional[int] = None) -> SnapshotProducts:
    """Verify a completed snapshot directory; refused with SnapshotIntegrityError (or, for the catalogue, the release
    loader's CatalogIntegrityError / CatalogProvenanceError). In order:
      1. meta.json is readable (a missing or unreadable one is refused, never skipped), of ``mode`` when given
         ("catalogue" or "full"), with that mode's completion marker;
      2. catalog.npz passes ``load_validated_catalog`` with the skewer length, ``settings``, ``raw_path`` and the
         skewer count (``n_skewers_built``; by default the file's count, or the recorded limit of ``settings``) and
         ``require_release_eligible``;
      3. meta.json lists the sha256 of every product the mode writes, and every listed product matches it;
      4. the snapshot record lists exactly those products; cddf.npz's record names the listed catalog.npz and the
         catalogue's raw file;
      5. the finder ids and release eligibility copied into the snapshot record and the cddf record equal those of the
         records they were copied from;
      6. the three name one (simulation, snapshot), equal to ``sim`` and ``snap`` when given;
      7. meta.json's own fields agree with the catalogue and its record: ``n_absorbers`` per class equals the
         catalogue's class counts (and their total the record's count), ``z`` and ``dv_kms`` equal the catalogue's and
         its record's, ``nbins`` the record's and the raw file's, ``n_skewers`` the raw file's (as the record states);
      8. with ``require_release_eligible``, the snapshot and cddf records are eligible; with ``rehash`` and
         ``raw_path``, the raw file is re-hashed against the catalogue's record."""
    snap_dir = Path(snap_dir)

    def refuse(why):
        raise SnapshotIntegrityError(f"{snap_dir}: completed snapshot refused ({why}); rebuild it")
    try:
        meta_bytes = (snap_dir / "meta.json").read_bytes()
        meta = json.loads(meta_bytes.decode("utf-8"))
        nbins, n_file = int(meta["nbins"]), int(meta["n_skewers"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        refuse(f"meta.json missing or unreadable: {type(e).__name__}: {e}")
    if not isinstance(meta, dict):
        refuse("meta.json is not an object")
    got_mode = meta.get("mode")
    if mode is not None and got_mode != mode:
        refuse(f"meta.json mode {got_mode!r}, this run {mode!r}")
    if got_mode not in MODE_MARKERS or not (snap_dir / MODE_MARKERS[got_mode]).exists():
        refuse(f"meta.json mode {got_mode!r} without its completion marker")
    if n_skewers_built is None and settings is not None:
        limit = settings.get("n_skewers")
        n_skewers_built = n_file if limit is None else min(int(limit), n_file)
    catalog = load_validated_catalog(snap_dir / "catalog.npz", nbins, settings=settings, raw_path=raw_path,
                                     n_skewers_built=n_skewers_built, require_release_eligible=require_release_eligible)
    prods = meta.get("products")
    if not isinstance(prods, dict) or not prods:
        refuse("meta.json lists no product hashes (written before snapshots recorded them)")
    need = {"catalog.npz", "cddf.npz"} | ({"p1d.npz"} if got_mode == "full" else set())
    if need - set(prods) or set(prods) - set(PRODUCT_KINDS):
        refuse(f"meta.json products {sorted(prods)}, expected {sorted(need)} and only known names")
    for name, ident in sorted(prods.items()):
        p = snap_dir / name
        if not (isinstance(ident, dict) and isinstance(ident.get("sha256"), str) and isinstance(ident.get("size"), int)):
            refuse(f"malformed product entry for {name}")
        if not p.is_file() or not L.same_file(p, {"sha256": ident["sha256"], "size": ident["size"]}):
            refuse(f"{name} does not match the sha256 meta.json records")
    try:
        rec = L.check_record(meta.get("lineage"), RELEASE_FINDER_IDS)
        if rec["kind"] != "snapshot" or {i["role"]: i["sha256"] for i in rec["inputs"]} != \
                {k: v["sha256"] for k, v in prods.items()}:
            refuse("the snapshot record does not list the products meta.json lists")
        cddf, crec = L.read_npz(snap_dir / "cddf.npz")
        L.check_record(crec, RELEASE_FINDER_IDS)
        if crec["kind"] != "cddf" or L.find_input(crec, "catalog")["sha256"] != prods["catalog.npz"]["sha256"]:
            refuse("cddf.npz was not made from the listed catalog.npz")
        raw = L.find_input(catalog.lineage, "raw_spectra")
        if L.find_input(crec, "raw_spectra_header")["sha256"] != raw["sha256"]:
            refuse("cddf.npz names another raw spectra file than the catalogue")
        copies = {"catalog.npz": catalog.lineage, "cddf.npz": crec}
        for inp in rec["inputs"]:
            if inp["role"] in copies and _flags(inp) != _flags(dict(copies[inp["role"]])):
                refuse(f"the snapshot record's copy of the {inp['role']} finder ids or eligibility differs from "
                       "that product's own record")
        if _flags(L.find_input(crec, "catalog")) != _flags(catalog.lineage):
            refuse("cddf.npz's copy of the catalogue's finder ids or eligibility differs from the catalogue's record")
    except L.LineageError as e:
        refuse(str(e))
    try:
        names = {(str(meta["sim_name"]), int(meta["snap"])), (catalog.sim_name, int(catalog.snap)),
                 (str(crec["product"]["sim_name"]), int(crec["product"]["snap"]))}
    except (KeyError, TypeError, ValueError) as e:
        refuse(f"a product does not name its simulation and snapshot ({type(e).__name__}: {e})")
    if len(names) != 1 or (sim is not None and names != {(str(sim), int(snap))}):
        refuse(f"the products name {sorted(map(str, names))}, expected {(sim, snap)}")
    _check_meta_fields(meta, catalog, refuse)
    if require_release_eligible:
        for what, r in (("snapshot record", rec), ("cddf.npz", crec)):
            if not r["release_eligible"]:
                refuse(f"{what} not release-eligible: " + "; ".join(r["ineligible_reasons"]))
    if rehash and raw_path is not None:
        try:
            L.verify_input(catalog.lineage, "raw_spectra", raw_path, rehash=True)
        except (L.LineageError, OSError) as e:
            refuse(f"raw spectra: {e}")
    meta_identity = {"sha256": hashlib.sha256(meta_bytes).hexdigest(), "size": len(meta_bytes)}
    return SnapshotProducts(meta=meta, record=rec, products=prods, meta_identity=meta_identity, catalog=catalog,
                            cddf=cddf, cddf_record=crec)


def _check_meta_fields(meta, catalog, refuse):
    """meta.json's own fields against the catalogue, its record and the raw file facts the record holds (the builder
    computes dN/dX from ``n_absorbers`` and labels the rows with ``z``, so neither may disagree with the catalogue)."""
    prod = catalog.lineage.get("product") or {}
    raw = L.find_input(catalog.lineage, "raw_spectra").get("info") or {}
    counts = meta.get("n_absorbers")
    want = catalog.summary()
    if not isinstance(counts, dict) or {str(k): v for k, v in counts.items()} != want:
        refuse(f"meta.json n_absorbers {counts} are not the catalogue's class counts {want}")
    if sum(want.values()) != prod.get("n_absorbers"):
        refuse(f"the catalogue holds {sum(want.values())} entries, its record states {prod.get('n_absorbers')}")
    pairs = (("z", meta.get("z"), (catalog.z, prod.get("z"))),
             ("dv_kms", meta.get("dv_kms"), (catalog.dv_kms, prod.get("dv_kms"))),
             ("nbins", meta.get("nbins"), (prod.get("nbins"), raw.get("nbins"))),
             ("n_skewers", meta.get("n_skewers"), (raw.get("n_skewers"),)))
    for name, got, refs in pairs:
        if got is None or any(r is None or r != got for r in refs):
            refuse(f"meta.json {name} = {got!r} disagrees with the catalogue or its record ({list(refs)})")
