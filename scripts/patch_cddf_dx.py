"""
Patch on-scratch ``cddf.npz`` and ``cddf_stacked.npz`` files for the
``(1+z)·h`` dX-normalisation bug (bug #7 in ``docs/bugs_found.md``).

Background
----------
Before commit ``c210990`` the function
``hcd_analysis.cddf.absorption_path_per_sightline`` computed dX with two
compounding errors that combined to::

    dX_buggy = dX_correct / [(1+z) · h]

The post-snap CDDF is computed as ``f(N) = n_abs / (dN · total_path)``
where ``total_path = n_sightlines · dX_per_sightline``, so the buggy
``f_nhi`` is **inflated** by the same factor ``(1+z)·h``.

The fix in commit ``c210990`` updated the in-code formula but did not
re-write the artefacts that had already been saved on
``/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/``.  Plotting
scripts recompute dX on the fly so figures from that commit are
correct, but anyone consuming the ``.npz`` files directly still sees
inflated values.

What this script does
---------------------
For every ``<sim>/snap_NNN/cddf.npz`` and ``<sim>/cddf_stacked.npz``
under the scratch hcd_outputs root (LF + HiRes), it writes a corrected
sibling file with the inflation removed:

* ``cddf_corrected.npz`` next to the original ``cddf.npz``
* ``cddf_stacked_corrected.npz`` next to ``cddf_stacked.npz``

Both corrected files carry three extra keys:

* ``dx_bug_patched`` — np.bool_ True
* ``patch_date``     — np.bytes_ ``b'2026-04-25'``
* ``patch_factor``   — float scalar = ``(1+z)·h`` for per-snap;
                       per-bin array ``(1+z_i)·h`` for stacked

Originals are left untouched by default.  ``--in-place`` overwrites
them but is gated behind an explicit confirmation token to prevent
accidents (the pre-audit backup at
``hcd_outputs_pre_audit_bak_2026_04_22/`` lives at the *sibling*
level, so even if --in-place is run it is unaffected).

Usage
-----
::

    python3 scripts/patch_cddf_dx.py --dry-run     # report counts only
    python3 scripts/patch_cddf_dx.py               # write corrected siblings
    python3 scripts/patch_cddf_dx.py --in-place YES_OVERWRITE   # destructive
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")

logging.basicConfig(format="%(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

PATCH_DATE = "2026-04-25"

# Path-segment-level exclusions (defensive — pre-audit backup is a *sibling*
# of hcd_outputs, but if anyone reorganises later we still want the safety).
EXCLUDED_PATH_TOKENS = ("_bak_", "pre_audit_bak", "_hcd_analysis_data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_excluded(path: Path) -> bool:
    s = str(path)
    return any(tok in s for tok in EXCLUDED_PATH_TOKENS)


def _read_hubble(snap_dir: Path) -> Optional[float]:
    """Read h from the snap dir's meta.json (key 'hubble')."""
    meta_p = snap_dir / "meta.json"
    if not meta_p.exists():
        return None
    try:
        with open(meta_p) as f:
            m = json.load(f)
        return float(m["hubble"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def _read_hubble_from_sim(sim_dir: Path) -> Optional[float]:
    """h is sim-level (cosmology), so any snap's meta.json works."""
    for snap_dir in sorted(sim_dir.glob("snap_*")):
        if snap_dir.is_dir():
            h = _read_hubble(snap_dir)
            if h is not None:
                return h
    return None


def _bin_keys(d_keys) -> List[str]:
    """Extract unique <bin> labels from cddf_stacked.npz keys (e.g. 'z2p0to2p5')."""
    labels = set()
    for k in d_keys:
        if "__" in k:
            labels.add(k.split("__", 1)[0])
    return sorted(labels)


# ---------------------------------------------------------------------------
# Per-snap patch
# ---------------------------------------------------------------------------

def patch_snap_cddf(npz_path: Path, hubble: float) -> Dict:
    """
    Read a per-snap cddf.npz and return a dict with the fix applied.

    Math
    ----
    factor = (1 + z) · h
    f_nhi_correct          = f_nhi_buggy / factor
    dX_per_sightline_corr  = dX_per_sightline_buggy * factor
    total_path_corr        = total_path_buggy * factor
    n_absorbers, log_nhi_* are unchanged.

    Idempotent: if the file already has dx_bug_patched=True, returns its
    contents unchanged.
    """
    d = np.load(npz_path)
    out = {k: d[k] for k in d.files}

    if "dx_bug_patched" in out and bool(out["dx_bug_patched"]):
        return out

    z = float(d["z"])
    factor = (1.0 + z) * hubble

    out["f_nhi"] = d["f_nhi"] / factor
    out["dX_per_sightline"] = d["dX_per_sightline"] * factor
    out["total_path"] = d["total_path"] * factor

    out["dx_bug_patched"] = np.bool_(True)
    out["patch_date"] = np.bytes_(PATCH_DATE)
    out["patch_factor"] = np.float64(factor)
    return out


# ---------------------------------------------------------------------------
# Per-sim stacked patch
# ---------------------------------------------------------------------------

def _gather_buggy_paths(sim_dir: Path, z_snapshots: np.ndarray) -> np.ndarray:
    """
    For each z in z_snapshots, find the matching <sim>/snap_NNN/cddf.npz
    and return its (buggy) total_path.

    We always read the *original* cddf.npz here, never the patched
    sibling, so we always get the buggy path and the formula stays
    correct regardless of the order in which patches run.
    """
    # Collect a *list* of (z, path) per file — the same z can appear more
    # than once in a sim (e.g. a partial/restarted snap that produced its
    # own row in cddf_stacked.npz alongside the proper one).
    per_file: List[Tuple[float, float]] = []
    for f in sorted(sim_dir.glob("snap_*/cddf.npz")):
        if _is_excluded(f):
            continue
        try:
            with np.load(f) as ds:
                z_f = float(ds["z"])
                if "dx_bug_patched" in ds.files and bool(ds["dx_bug_patched"]):
                    factor = float(ds["patch_factor"])
                    per_file.append((z_f, float(ds["total_path"]) / factor))
                else:
                    per_file.append((z_f, float(ds["total_path"])))
        except Exception as exc:  # pragma: no cover  (defensive)
            log.warning("        could not read %s: %s", f, exc)

    if not per_file:
        raise RuntimeError(f"no per-snap cddf.npz under {sim_dir}")

    # Match each requested z to a *unique* per-file entry — closest by |Δz|,
    # consumed so duplicate-z requests (e.g. [3.2, 3.2, 3.4]) split across
    # both files at z=3.2.
    out = np.empty_like(z_snapshots, dtype=np.float64)
    pool = list(per_file)
    for i, z in enumerate(z_snapshots):
        if not pool:
            raise RuntimeError(
                f"ran out of per-snap files under {sim_dir} matching "
                f"z_snapshots; requested {len(z_snapshots)}, available {len(per_file)}"
            )
        j = min(range(len(pool)), key=lambda k: abs(pool[k][0] - z))
        if abs(pool[j][0] - z) > 1e-6:
            raise RuntimeError(
                f"z={z} in stacked has no per-snap match under {sim_dir} "
                f"(nearest={pool[j][0]}, |Δz|={abs(pool[j][0] - z):.6f})"
            )
        out[i] = pool.pop(j)[1]
    return out


def patch_stacked_cddf(npz_path: Path, hubble: float) -> Dict:
    """
    Read a per-sim cddf_stacked.npz and return a dict with the fix
    applied to every z-bin.

    For each bin:
      f_nhi_per_snap_corr[i, :] = f_nhi_per_snap[i, :] / [(1+z_i)·h]
      buggy_path_per_snap_i     = total_path_i  (read from per-snap cddf.npz)
      total_path_corr           = Σ_i (1+z_i)·h · buggy_path_per_snap_i
      f_nhi_corr                = n_abs / (dN · total_path_corr)

    n_absorbers, log_nhi_*, z_snapshots, z_min, z_max are unchanged.
    """
    d = np.load(npz_path)
    out = {k: d[k] for k in d.files}

    if "dx_bug_patched" in out and bool(out["dx_bug_patched"]):
        return out

    sim_dir = npz_path.parent
    bins = _bin_keys(out.keys())
    if not bins:
        raise RuntimeError(f"no <bin>__... keys in {npz_path}")

    # Patch factors per bin (variable-length)
    patch_factors_per_bin: Dict[str, np.ndarray] = {}

    for label in bins:
        z_snapshots = np.asarray(d[f"{label}__z_snapshots"], dtype=np.float64)
        per_snap_factors = (1.0 + z_snapshots) * hubble  # shape (n_snaps,)

        # f_nhi_per_snap correction
        fps = np.asarray(d[f"{label}__f_nhi_per_snap"], dtype=np.float64)
        fps_corr = fps / per_snap_factors[:, None]
        out[f"{label}__f_nhi_per_snap"] = fps_corr

        # Reconstruct corrected total_path
        buggy_paths = _gather_buggy_paths(sim_dir, z_snapshots)
        # Sanity check: sum of buggy_paths should equal the file's total_path.
        recorded_total = float(np.asarray(d[f"{label}__total_path"]).ravel()[0])
        if not np.isclose(buggy_paths.sum(), recorded_total, rtol=1e-9):
            raise RuntimeError(
                f"{npz_path} bin {label}: per-snap path sum "
                f"{buggy_paths.sum():.6f} ≠ recorded total_path {recorded_total:.6f}"
            )

        total_path_corr = float(np.sum(per_snap_factors * buggy_paths))

        # Recompute stacked f_nhi
        log_nhi_edges = np.asarray(d[f"{label}__log_nhi_edges"], dtype=np.float64)
        n_abs = np.asarray(d[f"{label}__n_absorbers"], dtype=np.float64)
        dN = 10.0**log_nhi_edges[1:] - 10.0**log_nhi_edges[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            f_corr = np.where(dN > 0, n_abs / (dN * total_path_corr), 0.0)
        out[f"{label}__f_nhi"] = f_corr
        out[f"{label}__total_path"] = np.array([total_path_corr])

        patch_factors_per_bin[label] = per_snap_factors
        # Per-bin patch factor record
        out[f"{label}__patch_factor"] = per_snap_factors

    out["dx_bug_patched"] = np.bool_(True)
    out["patch_date"] = np.bytes_(PATCH_DATE)
    return out


# ---------------------------------------------------------------------------
# Walk + write
# ---------------------------------------------------------------------------

def discover(output_root: Path) -> Tuple[List[Path], List[Path]]:
    """Return (per_snap_files, per_sim_stacked_files), excluding backup paths."""
    snap_files = [
        p for p in output_root.rglob("snap_*/cddf.npz") if not _is_excluded(p)
    ]
    stacked_files = [
        p for p in output_root.rglob("cddf_stacked.npz") if not _is_excluded(p)
    ]
    return sorted(snap_files), sorted(stacked_files)


def _write_npz(path: Path, data: Dict, dry_run: bool) -> None:
    if dry_run:
        return
    # np.savez auto-appends ".npz" if not present, so save to a stem that
    # already ends in ".npz" to suppress the auto-suffix, then atomically
    # rename onto the target path.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **data)
    os.replace(tmp, path)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                   help=f"scratch hcd_outputs root (default: {DEFAULT_OUTPUT_ROOT})")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be done; do not write files")
    p.add_argument("--in-place", metavar="YES_OVERWRITE", default=None,
                   help="overwrite original .npz files (must literally pass 'YES_OVERWRITE')")
    p.add_argument("--limit", type=int, default=None,
                   help="process at most N per-snap and N stacked files (debug)")
    args = p.parse_args(argv)

    output_root = args.output_root
    if not output_root.is_dir():
        log.error("output root not found: %s", output_root)
        return 2

    in_place = args.in_place == "YES_OVERWRITE"
    if args.in_place is not None and not in_place:
        log.error("--in-place must be exactly 'YES_OVERWRITE' if you mean it")
        return 2

    log.info("Discovering files under %s ...", output_root)
    snap_files, stacked_files = discover(output_root)
    log.info("  found %d per-snap cddf.npz", len(snap_files))
    log.info("  found %d per-sim cddf_stacked.npz", len(stacked_files))
    if args.limit is not None:
        snap_files = snap_files[: args.limit]
        stacked_files = stacked_files[: args.limit]
        log.info("  --limit applied: %d / %d", len(snap_files), len(stacked_files))

    if args.dry_run:
        log.info("DRY-RUN — no files will be written.")
    elif in_place:
        log.info("IN-PLACE mode — original files will be overwritten.")
    else:
        log.info("Writing corrected siblings (originals untouched).")

    # ---- per-snap ----
    n_patched = n_skipped = n_failed = 0
    for f in snap_files:
        snap_dir = f.parent
        h = _read_hubble(snap_dir)
        if h is None:
            log.warning("  [snap] %s: missing meta.json/hubble — skip", f)
            n_failed += 1
            continue
        try:
            patched = patch_snap_cddf(f, h)
            if "dx_bug_patched" in np.load(f).files and bool(np.load(f)["dx_bug_patched"]):
                n_skipped += 1
                continue
            target = f if in_place else f.with_name("cddf_corrected.npz")
            log.info("  [snap] %s  z=%.3f h=%.3f  factor=%.4f -> %s",
                     str(f.relative_to(output_root)), float(np.load(f)["z"]),
                     h, float(patched["patch_factor"]),
                     "in-place" if in_place else target.name)
            _write_npz(target, patched, args.dry_run)
            n_patched += 1
        except Exception as exc:
            log.error("  [snap] %s: %s", f, exc)
            n_failed += 1

    # ---- per-sim stacked ----
    s_patched = s_skipped = s_failed = 0
    for f in stacked_files:
        sim_dir = f.parent
        h = _read_hubble_from_sim(sim_dir)
        if h is None:
            log.warning("  [stack] %s: no snap meta.json with hubble — skip", f)
            s_failed += 1
            continue
        try:
            existing = np.load(f).files
            if "dx_bug_patched" in existing and bool(np.load(f)["dx_bug_patched"]):
                s_skipped += 1
                continue
            patched = patch_stacked_cddf(f, h)
            target = f if in_place else f.with_name("cddf_stacked_corrected.npz")
            log.info("  [stack] %s  h=%.3f  bins=%d -> %s",
                     str(f.relative_to(output_root)), h,
                     len(_bin_keys(patched.keys())),
                     "in-place" if in_place else target.name)
            _write_npz(target, patched, args.dry_run)
            s_patched += 1
        except Exception as exc:
            log.error("  [stack] %s: %s", f, exc)
            s_failed += 1

    log.info("--- Summary ---")
    log.info("  per-snap  : patched=%d skipped(already)=%d failed=%d",
             n_patched, n_skipped, n_failed)
    log.info("  stacked   : patched=%d skipped(already)=%d failed=%d",
             s_patched, s_skipped, s_failed)
    log.info("  dry_run=%s  in_place=%s", args.dry_run, in_place)

    return 0 if (n_failed == 0 and s_failed == 0) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
