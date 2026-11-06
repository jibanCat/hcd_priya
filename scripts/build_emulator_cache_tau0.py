"""Build the tau0-extended HCD-emulator training cache (Phase 2, cache format 3.4).

For every fully-processed (sim, snap) pair, drive fake_spectra directly (via
hcd_analysis.priya_p1d) at each of N alpha-slope values to produce both:
  Tier P : PRIYA-compatible total P1D over all sightlines (bit-identical to
           PRIYA's flux_vectors), after the _filter_single_tau_complex mask.
  Tier C : per-class P1D (clean/LLS/subDLA/DLA) on the unfiltered tau, sharing
           Tier P's mean-flux normalisation (scale + target_F).
The CDDF / dN/dX are tau0-invariant and stored once per (sim, snap).

Inputs and lineage (hcd_analysis/tau0_cache.py): the snapshot root and the raw spectra root are required arguments.
Each snapshot directory (catalogue-only or full mode) is read through the snapshot verifier: a release-eligible
catalogue built with the release finder settings on the whole raw file, its CDDF and meta.json, all matching their
recorded sha256. The cache carries a lineage record naming every input by sha256 and is release-eligible only when
built from a clean, committed tree. ``--historical-reproduction REASON`` reads old (pre-lineage) snapshot directories
instead; its output is stamped not release-eligible.

Design: the internal phase-2 HCD emulator design spec.

Usage:
    python3 scripts/build_emulator_cache_tau0.py --hcd-root SNAPSHOT_ROOT --emu-root RAW_SPECTRA_ROOT \
        --output PATH [--fidelity {lf,hr}] [--alpha-refine 2] [--limit N] [--offset M] \
        [--select SIM_SUBSTRING:SNAP ...] [--rehash-raw] [--n-skewers N] [--spot-check] \
        [--historical-reproduction REASON]
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_emulator_cache as bec  # noqa: E402
from hcd_analysis import lineage as L  # noqa: E402
from hcd_analysis import tau0_cache as T  # noqa: E402
from hcd_analysis.tau0_rescale import make_alpha_grid_priya_aligned  # noqa: E402

_N_K = {"lf": 172, "hr": 525}
_COMPLETION_MARKERS = ("done_catalogue", "done")   # catalogue-only mode, full mode

# Canonical LLS / subDLA / DLA log_NHI class boundaries (single source of truth).
# 17.2 = clean->LLS, 19.0 = LLS->subDLA, 20.3 = subDLA->DLA. These MUST be members
# of hcd_analysis.priya_p1d.FINE_NHI_EDGES (the Tier-C fine edges) so that the
# dN/dX per-class partition and the Tier-C P1D partition land on identical N_HI
# boundaries -- otherwise power is silently misattributed between classes.
# Enforced at build time by _assert_class_edges() (called from main()).
# NOTE (follow-up): scripts/diag_wc_from_dndx.py and scripts/diag_crossclass_coupling.py
# duplicate these edges and should be refactored to import COARSE_CLASS_EDGES from here.
COARSE_CLASS_EDGES = (17.2, 19.0, 20.3)


def _assert_class_edges() -> None:
    """Fail the build if the coarse class edges are not Tier-C FINE_NHI_EDGES members.

    Guards against silent drift between the dN/dX class partition (this module)
    and the Tier-C P1D partition (hcd_analysis.priya_p1d.FINE_NHI_EDGES).
    """
    from hcd_analysis.priya_p1d import FINE_NHI_EDGES
    for edge in COARSE_CLASS_EDGES:
        if not np.isclose(FINE_NHI_EDGES, edge, atol=1e-9).any():
            raise ValueError(
                f"Class edge {edge} is not a member of FINE_NHI_EDGES "
                f"({np.asarray(FINE_NHI_EDGES)}); the dN/dX class partition and the "
                "Tier-C partition would disagree -> silent power misattribution.")


# Every Phase-1 catalogue directory snap_NNN was built on raw SPECTRA_NNN (same index; verified for every LF and HR
# directory), so the builder pairs them by that identity. Two directories carry a WRONG meta redshift (taken from an
# offset particle-snapshot list): ns0.907 snap_017 holds the z = 3.0 file's catalogue (meta says 2.799998) and snap_018
# the z = 2.8 file's (meta says 2.673742). Their label is corrected here and verified against the raw header at use.
# History: a 2026-06-01 fix re-pointed only the TAU of snap 17 to SPECTRA_018, leaving the z = 3.0 catalogue on z = 2.8
# rows (finding F8, emulator-debug campaign 2026-10; internal spec S4_REPAIR_SPEC).
_PHASE1_Z_OVERRIDE = {
    ("ns0.907", 17): 3.0,
    ("ns0.907", 18): 2.8,
}
_Z_LABEL_TOL = 2e-5          # largest |meta z - header z| of a correctly labelled directory is 1.3e-5


def vmax_from_header(hdr):
    """Box velocity width [km/s] exactly as fake_spectra (spectra.py:208-215): box * (a / h) * H(z) in kpc units."""
    a = 1.0 / (1.0 + float(hdr.redshift))
    return float(hdr.box) * (3.085678e21 * a / float(hdr.hubble)) * float(hdr.Hz) / 3.085678e24


def raw_header_z(raw_tau_path):
    """Header redshift of a raw spectra file (attributes only; no tau read)."""
    with h5py.File(raw_tau_path, "r") as f:
        return float(f["Header"].attrs["redshift"])


def catalogue_z_label(sim_name, snap, meta_z, raw_z):
    """Redshift label of a Phase-1 catalogue directory: its meta z, except for the enumerated mislabelled directories,
    whose corrected label must equal the paired raw file's header z (|delta| < 1e-6). An entry whose meta z already
    agrees with the header is stale and refused."""
    for (tag, s), z in _PHASE1_Z_OVERRIDE.items():
        if s == snap and tag in sim_name:
            if abs(float(meta_z) - float(raw_z)) <= _Z_LABEL_TOL:
                raise ValueError(f"stale _PHASE1_Z_OVERRIDE entry {(tag, s)}: meta z {meta_z} already equals the header")
            if abs(z - float(raw_z)) >= 1e-6:
                raise ValueError(f"_PHASE1_Z_OVERRIDE {(tag, s)} = {z} disagrees with the raw header z {raw_z}")
            return float(z)
    return float(meta_z)


def assert_label_matches_header(z_label, raw_z):
    if abs(float(z_label) - float(raw_z)) > _Z_LABEL_TOL:
        raise ValueError(f"redshift label {z_label} is not the paired raw file's header redshift {raw_z}")


def _catalogue_pix_max(catalog):
    """Largest absorber start pixel in a catalogue (-1 if it holds no absorbers)."""
    return max((int(a.pix_start) for a in catalog.absorbers), default=-1)


def assert_catalogue_built_on(meta, catalogue_pix_max, hdr):
    """Refuse a catalogue that was not built on the raw file it is paired with: its meta nbins and dv_kms must be the
    file's, and no catalogued absorber may start beyond the file's last pixel."""
    nb = int(hdr.nbins)
    if int(meta["nbins"]) != nb:
        raise ValueError(f"catalogue nbins {meta['nbins']} != paired raw file nbins {nb}")
    dv = vmax_from_header(hdr) / nb
    if abs(float(meta["dv_kms"]) / dv - 1.0) > 1e-12:
        raise ValueError(f"catalogue dv_kms {meta['dv_kms']} != paired raw file {dv}")
    if int(catalogue_pix_max) >= nb:
        raise ValueError(f"catalogue pixel index {catalogue_pix_max} beyond the paired raw file ({nb} pixels)")


def locate_raw_tau_file(emu_root, sim_name: str, snap: int):
    """Return the raw fake_spectra tau HDF5 for (sim_name, snap), or None.

    Layout: <emu_root>/<sim_name>/output/SPECTRA_<NNN>/
            lya_forest_spectra_grid_480.hdf5  (required; 691200-skewer PRIYA product)

    Only the 691200-skewer grid_480 file is accepted.  The low-res 32k
    lya_forest_spectra.hdf5 is never used (it cannot match PRIYA bit-identity).
    The raw SPECTRA dir index is always `snap`: the catalogue in snap_NNN was built on SPECTRA_NNN.
    """
    raw_snap = int(snap)
    found = _grid_in_dir(Path(emu_root) / sim_name / "output" / f"SPECTRA_{raw_snap:03d}")
    if found is not None:
        return found
    # Fallback: the Phase-1 (hcd_outputs) and raw (emu_full) folder names can
    # round the 9 params differently (e.g. the 4th HR sim is alphaq1.57/
    # omegamh20.141 in hcd_outputs but alphaq1.58/omegamh20.142 in emu_full).
    # Match the emu_root folder by parsed params instead of by exact name.
    alt = _match_emu_folder_by_params(emu_root, sim_name)
    if alt is not None:
        return _grid_in_dir(Path(emu_root) / alt / "output" / f"SPECTRA_{raw_snap:03d}")
    return None


def _grid_in_dir(spectra_dir):
    """Return the 691200-skewer PRIYA grid tau file in a SPECTRA dir, or None.

    The low-res `lya_forest_spectra.hdf5` fallback (32000 skewers) is
    intentionally NOT used: PRIYA bit-identity requires the grid_480 product.
    A dir with only the fallback (e.g. ns0.907/SPECTRA_015, a 32k z=3.2
    duplicate) is therefore skipped."""
    if not spectra_dir.is_dir():
        return None
    grid = spectra_dir / "lya_forest_spectra_grid_480.hdf5"
    return grid if grid.exists() else None


def _match_emu_folder_by_params(emu_root, sim_name, rtol=0.015):
    """Return the emu_root folder whose parsed 9 params match `sim_name`'s to
    within `rtol` on every param, or None if there is no unique match.

    Used when the exact-name path is absent because hcd_outputs and emu_full
    round the folder-name params differently. rtol=1.5% safely spans the
    <0.7% rounding gap while staying far below the inter-sim separation.
    """
    target = bec.parse_sim_params(sim_name)
    if target is None:
        return None
    match = None
    for d in sorted(Path(emu_root).iterdir()):
        if not d.is_dir():
            continue
        p = bec.parse_sim_params(d.name)
        if p is None:
            continue
        if all(abs(p[k] - target[k]) <= rtol * max(abs(target[k]), 1e-30)
               for k in target):
            if match is not None and match != d.name:
                return None  # ambiguous -> refuse to guess
            match = d.name
    return match


def discover_snapshot_dirs(root):
    """[(sim, snap, snap_dir)] of the completed snapshot directories under ``root``, in (sim, snap) order: a directory
    snap_NNN of a simulation folder holding meta.json, catalog.npz, cddf.npz and a completion marker of either mode
    (``done_catalogue`` or ``done``). The Phase-1 P1D products (p1d_per_class.h5) are not needed. Whether the products
    may be used is decided when they are read (tau0_cache.load_snapshot_inputs), never here."""
    out = []
    for sim_dir in sorted(Path(root).iterdir()):
        if not sim_dir.is_dir() or bec.parse_sim_params(sim_dir.name) is None:
            continue
        for snap_dir in sorted(sim_dir.iterdir()):
            m = bec.SNAP_DIR_RE.match(snap_dir.name)
            if not m or not snap_dir.is_dir():
                continue
            if not all((snap_dir / f).exists() for f in ("meta.json", "catalog.npz", "cddf.npz")):
                continue
            if not any((snap_dir / f).exists() for f in _COMPLETION_MARKERS):
                continue
            out.append((sim_dir.name, int(m.group(1)), snap_dir))
    return out


def discover_tau0_pairs(hcd_root, emu_root, fidelity="lf", max_off_grid=0.05):
    """Return [(sim, snap, snap_dir, raw_tau_path), ...] for one fidelity.

    fidelity='lf': sims directly under hcd_root; raw under emu_root/<sim>.
    fidelity='hr': pass hcd_root = the hcd_outputs BASE; /hires is appended
    automatically; raw under emu_root/<sim> (emu_full_hires_2). Pairs are
    returned in the deterministic (sim, snap) order from bec.discover_sim_snap_pairs.

    Deduplication: for each (sim, z_grid) only the snap whose meta z is CLOSEST
    to the PRIYA grid is kept. Any kept snap still more than max_off_grid from
    the grid is skipped with a WARN (protects against ~7% wrong mean-flux
    normalization from off-grid snaps getting the wrong target_F).
    """
    import json
    if fidelity == "hr":
        root = Path(hcd_root) / "hires"
    else:
        root = Path(hcd_root)
    out = []
    for sim, snap, snap_dir in discover_snapshot_dirs(root):
        raw = locate_raw_tau_file(emu_root, sim, snap)
        if raw is None:
            continue
        out.append((sim, snap, snap_dir, raw))

    best = {}   # (sim, round(z_grid,4)) -> (off_grid_distance, pair)
    for sim, snap, snap_dir, raw in out:
        z_meta = catalogue_z_label(sim, snap, float(json.load(open(snap_dir / "meta.json"))["z"]), raw_header_z(raw))
        z_grid = _snap_z_to_priya_grid(z_meta)
        off = abs(z_meta - z_grid)
        key = (sim, round(z_grid, 4))
        if key not in best or off < best[key][0]:
            best[key] = (off, (sim, snap, snap_dir, raw))
    deduped = []
    for (sim, z_grid), (off, pair) in best.items():
        if off > max_off_grid:
            print(f"  WARN: skip off-grid snap {pair[0][:20]} {pair[1]} "
                  f"(z_grid={z_grid}, off={off:.3f} > {max_off_grid})")
            continue
        deduped.append(pair)
    return sorted(deduped, key=lambda p: (p[0], p[1]))


def select_pairs(pairs, selectors):
    """Keep only the (sim, snap) groups named by selectors "SIM_SUBSTRING:SNAP" (targeted rebuilds). Each selector
    must match exactly one discovered pair; the discovered order is kept."""
    keep = set()
    for sel in selectors:
        tag, snap = sel.rsplit(":", 1)
        hits = [i for i, p in enumerate(pairs) if tag in p[0] and p[1] == int(snap)]
        if not hits:
            raise ValueError(f"selector {sel!r} matches no discovered pair")
        if len(hits) > 1:
            raise ValueError(f"selector {sel!r} matches more than one pair: {[pairs[i][0] for i in hits]}")
        keep.add(hits[0])
    return [p for i, p in enumerate(pairs) if i in keep]


# PRIYA's zout grid runs 2.0..5.4 in steps of 0.2; snapshot redshifts can land
# slightly off (e.g. 4.600013). For bit-compatibility with PRIYA's training
# data the mean-flux model must use the GRID z, not the snapshot's exact z
# (obs_mean_tau ∝ (1+z)^3.65; an off-grid z gives a ~1e-5 P1D bias — see
# the internal PRIYA P1D consistency check, section 6c).
def _snap_z_to_priya_grid(z):
    """Round a snapshot redshift to PRIYA's zout grid (nearest multiple of 0.2)."""
    return round(float(z) / 0.2) * 0.2


def _read_tau(raw_tau_path, n_skewers=None):
    """Read the full tau grid into memory as float64 (optionally first n_skewers)."""
    with h5py.File(raw_tau_path, "r") as f:
        ds = f["tau/H/1/1215"]
        if n_skewers is not None:
            return ds[:n_skewers].astype(np.float64)
        return ds[...].astype(np.float64)


# PRIYA's emulator parameter "Ap" is the primordial scalar amplitude at the
# Lya pivot k_p = pi/4 /Mpc, while CAMB's `scalar_amp` (As) is at k_0 = 0.05
# /Mpc. They are the same amplitude at different pivots:
#   Ap = As * (k_p / k_0)^(ns - 1),  with k_p/k_0 = (pi/4)/0.05 = 5*pi.
# Verified to machine precision against PRIYA's params array across the LF
# grid (sim 0/29/44).
_AP_PIVOT_RATIO = 5.0 * np.pi  # k_p/k_0


def _read_priya_params(raw_tau_path):
    """Return the PRIYA-exact 9-param vector (in bec.PARAM_ORDER) for a sim,
    read from its SimulationICs.json.

    Folder-name parsing (bec.parse_sim_params) rounds the params (off by up to
    ~0.4% from PRIYA's Latin-hypercube values), so we read full precision from
    SimulationICs.json instead. 8 cosmo params map directly; `Ap` is converted
    from CAMB `scalar_amp` (As) to PRIYA's Lya-pivot Ap via _AP_PIVOT_RATIO.

    `raw_tau_path` is <emu_root>/<sim>/output/SPECTRA_NNN/<grid>.hdf5, so the
    SimulationICs.json lives two parents up from the SPECTRA dir.
    """
    import json
    ics_path = Path(raw_tau_path).resolve().parents[2] / "SimulationICs.json"
    with open(ics_path) as f:
        ics = json.load(f)
    ns = float(ics["ns"])
    ap = float(ics["scalar_amp"]) * _AP_PIVOT_RATIO ** (ns - 1.0)
    vals = {
        "ns": ns,
        "Ap": ap,
        "herei": float(ics["here_i"]),
        "heref": float(ics["here_f"]),
        "alphaq": float(ics["alpha_q"]),
        "hub": float(ics["hubble"]),
        "omegamh2": float(ics["omega0"]) * float(ics["hubble"]) ** 2,
        "hireionz": float(ics["hireionz"]),
        "bhfeedback": float(ics["bhfeedback"]),
    }
    return np.array([vals[k] for k in bec.PARAM_ORDER], dtype=np.float64)


def build_tau0_rows(sim_name, snap, inputs, raw_tau_path, alpha_slope_grid,
                    n_k, n_skewers=None):
    """Build per-alpha rows (Tier-P total + Tier-C per-fine-bin P1D) and the
    per-snap CDDF block for one (sim, snap), driving fake_spectra directly.
    `inputs`: the snapshot's products as read by tau0_cache.load_snapshot_inputs
    (verified) or load_snapshot_inputs_historical: `.meta`, `.cddf`, `.catalog`.

    Tier P: PRIYA-compatible total P1D over all sightlines after the
    _filter_single_tau_complex DLA mask. Bit-identical to PRIYA's flux_vectors.
    Stores the first n_k native FFT bins.
    n_k: number of leading native-FFT k-bins to retain (must be <= the snap's
    native grid length, approximately nbins//2).
    Tier C: per-fine-NHI-bin (N_TIER_C_BINS=15) P1D on the UNFILTERED tau,
    sharing Tier P's scale + target_F (so they decompose against the total).
    Stores P_by_bin[:, :n_k] and n_by_bin counts.

    Returns (rows, snap_block).
    """
    from hcd_analysis.io import read_header
    from hcd_analysis.priya_p1d import (compute_tier_c_p1d, bin_sightlines_by_nhi,
                                        N_TIER_C_BINS, _apply_priya_filter)

    # PRIYA-exact params from SimulationICs.json (full precision; folder-name
    # parsing rounds them, and PRIYA's "Ap" is at a different pivot than CAMB As).
    params = _read_priya_params(raw_tau_path)

    meta, cddf, catalog = inputs.meta, inputs.cddf, inputs.catalog
    dndx = bec.compute_dndx_per_class(meta["n_absorbers"], float(cddf["total_path"]))

    hdr = read_header(raw_tau_path)
    nbins = int(hdr.nbins)
    z_raw = float(hdr.redshift)
    # Hard pairing invariant: the catalogue (and its dv_kms, CDDF) must have been built on THIS raw file.
    assert_catalogue_built_on(meta, _catalogue_pix_max(catalog), hdr)
    dv_kms = float(meta["dv_kms"])
    vmax = nbins * dv_kms

    z_meta = catalogue_z_label(sim_name, snap, float(meta["z"]), z_raw)
    assert_label_matches_header(z_meta, z_raw)
    z_grid = _snap_z_to_priya_grid(z_meta)

    tau_unfilt = _read_tau(raw_tau_path, n_skewers=n_skewers)
    rows = []
    for a_idx, alpha in enumerate(alpha_slope_grid):
        alpha = float(alpha)
        # --- Tier P + filtered Tier C (PRIYA path), MEMORY-OPTIMIZED ---
        # Tier P (the total) == count-weighted sum of the filtered per-class
        # pieces, verified bit-identical to fake_spectra flux_power to ~9e-15 on
        # real data (scripts/verify_tierp_sum_identity.py). So: filter tau in
        # place, compute the filtered Tier-C (which solves the mean-flux `scale`
        # internally), then SUM the pieces -> Tier P. This AVOIDS the monolithic
        # flux_power call, whose ~4-full-array peak (~34 GB) set the SLURM memory
        # bill. compute_tier_p_p1d (flux_power) is retained UNCHANGED as the gold
        # reference in hcd_analysis.priya_p1d + tests + verify_tierp_sum_identity.py.
        tau_filt = tau_unfilt.copy()
        _apply_priya_filter(tau_filt, tau_thresh=1.0e6)
        kf_c, P_by_bin_filt, n_by_bin_filt, target_F, scale = compute_tier_c_p1d(
            tau_filt, vmax, alpha_slope=alpha, z=z_grid, catalog=catalog)
        del tau_filt
        N_sl = int(n_by_bin_filt.sum())
        P_tier_p = (n_by_bin_filt[:, None] / N_sl * P_by_bin_filt).sum(axis=0)
        # Unfiltered Tier C (for the HCD add-back path), SAME scale/target_F:
        _, P_by_bin, n_by_bin, _, _ = compute_tier_c_p1d(
            tau_unfilt, vmax, alpha_slope=alpha, z=z_grid, catalog=catalog,
            external_scale=scale, external_target_F=target_F)
        assert np.array_equal(n_by_bin, n_by_bin_filt), \
            "N_HI class counts differ between filtered and unfiltered Tier C"
        if a_idx == 0:
            assert len(kf_c) >= n_k, \
                f"native grid {len(kf_c)} bins < n_k={n_k} for {sim_name} snap {snap}"
        # Per-class <F> under the uniform rescale (for the shared-vs-per-class
        # target_F disentanglement). NaN sentinel for empty classes (count==0) —
        # consumers MUST mask on tier_c_counts>0 before reducing over mean_F_by_bin.
        # NOTE: a per-pixel freeze ("freeze-core") of self-shielded gas was shown
        # to be a numerical no-op here (scripts/diag_tau_freeze_sensitivity.py:
        # self-shielded pixels have tau>>1 -> flux already saturated -> identical
        # whether frozen or scaled), so production uses the plain uniform rescale.
        # The tau_freeze knob remains in compute_tier_c_p1d for a future
        # absorber-level (Voigt-profile) wing treatment (Phase 3).
        cls = bin_sightlines_by_nhi(catalog, tau_unfilt.shape[0])
        mean_F_by_bin = np.full(N_TIER_C_BINS, np.nan)
        for c in range(N_TIER_C_BINS):
            sel = tau_unfilt[cls == c]
            if sel.size:
                mean_F_by_bin[c] = float(np.mean(np.exp(-scale * sel)))
        rows.append({
            "sim_name": sim_name, "snap": int(snap),
            "alpha_slope": alpha, "alpha_idx": int(a_idx),
            "z_meta": z_meta, "z_grid": float(z_grid),
            "dv_kms": dv_kms, "nbins_native": nbins,
            "target_F": float(target_F), "scale": float(scale),
            "params": params,
            "kfkms": kf_c[:n_k].astype(np.float64),
            "P_tier_p": P_tier_p[:n_k].astype(np.float64),
            "P_tier_c": P_by_bin[:, :n_k].astype(np.float64),
            "P_tier_c_filtered": P_by_bin_filt[:, :n_k].astype(np.float64),
            "tier_c_counts": n_by_bin.astype(np.int64),
            "mean_F_by_bin": mean_F_by_bin.astype(np.float64),
        })

    snap_block = {
        "sim_name": sim_name,
        "snap": int(snap),
        "f_nhi": np.asarray(cddf["f_nhi"], dtype=np.float64),
        "n_absorbers": np.asarray(cddf["n_absorbers"], dtype=np.int64),
        "log_nhi_centres": np.asarray(cddf["log_nhi_centres"], dtype=np.float64),
        "log_nhi_edges": np.asarray(cddf["log_nhi_edges"], dtype=np.float64),
        "total_path_dX": float(cddf["total_path"]),
        **dndx,
    }
    return rows, snap_block


_ROW_FLOAT_KEYS = T.ROW_FLOAT_KEYS
_ROW_INT_KEYS = T.ROW_INT_KEYS
_ROW_P1D_KEYS = T.ROW_P1D_KEYS          # 2-D (n_rows, n_k)
_ROW_TIERC_KEYS = T.ROW_TIERC_KEYS      # 3-D (n_rows, N_TIER_C_BINS, n_k)
_ROW_COUNT_KEYS = T.ROW_COUNT_KEYS      # 2-D (n_rows, N_TIER_C_BINS)
_ROW_MEANF_KEYS = T.ROW_MEANF_KEYS      # 2-D (n_rows, N_TIER_C_BINS)
_SNAP_FLOAT_KEYS = T.SNAP_FLOAT_KEYS
_SNAP_2D_KEYS = T.SNAP_2D_KEYS


def cache_attrs(n_k, alpha_range):
    """The descriptive attributes of a tau0 cache (the recipe in words; cache_version is set by the writer)."""
    return {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": L.code_identity().get("commit") or "unknown",
        "n_k": int(n_k),
        "priya_convention": (
            "Kim 2013 slope-alpha (obs_mean_tau=2.3e-3(1+z)^3.65); "
            "fake_spectra _filter_single_tau_complex(tau_thresh=1e6, thresh2=0.25); "
            "flux_power window=False spec_res=0; native k-grid (first n_k FFT bins)"),
        "tau_thresh": 1.0e6,
        "tier_c_recipe": "uniform",   # per-pixel freeze proven a no-op (diag_tau_freeze_sensitivity.py)
        "alpha_range": np.asarray(alpha_range, dtype=np.float64),
        "k_convention": "angular (rad*s/km) native FFT grid, PRIYA convention",
        "tier_c_note": (
            "P_tier_c = per-class P1D on UNFILTERED tau (uniform rescale); the HCD "
            "add-back delta is P_tier_c - P_tier_c_filtered. "
            "P_tier_c_filtered = on PRIYA tau=1e6-filtered tau "
            "(count-weighted sum == Tier P)."),
    }


RECIPE_ATTRS = ("n_k", "priya_convention", "tau_thresh", "tier_c_recipe", "alpha_range", "k_convention", "tier_c_note")


def stack_rows(rows, snap_blocks):
    """The cache datasets (tau0_cache layout) from per-alpha ``rows`` and per-snapshot ``snap_blocks``."""
    from hcd_analysis.priya_p1d import tier_c_labels, FINE_NHI_EDGES, N_TIER_C_BINS
    assert len(tier_c_labels()) == N_TIER_C_BINS == len(FINE_NHI_EDGES) + 1
    d = {
        "tier_c_labels": np.array(tier_c_labels(), dtype=object),
        "tier_c_nhi_edges": np.asarray(FINE_NHI_EDGES, dtype=np.float64),
        "param_names": np.array(list(bec.PARAM_ORDER), dtype=object),
        "log_nhi_centres": np.asarray(snap_blocks[0]["log_nhi_centres"], dtype=np.float64),
        "log_nhi_edges": np.asarray(snap_blocks[0]["log_nhi_edges"], dtype=np.float64),
        "sim_name": np.array([r["sim_name"] for r in rows], dtype=object),
        "params": np.stack([r["params"] for r in rows], axis=0),
        "snap_sim_name": np.array([b["sim_name"] for b in snap_blocks], dtype=object),
        "snap_snap": np.array([b["snap"] for b in snap_blocks], dtype=np.int32),
    }
    for b in snap_blocks[1:]:
        if not (np.array_equal(b["log_nhi_edges"], d["log_nhi_edges"])
                and np.array_equal(b["log_nhi_centres"], d["log_nhi_centres"])):
            raise ValueError(f"CDDF bins of {b['sim_name']} {b['snap']} differ from the first group's")
    for key in _ROW_FLOAT_KEYS:
        d[key] = np.array([r[key] for r in rows], dtype=np.float64)
    for key in _ROW_INT_KEYS:
        d[key] = np.array([r[key] for r in rows], dtype=np.int32)
    for key in _ROW_P1D_KEYS + _ROW_TIERC_KEYS:
        d[key] = np.stack([r[key] for r in rows], axis=0)
    for key in _ROW_COUNT_KEYS:
        d[key] = np.stack([r[key] for r in rows], axis=0).astype(np.int64)
    for key in _ROW_MEANF_KEYS:
        d[key] = np.stack([r[key] for r in rows], axis=0).astype(np.float64)
    for key in _SNAP_FLOAT_KEYS:
        d["snap_" + key] = np.array([b[key] for b in snap_blocks], dtype=np.float64)
    for key in _SNAP_2D_KEYS:
        d["snap_" + key] = np.stack([b[key] for b in snap_blocks], axis=0)
    return d


def write_cache_tau0(rows, snap_blocks, output_path, alpha_range, n_k, *, inputs, settings,
                     historical_reason=None, selection=None):
    """Stack per-alpha `rows` + per-snap `snap_blocks` into one cache (format 3.4, tau0_cache.write_cache), with a
    lineage record listing `inputs` (the groups' snapshot, catalogue, CDDF and raw entries, and the simulations'
    SimulationICs.json). Each row carries `snap_group_idx`, an index into the snap_* datasets. Returns the record."""
    if not rows:
        raise ValueError("write_cache_tau0 called with no rows.")
    if not snap_blocks:
        raise ValueError("write_cache_tau0 called with no snap_blocks.")
    d = stack_rows(rows, snap_blocks)
    z_of = {}
    for r in rows:
        z_of.setdefault((r["sim_name"], int(r["snap"])), float(r["z_grid"]))
    groups = [{"sim": b["sim_name"], "snap": int(b["snap"]), "z_grid": z_of[(b["sim_name"], int(b["snap"]))],
               "rows": T.ROWS_BUILT} for b in snap_blocks]
    attrs = dict(cache_attrs(n_k, alpha_range), n_rows=len(rows), n_snaps=len(snap_blocks))
    return T.write_cache(output_path, d, attrs, inputs=T.combine_inputs(inputs), settings=settings, groups=groups,
                         historical_reason=historical_reason, selection=selection)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the tau0-extended HCD-emulator cache (Phase 2, format 3.4).")
    parser.add_argument("--fidelity", choices=("lf", "hr"), default="lf")
    parser.add_argument("--hcd-root", type=Path, required=True,
                        help="Snapshot root (the pipeline's output root; for hr its base, /hires is appended).")
    parser.add_argument("--emu-root", type=Path, required=True, help="Raw spectra root (<sim>/output/SPECTRA_NNN).")
    parser.add_argument("--output", type=Path, required=True, help="New cache file (an existing file is refused).")
    parser.add_argument("--alpha-refine", type=int, default=2,
                        help="PRIYA-aligned grid: refine*10 alpha containing "
                             "PRIYA's 10 (default 2 -> 20 alpha).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N (sim, snap) pairs.")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip the first M pairs (for sharded array jobs).")
    parser.add_argument("--n-skewers", type=int, default=None,
                        help="Limit skewers per snap (dry runs only; the output is not release-eligible).")
    parser.add_argument("--select", action="append", default=None, metavar="SIM_SUBSTRING:SNAP",
                        help="Build only these (sim, snap) groups (repeatable; each must match one discovered pair).")
    parser.add_argument("--rehash-raw", action="store_true",
                        help="Re-hash every raw spectra file against its catalogue's record (default: a matching "
                             "size and mtime are accepted).")
    parser.add_argument("--historical-reproduction", metavar="REASON", default=None,
                        help="Read old snapshot directories without lineage (load_catalog_historical); the output "
                             "is stamped not release-eligible with this reason.")
    parser.add_argument("--spot-check", action="store_true",
                        help="After writing, verify row 0 P_tier_p is finite.")
    args = parser.parse_args(argv)

    _assert_class_edges()  # fail fast if dN/dX and Tier-C class edges have drifted
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; a cache is never overwritten")

    fid = args.fidelity
    n_k = _N_K[fid]
    alpha_slope_grid = make_alpha_grid_priya_aligned(refine=args.alpha_refine)
    alpha_range = (float(alpha_slope_grid[0]), float(alpha_slope_grid[-1]))
    output = args.output
    historical = args.historical_reproduction
    if historical is not None and not historical.strip():
        raise SystemExit("--historical-reproduction needs a non-empty reason")
    reason = historical
    if args.n_skewers is not None:
        reason = "; ".join(x for x in (historical, f"dry run on the first {args.n_skewers} skewers per snapshot") if x)

    pairs = discover_tau0_pairs(args.hcd_root, args.emu_root, fidelity=fid)
    print(f"[{fid}] Found {len(pairs)} tau0-buildable (sim, snap) pairs (n_k={n_k})")
    if args.select:
        pairs = select_pairs(pairs, args.select)
        print(f"[{fid}] --select kept {len(pairs)} pairs: {[(p[0][:24], p[1]) for p in pairs]}")
    pairs = pairs[args.offset:]
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"Processing {len(pairs)} pairs (offset={args.offset}, limit={args.limit}); "
          f"n_alpha={len(alpha_slope_grid)}")

    all_rows, snap_blocks, entries = [], [], []
    for i, (sim, snap, snap_dir, raw) in enumerate(pairs):
        if historical is not None:
            inputs, group_entries = T.load_snapshot_inputs_historical(snap_dir, reason=historical)
        else:
            inputs, group_entries = T.load_snapshot_inputs(snap_dir, raw, sim=sim, snap=snap,
                                                           rehash_raw=args.rehash_raw)
        rows, block = build_tau0_rows(sim, snap, inputs, raw, alpha_slope_grid,
                                      n_k=n_k, n_skewers=args.n_skewers)
        gi = len(snap_blocks)
        for r in rows:
            r["snap_group_idx"] = gi
        all_rows.extend(rows)
        snap_blocks.append(block)
        entries.extend(group_entries + [T.sim_ics_entry(raw, sim)])
        if (i + 1) % 10 == 0 or (i + 1) == len(pairs):
            print(f"  built {i+1}/{len(pairs)} pairs ({len(all_rows)} rows)")

    settings = T.cache_settings(fidelity=fid, n_k=n_k, alpha_refine=args.alpha_refine, alpha_range=alpha_range,
                                n_skewers=args.n_skewers)
    selection = {"select": args.select, "offset": int(args.offset), "limit": args.limit, "max_off_grid": 0.05}
    rec = write_cache_tau0(all_rows, snap_blocks, output, alpha_range=alpha_range, n_k=n_k, inputs=entries,
                           settings=settings, historical_reason=reason, selection=selection)
    print(f"Wrote {output} ({output.stat().st_size/1e6:.2f} MB, "
          f"{len(all_rows)} rows, {len(snap_blocks)} snaps); release_eligible={rec['release_eligible']}"
          + ("" if rec["release_eligible"] else f" ({'; '.join(rec['ineligible_reasons'][:3])}"
                                                  f"{'; ...' if len(rec['ineligible_reasons']) > 3 else ''})"))

    if args.spot_check and all_rows:
        with h5py.File(output, "r") as f:
            assert np.isfinite(f["P_tier_p"][0]).any(), \
                "spot-check failed: row 0 P_tier_p has no finite values"
            med_target_F = float(np.median(f["target_F"][...]))
            med_scale = float(np.median(f["scale"][...]))
            # Cheap integrity invariant (NO flux_power): the stored Tier P must
            # equal the count-weighted sum of the stored filtered Tier-C pieces
            # (it is, by construction) — guards against a write/desync bug. The
            # flux_power identity itself is checked offline by
            # scripts/verify_tierp_sum_identity.py.
            ptp = f["P_tier_p"][...]
            pcf = f["P_tier_c_filtered"][...]
            cnt = f["tier_c_counts"][...].astype(np.float64)
            w = cnt / cnt.sum(axis=1, keepdims=True)
            recon = np.einsum("rc,rck->rk", w, pcf)
            m = np.isfinite(ptp) & np.isfinite(recon) & (ptp != 0)
            rel = np.abs(recon[m] / ptp[m] - 1.0)
            assert rel.max() < 1e-10, \
                f"integrity: Tier P != sum(filtered pieces), max|r-1|={rel.max():.2e}"
        print(f"spot-check: row 0 P_tier_p finite OK; integrity max|r-1|={rel.max():.2e}; "
              f"median target_F={med_target_F:.4f} scale={med_scale:.4f}")


if __name__ == "__main__":
    main()
