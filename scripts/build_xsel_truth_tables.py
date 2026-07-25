"""STAGE-V conditional-P1D validation pass: X-battery truth tables in the DEPLOYED convention.

Campaign: cert-campaign-2026-07, PI decisions #7 (autonomous execution), runbook v2.1 step 1.
Design basis: 2026-07-24-conditional-p1d-feasibility.md + PROPOSAL-extreme-battery-v2.md
ROUND-2 REVISIONS 1-4. LAUNCH/ANALYSIS TOOLING ONLY: this script READS the frozen tree and the
lock-pinned tau0 caches; it edits nothing frozen.

WHAT IT COMPUTES, per (sim, z) over the full LF suite + hires, in the DEPLOYED convention
(round-2 MUST-FIX 1: global mean-flux normalization by the cache row's target_F, tau0-rescaled
by the cache row's scale, angular native FFT k grid with k=0 dropped, first n_k bins, and every
curve interpolated onto the KS leg's k grid):

  (a) the three class-conditional truth curves under the HIGHEST-CLASS partition
      (priya_p1d.bin_sightlines_by_nhi, the deployed template partition):
        X3  LLS-selected     : P_filt[LLS]    (deployed trough-fill convention)
        X2  subDLA-selected  : P_filt[subDLA] (deployed trough-fill convention)
        X1  DLA-selected     : P_filt[DLA]-equivalent recomputed (masked conditional)
                               + the dilution-CORRECTED variant (primary fork, PI annex OQ1)
  (b) the X2 decomposition (round-2 MUST-FIX 2: the trough-fill is NOT DLA-only): filled vs
      unfilled subDLA-class P1D, the mask/dilution part (transplanted fill windows on clean
      rows), and the absorber part (selector residual + companions; all-absorbers-masked arm)
  (c) the X1 dilution decomposition with BOTH estimators (round-2 cosmo MUST-FIX):
      transplanted-window DIVISION and direct PIXEL-EXCLUSION (surviving-segment periodogram),
      cross-checked on the KS band (k < 0.02 cyclic == 0.1257 angular; the whole KS band)
  (d) the mask-width sensitivity band ON THE CORRECTED FORK (thresh2=0.125, ~2x-wider windows)
  (e) at-least-one vs highest-class-partition companion tables for X2/X3 (round-2 MUST-FIX 3)

DEPLOYED-CONSISTENCY ANCHOR (diagnostic-deployment consistency rule): the per-(sim,z) pass
re-derives the filtered and unfiltered Tier-C P1Ds from the raw spectra with the deployed
fake_spectra filter and the deployed per-class math, at the cache row's EXACT (scale, target_F),
and FAILS LOUD unless they match the lock-pinned cache bytes (P_tier_c_filtered / P_tier_c /
tier_c_counts / mean_F_by_bin) to CACHE_RTOL. Licensed shortcut (verified on the deployed cache:
fine bins with zero trough-fill-triggered rows are bit-identical between P_tier_c and
P_tier_c_filtered): the unfiltered tier is recomputed only for fine bins containing triggered
rows; untriggered bins are checked against the cache identity instead.

MEMORY: single streaming IO pass over the 3.9 GB tau file (~2-3 GB peak RAM): clean rows are
FFT'd in-stream; non-clean rows plus the seeded clean control pools are collected in RAM
(float32, as stored) for the post-pass tiers, decomposition arms and transplants. A FULL
one-(sim,z) pass is login-node safe, so the smoke IS a full pass with the real cache anchor.

ENV: emu-3.9 + gsl (fake_spectra), exactly like the deployed cache builder:
  export LD_LIBRARY_PATH=/sw/pkgs/arc/stacks/gcc/10.3.0/gsl/2.7/lib:/home/mfho/.conda/envs/emu-3.9/lib:$LD_LIBRARY_PATH
  PY=/home/mfho/.conda/envs/emu-3.9/bin/python3

USAGE:
  $PY scripts/build_xsel_truth_tables.py --fidelity lf --sim-index 0            # one sim, all z
  $PY scripts/build_xsel_truth_tables.py --fidelity lf --sim-index 0 --snaps 19 # one (sim,z)
  $PY scripts/build_xsel_truth_tables.py --fidelity lf --list                   # array sizing
  $PY scripts/build_xsel_truth_tables.py --pool --outdir ... --fig-dir ...      # pool + figures

Pure functions (mixture arithmetic, segment estimator, stamps, KS-grid parsing) are numpy-only
and importable WITHOUT fake_spectra; tests/test_xsel_truth_tables.py exercises them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

# ---------------------------------------------------------------------------
# Pinned conventions (round-2 MUST-FIX 1: everything recomputed in the deployed
# frame; these constants only NAME the deployed choices, they do not redefine them)
# ---------------------------------------------------------------------------
ALPHA_IDX_DEFAULT = 10        # cache tau0 row: PRIYA-exact alpha (even index) nearest 1.0
TAU_THRESH = 1.0e6            # deployed trough-fill detection (cache attr tau_thresh)
THRESH2_DEPLOYED = 0.25       # deployed fill walk-out threshold (fake_spectra default)
THRESH2_WIDE = 0.125          # ~2x-wider window variant (mask-width band, pilot convention)
N_K = {"lf": 172, "hr": 525}  # native k bins retained, == cache attr n_k
SEED_DEFAULT = 20260724       # campaign seed (PI annex: fresh seed 20260724)
SEG_LMIN = 128                # min surviving-segment length [pixels] (k_min ~ 0.005 angular)
CACHE_RTOL = 1e-9             # cache byte-anchor tolerance (pilot precedent ~1e-15)
MEANF_RTOL = 5e-4             # |<F>_filt - target_F|/target_F bound (Newton abs tol 1e-5)
KS_BASE_DEFAULT = "/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/"
KS_Z_LO, KS_Z_HI, KS_KMAX = 2.4, 4.6, 0.069   # deployed load_ks_leg cuts (data_likelihood)

# Coarse class layout over the 15 Tier-C fine bins (single source of truth is
# priya_p1d.FINE_NHI_EDGES; asserted against it at runtime in process_snap).
CLASS_RANGES = {"clean": (0, 1), "LLS": (1, 8), "subDLA": (8, 13), "DLA": (13, 15)}
COARSE = ("clean", "LLS", "subDLA", "DLA")
N_FINE = 15

_HCD_OUTPUTS = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
_LF_EMU_ROOT = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")
_HR_EMU_ROOT = Path("/scratch/yueyingn_root/yueyingn0/mfho/priya/emu_full_hires_2")
_CACHE = {"lf": REPO_ROOT / "hcd_analysis/_emulator_data/observables_tau0_lf.h5",
          "hr": REPO_ROOT / "hcd_analysis/_emulator_data/observables_tau0_hr.h5"}
OUTDIR_DEFAULT = "/scratch/cavestru_root/cavestru0/mfho/cert_2026-07/xsel_truth"

# Curve keys written per (sim,z): native grid + "<key>_ks" on the KS leg k grid.
CURVE_KEYS = (
    "P_filt_clean", "P_filt_lls", "P_filt_sub", "P_filt_dla",
    "P_unf_lls", "P_unf_sub", "P_unf_dla",
    "X1_dla_masked_wide", "X1_dla_allmasked",
    "X1_ctrl_clean", "X1_ctrl_transplant", "X1_ctrl_transplant_wide",
    "X1_corrected", "X1_corrected_wide",
    "X2_ctrl_clean", "X2_ctrl_transplant", "X2_sub_allmasked",
    "X3_lls_allmasked",
    "P_alo_sub_filt", "P_alo_lls_filt",
)
SEG_KEYS = ("X1_seg_dla_ks", "X1_seg_ctrl_ks")   # KS grid only (segment estimator)
SCALAR_KEYS = (
    "z_meta", "z_grid", "snap", "scale", "target_F", "tau_eff_native",
    "n_clean", "n_lls", "n_sub", "n_dla",
    "n_trig_clean", "n_trig_lls", "n_trig_sub", "n_trig_dla",
    "trig_frac_sub", "trig_frac_dla", "multi_dla_frac",
    "maskfrac_dla", "masklen_med", "masklen_p90", "maskfrac_dla_wide",
    "n_alo_sub", "n_alo_lls", "n_dla_with_sub", "n_dla_with_lls", "n_sub_with_lls",
    "n_ctrl1", "n_ctrl2",
    "meanF_clean", "meanF_lls", "meanF_sub", "meanF_dla",
    "cache_row_idx", "crosscheck_filt_max", "crosscheck_unf_max",
    "crosscheck_meanF_max", "meanflux_target_reldiff",
)


# ===========================================================================
# Pure helpers (numpy only; no fake_spectra) -- unit-tested
# ===========================================================================
def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_describe(repo=REPO_ROOT):
    return subprocess.check_output(
        ["git", "describe", "--always", "--dirty", "--tags"], cwd=str(repo)).decode().strip()


def build_convention(fidelity, alpha_idx, alpha_slope, seed):
    """The convention-flag stamp written into every sidecar. Any mismatch across
    pooled members is a hard refusal (fail-loud, never silent)."""
    return {
        "frame": "deployed",
        "mean_flux_normalization": "global target_F (PRIYA Kim2013 slope-alpha), cache row",
        "tau0": f"uniform rescale by the cache row scale (alpha_idx={alpha_idx}, "
                f"alpha_slope={alpha_slope!r})",
        "k_convention": "angular rad s/km, native FFT grid, k=0 dropped, first n_k bins",
        "n_k": N_K[fidelity],
        "filter": f"fake_spectra _filter_single_tau_complex(tau_thresh={TAU_THRESH:g}, "
                  f"thresh2={THRESH2_DEPLOYED}) on NATIVE tau, then rescale",
        "wide_thresh2": THRESH2_WIDE,
        "partition": "highest-class fine-NHI bins (priya_p1d.bin_sightlines_by_nhi); "
                     "coarse = clean[0] LLS[1:8] subDLA[8:13] DLA[13:15]",
        "selection_unit": "per-sightline, one full 120 Mpc/h periodic skewer (PI annex OQ6)",
        "fill_value": "native tau_eff (deltaF=0 pre-rescale)",
        "companion_mask": "catalog pixrange windows filled with native tau_eff "
                          "(decomposition-attribution arm, NOT a deployed product)",
        "seg_lmin_pix": SEG_LMIN,
        "seed": int(seed),
        "fidelity": fidelity,
        "ks_cuts": {"z_lo": KS_Z_LO, "z_hi": KS_Z_HI, "k_max": KS_KMAX},
    }


def coarse_merge(P_by_bin, n_by_bin):
    """Count-weighted merge of the 15 fine Tier-C bins into the 4 coarse classes.
    Same math as priya_p1d.merge_fine_to_classes (zeros where a class is empty);
    the guarded test asserts equality against the deployed function."""
    P_by_bin = np.asarray(P_by_bin, float)
    n_by_bin = np.asarray(n_by_bin)
    out = {}
    for name, (lo, hi) in CLASS_RANGES.items():
        n = n_by_bin[lo:hi]
        ntot = int(n.sum())
        if ntot == 0:
            out[name] = np.zeros(P_by_bin.shape[1])
        else:
            out[name] = (n[:, None] / ntot * P_by_bin[lo:hi]).sum(axis=0)
    return out


def combine_unfiltered(P_filt_bin, n_bin, P_trig_filt, P_trig_unf, n_trig):
    """Mixture-arithmetic reconstruction of the unfiltered bin P1D (the B_eff identity):
    P_unf = P_filt + (n_trig/n) * (P_unf_trig - P_filt_trig).
    Exact because P1D is sightline-additive and the filter only alters triggered rows."""
    if n_trig == 0:
        return np.asarray(P_filt_bin, float).copy()
    if not (0 < n_trig <= n_bin):
        raise ValueError(f"combine_unfiltered: n_trig={n_trig} out of range for n={n_bin}")
    f = n_trig / float(n_bin)
    return np.asarray(P_filt_bin, float) + f * (np.asarray(P_trig_unf, float)
                                                - np.asarray(P_trig_filt, float))


def atleast_one_combine(parts):
    """Count-weighted combination of disjoint partition pieces into an at-least-one
    sample P1D: parts = [(n_i, P_i), ...]. Raises on empty total (fail-loud)."""
    ns = np.array([int(n) for n, _ in parts])
    if ns.sum() == 0:
        raise ValueError("atleast_one_combine: no rows in any part")
    P = np.stack([np.asarray(p, float) for _, p in parts])
    return (ns[:, None] * P).sum(axis=0) / ns.sum()


def dflux_rows(tau_rows, scale, target_F):
    """Deployed delta_F: exp(-scale*tau)/target_F - 1 (global-mean-flux convention)."""
    return np.exp(-float(scale) * np.asarray(tau_rows, np.float64)) / float(target_F) - 1.0


def powerspectrum_rows(delta):
    """fake_spectra._powerspectrum math: |rfft|^2 / npix^2 along the last axis."""
    delta = np.atleast_2d(delta)
    return np.abs(np.fft.rfft(delta, axis=1)) ** 2 / delta.shape[1] ** 2


def p1d_rows(tau_rows, vmax, scale, target_F, chunk=2048):
    """Mean P1D over rows at fixed (scale, target_F): vmax * mean(powerspectrum(dflux)).
    Identical math to priya_p1d._per_class_p1d_at_scale (uniform rescale); returns the
    FULL (npix//2+1,) spectrum INCLUDING k=0 (caller drops it). Zero rows -> zeros."""
    tau_rows = np.atleast_2d(tau_rows)
    n, npix = tau_rows.shape
    acc = np.zeros(npix // 2 + 1)
    if n == 0:
        return acc
    for s in range(0, n, chunk):
        d = dflux_rows(tau_rows[s:s + chunk], scale, target_F)
        acc += powerspectrum_rows(d).sum(axis=0)
    return float(vmax) * acc / n


def native_kgrid(npix, vmax, n_k):
    """Deployed native angular k grid: 2*pi*j/vmax for j=1..n_k (k=0 dropped),
    == fake_spectra._flux_power_bins(vmax, npix)[1:n_k+1]."""
    if n_k > npix // 2:
        raise ValueError(f"n_k={n_k} exceeds native grid ({npix//2}) -- wrong fidelity?")
    return 2.0 * np.pi * np.arange(1, n_k + 1) / float(vmax)


def mask_to_segments(mask):
    """Unmasked runs of a CYCLIC boolean mask (True = masked). Returns [(start, length)]
    in original coordinates; a fully unmasked row is one wrapped run of full length."""
    mask = np.asarray(mask, bool)
    n = mask.size
    if not mask.any():
        return [(0, n)]
    if mask.all():
        return []
    # rotate so index 0 is masked -> unmasked runs never wrap
    first = int(np.argmax(mask))
    r = np.roll(mask, -first)
    d = np.diff(np.concatenate(([1], r.view(np.int8), [1])))
    starts = np.where(d == -1)[0]
    ends = np.where(d == 1)[0]
    return [((int(s) + first) % n, int(e - s)) for s, e in zip(starts, ends)]


def segment_power_samples(delta_row, mask, dv, lmin=SEG_LMIN):
    """Direct pixel-exclusion estimator: per surviving segment (length >= lmin),
    the non-cyclic periodogram P(k_j) = (L*dv) * |rfft(d)|^2 / L^2 at angular
    k_j = 2*pi*j/(L*dv), j>=1. Returns (k, P, w) sample arrays, w = segment length."""
    ks, ps, ws = [], [], []
    n = delta_row.size
    for start, L in mask_to_segments(mask):
        if L < lmin:
            continue
        idx = (start + np.arange(L)) % n
        d = np.asarray(delta_row, np.float64)[idx]
        P = (L * dv) * (np.abs(np.fft.rfft(d)) ** 2 / L ** 2)
        k = 2.0 * np.pi * np.arange(L // 2 + 1) / (L * dv)
        ks.append(k[1:]); ps.append(P[1:]); ws.append(np.full(L // 2, float(L)))
    if not ks:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(ks), np.concatenate(ps), np.concatenate(ws)


def edges_from_centers_geom(c):
    """Geometric-midpoint bin edges from strictly-increasing positive centers."""
    c = np.asarray(c, float)
    if c.ndim != 1 or c.size < 2 or (np.diff(c) <= 0).any() or (c <= 0).any():
        raise ValueError("edges_from_centers_geom: need increasing positive centers")
    mid = np.sqrt(c[:-1] * c[1:])
    lo = c[0] ** 2 / mid[0]
    hi = c[-1] ** 2 / mid[-1]
    return np.concatenate([[lo], mid, [hi]])


def bin_weighted(k_s, P_s, w_s, edges):
    """Weighted mean of P samples per bin; NaN where a bin is empty."""
    out = np.full(len(edges) - 1, np.nan)
    if len(k_s) == 0:
        return out
    which = np.digitize(k_s, edges) - 1
    for b in range(len(edges) - 1):
        m = which == b
        if m.any():
            out[b] = np.average(P_s[m], weights=w_s[m])
    return out


def transplant_apply(tau_ctrl, donor_masks, tau_fill, trigger_frac, rng):
    """Transplant donor mask patterns onto control rows: each ctrl row receives, with
    probability trigger_frac, a randomly chosen donor row's mask ROLLED by a random
    cyclic offset, filled with tau_fill. Returns (tau_out float64, applied_masks list
    aligned to rows; None where no transplant). Deterministic given rng state."""
    tau_out = np.asarray(tau_ctrl, np.float64).copy()
    n, npix = tau_out.shape
    applied = [None] * n
    if len(donor_masks) == 0:
        return tau_out, applied
    for i in range(n):
        if rng.random() > trigger_frac:
            continue
        m = donor_masks[int(rng.integers(0, len(donor_masks)))]
        m = np.roll(m, int(rng.integers(0, npix)))
        tau_out[i][m] = tau_fill
        applied[i] = m
    return tau_out, applied


def interp_to_grid(k_native, P, k_target):
    """np.interp onto the leg grid with a hard bounds check (never extrapolate)."""
    k_native = np.asarray(k_native, float)
    k_target = np.asarray(k_target, float)
    if k_target.max() > k_native.max() + 1e-12 or k_target.min() < k_native.min() - 1e-12:
        raise ValueError(f"interp_to_grid: target [{k_target.min():g},{k_target.max():g}] "
                         f"outside native [{k_native.min():g},{k_native.max():g}]")
    return np.interp(k_target, k_native, np.asarray(P, float))


def ks_project(k_native, P, k_ks, z_grid):
    """Project a native curve onto the KS leg k grid, or NaN if unprojectable.

    Snaps outside the deployed KS z window [KS_Z_LO, KS_Z_HI] are never consumed
    on the KS grid (the analyzer masks to the leg's z bins), and at z > KS_Z_HI
    the native velocity-grid Nyquist in s/km shrinks with z and cosmology and can
    fall below the last KS bin centre for some sims -- so those rows are NaN, not
    a refusal. Inside the window the interp_to_grid bounds check stays fail-loud
    (an in-window coverage failure would corrupt the truth tables and must stop
    the run)."""
    if not (KS_Z_LO - 1e-6 <= z_grid <= KS_Z_HI + 1e-6):
        return np.full(np.asarray(k_ks).size, np.nan)
    if not np.isfinite(np.asarray(P, float)).all():
        return np.full(np.asarray(k_ks).size, np.nan)
    return interp_to_grid(k_native, P, k_ks)


def ks_leg_kgrid(base=KS_BASE_DEFAULT, z_lo=KS_Z_LO, z_hi=KS_Z_HI, k_max=KS_KMAX):
    """The KS leg's unique post-cut k grid, parsed directly from the conservative P1D
    table with the deployed load_ks_leg cuts. The analyzer re-verifies this against
    data_likelihood.load_ks_leg (jax env) -- a mismatch is a hard refusal there."""
    path = str(base).rstrip("/") + "/final-conservative-p1d-karacayli_etal2021.txt"
    z, k = [], []
    with open(path) as f:
        for ln in f.readlines()[1:]:
            parts = ln.split("|")
            if len(parts) < 4:
                continue
            try:
                zz, kk = float(parts[1]), float(parts[2])
            except ValueError:
                continue
            z.append(zz); k.append(kk)
    z, k = np.array(z), np.array(k)
    keep = (z >= z_lo - 1e-6) & (z <= z_hi + 1e-6) & (k <= k_max + 1e-9)
    ku = np.unique(k[keep])
    if ku.size == 0:
        raise RuntimeError(f"ks_leg_kgrid: no rows kept from {path}")
    return ku


def stamp_sidecar(npz_path, stamp):
    """Write <npz>.stamp.json carrying the npz sha256 + provenance. Fail-loud reader
    counterpart: verify_sidecar."""
    stamp = dict(stamp)
    stamp["npz_sha256"] = sha256_file(npz_path)
    stamp["npz_file"] = os.path.basename(str(npz_path))
    side = str(npz_path) + ".stamp.json"
    with open(side, "w") as f:
        json.dump(stamp, f, indent=1, sort_keys=True, default=str)
    return side


def verify_sidecar(npz_path):
    """Load + verify a stamped npz: sidecar exists, sha matches. Returns the stamp."""
    side = str(npz_path) + ".stamp.json"
    if not os.path.exists(side):
        raise FileNotFoundError(f"missing sidecar stamp: {side}")
    with open(side) as f:
        stamp = json.load(f)
    actual = sha256_file(npz_path)
    if stamp.get("npz_sha256") != actual:
        raise RuntimeError(f"sha mismatch for {npz_path}: sidecar {stamp.get('npz_sha256')} "
                           f"!= actual {actual} (stale or tampered output)")
    return stamp


def snap_rng(seed, sim_name, snap):
    """Deterministic per-(sim,snap) RNG (campaign seed + sim/snap entropy)."""
    return np.random.default_rng([int(seed), int(snap), zlib.crc32(sim_name.encode())])


def trig_counts_by_fine_bin(cls, trig):
    """Triggered-row count per fine Tier-C bin (int array, len N_FINE)."""
    out = np.zeros(N_FINE, np.int64)
    for c in range(N_FINE):
        out[c] = int((trig & (cls == c)).sum())
    return out


def _coarse_meanF(meanF_by_bin, n_by_bin, name):
    lo, hi = CLASS_RANGES[name]
    n = n_by_bin[lo:hi]
    if n.sum() == 0:
        return float("nan")
    m = meanF_by_bin[lo:hi]
    return float(np.sum(np.where(n > 0, m, 0.0) * n) / n.sum())


def _rel_max(a, b, mask=None):
    """max |a/b - 1| over finite, nonzero-b entries (optionally masked)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b) & (b != 0)
    if mask is not None:
        m &= mask
    if not m.any():
        return 0.0
    return float(np.max(np.abs(a[m] / b[m] - 1.0)))


# ===========================================================================
# Heavy per-(sim,z) pass (fake_spectra env required)
# ===========================================================================
def _filter_row(tau_row64, tau_eff, npix, thresh2):
    """Deployed trough-fill on ONE row (mutates + returns): the exact fake_spectra
    Spectra._filter_single_tau_complex via the priya_p1d SimpleNamespace pattern."""
    from types import SimpleNamespace
    from fake_spectra.spectra import Spectra
    Spectra._filter_single_tau_complex(SimpleNamespace(nbins=npix), tau_row64, tau_eff,
                                       tau_thresh=TAU_THRESH, thresh2=thresh2)
    return tau_row64


def _cache_row(cache_path, sim_name, snap, alpha_idx):
    """Locate + read the lock-pinned cache row for (sim, snap, alpha_idx). Fail-loud."""
    import h5py
    with h5py.File(cache_path, "r") as f:
        sims = f["sim_name"][...]
        snaps = f["snap"][...]
        aidx = f["alpha_idx"][...]
        hit = np.where((sims == sim_name.encode()) & (snaps == int(snap))
                       & (aidx == int(alpha_idx)))[0]
        if hit.size != 1:
            raise RuntimeError(f"cache row for ({sim_name}, snap {snap}, alpha_idx "
                               f"{alpha_idx}) not unique/found in {cache_path}: {hit}")
        r = int(hit[0])
        return dict(
            row=r,
            alpha_slope=float(f["alpha_slope"][r]),
            scale=float(f["scale"][r]),
            target_F=float(f["target_F"][r]),
            z_grid=float(f["z_grid"][r]),
            kfkms=f["kfkms"][r][...],
            P_tier_c=f["P_tier_c"][r][...],
            P_tier_c_filtered=f["P_tier_c_filtered"][r][...],
            tier_c_counts=f["tier_c_counts"][r][...],
            mean_F_by_bin=f["mean_F_by_bin"][r][...],
            cache_git_sha=str(f.attrs.get("git_sha", "")),
        )


def process_snap(sim_name, snap, snap_dir, raw_path, fidelity, *, alpha_idx, outdir,
                 seed=SEED_DEFAULT, batch_rows=4096, n_skewers=None, tau_sha=True,
                 ks_base=KS_BASE_DEFAULT, skip_existing=False):
    """Full STAGE-V pass for one (sim, z). Writes <outdir>/<fidelity>/<sim>/xsel_zZ.npz
    + .stamp.json. Returns the npz path. Every cross-check failure raises.

    n_skewers=None (default) is the FULL-SUITE pass with the hard cache byte-anchor.
    n_skewers=N is a plumbing smoke: the anchor checks are SKIPPED and the sidecar is
    stamped full_suite=False, which the pooler refuses (never silently poolable)."""
    import h5py
    from hcd_analysis.catalog import AbsorberCatalog
    from hcd_analysis.masking import build_skewer_mask
    from hcd_analysis.priya_p1d import FINE_NHI_EDGES, N_TIER_C_BINS, bin_sightlines_by_nhi
    from hcd_analysis.tau0_rescale import make_alpha_grid_priya_aligned

    t0 = time.time()
    assert N_TIER_C_BINS == N_FINE, "Tier-C bin-count drift vs deployed priya_p1d"
    n_k = N_K[fidelity]
    meta = json.load(open(Path(snap_dir) / "meta.json"))
    z_meta = float(meta["z"])
    dv = float(meta["dv_kms"])
    z_grid = round(z_meta / 0.2) * 0.2
    full = n_skewers is None
    out_sim_dir = Path(outdir) / fidelity / sim_name
    out_sim_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_sim_dir / f"xsel_z{z_grid:.1f}.npz"
    if skip_existing and npz_path.exists():
        try:
            st = verify_sidecar(npz_path)
            if st.get("full_suite", False):
                print(f"[skip] {npz_path} exists + sidecar verifies")
                return str(npz_path)
            print(f"[skip-guard] {npz_path} is a smoke output; recomputing full")
        except Exception as e:  # stale/partial output: recompute, loudly
            print(f"[skip-guard] {npz_path} present but UNVERIFIED ({e}); recomputing")

    # single source of truth for the class edges (mirrors the cache builder's assert)
    for edge in (17.2, 19.0, 20.3):
        if not np.isclose(FINE_NHI_EDGES, edge, atol=1e-9).any():
            raise RuntimeError(f"class edge {edge} not in FINE_NHI_EDGES -- partition drift")

    # --- cache anchor (deployed scale/target_F/z_grid + the byte reference) ----
    cache = _cache_row(_CACHE[fidelity], sim_name, snap, alpha_idx)
    grid = make_alpha_grid_priya_aligned(refine=2)
    if abs(cache["alpha_slope"] - grid[alpha_idx]) > 1e-12:
        raise RuntimeError(f"cache alpha_slope {cache['alpha_slope']} != grid[{alpha_idx}]"
                           f"={grid[alpha_idx]} -- wrong cache/alpha convention")
    if abs(cache["z_grid"] - z_grid) > 1e-9:
        raise RuntimeError(f"z_grid mismatch: meta {z_grid} vs cache {cache['z_grid']}")
    scale, target_F = cache["scale"], cache["target_F"]

    with h5py.File(raw_path, "r") as f:
        n_total_file, npix = f["tau/H/1/1215"].shape
    n_total = n_total_file if full else min(n_total_file, int(n_skewers))
    vmax = npix * dv
    kf = native_kgrid(npix, vmax, n_k)
    if _rel_max(kf, cache["kfkms"]) > 1e-12:
        raise RuntimeError("native k grid != cache kfkms row -- dv/nbins mismatch")
    k_ks = ks_leg_kgrid(ks_base)
    ks_edges = edges_from_centers_geom(k_ks)

    catalog = AbsorberCatalog.load_npz(Path(snap_dir) / "catalog.npz")
    cls = bin_sightlines_by_nhi(catalog, n_total)                # 15 fine bins
    coarse_of = np.zeros(N_FINE, int)
    for ci, name in enumerate(COARSE):
        lo, hi = CLASS_RANGES[name]
        coarse_of[lo:hi] = ci
    lab = coarse_of[cls]                                          # 0 clean..3 DLA

    # per-row absorber counts by class (companion incidence + at-least-one)
    n_abs = np.zeros((n_total, 3), np.int32)                      # cols LLS/sub/DLA
    cls_col = {"LLS": 0, "subDLA": 1, "DLA": 2}
    for ab in catalog.absorbers:
        if ab.skewer_idx < n_total and ab.absorber_class in cls_col:
            n_abs[ab.skewer_idx, cls_col[ab.absorber_class]] += 1

    rng = snap_rng(seed, sim_name, snap)
    idx_nonclean = np.where(lab > 0)[0]
    clean_idx = np.where(lab == 0)[0]
    n_dla = int((lab == 3).sum()); n_sub = int((lab == 2).sum()); n_lls = int((lab == 1).sum())
    # control pools (disjoint, seeded); sizes capped by the clean pool
    n_ctrl1 = int(min(3 * max(n_dla, 1), max(len(clean_idx) // 4, 1)))
    n_ctrl2 = int(min(3 * max(n_sub, 1), max(len(clean_idx) // 4, 1)))
    perm = rng.permutation(clean_idx)
    ctrl1_idx = np.sort(perm[:n_ctrl1])
    ctrl2_idx = np.sort(perm[n_ctrl1:n_ctrl1 + n_ctrl2])

    keep_idx = np.unique(np.concatenate([idx_nonclean, ctrl1_idx, ctrl2_idx]))
    slots = np.full(n_total, -1, np.int64)
    slots[keep_idx] = np.arange(len(keep_idx))
    store = np.empty((len(keep_idx), npix), np.float32)

    # ---- pass A: single stream over the tau file -------------------------------
    exp_sum_native = 0.0                     # sum exp(-tau) over ALL rows -> tau_eff
    row_max = np.empty(n_total)
    pow_clean = np.zeros(npix // 2 + 1)      # clean rows, deployed dflux power sum
    flux_clean = 0.0                         # sum exp(-scale*tau) over clean rows
    n_clean_trig = 0
    with h5py.File(raw_path, "r") as f:
        ds = f["tau/H/1/1215"]
        for s in range(0, n_total, batch_rows):
            e = min(s + batch_rows, n_total)
            tb32 = ds[s:e]
            tb = tb32.astype(np.float64)
            exp_sum_native += float(np.exp(-tb).sum())
            rm = tb.max(axis=1)
            row_max[s:e] = rm
            lb = lab[s:e]
            km = slots[s:e] >= 0
            if km.any():
                store[slots[s:e][km]] = tb32[km]
            n_clean_trig += int(((lb == 0) & (rm > TAU_THRESH)).sum())
            mc = lb == 0
            if mc.any():
                df = np.exp(-scale * tb[mc]) / target_F - 1.0
                pow_clean += powerspectrum_rows(df).sum(axis=0)
                flux_clean += (float(df.sum()) + df.size) * target_F
    if n_clean_trig:
        # A clean-labelled row above the trough-fill trigger breaks the partition
        # assumptions (catalog/label drift). Record loudly and stop -- never pool.
        raise RuntimeError(f"{n_clean_trig} clean-labelled rows trigger tau>"
                           f"{TAU_THRESH:g} at {sim_name} snap {snap} (z{z_grid:.1f}); "
                           "investigate catalog/label drift before pooling")
    tau_eff_native = -np.log(max(exp_sum_native / (n_total * npix), 1e-30))
    trig = row_max > TAU_THRESH
    trig_fine = trig_counts_by_fine_bin(cls, trig)
    n_trig_by_class = np.array([int((trig & (lab == c)).sum()) for c in range(4)])
    print(f"[{sim_name[:14]} z{z_grid:.1f}] pass A {time.time()-t0:.0f}s: "
          f"counts clean/LLS/sub/DLA = {len(clean_idx)}/{n_lls}/{n_sub}/{n_dla}, "
          f"trig by class {n_trig_by_class.tolist()}, tau_eff={tau_eff_native:.4f}",
          flush=True)

    # ---- deployed trough-fill on triggered stored rows --------------------------
    filt_of, mask_of = {}, {}
    for g in keep_idx[trig[keep_idx]]:
        row = store[slots[g]].astype(np.float64)
        orig = row.copy()
        _filter_row(row, tau_eff_native, npix, THRESH2_DEPLOYED)
        filt_of[int(g)] = row
        mask_of[int(g)] = row != orig

    def gather(global_idx, filtered):
        """(n, npix) float64 rows; filtered=True substitutes trough-filled rows."""
        global_idx = np.atleast_1d(global_idx)
        out = np.empty((len(global_idx), npix))
        for i, g in enumerate(global_idx):
            g = int(g)
            if slots[g] < 0:
                raise KeyError(f"row {g} not collected (bookkeeping bug)")
            if filtered and g in filt_of:
                out[i] = filt_of[g]
            else:
                out[i] = store[slots[g]].astype(np.float64)
        return out

    # ---- Tier-C fine-bin P1Ds (filtered everywhere; unfiltered where triggered) --
    P_filt_bins = np.zeros((N_FINE, npix // 2 + 1))
    P_unf_bins = np.zeros_like(P_filt_bins)
    n_by_bin = np.array([int((cls == c).sum()) for c in range(N_FINE)])
    meanF_by_bin = np.full(N_FINE, np.nan)
    flux_filt_sum = flux_clean                # global FILTERED <F> check accumulator
    if n_by_bin[0] > 0:
        P_filt_bins[0] = vmax * pow_clean / n_by_bin[0]
        P_unf_bins[0] = P_filt_bins[0]
        meanF_by_bin[0] = flux_clean / (n_by_bin[0] * npix)
    for c in range(1, N_FINE):
        rows_c = np.where(cls == c)[0]
        if rows_c.size == 0:
            continue
        tf = gather(rows_c, filtered=True)
        P_filt_bins[c] = p1d_rows(tf, vmax, scale, target_F)
        flux_filt_sum += float(np.exp(-scale * tf).sum())
        if trig_fine[c]:
            meanF_by_bin[c] = float(np.mean(np.exp(-scale * gather(rows_c, False))))
            tr = rows_c[trig[rows_c]]
            Pt_f = p1d_rows(gather(tr, True), vmax, scale, target_F)
            Pt_u = p1d_rows(gather(tr, False), vmax, scale, target_F)
            P_unf_bins[c] = combine_unfiltered(P_filt_bins[c], rows_c.size,
                                               Pt_f, Pt_u, tr.size)
        else:
            meanF_by_bin[c] = float(np.mean(np.exp(-scale * tf)))
            P_unf_bins[c] = P_filt_bins[c]
        del tf

    # global filtered mean flux vs the cache row's Newton target (deployed solve
    # ran on the FILTERED tau; abs tol 1e-5)
    mf_filt = flux_filt_sum / (n_total * npix)
    mf_rel = abs(mf_filt / target_F - 1.0)

    # ---- cache byte-anchor cross-checks (fail-loud; FULL suite only) -------------
    if full:
        if not np.array_equal(n_by_bin, cache["tier_c_counts"]):
            raise RuntimeError(f"tier_c_counts mismatch vs cache: {n_by_bin.tolist()} "
                               f"vs {cache['tier_c_counts'].tolist()}")
        chk_f = max(_rel_max(P_filt_bins[c][1:n_k + 1], cache["P_tier_c_filtered"][c])
                    for c in range(N_FINE) if n_by_bin[c] > 0)
        if chk_f > CACHE_RTOL:
            raise RuntimeError(f"filtered Tier-C vs cache max|r-1|={chk_f:.3e} > "
                               f"{CACHE_RTOL} -- NOT the deployed pipeline; refusing")
        chk_u = 0.0
        for c in range(N_FINE):
            if n_by_bin[c] == 0:
                continue
            if trig_fine[c] == 0:
                # licensed shortcut: the cache itself must be bit-identical here
                if not np.array_equal(cache["P_tier_c"][c], cache["P_tier_c_filtered"][c]):
                    raise RuntimeError(f"cache bin {c}: unfiltered != filtered but no "
                                       "triggered rows found -- trigger bookkeeping bug")
            chk_u = max(chk_u, _rel_max(P_unf_bins[c][1:n_k + 1], cache["P_tier_c"][c]))
        if chk_u > CACHE_RTOL:
            raise RuntimeError(f"unfiltered Tier-C vs cache max|r-1|={chk_u:.3e} > "
                               f"{CACHE_RTOL}")
        chk_mf = _rel_max(meanF_by_bin, cache["mean_F_by_bin"])
        if chk_mf > 1e-9:
            raise RuntimeError(f"mean_F_by_bin vs cache max|r-1|={chk_mf:.3e} > 1e-9")
        if mf_rel > MEANF_RTOL:
            raise RuntimeError(f"filtered <F>={mf_filt:.6f} vs target_F={target_F:.6f} "
                               f"rel diff {mf_rel:.2e} > {MEANF_RTOL} -- rescale drift")
        print(f"[{sim_name[:14]} z{z_grid:.1f}] cache anchor OK: filt {chk_f:.2e} "
              f"unf {chk_u:.2e} meanF {chk_mf:.2e} <F>filt/target-1 {mf_rel:.2e} "
              f"{time.time()-t0:.0f}s", flush=True)
    else:
        chk_f = chk_u = chk_mf = float("nan")
        print(f"[{sim_name[:14]} z{z_grid:.1f}] SMOKE subset: cache anchor SKIPPED "
              f"(<F>filt/target-1 = {mf_rel:.2e}, meaningless on a subset)", flush=True)

    Pf = coarse_merge(P_filt_bins, n_by_bin)
    Pu = coarse_merge(P_unf_bins, n_by_bin)

    # ---- decomposition arms -------------------------------------------------------
    curves = {
        "P_filt_clean": Pf["clean"], "P_filt_lls": Pf["LLS"],
        "P_filt_sub": Pf["subDLA"], "P_filt_dla": Pf["DLA"],
        "P_unf_lls": Pu["LLS"], "P_unf_sub": Pu["subDLA"], "P_unf_dla": Pu["DLA"],
    }
    dla_idx = np.where(lab == 3)[0]
    sub_idx = np.where(lab == 2)[0]
    lls_idx = np.where(lab == 1)[0]
    dla_trig = dla_idx[trig[dla_idx]]
    sub_trig = sub_idx[trig[sub_idx]]
    donor_masks_dla = [mask_of[int(g)] for g in dla_trig]
    donor_masks_sub = [mask_of[int(g)] for g in sub_trig]
    trig_frac_dla = len(dla_trig) / max(len(dla_idx), 1)
    trig_frac_sub = len(sub_trig) / max(len(sub_idx), 1)
    masklens = (np.array([int(m.sum()) for m in donor_masks_dla])
                if donor_masks_dla else np.array([0]))
    nan_curve = np.full(npix // 2 + 1, np.nan)

    abs_by_row = {}
    for ab in catalog.absorbers:
        if ab.skewer_idx < n_total:
            abs_by_row.setdefault(int(ab.skewer_idx), []).append(ab)

    def companions_of(g, classes):
        return [ab for ab in abs_by_row.get(int(g), ())
                if ab.absorber_class in classes]

    # X1: wide-mask variant on DLA rows (thresh2=0.125) + all-absorbers-masked arm
    if len(dla_idx):
        wide_rows = gather(dla_idx, filtered=False)
        wide_masks = []
        for i, g in enumerate(dla_idx):
            if trig[g]:
                orig = wide_rows[i].copy()
                _filter_row(wide_rows[i], tau_eff_native, npix, THRESH2_WIDE)
                wide_masks.append(wide_rows[i] != orig)
        curves["X1_dla_masked_wide"] = p1d_rows(wide_rows, vmax, scale, target_F)
        maskfrac_wide = (sum(int(m.sum()) for m in wide_masks)
                         / max(len(dla_idx) * npix, 1))
        del wide_rows
        allm = gather(dla_idx, filtered=True)
        for i, g in enumerate(dla_idx):
            comp = companions_of(g, ("LLS", "subDLA"))
            if comp:
                allm[i][build_skewer_mask(npix, comp)] = tau_eff_native
        curves["X1_dla_allmasked"] = p1d_rows(allm, vmax, scale, target_F)
        del allm
    else:
        curves["X1_dla_masked_wide"] = nan_curve
        curves["X1_dla_allmasked"] = nan_curve
        wide_masks, maskfrac_wide = [], float("nan")

    # X1 control: transplanted deployed windows on the clean pool
    ctrl1 = gather(ctrl1_idx, filtered=False)
    curves["X1_ctrl_clean"] = p1d_rows(ctrl1, vmax, scale, target_F)
    tw, applied1 = transplant_apply(ctrl1, donor_masks_dla, tau_eff_native,
                                    trig_frac_dla, rng)
    curves["X1_ctrl_transplant"] = p1d_rows(tw, vmax, scale, target_F)
    del tw
    tww, _ = transplant_apply(ctrl1, wide_masks, tau_eff_native, trig_frac_dla, rng)
    curves["X1_ctrl_transplant_wide"] = p1d_rows(tww, vmax, scale, target_F)
    del tww
    with np.errstate(divide="ignore", invalid="ignore"):
        R_dil = curves["X1_ctrl_transplant"] / curves["X1_ctrl_clean"]
        R_dil_w = curves["X1_ctrl_transplant_wide"] / curves["X1_ctrl_clean"]
        curves["X1_corrected"] = curves["P_filt_dla"] / R_dil
        curves["X1_corrected_wide"] = curves["X1_dla_masked_wide"] / R_dil_w

    # X1 pixel-exclusion (surviving-segment) estimator, on the KS grid
    def seg_accumulate(pairs):
        ks_, ps_, ws_ = [], [], []
        for delta, mask in pairs:
            k_s, p_s, w_s = segment_power_samples(delta, mask, dv)
            ks_.append(k_s); ps_.append(p_s); ws_.append(w_s)
        if not ks_:
            return np.full(k_ks.size, np.nan)
        return bin_weighted(np.concatenate(ks_), np.concatenate(ps_),
                            np.concatenate(ws_), ks_edges)

    seg_dla = seg_accumulate(
        (dflux_rows(gather([g], False), scale, target_F)[0], mask_of[int(g)])
        for g in dla_trig)
    seg_ctrl = seg_accumulate(
        (dflux_rows(ctrl1[i:i + 1], scale, target_F)[0], applied1[i])
        for i in range(len(ctrl1_idx)) if applied1[i] is not None)
    del ctrl1

    # X2: control transplant of subDLA fill windows + all-absorbers-masked arm
    ctrl2 = gather(ctrl2_idx, filtered=False)
    curves["X2_ctrl_clean"] = p1d_rows(ctrl2, vmax, scale, target_F)
    tw2, _ = transplant_apply(ctrl2, donor_masks_sub, tau_eff_native, trig_frac_sub, rng)
    curves["X2_ctrl_transplant"] = p1d_rows(tw2, vmax, scale, target_F)
    del tw2, ctrl2
    if len(sub_idx):
        allm2 = gather(sub_idx, filtered=True)
        for i, g in enumerate(sub_idx):
            syst = companions_of(g, ("LLS", "subDLA"))
            if syst:
                allm2[i][build_skewer_mask(npix, syst)] = tau_eff_native
        curves["X2_sub_allmasked"] = p1d_rows(allm2, vmax, scale, target_F)
        del allm2
    else:
        curves["X2_sub_allmasked"] = nan_curve

    # X3: all-absorbers-masked arm on LLS rows
    if len(lls_idx):
        allm3 = gather(lls_idx, filtered=True)
        for i, g in enumerate(lls_idx):
            syst = companions_of(g, ("LLS",))
            if syst:
                allm3[i][build_skewer_mask(npix, syst)] = tau_eff_native
        curves["X3_lls_allmasked"] = p1d_rows(allm3, vmax, scale, target_F)
        del allm3
    else:
        curves["X3_lls_allmasked"] = nan_curve

    # at-least-one samples (round-2 MUST-FIX 3): filtered P1D over rows hosting >= 1
    # system of the class, regardless of the highest-class partition
    alo_sub = np.where(n_abs[:, 1] > 0)[0]
    alo_lls = np.where(n_abs[:, 0] > 0)[0]
    dla_with_sub = dla_idx[n_abs[dla_idx, 1] > 0]
    dla_with_lls = dla_idx[n_abs[dla_idx, 0] > 0]
    sub_with_lls = sub_idx[n_abs[sub_idx, 0] > 0]
    if alo_sub.size != len(sub_idx) + len(dla_with_sub):
        raise RuntimeError(f"at-least-one subDLA bookkeeping mismatch: {alo_sub.size} "
                           f"!= {len(sub_idx)} + {len(dla_with_sub)} (catalog class vs "
                           "NHI-partition drift)")
    if alo_lls.size != len(lls_idx) + len(sub_with_lls) + len(dla_with_lls):
        raise RuntimeError("at-least-one LLS bookkeeping mismatch (catalog class vs "
                           "NHI-partition drift)")
    parts_sub = [(len(sub_idx), curves["P_filt_sub"])]
    if len(dla_with_sub):
        parts_sub.append((len(dla_with_sub),
                          p1d_rows(gather(dla_with_sub, True), vmax, scale, target_F)))
    curves["P_alo_sub_filt"] = (atleast_one_combine(parts_sub)
                                if alo_sub.size else nan_curve)
    parts_lls = [(len(lls_idx), curves["P_filt_lls"])]
    if len(sub_with_lls):
        parts_lls.append((len(sub_with_lls),
                          p1d_rows(gather(sub_with_lls, True), vmax, scale, target_F)))
    if len(dla_with_lls):
        parts_lls.append((len(dla_with_lls),
                          p1d_rows(gather(dla_with_lls, True), vmax, scale, target_F)))
    curves["P_alo_lls_filt"] = (atleast_one_combine(parts_lls)
                                if alo_lls.size else nan_curve)

    # companion incidence table (catalog-only): mean systems/row by host label
    inc = np.zeros((5, 3))
    for hi_, sel in enumerate([lab == 0, lab == 1, lab == 2, lab == 3,
                               np.ones(n_total, bool)]):
        if sel.any():
            inc[hi_] = n_abs[sel].mean(axis=0)
    multi_dla_frac = (float((n_abs[dla_idx, 2] >= 2).mean())
                      if len(dla_idx) else float("nan"))

    # ---- assemble + write ----------------------------------------------------------
    out = {"k_native": kf, "k_ks": k_ks,
           "inc_host_class": inc, "trig_by_fine_bin": trig_fine}
    for key in CURVE_KEYS:
        cur = np.asarray(curves[key], float)
        if cur.shape != (npix // 2 + 1,):
            raise RuntimeError(f"curve {key}: shape {cur.shape} != full native spectrum")
        cur = cur[1:n_k + 1]
        out[key] = cur
        out[key + "_ks"] = ks_project(kf, cur, k_ks, z_grid)
    out["X1_seg_dla_ks"] = seg_dla
    out["X1_seg_ctrl_ks"] = seg_ctrl
    scalars = dict(
        z_meta=z_meta, z_grid=z_grid, snap=int(snap), scale=scale, target_F=target_F,
        tau_eff_native=float(tau_eff_native),
        n_clean=len(clean_idx), n_lls=n_lls, n_sub=n_sub, n_dla=n_dla,
        n_trig_clean=int(n_trig_by_class[0]), n_trig_lls=int(n_trig_by_class[1]),
        n_trig_sub=int(n_trig_by_class[2]), n_trig_dla=int(n_trig_by_class[3]),
        trig_frac_sub=float(trig_frac_sub), trig_frac_dla=float(trig_frac_dla),
        multi_dla_frac=multi_dla_frac,
        maskfrac_dla=float(masklens.sum() / max(len(dla_idx) * npix, 1)),
        masklen_med=float(np.median(masklens)),
        masklen_p90=float(np.percentile(masklens, 90)),
        maskfrac_dla_wide=float(maskfrac_wide),
        n_alo_sub=int(alo_sub.size), n_alo_lls=int(alo_lls.size),
        n_dla_with_sub=int(len(dla_with_sub)), n_dla_with_lls=int(len(dla_with_lls)),
        n_sub_with_lls=int(len(sub_with_lls)),
        n_ctrl1=int(n_ctrl1), n_ctrl2=int(n_ctrl2),
        meanF_clean=float(meanF_by_bin[0]),
        meanF_lls=_coarse_meanF(meanF_by_bin, n_by_bin, "LLS"),
        meanF_sub=_coarse_meanF(meanF_by_bin, n_by_bin, "subDLA"),
        meanF_dla=_coarse_meanF(meanF_by_bin, n_by_bin, "DLA"),
        cache_row_idx=int(cache["row"]),
        crosscheck_filt_max=float(chk_f), crosscheck_unf_max=float(chk_u),
        crosscheck_meanF_max=float(chk_mf), meanflux_target_reldiff=float(mf_rel),
    )
    for key in SCALAR_KEYS:
        v = scalars[key]
        out[key] = np.int64(v) if isinstance(v, (int, np.integer)) else np.float64(v)
    np.savez(npz_path, **out)

    stamp = dict(
        kind="xsel_truth_snap",
        sim=sim_name, snap=int(snap), z_grid=z_grid, fidelity=fidelity,
        alpha_idx=int(alpha_idx),
        convention=build_convention(fidelity, alpha_idx, cache["alpha_slope"], seed),
        git_describe=git_describe(),
        cache_file=str(_CACHE[fidelity]), cache_row=int(cache["row"]),
        cache_git_sha=cache["cache_git_sha"],
        catalog_sha256=sha256_file(Path(snap_dir) / "catalog.npz"),
        tau_file=str(raw_path),
        tau_sha256=(sha256_file(raw_path) if (tau_sha and full) else "SKIPPED"),
        n_skewers=("ALL" if full else int(n_skewers)),
        full_suite=bool(full),
        crosschecks=dict(filt=float(chk_f), unf=float(chk_u), meanF=float(chk_mf),
                         meanflux_target_reldiff=float(mf_rel), rtol=CACHE_RTOL),
        wall_s=round(time.time() - t0, 1),
    )
    side = stamp_sidecar(npz_path, stamp)
    print(f"[{sim_name[:14]} z{z_grid:.1f}] wrote {npz_path} "
          f"(+{os.path.basename(side)}) wall={time.time()-t0:.0f}s", flush=True)
    return str(npz_path)


# ===========================================================================
# Discovery / driver
# ===========================================================================
def discover(fidelity, hcd_root=None, emu_root=None):
    """(sim, snap, snap_dir, raw_path) list via the DEPLOYED cache builder's discovery
    (same dedup, same off-grid skip, same raw-dir override), plus the sorted sim list."""
    import build_emulator_cache_tau0 as bct
    hcd_root = Path(hcd_root or _HCD_OUTPUTS)
    emu_root = Path(emu_root or (_HR_EMU_ROOT if fidelity == "hr" else _LF_EMU_ROOT))
    pairs = bct.discover_tau0_pairs(hcd_root, emu_root, fidelity=fidelity)
    sims = sorted({p[0] for p in pairs})
    return pairs, sims


def run_sims(args):
    pairs, sims = discover(args.fidelity, args.hcd_root, args.emu_root)
    if args.list:
        for i, s in enumerate(sims):
            n = sum(1 for p in pairs if p[0] == s)
            print(f"{i:3d} {s} ({n} snaps)")
        print(f"total {len(sims)} sims, {len(pairs)} (sim,z) pairs [{args.fidelity}]")
        return
    if args.sim is not None:
        target = [s for s in sims if s == args.sim or s.startswith(args.sim)]
        if len(target) != 1:
            raise SystemExit(f"--sim {args.sim!r} matches {len(target)} sims")
        sim = target[0]
    else:
        if not (0 <= args.sim_index < len(sims)):
            raise SystemExit(f"--sim-index {args.sim_index} out of range 0..{len(sims)-1}")
        sim = sims[args.sim_index]
    todo = [p for p in pairs if p[0] == sim]
    if args.snaps:
        want = {int(s) for s in args.snaps.split(",")}
        todo = [p for p in todo if p[1] in want]
        if not todo:
            raise SystemExit(f"--snaps {args.snaps} matches nothing for {sim}")
    print(f"[{args.fidelity}] {sim}: {len(todo)} snaps", flush=True)
    for _, snap, snap_dir, raw in todo:
        process_snap(sim, snap, snap_dir, raw, args.fidelity,
                     alpha_idx=args.alpha_idx, outdir=args.outdir, seed=args.seed,
                     batch_rows=args.batch_rows, n_skewers=args.n_skewers,
                     tau_sha=not args.no_tau_sha, ks_base=args.ks_base,
                     skip_existing=args.skip_existing)


# ===========================================================================
# Pooling + figures
# ===========================================================================
def pool_suite(outdir, fidelity):
    """Stack all per-(sim,z) npzs of one suite into a table npz. Refuses mixed
    conventions, unverifiable sidecars, or subset (non-full-suite) members."""
    root = Path(outdir) / fidelity
    files = sorted(root.glob("*/xsel_z*.npz"))
    if not files:
        raise RuntimeError(f"pool_suite: no member npzs under {root}")
    ref_conv = None
    sims, shas = [], []
    arrays = {}
    keys = (("k_native", "k_ks", "inc_host_class", "trig_by_fine_bin")
            + CURVE_KEYS + tuple(k + "_ks" for k in CURVE_KEYS)
            + SEG_KEYS + SCALAR_KEYS)
    for f in files:
        stamp = verify_sidecar(f)
        if not stamp.get("full_suite", False):
            raise RuntimeError(f"{f}: smoke/subset output (full_suite=False) in the pool "
                               "-- delete it or rerun full")
        conv = dict(stamp["convention"])
        if ref_conv is None:
            ref_conv = conv
        elif conv != ref_conv:
            raise RuntimeError(f"{f}: convention flags differ from the pool reference")
        d = np.load(f)
        sims.append(stamp["sim"])
        shas.append(stamp["npz_sha256"])
        for key in keys:
            if key not in d:
                raise RuntimeError(f"{f}: missing key {key}")
            arrays.setdefault(key, []).append(np.asarray(d[key]))
    out = {k: np.stack(v) for k, v in arrays.items()}
    out["sim"] = np.array(sims)
    out_path = Path(outdir) / f"xsel_truth_{fidelity}.npz"
    np.savez(out_path, **out)
    stamp = dict(kind="xsel_truth_suite", fidelity=fidelity, n_members=len(files),
                 members=[str(f) for f in files], member_sha256=shas,
                 convention=ref_conv, git_describe=git_describe())
    stamp_sidecar(out_path, stamp)
    print(f"[pool] {out_path}: {len(files)} members, sims={len(set(sims))}", flush=True)
    return out_path


def pool_all(outdir, fig_dir=None, suites=("lf", "hr")):
    suite_paths, pooled, member_info = {}, {}, {}
    for fid in suites:
        p = pool_suite(outdir, fid)
        suite_paths[fid] = p
        st = verify_sidecar(p)
        member_info[fid] = dict(n_members=st["n_members"], sha256=st["npz_sha256"],
                                convention=st["convention"])
        d = np.load(p)
        for k in d.files:
            pooled[f"{fid}_{k}"] = d[k]
    out_path = Path(outdir) / "xsel_truth_pooled.npz"
    np.savez(out_path, **pooled)
    stamp = dict(kind="xsel_truth_pooled", suites=member_info,
                 suite_files={k: str(v) for k, v in suite_paths.items()},
                 git_describe=git_describe())
    stamp_sidecar(out_path, stamp)
    print(f"[pool] {out_path} written")
    if fig_dir:
        for fid in suites:
            make_figures(suite_paths[fid], fid, fig_dir)
    return out_path


# ---------------------------------------------------------------------------
# Figures (Okabe-Ito fixed arm identity; viridis sequential for z)
# ---------------------------------------------------------------------------
C_TOTAL, C_CLUST, C_COMP = "#0072B2", "#009E73", "#E69F00"
C_MASK, C_UNM, C_WIDE, C_CORR = "#999999", "#CC79A7", "#56B4E9", "#D55E00"


def _zmean(d, key, zvals, z):
    """Mean curve over sims at one z (rows where z_grid == z), NaN-safe."""
    m = np.isclose(zvals, z)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.asarray(d[key], float)[m], axis=0)


def _ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.asarray(b) != 0, np.asarray(a, float) / np.asarray(b, float),
                        np.nan)


def make_figures(suite_npz, fidelity, fig_dir):
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    warnings.filterwarnings("ignore", message="Mean of empty slice")
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    d = np.load(suite_npz)
    zvals = np.asarray(d["z_grid"], float)
    zs = np.unique(np.round(zvals, 1))
    kf = np.asarray(d["k_native"], float)[0]
    norm = matplotlib.colors.Normalize(vmin=zs.min(), vmax=zs.max())
    ks_lo, ks_hi = float(d["k_ks"][0][0]), float(d["k_ks"][0][-1])
    zrep = [z for z in (2.6, 3.4, 4.2) if np.isclose(zs, z).any()] or [zs[0]]

    def zcurves(ax, num_key, title):
        for z in zs:
            r = _ratio(_zmean(d, num_key, zvals, z), _zmean(d, "P_filt_clean", zvals, z))
            ax.plot(kf, r, color=cm.viridis(norm(z)), lw=1.1)
        ax.axvspan(ks_lo, ks_hi, color="#0072B2", alpha=0.06, lw=0)
        ax.axhline(1.0, color="#cccccc", lw=0.8)
        ax.set_xscale("log"); ax.set_xlabel("k [rad s/km, angular]")
        ax.set_ylabel(r"$P_X/P_{\rm clean}$"); ax.set_title(title, fontsize=10)

    # X1 truth: diluted + corrected panels
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    zcurves(axes[0], "P_filt_dla", "X1 diluted (deployed trough-fill; X1b overlay)")
    zcurves(axes[1], "X1_corrected", "X1 dilution-CORRECTED (primary fork)")
    for z in zrep:
        rw = _ratio(_zmean(d, "X1_corrected_wide", zvals, z),
                    _zmean(d, "P_filt_clean", zvals, z))
        axes[1].plot(kf, rw, color=C_WIDE, lw=1.0, ls=":")
    axes[1].plot([], [], color=C_WIDE, ls=":", label="2x-wide mask (band, rep. z)")
    axes[1].legend(fontsize=8, frameon=False)
    sm = cm.ScalarMappable(norm=norm, cmap="viridis"); sm.set_array([])
    fig.colorbar(sm, ax=axes, label="z", fraction=0.03)
    fig.suptitle(f"X1_dla100 truth, {fidelity.upper()} suite (deployed convention; "
                 "shaded = KS band)", fontsize=11)
    fig.savefig(Path(fig_dir) / f"xsel_{fidelity}_X1_truth.png", bbox_inches="tight")
    plt.close(fig)

    # X1 decomposition at representative z (both dilution estimators)
    fig, axes = plt.subplots(1, len(zrep), figsize=(4.2 * len(zrep), 4.0), dpi=150,
                             squeeze=False)
    for ax, z in zip(axes[0], zrep):
        Pc = _zmean(d, "P_filt_clean", zvals, z)
        ax.plot(kf, _ratio(_zmean(d, "P_filt_dla", zvals, z), Pc), color=C_TOTAL,
                lw=2.0, label="total (diluted)")
        ax.plot(kf, _ratio(_zmean(d, "P_unf_dla", zvals, z), Pc), color=C_UNM,
                lw=1.2, ls=":", label="unmasked (context)")
        ax.plot(kf, _ratio(_zmean(d, "X1_dla_allmasked", zvals, z), Pc), color=C_CLUST,
                lw=1.5, label="all-absorbers-masked (clustering)")
        ax.plot(kf, _ratio(_zmean(d, "X1_ctrl_transplant", zvals, z),
                           _zmean(d, "X1_ctrl_clean", zvals, z)), color=C_MASK,
                lw=1.5, label="dilution control (transplant)")
        ax.plot(kf, _ratio(_zmean(d, "X1_corrected", zvals, z), Pc), color=C_CORR,
                lw=2.0, ls="--", label="dilution-corrected (primary)")
        kks = np.asarray(d["k_ks"], float)[0]
        m = np.isclose(zvals, z)
        seg = np.nanmean(_ratio(np.asarray(d["X1_seg_dla_ks"], float)[m],
                                np.asarray(d["X1_seg_ctrl_ks"], float)[m]), axis=0)
        ax.plot(kks, seg, "o", color=C_CORR, ms=4, mfc="none",
                label="pixel-exclusion estimator")
        ax.axvspan(ks_lo, ks_hi, color="#0072B2", alpha=0.06, lw=0)
        ax.axhline(1.0, color="#cccccc", lw=0.8)
        ax.set_xscale("log"); ax.set_title(f"z = {z:.1f}", fontsize=10)
        ax.set_xlabel("k [rad s/km]")
        ax.set_ylim(0.0, 2.0)   # the unmasked-context curve blows up below the KS band
    axes[0][0].set_ylabel("ratio"); axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"X1 dilution decomposition, {fidelity.upper()} (both estimators)",
                 fontsize=11)
    fig.savefig(Path(fig_dir) / f"xsel_{fidelity}_X1_decomposition.png",
                bbox_inches="tight")
    plt.close(fig)

    # X2 truth + trough-fill decomposition (round-2: fill is NOT DLA-only)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), dpi=150)
    zcurves(axes[0], "P_filt_sub", "X2 truth: subDLA filled (deployed)")
    z0 = zrep[len(zrep) // 2]
    Pc = _zmean(d, "P_filt_clean", zvals, z0)
    axes[1].plot(kf, _ratio(_zmean(d, "P_filt_sub", zvals, z0), Pc), color=C_TOTAL,
                 lw=2.0, label="filled (deployed truth)")
    axes[1].plot(kf, _ratio(_zmean(d, "P_unf_sub", zvals, z0), Pc), color=C_UNM,
                 lw=1.5, label="unfilled (wings+cores)")
    axes[1].plot(kf, _ratio(_zmean(d, "X2_ctrl_transplant", zvals, z0),
                            _zmean(d, "X2_ctrl_clean", zvals, z0)), color=C_MASK,
                 lw=1.5, label="mask/dilution part (transplant)")
    axes[1].plot(kf, _ratio(_zmean(d, "X2_sub_allmasked", zvals, z0), Pc),
                 color=C_CLUST, lw=1.5, label="all-masked (clustering)")
    axes[1].axhline(1.0, color="#cccccc", lw=0.8)
    axes[1].axvspan(ks_lo, ks_hi, color="#0072B2", alpha=0.06, lw=0)
    axes[1].set_xscale("log")
    axes[1].set_title(f"X2 decomposition, z={z0:.1f}", fontsize=10)
    axes[1].legend(fontsize=7, frameon=False); axes[1].set_xlabel("k [rad s/km]")
    lowk_loss, trigf = [], []
    j = int(np.argmin(np.abs(kf - ks_lo)))
    for z in zs:
        fl = _zmean(d, "P_filt_sub", zvals, z); un = _zmean(d, "P_unf_sub", zvals, z)
        lowk_loss.append(1.0 - fl[j] / un[j] if un[j] else np.nan)
        trigf.append(float(np.nanmean(
            np.asarray(d["trig_frac_sub"], float)[np.isclose(zvals, z)])))
    axes[2].plot(zs, lowk_loss, "o-", color=C_TOTAL, label="low-k power lost to fill")
    axes[2].plot(zs, trigf, "s--", color=C_COMP, label="subDLA rows altered (trig frac)")
    axes[2].set_xlabel("z"); axes[2].set_ylabel("fraction")
    axes[2].set_title("sub-DLA trough-fill effect (OQ9 inputs)", fontsize=10)
    axes[2].legend(fontsize=8, frameon=False)
    fig.suptitle(f"X2_sub100, {fidelity.upper()} (round-2: fill is NOT DLA-only)",
                 fontsize=11)
    fig.savefig(Path(fig_dir) / f"xsel_{fidelity}_X2_truth_fill.png", bbox_inches="tight")
    plt.close(fig)

    # X3 truth + absorber part
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=150)
    zcurves(axes[0], "P_filt_lls", "X3 truth: LLS-selected (deployed)")
    from matplotlib import cm as _cm
    for z in zrep:
        axes[1].plot(kf, _ratio(_zmean(d, "P_filt_lls", zvals, z),
                                _zmean(d, "X3_lls_allmasked", zvals, z)),
                     color=_cm.viridis(norm(z)), lw=1.4, label=f"z={z:.1f}")
    axes[1].axhline(1.0, color="#cccccc", lw=0.8)
    axes[1].axvspan(ks_lo, ks_hi, color="#0072B2", alpha=0.06, lw=0)
    axes[1].set_xscale("log"); axes[1].legend(fontsize=8, frameon=False)
    axes[1].set_title("X3 absorber part (selected / all-masked)", fontsize=10)
    axes[1].set_xlabel("k [rad s/km]")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes[0], label="z",
                 fraction=0.04)
    fig.savefig(Path(fig_dir) / f"xsel_{fidelity}_X3_truth.png", bbox_inches="tight")
    plt.close(fig)

    # at-least-one vs partition (X2/X3 semantic band, round-2 OQ10)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=150)
    for ax, alo, part, lbl in ((axes[0], "P_alo_sub_filt", "P_filt_sub", "subDLA"),
                               (axes[1], "P_alo_lls_filt", "P_filt_lls", "LLS")):
        for z in zs:
            r = _ratio(_zmean(d, alo, zvals, z), _zmean(d, part, zvals, z)) - 1.0
            ax.plot(kf, 100 * r, color=cm.viridis(norm(z)), lw=1.1)
        ax.axhline(0.0, color="#cccccc", lw=0.8)
        ax.axvspan(ks_lo, ks_hi, color="#0072B2", alpha=0.06, lw=0)
        ax.set_xscale("log"); ax.set_xlabel("k [rad s/km]")
        ax.set_title(f"at-least-one vs partition, {lbl} [%]", fontsize=10)
    fig.suptitle(f"Selection-semantics difference (round-2 OQ10), {fidelity.upper()}",
                 fontsize=11)
    fig.savefig(Path(fig_dir) / f"xsel_{fidelity}_alo_vs_partition.png",
                bbox_inches="tight")
    plt.close(fig)
    print(f"[figs] wrote 5 {fidelity} figures to {fig_dir}", flush=True)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fidelity", choices=("lf", "hr"), default="lf")
    ap.add_argument("--sim-index", type=int, default=None)
    ap.add_argument("--sim", type=str, default=None)
    ap.add_argument("--snaps", type=str, default=None,
                    help="comma-separated snap indices (default: all for the sim)")
    ap.add_argument("--outdir", default=os.environ.get("OUTDIR", OUTDIR_DEFAULT))
    ap.add_argument("--fig-dir", default=None)
    ap.add_argument("--alpha-idx", type=int, default=ALPHA_IDX_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--batch-rows", type=int, default=4096)
    ap.add_argument("--n-skewers", type=int, default=None,
                    help="PLUMBING SMOKE ONLY: subset rows; cache anchor skipped, "
                         "sidecar stamped full_suite=False (pool refuses it)")
    ap.add_argument("--no-tau-sha", action="store_true",
                    help="skip the full tau-file sha256 (saves one IO pass)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip (sim,z) whose npz exists AND sidecar-verifies")
    ap.add_argument("--hcd-root", default=None)
    ap.add_argument("--emu-root", default=None)
    ap.add_argument("--ks-base", default=KS_BASE_DEFAULT)
    ap.add_argument("--list", action="store_true", help="list sims + indices, exit")
    ap.add_argument("--pool", action="store_true",
                    help="pool per-(sim,z) npzs into suite + pooled tables (+figures)")
    ap.add_argument("--suites", default="lf,hr", help="suites to pool (with --pool)")
    args = ap.parse_args()

    if args.pool:
        pool_all(args.outdir, fig_dir=args.fig_dir, suites=tuple(args.suites.split(",")))
        return
    if not args.list and args.sim is None and args.sim_index is None:
        ap.error("need --sim-index/--sim (or --list / --pool)")
    run_sims(args)


if __name__ == "__main__":
    main()
