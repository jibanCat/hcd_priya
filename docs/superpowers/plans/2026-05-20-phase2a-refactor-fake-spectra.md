# Phase-2a refactor — drive fake_spectra directly, two-tier τ₀ cache

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Supersedes** `docs/superpowers/plans/2026-05-17-phase2-tau0-cache.md`. That
plan's `compute_p1d_per_class` + `tau_transform` pipeline produced P1D off by
~7 % from PRIYA's published training data at the same `(sim, z, α)` (see
`docs/superpowers/2026-05-20-priya-p1d-consistency-check.md`). This plan
replaces it with a thin wrapper around fake_spectra's actual machinery, which
matches PRIYA to floating-point precision (median ratio 1.00000015,
max\|r−1\| 1.2 × 10⁻⁶).

The Phase-2a code already on `phase2-emulator-jax` (PR #10) WILL be modified.

---

## Goal

Build a τ₀-extended HCD-emulator training cache that:

* **Tier P** (PRIYA-compatible total P1D): bit-identical to
  `mf_emulator_flux_vectors_tau1000000.hdf5` row-by-row for every
  matching `(sim, z, α_slope)` triple. This is the survey-equivalent training
  target for the *main forest emulator* — the survey P1D measurement
  already includes residual DLAs after their imperfect masking.
* **Tier C** (per-class P1D, HCD adds-on): four separately-stored P1Ds —
  P_clean / P_LLS / P_subDLA / P_DLA — built on the *same* sightlines, the
  *same* `target_F` and `scale` as Tier P, but partitioned by the absorber
  catalog instead of filtered. Used to build the HCD adds-on emulator
  module. Storing each class separately (not as ratios) so the user can also
  test the "P_clean + HCD-adds-on" inference path against the
  "PRIYA Tier-P + HCD-adds-on" path. Future tier C′ (Rahmati partial
  self-shielding) is a swap-out, not a rewrite.

Both tiers share: fake_spectra's `_filter_single_tau_complex` (Tier P only),
`_rescale_mean_flux`, `fluxstatistics.flux_power` / `_powerspectrum` /
`_flux_power_bins`, and `obs_mean_tau` (Kim 2013 slope-α convention).

---

## Tech stack

- **Python 3.9** with `fake_spectra` (in conda env `emu-3.9`, needs GSL loaded).
- `numpy`, `h5py`.
- Reuses Phase-1 `scripts/build_emulator_cache.py` for sim discovery,
  `parse_sim_params`, `PARAM_ORDER`, `compute_dndx_per_class`,
  `interp_p1d_loglog`, `_git_sha`.
- Reuses Phase-1 `hcd_analysis/catalog.py` `AbsorberCatalog.load_npz` for
  per-class sightline labels (drives Tier C).
- Tests are plain assert-scripts run with `python3 tests/test_*.py`.

**Critical environment caveat.** The fake_spectra import requires both
`emu-3.9` and `gcc/10.3.0 + gsl/2.7` modules (the C extension links libgsl).
Every test that touches fake_spectra must run in that environment. We
introduce a `pytest_skip_no_fake_spectra` helper and explicit env-load
documentation in each touching test file so a stray `python3 tests/test_*.py`
under the wrong env gives a clean skip instead of an `ImportError`.

---

## Commit convention

every commit message ends with the trailer
`Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.

---

## File structure

**Modified:**
- `scripts/build_emulator_cache_tau0.py` — replaces all Phase-1
  `compute_p1d_per_class` calls; pivots to fake_spectra. Cache schema gains
  Tier-P datasets alongside the existing Tier-C per-class arrays.
- `hcd_analysis/p1d.py` — keep `compute_p1d_per_class`'s `tau_transform`
  hook untouched (Phase-1 callers still use it). Do NOT call it from Phase-2.
- `hcd_analysis/tau0_rescale.py` — keep `make_alpha_grid`, mark
  `freeze_core_rescale` and `tau0_from_mean_flux` as legacy with deprecation
  notes; not used by the new builder but retained for backward-compat with
  Phase-1 tests.

**New:**
- `hcd_analysis/priya_p1d.py` — thin wrapper around fake_spectra. Two public
  functions: `compute_tier_p_p1d(tau, vmax, alpha_slope, z)` and
  `compute_tier_c_p1d(tau, vmax, alpha_slope, z, catalog)`. Internally calls
  `_filter_single_tau_complex` (Tier P) or class-slicing (Tier C). Returns
  `(kf_kms, P1D_or_dict, target_F, scale)`.
- `tests/test_priya_p1d.py` — bit-identity regression test against PRIYA's
  flux_vectors HDF5 (the v3 test, formalised + parametrised over multiple
  sims/z/α).
- `scripts/consistency_checks/priya_p1d_consistency_v3.py` already shipped
  this session; tests reuse the data path.

**Cache schema bumped** (new top-level attr `cache_version = "2.0"`; old
caches built by the 2026-05-17 plan are no longer schema-compatible).
Existing `observables_tau0.h5` files in `_emulator_data/` should be deleted
before running the new builder (they will mismatch the loader's schema check).

---

## Task 0 — Confirm v3 multi-point verdict before refactoring ✅ DONE 2026-05-20

Reads `docs/superpowers/figs/multipoint/sim{idx}_snap{N}.npz` (12 files from
sbatch array `50565913` + `50566111`, aggregated via
`scripts/consistency_checks/aggregate_multipoint.py`).

- [x] **Step 1:** Multi-point verification ran via sbatch array (3 sims × 4 z
  × 10 α = 120 points). All passed: worst `max|r-1| = 1.89e-5`, worst std =
  `1.58e-6`. See consistency-check doc §6b for full table and plot.
- [x] **Step 2:** Verdict documented in
  `docs/superpowers/2026-05-20-priya-p1d-consistency-check.md` §6b.
  **Refactor is cleared to proceed.**

---

## Task 1 — `hcd_analysis/priya_p1d.py`: Tier P helper

**Files:**
- Create: `hcd_analysis/priya_p1d.py`
- Create: `tests/test_priya_p1d.py`

- [ ] **Step 1: Write the failing test.**

```python
"""Bit-identity test against PRIYA's flux_vectors HDF5.

Runs in emu-3.9 + GSL env:
    module load gcc/10.3.0 gsl/2.7 && conda activate emu-3.9
    python3 tests/test_priya_p1d.py
"""
import os, sys, json
from pathlib import Path
import numpy as np
import h5py

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

try:
    from hcd_analysis.priya_p1d import compute_tier_p_p1d
    from fake_spectra.fluxstatistics import obs_mean_tau
    _has_fs = True
except ImportError as e:
    print(f"SKIP — fake_spectra unavailable ({e}); requires emu-3.9 env with gsl")
    sys.exit(0)


PRIYA_FILE = "/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/mf_emulator_flux_vectors_tau1000000.hdf5"
SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
SNAP = 17
PRIYA_ROW = 344
PRIYA_Z_IDX = 8
N_K = 172


def test_tier_p_bit_identical_to_priya_at_sim0_z3_alpha1():
    tau_p = f"/nfs/turbo/umor-yueyingn/mfho/emu_full/{SIM}/output/SPECTRA_{SNAP:03d}/lya_forest_spectra_grid_480.hdf5"
    meta_p = f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/meta.json"
    with open(meta_p) as f: m = json.load(f)
    nbins, dv_kms, z = int(m["nbins"]), float(m["dv_kms"]), float(m["z"])
    vmax = nbins * dv_kms

    with h5py.File(tau_p, "r") as f:
        tau = f["tau/H/1/1215"][...].astype(np.float64)
    with h5py.File(PRIYA_FILE, "r") as f:
        alpha = float(f["params"][PRIYA_ROW, 0])
        P_priya = f["flux_vectors"][PRIYA_ROW, PRIYA_Z_IDX*N_K:(PRIYA_Z_IDX+1)*N_K].astype(np.float64)
        kp = f["kfkms"][PRIYA_ROW, PRIYA_Z_IDX, :].astype(np.float64)

    kf, P_mine, target_F, scale = compute_tier_p_p1d(tau, vmax, alpha_slope=alpha, z=z)
    # k-grid first
    assert np.max(np.abs(kf[:N_K] - kp) / kp) < 1e-12, "k-grid mismatch"
    # P1D bit-identical
    ratio = P_mine[:N_K] / P_priya
    assert np.max(np.abs(ratio - 1)) < 1e-4, \
        f"P1D mismatch: max|r-1| = {np.max(np.abs(ratio-1)):.3e}"
    # Sanity: scale and target_F roughly match expectations
    assert 0.6 < target_F < 0.8
    assert 0.85 < scale < 1.0
    print(f"OK  α={alpha:.4f}  z={z}  target_F={target_F:.4f}  scale={scale:.4f}  "
          f"med(r)={np.median(ratio):.7f}  max|r-1|={np.max(np.abs(ratio-1)):.3e}")


if __name__ == "__main__":
    test_tier_p_bit_identical_to_priya_at_sim0_z3_alpha1()
    print("OK")
```

- [ ] **Step 2: Run the test to verify it fails** with
  `ModuleNotFoundError: hcd_analysis.priya_p1d`.

- [ ] **Step 3: Write the module.**

```python
"""Thin wrapper around fake_spectra's P1D pipeline, used by the Phase-2
emulator cache builder. The functions in this module are intentionally
shallow — every P1D-relevant choice (filter, mean-flux inversion, FFT) is
delegated to fake_spectra so the cache rows are bit-identical to PRIYA's
flux_vector training data at matching (sim, z, alpha_slope).

Requires the emu-3.9 conda env with gsl loaded (libgsl.so.25 is dlopen'd by
fake_spectra._spectra_priv).
"""
from __future__ import annotations
from types import SimpleNamespace
from typing import Tuple

import numpy as np

from fake_spectra.fluxstatistics import (
    flux_power, obs_mean_tau, _powerspectrum, _flux_power_bins,
)
from fake_spectra.spectra import Spectra
# Unbound method, reads only `self.nbins`. Calling via a SimpleNamespace
# stand-in avoids constructing a full Spectra (which needs a snapshot dir).
_filter_single_tau_complex = Spectra._filter_single_tau_complex


def _apply_priya_filter(tau: np.ndarray, tau_thresh: float = 1.0e6,
                        thresh2: float = 0.25) -> np.ndarray:
    """In-place: apply fake_spectra._filter_single_tau_complex to every
    sightline with `max(tau) > tau_thresh`. Returns the same array."""
    tau_eff = -np.log(np.mean(np.exp(-tau)))
    ii = np.where(np.max(tau, axis=1) > tau_thresh)[0]
    if len(ii) == 0:
        return tau
    self_stub = SimpleNamespace(nbins=tau.shape[1])
    for i in ii:
        tau[i], _ = _filter_single_tau_complex(self_stub, tau[i], tau_eff,
                                               tau_thresh=tau_thresh, thresh2=thresh2)
    return tau


def compute_tier_p_p1d(tau: np.ndarray, vmax: float,
                       alpha_slope: float, z: float,
                       tau_thresh: float = 1.0e6,
                       ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """PRIYA-compatible total P1D over all sightlines, after filter.

    Pipeline:
      1. mean_flux_desired = exp(-alpha_slope * obs_mean_tau_Kim2013(z))
      2. Apply _filter_single_tau_complex to tau (modifies in place).
      3. fake_spectra.fluxstatistics.flux_power(tau_filtered, vmax,
         mean_flux_desired=target_F, window=False, spec_res=0.0).
      4. Drop the k=0 mode (PRIYA convention).

    Returns (kf_kms, P1D, target_F, scale).
    """
    target_F = float(np.exp(-alpha_slope * obs_mean_tau(z)))
    _apply_priya_filter(tau, tau_thresh=tau_thresh)
    kf, P = flux_power(tau, vmax, spec_res=0.0,
                       mean_flux_desired=target_F, window=False)
    # scale (the actual tau-multiplier) is solved inside flux_power; we can
    # recover it by calling mean_flux explicitly (cheap — Newton iteration on
    # the same tau).
    from fake_spectra.fluxstatistics import mean_flux
    scale = float(mean_flux(tau, target_F))
    return kf[1:], P[1:], target_F, scale
```

- [ ] **Step 4: Run the test to verify it passes.**
  `module load gcc/10.3.0 gsl/2.7 && conda activate emu-3.9 && python3 tests/test_priya_p1d.py`
  Expected: `OK  α=1.0112  z=3.0  target_F=0.6932  scale=0.9385  med(r)≈1.0  max|r-1|≈1e-6`.

- [ ] **Step 5: Commit.**

```bash
git add hcd_analysis/priya_p1d.py tests/test_priya_p1d.py
git commit -m "$(cat <<'EOF'
feat: hcd_analysis.priya_p1d — fake_spectra wrapper for Tier P P1D

Bit-identical to PRIYA's mf_emulator_flux_vectors_tau1000000.hdf5 at
floating-point precision (verified at sim 0, snap_017, alpha=1.011).
Drops the rolled-own _filter_single_tau_complex port from 2026-05-17
plan; calls fake_spectra's unbound method via SimpleNamespace stub.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — Per-class P1D primitive (`compute_tier_c_p1d`)

**Files:**
- Modify: `hcd_analysis/priya_p1d.py` (append `compute_tier_c_p1d`)
- Modify: `tests/test_priya_p1d.py` (append per-class test)

The Tier-C P1D shares Tier P's `scale` (computed on the **filtered** tau —
matching PRIYA's mean-flux convention) but slices the **unfiltered** tau by
sightline class. δF normalisation uses the same `target_F`, so each class's
P1D is on the same observable axis as Tier P (literally: the Tier P P1D is
the sightline-weighted average of the four Tier-C P1Ds when no filter is
applied; the filter then shifts low-class to clean, but the comparison is
well-defined).

### Sightline classification (uses Phase-1 `AbsorberCatalog`)

For each sightline, the "highest class" is assigned from
`AbsorberCatalog.load_npz`:

- **DLA** if any absorber on this sightline has `log N_HI >= 20.3`
- else **subDLA** if any `log N_HI >= 19.0`
- else **LLS** if any `log N_HI >= 17.2`
- else **clean**

This matches Phase-1's existing classifier in `compute_p1d_per_class` (we'll
import the labelling routine, not the P1D path).

- [ ] **Step 1: Write the failing test.**

Append to `tests/test_priya_p1d.py`:

```python
def test_tier_c_matches_filter_free_sum_at_alpha_one():
    """When tau_thresh=inf (no filter), the sightline-weighted sum of the four
    Tier-C P1Ds must equal the Tier-P-without-filter P1D — to floating-point.
    This guards against bugs in the per-class accumulator."""
    from hcd_analysis.priya_p1d import compute_tier_c_p1d, compute_tier_p_p1d
    from hcd_analysis.catalog import AbsorberCatalog

    tau_p = f"/nfs/turbo/umor-yueyingn/mfho/emu_full/{SIM}/output/SPECTRA_{SNAP:03d}/lya_forest_spectra_grid_480.hdf5"
    meta_p = f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/meta.json"
    cat_p = f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/catalog.npz"
    with open(meta_p) as f: m = json.load(f)
    vmax = int(m["nbins"]) * float(m["dv_kms"])
    z = float(m["z"])
    with h5py.File(tau_p, "r") as f:
        tau = f["tau/H/1/1215"][...].astype(np.float64)
    catalog = AbsorberCatalog.load_npz(cat_p)

    alpha = 1.0
    # Tier P with filter disabled => total P1D over all sightlines, no DLA mask
    kf_P, P_P, target_F, scale = compute_tier_p_p1d(
        tau.copy(), vmax, alpha_slope=alpha, z=z, tau_thresh=np.inf)
    # Tier C: per-class P1Ds with the same scale + target_F
    kf_C, by_class, n_by_class, _, _ = compute_tier_c_p1d(
        tau, vmax, alpha_slope=alpha, z=z, catalog=catalog,
        external_scale=scale, external_target_F=target_F)
    # weights = n_class / n_total
    n_total = sum(n_by_class.values())
    P_recombined = sum((n_by_class[c]/n_total) * by_class[c] for c in by_class)
    assert np.allclose(P_recombined, P_P, rtol=1e-10), \
        f"per-class sum != total P1D at α=1 (no filter), worst rel diff "\
        f"= {np.max(np.abs(P_recombined/P_P - 1)):.3e}"
    print("OK — Tier C sums to Tier P at α=1 with no filter")
```

- [ ] **Step 2: Run; expect** `ImportError: cannot import name 'compute_tier_c_p1d'`.

- [ ] **Step 3: Add `compute_tier_c_p1d` and helpers.**

Append to `hcd_analysis/priya_p1d.py`:

```python
from typing import Dict, Optional


def _classify_sightlines(catalog, n_skewers: int) -> Dict[str, np.ndarray]:
    """Return a dict of boolean masks, one per class label.

    Mirrors the highest-class promotion rule already used by
    hcd_analysis.p1d.compute_p1d_per_class (DLA wins over subDLA wins over
    LLS; everything else is "clean"). Uses each Absorber's pre-computed
    `absorber_class` field, NOT a re-thresholded log_NHI — the absorber's
    class is the authoritative label (Phase-1 catalog builder applied the
    thresholds at fit time).
    """
    labels = np.full(n_skewers, "clean", dtype=object)
    for ab in catalog.absorbers:
        if ab.skewer_idx >= n_skewers:
            continue
        if ab.absorber_class == "DLA":
            labels[ab.skewer_idx] = "DLA"
        elif ab.absorber_class == "subDLA":
            if labels[ab.skewer_idx] != "DLA":
                labels[ab.skewer_idx] = "subDLA"
        elif ab.absorber_class == "LLS":
            if labels[ab.skewer_idx] not in ("DLA", "subDLA"):
                labels[ab.skewer_idx] = "LLS"
    return {c: (labels == c) for c in ("clean", "LLS", "subDLA", "DLA")}


def _per_class_p1d_at_scale(tau_class: np.ndarray, vmax: float,
                             scale: float, target_F: float):
    """Mirror fake_spectra.fluxstatistics.flux_power but with predetermined
    (scale, target_F) — used by Tier C so all four classes share Tier P's
    mean-flux normalisation. Returns (kf[1:], P[1:]) like flux_power does."""
    nspec, npix = tau_class.shape
    if nspec == 0:
        kf = _flux_power_bins(vmax, npix)
        return kf[1:], np.zeros(npix//2 + 1)[1:]
    mfp = np.zeros(npix // 2 + 1, dtype=tau_class.dtype)
    for i in range(10):
        end = min((i + 1) * nspec // 10, nspec)
        s = i * nspec // 10
        if end == s:
            continue
        dflux = np.exp(-scale * tau_class[s:end]) / target_F - 1.0
        mfp += vmax * np.sum(_powerspectrum(dflux, axis=1), axis=0)
    mfp /= nspec
    kf = _flux_power_bins(vmax, npix)
    return kf[1:], mfp[1:]


def compute_tier_c_p1d(tau: np.ndarray, vmax: float,
                       alpha_slope: float, z: float,
                       catalog,
                       external_scale: Optional[float] = None,
                       external_target_F: Optional[float] = None,
                       ) -> "Tuple[np.ndarray, Dict[str,np.ndarray], Dict[str,int], float, float]":
    """Per-class P1Ds (clean, LLS, subDLA, DLA) on the UNFILTERED tau.

    If `external_scale` / `external_target_F` are given (the production path,
    set from a prior Tier-P call), use those so the four classes share Tier
    P's mean-flux normalisation. Otherwise compute target_F from
    obs_mean_tau_Kim2013(z) and solve scale on the full tau (independent
    Tier-C-only path, useful for unit tests).

    Returns:
        kf      : k-grid (s/km, angular), shape (npix//2,)
        by_class: dict[label] -> P1D array, shape (npix//2,)
        n_by_class: dict[label] -> sightline count
        target_F, scale (the values used)
    """
    from fake_spectra.fluxstatistics import mean_flux

    if external_target_F is None:
        target_F = float(np.exp(-alpha_slope * obs_mean_tau(z)))
    else:
        target_F = float(external_target_F)
    if external_scale is None:
        scale = float(mean_flux(tau, target_F))
    else:
        scale = float(external_scale)

    masks = _classify_sightlines(catalog, tau.shape[0])
    out: Dict[str, np.ndarray] = {}
    n_out: Dict[str, int] = {}
    kf_ref = None
    for label, mask in masks.items():
        kf, P = _per_class_p1d_at_scale(tau[mask], vmax, scale, target_F)
        if kf_ref is None:
            kf_ref = kf
        out[label] = P
        n_out[label] = int(mask.sum())
    return kf_ref, out, n_out, target_F, scale
```

- [ ] **Step 4: Run the test.** Expect `OK` for the per-class sum identity.

- [ ] **Step 5: Commit.**

```bash
git add hcd_analysis/priya_p1d.py tests/test_priya_p1d.py
git commit -m "feat: priya_p1d.compute_tier_c_p1d — per-class P1Ds sharing Tier-P scale/target_F

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3 — Slope-α grid + Kim 2013 inversion helper

Phase-2 stores `α_slope` in the cache (PRIYA convention). The α grid is
`np.linspace(0.66, 1.36, 20)` — same range as PRIYA's KODIAQ default but
double the density. The "actual" τ-multiplier scale per (sim, z, α) is
recovered at cache-load time from `target_F = exp(-α · obs_mean_tau(z))`
and the simulation's filtered τ distribution; we cache the value to save the
loader a re-solve.

**Files:**
- Modify: `hcd_analysis/tau0_rescale.py`
- Modify: `tests/test_tau0_rescale.py`

- [ ] **Step 1: Add a failing test.** Append to `tests/test_tau0_rescale.py`:

```python
def test_kim_slope_alpha_to_target_F_matches_obs_mean_tau():
    """The Tier P / Tier C convention is target_F = exp(-α · obs_mean_tau_Kim(z)).
    Verify the helper agrees with fake_spectra.fluxstatistics.obs_mean_tau."""
    from hcd_analysis.tau0_rescale import slope_alpha_to_target_F
    # at z=3, Kim 2013: obs_mean_tau = 2.3e-3 * 4^3.65 ≈ 0.3625
    target = slope_alpha_to_target_F(alpha_slope=1.0, z=3.0)
    expected = np.exp(-2.3e-3 * 4.0 ** 3.65)
    assert np.isclose(target, expected, rtol=1e-12)
```

- [ ] **Step 2: Add the helper** in `hcd_analysis/tau0_rescale.py`:

```python
def obs_mean_tau_kim2013(z):
    """Kim 2013 (arXiv 0711.1862) fit: tau_obs(z) = 2.3e-3 * (1+z)^3.65.
    Same formula as fake_spectra.fluxstatistics.obs_mean_tau; duplicated
    here so it can be called without loading fake_spectra (which needs gsl)."""
    return 2.3e-3 * (1.0 + np.asarray(z, dtype=np.float64)) ** 3.65


def slope_alpha_to_target_F(alpha_slope, z):
    """PRIYA convention: target_F(alpha, z) = exp(-alpha * tau_obs_Kim2013(z))."""
    return np.exp(-alpha_slope * obs_mean_tau_kim2013(z))
```

- [ ] **Step 3: Mark legacy functions.** Add to the docstrings of
  `freeze_core_rescale` and `tau0_from_mean_flux`:
  `"Legacy (2026-05-17 plan); superseded by the Tier-P PRIYA-compatible
  recipe in hcd_analysis.priya_p1d.  Retained for Phase-1 regression
  tests."`

- [ ] **Step 4: Run all tests** (`python3 tests/test_tau0_rescale.py`) — pass.

- [ ] **Step 5: Commit.**

```bash
git add hcd_analysis/tau0_rescale.py tests/test_tau0_rescale.py
git commit -m "feat: slope-alpha → target_F (Kim 2013) helper; mark freeze-core legacy

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4 — Builder: `build_emulator_cache_tau0.py` rewrite

The Phase-2a-from-2026-05-17 builder's `build_tau0_rows` currently loops α
calling `compute_p1d_per_class(tau_transform=...)`. Replace it with a
single-pass version that, for each (sim, snap):

1. Read raw τ once.
2. Apply `_apply_priya_filter` once → `tau_P` (a copy or in-place).
3. For each α in `make_alpha_grid()`:
   - target_F = `slope_alpha_to_target_F(α, z)`.
   - Tier P: `kf, P_P, _, scale = compute_tier_p_p1d(tau_P.copy_or_not, vmax, α, z)`.
   - Tier C: `kf_c, by_class, n_by_class, _, _ = compute_tier_c_p1d(tau_unfiltered, vmax, α, z, catalog, external_scale=scale, external_target_F=target_F)`.
   - Pack one cache row per α.

**Files:**
- Modify: `scripts/build_emulator_cache_tau0.py` (significant rewrite).
- Modify: `tests/test_emulator_cache_tau0.py` (rewrite per-class assertions
  to use the new function names + bit-identity probe vs PRIYA).

The old `--tier A/B` flag is dropped (Phase-2 always builds both tiers in
one pass; the cost is dominated by I/O, not the second flux_power call).

- [ ] **Step 1: Failing test.** Append a small integration test that runs
  the new `build_tau0_rows` on (sim 44, snap_017) with one α and asserts:
  - `row["P_tier_p"]` matches PRIYA's flux_vector at the matching row.
  - `row["P_clean"] + row["P_LLS"] + row["P_subDLA"] + row["P_DLA"]` weighted
    by sightline counts ≈ `row["P_tier_p_unfiltered"]` (a diagnostic
    no-filter Tier P that the builder also stores for the cache, to enable
    the user's "P_clean + HCD adds-on" inference path).

- [ ] **Step 2: Run; expect failure** because the builder still calls
  `compute_p1d_per_class`.

- [ ] **Step 3: Rewrite `build_tau0_rows`.** Pseudocode:

```python
def build_tau0_rows(sim_name, snap, snap_dir, raw_tau_path, alpha_slope_grid,
                    k_target, n_skewers=None):
    from hcd_analysis.catalog import AbsorberCatalog
    from hcd_analysis.io import read_header, read_meta, read_cddf
    from hcd_analysis.priya_p1d import (
        compute_tier_p_p1d, compute_tier_c_p1d, _apply_priya_filter,
    )

    meta = read_meta(snap_dir)
    cddf = read_cddf(snap_dir)
    dndx = bec.compute_dndx_per_class(meta["n_absorbers"], float(cddf["total_path"]))
    catalog = AbsorberCatalog.load_npz(snap_dir / "catalog.npz")
    header = read_header(raw_tau_path)
    nbins = int(header.nbins)
    dv_kms = float(meta["dv_kms"])
    vmax = nbins * dv_kms

    # 1. Read tau (subsample if requested)
    tau_unfilt = _read_tau(raw_tau_path, n_skewers=n_skewers)  # (n, nbins) float64
    # 2. Tier-P filter on a separate copy
    tau_filt = tau_unfilt.copy()
    _apply_priya_filter(tau_filt)

    rows = []
    for a_idx, alpha in enumerate(alpha_slope_grid):
        z = float(meta["z"])
        kf_p, P_P, target_F, scale = compute_tier_p_p1d(
            tau_filt, vmax, alpha_slope=alpha, z=z)
        kf_c, by_class, n_by_class, _, _ = compute_tier_c_p1d(
            tau_unfilt, vmax, alpha_slope=alpha, z=z, catalog=catalog,
            external_scale=scale, external_target_F=target_F)
        rows.append(_pack_row(sim_name, snap, alpha, a_idx, z, dv_kms, nbins,
                              kf_p, P_P, by_class, n_by_class,
                              target_F, scale, k_target))
    snap_block = _pack_snap_block(sim_name, snap, cddf, dndx)
    return rows, snap_block
```

- [ ] **Step 4: Rewrite `write_cache_tau0`** with the new schema (renamed
  datasets):
  - `P_tier_p[n_rows, k]`           — bit-PRIYA total
  - `P_clean[n_rows, k]`            — per-class
  - `P_LLS[n_rows, k]`              — per-class
  - `P_subDLA[n_rows, k]`           — per-class
  - `P_DLA[n_rows, k]`              — per-class
  - `n_clean[n_rows], n_LLS, n_subDLA, n_DLA` — sightline counts per class
    (so the loader can reweight)
  - `target_F[n_rows], scale[n_rows]` — per-row mean-flux convention
  - `alpha_slope[n_rows]`, `z[n_rows]`, `params[n_rows, 9]`, plus existing
    snap_* CDDF block
  - top-level attrs: `cache_version = "2.0"`,
    `priya_convention = "Kim 2013 slope-alpha, _filter_single_tau_complex(1e6,0.25)"`,
    `git_sha`, `created_utc`, `n_rows`, `n_snaps`, `alpha_range`,
    `k_convention = "angular (rad s/km)"`, `tau_thresh = 1e6`.

- [ ] **Step 5: CLI.** Strip the `--tier A/B` flag; Phase-2 always builds
  both Tier P and Tier C in one pass. Default output:
  `hcd_analysis/_emulator_data/observables_tau0.h5`.

- [ ] **Step 6: Run the new integration test.** Expect:
  - `P_tier_p` row 0 matches PRIYA's flux_vector at the matching `(sim_idx,
    z_idx, α_row)` to `max|r-1| < 1e-4`.
  - Per-class sum (un-filter probe) consistency to `rtol < 1e-10`.

- [ ] **Step 7: Smoke-test on 1 pair.**

```bash
python3 scripts/build_emulator_cache_tau0.py --limit 1 --n-alpha 3 \
    --n-skewers 4096 --output /tmp/tau0_smoke.h5 --spot-check
```

- [ ] **Step 8: Commit.**

```bash
git add scripts/build_emulator_cache_tau0.py tests/test_emulator_cache_tau0.py
git commit -m "refactor: pivot Phase-2a builder to fake_spectra (Tier P + Tier C)

Bit-identical to PRIYA flux_vectors for Tier P; per-class P1D for HCD adds-on.
Drops compute_p1d_per_class + tau_transform path entirely. Cache schema bumped
to v2.0 with separate per-class P arrays.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5 — Remove dead code, update docs

- [ ] Mark `hcd_analysis/tau0_rescale.freeze_core_rescale` as deprecated.
  Keep the function (Phase-1 tests still import it) but raise a
  `DeprecationWarning` when called from outside `tests/`.
- [ ] Update `docs/superpowers/specs/2026-05-17-phase2-hcd-emulator-design.md`:
  - §3 (mean-flux): rewrite "freeze-core" → "Tier P + Tier C" framework.
    The freeze-core rescale becomes one of several future Tier-C
    masking-recipe options, not the production recipe.
  - §4 (cache schema): bump to v2.0 dataset list.
  - §9 (validation): swap "freeze-core physics unit tests" for "bit-identity
    multi-point verification against PRIYA flux_vectors".
- [ ] Cross-reference both docs in
  `docs/SESSION_HANDOVER_2026_05_19.md` §5 (or write a new handover for
  2026-05-20).

- [ ] Commit.

```bash
git commit -m "docs: spec + handover updated for Tier P/C refactor (supersedes 2026-05-17)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6 — Multi-point regression test

Codify the multi-point bit-identity check from §10 of the consistency-check
doc as a real test (skipped if fake_spectra is unavailable):

```python
def test_priya_p1d_bit_identical_multipoint():
    """Run compute_tier_p_p1d at every (sim, snap, alpha) PRIYA stored for
    the three reference sims (44, 0, 29) at z ∈ {4.6, 4.0, 3.0, 2.4}.
    Assert max|r-1| < 1e-4 across all 120 test points.

    Slow (~80 min); skipped by default; run via
        SLOW_TESTS=1 python3 tests/test_priya_p1d.py
    """
```

- [ ] **Step 1:** Implement, guarded by `SLOW_TESTS` env var.
- [ ] **Step 2:** Document in the test file how to run.
- [ ] **Step 3:** Commit.

---

## Self-review notes

- The C-extension dependency on libgsl means the Phase-2a tests CANNOT run
  in the base `mfho/.conda/envs` Python (they need `emu-3.9` with gsl).
  Document this prominently in `tests/test_priya_p1d.py` and the plan
  itself; the ImportError-guard at the top of the test makes a stray run
  print SKIP instead of fail.
- The cache schema bump (v2.0) is breaking. Any Tier-A / Tier-B caches
  built under the 2026-05-17 plan must be deleted before running the new
  builder; the loader's `cache_version` check will refuse to read v1.0.
- `build_tau0_rows` is now I/O-bound on the τ read. The α loop is cheap by
  comparison (filter once, flux_power 20 times — flux_power is the only
  re-cost). Estimated wallclock per (sim, snap): ~16 s read + ~12 s filter
  + 20 × ~40 s = ~14 minutes. 1076 pairs × 14 min = ~250 hours single-core,
  → must shard via `--offset/--limit` on the cluster (no parallelism added
  in this plan).
- The freeze-core recipe IS NOT discarded; it lives in
  `hcd_analysis/tau0_rescale.py` and the design spec records it as a
  future Tier-C variant. The user's framing was clear: the survey baseline
  is `_filter_tau`; freeze-core is a future Tier-C swap-out option for the
  HCD adds-on emulator when we want to explore wing-preservation effects.

---

## What this plan DOES NOT cover

- Phase 2b (JAX loader + model + training + likelihood). Separate plan.
- Tier-C variants beyond the labelled-sightline split (e.g. Rahmati partial
  self-shielding, NHI ≥ 20.3 cut without τ-cap). Separate future plans;
  the `priya_p1d.py` module is structured so each variant is a swap of
  `_apply_priya_filter` for an alternative filter function.
- The production sharded cache build (1076 pairs × 20 α). That's an
  execution step, not a plan step; it runs *after* this plan is green.
