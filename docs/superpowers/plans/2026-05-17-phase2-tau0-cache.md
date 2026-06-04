# Phase 2a — τ₀-extended emulator cache builder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `observables_tau0.h5` — a mean-flux-extended HCD-emulator training cache — by rescaling the Lyα optical depth with the physically-correct "freeze-core" recipe and recomputing per-class P1D at 20 τ₀ points per (sim, snap).

**Architecture:** A small pure module (`hcd_analysis/tau0_rescale.py`) provides the freeze-core rescale and the α grid. `compute_p1d_per_class` gains an opt-in `tau_transform` hook so the existing two-pass per-class P1D code is reused unchanged. A new builder script (`scripts/build_emulator_cache_tau0.py`) discovers (sim, snap) pairs, locates the raw `fake_spectra` τ grids, loops α, and stacks the result into a two-level HDF5 cache (per-(sim,snap,α) P1D rows + per-(sim,snap) τ₀-invariant CDDF blocks). A `--tier` flag produces either the freeze-core production cache (Tier B) or the uniform-rescale twin (Tier A).

**Tech Stack:** Python 3.9, NumPy, h5py. Reuses `hcd_analysis/{p1d,catalog,io}.py` and `scripts/build_emulator_cache.py`. Tests are plain assert-scripts run with `python3 tests/test_*.py` (repo convention — not pytest).

**Spec:** `docs/superpowers/specs/2026-05-17-phase2-hcd-emulator-design.md` (§3 recipe, §4 schema, §9 unit tests).

**Commit convention:** every commit message ends with the trailer
`Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.

**Scope note:** This plan covers Phase 2a only — the cache builder. Phase 2b
(the JAX emulator, training loop, likelihood) and the Tier-C `fake_spectra`
re-extraction spot-check harness are separate follow-on plans, written once
this cache schema is proven against real data.

---

## File Structure

- Create `hcd_analysis/tau0_rescale.py` — pure τ₀-rescale functions, no I/O.
- Modify `hcd_analysis/p1d.py` — add `tau_transform` param to `compute_p1d_per_class`.
- Create `scripts/build_emulator_cache_tau0.py` — the builder.
- Create `tests/test_tau0_rescale.py` — tests for the pure module + the
  `tau_transform` hook.
- Create `tests/test_emulator_cache_tau0.py` — tests for the builder
  (helpers, schema, real-data integration, physics unit tests).

---

## Task 1: The τ₀-rescale pure module

**Files:**
- Create: `hcd_analysis/tau0_rescale.py`
- Test: `tests/test_tau0_rescale.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tau0_rescale.py`:

```python
"""Tests for hcd_analysis/tau0_rescale.py and the compute_p1d_per_class
tau_transform hook.

Run with: python3 tests/test_tau0_rescale.py
"""
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hcd_analysis.tau0_rescale import (
    freeze_core_rescale, make_alpha_grid, tau0_from_mean_flux,
    TAU_FREEZE_DEFAULT, N_ALPHA_DEFAULT,
)


def test_freeze_core_rescale_freezes_cores_scales_thin():
    tau = np.array([1.0e3, 5.0e6, 1.0e6, 2.0e6, 0.5])
    out = freeze_core_rescale(tau, alpha=1.3, tau_freeze=1.0e6)
    # tau > 1e6 frozen; tau <= 1e6 scaled by alpha
    expected = np.array([1.3e3, 5.0e6, 1.3e6, 2.0e6, 0.65])
    assert np.allclose(out, expected), f"got {out}"


def test_freeze_core_rescale_uniform_when_tau_freeze_inf():
    tau = np.array([1.0e3, 5.0e6, 2.0e6])
    out = freeze_core_rescale(tau, alpha=0.7, tau_freeze=np.inf)
    assert np.allclose(out, 0.7 * tau), f"got {out}"


def test_freeze_core_rescale_alpha_one_is_identity():
    tau = np.array([1.0e3, 5.0e6, 2.0e6, 0.1])
    out = freeze_core_rescale(tau, alpha=1.0, tau_freeze=TAU_FREEZE_DEFAULT)
    assert np.allclose(out, tau), f"got {out}"


def test_make_alpha_grid_spans_range_inclusive():
    grid = make_alpha_grid(n=20, lo=0.66, hi=1.36)
    assert grid.shape == (20,)
    assert np.isclose(grid[0], 0.66)
    assert np.isclose(grid[-1], 1.36)
    assert np.all(np.diff(grid) > 0), "grid must be strictly increasing"


def test_make_alpha_grid_default_count():
    assert make_alpha_grid().shape == (N_ALPHA_DEFAULT,)


def test_tau0_from_mean_flux_inverts_exp():
    assert np.isclose(tau0_from_mean_flux(np.exp(-2.0)), 2.0)
    assert np.isclose(tau0_from_mean_flux(1.0), 0.0)


if __name__ == "__main__":
    test_freeze_core_rescale_freezes_cores_scales_thin()
    test_freeze_core_rescale_uniform_when_tau_freeze_inf()
    test_freeze_core_rescale_alpha_one_is_identity()
    test_make_alpha_grid_spans_range_inclusive()
    test_make_alpha_grid_default_count()
    test_tau0_from_mean_flux_inverts_exp()
    print("OK")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 tests/test_tau0_rescale.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'hcd_analysis.tau0_rescale'`

- [ ] **Step 3: Write the module**

Create `hcd_analysis/tau0_rescale.py`:

```python
"""Mean-flux (tau0) rescaling of the Lya optical-depth field.

The emulator's mean-flux dimension is generated by post-processing the
optical depth tau. The physically-correct recipe ("freeze-core") rescales
optically-thin pixels by a factor alpha while *freezing* self-shielded
saturated cores (native tau > TAU_FREEZE_DEFAULT) at their native value:
self-shielded gas does not respond to the UV background. See
docs/superpowers/specs/2026-05-17-phase2-hcd-emulator-design.md sec. 3.

A uniform rescale ("Tier A", a deliberately-wrong systematic twin) is
recovered with tau_freeze = inf.
"""
from __future__ import annotations

import numpy as np

# Self-shielding freeze boundary. A native per-pixel Lya optical depth above
# this is a saturated DLA / strong-sub-DLA core (log N_HI >~ 19.3, verified)
# and is held fixed under rescaling. PRIYA's _filter_tau core threshold.
TAU_FREEZE_DEFAULT = 1.0e6

# Mean-flux factor grid: PRIYA's KODIAQ-slope range (coarse_grid.py).
ALPHA_LO_DEFAULT = 0.66
ALPHA_HI_DEFAULT = 1.36
N_ALPHA_DEFAULT = 20


def freeze_core_rescale(tau, alpha, tau_freeze=TAU_FREEZE_DEFAULT):
    """Rescale optically-thin pixels by `alpha`; freeze self-shielded cores.

    Parameters
    ----------
    tau : np.ndarray
        Native Lya optical depth (any shape). The freeze condition is
        evaluated on these native values.
    alpha : float
        Mean-flux multiplicative factor applied to non-frozen pixels.
    tau_freeze : float
        Pixels with native tau > tau_freeze keep their native value.
        Pass np.inf for a uniform rescale (Tier A).

    Returns
    -------
    np.ndarray
        Rescaled optical depth as float64, same shape as `tau`.
    """
    tau = np.asarray(tau, dtype=np.float64)
    return np.where(tau > tau_freeze, tau, alpha * tau)


def make_alpha_grid(n=N_ALPHA_DEFAULT, lo=ALPHA_LO_DEFAULT, hi=ALPHA_HI_DEFAULT):
    """Return `n` mean-flux factors, uniform in alpha over [lo, hi] inclusive."""
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    return np.linspace(lo, hi, n)


def tau0_from_mean_flux(mean_flux):
    """Mean optical depth tau0 = -ln<F> from a (rescaled) state's clean <F>."""
    mf = float(mean_flux)
    if mf <= 0.0:
        raise ValueError(f"mean_flux must be > 0, got {mf}")
    return float(-np.log(mf))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 tests/test_tau0_rescale.py`
Expected: PASS — prints `OK`

- [ ] **Step 5: Commit**

```bash
git add hcd_analysis/tau0_rescale.py tests/test_tau0_rescale.py
git commit -m "feat: tau0-rescale module (freeze-core, alpha grid)"
```

---

## Task 2: Add the `tau_transform` hook to `compute_p1d_per_class`

`compute_p1d_per_class` (`hcd_analysis/p1d.py:374-472`) reads τ batches from a
file in two passes. We add an opt-in `tau_transform` callable applied to each
batch in *both* passes, so the builder can inject the freeze-core rescale
without duplicating the per-class labelling and accumulation logic.

**Files:**
- Modify: `hcd_analysis/p1d.py:374-453`
- Test: `tests/test_tau0_rescale.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tau0_rescale.py`, before the `if __name__` block:

```python
def _make_synthetic_tau_hdf5(path, tau):
    """Write a (n_skewers, nbins) tau array as a SPECTRA-style HDF5 file."""
    with h5py.File(path, "w") as f:
        f.create_dataset("tau/H/1/1215", data=tau.astype(np.float32))


def test_compute_p1d_per_class_tau_transform_changes_mean_flux():
    from hcd_analysis.p1d import compute_p1d_per_class
    from hcd_analysis.catalog import AbsorberCatalog

    rng = np.random.default_rng(0)
    nbins = 128
    tau = rng.uniform(0.0, 2.0, size=(64, nbins))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spec.hdf5"
        _make_synthetic_tau_hdf5(path, tau)
        cat = AbsorberCatalog(sim_name="syn", snap=0, z=3.0, dv_kms=10.0)

        base = compute_p1d_per_class(path, nbins=nbins, dv_kms=10.0, catalog=cat)
        scaled = compute_p1d_per_class(
            path, nbins=nbins, dv_kms=10.0, catalog=cat,
            tau_transform=lambda t: 2.0 * t,
        )
        ident = compute_p1d_per_class(
            path, nbins=nbins, dv_kms=10.0, catalog=cat,
            tau_transform=lambda t: t,
        )

    # empty catalog => every sightline is "clean"
    assert base["n_sightlines_clean"] == 64
    assert np.isclose(base["mean_F_clean"], np.exp(-tau).mean())
    assert np.isclose(scaled["mean_F_clean"], np.exp(-2.0 * tau).mean())
    # identity transform reproduces the no-transform result exactly
    assert np.allclose(ident["P_clean"], base["P_clean"])
    assert np.isclose(ident["mean_F_clean"], base["mean_F_clean"])
```

Add its call inside the `if __name__ == "__main__":` block (before `print("OK")`):

```python
    test_compute_p1d_per_class_tau_transform_changes_mean_flux()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 tests/test_tau0_rescale.py`
Expected: FAIL — `TypeError: compute_p1d_per_class() got an unexpected keyword argument 'tau_transform'`

- [ ] **Step 3: Modify `compute_p1d_per_class`**

In `hcd_analysis/p1d.py`, change the signature (currently lines 374-382):

```python
def compute_p1d_per_class(
    hdf5_path,
    nbins: int,
    dv_kms: float,
    catalog,
    batch_size: int = 4096,
    n_skewers: Optional[int] = None,
    k_bins: Optional[np.ndarray] = None,
    tau_transform=None,
) -> Dict[str, object]:
```

Add to the docstring's Parameters (after the existing summary, before `Returns`):

```
    tau_transform : callable or None
        If given, applied to each native float64 tau batch before <F> and
        P1D accumulation, in BOTH passes. Used to inject the freeze-core
        mean-flux rescale (see hcd_analysis.tau0_rescale).
```

In **pass 1** (currently lines 435-442), replace:

```python
    for s, e, tau in iter_tau_batches(hdf5_path, batch_size=batch_size, n_skewers=n_total):
        F = np.exp(-tau.astype(np.float64))
        lab = labels[s:e]
```

with:

```python
    for s, e, tau in iter_tau_batches(hdf5_path, batch_size=batch_size, n_skewers=n_total):
        tau = tau.astype(np.float64)
        if tau_transform is not None:
            tau = tau_transform(tau)
        F = np.exp(-tau)
        lab = labels[s:e]
```

In **pass 2** (currently lines 447-453), replace:

```python
    for s, e, tau in iter_tau_batches(hdf5_path, batch_size=batch_size, n_skewers=n_total):
        tau = tau.astype(np.float64)
        lab = labels[s:e]
```

with:

```python
    for s, e, tau in iter_tau_batches(hdf5_path, batch_size=batch_size, n_skewers=n_total):
        tau = tau.astype(np.float64)
        if tau_transform is not None:
            tau = tau_transform(tau)
        lab = labels[s:e]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 tests/test_tau0_rescale.py`
Expected: PASS — prints `OK`

- [ ] **Step 5: Run the Phase 1 cache tests to confirm no regression**

Run: `python3 tests/test_emulator_cache.py`
Expected: PASS — prints `OK` (the new param defaults to `None`, so Phase 1 behaviour is unchanged).

- [ ] **Step 6: Commit**

```bash
git add hcd_analysis/p1d.py tests/test_tau0_rescale.py
git commit -m "feat: opt-in tau_transform hook in compute_p1d_per_class"
```

---

## Task 3: Builder skeleton and raw-τ file locator

**Files:**
- Create: `scripts/build_emulator_cache_tau0.py`
- Test: `tests/test_emulator_cache_tau0.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_emulator_cache_tau0.py`:

```python
"""Tests for scripts/build_emulator_cache_tau0.py.

Run with: python3 tests/test_emulator_cache_tau0.py
"""
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_emulator_cache_tau0 as bt0


def test_locate_raw_tau_file_finds_grid_file():
    with tempfile.TemporaryDirectory() as tmp:
        emu_root = Path(tmp)
        sim = "ns0.8Ap2e-09herei4heref3alphaq2hub0.7omegamh20.14hireionz7bhfeedback0.03"
        spectra_dir = emu_root / sim / "output" / "SPECTRA_010"
        spectra_dir.mkdir(parents=True)
        grid = spectra_dir / "lya_forest_spectra_grid_480.hdf5"
        grid.write_bytes(b"")  # presence is all locate checks

        found = bt0.locate_raw_tau_file(emu_root, sim, 10)
        assert found == grid, f"got {found}"


def test_locate_raw_tau_file_returns_none_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        found = bt0.locate_raw_tau_file(Path(tmp), "ns0.8Ap2e-09", 10)
        assert found is None


if __name__ == "__main__":
    test_locate_raw_tau_file_finds_grid_file()
    test_locate_raw_tau_file_returns_none_when_missing()
    print("OK")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 tests/test_emulator_cache_tau0.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_emulator_cache_tau0'`

- [ ] **Step 3: Write the builder skeleton with `locate_raw_tau_file`**

Create `scripts/build_emulator_cache_tau0.py`:

```python
"""Build the tau0-extended HCD-emulator training cache (Phase 2).

For every fully-processed (sim, snap) pair, rescale the raw fake_spectra
optical-depth grid with the freeze-core recipe at each of N alpha values
and stack the per-class P1D into observables_tau0.h5. The CDDF / dN/dX are
tau0-invariant and stored once per (sim, snap).

A --tier flag selects the rescale recipe:
  B (default) : freeze-core (tau_freeze = 1e6)        -> observables_tau0.h5
  A           : uniform rescale (tau_freeze = inf)    -> observables_tau0_uniform.h5

See docs/superpowers/specs/2026-05-17-phase2-hcd-emulator-design.md.

Usage:
    python3 scripts/build_emulator_cache_tau0.py \
        [--hcd-root /scratch/cavestru_root/cavestru0/mfho/hcd_outputs] \
        [--emu-root /nfs/turbo/umor-yueyingn/mfho/emu_full] \
        [--tier B] [--n-alpha 20] [--limit N] [--offset M] \
        [--n-skewers N] [--output PATH] [--spot-check]
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from functools import partial
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_emulator_cache as bec  # noqa: E402
from hcd_analysis.tau0_rescale import (  # noqa: E402
    freeze_core_rescale, make_alpha_grid, tau0_from_mean_flux,
    TAU_FREEZE_DEFAULT,
)

_DEFAULT_HCD_ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
_DEFAULT_EMU_ROOT = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")


def locate_raw_tau_file(emu_root, sim_name: str, snap: int):
    """Return the raw fake_spectra tau HDF5 for (sim_name, snap), or None.

    Layout: <emu_root>/<sim_name>/output/SPECTRA_<NNN>/
            lya_forest_spectra_grid_480.hdf5  (preferred)
            lya_forest_spectra.hdf5            (fallback)
    """
    spectra_dir = Path(emu_root) / sim_name / "output" / f"SPECTRA_{snap:03d}"
    if not spectra_dir.is_dir():
        return None
    grid = spectra_dir / "lya_forest_spectra_grid_480.hdf5"
    if grid.exists():
        return grid
    fallback = spectra_dir / "lya_forest_spectra.hdf5"
    if fallback.exists():
        return fallback
    return None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 tests/test_emulator_cache_tau0.py`
Expected: PASS — prints `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/build_emulator_cache_tau0.py tests/test_emulator_cache_tau0.py
git commit -m "feat: tau0 cache builder skeleton + raw-tau locator"
```

---

## Task 4: Discover τ₀-buildable (sim, snap) pairs

A τ₀ row needs three things: the Phase-1 outputs (`discover_sim_snap_pairs`
checks `meta.json`, `cddf_corrected.npz`, `p1d_per_class.h5`, `done`), the
native `catalog.npz` (for per-class sightline labels), and a locatable raw τ
grid in `emu_root`.

**Files:**
- Modify: `scripts/build_emulator_cache_tau0.py`
- Test: `tests/test_emulator_cache_tau0.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_emulator_cache_tau0.py`, before the `if __name__` block:

```python
_HCD_ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
_EMU_ROOT = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")


def test_discover_tau0_pairs_returns_nonempty():
    pairs = bt0.discover_tau0_pairs(_HCD_ROOT, _EMU_ROOT)
    assert len(pairs) >= 1, "no tau0-buildable (sim, snap) pairs found"
    sim, snap, snap_dir, raw = pairs[0]
    assert isinstance(sim, str) and sim.startswith("ns")
    assert isinstance(snap, int)
    assert (snap_dir / "catalog.npz").exists()
    assert raw.exists() and raw.suffix == ".hdf5"
    print(f"discover_tau0_pairs: {len(pairs)} pairs; first = ({sim}, snap_{snap:03d})")
```

Add its call inside `if __name__ == "__main__":` (before `print("OK")`):

```python
    test_discover_tau0_pairs_returns_nonempty()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 tests/test_emulator_cache_tau0.py`
Expected: FAIL — `AttributeError: module 'build_emulator_cache_tau0' has no attribute 'discover_tau0_pairs'`

- [ ] **Step 3: Add `discover_tau0_pairs`**

Append to `scripts/build_emulator_cache_tau0.py`:

```python
def discover_tau0_pairs(hcd_root, emu_root):
    """Return [(sim_name, snap, snap_dir, raw_tau_path), ...] for every
    (sim, snap) that has Phase-1 outputs, a native catalog.npz, AND a
    locatable raw fake_spectra tau grid."""
    out = []
    for sim, snap, snap_dir in bec.discover_sim_snap_pairs(Path(hcd_root)):
        if not (snap_dir / "catalog.npz").exists():
            continue
        raw = locate_raw_tau_file(emu_root, sim, snap)
        if raw is None:
            continue
        out.append((sim, snap, snap_dir, raw))
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 tests/test_emulator_cache_tau0.py`
Expected: PASS — prints the pair count and `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/build_emulator_cache_tau0.py tests/test_emulator_cache_tau0.py
git commit -m "feat: discover_tau0_pairs (Phase-1 + catalog + raw-tau gate)"
```

---

## Task 5: Build the per-(sim, snap) τ₀ rows

`build_tau0_rows` is the core: for one (sim, snap) it loops the α grid,
applies the freeze-core rescale through `compute_p1d_per_class`, interpolates
each per-class P1D onto the shared angular k-grid, and returns the per-α rows
plus the single τ₀-invariant CDDF block.

**Files:**
- Modify: `scripts/build_emulator_cache_tau0.py`
- Test: `tests/test_emulator_cache_tau0.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_emulator_cache_tau0.py`, before the `if __name__` block:

```python
def test_build_tau0_rows_integration_small():
    """Real-data integration: 1 pair, 2 alpha, limited skewers."""
    from hcd_analysis.p1d import _DEFAULT_K_BINS
    pairs = bt0.discover_tau0_pairs(_HCD_ROOT, _EMU_ROOT)
    sim, snap, snap_dir, raw = pairs[0]
    k_target = 2.0 * np.pi * _DEFAULT_K_BINS
    alpha_grid = np.array([0.8, 1.2])

    rows, snap_block = bt0.build_tau0_rows(
        sim, snap, snap_dir, raw, alpha_grid, k_target,
        tau_freeze=1.0e6, n_skewers=4096,
    )

    assert len(rows) == 2
    for a_idx, row in enumerate(rows):
        assert row["sim_name"] == sim and row["snap"] == snap
        assert row["alpha_idx"] == a_idx
        assert row["params"].shape == (9,)
        for key in ("P_clean", "P_LLS_only", "P_subDLA_only", "P_DLA_only"):
            assert row[key].shape == (50,)
        # tau0 == -ln(mean_F_clean) exactly (spec sec.9 unit test)
        assert np.isclose(row["tau0"], -np.log(row["mean_F_clean"]))
    # the two alpha rows must differ (mean flux moved)
    assert not np.isclose(rows[0]["tau0"], rows[1]["tau0"])

    # one tau0-invariant CDDF block per (sim, snap)
    assert snap_block["f_nhi"].shape == (30,)
    assert snap_block["n_absorbers"].shape == (30,)
    for key in ("dNdX_LLS", "dNdX_subDLA", "dNdX_DLA"):
        assert np.isfinite(snap_block[key])
    print(f"build_tau0_rows: ({sim}, snap_{snap:03d}) "
          f"tau0={rows[0]['tau0']:.3f},{rows[1]['tau0']:.3f}")
```

Add its call inside `if __name__ == "__main__":` (before `print("OK")`):

```python
    test_build_tau0_rows_integration_small()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 tests/test_emulator_cache_tau0.py`
Expected: FAIL — `AttributeError: module 'build_emulator_cache_tau0' has no attribute 'build_tau0_rows'`

- [ ] **Step 3: Add `build_tau0_rows`**

Append to `scripts/build_emulator_cache_tau0.py`:

```python
def build_tau0_rows(sim_name, snap, snap_dir, raw_tau_path, alpha_grid,
                    k_target, tau_freeze, n_skewers=None):
    """Build the per-alpha rows and the per-snap CDDF block for one (sim, snap).

    Returns (rows, snap_block):
      rows       : list of dicts, one per alpha (per-class P1D + scalars)
      snap_block : one dict with the tau0-invariant CDDF / dN/dX
    """
    from hcd_analysis.catalog import AbsorberCatalog
    from hcd_analysis.io import read_header
    from hcd_analysis.p1d import compute_p1d_per_class

    params_dict = bec.parse_sim_params(sim_name)
    if params_dict is None:
        raise ValueError(f"cannot parse params from sim folder name: {sim_name!r}")
    params = np.array([params_dict[k] for k in bec.PARAM_ORDER], dtype=np.float64)

    meta = bec.read_meta(snap_dir)
    cddf = bec.read_cddf(snap_dir)
    dndx = bec.compute_dndx_per_class(meta["n_absorbers"], float(cddf["total_path"]))
    catalog = AbsorberCatalog.load_npz(snap_dir / "catalog.npz")

    header = read_header(raw_tau_path)
    nbins = int(header.nbins)
    dv_kms = float(meta["dv_kms"])

    rows = []
    for a_idx, alpha in enumerate(alpha_grid):
        per_class = compute_p1d_per_class(
            raw_tau_path, nbins=nbins, dv_kms=dv_kms, catalog=catalog,
            n_skewers=n_skewers,
            tau_transform=partial(freeze_core_rescale, alpha=float(alpha),
                                  tau_freeze=tau_freeze),
        )
        k_src_angular = 2.0 * np.pi * per_class["k"]
        rows.append({
            "sim_name": sim_name,
            "snap": int(snap),
            "alpha": float(alpha),
            "alpha_idx": int(a_idx),
            "tau0": tau0_from_mean_flux(per_class["mean_F_clean"]),
            "params": params,
            "z": float(meta["z"]),
            "dv_kms": dv_kms,
            "nbins_native": nbins,
            "P_clean":       bec.interp_p1d_loglog(k_src_angular, per_class["P_clean"], k_target),
            "P_LLS_only":    bec.interp_p1d_loglog(k_src_angular, per_class["P_LLS_only"], k_target),
            "P_subDLA_only": bec.interp_p1d_loglog(k_src_angular, per_class["P_subDLA_only"], k_target),
            "P_DLA_only":    bec.interp_p1d_loglog(k_src_angular, per_class["P_DLA_only"], k_target),
            "mean_F_clean":  float(per_class["mean_F_clean"]),
            "mean_F_LLS":    float(per_class["mean_F_LLS"]),
            "mean_F_subDLA": float(per_class["mean_F_subDLA"]),
            "mean_F_DLA":    float(per_class["mean_F_DLA"]),
            "n_sightlines_clean":  int(per_class["n_sightlines_clean"]),
            "n_sightlines_LLS":    int(per_class["n_sightlines_LLS"]),
            "n_sightlines_subDLA": int(per_class["n_sightlines_subDLA"]),
            "n_sightlines_DLA":    int(per_class["n_sightlines_DLA"]),
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 tests/test_emulator_cache_tau0.py`
Expected: PASS — prints the τ₀ values and `OK`. (This test reads a real raw
τ grid; it may take ~10-30 s.)

- [ ] **Step 5: Commit**

```bash
git add scripts/build_emulator_cache_tau0.py tests/test_emulator_cache_tau0.py
git commit -m "feat: build_tau0_rows (freeze-core per-class P1D per alpha)"
```

---

## Task 6: Write the two-level τ₀ cache

The cache has two row-axes: per-(sim,snap,α) P1D rows, and per-(sim,snap)
τ₀-invariant CDDF blocks. Each P1D row carries a `snap_group_idx` pointing
into the CDDF block arrays.

**Files:**
- Modify: `scripts/build_emulator_cache_tau0.py`
- Test: `tests/test_emulator_cache_tau0.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_emulator_cache_tau0.py`, before the `if __name__` block:

```python
def _fake_row(sim, snap, a_idx, alpha, snap_group_idx):
    return {
        "sim_name": sim, "snap": snap, "alpha": alpha, "alpha_idx": a_idx,
        "snap_group_idx": snap_group_idx,
        "tau0": 2.0 + 0.1 * a_idx, "z": 3.0, "dv_kms": 10.0, "nbins_native": 1500,
        "params": np.arange(9, dtype=np.float64),
        "P_clean": np.full(50, 1.0), "P_LLS_only": np.full(50, 2.0),
        "P_subDLA_only": np.full(50, 3.0), "P_DLA_only": np.full(50, 4.0),
        "mean_F_clean": 0.3, "mean_F_LLS": 0.2,
        "mean_F_subDLA": 0.1, "mean_F_DLA": 0.05,
        "n_sightlines_clean": 600, "n_sightlines_LLS": 50,
        "n_sightlines_subDLA": 20, "n_sightlines_DLA": 10,
    }


def _fake_snap_block(sim, snap):
    return {
        "sim_name": sim, "snap": snap,
        "f_nhi": np.full(30, 1e-21), "n_absorbers": np.arange(30, dtype=np.int64),
        "log_nhi_centres": np.linspace(17.1, 22.9, 30),
        "log_nhi_edges": np.linspace(17.0, 23.0, 31),
        "total_path_dX": 1234.5,
        "dNdX_LLS": 0.5, "dNdX_subDLA": 0.2, "dNdX_DLA": 0.1,
    }


def test_write_cache_tau0_round_trip():
    sim = "ns0.8Ap2e-09herei4heref3alphaq2hub0.7omegamh20.14hireionz7bhfeedback0.03"
    k_target = np.geomspace(0.007, 0.31, 50)
    # 1 snap, 3 alpha
    rows = [_fake_row(sim, 10, i, 0.7 + 0.3 * i, snap_group_idx=0) for i in range(3)]
    snap_blocks = [_fake_snap_block(sim, 10)]

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "observables_tau0.h5"
        bt0.write_cache_tau0(rows, snap_blocks, k_target, out,
                             tier="B", tau_freeze=1.0e6,
                             alpha_range=(0.66, 1.36))
        with h5py.File(out, "r") as f:
            assert f["P_clean"].shape == (3, 50)
            assert f["alpha"].shape == (3,)
            assert f["tau0"].shape == (3,)
            assert f["snap_group_idx"].shape == (3,)
            assert np.all(f["snap_group_idx"][...] == 0)
            assert f["snap_f_nhi"].shape == (1, 30)
            assert f["snap_dNdX_DLA"].shape == (1,)
            assert f["k_target"].shape == (50,)
            assert f.attrs["rescale_tier"] == "B"
            assert f.attrs["tau_freeze"] == 1.0e6
            assert np.allclose(f["P_DLA_only"][0], 4.0)
            # per-row CDDF is reachable via snap_group_idx
            gi = f["snap_group_idx"][1]
            assert f["snap_f_nhi"][gi].shape == (30,)
    print("write_cache_tau0: round-trip OK")
```

Add its call inside `if __name__ == "__main__":` (before `print("OK")`):

```python
    test_write_cache_tau0_round_trip()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 tests/test_emulator_cache_tau0.py`
Expected: FAIL — `AttributeError: module 'build_emulator_cache_tau0' has no attribute 'write_cache_tau0'`

- [ ] **Step 3: Add `write_cache_tau0`**

Append to `scripts/build_emulator_cache_tau0.py`:

```python
_ROW_FLOAT_KEYS = (
    "alpha", "tau0", "z", "dv_kms",
    "mean_F_clean", "mean_F_LLS", "mean_F_subDLA", "mean_F_DLA",
)
_ROW_INT_KEYS = (
    "snap", "alpha_idx", "nbins_native", "snap_group_idx",
    "n_sightlines_clean", "n_sightlines_LLS",
    "n_sightlines_subDLA", "n_sightlines_DLA",
)
_ROW_P1D_KEYS = ("P_clean", "P_LLS_only", "P_subDLA_only", "P_DLA_only")
_SNAP_FLOAT_KEYS = ("total_path_dX", "dNdX_LLS", "dNdX_subDLA", "dNdX_DLA")
_SNAP_2D_KEYS = ("f_nhi", "n_absorbers")


def write_cache_tau0(rows, snap_blocks, k_target, output_path,
                     tier, tau_freeze, alpha_range):
    """Stack per-alpha `rows` + per-snap `snap_blocks` into one HDF5 cache.

    Each row carries `snap_group_idx`, an index into the snap_* datasets.
    """
    if not rows:
        raise ValueError("write_cache_tau0 called with no rows; nothing to write.")
    if not snap_blocks:
        raise ValueError("write_cache_tau0 called with no snap_blocks.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_nhi_centres = snap_blocks[0]["log_nhi_centres"]
    log_nhi_edges = snap_blocks[0]["log_nhi_edges"]
    for b in snap_blocks[1:]:
        assert np.array_equal(b["log_nhi_centres"], log_nhi_centres), \
            "log_nhi_centres mismatch across snap_blocks"

    with h5py.File(output_path, "w") as f:
        f.attrs["created_utc"] = (
            datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z")
        f.attrs["git_sha"] = bec._git_sha(REPO_ROOT)
        f.attrs["n_rows"] = len(rows)
        f.attrs["n_snaps"] = len(snap_blocks)
        f.attrs["rescale_tier"] = tier
        f.attrs["tau_freeze"] = float(tau_freeze)
        f.attrs["alpha_range"] = np.asarray(alpha_range, dtype=np.float64)
        f.attrs["k_convention"] = "angular (rad*s/km), PRIYA convention"

        f.create_dataset("k_target", data=np.asarray(k_target, dtype=np.float64))
        f.create_dataset("param_names",
                         data=np.array(list(bec.PARAM_ORDER), dtype=h5py.string_dtype()))
        f.create_dataset("log_nhi_centres", data=log_nhi_centres)
        f.create_dataset("log_nhi_edges", data=log_nhi_edges)

        # --- per-row (sim, snap, alpha) datasets ---
        f.create_dataset("sim_name",
                         data=np.array([r["sim_name"] for r in rows],
                                       dtype=h5py.string_dtype()))
        f.create_dataset("params", data=np.stack([r["params"] for r in rows], axis=0))
        for key in _ROW_FLOAT_KEYS:
            f.create_dataset(key, data=np.array([r[key] for r in rows], dtype=np.float64))
        for key in _ROW_INT_KEYS:
            f.create_dataset(key, data=np.array([r[key] for r in rows], dtype=np.int32))
        for key in _ROW_P1D_KEYS:
            f.create_dataset(key, data=np.stack([r[key] for r in rows], axis=0))

        # --- per-(sim, snap) tau0-invariant CDDF datasets ---
        f.create_dataset("snap_sim_name",
                         data=np.array([b["sim_name"] for b in snap_blocks],
                                       dtype=h5py.string_dtype()))
        f.create_dataset("snap_snap",
                         data=np.array([b["snap"] for b in snap_blocks], dtype=np.int32))
        for key in _SNAP_FLOAT_KEYS:
            f.create_dataset("snap_" + key,
                             data=np.array([b[key] for b in snap_blocks], dtype=np.float64))
        for key in _SNAP_2D_KEYS:
            f.create_dataset("snap_" + key,
                             data=np.stack([b[key] for b in snap_blocks], axis=0))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 tests/test_emulator_cache_tau0.py`
Expected: PASS — prints `write_cache_tau0: round-trip OK` and `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/build_emulator_cache_tau0.py tests/test_emulator_cache_tau0.py
git commit -m "feat: write_cache_tau0 two-level schema (rows + snap CDDF)"
```

---

## Task 7: Builder CLI (`main`) with the `--tier` flag

**Files:**
- Modify: `scripts/build_emulator_cache_tau0.py`
- Test: `tests/test_emulator_cache_tau0.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_emulator_cache_tau0.py`, before the `if __name__` block:

```python
def test_tier_to_tau_freeze_mapping():
    assert bt0.tier_to_tau_freeze("B") == 1.0e6
    assert bt0.tier_to_tau_freeze("A") == np.inf


def test_run_build_one_pair_end_to_end():
    """End-to-end: discover -> build 2 alpha -> write -> reload."""
    from hcd_analysis.p1d import _DEFAULT_K_BINS
    pairs = bt0.discover_tau0_pairs(_HCD_ROOT, _EMU_ROOT)[:1]
    k_target = 2.0 * np.pi * _DEFAULT_K_BINS
    alpha_grid = np.array([0.8, 1.2])

    all_rows, snap_blocks = [], []
    for sim, snap, snap_dir, raw in pairs:
        rows, block = bt0.build_tau0_rows(
            sim, snap, snap_dir, raw, alpha_grid, k_target,
            tau_freeze=1.0e6, n_skewers=4096)
        gi = len(snap_blocks)
        for r in rows:
            r["snap_group_idx"] = gi
        all_rows.extend(rows)
        snap_blocks.append(block)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "observables_tau0.h5"
        bt0.write_cache_tau0(all_rows, snap_blocks, k_target, out,
                             tier="B", tau_freeze=1.0e6, alpha_range=(0.8, 1.2))
        with h5py.File(out, "r") as f:
            assert f["P_clean"].shape == (2, 50)
            assert f["snap_f_nhi"].shape == (1, 30)
    print("run_build one-pair end-to-end: OK")
```

Add both calls inside `if __name__ == "__main__":` (before `print("OK")`):

```python
    test_tier_to_tau_freeze_mapping()
    test_run_build_one_pair_end_to_end()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 tests/test_emulator_cache_tau0.py`
Expected: FAIL — `AttributeError: module 'build_emulator_cache_tau0' has no attribute 'tier_to_tau_freeze'`

- [ ] **Step 3: Add `tier_to_tau_freeze` and `main`**

Append to `scripts/build_emulator_cache_tau0.py`:

```python
def tier_to_tau_freeze(tier: str) -> float:
    """Map a --tier letter to its tau_freeze boundary.

    B = freeze-core production (tau_freeze = 1e6).
    A = uniform-rescale twin   (tau_freeze = inf) — a deliberately-wrong
        systematic baseline; see spec sec. 3.
    """
    if tier == "B":
        return TAU_FREEZE_DEFAULT
    if tier == "A":
        return np.inf
    raise ValueError(f"unknown tier {tier!r}; expected 'A' or 'B'")


def _default_output(tier: str) -> Path:
    name = "observables_tau0.h5" if tier == "B" else "observables_tau0_uniform.h5"
    return REPO_ROOT / "hcd_analysis" / "_emulator_data" / name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the tau0-extended HCD-emulator cache (Phase 2).")
    parser.add_argument("--hcd-root", type=Path, default=_DEFAULT_HCD_ROOT)
    parser.add_argument("--emu-root", type=Path, default=_DEFAULT_EMU_ROOT)
    parser.add_argument("--tier", choices=["A", "B"], default="B",
                        help="B = freeze-core (default); A = uniform-rescale twin.")
    parser.add_argument("--n-alpha", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N (sim, snap) pairs.")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip the first M pairs (for sharded array jobs).")
    parser.add_argument("--n-skewers", type=int, default=None,
                        help="Limit skewers per snap (dry runs only).")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--spot-check", action="store_true",
                        help="After writing, re-verify row 0: tau0 == -ln(mean_F_clean).")
    args = parser.parse_args()

    from hcd_analysis.p1d import _DEFAULT_K_BINS
    k_target = 2.0 * np.pi * _DEFAULT_K_BINS

    tau_freeze = tier_to_tau_freeze(args.tier)
    alpha_grid = make_alpha_grid(n=args.n_alpha)
    output = args.output if args.output is not None else _default_output(args.tier)

    pairs = discover_tau0_pairs(args.hcd_root, args.emu_root)
    print(f"Found {len(pairs)} tau0-buildable (sim, snap) pairs")
    pairs = pairs[args.offset:]
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"Processing {len(pairs)} pairs (offset={args.offset}, limit={args.limit}); "
          f"tier={args.tier} (tau_freeze={tau_freeze}); n_alpha={args.n_alpha}")

    all_rows, snap_blocks = [], []
    for i, (sim, snap, snap_dir, raw) in enumerate(pairs):
        rows, block = build_tau0_rows(
            sim, snap, snap_dir, raw, alpha_grid, k_target,
            tau_freeze=tau_freeze, n_skewers=args.n_skewers)
        gi = len(snap_blocks)
        for r in rows:
            r["snap_group_idx"] = gi
        all_rows.extend(rows)
        snap_blocks.append(block)
        if (i + 1) % 10 == 0 or (i + 1) == len(pairs):
            print(f"  built {i + 1}/{len(pairs)} pairs ({len(all_rows)} rows)")

    write_cache_tau0(all_rows, snap_blocks, k_target, output,
                     tier=args.tier, tau_freeze=tau_freeze,
                     alpha_range=(float(alpha_grid[0]), float(alpha_grid[-1])))
    print(f"Wrote {output}  ({output.stat().st_size / 1e6:.2f} MB, "
          f"{len(all_rows)} rows, {len(snap_blocks)} snaps)")

    if args.spot_check and all_rows:
        with h5py.File(output, "r") as f:
            tau0_0 = float(f["tau0"][0])
            mfc_0 = float(f["mean_F_clean"][0])
        assert np.isclose(tau0_0, -np.log(mfc_0)), \
            f"spot-check failed: tau0={tau0_0} != -ln(mean_F_clean)={-np.log(mfc_0)}"
        print(f"spot-check: row 0 tau0={tau0_0:.4f} == -ln(mean_F_clean) OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 tests/test_emulator_cache_tau0.py`
Expected: PASS — prints `run_build one-pair end-to-end: OK` and `OK`

- [ ] **Step 5: Smoke-test the CLI on one pair**

Run:
```bash
python3 scripts/build_emulator_cache_tau0.py --limit 1 --n-alpha 3 \
    --n-skewers 4096 --output /tmp/tau0_smoke.h5 --spot-check
```
Expected: prints the pair count, `built 1/1 pairs`, `Wrote /tmp/tau0_smoke.h5`,
and `spot-check: row 0 ... OK`.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_emulator_cache_tau0.py tests/test_emulator_cache_tau0.py
git commit -m "feat: tau0 cache builder CLI with --tier A/B flag"
```

---

## Task 8: Freeze-core physics unit tests

Spec §9 requires unit tests pinning the physics. Task 1 covered the pure
function; this task pins the freeze-core behaviour *through the P1D pipeline*
on synthetic data — no `catalog.npz` dependency.

**Files:**
- Test: `tests/test_emulator_cache_tau0.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_emulator_cache_tau0.py`, before the `if __name__` block:

```python
def _write_tau_hdf5(path, tau):
    with h5py.File(path, "w") as f:
        f.create_dataset("tau/H/1/1215", data=tau.astype(np.float32))


def test_freeze_core_equals_uniform_when_no_core_pixels():
    """With all native tau < 1e6, freeze-core and uniform rescale are
    identical — nothing is frozen (spec sec. 9)."""
    from hcd_analysis.p1d import compute_p1d_per_class
    from hcd_analysis.catalog import AbsorberCatalog
    from hcd_analysis.tau0_rescale import freeze_core_rescale

    rng = np.random.default_rng(1)
    nbins = 128
    tau = rng.uniform(0.0, 5.0, size=(64, nbins))  # all << 1e6
    cat = AbsorberCatalog(sim_name="syn", snap=0, z=3.0, dv_kms=10.0)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spec.hdf5"
        _write_tau_hdf5(path, tau)
        frozen = compute_p1d_per_class(
            path, nbins=nbins, dv_kms=10.0, catalog=cat,
            tau_transform=partial(freeze_core_rescale, alpha=1.3, tau_freeze=1.0e6))
        uniform = compute_p1d_per_class(
            path, nbins=nbins, dv_kms=10.0, catalog=cat,
            tau_transform=partial(freeze_core_rescale, alpha=1.3, tau_freeze=np.inf))
    assert np.allclose(frozen["P_clean"], uniform["P_clean"])
    assert np.isclose(frozen["mean_F_clean"], uniform["mean_F_clean"])
    print("freeze-core == uniform when no core pixels: OK")


def test_freeze_core_differs_from_uniform_with_core_pixels():
    """With saturated pixels (tau > 1e6) present, freeze-core and uniform
    rescale give different mean flux — the cores are held fixed."""
    from hcd_analysis.p1d import compute_p1d_per_class
    from hcd_analysis.catalog import AbsorberCatalog
    from hcd_analysis.tau0_rescale import freeze_core_rescale

    rng = np.random.default_rng(2)
    nbins = 128
    tau = rng.uniform(0.0, 5.0, size=(64, nbins))
    tau[:8, 64] = 5.0e6  # 8 sightlines get a saturated core pixel

    cat = AbsorberCatalog(sim_name="syn", snap=0, z=3.0, dv_kms=10.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spec.hdf5"
        _write_tau_hdf5(path, tau)
        frozen = compute_p1d_per_class(
            path, nbins=nbins, dv_kms=10.0, catalog=cat,
            tau_transform=partial(freeze_core_rescale, alpha=1.3, tau_freeze=1.0e6))
        uniform = compute_p1d_per_class(
            path, nbins=nbins, dv_kms=10.0, catalog=cat,
            tau_transform=partial(freeze_core_rescale, alpha=1.3, tau_freeze=np.inf))
    # the saturated cores stay black (F=0) under freeze; uniform also leaves
    # F~0 there, but the difference shows in pixels just under the threshold —
    # so require the per-class P1D to differ somewhere.
    assert not np.allclose(frozen["P_clean"], uniform["P_clean"]), \
        "freeze-core and uniform must differ when core pixels are present"
    print("freeze-core != uniform with core pixels: OK")


def test_cddf_block_is_single_per_snap():
    """The CDDF block is built once per (sim, snap) and is therefore
    tau0-invariant by construction (spec sec. 3)."""
    pairs = bt0.discover_tau0_pairs(_HCD_ROOT, _EMU_ROOT)
    sim, snap, snap_dir, raw = pairs[0]
    from hcd_analysis.p1d import _DEFAULT_K_BINS
    k_target = 2.0 * np.pi * _DEFAULT_K_BINS
    rows, block = bt0.build_tau0_rows(
        sim, snap, snap_dir, raw, np.array([0.7, 1.0, 1.3]), k_target,
        tau_freeze=1.0e6, n_skewers=4096)
    assert len(rows) == 3
    assert isinstance(block, dict)
    assert block["f_nhi"].shape == (30,)  # one block, not 3
    print("CDDF block single-per-snap (tau0-invariant): OK")
```

Add the three calls inside `if __name__ == "__main__":` (before `print("OK")`):

```python
    test_freeze_core_equals_uniform_when_no_core_pixels()
    test_freeze_core_differs_from_uniform_with_core_pixels()
    test_cddf_block_is_single_per_snap()
```

Also add `from functools import partial` to the imports at the top of
`tests/test_emulator_cache_tau0.py`.

- [ ] **Step 2: Run the tests to verify they fail or pass**

Run: `python3 tests/test_emulator_cache_tau0.py`
Expected: the three new tests PASS (they exercise already-implemented code).
If `test_freeze_core_differs_from_uniform_with_core_pixels` fails, that is a
real signal the freeze logic is wrong — debug `freeze_core_rescale` before
proceeding.

- [ ] **Step 3: Run the full new test suite**

Run: `python3 tests/test_tau0_rescale.py && python3 tests/test_emulator_cache_tau0.py`
Expected: both print `OK`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_emulator_cache_tau0.py
git commit -m "test: freeze-core physics unit tests (spec sec.9)"
```

---

## Self-Review notes

- **Spec coverage.** §3 freeze-core recipe → Tasks 1, 2, 5. §3 Tier-A twin →
  Task 7 (`--tier A`). §3 α grid (20, [0.66,1.36]) → Task 1. §3 CDDF
  τ₀-invariance → Task 5 (single `snap_block`), Task 8. §4 schema /
  `observables_tau0.h5` → Task 6. §9 unit tests (frozen-pixel identity,
  mean_F=exp(−τ₀), CDDF invariance, k-convention) → Tasks 1, 5, 8.
  **Deliberately out of this plan** (Phase 2b / separate plans): the JAX
  loader/model/training/likelihood, the Tier-C `fake_spectra` re-extraction
  spot-checks, LOSO/PRIYA-cross-check validation. NaN/Nyquist masking is a
  loader concern (Phase 2b) — `interp_p1d_loglog` already emits NaN outside
  the native k-range, which the loader will mask.
- **Performance note (not a code bug).** `build_tau0_rows` calls
  `compute_p1d_per_class` once per α, each a two-pass read of the raw τ grid
  (~4 GB). The OS page cache absorbs the repeated reads within a snap, so real
  disk I/O is ~once per snap; the cost is CPU (≈ 2·n_alpha FFT passes/snap).
  Pairs are independent — shard the full run with `--offset/--limit` as an
  array job. An in-memory single-read optimisation is a possible later speedup.
- **Type consistency.** `freeze_core_rescale(tau, alpha, tau_freeze)`,
  `make_alpha_grid(n, lo, hi)`, `tau0_from_mean_flux(mean_flux)`,
  `compute_p1d_per_class(..., tau_transform=None)`,
  `locate_raw_tau_file(emu_root, sim_name, snap)`,
  `discover_tau0_pairs(hcd_root, emu_root)`,
  `build_tau0_rows(sim_name, snap, snap_dir, raw_tau_path, alpha_grid, k_target, tau_freeze, n_skewers=None)`,
  `write_cache_tau0(rows, snap_blocks, k_target, output_path, tier, tau_freeze, alpha_range)`,
  `tier_to_tau_freeze(tier)` — names and signatures are consistent across tasks.
  `build_tau0_rows` returns rows *without* `snap_group_idx`; `main` (Task 7)
  and the Task-7 test inject it before `write_cache_tau0` — intentional and
  consistent.

---

## After the plan: producing the real caches

These are execution steps, run after the plan's tasks are complete and green:

1. **Tier B production cache** (sharded array job recommended):
   `python3 scripts/build_emulator_cache_tau0.py --tier B --n-alpha 20`
   → `hcd_analysis/_emulator_data/observables_tau0.h5`
2. **Tier A uniform twin:**
   `python3 scripts/build_emulator_cache_tau0.py --tier A --n-alpha 20`
   → `hcd_analysis/_emulator_data/observables_tau0_uniform.h5`
3. Confirm both caches are gitignored (the `_emulator_data/` directory already
   is, from Phase 1).

Then: write the Phase 2b plan (JAX emulator + likelihood) and the Tier-C
`fake_spectra` spot-check plan.
