# Phase 1 HCD-Emulator Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Walk all 1076 `(sim, snap)` outputs under `/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/`, stack their per-class P1D + CDDF + dN/dX + parameter-vector observables into one in-repo HDF5 (`hcd_analysis/_emulator_data/observables.h5`, gitignored), interpolated onto the shared 50-bin `_DEFAULT_K_BINS` grid. This produces the training-ready cache that Phase 2 (the JAX two-head emulator) will read.

**Architecture:** A single Python module `scripts/build_emulator_cache.py` containing library-style functions plus a `main()` CLI entry. All work is read-only over existing on-disk files (no recomputation from raw spectra — mean-flux dimension is Phase 3, not Phase 1). The k-grid is unified via log–log linear interpolation onto `hcd_analysis.p1d._DEFAULT_K_BINS` with NaN-fill for native k-ranges that don't reach the Nyquist tail. A companion test file imports the script's functions and verifies a tiny 1–2 row cache round-trip.

**Tech Stack:** Python 3, NumPy, h5py, `hcd_analysis.io.parse_sim_params`, `hcd_analysis.p1d._DEFAULT_K_BINS`, stdlib `pathlib` / `json`. No new dependencies.

---

## File structure

| Path | Responsibility | New / Modified |
|---|---|---|
| `.gitignore` | Add `hcd_analysis/_emulator_data/` so the cache file is not tracked | Modified |
| `scripts/build_emulator_cache.py` | All cache-build logic: discovery → per-file readers → interpolation → row builder → aggregator → HDF5 writer → CLI | New |
| `tests/test_emulator_cache.py` | Imports the script as a module, builds a 1–2-row test cache, verifies schema + round-trip | New |
| `hcd_analysis/_emulator_data/observables.h5` | The cache itself, produced by the script. ~2–3 MB. Gitignored. | New (generated, untracked) |

Reusable function boundaries inside `scripts/build_emulator_cache.py`:

1. `discover_sim_snap_pairs(root: Path) -> list[tuple[str, int, Path]]` — yields `(sim_folder_name, snap_number, snap_dir)` triples for every valid `(sim, snap)`.
2. `read_meta(snap_dir: Path) -> dict` — parse `meta.json`.
3. `read_cddf(snap_dir: Path) -> dict` — load `cddf_corrected.npz` arrays into a dict.
4. `read_p1d_per_class(snap_dir: Path) -> dict` — load `p1d_per_class.h5` datasets into a dict.
5. `interp_p1d_loglog(k_src, P_src, k_target) -> np.ndarray` — log–log linear interpolation with NaN-fill outside the source range.
6. `compute_dndx_per_class(cddf: dict) -> dict[str, float]` — sum `n_absorbers` over `[17.2, 19.0)`, `[19.0, 20.3)`, `[20.3, ∞)` log-NHI ranges; divide by `total_path`.
7. `build_row(sim_dir: Path, snap_dir: Path, k_target: np.ndarray) -> dict` — single (sim, snap) row, returning a dict of arrays/scalars matching the output schema.
8. `write_cache(rows: list[dict], k_target: np.ndarray, output_path: Path) -> None` — stack and write the HDF5.
9. `verify_round_trip(cache_path: Path, sim: str, snap: int) -> None` — re-load one row, compare back to source files; raise on mismatch.
10. `main()` — CLI: `--root`, `--output`, `--limit` (for test runs), `--spot-check`.

---

## Output HDF5 schema

Root datasets (all 1076 rows in `params` order):

| Dataset | Shape | dtype | Source |
|---|---|---|---|
| `params` | `(N, 9)` | float64 | `parse_sim_params(sim_folder)` |
| `param_names` | `(9,)` | bytes (utf-8) | Hardcoded: `["ns","Ap","herei","heref","alphaq","hub","omegamh2","hireionz","bhfeedback"]` |
| `sim_name` | `(N,)` | bytes (utf-8) | sim folder name |
| `snap` | `(N,)` | int32 | parsed from `snap_NNN` |
| `z` | `(N,)` | float64 | `meta["z"]` |
| `dv_kms` | `(N,)` | float64 | `meta["dv_kms"]` |
| `nbins_native` | `(N,)` | int32 | `meta["nbins"]` |
| `k_target` | `(50,)` | float64 | `_DEFAULT_K_BINS` |
| `P_clean` | `(N, 50)` | float64 | interp(k_src, P_clean_src) |
| `P_LLS_only` | `(N, 50)` | float64 | interp(k_src, P_LLS_only_src) |
| `P_subDLA_only` | `(N, 50)` | float64 | interp(k_src, P_subDLA_only_src) |
| `P_DLA_only` | `(N, 50)` | float64 | interp(k_src, P_DLA_only_src) |
| `mean_F_clean` / `mean_F_LLS` / `mean_F_subDLA` / `mean_F_DLA` | `(N,)` each | float64 | from `p1d_per_class.h5` scalars |
| `n_sightlines_clean` / `n_sightlines_LLS` / `n_sightlines_subDLA` / `n_sightlines_DLA` | `(N,)` each | int32 | from `p1d_per_class.h5` scalars |
| `n_total_sightlines` | `(N,)` | int32 | `meta["n_skewers"]` |
| `log_nhi_centres` | `(30,)` | float64 | from one canonical `cddf_corrected.npz` (same grid for all sims) |
| `log_nhi_edges` | `(31,)` | float64 | same |
| `f_nhi` | `(N, 30)` | float64 | `cddf["f_nhi"]` |
| `n_absorbers` | `(N, 30)` | int64 | `cddf["n_absorbers"]` |
| `total_path_dX` | `(N,)` | float64 | `cddf["total_path"]` |
| `dNdX_LLS` / `dNdX_subDLA` / `dNdX_DLA` | `(N,)` each | float64 | `sum(n_absorbers in class range) / total_path` |

Root attrs: `created_utc`, `git_sha`, `n_rows`, `k_target_source = "hcd_analysis.p1d._DEFAULT_K_BINS"`, `interp_method = "loglog_linear_NaN_outside"`.

---

## Task 0: Gitignore entry

**Files:**
- Modify: `.gitignore` (under the "Data artifacts" section near line 220)

- [ ] **Step 1: Append gitignore entry**

Add this line to `.gitignore` under the existing "Data artifacts" comment block:

```
# Emulator training cache (Phase 1 — built by scripts/build_emulator_cache.py)
hcd_analysis/_emulator_data/
```

- [ ] **Step 2: Verify**

Run: `git check-ignore -v hcd_analysis/_emulator_data/observables.h5`
Expected: prints the .gitignore line that matched (non-empty exit-code-0 line).

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "gitignore: exclude hcd_analysis/_emulator_data/ (Phase 1 emulator cache)"
```

---

## Task 1: Script skeleton + discovery

**Files:**
- Create: `scripts/build_emulator_cache.py`
- Create: `tests/test_emulator_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_emulator_cache.py
"""Tests for scripts/build_emulator_cache.py.

Run with: python3 tests/test_emulator_cache.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_emulator_cache as bec

HCD_ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")


def test_discover_sim_snap_pairs_returns_nonempty():
    pairs = bec.discover_sim_snap_pairs(HCD_ROOT)
    assert len(pairs) >= 1, f"no (sim, snap) pairs found under {HCD_ROOT}"
    sim_name, snap, snap_dir = pairs[0]
    assert isinstance(sim_name, str) and sim_name.startswith("ns")
    assert isinstance(snap, int)
    assert snap_dir.is_dir()
    assert (snap_dir / "meta.json").exists()
    print(f"discover_sim_snap_pairs: {len(pairs)} pairs found; first = ({sim_name}, snap_{snap:03d})")


if __name__ == "__main__":
    test_discover_sim_snap_pairs_returns_nonempty()
    print("OK")
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python3 tests/test_emulator_cache.py`
Expected: `ModuleNotFoundError: No module named 'build_emulator_cache'`

- [ ] **Step 3: Write minimal `scripts/build_emulator_cache.py`**

```python
"""Build the HCD-emulator training cache (Phase 1).

Walks the on-disk (sim, snap) outputs and stacks their per-class P1D,
CDDF, dN/dX, and parameter-vector observables into one HDF5 file at
`hcd_analysis/_emulator_data/observables.h5`. Read-only over the
source data; no recomputation from raw spectra.

Usage:
    python3 scripts/build_emulator_cache.py \
        [--root /scratch/cavestru_root/cavestru0/mfho/hcd_outputs] \
        [--output hcd_analysis/_emulator_data/observables.h5] \
        [--limit N] [--spot-check]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hcd_analysis.io import parse_sim_params  # noqa: E402

SNAP_DIR_RE = re.compile(r"^snap_(\d{3})$")


def discover_sim_snap_pairs(root: Path) -> list[tuple[str, int, Path]]:
    """Return [(sim_folder_name, snap_number, snap_dir_path), ...] for every
    valid (sim, snap) under `root`. A pair is valid only if `meta.json`,
    `cddf_corrected.npz`, and `p1d_per_class.h5` all exist in the snap dir.
    """
    pairs: list[tuple[str, int, Path]] = []
    for sim_dir in sorted(root.iterdir()):
        if not sim_dir.is_dir():
            continue
        if parse_sim_params(sim_dir.name) is None:
            continue
        for snap_dir in sorted(sim_dir.iterdir()):
            m = SNAP_DIR_RE.match(snap_dir.name)
            if not m or not snap_dir.is_dir():
                continue
            required = ["meta.json", "cddf_corrected.npz", "p1d_per_class.h5"]
            if not all((snap_dir / f).exists() for f in required):
                continue
            pairs.append((sim_dir.name, int(m.group(1)), snap_dir))
    return pairs
```

- [ ] **Step 4: Run test, verify it passes**

Run: `python3 tests/test_emulator_cache.py`
Expected: `discover_sim_snap_pairs: 1076 pairs found; first = (ns…, snap_0NN)\nOK`

- [ ] **Step 5: Commit**

```bash
git add scripts/build_emulator_cache.py tests/test_emulator_cache.py
git commit -m "emulator-cache: skeleton + (sim, snap) discovery"
```

---

## Task 2: Per-file readers

**Files:**
- Modify: `scripts/build_emulator_cache.py` — add `read_meta`, `read_cddf`, `read_p1d_per_class`.
- Modify: `tests/test_emulator_cache.py` — add a test reading one snap's three files.

- [ ] **Step 1: Add the failing test**

Append to `tests/test_emulator_cache.py`:

```python
def test_per_file_readers_on_first_pair():
    pairs = bec.discover_sim_snap_pairs(HCD_ROOT)
    _, _, snap_dir = pairs[0]

    meta = bec.read_meta(snap_dir)
    assert "z" in meta and "dv_kms" in meta and "nbins" in meta and "n_skewers" in meta
    assert meta["nbins"] > 0 and 0 < meta["z"] < 10

    cddf = bec.read_cddf(snap_dir)
    for key in ("log_nhi_centres", "log_nhi_edges", "f_nhi", "n_absorbers", "total_path"):
        assert key in cddf, f"missing {key} in cddf_corrected.npz"
    assert cddf["log_nhi_centres"].shape == (30,)
    assert cddf["log_nhi_edges"].shape == (31,)
    assert cddf["f_nhi"].shape == (30,)
    assert cddf["n_absorbers"].shape == (30,)

    p1d = bec.read_p1d_per_class(snap_dir)
    for key in ("k", "P_clean", "P_LLS_only", "P_subDLA_only", "P_DLA_only",
                "mean_F_clean", "mean_F_LLS", "mean_F_subDLA", "mean_F_DLA",
                "n_sightlines_clean", "n_sightlines_LLS", "n_sightlines_subDLA",
                "n_sightlines_DLA", "n_total"):
        assert key in p1d, f"missing {key} in p1d_per_class.h5"
    assert p1d["k"].ndim == 1
    assert p1d["P_clean"].shape == p1d["k"].shape
    print(f"readers: meta z={meta['z']:.3f} nbins={meta['nbins']} | "
          f"cddf bins={len(cddf['log_nhi_centres'])} | "
          f"p1d k.shape={p1d['k'].shape}")


if __name__ == "__main__":
    test_discover_sim_snap_pairs_returns_nonempty()
    test_per_file_readers_on_first_pair()
    print("OK")
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python3 tests/test_emulator_cache.py`
Expected: `AttributeError: module 'build_emulator_cache' has no attribute 'read_meta'`

- [ ] **Step 3: Add the readers**

Append to `scripts/build_emulator_cache.py`:

```python
import json
import numpy as np
import h5py


def read_meta(snap_dir: Path) -> dict:
    with open(snap_dir / "meta.json") as f:
        return json.load(f)


def read_cddf(snap_dir: Path) -> dict:
    with np.load(snap_dir / "cddf_corrected.npz") as data:
        return {key: data[key] for key in data.files}


def read_p1d_per_class(snap_dir: Path) -> dict:
    out: dict = {}
    with h5py.File(snap_dir / "p1d_per_class.h5", "r") as f:
        for key in f.keys():
            arr = f[key][...]
            out[key] = arr.item() if arr.shape == () else arr
    return out
```

- [ ] **Step 4: Run test, verify it passes**

Run: `python3 tests/test_emulator_cache.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/build_emulator_cache.py tests/test_emulator_cache.py
git commit -m "emulator-cache: meta.json / cddf_corrected.npz / p1d_per_class.h5 readers"
```

---

## Task 3: Log–log P1D interpolation onto the shared k-grid

**Files:**
- Modify: `scripts/build_emulator_cache.py` — add `interp_p1d_loglog`.
- Modify: `tests/test_emulator_cache.py` — add an interpolation correctness test.

- [ ] **Step 1: Add the failing test**

Append to `tests/test_emulator_cache.py`:

```python
import numpy as np

def test_interp_p1d_loglog_pure_power_law_recovers_input():
    # P(k) = A k^n on a fine source grid; interpolation onto a coarse
    # subgrid should reproduce A k^n exactly (log-log interp is exact
    # for power laws).
    k_src = np.geomspace(1e-3, 5e-2, 200)
    A, n = 7.3, -1.4
    P_src = A * k_src**n

    k_target = np.geomspace(2e-3, 3e-2, 25)
    P_interp = bec.interp_p1d_loglog(k_src, P_src, k_target)
    assert np.allclose(P_interp, A * k_target**n, rtol=1e-10)


def test_interp_p1d_loglog_out_of_range_is_nan():
    k_src = np.geomspace(1e-3, 2e-2, 100)
    P_src = np.ones_like(k_src)
    k_target = np.array([5e-4, 1e-2, 5e-2])  # below, inside, above source range
    P_interp = bec.interp_p1d_loglog(k_src, P_src, k_target)
    assert np.isnan(P_interp[0])
    assert P_interp[1] == 1.0
    assert np.isnan(P_interp[2])


if __name__ == "__main__":
    test_discover_sim_snap_pairs_returns_nonempty()
    test_per_file_readers_on_first_pair()
    test_interp_p1d_loglog_pure_power_law_recovers_input()
    test_interp_p1d_loglog_out_of_range_is_nan()
    print("OK")
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python3 tests/test_emulator_cache.py`
Expected: `AttributeError: module 'build_emulator_cache' has no attribute 'interp_p1d_loglog'`

- [ ] **Step 3: Implement the interpolator**

Append to `scripts/build_emulator_cache.py`:

```python
def interp_p1d_loglog(k_src: np.ndarray, P_src: np.ndarray,
                      k_target: np.ndarray) -> np.ndarray:
    """Log-log linear interpolation of P(k) from `k_src` onto `k_target`.

    Out-of-range targets (k_target < k_src.min() or > k_src.max()) are
    set to NaN. Non-positive P values are also set to NaN at the output
    (cannot interpolate in log space).
    """
    k_src = np.asarray(k_src, dtype=np.float64)
    P_src = np.asarray(P_src, dtype=np.float64)
    k_target = np.asarray(k_target, dtype=np.float64)

    # Mask non-positive source points (cannot log them)
    pos = P_src > 0
    if pos.sum() < 2:
        return np.full(k_target.shape, np.nan)

    log_k_src = np.log(k_src[pos])
    log_P_src = np.log(P_src[pos])

    # In-range mask on k_target
    k_lo, k_hi = k_src[pos].min(), k_src[pos].max()
    in_range = (k_target >= k_lo) & (k_target <= k_hi)

    out = np.full(k_target.shape, np.nan)
    out[in_range] = np.exp(np.interp(np.log(k_target[in_range]), log_k_src, log_P_src))
    return out
```

- [ ] **Step 4: Run test, verify it passes**

Run: `python3 tests/test_emulator_cache.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/build_emulator_cache.py tests/test_emulator_cache.py
git commit -m "emulator-cache: log-log P1D interpolation onto shared k-grid"
```

---

## Task 4: dN/dX per class from CDDF

**Files:**
- Modify: `scripts/build_emulator_cache.py` — add `compute_dndx_per_class`.
- Modify: `tests/test_emulator_cache.py` — verify per-class sums on the first pair.

- [ ] **Step 1: Add the failing test**

Append to `tests/test_emulator_cache.py`:

```python
def test_dndx_per_class_matches_manual_sum_on_first_pair():
    pairs = bec.discover_sim_snap_pairs(HCD_ROOT)
    _, _, snap_dir = pairs[0]
    cddf = bec.read_cddf(snap_dir)

    result = bec.compute_dndx_per_class(cddf)

    # Manual reference computation using bin centres
    centres = cddf["log_nhi_centres"]
    n_abs = cddf["n_absorbers"]
    total_path = float(cddf["total_path"])
    lls_mask = (centres >= 17.2) & (centres < 19.0)
    sub_mask = (centres >= 19.0) & (centres < 20.3)
    dla_mask = (centres >= 20.3)
    ref = {
        "dNdX_LLS":    float(n_abs[lls_mask].sum() / total_path),
        "dNdX_subDLA": float(n_abs[sub_mask].sum() / total_path),
        "dNdX_DLA":    float(n_abs[dla_mask].sum() / total_path),
    }
    for k in ref:
        assert np.isclose(result[k], ref[k], rtol=1e-12), \
            f"{k}: got {result[k]}, expected {ref[k]}"
    print(f"dNdX: LLS={result['dNdX_LLS']:.4f}  subDLA={result['dNdX_subDLA']:.4f}  DLA={result['dNdX_DLA']:.4f}")


if __name__ == "__main__":
    test_discover_sim_snap_pairs_returns_nonempty()
    test_per_file_readers_on_first_pair()
    test_interp_p1d_loglog_pure_power_law_recovers_input()
    test_interp_p1d_loglog_out_of_range_is_nan()
    test_dndx_per_class_matches_manual_sum_on_first_pair()
    print("OK")
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python3 tests/test_emulator_cache.py`
Expected: `AttributeError: module 'build_emulator_cache' has no attribute 'compute_dndx_per_class'`

- [ ] **Step 3: Implement**

Append to `scripts/build_emulator_cache.py`:

```python
_LLS_LOW, _LLS_HIGH = 17.2, 19.0
_SUB_LOW, _SUB_HIGH = 19.0, 20.3
_DLA_LOW = 20.3


def compute_dndx_per_class(cddf: dict) -> dict[str, float]:
    centres = cddf["log_nhi_centres"]
    n_abs = cddf["n_absorbers"]
    total_path = float(cddf["total_path"])

    lls = float(n_abs[(centres >= _LLS_LOW) & (centres < _LLS_HIGH)].sum() / total_path)
    sub = float(n_abs[(centres >= _SUB_LOW) & (centres < _SUB_HIGH)].sum() / total_path)
    dla = float(n_abs[centres >= _DLA_LOW].sum() / total_path)
    return {"dNdX_LLS": lls, "dNdX_subDLA": sub, "dNdX_DLA": dla}
```

- [ ] **Step 4: Run test, verify it passes**

Run: `python3 tests/test_emulator_cache.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/build_emulator_cache.py tests/test_emulator_cache.py
git commit -m "emulator-cache: dN/dX per class from CDDF bin sums"
```

---

## Task 5: Single-row builder

**Files:**
- Modify: `scripts/build_emulator_cache.py` — add `build_row` and `PARAM_ORDER`.
- Modify: `tests/test_emulator_cache.py` — verify schema of one row.

- [ ] **Step 1: Add the failing test**

Append to `tests/test_emulator_cache.py`:

```python
def test_build_row_schema_first_pair():
    from hcd_analysis.p1d import _DEFAULT_K_BINS
    pairs = bec.discover_sim_snap_pairs(HCD_ROOT)
    sim_name, snap, snap_dir = pairs[0]

    row = bec.build_row(sim_name, snap, snap_dir, _DEFAULT_K_BINS)

    # scalars
    assert row["sim_name"] == sim_name
    assert row["snap"] == snap
    assert row["params"].shape == (9,)
    assert 0 < row["z"] < 10
    # P1D arrays
    for key in ("P_clean", "P_LLS_only", "P_subDLA_only", "P_DLA_only"):
        assert row[key].shape == (50,)
    # CDDF arrays
    assert row["f_nhi"].shape == (30,)
    assert row["n_absorbers"].shape == (30,)
    # dN/dX scalars
    for key in ("dNdX_LLS", "dNdX_subDLA", "dNdX_DLA"):
        assert key in row and np.isfinite(row[key])
    print(f"build_row: z={row['z']:.3f}  "
          f"P_clean finite frac = {np.isfinite(row['P_clean']).mean():.2f}")


if __name__ == "__main__":
    test_discover_sim_snap_pairs_returns_nonempty()
    test_per_file_readers_on_first_pair()
    test_interp_p1d_loglog_pure_power_law_recovers_input()
    test_interp_p1d_loglog_out_of_range_is_nan()
    test_dndx_per_class_matches_manual_sum_on_first_pair()
    test_build_row_schema_first_pair()
    print("OK")
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python3 tests/test_emulator_cache.py`
Expected: `AttributeError: module 'build_emulator_cache' has no attribute 'build_row'`

- [ ] **Step 3: Implement**

Append to `scripts/build_emulator_cache.py`:

```python
PARAM_ORDER = ("ns", "Ap", "herei", "heref", "alphaq",
               "hub", "omegamh2", "hireionz", "bhfeedback")


def build_row(sim_name: str, snap: int, snap_dir: Path,
              k_target: np.ndarray) -> dict:
    params_dict = parse_sim_params(sim_name)
    if params_dict is None:
        raise ValueError(f"could not parse params from sim folder name: {sim_name!r}")
    params = np.array([params_dict[k] for k in PARAM_ORDER], dtype=np.float64)

    meta = read_meta(snap_dir)
    cddf = read_cddf(snap_dir)
    p1d = read_p1d_per_class(snap_dir)
    dndx = compute_dndx_per_class(cddf)

    k_src = p1d["k"]
    row = {
        "sim_name": sim_name,
        "snap": int(snap),
        "params": params,
        "z": float(meta["z"]),
        "dv_kms": float(meta["dv_kms"]),
        "nbins_native": int(meta["nbins"]),
        "n_total_sightlines": int(meta["n_skewers"]),
        # per-class P1D, interpolated
        "P_clean":       interp_p1d_loglog(k_src, p1d["P_clean"],       k_target),
        "P_LLS_only":    interp_p1d_loglog(k_src, p1d["P_LLS_only"],    k_target),
        "P_subDLA_only": interp_p1d_loglog(k_src, p1d["P_subDLA_only"], k_target),
        "P_DLA_only":    interp_p1d_loglog(k_src, p1d["P_DLA_only"],    k_target),
        # per-class scalars
        "mean_F_clean":  float(p1d["mean_F_clean"]),
        "mean_F_LLS":    float(p1d["mean_F_LLS"]),
        "mean_F_subDLA": float(p1d["mean_F_subDLA"]),
        "mean_F_DLA":    float(p1d["mean_F_DLA"]),
        "n_sightlines_clean":  int(p1d["n_sightlines_clean"]),
        "n_sightlines_LLS":    int(p1d["n_sightlines_LLS"]),
        "n_sightlines_subDLA": int(p1d["n_sightlines_subDLA"]),
        "n_sightlines_DLA":    int(p1d["n_sightlines_DLA"]),
        # CDDF arrays + per-class dN/dX
        "log_nhi_centres": np.asarray(cddf["log_nhi_centres"], dtype=np.float64),
        "log_nhi_edges":   np.asarray(cddf["log_nhi_edges"], dtype=np.float64),
        "f_nhi":           np.asarray(cddf["f_nhi"], dtype=np.float64),
        "n_absorbers":     np.asarray(cddf["n_absorbers"], dtype=np.int64),
        "total_path_dX":   float(cddf["total_path"]),
        **dndx,
    }
    return row
```

- [ ] **Step 4: Run test, verify it passes**

Run: `python3 tests/test_emulator_cache.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/build_emulator_cache.py tests/test_emulator_cache.py
git commit -m "emulator-cache: single-row builder (params + P1D + CDDF + dN/dX)"
```

---

## Task 6: Aggregator + HDF5 writer

**Files:**
- Modify: `scripts/build_emulator_cache.py` — add `write_cache`.
- Modify: `tests/test_emulator_cache.py` — build a 2-row mini-cache to a tmp path and verify the round-trip.

- [ ] **Step 1: Add the failing test**

Append to `tests/test_emulator_cache.py`:

```python
import tempfile
import h5py

def test_write_cache_two_row_round_trip():
    from hcd_analysis.p1d import _DEFAULT_K_BINS
    pairs = bec.discover_sim_snap_pairs(HCD_ROOT)[:2]
    k_target = _DEFAULT_K_BINS
    rows = [bec.build_row(s, sn, sd, k_target) for s, sn, sd in pairs]

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "observables.h5"
        bec.write_cache(rows, k_target, out)
        assert out.exists()

        with h5py.File(out, "r") as f:
            assert f["params"].shape == (2, 9)
            assert f["P_clean"].shape == (2, 50)
            assert f["k_target"].shape == (50,)
            assert list(f["param_names"][...].astype(str)) == list(bec.PARAM_ORDER)
            # round-trip one scalar
            assert float(f["z"][0]) == rows[0]["z"]
            # round-trip one array
            assert np.array_equal(f["params"][0], rows[0]["params"])
            assert np.allclose(f["P_clean"][0], rows[0]["P_clean"], equal_nan=True)
    print("write_cache: 2-row round-trip OK")


if __name__ == "__main__":
    test_discover_sim_snap_pairs_returns_nonempty()
    test_per_file_readers_on_first_pair()
    test_interp_p1d_loglog_pure_power_law_recovers_input()
    test_interp_p1d_loglog_out_of_range_is_nan()
    test_dndx_per_class_matches_manual_sum_on_first_pair()
    test_build_row_schema_first_pair()
    test_write_cache_two_row_round_trip()
    print("OK")
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python3 tests/test_emulator_cache.py`
Expected: `AttributeError: module 'build_emulator_cache' has no attribute 'write_cache'`

- [ ] **Step 3: Implement**

Append to `scripts/build_emulator_cache.py`:

```python
import datetime
import subprocess


def _git_sha(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


_VECTOR_FLOAT_KEYS = (
    "z", "dv_kms", "total_path_dX",
    "mean_F_clean", "mean_F_LLS", "mean_F_subDLA", "mean_F_DLA",
    "dNdX_LLS", "dNdX_subDLA", "dNdX_DLA",
)
_VECTOR_INT_KEYS = (
    "snap", "nbins_native", "n_total_sightlines",
    "n_sightlines_clean", "n_sightlines_LLS",
    "n_sightlines_subDLA", "n_sightlines_DLA",
)
_PER_CLASS_P1D_KEYS = ("P_clean", "P_LLS_only", "P_subDLA_only", "P_DLA_only")
_PER_ROW_2D_KEYS = ("f_nhi", "n_absorbers")  # CDDF per-row 2D arrays


def write_cache(rows: list[dict], k_target: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(rows)

    sim_names = np.array([r["sim_name"] for r in rows], dtype=h5py.string_dtype())
    params = np.stack([r["params"] for r in rows], axis=0)  # (n, 9)
    log_nhi_centres = rows[0]["log_nhi_centres"]
    log_nhi_edges = rows[0]["log_nhi_edges"]
    # Sanity: assume all rows share the same NHI grid (they do — module-level constant).
    for r in rows[1:]:
        assert np.array_equal(r["log_nhi_centres"], log_nhi_centres), \
            "log_nhi_centres mismatch across rows"

    with h5py.File(output_path, "w") as f:
        f.attrs["created_utc"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        f.attrs["git_sha"] = _git_sha(REPO_ROOT)
        f.attrs["n_rows"] = n
        f.attrs["k_target_source"] = "hcd_analysis.p1d._DEFAULT_K_BINS"
        f.attrs["interp_method"] = "loglog_linear_NaN_outside"

        f.create_dataset("k_target", data=np.asarray(k_target, dtype=np.float64))
        f.create_dataset("param_names",
                         data=np.array(list(PARAM_ORDER), dtype=h5py.string_dtype()))
        f.create_dataset("sim_name", data=sim_names)
        f.create_dataset("params", data=params)
        f.create_dataset("log_nhi_centres", data=log_nhi_centres)
        f.create_dataset("log_nhi_edges", data=log_nhi_edges)

        for key in _VECTOR_FLOAT_KEYS:
            f.create_dataset(key, data=np.array([r[key] for r in rows], dtype=np.float64))
        for key in _VECTOR_INT_KEYS:
            f.create_dataset(key, data=np.array([r[key] for r in rows], dtype=np.int32))
        for key in _PER_CLASS_P1D_KEYS:
            f.create_dataset(key, data=np.stack([r[key] for r in rows], axis=0))
        for key in _PER_ROW_2D_KEYS:
            f.create_dataset(key, data=np.stack([r[key] for r in rows], axis=0))
```

- [ ] **Step 4: Run test, verify it passes**

Run: `python3 tests/test_emulator_cache.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/build_emulator_cache.py tests/test_emulator_cache.py
git commit -m "emulator-cache: HDF5 aggregator/writer with provenance attrs"
```

---

## Task 7: CLI + spot-check verifier

**Files:**
- Modify: `scripts/build_emulator_cache.py` — add `verify_round_trip`, `main`, argparse, `if __name__ == "__main__"` guard.

- [ ] **Step 1: Add the failing test**

Append to `tests/test_emulator_cache.py`:

```python
def test_verify_round_trip_against_source_first_pair():
    from hcd_analysis.p1d import _DEFAULT_K_BINS
    pairs = bec.discover_sim_snap_pairs(HCD_ROOT)[:1]
    k_target = _DEFAULT_K_BINS
    rows = [bec.build_row(s, sn, sd, k_target) for s, sn, sd in pairs]

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "observables.h5"
        bec.write_cache(rows, k_target, out)
        # Should not raise.
        bec.verify_round_trip(out, sim=pairs[0][0], snap=pairs[0][1])
    print("verify_round_trip: OK")


if __name__ == "__main__":
    test_discover_sim_snap_pairs_returns_nonempty()
    test_per_file_readers_on_first_pair()
    test_interp_p1d_loglog_pure_power_law_recovers_input()
    test_interp_p1d_loglog_out_of_range_is_nan()
    test_dndx_per_class_matches_manual_sum_on_first_pair()
    test_build_row_schema_first_pair()
    test_write_cache_two_row_round_trip()
    test_verify_round_trip_against_source_first_pair()
    print("OK")
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python3 tests/test_emulator_cache.py`
Expected: `AttributeError: module 'build_emulator_cache' has no attribute 'verify_round_trip'`

- [ ] **Step 3: Implement verifier + CLI**

Append to `scripts/build_emulator_cache.py`:

```python
import argparse


def verify_round_trip(cache_path: Path, sim: str, snap: int,
                      root: Path = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")) -> None:
    """Re-load one row from the cache and compare back to source files."""
    with h5py.File(cache_path, "r") as f:
        sim_names = f["sim_name"][...].astype(str)
        snaps = f["snap"][...]
        mask = (sim_names == sim) & (snaps == snap)
        if mask.sum() != 1:
            raise AssertionError(f"row ({sim}, {snap}) not unique in cache "
                                 f"({mask.sum()} matches)")
        i = int(np.where(mask)[0][0])
        cached = {
            "z": float(f["z"][i]),
            "params": f["params"][i],
            "P_clean": f["P_clean"][i],
            "dNdX_DLA": float(f["dNdX_DLA"][i]),
            "k_target": f["k_target"][...],
        }

    # Rebuild the row directly from source and compare
    match = [(s, sn, sd) for s, sn, sd in discover_sim_snap_pairs(root)
             if s == sim and sn == snap]
    if not match:
        raise AssertionError(f"({sim}, {snap}) not found under {root}")
    _, _, snap_dir = match[0]
    row = build_row(sim, snap, snap_dir, cached["k_target"])

    assert np.isclose(cached["z"], row["z"]), f"z mismatch: {cached['z']} vs {row['z']}"
    assert np.allclose(cached["params"], row["params"]), "params mismatch"
    assert np.allclose(cached["P_clean"], row["P_clean"], equal_nan=True), "P_clean mismatch"
    assert np.isclose(cached["dNdX_DLA"], row["dNdX_DLA"]), "dNdX_DLA mismatch"
    print(f"verify_round_trip: ({sim}, snap_{snap:03d}) matches source OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the HCD-emulator observables cache.")
    parser.add_argument("--root", type=Path,
                        default=Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs"),
                        help="Root scratch directory containing per-sim folders.")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "hcd_analysis" / "_emulator_data" / "observables.h5",
                        help="Output HDF5 path (will be overwritten).")
    parser.add_argument("--limit", type=int, default=None,
                        help="If set, only process the first N (sim, snap) pairs (for dry runs).")
    parser.add_argument("--spot-check", action="store_true",
                        help="After building, re-verify the first row against source files.")
    args = parser.parse_args()

    from hcd_analysis.p1d import _DEFAULT_K_BINS
    k_target = _DEFAULT_K_BINS

    pairs = discover_sim_snap_pairs(args.root)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"Found {len(pairs)} (sim, snap) pairs under {args.root}")

    rows = []
    for i, (sim, snap, snap_dir) in enumerate(pairs):
        rows.append(build_row(sim, snap, snap_dir, k_target))
        if (i + 1) % 100 == 0 or (i + 1) == len(pairs):
            print(f"  built {i + 1}/{len(pairs)}")

    write_cache(rows, k_target, args.output)
    print(f"Wrote cache to {args.output}  ({args.output.stat().st_size / 1e6:.2f} MB)")

    if args.spot_check and rows:
        verify_round_trip(args.output, sim=rows[0]["sim_name"], snap=rows[0]["snap"],
                          root=args.root)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, verify it passes**

Run: `python3 tests/test_emulator_cache.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/build_emulator_cache.py tests/test_emulator_cache.py
git commit -m "emulator-cache: CLI + spot-check verifier (full Phase 1)"
```

---

## Task 8: Build the real cache + smoke-check

- [ ] **Step 1: Dry run on 5 pairs**

Run:
```
python3 scripts/build_emulator_cache.py --limit 5 --spot-check \
    --output /tmp/observables_smoke.h5
```
Expected: prints `Found 1076 (sim, snap) pairs`, then `built 5/5`, then `Wrote cache to /tmp/observables_smoke.h5`, then `verify_round_trip: (ns…, snap_NNN) matches source OK`.

- [ ] **Step 2: Full build**

Run:
```
python3 scripts/build_emulator_cache.py --spot-check
```
Expected: prints progress every 100 rows, ends with `Wrote cache to hcd_analysis/_emulator_data/observables.h5  (~2-3 MB)` and the spot-check `OK` line.

- [ ] **Step 3: Schema audit**

Run:
```
python3 -c "
import h5py
with h5py.File('hcd_analysis/_emulator_data/observables.h5', 'r') as f:
    print('attrs:', dict(f.attrs))
    print('datasets:')
    for k in f.keys():
        print(f'  {k}: shape={f[k].shape}  dtype={f[k].dtype}')
"
```
Expected: 1076 rows everywhere; `k_target` shape `(50,)`; `params` shape `(1076, 9)`; non-zero `git_sha`.

- [ ] **Step 4: Confirm cache is gitignored**

Run: `git status --short`
Expected: only `scripts/build_emulator_cache.py` and `tests/test_emulator_cache.py` (already committed) appear; the `.h5` file is NOT listed.

- [ ] **Step 5: Push branch**

```bash
git push origin joint-emulator-scaffold
```
