# Phase-2 tau0 Cache Schema Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the tau0 emulator cache to store P1D on PRIYA's *native* k-grid in **separate LF and HR caches**, and replace the 4-class Tier C with **uniform 0.25-dex fine-N_HI bins + sightline counts**, so any absorber class reconstructs as a count-weighted sum at inference.

**Architecture:** `compute_tier_c_p1d` (in `hcd_analysis/priya_p1d.py`) gains a fine-N_HI binner returning a per-bin P1D stack + counts on the native grid. `build_emulator_cache_tau0.py` stops interpolating to a fixed 50-bin angular grid and instead stores the first `N_K` native FFT bins (172 LF / 522 HR) plus a per-row `kfkms`, driven by a new `--fidelity {lf,hr}` flag that selects the data roots, `N_K`, and output path. `merge_tau0_cache.py` and the tests follow the new schema. Bit-identity to PRIYA (LF 172, HR 522) is the validation anchor.

**Tech Stack:** Python 3.9 (emu-3.9 conda env), h5py, numpy, fake_spectra (GSL-linked). Tests that import fake_spectra MUST run under:
```bash
export LD_LIBRARY_PATH=/sw/pkgs/arc/stacks/gcc/10.3.0/gsl/2.7/lib:/home/mfho/.conda/envs/emu-3.9/lib:$LD_LIBRARY_PATH
PY=/home/mfho/.conda/envs/emu-3.9/bin/python3
```
A SKIP under the wrong env is **not** a pass.

---

## Key facts (verified 2026-05-21)

- **Native grid = first `N_K` FFT bins (follow `flux_power`; read N_K from the PRIYA ref, don't guess).** `compute_tier_p_p1d` returns `kf[1:], P[1:]` (k=0 dropped, the native `flux_power` grid). PRIYA stores the first `N_K` of these per row. **Confirmed from the ref files (2026-05-21):** LF `kfkms` is `(600, 13, 172)` → **N_K=172**, 13 z (zout 4.6→2.2); HR 6-sim ref `kfkms` is `(60, 17, 525)` → **N_K=525**, 17 z (zout 5.4→2.2). (The handover's "522" was a wrong guess.) The k-*values* vary per (sim,snap) because `vmax=nbins·dv` varies with z, so PRIYA stores `kfkms` per row; we do the same. `N_K` is safely below `min(nbins)//2` (HR z2.0 nbins=1175→587≥525; LF z2.0 nbins≈1228→614≥172). Both refs share layout: `params[N,10]`=[α, 9 cosmo], `kfkms[row,z,N_K]`, `flux_vectors` row = n_z blocks of N_K, explicit `zout`.
- **Fine-N_HI bins (anchored piecewise-linear, user decision 2026-05-21).** Edges are anchored EXACTLY at the physical class boundaries (LLS 17.2, subDLA 19.0, DLA 20.3) with linear ~0.26-dex spacing inside the LLS and subDLA bands and a COARSE DLA tail (most DLAs are survey-masked; only the near-20.3 edge leaks in). Class index per sightline (0..14, `N_TIER_C_BINS = 15`):
  `0 = clean` (no absorber ≥17.2); `1..7 = LLS [17.2,19.0)` (7 bins); `8..12 = subDLA [19.0,20.3)` (5 bins); `13 = DLA edge [20.3,21.0)`; `14 = DLA tail ≥21.0`. `Absorber.log_NHI` is the field; `min_log_nhi=17.2` means every catalog absorber is ≥17.2, so clean = "no absorber".
  - Boundary alignment: 17.2 (handled by the clean/has-absorber split), 19.0, and 20.3 are ALL exact edges, so LLS, subDLA, and DLA each reconstruct exactly as a count-weighted sum. The 1.3-dex subDLA band doesn't divide evenly by 0.25, so each band uses its own uniform spacing (per-band `linspace`), anchored at the boundaries.
- **HR data is split across roots.** Raw τ for all 6 HR sims: `/scratch/yueyingn_root/yueyingn0/mfho/priya/emu_full_hires_2/<sim>/output/SPECTRA_NNN/`. Phase-1 catalogs: `/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/hires/<sim>/snap_NNN/` (the 4 original + the 2 new ones from SLURM job 50612177). Old vs new raw τ are **bit-identical** for the 4 overlapping sims, and Phase-1 `snap_NNN` ↔ raw `SPECTRA_NNN` align *within each sim* (numbering only differs *between* sims). Pair by snap-per-sim; assert z agreement defensively.
- **LF data roots (unchanged):** Phase-1 `/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/<sim>/snap_NNN/`, raw τ `/nfs/turbo/umor-yueyingn/mfho/emu_full/<sim>/output/SPECTRA_NNN/`.

---

## File Structure

- **Modify** `hcd_analysis/priya_p1d.py` — add `FINE_NHI_EDGES`, `N_TIER_C_BINS`, `tier_c_labels()`, `bin_sightlines_by_nhi()`, `merge_fine_to_classes()`; rewrite `compute_tier_c_p1d` to return a per-bin P1D stack + counts. (`_classify_sightlines` is removed — superseded by `bin_sightlines_by_nhi`; the reconstruction test uses the latter.)
- **Modify** `scripts/build_emulator_cache_tau0.py` — `build_tau0_rows` (native grid + fine Tier C + z-assert, `n_k` param, drop `k_target`/interp), `write_cache_tau0` (schema v3.0), `discover_tau0_pairs` (fidelity-aware roots), `main` (`--fidelity`).
- **Modify** `scripts/merge_tau0_cache.py` — new key tuples, per-row `kfkms`, 3-D Tier-C concat.
- **Modify** `tests/test_priya_p1d.py` — fine-bin sum-identity + 4-class reconstruction.
- **Modify** `tests/test_emulator_cache_tau0.py` — v3.0 round-trip, native-grid Tier-P-vs-PRIYA (LF 172), HR-vs-6-sim-ref (522), fidelity discovery.

---

### Task 1: Fine-N_HI sightline binning (pure, no fake_spectra)

**Files:**
- Modify: `hcd_analysis/priya_p1d.py` (add after `_classify_sightlines`, ~line 89)
- Test: `tests/test_priya_p1d.py` (new `test_bin_sightlines_by_nhi`)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_priya_p1d.py` (top-level, runs without fake_spectra — it only needs numpy + the new helpers; guard import):

```python
def test_bin_sightlines_by_nhi():
    from types import SimpleNamespace
    from hcd_analysis.priya_p1d import (
        bin_sightlines_by_nhi, tier_c_labels, N_TIER_C_BINS, FINE_NHI_EDGES)
    # Edges (14): [17.2, ...7 LLS..., 19.0, ...5 subDLA..., 20.3, 21.0]
    # classes: 0 clean | 1..7 LLS | 8..12 subDLA | 13 DLA-edge [20.3,21.0) | 14 >=21.0
    ab = lambda i, lognhi: SimpleNamespace(skewer_idx=i, log_NHI=lognhi)
    cat = SimpleNamespace(absorbers=[
        ab(1, 17.3),    # LLS, [17.2,17.457) -> class 1
        ab(2, 19.0),    # subDLA floor (exact edge) -> class 8
        ab(3, 20.4),    # DLA edge [20.3,21.0) -> class 13
        ab(4, 23.1),    # DLA tail >=21.0 -> class 14
        ab(5, 18.0), ab(5, 20.9),  # highest wins: 20.9 -> DLA edge -> class 13
    ])
    cls = bin_sightlines_by_nhi(cat, n_skewers=6)
    assert cls.shape == (6,)
    assert cls[0] == 0          # clean (no absorber)
    assert cls[1] == 1          # 17.3
    assert cls[2] == 8          # 19.0 exact subDLA edge
    assert cls[3] == 13         # 20.4
    assert cls[4] == 14         # overflow tail
    assert cls[5] == 13         # max(18.0,20.9)=20.9
    assert len(FINE_NHI_EDGES) == 14 and N_TIER_C_BINS == 15
    assert tier_c_labels()[0] == "clean"
    assert tier_c_labels()[-1].startswith(">=")
    print("OK bin_sightlines_by_nhi")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/mfho/hcd_priya && python3 -c "import tests.test_priya_p1d as t; t.test_bin_sightlines_by_nhi()"`
Expected: FAIL with `ImportError: cannot import name 'bin_sightlines_by_nhi'`.

- [ ] **Step 3: Write minimal implementation**

In `hcd_analysis/priya_p1d.py`, after the imports/`_classify_sightlines`:

```python
# Fine-N_HI Tier-C bins, anchored at the physical class boundaries (user
# decision 2026-05-21): LLS 17.2, subDLA 19.0, DLA 20.3 are EXACT edges, with
# linear ~0.26-dex spacing inside the LLS and subDLA bands and a coarse DLA
# tail (most DLAs are survey-masked; only the near-20.3 edge leaks in).
# Sightlines are binned by their HIGHEST absorber's log_NHI. Class layout (15):
#   0      : clean (no absorber >= 17.2)
#   1..7   : LLS    [17.2, 19.0)  (7 uniform bins)
#   8..12  : subDLA [19.0, 20.3)  (5 uniform bins)
#   13     : DLA edge [20.3, 21.0)
#   14     : DLA tail >= 21.0
# 17.2/19.0/20.3 are exact edges -> LLS/subDLA/DLA reconstruct exactly. P1D is
# sightline-additive, so any class is a count-weighted sum (merge_fine_to_classes).
_LLS_EDGES = np.linspace(17.2, 19.0, 8)        # 7 bins (~0.257 dex)
_SUBDLA_EDGES = np.linspace(19.0, 20.3, 6)     # 5 bins (0.26 dex)
_DLA_EDGES = np.array([20.3, 21.0])            # DLA edge bin; >=21.0 overflow
FINE_NHI_EDGES = np.round(
    np.unique(np.concatenate([_LLS_EDGES, _SUBDLA_EDGES, _DLA_EDGES])), 4)  # 14 edges
N_TIER_C_BINS = len(FINE_NHI_EDGES) + 1                                     # 15


def tier_c_labels():
    """Human-readable label per Tier-C class index (len == N_TIER_C_BINS)."""
    labs = ["clean"]
    for lo, hi in zip(FINE_NHI_EDGES[:-1], FINE_NHI_EDGES[1:]):
        labs.append(f"{lo:.2f}-{hi:.2f}")
    labs.append(f">={FINE_NHI_EDGES[-1]:.2f}")
    return labs


def bin_sightlines_by_nhi(catalog, n_skewers: int) -> np.ndarray:
    """Return an int class index (0..N_TIER_C_BINS-1) per sightline, by the
    sightline's highest-log_NHI absorber. 0 = clean (no absorber)."""
    maxnhi = np.full(n_skewers, -np.inf)
    for ab in catalog.absorbers:
        if ab.skewer_idx < n_skewers and ab.log_NHI > maxnhi[ab.skewer_idx]:
            maxnhi[ab.skewer_idx] = ab.log_NHI
    cls = np.zeros(n_skewers, dtype=np.int64)         # clean = 0
    finite = np.isfinite(maxnhi)
    d = np.digitize(maxnhi[finite], FINE_NHI_EDGES)   # 0..23
    cls[finite] = np.clip(d, 1, len(FINE_NHI_EDGES))  # <17->1; >=22.5->23
    return cls
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/mfho/hcd_priya && python3 -c "import tests.test_priya_p1d as t; t.test_bin_sightlines_by_nhi()"`
Expected: `OK bin_sightlines_by_nhi`

- [ ] **Step 5: Commit**

```bash
git add hcd_analysis/priya_p1d.py tests/test_priya_p1d.py
git commit -m "feat: uniform 0.25-dex fine-N_HI sightline binning for Tier C"
```

---

### Task 2: Per-bin Tier C P1D + counts + class reconstruction

**Files:**
- Modify: `hcd_analysis/priya_p1d.py:114-155` (`compute_tier_c_p1d`); add `merge_fine_to_classes`
- Test: `tests/test_priya_p1d.py` (`test_tier_c_matches_filter_free_sum_at_alpha_one` — rewrite; add `test_fine_bins_reconstruct_subdla_split`)

- [ ] **Step 1: Write the failing tests**

Replace `test_tier_c_matches_filter_free_sum_at_alpha_one` body with the fine-bin version and add a reconstruction test (both need fake_spectra + real data; keep the existing module-level SKIP guard):

```python
def test_tier_c_fine_bins_sum_to_total_at_alpha_one():
    """Count-weighted sum of all fine-N_HI bin P1Ds == Tier-P-no-filter P1D."""
    from hcd_analysis.priya_p1d import compute_tier_c_p1d, compute_tier_p_p1d
    from hcd_analysis.catalog import AbsorberCatalog
    tau_p = f"/nfs/turbo/umor-yueyingn/mfho/emu_full/{SIM}/output/SPECTRA_{SNAP:03d}/lya_forest_spectra_grid_480.hdf5"
    meta_p = f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/meta.json"
    cat_p = f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/catalog.npz"
    with open(meta_p) as fh: m = json.load(fh)
    vmax = int(m["nbins"]) * float(m["dv_kms"]); z = float(m["z"])
    with h5py.File(tau_p, "r") as fh:
        tau = fh["tau/H/1/1215"][...].astype(np.float64)
    catalog = AbsorberCatalog.load_npz(cat_p)
    kf_P, P_P, target_F, scale = compute_tier_p_p1d(
        tau.copy(), vmax, alpha_slope=1.0, z=z, tau_thresh=np.inf)
    kf_C, P_by_bin, n_by_bin, _, _ = compute_tier_c_p1d(
        tau, vmax, alpha_slope=1.0, z=z, catalog=catalog,
        external_scale=scale, external_target_F=target_F)
    n_total = int(n_by_bin.sum())
    P_recombined = (n_by_bin[:, None] / n_total * P_by_bin).sum(axis=0)
    assert np.allclose(P_recombined, P_P, rtol=1e-10), \
        f"fine-bin sum != total; worst {np.max(np.abs(P_recombined/P_P-1)):.3e}"
    assert np.array_equal(kf_C, kf_P)
    print(f"OK fine-bin sum==total ({P_by_bin.shape[0]} bins, n={n_total})")


def test_fine_bins_reconstruct_subdla_split():
    """Merging fine bins at the aligned 19.0 edge reproduces a direct
    clean+LLS vs subDLA+DLA split (mechanics of count-weighted reconstruction)."""
    from hcd_analysis.priya_p1d import (
        compute_tier_c_p1d, compute_tier_p_p1d, merge_fine_to_classes,
        _per_class_p1d_at_scale, bin_sightlines_by_nhi, FINE_NHI_EDGES)
    from hcd_analysis.catalog import AbsorberCatalog
    tau_p = f"/nfs/turbo/umor-yueyingn/mfho/emu_full/{SIM}/output/SPECTRA_{SNAP:03d}/lya_forest_spectra_grid_480.hdf5"
    meta_p = f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/meta.json"
    cat_p = f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/catalog.npz"
    with open(meta_p) as fh: m = json.load(fh)
    vmax = int(m["nbins"]) * float(m["dv_kms"]); z = float(m["z"])
    with h5py.File(tau_p, "r") as fh:
        tau = fh["tau/H/1/1215"][...].astype(np.float64)
    catalog = AbsorberCatalog.load_npz(cat_p)
    _, _, tF, sc = compute_tier_p_p1d(tau.copy(), vmax, 1.0, z, tau_thresh=np.inf)
    kf_C, P_by_bin, n_by_bin, _, _ = compute_tier_c_p1d(
        tau, vmax, 1.0, z, catalog=catalog, external_scale=sc, external_target_F=tF)
    # edge index 8 == 19.0 -> class index 9 (digitize). Merge [0..8] vs [9..23].
    edge9 = int(np.where(np.isclose(FINE_NHI_EDGES, 19.0))[0][0]) + 1   # = 9
    P_lo = merge_fine_to_classes(P_by_bin, n_by_bin, [(0, edge9)])[0]
    # direct: sightlines with max log_NHI < 19.0 (cls < 9)
    cls = bin_sightlines_by_nhi(catalog, tau.shape[0])
    _, P_direct = _per_class_p1d_at_scale(tau[cls < edge9], vmax, sc, tF)
    assert np.allclose(P_lo, P_direct, rtol=1e-10)
    print("OK fine->class reconstruction at 19.0 edge")
```

Note: update the `if __name__` block to call the renamed/added tests.

- [ ] **Step 2: Run tests to verify they fail**

Run (emu-3.9 env): `cd /home/mfho/hcd_priya && $PY tests/test_priya_p1d.py`
Expected: FAIL — `compute_tier_c_p1d` still returns the old 4-class dict / `merge_fine_to_classes` undefined.

- [ ] **Step 3: Rewrite `compute_tier_c_p1d` and add `merge_fine_to_classes`**

Replace `compute_tier_c_p1d` (lines 114-155) with:

```python
def compute_tier_c_p1d(tau: np.ndarray, vmax: float,
                       alpha_slope: float, z: float,
                       catalog,
                       external_scale: Optional[float] = None,
                       external_target_F: Optional[float] = None,
                       ):
    """Per-fine-N_HI-bin P1Ds on the UNFILTERED tau, sharing Tier P's
    (scale, target_F) when given. Returns:
        kf        : native k-grid (s/km, angular), shape (npix//2,)
        P_by_bin  : (N_TIER_C_BINS, npix//2) per-bin P1D (count-weighted sums
                    reconstruct any class; see merge_fine_to_classes)
        n_by_bin  : (N_TIER_C_BINS,) int sightline counts
        target_F, scale
    """
    from fake_spectra.fluxstatistics import mean_flux
    if external_target_F is None:
        target_F = float(np.exp(-alpha_slope * obs_mean_tau(z)))
    else:
        target_F = float(external_target_F)
    scale = float(mean_flux(tau, target_F)) if external_scale is None else float(external_scale)

    cls = bin_sightlines_by_nhi(catalog, tau.shape[0])
    P_by_bin, n_by_bin = [], np.zeros(N_TIER_C_BINS, dtype=np.int64)
    kf_ref = None
    for c in range(N_TIER_C_BINS):
        mask = (cls == c)
        kf, P = _per_class_p1d_at_scale(tau[mask], vmax, scale, target_F)
        if kf_ref is None:
            kf_ref = kf
        P_by_bin.append(P)
        n_by_bin[c] = int(mask.sum())
    return kf_ref, np.stack(P_by_bin, axis=0), n_by_bin, target_F, scale


def merge_fine_to_classes(P_by_bin, n_by_bin, bin_ranges):
    """Count-weighted merge of fine bins into coarse classes.
    `bin_ranges` is a list of (lo, hi) half-open class-index ranges. Returns a
    list of P1D arrays, one per range (zeros where a range has no sightlines)."""
    out = []
    for lo, hi in bin_ranges:
        n = n_by_bin[lo:hi]
        ntot = int(n.sum())
        if ntot == 0:
            out.append(np.zeros(P_by_bin.shape[1]))
        else:
            out.append((n[:, None] / ntot * P_by_bin[lo:hi]).sum(axis=0))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run (emu-3.9 env): `cd /home/mfho/hcd_priya && $PY tests/test_priya_p1d.py`
Expected: all `OK` lines, including `OK fine-bin sum==total` and `OK fine->class reconstruction at 19.0 edge`.

- [ ] **Step 5: Commit**

```bash
git add hcd_analysis/priya_p1d.py tests/test_priya_p1d.py
git commit -m "feat: Tier C returns per-fine-bin P1D stack + counts; class reconstruction"
```

---

### Task 3: `build_tau0_rows` — native grid, fine Tier C, z-assert

**Files:**
- Modify: `scripts/build_emulator_cache_tau0.py:189-267` (`build_tau0_rows`)
- Test: `tests/test_emulator_cache_tau0.py` (`test_build_tau0_rows_tier_p_matches_priya` — update to native grid)

- [ ] **Step 1: Update the failing test to native grid**

Rewrite `test_build_tau0_rows_tier_p_matches_priya` (in `tests/test_emulator_cache_tau0.py`) so it builds with `n_k=172` and compares the stored native row directly (no interpolation):

```python
def test_build_tau0_rows_tier_p_matches_priya():
    SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
    SNAP = 17; NK = 172; ZIDX = 8; ROW = 344
    snap_dir = Path(f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}")
    raw = Path(f"/nfs/turbo/umor-yueyingn/mfho/emu_full/{SIM}/output/SPECTRA_{SNAP:03d}/lya_forest_spectra_grid_480.hdf5")
    if not (snap_dir.exists() and raw.exists()):
        print("SKIP test_build_tau0_rows_tier_p_matches_priya (data unavailable)"); return
    priya = "/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/mf_emulator_flux_vectors_tau1000000.hdf5"
    with h5py.File(priya, "r") as f:
        alpha = float(f["params"][ROW, 0])
        P_priya = f["flux_vectors"][ROW, ZIDX*NK:(ZIDX+1)*NK].astype(np.float64)
        kp = f["kfkms"][ROW, ZIDX, :].astype(np.float64)
    rows, _ = bt0.build_tau0_rows(SIM, SNAP, snap_dir, raw,
                                  alpha_slope_grid=np.array([alpha]), n_k=NK)
    r = rows[0]
    assert r["kfkms"].shape == (NK,) and r["P_tier_p"].shape == (NK,)
    assert np.max(np.abs(r["kfkms"] / kp - 1)) < 1e-12, "native k-grid mismatch"
    assert np.max(np.abs(r["P_tier_p"] / P_priya - 1)) < 1e-4, "Tier-P not bit-identical"
    assert r["P_tier_c"].shape == (bt0_pp.N_TIER_C_BINS, NK)
    assert r["tier_c_counts"].shape == (bt0_pp.N_TIER_C_BINS,)
    print("OK build_tau0_rows native-grid Tier-P bit-identical")
```

Add near the top of the test file: `from hcd_analysis import priya_p1d as bt0_pp`.

- [ ] **Step 2: Run test to verify it fails**

Run (emu-3.9 env): `cd /home/mfho/hcd_priya && $PY tests/test_emulator_cache_tau0.py`
Expected: FAIL — `build_tau0_rows` takes `k_target`, not `n_k`, and rows have no `kfkms`/`P_tier_c`.

- [ ] **Step 3: Rewrite `build_tau0_rows`**

Change the signature and body (lines 189-267). New signature:
`def build_tau0_rows(sim_name, snap, snap_dir, raw_tau_path, alpha_slope_grid, n_k, n_skewers=None):`

Replace the per-alpha loop body and `rows.append({...})` with native-grid + fine Tier C:

```python
    z_meta = float(meta["z"])
    z_grid = _snap_z_to_priya_grid(z_meta)
    # Defensive: raw header z must agree with Phase-1 meta z (catches mis-pairing
    # across the cross-sim SPECTRA-numbering differences; see plan Key facts).
    with h5py.File(raw_tau_path, "r") as _f:
        z_raw = float(_f["Header"].attrs["redshift"])
    assert abs(z_raw - z_meta) < 1e-2, \
        f"z mismatch raw={z_raw} meta={z_meta} for {sim_name} snap {snap}"

    tau_unfilt = _read_tau(raw_tau_path, n_skewers=n_skewers)
    rows = []
    for a_idx, alpha in enumerate(alpha_slope_grid):
        alpha = float(alpha)
        tau_filt = tau_unfilt.copy()
        kf_p, P_tier_p, target_F, scale = compute_tier_p_p1d(
            tau_filt, vmax, alpha_slope=alpha, z=z_grid)
        del tau_filt
        kf_c, P_by_bin, n_by_bin, _, _ = compute_tier_c_p1d(
            tau_unfilt, vmax, alpha_slope=alpha, z=z_grid, catalog=catalog,
            external_scale=scale, external_target_F=target_F)
        assert len(kf_p) >= n_k and len(kf_c) >= n_k, \
            f"native grid {len(kf_p)} bins < n_k={n_k} for {sim_name} snap {snap}"
        rows.append({
            "sim_name": sim_name, "snap": int(snap),
            "alpha_slope": alpha, "alpha_idx": int(a_idx),
            "z_meta": z_meta, "z_grid": float(z_grid),
            "dv_kms": dv_kms, "nbins_native": nbins,
            "target_F": float(target_F), "scale": float(scale),
            "params": params,
            "kfkms": kf_p[:n_k].astype(np.float64),
            "P_tier_p": P_tier_p[:n_k].astype(np.float64),
            "P_tier_c": P_by_bin[:, :n_k].astype(np.float64),
            "tier_c_counts": n_by_bin.astype(np.int64),
        })
```

Remove the old `from ... import compute_tier_c_p1d` line only if it changes; the imports `compute_tier_p_p1d, compute_tier_c_p1d` stay. The `snap_block` construction below is unchanged. Keep `compute_tier_c_p1d` import; it now returns the new tuple.

- [ ] **Step 4: Run test to verify it passes**

Run (emu-3.9 env): `cd /home/mfho/hcd_priya && $PY tests/test_emulator_cache_tau0.py`
Expected: `OK build_tau0_rows native-grid Tier-P bit-identical` (other tests may still fail until Task 4 — that's expected; this asserts the row shapes/values).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_emulator_cache_tau0.py tests/test_emulator_cache_tau0.py
git commit -m "feat: build_tau0_rows stores native k-grid + per-fine-bin Tier C (no interp)"
```

---

### Task 4: `write_cache_tau0` — schema v3.0

**Files:**
- Modify: `scripts/build_emulator_cache_tau0.py:270-343` (key tuples + `write_cache_tau0`)
- Test: `tests/test_emulator_cache_tau0.py` (`test_write_cache_tau0_round_trip` — update; `_make_fake_rows` helper)

- [ ] **Step 1: Update the round-trip test for v3.0**

Rewrite the fake-row builder and `test_write_cache_tau0_round_trip` so rows carry `kfkms`/`P_tier_p`/`P_tier_c`/`tier_c_counts`:

```python
def _fake_rows(nk, nrows=3):
    from hcd_analysis.priya_p1d import N_TIER_C_BINS as NB
    rows = []
    for i in range(nrows):
        rows.append({
            "sim_name": f"sim{i%2}", "snap": 10 + i, "alpha_slope": 1.0 + 0.1*i,
            "alpha_idx": i, "z_meta": 3.0, "z_grid": 3.0, "dv_kms": 10.0,
            "nbins_native": 1228, "target_F": 0.7, "scale": 0.9,
            "params": np.arange(9, dtype=np.float64),
            "kfkms": np.linspace(5e-4, 0.08, nk),
            "P_tier_p": np.full(nk, 5.0),
            "P_tier_c": np.tile(np.arange(NB)[:, None], (1, nk)).astype(float),
            "tier_c_counts": np.arange(NB, dtype=np.int64),
            "snap_group_idx": i,
        })
    return rows

def test_write_cache_tau0_round_trip(tmp_path=Path("/tmp")):
    import hcd_analysis.priya_p1d as pp
    nk = 172
    rows = _fake_rows(nk)
    snap_blocks = [_fake_snap_block(i) for i in range(3)]  # existing helper
    out = tmp_path / "rt_tau0.h5"
    bt0.write_cache_tau0(rows, snap_blocks, out, alpha_range=(1.0, 1.2), n_k=nk)
    with h5py.File(out, "r") as f:
        assert f.attrs["cache_version"] == "3.0"
        assert f["P_tier_p"].shape == (3, nk)
        assert f["kfkms"].shape == (3, nk)
        assert f["P_tier_c"].shape == (3, pp.N_TIER_C_BINS, nk)
        assert f["tier_c_counts"].shape == (3, pp.N_TIER_C_BINS)
        assert list(f["tier_c_labels"].asstr()[...]) == pp.tier_c_labels()
        assert np.allclose(f["P_tier_p"][0], 5.0)
    print("OK write_cache_tau0 v3.0 round-trip")
```

(Keep/define `_fake_snap_block` as in the current test — CDDF block is unchanged.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/mfho/hcd_priya && python3 tests/test_emulator_cache_tau0.py`
Expected: FAIL — `write_cache_tau0` signature still takes `k_target`; no `kfkms`/`P_tier_c` datasets.

- [ ] **Step 3: Rewrite key tuples + `write_cache_tau0`**

Replace the row-key tuples (lines 270-277) with:

```python
_ROW_FLOAT_KEYS = ("alpha_slope", "target_F", "scale", "z_meta", "z_grid", "dv_kms")
_ROW_INT_KEYS = ("snap", "alpha_idx", "nbins_native", "snap_group_idx")
_ROW_P1D_KEYS = ("kfkms", "P_tier_p")          # 2-D (n_rows, n_k)
_ROW_TIERC_KEYS = ("P_tier_c",)                 # 3-D (n_rows, N_TIER_C_BINS, n_k)
_ROW_COUNT_KEYS = ("tier_c_counts",)            # 2-D (n_rows, N_TIER_C_BINS)
_SNAP_FLOAT_KEYS = ("total_path_dX", "dNdX_LLS", "dNdX_subDLA", "dNdX_DLA")
_SNAP_2D_KEYS = ("f_nhi", "n_absorbers")
```

Replace `write_cache_tau0` signature/body (drop `k_target`, add `n_k`):

```python
def write_cache_tau0(rows, snap_blocks, output_path, alpha_range, n_k):
    from hcd_analysis.priya_p1d import tier_c_labels, FINE_NHI_EDGES, N_TIER_C_BINS
    if not rows:
        raise ValueError("write_cache_tau0 called with no rows.")
    if not snap_blocks:
        raise ValueError("write_cache_tau0 called with no snap_blocks.")
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    log_nhi_centres = snap_blocks[0]["log_nhi_centres"]
    log_nhi_edges = snap_blocks[0]["log_nhi_edges"]
    with h5py.File(output_path, "w") as f:
        f.attrs["created_utc"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        f.attrs["git_sha"] = bec._git_sha(REPO_ROOT)
        f.attrs["n_rows"] = len(rows)
        f.attrs["n_snaps"] = len(snap_blocks)
        f.attrs["cache_version"] = "3.0"
        f.attrs["n_k"] = int(n_k)
        f.attrs["priya_convention"] = (
            "Kim 2013 slope-alpha (obs_mean_tau=2.3e-3(1+z)^3.65); "
            "fake_spectra _filter_single_tau_complex(tau_thresh=1e6, thresh2=0.25); "
            "flux_power window=False spec_res=0; native k-grid (first n_k FFT bins)")
        f.attrs["tau_thresh"] = 1.0e6
        f.attrs["alpha_range"] = np.asarray(alpha_range, dtype=np.float64)
        f.attrs["k_convention"] = "angular (rad*s/km) native FFT grid, PRIYA convention"
        # Tier-C schema descriptors
        f.create_dataset("tier_c_labels",
                         data=np.array(tier_c_labels(), dtype=h5py.string_dtype()))
        f.create_dataset("tier_c_nhi_edges", data=np.asarray(FINE_NHI_EDGES, dtype=np.float64))
        f.create_dataset("param_names",
                         data=np.array(list(bec.PARAM_ORDER), dtype=h5py.string_dtype()))
        f.create_dataset("log_nhi_centres", data=log_nhi_centres)
        f.create_dataset("log_nhi_edges", data=log_nhi_edges)
        # per-row
        f.create_dataset("sim_name", data=np.array([r["sim_name"] for r in rows],
                                                    dtype=h5py.string_dtype()))
        f.create_dataset("params", data=np.stack([r["params"] for r in rows], axis=0))
        for key in _ROW_FLOAT_KEYS:
            f.create_dataset(key, data=np.array([r[key] for r in rows], dtype=np.float64))
        for key in _ROW_INT_KEYS:
            f.create_dataset(key, data=np.array([r[key] for r in rows], dtype=np.int32))
        for key in _ROW_P1D_KEYS:
            f.create_dataset(key, data=np.stack([r[key] for r in rows], axis=0))
        for key in _ROW_TIERC_KEYS:
            f.create_dataset(key, data=np.stack([r[key] for r in rows], axis=0))
        for key in _ROW_COUNT_KEYS:
            f.create_dataset(key, data=np.stack([r[key] for r in rows], axis=0).astype(np.int64))
        # per-(sim,snap) CDDF (unchanged)
        f.create_dataset("snap_sim_name", data=np.array([b["sim_name"] for b in snap_blocks],
                                                        dtype=h5py.string_dtype()))
        f.create_dataset("snap_snap", data=np.array([b["snap"] for b in snap_blocks], dtype=np.int32))
        for key in _SNAP_FLOAT_KEYS:
            f.create_dataset("snap_" + key, data=np.array([b[key] for b in snap_blocks], dtype=np.float64))
        for key in _SNAP_2D_KEYS:
            f.create_dataset("snap_" + key, data=np.stack([b[key] for b in snap_blocks], axis=0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/mfho/hcd_priya && python3 tests/test_emulator_cache_tau0.py`
Expected: `OK write_cache_tau0 v3.0 round-trip`.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_emulator_cache_tau0.py tests/test_emulator_cache_tau0.py
git commit -m "feat: tau0 cache schema v3.0 (native kfkms + 3-D Tier C + counts)"
```

---

### Task 5: `--fidelity {lf,hr}` discovery + `main`

**Files:**
- Modify: `scripts/build_emulator_cache_tau0.py:38-126,345-412` (roots, `discover_tau0_pairs`, `_default_output`, `main`)
- Test: `tests/test_emulator_cache_tau0.py` (`test_discover_tau0_pairs_includes_hires_and_is_sorted` — split into LF/HR fidelity tests)

- [ ] **Step 1: Update the discovery test for fidelity**

Replace the discovery test with two:

```python
_HCD_ROOT = "/scratch/cavestru_root/cavestru0/mfho/hcd_outputs"
_EMU_LF = "/nfs/turbo/umor-yueyingn/mfho/emu_full"
_EMU_HR = "/scratch/yueyingn_root/yueyingn0/mfho/priya/emu_full_hires_2"

def test_discover_lf_pairs_sorted_no_hires():
    pairs = bt0.discover_tau0_pairs(_HCD_ROOT, _EMU_LF, fidelity="lf")
    assert all("/hires/" not in str(p[2]) for p in pairs)
    assert pairs == sorted(pairs, key=lambda p: (p[0], p[1]))
    print(f"OK LF discovery: {len(pairs)} pairs")

def test_discover_hr_pairs_six_sims():
    pairs = bt0.discover_tau0_pairs(_HCD_ROOT, _EMU_HR, fidelity="hr")
    sims = sorted({p[0] for p in pairs})
    assert all("/hires/" in str(p[2]) for p in pairs)
    # raw tau resolves under emu_full_hires_2 (no /hires/ segment there)
    for _s, _snap, _sd, raw in pairs[:3]:
        assert raw.exists() and "emu_full_hires_2" in str(raw)
    print(f"OK HR discovery: {len(sims)} sims, {len(pairs)} pairs")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/mfho/hcd_priya && python3 tests/test_emulator_cache_tau0.py`
Expected: FAIL — `discover_tau0_pairs` has no `fidelity` kwarg.

- [ ] **Step 3: Add fidelity to discovery + roots + main**

Add fidelity roots near the top (after line 39):

```python
_LF_HCD_ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
_LF_EMU_ROOT = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")
_HR_HCD_ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/hires")
_HR_EMU_ROOT = Path("/scratch/yueyingn_root/yueyingn0/mfho/priya/emu_full_hires_2")
_N_K = {"lf": 172, "hr": 525}
```

Replace `discover_tau0_pairs` (lines 101-126) so fidelity selects exactly one root and never mixes LF+HR:

```python
def discover_tau0_pairs(hcd_root, emu_root, fidelity="lf"):
    """Return [(sim, snap, snap_dir, raw_tau_path), ...] for one fidelity.

    fidelity='lf': sims directly under hcd_root; raw under emu_root/<sim>.
    fidelity='hr': pass hcd_root=<.../hcd_outputs> -> uses hcd_root/hires; raw
    under emu_root/<sim> (emu_full_hires_2). Sorted by (sim, snap)."""
    if fidelity == "hr":
        root = Path(hcd_root) / "hires" if Path(hcd_root).name != "hires" else Path(hcd_root)
    else:
        root = Path(hcd_root)
    out = []
    for sim, snap, snap_dir in bec.discover_sim_snap_pairs(root):
        if not (snap_dir / "catalog.npz").exists():
            continue
        raw = locate_raw_tau_file(emu_root, sim, snap)
        if raw is None:
            continue
        out.append((sim, snap, snap_dir, raw))
    return out
```

Replace `_default_output` and the relevant part of `main`:

```python
def _default_output(fidelity) -> Path:
    return REPO_ROOT / "hcd_analysis" / "_emulator_data" / f"observables_tau0_{fidelity}.h5"
```

In `main`, replace the `--hcd-root`/`--emu-root` defaults and the build call:

```python
    parser.add_argument("--fidelity", choices=("lf", "hr"), default="lf")
    parser.add_argument("--hcd-root", type=Path, default=None)
    parser.add_argument("--emu-root", type=Path, default=None)
    # ... (keep --alpha-refine/--limit/--offset/--n-skewers/--output/--spot-check)
    args = parser.parse_args()
    fid = args.fidelity
    hcd_root = args.hcd_root or (_HR_HCD_ROOT.parent if fid == "hr" else _LF_HCD_ROOT)
    emu_root = args.emu_root or (_HR_EMU_ROOT if fid == "hr" else _LF_EMU_ROOT)
    n_k = _N_K[fid]
    alpha_slope_grid = make_alpha_grid_priya_aligned(refine=args.alpha_refine)
    output = args.output or _default_output(fid)
    pairs = discover_tau0_pairs(hcd_root, emu_root, fidelity=fid)
    print(f"[{fid}] Found {len(pairs)} tau0-buildable (sim, snap) pairs (n_k={n_k})")
    pairs = pairs[args.offset:]
    if args.limit is not None:
        pairs = pairs[:args.limit]
    all_rows, snap_blocks = [], []
    for i, (sim, snap, snap_dir, raw) in enumerate(pairs):
        rows, block = build_tau0_rows(sim, snap, snap_dir, raw, alpha_slope_grid,
                                      n_k=n_k, n_skewers=args.n_skewers)
        gi = len(snap_blocks)
        for r in rows:
            r["snap_group_idx"] = gi
        all_rows.extend(rows); snap_blocks.append(block)
        if (i + 1) % 10 == 0 or (i + 1) == len(pairs):
            print(f"  built {i+1}/{len(pairs)} pairs ({len(all_rows)} rows)")
    write_cache_tau0(all_rows, snap_blocks, output,
                     alpha_range=(float(alpha_slope_grid[0]), float(alpha_slope_grid[-1])),
                     n_k=n_k)
    print(f"Wrote {output} ({output.stat().st_size/1e6:.2f} MB, "
          f"{len(all_rows)} rows, {len(snap_blocks)} snaps)")
```

Delete the now-unused `from hcd_analysis.p1d import _DEFAULT_K_BINS` / `k_target` lines.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/mfho/hcd_priya && python3 tests/test_emulator_cache_tau0.py`
Expected: `OK LF discovery` and `OK HR discovery` (HR test needs the 2 new sims' catalogs from job 50612177; if Phase-1 hasn't finished, it still finds the 4 originals — assert `len(sims) >= 4`, and tighten to 6 once the job completes).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_emulator_cache_tau0.py tests/test_emulator_cache_tau0.py
git commit -m "feat: --fidelity {lf,hr} drives roots/N_K/output; HR uses emu_full_hires_2"
```

---

### Task 6: Update `merge_tau0_cache.py` for v3.0

**Files:**
- Modify: `scripts/merge_tau0_cache.py:35-110`
- Test: `tests/test_emulator_cache_tau0.py` (`test_merge_v3_synthetic`)

- [ ] **Step 1: Write the failing test (synthetic 2-shard merge)**

```python
def test_merge_v3_synthetic(tmp_path=Path("/tmp")):
    import scripts.merge_tau0_cache as mg
    nk = 172
    for si in range(2):
        rows = _fake_rows(nk, nrows=2)
        for r in rows: r["snap_group_idx"] = 0  # one snap per shard
        blocks = [_fake_snap_block(0)]
        bt0.write_cache_tau0(rows, blocks, tmp_path / f"shard_{si}.h5",
                             alpha_range=(1.0, 1.2), n_k=nk)
    out = tmp_path / "merged.h5"
    mg.merge_shards([str(tmp_path / f"shard_{i}.h5") for i in range(2)], out)
    with h5py.File(out, "r") as f:
        assert f["P_tier_p"].shape == (4, nk)
        assert f["P_tier_c"].shape[0] == 4
        assert f["kfkms"].shape == (4, nk)
        gi = f["snap_group_idx"][...]
        assert gi.tolist() == [0, 0, 1, 1]   # second shard's snap remapped
    print("OK merge v3.0 synthetic")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/mfho/hcd_priya && python3 tests/test_emulator_cache_tau0.py`
Expected: FAIL — merge references `k_target` / old `_ROW_P1D_KEYS` only.

- [ ] **Step 3: Update merge key tuples + top-level**

In `scripts/merge_tau0_cache.py`, replace lines 35-43:

```python
_TOP_LEVEL = ("tier_c_labels", "tier_c_nhi_edges", "param_names",
              "log_nhi_centres", "log_nhi_edges")
_ROW_STR = ("sim_name",)
_ROW_ARR = ("params",) + bt0._ROW_P1D_KEYS + bt0._ROW_TIERC_KEYS + bt0._ROW_COUNT_KEYS
_ROW_FLOAT = bt0._ROW_FLOAT_KEYS
_ROW_INT = bt0._ROW_INT_KEYS
_SNAP_STR = ("snap_sim_name",)
_SNAP_INT = ("snap_snap",)
_SNAP_FLOAT = tuple("snap_" + k for k in bt0._SNAP_FLOAT_KEYS)
_SNAP_2D = tuple("snap_" + k for k in bt0._SNAP_2D_KEYS)
```

In `merge_shards`, change the shard-0 consistency check (line 70) from `("k_target", ...)` to:
`for k in ("tier_c_nhi_edges", "log_nhi_centres", "log_nhi_edges"):` and bump `cache_version` to `"3.0"`, copy `n_k` from shard 0 (`f.attrs["n_k"] = first_n_k`), and update the `priya_convention`/`k_convention` attrs to the v3.0 strings (copy from `write_cache_tau0`). Concatenation logic (`np.concatenate(lst, axis=0)`) is shape-agnostic and already handles the 3-D `P_tier_c`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/mfho/hcd_priya && python3 tests/test_emulator_cache_tau0.py`
Expected: `OK merge v3.0 synthetic`.

- [ ] **Step 5: Commit**

```bash
git add scripts/merge_tau0_cache.py tests/test_emulator_cache_tau0.py
git commit -m "feat: merge_tau0_cache handles v3.0 schema (kfkms + 3-D Tier C)"
```

---

### Task 7: HR native-grid bit-identity verification (N_K=522)

**Files:**
- Test: `tests/test_emulator_cache_tau0.py` (`test_build_tau0_rows_hr_matches_priya_6sim`)

This is the HR validation anchor: a built HR row must be bit-identical to the **new 6-sim** PRIYA HR reference on the native 522-bin grid.

- [ ] **Step 1: Write the test**

```python
def test_build_tau0_rows_hr_matches_priya_6sim():
    """One HR (sim, z, alpha) row vs the new 6-sim PRIYA HR reference, native grid.
    N_K is READ from the ref (525 as of 2026-05-21), not hardcoded — we follow
    flux_power's native grid and confirm bit-identity to PRIYA's hires vectors."""
    HR_REF = "/scratch/yueyingn_root/yueyingn0/mfho/priya/emu_full_hires_2/mf_emulator_flux_vectors_tau1000000.hdf5"
    EMU_HR = "/scratch/yueyingn_root/yueyingn0/mfho/priya/emu_full_hires_2"
    HCD_BASE = "/scratch/cavestru_root/cavestru0/mfho/hcd_outputs"   # discover appends /hires for fidelity=hr
    if not Path(HR_REF).exists():
        print("SKIP HR bit-identity (6-sim ref unavailable)"); return
    pairs = bt0.discover_tau0_pairs(HCD_BASE, EMU_HR, fidelity="hr")
    if not pairs:
        print("SKIP HR bit-identity (no HR Phase-1 catalogs yet — reprocess pending)"); return
    cand = None
    for sim, snap, sd, raw in pairs:
        z = bt0._snap_z_to_priya_grid(json.load(open(sd / "meta.json"))["z"])
        if abs(z - 3.0) < 1e-6:
            cand = (sim, snap, sd, raw); break
    assert cand, "no HR z=3.0 pair"
    sim, snap, sd, raw = cand
    my_params = bt0._read_priya_params(raw)            # 9 cosmo in bec.PARAM_ORDER
    with h5py.File(HR_REF, "r") as f:
        params = f["params"][...]; fv = f["flux_vectors"][...]
        zout = f["zout"][...]
        NK = int(f["kfkms"].shape[-1])                 # 525, read from the ref
        sim_rows = np.where(np.all(np.isclose(params[:, 1:], my_params[None, :],
                                              rtol=1e-4), axis=1))[0]
        assert len(sim_rows) > 0, "HR sim not found in PRIYA ref by params"
        row = int(sim_rows[0]); alpha = float(params[row, 0])
        zidx = int(np.argmin(np.abs(zout - 3.0)))
        P_priya = fv[row, zidx*NK:(zidx+1)*NK].astype(np.float64)
        kp = f["kfkms"][row, zidx, :].astype(np.float64)
    rows, _ = bt0.build_tau0_rows(sim, snap, sd, raw,
                                  alpha_slope_grid=np.array([alpha]), n_k=NK)
    r = rows[0]
    assert r["kfkms"].shape == (NK,), f"expected {NK} k-bins, got {r['kfkms'].shape}"
    assert np.max(np.abs(r["kfkms"] / kp - 1)) < 1e-10, "HR native k-grid mismatch"
    rel = np.max(np.abs(r["P_tier_p"] / P_priya - 1))
    assert rel < 1e-4, f"HR Tier-P not bit-identical to PRIYA hires vectors: max|r-1|={rel:.3e}"
    print(f"OK HR bit-identity sim={sim[:24]} z=3.0 alpha={alpha:.4f} N_K={NK} max|r-1|={rel:.3e}")
```

> N_K is read from the ref (`kfkms.shape[-1]` = 525) — no hardcoded guess. Layout confirmed identical to the LF ref. **This test needs the HR Phase-1 catalogs from the all-6 reprocess (job 50639490); it SKIPs cleanly until they exist, so run it AFTER that job completes** (a SKIP here is acceptable pre-reprocess, but the final run must print OK). Set `_N_K["hr"]` in build_emulator_cache_tau0 to this same 525.

- [ ] **Step 2: Run test**

Run (emu-3.9 env): `cd /home/mfho/hcd_priya && $PY tests/test_emulator_cache_tau0.py`
Expected: `OK HR bit-identity ...` (or `SKIP` only if the ref is missing — a SKIP is not a pass; the ref exists as of 2026-05-21).

- [ ] **Step 3: If it fails on N_K**, the HR grid count differs from 522. Read `f["kfkms"].shape[-1]` from `HR_REF`, set `_N_K["hr"]` to that value, re-run. Commit the corrected constant with a note.

- [ ] **Step 4: Commit**

```bash
git add tests/test_emulator_cache_tau0.py scripts/build_emulator_cache_tau0.py
git commit -m "test: HR native-grid Tier-P bit-identity vs 6-sim PRIYA ref (N_K=522)"
```

---

### Task 8: Tier-C-vs-PRIYA decomposition check (forest+LLS+subDLA reproduce the filtered P1D)

Per user request 2026-05-21: verify the per-class decomposition against PRIYA's
τ=1e6-filtered P1D. The per-class P1Ds are computed on the SAME *filtered* τ as
Tier P (so the partition is exact). Two checks: (1) ALL classes count-weighted ==
Tier P (=PRIYA), exact to floating point; (2) the non-DLA-only sum
(forest+LLS+subDLA, classes 0..12) reproduces PRIYA up to the residual
`(n_DLA/N)·P_DLA^filt` (the healed-DLA power that survives filtering).
**User expectation: this non-DLA residual should be <1%** at z≈3 (the class
P1Ds come from exactly the same snapshot, so the only difference is the DLA
contribution). The test prints the residual and the n_DLA fraction; a result
>1% is a real finding to report (DLAs not negligible / may motivate storing a
filtered Tier C), not necessarily a code bug.

**Files:**
- Test: `tests/test_priya_p1d.py` (`test_tier_c_nonDLA_reproduces_priya_filtered`)

- [ ] **Step 1: Write the test**

```python
def test_tier_c_nonDLA_reproduces_priya_filtered():
    from hcd_analysis.priya_p1d import compute_tier_p_p1d, compute_tier_c_p1d
    from hcd_analysis.catalog import AbsorberCatalog
    SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
    SNAP=17; NK=172; ZIDX=8; ROW=344
    tau_p=f"/nfs/turbo/umor-yueyingn/mfho/emu_full/{SIM}/output/SPECTRA_{SNAP:03d}/lya_forest_spectra_grid_480.hdf5"
    cat_p=f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/catalog.npz"
    meta_p=f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}/meta.json"
    priya="/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/mf_emulator_flux_vectors_tau1000000.hdf5"
    if not all(os.path.exists(p) for p in (tau_p,cat_p,meta_p,priya)):
        print("SKIP tier_c_nonDLA (data unavailable)"); return
    with open(meta_p) as fh: m=json.load(fh)
    vmax=int(m["nbins"])*float(m["dv_kms"]); z=round(float(m["z"])/0.2)*0.2  # PRIYA grid z
    with h5py.File(tau_p,"r") as fh: tau=fh["tau/H/1/1215"][...].astype(np.float64)
    catalog=AbsorberCatalog.load_npz(cat_p)
    with h5py.File(priya,"r") as fh:
        alpha=float(fh["params"][ROW,0])
        P_priya=fh["flux_vectors"][ROW,ZIDX*NK:(ZIDX+1)*NK].astype(np.float64)
    # Tier P filters tau in place; reuse the SAME filtered tau for Tier C so the
    # per-class partition is exact against Tier P.
    tau_f=tau.copy()
    kf_p,P_p,tF,scale=compute_tier_p_p1d(tau_f, vmax, alpha_slope=alpha, z=z)
    assert np.max(np.abs(P_p[:NK]/P_priya-1))<1e-4, "Tier P != PRIYA (precondition)"
    kf_c,P_by_bin,n_by_bin,_,_=compute_tier_c_p1d(
        tau_f, vmax, alpha, z, catalog=catalog, external_scale=scale, external_target_F=tF)
    N=int(n_by_bin.sum())
    P_all=(n_by_bin[:,None]/N*P_by_bin).sum(0)
    P_nonDLA=(n_by_bin[:13,None]/N*P_by_bin[:13]).sum(0)   # classes 0..12 = forest+LLS+subDLA
    n_dla=int(n_by_bin[13:].sum())
    # EXACT: all classes on the filtered tau reconstruct Tier P (= PRIYA)
    assert np.allclose(P_all, P_p, rtol=1e-10), \
        f"filtered partition != Tier P, worst {np.max(np.abs(P_all/P_p-1)):.2e}"
    # DIAGNOSTIC: non-DLA classes vs PRIYA's filtered P1D
    res=np.abs(P_nonDLA[:NK]/P_priya-1)
    print(f"OK Tier-C decomposition: filtered partition EXACT; "
          f"non-DLA vs PRIYA max|r-1|={res.max():.3e} med={np.median(res):.3e} "
          f"(n_DLA={n_dla}/{N}={n_dla/N:.2%})  [target <1%]")
    assert res.max() < 0.05, f"non-DLA residual {res.max():.3e} >5% — investigate (bug or DLA-heavy)"
```

> The hard partition check (`P_all == P_p`, rtol 1e-10) is the bug-catcher. The `< 0.05` ceiling is a gross guard; the REAL result is the printed `max|r-1|` against the user's <1% target. Report the actual number: if <1%, consistent; if 1–5%, report it as a finding (healed-DLA contribution larger than expected — may motivate storing a filtered Tier C for exact non-DLA reconstruction). Per-class P1Ds here are on the FILTERED τ (Tier P filters `tau_f` in place; the same `tau_f` is reused for Tier C).

- [ ] **Step 2: Run the test**

Run (emu-3.9 env): `cd /home/mfho/hcd_priya && $PY tests/test_priya_p1d.py`
Expected: `OK Tier-C decomposition: filtered partition EXACT; non-DLA vs PRIYA max|r-1|=...`

- [ ] **Step 3: Commit**

```bash
git add tests/test_priya_p1d.py
git commit -m "test: Tier-C non-DLA classes reproduce PRIYA tau=1e6 P1D (decomposition check)"
```

---

### Task 9: Filtered Tier-C tier (exact PRIYA reconstruction from per-class)

Per user 2026-05-22: the unfiltered Tier C (for "PRIYA + HCD add-back") sums to
the *unfiltered* total, not PRIYA. To let the per-class decomposition reproduce
PRIYA exactly (and define a clean non-DLA forest), ALSO store a **filtered Tier
C**: the 15-bin per-class P1D computed on PRIYA's τ=1e6-filtered τ (the SAME
whole-array filter Tier P uses), then split by N_HI class. Its count-weighted
sum == Tier P (= PRIYA) to 1e-10. **Consistency requirement (user-flagged):** the
filter MUST be the τ-based whole-array PRIYA filter applied BEFORE the N_HI split
— never a per-N_HI-class filter (that would drop the τ-saturated subDLAs PRIYA
heals and break the identity). The classification (counts) is by N_HI and is
shared with the unfiltered tier.

**Files:** `scripts/build_emulator_cache_tau0.py` (build_tau0_rows + _ROW_TIERC_KEYS),
`tests/test_emulator_cache_tau0.py`. (merge needs no change — it composes from
`bt0._ROW_TIERC_KEYS`.) Cost: +1 per-class flux_power-equiv per α (~+50% Tier C;
reuses Tier P's already-filtered τ, no extra filtering).

- [ ] **Step 1: failing test** — in `tests/test_emulator_cache_tau0.py`, add a
  test that builds one LF row (ns0.803 snap_017, n_k=172, the bit-identity sim)
  and asserts the FILTERED Tier C reconstructs Tier P (=PRIYA) exactly:
```python
def test_filtered_tier_c_reconstructs_priya():
    SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
    SNAP, NK = 17, 172
    sd = Path(f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}")
    raw = Path(f"/nfs/turbo/umor-yueyingn/mfho/emu_full/{SIM}/output/SPECTRA_{SNAP:03d}/lya_forest_spectra_grid_480.hdf5")
    if not (sd.exists() and raw.exists()):
        print("SKIP filtered_tier_c (data unavailable)"); return
    rows, _ = bt0.build_tau0_rows(SIM, SNAP, sd, raw, np.array([1.0]), n_k=NK)
    r = rows[0]
    assert r["P_tier_c_filtered"].shape == (bt0_pp.N_TIER_C_BINS, NK)
    N = int(r["tier_c_counts"].sum())
    recon = (r["tier_c_counts"][:, None] / N * r["P_tier_c_filtered"]).sum(0)
    rel = np.max(np.abs(recon / r["P_tier_p"] - 1))
    assert rel < 1e-9, f"filtered Tier-C != Tier P (=PRIYA): max|r-1|={rel:.2e}"
    print(f"RESULT OK filtered Tier-C reconstructs PRIYA exactly: max|r-1|={rel:.2e}")
```
- [ ] **Step 2: run → fail** (`P_tier_c_filtered` KeyError).
- [ ] **Step 3:** in `build_tau0_rows`, keep `tau_filt` after Tier P (don't `del`
  it) and compute a SECOND `compute_tier_c_p1d(tau_filt, ..., external_scale=scale,
  external_target_F=target_F)` → `P_by_bin_filt, n_by_bin_filt`; assert
  `np.array_equal(n_by_bin, n_by_bin_filt)` (same N_HI classification); add row key
  `"P_tier_c_filtered": P_by_bin_filt[:, :n_k].astype(np.float64)`. Add
  `"P_tier_c_filtered"` to `_ROW_TIERC_KEYS`. Bump `cache_version` to `"3.1"` and
  extend the Tier-C docstring/labels note: `P_tier_c` = unfiltered (HCD add-back),
  `P_tier_c_filtered` = PRIYA-filtered (sums to Tier P). Update the round-trip +
  `_fake_rows` to include `P_tier_c_filtered`.
- [ ] **Step 4: run → pass** (emu-3.9 env; expect `RESULT OK filtered Tier-C
  reconstructs PRIYA exactly: max|r-1|=...e-1x`, plus the existing tests still green).
- [ ] **Step 5: commit** `feat: filtered Tier-C tier — per-class reconstructs PRIYA exactly (v3.1)`.

---

## Post-rework (not gating the schema; tracked in handover §4)

- Size + launch the production sbatch arrays per fidelity (LF ~1076 pairs; HR 6 sims) once HR Phase-1 (job 50612177) is complete. ~70 min/pair @ 20 α.
- Merge shards → `observables_tau0_lf.h5` + `observables_tau0_hr.h5`.
- Full PRIYA-overlap validation (z2.2–4.6 × the 10 PRIYA α) on both caches.

---

## Self-Review

**Spec coverage** (handover §3 rework items):
1. Native k-grid, separate LF/HR — Tasks 3 (native rows), 4 (schema), 5 (fidelity/output). ✅
2. Tier C fine 0.25-dex bins + counts — Tasks 1, 2, 4. ✅
3. DLA α-response (measure-then-correct) — out of scope for the cache; global-α baseline is what Tier P/C already produce (handover: this study informs a *later* correction). Noted, not a task here. ✅ (intentional)
4. 6-HR source + per-z pairing — Task 5 (HR roots), Task 3 (z-assert). ✅
5. Re-verify Tier-P bit-identity on native grid — Task 3 (LF 172), Task 7 (HR 522). ✅
6. Update merge for new schema — Task 6. ✅
7. Tier-C-vs-PRIYA decomposition check (user 2026-05-21: forest+LLS+subDLA reproduce τ=1e6 P1D) — Task 8. ✅

**Placeholder scan:** no TBD/"handle edge cases"; every code step has real code. Task 7 flags a verify-the-ref-layout step (legitimate — the HR ref's exact index layout is unconfirmed) with a concrete fallback.

**Type consistency:** `n_k` threaded through `build_tau0_rows`→`write_cache_tau0`/`merge`; row keys `kfkms/P_tier_p/P_tier_c/tier_c_counts` consistent across Tasks 3/4/6; `N_TIER_C_BINS`/`tier_c_labels`/`FINE_NHI_EDGES`/`merge_fine_to_classes`/`bin_sightlines_by_nhi` defined in Task 1-2 and referenced consistently after.

**Bin scheme (resolved 2026-05-21):** edges anchored exactly at 17.2/19.0/20.3 (LLS/subDLA/DLA all reconstruct exactly), linear ~0.26-dex within the LLS (7) and subDLA (5) bands, coarse DLA tail ([20.3,21.0) + ≥21.0). `N_TIER_C_BINS = 15`. Within-band bin counts (7/5) are easy to retune via `_LLS_EDGES`/`_SUBDLA_EDGES` if finer/coarser forest granularity is wanted.
