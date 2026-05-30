# Phase-2b A — Tier-C freeze-core cache fix + production τ₀ build — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the unfiltered Tier-C per-class P1D physically faithful (freeze self-shielded gas before mean-flux rescaling), store a uniform twin + per-class mean-F for diagnostics, calibrate the freeze threshold against ground truth, then build and merge the production τ₀ cache.

**Architecture:** A small, backward-compatible `tau_freeze` knob is threaded through the per-class P1D path in `hcd_analysis/priya_p1d.py`; the cache builder computes the unfiltered Tier-C **both** frozen (production) and uniform (systematic twin) and stores per-class ⟨F⟩; a calibration step pins `τ_freeze` against a fake_spectra re-extraction; then a sharded SLURM array builds the cache and `merge_tau0_cache.py` stitches it.

**Tech Stack:** Python 3.9 (`emu-3.9` conda env for anything importing `hcd_analysis.priya_p1d`; mamba py3.11 for discovery-only), `fake_spectra` v2.2.3, numpy/h5py, SLURM (account `cavestru0`).

**Scope:** This is **Phase A only** (cache). The emulator model+loss (Phase B) and likelihood+validation (Phase C) get separate plans; the loader can be TDD'd against a synthetic fixture in parallel once the v3.1+frozen schema here is locked.

### ⚠️ CRITICAL env (every test/build step that imports `hcd_analysis.priya_p1d`)
```bash
export LD_LIBRARY_PATH=/sw/pkgs/arc/stacks/gcc/10.3.0/gsl/2.7/lib:/home/mfho/.conda/envs/emu-3.9/lib:$LD_LIBRARY_PATH
PY=/home/mfho/.conda/envs/emu-3.9/bin/python3   # do NOT `conda activate`
```
A SKIP / ImportError is NOT a pass.

### Key references
- Spec: `docs/superpowers/specs/2026-05-29-phase2b-emulator-design.md` (§2a, §8).
- Code: `hcd_analysis/priya_p1d.py` (`_per_class_p1d_at_scale` L111-130, `compute_tier_c_p1d` L133-166, `build_tau0_rows`/`write_cache_tau0` in `scripts/build_emulator_cache_tau0.py` L211-384).
- Existing freeze helper (legacy, alpha-based): `hcd_analysis/tau0_rescale.py:37`.
- Data: LF raw τ `/nfs/turbo/umor-yueyingn/mfho/emu_full/<sim>/output/SPECTRA_NNN/lya_forest_spectra_grid_480.hdf5`; Phase-1 `/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/<sim>/snap_NNN/`.

---

## File structure

- **Modify** `hcd_analysis/priya_p1d.py` — add scale-based `tau_freeze` to `_per_class_p1d_at_scale` and `compute_tier_c_p1d`; add module constant `TAU_FREEZE_TIERC`.
- **Modify** `scripts/build_emulator_cache_tau0.py` — `build_tau0_rows` computes frozen + uniform unfiltered Tier-C and per-class ⟨F⟩; `write_cache_tau0` stores the new datasets; bump `cache_version`.
- **Modify** `scripts/merge_tau0_cache.py` — carry the new per-row keys.
- **Modify** `tests/test_priya_p1d.py` — freeze-core unit tests.
- **Modify** `tests/test_emulator_cache_tau0.py` — schema round-trip for the frozen tier + per-class ⟨F⟩.
- **Create** `scripts/calibrate_tau_freeze.py` — τ_freeze scan vs ground truth.
- **Create** `scripts/spot_check_tau0_fake_spectra.py` — fake_spectra re-extraction at genuinely scaled Γ (ground truth) + a feasibility check.
- **Create** `scripts/batch_tau0_production.sh` — sharded production array.

---

## Task 1: Scale-based freeze-core in `_per_class_p1d_at_scale`

**Files:**
- Modify: `hcd_analysis/priya_p1d.py:111-130`
- Test: `tests/test_priya_p1d.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_priya_p1d.py`:

```python
def test_per_class_freeze_core_thin_and_frozen():
    """tau_freeze=inf reproduces the uniform rescale exactly; a finite
    tau_freeze leaves frozen pixels at native exp(-tau) and scales the rest."""
    import numpy as np
    from hcd_analysis.priya_p1d import _per_class_p1d_at_scale
    rng = np.random.default_rng(0)
    # one core pixel (tau huge) + thin forest pixels
    tau = rng.uniform(0.0, 5.0, size=(64, 128))
    tau[:, 0] = 1.0e7  # saturated core column
    vmax, scale, tF = 1000.0, 0.8, 0.7

    kf_u, P_u = _per_class_p1d_at_scale(tau, vmax, scale, tF)                 # uniform
    kf_i, P_i = _per_class_p1d_at_scale(tau, vmax, scale, tF, tau_freeze=np.inf)
    assert np.allclose(P_u, P_i, rtol=0, atol=0), "tau_freeze=inf must equal uniform"

    # with a finite freeze below the core, the core pixel must NOT scale:
    kf_f, P_f = _per_class_p1d_at_scale(tau, vmax, scale, tF, tau_freeze=1.0e4)
    # reconstruct expected flux for row 0 manually
    chunk = tau[:64]
    tau_eff = np.where(chunk > 1.0e4, chunk, scale * chunk)
    assert np.all(tau_eff[:, 0] == tau[:64, 0]), "core frozen at native tau"
    assert np.allclose(tau_eff[:, 1:], scale * chunk[:, 1:]), "thin pixels scaled"
    assert not np.allclose(P_f, P_u), "freeze must change the per-class P1D"
    print("OK freeze-core: inf==uniform, finite freezes cores")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export LD_LIBRARY_PATH=/sw/pkgs/arc/stacks/gcc/10.3.0/gsl/2.7/lib:/home/mfho/.conda/envs/emu-3.9/lib:$LD_LIBRARY_PATH; /home/mfho/.conda/envs/emu-3.9/bin/python3 -c "import tests.test_priya_p1d as t; t.test_per_class_freeze_core_thin_and_frozen()"`
Expected: FAIL — `TypeError: _per_class_p1d_at_scale() got an unexpected keyword argument 'tau_freeze'`.

- [ ] **Step 3: Write minimal implementation**

In `hcd_analysis/priya_p1d.py`, change the signature and the `dflux` line:

```python
def _per_class_p1d_at_scale(tau_class: np.ndarray, vmax: float,
                             scale: float, target_F: float,
                             tau_freeze: float = np.inf):
    """Mirror fake_spectra.fluxstatistics.flux_power but with predetermined
    (scale, target_F) — used by Tier C so all four classes share Tier P's
    mean-flux normalisation. `tau_freeze`: pixels with native tau > tau_freeze
    keep their native optical depth (self-shielded, UVB-insensitive); the rest
    are rescaled by `scale`. tau_freeze=inf reproduces the uniform rescale.
    Returns (kf[1:], P[1:]) like flux_power does."""
    nspec, npix = tau_class.shape
    if nspec == 0:
        kf = _flux_power_bins(vmax, npix)
        return kf[1:], np.zeros(npix // 2 + 1)[1:]
    mfp = np.zeros(npix // 2 + 1, dtype=tau_class.dtype)
    for i in range(10):
        end = min((i + 1) * nspec // 10, nspec)
        s = i * nspec // 10
        if end == s:
            continue
        chunk = tau_class[s:end]
        tau_eff = np.where(chunk > tau_freeze, chunk, scale * chunk)
        dflux = np.exp(-tau_eff) / target_F - 1.0
        mfp += vmax * np.sum(_powerspectrum(dflux, axis=1), axis=0)
    mfp /= nspec
    kf = _flux_power_bins(vmax, npix)
    return kf[1:], mfp[1:]
```

- [ ] **Step 4: Run test to verify it passes**

Run: same command as Step 2.
Expected: `OK freeze-core: inf==uniform, finite freezes cores`.

- [ ] **Step 5: Commit**

```bash
git add hcd_analysis/priya_p1d.py tests/test_priya_p1d.py
git commit -m "feat(tier-c): scale-based tau_freeze in _per_class_p1d_at_scale (inf=uniform)"
```

---

## Task 2: Thread `tau_freeze` through `compute_tier_c_p1d` + define `TAU_FREEZE_TIERC`

**Files:**
- Modify: `hcd_analysis/priya_p1d.py` (`compute_tier_c_p1d` L133-166; add a constant near `FINE_NHI_EDGES`)
- Test: `tests/test_priya_p1d.py`

- [ ] **Step 1: Write the failing test**

```python
def test_compute_tier_c_freeze_passthrough():
    """compute_tier_c_p1d forwards tau_freeze; default inf == current behavior."""
    import numpy as np
    from hcd_analysis.priya_p1d import compute_tier_c_p1d, N_TIER_C_BINS

    class _Ab:  # minimal absorber
        def __init__(self, idx, nhi): self.skewer_idx, self.log_NHI = idx, nhi
    class _Cat:
        absorbers = [_Ab(0, 20.5), _Ab(1, 18.0)]  # sightline0=DLA, sightline1=LLS
    rng = np.random.default_rng(1)
    tau = rng.uniform(0.0, 5.0, size=(4, 64)); tau[0, 0] = 1e7
    kw = dict(vmax=1000.0, alpha_slope=1.0, z=3.0, catalog=_Cat(),
              external_scale=0.8, external_target_F=0.7)
    _, P_inf, n_inf, _, _ = compute_tier_c_p1d(tau, **kw)                    # default inf
    _, P_frz, n_frz, _, _ = compute_tier_c_p1d(tau, tau_freeze=1e4, **kw)
    assert P_inf.shape == (N_TIER_C_BINS, 64 // 2)
    assert np.array_equal(n_inf, n_frz), "counts are tau_freeze-invariant"
    assert not np.allclose(P_inf, P_frz), "freeze changes the per-bin P1D"
    print("OK compute_tier_c_p1d tau_freeze passthrough")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export LD_LIBRARY_PATH=...; $PY -c "import tests.test_priya_p1d as t; t.test_compute_tier_c_freeze_passthrough()"`
Expected: FAIL — unexpected keyword `tau_freeze`.

- [ ] **Step 3: Write minimal implementation**

In `hcd_analysis/priya_p1d.py`, add the constant after `FINE_NHI_EDGES`/`N_TIER_C_BINS`:

```python
# Self-shielding freeze threshold for the UNFILTERED Tier-C path. NOT the PRIYA
# 1e6 trough-fill threshold (that is the *filtered* tier). Default value pinned
# by scripts/calibrate_tau_freeze.py; override per-call. See spec 2026-05-29 §2a.
TAU_FREEZE_TIERC = 1.0e4
```

Change `compute_tier_c_p1d`'s signature to accept `tau_freeze: float = np.inf` and forward it:

```python
def compute_tier_c_p1d(tau: np.ndarray, vmax: float,
                       alpha_slope: float, z: float,
                       catalog,
                       external_scale: Optional[float] = None,
                       external_target_F: Optional[float] = None,
                       tau_freeze: float = np.inf,
                       ):
    ...
    for c in range(N_TIER_C_BINS):
        mask = (cls == c)
        kf, P = _per_class_p1d_at_scale(tau[mask], vmax, scale, target_F,
                                        tau_freeze=tau_freeze)
        ...
```

(Leave the rest of the body unchanged. Update the docstring to mention `tau_freeze`.)

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: `OK compute_tier_c_p1d tau_freeze passthrough`.

- [ ] **Step 5: Re-run the existing priya_p1d suite to confirm no regression**

Run: `export LD_LIBRARY_PATH=...; $PY tests/test_priya_p1d.py`
Expected: all existing tests still pass (Tier-P bit-identity, fine-bin sum at α=1, filtered reconstruction) — the defaults (`tau_freeze=inf`) preserve current behavior.

- [ ] **Step 6: Commit**

```bash
git add hcd_analysis/priya_p1d.py tests/test_priya_p1d.py
git commit -m "feat(tier-c): tau_freeze passthrough + TAU_FREEZE_TIERC constant"
```

---

## Task 3: Cache builder stores frozen + uniform unfiltered Tier-C + per-class ⟨F⟩

**Files:**
- Modify: `scripts/build_emulator_cache_tau0.py` (`build_tau0_rows` L211-298, `write_cache_tau0` L310-383, row-key tuples L301-307)
- Modify: `scripts/merge_tau0_cache.py` (key lists)
- Test: `tests/test_emulator_cache_tau0.py`

- [ ] **Step 1: Write the failing test (schema round-trip)**

Add to `tests/test_emulator_cache_tau0.py` (reuse the file's synthetic-row helpers; if none, build a 2-row dict matching `_ROW_*`):

```python
def test_v32_frozen_tier_roundtrip(tmp_path=None):
    """write_cache_tau0 stores P_tier_c_frozen[15,nk] and mean_F_by_bin[15];
    round-trips; cache_version bumped to 3.2."""
    import numpy as np, h5py, tempfile, os
    import scripts.build_emulator_cache_tau0 as bt0
    nk = bt0._N_K["lf"]
    def _row(a):
        return dict(sim_name="nsTEST", snap=1, alpha_slope=float(a), alpha_idx=a,
                    z_meta=3.0, z_grid=3.0, dv_kms=10.0, nbins_native=2*nk,
                    snap_group_idx=0, target_F=0.7, scale=0.8,
                    params=np.zeros(9), kfkms=np.linspace(1e-3,0.1,nk),
                    P_tier_p=np.ones(nk), P_tier_c=np.ones((15,nk)),
                    P_tier_c_frozen=2*np.ones((15,nk)),
                    P_tier_c_filtered=np.ones((15,nk)),
                    tier_c_counts=np.arange(15),
                    mean_F_by_bin=np.linspace(0.5,0.9,15))
    rows = [_row(0), _row(1)]
    snap_blocks = [dict(sim_name="nsTEST", snap=1, f_nhi=np.ones(30),
                        n_absorbers=np.ones(30), log_nhi_centres=np.zeros(30),
                        log_nhi_edges=np.zeros(31), total_path_dX=1.0,
                        dNdX_LLS=0.1, dNdX_subDLA=0.05, dNdX_DLA=0.02)]
    out = os.path.join(tempfile.mkdtemp(), "c.h5")
    bt0.write_cache_tau0(rows, snap_blocks, out, alpha_range=(0.0,1.0), n_k=nk)
    with h5py.File(out, "r") as f:
        assert f.attrs["cache_version"] == "3.2"
        assert f["P_tier_c_frozen"].shape == (2, 15, nk)
        assert f["mean_F_by_bin"].shape == (2, 15)
        assert np.allclose(f["P_tier_c_frozen"][0], 2.0)
    print("OK v3.2 frozen-tier + mean_F round-trip")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export LD_LIBRARY_PATH=...; $PY -c "import tests.test_emulator_cache_tau0 as t; t.test_v32_frozen_tier_roundtrip()"`
Expected: FAIL — `KeyError`/missing dataset `P_tier_c_frozen`.

- [ ] **Step 3: Implement — `build_tau0_rows` computes the frozen tier + ⟨F⟩**

In `scripts/build_emulator_cache_tau0.py`, inside `build_tau0_rows`, after the existing unfiltered Tier-C call (the `compute_tier_c_p1d(tau_unfilt, ...)` block, ~L266), add the frozen variant and the per-class ⟨F⟩, and add the new keys to the appended row dict:

```python
        from hcd_analysis.priya_p1d import TAU_FREEZE_TIERC
        # Frozen unfiltered Tier C (PRODUCTION HCD add-back): self-shielded gas
        # held at native tau; thin pixels scaled. tau_freeze pinned by calibration.
        _, P_by_bin_frozen, _, _, _ = compute_tier_c_p1d(
            tau_unfilt, vmax, alpha_slope=alpha, z=z_grid, catalog=catalog,
            external_scale=scale, external_target_F=target_F,
            tau_freeze=TAU_FREEZE_TIERC)
        # Per-class <F> on the frozen field (for the shared-vs-per-class target_F
        # disentanglement; computed once via the same freeze recipe).
        cls = bin_sightlines_by_nhi(catalog, tau_unfilt.shape[0])
        mean_F_by_bin = np.array([
            float(np.mean(np.exp(-np.where(tau_unfilt[cls == c] > TAU_FREEZE_TIERC,
                                           tau_unfilt[cls == c],
                                           scale * tau_unfilt[cls == c]))))
            if (cls == c).any() else np.nan
            for c in range(N_TIER_C_BINS)])
```

Add `bin_sightlines_by_nhi` and `N_TIER_C_BINS` to the existing `from hcd_analysis.priya_p1d import ...` line at the top of `build_tau0_rows`. Then add to the `rows.append({...})` dict:

```python
            "P_tier_c_frozen": P_by_bin_frozen[:, :n_k].astype(np.float64),
            "mean_F_by_bin": mean_F_by_bin.astype(np.float64),
```

- [ ] **Step 4: Implement — register the new datasets in `write_cache_tau0`**

Update the key tuples (L303-305 region):

```python
_ROW_TIERC_KEYS = ("P_tier_c", "P_tier_c_frozen", "P_tier_c_filtered")  # 3-D (n_rows, 15, n_k)
_ROW_MEANF_KEYS = ("mean_F_by_bin",)                                    # 2-D (n_rows, 15)
```

In `write_cache_tau0`, bump the version and write the new 2-D block:

```python
        f.attrs["cache_version"] = "3.2"
        f.attrs["tau_freeze_tierc"] = float(bt0_tau_freeze())  # see note below
```

(Define a tiny helper or import `TAU_FREEZE_TIERC` and write it directly: `f.attrs["tau_freeze_tierc"] = float(__import__("hcd_analysis.priya_p1d", fromlist=["TAU_FREEZE_TIERC"]).TAU_FREEZE_TIERC)`.)

Add a loop to write `_ROW_MEANF_KEYS` next to the existing `_ROW_COUNT_KEYS` loop:

```python
        for key in _ROW_MEANF_KEYS:
            f.create_dataset(key, data=np.stack([r[key] for r in rows], axis=0))
```

(`_ROW_TIERC_KEYS` already loops via the existing `for key in _ROW_TIERC_KEYS` block — adding `P_tier_c_frozen` to the tuple is sufficient.)

- [ ] **Step 5: Update the merger**

In `scripts/merge_tau0_cache.py`, ensure it composes keys from `bt0._ROW_TIERC_KEYS` and add `bt0._ROW_MEANF_KEYS` wherever `_ROW_COUNT_KEYS` is consumed, and carry `cache_version="3.2"` + the `tau_freeze_tierc` attr.

- [ ] **Step 6: Run the schema test to verify it passes**

Run: `export LD_LIBRARY_PATH=...; $PY -c "import tests.test_emulator_cache_tau0 as t; t.test_v32_frozen_tier_roundtrip()"`
Expected: `OK v3.2 frozen-tier + mean_F round-trip`.

- [ ] **Step 7: Run the full tau0 cache suite (no regression on the existing anchors)**

Run: `export LD_LIBRARY_PATH=...; $PY tests/test_emulator_cache_tau0.py`
Expected: existing anchors still pass (native-grid Tier-P bit-identity, filtered Tier-C reconstructs PRIYA ~8e-15, HR bit-identity, discovery counts, v3.x merge) + the new round-trip.

- [ ] **Step 8: Commit**

```bash
git add scripts/build_emulator_cache_tau0.py scripts/merge_tau0_cache.py tests/test_emulator_cache_tau0.py
git commit -m "feat(cache): v3.2 stores frozen unfiltered Tier-C + per-class <F> (uniform kept as twin)"
```

---

## Task 4: Calibrate `τ_freeze` against fake_spectra ground truth

**Files:**
- Create: `scripts/spot_check_tau0_fake_spectra.py` (ground-truth re-extraction at genuinely scaled Γ)
- Create: `scripts/calibrate_tau_freeze.py` (τ_freeze scan vs ground truth)

- [ ] **Step 1: Confirm ground-truth feasibility FIRST (gating)**

The ground truth = re-running `fake_spectra` on the particle snapshots at a genuinely scaled UV rate Γ (not a τ post-rescale). Confirm the PART snapshots exist and a re-extraction runs:
Run: `ls /scratch/yueyingn_root/yueyingn0/mfho/priya/PART/emu_full/ 2>/dev/null | head` and locate one (sim, z≈3) PART snapshot.
Expected: at least one usable PART snapshot. **If PART data is unavailable/infeasible**, STOP and record that the ground-truth gate cannot run; fall back to the interim calibration in Step 4 (internal self-consistency + physical priors) and flag the spot-check as deferred in the cache attrs. Surface this to the user before launching the production build.

- [ ] **Step 2: Write `spot_check_tau0_fake_spectra.py`**

For one (sim, z≈3): extract per-class P1D with `fake_spectra` at the native Γ and at Γ·k for a few k matching the α grid (genuinely scaled UVB → re-derived neutral fractions → new τ). Save `P_c^truth(α)` per class to an `.npz`. (Use the existing `fake_spectra` extraction path used by Phase-1; reuse `AbsorberCatalog` for classification on the native field — classification is τ₀-invariant.)

- [ ] **Step 3: Write `calibrate_tau_freeze.py`**

For the same (sim, z): for `τ_freeze ∈ {1e3, 3e3, 1e4, 3e4, 1e5}`, compute the frozen unfiltered Tier-C per-class P1D (via `compute_tier_c_p1d(..., tau_freeze=τ)`) at the matching α, and compare each to `P_c^truth(α)` from Step 2. Report, per class and per τ_freeze, the rms fractional residual vs ground truth across k and α. Pick the `τ_freeze` minimizing the HCD-class (subDLA+DLA) residual.

- [ ] **Step 4: Run the calibration and set `TAU_FREEZE_TIERC`**

Run (emu-3.9 env): `$PY scripts/calibrate_tau_freeze.py`
Expected: a table of residual vs τ_freeze; a clear minimum. Set `TAU_FREEZE_TIERC` in `hcd_analysis/priya_p1d.py` to the winning value.
**Interim fallback (if Step 1 failed):** set `TAU_FREEZE_TIERC` at the Rahmati+2013 self-shielding onset (map `n_{H,SSh}(z=3)` to per-pixel τ; document the mapping + that it is uncalibrated), and keep the uniform twin so the choice is revisitable.

- [ ] **Step 5: Commit**

```bash
git add scripts/spot_check_tau0_fake_spectra.py scripts/calibrate_tau_freeze.py hcd_analysis/priya_p1d.py
git commit -m "feat(calib): tau_freeze calibration vs fake_spectra ground truth; pin TAU_FREEZE_TIERC"
```

---

## Task 5: 1-pair timing run to size the production array

**Files:** (none new; uses `build_emulator_cache_tau0.py`)

- [ ] **Step 1: Time one LF pair end-to-end with the frozen tier**

Run a single (sim, snap) at the full α grid via a short sbatch (account `cavestru0`, `standard`, `--mem=24G`, `--cpus-per-task=8`, `--time=04:00:00`):
```bash
$PY scripts/build_emulator_cache_tau0.py --fidelity lf --offset 0 --limit 1 \
    --output /tmp/timing_one_pair.h5 --spot-check
```
Record `sacct -j <id> --format=Elapsed,AllocCPUS,TotalCPU,MaxRSS`.

- [ ] **Step 2: Compute the array size**

From the per-pair Elapsed + MaxRSS: confirm per-pair wall (~2 h expected with the frozen tier), set `--mem` to MaxRSS×1.3, choose shard width so total wall on the array is ≤1–2 days (LF 1072 + HR 103 pairs). Record the numbers in a comment block at the top of `batch_tau0_production.sh` (Task 6).

- [ ] **Step 3: (no commit — measurement only; record numbers in Task 6 script)**

---

## Task 6: Production sharded SLURM array (gated on Tasks 4 + 5)

**Files:**
- Create: `scripts/batch_tau0_production.sh`

- [ ] **Step 1: Write `batch_tau0_production.sh`**

A `#SBATCH --array=` sharded driver: each task processes a contiguous shard via `--offset/--limit`, writing `observables_tau0_lf.shardNNN.h5`. Parameterize `--fidelity` (lf|hr). Include the env preamble (LD_LIBRARY_PATH + emu-3.9 python), `--account=cavestru0`, `--partition=standard`, `--mem` and `--cpus-per-task` from Task 5, `--time` per the per-pair estimate × shard size with margin. Mirror the structure of `scripts/batch_lf_rerun_all.sh`.

- [ ] **Step 2: Dry-run one shard**

Run the array with a 1-shard, `--limit 2` smoke test; confirm output shard files appear with `cache_version=3.2` and `P_tier_c_frozen` present.

- [ ] **Step 3: Launch the full LF + HR arrays**

`sbatch scripts/batch_tau0_production.sh` (LF), then the HR variant. Record job IDs. (SLURM emails on completion; the next session must check — no in-session watcher survives.)

- [ ] **Step 4: Commit the script**

```bash
git add scripts/batch_tau0_production.sh
git commit -m "infra: sharded production tau0 cache array (v3.2 frozen Tier-C)"
```

---

## Task 7: Merge + validate the production cache

**Files:** (uses `merge_tau0_cache.py`)

- [ ] **Step 1: Wait for the arrays, re-submit any FAILED shards**

`squeue -j <id>` empty + `sacct -j <id> -n --format=State | sort | uniq -c` all COMPLETED. Re-run failed shards.

- [ ] **Step 2: Merge**

```bash
$PY scripts/merge_tau0_cache.py --fidelity lf --inputs '<shard glob>' \
    --output hcd_analysis/_emulator_data/observables_tau0_lf.h5
```
(and HR → `_hr.h5`). Expected: `cache_version=3.2`, `n_rows` ≈ 1072×20 (LF), `tau_freeze_tierc` attr set.

- [ ] **Step 3: Validation anchors**

Run `tests/test_emulator_cache_tau0.py` against the production cache: Tier-P bit-identity vs PRIYA, filtered Tier-C reconstructs PRIYA ~8e-15, the freeze-vs-uniform systematic (`P_tier_c_frozen` vs `P_tier_c`) is nonzero and concentrated in the HCD classes, `snap_group_idx` contiguous.

- [ ] **Step 4: Commit a short build report + update the handover**

```bash
git add docs/  # build report noting tau_freeze value, job IDs, anchor results
git commit -m "docs: production tau0 cache (v3.2) build report + anchors"
```

---

## Self-review notes (spec coverage)

- Spec §2a items 1–4 (freeze-core, τ_freeze≠1e6, uniform twin, per-class ⟨F⟩) → Tasks 1–4.
- Spec §8 fake_spectra ground-truth spot-checks (gating) + freeze-vs-uniform systematic → Tasks 4 + 7.3.
- Production build + merge (§11 prerequisite) → Tasks 5–7.
- Phases B (model/loss) and C (likelihood/validation) are intentionally separate plans (scope check); the loader can be TDD'd against the v3.2 schema fixture once Task 3 lands.
- Risk carried: PART-data feasibility for the ground truth (Task 4 Step 1 gates it; interim fallback documented).
```
