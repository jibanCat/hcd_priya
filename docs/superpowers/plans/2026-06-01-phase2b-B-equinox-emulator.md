# Phase-2b B — Equinox HCD P1D + CDDF Emulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the JAX/Equinox emulator (shared encoder + Head A + Head B), its NaN-safe masked loss, the dN/dX→w_c coupling, the training loop with k-fold LOSO + static error vector, and the total-P1D likelihood contract — all TDD'd against a synthetic v3.3 cache fixture so most of it lands before the production cache merge.

**Architecture:** `params(9)⊕z → encoder[10→256→128→64] → latent`; Head A (latent-only branch) → `f_nhi[30]`+`dN/dX[3]` (τ₀-invariant); Head B (`concat(latent,τ₀)`) → 4 filtered class P1D + 3 sign-safe HCD deltas `Δ_c`. `P_tier_p` is the **structural** sum `Σ_c w_c·P_c^filt` (not a free output); `w_c` is **derived** from `dN/dX` via the diagonal telescoping-Poisson map `M₀` + a JAX-pure `δ_c(z)` polynomial. The whole `dN/dX→w_c→P_obs` path is one differentiable `jit`.

**Tech Stack:** Python 3.9+, JAX (CUDA jaxlib on the `cavestru0` GPU allocation), Equinox, optax, numpy, h5py. Tests use pytest + `jax.grad` finite-gradient assertions.

**Spec:** `docs/superpowers/specs/2026-05-29-phase2b-emulator-design.md` (§3–§10). **Coupling note:** `docs/superpowers/2026-05-29-wc-dndx-poisson-coupling.md`. This plan is Phase B + C of that spec; Phase A (the v3.3 cache) is DONE.

**Cache schema (v3.3, the loader's input).** Per-row datasets (R rows = sim×snap×α): `params[9]`, `alpha_slope`, `alpha_idx`, `target_F`, `scale`, `z_meta`, `z_grid`, `dv_kms`, `nbins_native`, `snap_group_idx`, `sim_name`, `kfkms[172]`, `P_tier_p[172]`, `P_tier_c[15,172]` (unfiltered), `P_tier_c_filtered[15,172]`, `tier_c_counts[15]`, `mean_F_by_bin[15]`. Per snap-block (G groups, τ₀-invariant): `snap_f_nhi[30]`, `snap_n_absorbers[30]`, `snap_total_path_dX`, `snap_dNdX_{LLS,subDLA,DLA}`, `snap_sim_name`, `snap_snap`. Attrs include `n_k`, `cache_version='3.3'`, `tier_c_nhi_edges[14]`, `log_nhi_edges[31]`. (Verified against `observables_tau0_lf.shard000.h5`.)

**Class collapse 15→4** (`merge_fine_to_classes` convention): clean=index 0; LLS=1–7; subDLA=8–12; DLA=13–14. **τ₀ coordinate:** `τ₀ = −ln(target_F)` per row.

---

## File Structure

- Create `hcd_analysis/emulator/__init__.py` — package marker, exports.
- Create `hcd_analysis/emulator/data.py` — cache loader: read v3.3 → arrays; 15→4 collapse; τ₀; native-Nyquist NaN masks; analytic `1/n_c` weights; k-fold LOSO + τ₀-edge holdout; train-split normalisation stats + transforms.
- Create `hcd_analysis/emulator/dndx_wc.py` — `M₀` telescoping-Poisson `w_c(dN/dX)`; JAX-pure `δ_c(z)` polynomial; corrected `w_c`; class-specific incidence prior forms.
- Create `hcd_analysis/emulator/model.py` — Equinox `Encoder`, `HeadA`, `HeadB`, `Emulator`; structural `P_tier_p`; sign-safe `Δ_c` transform; the joint NaN-safe loss.
- Create `hcd_analysis/emulator/train.py` — optax loop, LR schedule, early stop, seeding, checkpoint bundle, k-fold LOSO driver, static error vector.
- Create `hcd_analysis/emulator/likelihood.py` — §6 total-P1D contract: difference form (default) + ratio toggle, covariance assembly, closure test helper.
- Create `tests/emulator/_fixture.py` — synthetic v3.3 cache builder (writes a tiny valid .h5).
- Create `tests/test_emulator_data.py`, `tests/test_emulator_dndx_wc.py`, `tests/test_emulator_model.py`, `tests/test_emulator_likelihood.py`.
- Modify `scripts/build_emulator_cache_tau0.py` — add the class-edge source-of-truth assertion (Task 17; referee hardening).

**Conventions:** numpy for the loader/IO (host side); JAX (`jnp`) for everything inside the differentiable path (model, loss, `dndx_wc`, likelihood). The loader returns numpy; the train step converts to `jnp`. Keep `dndx_wc.w_c_from_dndx` JAX-pure (no python branching on traced values).

**Env activation (all test/run commands assume).** ⚠️ `PYTHONNOUSERSITE=1` is
MANDATORY — without it the interpreter silently borrows stale packages (numpy<2,
h5py) from `~/.local`. See `docs/superpowers/2026-06-01-emu-jax-env.md`.
```bash
export EMUJAX=/home/mfho/.conda/envs/emu-jax/bin/python3
PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya $EMUJAX -m pytest <test> -v
```

---

## Task 0: emu-jax GPU environment + package skeleton

**Files:**
- Create: `hcd_analysis/emulator/__init__.py`
- Create: `docs/superpowers/2026-06-01-emu-jax-env.md` (env provenance note)

- [ ] **Step 1: Create the GPU conda env on cavestru0.** JAX CUDA wheels bundle CUDA; pick the jaxlib CUDA build matching Great Lakes' driver (check `nvidia-smi` on a GPU node first via `srun -A cavestru0 -p gpu --gres=gpu:1 --pty nvidia-smi`).

```bash
conda create -y -n emu-jax python=3.11
conda activate emu-jax
# CUDA 12 wheels (adjust to the cluster driver; CPU fallback: pip install -U "jax[cpu]")
pip install -U "jax[cuda12]" equinox optax h5py numpy
python -c "import jax; print('devices:', jax.devices())"
```
Expected: prints a `gpu`/`cuda` device on a GPU node (or `cpu` on a login node — that's fine for the TDD tasks; only Task 13 training needs the GPU).

- [ ] **Step 2: Record env provenance.** Write `docs/superpowers/2026-06-01-emu-jax-env.md` with: the exact `pip freeze | grep -Ei 'jax|equinox|optax|numpy|h5py'` output, the CUDA wheel chosen, the `nvidia-smi` driver/CUDA version, and a one-line note "GPU jobs run under SLURM account cavestru0, partition gpu — cavestru0 budget is tight (~4000 CPU-h equiv); profile before sweeps (see memory cavestru0-compute-budget)."

- [ ] **Step 3: Create the package marker.**

```python
# hcd_analysis/emulator/__init__.py
"""Phase-2b JAX/Equinox HCD P1D + CDDF emulator.

See docs/superpowers/specs/2026-05-29-phase2b-emulator-design.md.
"""
CLASS_NAMES = ("clean", "LLS", "subDLA", "DLA")  # coarse 4-class order
N_CLASSES = 4
HCD_CLASSES = ("LLS", "subDLA", "DLA")  # the 3 with a non-zero Delta_c
```

- [ ] **Step 4: Verify import.**

Run: `PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3 -c "import hcd_analysis.emulator as e; print(e.CLASS_NAMES, e.N_CLASSES)"`
Expected: `('clean', 'LLS', 'subDLA', 'DLA') 4`

- [ ] **Step 5: Commit.**

```bash
git add hcd_analysis/emulator/__init__.py docs/superpowers/2026-06-01-emu-jax-env.md
git commit -m "infra(phase2b): emu-jax GPU env + emulator package skeleton"
```

---

## Task 1: Synthetic v3.3 cache fixture (TDD enabler)

A tiny valid v3.3 cache so the loader, model, and likelihood can be TDD'd before the real merge. The fixture must satisfy the structural identity `Σ_c (counts/N)·P_filt == P_tier_p` and carry NaN bins above each row's Nyquist, because the loader's masking + the model's structural sum are tested against it.

**Files:**
- Create: `tests/emulator/__init__.py` (empty)
- Create: `tests/emulator/_fixture.py`
- Test: `tests/test_emulator_data.py` (first test below)

- [ ] **Step 1: Write the failing test.**

```python
# tests/test_emulator_data.py
import h5py, numpy as np
from tests.emulator._fixture import write_synthetic_cache

def test_fixture_is_valid_v33(tmp_path):
    path = tmp_path / "obs.h5"
    write_synthetic_cache(path, n_sims=3, snaps_per_sim=2, n_alpha=4, n_k=8)
    with h5py.File(path, "r") as h:
        assert h.attrs["cache_version"] == "3.3"
        R = h["P_tier_p"].shape[0]
        assert R == 3 * 2 * 4
        cnt = h["tier_c_counts"][:].astype(float)          # (R,15)
        Pf = h["P_tier_c_filtered"][:]                      # (R,15,n_k)
        Pp = h["P_tier_p"][:]                               # (R,n_k)
        w = cnt / cnt.sum(1, keepdims=True)
        recon = np.einsum("rc,rck->rk", w, Pf)
        ok = np.isfinite(Pp)
        assert np.allclose(recon[ok], Pp[ok], rtol=0, atol=1e-12)
        # NaNs above Nyquist present in at least one row
        assert np.isnan(Pp).any()
```

- [ ] **Step 2: Run to verify it fails.**

Run: `PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_emulator_data.py::test_fixture_is_valid_v33 -v`
Expected: FAIL — `ModuleNotFoundError: tests.emulator._fixture`.

- [ ] **Step 3: Implement the fixture.** Build per-class filtered P1Ds, set `P_tier_p` = count-weighted sum exactly, set unfiltered = filtered + a small signed delta, and NaN out bins above a per-row Nyquist cut.

```python
# tests/emulator/_fixture.py
import h5py, numpy as np

PARAM_NAMES = ["ns","Ap","herei","heref","alphaq","hub","omegamh2","hireionz","bhfeedback"]

def write_synthetic_cache(path, n_sims=3, snaps_per_sim=2, n_alpha=4, n_k=8, seed=0):
    rng = np.random.default_rng(seed)
    G = n_sims * snaps_per_sim                 # snap-blocks
    R = G * n_alpha                            # rows
    P_tier_c_filt = np.empty((R, 15, n_k))
    P_tier_c_unfilt = np.empty((R, 15, n_k))
    counts = np.empty((R, 15), dtype=np.int64)
    P_tier_p = np.empty((R, n_k))
    kf = np.tile(np.linspace(1e-3, 0.1, n_k), (R, 1))
    params = np.empty((R, 9)); alpha_slope = np.empty(R); alpha_idx = np.empty(R, np.int32)
    target_F = np.empty(R); scale = np.empty(R); z_meta = np.empty(R); z_grid = np.empty(R)
    dv_kms = np.full(R, 7.0); nbins_native = np.empty(R, np.int32); group_idx = np.empty(R, np.int32)
    sim_name = np.empty(R, object)
    snap_f_nhi = np.empty((G, 30)); snap_n_abs = np.empty((G, 30), np.int64)
    snap_path = np.empty(G); snap_sim = np.empty(G, object); snap_snap = np.empty(G, np.int32)
    snap_dndx = {c: np.empty(G) for c in ("LLS","subDLA","DLA")}
    alpha_grid = np.linspace(0.66, 1.33, n_alpha)
    r = 0
    for s in range(n_sims):
        p = rng.uniform(0.5, 1.5, 9)
        for j in range(snaps_per_sim):
            g = s * snaps_per_sim + j
            z = 2.0 + 0.4 * j
            cnt15 = np.array([700, 60,40,30,20,15,10,8, 12,9,7,5,4, 3,2], np.int64)  # clean-heavy
            snap_f_nhi[g] = rng.uniform(1e-22, 1e-20, 30)
            snap_n_abs[g] = rng.integers(0, 50, 30)
            snap_path[g] = 1000.0; snap_sim[g] = f"sim{s}"; snap_snap[g] = j
            for c in snap_dndx: snap_dndx[c][g] = rng.uniform(0.01, 0.4)
            nyq = n_k - (g % 2)   # alternate rows lose the top bin above Nyquist
            for a in range(n_alpha):
                base = rng.uniform(0.5, 2.0, (15, n_k))
                P_tier_c_filt[r] = base
                P_tier_c_unfilt[r] = base + rng.uniform(-0.05, 0.05, (15, n_k))  # signed delta
                counts[r] = cnt15
                w = cnt15 / cnt15.sum()
                Pp = np.einsum("c,ck->k", w, base)
                if nyq < n_k:
                    Pp[nyq:] = np.nan
                    P_tier_c_filt[r, :, nyq:] = np.nan
                    P_tier_c_unfilt[r, :, nyq:] = np.nan
                P_tier_p[r] = Pp
                params[r] = p; alpha_slope[r] = alpha_grid[a]; alpha_idx[r] = a
                z_meta[r] = z; z_grid[r] = round(z/0.2)*0.2; nbins_native[r] = 100 + g
                target_F[r] = np.exp(-alpha_grid[a] * 2.3e-3 * (1+z)**3.65)
                scale[r] = 1.0; group_idx[r] = g; sim_name[r] = f"sim{s}"
                r += 1
    with h5py.File(path, "w") as h:
        h.attrs.update(cache_version="3.3", n_k=n_k, n_rows=R, n_snaps=G,
                       tau_thresh=1e6, tier_c_recipe="uniform")
        h["P_tier_c_filtered"] = P_tier_c_filt; h["P_tier_c"] = P_tier_c_unfilt
        h["tier_c_counts"] = counts; h["P_tier_p"] = P_tier_p; h["kfkms"] = kf
        h["params"] = params; h["alpha_slope"] = alpha_slope; h["alpha_idx"] = alpha_idx
        h["target_F"] = target_F; h["scale"] = scale; h["z_meta"] = z_meta; h["z_grid"] = z_grid
        h["dv_kms"] = dv_kms; h["nbins_native"] = nbins_native; h["snap_group_idx"] = group_idx
        h.create_dataset("sim_name", data=np.array(sim_name, dtype=object),
                         dtype=h5py.string_dtype())
        h.create_dataset("param_names", data=np.array(PARAM_NAMES, dtype=object),
                         dtype=h5py.string_dtype())
        h["snap_f_nhi"] = snap_f_nhi; h["snap_n_absorbers"] = snap_n_abs
        h["snap_total_path_dX"] = snap_path; h["snap_snap"] = snap_snap
        h.create_dataset("snap_sim_name", data=np.array(snap_sim, dtype=object),
                         dtype=h5py.string_dtype())
        for c in snap_dndx: h[f"snap_dNdX_{c}"] = snap_dndx[c]
```

- [ ] **Step 4: Run to verify it passes.**

Run: `PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_emulator_data.py::test_fixture_is_valid_v33 -v`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add tests/emulator/__init__.py tests/emulator/_fixture.py tests/test_emulator_data.py
git commit -m "test(phase2b): synthetic v3.3 cache fixture for emulator TDD"
```

---

## Task 2: Loader — read v3.3, 15→4 collapse, τ₀, masks, 1/n_c weights

**Files:**
- Create: `hcd_analysis/emulator/data.py`
- Test: `tests/test_emulator_data.py`

- [ ] **Step 1: Write the failing test.**

```python
# append to tests/test_emulator_data.py
from hcd_analysis.emulator.data import load_cache, COARSE_SLICES

def test_load_collapses_15_to_4_and_builds_tau0_masks_weights(tmp_path):
    path = tmp_path / "obs.h5"; write_synthetic_cache(path, n_k=8)
    d = load_cache(path)
    R = d["P_tier_p"].shape[0]
    assert d["P_filt"].shape == (R, 4, 8)     # 4 coarse classes
    assert d["delta"].shape == (R, 3, 8)      # 3 HCD deltas (LLS, subDLA, DLA)
    # tau0 = -ln(target_F)
    assert np.allclose(d["tau0"], -np.log(d["target_F"]))
    # coarse counts: clean = fine[0]; LLS = fine[1:8]; subDLA = fine[8:13]; DLA = fine[13:15]
    assert COARSE_SLICES == (slice(0,1), slice(1,8), slice(8,13), slice(13,15))
    # mask is True where finite
    assert d["mask"].shape == (R, 8)
    assert d["mask"].dtype == bool
    assert (d["mask"] == np.isfinite(d["P_tier_p"])).all()
    # analytic 1/n_c weights, NaN-safe (0 weight for empty class)
    assert d["inv_nc"].shape == (R, 4)
    assert np.all(np.isfinite(d["inv_nc"]))
```

- [ ] **Step 2: Run to verify it fails.**

Run: `... -m pytest tests/test_emulator_data.py::test_load_collapses_15_to_4_and_builds_tau0_masks_weights -v`
Expected: FAIL — `ModuleNotFoundError: hcd_analysis.emulator.data`.

- [ ] **Step 3: Implement the loader.**

```python
# hcd_analysis/emulator/data.py
"""v3.3 cache loader for the Phase-2b emulator. Returns numpy arrays (host side).

15->4 coarse-class collapse, tau0 coordinate, native-Nyquist NaN masks, and the
analytic 1/n_c sample-variance weights. See spec sec.4, sec.7, sec.9.
"""
from __future__ import annotations
import h5py, numpy as np

# Fine 15-bin -> coarse 4-class (clean, LLS, subDLA, DLA). See spec sec.2.
COARSE_SLICES = (slice(0, 1), slice(1, 8), slice(8, 13), slice(13, 15))
COARSE_NAMES = ("clean", "LLS", "subDLA", "DLA")

def _collapse_counts(counts15):                      # (R,15) -> (R,4)
    return np.stack([counts15[:, s].sum(1) for s in COARSE_SLICES], axis=1)

def _collapse_p1d(p15, counts15):
    """Count-weighted collapse of fine-bin P1D to coarse classes (R,15,K)->(R,4,K).

    A coarse class P1D is the count-weighted mean of its fine bins (each fine
    P1D is already normalised by its own count). An empty coarse class (zero
    counts) collapses to 0.0 (the cache stores finite zeros for empty classes;
    this is REQUIRED so the structural sum Sum_c w_c*P_filt stays finite, since
    w_c=0 there and 0*NaN would poison the total). A bin that is NaN across ALL
    fine sub-bins (above native Nyquist) stays NaN for masking.
    """
    R, _, K = p15.shape
    out = np.full((R, 4, K), np.nan)
    for ci, s in enumerate(COARSE_SLICES):
        c = counts15[:, s].astype(float)             # (R, nf)
        w = c / np.where(c.sum(1, keepdims=True) == 0, 1.0, c.sum(1, keepdims=True))
        seg = p15[:, s, :]                           # (R, nf, K)
        # treat NaN fine-bin P1D as 0 contribution but keep all-NaN -> NaN
        contrib = np.where(np.isfinite(seg), seg, 0.0) * w[:, :, None]
        summ = contrib.sum(1)
        allnan = (~np.isfinite(seg)).all(1)          # (R,K) all fine bins NaN
        summ[allnan] = np.nan
        out[:, ci, :] = summ
    return out

def load_cache(path):
    with h5py.File(path, "r") as h:
        assert h.attrs["cache_version"] == "3.3", h.attrs.get("cache_version")
        g = lambda k: h[k][:]
        counts15 = g("tier_c_counts")
        Pf15, Pu15 = g("P_tier_c_filtered"), g("P_tier_c")
        out = dict(
            params=g("params").astype(np.float64),
            kfkms=g("kfkms").astype(np.float64),
            P_tier_p=g("P_tier_p").astype(np.float64),
            target_F=g("target_F").astype(np.float64),
            scale=g("scale").astype(np.float64),
            z_grid=g("z_grid").astype(np.float64),
            z_meta=g("z_meta").astype(np.float64),
            alpha_idx=g("alpha_idx").astype(np.int32),
            snap_group_idx=g("snap_group_idx").astype(np.int32),
            sim_name=np.array([s.decode() if isinstance(s, bytes) else s
                               for s in h["sim_name"][:]]),
            snap_dNdX=np.stack([g("snap_dNdX_LLS"), g("snap_dNdX_subDLA"),
                                g("snap_dNdX_DLA")], axis=1),   # (G,3) class order LLS,subDLA,DLA
            snap_f_nhi=g("snap_f_nhi").astype(np.float64),
            snap_total_path_dX=g("snap_total_path_dX").astype(np.float64),
            snap_n_absorbers=g("snap_n_absorbers").astype(np.float64),
        )
    coarse_counts = _collapse_counts(counts15)       # (R,4)
    Pf4 = _collapse_p1d(Pf15, counts15)              # (R,4,K)
    Pu4 = _collapse_p1d(Pu15, counts15)
    out["P_filt"] = Pf4
    out["delta"] = (Pu4 - Pf4)[:, 1:, :]             # (R,3,K): HCD classes only (clean Delta=0)
    out["coarse_counts"] = coarse_counts
    out["tau0"] = -np.log(out["target_F"])
    out["mask"] = np.isfinite(out["P_tier_p"])       # (R,K) native-Nyquist mask
    N = coarse_counts.sum(1, keepdims=True)
    out["w_c_cache"] = coarse_counts / N             # counted sightline weights (ground truth)
    # analytic 1/n_c (NaN-safe: empty class -> 0 weight, never inf)
    out["inv_nc"] = np.where(coarse_counts > 0, 1.0 / np.maximum(coarse_counts, 1), 0.0)
    return out
```

- [ ] **Step 4: Run to verify it passes.**

Run: `... -m pytest tests/test_emulator_data.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit.**

```bash
git add hcd_analysis/emulator/data.py tests/test_emulator_data.py
git commit -m "feat(phase2b): v3.3 loader — 15->4 collapse, tau0, Nyquist masks, 1/n_c weights"
```

---

## Task 3: Transforms + train-split normalisation

Channel transforms (log for `f_nhi`/`dN/dX`/filtered P1D; `arcsinh`/signed-log for `Δ_c`), standardised by **training-split** mean/std in transformed space. Stats are stored for the checkpoint and inverted at predict time.

**Files:**
- Modify: `hcd_analysis/emulator/data.py`
- Test: `tests/test_emulator_data.py`

- [ ] **Step 1: Write the failing test.**

```python
# append to tests/test_emulator_data.py
from hcd_analysis.emulator.data import signed_log, signed_log_inv, fit_norm, apply_norm

def test_signed_log_roundtrip_through_zero():
    x = np.array([-5.0, -1e-9, 0.0, 1e-9, 3.0])
    assert np.allclose(signed_log_inv(signed_log(x)), x, atol=1e-12)

def test_norm_uses_only_train_rows(tmp_path):
    x = np.arange(20.0).reshape(10, 2)
    stats = fit_norm(x, train_idx=np.arange(5))     # first 5 rows only
    assert np.allclose(stats["mean"], x[:5].mean(0))
    z = apply_norm(x, stats)
    assert np.allclose(z[:5].mean(0), 0.0, atol=1e-9)
```

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL — names not defined.

- [ ] **Step 3: Implement.**

```python
# append to hcd_analysis/emulator/data.py
def signed_log(x):                 # sign-safe transform for Delta_c (smooth through 0)
    return np.arcsinh(x)
def signed_log_inv(y):
    return np.sinh(y)
def safe_log(x, floor=1e-30):
    return np.log(np.maximum(x, floor))

def fit_norm(x, train_idx):
    xt = x[train_idx]
    mean = np.nanmean(xt, axis=0)
    std = np.nanstd(xt, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return {"mean": mean, "std": std}

def apply_norm(x, stats):
    return (x - stats["mean"]) / stats["std"]

def invert_norm(z, stats):
    return z * stats["std"] + stats["mean"]
```

- [ ] **Step 4: Run to verify it passes.** Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add hcd_analysis/emulator/data.py tests/test_emulator_data.py
git commit -m "feat(phase2b): channel transforms + train-split normalisation"
```

---

## Task 4: k-fold LOSO splits + τ₀-edge holdout

Rotate held-out sims so every sim is held out once (~8 folds); within a fold, mark whole (sim,snap) validation blocks (no α-sibling leakage); and a separate **τ₀-edge** holdout for the extrapolation probe (hold the τ₀ extremes, not raw α). See spec §7.

**Files:**
- Modify: `hcd_analysis/emulator/data.py`
- Test: `tests/test_emulator_data.py`

- [ ] **Step 1: Write the failing test.**

```python
# append to tests/test_emulator_data.py
from hcd_analysis.emulator.data import kfold_loso, tau0_edge_holdout

def test_kfold_loso_every_sim_held_once(tmp_path):
    path = tmp_path / "obs.h5"; write_synthetic_cache(path, n_sims=6, snaps_per_sim=2, n_alpha=4)
    d = load_cache(path)
    folds = kfold_loso(d["sim_name"], n_folds=3)
    held = set()
    for tr, va in folds:
        assert set(tr).isdisjoint(va)               # no row in both
        held |= set(d["sim_name"][va])
    assert held == set(d["sim_name"])               # every sim held out exactly once across folds

def test_tau0_edge_holdout_picks_extremes(tmp_path):
    path = tmp_path / "obs.h5"; write_synthetic_cache(path, n_sims=4, snaps_per_sim=3, n_alpha=4)
    d = load_cache(path)
    tr, ho = tau0_edge_holdout(d["tau0"], frac=0.2)
    assert d["tau0"][ho].min() <= d["tau0"][tr].min()   # holdout reaches the low edge
    assert d["tau0"][ho].max() >= d["tau0"][tr].max()   # and the high edge
```

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL — names not defined.

- [ ] **Step 3: Implement.**

```python
# append to hcd_analysis/emulator/data.py
def kfold_loso(sim_name, n_folds=8):
    """Return [(train_row_idx, val_row_idx), ...]; sims partitioned into n_folds
    disjoint groups, each group held out as validation once."""
    sims = np.array(sorted(set(sim_name)))
    groups = np.array_split(sims, n_folds)
    all_idx = np.arange(len(sim_name))
    folds = []
    for held in groups:
        if len(held) == 0:
            continue
        is_val = np.isin(sim_name, held)
        folds.append((all_idx[~is_val], all_idx[is_val]))
    return folds

def tau0_edge_holdout(tau0, frac=0.15):
    """Hold out the low+high tau0 tails (the extrapolation probe in tau0-space)."""
    order = np.argsort(tau0)
    k = max(1, int(round(frac * len(tau0) / 2)))
    ho = np.concatenate([order[:k], order[-k:]])
    tr = np.setdiff1d(np.arange(len(tau0)), ho)
    return tr, ho
```

- [ ] **Step 4: Run to verify it passes.** Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add hcd_analysis/emulator/data.py tests/test_emulator_data.py
git commit -m "feat(phase2b): k-fold LOSO + tau0-edge holdout splits"
```

---

## Task 5: dndx_wc — M₀ telescoping-Poisson + sum-to-1

The diagonal map `w_c(dN/dX)` (coupling note §"The math"). Class order **(clean, LLS, subDLA, DLA)**; `dN/dX` input is the 3 HCD classes; `μ_c = (dN/dX)_c · X̄`, `X̄ = total_path_dX / N_sl`.

**Files:**
- Create: `hcd_analysis/emulator/dndx_wc.py`
- Test: `tests/test_emulator_dndx_wc.py`

- [ ] **Step 1: Write the failing test.**

```python
# tests/test_emulator_dndx_wc.py
import numpy as np, jax.numpy as jnp
from hcd_analysis.emulator.dndx_wc import w_c_from_mu

def test_wc_sums_to_one_and_matches_paper_point():
    # coupling note single point: dN/dX LLS=0.360 subDLA=0.100 DLA=0.050, X_bar s.t. mu=dndx*Xbar
    # use the note's mu directly (Xbar=0.642): mu = [0.231, 0.064, 0.032]
    mu = jnp.array([0.231, 0.064, 0.032])           # (LLS, subDLA, DLA)
    w = w_c_from_mu(mu)                              # (clean, LLS, subDLA, DLA)
    assert abs(float(w.sum()) - 1.0) < 1e-12
    # paper w_poisson: clean 0.72121, LLS 0.18715, subDLA 0.06027, DLA 0.03136
    assert np.allclose(np.array(w), [0.72121, 0.18715, 0.06027, 0.03136], atol=5e-4)
```

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL — module not found.

- [ ] **Step 3: Implement.**

```python
# hcd_analysis/emulator/dndx_wc.py
"""dN/dX -> w_c diagonal telescoping-Poisson map (M0) + JAX-pure delta_c(z).

See docs/superpowers/2026-05-29-wc-dndx-poisson-coupling.md. JAX-pure: no python
branching on traced values, so the whole dN/dX -> w_c -> P_obs path is one jit.
Class order throughout: (clean, LLS, subDLA, DLA); mu/dN-dX inputs are the 3 HCD
classes in order (LLS, subDLA, DLA).
"""
from __future__ import annotations
import jax.numpy as jnp

def w_c_from_mu(mu):
    """mu: (...,3) mean absorbers/sightline for (LLS, subDLA, DLA). Returns (...,4)."""
    mu_LLS, mu_sub, mu_DLA = mu[..., 0], mu[..., 1], mu[..., 2]
    w_DLA = 1.0 - jnp.exp(-mu_DLA)
    w_sub = (1.0 - jnp.exp(-mu_sub)) * jnp.exp(-mu_DLA)
    w_LLS = (1.0 - jnp.exp(-mu_LLS)) * jnp.exp(-(mu_sub + mu_DLA))
    w_clean = jnp.exp(-(mu_LLS + mu_sub + mu_DLA))
    return jnp.stack([w_clean, w_LLS, w_sub, w_DLA], axis=-1)

def mu_from_dndx(dndx, Xbar):
    """dndx: (...,3) per-class incidence; Xbar: (...) mean path per sightline."""
    return dndx * Xbar[..., None]
```

- [ ] **Step 4: Run to verify it passes.** Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add hcd_analysis/emulator/dndx_wc.py tests/test_emulator_dndx_wc.py
git commit -m "feat(phase2b): M0 telescoping-Poisson w_c(dN/dX) + sum-to-1 test"
```

---

## Task 6: dndx_wc — JAX-pure δ_c(z) correction + finite-grad

The diagonal map is ~2.5% biased; a small `δ_c(z)` correction (frozen polynomial in z) brings it to <0.5%. **Calibrate** the coefficients from `scripts/diag_crossclass_coupling.py` output (real catalogs), then freeze them as compile-time constants. The finite-grad test pins the whole `dN/dX→w_c` path differentiable.

**Files:**
- Modify: `hcd_analysis/emulator/dndx_wc.py`
- Create: `scripts/calibrate_delta_c.py`
- Test: `tests/test_emulator_dndx_wc.py`

- [ ] **Step 1: Calibrate δ_c(z) from real catalogs (produces the frozen coeffs).** Extend `diag_crossclass_coupling.py`'s residual table into a fit. Run across ≥3 sims spanning cosmology (referee R2 #4 wanted >1 sim) and fit `δ_c(z) = w_counted/w_poisson − 1` with a low-order polynomial in z.

```python
# scripts/calibrate_delta_c.py  (run with emu-3.9 env; reads catalogs + cache)
# Emits coeffs to docs/superpowers/2026-06-01-delta_c-coeffs.md and prints a dict.
# For each (sim,snap,z): counted w_c from catalog max-class; poisson w_c from M0.
# delta_c(z) = counted/poisson - 1 ; np.polyfit(z, delta_c, deg=2) per class.
# Assert max|residual after correction| < 0.5% across all sampled (sim,z).
```
Record the resulting per-class deg-2 coefficients (clean/LLS/subDLA/DLA) in a note. **Do not invent them** — they come from this run.

- [ ] **Step 2: Write the failing test.**

```python
# append to tests/test_emulator_dndx_wc.py
import jax
from hcd_analysis.emulator.dndx_wc import w_c_corrected

def test_delta_c_correction_keeps_sum_to_one_and_is_differentiable():
    dndx = jnp.array([0.36, 0.10, 0.05]); Xbar = jnp.array(0.642); z = jnp.array(3.0)
    w = w_c_corrected(dndx, Xbar, z)
    assert w.shape == (4,)
    assert abs(float(w.sum()) - 1.0) < 1e-10            # renormalised after correction
    # whole dN/dX -> w path is differentiable (finite grads, no NaN)
    g = jax.grad(lambda d: w_c_corrected(d, Xbar, z)[3])(dndx)   # d w_DLA / d dndx
    assert jnp.all(jnp.isfinite(g))
```

- [ ] **Step 3: Run to verify it fails.** Expected: FAIL — `w_c_corrected` not defined.

- [ ] **Step 4: Implement** (substitute the calibrated coeffs from Step 1).

```python
# append to hcd_analysis/emulator/dndx_wc.py
# Frozen delta_c(z) deg-2 coeffs from scripts/calibrate_delta_c.py (REPLACE with run output).
# Order (clean, LLS, subDLA, DLA); each row = (a2, a1, a0) for a2*z^2+a1*z+a0.
_DELTA_C_COEFFS = jnp.array([
    [0.0, 0.0, 0.0],   # clean   <- fill from calibration
    [0.0, 0.0, 0.0],   # LLS
    [0.0, 0.0, 0.0],   # subDLA
    [0.0, 0.0, 0.0],   # DLA
])

def delta_c(z):
    zz = jnp.stack([z**2, z, jnp.ones_like(z)], axis=-1)   # (...,3)
    return zz @ _DELTA_C_COEFFS.T                           # (...,4)

def w_c_corrected(dndx, Xbar, z):
    w0 = w_c_from_mu(mu_from_dndx(dndx, Xbar))             # (...,4)
    w = w0 * (1.0 + delta_c(z))
    return w / w.sum(axis=-1, keepdims=True)               # renormalise to sum-to-1
```

- [ ] **Step 5: Run to verify it passes.** Expected: PASS. **Commit.**

```bash
git add hcd_analysis/emulator/dndx_wc.py scripts/calibrate_delta_c.py tests/test_emulator_dndx_wc.py docs/superpowers/2026-06-01-delta_c-coeffs.md
git commit -m "feat(phase2b): JAX-pure delta_c(z) correction (calibrated) + finite-grad test"
```

---

## Task 7: dndx_wc — class-specific incidence prior forms (from sbird/dla_data)

The likelihood marginalises `dN/dX_c(z)=A_c·(1+z)^{γ_c}` (DLA), a broken-power-law/turnover (LLS), Crighton+2015-style (subDLA). **Pull exact forms + fiducial `(A_c,γ_c)` + prior widths from `sbird/dla_data` / the papers — no invented numbers** (spec §5).

**Files:**
- Modify: `hcd_analysis/emulator/dndx_wc.py`
- Test: `tests/test_emulator_dndx_wc.py`

- [ ] **Step 1: Locate the reference numbers.** `pip show sbird` / find the `dla_data` module; read its dN/dX fiducials (PW14 DLA, Crighton+2015 subDLA, LLS turnover). If `sbird/dla_data` is not importable in emu-jax, read the published values from the papers and cite them in `docs/superpowers/2026-06-01-dndx-priors.md`. Record source + value + width for each class.

- [ ] **Step 2: Write the failing test** (use the actual fiducials from Step 1; placeholder values shown — replace).

```python
# append to tests/test_emulator_dndx_wc.py
from hcd_analysis.emulator.dndx_wc import dndx_DLA_pw14, INCIDENCE_PRIORS

def test_dndx_dla_powerlaw_and_priors_present():
    # PW14 power law dN/dX_DLA(z) = A*(1+z)^gamma — exact A,gamma from sbird/dla_data
    A, gamma = INCIDENCE_PRIORS["DLA"]["fiducial"]
    z = jnp.array([2.0, 3.0, 4.0])
    out = dndx_DLA_pw14(z, A, gamma)
    assert out.shape == (3,)
    assert jnp.all(out > 0)
    for c in ("LLS", "subDLA", "DLA"):
        assert "fiducial" in INCIDENCE_PRIORS[c] and "width" in INCIDENCE_PRIORS[c]
        assert "source" in INCIDENCE_PRIORS[c]      # provenance recorded, not invented
```

- [ ] **Step 3: Run to verify it fails.** Expected: FAIL.

- [ ] **Step 4: Implement** (fill `INCIDENCE_PRIORS` from Step 1's sourced numbers).

```python
# append to hcd_analysis/emulator/dndx_wc.py
def dndx_DLA_pw14(z, A, gamma):
    return A * (1.0 + z) ** gamma

# REPLACE fiducial/width with the values sourced in Step 1 (docs/.../2026-06-01-dndx-priors.md).
INCIDENCE_PRIORS = {
    "DLA":    {"form": "powerlaw", "fiducial": (None, None), "width": None, "source": "PW14 / sbird.dla_data"},
    "subDLA": {"form": "crighton2015", "fiducial": None, "width": None, "source": "Crighton+2015"},
    "LLS":    {"form": "broken_powerlaw", "fiducial": None, "width": None, "source": "Ribaudo+2011 / DESI"},
}
```

- [ ] **Step 5: Run to verify it passes.** Expected: PASS. **Commit.**

```bash
git add hcd_analysis/emulator/dndx_wc.py tests/test_emulator_dndx_wc.py docs/superpowers/2026-06-01-dndx-priors.md
git commit -m "feat(phase2b): class-specific dN/dX incidence prior forms (sourced, not invented)"
```

---

## Task 8: model — Equinox encoder + Head A branch (τ₀-invariant)

**Files:**
- Create: `hcd_analysis/emulator/model.py`
- Test: `tests/test_emulator_model.py`

- [ ] **Step 1: Write the failing test.**

```python
# tests/test_emulator_model.py
import jax, jax.numpy as jnp, equinox as eqx
from hcd_analysis.emulator.model import Encoder, HeadA

def test_encoder_headA_shapes_and_tau0_independence():
    key = jax.random.PRNGKey(0)
    enc = Encoder(in_dim=10, key=key)               # params(9)+z
    ha = HeadA(latent=64, n_k_cddf=30, key=key)
    x = jnp.zeros(10)
    lat = enc(x); assert lat.shape == (64,)
    out = ha(lat)
    assert out["f_nhi"].shape == (30,)
    assert out["dndx"].shape == (3,)                # LLS, subDLA, DLA
    # Head A takes only latent (no tau0) -> structurally tau0-invariant by construction
```

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL — module not found.

- [ ] **Step 3: Implement.**

```python
# hcd_analysis/emulator/model.py
"""Equinox emulator: encoder + Head A (tau0-invariant) + Head B (tau0-dependent).

Head A consumes ONLY the latent (no tau0) so its outputs (f_nhi, dN/dX) are
tau0-invariant by construction (spec sec.3). Head B consumes concat(latent, tau0).
P_tier_p is the structural sum Sum_c w_c * P_c^filt (spec sec.3/sec.6).
"""
from __future__ import annotations
import jax, jax.numpy as jnp, equinox as eqx

class Encoder(eqx.Module):
    layers: list
    def __init__(self, in_dim=10, widths=(256, 128, 64), key=None):
        ks = jax.random.split(key, len(widths)); dims = [in_dim, *widths]
        self.layers = [eqx.nn.Linear(dims[i], dims[i+1], key=ks[i]) for i in range(len(widths))]
    def __call__(self, x):
        for lin in self.layers[:-1]:
            x = jax.nn.gelu(lin(x))
        return self.layers[-1](x)                   # latent (no final activation)

class HeadA(eqx.Module):
    trunk: eqx.nn.Linear
    cddf: eqx.nn.Linear
    dndx: eqx.nn.Linear
    def __init__(self, latent=64, n_k_cddf=30, key=None):
        k1, k2, k3 = jax.random.split(key, 3)
        self.trunk = eqx.nn.Linear(latent, 64, key=k1)
        self.cddf = eqx.nn.Linear(64, n_k_cddf, key=k2)
        self.dndx = eqx.nn.Linear(64, 3, key=k3)
    def __call__(self, latent):
        h = jax.nn.gelu(self.trunk(latent))
        return {"f_nhi": self.cddf(h), "dndx": self.dndx(h)}   # transformed (log) space
```

- [ ] **Step 4: Run to verify it passes.** Expected: PASS. **Commit.**

```bash
git add hcd_analysis/emulator/model.py tests/test_emulator_model.py
git commit -m "feat(phase2b): Equinox encoder + tau0-invariant Head A branch"
```

---

## Task 9: model — Head B + structural P_tier_p + sign-safe Δ_c

**Files:**
- Modify: `hcd_analysis/emulator/model.py`
- Test: `tests/test_emulator_model.py`

- [ ] **Step 1: Write the failing test.**

```python
# append to tests/test_emulator_model.py
from hcd_analysis.emulator.model import HeadB, Emulator, structural_tier_p

def test_headB_outputs_and_structural_tier_p():
    key = jax.random.PRNGKey(1)
    hb = HeadB(latent=64, n_k=8, key=key)
    lat = jnp.zeros(64); tau0 = jnp.array(0.3)
    out = hb(lat, tau0)
    assert out["P_filt"].shape == (4, 8)            # 4 classes (clean,LLS,subDLA,DLA) transformed
    assert out["delta"].shape == (3, 8)             # HCD deltas, sign-safe space
    # structural total: Sum_c w_c * P_filt (linear-space P_filt, linear w)
    w = jnp.array([0.7, 0.18, 0.07, 0.05]); P_filt_lin = jnp.ones((4, 8))
    tp = structural_tier_p(w, P_filt_lin)
    assert tp.shape == (8,)
    assert jnp.allclose(tp, w.sum())                # all-ones P -> sum of weights

def test_emulator_endtoend_runs():
    key = jax.random.PRNGKey(2)
    m = Emulator(in_dim=10, n_k=8, key=key)
    pred = m(jnp.zeros(10), tau0=jnp.array(0.3))
    assert set(pred) >= {"f_nhi", "dndx", "P_filt", "delta"}
```

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL.

- [ ] **Step 3: Implement.**

```python
# append to hcd_analysis/emulator/model.py
class HeadB(eqx.Module):
    trunk: eqx.nn.Linear
    out: eqx.nn.Linear
    n_k: int
    def __init__(self, latent=64, n_k=172, key=None):
        k1, k2 = jax.random.split(key, 2); self.n_k = n_k
        self.trunk = eqx.nn.Linear(latent + 1, 256, key=k1)
        self.out = eqx.nn.Linear(256, 7 * n_k, key=k2)   # 4 filtered class P1D + 3 deltas
    def __call__(self, latent, tau0):
        h = jax.nn.gelu(self.trunk(jnp.concatenate([latent, jnp.atleast_1d(tau0)])))
        y = self.out(h).reshape(7, self.n_k)
        return {"P_filt": y[:4], "delta": y[4:]}     # P_filt in log-space, delta in arcsinh-space

class Emulator(eqx.Module):
    enc: Encoder
    head_a: HeadA
    head_b: HeadB
    def __init__(self, in_dim=10, n_k=172, key=None):
        k1, k2, k3 = jax.random.split(key, 3)
        self.enc = Encoder(in_dim=in_dim, key=k1)
        self.head_a = HeadA(latent=64, key=k2)
        self.head_b = HeadB(latent=64, n_k=n_k, key=k3)
    def __call__(self, x, tau0):
        lat = self.enc(x)
        return {**self.head_a(lat), **self.head_b(lat, tau0)}

def structural_tier_p(w_c, P_filt_lin):
    """w_c: (...,4); P_filt_lin: (...,4,K) LINEAR space. Returns (...,K)."""
    return jnp.einsum("...c,...ck->...k", w_c, P_filt_lin)
```

- [ ] **Step 4: Run to verify it passes.** Expected: PASS. **Commit.**

```bash
git add hcd_analysis/emulator/model.py tests/test_emulator_model.py
git commit -m "feat(phase2b): Head B (filtered P1D + sign-safe Delta) + structural P_tier_p"
```

---

## Task 10: loss — NaN-safe masked loss + finite-grad test (the load-bearing test)

**Files:**
- Modify: `hcd_analysis/emulator/model.py`
- Test: `tests/test_emulator_model.py`

- [ ] **Step 1: Write the failing test (NaN targets must yield finite grads).**

```python
# append to tests/test_emulator_model.py
from hcd_analysis.emulator.model import masked_mse

def test_masked_mse_nan_safe_gradients():
    pred = jnp.array([1.0, 2.0, 3.0, 4.0])
    targ = jnp.array([1.0, jnp.nan, 3.0, jnp.nan])      # NaN above Nyquist
    mask = jnp.isfinite(targ)
    val, grad = jax.value_and_grad(lambda p: masked_mse(p, targ, mask))(pred)
    assert jnp.isfinite(val)
    assert jnp.all(jnp.isfinite(grad))                  # the double-where NaN-grad trap must not bite
    assert grad[1] == 0.0 and grad[3] == 0.0            # masked elements contribute no gradient
```

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL — `masked_mse` not defined.

- [ ] **Step 3: Implement (sanitise BEFORE masking — spec §4).**

```python
# append to hcd_analysis/emulator/model.py
def masked_mse(pred, target, mask, weight=None):
    """NaN-safe masked MSE. Sanitise target to finite BEFORE the masked diff so the
    jnp.where double-NaN-gradient trap never fires; guard the denominator with max(n,1)."""
    target_safe = jnp.nan_to_num(target, nan=0.0)
    diff = jnp.where(mask, pred - target_safe, 0.0)
    sq = diff ** 2
    if weight is not None:
        sq = sq * weight
    denom = jnp.maximum(jnp.sum(mask.astype(sq.dtype)), 1.0)
    return jnp.sum(sq) / denom
```

- [ ] **Step 4: Run to verify it passes.** Expected: PASS. **Commit.**

```bash
git add hcd_analysis/emulator/model.py tests/test_emulator_model.py
git commit -m "feat(phase2b): NaN-safe masked MSE + finite-gradient test"
```

---

## Task 11: loss — joint loss (1/n_c, per-element term weights, α-multiplicity, clean mean-flux)

Assemble the single joint scalar (spec §4): Head A (`f_nhi`,`dN/dX`) reweighted by `1/n_α(sim,snap)`; Head B per-class P1D + Δ weighted by `1/n_c`; **per-element** (mean) term weights so log-P1D[172] doesn't swamp log-dN/dX[3]; clean mean-flux consistency `mean_F_clean ≈ exp(−τ₀)`.

**Files:**
- Modify: `hcd_analysis/emulator/model.py`
- Test: `tests/test_emulator_model.py`

- [ ] **Step 1: Write the failing test.**

```python
# append to tests/test_emulator_model.py
from hcd_analysis.emulator.model import joint_loss

def test_joint_loss_scalar_finite_and_grads_finite():
    key = jax.random.PRNGKey(3)
    m = Emulator(in_dim=10, n_k=8, key=key)
    batch = {  # one synthetic row, with NaN above Nyquist and an empty DLA class
        "x": jnp.zeros((2,10)), "tau0": jnp.array([0.3, 0.5]),
        "t_f_nhi": jnp.zeros((2,30)), "t_dndx": jnp.zeros((2,3)),
        "t_P_filt": jnp.where(jnp.arange(8) < 6, 1.0, jnp.nan)[None,None,:]*jnp.ones((2,4,8)),
        "t_delta": jnp.zeros((2,3,8)),
        "mask": (jnp.arange(8) < 6)[None,:]*jnp.ones((2,8), bool),
        "inv_nc": jnp.array([[1.,1/10,1/7,0.],[1.,1/10,1/7,1/3]]),   # row0 empty DLA
        "inv_nalpha": jnp.array([0.25, 0.25]),
        "mean_F_clean": jnp.array([0.7, 0.6]),
    }
    val, grad = jax.value_and_grad(lambda mm: joint_loss(mm, batch))(m)
    assert jnp.isfinite(val)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grad, eqx.is_array))
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
```

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL — `joint_loss` not defined.

- [ ] **Step 3: Implement** (uses `masked_mse`; `vmap` the model over the batch).

```python
# append to hcd_analysis/emulator/model.py
def joint_loss(model, batch, term_w=None):
    """Single joint scalar (spec sec.4). Per-element means already balance the
    172 vs 3 channel counts; term_w optionally rescales the named terms."""
    term_w = term_w or {"f_nhi": 1.0, "dndx": 1.0, "P_filt": 1.0, "delta": 1.0, "meanF": 0.1}
    preds = jax.vmap(model)(batch["x"], batch["tau0"])          # each value (B, ...)
    B = batch["x"].shape[0]
    # Head A: reweight per-row by 1/n_alpha (identical across alpha-siblings)
    la_cddf = masked_mse(preds["f_nhi"], batch["t_f_nhi"],
                         jnp.ones_like(batch["t_f_nhi"], bool),
                         weight=batch["inv_nalpha"][:, None])
    la_dndx = masked_mse(preds["dndx"], batch["t_dndx"],
                         jnp.ones_like(batch["t_dndx"], bool),
                         weight=batch["inv_nalpha"][:, None])
    # Head B: per-class 1/n_c weight, broadcast over k; mask = native Nyquist (B,K)
    m3 = batch["mask"][:, None, :]                              # (B,1,K)
    wcls4 = batch["inv_nc"][:, :, None]                         # (B,4,1)
    wcls3 = batch["inv_nc"][:, 1:, None]                        # HCD classes for delta
    lb_pf = masked_mse(preds["P_filt"], batch["t_P_filt"],
                       m3 & jnp.ones_like(batch["t_P_filt"], bool), weight=wcls4)
    lb_dl = masked_mse(preds["delta"], batch["t_delta"],
                       m3 & jnp.ones_like(batch["t_delta"], bool), weight=wcls3)
    # clean mean-flux consistency: predicted clean filtered mean ~ exp(-tau0)
    # (uses P_filt clean channel as a proxy handle; full hook wired in Task 16)
    l_meanF = jnp.mean((batch["mean_F_clean"] - jnp.exp(-batch["tau0"])) ** 2)
    return (term_w["f_nhi"]*la_cddf + term_w["dndx"]*la_dndx
            + term_w["P_filt"]*lb_pf + term_w["delta"]*lb_dl
            + term_w["meanF"]*l_meanF)
```

- [ ] **Step 4: Run to verify it passes.** Expected: PASS. **Commit.**

```bash
git add hcd_analysis/emulator/model.py tests/test_emulator_model.py
git commit -m "feat(phase2b): joint loss — 1/n_c + per-element terms + alpha-multiplicity + clean mean-F"
```

---

## Task 12: Head-A τ₀-invariance gradient test

Pin the structural invariance: `d(f_nhi, dN/dX)/d τ₀ == 0` exactly (Head A never sees τ₀). A gradient test, not a value test (spec §8).

**Files:**
- Test: `tests/test_emulator_model.py`

- [ ] **Step 1: Write the test.**

```python
# append to tests/test_emulator_model.py
def test_headA_outputs_are_tau0_invariant_by_gradient():
    key = jax.random.PRNGKey(4)
    m = Emulator(in_dim=10, n_k=8, key=key)
    x = jnp.ones(10)
    def fa(tau0):
        p = m(x, tau0); return jnp.concatenate([p["f_nhi"], p["dndx"]])
    J = jax.jacfwd(fa)(jnp.array(0.4))      # d(headA)/d tau0
    assert jnp.allclose(J, 0.0, atol=0.0)   # exactly zero — Head A does not consume tau0
```

- [ ] **Step 2: Run to verify it passes** (no implementation needed — invariance is structural).

Run: `... -m pytest tests/test_emulator_model.py::test_headA_outputs_are_tau0_invariant_by_gradient -v`
Expected: PASS. If FAIL, Head A wrongly consumes τ₀ — fix the architecture.

- [ ] **Step 3: Commit.**

```bash
git add tests/test_emulator_model.py
git commit -m "test(phase2b): Head-A tau0-invariance gradient test (structural)"
```

---

## Task 13: train.py — optax loop, LR schedule, early stop, checkpoint bundle

**Files:**
- Create: `hcd_analysis/emulator/train.py`
- Create: `scripts/train_emulator.py` (CLI + SLURM-launchable on cavestru0 gpu)
- Test: `tests/test_emulator_model.py` (a tiny overfit-one-batch test)

- [ ] **Step 1: Write the failing test (loss decreases on a tiny batch).**

```python
# append to tests/test_emulator_model.py
from hcd_analysis.emulator.train import make_optimizer, train_step

def test_one_train_step_reduces_loss():
    import optax
    key = jax.random.PRNGKey(5); m = Emulator(in_dim=10, n_k=8, key=key)
    batch = {  # reuse the joint_loss batch shape
        "x": jax.random.normal(key, (4,10)), "tau0": jnp.linspace(0.2,0.6,4),
        "t_f_nhi": jnp.zeros((4,30)), "t_dndx": jnp.zeros((4,3)),
        "t_P_filt": jnp.ones((4,4,8)), "t_delta": jnp.zeros((4,3,8)),
        "mask": jnp.ones((4,8), bool), "inv_nc": jnp.ones((4,4)),
        "inv_nalpha": jnp.ones(4), "mean_F_clean": jnp.exp(-jnp.linspace(0.2,0.6,4)),
    }
    opt = make_optimizer(lr=1e-3); opt_state = opt.init(eqx.filter(m, eqx.is_array))
    l0 = joint_loss(m, batch)
    for _ in range(50):
        m, opt_state, l = train_step(m, opt, opt_state, batch)
    assert float(l) < float(l0)              # optimiser actually reduces the loss
```

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL — `train` module not found.

- [ ] **Step 3: Implement.**

```python
# hcd_analysis/emulator/train.py
"""optax training loop + checkpoint bundle (leaves + arch cfg + norm stats + seed)."""
from __future__ import annotations
import json, pickle
import jax, jax.numpy as jnp, equinox as eqx, optax
from hcd_analysis.emulator.model import Emulator, joint_loss

def make_optimizer(lr=1e-3, steps=None):
    sched = optax.cosine_decay_schedule(lr, steps) if steps else optax.constant_schedule(lr)
    return optax.adamw(sched, weight_decay=1e-4)

@eqx.filter_jit
def train_step(model, opt, opt_state, batch):
    loss, grads = eqx.filter_value_and_grad(joint_loss)(model, batch)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def save_checkpoint(path, model, arch_cfg, norm_stats, seed):
    eqx.tree_serialise_leaves(str(path) + ".eqx", model)
    with open(str(path) + ".meta.json", "w") as f:
        json.dump({"arch_cfg": arch_cfg, "seed": seed}, f)
    with open(str(path) + ".norm.pkl", "wb") as f:
        pickle.dump(norm_stats, f)

def load_checkpoint(path):
    with open(str(path) + ".meta.json") as f:
        meta = json.load(f)
    skeleton = Emulator(**meta["arch_cfg"], key=jax.random.PRNGKey(meta["seed"]))
    model = eqx.tree_deserialise_leaves(str(path) + ".eqx", skeleton)
    with open(str(path) + ".norm.pkl", "rb") as f:
        norm = pickle.load(f)
    return model, meta, norm
```

- [ ] **Step 4: Run to verify it passes.** Expected: PASS.

- [ ] **Step 5: Write `scripts/train_emulator.py`** — argparse (`--cache`, `--fold`, `--epochs`, `--lr`, `--batch`, `--out`), build batches via `data.load_cache` + the splits, early-stop on a held (sim,snap) block, call `save_checkpoint`. Add a SLURM header note: run on `-A cavestru0 -p gpu --gres=gpu:1`; **profile one fold first** (cavestru0 budget is tight).

- [ ] **Step 6: Commit.**

```bash
git add hcd_analysis/emulator/train.py scripts/train_emulator.py tests/test_emulator_model.py
git commit -m "feat(phase2b): optax train loop + checkpoint bundling + train CLI"
```

---

## Task 14: k-fold LOSO error vector (+ DLA high-k shot flags)

Aggregate per-cell emulator error over folds → a static `(class, k, z, τ₀-band)` error vector; flag k-bins where the DLA class is shot-limited (near-zero effective sightlines). Feeds the likelihood covariance (spec §6, §8).

**Files:**
- Modify: `hcd_analysis/emulator/train.py`
- Create: `scripts/build_error_vector.py`
- Test: `tests/test_emulator_model.py`

- [ ] **Step 1: Write the failing test.**

```python
# append to tests/test_emulator_model.py
from hcd_analysis.emulator.train import aggregate_error_vector
def test_error_vector_shape_and_dla_shotflag():
    import numpy as np
    # per-fold residuals: list of (class, k, z-band) arrays + per-cell effective counts
    resid = [np.ones((4, 8, 3)) * 0.1, np.ones((4, 8, 3)) * 0.2]
    neff  = [np.ones((4, 8, 3)) * 100, np.ones((4, 8, 3)) * 100]
    neff[0][3, 6:, :] = 0.0    # DLA class, high-k, fold0 -> shot-limited
    ev = aggregate_error_vector(resid, neff)
    assert ev["sigma"].shape == (4, 8, 3)
    assert ev["dla_shot_flag"].shape == (8,)
    assert ev["dla_shot_flag"][6] and ev["dla_shot_flag"][7]
    assert not ev["dla_shot_flag"][0]
```

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL.

- [ ] **Step 3: Implement.**

```python
# append to hcd_analysis/emulator/train.py
import numpy as np
def aggregate_error_vector(resid_folds, neff_folds, shot_thresh=1.0):
    """resid_folds, neff_folds: lists of (4,K,Zb) arrays. Returns smoothed sigma +
    a DLA high-k shot-noise flag (True where DLA effective counts are ~0)."""
    R = np.stack(resid_folds, 0)          # (F,4,K,Zb)
    N = np.stack(neff_folds, 0)
    sigma = np.sqrt(np.nanmean(R**2, axis=0))         # RMS over folds -> (4,K,Zb)
    dla_neff = N[:, 3, :, :].min(axis=(0, 2))         # worst-case DLA neff per k
    return {"sigma": sigma, "dla_shot_flag": dla_neff < shot_thresh}
```

- [ ] **Step 4: Run to verify it passes.** Expected: PASS.

- [ ] **Step 5: Write `scripts/build_error_vector.py`** — runs all LOSO folds (Task 13), collects per-cell residuals + effective DLA counts, calls `aggregate_error_vector`, saves `error_vector.npz`. **Commit.**

```bash
git add hcd_analysis/emulator/train.py scripts/build_error_vector.py tests/test_emulator_model.py
git commit -m "feat(phase2b): k-fold LOSO error vector + DLA high-k shot flags"
```

---

## Task 15: validation — τ₀-response (reframed), cross-class additivity, DLA-shape clustering

**Files:**
- Create: `scripts/validate_emulator.py`
- Test: `tests/test_emulator_likelihood.py` (additivity unit test) + `tests/test_emulator_model.py`

- [ ] **Step 1: Write the cross-class additivity unit test (spec §8).**

```python
# tests/test_emulator_likelihood.py
import numpy as np, jax.numpy as jnp
from hcd_analysis.emulator.likelihood import total_p1d_difference

def test_cross_class_additivity():
    # Sum_c w_c*Delta_c  ==  (P_total^unfilt - P_tier_p), by construction
    K = 8; w = jnp.array([0.7,0.18,0.07,0.05])
    P_filt = jnp.abs(jnp.ones((4,K)))
    delta = jnp.array(np.random.default_rng(0).normal(0,0.01,(3,K)))   # HCD only
    A = jnp.ones(3)
    P_tier_p = jnp.einsum("c,ck->k", w, P_filt)
    P_total = total_p1d_difference(P_tier_p, w[1:], A, delta)
    # P_total - P_tier_p must equal Sum_hcd w_c*A_c*delta_c
    assert jnp.allclose(P_total - P_tier_p, jnp.einsum("c,c,ck->k", w[1:], A, delta), atol=1e-12)
```

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL — module/fn not defined (created in Task 16; this test will pass once Task 16 lands — keep it here and run after Task 16, or stub `total_p1d_difference` now).

- [ ] **Step 3: Write `scripts/validate_emulator.py`** — on a trained checkpoint + the merged cache: (a) **reframed τ₀-response** (predicted per-class P1D vs the shared-`target_F` control across τ₀ — internal consistency, spec decision #4); (b) Tier-P/clean vs PRIYA bit-anchor cross-check; (c) **DLA-shape clustering** — `P_DLA` from DLA-only vs all-max-DLA sightlines across z, pass if within the Task-14 error vector. Emit a figure per check under `figures/analysis/03_templates_and_p1d/`.

- [ ] **Step 4: Commit.**

```bash
git add scripts/validate_emulator.py tests/test_emulator_likelihood.py
git commit -m "feat(phase2b): validation — reframed tau0-response, additivity, DLA-shape clustering"
```

---

## Task 16: likelihood.py — total-P1D contract (difference default + ratio toggle) + covariance

**Files:**
- Create: `hcd_analysis/emulator/likelihood.py`
- Test: `tests/test_emulator_likelihood.py`

- [ ] **Step 1: Write the failing tests.**

```python
# append to tests/test_emulator_likelihood.py
import jax
from hcd_analysis.emulator.likelihood import total_p1d_difference, total_p1d_ratio, assemble_covariance

def test_difference_form_differentiable_through_wc():
    K = 8
    P_filt = jnp.ones((4, K)); delta = jnp.zeros((3, K)); A = jnp.ones(3)
    w = jnp.array([0.7, 0.18, 0.07, 0.05])
    def f(d):  # gradient through the whole reconstruction
        P_tier_p = jnp.einsum("c,ck->k", w, P_filt)
        return total_p1d_difference(P_tier_p, w[1:], A, d).sum()
    g = jax.grad(f)(delta)
    assert jnp.all(jnp.isfinite(g))

def test_covariance_inflates_dla_high_k():
    K = 8
    sigma = jnp.ones((4, K)) * 0.01
    w = jnp.array([0.7, 0.18, 0.07, 0.05])
    cov_noflag = assemble_covariance(jnp.ones(K)*1e-3, sigma, w, dla_shot_flag=jnp.zeros(K, bool))
    flag = jnp.arange(K) >= 6
    cov_flag = assemble_covariance(jnp.ones(K)*1e-3, sigma, w, dla_shot_flag=flag)
    assert jnp.all(jnp.diag(cov_flag)[6:] > jnp.diag(cov_noflag)[6:])   # flagged bins inflated
```

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL.

- [ ] **Step 3: Implement (spec §6).**

```python
# hcd_analysis/emulator/likelihood.py
"""Total-P1D likelihood contract (spec sec.6). Difference form is the DEFAULT."""
from __future__ import annotations
import jax.numpy as jnp

def total_p1d_difference(P_tier_p, w_hcd, A_hcd, delta_hcd):
    """P_obs = P_tier_p + Sum_{c in HCD} w_c*A_c*Delta_c. w_hcd,A_hcd: (3,); delta:(3,K)."""
    return P_tier_p + jnp.einsum("c,c,ck->k", w_hcd, A_hcd, delta_hcd)

def total_p1d_ratio(P_tier_p, w_hcd, A_hcd, ratio_hcd):
    """Alternative toggle: multiplicative per-class ratio form."""
    factor = 1.0 + jnp.einsum("c,c,ck->k", w_hcd, (A_hcd - 1.0), ratio_hcd)
    return P_tier_p * factor

def assemble_covariance(cosmic_var, sigma_emu, w_c, dla_shot_flag, shot_inflate=10.0):
    """Total cov = diag(cosmic variance) + per-class emulator error propagated through w_c
    (quadrature). DLA high-k shot-limited bins inflated so uncertainty isn't understated."""
    # per-class emu error -> total via weights, in quadrature over classes
    emu_var = jnp.einsum("c,ck->k", w_c**2, sigma_emu**2)
    emu_var = jnp.where(dla_shot_flag, emu_var * shot_inflate, emu_var)
    return jnp.diag(cosmic_var + emu_var)
```

- [ ] **Step 4: Run to verify it passes** (and re-run Task-15 Step-1 additivity test, now green).

Run: `... -m pytest tests/test_emulator_likelihood.py -v`
Expected: PASS (all 3 tests).

- [ ] **Step 5: Commit.**

```bash
git add hcd_analysis/emulator/likelihood.py tests/test_emulator_likelihood.py
git commit -m "feat(phase2b): total-P1D likelihood — difference default + ratio toggle + covariance"
```

> **A_subDLA semantics (spec §6):** the filtered tier masks 100% of DLAs but only ~56% of subDLAs, so `P_subDLA^filt` is a partial hybrid and `A_subDLA` absorbs the amplitude ambiguity. Re-derive its prior width against the merged v3.3 cache when the likelihood priors are finalised (a short follow-up once Task 2 of the *cache* plan lands the merge).

---

## Task 17: Hardening — class-edge source-of-truth assertion (referee finding)

Both science referees flagged that the N_HI class edges (17.2/19.0/20.3) are duplicated across `priya_p1d.FINE_NHI_EDGES`, `diag_wc_from_dndx.py`, and `diag_crossclass_coupling.py` with no single source of truth, risking silent misattribution. Add a post-build assertion that the cache's `n_absorbers` partition edges match `FINE_NHI_EDGES`.

**Files:**
- Modify: `scripts/build_emulator_cache_tau0.py`
- Test: `tests/test_emulator_cache_tau0.py`

- [ ] **Step 1: Write the failing test.**

```python
# append to tests/test_emulator_cache_tau0.py
def test_dndx_class_edges_match_fine_nhi_edges():
    from hcd_analysis.priya_p1d import FINE_NHI_EDGES
    import scripts.build_emulator_cache_tau0 as bt0
    # the coarse class boundaries (17.2,19.0,20.3) must be members of FINE_NHI_EDGES
    for edge in (17.2, 19.0, 20.3):
        assert np.isclose(FINE_NHI_EDGES, edge, atol=1e-9).any(), f"{edge} not in FINE_NHI_EDGES"
    assert bt0.COARSE_CLASS_EDGES == (17.2, 19.0, 20.3)
```

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL — `COARSE_CLASS_EDGES` not defined.

- [ ] **Step 3: Implement** — add a single `COARSE_CLASS_EDGES = (17.2, 19.0, 20.3)` constant in `build_emulator_cache_tau0.py`, assert at build time that each is in `FINE_NHI_EDGES`, and make `compute_dndx_per_class` reference it. (Follow-up: refactor the diag scripts to import this constant instead of re-hardcoding — a DRY cleanup.)

- [ ] **Step 4: Run to verify it passes.** Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add scripts/build_emulator_cache_tau0.py tests/test_emulator_cache_tau0.py
git commit -m "harden(phase2b): single-source class edges + n_absorbers boundary assertion"
```

---

## Validation gates carried from the science-referee review (2026-06-01)

Folded into the tasks above (not dropped):
- **Freeze-core no-op breadth** (referee R2 #1): the core claim was bit-identical on the 3 sampled shards (LF 000/040, HR 000) at 0.000e+00. A z×α-grid extension of `diag_tau_freeze_sensitivity.py` is **optional added confidence**, not a blocker — run it opportunistically before the likelihood is finalised.
- **w_c round-trip across sims** (referee R2 #4): Task 6 Step 1 calibrates `δ_c(z)` over ≥3 sims (not the single sim the diagnostic used) and asserts <0.5% post-correction.
- **Class-edge single source of truth** (both referees): Task 17.
- **Per-class shared `target_F`** (referee R2 #2): by design (spec); `mean_F_by_bin` is the stored consistency handle — surfaced in the Task-15 reframed τ₀-response (per-class vs shared control).
- **Damping-wing systematic**: deferred to Phase 3 per spec; carried as a likelihood model-error term, not closed here.

---

## Self-review

- **Spec coverage:** §3 arch → Tasks 8,9; §4 loss → Tasks 10,11; §5 w_c/dN/dX → Tasks 5,6,7; §6 likelihood → Task 16; §7 splits → Task 4; §8 validation → Tasks 12,14,15; §9 modules → all; §10 B.5–B.11 → Tasks 2–13; C.12–C.14 → Tasks 14–16. Cache-edge hardening (referee) → Task 17.
- **Reference-data tasks (6 Step 1, 7 Step 1):** not placeholders — concrete fetch/calibrate actions producing recorded, sourced numbers (`δ_c` coeffs; PW14/Crighton incidence fiducials). The plan flags these as the only steps that read external sources; the spec forbids inventing them.
- **Type consistency:** class order `(clean,LLS,subDLA,DLA)` everywhere; `dndx`/`delta`/`w_hcd` are the 3 HCD classes in `(LLS,subDLA,DLA)`; `P_filt` is 4-class; `structural_tier_p` consumes 4-class `w_c` + linear `P_filt`.
- **Open dependency:** training (Task 13) + validation (Task 15) + error vector (Task 14) consume the **merged production cache** (the separate cache plan's merge step). Tasks 0–12, 16, 17 are fully TDD-able now against the synthetic fixture.
