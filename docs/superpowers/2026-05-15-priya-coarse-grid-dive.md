# PRIYA `coarse_grid.py` τ₀-sampling and training-set construction — 2026-05-15

> Background-agent output (`a3d4a2beecd3e5cd6`) reverse-engineering
> `/home/mfho/lya_emulator_full/lyaemu/coarse_grid.py` and its
> dependencies to document how PRIYA builds its P1D training set with
> mean-flux sampling. Read alongside
> [`2026-05-15-phase2-design-memo.md`](2026-05-15-phase2-design-memo.md)
> (HCD-emulator side) and
> [`../SESSION_HANDOVER_2026_05_15.md`](../SESSION_HANDOVER_2026_05_15.md)
> §9.

## 1. The τ₀ sampling mechanism

The τ₀ (mean-flux) sampling lives in `MeanFluxFactor` in `/home/mfho/lya_emulator_full/lyaemu/mean_flux.py:116-176`. The user is correct: **10 τ₀ values** per (sim, snap). The mechanism:

- **Parameterisation.** τ₀ is sampled as a *dimensionless multiplicative factor* α applied to the observed Becker-style power-law mean optical depth `obs_mean_tau(z) = 2.3e-3 × (1+z)**3.65` (mean_flux.py:6-32, the KODIAQ branch is the production path). At each redshift, the simulation's per-pixel τ is rescaled by α — so the "tau0" parameter is really `α = τ₀_target / τ₀_obs(z)`, redshift-independent.
- **Number of samples.** Default `dense_samples=10` (mean_flux.py:121). Used unchanged everywhere it is constructed (`likelihood.py:613, 638, 672, 675`; `gp_wrap.py:270`; `fakespectra2flux_power.py:30`).
- **Range.** When `dense_limits` is supplied (the production path used in `likelihood.py:610-613`), the range is `t0_factor * [slopelow, slopehigh]` with `t0_factor = [0.75, 1.25]` and the slope-induced multipliers computed by `mean_flux_slope_to_factor` for `slope ∈ [-0.4, 0.25]` over the z-grid. Numerically this is ≈ `[0.66, 1.36]` for the KODIAQ-SQUAD-XQ100 fit. The default no-arg case (mean_flux.py:128-140) uses the same construction internally.
- **Spacing.** **Uniform in α**, via `np.linspace(dlim[0], dlim[1], dense_samples)` (mean_flux.py:169). Not log-spaced, not Latin-hypercubed, not uniform in ⟨F⟩.
- **Rescaling application.** Done at *training-set construction time*, not on-the-fly inside the GP. The α-factor is converted to a per-z optical depth (`get_t0`) then to a target mean flux (`get_mean_flux`); that target is passed via `mean_flux_desired=mf` into `fake_spectra`'s `get_flux_power_1D` (flux_power.py:64-65, 98-99), which internally re-solves for α and recomputes `F = exp(-α τ)` on each sightline before forming P1D. The τ field is read from disk once per snap and held in memory while looping over the 10 α values (see `FluxPower.drop_table()` in flux_power.py:135-138, which clears tau after each parameter sweep).

Key sampling snippet (`mean_flux.py:159-172`):
```python
def get_params(self):
    ndense = np.shape(self.dense_param_limits)[0]
    pvals = np.nan * np.zeros((self.dense_samples, ndense))
    for dd in range(ndense):
        dlim = self.dense_param_limits[dd]
        dense = np.linspace(dlim[0], dlim[1], self.dense_samples)
        pvals[:, dd] = dense
    return pvals
```

There is also a subtle "interior shift" of the endpoints in `coarse_grid.py:540-545` so the sampled α values do not sit exactly on the prior boundary:
```python
newdp = dpvals[0] + (dpvals - dpvals[0]) / (np.size(dpvals)+1) * np.size(dpvals)
```
This compresses the 10 linearly-spaced points to lie strictly inside `[lo, hi]`.

## 2. Training-set construction call-chain

Numbered, in execution order:

1. **`coarse_grid.Emulator.__init__`** (coarse_grid.py:63-163) — sets `param_names`, `param_limits`, `mf=MeanFluxFactor(...)`, `kf=BOSSData().get_kf()`, base directory, and `tau_thresh` (the τ-clipping value used for HCD masking in `fake_spectra`).
2. **`Emulator.load`** (coarse_grid.py:281-301) — reads `emulator_params.json` and instantiates `self.myspec = flux_power.MySpectra(max_z, min_z, max_k=self.maxk)` (ngrid=480 sightlines per side).
3. **`Emulator.get_flux_vectors`** (coarse_grid.py:527-606) — the orchestrator. Calls `self.mf.get_params()` → `dpvals` (the 10 α values), applies the endpoint-shift, then either loads `mf_emulator_flux_vectors.hdf5` via `load_flux_vectors` or rebuilds it.
4. **`Emulator._get_fv(pp)`** (coarse_grid.py:425-432) — for each sim `pp`, calls `self.myspec.get_snapshot_list(base=di)` to build a `FluxPower` object that holds one `fake_spectra.Spectra` per snapshot (per redshift).
5. **`MySpectra.get_snapshot_list`** (flux_power.py:217-267) — iterates over snap indices, instantiates a `fake_spectra.Spectra` (loading the pre-computed `lya_forest_spectra_grid_480.hdf5` if present, else a fresh `GriddedSpectra` with `nspec=480`, `axis=-1`, `pix_res=10 km/s`, `spec_res=0`), accumulates them into the `FluxPower` until `nz` snapshots are collected.
6. **Per-(sim, α) flux-power evaluation** (coarse_grid.py:561-572):
   ```python
   flux_vectors = np.array([
       powers[i].get_power_native_binning(
           mean_fluxes=mef(dp), tau_thresh=self.tau_thresh, ith_sim=i)
       for dp in dpvals for i in range(nsims)
   ])
   ```
   This is the **per-α outer loop, per-sim inner loop**. For each (α, sim) pair `FluxPower.get_power_native_binning` (flux_power.py:78-119) loops over snapshots, calls `ss.get_flux_power_1D("H",1,1215, mean_flux_desired=mf[i_z], tau_thresh=tau_thresh)`, concatenates the z-slabs along the k-axis after clipping to `kf_sim <= self.maxk` (Mpc/h units).
7. **`Emulator.save_flux_vectors`** (coarse_grid.py:608-630) — writes one HDF5 (see §3).
8. **`Emulator.get_emulator`** (coarse_grid.py:434-457) — wraps `(aparams, kf, flux_vectors)` in `gpemulator.MultiBinGP` which trains one `SkLearnGP` per z-bin.

So the τ₀ loop is the **outer-most loop in the training-set build** (`for dp in dpvals for i in range(nsims)`), giving `n_α × n_sim` rows.

## 3. P1D data layout on disk and in memory

**On disk** — single HDF5 file at `{basedir}/{mfc}_emulator_flux_vectors[_tau{tau_thresh}].hdf5`, with `mfc="mf"` when τ₀ is being sampled, `"cc"` for the constant-mean-flux variant (coarse_grid.py:608-630, 632-670). Datasets:

| dataset | shape | units / meaning |
|---|---|---|
| `params` | `(n_α · n_sim, ndense + n_params)` | dense params first (α), then 9 cosmo/astro |
| `flux_vectors` | `(n_α · n_sim, n_z · n_kf_native)` | flux power, native Mpc/h-converted km/s P_F |
| `kfmpc` | `(n_kf_native,)` | shared k grid in comoving h/Mpc |
| `kfkms` | `(n_α · n_sim, n_z, n_kf_native)` | per-sim, per-z velocity-grid k in s/km |
| `zout` | `(n_z,)` | snapshot redshifts, linear z spacing of 0.2 |
| `attrs/classname` | scalar | identifies emulator subclass |

A **row** of `flux_vectors` is one (α, sim) pair; columns within the row are `n_z` blocks of length `n_kf_native` concatenated (z-major order). One snapshot is implicitly one redshift since each sim writes one snapshot per `zout` value.

There is no per-snapshot, per-α cache file. The raw sightline tau arrays (`lya_forest_spectra_grid_480.hdf5`) inside each simulation's `output/SPECTRA_NNN/` are reused; the 10 α rescalings are looped in memory after one disk read per snap.

## 4. k-grid handling

- **k_F grid stored in the training file** is `kfmpc` — Mpc/h comoving units (flux_power.py:101-103: `kf_sim *= vscale` converts the native km/s of `fake_spectra` to Mpc/h).
- **Per-(sim, snap) k_F in km/s** (`kfkms`) is recovered from `kfmpc` by dividing by each snap's `velfac`. So the actual modelled k-axis is the *cyclic / Mpc-h native* grid produced by `fake_spectra` (box-mode-spaced — the simulation box is L = 60 Mpc/h, so Δk = 2π/L).
- **Sharing across sims.** Within one resolution tier the box size is identical, so `kfmpc` is the same up to 1 part in 10⁶ — asserted in coarse_grid.py:590 (`np.all(np.abs(powers[0].kf / powers[-1].kf - 1) < 1e-6)`).
- **kmax cut.** `MySpectra` is constructed with `max_k = self.maxk` (coarse_grid.py:165-177), which sets the kmax to `kf_BOSS_max × velfac(z=4.4) × 2 ≈ 5 h/Mpc`. Modes above that are dropped in `FluxPower.get_power_native_binning` (flux_power.py:104).
- **Mismatched nbins between LF / HF.** Handled in `get_MFemulator` (coarse_grid.py:490-508): both LF and HF arrays are truncated to the common shorter `n_kf` after the kmax filter; the assert `np.all(kf - HRkf < 1e-3)` then enforces equality. No interpolation to a unified grid; HF and LF are made to share by truncation only.
- **Mapping to data k-bins** (BOSS / KODIAQ) is **not** done at training time; it is done at *prediction* time inside the likelihood via `flux_power.rebin_power_to_kms` (flux_power.py:14-35) using `scipy.interpolate.interp1d` on a per-z basis.

## 5. Emulator inputs / outputs

`MultiBinGP` (`gpemulator.py:18-61`) trains **one independent `SkLearnGP` per redshift bin** — i.e. `n_z` separate GPs, not one big joint emulator.

**Input vector** to each per-z GP (`SkLearnGP._get_interp`, gpemulator.py:85-127):
- `tau0` (α factor, dimensionless),
- `ns`, `Ap`, `herei`, `heref`, `alphaq`, `hub`, `omegamh2`, `hireionz`, `bhfeedback` (9 cosmo/astro).
- Mapped to unit cube via `map_to_unit_cube_list`.
- **Note:** z is NOT in the input — it is implicit in *which* GP you query.

**Output vector**: `P_F(k) / P_F^median(k) − 1` at the `n_kf_native ≈ 30-50` k-bins of that z-bin, where `P_F^median` is the median power spectrum across training rows (used as a per-bin scale factor, gpemulator.py:96-100).

**Kernel / loss**: GPy `Linear(ARD) + RBF(ARD)`, zero-mean Gaussian likelihood, noise fixed to `1e-10`. Trained by marginal-likelihood maximisation (`gp.optimize`), with 10 restarts on failure. The multi-fidelity variant (`SingleBinAR1`, gpemulator.py:170-287) layers a linear-multifidelity kernel on top.

## 6. HCD treatment in the existing PRIYA emulator

PRIYA is **forest-only at the emulator level**. HCDs enter only:

1. **At the simulation/spectrum stage** via `tau_thresh` (coarse_grid.py:72, 161; flux_power.py:99): passed to `fake_spectra.get_flux_power_1D`, where any sightline pixel with τ above the threshold is masked out before computing P1D. The user choose `tau_thresh=1e6` (effectively off) or a finite value to suppress saturated DLAs at the spectrum level. This is a coarse, single-class HCD mask — *not* a CDDF-aware four-class treatment.
2. **At the post-prediction stage** via `DLAcorr` / `DLA4corr` (gp_wrap.py:27-58, 61-113): a Rogers+2018 (`arXiv:1706.08532`) analytic fitting formula multiplies the GP-predicted P_F by `(1 + Σ_class α_class · f_class(k,z))`. Four classes: LLS, sub-DLA, small-DLA, large-DLA. `α_class` are nuisance parameters in the likelihood, not training inputs.

There is **no per-class P1D training signal**, no CDDF coupling inside training, and no HCD-class-aware emulator output. The DLA correction is a post-hoc analytic multiplier with priors on `α_lls, α_sub, α_sdla, α_ldla` (see `gp_wrap.py:304` for the limits).

## 7. What we can directly port vs adapt

**Direct port (copy with cosmetic edits):**
- `mean_flux.py` whole module — `obs_mean_tau`, `MeanFluxFactor`, `mean_flux_slope_to_factor`. The 10-point linspace in α is exactly the recipe we want.
- The endpoint-shift trick (coarse_grid.py:540-545).
- `gpemulator.SkLearnGP` / `MultiBinGP` skeleton — one GP per z-bin, median-normalised output, ARD Linear+RBF kernel. Even if we change inputs, the wrapper is reusable.
- `flux_power.rebin_power_to_kms` for mapping native sim k-grid → data k-bins.
- HDF5 schema (`params`, `flux_vectors`, `kfmpc`, `kfkms`, `zout`) — extend with per-class blocks.

**Substantive adaptation needed:**
- Four-class output structure. Today `flux_vectors[row]` is one P_F vector; we will need to store four P1D vectors (forest-only / LLS / sub-DLA / DLA — whichever class partition) per row, or reshape `flux_vectors` to `(n_rows, n_class, n_z·n_kf)`.
- The "row" definition expands from (α, sim) to (α, sim, HCD-class) or (α, sim) × per-class output channel.
- CDDF coupling: currently absent. We need a mechanism either to (a) condition each class's P1D on CDDF moments / column-density-window summary statistics, or (b) emit per-class P1D plus a CDDF and let the consumer combine them. PRIYA has no precedent here — the Rogers-2018 `DLA4corr` is the closest analogue and is purely analytic.
- HCD detection / per-class spectrum generation. Today PRIYA only masks pixels with `tau > tau_thresh`; for the four-class build we will need to *classify* each sightline by integrated column density and bin spectra into classes before forming P1D. This requires `fake_spectra` capabilities not used in the current pipeline (per-sightline `N_HI` integrals).
- Per-class kmax may differ — LLS / DLA wings carry low-k power that the forest mask removes, so the `maxk` heuristic in coarse_grid.py:165-177 may need a class-aware variant.

**Dependencies to replicate locally:**
- `fake_spectra` — the heavyweight one. Provides `spectra.Spectra`, `griddedspectra.GriddedSpectra`, `get_flux_power_1D`, `get_tau`, `velfac`, `units`. Same author (Bird); pip-installable (`fake-spectra` on PyPI) but `hcd_priya` will need its sightline-classification utilities.
- `GPy`, `emukit` — only if we keep the GP backbone.
- `h5py`, `mpi4py` (the latter is imported unconditionally in flux_power.py:8; can be stubbed).
- The PRIYA-internal helpers we'd actually copy are tiny: `latin_hypercube.py` (for any new dense sampling), `mean_flux.py`, `lyman_data.py` (k-grid definitions), and parts of `coarse_grid.py`.

## 8. Open questions for the user

1. **"Tightly coupled to CDDF statistics" — what is the intended coupling?** PRIYA has *no* precedent for putting CDDF moments into the emulator's input vector. Options I see: (a) condition GP inputs on CDDF parameters (a few f(N) amplitude/slope numbers); (b) emit per-class P1D + a separate CDDF predictor and combine analytically downstream; (c) train a joint emulator whose output is (P1D_forest, P1D_LLS, P1D_subDLA, P1D_DLA, CDDF_bins). Which is the design?

2. **Four-class partition definition.** Are the four classes the Rogers-2018 LLS / sub-DLA / small-DLA / large-DLA, or a different `N_HI` partition? The α-correction in `gp_wrap.py:78-113` uses Rogers's four-class binning — should we mirror it exactly?

3. **τ₀ count for the new emulator: keep 10, or change?** 10 was tuned for the forest in 2018; a four-class build may want more (because each class is smaller signal-to-noise). User said 10, so I will assume 10 unless told otherwise.

4. **Mean-flux model in the per-class emulator.** Does α act on the *forest* class only (because HCDs by construction saturate)? Or per-class with separate α per class? PRIYA's current `mean_flux_desired` is global — one α rescales all pixels uniformly.

5. **k-grid.** Do we keep the Mpc/h native grid (box-mode spacing) and rebin to data k at predict-time, or pre-rebin to a shared km/s grid at training-set construction? PRIYA does the former; a per-class build may want the latter to avoid four interp1d calls per predict.

6. **HCD masking semantics.** Right now `tau_thresh` masks pixels. For a per-class build we want to *select* sightlines (or sub-segments) by `N_HI` class, not just clip-and-keep. Is there an existing `fake_spectra` helper, or do we need to write the sightline classifier from scratch?

7. **Multi-fidelity?** PRIYA has LF (`fakespectra_basedir`) and HF (`HRfakespectra_basedir`) tiers and `SingleBinAR1`. Are we porting the multi-fidelity layer, or building single-fidelity first?

Relevant absolute paths used in this analysis:
- `/home/mfho/lya_emulator_full/lyaemu/mean_flux.py`
- `/home/mfho/lya_emulator_full/lyaemu/coarse_grid.py`
- `/home/mfho/lya_emulator_full/lyaemu/flux_power.py`
- `/home/mfho/lya_emulator_full/lyaemu/gpemulator.py`
- `/home/mfho/lya_emulator_full/lyaemu/gp_wrap.py`
- `/home/mfho/lya_emulator_full/lyaemu/likelihood.py`
- `/home/mfho/lya_emulator_full/lyaemu/fakespectra2flux_power.py`
- `/home/mfho/lya_emulator_full/lyaemu/priya_explorer.py`
- `/home/mfho/lya_emulator_full/lyaemu/lyman_data.py`
