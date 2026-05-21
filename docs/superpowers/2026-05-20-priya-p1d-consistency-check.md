# PRIYA P1D consistency check — 2026-05-20 (v3, machine-precision match)

**Question.** Does our P1D pipeline reproduce PRIYA's published training data
`/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/mf_emulator_flux_vectors_tau1000000.hdf5`
to better than 1 %?

**Headline result.** **Yes — at floating-point precision.**

```
median(P_mine / P_PRIYA) = 1.00000015
std                       = 3.1e-7
max|r-1|                  = 1.21e-6
```

across all 172 k-bins (3 decades in k, modes m=1..172) — when we drive
fake_spectra's pipeline directly:

```python
from fake_spectra.fluxstatistics import flux_power, mean_flux, obs_mean_tau
from fake_spectra.spectra import Spectra
filter_single = Spectra._filter_single_tau_complex   # unbound method

tau_eff = -np.log(np.mean(np.exp(-tau_raw)))
self_stub = SimpleNamespace(nbins=tau_raw.shape[1])
for i in np.where(tau_raw.max(axis=1) > 1e6)[0]:
    tau_raw[i], _ = filter_single(self_stub, tau_raw[i], tau_eff,
                                  tau_thresh=1e6, thresh2=0.25)
target_F = np.exp(-alpha * obs_mean_tau(z))   # Kim 2013 / 0711.1862
kf, P = flux_power(tau_raw, vmax, spec_res=0.0,
                   mean_flux_desired=target_F, window=False)
```

Match point: sim 0
(`ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056`),
snap 17 (z = 3.0), PRIYA row 344 with α = 1.011183.

Plots:
- **Headline (v3 vs v2 overlay + 1e-6 zoom):**
  `docs/superpowers/figs/2026-05-20-priya-p1d-ratio-v3.png`
- v2 ratio (with the 0.1 % slope from my non-periodic port):
  `docs/superpowers/figs/2026-05-20-priya-p1d-ratio.png`
- v2 P1D overlay: `docs/superpowers/figs/2026-05-20-priya-p1d-overlay.png`

## v1 → v2 → v3 summary

| | v1 (this-repo plumbing) | v2 (fake_spectra + Python port of filter) | v3 (fake_spectra direct) |
|--|--|--|--|
| Median ratio mine/PRIYA | 1.0702 | 0.99949 | **1.00000015** |
| std | 3.4e-3 | 8.7e-4 (0.087 %) | **3.1e-7** |
| max\|r−1\| | 8.7 % | 1.3 % (mode m=1) | **1.2e-6** |
| Slope in log k? | yes (7 % flat) | yes, ~0.1 % | **none, machine precision** |

The v1 → v2 jump (7 % flat offset removed):
- ~6 % from interpreting α as the τ-multiplier instead of the Kim 2013 slope-α
  (PRIYA's actual scale at row 344 is **0.9385**, not 1.0112 — found by
  brentq on `<exp(−scale·τ_filtered)> = exp(−α·obs_mean_tau_Kim2013(z))`).
- ~0.6 % from Kim 2013 vs Becker `obs_mean_tau`
  (z = 3.0: Kim = 0.3625; Becker = 0.3745).

The v2 → v3 jump (the residual 0.1 % slope + m=1 spike removed):
- My Python port of `_filter_single_tau_complex` truncated walks at the box
  boundary; fake_spectra uses Python negative indexing on the left walk and
  explicit `j -= self.nbins` on the right walk, so DLAs near box edges wrap
  periodically. The fix was to import the unbound method from
  `fake_spectra.spectra.Spectra` and call it directly with a
  `SimpleNamespace(nbins=…)` stand-in for `self`. With that, every PRIYA mode
  agrees to floating-point precision.

## What v3 means for Phase 2a

The reference pipeline is now pinned: drive fake_spectra's actual machinery
directly. The Phase-2a builder I shipped (`scripts/build_emulator_cache_tau0.py`)
uses this repo's `compute_p1d_per_class` + `tau_transform=partial(freeze_core_rescale,
α=α, ...)` — which is the v1 pipeline (7 % off PRIYA at α = 1.011). It needs
to be rewritten as a thin wrapper that:

1. Slices τ by sightline class (clean / LLS / subDLA / DLA) before P1D.
2. Calls `fake_spectra.fluxstatistics.flux_power` per class (and per global,
   PRIYA-compatible variant).
3. Uses Kim 2013 slope-α parameterisation: stores `α_slope` in the cache and
   computes `mean_flux_desired = exp(-α_slope · obs_mean_tau(z))` per (sim, z).
4. Provides TWO tau-handling tiers (per the user's framing — see §7):
   - **Tier "P" (PRIYA-compatible, baseline):** `_filter_single_tau_complex`
     with `tau_thresh = 1e6, thresh2 = 0.25`. Single total-P1D output.
     Survey-equivalent training target for the main forest emulator.
   - **Tier "C" (per-class, HCD adds-on):** no `_filter_tau`; instead
     per-class P1Ds (clean / LLS / subDLA / DLA) using the catalog labels.
     Used to build the P_DLA/P_clean ratio emulator. The user explicitly
     wants per-class P1Ds stored *separately* (not as ratios) so that clean
     forest can also be evaluated standalone.

Phase-2a-fix plan to follow (see §7 for the user's framing).

## Sim/snap assertion

PRIYA row 344 vs my `SimulationICs.json`:

| param | mine | PRIYA | rel diff |
|--|--|--|--|
| ns | 0.80325 | 0.80325 | 0 |
| Ap | 2.2e-9 | 2.20333e-9 | 0.0015 (Ap is a different parameterisation; see Q1 below) |
| herei | 4.05 | 4.05 | 0 |
| heref | 2.67 | 2.67 | 0 |
| alphaq | 2.215 | 2.215 | 0 |
| hub | 0.735 | 0.735 | 0 |
| omegamh2 | 0.1405 | 0.1405 | 0 |
| hireionz | 7.175 | 7.175 | 0 |
| bhfeedback | 0.056 | 0.056 | 0 |

All non-Ap parameters agree exactly. Same sim, same snap.

---

---

## 1. PRIYA's τ-mask recipe — what `_filter_tau` *actually* does

The 2026-05-15 dive doc described `_filter_tau` as a pixel cap; the
re-reading in this session (cross-checked against
`hcd_analysis/masking.py:416-455` which implements the same recipe locally as
`priya_dla_mask_row`) shows it is **destructive trough-fill**, not a cap.

Algorithm (PRIYA paper, arXiv:2306.05471 §3.3, ported in `priya_dla_mask_row`):

1. **Detect.** Sightline is "DLA-contaminated" if `max(τ_pixel) > τ_dla_detect`
   with `τ_dla_detect = 10⁶`.  (The dive doc's *empirical* finding still holds:
   100 % of DLAs and ~56 % of sub-DLAs satisfy this; it is a real DLA-strength
   cut, not a no-op.)
2. **Locate** the peak pixel — index of the max-τ pixel on that sightline.
3. **Walk outward** from the peak in both directions until τ falls below
   `τ_mask_scale + τ_eff` with `τ_mask_scale = 0.25` and
   `τ_eff = -ln⟨F⟩_unmasked` (computed in a separate first pass over *all*
   sightlines). The mask is the **single contiguous region** around the peak;
   scattered IGM pixels above the threshold elsewhere on the same sightline are
   *not* masked.
4. **Fill** masked pixels with `τ_eff`, so `δF = F/⟨F⟩_global − 1 = 0` in the
   masked region.
5. **Compute P1D** over all sightlines (modified + unmodified) using the
   *unmasked* `⟨F⟩_global` in the `δF` normalisation.

This is **not** a per-pixel cap. It zeroes the entire damping trough on
DLA-contaminated sightlines.

The repo's `compute_p1d_priya_masked` (`hcd_analysis/p1d.py:556-600`) is a
faithful port of this algorithm.

## 2. My Phase-2 "freeze-core" recipe — what `freeze_core_rescale` does

For each pixel with native τ:

* if τ > τ_freeze (= 10⁶): output τ (**unchanged**, native).
* otherwise: output α · τ.

This is **not** equivalent to `_filter_tau`. The two recipes agree on
**saturated-core pixels** (because exp(−10⁶) underflows to 0 in float64 under
either recipe and a cap or freeze leaves F = 0 either way) but they
**disagree** on:

* **pixels just under the threshold** (τ ∈ ~[10⁴, 10⁶]) — damped wings, strong
  sub-DLA bodies. `_filter_tau` zeroes them as part of the contiguous DLA
  trough; `freeze_core_rescale` rescales them by α.
* **the structure of the rest of the sightline that contains a DLA** — under
  `_filter_tau` an entire DLA-contaminated sightline contributes only the
  non-DLA pixels; under `freeze_core_rescale` the whole sightline (with native
  τ above-threshold) is kept in P_clean or whatever class it belongs to.

The freeze-core recipe was chosen *because* the design memo argued
`_filter_tau`'s wing-zeroing is physically wrong (self-shielded cores +
damping wings carry real Lyα-forest information that PRIYA discards). So
"reproduces PRIYA exactly" is **not** a success criterion for the freeze-core
code — the two are *intended* to diverge on damped wings. The consistency
check below uses a different lever (run the **PRIYA recipe** through the
shared plumbing) to isolate the plumbing from the recipe.

## 3. The P1D math, as implemented

`P1DAccumulator.add_batch` (`hcd_analysis/p1d.py:115-148`):

```python
delta_F = F / mean_F_global - 1                     # dimensionless
ft = np.fft.rfft(delta_F * dv_kms, axis=1)          # units km/s
power_per_mode = |ft|² / (nbins * dv_kms)           # units km/s
P1D[k] = mean_over_skewers(power_per_mode[k])
```

* **`mean_F_global`**: for `compute_p1d_priya_masked` this is `⟨F⟩_unmasked`
  computed in a first streaming pass over all native τ pixels.
* **Native k grid**: `np.fft.rfftfreq(nbins, dv_kms)` — **cyclic** k in (s/km).
* **What's stored where**:
  - Phase-2 cache `observables_tau0.h5` interpolates `per_class["P_*"]` onto
    `k_target = 2π · _DEFAULT_K_BINS` — i.e. the **k axis is angular** but the
    **P values are the cyclic-FFT outputs** (no /2π applied).
  - PRIYA's `mf_emulator_flux_vectors_tau1000000.hdf5` uses the **same hybrid
    convention** (`kfkms` ≡ angular `2π · k_cyc`; `flux_vectors` is the cyclic
    FFT output). Verified empirically below (a /2π division gave an
    off-by-2π ratio of 0.17, removing the division gave 1.07).

## 4. The comparison

**Anchor.** sim 0 = `ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056`,
snap 17 (z = 3.0). Nearest PRIYA training row (col-1 onward matches the 9
cosmology params to 3.5 × 10⁻³ relative, the precision of my folder-name
parser): row 344 with `params[344, 0] = α = 1.011183`. PRIYA's z = 3.0 lives at
`zout[8]`.

**My side.** Raw τ from
`/nfs/turbo/umor-yueyingn/mfho/emu_full/<sim>/output/SPECTRA_017/lya_forest_spectra_grid_480.hdf5`
— 691 200 skewers × 1250 pixels at dv = 10.0068 km/s. Pre-multiplied
τ → α · τ with α = 1.011183, then ran the PRIYA recipe inline (mean-F first
pass + `priya_dla_mask_row` + `P1DAccumulator`).

**PRIYA side.** `flux_vectors[344, 8·172 : 9·172]` and `kfkms[344, 8, :]`.

**Result.**

```
my mean_F at α=1.0112  : 0.6756
PRIYA-target mean_F     : exp(-1.0112 × τ_obs_Becker(z=3)) = 0.6849
                          (5 % offset — see §5.Q1)
DLA-flagged sightlines  : 44 506 / 691 200  (6.4 %)
k bins compared         : 171 / 172 (k_min = 5.0231 × 10⁻⁴ rad·s/km exactly matches
                          both grids' fundamental, as expected for matched nbins,dv)
ratio (mine / PRIYA)    : median 1.0702, mean 1.0715, IQR 0.0063,
                          min 1.067, max 1.087
```

The k-shape match is essentially perfect (IQR 0.6 %); the offset is a single
flat number (7 % high) across 3 decades of k. Plot:
`/tmp/priya_consistency_plot.png`.

## 5. Urgent questions for review (v1 — historical, since superseded by v2)

The v2 pipeline (above) resolved 6 of the 7 % offset, with the answers
re-derived from reading `fake_spectra/fluxstatistics.py`,
`fake_spectra/spectra.py`, and `lya_emulator_full/lyaemu/mean_flux.py`.

Original questions Q1–Q6 with answers from the source code:

* **Q1 (slope-α vs τ-multiplier α).** **Slope-α**, confirmed:
  `MeanFluxFactor.get_t0(zzs, params=t0)` returns `t0 * obs_mean_tau(zzs)`
  (`mean_flux.py:155-157`); this becomes `mean_flux_desired = exp(-t0 · τ_obs(z))`,
  and `_rescale_mean_flux` iteratively solves for `scale` such that
  `<exp(-scale·τ_sim)> = mean_flux_desired`. The scale is what's actually
  multiplied into τ. At α = 1.011183, z = 3.0, sim 0: scale = 0.9349.
* **Q2 (target vs measured ⟨F⟩).** **Target** ⟨F⟩ used in `δF = exp(-scale·τ)/mean_flux_desired - 1`
  (`fluxstatistics.py:88`). Not the simulation's measured ⟨F⟩.
* **Q3 (per-skewer mean-F).** **Global** mean-F. PRIYA does NOT subtract
  per-skewer ⟨F⟩ (`fluxstatistics.py:88` uses the scalar `mean_flux_desired`).
* **Q4 (sightline subset).** **All 691 200** sightlines (3 × 480² axes — confirmed by user).
* **Q5 (shot-noise / window).** **No** shot-noise subtraction.  `MySpectra.spec_res = 0.0`
  (`flux_power.py:155`) and PRIYA uses `window=False`, so no window correction.
* **Q6 (redo at exact PRIYA-α).** Done — see v2 above.

### Remaining sub-questions for the user

**Q1' (this is now the only physics-relevant open question).**  Now that we
know PRIYA's α (col 0 of `params`) is the **slope-multiplier on Kim 2013**,
the Phase-2 design should follow the same convention.  My Phase-2 spec
specified α uniform over [0.66, 1.36] applied directly to τ (the freeze-core
rescale).  Should Phase-2 instead sample slope-α (same convention) and
iteratively solve for the τ-multiplier per (sim, z), as PRIYA does?

Pro: bit-compatible with PRIYA's training data; downstream emulator can be
benchmarked directly against PRIYA's emulator on the same params.

Con: τ₀ stored in our cache is no longer `tau0_from_mean_flux(mean_F_clean)`
trivially; it must come from `target_tau = α · obs_mean_tau_Kim2013(z)`.

**Q2'.** The "Ap" parameterisation difference (0.15 % between
`SimulationICs.json` "scalar_amp=3.788e-9" and PRIYA's `params[:, 2] = 2.2e-9`).
Is "Ap" the value at a specific pivot scale, or a different power-spectrum
normalisation, or just the folder-name label? Worth nailing down before
declaring sim/snap match "perfect" — for the consistency check we matched
to 0.15 % which is fine, but for downstream emulator training the input
features must be defined identically to PRIYA's.

## 6b. Multi-point verdict (2026-05-20 sbatch fan-out)

After the single-point v3 result (sim 44, z=3.0, α=1.0112), we ran the same
fake_spectra-direct pipeline across a wider parameter grid using a sbatch
array (`scripts/consistency_checks/sbatch_multipoint.sh`):

- **3 sims** — sim_idx 0 (`ns0.842Ap1.36e-09...`), 29 (`ns0.901Ap1.22e-09...`),
  and 44 (`ns0.803Ap2.2e-09...` — the original v3 anchor).
- **4 redshifts** per sim — z = 4.6, 4.0, 3.0, 2.4.
- **10 α per (sim, snap)** — every PRIYA training point at that sim.
- **120 test points total** (`P_mine[i] / P_priya[i]` at each of 172 k-bins).

| metric | value |
|--|--|
| worst max\|r−1\| across all 120 points | **1.89 × 10⁻⁵ (0.0019 %)** |
| worst std across all 120 points | **1.58 × 10⁻⁶** |
| median of medians | 1.00000002 |
| all max\|r−1\| < 10⁻⁴ ? | yes |
| all max\|r−1\| < 10⁻³ ? | yes |
| all max\|r−1\| < user's 1 % target ? | yes (by 3 orders of magnitude) |

**Verdict: the v3 recipe reproduces PRIYA's flux_vectors at floating-point
precision across the LF training parameter space.** Slight elevation of
max\|r−1\| at the highest z (z = 4.6) — up to 1.89 × 10⁻⁵ for sim 44 —
correlates with DLA-flagged sightline count (more filter calls → more FP
accumulation). Even there the deviation is 530× below the user's 1 %
acceptance bar.

Plot: `docs/superpowers/figs/2026-05-20-priya-p1d-multipoint-ratios.png`
(3 sims × 4 z grid, 10 α overlaid per panel, y-axis ±10⁻⁴).
Full per-point summary table:
`docs/superpowers/figs/2026-05-20-priya-p1d-multipoint-summary.csv`
(120 rows, fields: sim_idx, snap, z, alpha, target_F, median, std, max_abs_dev).

**Implication for the refactor plan** (`docs/superpowers/plans/2026-05-20-phase2a-refactor-fake-spectra.md`):
- Task 0 (multi-point gate) is **GREEN** — proceed with the refactor.
- The "<1 %" requirement from the user is satisfied with ~530× headroom.

### sbatch infrastructure notes (for the production cache build)

- Per-(sim, snap) wallclock: ~8–17 min (1 GB I/O on NFS dominates short
  jobs; 10 × flux_power dominates the rest).
- Array submission: 12 tasks at `--cpus-per-task=4 --mem=32G --time=30:00`
  on partition `standard`. Queued + completed all 12 in ~20 min wallclock.
- Two env caveats baked into `sbatch_multipoint.sh` (so the production
  builder inherits the same pattern):
  1. **DON'T** `conda activate emu-3.9` from inside the slurm script —
     mamba module reload corrupts the env. Use the python binary's
     absolute path: `/home/mfho/.conda/envs/emu-3.9/bin/python3`.
  2. **Set `LD_LIBRARY_PATH` to put GSL first, then the conda env's lib
     directory, then the system path.** This is needed so (a)
     `fake_spectra._spectra_priv` finds `libgsl.so.25`, and (b) scipy's
     `_highs` C extension finds the conda env's newer `libstdc++.so.6`
     (which has `GLIBCXX_3.4.29`) rather than the gcc/10.3.0 module's
     older one.
- Snap-numbering varies sim-to-sim. The first array submission failed for
  4/12 tasks because I had hard-coded snap numbers from sim 44; sims 0
  and 29 have different snap_NNN at the same target z (e.g., z=3.0 is
  snap_017 for sim 44 but snap_016 for sim 0). The fix-up array job used
  per-sim correct snap numbers. Production builder must look up snap by
  target z from meta.json, not assume a fixed snap-to-z map.

---

## 7. User framing of the two tiers (2026-05-20 session)

The user confirmed:

* **Baseline = Tier P (PRIYA-compatible).** Use fake_spectra's `_filter_tau` at
  `tau_thresh = 1e6`. Reasoning: real surveys do not perfectly remove all DLAs;
  the survey P1D measurement IS Tier P. The main forest emulator must be
  trained against the same observable the survey delivers, so the survey-side
  P1D *includes* the residual DLA contribution after their imperfect masking.
  Keep `tau_thresh = 1e6` as baseline. Keep `NHI ≥ 20.3` (no τ-cap) as a future
  backup tier — most surveys cut on NHI directly. Future option: differentiated
  treatment of partial self-shielding for sub-DLAs / LLS (Rahmati-style).

* **Adds-on = Tier C (per-class, HCD module).** Same `flux_power` + slope-α +
  iterative inversion, but replace `_filter_tau` with the freeze-core per-class
  mask. Used to build the P_DLA/P_clean (and analogous LLS, sub-DLA) ratios as
  separate emulator heads. At inference time: PRIYA's main emulator gives the
  survey-equivalent P1D, the adds-on module gives the HCD contribution that
  was suppressed by survey masking, the two are combined to predict the
  observable.

* **Store P_DLA and P_clean SEPARATELY (not as ratios).** Reason from the user:
  we may want to validate "P_clean + HCD emulator" as an alternative inference
  path to "PRIYA P1D (with residual DLAs / capped τ at 1e6) + HCD emulator".
  Especially since some high-res surveys can mask sub-DLAs as well. Storing
  the per-class P1Ds individually lets us form whatever ratio at inference
  time without rebuilding the cache.

## 8. Implications for Phase-2a (the code already shipped)

The PR (#10) currently uses this repo's `compute_p1d_per_class` with a
`tau_transform=partial(freeze_core_rescale, alpha=α, ...)` applied to τ
directly. The v1 7 % offset means **the τ₀ cache built by the PR is not
bit-compatible with PRIYA's flux_vector training data**. Three follow-up tasks
are needed before the production sharded cache is built:

1. **Refactor `build_tau0_rows`** to use `fake_spectra.fluxstatistics.flux_power`
   directly (or, equivalently, mimic it exactly). The per-class extension
   becomes: slice `tau` by sightline class first, then call `flux_power` on
   each subset.
2. **Switch from direct-α to slope-α**: store `params[:, 0] = α_slope` and
   compute `target_tau = α · obs_mean_tau_Kim2013(z)` per row. The
   `tau_transform` becomes a (sim, z)-specific `partial(rescale_to_target,
   target_F=…)` rather than a fixed `α · τ`.
3. **Multi-DLA mask:** replace the repo's `priya_dla_mask_row` with the
   `while max(tt) > tau_thresh` loop that fake_spectra uses.

The user has already authorised this direction: "first principle is to build
on top of existing P1D code and add HCD dimensions" (2026-05-20 session). I'll
draft these as a Phase-2a-fix plan once user confirms Q1' / Q2'.

---

## 7. Historical: v1 questions (kept for reference)

I have **strong hypotheses** for what the 7 % is — I want to confirm them
with you before changing any production code, because each hypothesis points
at a different fix (or no fix, if the convention is genuinely PRIYA-side).

### Q1. `params[:, 0]` α — is it slope-α or τ-multiplicative-α?

The dive doc said PRIYA's `mean_flux_desired` "internally re-solves for α and
recomputes F = exp(−α τ)" — i.e. the **slope-α** stored in `params` is
converted *inside* `fake_spectra` to whatever τ-multiplier makes
⟨exp(−α' τ_sim)⟩ equal `exp(−α_slope · τ_obs_Becker(z))`.

I applied α = 1.0112 directly to τ_sim, which gives ⟨F⟩ = 0.6756, but PRIYA's
target ⟨F⟩ is `exp(−1.0112 × 0.3745) = 0.6849`. A rough log-power
linearisation predicts PRIYA's effective τ-multiplier is α' ≈ 0.980 (3 % below
my 1.011); the resulting P1D shift is `2 · ⟨τ⟩ · Δα'/α' ≈ 2.4 %`. That's about
**a third of my 7 % offset**.

**Question:** is "α in `params[:, 0]` = slope-α, inverted by fake_spectra to
get the τ-multiplier" the right reading? If yes, my Phase-2 builder should
either (a) re-solve the inversion per sim/z (PRIYA-compatible), or (b) keep
the direct-multiplier definition and document the offset (cheaper). My
preference is (b) because it keeps `tau0_from_mean_flux(⟨F⟩_α) = -ln⟨F⟩_α`
*exactly* invertible from the cache, but I want you to weigh in.

### Q2. Does PRIYA's `δF` use the simulation's `⟨F⟩` or the *target* `⟨F⟩`?

The mismatched `⟨F⟩` (0.676 vs 0.685) couples into `δF = F/⟨F⟩ − 1` and
quadratically into P. If PRIYA uses the **target** `⟨F⟩_obs(z)` in the
normalisation while I use the **simulation's** measured `⟨F⟩`, that alone
flips both numerator and denominator — net effect a few percent. The repo's
`compute_p1d_priya_masked` uses the measured `mean_F_all`; I have not seen
which fake_spectra picks. Could you confirm from `flux_power.py`?

### Q3. Does fake_spectra subtract per-skewer mean-F before FFT?

Some pipelines normalise per-sightline (`δF_i = F_i / ⟨F⟩_i − 1`) instead of
globally. Per-sightline normalisation zeros mode 0 by construction and shifts
low-k power. The repo uses a **global** `⟨F⟩`. If PRIYA uses per-sightline,
that's another <few % effect at most-k but could explain a fraction of the
7 %. (Specifically, the dive doc's tutorial conventions said
"per-class P1D uses *per-subset* mean flux (Rogers convention)" — but the
*total* P1D in PRIYA's flux_vectors may use the global one.)

### Q4. Sightline subset — is PRIYA's training file built on all 691 200 or a
sub-sample?

My pipeline ran on all 691 200 skewers in the raw HDF5. PRIYA may build the
training cache from a subset (every-nth-skewer, or a random sub-sample). If
so, P1D variance differs but mean should not — *unless* the sub-sample is
biased (e.g. skipping DLA-rich sightlines). Worth a sanity check at the same
DLA-flagged fraction (`44 506 / 691 200 = 6.4 %`) — does that match PRIYA's
expectation?

### Q5. Shot-noise / window-function subtraction?

Some P1D pipelines subtract per-mode shot noise `1/N_pixels` or apply a
Hanning window. The repo does **neither**. If PRIYA does, a sub-% effect at
intermediate k.

### Q6. Should I redo the check at exact-PRIYA-α before declaring victory?

To eliminate Q1 as a confounder I could re-run with the (estimated) α' = 0.980
that PRIYA would have used internally. That would tell us if Q1 alone
accounts for the 7 % or whether there's a second piece. Want me to do this
re-run? It's another ~1 min of compute on real data.

---

## 6. What this means for Phase-2 production

* **Plumbing is sound.** I/O, FFT normalisation, k convention, mean-flux pass,
  DLA-mask recipe in this repo all reproduce PRIYA's flux-vector to 7 %
  absolute / <1 % in shape. The Phase-2a cache builder reuses every one of
  these pieces (`compute_p1d_per_class` is structurally the same accumulator
  with a class-label dispatch).
* **The 7 % offset is a *normalisation* discrepancy, not a *shape* one.** It
  doesn't propagate as a k-dependent bias and won't bend the emulator's
  predictions; it will at most rescale the absolute Lyα-forest amplitude. The
  emulator trains *its own* `A_c` template amplitudes anyway (per the design
  spec); a uniform 7 % is something `A_c` would absorb on the first iteration.
* **Recommendation:** proceed with the production sharded cache build using
  the current code, after we agree on Q1 (slope-α vs multiplicative-α). If we
  pick option (b) — keep direct multiplicative-α and document — no code
  change. If (a), I need to add a per-(sim, z) α-resolve inside
  `build_tau0_rows`.

## Annex A. Scripts + cached arrays (all in repo)

**v3 (THE authoritative reference — machine-precision match):**
* Driver: `scripts/consistency_checks/priya_p1d_consistency_v3.py`
* Ratio plot driver: `scripts/consistency_checks/plot_ratio_v3.py`
* Raw arrays: `docs/superpowers/figs/2026-05-20-priya-p1d-consistency-v3.npz`
* Ratio PNG: `docs/superpowers/figs/2026-05-20-priya-p1d-ratio-v3.png`
* Diagnostic for the v2 slope: `scripts/consistency_checks/check_k_grid_alignment.py`
  (confirms k-grids match to 1e-16 across all 172 modes — slope was not interpolation).
* Run with: `module load gcc/10.3.0 gsl/2.7 && conda activate emu-3.9 &&
  python3 scripts/consistency_checks/priya_p1d_consistency_v3.py` (the GSL
  load is needed for `fake_spectra._spectra_priv._rescale_mean_flux`).

**v2 (Python port of `_filter_single_tau_complex`, residual 0.1 % slope):**
* Driver: `scripts/consistency_checks/priya_p1d_consistency.py`
* Overlay-plot driver: `scripts/consistency_checks/plot_overlay.py`
* Ratio-plot driver: `scripts/consistency_checks/plot_ratio.py`
* Raw arrays: `docs/superpowers/figs/2026-05-20-priya-p1d-consistency.npz`
* Overlay PNG: `docs/superpowers/figs/2026-05-20-priya-p1d-overlay.png`
* Ratio PNG: `docs/superpowers/figs/2026-05-20-priya-p1d-ratio.png`

**v1 (historical, this-repo plumbing → 7 % offset from rolling our own):**
Not re-saved into the repo. Reconstructable from §3 of this doc if needed.
