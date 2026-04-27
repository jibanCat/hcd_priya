# HCD clustering — definitions, conventions, validation

Authoritative spec for the DLA × Lyα cross-correlation and the DLA × DLA
3D auto-correlation. Written 2026-04-27 ahead of implementation; **read
this and the unit-test list (`tests/test_clustering.py`) before
trusting any number from `hcd_analysis/clustering.py`**.

References
----------

- **Font-Ribera et al. 2012** ([arXiv:1209.4596](https://arxiv.org/abs/1209.4596)) — first measurement of DLA × Lyα ξ(r_∥, r_⊥) on BOSS DR9, recovers `b_DLA = 2.17 ± 0.20` at z ≈ 2.3.
- **Pérez-Ràfols et al. 2014** ([arXiv:1405.3994](https://arxiv.org/abs/1405.3994)) — same observable measured in hydrodynamic simulations; provides the sim-validation pattern we follow.

## 1. Coordinate system

PRIYA stores per-sim 691 200 sightlines on a regular 480 × 480 lateral
grid replicated along three orthogonal LOS axes (x, y, z). The HDF5
schema (verified 2026-04-27 against `ns0.803…/output/SPECTRA_017/lya_forest_spectra_grid_480.hdf5`) is:

| Group / dataset | Shape | Meaning |
|---|---|---|
| `Header.box` | scalar | comoving box side in **kpc/h** (= 120 000 in production) |
| `Header.hubble` | scalar | dimensionless `h` |
| `Header.redshift` | scalar | `z_snap` |
| `Header.Hz` | scalar | `H(z)` in km/s/Mpc |
| `spectra/axis` | (691 200,) int32 | LOS axis ∈ {1, 2, 3}; **1-indexed** (per fake_spectra) |
| `spectra/cofm` | (691 200, 3) float64 | sightline anchor (x, y, z) in **kpc/h** |
| `tau/H/1/1215` | (691 200, 1250) float32 | τ per (sightline, pixel) |

The pixel pitch along the LOS is

```
dx_pix = box_kpch / nbins                   # 120000 / 1250 = 96 kpc/h
dv_pix = dx_pix · H(z) / (a · 1000)          # ≈ 10 km/s at z=3
```

so the LOS runs the full box (12 510 km/s at z = 3) and the periodic
box is closed exactly.

For a sightline `i` with `axis = a` (1-indexed), pixel `j ∈ [0, 1249]`,
the comoving 3D position is

```
x[k] = cofm[i, k]                 if k ≠ a − 1
x[a−1] = (cofm[i, a−1] + j · dx_pix) mod box
```

(`mod box` because PRIYA writes `cofm[i, a−1] = 0` and the pixel index
runs from 0 to box).

All distances below are in **comoving Mpc/h**. We convert from kpc/h
inside the loader.

## 2. The Lyman-α flux field — all HCDs masked

Per pixel, after **masking every absorber in the catalog** (LLS + subDLA + DLA):

```
mask_ij = True  if pixel j on sightline i lies in [pix_start, pix_end]
                of ANY catalog row whose NHI ≥ 10^17.2

F_ij    = exp(−τ_ij)             on unmasked pixels
        = ⟨F⟩                     on masked pixels       # fill so δ_F = 0
⟨F⟩     = mean of F over UNMASKED pixels in the snap
δ_F_ij  = F_ij / ⟨F⟩ − 1
```

This is the production "all-HCD-masked" δ_F. We construct it at
analysis time directly from `tau/H/1/1215` + `catalog.npz`, **without
requiring a new saved variant** (`p1d.npz` only has `all` and
`no_DLA_priya`; neither matches our spec).

The motivation is that LLS / subDLA pixels carry their own clustering
signal which would otherwise contaminate the DLA × Lyα cross-corr at
small r. Masking all HCDs gives a "pure forest" δ_F whose only
remaining bias is the underlying intergalactic Lyα bias `b_Lyα`. The
PRIYA-style `no_DLA_priya` mask is **not enough** for this purpose
because it leaves LLS / subDLA pixels in place.

We do **not** rescale τ to a target `⟨F⟩_obs` — PRIYA's native UVB
mean flux is used. (See `docs/assumptions.md` item 11.)

## 3. The DLA point catalog

For each absorber row in `catalog.npz` with `NHI ≥ 10^20.3`:

* lateral 3D position: `cofm[skewer_idx]` projected onto the two axes
  perpendicular to `axis`;
* LOS coordinate: pixel-flux-weighted centre of `[pix_start, pix_end]`
  along `axis`, then `(cofm[…, a-1] + j_centre · dx_pix) mod box`.

Each DLA is therefore a single 3D point in the box. We do **not**
deconvolve the 100 km/s `merge_dv_kms` window; the merged-system
centre is the catalog's reported pixel range.

The position is in **redshift space** because PRIYA τ already includes
peculiar velocities (assumption 19). This is the right thing — FR+2012
also measure in redshift space.

## 4. The cross-correlation ξ(r_∥, r_⊥)

For each (DLA d, Lyα-pixel ℓ) pair we form

```
Δ⃗   = x_d − x_ℓ                        (apply minimum-image periodic wrap)
r_∥ = | Δ⃗ · ê_LOS,ℓ |                   (along ℓ's sightline axis)
r_⊥ = √(|Δ⃗|² − r_∥²)                    (transverse to that axis)
```

**Sign convention.** The pair counter tracks **signed** `r_∥`
internally (sign of the dot product `Δ⃗ · ê_LOS,ℓ`). For the science
panel and the bias fit we fold to `|r_∥|`. The signed version is
preserved so we can run a symmetry test
`ξ(+r_∥, r_⊥) = ξ(−r_∥, r_⊥)` — any deviation flags a systematic
(light-cone evolution across the box, asymmetric mask leakage, or a
bug in the pair coding). This goes in `tests/test_clustering.py`
(test 7b).

Estimator (continuous-field × point-set):

```
ξ_×(r_∥, r_⊥) =  ⟨ δ_F · 𝟙_DLA-near ⟩  /  ⟨ 𝟙_DLA-near ⟩

              =  Σ_{(d, ℓ) ∈ bin} δ_F_ℓ
                 ──────────────────────
                  N_pairs in bin
```

(no random catalog needed because ⟨δ_F⟩ = 0 by construction). This is
the form FR+2012 use (their eq. 5).

**Binning**

* `r_⊥` ∈ [0, 50] Mpc/h, 25 linear bins of width 2 Mpc/h
* `r_∥` ∈ [0, 50] Mpc/h, 25 linear bins of width 2 Mpc/h

Linear-bias fitting window: `r_⊥ ∈ [10, 40]`, `r_∥ ∈ [10, 40]` Mpc/h
(see §6).

## 5. The auto-correlation ξ_DD(r_∥, r_⊥)

Standard Landy–Szalay on a periodic box. For a population of N DLAs:

```
ξ_DD(r_∥, r_⊥) = (DD − 2·DR + RR) / RR
```

with `DD`, `DR`, `RR` natural-pair counts. On a periodic box we can
short-cut by using the analytic random-pair count `RR_analytic =
n̄_DLA² · V_bin · V_box`, which removes the largest source of randomness
in `RR`. Implementation will start with this and fall back to a
finite random catalog if the analytic form looks suspect at large r.

We bin in the same (r_∥, r_⊥) grid as the cross.

## 6. Linear bias extraction

Both estimators reduce to the linear-theory prediction at large r:

```
ξ_×  (r) ≈ b_DLA · b_Lyα · (1 + (β_DLA + β_Lyα) μ² + β_DLA β_Lyα μ⁴) · ξ_lin(r)
ξ_DD (r) ≈ b_DLA²        · (1 + 2 β_DLA μ² + β_DLA² μ⁴)              · ξ_lin(r)
```

where `μ = r_∥ / r`. For the **monopole** (μ-averaged):

```
ξ_×^(0)  ≈ b_DLA · b_Lyα · (1 + ⅓(β_×) + ⅕(β_×')) · ξ_lin^(0)
ξ_DD^(0) ≈ b_DLA²        · (1 + ⅔ β_DLA + ⅕ β_DLA²) · ξ_lin^(0)
```

For the **first-pass validation** we fit the monopole only (ignoring
β by absorbing it into an effective bias `b_eff`); then we redo with
the full Kaiser model.

`ξ_lin(r)` is the linear matter correlation function at the snap's
redshift. We compute it from CAMB power spectra, which are saved per
sim in
`/nfs/turbo/umor-yueyingn/mfho/emu_full/<sim>/output/powerspectrum-<a>.txt`
(see `_class_params.ini`).

## 7. Cross-validation: the auto–cross consistency check

The acid test is

```
b_DLA from auto    =    b_DLA from cross
```

If those numbers disagree at the > 1σ level, *something is wrong with
one of the pipelines* (likely the `b_Lyα` we plug in for the cross,
or peculiar-velocity contamination in the auto). PR+2014 do this
exact comparison.

For the cross we need an independent `b_Lyα`. We compute it
**in-house** from the same snap, in `hcd_analysis/lya_bias.py`, by
fitting linear theory to the **all-HCD-masked P1D** that we
reconstruct on the fly (same masking as in §2 — not the saved
`no_DLA_priya` variant, which leaves LLS / subDLA pixels in place).
The fit is on linear scales (`k_angular ∈ [0.001, 0.005] rad·s/km` at
z = 2.3, scaling with H(z)) using a McDonald 2003-style template
`P1D(k) = b_F² · D(k, β_F) · P_lin(k)`.

The fit comes with its own unit tests + hypothesis tests
(`tests/test_lya_bias.py`):
* recovery of an injected `b_Lyα` from a simulated linear forest at
  the 2 % level;
* sanity of the recovered number against the literature reference
  `b_F(z=2.3) ≈ −0.18 ± 0.02` (Slosar et al. 2011, McDonald 2003) —
  must agree to within factor 2 or we abort and investigate.

## 8. Validation plan (must all pass before the production run)

`tests/test_clustering.py` will lock these claims **before** any
science number is published. Pattern follows
`tests/test_absorption_path.py`.

| # | Test | Pass criterion |
|---|---|---|
| 1 | Coordinate round-trip | `(skewer, axis, pixel) → (x,y,z) → nearest sightline & pixel` recovers input on a 1-pixel grid |
| 2 | Periodic minimum-image | random `Δx` in `[−box, box]` returns `|Δx| ≤ box/2` after wrap |
| 3 | r_∥ + r_⊥ identity | `r_∥² + r_⊥² = |Δ⃗|²` to FP for 10 000 random pairs |
| 4 | Random Poisson DLAs vs random Lyα → ξ_× = 0 | mean over (r_∥, r_⊥) bins consistent with 0 within Poisson error |
| 5 | Random Poisson DLAs auto → ξ_DD = 0 | same, after subtracting Poisson shot noise |
| 6 | Periodic-box closure | sum of `RR_analytic` over all bins equals total point pairs in box |
| 7a | Coordinate symmetry | swapping x and y axes leaves ξ unchanged on a uniform field |
| 7b | LOS symmetry | `ξ(+r_∥, r_⊥) − ξ(−r_∥, r_⊥)` consistent with 0 across all bins (signed pair counter check) |
| 8 | Bias recovery on lognormal mock | London 2019-style mock (Gaussian field → exponentiate → Poisson-sample tracers with input b_DLA, β_DLA, b_F, β_F → impose Kaiser RSDs) recovers `b_DLA = 2.0 ± 0.1` from cross monopole. **Fallback** if lognormal proves too brittle: GRF mock (no Poisson, just `δ_g = b·δ_m` on a grid) — same recovery target, simpler stochastics. |
| 9 | Auto–cross consistency on the same mock | `b_DLA(auto) − b_DLA(cross)` consistent with 0 at 1σ |
| 10 | FR+2012 sanity on real PRIYA | one snap at z ≈ 2.3 gives `b_DLA ∈ [1.7, 2.5]` (FR+2012's central value ± 1.5 σ) |

Tests 4, 5, 6, 7 are deterministic and run in CI. Tests 8, 9 are
seeded-stochastic and must be reproducible. Test 10 runs against
`/scratch/.../hcd_outputs` data and is not a CI gate but is required
before the 60-sim production sweep.

## 9. Conventions and gotchas to remember

* **`spectra/axis` is 1-indexed** (1 = x, 2 = y, 3 = z). Every loader
  must subtract 1 before using as a NumPy index.
* **`cofm` is in kpc/h, not Mpc/h.** Convert immediately on load.
* **Mean flux uses the masked field**, not the raw τ-derived flux.
  Forgetting this introduces a spurious DLA bias.
* **`r_∥` is computed against the *Lyα pixel's* sightline axis**, not
  the DLA's sightline axis. They are usually different (the DLA can be
  far from the pixel laterally).
* **Periodic minimum-image always.** The box is small (120 Mpc/h);
  ignoring periodicity at r ≳ 60 Mpc/h gives wrong ξ.
* **The 100 km/s `merge_dv_kms`** in the absorber finder means two
  closely-spaced systems become one DLA — so the catalog systematically
  under-counts DLA pairs at small r_∥. We do not correct this; we just
  exclude r_∥ < 10 Mpc/h from the bias fit.

## 10. Implementation plan

This document gates the implementation. After user approval:

1. `hcd_analysis/clustering.py` — coordinate conversion, ξ_× and ξ_DD pair counters, bias fitter.
2. `hcd_analysis/lya_bias.py` — `b_Lyα` calibrator from `no_DLA_priya` P1D.
3. `tests/test_clustering.py` — tests 1–9 above (test 10 is in `tests/validate_*.py`).
4. `scripts/run_clustering_one_snap.py` — one-snap driver.
5. `scripts/plot_clustering_validation.py` — produces `figures/analysis/04_clustering/{xi_cross_2d.png, xi_auto_2d.png, bias_recovery.png, fr2012_sanity.png}`.

Production (60 LF sims × all snaps) is **deferred** until tests 1–9
pass and test 10 lands within the FR+2012 window.
