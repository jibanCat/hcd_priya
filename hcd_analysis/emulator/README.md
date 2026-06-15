# HCD-marginalised Lyα P1D emulator (`hcd_analysis.emulator`)

This is the emulator at the centre of the Lyman-α cosmology analysis. The document describes what
it does, why it is built the way it is, and how to run it.

### Overview

We measure cosmology from the Lyα forest, the absorption lines that intervening hydrogen imprints
on the spectrum of a distant quasar. The summary statistic is the P1D, the one-dimensional flux
power spectrum, which records how much absorption variance lives on each spatial scale along the
line of sight; the cosmology is encoded in its shape.

Some of those absorption lines arise in dense gas systems, the HCDs (high-column-density
absorbers: Lyman-limit systems, subDLAs and damped Lyα systems). We split the three classes by
neutral-hydrogen column density N_HI (`hcd_analysis/catalog.py`): LLS at 10^17.2–10^19, subDLA at
10^19–10^20.3, DLA at N_HI ≥ 10^20.3 cm⁻². They contaminate the P1D and bias the cosmology if they
are ignored, so we require a model of the P1D that separates the clean forest from the HCD
contamination and lets us marginalise the HCDs away.

A full hydrodynamic simulation is far too slow to run inside a sampler. We therefore train a fast
emulator, a neural network that learns the simulation's P1D as a smooth function of its inputs, and
make it differentiable so that a gradient-based sampler can explore it efficiently.

This package is that emulator. It predicts the per-class P1D `P_filt(θ, z, τ₀)`, one spectrum each
for the clean forest, LLS, subDLA and DLA, as a function of:
- `θ`, the 9 cosmological and astrophysical parameters,
- `z`, the redshift,
- `τ₀`, the mean-flux optical depth (how absorbed the forest is on average; defined below).

It mirrors the PRIYA simulation suite (Bird, Fernandez, Ho et al. 2023, JCAP 10 037,
[arXiv:2306.05471](https://arxiv.org/abs/2306.05471); extended box from Fernandez, Bird & Ho 2024,
JCAP 07 029, [arXiv:2309.03943](https://arxiv.org/abs/2309.03943)), so that anyone familiar with
PRIYA can drive it. It is end-to-end autodifferentiable, so the likelihood
(`inference.log_lik_multiz`) is ready to pass to a gradient sampler: NUTS, the No-U-Turn Sampler, a
Hamiltonian Monte Carlo variant, via numpyro/blackjax, or a PRIYA-style Cobaya adapter.

Note that the sampler and Cobaya wrappers are still forthcoming (Phase-C T4, not yet shipped).
For now the differentiable likelihood is called directly, which is sufficient for real inference.

This README serves both as a reproduction, install and quickstart guide and as a tour of the
validation evidence. The full validation verdicts (the held-out error, the coverage and the
cosmology-bias gate) are kept in the private notes repository and linked from each demo below; the
headline validation document is
`hcd_priya_notes/docs/superpowers/2026-06-14-validation-loso-emulator-lf-mf.md`.

### Where to go

To build and run it:
1. [Environment](#1-environment-mandatory) — the environment string it must be run with
2. [The τ₀ cache](#2-the-τ₀-cache) — the training data: what it is and how to load it
3. [Architecture](#3-architecture) — the structure of the network and the reasons for it
4. [Training](#4-training) — how to run it and what a healthy run looks like
5. [Forward model / prediction](#5-forward-model--prediction) — turning the network into a P1D
6. [Quickstart](#6-quickstart) — a self-contained snippet
7. [Conventions & gotchas](#7-conventions--gotchas) — the points that commonly cause trouble
8. [Likelihood & inference](#8-likelihood--inference) — the differentiable log-likelihood and priors
9. [Module map](#9-module-map)

To see that it works, the demos are:
- [Demo A — Mock inference (closure / SBC)](#demo-a--does-the-inference-recover-the-truth-closure--sbc)
- [Demo B — A corner plot, reproduced](#demo-b--a-posterior-corner-plot-and-how-to-make-one)
- [Demo C — Does emulator error bias the cosmology?](#demo-c--does-emulator-error-bias-the-cosmology-the-loso-fisher-gate)
- [Demo D — HCD statistics vs the literature](#demo-d--hcd-statistics-the-emulator-vs-the-literature)
- [Demo E — How HCDs nudge the cosmology](#demo-e--how-hcds-nudge-the-cosmology-the-llsforest-degeneracy)

---

## 1. Environment (mandatory)

The emulator runs in float64 (double precision) because several of its identities hold only
exactly and break under the default float32. It therefore lives in a dedicated conda environment,
and every script must be launched with the same environment string.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3  <script>
```

What each piece does:
- `import hcd_analysis.emulator` runs `jax.config.update("jax_enable_x64", True)`; x64 is mandatory.
  The structural identities (`P_tier_p = Σ_c w_c·P_filt`, the telescoping `Σ w_c = 1`) hold at the
  bit level and break under float32.
- `PYTHONNOUSERSITE=1` prevents stray packages in the home directory from leaking into the import
  path.
- `JAX_PLATFORMS=cpu` and `CUDA_VISIBLE_DEVICES=""` pin execution to the CPU, which is deterministic
  and avoids GPU surprises.
- The JAX/Equinox traps we have already encountered are logged in
  `docs/superpowers/jax-traps-log.md` (for example trap #29: sanitise `jnp.interp` y-data before
  the call, not after).

This yields a working float64 JAX runtime that can `import` the package.

---

## 2. The τ₀ cache

The cache is the emulator's entire training set, held in a single HDF5 file. It is a table of P1D
spectra measured from the PRIYA low-fidelity (LF, the cheaper, larger-volume simulation box) runs,
augmented by the τ₀ ladder described next.

τ₀ is the mean-flux optical depth, `τ₀ = −ln(target_F)`, where `target_F` is the average
transmitted flux of the forest. It sets how absorbed the forest is on average, and its observed
value drifts with redshift. Rather than rely on the simulation having landed on the correct mean
flux, we resample each simulation's mean flux up and down a ladder of rungs (PRIYA's
`mean_flux="per_z"` post-processing). The emulator then learns the τ₀ dependence directly and can
be evaluated at whatever mean flux the data require.

The ladder is anchored on the Kim et al. 2007 mean-flux fit `τ_eff(z) = 0.0023·(1+z)^3.65`
(MNRAS 382, 1657, [arXiv:0711.1862](https://arxiv.org/abs/0711.1862)) and built wide enough to
bracket the observed mean flux across the analysis redshift range.

The figure below shows that the trained ladder (the shaded band) brackets every observed mean-flux
measurement across z = 2.2–4.6, so the emulator never extrapolates in mean flux inside the
analysis.

![Trained τ₀ ladder bracketing observed mean flux](../../figures/analysis/04_emulator/tau0_ladder_vs_obs_meanflux.png)

*The shaded band is the trained τ₀ ladder; the lines/points are observed-⟨F⟩ fits and data
(Kim+2007, Becker+2013, Turner+2024, XQ-100). The ladder brackets the data across z = 2.2–4.6.*

The cache file is `hcd_analysis/_emulator_data/observables_tau0_lf.h5`, the v3.3 LF cache: 21 440
rows, n_k = 172 k-bins, angular k, a 20-rung τ₀ ladder, 60 LF sims and 4 HCD classes. It is built
by `scripts/build_emulator_cache_tau0.py`.

Load it with:

```python
from hcd_analysis.emulator.data import load_cache
d = load_cache("hcd_analysis/_emulator_data/observables_tau0_lf.h5")
```

`load_cache` returns a dict; the keys you will reach for most:

| key | shape | meaning |
|---|---|---|
| `params` / `params_unit` / `x` | `(R,9)` / `(R,9)` / `(R,10)` | physical / unit-cube params; `x` = `[θ9, z_unit]` |
| `kfkms` | `(K,)` | angular k-grid in s/km (`K = n_k = 172`) |
| `P_filt` | `(R,4,K)` | per-class P1D: clean, LLS, subDLA, DLA |
| `delta` | `(R,3,K)` | per-class core add-back; `delta[:,2]` = `dla_core` |
| `coarse_counts` / `w_c_cache` | — | per-class sightline counts / incidence weights `w_c` |
| `target_F`, `tau0` | `(R,)` | global mean flux, `tau0 = −ln(target_F)` |
| `z_grid` | `(R,)` | redshift per row |

The four classes are absorber types, ordered by neutral-hydrogen content: clean (no HCD), LLS
(Lyman-limit systems), subDLA and DLA (damped Lyα). PRIYA internally carries many fine tiers, which
we collapse into these four coarse classes (`COARSE_SLICES = clean=tier0, LLS=1–7, subDLA=8–12,
DLA=13–14`).

The 9 input parameters span very different scales (the amplitude A_p ≈ 1e-9 against the tilt
n_s ≈ 0.9). A network cannot learn well from raw inputs that differ by nine orders of magnitude, so
we map everything to a unit cube before training (see §7).

![Training-data parameter distributions](../../figures/analysis/04_emulator/data_params_hist.png)

*The 9 PRIYA design parameters + redshift across the cache rows. The mismatched scales are exactly
why `data.normalize_params` maps to the unit cube before training.*

The loader returns a dict of arrays, the per-class P1D over `(θ, z, τ₀)` together with the
structural weights, which constitutes the emulator's complete training set.

---

## 3. Architecture

The design shares one feature extractor and then splits into three small heads, each responsible
for one well-defined physical task: `model.Emulator` (an `eqx.Module`) is
Encoder → {HeadA, BaselineHead, HeadB}.

![Emulator architecture](../../figures/analysis/04_emulator/emulator_architecture.png)

*The Encoder builds a shared latent; HeadA emits the τ₀-invariant CDDF/incidence; the BaselineHead
emits a θ-blind P_filt baseline and HeadB the θ-dependent residual; the structured mean recombines
them and the downstream block assembles `P_tier_p` / `P_obs` for the likelihood.*

```
x = [θ9 (9), z_unit (1)]  ─► Encoder (MLP 256-128-64) ─► latent (64)
                                         │
   HeadA(latent)            ── τ₀-INVARIANT  ─► f_nhi, dN/dX (CDDF / incidence; n_k_cddf=30)
   BaselineHead(z, τ₀)      ── θ-BLIND       ─► m̂  : P_filt baseline   (4×n_basis coeffs)
   HeadB(latent, τ₀)        ── θ-DEPENDENT   ─► r̂  : P_filt residual    (4×n_basis coeffs)
```

The reason for this split is the Kennedy–O'Hagan structured mean (Kennedy & O'Hagan 2001, JRSS-B
63, 425), a θ-blind simulator term plus a learned discrepancy. Rather than predict the whole
spectrum with a single black box, we predict a fixed baseline that does not depend on cosmology
plus a small learned correction that carries all of the cosmology dependence, so that the cosmology
signal is concentrated in a single term we can isolate and check. We write each per-class log-power
as a θ-blind baseline plus a whitened cosmology residual:

```
logP̂(θ,z,τ₀) = ( m̂·σ_marg + μ_marg )  +  σ_cosmo · r̂
P_filt        = exp(logP̂)                         # (4,K) LINEAR: clean, LLS, subDLA, DLA
```

The advantages of this construction are:
- The cosmology enters only through `r̂` (the baseline is θ-blind), so
  `∂logP̂/∂θ = σ_cosmo·∂r̂/∂θ`. The entire cosmology response is concentrated in one isolatable
  term, which is both easy to check (the gradient gate `scripts/diag_grad_fidelity.py`) and cleanly
  identifiable.
- `μ_marg, σ_marg, σ_cosmo` are per-`(class, k)` normalisation statistics fit on the train split and
  stored in the checkpoint's `norm["P_filt"]` dict (see §4 and §6).
- The low-rank P_filt bottleneck (`n_basis = 24`): rather than predict all 172 k-values directly,
  both heads emit `4×n_basis` coefficients that decode through a trainable, SVD-warm-started basis
  `(n_basis, n_k)`. Fewer free parameters give smoother spectra and less over-fitting than a dense
  `4×n_k` output.
- `K = n_k = 172` k-bins (LF cache), in the angular k convention (see §7).
- HeadB also carries a legacy dense `delta` head, which the live forward model does not use (the
  HCD excess is computed from `P_filt` instead, see §5). It is harmless dead weight.

The multi-fidelity layer (`multifidelity.py`) addresses the fact that the LF backbone above is
LF-only and never sees the expensive high-resolution (HF / HiRes) box. A separate, frozen `ρ(k,z)`
correction layer lifts each LF prediction to HR resolution. The default deployed correction is the
ρ(k,z)-only `FixedMeanHead`, a mean-correction head (it returns `ḡ(z,k) − log ρ`) rather than a
mean-flux head; the mean flux is handled structurally through τ₀. The quantitative importance of
the LF→HR correction is shown in the notes validation document §3.

The result is a differentiable per-class P1D predictor whose cosmology response lives in a single,
isolatable residual term.

---

## 4. Training

Training is driven by the eight-fold LOSO sweep `scripts/run_loso_sweep.py`. LOSO,
leave-one-simulation-out, partitions the 60 simulations into eight groups; for each fold we hold
out a whole group of simulations, train on the rest, and test on the held-out group. Because the
held-out simulations are entire cosmologies the network has never seen, this is an honest test of
generalisation rather than of memorisation. Each fold trains one emulator (`train.train_fold`),
reuses the SVD warm-start, collects stratified fractional residuals, and writes `error_vector.npz`
and figures into `figures/analysis/04_emulator/`.

The frozen production recipe (`FINAL_RECIPE`):

```python
n_basis=24, p_resid_w=8.0, edge_gain=3.0, lowk_extra=2.0,    # low-rank + k-weighting
w_coh=80.0, weight_decay=3e-4, datarange=True,               # de-bias + soft data-range
epochs=180, patience=25, lr=1e-3, batch=512,                 # optax AdamW + cosine decay
```

Run it:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/run_loso_sweep.py \
    --out checkpoints/final --histdir checkpoints     # (defaults == FINAL_RECIPE)
```

Add `--smoke` for a fast shape and finiteness check. A θ-blind baseline pre-fit
(`train._prefit_baseline`) runs before the joint fit so that the baseline is structurally
identifiable.

In a healthy run the training loss and the validation loss fall together and flatten with no gap
between them. A gap would indicate over-fitting, the network memorising the training simulations,
which the regularised residual head is designed to prevent.

![Fold-0 train/val loss](../../figures/analysis/04_emulator/train_val_loss_fold0.png)

*A clean joint-loss curve: train and val track each other and plateau — no over-fitting.*

There is an important distinction between the folds and the production model. The eight LOSO folds
exist for closure validation and for calibrating the emulator-error budget (`C_emu`). The deployed
point prediction uses a single model; the folds should not be ensembled at inference. The
fold-to-fold spread is already captured as the σ budget in `error_vector.npz` and enters the
likelihood through `C_emu`. Before the real fit we train an all-sims production emulator with no
hold-out; `final_fold0` is the canonical stand-in for now.

This stage yields one checkpoint bundle per fold (see §6) together with the LOSO error vector.

---

## 5. Forward model / prediction

The forward model turns the network's per-class spectra into the single contaminated P1D the data
actually measure, while allowing the HCD contamination to be dialled up and down. In the
2026-06-04 redesign it is a clean-forest baseline plus a free-amplitude per-class HCD excess:

```
R_c   = P_c − P_clean          # (3,K) excess; FILTERED for LLS/subDLA, UNFILTERED for DLA
P_obs = P_clean + Σ_{c∈HCD} α_c · R_c
      ≡ P_clean · [ 1 + Σ_c α_c (P_c/P_clean − 1) ]      # additive ≡ multiplicative
```

Reading it:
- `α_c` is the effective post-masking per-class incidence, how often each HCD type survives into the
  data. Setting `α_c = w_c` reproduces the simulation's contaminated `P_tier_p` (up to the DLA-core
  add-back). The prior on `α_c` is centred on the observed dN/dX rather than PRIYA's simulation
  value, since the simulations and the data differ; see `inference.hcd_incidence_prior`.
- DLA uses the **unfiltered** template: `P_DLA^unf = P_filt[DLA] + dla_core`, where `dla_core`
  (`= P_DLA^unf − P_DLA^filt`) is `cache["delta"][row, 2]`.
- `∂P_obs/∂α_c = (P_c − P_clean) ≠ 0` for LLS now (this fixes an old `Δ_LLS ≡ 0` filter-residual
  bug).
- This is the field-standard **fixed-shape / free-amplitude HCD template** (cf. Rogers, Bird,
  Peiris et al. 2018, MNRAS 474, 3032, [arXiv:1706.08532](https://arxiv.org/abs/1706.08532); the
  4-class form is also what PRIYA-on-KODIAQ-SQUAD uses,
  [arXiv:2509.18271](https://arxiv.org/abs/2509.18271)). Unlike the fixed Rogers kernel, ours
  carries the θ, τ₀ sensitivity and preserves the global-⟨F⟩ normalisation. See
  `docs/superpowers/2026-06-04-phase-c-walkthrough.md` §6 and
  `[[hcd-template-rogers-normalization]]`.

The figure below shows an example fold-0 prediction (dashed) over the cache truth (solid), per
class. The top panels are the spectra and the bottom panels are the fractional residual
`pred/cache − 1`, restricted to the band we actually use (a ±5% axis). The grey shading marks k
above the LF Nyquist (k ≈ 0.069 s/km), where the network is neither used nor validated. The dotted
and dashed vertical lines mark the two analysis k-cuts, DESI k ≤ 0.041 and KODIAQ-SQUAD (KS)
k ≤ 0.06 s/km.

![Per-class predicted vs cache P1D, fold 0 — residual on the used band](../../figures/analysis/04_emulator/pred_vs_true_p1d_fold0_inrange.png)

***(a) What the figure shows.*** A fold-0 example prediction (dashed) vs the cache (solid), per HCD
class. Inside the analysis band the residual is ~1–2% (median ≈ 0.7%); the grey region (above the
Nyquist) is not used. The rigorous all-folds error lives in the notes validation doc
(`2026-06-14-validation-loso-emulator-lf-mf.md` §2), which resolves the held-out error per
wavenumber k — the rigorous "where in k does the error live" answer. Demo C below shows the
bottom line for cosmology: the per-fold A_p/n_s bias.

***(b) Why an older plot looked ±20%.*** The previous version of this plot showed the
residual over the full raw k-grid and reached ±20% at the k-extremes. That ±20% is entirely at
the lowest, cosmic-variance-sparse modes (k < 0.005, where individual sims are noisy) and above
the Nyquist (k > 0.069, where the LF emulator is not used, and is never in the analysis band). The
plot above restricts to the band we actually use, which is why the residual is smaller.

***(c) Reproduce.*** Diagnosis numbers and this figure are produced by
`scripts/diag_pred_vs_true_honest.py`.*

The functions you will call (all JAX-pure, differentiable in `θ9, τ₀, α`):

| function | returns |
|---|---|
| `predict_P_filt(model, θ9, z_unit, τ₀, pf)` | `(4,K)` per-class P_filt |
| `predict_excess(model, θ9, z_unit, τ₀, pf, dla_core)` | `(3,K)` excess R_c |
| `predict_P_obs(model, θ9, z_unit, τ₀, α_hcd, pf, dla_core)` | `(K,)` total P_obs |
| `predict_P_tier_p(model, θ9, z_unit, τ₀, w_c, pf)` | `(K,)` structural Σ w_c·P_filt (clean-path diagnostics only) |

`predict_P_obs` and `predict_excess` are differentiable in `(θ9, τ₀, α)`; `predict_P_filt` in
`(θ9, τ₀)`. Here `pf = norm["P_filt"]` (the structured norm dict, see §6).

This yields the total contaminated P1D `P_obs(k)`, and its building blocks, differentiable in the
cosmology, the mean flux and the HCD incidence, ready to feed the likelihood.

---

## 6. Quickstart

The snippet below is self-contained; it builds the inputs from a cache row so that the whole flow
is visible at once.

```python
import hcd_analysis.emulator                        # enables JAX float64 on import
import jax.numpy as jnp
from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.data import load_cache
from hcd_analysis.emulator.predict import predict_P_obs, predict_P_filt

model, meta, norm = T.load_checkpoint("checkpoints/final_fold0")
pf = norm["P_filt"]                                  # structured P_filt norm dict (mu/sig_marg, sig_cosmo)

d   = load_cache("hcd_analysis/_emulator_data/observables_tau0_lf.h5")
row = 0                                              # or construct your own inputs — see below
theta9 = jnp.asarray(d["params_unit"][row])          # (9,) UNIT cube; else (θ_phys−lo)/(hi−lo)
z_unit = float(d["x"][row, 9])                       # = (z − 2.0)/3.4
tau0   = float(d["tau0"][row])                       # = −ln(target_F)
alpha  = jnp.asarray(d["w_c_cache"][row, 1:])        # (3,) per-class incidence (LLS,subDLA,DLA)
dla_core = jnp.asarray(d["delta"][row, 2])           # (K,) DLA-core add-back

P_filt = predict_P_filt(model, theta9, z_unit, tau0, pf)                  # (4,K) clean,LLS,subDLA,DLA
P_obs  = predict_P_obs(model, theta9, z_unit, tau0, alpha, pf, dla_core)  # (K,)  total
```

All emulator inputs are on the unit cube.

The checkpoints (`checkpoints/`) are four-file bundles per fold,
`final_fold{0..7}.{eqx, meta.json, norm.pkl, hist.json}`:

- `.eqx` — Equinox leaves; `.meta.json` — `arch_cfg` + `seed`; `.norm.pkl` — train-split norm stats
  (the `P_filt` dict with `mu_marg / sig_marg / sig_cosmo`, + f_nhi/dndx/delta stats);
  `.hist.json` — per-epoch loss history.
- `T.load_checkpoint(path)` → `(model, meta, norm)`.

**Error vector** (`checkpoints/error_vector.npz`) — the emulator-error budget for the likelihood:
`sigma (4, K=172, Zb=3, Tb=4)` = per-(class, k, z-band, τ₀-band) RMS fractional LOSO residual,
plus `tau0_band_centres (4,)`, `z_band_edges`, `dla_shot_flag (K,)`. Consumed by
`likelihood.sigma_at_tau0` + the per-class `C_emu`.

To build your own inputs instead of reading a cache row:

- Parameters (`data.PARAM_LIMITS`, in the same order as PRIYA's `coarse_grid`):
  `[ns, Ap, herei, heref, alphaq, hub, omegamh2, hireionz, bhfeedback]`. Feed the emulator the
  **unit-cube** `θ9 ∈ [0,1]^9`: `θ9 = (θ_phys − lo)/(hi − lo)` (or use `data.normalize_params`).
- `z_unit = (z − 2.0)/(5.4 − 2.0)` — a linear map over `data.Z_LIMITS = (2.0, 5.4)` (e.g. z = 3 →
  0.2941).
- `τ₀ = −ln(target_F)` (the per-z mean-flux optical depth).
- `α_hcd` = `(3,)` per-class incidence (LLS, subDLA, DLA).

---

## 7. Conventions & gotchas

The following points have repeatedly cost time, and are worth reviewing before debugging.

- k is angular, `k = 2π/λ_v` in s/km, the community-wide convention (Croft et al., McDonald et al.,
  Palanque-Delabrouille et al., Rogers et al., PRIYA, DESI). `cache["kfkms"] = 2π·rfftfreq/dv`; pass
  it to the Rogers templates directly, with no extra `/2π`. See
  `[[hcd-template-rogers-normalization]]`.
- x64 is mandatory (see §1); the structural identities break under float32.
- PRIYA's n_s and A_p are forest-pivot quantities, defined at `k = 0.78 Mpc⁻¹` rather than the CMB
  pivot 0.05 Mpc⁻¹. Do not confuse `PARAM_LIMITS[ns/Ap]` with the CMB `n_s, A_s`.
- The τ₀ ladder coordinate `α_factor = τ₀ / Kim2007(z)` is the z-independent ladder axis
  (`data.tau0_ladder_factor`, with `Kim2007(z) = 0.0023·(1+z)^3.65`), the natural axis for the
  smooth `σ(τ₀)` interpolation. The code uses a `Kim2013` symbol name for this curve, but the
  underlying fit is Kim et al. 2007, [arXiv:0711.1862](https://arxiv.org/abs/0711.1862).
- The data range (`data.DATA_RANGE`) is z ∈ [2.2, 4.6], `k_min = 1e-3`. The cache is wider
  (z ∈ {2.0..5.4}, angular k ∈ [3.5e-4, 0.098]); out-of-range bins are softly down-weighted in
  training and excluded from `C_emu`. The 2026-06-04 decision was to keep `k_min = 1e-3`.
- The LF Nyquist is k ≈ 0.069 s/km. Above it the LF emulator is neither used nor validated. Both
  analysis k-cuts sit safely below it: DESI k ≤ 0.041 and KS k ≤ 0.06.
- The per-class power uses a single shared global ⟨F⟩ (`target_F`), not a per-subset mean, so the
  class offsets are real physics rather than a sightline-count artefact.

---

## Demo A — Does the inference recover the truth? (closure / SBC)

Before an inference pipeline can be trusted on real data, we run a closure test and its stricter
counterpart, simulation-based calibration (SBC): we feed the pipeline mock data drawn from a known
truth and check that it recovers that truth, within the stated error bars, without the sampler
misbehaving. A pipeline that fails closure has a bug or a mis-calibrated error model; one that
passes is trustworthy where it was tested. On held-out-sim mocks the pipeline recovers `n_s` in 8
of 8 cases within ±1σ, with no divergences (a divergence is the NUTS sampler's signal of a
pathological posterior geometry). For `A_p`, 6 of 8 fall within ±1σ; the two outside are a
high-n_s box-corner LOSO outlier and one +1.1σ mock, not a likelihood bug. The corner is the
sparse-design n_s ≳ 0.98 ridge where the held-out emulator extrapolates, and the production
interior cosmology sits away from it.

These closures validate the architecture, the forward model, the covariance and the sampler, that
the production all-sims N=5 ensemble inherits. The ensemble-level production SBC (rank-based
simulation-based calibration on the production ensemble itself) is the final inference-calibration
gate, and it remains pending before the real fit (notes
`2026-06-14-validation-likelihood-production.md` §6).

![Per-mock closure coverage detail](../../figures/analysis/06_validation_summary/coverage_summary.png)

*The **per-mock closure-coverage detail** (distinct from the root README's Leg-B / closure coverage
headline, despite the similar filename). Left: the HCD DESI-only closure bias `z = (θ̂ − θ_true)/σ`
per mock — `n_s` is 8/8 inside ±1σ. Middle: the α_subDLA mean bias (a known subDLA↔DLA degeneracy
that is harmless to cosmology and covered by the error model). Right: the eBOSS DR14 low-k recovery.
0 divergences throughout.*

To reproduce, the closure and SBC machinery is `hcd_analysis/emulator/closure_diagnostics.py`
(PIT-ECDF bands, rank tests, coverage). The coverage figure is built by
`hcd_priya_notes/figures/analysis/06_validation_summary/make_coverage_summary.py` from the
closure-chain outputs (`checkpoints/stepA/`). The full writeup is in notes
`docs/superpowers/2026-06-14-validation-loso-emulator-lf-mf.md` (validation summary) and the
closure plan `docs/superpowers/plans/2026-06-04-phase-c-t4-closure-plan.md`.

---

## Demo B — A posterior corner plot, and how to make one

A corner plot is the standard way to inspect a posterior: each diagonal panel is the 1D marginal of
one parameter and each off-diagonal panel is the 2D joint of a pair, so that the constraints and the
correlations between parameters can be read at once. The figure below is one of our eBOSS DR14
closure-certification posteriors; it uses mock data, so the dashed lines mark the known truth. It is
a healthy result: every truth line lies inside its contour, and the cosmology (`n_s`, `A_p`) and the
HCD incidences (`α_LLS`, `α_subDLA`, `α_DLA`) are recovered together.

![eBOSS closure corner — cosmology + HCD](../../figures/analysis/04_emulator/eboss_corner_cosmo_E_f6.png)

*The cosmo+HCD corner for the eBOSS E_f6 (Planck-`n_s`) closure mock. Dashed lines = truth. `n_s =
0.958 ± 0.020` recovers the input; the HCD incidences are constrained and consistent with truth.*

The exact steps to reproduce it are:

```bash
# 1. Run the eBOSS closure-cert chains (writes checkpoints/stepA/E_f6_c*.npz).
#    (Concurrent-pool launcher; the fiducials E_f5/E_f6/E_f7 are defined in run_stepA.build_config.)
PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya OMP_NUM_THREADS=1 \
  /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/run_eboss_cert.py --workers 10

# 2. Make the corner from those chains (getdist env). FID defaults to E_f6.
PYTHONPATH=/home/mfho/hcd_priya \
  /home/mfho/.conda/envs/emu-3.9/bin/python3 scripts/plot_eboss_corners.py E_f6
```

`plot_eboss_corners.py` emits two corners per fiducial (cosmology plus HCD, and the IGM nuisances)
with truth markers, into the notes `05_truth_validation/` directory. The full writeup is the
eBOSS-certification section of the notes validation docs (`05_truth_validation/`).

---

## Demo C — Does emulator error bias the cosmology? (the LOSO Fisher gate)

A small RMS error is necessary but not sufficient. What we require is that whatever residual error
the emulator carries does not push the recovered cosmology in a consistent direction; a
biased-but-precise emulator is worse than a noisy-but-unbiased one. We therefore project each
held-out fold's residual onto the (`A_p`, `n_s`) Fisher directions (the directions the data
actually constrain) and ask how many σ of cosmology bias the emulator error injects. This is the
appropriate measure of cosmological accuracy, more meaningful than any raw per-row spectrum.

The per-fold cosmology bias is everywhere well inside the ±0.2σ acceptance gate; none of the 8
folds fail on either parameter, with an RMS bias of 0.067σ (A_p) and 0.071σ (n_s). The emulator is
thus validated as unbiased in the cosmology on held-out simulations (the per-fold LOSO gate), not
merely as small-RMS. This is the held-out-sim cosmology gate, subject to the production-SBC caveat
of Demo A: the per-fold LOSO result is what the production ensemble inherits, and the
ensemble-level SBC is still pending.

![Per-fold A_p / n_s emulator bias across all folds](../../figures/analysis/04_emulator/emu_bias_allfolds.png)

*Per-sim `A_p` and `n_s` Fisher bias (in σ) over all 60 honestly-held-out sims (each scored by its
own held-out emulator), with the ±0.2σ gate drawn — i.e. unbiased *on held-out sims* (the per-fold
LOSO gate). The deployed forward (with the MF correction) tightens the pooled mean bias to n_s
+0.026σ / A_p +0.055σ (see the `_mf` summary). The per-fold RMS is 0.067σ / 0.071σ, 0/8 fail.*

To reproduce:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_emu_bias_allfolds.py
# the deployed forward (through the multi-fidelity layer):
#   scripts/diag_emu_bias_allfolds_mf.py   → figures/analysis/04_emulator/emu_bias_allfolds_mf.txt
```

The full writeup, with the acceptance gate, the per-fold numbers and the MF-forward G-gates, is in
notes `docs/superpowers/2026-06-14-validation-loso-emulator-lf-mf.md` §2b.

---

## Demo D — HCD statistics: the emulator vs the literature

Besides the P1D, the emulator's HeadA predicts the HCD statistics themselves: the dN/dX (the
incidence, how many absorbers of each class per unit absorption path) and the CDDF (the
column-density distribution function, `f(N_HI)`, how absorbers are distributed in neutral-hydrogen
column density). These are external, observable quantities and so provide a clean check: do the
simulation's HCD populations match what surveys see? They should, because the HCD-incidence prior
in the real fit is centred on these observed values; were the simulation badly off, the prior
rather than the data would be doing the work.

The per-class dN/dX tracks the observed literature across redshift (median ratios ≈ 0.98 for LLS,
1.31 for subDLA and 0.70 for DLA, so the simulation is in the right range, which is precisely why
the prior centre is the observed value rather than the simulation's), and the CDDF lies on top of
the standard literature fits across five decades of column density.

![PRIYA dN/dX vs observed literature, per class](../../figures/analysis/06_performance_walkthrough/A6_dndx_vs_literature.png)

*Per-class dN/dX vs z against the observed literature (LLS: O'Meara+2013 / Fumagalli+2013 /
Prochaska+2010; subDLA: Zafar+2013; DLA: Prochaska & Wolfe 2009), with the PRIYA/obs ratio below.
The companion CDDF-vs-literature figure is `06_performance_walkthrough/A7_cddf_vs_literature.png`.*

To reproduce:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/plot_dndx_vs_literature.py
PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/plot_cddf_vs_literature.py
```

HeadA's own prediction-versus-truth accuracy is in
`figures/analysis/04_emulator/head_a_dndx_pred_vs_true.png` and `head_a_fnhi_pred_vs_true.png`. The
background on the dN/dX → w_c → α maps is in module `dndx_wc.py` and `[[phase2c-hcd-redesign]]`.

---

## Demo E — How HCDs nudge the cosmology (the LLS↔forest degeneracy)

This is the underlying reason for the whole exercise. The HCD contamination is degenerate with the
cosmology, and the cleanest way to read that off is the posterior correlation matrix below, which is
the standard summary of a parameter degeneracy: each entry is the correlation between two parameters
under the data Fisher information plus the HCD priors. The cell that matters is `corr(A_HCD, n_s) =
+0.83`: the HCD amplitude and the tilt move together. The same column shows `corr(A_HCD, A_p) =
−0.84`, so `A_HCD` is strongly coupled to both forest-amplitude directions at once.

![Posterior correlation matrix (A_p, n_s, A_HCD, B_HCD, r_sub)](../../figures/analysis/04_emulator/degeneracy_fisher_corr_matrix.png)

*Posterior correlation matrix (data Fisher + HCD priors) over the two cosmology parameters (`A_p`,
`n_s`), the HCD amplitude `A_HCD`, its z-tilt `B_HCD` and the subDLA-to-LLS ratio `r_sub`. The
boxed cell is `corr(A_HCD, n_s) = +0.83`. The data-only correlation (no HCD prior) is ≈ 0: the
coupling is prior-mediated. `A_HCD` is pinned by its external prior, and that pinned amplitude reads
to the forest as a change in `A_p`/`n_s`. The z-tilt `B_HCD` is nearly orthogonal to `A_p`
(`corr ≈ −0.27`) but is data-starved, so it cannot break the coupling (see below).*

The mechanism is the damping wing. An HCD's low-k power excess, from its damping wing, looks over the
DESI band almost exactly like a change in the forest amplitude, and that low-k confusion is the
degeneracy: the data alone cannot fully separate "more LLS" from "more `A_p` or a different `n_s`."
That is why the off-diagonal correlation is large, and why it is *prior*-mediated rather than
intrinsic to the data. With no HCD prior the data place almost no constraint on `A_HCD`, so its
correlation with the cosmology is ≈ 0; it is only once the external incidence prior pins `A_HCD` that
the pinned amplitude propagates into the forest plane and the `+0.83` correlation appears. The
practical consequence is that a ±1σ shift in where we centre the HCD-amplitude prior drags `n_s` by
roughly 0.5σ. This factorises exactly as 0.5σ = corr(A_HCD, n_s) × prior-dominance ≈ 0.83 × 0.62, so
it is a real and irreducible systematic: a tighter prior shrinks the prior-dominance factor but
leaves the correlation untouched, and therefore cannot remove the coupling.

The geometry behind the correlation is the projection of the LLS template into the forest plane,
shown next. It quantifies the same degeneracy as an in-plane fraction: how much of the whitened LLS
template lies inside the forest (`A_p`, `n_s`) directions the data constrain.

![LLS template projected into the forest (A_p, n_s) plane](../../figures/analysis/04_emulator/degeneracy_lls_inplane_projection.png)

*The pure-LLS amplitude template projected into the C_data-whitened forest (`A_p`, `n_s`) plane.
Left (DESI leg): the template lies 93% inside the forest plane (`|cos(LLS, A_p)| = 0.84`; only ~7%
orthogonal). Right: how much of the template the forest can absorb — 93% on DESI's (`A_p`, `n_s`),
98% once the τ₀ nuisances are added, dropping to 65% on the high-k KS leg. These are the
single-fiducial Fisher ref-script (DESI-leg) values; the joint DESI+KS panel headline is 91%
in-plane on DESI / 64% on KS with `cos(LLS, A_p) = 0.81`, consistent to ~2% (the small offset is the
single-fiducial-vs-joint method difference, not a disagreement).*

The drop from 91% in-plane on DESI to 64% on the high-k KS leg is the degeneracy-breaker. It is not
the damping wing in general but specifically its high-k Voigt deficit: that high-k shape is what KS
sees and what separates LLS from a pure amplitude change (the cure), whereas the low-k excess is what
mimics the amplitude in the first place (the confusion). The z-tilt of the HCD incidence (`B_HCD`) is
genuinely orthogonal to `A_p`, but it carries far too little signal to help, which is why the
correlation matrix above is unchanged when it is marginalised.

To see this degeneracy in a full posterior rather than at the Fisher level, the eBOSS closure corner
of Demo B above (`eboss_corner_cosmo_E_f6.png`) shows the `n_s`–`α` contours directly. The
correlation matrix here is the primary summary; the corner is the worked example.

Two points follow for the real fit. First, the accuracy of the external LLS-incidence prior centre
matters, because a wrong centre is the 0.5σ-per-1σ systematic. Second, the genuine
degeneracy-breaker is the high-k KS leg together with the damping-wing high-k Voigt deficit (the
low-k excess is the confusion, the high-k deficit is the cure), not a tighter incidence prior. The
full derivation, the correlation matrix and the factorisation 0.5σ = corr × prior-dominance are in
notes `docs/superpowers/2026-06-14-hcd-cosmology-degeneracy.md`, with the related figures
`degeneracy_prior_center_law.png` and `degeneracy_snr_amplitude_vs_ztilt.png` in the notes
`04_emulator/`. All three correlation/projection figures are reproduced by
`scripts/diag_hcd_cosmo_degeneracy_ref.py`.

---

## 8. Likelihood & inference

This is the layer the demos above exercise: the differentiable Gaussian likelihood together with
the priors, which is everything a sampler needs.

- `inference.log_lik_single_z` / `log_lik_multiz` — the differentiable per-z / multi-z Gaussian
  log-likelihood. The per-class **emulator covariance** is
  `emu_var = Σ_c coef_c²·σ_c(k,z,τ₀)²·P_c²` with `coef = [1−Σα, α_LLS, α_subDLA, α_DLA]`;
  it is logdet-bearing (`likelihood.gaussian_loglik`, SPD jitter + Cholesky) and τ₀-aware
  (`likelihood.sigma_at_tau0`).
- `inference.hcd_incidence_prior` — the observed-centred, z-slope HCD incidence priors
  (`HCD_LIT_OVER_SIM`, `HCD_PRIOR_FRAC_SIGMA`, …). The θ prior is smooth and bounded (no `-inf`
  wall, so NUTS always gets finite gradients).
- `closure_diagnostics.py` — the SBC / closure machinery: `ecdf_pit_bands` (Säilynoja, Bürkner &
  Vehtari 2022, Stat. Comput. 32, 32 — simultaneous ECDF bands, our primary calibration gate),
  `loglik_rank` (Modrák et al. 2023, Bayesian Analysis,
  [arXiv:2211.02383](https://arxiv.org/abs/2211.02383)), `whitening_test`, `empirical_coverage`,
  `calibrate_cemu_inflate`. All NUTS-free and fast.

The closures these tools drive validate the architecture that the production all-sims N=5 ensemble
inherits; the ensemble-level production SBC is the final inference-calibration gate and remains
pending before the real fit (notes `2026-06-14-validation-likelihood-production.md` §6).

See `docs/superpowers/plans/2026-06-04-phase-c-t4-closure-plan.md` for the closure/SBC harness
(Phase-C T4, in progress) and `docs/superpowers/2026-06-04-phase-c-checkpoint-review.md` for the
latest 4-agent review status.

---

## 9. Module map

| file | role |
|---|---|
| `__init__.py` | enables x64; exports `CLASS_NAMES`, `HCD_CLASSES` |
| `data.py` | cache loader, `PARAM_LIMITS`, `DATA_RANGE`, coarse collapse, τ₀ bands, splits/batches, norm-stat fitting |
| `model.py` | `Emulator` = Encoder + HeadA + BaselineHead + HeadB; SVD basis; `structural_tier_p` |
| `multifidelity.py` | LF+HF combination, ρ(k,z) HiRes correction, `FixedMeanHead` |
| `train.py` | `train_fold`, optimizer/loss, baseline pre-fit, `save/load_checkpoint` |
| `predict.py` | differentiable forward: `predict_P_filt/excess/P_obs/P_tier_p` |
| `likelihood.py` | `sigma_at_tau0`, `gaussian_loglik`, covariance assembly |
| `inference.py` | per-z/multi-z log-likelihood + HCD incidence priors + θ/τ₀/α priors |
| `closure_diagnostics.py` | SBC/PIT-ECDF bands, whitening, coverage, cemu_inflate calibration |
| `dndx_wc.py` | dN/dX ↔ w_c ↔ α structural maps (M₀-inverse) |
