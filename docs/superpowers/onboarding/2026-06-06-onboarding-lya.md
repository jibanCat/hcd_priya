# Lyman-alpha forest / IGM physics / PRIYA sims onboarding report — 2026-06-06

Lens: is the forest physics modeling FAITHFUL — to the PRIYA simulations it is trained on,
to Rogers&Bird 2018 for HCD, to the literature for incidence/mean-flux, and to the data's
DLA-masked state? And is the documented "closing step" (thread the exact sim per-z w_c(z)
into the closure forward) physically defensible or circular?

All claims below are either (a) verified against the PRIYA paper text (extracted to
`/tmp/priya.txt` via pymupdf, 105835 chars — `/sw/pkgs/arc/mamba/py3.11/bin/python3` has
`fitz`, the emu-jax env does NOT), (b) verified against repo code (file:line), or (c)
reproduced live by me (cache w_c(z) slope + the forward-only alpha-swap). Where a number
is only stated in a doc/handoff I say so.

---

## 1. What this subsystem does (the forest-physics lens)

The end product is a **differentiable forward model `P_obs(θ, z, τ₀, α)`** for the Lyα-forest
1D flux power spectrum (P1D), trained to reproduce the PRIYA SPH suite, marginalized over
high-column-density (HCD) contamination, and wired to fit DESI DR1 + KODIAQ-SQUAD. The
physics stack (README.md §2–3, walkthrough §1):

```
P_filt_c(k; θ, z, τ₀)  ← emulator, per HCD class c ∈ {clean, LLS, subDLA, DLA}   (4,K)
R_c   = P_c − P_clean   (excess over clean forest; filtered LLS/subDLA, UNFILTERED DLA)
P_obs = P_clean + Σ_{c∈HCD} α_c · R_c    ≡    P_clean·[1 + Σ_c α_c(P_c/P_clean − 1)]
```

The forest-physics content lives in five places:
1. **The emulator's fidelity to PRIYA** — the 9-param contract, the τ₀ mean-flux rescaling,
   the multi-fidelity (LF→HF) resolution correction. (PRIYA paper §2.)
2. **The HCD template** — `predict._excess_from_P_filt` / `predict_P_obs` (`predict.py:59,83`):
   the Rogers&Bird-2018 fixed-shape/free-amplitude form, the filtered/unfiltered split, the
   global-⟨F⟩ normalization.
3. **The incidence prior** — `inference.hcd_incidence_prior` (`inference.py:86`): α_c centered
   on the OBSERVED dN/dX (not PRIYA's sim) via a z-slope power-law `lit_over_sim_at_z`
   (`inference.py:77`).
4. **The DLA-masked knob** — the ONE load-bearing HCD assumption: data are DLA-masked, so
   α_DLA residual ≈ 0 (plan §0c). Currently still `HCD_DLA_RESIDUAL_FRAC=0.30` + softplus
   (`inference.py:61`, `sampler_numpyro.py:67`) — the Gaussian-at-0 redesign is NOT yet coded.
5. **Metals + mean-flux anchors** — `data_likelihood._metal_factor` (`data_likelihood.py:226`,
   SiIII/SiII companion Eq 4.2–4.3), `meanflux_prior.becker13_tau0` (`meanflux_prior.py:49`).

The closure (`closure_legb.py`) is the certification harness: it builds a mock from a
HELD-OUT sim's measured P1D, adds cosmic noise, and runs NUTS with this forward model to
check coverage/bias before touching real data.

---

## 2. Load-bearing design decisions & WHY

### 2.1 PRIYA suite — VERIFIED from the paper (`/tmp/priya.txt`)

All of the memory `reference-priya-sims` and plan §🔒 claims check out against the paper:

- **Box: 120 Mpc/h** for both tiers (`priya.txt:18-19,182`).
- **LF = 48 sims @ 1536³** (gas+CDM each), mean interparticle spacing 78–79 kpc/h
  (`priya.txt:18,180-182,805`). **HF = 3 sims @ 2×3072³**, 39 kpc/h (`priya.txt:19,182,673,703`).
- **9 params = 4 cosmology + 3 He-II reion + 1 H-I reion + 1 AGN feedback** (`priya.txt:425`,
  Table 2). Names + ranges (`priya.txt:255-432`):
  - `n_P` (= our `ns`) ∈ [0.8, 0.995], small-scale slope at k=0.78 Mpc⁻¹ (`priya.txt:261`).
  - `A_P` (= our `Ap`) ∈ [1.2e-9, 2.6e-9], amplitude at **k=0.78 Mpc⁻¹** (NOT k₀=0.05 — chosen
    to decorrelate amp/slope as measured by the forest; `priya.txt:261-265`).
  - `Ω_M h²` ∈ [0.14, 0.146]; `h` ∈ [0.65, 0.75] (`priya.txt:267-270`).
  - `z_HeI` (herei) ∈ [3.5, 4.1] start of HeII reion; `z_Hef` (heref) ∈ [2.6, 3.2] end;
    `α_q` (alphaq) ∈ [1.3, 2.5] quasar spectral index (`priya.txt:407-415`).
  - `z_HI` (hireionz) ∈ [6.5, 8.0]; `ε_AGN` (bhfeedback) ∈ [0.03, 0.07] (`priya.txt`, Table 2).
  - **Repo order** `[ns, Ap, herei, heref, alphaq, hub, omegamh2, hireionz, bhfeedback]`
    (`data.PARAM_LIMITS`, README §4 line 132; `inference.PARAM_NAMES` line 25) — matches PRIYA's
    `coarse_grid`. **Note: the repo emulates `omegamh2` though PRIYA notes Ω_M h² is weakly
    constrained by the forest and better measured elsewhere — keep its informative prior.**

- **Multi-fidelity resolution correction = PRIYA Eq 2.18** (`priya.txt:743-753`), VERBATIM:
  `P_HF_F(k,z,θ) = ρ(k,z)·P_LF_F(k,z,θ) + δ(k,z,θ)`, ρ a multiplicative scaling parameter,
  δ a GP independent of the LF output. **"the cosmology dependence comes from δ"** and **"the
  cosmology dependence of the resolution correction is fairly small"** → the linear MF model
  is justified (`priya.txt:751-753`). Separate GP per redshift bin. This is EXACTLY the memory
  claim "ρ cosmology-independent, δ small cosmology-dependent." The repo `multifidelity.py`
  default `delta_mode='none'` DROPS δ(θ) → θ-independent correction; the dropped δ residual is
  to be carried in C_emu (plan §0 item 2). **Faithful, with the caveat that dropping δ(θ) is a
  deliberate simplification beyond PRIYA — PRIYA keeps δ(θ); we argue it overfits 6 sims.**

- **Mean-flux rescaling — VERIFIED** (`priya.txt:608-629`): **10 optical-depth samples per z**
  ("dramatically over-sample it, generating a dense grid of 10 optical depth samples per
  redshift"), specialized to a power law `τ_Kim(z) = 0.0023·(1+z)^3.65` (`priya.txt:619`). The
  repo uses this as `KIM_AMP·(1+z)^KIM_SLOPE` (cache τ₀ ladder anchor, `data.KIM_AMP/KIM_SLOPE`,
  walkthrough §2). **NB: the repo LF cache uses a 20-point τ₀ ladder (README §6), MORE than
  PRIYA's 10 — a finer rescale grid; this is a repo choice, not a fidelity break.**

- **Convergence — VERIFIED** (`priya.txt:1110-1119,166-170,29`): "flux power converged at the
  percent level for z = 5.4–2.2"; DESI scales (k≲0.05) at the percent level; high-res scales
  (k≲0.1) at the ten-percent level. The 15 Mpc/h n384/n512 diagnostic: **2% level for z>2.6,
  sub-percent z≤2.6**, "approximate limit of 2% convergence is 0.07 s/km, converged to about
  **7% at 0.1 s/km**" (`priya.txt:1117-1119`). **WORST during He-II reionization z=4→3.1**
  (`priya.txt:1108-1110`) — NOT a monotonic high-z blow-up. This is the basis for the locked
  claim "k_max=0.1 trustworthy, modest high-k floor" and for rejecting Khan+2023's ~50%@0.1
  (a LOW-res 256³–1024³ result, per memory `reference-khan2023`).

### 2.2 The 6-HR-vs-3-HF reconciliation — RESOLVED (the MF-LOSO DOF question)

The PRIYA *paper* used **3 HF sims** for its published MF GP (`priya.txt:19,673,703,1100`:
"only 3 high fidelity simulations"). The *repo HR cache* (`observables_tau0_hr.h5`) has **6
unique sims / 2060 rows / n_k=525** — I verified this directly: the 6 HR sims are
`ns0.859Ap1.29e-09…`, `ns0.885Ap2.32e-09…`, `ns0.909…`, `ns0.914…`, `ns0.972…`, `ns0.979…`.
Cross-checked: `scripts/diag_tilt_bias_lf_hr.py:74` hardcodes `("…hr.h5", 6, 525)`;
`multifidelity.py:16,243,268` and `scripts/build_mf_delta.py:306` all say "6 HF sims."

**Interpretation (forest-physics lens):** these 6 HR cosmologies are described in the code as
"EXACT LF design points" (`multifidelity.py:18`, `scripts/diag_mf_complexity.py:14`) — i.e.
the repo re-ran 6 of the 60 LF design points at HF (3072³) resolution to MEASURE the LF→HF
correction at MATCHED (θ,z,τ₀). This is a SUPERSET of PRIYA's published 3 HF nodes (PRIYA's 3
were chosen by Bayesian-optimization L2-norm minimization, `priya.txt:692-703`). So the repo's
MF correction has **6** matched LF↔HF pairs, not 3, for its own LOSO. **This is a strength, not
a discrepancy** — but it is NOT what the PRIYA paper validated; the repo MF rests on the repo's
own 6-sim measurement (plan §0.4 Phase-0 pre-flight flags this as MUST-verify). **Risk: 6 sims
is still few-DOF — the learned δ-head overfit n_s to 4σ (`multifidelity.py:315,416`), which is
why production drops δ(θ). Any MF-LOSO C_emu scale s(k) must stay low-DOF/smooth (plan §2.1).**
The LF cache I confirmed has **60 unique sims** (not 48) — the repo added a Bayesian-opt 8-sim
LHC + 3 more (`priya.txt:684-692`); 60 = 48 published + 12 extra. Faithful superset.

### 2.3 The HCD template — faithful to Rogers&Bird 2018, strictly more general

Code: `predict._excess_from_P_filt` (`predict.py:59-73`) + `predict_P_obs` (`predict.py:83-103`).
Walkthrough §6 documents the 2026-06-04 redesign (the user flagged a real bug).

- **The fixed bug:** the OLD template was `Δ_c = P_c^unfiltered − P_c^filtered` (the power the
  sim's internal τ=1e6 filter removes). That is wrong physics — the data has DLA *masking*, not
  a τ-filter — and worse, the filter never touches LLS, so **Δ_LLS ≡ 0** and α_LLS was inert in
  the likelihood (walkthrough §6.1, table; LLS old=0.0000 vs correct=0.066 at z=3). FIXED to
  `R_c = P_c − P_clean` (excess over the clean forest), the Rogers/DESI/PRIYA contamination
  object. `∂P_obs/∂α_LLS = (P_LLS − P_clean) ≠ 0` now (`predict.py:98`).

- **The filtered/unfiltered split (physically motivated):** LLS/subDLA use the FILTERED template
  (filt≈unfilt there); DLA uses the UNFILTERED template `P_DLA^unf = P_filt[DLA] + dla_core`
  (`predict.py:70`), because the data's residual *unmasked* DLAs are full systems. `dla_core` =
  `cache["delta"][row,2]` = the DLA-core add-back (`P_DLA^unf − P_DLA^filt`). README §3 line 110.
  **This is internally consistent: the truth construction `make_truth_from_sim` uses the SAME
  unfiltered-DLA add-back** (`closure_legb.py:276-277`: `P_filt[r,3] + core_r`), so forward and
  truth cores cancel in the emulator-error sizing.

- **additive ≡ multiplicative is an ALGEBRAIC IDENTITY** (`predict.py:88-89`, walkthrough §6.3):
  `P_clean + Σα_c(P_c−P_clean) ≡ P_clean·[1+Σα_c(P_c/P_clean−1)]`. This CANNOT bias — the
  Lyα+CS agents proved the "Δ-vs-reweight" and "additive-vs-multiplicative" forks are non-forks.
  The only real design choice was the template *definition* (fixed) + the filtered/unfiltered
  split (fixed). The dense learned Δ head is DROPPED (over-fit risk, 516/1575 unused DOF) —
  the excess comes free from the emulated P_filt (walkthrough §6.3; README §2 line 86).

- **Global-⟨F⟩ normalization (offsets are REAL):** per-class power uses ONE shared global ⟨F⟩
  (`target_F`), not per-subset (README §5 line 171, memory `hcd-template-rogers-normalization`).
  So the class offsets are real physics matching Rogers/Croft, not a sightline-count artifact.
  k is ANGULAR (`k=2π/λ_v`, s/km) community-wide — feed Rogers templates directly, no /2π
  (README §5 line 162).

- **Rogers Eq-6 +c(z) plateau the repo omits:** memory `hcd-template-rogers-normalization` notes
  Rogers Eq-6 has a `+c(z)` plateau term the repo does not implement. The repo uses the EXACT
  emulated excess instead of a smooth kernel — justified for DLA where the Rogers low-order
  kernel cross-check is WORSE (RMSE ~0.26–0.33 / ~30%, walkthrough §6.4) than for LLS/subDLA
  (0.03–0.13). **Forest-physics verdict: the missing +c(z) is a low-k DC offset; using the
  measured excess (which carries any plateau) is more faithful than a kernel that omits it. Not
  a bias risk, but document it — see §4.**

### 2.4 The dN/dX → α incidence prior (sim ≠ data)

`inference.hcd_incidence_prior` (`inference.py:86-106`) + `lit_over_sim_at_z` (`inference.py:77`).
The KEY physics: **PRIYA does NOT match the observed dN/dX.** Walkthrough §4 / §6.5:
PRIYA reproduces LLS (~0.98) but OVER-predicts subDLA (×1.31) and UNDER-predicts DLA (×0.70) at
z=3 — and its z-slope differs. So the prior centers on `(lit/sim)(z)·w_c`, NOT on α=w_c:
- `HCD_LIT_OVER_SIM = (1.06, 0.76, 1.34)` (LLS, subDLA, DLA) data/sim at z_pivot=3 (`inference.py:70`).
- `HCD_LIT_OVER_SIM_SLOPE = (0.95, 0.15, 0.40)` = `d ln(lit/sim)/d ln(1+z)` (`inference.py:74`)
  — a power-law in (1+z), the τ₀-analog (a curve+slope, not a single number). DLA slope
  deliberately weakened to 0.4 (raw fit +1.08 is dominated by z>3.5 DLA dN/dX the literature
  does not measure reliably; `inference.py:71-73`).
- The structural `w_c` come from PRIYA's per-class sightline fractions (cache `w_c_cache`), built
  from dN/dX via the telescoping-Poisson M₀ map `dndx_wc.w_c_from_mu` (`dndx_wc.py:12`).

This is the right posture: the sim is a *template generator*, the observed incidence sets the
*amplitude prior*.

### 2.5 The closure-bias physics — the closing step (THE central question)

**The bug (handoff §6d, plan §0b-1):** the closure forward applied ONE z-constant α while the
mock truth carried the sim's per-z `w_c(z)`. The sim's HCD incidence RISES STEEPLY with z, so a
z-constant α over-counts HCD at low z → coherent low-k over-prediction the diagonal-in-k C_emu
can't whiten → NUTS lowers HCD-α + cosmology amplitude (A_p ≈10σ low, α_subDLA ≈19σ low in the
buggy pilot).

**I REPRODUCED the magnitude live** (`scripts/diag_legb_zresolved_alpha_check.py`, forward-only,
no NUTS), on the DESI leg, sim `ns0.803Ap2.2e-09…`, 44 low-k<3e-3 bins:
```
  OLD z-median α (3,)   : low-k whitened Δ mean −0.780σ  coherent −5.176σ  (frac −2.01%)
  LIT-shape α(z)        : low-k whitened Δ mean −0.239σ  coherent −1.589σ  (frac −0.64%)
  EXACT per-z w_c(z)    : low-k whitened Δ mean +0.034σ  coherent +0.227σ  (frac +0.18%)
```
These match the handoff numbers (−0.78 → −0.24 → +0.03σ) EXACTLY. KS has 0 low-k<3e-3 bins
(NaN) → this is a **DESI-low-k-specific** problem, consistent with "KS clean."

**I ALSO ground-truthed the root-cause physics** via the cache loader (`load_cache`):
per-z median `w_c(z)` over z∈[2.2,4.6], absolute slope `d ln w_c/d ln(1+z)`:
```
  LLS    : w_c(2.2)=0.0933 → w_c(4.6)=0.4105   4.40× rise   slope +2.60
  subDLA : w_c(2.2)=0.0319 → w_c(4.6)=0.1485   4.65× rise   slope +2.72
  DLA    : w_c(2.2)=0.0175 → w_c(4.6)=0.0691   3.94× rise   slope +2.42
  SUM HCD: 0.1426 → 0.6273                      4.40× rise   slope +2.61
```
So the sim's per-z HCD incidence rises **~4–4.7× over the data z-range with an absolute slope
~2.4–2.7** in (1+z). The literature *ratio* slope `HCD_LIT_OVER_SIM_SLOPE` is only **0.15–0.95**
— that is the slope of `(lit/sim)`, a SMALL DIFFERENTIAL on top of the sim's own steep w_c(z).
The forward's `_legb_model` (`closure_legb.py:453`) builds `α(z) = α_pivot·((1+z)/(1+z_p))^s_c`
with `s_c = HCD_LIT_OVER_SIM_SLOPE` — i.e. it uses the SMALL ratio-slope (0.15–0.95) as the FULL
z-shape, which is why it only removes 70% of the bias (the −0.24σ). The EXACT case threads the
full sim w_c(z) shape (slope ~2.6) → +0.03σ.

**The documented closing step (handoff lines 32–37, plan §0b-1):** option (B) thread the per-z
shape `g_c(z) = w_c(z)/w_c(z_p)` from the truth into `_legb_model`, so `α(z)=α_pivot·g(z)` = the
EXACT verified case. Option (A) sample a per-class z-slope with a wide prior.

---

## 3. Verified vs assumed

### VERIFIED (test/figure/paper/reproduction behind it)
- **PRIYA box/sims/params/Eq-2.18/mean-flux/convergence** — all read VERBATIM from the paper
  text (§2.1 above; `/tmp/priya.txt` line refs given). High confidence.
- **6 HR cache sims** — I listed them from the HDF5 directly (§2.2). LF cache = 60 sims.
- **The closure-bias magnitude (−0.78/−0.24/+0.03σ)** — I reproduced it live (§2.5).
- **The sim w_c(z) slope ~2.6 vs lit ratio-slope 0.15–0.95** — I computed it live (§2.5). This is
  the load-bearing physics of the closing step and it holds.
- **The HCD template identities** — additive≡multiplicative is algebraic (`predict.py:88`);
  ∂P_obs/∂α_LLS ≠ 0 (the Δ_LLS≡0 fix); forward/truth DLA-core cancel (`closure_legb.py:276`).
- **Metal term = companion Eq 4.2–4.3** — `data_likelihood._metal_factor` (`data_likelihood.py:242`):
  `C = a²+2a·cos(kΔv)·D(k)`, D the sigmoid decorrelation on the cosine ONLY (commit `51d22eb`,
  "Lyα-confirmed"). Tests `test_metal_factor_*` (`tests/test_data_likelihood.py:183-199`) pin
  identity-at-0 + oscillation + differentiability. Metals OFF for closure (correct: sim truth has
  no metals; `closure_legb.py:124` comment).
- **z-resolved α path is implemented + backward-compat** — `predict_P_obs_on_leg` takes (3,) OR
  (n_z,3) α (`data_likelihood.py:328,346`); `_legb_model` samples α_pivot×lit-slope shape
  (`closure_legb.py:450-454`); 28 tests pass (handoff §3).
- **DLA masking in PRIYA ~2% of spectra, <1% P1D change** (`priya.txt:566-572`) — the sim is
  built DLA-masked already.

### ASSUMED / STATED (no proof yet, or a deliberate MVP)
- **`delta_mode='none'` (drop PRIYA's δ(θ))** — assumed the dropped cosmology-dependent residual
  is small enough to carry in C_emu. PRIYA KEEPS δ(θ). Stated, not yet certified on this suite.
- **α_DLA Gaussian-at-0 / mock-truth DLA-masked (plan §0c)** — DECIDED but NOT YET CODED. The
  live code still has `HCD_DLA_RESIDUAL_FRAC=0.30` (`inference.py:61`) + softplus
  (`sampler_numpyro.py:67`, `closure_legb.py:447-449`), and `make_truth_from_sim` still builds the
  truth with the sim's FULL w_c including DLA (`closure_legb.py:273-278`, NOT DLA-masked). This is
  the single biggest "documented-but-unimplemented" gap from this lens.
- **Becker+2013 τ₀ anchor + 5% width** — MVP; `meanflux_prior.py:76` LYA-CONSULT explicitly says
  the production ⟨F⟩(z) + z-dependent error budget must be picked by a Lyα expert and NOT shipped
  on the Kim MVP without sign-off.
- **DESI–KS block-diagonal (no cross-covariance)** — assumed independent systematics
  (`data_likelihood.py:407` LYA-CONSULT). Reasonable (different instruments) but unconfirmed.
- **KS conservative mode carries no model systematics** — assumed (`data_likelihood.py:162`
  LYA-CONSULT). The off-by-one drop (5 not 4 bins) + CACHE_KMAX=0.069 cap (drops KS's unique
  0.079/0.099) are KNOWN bugs (`data_likelihood.py:60-62,156-158`; handoff §6b/§6c, plan §4.2).

---

## 4. Risks / open questions (ranked; flagged if they could bias the real fit)

**R1 [HIGH — could bias real fit] The closing step risks CIRCULARITY if applied as written.**
The documented option (B) — thread the *exact held-out sim's* per-z `w_c(z)` shape into the
closure forward — makes the closure pass by construction: you are telling the forward model the
truth's incidence z-shape. For the CLOSURE this is defensible *only if* you also use a wide,
honestly-marginalized z-slope in the SAME forward when you fit real data (because for real data
you do NOT know the true incidence shape — you only have the observed dN/dX with its z-slope
uncertainty). My read of the forest physics:
  - The sim's w_c(z) is a known POPULATION SELECTION FUNCTION (like τ₀): it is set by the sim's
    self-shielding + dN/dX, which is a measured quantity of the sim. Threading it is NOT cheating
    in the sense of leaking cosmology — w_c(z) is θ-weakly-dependent (it's an abundance, Head A is
    τ₀-invariant). So as a *closure construction* it is like fixing τ₀ to the truth: legitimate to
    isolate the cosmology coverage.
  - BUT the real fit will NOT have the exact w_c(z); it will use `lit_over_sim_at_z` × an observed
    w_c with a z-slope PRIOR. If that prior's slope range does not BRACKET the true incidence
    z-shape (sim slope ~2.6 absolute; the differential lit/sim slope is 0.15–0.95), the same −0.24σ
    coherent low-k pull will reappear on real data — and low-k DESI is exactly the A_p/n_s regime.
  - **Recommendation (defensible path):** prefer option (A) — SAMPLE a per-class z-slope with a
    prior wide enough to contain the sim's measured absolute slope (~2.6) — and certify that the
    closure passes when the slope is MARGINALIZED, not fixed to truth. If you ship option (B)
    (exact w_c(z) fixed), the closure +0.03σ is a NECESSARY-but-not-SUFFICIENT check; you MUST
    additionally run a closure where the forward's z-slope is sampled (the real-fit configuration)
    and show coverage there. Otherwise the real-data A_p/n_s posterior inherits an un-tested
    incidence-shape systematic. This is the #1 thing the meta-reviewer should escalate.

**R2 [HIGH — load-bearing, not yet coded] The DLA-masked knob is decided but unimplemented.**
Plan §0c LOCKED: α_DLA residual ≈ 0, Gaussian-at-0, mock truth DLA-masked (KS ~0%, DESI ≥90%
masked). The CODE still uses the 0.30 residual fraction + one-sided softplus + a FULL-w_c truth.
Physically the §0c decision is RIGHT: PRIYA masks DLAs to ⟨δ_F⟩=0 (`priya.txt:566-572`), DESI's
DLA-finder is >85% complete and missed DLAs are misclassified as subDLAs (a small subDLA residual,
not a DLA one). So the closure mock truth SHOULD be DLA-masked to match. **Until this is coded,
the closure mock carries a DLA contamination the data does not have, and α_DLA has a one-sided
prior pulling it positive — both inconsistent with the data's masked state. This could bias the
DLA sector (minor for cosmology, since DLA P1D leverage is mostly low-k where masking removes it,
but it muddies the closure verdict). VERIFY this is fixed before the cert run.**

**R3 [MEDIUM] `delta_mode='none'` drops PRIYA's cosmology-dependent δ(θ).** PRIYA Eq 2.18 keeps
δ(θ) and says its cosmology dependence is "fairly small" (`priya.txt:751`) — but "small" is not
"zero," and the repo dropped it because the learned δ-head overfit n_s to 4σ on 6 sims
(`multifidelity.py:315,416`). The plan carries the dropped δ in C_emu via the MF-LOSO scale s(k).
**Risk: if the dropped δ(θ) has a COHERENT (not random) k-shape correlated with n_s/A_p, scaling a
diagonal variance does not marginalize it — it needs a correlated/low-rank C_emu term (plan §0b-2
acknowledges this for the LF residual). For the production MF run, confirm the dropped-δ residual
is verified incoherent-in-cosmology, not just small in RMS.** Note: MF is KS-leg-only for the
high-k EXTENSION but BOTH-legs for the correction (plan §0 Q1); DESI maxes at k≤0.041 < LF Nyquist
0.069, so the MF correction on DESI is the resolution back-coupling, not an extrapolation.

**R4 [MEDIUM] KS leg is near-redundant + has loader bugs (provisional).** Handoff §6b/§6c, plan §4:
`load_ks_leg` drops 5 bins not 4 (off-by-one `KS_DROP_KMAX=0.0158`, `data_likelihood.py:62`), the
"2306.06316 Fig 11" citation is mis-attributed (that's DESI, KS is 2108.10870), and `CACHE_KMAX=0.069`
caps off KS's UNIQUE small-scale bins 0.079/0.099 — exactly what DESI can't reach. With the
production emulator LF-only (Nyquist 0.069, no HR emulator trained — `multifidelity.py` is a stub),
KS as-configured ≈ redundant with DESI. **For the forest physics this means: KS currently
contributes ~nothing unique; its thermal/small-scale value is locked behind training an HR/MF
emulator (real work). Not a bias risk for the DESI-primary fit, but do not claim KS small-scale
constraining power until the loader is fixed AND the MF emulator reaches 0.08–0.1.**

**R5 [LOW-MEDIUM] The omitted Rogers Eq-6 +c(z) plateau.** The repo's measured-excess template
carries any plateau implicitly, so this is more faithful than a kernel — but the DLA Rogers-kernel
cross-check is ~30% RMSE (walkthrough §6.4), the noisiest class. Since DLA is being masked to ≈0
(R2), this de-risks. Document that the template definition is the measured excess, not Rogers Eq-6,
so a future reader does not "add back" a plateau and double-count.

**R6 [LOW] Becker13 MVP mean-flux anchor.** `meanflux_prior.py:76` LYA-CONSULT flags this. The
τ_eff–cosmology degeneracy is the classic P1D systematic; anchoring on the wrong ⟨F⟩(z) curve or
too-tight a width biases A_p. The 5% width is "conservative MVP" — Becker+2013 reports 3–8%
z-dependent. **Before the real fit, set the production ⟨F⟩(z) + per-z error budget in the data's
DLA-masked/metal-corrected state. Not a closure-stage risk (closure uses the sim's own τ₀).**

**R7 [LOW] omegamh2 + weak directions.** PRIYA notes Ω_M h² is weakly forest-constrained
(`priya.txt:267`); walkthrough §8 notes bhfeedback/hireionz show "mild SVD-basis ringing" in the
gradient, neutralized by informative priors. Keep PRIYA's informative priors on these — do not
let them float wide, or the ringing could leak into A_p/n_s.

---

## 5. Pointers for the main agent (the files/functions you MUST know)

1. **`hcd_analysis/emulator/predict.py:59-103`** — `_excess_from_P_filt` + `predict_P_obs`: THE HCD
   forward model. `R_c = P_c − P_clean` (filtered LLS/subDLA, unfiltered DLA via `dla_core`);
   `P_obs = P_clean + Σα_c·R_c`. additive≡multiplicative (line 88), Δ_LLS≡0 bug fixed.

2. **`hcd_analysis/emulator/inference.py:61-106`** — the incidence prior. `HCD_LIT_OVER_SIM` (1.06,
   0.76, 1.34) + `HCD_LIT_OVER_SIM_SLOPE` (0.95, 0.15, 0.40) are the data/sim dN/dX ratio + z-slope.
   `HCD_DLA_RESIDUAL_FRAC=0.30` (line 61) is the OUTDATED 0.30 fraction the plan §0c supersedes —
   CHECK whether it's been replaced by Gaussian-at-0 before any cert run.

3. **`hcd_analysis/emulator/closure_legb.py:214-283` (`make_truth_from_sim`) + `:432-456`
   (`_legb_model`)** — the closure forward. `make_truth_from_sim` builds the mock truth from a
   held-out sim's measured P1D at α=w_c (line 273-278) and currently returns a z-MEDIAN w_c (line
   283) — the closing step needs the per-z shape. `_legb_model:453` builds α(z) with the
   lit-slope shape (the −0.24σ partial fix); the +0.03σ EXACT case needs the truth's w_c(z) g(z)
   threaded in. THIS IS WHERE THE CLOSING STEP LANDS.

4. **`scripts/diag_legb_zresolved_alpha_check.py`** — the forward-only verifier (no NUTS, ~minutes).
   Reproduces −0.78/−0.24/+0.03σ on DESI. Run this to re-confirm any α(z) change before a NUTS cert.
   Uses the de-circularized `error_vector_xclass_holdout0.npz`.

5. **`hcd_analysis/emulator/data_likelihood.py:226-251` (metals/resolution) + `:294-378`
   (`predict_P_obs_on_leg`) + `:384-449` (`data_loglik`)** — the real-data binding. Metal term =
   companion Eq 4.2–4.3 (line 242); per-z α supported (line 328,346); block-diagonal multi-leg.
   KS loader bugs at lines 60-62,149-179 (off-by-one + mis-cite + kmax cap).

6. **`hcd_analysis/emulator/meanflux_prior.py:49-95`** — `becker13_tau0` (Eq 6: 0.751·((1+z)/4.5)^2.90
   − 0.132) is the production anchor; `kim_tau0` is the cache-ladder MVP. LYA-CONSULT at line 76:
   pick the production ⟨F⟩(z) before the real fit.

7. **`hcd_analysis/emulator/multifidelity.py:1-90`** — the MF layer mirroring PRIYA Eq 2.18.
   `delta_mode='none'` (θ-independent, drops δ(θ)) is the default; the 6-HR-sim few-DOF caution is
   documented at lines 268,315,416. It is a STUB for the likelihood (not yet wired; handoff §6c).

8. **`/tmp/priya.txt`** — full PRIYA paper text (re-extract anytime with
   `/sw/pkgs/arc/mamba/py3.11/bin/python3 -c "import fitz; ..."`; the emu-jax env lacks fitz).
   Eq 2.18 at lines 743-753; mean-flux at 608-629; convergence at 1110-1119; params at 255-432.

9. **`docs/superpowers/plans/2026-06-05-mf-likelihood-wiring-plan.md` §0/§0b/§0c** — the LOCKED MF +
   closure-bias + DLA-residual decisions. §0c (DLA Gaussian-at-0, mock truth DLA-masked) is the
   load-bearing physics decision that is decided-but-not-yet-coded.

10. **`docs/superpowers/2026-06-04-phase-c-walkthrough.md` §4-6** — the HCD redesign rationale +
    the PRIYA-vs-literature dN/dX (§4: LLS 0.98 / subDLA ×1.31 / DLA ×0.70) + the Rogers cross-check
    (§6.4). The physics narrative for everything in #1-#2 above.
