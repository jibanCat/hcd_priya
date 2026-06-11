# Phase-4d plan — HCD-template + metal-line marginalization robustness tests

> Three mock-injection closure tests the PI requested (2026-06-11): stress the forward against
> contamination it does NOT natively model (the external Rogers 2018 HCD template; metal/SiIII),
> and check the baseline cosmology stays unbiased. All designs confirmed in dialogue.

**Goal:** certify that the real-fit baseline (PRIYA emulator + per-survey LLS pin + `_metal_factor`)
recovers cosmology when the *data* carry contamination modeled differently than our forward.

**Shared mechanism:** inject the contamination into the closure MOCK (`make_legb_mock` /
`make_truth_from_sim`), fit with the (existing) baseline forward, read the A_p/n_s bias_z. Reuse the
`run_stepA` + battery + `analyze_*` infrastructure; new per-chain knobs gate each injection.

---

## Test 1 — Rogers template vs PRIYA in-situ HCD (consistency)

- **MOCK:** PRIYA tier_p, FULL (unfiltered) LLS+subDLA (cache `P_tier_c`, not filtered) — the data analog.
- **FORWARD:** PRIYA **clean** baseline (`P_filt[0]`) + the Rogers `template_factor` (Eq.8, floating α_i)
  as the SOLE HCD model. Marginalize α_i (LLS, Sub-DLA, Small-DLA, Large-DLA).
- **Question:** can Rogers absorb PRIYA's in-situ HCD → recover cosmology? (Rogers ≈ PRIYA in-situ?)
- **Build:** a forward variant `hcd_model="rogers"` in the closure that swaps PRIYA's per-class delta
  for `hcd_template.template_factor(k, z)` with sampled α_i; new α_i nuisances in `_legb_model`.

## Test 2 — can PRIYA safely USE Rogers (validate Fernandez2024 / Ho2025)

- **MOCK:** PRIYA tier_p, full (same as test 1).
- **FORWARD:** PRIYA **tier_p** baseline (in-situ HCD) + Rogers `template_factor` marginalized with
  **α_Rogers prior CENTERED ON 0** (the Fernandez24/Ho25 recipe: the emulator carries the HCD, Rogers
  is the extra marginalized systematic that should fit to ~0).
- **Question:** does adding the Rogers nuisance (α≈0) keep cosmology unbiased + how much σ inflation?
- **Build:** same Rogers forward, but additive-on-tier_p with a N(0, σ_Rogers) prior on α_i.

## Test 3 — metal-line (SiIII) marginalization  [START HERE — `_metal_factor` exists]

- **MOCK (two arms, PI 2026-06-11):**
  - **DESI arm:** inject the FULL DESI model (arXiv:2601.21432): Lyα–SiIII (mult, our form) + Lyα–SiII
    DOUBLET (1190.42/1193.28 Å, ratio r) + **ADDITIVE same-ion SiII–SiII** `+f[1+r²+2r·cos(k·Δv)]·exp(−k²/k²)`
    + cross-ion SiII–SiIII. `f_SiIII≈0.009` (PRIYA-eBOSS), `A=f/(1−⟨F⟩)≈0.03`.
  - **eBOSS arm:** inject the McDonald/eBOSS form (`lya_emulator_full SiIIIcorr`): `1+aa²+2aa·cos(2271k)`,
    `aa=f_SiIII/(1−⟨F⟩)`, NO decorrelation, SiIII only.
- **FORWARD:** baseline + our reduced multiplicative `_metal_factor` (sample a_SiIII, a_SiII), `metals_on=True`.
- **Question:** is cosmology unbiased when the data's metals follow a model our multiplicative factor
  CANNOT fully reproduce (esp. the additive SiII–SiII term)?
- **KS:** `metals_on=False` stays (KS side-band-SUBTRACTS metals — modeling would double-count).
- **Build:** a `_metal_inject(k, z, ⟨F⟩, form=...)` mock helper (full-DESI + McDonald forms) + a
  `metal_inject` chain knob in `make_legb_mock`; turn `metals_on=True` in the fit for these fiducials;
  new fiducials `D_metalDESI`, `D_metaleBOSS` (DESI leg, fold6).

---

## Order + cost

1. **Test 3** first (forward `_metal_factor` already exists; only the mock injection is new). ~2 fiducials
   × 4 chains ≈ 8 DESI chains (~10 CPU-h). Pull exact DESI f_SiIII/k_x from arXiv:2601.21432 §6.3 /
   igmhub/cobaya_lya_p1d if precision needed; `f_SiIII≈0.009 ±50%` central otherwise.
2. **Tests 1 + 2** next (share the Rogers forward variant). ~2–3 fiducials.

## Open items / caveats

- "Full LLS+subDLA" mock = the unfiltered `P_tier_c` (the deferred subDLA add-back, now needed here).
- Our `_metal_factor` lacks the additive SiII–SiII + the SiII doublet; test 3 measures whether that gap
  biases cosmology, and thus whether to ADD an additive term to `_metal_factor` before the real fit.
- Rogers `template_factor` classes are (LLS, Sub-DLA, Small-DLA, Large-DLA) — map to our (LLS, subDLA,
  DLA) for the prior; the two DLA sub-classes are the masked regime.
