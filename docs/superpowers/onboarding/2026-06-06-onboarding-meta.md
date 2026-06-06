# Meta onboarding synthesis — 2026-06-06

Synthesis of the four onboarding lenses (Bayesian/PPL, CS/JAX, cosmology/P1D, Lyα/IGM)
for the `phase2c-likelihood` branch. Source reports:
`2026-06-06-onboarding-{bayesian,cs,cosmology,lya}.md`. I cite each as [BAY], [CS],
[COS], [LYA] with their file:line where it matters. This is a critical synthesis, not a
recap: §2 surfaces the disagreements, §3 separates proof from assumption, §4 ranks the
real-fit risks, §5 adjudicates the closing-step decision.

---

## 0. One-screen orientation for the main agent

**What this is.** A JAX/Equinox differentiable emulator of the Lyα-forest 1D flux power
spectrum P1D(k), trained on the PRIYA SPH suite (120 Mpc/h; LF 48×1536³, HF 3×3072³),
plus a Phase-C numpyro/NUTS likelihood that fits it to DESI DR1 + KODIAQ-SQUAD to recover
cosmology (primarily **A_p, n_s**). Phase-2 (emulator) is MERGED to `main` (PR #10);
Phase-C (this branch) is the inference/closure stack.

**Forward model.** `P_obs(k) = P_clean + Σ_{c∈HCD} α_c·(P_c − P_clean)` over 4 classes
(clean/LLS/subDLA/DLA) [predict.py:83-103]; per-class P_filt reconstructed as a θ-blind
baseline + a σ_cosmo-whitened residual so ALL cosmology flows through one head
[predict.py:29-39, model.py:241-247]. Then interp cache-k → DESI/KS leg-k, × metals ×
resolution, into a Gaussian logL with a carried logdet [likelihood.py:154-175].

**Where we are.** 28 tests pass (re-run this session by [CS] and confirmed). The structural
machinery is sound and well-tested. The project is at the **closing step before the
certified Leg-B coverage run**: a documented chain of fixes (golden guard → z-resolved
α(z) → DLA-mask → α_DLA Gaussian-at-0 → correlated-in-k C_emu) is PLANNED but the live
code implements NONE of them yet. A forward-only diagnostic shows the central bias
(coherent low-k over-prediction) drops from −0.78σ → −0.24σ (current lit-slope code) →
+0.03σ (if the truth's exact per-z shape is threaded). No NUTS coverage verdict exists yet.

**The next step (all four lenses converge here).** Do NOT jump to a NUTS cert. The
sequenced next actions are: (1) write the golden-regression guard [CS-R1]; (2) decide the
α(z)-shape treatment — and the lenses agree the closure must certify with the
**production** α(z) parametrization (sampled amplitude **and** slope), NOT by hard-wiring
the held-out sim's truth shape (see §5); (3) build the low-rank correlated-in-k C_emu that
the plan itself requires (currently C_emu is diagonal-in-k everywhere and cannot whiten a
coherent low-k mode); (4) code the §0c DLA-masked decision; then (5) run Leg-A SBC at
production scale and Leg-B coverage at N≥99. Compute budget is tight (§4-R6).

---

## 1. Cross-lens AGREEMENTS (load-bearing, confirmed by multiple lenses)

These are the facts you can act on with confidence — each is confirmed independently by
≥2 lenses, and several by direct re-execution this session.

1. **The numpyro model is structurally clean and the contract is tested.**
   `potential == loglik + Σ logprior` with no double-count [BAY §2.1, test_closure_sbc.py:93];
   priors are `sample` sites, likelihood is one `factor`; θ~Uniform^9 with auto-bijector is
   exactly flat in-box (no soft-wall double-penalty) [BAY §2.2, COS §2.5]. 28 tests pass —
   **[CS] re-ran them this session, exit 0** (137 s), confirming the handoff claim.

2. **x64 is enforced on package import and re-asserted everywhere** [CS §x64,
   __init__.py:11]. The structural identities are bit-level and die under float32. The one
   fragility (any new entry-point importing `jax` before `hcd_analysis.emulator`) is noted.

3. **The NUTS-killer JAX traps are correctly guarded.** The interp-slope NaN trap
   (sanitise YDATA *before* `jnp.interp`, never the output after) is the single most
   dangerous likelihood trap and is correctly handled in `sigma_at_tau0`/`rho_at_tau0`
   [CS §NaN-traps, likelihood.py:98-151; COS §pointer-5; jax-traps #29]. `gaussian_loglik`
   jitters before Cholesky and carries the logdet (Cholesky returns NaN, not raises)
   [CS, BAY, COS all cite likelihood.py:154-175].

4. **The cross-class C_emu einsum is SPD-guaranteed non-negative** and recovers the
   diagonal form exactly at band centre (rtol 1e-10) [CS §einsum, BAY §2.6,
   data_likelihood.py:285]. Whitening improves 1.28 (diagonal) → 0.93 (cross-class).

5. **The θ-blind baseline split is the load-bearing identifiability move and is verified.**
   ∂m̂/∂θ ≡ 0 forces all θ-response through r̂; jacfwd(baseline)=0, ∂P/∂θ FD-fidelity 7e-8,
   round-trip 8e-16 [COS §2.2, CS §θ-blind, model.py:125-135/241-247; jax-traps #14]. ~99.5%
   of per-k log-variance is z/τ₀, ~0.4% is cosmology — the redesign un-buries that 0.4%.

6. **The HCD template is faithful to Rogers&Bird 2018 and the old Δ_LLS≡0 bug is fixed.**
   `R_c = P_c − P_clean`; additive≡multiplicative is an algebraic identity (cannot bias);
   filtered LLS/subDLA + unfiltered DLA; forward/truth DLA-cores cancel [LYA §2.3, COS §pointer-2,
   predict.py:59-103]. The incidence prior correctly centers on observed dN/dX
   (lit/sim = 1.06/0.76/1.34), not on α=w_c [LYA §2.4, BAY §2.5, COS §pointer-3].

7. **The Leg-A vs Leg-B distinction is correctly enforced.** Leg-A null IS rank-uniform
   (ECDF gate valid, L_FLOOR=99); Leg-B null is NOT (truth = held-out sim) → coverage≥nominal
   + bias≈0 are the ONLY valid headlines, ECDF hard-flagged diagnostic-only
   [BAY §2.8, CS §1, COS §pointer-6, closure_legb.py:655-676]. De-circularized ρ
   (`error_vector_xclass_holdout0.npz`, folds 1-7) exists; in-sample circularity shown mild
   (+0.917 vs +0.926) by [BAY], [CS], [COS].

8. **The closing-step forward-only ladder is real and reproduced.** OLD z-median −0.78σ →
   lit-slope α(z) −0.24σ → exact per-z w_c(z) +0.03σ. **[LYA] reproduced this LIVE** on the
   DESI leg (44 low-k<3e-3 bins) matching the handoff exactly; KS has 0 such bins → this is
   **DESI-low-k-specific** [LYA §2.5]. [BAY], [CS], [COS] all confirm it is forward-only
   (no NUTS).

9. **The PRIYA paper claims check out** — [LYA] verified box/sims/Eq-2.18/mean-flux
   (10 τ-samples/z, Kim 0.0023·(1+z)^3.65)/convergence (percent k<0.05, ~7%@0.1, worst at
   HeII reion z=4→3.1) VERBATIM against the extracted paper text [LYA §2.1]. The 6-HR-vs-3-HF
   question is RESOLVED: PRIYA published 3 HF; the repo re-ran 6 LF design points at 3072³ for
   its OWN MF-LOSO (a superset, but still few-DOF) [LYA §2.2].

10. **The cosmology contract is PRIYA-exact** (A_p = As·(5π)^(ns−1) verified to machine
    precision; blinding is parameter-blind on (A_p,n_s) only, correctly NOT a data-vector
    shift because the nuisances would self-unblind it) [COS §2.1/2.5, BAY §R7, LYA].

---

## 2. Cross-lens CONFLICTS / tensions (adjudicated where possible)

**C1 — PRIYA parameter RANGES differ between [COS] and [LYA]. Real discrepancy; must be
resolved before quoting a box.** [COS §1] lists the *repo's* design box `data.py:31-41`:
ns∈[0.8,1.05], herei∈[3.5,4.5], heref∈[2.2,3.2], alphaq∈[1.3,3.0], hireionz∈[6.5,8.0].
[LYA §2.1] lists the *PRIYA paper* Table 2: ns(n_P)∈[0.8,0.995], herei∈[3.5,4.1],
heref∈[2.6,3.2], alphaq∈[1.3,2.5]. **These are not the same numbers.** Adjudication: the
two lenses are quoting two different objects — [COS] read the repo `PARAM_LIMITS`, [LYA]
read the PRIYA PDF. The repo box is WIDER than the published PRIYA grid on at least ns,
herei, heref, alphaq. This is most likely the repo's *design/emulation* box (which can
legitimately bound a superset, esp. with the 60-sim Bayes-opt extension [LYA §2.2]), but
it is NOT verified that the wider box is covered by training points — extrapolation outside
the PRIYA grid would be silent and dangerous near the box edges (and n_s edge pile-up is
already a flagged coverage caveat, [COS §R5]). **ACTION for main agent: reconcile
`data.PARAM_LIMITS` against the PRIYA published grid; confirm whether the wider repo box
is backed by training points or is an extrapolation hazard.** This is the one factual
conflict the meta-review cannot resolve from the reports alone.

> **RESOLVED 2026-06-06 (main agent, direct read).** `data.PARAM_LIMITS` [data.py:31-41] is
> byte-identical to `emulator_params.json::param_limits` (e.g.
> `/home/mfho/lya_emulator_full/dtau-48-48/emulator_params.json`) — the PRIYA emulator
> suite's actual **Latin-hypercube DESIGN box** (the box the 48 + 12 Bayes-opt LF training
> sims were drawn from). So the wider repo box IS backed by training points by construction;
> it is NOT an extrapolation hazard. [LYA]'s narrower "PRIYA Table 2" ranges
> (ns≤0.995, herei≤4.1, alphaq≤2.5) are the published-node / marginalized-posterior ranges,
> NOT the design box. No action — but keep the informative priors on the weak directions
> (omegamh2, bhfeedback, hireionz) so the fit is not pushed to box corners where the
> SVD-basis ringing [LYA-R7] lives.

**C2 — The A_p pivot scale: π/4 /Mpc [COS] vs 0.78 Mpc⁻¹ [LYA]. NOT a conflict (consistent).**
[COS §2.1] says k_p = π/4 /Mpc ≈ 0.785; [LYA §2.1] says k=0.78 Mpc⁻¹. These agree to
rounding — π/4 = 0.7854. The Lyα pivot is the same scale; A_p = As·(5π)^(ns−1) with
5π = (π/4)/0.05 [COS §2.1]. No action.

**C3 — Severity rating of the closing-step / coherent-low-k risk: [CS] rates it "LOW-risk,
JAX-clean" while [BAY], [COS], [LYA] rate it the #1 risk. NOT a real disagreement — they
are rating different things.** [CS-R3] is rating the *code change* (threading g_zc into
`_legb_model`) — and is correct that the mechanics are near-zero-risk (the (n_z,3) path
already exists, the multiply is differentiable, no new tracer control flow). [BAY-R1],
[COS-R1], [LYA-R1] are rating the *statistical/physical validity* of using the exact-truth
shape as the certification — and are correct that this is the central methodological trap.
**Adjudication: both are right. The change is safe to MAKE; the question is whether the
resulting closure CERTIFIES the real fit. It does not, by itself (see §5).** The main agent
must not read [CS]'s "low-risk" as "go ahead and certify on the exact shape."

**C4 — Is the in-sample circularity of the 0.93 whitening number a concern?** [BAY-R4]
flags it as "mildly circular" and asks for per-(k-band,z) whitening rather than a pooled
scalar; [CS] and [COS] note it is mild (0.917 vs 0.926) and largely de-risked by holdout0.
Adjudication: agreement that it is MILD, with [BAY] adding the sharper point that a pooled
scalar var≈1 can MASK a low-k var≫1 cancelled by high-k var≪1 — which is exactly the
coherent-mode failure (§4-R1). **The pooled 0.93 is necessary but not sufficient; report
whitening split by (k-band, z).** No conflict, [BAY] is strictly more conservative and correct.

**C5 — "Is HCD marginalization free?"** [COS] and [BAY] both cite the Fisher forecast
(A_p ×1.10, "nearly free"). But [BAY-R1] and [LYA-R1] both warn a SLOPE nuisance
specifically targets the A_p-degenerate low-k mode, so the "free" result may NOT hold once
a z-slope is sampled. Adjudication: no contradiction — the Fisher "free" applies to the
α-AMPLITUDE marginalization at fixed shape; it says nothing about a slope nuisance. Treat
"HCD marg is free" as true for amplitude only; the slope is the untested lever (§4-R1, §5).

**C6 — Fisher forecast status.** All lenses agree it is DIAGNOSTIC-ONLY (single fiducial,
linearized, n_s truth near box edge, 6-sim MF Fisher rank-deficient) [COS §R4, BAY §3].
No conflict — unanimous: trust it as a *ranking* ("τ₀ is the cost, HCD-amplitude is free"),
NEVER quote σ(A_p)=0.42·box as a result.

No lens contradicts another on the core mechanism. The only genuine factual discrepancy
needing resolution is **C1 (the parameter box)**.

---

## 3. Verified vs assumed — the consolidated ledger

### VERIFIED (test / figure / paper / live re-execution behind it)
| Claim | Evidence | Lens(es) |
|---|---|---|
| 28 tests pass | re-run this session, exit 0, 137 s | [CS] (ran it) |
| potential == loglik + Σ logprior | test_closure_sbc.py:93, atol 1e-6 | [BAY] |
| Uniform exactly flat in-box (no soft wall) | test_closure_sbc.py:119 | [BAY], [COS] |
| C_mock == C_like at truth; vmap==loop; softplus α_dla>0 | test_closure_sbc.py:178/132/172 | [BAY] |
| cross-class einsum SPD ⇒ emu_var≥0; ==diagonal at band centre | rtol 1e-10 | [CS], [BAY] |
| θ-blind baseline ∂m̂/∂θ≡0; ∂P/∂θ FD 7e-8; round-trip 8e-16 | jacfwd=0, walkthrough §8 | [COS], [CS] |
| A_p = As·(5π)^(ns−1) to machine precision vs PRIYA params | sims 0/29/44 | [COS] |
| HCD bug fix (Δ_LLS≡0 → P_c−P_clean); additive≡multiplicative | predict.py:88-98 (algebraic) | [LYA], [COS] |
| NaN-before-interp guard; gaussian logdet carried | jax-traps #29, likelihood.py:154-175 | [CS], [COS] |
| Closing-step ladder −0.78 → −0.24 → +0.03σ | **reproduced live** by [LYA]; forward-only | [LYA], all confirm |
| Sim w_c(z) abs slope ~2.6 vs lit RATIO slope 0.15–0.95 | **computed live** from cache | [LYA] |
| C_emu sub-dominant (median 0.014·C_data) EXCEPT low-k DESI max 0.52 | handoff §3a-1 | [COS], [BAY], [CS] |
| PRIYA box/sims/Eq-2.18/mean-flux/convergence | VERBATIM from /tmp/priya.txt | [LYA] |
| 6 HR cache sims (superset of PRIYA's 3 HF); LF cache 60 sims | listed from HDF5 | [LYA] |
| de-circularized ρ (folds 1-7) exists, circularity mild | files on disk | [BAY], [CS], [COS] |

### ASSUMED / NOT-YET-IMPLEMENTED / STATED-ONLY (could bite)
| Claim | Status | Lens(es) |
|---|---|---|
| **Golden-regression guard** | **does NOT exist on disk** (`tests/golden/` absent) — a plan | [CS-R1] |
| **Exact per-z g_c(z) closing step** | NOT wired into `_legb_model` (code uses lit-slope, −0.24σ) | all four |
| **§0c DLA Gaussian-at-0 + DLA-masked mock truth** | DECIDED, NOT CODED (live: HCD_DLA_RESIDUAL_FRAC=0.30 + softplus + full-w_c truth) | [LYA-R2], [BAY-R2], [COS], [CS] |
| **Low-rank correlated-in-k C_emu (§0b-2)** | does NOT exist (C_emu diagonal-in-k everywhere) | [BAY], [COS-R1], [CS], [LYA] |
| **MF wired into `data_loglik`** | BUILT but NOT wired (cert is LF-only; MF must be re-certified) | [CS-R2], [LYA-R3] |
| **Leg-B coverage as a verdict** | NEVER read (only smoke N≈4-8 + a buggy N=50 pilot) | [BAY], [COS] |
| **Leg-A SBC at production scale (L≥99, N≥400)** | NOT re-run (only --smoke) | [BAY-R6] |
| **(A_p,n_s) coverage under real NUTS** | NOT certified (only single-fiducial Fisher, diagnostic-only) | [COS], [BAY] |
| **Production ⟨F⟩(z) anchor + σ(z)** | Becker13 MVP; LYA-CONSULT open (meanflux_prior.py:76) | [COS-R3], [LYA-R6], [BAY] |
| **PRIYA param box vs repo PARAM_LIMITS (C1)** | DISCREPANT ranges; coverage of wider repo box unverified | [COS]/[LYA] (meta C1) |
| **`delta_mode='none'` (drop PRIYA's δ(θ))** | assumed small enough to carry in C_emu; PRIYA keeps it | [LYA-R3] |
| **KS leg loader bugs** (off-by-one drop 5 not 4; mis-cite; CACHE_KMAX caps unique 0.079/0.099) | KNOWN bugs; KS ≈ redundant with DESI as-configured | [LYA-R4] |
| **_data_loglik_legcore z-mean DLA core** | documented MVP, not per-z | [BAY], [CS] |
| **cemu_inflate calibration** | machinery exists, uses Gaussian-posterior surrogate, default 1.0, not run | [BAY] |
| **σ=2% cosmology mis-centering bias** | 0.192σ — AT the 0.2σ gate single-z; multi-z stacking unquantified | [COS-R6] |

---

## 4. Prioritized OPEN QUESTIONS / risks for the real-data fit

Ranked by impact on the (A_p, n_s) result. Each: which lens(es), why it matters, what resolves it.

**R1 [CRITICAL] — The coherent low-k closure residual sits in the exact A_p/n_s regime, and
the documented fix is (a) verified forward-only and (b) potentially circular.**
Raised by ALL FOUR lenses as their #1 (or co-#1). Why: low-k is exactly where A_p (a low-k
boost) and n_s (pivot k≈0.005) live; it is the ONE regime where C_emu reaches ~0.5·C_data
(low-k DESI, z~3.2) [COS-R2]. A diagonal/k-diagonal-cross-class C_emu can RESIZE but cannot
WHITEN a coherent mode → NUTS absorbs it onto A_p amplitude + HCD-α (the buggy pilot showed
A_p ≈10σ low, α_subDLA ≈19σ low). The "+0.03σ" fix threads the SIM's exact w_c(z), but the
real fit uses the literature dN/dX z-slope prior — so the closure can pass against a shape
production never runs (§5). **Resolves by:** (i) certify with the PRODUCTION α(z)
parametrization (sampled amplitude + a sampled z-slope marginalized, NOT fixed to truth);
(ii) BUILD the low-rank correlated-in-k C_emu (plan §0b-2, currently absent); (iii) re-run
Leg-B coverage at N≥99 and show the A_p/n_s posterior pull collapses through the POSTERIOR,
not just the construction. (See §5 for the decision.)

**R2 [HIGH] — No golden-regression guard exists; the closing-step α-threading must not be
touched until it does.** [CS-R1]. Why: `tests/golden/` is absent; a refactor of the (3,)
broadcast path (mis-indexing `alpha_hcd[iz]`, changing the einsum) would be caught only by
the looser functional asserts, silently biasing the real fit. **Resolves by:** implement the
handoff RESUME step-1 guard FIRST — freeze (P_model, C_total) at a fixed (θ,τ₀,α=(3,)),
assert allclose rtol=1e-12 on the legacy LF path. This is the cheapest, highest-leverage
single CS action and is a hard prerequisite to the α work.

**R3 [HIGH] — The §0c DLA decision is decided but UNIMPLEMENTED, and it interacts with R1.**
[LYA-R2], [BAY-R2], also [COS], [CS]. Why: live code uses a one-sided softplus α_DLA centred
at 0.30·r·w_DLA and a FULL-w_c (un-masked) mock truth — inconsistent with the DLA-masked data
(DESI ≥90% masked, KS ~0%). The closure mock therefore carries a DLA contamination the data
does not have, and the one-sided prior inflates the (low-k) DLA contribution → feeds the same
A_p mode as R1. **Resolves by:** code §0c (α_DLA Gaussian-at-0 per leg; DLA-masked mock truth)
BEFORE the cert; then verify the DLA posterior does not rail negative and A_p coverage is
unchanged with the DLA sector ~removed [BAY-R2].

**R4 [HIGH] — MF is not wired into the certified likelihood.** [CS-R2], [LYA-R3]. Why: the
Leg-B cert is of the LF likelihood (`data_loglik` calls the LF forward directly, no `mf=`
kwarg). When MF is wired, the dropped δ(θ) must be carried in C_emu, and a real fit run with
MF but certified only on LF has an under-validated high-k channel. Compounding [LYA-R3]: if
the dropped δ(θ) has a COHERENT k-shape correlated with n_s/A_p, scaling a diagonal variance
does not marginalize it. **Resolves by:** re-run the closure after MF wiring; confirm the
dropped-δ residual is incoherent-in-cosmology, not just small in RMS. Do NOT assume the LF
cert transfers.

**R5 [MEDIUM-HIGH] — τ₀ is the dominant cost on n_s (Fisher ×1.28) and the production ⟨F⟩(z)
anchor is unconfirmed.** [COS-R3], [LYA-R6], [BAY]. Why: A_p↔τ₀ is the classic mean-flux
degeneracy; a wrong τ₀ prior width biases A_p. The closure uses Becker13/Kim MVP; production
needs the observed −ln⟨F⟩(z) in the data's exact DLA-masked/metal-corrected state. [COS-R3]
adds a center-mismatch check: confirm the becker13 prior center maps to a ladder α inside
[0.656,1.331] interior, not near a band edge. **Resolves by:** the open LYA-CONSULT
(meanflux_prior.py:76) — set the production ⟨F⟩(z) + per-z error budget with Lyα sign-off
before the real fit. (Not a closure-stage blocker; the closure uses the sim's own τ₀.)

**R6 [MEDIUM] — NUTS practicalities could silently degrade the verdict, and the cost exceeds
budget.** [BAY-R3]. Why: ESS ≈ 0.18/sample → at n_samples=250 the thinned L falls below
L_FLOOR=99, so coverage is uncomputable as a verdict; one cfg3 mock >35 min; N=600 ≈ 4800
CPU-h EXCEEDS the ~4000 budget (memory cavestru0-compute-budget). A slope×amplitude product
(if R1's slope nuisance is added) is a classic funnel → watch divergences. **Resolves by:**
profile before any N≥150 launch; bump n_samples to L≥99; consider max_tree_depth 8→6,
diagonal-mass warmup, non-centered parametrization for the slope×amplitude product.

**R7 [MEDIUM] — Parameter-box discrepancy (C1).** [COS]/[LYA] meta-C1. Why: repo
`PARAM_LIMITS` is wider than the published PRIYA grid on ns/herei/heref/alphaq; if the wider
box is not backed by training points, the emulator extrapolates near the edges, and n_s edge
pile-up [COS-R5] already distorts coverage there. **Resolves by:** reconcile the box vs the
PRIYA grid + the 60-sim design; confirm training coverage of the wider box.

**R8 [LOW-MEDIUM] — Leg-A SBC has not been run at production scale.** [BAY-R6]. Why: Leg-A
certifies the SAMPLER; "if Leg A fails, every closure number is uninterpretable." Only
--smoke exists. **Resolves by:** run the full Leg-A SBC (L≥99, N≥400) before trusting any
Leg-B coverage — the sampler must be certified independent of the mean-model fix.

**R9 [LOW-MEDIUM] — Whitening reported as a pooled scalar can mask the coherent low-k mode.**
[BAY-R4]. Why: var≈1 pooled can hide low-k var≫1 cancelled by high-k var≪1 — exactly R1's
failure. **Resolves by:** report whitening var per (k-band, z), not just pooled.

**R10 [LOW] — KS leg loader bugs make KS ≈ redundant with DESI as-configured.** [LYA-R4].
Off-by-one (drops 5 not 4 bins), mis-cite, CACHE_KMAX=0.069 caps KS's unique 0.079/0.099.
Not a bias risk for the DESI-primary fit, but do not claim KS small-scale constraining power
until the loader is fixed AND an HR/MF emulator reaches 0.08–0.1.

---

## 5. The closing-step decision (the α(z)-shape fix)

**The situation.** The closure forward currently applies α(z) = α_pivot·((1+z)/(1+z_p))^s_c
with s_c = the LITERATURE ratio-slope `HCD_LIT_OVER_SIM_SLOPE` (0.95,0.15,0.40)
[closure_legb.py:451-454]. But the held-out sim's *absolute* incidence slope is ~2.6 [LYA,
computed live] — the lit slope is a small DIFFERENTIAL on top of the sim's own steep w_c(z),
which is why the lit-slope shape only removes 70% of the bias (−0.24σ). Threading the truth's
exact per-z g_c(z) drives the construction residual to +0.03σ. The decision is HOW to treat
the z-shape in the certification.

**The three options (synthesized from all four reports):**

- **(B-alone) τ₀-analog: fixed known g_c(z) shape × sampled amplitude.** Thread a fixed
  shape (either the truth's exact g(z), or a committed literature shape) and sample only the
  amplitude. — *Stances:* [BAY] "principled IF g_c(z) is genuinely known with negligible
  uncertainty (like the Kim curve for τ₀); but the closure itself proved the slope is
  uncertain by ~3×, so B-alone is **under-conservative**." [LYA] "defensible ONLY as a
  fixed-selection-function check; for the real fit the same −0.24σ pull reappears if the
  fixed shape's slope doesn't bracket the truth." [COS] "matching the SIM's w_c(z) certifies
  a shape production never runs — the coverage may not transfer." [CS] neutral (the code
  change is low-risk either way). **Consensus: NOT sufficient alone.**

- **(B+slope / option A) sample a per-class z-slope nuisance with a WIDE prior** bracketing
  [lit-slope, sim-slope ~2.6], and certify coverage with the slope MARGINALIZED. — *Stances:*
  [BAY] "this is the statistically honest marginalization; **recommended primary**, with a
  coverage gate on A_p; watch the slope×amplitude funnel." [LYA] "**prefer this (option A)**;
  certify the closure passes when the slope is marginalized, not fixed to truth; this is the
  #1 thing to escalate." [COS] "certify with the production α(z) parametrization (sampled
  amplitude + slope)." [CS] mechanically clean (adds 1-3 params, no JAX trap; funnel →
  non-centered). **Consensus: this is the principled primary.**

- **(C) keep the lit-slope, absorb the residual into a low-rank correlated-in-k C_emu.** —
  *Stances:* [BAY] "sound as a marginalization ONLY if the low-rank amplitude is calibrated
  to the full slope-prior RANGE, not to the in-sample residual of one fold (that would be
  circular and too tight) — i.e. it is really (B+slope) re-expressed as a covariance." [COS]
  "the correlated-in-k C_emu is REQUIRED regardless (it is the only thing that can whiten the
  residual coherent mode), but it is NOT yet built." [LYA]/[CS] do not oppose; treat as
  belt-and-braces. **Consensus: C is necessary infrastructure (R1) AND a valid
  belt-and-braces marginalization IF its amplitude covers the slope-prior range — but it is
  not a substitute for B+slope.**

**Cross-lens recommendation (strong, near-unanimous):**
**Adopt (B+slope) as the primary** — sample a per-class HCD z-slope nuisance with a prior
wide enough to bracket [lit-slope 0.15–0.95, sim absolute-slope ~2.6], and **certify Leg-B
coverage with the slope MARGINALIZED, gated on A_p coverage** — NOT by hard-wiring the
held-out sim's exact shape. Use **(C) the low-rank correlated-in-k C_emu as belt-and-braces**,
with its amplitude tied to the slope-prior range (NOT the holdout-0 in-sample residual). The
exact-per-z-g(z) construction (+0.03σ) is a NECESSARY sampler-side sanity check but is
NECESSARY-not-SUFFICIENT: it validates the sampler given a correct mean model, it does NOT
certify the real-data mean model. Sequence it AFTER the golden guard (R2). All four lenses
converge on this; [BAY] and [LYA] state it most explicitly, [COS] frames it as "certify with
the production parametrization," [CS] confirms it is mechanically clean.

**The ONE design question that needs the PI's call:**
**What prior do we put on the per-class HCD z-slope nuisance, and is the resulting A_p/n_s
inflation acceptable?** Specifically: should the slope prior be (a) wide enough to bracket
the full [lit, sim≈2.6] range — maximally honest but maximally degenerate with A_p at low-k,
likely inflating σ(A_p) and risking funnels; or (b) tied to the literature dN/dX z-evolution
with a tighter, physically-motivated width — less conservative but cleaner sampling. This is
a science-vs-conservatism tradeoff on the cosmology error bar that only the PI can set,
because it trades real σ(A_p) constraining power against robustness to the HCD-incidence
z-shape systematic. (Secondary, follow-on: confirm the §0c DLA decision and the production
⟨F⟩(z) anchor are settled before the cert, since both feed the same low-k A_p mode.)

---

## 6. "What the main agent MUST know" — tight digest

1. **State:** Phase-2 emulator MERGED (main, PR #10). Phase-C likelihood on
   `phase2c-likelihood`. 28 tests pass [CS ran them]. The machinery is sound + tested; the
   FIXES are planned but UNIMPLEMENTED.

2. **Forward model:** `P_obs = P_clean + Σ_c α_c·(P_c − P_clean)` [predict.py:83-103];
   θ-blind baseline + σ_cosmo·r̂ reconstruction [predict.py:29-39] — ALL cosmology flows
   through r̂ (the load-bearing identifiability move, ∂m̂/∂θ≡0 verified).

3. **Two legs, two nulls:** Leg-A (cache grid) IS rank-uniform → ECDF gate valid. Leg-B
   (real DESI/KS grid, held-out-sim truth) is NOT → **coverage≥nominal + bias≈0 are the ONLY
   valid headlines; ECDF is diagnostic-only** [closure_legb.py:655-676]. Never gate Leg-B on
   ECDF.

4. **The next-actions sequence (handoff RESUME, all lenses agree):**
   (1) golden-regression guard FIRST [CS-R1] → (2) z-resolved α(z) as a SAMPLED slope nuisance,
   NOT hard-wired truth (§5) → (3) build the low-rank correlated-in-k C_emu (plan §0b-2) →
   (4) code §0c DLA-masked + α_DLA Gaussian-at-0 → (5) Leg-A SBC at scale, then Leg-B coverage
   at N≥99. Do NOT certify before (1)-(4).

5. **The #1 risk (R1):** a coherent low-k over-prediction sits in the exact A_p/n_s regime;
   diagonal/k-diagonal C_emu can resize but not whiten it; the buggy pilot showed A_p ≈10σ
   low. The fix is verified forward-only (−0.78→−0.24→+0.03σ, [LYA] reproduced live), NOT
   through NUTS.

6. **The closing-step decision (§5):** adopt **(B+slope)** — sample a per-class HCD z-slope
   (wide prior bracketing lit 0.15–0.95 and sim ~2.6), certify with the slope MARGINALIZED,
   NOT fixed to the held-out sim's truth. (C) correlated C_emu as belt-and-braces. The PI must
   set the slope-prior width (science-vs-conservatism tradeoff on σ(A_p)).

7. **Closing-step lands here:** `_legb_model` α(z) at closure_legb.py:451-454; the z-median
   collapse to UNDO at closure_legb.py:283 (ADD a per-z `w_c_perz`, do NOT replace the (3,)
   `w_c` the α-coverage truth vector needs — [CS-R4]). Forward-only verifier:
   `scripts/diag_legb_zresolved_alpha_check.py` (uses holdout0 ρ; run before any NUTS).

8. **JAX traps that kill NUTS (do not break these):** sanitise YDATA BEFORE `jnp.interp`,
   never the output after [likelihood.py:98-151, jax-traps #29]; gaussian_loglik jitters
   before Cholesky + carries the logdet [likelihood.py:154-175]; x64 on package import — always
   `import hcd_analysis.emulator` before `import jax` [__init__.py:11]; LF backbone frozen via
   stop_gradient not static field [multifidelity.py:153-166, #26]; never `jax.jit` a bound MF
   method, use `eqx.filter_jit` [#27/#28].

9. **DLA §0c is decided but NOT coded** [LYA-R2]: live code = HCD_DLA_RESIDUAL_FRAC=0.30
   [inference.py:61] + softplus + full-w_c truth — inconsistent with DLA-masked data. Code it
   before the cert.

10. **MF is BUILT but NOT wired into `data_loglik`** [CS-R2]: the cert is LF-only; MF must be
    re-certified after wiring; the dropped δ(θ) must be carried in C_emu and shown
    incoherent-in-cosmology [LYA-R3].

11. **C_emu:** sub-dominant (median 0.014·C_data) EXCEPT low-k DESI z~3.2 where it reaches
    0.52·C_data — the cosmology-critical regime. Cross-class form is production (whitening
    1.28→0.93, in-sample mild-circular; report split per (k-band,z) [BAY-R4]). The
    correlated-in-k form needed to whiten the coherent mode does NOT exist yet.

12. **τ₀:** sampled on the ladder coordinate α=τ₀/Kim(z) (Jacobian-free); Becker13 production
    anchor is an MVP — LYA-CONSULT open [meanflux_prior.py:76]; τ₀ is the dominant n_s cost
    (Fisher ×1.28) and A_p↔τ₀ degenerate → set the production ⟨F⟩(z)+σ(z) before the real fit.

13. **Cosmology contract:** A_p = As·(5π)^(ns−1) at the Lyα pivot (verified machine-precision).
    Blinding is parameter-blind on (A_p,n_s) ONLY (commit the SHA256 seed before the first real
    fit; unblind ONCE). DESI real cosmology stays LOCAL/gitignored; KS + code + mocks committable.

14. **OPEN factual conflict to resolve (C1):** repo `data.PARAM_LIMITS` [data.py:31-41] is
    WIDER than the published PRIYA grid on ns/herei/heref/alphaq [LYA §2.1 vs COS §1]. Confirm
    the wider box is backed by training points, not an extrapolation hazard.

15. **Compute is tight:** N=600 ≈ 4800 CPU-h > ~4000 budget; ESS≈0.18/sample; profile before
    any N≥150 launch [BAY-R6, memory cavestru0-compute-budget].

16. **Key files:** `closure_legb.py` (Leg-B model + driver), `data_likelihood.py`
    (real-data binding, predict_P_obs_on_leg:294, data_loglik:384), `inference.py`
    (incidence prior :77-106, log_lik_multiz:209), `likelihood.py` (gaussian_loglik:154,
    interp guards :98-151), `sampler_numpyro.py` (Leg-A model), `predict.py` (forward),
    `closure_diagnostics.py` (verdict math). Plan: `2026-06-05-mf-likelihood-wiring-plan.md`
    §0/§0b/§0c. Handoff: `SESSION_HANDOVER_2026_06_05.md` RESUME block. PRIYA text: `/tmp/priya.txt`.
