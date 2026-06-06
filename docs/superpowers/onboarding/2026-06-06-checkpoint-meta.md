# Checkpoint meta-review — HCD slope-prior spec — 2026-06-06

Synthesizes the four lens critiques of `docs/superpowers/specs/2026-06-06-hcd-slope-prior-spec.md`:
- Bayesian/PPL: `2026-06-06-checkpoint-bayesian.md`
- CS/JAX: `2026-06-06-checkpoint-cs.md`
- Cosmology/P1D: `2026-06-06-checkpoint-cosmology.md`
- Lyα/IGM: `2026-06-06-checkpoint-lya.md`

All four lenses returned **GO-WITH-CHANGES**. None said NO-GO; none said clean GO.

---

## Overall verdict: **GO-WITH-CHANGES**

The factorized model `α_c(z) = α_pivot,c · exp(δ_{c,leg}) · g_fixed,c(z) · ((1+z)/(1+z_p))^{s_c}`
(spec §1, line 28) is the statistically correct fix for the coherent low-k leak, and the
**variance** conclusion (marginalizing the slope costs ×1.04 on σ(A_p)) is framing-robust and
stands across all four lenses. But the spec **certifies the wrong configuration**: every headline
number (×1.04, +0.167σ A_p bias) was measured with `g_fixed` set to the SIM's own shape
(`diag_legb_slope_prior_tradeoff.py:90`, `g_fixed = wc_perz[sel_sim]/wc_pivot`), i.e. the
"closure-only" framing the spec itself recommends AGAINST (§4 lines 83-87). All four lenses caught
this independently. Two further LOCKED-part defects compound it: the cosmology lens found the n_s
bias is **−0.223σ — already over the 0.2σ gate** and entirely unmentioned in §2; and the CS lens
found the §1/§3 model **cannot be implemented at the site it names** (`_legb_model` builds one
global-z α array; the per-leg `δ_leg` forces a real signature change to `_data_loglik_legcore`, not
a 1-line multiply). The fixes are cheap and forward-only — one sweep re-run on the production
`g_fixed` with the §0c mask, computing the exact (non-linearized) bias and reporting n_s — plus a
golden guard and a contract-change refactor that must precede coding. The width's variance rationale
survives; its bias rationale and the gate claim do not, until re-measured. Hence GO-WITH-CHANGES,
not GO.

---

## Resolved verdicts on the 4 open questions

### Q1 — Is the closure `g_fixed` production-faithful? Should it be lit-derived, with sim-truth as the survival target?

**Consensus: UNANIMOUS YES — production-faithful (lit-derived) `g_fixed` is REQUIRED.** All four
lenses agree, and three of the four independently flagged the same trap: the sweep that produced the
locked numbers used the SIM shape (sweep:90), not the production one, so the locked numbers describe
a config the real fit never runs.

- **Bayesian** (lines 22-78): YES; the sim-`g_fixed` framing makes `s=0` the +0.03σ exact point "by
  construction" → "certifies a configuration that will never run." Adds the load-bearing caveat: in
  the production framing `s=0` is NO LONGER exact (sim slope ~2.6 vs lit center is a 3–5σ_s offset
  the lit width cannot absorb), so the bias likely GROWS and the correlated-in-k C_emu may become
  required (his option (b)).
- **Cosmology** (lines 25-63): "REQUIRED — not merely 'recommended' as the spec hedges (line 97)."
  The low-k excess lives in the A_p/n_s regime; a z-shape error tilts it in z, which a k-diagonal
  C_emu cannot whiten. Calls the sim-shape sweep "a real inconsistency."
- **Lyα** (lines 22-63): YES, and **verified the construction LIVE** (`/tmp/q1_gfixed.py`):
  literature dN/dX × X̄(z) through `w_c_from_mu` gives LLS w_c slope **+3.20**, of which +2.02 is the
  X̄(z) path-length geometry (θ-independent, identical in sim and data) and +1.18 is incidence; the
  sim empirical slope is +2.60 → **physically-bounded shape gap −0.60** ≈ the literature dN/dX
  ratio-slope. The map is faithful and the gap is within the lit dN/dX spread, so the slope nuisance
  + anchor have the right magnitude to absorb it.
- **CS** (lines 24-63): defers the science but is decisive on mechanics — the lit-derived `g_fixed`
  is a PRECOMPUTED constant (the `(A_c, γ_c)` lit coefficients and X̄(z) have no θ dependence), and
  in the lit framing it is the SAME for every mock → store it as a frozen field on `LegBCtx`, never
  recompute in the traced model. This makes the closure forward byte-identical to the real fit.

**Dissent:** none on the direction. The only nuance is Bayesian's option (a)-vs-(b) fork: whether the
lit-vs-sim residual at the truth is already < gate (a), or whether the correlated C_emu becomes
required (b). That fork is *answered by the rerun*, not by argument.

**Actionable decision:** Adopt the production lit-derived `g_fixed`. Build it once, host-side, via
`dndx_wc.w_c_from_mu(dN/dX · X̄(z))` (Lyα: NEVER from dN/dX alone — the incidence-only +1.2 slope IS
the original bug in disguise; Lyα A2), store it as a `(n_z_leg,3)` frozen constant on `LegBCtx` per
leg (CS rec 3), and document the X̄(z) source (use the sim's `snap_total_path_dX/n_skewers`; it is
near-invariant geometry — Lyα A2). Re-run the sweep with this `g_fixed` before locking any number.
**No PI call needed** — this is a unanimous technical resolution.

### Q2 — Slope center, and fixed-constant vs hyper-prior

**Consensus: STRONG. Fixed constant, NO hyper-prior. Center FOLLOWS from the Q1 framing — it is not
a free parameter.** All four lenses converge.

- The center is **coupled to Q1** and this resolves the apparent WLS-vs-code disagreement: in the
  production framing (Q1=lit-derived), `g_fixed` ALREADY encodes the lit z-shape, so `s_c` is the
  *residual correction on top of it* and its honest center is **0** ("no correction"). The spec's own
  §4 line 92-93 says this, and Bayesian (108-110), CS (74-76), and Lyα (88-95) all confirm: centering
  at WLS 0.76 in the production framing would **double-count** the lit slope (once in `g_fixed`, once
  in the center).
- **WLS vs code, as a provenance fact** (all four): the code constants
  `HCD_LIT_OVER_SIM_SLOPE=(0.95,0.15,0.40)` (`inference.py:74`) are an **unweighted** `np.polyfit`
  (`plot_dndx_vs_literature.py:102-104`, no error bars); the WLS re-fit (0.76/0.05/1.16) propagates
  the quoted dN/dX errors. **If a literal lit-slope is ever needed as a fixed value** (i.e. if the PI
  keeps the sim-`g_fixed` framing — which Q1 rejects), use the **WLS 0.76/0.05**, not the OLS 0.95/0.15.
- **Hyper-prior: rejected, 4/4.** It buys nothing (the data cannot inform a hierarchical center with
  ~12 z-bins and a slope that barely competes with A_p) and adds a second funnel level (the classic
  centered-hierarchical funnel) on top of the amplitude×slope product funnel (Bayesian 112-122, CS
  77-81, Cosmology 81-85, Lyα 88-92). If robustness to the center is wanted, WIDEN the fixed width
  (the z-edge inflation already does this), don't add a level.

**Dissent:** none. Lyα adds a worthwhile refinement (lines 96-104, rec 6): report the
**in-window (z=2.5–3.5) WLS center** alongside the full-window one; if they differ by >0.2 the
full-window center is being set by the untrusted z=4.23 edge point (±0.19). Cheap re-fit; do it.

**Actionable decision:** `s_c ~ Normal(0, σ_s)`, FIXED center at 0 under the production `g_fixed`,
no hyper-prior. Width = the WLS lit 1σ (0.52/0.53/0.33) with z-edge inflation. Report the in-window
WLS center as a check. **No PI call needed** — the center is determined by Q1.

### Q3 — Trust the Fisher bias (0.167σ), or compute the exact construction bias?

**Consensus: UNANIMOUS — compute the EXACT (non-linearized) bias first. Do NOT lock the width on the
linearized 0.167σ.** This was the single strongest shared recommendation; CS, Cosmology, and Lyα each
called it their sharpest or strongest point.

- **The linearization is measured-wrong on the load-bearing axis:** the sweep's own cross-check
  (`legb_slope_prior_tradeoff.txt:6-8`) reports `||J_s·gap − (P(gap)−P(0))||/||·|| = 0.129 (DESI),
  0.152 (KS)` — 13–15% off. The Fisher bias is exactly `J_s·gap` propagated. 0.167σ is already 83% of
  the 0.2σ gate; a 13–15% under-estimate (and `(1+z)^s` is convex over the large gap, so `J_s·gap`
  *under*-estimates ΔP) plausibly pushes it to ~0.19σ, at the gate (CS 88-114, Lyα 110-117, Cosmology
  point 4).
- **Bayesian adds two reasons beyond linearization** (lines 124-162): the Fisher bias ignores the
  tight LLS prior boundary (σ/μ=0.15, the most cosmology-relevant prior) and the n_s box edge, and it
  is a MAP-shift surrogate for a coverage quantity — they coincide only in the Gaussian-interior limit,
  which is exactly where A_p is NOT (it sits on the (A_p, α_LLS) degeneracy ridge).
- **Cosmology adds the deepest finding here** (lines 92-120): (1) the bias is **width-INSENSITIVE**
  (+0.185σ at narrowest → +0.167σ at lit → +0.056σ only at 5× lit), so the spec's safety logic is
  INVERTED — you cannot narrow your way to safety, you must WIDEN; the spec's "free headroom to go
  wider" (line 49) is the only bias-reducing lever and should be the recommended direction. (2) the
  **n_s bias is −0.223σ at lit — over the gate** (see Errors §below). (3) the single fiducial's n_s
  sits ON the box edge (n_s=0.803), making the n_s number the least trustworthy in the table.
- **Lyα and Bayesian both note** the bias must also be recomputed on the *production* `g_fixed` (not
  just exactly), and Lyα that the sweep was run WITHOUT the §0c DLA mask (Lyα E2 — overstates by the
  +0.062σ DLA term). So the rerun is one job satisfying Q1 + Q3 + the §0c coupling at once.

**Dissent:** none on "compute exact." CS's honest caveat (lines 116-118): the exact-ΔP Fisher bias is
itself still a posterior-linearization; the *fully* honest number is the NUTS closure — but the
exact-ΔP Fisher is strictly better than `J_s·gap` and is the right pre-NUTS gate.

**Actionable decision:** Extend `diag_legb_slope_prior_tradeoff.py` (~15 lines per CS 102-114; `P_s`,
`P_0`, `J`, `Cinv`, `Cpost` already in scope) to propagate the EXACT `ΔP = P_s − P_0` through
`(F+P)⁻¹ Jᵀ Cinv` and read the A_p AND n_s components, on the production `g_fixed`, with the §0c mask,
at ≥1 interior-n_s fiducial. Gate the width on `bias_exact < 0.2σ` for BOTH A_p and n_s.
**No PI call on the method.** The PI call is downstream (see "Open items").

### Q4 — Anchor float on amplitude vs slope (vs both)?

**Consensus: UNANIMOUS — AMPLITUDE only, slope stays GLOBAL.** 4/4, with three independent
justifications that reinforce each other.

- **Bayesian** (164-191): the bias is amplitude(LLS)-driven (+0.106σ of +0.167σ); a per-leg slope is
  an "unidentifiable second amplitude-like direction" — it would add up to 4 latents the data cannot
  separate, inflating ESS cost and funnel risk for zero bias reduction.
- **CS** (120-132): adds that a per-leg slope is degenerate with the per-leg amplitude *at the leg's
  z-mean* (a banana), so the two would fight in the sampler — inflating warmup for no identifiability
  gain.
- **Cosmology** (131-154): a per-leg slope opens a per-leg low-k *tilt* that competes with n_s on each
  leg independently — directly eroding n_s, the weaker-protected cosmology parameter. Amplitude float
  competes mainly with A_p (cheap, ×1.04). So amplitude-only is also the n_s-protective choice.
- **Lyα** (140-178): the physics is decisive — KS's absorber-targeted selection shifts *incidence
  amplitude* (more absorbers per sightline), NOT z-*evolution* (which is the same cosmological
  self-shielding physics in both surveys' lines of sight). A per-leg slope models a mechanism that
  does not exist.

**Dissent:** none. Three refinements all four would endorse:
1. **Lyα** (171-178, rec 5): the LLS τ≥1-vs-τ≥2 definitional offset (sim ~25–40% high) is a GLOBAL
   sim-vs-lit offset → it belongs in the amplitude CENTER (`HCD_LIT_OVER_SIM[0]`, `inference.py:70`),
   NOT in `δ_leg` (which would double-count it as a survey difference).
2. **Bayesian** (188-191) and **Cosmology** (149-154): the σ_anchor values (KS 0.25–0.30, DESI
   0.10–0.15) are un-measured "defensible expectations" doing real work — they need their own coverage
   line and a mini σ_anchor sweep before locking.
3. **Bayesian F2** (233-246): a concrete inequality to enforce — `σ_anchor < σ_α` per class (in
   matched units; LLS σ_α≈0.15·μ vs σ_anchor 0.25–0.30 *looks inverted* — pin whether σ_anchor is a
   log-offset or a fractional width), else the global `α_pivot` loses its anchoring role and the
   per-leg amplitudes float freely, re-opening the A_p degeneracy.

**Actionable decision:** Amplitude-only per-leg float, slope global. Keep the LLS definitional offset
in the amplitude center. **One sub-item for the PI:** the σ_anchor magnitudes (Open item 3 below).

---

## Errors found in the LOCKED parts

Three real defects in LOCKED material. They change the **bias / gate** conclusion (and the
implementation claim), but NOT the **width / ×1.04 / LLS-binding** conclusions.

**E-A (all 4 lenses) — the locked ×1.04 and +0.167σ were measured with the SIM `g_fixed`, the
framing the spec rejects.** `diag_legb_slope_prior_tradeoff.py:90` uses `wc_perz[sel_sim]/wc_pivot`;
the docstring (:4-8) says so. The spec's §6 lists "variance ×1.04; bias 0.167σ" as VERIFIED, but they
are verified for the closure-only framing only. **Effect on conclusions:** the ×1.04 VARIANCE is a
marginal-σ(A_p) statement, framing-robust — it STANDS (Bayesian 211, Lyα 191-193, Cosmology 174-178,
all agree). The BIAS does NOT — re-measure. (Lyα E1 sharpens this: the sweep actually MIXES two
framings — variance uses `g_fixed`=sim-w_c, while the bias injects the production-scenario offset
`SIM_LIT_GAP` — so +0.167σ is "a hypothetical, not the construction's bias.")

**E-B (cosmology, lines 102-107, 160-166) — the n_s bias is −0.223σ at the lit width, OVER the 0.2σ
gate, and §2 never mentions it.** Same sweep, column 6: −0.235σ at narrow, −0.223σ at lit, crossing
under the gate only at ≈2× lit width. §2 reports ONLY the A_p bias and asserts "σ(n_s) ≈ flat" (line
45) — true for the VARIANCE, silent on the n_s BIAS, which is LARGER than A_p's and violates the same
gate the spec invokes. **Effect on conclusions:** the locked claim "keeps bias < gate" (§2 line 48) is
FALSE for n_s. This does NOT change the width as a variance choice, but it removes the spec's stated
justification for the lock and means the gate is currently FAILED on n_s. The cosmology lens notes the
single fiducial's n_s is ON the box edge (n_s=0.803), so the n_s number itself needs an interior
re-check — which is why the rerun must add an interior-n_s fiducial.

**E-C (CS, lines 138-161, "E1") — the §1/§3 model cannot be implemented at the site it names.**
`_legb_model` (`closure_legb.py:450-456`) builds ONE `alpha_hcd` of shape `(n_zg,3)` on the global
z-grid; `_data_loglik_legcore` re-slices it per leg by nearest-z (line 403). There is no per-leg α
path. The per-leg `δ_{c,leg}` (§3) forces moving α(z) assembly INTO `_data_loglik_legcore`'s per-leg
loop (CS route A) — a ~40-60 line signature/contract change, NOT the "1-line multiply" the earlier
onboarding synthesis implied. **Effect on conclusions:** does not change the width/×1.04/binding
science, but it makes the **golden guard a hard gate** (confirmed ABSENT — `tests/golden/` does not
exist, CS F1) and re-sequences the implementation. CS E2 is the matching truth-packing fix
(`make_truth_from_sim` must return `w_c_pivot` at z≈3 to rank `a_pivot`, and the new `s`/`δ` sites
have NO sim truth → their coverage is a degenerate null; report θ + a_pivot coverage only).

**Not errors but provisional-flags (carry into the spec):**
- The locked width is **provisional on the correlated-in-k C_emu** (Cosmology finding 1-2, Bayesian
  Q1 option (b)). The slope nuisance + diagonal C_emu leaves a coherent low-k residual on n_s; the
  correlated C_emu is the real whitener. The Fisher used the still-k-diagonal C_emu
  (sweep:124-127), so the n_s bias must be re-measured WITH the correlated C_emu before the final
  cert. The slope width and the correlated C_emu are two tools for the SAME mode and must be co-designed.
- `cemu_inflate` defaults to 1.0 (uncalibrated); if >1, σ(A_p) grows and bias/σ shrinks — so the
  locked bias/σ is conservative on that one axis (Cosmology finding 3).

The width-LOCK math itself is correct: all four checked the `(F+P)⁻¹`, the slope-block prior
injection (sweep:148-158), the per-class decomposition, and the SPD/endpoint handling — no arithmetic
errors (Bayesian 211-215, CS 188-191, Lyα 214-220). The LLS-as-binding-class call (tightest amp prior
0.15 + biggest gap +0.95) is correct and confirmed live by Lyα.

---

## Changes to fold into the spec before coding

1. **Replace the sim-`g_fixed` framing with the production lit-derived `g_fixed` as the LOCKED
   framing** (Q1, E-A). `g_fixed,c(z) = w_c_from_mu(dN/dX_lit · X̄(z))` normalized to z=3, a frozen
   `(n_z_leg,3)` constant on `LegBCtx`, computed once host-side. Document the X̄(z) source (sim
   `snap_total_path_dX/n_skewers`) and add the warning: g_fixed MUST go through `w_c_from_mu`, never
   dN/dX alone (Lyα A2).

2. **Re-run `diag_legb_slope_prior_tradeoff.py` on the production `g_fixed`, with the §0c DLA mask,
   computing the EXACT non-linearized bias for BOTH A_p AND n_s, at ≥1 interior-n_s fiducial** (Q3,
   E-A, E-B). Replace the LOCKED §6 numbers with the production-framing results. This single rerun
   discharges Q1's "re-measure," Q3's "exact," the n_s omission, the §0c overstatement, and the
   box-edge fiducial at once. Gate: `bias_exact < 0.2σ` on both parameters.

3. **Add the n_s bias to §2 and correct the safety logic** (E-B, Cosmology rec 1, 5): report the n_s
   bias, state plainly the lit width currently fails the gate on n_s in the closure-only framing, and
   reframe "free headroom to go wider" as the *recommended bias-reducing direction* (the bias is
   width-INSENSITIVE — tightening makes it worse).

4. **State that the locked width is PROVISIONAL on the correlated-in-k C_emu (§5)** and that the n_s
   bias must be re-measured WITH it (Cosmology rec 6, Bayesian Q1(b)). Co-design, not two steps.

5. **Set `s_c ~ Normal(0, σ_s)`, FIXED center at 0, no hyper-prior** (Q2). Width = WLS lit
   0.52/0.53/0.33 with z-edge inflation. Report the in-window (z=2.5–3.5) WLS center as a check
   (Lyα rec 6).

6. **Rewrite §1/§3 to put α(z) assembly in `_data_loglik_legcore`'s per-leg loop (CS route A), and
   state it is a ~40-60 line signature change, NOT a 1-line multiply** (E-C). Sample `a_pivot (3,)`,
   `s (3,)`, `δ_leg (3,)` per leg; build `alpha_leg = a_pivot·exp(δ_leg)·g_fixed_leg·exp(lnz_leg·s)`.
   Update `make_truth_from_sim` to return `w_c_pivot` at z≈3 and the `_draws_matrix` truth packing
   (CS E2).

7. **Reparametrize NON-CENTERED in log-space** (Bayesian F1, rec 5 — HIGH): sample `δ_leg, s` as
   standard normals and reconstruct `log α_c(z) = log α_pivot,c + σ_anchor·δ̃ + lnz·(0 + σ_s·s̃)`, so
   the amplitude×slope product becomes a SUM of Gaussians and the funnel on the ~28-30 dim posterior
   disappears. The current code builds α multiplicatively (`closure_legb.py:453-454`); this is a small
   differentiable change and the single highest-leverage geometry fix. Put it in the plan, not at
   NUTS time.

8. **Specify σ_anchor as a log-offset or a fractional width, and enforce `σ_anchor < σ_α` per class**
   (Bayesian F2): LLS σ_α≈0.15 vs σ_anchor 0.25–0.30 looks inverted; resolve the units. Keep the LLS
   τ≥1-vs-τ≥2 definitional offset in the amplitude CENTER, not `δ_leg` (Lyα rec 5).

9. **Make the edge-inflation a prior-builder change with its own unit test** (CS E3): extending the
   DLA-only one-sided `dla_inflate` (`inference.py:104`) to two-sided LLS/subDLA σ AND σ_anchor touches
   `hcd_incidence_prior`, not just the closure — unit-test (μ,σ) inside vs outside [2.5,3.5].

10. **State that s/δ coverage is NOT a valid null** (CS E2, Bayesian F3): the sim has no
    lit-correction slope, so s/δ truth = prior center by construction. Report s/δ coverage only as a
    *diagnostic* (railing = unbroken degeneracy); the PRIMARY gate stays A_p/n_s coverage.

---

## Sequenced implementation order this checkpoint endorses

This ordering is the intersection of CS recs 1-9, Bayesian recs 1-9, and the spec §5 prerequisites.
Gate conditions in **bold**.

1. **Golden guard FIRST** (CS rec 1, F1; Bayesian rec 9; Lyα A3, rec 8). `tests/golden/` is confirmed
   ABSENT. Freeze `(P_model, C_total)` on BOTH legs at the test fixture's fixed `(θ9, τ₀, α=(3,))`
   with production `sigma_zb/rho_zb` to `tests/golden/legb_lf_golden.npz`; add
   `test_legb_golden_unchanged` asserting `allclose(rtol=1e-12, atol=0)`. **Hard gate: nothing below
   lands until this exists** — E-C makes the refactor a real contract change, so the legacy (3,)-path
   must be pinned byte-for-byte first.

2. **The forward-only rerun (Q1+Q3+E-A+E-B+§0c), BEFORE any model code.** Re-run the sweep on the
   production lit-derived `g_fixed`, §0c-masked, exact non-linearized bias for A_p AND n_s, at ≥1
   interior-n_s fiducial. **Gate: if exact bias < 0.2σ on both, lock the width. If 0.18–0.25σ, the
   width is NOT free — WIDEN (do not narrow), or escalate to require the correlated C_emu.** This is
   the gating measurement for the whole spec and is independent of the refactor (it reuses the
   existing Fisher harness), so it can run in parallel with step 1.

3. **§0c DLA mask** (spec §5, Lyα A1, rec 4). Sequenced BEFORE finalizing the width because the
   rerun in step 2 depends on the masked truth (`make_truth_from_sim` currently builds full-w_c
   including DLA, `closure_legb.py:273-278`). The DLA-MOOT claim is correct only once this lands.

4. **The per-leg α refactor (CS route A) + non-centered log-space reparam (Bayesian F1).** Move α(z)
   assembly into `_data_loglik_legcore`'s per-leg loop; store `g_fixed` per leg on `LegBCtx`; sample
   `a_pivot/s/δ_leg`; reconstruct in log-space. Update truth packing (CS E2). **Gate: golden guard
   (step 1) green + the CS unit tests** — `test_alpha_factorized_shape`,
   `test_alpha_factorized_finite_and_grad`, `test_alpha_s0_delta0_equals_g_fixed` (rtol 1e-12),
   `test_alpha_g_fixed_precomputed_no_trace`, and all existing (3,)-broadcast tests UNCHANGED (CS rec 9).

5. **Prior-builder edge-inflation extension** (CS E3) + the **σ_anchor inequality check / mini-sweep**
   (Bayesian F2, Cosmology Q4 sub-point) + the **in-window WLS center check** (Lyα rec 6). These are
   small and can fold into step 4's PR.

6. **Re-profile ESS/sample on ONE mock with the new ~28-30 dim latents BEFORE choosing N** (Bayesian
   F4). **Gate: do not assume the LF cert's ESS≈0.18 transfers** to the larger posterior; the
   non-centered reparam should help but must be measured against the ~4000 CPU-h budget.

7. **NUTS Leg-B closure** over multiple held-out sims (Lyα A5 — the gap is sim-dependent; certify over
   the folds, not the single fiducial). **Gate: coverage ≥ nominal AND |bias| < 0.2σ on BOTH A_p and
   n_s** (Cosmology rec 8 — carry the n_s bias through the gate, not A_p alone); report s/δ coverage
   as a diagnostic only (step-10 change). The single-fiducial +0.03σ sim-`g_fixed` run is retained as
   a *sampler sanity check*, not the verdict (Lyα rec 7).

**Provisional flag spanning steps 2 and 7:** the locked width is conditional on the correlated-in-k
C_emu (§5, out of scope here). If step 2's n_s bias stays over the gate with a diagonal C_emu, the
final cert (step 7) must wait for the correlated C_emu and re-measure — the slope width alone may not
close the n_s gate.

---

## Open items that remain the PI's decision

The lenses resolved Q1, Q2, Q3-method, and Q4 with no material dissent. What genuinely needs the PI:

1. **The gate-vs-widen-vs-C_emu call, AFTER step-2's rerun.** The method is decided (compute the exact
   bias). The DECISION the number forces is the PI's: if the exact production-framing n_s (and/or A_p)
   bias lands in 0.18–0.25σ, do we (a) widen the slope prior (cheap, ×1.04 variance, but
   width-insensitive so limited leverage), or (b) block the cert on building the correlated-in-k C_emu
   (the real whitener for the coherent mode, but more work). All four lenses point at this fork;
   none can resolve it without the number. **This is the load-bearing PI call.**

2. **Acceptance of the conditional coverage claim** (Cosmology line 50-54, Lyα A5). The closure truth
   is held-out SIMS, whose w_c(z) gap to the lit `g_fixed` is a proxy for the *real-data*-vs-lit gap,
   which is unknowable. The PI must accept that the certified coverage is "conditional on
   sim-vs-lit gap ≈ real-vs-lit gap" — the best available test, but a stated assumption, not a proof.

3. **The σ_anchor magnitudes (KS 0.25–0.30, DESI 0.10–0.15).** The lenses agree these are un-measured,
   are doing real work on the A_p bias, and should get a mini-sweep + the `σ_anchor < σ_α` check
   (item 8 above). But the PI owns the physical prior on "how much can KS's absorber-targeted selection
   bias incidence high" — Lyα calls 0.25–0.30 "borderline-generous but acceptable." The final numbers
   are the PI's to set, informed by the mini-sweep.

Everything else — production `g_fixed`, center=0/fixed/no-hyperprior, compute exact bias, amplitude-only
float, non-centered log-space reparam, golden guard first — is a cross-lens technical resolution and
does not need the PI to adjudicate, only to approve the plan.
