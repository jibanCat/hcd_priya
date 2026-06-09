# TASK A — the τ₀–n_s correlation-rotation flag: coverage ruling

**Author:** Bayesian/PPL statistician (certification lens). **Date:** 2026-06-09.
**Branch:** `phase2c-likelihood`. **Object:** rule on the T3-gate flag that the τ₀–n_s
posterior correlation rotates ~0.13 (|r(ON)−r(LF)| worst = 0.129) under the MF correction —
the one threshold the gate exceeded (G4.2 literal). The gate already established the rotation
is from the SEPARABLE correction (|r(OFF−LF)| ≈ |r(ON−LF)| every fold), NOT the rank-1 term
(|r(ON−OFF)| worst 0.014 ≤ 0.05). This note quantifies the σ-impact and rules on coverage.

**Inputs:** the gate `emu_bias_allfolds_mf.{txt,npz}` + `..._mf_aliasing.txt`, and a focused
re-run `scripts/diag_mf_tau0ns_rotation.py` → `figures/analysis/04_emulator/mf_tau0ns_rotation.{txt,npz,png}`
(the SAME Fisher C_post = (F+P)⁻¹ the gate built, under LF / MF-OFF / MF-ON, per fold).

---

## 1. The numbers (8 folds, one representative held-out sim each)

| quantity | mean | range over folds |
|---|---|---|
| σ(n_s) ratio **ON/LF** | **0.939** | [0.920, 0.955] |
| σ(n_s) ratio OFF/LF | 0.938 | [0.920, 0.956] |
| σ(n_s) ratio **ON/OFF** | **1.0004** | [0.998, 1.002] |
| σ(τ₀) ratio **ON/LF** | **0.975** | [0.953, 0.999] |
| σ(τ₀) ratio OFF/LF | 0.981 | [0.960, 1.004] |
| σ(τ₀) ratio **ON/OFF** | 0.993 | [0.984, 0.998] |
| Δr(ON−LF) | −0.033 | worst\|·\| 0.107 |
| Δr(OFF−LF) | −0.028 | worst\|·\| 0.100 |
| Δr(ON−OFF) | −0.005 | worst\|·\| **0.014** |

(τ₀-block = the z∈[2.4,3.4] data-constraining diagonal; r reported at the worst band z-bin.)

## 2. What the rotation IS: a small TIGHTENING, not a widening

The headline answer to the spec's first question — **is the 0.13 correlation shift a WIDENING
(conservative) or a TIGHTENING (could under-cover)?**

- It is a **TIGHTENING**: σ(n_s) shrinks to **~0.94× LF (≈ −6%)** and σ(τ₀) to **~0.975× LF
  (≈ −2.5%)**, in EVERY fold (the σ(n_s) ON/LF ratio is ≤ 0.955 in all 8 folds; it never
  exceeds 1.0). So the MF correction does NOT widen the marginals; it sharpens them.
- This is the direction the spec flagged as the one to watch ("could under-cover"). It is NOT,
  however, a *rotation-induced* under-coverage; it is a Fisher-information increase (see §3).
- **It is essentially ALL the SEPARABLE correction.** The rank-1 increment is negligible:
  σ(n_s) ON/OFF = 1.0004 (the rank-1 term changes σ(n_s) by <0.05%); σ(τ₀) ON/OFF = 0.993;
  Δr(ON−OFF) worst = 0.014. This confirms the gate's attribution: the geometry change is the
  separable (gbar_z + gbar_tau + log_rho + res_corr) part, present in the SAFE-FLOOR pure-separable
  arm too, and the rank-1 term — the only piece that *could* alias — does essentially nothing to
  the joint geometry. **The rank-1 term is clean** (consistent with G4.1 Δn_s = −0.0003σ and
  G4.3 ∂g/∂n_s ≡ 0).

## 3. Why the tightening is LEGITIMATE information, not spurious under-coverage

The mechanism: the MF correction adds the LF→HR high-k power (the +6% resolution correction +
res_corr) on top of f_LF. This **steepens the predicted P(k) response at high k**, which is where
the n_s tilt lever-arm lives, so the Fisher matrix gains information on n_s (and, through the
n_s–τ₀ degeneracy, modestly on τ₀). A sharper, more-correct forward → a tighter posterior. That
is the expected and *desirable* behaviour of a forward that now resolves the band it previously
under-predicted; it is the same reason adding KODIAQ high-k rows tightens n_s.

The danger the spec names — a tightening that **under-covers** — would arise only if the extra
information were built on an MF forward that is WRONG (i.e. the tightening is spurious because the
correction does not generalize). Three independent facts rule that out *for the closure*, and the
residual generalization risk is budgeted SEPARATELY:

1. **The closure n_s bias is in-gate and unchanged.** G1: pooled 60-sim n_s through MF = +0.0264σ
   (ON) / +0.0267σ (OFF), vs the LF reference +0.030σ — i.e. the tighter posterior is **correctly
   centered**. Re-expressed in LF-posterior σ units the bias is +0.0248σ — still negligible. A
   tightening that under-covered would show up as an n_s bias that GROWS when measured in the
   tighter σ units; it does not. **The closure is the arbiter, and it passes.**
2. **The rotation is θ-blind by construction.** ∂g/∂n_s ≡ 0 (G4.3, exact). The correction cannot
   inject an n_s-direction tilt; it only reweights which (z,τ₀,k) cells the Fisher trusts. A
   θ-blind reweighting applied identically to all 60 closure sims cannot bias the closure n_s
   (G4.1, Δn_s = −0.0003σ) — verified.
3. **The LF→HR generalization residual that the tightening *could* over-trust is covered by the
   TASK-B C_emu floor**, which INFLATES C_emu on the small-scale leg (sigma_floor ≈ 1.23% LF-resolvable;
   larger extrapolated) and thus WIDENS the posterior back where the correction is least reliable —
   AND by the n_s-edge term for ns∉[0.86,0.98]. The floor is the conservative counter-weight to the
   information the MF correction adds; together they are coverage-consistent.

So: the tightening is real Fisher information from a more-correct forward (closure-certified
center), and the part of it that rests on an under-tested correction is re-widened by the TASK-B
floor. The 0.13 r-rotation is a *consequence* of the same response-reshaping, not an independent
pathology.

## 4. Is it captured by the through-MF Fisher? Is the closure the arbiter?

- **Captured: YES.** The gate's F-matrix is built from the **MF** forward Jacobian (J = ∂P_MF/∂θ),
  so the reported σ(n_s), σ(τ₀) and r **already include the rotated/tightened geometry**. Coverage
  within the Fisher scout is self-consistent — there is no separate "un-rotated" Fisher that the
  pipeline silently uses. No additional Fisher treatment is needed.
- **The arbiter is the closure (Leg-B NUTS), not the Fisher.** The Fisher is a linear-Gaussian
  scout; it cannot see non-Gaussian τ₀–n_s degeneracy curvature or prior-edge effects. The actual
  coverage statement comes from the SBC / closure NUTS run, which samples the FULL MF posterior
  (rotated geometry included) and is checked against the injected truth. The Fisher tightening (−6%)
  is a forecast; the closure +0.026σ bias (already through the MF forward, per the gate's
  construction) is the measurement. **The closure is the binding check and it is in-gate.**

## 5. RECOMMENDATION — carry-as-coverage-note (NO C_emu term, NO action beyond the TASK-B floor)

**Ruling: carry-as-coverage-note.** Do NOT add a dedicated C_emu term for the rotation, and do NOT
fall back from the separable+rank-1 form.

Reasons:
1. **It is not a bias.** The closure n_s is +0.026σ ON and +0.027σ OFF — the rotation does not move
   the n_s error bar's CENTER materially, and the marginal σ(n_s) change is a *tightening* the
   closure certifies as correctly-centered, not an inflation that needs covering.
2. **The rank-1 term is exonerated.** The rotation is the separable correction (ON/OFF σ-ratio
   1.0004; Δr(ON−OFF) 0.014 ≤ 0.05). Pure-separable cannot alias (no z×τ₀ cross term), so the
   geometry change is a benign property of applying the (physically-motivated) resolution + res_corr
   tilt, not an information-trading aliasing pathology. Dropping the rank-1 term would NOT remove the
   rotation (it lives in OFF too) and would discard the validated high-z sign-flip representation.
3. **A C_emu term would be the WRONG instrument.** C_emu inflates the data covariance to cover
   *forward-model error*. The rotation is not forward-model error — it is a correct change in the
   forward's information content. Inflating C_emu to "undo" a legitimate tightening would *discard
   real information* and is not justified by any measured bias. The only generalization risk that
   warrants C_emu inflation (the LF→HR residual the tightened posterior could over-trust) is ALREADY
   handled by the TASK-B sigma_floor + sigma_edge on the small-scale leg.
4. **Coverage is self-consistent through the MF Fisher AND the closure.** Both already carry the
   rotated geometry; no separate treatment is needed.

### The coverage note to carry (for the blinding / real-fit report)
> The MF correction tightens the marginal σ(n_s) by ≈6% and σ(τ₀) by ≈2.5% relative to the LF-only
> forward, and rotates the τ₀–n_s posterior correlation by ≈0.13 (worst fold). This is a Fisher-
> information gain from a forward that now resolves the high-k band (correctly centered: closure
> n_s +0.026σ, in-gate), driven by the SEPARABLE resolution+res_corr correction; the rank-1
> interaction contributes negligibly (σ ON/OFF = 1.0004, Δr(ON−OFF) = 0.014). The information the
> tightening adds where the MF correction is least tested (LF→HR generalization, ns-edge) is
> re-widened by the TASK-B small-scale C_emu floor (sigma_floor + sigma_edge). No dedicated rotation
> C_emu term is warranted. The binding coverage check is the Leg-B closure (NUTS / SBC), which
> samples the full rotated MF posterior; the Fisher tightening is a scout-level forecast.

---

## 6. What n=6 (HR) does and does not license here
- **Captured/licensed:** the σ-change and r-rotation are measured on all 60 LF-truth sims through
  the MF Fisher (the 6-HR limit only constrains the θ-blind correction, applied identically). The
  rank-1 exoneration (ON/OFF) is exact-by-construction + measured. CAN rule the rotation is a
  closure-certified tightening, not a bias.
- **NOT licensed here:** that the tightened MF posterior generalizes to the real fit's n_s (eBOSS
  ~1.009, above the HR ceiling 0.979). The tightening at ns≈1.0 rests on the MF correction's
  EXTRAPOLATION; that residual is the TASK-B n_s-edge term (sigma_edge), and ns>0.98 posterior is
  GUARDED, not nominal. This note rules on the rotation as a *closure* coverage property; the
  real-fit edge is the separate TASK-B object.

**Files:** `scripts/diag_mf_tau0ns_rotation.py`,
`figures/analysis/04_emulator/mf_tau0ns_rotation.{txt,npz,png}`. (Diag-only; not committed.)
