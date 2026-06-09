# T3 — the through-MF n_s-Fisher adoption gate (HARD) + aliasing detector

**Author:** Bayesian/PPL statistician (certification lens). **Date:** 2026-06-08.
**Branch:** `phase2c-likelihood`. **Object:** the HARD adoption gate for the CS agent's
separable + rank-1 MF correction `g(z,τ₀,k) = gbar_z(z,k) + gbar_tau(τ₀,k) + a(k)·u_z(z)·u_τ(τ₀)`.

This spec defines (1) HOW to re-run the all-folds n_s/A_p Fisher MAP-shift **through
`MultiFidelity.logP_mf`** (not the raw cache); (2) the EXACT pass criterion; (3) the
**n_s-tilt → τ₀-axis ALIASING detector** that decides whether the rank-1 interaction term is
admitted. **No code is committed by this spec** — it is the contract the CS agent's
`scripts/diag_emu_bias_allfolds_mf.py` must satisfy before the separable+rank-1 form is adopted.

Anchors (committed, verified): all-folds z_lo=2.0 n_s = **−0.646σ** (`emu_bias_allfolds.txt`);
all-folds z_lo=2.6 = **+0.038σ**; shipped z_lo=2.4 60-sim = **see C1**
(`emu_bias_allfolds_zlo24.txt`). MF LOSO worst-per-sim coherent low-z high-k residual **+0.91%**
(`mf_rescorr_loso.txt`). τ₀ swing of ρ ~9.1pp, z swing ~12.2pp at k~0.05 (`mf_rescorr_vs_tau0_z.txt`).

---

## 0. Why this gate exists (the failure mode it guards)

The retired `delta_mode='mlp'` head over-fit the LF→HR θ-gradient on 6 clustered HR sims and produced
a **4σ n_s Fisher bias** (`multifidelity.py:316-318`, the documented reason `'none'` is the default).
The new form adds ONE rank-1 (z×τ₀) interaction to capture a REAL (z,τ₀) sign-flip in ρ that the
pure-separable form cannot represent. The danger is identical in kind to the mlp head: with only **6
HR sims spanning ns∈[0.859,0.979]** (clustered, max NN gap 0.058), a (z,τ₀) interaction term has just
enough freedom to **absorb the n_s tilt into the τ₀ axis** (because n_s and τ₀ are ~−0.66 correlated
through the LF emulator's two-stage head and the HR sims' n_s and τ₀ are not orthogonalized). If it
does, the MF "correction" silently re-injects an n_s bias that the closure cannot see (the rank-1 term
is θ-independent by construction, so it cannot bias the *closure*; it biases the *real fit* by
mis-shaping the (z,τ₀) response that multiplies f_LF). **The gate must therefore test not just the net
n_s bias but whether the rank-1 term is doing n_s-tilt work.**

---

## 1. The through-MF Fisher run (the gate's measurement)

### 1.1 Construction
Clone `scripts/diag_emu_bias_allfolds.py` → `scripts/diag_emu_bias_allfolds_mf.py`. Keep the
machinery byte-identical EXCEPT swap the forward `fwd(th, t0g, apv)` so the per-leg P prediction
goes through `MultiFidelity.logP_mf` instead of the raw LF cache path
`DL.predict_P_obs_on_leg(... LF ...)`. Concretely:

- Build a `MultiFidelity` per fold from that fold's **held-out** LF backbone
  (`load_lf_backbone(fold)`) AND the **HF-LOSO** correction: the `FixedMeanHead`/rank-1 head must be
  fit on the 5 HR sims **excluding** the held-out fold's HR sim **if** the held-out sim is one of the
  6 HR sims (otherwise all 6 HR sims train the head — but then state that this fold's MF correction is
  in-sample w.r.t. the HR set; only the LF backbone is honestly held out). Because only 6/60 sims have
  HR truth, **most folds will have an in-HR-sample correction** — this is unavoidable and must be
  reported per-fold, not hidden. The honest LF→HR generalization number lives in T4's 6-fold HR-LOSO,
  not here; THIS gate measures the n_s response of the *production* MF forward.
- Evaluate `logP_mf` on the leg k-grid via the eval-grid interp the production forward uses (the LF
  tail-extrapolation above k=0.069 s/km is IN the forward and MUST be exercised — KS reaches into the
  extrapolated band even after the k<0.06 analysis cap, because the cap is applied to the *data* rows
  not the emulator's internal grid; verify the kept KS rows' k against 0.069).
- Keep KS at the **shipped z_lo=2.4** and the locked **k<0.06** cap. Keep the DLA mask, the per-leg
  C_emu (`predict_P_obs_on_leg`'s `Ctot`), the Jacobian wrt (theta9, τ₀, a_pivot(3)), and the
  (F+P)⁻¹JᵀC⁻¹ΔP MAP-shift identical. ΔP = P_truth_masked − P_MF_forward (pure MF emulator error).
- **z-resolved output:** accumulate the per-z gradient block `g_z = Σ_{rows∈z} JᵀC⁻¹ΔP` exactly as
  `nsbias_z_attribution.npz` does, and report `bias_z = (g_z·C_post[:,ns])/σ_ns` per z-bin. Σ_z bias_z
  must equal the pooled n_s bias to <1e-6 (the exactness gate — this also validates the run did not
  silently drop rows).

### 1.2 Write to a NEW file
`figures/analysis/04_emulator/emu_bias_allfolds_mf.txt` (+ `..._mf.npz` with `bias_z` arrays). **Do NOT
overwrite** `emu_bias_allfolds.txt` (z_lo=2.0 LF reference) or `emu_bias_allfolds_zlo24.txt` (C1, LF
shipped). The MF number must sit NEXT TO the LF numbers so the reduction is auditable.

---

## 2. The PASS criterion (HARD gate — all four clauses must hold)

The form is **adopted** iff ALL of:

| # | Clause | Threshold | Why |
|---|--------|-----------|-----|
| G1 | pooled 60-sim **n_s** bias through MF | **\|bias\| ≤ 0.2σ** | net n_s unbiased through the production forward |
| G2 | pooled 60-sim **A_p** bias through MF | **\|bias\| ≤ 0.2σ** | A_p is the priority param; must not regress |
| G3 | z=2.8–3.4 **high-k** residual REDUCED vs the LF-only forward | the per-z signed coherent logP residual at k≥0.0442 s/km in z∈[2.8,3.4] must have **smaller \|mean\|** under MF than under LF | the MF correction's job is to *reduce* the high-k deficit, not just rebalance the bias |
| G4 | **no aliasing** (§3) | the rank-1 ON-vs-OFF n_s delta ≤ 0.1σ AND the correction's n_s-derivative ≈ 0 (§3.3) AND no τ₀-width inflation (§3.2) | the rank-1 term must capture the (z,τ₀) sign-flip WITHOUT doing n_s-tilt work |

**If G1 or G2 fails:** STOP. Do not adopt. The form leaves a coherent cosmology bias.
**If G3 fails but G1/G2 pass:** the correction is null/cosmetic in the target band — revisit whether
the z+τ₀ resolution actually helps (the whole point was the z=2.8–3.4 high-k deficit); do not ship as
"the fix" until G3 holds.
**If G4 fails (aliasing detected):** STOP and DROP the rank-1 term — fall back to **pure separable**
`gbar_z + gbar_tau` (no interaction). Per the plan, the fallback is "revisit the interaction rank,
never a free 2-D table." Pure separable cannot alias (it has no (z×τ₀) cross term), so it is the safe
floor even though it under-represents the high-z sign-flip; the residual sign-flip is then carried by
the T4 C_emu floor instead.

**Reference for "in-gate":** the LF shipped z_lo=2.4 number (C1) is the n_s the closure already
certifies WITHOUT MF. The MF n_s must NOT be WORSE than that — i.e. the gate is not just \|·\|≤0.2σ in
absolute terms but **also** must not regress relative to C1 by more than the 6-sim noise (~0.05σ).
State both: the absolute gate (±0.2σ) and the no-regression-vs-C1 check.

---

## 3. The ALIASING DETECTOR (the part this gate adds beyond a net-bias check)

A net n_s bias inside ±0.2σ is **necessary but NOT sufficient**: the rank-1 term could be doing
n_s-tilt work that *cancels* against the LF backbone's own residual in the pooled mean while still
mis-shaping individual (z,τ₀) cells (and therefore mis-covering the real fit, whose n_s is a single
point, not a 60-sim average). Three orthogonal probes — run all three:

### 3.1 Rank-1 ON vs OFF (the primary aliasing probe)
Run the §1 Fisher gate TWICE, with the SAME fold emulators and SAME HF-LOSO head fit:
- **ON:**  `g = gbar_z + gbar_tau + a(k)·u_z·u_τ` (full form).
- **OFF:** `g = gbar_z + gbar_tau` (zero out the rank-1 term: set `a(k)≡0`).

Report `Δn_s = bias_ns(ON) − bias_ns(OFF)` and `Δn_s,z` per z-bin.
- **PASS:** `|Δn_s| ≤ 0.1σ`. The rank-1 term changes the n_s bias by less than half the gate width →
  it is doing (z,τ₀)-shape work, not n_s-tilt work.
- **FAIL:** `|Δn_s| > 0.1σ`, especially if `Δn_s` is concentrated in the high-z bins (z≥4) where the
  sign-flip lives — that is the signature of the interaction term absorbing the n_s tilt. If the
  rank-1 term *improves* A_p (G2) but its n_s-delta is large and z-localized, that is aliasing dressed
  as improvement.
RATIONALE: the rank-1 term exists ONLY to represent the (z,τ₀) sign-flip (a θ-independent shape). A
θ-independent shape, applied identically to all 60 closure sims, **cannot** change the *closure*
n_s bias by construction — so a large `Δn_s` can ONLY come from the term mis-shaping the (z,τ₀)
response in a way that interacts with the Fisher Jacobian's n_s direction. That is precisely aliasing.

### 3.2 τ₀-posterior-width inflation
The aliasing route is: the rank-1 term soaks up n_s information into the τ₀ axis, which the Fisher then
must "pay back" by widening the τ₀ posterior (or shifting the τ₀–n_s degeneracy direction). Compute,
per fold, the marginal τ₀-posterior width `σ_τ0 = sqrt(diag(C_post)[τ₀])` and the τ₀–n_s correlation
`r = C_post[τ₀,ns]/sqrt(C_post[τ₀,τ₀]C_post[ns,ns])` under MF-ON, MF-OFF, and LF.
- **PASS:** `σ_τ0(ON)/σ_τ0(OFF) ≤ 1.05` (≤5% inflation) AND `|r(ON) − r(LF)| ≤ 0.05`. The interaction
  term does not inflate the τ₀ posterior or rotate the degeneracy.
- **FAIL:** τ₀ width inflates >5% or the τ₀–n_s correlation rotates by >0.05 — the term is trading
  n_s information into τ₀ (the classic aliasing signature: a parameter that should be data-constrained
  instead absorbs an unmodeled correction).
NOTE: `C_post` already marginalizes τ₀ (it is the full information matrix), so this reads off the
joint geometry directly — no extra sampling needed.

### 3.3 The correction's n_s-derivative ≈ 0 (the cleanest probe)
The separable+rank-1 `g(z,τ₀,k)` is **θ-independent BY CONSTRUCTION** — it reads only `cond[9]=z_unit`
and `cond[10]=τ₀`, never the 9 cosmology params. Therefore `∂g/∂n_s` should be **identically zero**
(machine epsilon). Verify directly with `jax.grad`:
```
gfun = lambda th: mf.g(jnp.concatenate([th, z_unit_arr]), tau0)   # g as fn of theta9
dgn = jax.grad(lambda th: gfun(th).sum())(theta9)[NS_I]
assert abs(dgn) < 1e-10   # g must NOT depend on n_s
```
- **PASS:** `|∂g/∂n_s| < 1e-10` at a grid of (z,τ₀) eval points. This is the *structural* guarantee
  that the head cannot do n_s-tilt work — if the CS agent accidentally wired the head to read a
  cosmology param (or τ₀ became a function of n_s upstream), this catches it immediately.
- **FAIL:** any nonzero `∂g/∂n_s` means the correction is θ-dependent — that is the retired mlp
  failure mode and an instant NO_GO for the form as built.
CAVEAT: this probe certifies the *head* is θ-blind, but the n_s bias can STILL move (3.1) because
`logP_mf = logP_LF + g + log res_corr` — the n_s response flows through `logP_LF` (f_LF), and `g`
re-weights *which* (z,τ₀) cells the Fisher trusts. So 3.3 is necessary (head θ-blindness) but 3.1/3.2
are what test whether that θ-blind reweighting nonetheless aliases through the Jacobian. **Run all
three.**

### 3.4 Reading the three together
- 3.3 PASS + 3.1 PASS + 3.2 PASS → clean: adopt the rank-1 form.
- 3.3 PASS but 3.1 FAIL → the head is θ-blind but its (z,τ₀) reshaping aliases the tilt through the
  Fisher geometry → the rank-1 *direction* is the problem; try constraining `u_z(z)` to the measured
  ρ-sign-flip principal vector (§4 of the form-safety memo) rather than a free SVD direction, OR drop
  to pure separable.
- 3.3 FAIL → the head is not θ-blind: a wiring bug or the form was changed; reject outright.

---

## 4. What n=6 (HR) can and cannot certify here — explicit

- **CAN:** that the *production* MF forward's net n_s/A_p Fisher bias over all 60 LF-truth sims is
  in-gate (G1/G2) — because the Fisher run uses all 60 LF sims; the 6-HR limit only constrains the
  *correction*, which is θ-independent and applied identically. CAN: that the head is structurally
  θ-blind (3.3, exact). CAN: that the rank-1 term is not aliasing *within the closure* (3.1/3.2,
  measured on 60 sims).
- **CANNOT:** that the MF correction GENERALIZES to the real fit's n_s (eBOSS ~1.009, above the HR
  ceiling 0.979). The 6 HR sims are clustered in ns∈[0.859,0.979]; the rank-1 `u_τ(τ₀)`,`u_z(z)`
  directions are fit on those 6 — outside that box the interaction is extrapolated. The gate certifies
  the form does not alias *on the closure*; it does NOT certify the (z,τ₀) interaction is correct at
  ns≈1.0. That residual is the T4 n_s-edge MF-extrapolation budget (separate object). State in the run
  output: "G1–G4 certify no-aliasing within the closure + HR cluster; ns>0.98 generalization is the T4
  edge budget, NOT certified here."
- **CANNOT:** distinguish a true (z,τ₀) physical sign-flip from a 6-sim noise fluctuation in the
  rank-1 direction. Mitigate by requiring the rank-1 singular value to exceed the per-sim ρ noise
  (CoV ~0.6–1.3%, `mf_rescorr_per_class.txt`) — if `a(k)` is below the noise, drop it (it is fitting
  noise, and a noise-direction interaction is the most dangerous aliasing vector).

---

## 5. Compute / discipline
Forward-only Fisher (no NUTS, no LF retrain). Per fold: one `MultiFidelity` build + 60-sim Jacobians,
×2 (ON/OFF) ×8 folds ≈ the cost of `diag_emu_bias_allfolds.py` doubled (~minutes–tens of minutes on
CPU). Profile before any NUTS at scale (cavestru0 ~4000 CPU-h budget). The CS agent owns
`diag_emu_bias_allfolds_mf.py`; this spec is the acceptance contract.
