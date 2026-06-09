# T4 — the small-scale-leg C_emu floor estimator (LF→HR generalization) + n_s-edge MF-extrapolation budget

**Author:** Bayesian/PPL statistician (certification lens). **Date:** 2026-06-08.
**Branch:** `phase2c-likelihood`. **Object:** the spec for the C_emu floor term the CS agent adds to
the small-scale-leg covariance assembly in `data_likelihood.py` (consumed by `inference.py`'s
`emu_var`), covering (A) the LF→HR generalization residual of the MF correction, and (B) the
**separately-budgeted** n_s-edge MF-extrapolation inflation. This is design item 3 + item 4 of the MF
plan and I2 + I3 of the meta-review.

Anchors (committed, verified — `mf_rescorr_loso.txt`, `mf_rescorr_vs_tau0_z.txt`):
worst-per-sim coherent low-z high-k residual **+0.91%** (sim 3, ns=0.914, mid-cluster); pooled signed
mean **+0.01%** (cancels — NOT usable); per-sim spread [−0.70%, +0.91%]; per-sim RMS up to ~1.5%; the
LF→HR deficit **sign-flips with z** (ρ>1 low-z high-k → ρ<1 at z=5 high-τ₀). 6 HR sims, ns∈[0.859,0.979].

---

## 1. What this floor IS and IS NOT (scope — say it on every plot/output)

- **IS:** an extra DIAGONAL-in-k variance term on the **small-scale leg** (KS; and any high-k DESI
  rows above the LF Nyquist) that inflates C_emu to cover the residual the FIXED θ-independent MF
  correction (`g = gbar_z + gbar_tau + rank-1`) leaves **after** it is applied — i.e. the LF→HR
  *generalization* error: "given the MF correction trained on the HR set, how wrong is it on a held-out
  cosmology?"
- **IS NOT:** a model of HR→truth non-convergence. The 6 HR sims are themselves not converged to
  infinite resolution; that gap is covered by the **k<0.06 analysis cap** (the locked cap inside which
  HR is trustworthy per the PRIYA convergence memo). **State on the C_emu assembly and the real-fit
  plan: "this floor covers LF→HR generalization ONLY; HR→truth is covered by k<0.06."** Do NOT claim MF
  "resolves to truth."
- **IS NOT** the n_s-edge extrapolation budget (§4) — that is a SECOND, separately-budgeted term for
  ns outside the HR cluster. Keep them additive and labeled separately.

---

## 2. The floor estimator (LF→HR generalization, design item 3)

### 2.1 Measure THROUGH the production forward, z-resolved, per-band
6-fold HR-LOSO of the MF correction, through `MultiFidelity.logP_mf` (NOT the raw cache — the
production forward includes the LF tail-extrapolation above k=0.069 s/km, which is where the worst
residual lives). For each held-out HR sim `s` and each (z, k) the eval grid touches:

```
eps_s(z,k) = P_MF_pred(held-out s, with gbar/rank-1 fit on the OTHER 5 HR sims) / P_HR_true_s(z,k) − 1
```

This mirrors `mf_rescorr_loso.txt`'s `eps`, but evaluated through `logP_mf` (with the rank-1 head and
res_corr in the forward) rather than on raw cache ρ, so the tail-extrapolation error is captured.

### 2.2 The floor: WORST-per-sim COHERENT residual, z-resolved, per-band
For each (z-bin, k-band) cell, the floor's *standard deviation* is:

```
sigma_floor(z, band) = INFLATE × max_s | coherent_s(z, band) |

coherent_s(z, band) = mean_{k ∈ band} eps_s(z, k)          # the SIGNED mean over k in the band
                                                             #  (the n_s-bias-relevant coherent offset)
INFLATE = 1.35                                               # finite-sample inflation, n=6 (see §3)
```

KEY choices (all four cross-lens-agreed):
- **WORST per sim, never the pooled signed mean.** The pooled mean (+0.01%) cancels opposite-sign
  sims and would understate the per-cosmology residual ~90×. Use `max_s |coherent_s|`. (Verified: the
  worst sim is +0.91%, mid-cluster ns=0.914 — a genuine in-cluster generalization residual, not an
  edge artifact, so it is irreducible at this fidelity.)
- **COHERENT (band-mean over k), not RMS.** The n_s bias is driven by the *coherent* (sign-stable
  across k) logP offset — a tilt, not per-k scatter. The per-k RMS (~1.5%) over-counts incoherent
  noise that the cosmic_cov + the existing per-k C_emu already carry; the coherent band-mean is the
  mode that leaks into n_s. (If you also want a per-k incoherent floor, that is the *existing* σ_c(k)
  C_emu; do not double it.)
- **Z-RESOLVED, never pooled over z.** The deficit sign-flips with z (ρ>1 at low z, ρ<1 at z=5
  high-τ₀); a z-pooled floor cancels the flip and under-covers BOTH ends. Compute one
  `sigma_floor(z, band)` per z-bin the leg uses.
- **PER-BAND.** Report and apply separately for:
  - **LF-resolvable band:** k < 0.069 s/km (the LF Nyquist) — the MF correction interpolates here.
  - **Extrapolated band:** k ≥ 0.07 s/km (the KODIAQ reach above the LF Nyquist) — the LF backbone is
    tail-EXTRAPOLATED and the MF correction is least reliable; expect a larger floor here. (The k<0.06
    analysis cap means the *data* rows mostly sit in the LF-resolvable band, but the eval grid and the
    cosmic-variance leakage touch the extrapolated band, so size it.)

### 2.3 How it plugs into the C_emu assembly
The covariance is built per-z in `inference.py:predict_P_obs_and_cov_single_z`:
`emu_var(k) = Σ_c coef² σ_c(k,z,τ₀)² P_c² ` (diagonal) `× cemu_inflate`. Add the floor as an **additive
diagonal variance** on the small-scale leg, in ABSOLUTE P² units (the floor is a *fractional* logP
residual → multiply by P_obs²):

```
emu_var(k, z) += [ sigma_floor(z, band(k)) · P_obs(k) ]²
```

i.e. a fractional floor `sigma_floor` on the *total* P_obs (not per class — the LF→HR resolution
correction acts on the whole clean+excess forward, and the per-class spread is ≤1.1pp < the floor, so
a single P_obs-level floor is adequate per the per-class diag). Plumb `sigma_floor(z, band)` as a
small `(Nz, 2)` table (LF-resolvable / extrapolated) into `data_likelihood.py`'s leg assembly and pass
it down to `predict_P_obs_and_cov_single_z`. It is a FIXED, θ-independent, data-side table (like
res_corr) — it carries NO gradient and cannot bias the MAP; it only widens the posterior. **Apply it
ONLY to the small-scale leg(s)** (KS, and high-k DESI rows above the LF Nyquist); the low-k DESI leg
is below the resolution regime and must not be inflated.

### 2.4 Concrete formula (copy-ready)
```
# per z-bin, per band; band ∈ {LFres: k<0.069, extrap: k≥0.07}
coherent_s(z, band)  = nanmean_{k in band}( P_MF_pred_LOSO(s; z,k) / P_HR_true(s; z,k) - 1 )
sigma_floor(z, band) = 1.35 * max over the 6 HR sims of | coherent_s(z, band) |
# assembly (small-scale leg only):
emu_var(k, z) += ( sigma_floor(z, band(k)) * P_obs(k, z) )**2
```
Lower bound: never let `sigma_floor` go below the worst observed +0.91% × 1.35 ≈ **1.23%** in any
band where the LOSO is noise-limited (so a fluke-small per-sim coherent in one z-bin does not zero the
floor there). i.e. `sigma_floor(z, band) = max( 1.35·max_s|coherent_s|, 0.0123 )` on the small-scale
leg. (Document: 1.23% is the global worst-per-sim ×1.35; per-z values may exceed it where the sign-flip
is strongest, e.g. the z=5 high-τ₀ band.)

---

## 3. The finite-sample inflation INFLATE=1.35 — justification

n=6 gives a very noisy estimate of the worst-per-sim residual: the max over 6 draws is a downward-biased
estimator of the population worst (the true tail is under-sampled), and the per-cell coherent residual
itself has ~1/√(rows-per-cell) sampling noise on top. The plan specifies ×1.2–1.5; **1.35 is the
mid-point**, defensible as:
- the max of n=6 half-normal-ish draws sits ~0.8–0.9× the population 95th percentile → inflate ~1.2–1.3
  just to reach the population worst;
- plus the per-cell coherent residual's own sampling error (~1.1×);
- product ≈ 1.3–1.4. Use **1.35**; if a later diagnostic shows the per-z LOSO residuals are unusually
  noisy (per-sim spread > 2× the worst |coherent|), bump to 1.5 and state why. Do NOT go below 1.2
  (the plan floor) — 6 clustered sims cannot justify a tighter inflation.

**What n=6 CANNOT certify:** that 1.35× covers the LF→HR residual *outside* the HR cluster
(ns∈[0.859,0.979]). The inflation covers sampling noise WITHIN the cluster; the OUT-of-cluster
extrapolation is §4, budgeted separately. Do not let §4's job leak into INFLATE.

---

## 4. The n_s-edge MF-extrapolation budget (design item 4 / I3) — SEPARATE term

The 6 HR sims top out at **ns=0.979**; eBOSS lands at n_P≈**1.009**, i.e. AT/ABOVE the HR ceiling, and
the real fit extrapolates the MF correction *in the tilt direction* — the one we cannot afford to get
wrong. This is a DIFFERENT object from §2 (which is in-cluster generalization) and from the existing
LF-emulator sparse-edge **step-C_emu above 0.995** (which covers the LF backbone's 2/60-point training
sparsity, not the MF correction's HR-coverage gap). The real fit hits BOTH simultaneously — the
highest-risk corner — so they must be budgeted SEPARATELY and ADDED.

### 4.1 Estimator
The MF correction `g(z,τ₀,k)` and its rank-1 `u_τ`,`u_z` are fit on ns∈[0.859,0.979]. Outside that, it
is extrapolated. Budget the extrapolation as a ramp keyed to the distance of the query ns from the HR
cluster's edge:

```
d_ns(ns) = max( ns - 0.98 , 0.86 - ns , 0 )        # unit-cube distance OUTSIDE the HR ns box [0.86, 0.98]
sigma_edge(z, band; ns) = SLOPE(z, band) * d_ns(ns)
```

where `SLOPE(z, band)` = the measured *rate* at which the LF→HR coherent residual changes with ns
INSIDE the cluster, per unit ns, extrapolated outward. Estimate `SLOPE` from the 6-sim LOSO: regress
`coherent_s(z, band)` on `ns_s` over the 6 HR sims (corr(ns, residual) = −0.33 in the pooled low-z
high-k band, `mf_rescorr_loso.txt` — so there IS an ns-trend to extrapolate). Use the magnitude of the
fitted slope (not its sign — we are budgeting uncertainty, not predicting a mean shift):
```
SLOPE(z, band) = | d(coherent_s)/d(ns_s) |  fit over the 6 HR sims, per (z, band)
sigma_edge(z, band; ns) = 2.0 * SLOPE(z, band) * d_ns(ns)     # 2× because a 6-sim slope is itself noisy
```
Floor it so that even a near-zero fitted slope leaves SOME edge budget once ns leaves the cluster:
```
sigma_edge(z, band; ns) = max( 2.0·SLOPE·d_ns , 0.5·sigma_floor(z,band)·(d_ns / 0.03) )
```
(at d_ns = 0.03 — one cluster-NN-gap outside the box — the edge term reaches ≥0.5× the in-cluster
floor; it grows linearly beyond.) **This term DEPENDS ON ns** (unlike §2's fixed floor), so it must be
evaluated at the sampled ns inside the likelihood — it is a fixed *function* of ns, still θ-blind in
the sense that it carries no gradient toward the MAP (pass it as a `stop_gradient`'d variance; it
widens the posterior near the edge, it does not pull the MAP).

### 4.2 Assembly
```
emu_var(k, z) += ( sigma_edge(z, band(k); ns) * P_obs(k) )**2     # ADD to §2.3, small-scale leg only
```
Total small-scale floor = §2 (in-cluster generalization, fixed) ⊕ §4 (out-of-cluster ns-extrapolation,
ns-dependent), added in quadrature on the variance. Keep BOTH separate from the existing step-C_emu
above 0.995 (LF-emulator sparsity) — three distinct objects, three distinct additive variance terms.

### 4.3 Real-fit/blinding gate flag (mandatory)
State in the blinding/real-fit report: **"any posterior mass above ns≈0.98 rests on MF-correction
EXTRAPOLATION (no HR sims there); the sigma_edge term inflates C_emu accordingly, but the (z,τ₀)
interaction itself is untested at ns≥0.98. Treat ns>0.98 posterior as guarded, not nominal."** This is
the honest coverage statement; do not let the inflated C_emu be read as "certified."

---

## 5. What n=6 can and cannot certify (the discipline statement for the output)

- **CAN:** size an in-cluster generalization floor (§2) and its sampling-noise inflation (§3) — these
  use the 6-sim LOSO honestly (worst-per-sim, not pooled, z-resolved, per-band).
- **CANNOT:** certify the floor's amplitude OUTSIDE ns∈[0.86,0.98] (that is the §4 extrapolation
  budget, which is an *inflation*, not a *measurement* — a 6-sim slope cannot be trusted as a point
  estimate, hence the 2× and the floor). CANNOT distinguish a real z=5 sign-flip from 6-sim noise in
  any single z-bin (mitigate: the per-z floor is `max`'d with the global 1.23%). CANNOT cover HR→truth
  (the k<0.06 cap does that). 
- **Report all three in the floor diagnostic output** so the next reviewer sees the boundary of what
  the 6 sims license.

---

## 6. Compute / discipline
6-fold HR-LOSO through `logP_mf` (no LF retrain; the head is re-fit on 5 HR sims per fold, cheap). The
CS agent owns `scripts/diag_mf_cemu_floor.py` and the `data_likelihood.py` assembly; this spec is the
estimator contract. Forward-only; profile before NUTS.
