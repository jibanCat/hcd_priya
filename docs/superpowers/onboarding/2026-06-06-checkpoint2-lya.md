# Lyα/IGM/Emulator-fidelity checkpoint — low-k A_p bias investigation — 2026-06-06

Lens: emulator FIDELITY + IGM/PRIYA physics. Did the emulator-error claims (B/C/D) hold up
when I re-ran them? Is the high-k coherent error PHYSICAL (a real LF mis-prediction of the
high-k P1D shape, → MF/HR fix) or a TRAINING/PROJECTION artifact (→ covariance band-aid)? And
the load-bearing physics question the PI posed: is the IN-SCOPE 0.02–0.06 band ALSO
systematics-limited, contradicting the "validated band" claim?

Everything below is read from code (file:line), from the two result `.txt` files, or
REPRODUCED LIVE this session (`/tmp/band_finer.py`, `/tmp/zaccum.py`, `/tmp/zk_coherent.py`,
`/tmp/mf_kshape.py`, `/tmp/kscheck.py`, direct basis inspection).

## Verdict (one line)
**GO-WITH-CHANGES** — the headline numbers reproduce and the diagnosis is directionally right,
but three claims are over-stated (z-coherence, "6–18× in every band", the validated-band
framing) and the +0.6σ is statistically marginal (n=8, ~1.9σ from zero); the fork must NOT be
decided as "covariance band-aid" until MF is wired, because half the bias is in-scope physics.

---

## Claims I VERIFIED (re-ran or re-derived) vs claims I could not check

**VERIFIED by independent re-run (numbers matched):**
- Basis orthonormality `||BBᵀ−I||_max = 2.68` — reproduced exactly (direct inspection of
  `model.head_b.p_filt_basis`, shape (24,172), row norms 1.05–1.92). The low-rank path IS active
  in production (`head_b.n_basis=24`, basis not None). The basis the script reads is the CORRECT
  one (HeadB's θ-dependent residual basis), even though the doc cites the wrong line (see Errors).
- PART B band-attribution: FULL +0.617σ A_p / −0.191σ n_s; high-k +0.723σ; mid −0.05σ; low −0.056σ
  — reproduced to 3 digits in `/tmp/band_finer.py`.
- PART D leakage r = −0.612 — matches.
- PART A/C per-band emu-err and truth-floor table — matches the `.txt`.
- **z-ACCUMULATION (claim C, the "sharpest finding")**: I rebuilt the single-z=3 vs full-z bias
  MYSELF (`/tmp/zaccum.py`, same forward, same C): **full-z +0.617σ, z=3-only +0.033σ, ×18.6** —
  reproduced EXACTLY. The mechanism (bias does not average down over z) is real at the bias level.
- The current C_emu is DIAGONAL in k (cross-class option is k-diagonal 4×4 blocks; no k–k or z–z
  correlation — `data_likelihood.py:258-291,335`). So "a diagonal C_emu cannot whiten a coherent
  low-rank mode, the correlated C_emu is the right tool" is STRUCTURALLY CORRECT.

**Could NOT check / not yet possible:**
- Whether the +0.6σ survives as a NUTS coverage failure (doc itself flags this as the arbiter —
  §5 caveats). Fisher MAP-shift ≠ NUTS bias on a 28–30-dim funnel.
- Whether a TARGETED retrain actually removes it (the "above the rank floor ⇒ retrainable in
  principle" is an existence argument, not a demonstration).
- Whether MF wiring reduces or increases it (structurally re-measured only post-wiring).

---

## Errors / overclaims / things the doc gets wrong (specific)

**(1) The doc's central physics claim hides that HALF the bias is IN the validated band.**
Doc §2/§0 lumps the source as "high-k (k>0.02): +0.72σ", contrasted against the out-of-scope
0.06–0.1 worry. My finer split (`/tmp/band_finer.py`, splitting 0.02–0.069 into in-scope
0.02–0.06 and out 0.06–0.069):
```
band             A_p bias   n_s bias
low  <5e-3        -0.056     +0.117
mid  5e-3-0.02    -0.050     +0.006
IN   0.02-0.06    +0.312     -0.248      <-- IN the "validated, unbiased" band
OUT  k>=0.06      +0.412     -0.066      <-- out of scope
```
So **+0.31σ of the A_p bias is sourced from the IN-SCOPE 0.02–0.06 band** — the band the memory
`phase2-analysis-scope-dla` calls "entirely inside the LF-backbone's validated, unbiased band
(A_p Fisher-bias RMS 0.067σ, 0/8 gate failures)". This is the PI's most important question and
the answer is: **yes, the in-scope band is also carrying a coherent A_p residual once z-summed.**
The "validated band" claim was a per-z, full-k-pooled RMS; it does not certify the z-summed
coherent low-rank mode. The doc should say this explicitly instead of routing everything to
"high-k" (which reads as "the out-of-scope tail").

**(2) Scope-cap violation in the rerun: KS keep extends to k=0.0627 (14 bins at k≥0.06).**
`/tmp/kscheck.py`: KS keep k-max = 0.0627, with 14 bins ≥0.06. The memory locks KS at **k<0.06**.
The rerun (and PART B's +0.41σ OUT-band) therefore includes 14 out-of-scope bins that should be
masked per the analysis-scope decision. This inflates the "high-k" bias relative to the actual
analysis configuration. **Re-run with the KS k<0.06 cap enforced before quoting the bias as the
production number.** (DESI is clean: keep k-max = 0.0409, all in-scope.)

**(3) "z-COHERENT (the high-k response is biased the same way at every z)" is too strong.**
`/tmp/zk_coherent.py` (signed coherent emu error, std-log, clean+LLS, per z & band): the
IN-band error FLIPS SIGN with z (negative at z<2.7, positive z=2.8–4.2; frac z>0 = 0.69, mean
≈ +0.0005 ≈ 0); OUT-band frac z>0 = 0.46, mean −0.0037. At the P_filt level the high-k error is
NOT a clean monotone same-sign residual. The ×18.6 amplification is real but is a
PROJECTION/accumulation effect through C⁻¹·J over ~13 z-slices, NOT "the same sign at every z."
The doc's mechanism sentence over-claims the physical coherence; the bias-level statement
(full ≠ z·single because it does not average down) is the defensible version.

**(4) "emu error is 6–18× the floor in EVERY band" is wrong for the mid band.**
From the `.txt`: emu/floor ratios are LLS mid 3.18×, subDLA mid 2.15×, DLA mid 3.68× — i.e.
2–4× in the mid band, not 6×. The 6–18× holds for low and high bands. Minor, but "every band"
is false; say "6–18× in the low/high bands, 2–4× mid."

**(5) The +0.6σ is statistically MARGINAL and the doc under-states this.**
Per-sim full-z A_p bias (`/tmp/zaccum.py`): `[-0.65, 1.48, 1.98, -0.03, 1.03, -0.24, 1.32,
0.04]`, mean +0.62, SD ≈ 0.94, **SE ≈ 0.33 ⇒ mean is ~1.9σ from zero on n=8.** The spec §8
quotes "SE 0.31 → ~1.7σ". So the headline is a ~2σ central value, not an established bias. PART
B's high-k +0.72σ has scatter 0.72 (SE 0.25, ~2.9σ). PART D's r=−0.61 is 8 points with one
leverage outlier. None of this is a clean detection — it is a "real-enough to budget, not
real-enough to retrain on" signal. The doc's confident "coherent +0.6σ" framing should carry the
n=8 / ~2σ uncertainty front-and-center.

**(6) Citation error: `model.py:135` is the θ-BLIND BaselineHead, not the residual basis.**
Doc §2 and spec §7 cite `model.py:135` ("the GLOBAL SVD output basis") — but line 135 is
`BaselineHead.__call__` (`coeffs @ self.p_filt_basis`), the θ-BLIND baseline m̂. The θ-dependent
residual r̂ (which is what the A_p Jacobian flows through, `predict.py:38`
`logP = baseline + sig_cosmo·r̂`) goes through `HeadB.p_filt_basis` (~model.py:185+,
`__call__` near the HeadB block). The SCRIPT reads the right object (`head_b.p_filt_basis`); only
the doc's line cite is wrong. Worth fixing so the mechanism story points at the basis that
actually carries the cosmology gradient. (Note also: there are TWO global bases — head_base and
head_b — each (24,172); the baseline carries ~99.5% of the variance but is θ-blind, so the
leakage-relevant one is head_b's, correctly used.)

**(7) Reproducibility gap: the §3 ruler table is not in the cited scripts.**
The §3 table ("correlated vs diagonal 1.26×", "diagonal 3%-frac −0.60σ", "z=3 +0.033σ /
×18.6") is presented as measured, but `diag_emu_lowk_investigation.py` (which I read fully,
223 lines) computes NONE of it — it only mentions "ruler" in its docstring goal. `grep` for
`18.6`/`0.033`/`1.26`/`ruler` in both cited scripts finds only the docstring. I had to REBUILD
the z=3↔full-z number myself to verify it (it checks out). **The ruler computation should be
committed as a script** so the 1.26× / 3%-frac numbers are reproducible too (I verified only the
z-accumulation leg of the table).

---

## My lens's take on the fork: correlated-in-k/z C_emu vs targeted emulator retrain (vs MF)

The doc frames a 2-way fork (correlated C_emu now; retrain later). From the IGM/fidelity side it
is really a **3-way fork, and the right answer is gated on MF — which the doc correctly says but
then under-weights.**

- **The bias is half in-scope, half out-of-scope, sign-changing in z, ~2σ marginal, and the
  OUT half includes a scope-cap violation.** This is NOT the signature of a clean physical
  "LF can't resolve the high-k P1D shape vs cosmology" residual that would mandate HR. If it were
  pure HF-resolution physics, the P_filt error would be monotone and same-sign in z (it is not —
  Error 3) and would live mostly above the LF Nyquist (it is split, with ~0.31σ at k<0.06 where
  the LF is supposed to be converged). So I do **not** read this as "the LF emulator structurally
  cannot capture HF-resolution physics" — it reads as a TRAINING-limited residual (6–18× the rank
  floor, basis 2.7-from-orthonormal) projected through a global basis and accumulated over z.

- **MF is the pivotal unknown and it is NOT free.** `res_corr` (the FIXED particle-convergence
  factor) is already ~2–4% at k=0.02–0.06 at z=3 (`/tmp/mf_kshape.py`: 0.985→0.964) and 6–9% at
  k>0.06 — i.e. MF reshapes the EXACT band that sources the bias, including the in-scope half.
  So "re-measure after MF" is right and load-bearing. BUT the MF default itself carries an A_p/n_s
  Fisher-bias budget (`diag_mf_complexity.py`: MLP head n_s +3.84σ → rho(k,z)-only chosen because
  it gives |A_p|~1.4σ, |n_s|~2σ). MF wiring could REDUCE this LF-only bias (HF more converged at
  high-k) OR ADD its own. The doc's optimistic "MF acts on the same channel → may absorb it"
  should be balanced against "MF brings its own ~1.4σ A_p budget."

- **My ranking:**
  1. **MF-wiring FIRST, then re-measure** — non-negotiable. The LF-only +0.6σ is a number for a
     forward model the production fit will not run (no res_corr, no log_rho). Quoting it as THE
     risk is premature. Half the bias band is reshaped 2–9% by res_corr.
  2. **Correlated-in-k/z C_emu** as the marginalization tool for whatever coherent mode SURVIVES
     post-MF — this is the right structural tool (diagonal C_emu provably cannot whiten a
     low-rank mode) and the cheap/robust path. Build it, but parameterize it from the POST-MF
     residual, not the LF-only one.
  3. **Targeted HR/retrain** — only if a coherent mode survives BOTH MF and the correlated C_emu
     AND is shown to be monotone-in-z physical (the current evidence says it is not). The basis
     non-orthonormality (||BBᵀ−I||=2.7) is a real, cheap retrain lead (re-orthonormalize / SVD
     warm-start is already coded in `svd_basis_init`) and worth doing regardless, but as a
     fidelity hygiene item, not as the bias fix.

**Bottom line for the fork:** do NOT pick "covariance band-aid vs retrain" yet. The honest
sequence is MF-wire → re-measure (with KS capped at k<0.06) → correlated C_emu on the survivor →
retrain only if a monotone-z physical mode persists. The doc's "load-bearing risk moved onto the
correlated C_emu" is half-right: it moved onto **the post-MF residual**, and we don't yet know its
size or whether it is in-scope.

---

## Additional risks or missing checks

- **n=8 is too few for any of these means.** Every headline (+0.62σ A_p, +0.72σ high-k, r=−0.61)
  is a ~2σ statement on 8 sims with large scatter. Before the fork is decided, run the bias over
  ALL held-out sims across folds (the spec's interior-n_s spot-checks are in-sample and under-
  state emu error, as it admits). A bootstrap CI on the mean would make the marginality explicit.
- **The in-scope coherent A_p mode partially invalidates the 0.067σ "validated band" memory.**
  This should be reconciled in `phase2-analysis-scope-dla`: the band is unbiased PER-Z but carries
  a z-summed coherent A_p residual. Not a contradiction in the emulator's per-z spec, but the
  "0/8 gate failures" was a single-z RMS gate that is blind to z-accumulation (same blind spot the
  doc identifies for Phase-2's 0.067σ).
- **per-class check (PI asked): the doc shows LLS only.** The `.txt` has all 4: the emu/floor
  ratio is LOWEST for DLA (4–5×) and subDLA (2–7×), highest for clean (7–18×). The "training-
  limited" verdict holds per-class, but DLA/subDLA are closer to the floor — and DLA is masked
  anyway, so the binding classes for the in-scope bias are clean+LLS (which is what PART D
  correctly uses). No error here, but the per-class spread (clean worst) suggests the residual is
  a CLEAN-forest baseline issue, not an HCD-template issue — another point against "HR/HCD retrain."
- **Missing: is the +0.6σ even present with the production multi-leg cov + SiIII metals + real
  C_data?** The rerun uses a forward-only Fisher with the cache C; the real Leg-B has the full
  DESI 1020×1020 cov + KS HR cov + metal terms. The covariance ruler being "only 1.26×" was
  asserted but not reproducibly computed (Error 7).

---

## Recommendations (numbered, actionable)

1. **Re-run the band-attribution and the +0.6σ with KS capped at k<0.06** (drop the 14 bins
   ≥0.06). Quote the in-scope-only A_p bias as the production-relevant number. (Error 2.)
2. **Separate the in-scope (0.02–0.06) and out-of-scope (≥0.06) contributions in the doc** and
   state plainly that ~+0.3σ is sourced INSIDE the validated band — answer the PI's Q1 honestly
   rather than routing it all to "high-k". (Error 1.)
3. **Soften the z-coherence language** to "the per-z error does not average down (projection
   accumulation), not "biased the same way at every z" — the P_filt error is sign-changing in z.
   (Error 3.)
4. **Carry the n=8 / ~2σ uncertainty on every headline** (+0.62σ ⇒ mean ± SE 0.33, ~1.9σ). Run
   the bias over all folds' held-out sims + bootstrap CI before the fork is decided. (Error 5.)
5. **Commit the ruler script** that produced the §3 table (1.26×, 3%-frac, z=3/×18.6) — currently
   not reproducible from the cited scripts. (Error 7.) I verified the z-accumulation leg; the
   covariance-ruler legs are unverified.
6. **Do not decide the fork pre-MF.** Sequence: MF-wire → re-measure (KS<0.06) → correlated C_emu
   on the post-MF residual → HR/retrain only if a monotone-z physical mode survives. Weigh MF's
   own ~1.4σ A_p budget against any reduction it brings. (Fork section.)
7. **Fix the `model.py:135` cite → HeadB's `p_filt_basis`** (the θ-dependent residual basis the
   A_p gradient flows through); note both head_base and head_b carry a (24,172) basis. (Error 6.)
8. **Re-orthonormalize the HeadB basis as a cheap fidelity hygiene step** (||BBᵀ−I||=2.7;
   `svd_basis_init` already exists) and re-measure — independent of the bias fork, this is a
   concrete, low-cost lead for the coupling channel.
9. **Reconcile the `phase2-analysis-scope-dla` memory**: the "validated, unbiased (0.067σ)" band
   carries a z-summed coherent A_p residual; note the gate was single-z and blind to it.
