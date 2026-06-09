# Reference notes — LaCE P1D emulator normalization (arXiv:2305.19064)

**For:** User feedback #3 — how does LaCE normalize/transform its P1D target, how does it treat
amplitude vs tilt, and what does that imply for our n_s (-0.65 sigma) under-prediction?
**Date:** 2026-06-08. Sources fetched + text-extracted locally via pymupdf.

## Papers read
- **LaCE-NN (2305.19064)** Cabayol-Garcia, Chaves-Montero, Font-Ribera, Pedersen, *"A neural network
  emulator for the Lyman-alpha forest 1D flux power spectrum"*, MNRAS 525, 3499 (2023). This is the
  paper the user cites for DATA NORMALIZATION. Full text at /tmp/nn2305.txt (18 pp, 94 kchar).
- **P21 (2011.15127)** Pedersen et al. 2021, JCAP, *"An emulator for the Lyman-alpha forest in
  beyond-LCDM cosmologies"* — the GP predecessor that introduced the Delta2_p/n_p compression. Full
  text at /tmp/ped2011.txt.
- Polyfit (the polynomial decomposition) was actually introduced in **Pedersen et al. 2023** (the
  cosmic-variance fix), inherited/extended by LaCE-NN. P21 itself is per-k, not polyfit.

---

## (a) HOW LaCE normalizes / transforms the P1D target

**Two-layer transform: (i) a scalar median rescale, then (ii) a polynomial-in-log-k decomposition of
log10 P1D. It is NOT per-k standardization and NOT a ratio to linear power and NOT k*P1D/pi.**

1. **Scalar median normalization (Eq. 6):**
   `P'_1D = log10( P1D / A_scalings )`, where `A_scalings` = **the median of ALL P1D measurements in
   the training sample** (a single global scalar, z- and k- and theta-independent). In the GP version
   (P21, Eq. 4.1) the same idea normalizes P1D by the median to approximate the GP zero-mean prior.

2. **Polynomial-in-log10(k) decomposition of log10 P1D (Eq. 5):**
   `P'_1D = sum_{i=0}^{n-1} alpha_i * (log10 k_parallel)^i`, a **fifth-order polynomial (n=5)** for
   k in (0, 4] Mpc^-1 (P21/Pedersen2023 used 4th order to k=3). The NN **emulates the polynomial
   COEFFICIENTS alpha_i**, not per-k values. (P21's GP emulated per-k values directly in 85 linear
   k-bins; Pedersen 2023 switched to coefficients because the per-k target was "significantly affected
   by cosmic variance" — the polynomial smooths the large-scale cosmic-variance noise.)

3. **It is a Mixture Density Network (MDN):** maps the 6 params to a Gaussian over the 6 polynomial
   coefficients, predicting both mean alpha_i and per-coefficient uncertainty sigma_i, which propagate
   to a per-k log10 P1D variance (Eq. 7): `sigma^2_logP1D = sum_i sigma_i^2 (log10 k)^{2i}`.

4. **Loss is a per-k inverse-variance / beta-NLL (Eq. 8), evaluated in log10 P1D space**, NOT in
   coefficient space:
   `L = (1/N) sum_samples sum_k [ ((log10 P'_pred - log10 P'_true)/sigma_logP1D')^2 + 2 log sigma_logP1D' ]`.
   They explicitly say evaluating the loss in log10 P1D space (rather than coefficient space) "enables
   the neural network to learn the importance of each coefficient ... and weight the learning
   accordingly." So the per-k log-space residual is what is minimized; the heteroscedastic sigma is
   learned, not a fixed marginal std.

5. **Input parameter normalization:** the 6 inputs are min-max rescaled to a unit volume (~[-0.5, 0.5]);
   inputs = `theta = [Delta2_p, n_p, mean_F, sigma_T, gamma, k_F]`.

**Architecture ablation (Fig. 4) — directly load-bearing for our diagnosis:** they ablated the target
representation. Starting from a polynomial in **linear** k predicting P'_1D gave **10% error**; switching
to a polynomial in **log10 k** (their modification "A") dropped it to **1.4%** — the single biggest
improvement. Further modifications: B = decompose log10 P1D (vs P1D), C = LeakyReLU, D = "parameter-space
shift". Final stack reaches sub-percent. The takeaway is that **the CHOICE of target decomposition basis
(log-k powers of log P1D) was the dominant accuracy lever**, far more than architecture.

## (b) Amplitude vs tilt treated differently?

**Not by a separate amplitude/tilt branch in the TARGET, but by the COMPRESSED INPUT
parameterization.** LaCE does not emulate raw cosmological params; it emulates P1D as a function of the
amplitude **Delta2_p** and slope **n_p** of the *linear* matter power at a pivot:
- `Delta2_p(z) = k_p^3 P_lin(k_p, z)` (amplitude), `n_p(z) = d ln P_lin / d ln k |_{k_p}` (slope), at
  pivot `k_p = 0.7 Mpc^-1`, defined at `z_star = 3`. Training ranges Delta2_p in [0.25, 0.45], n_p in
  [-2.35, -2.25].
- P21 computes Delta2_p and n_p by **fitting a SECOND-ORDER log polynomial to P_lin over
  0.5 k_p < k < 2 k_p** (a local-in-k amplitude+slope+curvature fit, then read amplitude & slope at
  k_p). A natural extension noted is a running alpha_p (2nd derivative); NOT currently emulated.

So **the tilt direction is carried by an explicit, low-dimensional, physically-defined SLOPE input
(n_p)**, and the amplitude by Delta2_p — they are *separated at the input* by construction, exactly the
"compress amplitude and slope" idiom. The output side does NOT separately whiten/standardize amplitude
vs tilt; the polynomial-coefficient target mixes them.

## (c) What they report about emulating SLOPE vs AMPLITUDE, and any accuracy asymmetry

**Neither LaCE-NN nor P21 reports a quantified amplitude-vs-slope accuracy ASYMMETRY.** Accuracy is
reported as overall percent error on P1D:
- LaCE-NN: **sub-percent across k = 0.1-4 Mpc^-1 and z = 2-4.5**; ~1% for unseen thermal/ionization
  histories; sub-percent for LCDM extensions (neutrinos, running, curvature) outside the training set;
  degrades to ~4% at z=2 when params fall outside the convex hull.
- P21 GP: ~1% leave-one-out.
- The only "slope" discussion is (i) the *input* slope n_p and (ii) the *polynomial order* in log-k.
  P21 motivates the Delta2_p/n_p compression by noting forest params are "strongly degenerate" and that
  "minimising parameter degeneracies is important" / "interpolation errors can artificially break
  degeneracies" — i.e. low-D compression is what protects directions like the tilt.
- Implicit asymmetry handling: the **log10-k polynomial basis is itself a tilt-friendly basis** — the
  low-order coefficients (alpha_0 amplitude-like, alpha_1 slope-like in log P vs log k) carry most
  signal, and the log-space inverse-variance loss reweights so the network "learns the importance of
  each coefficient". That is a structural, not statistical, way to keep the slope coefficient resolved.

## (d) Normalization change that helps a TILT (n_s) direction a global log-P1D standardization under-resolves

The LaCE recipe points to concrete fixes for exactly our failure mode (global per-k standardization
buries the ~0.4% theta-signal so the tilt/n_s direction is under-resolved):

1. **Decompose in a SMOOTH log-k BASIS and emulate the coefficients, not per-k values.** LaCE's
   alpha_i are an orthogonalizable, low-D representation where the **slope-like coefficient (alpha_1,
   roughly d log P / d log k) is an explicit, separately-weighted DOF**. A global per-k standardization
   spreads the tilt signal across all k and lets the amplitude/mean dominate the variance; a
   coefficient target makes the tilt a named, individually-loss-weighted output. This is the single
   biggest accuracy lever they found (Fig. 4: linear-k -> log-k = 10% -> 1.4%).

2. **Use log10 P1D (not P1D) and a per-k INVERSE-VARIANCE (heteroscedastic) loss in log space**
   (Eq. 8), with a LEARNED per-k sigma — NOT a fixed marginal std. This is the same inverse-variance
   reweighting our redesign #1 calls for (sigma_cosmo whitening / beta-NLL), and it is what lets the
   loss "weight the learning" toward the under-represented coefficients/directions.

3. **A single scalar (median) normalization, NOT per-k marginal standardization.** LaCE deliberately
   divides P1D by ONE global median (A_scalings) and lets the polynomial absorb the k-shape. Our bug
   is the *opposite*: per-k marginal standardization by the (z,tau0)-dominated std, which is precisely
   what buries the tilt. A median scalar + smooth-basis decomposition + inverse-variance loss avoids
   that burial.

---

## Compare to OUR redesign (2026-06-02-normalization-fix-research.md)

| Aspect | OUR redesign | LaCE (2305.19064) |
|---|---|---|
| Dominant-variation removal | in-network theta-BLIND baseline head m_hat(z,tau0,k) + zero-mean residual head | scalar median A_scalings + **smooth log-k polynomial** absorbs the k-shape; coefficients emulated |
| Re-whitening | residual whitened by **conditional (within-(z,tau0)-cell) sigma_cosmo,k** | **learned per-k heteroscedastic sigma** (MDN), inverse-variance log-loss (Eq. 8) |
| Loss | inverse-variance (sigma_cosmo) weighted | per-k beta-NLL in log10 P1D space (Eq. 8) — same family |
| Amplitude/tilt split | structural: theta-only in the residual head | structural: amplitude/slope compressed at the INPUT (Delta2_p, n_p) + tilt-friendly log-k coeff basis |
| Stored reference / interpolation | explicitly rejected (in-network baseline) | none — single scalar median + analytic polynomial; consistent with our "no stored grid" rule |
| Target basis | per-k residual (we standardize per-k after baseline removal) | **fifth-order polynomial in log10 k** of log10(P1D/median) — a smooth low-D basis |

**Already adopted (convergent):** log-space target; inverse-variance / heteroscedastic loss; removal of
the dominant predictable variation before whitening; a single (not stored-grid) reference; structural
amplitude/tilt separation.

**LaCE does DIFFERENTLY (candidate to adopt for the tilt problem):**
- **A SMOOTH log-k POLYNOMIAL coefficient target** rather than emulating per-k values then standardizing
  per-k. This is the piece our redesign does NOT currently have, and it is exactly the mechanism that
  gives the SLOPE/tilt its own named, separately-weighted degree of freedom (alpha_1) instead of
  diluting it across all k. Our per-k residual target, even after baseline removal + sigma_cosmo
  whitening, still lacks an explicit slope coefficient. Worth checking whether projecting our residual
  onto a low-order log-k (Legendre/polynomial) basis and weighting the slope coefficient up resolves the
  n_s under-prediction.
- A single **scalar median** normalization rather than per-k standardization (we should make sure our
  baseline removal does not reintroduce per-k marginal-std division on the residual in a way that
  re-buries the tilt).

**Caveats / non-transfers:**
- LaCE's amplitude/slope split lives in the *linear-power compression* (Delta2_p, n_p), which is a
  proxy for cosmology; our analysis emulates in (A_p, n_s)-like cosmology directly, so the analog of
  their n_p in OUR target space is whatever low-order log-k coefficient carries the response to n_s.
- LaCE reports no quantified amplitude-vs-slope accuracy asymmetry, so it does NOT directly confirm a
  generic "tilt is harder" claim; it instead shows the *target basis choice* is what governs whether the
  tilt is resolvable. CANNOT-DETERMINE on a measured asymmetry; the relevance is the mechanism.
