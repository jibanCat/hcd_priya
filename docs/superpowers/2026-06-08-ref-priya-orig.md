# Reference notes: PRIYA original suite (Bird et al. 2023, arXiv:2306.05471)

Fetched 2026-06-08. Source: arXiv:2306.05471 = JCAP 10 (2023) 037,
"PRIYA: a new suite of Lyman-alpha forest simulations for cosmology",
Bird, Fernandez, Ho, Qezlou, Garcia, Tonnesen.
Primary text verified against the arXiv PDF (gs txtwrite extract, /tmp/priya.txt)
AND the arXiv HTML rendering of Table 2 — numbers agree.

## Purpose for this project
Establish the ORIGINAL PRIYA low-fidelity (LF) Latin-hypercube design ranges,
to test USER FEEDBACK #2: that n_s > 1.0 was added LATER and was NOT in the
original LHD. VERDICT: CONFIRMED. Original n_s (n_p) upper bound = 0.995 < 1.0.

## (a) Design / fidelities (abstract + Sec 2)
- 9-dimensional parameter space: 4 cosmology + 3 HeII reion + 1 HI reion + 1 AGN.
  (A 10th parameter, the MEAN FLUX / tau0, is generated in post-processing, so the
  *simulation* space is 9-D. Text line 508-510.)
- 48 LOW-FIDELITY sims: 1536^3 particles, 120 Mpc/h box (mean interparticle
  spacing ~78 kpc/h).
- 3 HIGH-FIDELITY sims: 3072^3 particles, 120 Mpc/h box.
- Multi-fidelity GP emulator (doubles effective resolution).

## (b) Space-filling design — IS a Latin Hypercube (Sec 2.8)
"We generate simulations at specific points in parameter space using a Latin
Hypercube design, following Ref. [14]. Latin Hypercube samples are generated at
random on a normalised unit cube, and the design which maximises the spread
between parameter points is chosen for the final emulator."

CONSTRUCTION HISTORY of the 48 LF sims (line 514-521), important for FEEDBACK #2:
1. 30 sims: INITIAL Latin Hypercube design.
2. +2 sims: Bayesian optimisation (BO kept hitting the extreme boundaries).
3. +8 sims: an EXTRA Latin Hypercube (because BO chased the edges).
4. +3 sims: more LF chosen by Bayesian Optimisation.
5. +6 sims: UNIFORMLY spaced in alpha_q, an EXPANSION of the alpha_q lower bound.
   "Our initial suite covered 1.6 < alpha_q < 2.5. A comparison ... to observed
   mean IGM temperatures suggested expanding the lower limit ... 6 simulations
   uniformly spaced between 1.325 <= alpha_q <= 1.575."
Sum = 30+2+8+3+6 = 49 nominal entries; paper/abstract state 48 LF used.

KEY: The ONLY documented range EXPANSION is alpha_q (quasar spectral index),
lowered from 1.6 to ~1.325. There is NO mention anywhere of expanding n_s.
n_s stayed within [0.8, 0.995] in the original suite.

## (c) The 9 parameters and exact ranges — Table 2 (verified, lines 305-317)
| Parameter        | Min        | Max        | Description                                   |
|------------------|------------|------------|-----------------------------------------------|
| n_p (n_s)        | 0.8        | 0.995      | Scalar spectral index (small-scale slope)     |
| A_p              | 2.2e-9*    | 2.6e-9     | Power amplitude at k = 0.78 Mpc^-1            |
| h                | 0.65       | 0.75       | Hubble parameter                              |
| Omega_M h^2      | 0.14       | 0.146      | Total matter density                          |
| z_HeI (herei)    | 3.5        | 4.1        | START redshift of HeII reionization           |
| z_HeF (heref)    | 2.6        | 3.2        | END redshift of HeII reionization             |
| alpha_q          | 1.3        | 2.5        | Quasar spectral index during HeII reion (heat)|
| z_HI             | 6.5        | 8          | Median redshift of HI reionization            |
| eps_AGN          | 0.03       | 0.07       | Thermal efficiency of black-hole feedback     |

NOTE on n_s/A_p definition (Sec 2.2, line 186-194): PRIYA defines the primordial
power spectrum at a PIVOT k = 0.78 Mpc^-1 (NOT the usual k0 = 0.05 Mpc^-1).
P(k) = A_p (k/0.78 Mpc)^{n_p - 1}. So n_p and A_p are the SMALL-SCALE slope and
amplitude, *distinct* from the CMB n_s, A_s at k0=0.05. The pivot 0.78 Mpc^-1 is
chosen to decorrelate amplitude and slope as seen by the forest [Ref 14].
=> "n_s = 0.995 max" is the FOREST pivot slope, not literally the CMB n_s; but it
is the parameter the emulator/LHD is built on and the one the user is referring to.

* A_p lower-bound DISCREPANCY in the paper: Table 2 lists A_p min = 2.2e-9, but
  the PROSE (line 192) says "We vary A_p in the range 1.2e-9 and 2.6e-9." The two
  disagree on the lower bound (1.2 vs 2.2). The HTML render also shows 2.2e-9.
  Likely a typo in one place; the leave-one-out caveat (below) references a sim at
  A_p = 2.57e-9, consistent with the 2.6e-9 UPPER bound. Flag, not resolved here.
  This does NOT affect the n_s conclusion.

n_s rationale (line 193-194): "These ranges are chosen to include the posterior
constraint from Planck [66]." => The 0.8-0.995 window was deliberately set to
bracket Planck's n_s ~ 0.965 (at the CMB pivot). Planck n_s sits comfortably
inside [0.8, 0.995]; the design did NOT need n_s > 1.

## (d) Emulator-accuracy / near-edge generalization caveats
- LF flux-power emulator: average abs. relative error ~2e-3. HF (multi-fidelity)
  median error ~1e-2. (Sec 2.9, lines 603-606.)
- WORST-CASE leave-one-out error predicting LF flux power = ~0.06 (6%). The paper
  attributes this directly to a sim "on the extreme edge of parameter space
  (A_p = 2.57e-9), and so an extrapolation for the leave-one-out emulator."
  (lines 608-611.) => explicit statement that ACCURACY DEGRADES AT BOX EDGES /
  for extrapolation beyond the design hull.
- Bayesian optimisation "frequently chose points on the extreme boundaries of the
  parameter space" (line 516) — which is why they added the extra LHD; another
  signal that edges are where the emulator is least constrained.
- All errors stated to be smaller than DR14 observational uncertainties (1.5-15%).
- Resolution convergence: ~7% at k=0.07 s/km for HF; worst convergence during
  HeII reionization (z~3-4). (Consistent with project MEMORY reference-priya-sims.)

## Bottom line for FEEDBACK #2 and the n_s under-prediction
- ORIGINAL LHD n_s window = [0.8, 0.995], hard upper bound 0.995 < 1.0. CONFIRMED.
- If the current analysis uses an emulator/prior with n_s >= 1.0, that region is
  an EXTENSION beyond the original 2306.05471 design hull. The paper itself warns
  that LOO/extrapolation error grows at the edges (the A_p=2.57e-9 case, 6% error).
- Therefore any data preference pushing n_s toward/over 1.0 would sit at/over the
  TOP edge of the original training box, where emulator accuracy is least trusted.
  A -0.65 sigma LOW n_s (data pulling n_s DOWN, away from 1.0) moves it toward the
  interior/lower part of the box — but the relevant point is that the box is
  ASYMMETRIC about Planck (0.8 to 0.995, i.e. lots of headroom below 0.965, only
  0.03 above). An under-predicting (low) n_s is well inside the design; it is the
  HIGH side that is edge/extension-limited. See verdict field.
