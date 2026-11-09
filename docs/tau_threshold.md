# The detection threshold `absorber.tau_threshold`

The catalogue (`hcd_analysis/catalog.py`) finds high-column-density absorbers (HCDs) in each simulated skewer from its
Lyman-alpha optical depth. `tau_threshold` is the optical depth above which a pixel belongs to an absorption system.
Its production value is 100 (`config/default.yaml`). This page explains:

- what that value means physically;
- how much the HCD catalogue and the class templates change if it is moved;
- how to rerun the check.

## What the threshold does

- **Which skewers are searched.** A skewer is searched only if its maximum optical depth reaches
  `max(tau_threshold, tau_min)`.
  - `tau_min` is 0.1 times the line-centre optical depth of an absorber with `N_HI = 10^min_log_nhi` and `b = 30 km/s`.
  - With the production `min_log_nhi = 17.2`, that is about 400.
  - So for any threshold up to about 400, the same skewers are searched.
- **Where a system starts and ends.**
  - A system is a run of pixels with `tau > tau_threshold`.
  - Runs closer than `merge_dv_kms` (100 km/s) are merged.
  - In the production `fast_mode`, the system's `N_HI` is the optical-depth integral over that run divided by the
    Lyman-alpha cross-section.
  - Systems below `min_log_nhi` are dropped.
- **Its physical scale.**
  - A line-centre optical depth of 100 is reached at `N_HI ~ 10^15.6` for `b = 30 km/s`.
  - The lowest-column HCD (`N_HI = 10^17.2`) has a line-centre optical depth of about 4000.
  - So at 100 the whole core of every HCD lies above the threshold, and the integral misses only its far line wings.
- **What it changes.**
  - Moving the threshold changes the extent of each system, and with it the integrated `N_HI`.
  - That can move a system across a class edge (17.2, 19.0, 20.3), or merge nearby runs.
  - The class templates see the threshold only through each sightline's class, which is the class of its
    highest-`N_HI` system.

The value 100 was chosen for a practical reason: it identifies essentially all HCDs quickly. It is far below the HCD
regime, so the core of every HCD is found. The weaker forest systems it also finds are removed by the `min_log_nhi` cut.

The threshold acts on the simulation's own (native) optical depth, before the mean-flux rescaling. Across the 60
low-fidelity PRIYA simulations, that rescaling varies by a factor of 1.6 to 2.2 at a fixed redshift (z = 2.2 to 4.6, at
the mean-flux scaling `alpha_slope = 1.011` used in the check). A fixed threshold
of 100 therefore already acts like an effective threshold varying by about a factor of 2 from one simulation to
another. The check below moves it by a factor of 2 each way.

## How much it matters

**Small over the tested setup.** Relative to its own template, LLS is the most sensitive class, and within narrow k
ranges the templates change by several times more than the global template measures. The results hold only for the
tested domain:

- one low-fidelity PRIYA simulation, near the centre of the parameter box, with all 691200 skewers of each snapshot;
- z = 2.2, 3.0, 4.0 and 4.6;
- one mean-flux scaling (`alpha_slope = 1.011`);
- thresholds 30, 50, 100, 200 and 300 on the same skewers: 100 is the reference, 50 and 200 the headline comparison,
  30 and 300 a wider stress;
- the 172 k bins of the emulator's training grid.

**Global template measures.** The largest change from threshold 100 to 50 or 200, over the four redshifts:

| largest change | LLS | subDLA | DLA |
|---|---|---|---|
| absorber count | 0.81% | 0.22% | 0.05% |
| template amplitude `a` | 0.55% | 0.14% | 0.04% |
| template shape `s` (what remains after the amplitude) | 1.03% | 0.18% | 0.05% |

The clean-sightline power changed by at most 0.07% at any k.

**Local changes in k.** Within each k window, the largest change of a template relative to the template's largest
absolute value in that window (threshold 100 to 50 or 200):

| window (s/km) | LLS | subDLA | DLA |
|---|---|---|---|
| 0.001 <= k < 0.01 | 3.5% | 0.69% | 0.23% |
| 0.01 <= k < 0.02 | 3.5% | 0.40% | 0.11% |
| 0.02 <= k < 0.065 | 2.7% | 0.47% | 0.15% |

A template is a class's P1D divided by the clean P1D, minus one, so a template change is a change of the class's P1D
in units of the clean power. In those units the largest local change is 1.2e-3 for LLS, 1.5e-3 for subDLA and 3.3e-3
for DLA (at the low-k end at z = 2.2, where the DLA template is largest). In these units LLS is not the largest.

- **Effect on the predicted P1D.** The threshold changes no spectrum; it only moves sightlines between classes, so the
  power summed over all sightlines is unchanged. With the class amplitudes held at the sightline fractions of
  threshold 100, and the clean power and templates taken from each threshold, the predicted P1D moves by at most
  0.03% at any k at thresholds 50 and 200 (0.06% at 30 and 300). In a fit that varies the
  class amplitudes, a pure amplitude change of a template is taken up by its amplitude parameter; the shape and local
  changes above are not.
- **Where the effect is largest.** Relative to its own template it is largest for LLS, and the global LLS measures
  grow with redshift: the LLS shape change at threshold 200 is 0.18%, 0.46%, 0.88% and 1.03% at z = 2.2, 3.0, 4.0
  and 4.6, probably because the lowest column densities sit closest to the threshold and a more saturated forest puts
  more pixels near it. The local changes show no such simple trend with redshift.
- **The wider stress (30 and 300).** Counts, amplitude and shape change by less than 1.5%. Locally, the LLS template
  changes by up to 6.0% (0.01 <= k < 0.02 s/km, threshold 300) and the other classes by up to 1.1%.
- **A note on the DLA numbers.** `a` and `s` weight the k bins by the square of the template. At z = 2.2 and 3.0 the
  DLA template is so steep at the lowest k that its `a` and `s` are dominated by the first one or two bins of the grid,
  which lie below the data range (k < 0.001 s/km). Restricted to the data range, they stay below 0.05%.

Two checks show the comparison measures the production templates:

- at the production value, the script reproduces the sightline counts and per-class P1D of the emulator training cache
  to rounding, at every redshift above;
- its blocked computation reproduces the whole-array production functions to rounding, checked on the first 40000
  skewers of each snapshot at the production value.

**What this means for users.** Over the tested domain, moving `tau_threshold` from 100 to any of the tested values 30,
50, 200 and 300 changes the HCD class templates by a small amount, largest for LLS relative to its own template and
within local k ranges. Outside the tested domain nothing was measured:

- other simulations: the effect is expected to be similar, because the threshold acts on the native optical depth and
  the mean-flux rescaling already makes the effective threshold vary by about a factor of 2 between simulations; this
  is an argument, not a test;
- the high-resolution simulations, other mean-flux scalings, and redshifts outside 2.2 to 4.6 were not tested.

Values far outside 30 to 300 were not tested either:

- much lower, a system's run of pixels takes in more of the surrounding forest absorption;
- much higher, the integral loses more of each system's wings, and above a few thousand the weakest LLS (line-centre
  optical depth about 4000) are missed entirely.

Rerun the check before relying on such a value. The built-in default of `AbsorberConfig.tau_threshold` (1.0) is far
outside this range, so always run the pipeline with `config/default.yaml` or a copy of it.

## Reproducing the check

`scripts/tau_threshold_sensitivity.py` reruns the check on any raw fake_spectra file. It needs fake_spectra, so run it
in the environment that builds the emulator cache. Its docstring gives an example command.

What it does:

- builds the catalogue at several thresholds on the same skewers;
- runs the production template path for each threshold: the per-class P1D of
  `hcd_analysis.priya_p1d.compute_tier_c_p1d` on the PRIYA-filtered and unfiltered optical depths, collapsed to the
  four classes as the emulator cache loader does.

What it reports, for every threshold and class:

- the absorber count ratio;
- the template amplitude ratio `a`, the least-squares scaling of the template at the reference threshold;
- the shape residual `s`, the root-mean-square change of the template after removing that scaling, relative to the
  template's root-mean-square;
- the largest template change at 0.001 <= k < 0.01 s/km (the low-k end of the data range), relative to the template's
  largest absolute value there;
- the largest fractional change of the clean power.

How it reads the result, from `a`, `s` and the clean change at the headline thresholds (50 and 200):

- **negligible:** every `abs(a - 1) < 0.05`, every `s < 0.05` and every clean change below 1%;
- **modest:** every `abs(a - 1) <= 0.29` and every `s <= 0.10`;
- **important:** otherwise.

0.29 is the LLS fractional amplitude prior width (0.287) that the code uses on its DESI and eBOSS likelihood paths
(`HCD_LLS_SURVEY_FRAC_SIGMA` in `hcd_analysis/emulator/inference.py`); the rule was fixed before the check was run.
The per-class readings apply the same thresholds to `a` and `s` of that class. The rule's labels cover only these
global measures at thresholds 50 and 200, not the local changes in k; the conclusion of this page is "small over the
tested setup", as stated above.

Its two built-in checks:

- `--self-check N` compares its blocked computation with the whole-array production functions on the first N skewers
  of each input. Once any input has a self-check (always, with `--self-check`), every input needs a passed one at the
  reference threshold; a failed or missing self-check makes the reading "invalid".
- `--anchor-cache` compares the production value with the emulator training cache.

A comparison that is undefined at a headline threshold or the reference (a class or the clean set without sightlines,
possible in a quick look with `--n-skewers`) also makes the reading "invalid". Any invalid reading sets `valid: false`
in `summary.json` and exits with status 2.

## Run settings

The script records its settings in `run_settings.json` in its output directory. A rerun into that directory with any
setting the results depend on changed is refused, so results made with different settings never mix.

The pipeline does the same at its output root:

- a rerun of `hcd` with another `tau_threshold`, or any other product setting, into an existing output root is refused
  before anything is read or written;
- an output root that already holds any file but no record is refused too, because the settings of what it holds
  are unknown. Use a new output root.
