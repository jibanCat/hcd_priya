# X-battery truth-table contract (X-battery-2, 2026-07-24)

Interface contract between the stage-V validation pass (`scripts/analyze_xsel_truth_tables.py`,
built in parallel; it EMITS the table) and the X-battery tooling (`scripts/ks_xsel_arms.py`
loads it; `scripts/run_xsel_shard.py` consumes it for the X1 data-side swap;
`scripts/analyze_xsel.py` consumes it for D, gate power, and the X1b/mask-width overlays).
Design basis: PROPOSAL-extreme-battery-v2 (round-2 revisions 1, 2, 6, 8) + PI decisions of
record #7 execution annex (OQ1/3/9/10 resolutions). Everything here is in the DEPLOYED
convention (round-2 revision 1): global-mean-flux, tau0-rescaled, the deployed KS leg's
native angular k grid (echelle-float window, k_max 0.065). Pilot-convention numbers are
barred from this file's payload.

## File

One `.npz` (numpy savez, no pickled objects except the JSON strings noted below):

    /home/mfho/hcd_priya_notes/docs/superpowers/xsel-truth-artifacts/xsel_truth_tables.npz

Provenance: sha256 of the FILE BYTES is pinned in `scripts/ks_xsel_arms.py`
(`XSEL_TRUTH_TABLE_SHA256`). The pin ships as `None`; after the PI-delegated validation
review (decision record #7, delegated truth approval) the sha is set DELIBERATELY in that
module (runs-branch file). Until then every loader call fails loud (RuntimeError). A missing
file fails loud (FileNotFoundError). A sha mismatch fails loud (AssertionError). The loaded
sha is stamped into every shard pkl and re-asserted at analyzer ingest.

## Required keys

| key | shape/type | meaning |
|---|---|---|
| `leg_k` | (N,) float64 | the deployed KS leg k rows, EXACTLY `leg.k` of the deployed ctx (row-aligned; rows are (z,k) pairs) |
| `leg_z` | (N,) float64 | per-row z, EXACTLY `leg.z[leg.z_idx]` of the deployed ctx |
| `ratio_rows_X1_dla100` | (N,) float64 > 0 | dilution-CORRECTED masked DLA-conditional P1D / clean P1D (the PRIMARY X1 fork, annex OQ1) |
| `ratio_rows_X1b_dla100_diluted` | (N,) float64 > 0 | dilution-INCLUDED fork (trough-fill cache convention); READOUT OVERLAY ONLY, no fits (annex OQ1) |
| `ratio_rows_X2_sub100` | (N,) float64 > 0 | deployed trough-fill subDLA-selected conditional P1D / clean (annex OQ9: deployed convention; window-mask sensitivity documented in the readout) |
| `ratio_rows_X3_lls100` | (N,) float64 > 0 | LLS-selected conditional P1D / clean (highest-class partition, annex OQ10) |
| `convention_json` | 0-d str (JSON dict) | convention stamp; REQUIRED entries must equal `ks_xsel_arms.REQUIRED_CONVENTION` exactly (see below); extra provenance entries allowed |
| `gate_power_json` | 0-d str (JSON dict) | per-arm gate-power inputs (round-2 revision 4b), see below |

Optional keys:

| key | shape | meaning |
|---|---|---|
| `band_lo_rows_X1_dla100`, `band_hi_rows_X1_dla100` | (N,) | wide-mask systematic BAND recomputed on the corrected fork (annex OQ3, overlay only) |
| `meta_json` | 0-d str | free-form provenance (commit, sim list, snapshot list, wall clock) |

## convention_json REQUIRED entries (exact values)

    {"frame": "deployed", "mean_flux": "global", "tau0_rescale": true,
     "k_convention": "leg_native_angular", "x1_fork": "dilution_corrected",
     "x1b_fork": "dilution_included", "x2_trough_fill": "deployed",
     "partition": "highest_class", "selection_unit": "per_sightline_120mpch",
     "mask_width": "nominal_band_overlay", "metal_free_ks": true}

## gate_power_json

One entry per registry arm PLUS the X1b overlay (8 entries total):
`X1_dla100, X1b_dla100_diluted, X2_sub100, X3_lls100, X4_prof, K8a_eps_hi, K8b_eps_lo,
K8c_kap_hi`, each

    {"D": <float > 0>, "sigma_pair_expected": <float > 0>, "p_part1_fail_null": <float in [0,1]>}

- `D`: the arm's displacement for the Part-2 disclosure line, computed BY STAGE V from the
  deployed-convention truth curves (the "D computed at readout from the stamped truth curve"
  object, produced once, PI-reviewed with the validation figures, then frozen in this table).
- `sigma_pair_expected` / `p_part1_fail_null`: the round-2 revision 4b pre-launch statements
  (expected paired-delta sd in sigma_post units at the arm's n, and P(Part-1 fail | zero
  bias)). The analyzer prints them beside the gate lines; if `p_part1_fail_null` is not small
  the n or gate form is revised PRE-registration (revision 4b), so the analyzer flags any
  value > 0.05 loudly.

## Semantics the consumers code to

1. X1 draw mapping ("ratio transport"): the per-mock X1 truth vector is
   `P_swap = truth_on_leg_clean(draw) * ratio_rows_X1_dla100`, i.e. the pinned ratio curve
   transported to each mock's drawn mean-flux/cosmology truth by multiplying the deployed
   clean forward at the draw. This is the theta-continuous form of proposal Sec 8 option (1):
   deployed-consistent, continuous in the truth cosmology, exact pairing with K0. Stage V
   must therefore emit the ratio TO CLEAN at the reference truth, in the deployed convention,
   with any mask-edge artifact regime handled inside the deployed-convention recomputation
   (round-2 revision 8), never extrapolated from the pilot.
2. X4 derived curve: `ratio_X4(z,k) = 1 + f_sel(z) * (ratio_rows_X3_lls100 - 1)` (exact
   mixture identity); computed analyzer-side, NOT stored.
3. K8 arms carry no truth curve here (dN/dX-space displacements through the frozen map);
   only their gate-power entries live in this table.
4. Row alignment is the ONLY grid contract: consumers verify `leg_k`/`leg_z` against the
   built deployed leg to rtol 1e-10 and refuse on mismatch. No interpolation anywhere.

## Change control

Any change to this contract is a coordinated edit of this file, the stage-V emitter, and
`scripts/ks_xsel_arms.py`, followed by a NEW sha pin. The registry signature covers the pin,
so a silently swapped table refuses to pool at the analyzer.
