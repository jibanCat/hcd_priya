# Models and data: what is distributed, the release format, and the runtime data inputs

This page holds the detail that the [README](../README.md) points to. The first sections were part of the README
until 2026-11 and are kept word for word.

The repository ships the code, not the data or the trained models. To set
expectations:

- **The raw catalogs and skewers are not distributed here.** The PRIYA τ skewers
  and the per-(sim, snap) absorber catalogs are large and live on the cluster;
  they are gitignored (`outputs/`, `*.h5`). Building them is a Great Lakes job.
- **The full τ₀ training cache is not distributed here.** The v3.3 LF cache,
  `hcd_analysis/_emulator_data/observables_tau0_lf.h5`, is 1.1 GB and gitignored.
- **The trained checkpoints are not distributed here.** A checkpoint bundle is
  only ~6 MB, but its binary leaves (`checkpoints/*.eqx`, `*.norm.pkl`,
  `*.meta.json`) are gitignored, so a usable checkpoint is not present in a fresh
  clone either.

**Runtime data inputs.** The likelihood's observed P1D legs and the historical multi-fidelity resolution correction
are read from files the package does not ship. Each loader takes an explicit path; without one it reads the directory
an environment variable names, and with neither it stops and names the variable. There is no built-in path. Every
file is checked against the sha256 the package records before it is read, and a different file is refused
(`hcd_analysis.runtime_data`). An empty path, and a variable that is set but empty, are refused too. For DESI,
`use_snr3` (or `HCD_DESI_SNR3=1`) chooses the SNR>3 sample and an explicit path says where it is: a file of the other
sample is refused, never swapped. Each loaded leg records what it read (`leg.data_products`: the product, its version
and sha256, how the file was found and its path), and the likelihood's command-line entry points print one line per
verified file on stderr; in your own script, call `hcd_analysis.runtime_data.show_loads()` for the same lines.

| Variable | Holds |
|---|---|
| `HCD_P1D_DATA_DIR` | `desi_dr1_p1d/` (DESI DR1 P1D, SNR>1 and SNR>3 samples) and `eboss_dr14_p1d/` (eBOSS DR14 P1D) |
| `HCD_LYAEMU_PARENT` | `lyaemu/data/kodiaq_squad/` (KODIAQ-SQUAD P1D, conservative mode) |
| `HCD_KS_BASEDIR` | `res_corr/` (the historical multi-fidelity layer's resolution-correction table) |
| `HCD_LF_RAW_DIR`, `HCD_HR_RAW_DIR` | the LF and HiRes simulation outputs, used when the config file gives no `data_root` or `hires_data_root` |

What you need to run the emulator's low-level forward model yourself is modest:
a current-format trained checkpoint plus at least a slice of the current-format
τ₀ cache (one fold is enough to exercise the forward model on real cache rows);
neither is public. You do not need the full 1.1 GB cache or a cluster for that.
The likelihood needs more, and no inference driver runs at this commit (see
[Scope and limitations of this release](../README.md#scope-and-limitations-of-this-release)).

A data and model release is planned but not yet available (work in progress). The
natural home for the one-fold cache slice and a checkpoint is a dedicated data
repository, which the PI (M. Ho) will create; it is deferred for now. Until then,
request the checkpoint and a cache slice from the PI (mfho@umich.edu) or open an
issue. Please do not commit any large data or model artefacts to this code
repository.

### Release format of the emulator (HDF5 + JSON)

Emulator members are exported in a pickle-free, single-fidelity format, `hcd_priya.h5/2` (scope below; the
multi-fidelity correction is a separate record, see "LF+MF prediction" below): per member one HDF5 file (the network
weights as datasets named by their JAX tree paths, and the output normalizer) and one JSON sidecar (the meta and the
file's sha256), plus the map from absorber incidence (dN/dX) to the class weights w_c, and a bundle `manifest.json`
whose lineage block names the training cache,
its catalogues and the code. Loading builds the network from code, checks every array by name, shape and dtype,
verifies the sha256 before the HDF5 library parses the file, refuses a bundle whose lineage it cannot establish, and
never unpickles anything:

```python
from hcd_analysis.emulator import safe_h5
ensemble, metas, norm = safe_h5.load_ensemble("path/to/bundle", expected_manifest_sha256="<published digest>")
```

Most users load through the public predictor instead, which wraps this call and refuses an unanchored bundle:

```python
import hcd_analysis.predictor as hp
emu = hp.load_bundle("path/to/bundle", "<published digest>")
```

`load_bundle` serves **production** models only: every member must declare training on every simulation of its
training cache (verified against the cache at export and acceptance), and the bundle's w_c map must be fitted on the
whole cache. A bundle that declares fold or research members, or carries a fold map, is refused with a coded reason
(for example `ALL_SIMS_NOT_TRUE`, `RSV_KEY:recipe.holdout_sims`, `SIM_SET:recipe.train_sims` for a simulation list
under any key, or `WCMAP_NOT_FULL_CACHE`). Cross-validation folds
and other **research** models travel in research bundles (`scripts/export_research_bundle.py`, which needs a
repository checkout and refits the w_c map on the training simulations only) and load through their own explicit
path, never as production:

```python
rp = hp.load_research_bundle("path/to/research_bundle", "<its digest>", purpose="evaluate fold 3")
pred = rp.predict(params, 3.0, tau_eff=0.36)
pred.flags.get("research_model", False)             # True here; production predictions carry no such key
```

Arrays returned by `hp.add_hcd` do not carry the flag: keep the `Prediction` (or its `provenance`) with them.

**Scope: single fidelity.** An `hcd_priya.h5/2` bundle holds the members of one training cache and the w_c map fitted
to it, and no multi-fidelity (LF to HR) correction: loaded alone, a bundle trained on the low-fidelity cache gives
low-fidelity predictions; each prediction's labels say so (`product`, `mf_correction`). The multi-fidelity correction
is not part of this format: it is a separate MF record bound to one bundle, and an LF+MF release publishes the two
under one release manifest (next section). An `hcd_priya.h5/2` bundle alone is therefore not the complete release
model.

The files open with standard tools (`h5ls -r member.h5`). Contract and full usage: `docs/release_format_h5.md`; a
step-by-step guide to production and research bundles, with executable examples:
[`docs/tutorials/production_and_research_bundles.md`](tutorials/production_and_research_bundles.md).

### LF+MF prediction (the multi-fidelity correction)

The low-fidelity (LF) emulator predicts the per-class P1D of the low-resolution PRIYA simulations. A multi-fidelity
(MF) record, fitted on matched high- and low-resolution simulation pairs, calibrates it to high resolution:
P_MF = P_LF exp(g(z, tau0, k)), where g does not depend on the cosmological parameters and is bound to exactly one LF
backbone bundle. The release MF record has been produced, and its grouped (K-fold) cross-validation results are
under review; no MF is released before that review is complete. The
calls below show the interface only; every path and digest in them is a placeholder.

```python
import hcd_analysis.predictor as hp
emu = hp.load_release("path/to/release", "<published release sha256>")    # release.json names both parts
# the same, with the two digests given directly:
emu = hp.load_bundle("path/to/release/backbone", "<backbone manifest sha256>",
                     mf_dir="path/to/release/mf", mf_record_sha256="<MF record sha256>")
pred = emu.predict(params, 3.0, tau_eff=0.36)                 # LF+MF: the default once an MF record is loaded
lf = emu.predict(params, 3.0, tau_eff=0.36, fidelity="LF")    # the LF backbone alone, bit for bit
pred.flags["mf_applied"], pred.provenance["mf"]["record_sha256"]
```

`provenance["mf"]` names the record, its composition and the rulings that set it, the fit commit, the mean-flux
convention of its targets (the settled global convention), the HR inputs used, the HR/LF mode-mapping check, the
lower edge of the MF's validation domain (`validated_z_min`, 2.2 by default; the domain edge, not a passed
validation), and the two cache digests with what each establishes:
the LF cache is lineage-verified through the backbone, while the HR cache is digest-anchored only and never
lineage-verified. Below the larger of the MF tables' lowest redshift and `validated_z_min` an LF+MF prediction is
refused (`MFRangeError`); ask for `fidelity="LF"` there. A bundle loaded without an MF record predicts exactly as before, and `fidelity="LF+MF"` on it
raises `MFUnavailableError`. Details: `docs/release_format_h5.md`, sections "The multi-fidelity record" and "The
release manifest".

**The incidence.** The record above corrects the class spectra. A release may also name, as an optional part of its
manifest, an incidence MF record: a fixed table `R_c(z)`, the mean HR/LF ratio of dN/dX per HCD class and redshift over the
matched simulation pairs. With it, an LF+MF prediction returns `dndx = dndx_LF * R_c(z)`, and `w_c` is the shipped
map of that corrected `dndx`; `pred.flags["incidence_mf_applied"]` is then True and `provenance["incidence_mf"]`
names the record, its two inputs and the HR simulations it was fitted on. A release without that part has no such record:
its `dndx` and `w_c` are the LF model's, as before, and it has no `cddf`. With the record a prediction also
carries the column-density distribution `pred.cddf` at the bin centres `pred.log_nhi`, with NaN in every bin where
no HR/LF simulation pair has absorbers at any tabulated redshift and, under LF+MF, where none has them at the
redshift used. A record may also carry a held-out error of the dN/dX and the CDDF; a prediction then returns it as
`dndx_err` and `cddf_err`, and an LF+MF prediction without one warns (`HeldOutErrorWarning`). No such record is
released yet. Details:
`docs/release_format_h5.md`, section "The incidence MF record".
