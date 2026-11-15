# API reference

The modules a user of a released model needs. The package's other modules (the simulation pipeline, training,
inference and the validation tools) are documented in their docstrings and are not part of this reference, except
the data-leg loaders below, which are listed as a research interface.

## The predictor

The supported public interface of this release. `predict` and `predict_many` are methods of a loaded model
(`Bundle`, or `ResearchPredictor` for a research model); `load_published`, `load_release`, `load_bundle` and
`load_research_bundle` return one. The loaders, `predict`, `predict_many`, `add_hcd`, `Prediction` and the helper
functions have examples in their docstrings, and a test runs them (`tests/test_docstring_examples.py`).

```{eval-rst}
.. autoapimodule:: hcd_analysis.predictor
   :members:
   :exclude-members: Bundle, ResearchPredictor

.. autoapiclass:: hcd_analysis.predictor.Bundle
   :members:
   :inherited-members:

.. autoapiclass:: hcd_analysis.predictor.ResearchPredictor
   :members:
   :inherited-members:
```

## From dN/dX to sightline fractions

How a model's incidence dN/dX becomes the sightline fractions `w_c` that `add_hcd` mixes with. A loaded model
applies it through `emu.wc_map.w_c(dndx, z)`.

```{eval-rst}
.. autoapimethod:: hcd_analysis.emulator.wc_map.WcMap.w_c
.. autoapifunction:: hcd_analysis.emulator.dndx_wc.w_c_from_mu
.. autoapifunction:: hcd_analysis.emulator.dndx_wc.w_c_corrected
.. autoapifunction:: hcd_analysis.emulator.dndx_wc.dndx_powerlaw
```

## The release format loaders

The loaders of `hcd_analysis.emulator.safe_h5`; the format itself is described in
[the release format page](release_format_h5.md).

```{eval-rst}
.. autoapifunction:: hcd_analysis.emulator.safe_h5.load_release
.. autoapifunction:: hcd_analysis.emulator.safe_h5.load_research
.. autoapifunction:: hcd_analysis.emulator.safe_h5.load_ensemble
.. autoapifunction:: hcd_analysis.emulator.safe_h5.load_bundle
.. autoapifunction:: hcd_analysis.emulator.safe_h5.verify_manifest
.. autoapiclass:: hcd_analysis.emulator.safe_h5.ReleaseBundle
   :members:
.. autoapiclass:: hcd_analysis.emulator.safe_h5.ResearchBundle
   :members:
.. autoapiclass:: hcd_analysis.emulator.safe_h5.MemberMetas
   :members:
.. autoapiexception:: hcd_analysis.emulator.safe_h5.UnanchoredBundleWarning
```

## Runtime data inputs

Where the package finds the observed P1D files and how it checks them: an explicit path, else the product's
environment variable, with every file verified by sha256 before it is read. The variables and folders are listed in
the repository file `docs/models_and_data.md`; the product ids are in the module docstring of
`hcd_analysis.runtime_data`.

```{eval-rst}
.. autoapifunction:: hcd_analysis.runtime_data.resolve_file
.. autoapifunction:: hcd_analysis.runtime_data.resolve_dir
.. autoapifunction:: hcd_analysis.runtime_data.product_file
.. autoapifunction:: hcd_analysis.runtime_data.product_dir
.. autoapifunction:: hcd_analysis.runtime_data.show_loads
.. autoapiclass:: hcd_analysis.runtime_data.LoadRecord
.. autoapiexception:: hcd_analysis.runtime_data.DataProductError
.. autoapiexception:: hcd_analysis.runtime_data.DataProductConflict
```

## Data-leg loaders (research interface)

**Outside the supported scope of this release.** These functions of `hcd_analysis.emulator.data_likelihood` load a
measured P1D with its cuts and covariance for the research likelihood. They are listed here because they are how the
verified data files are read. No fit runs from this repository at this commit (README, "Scope and limitations of
this release"). Their full docstrings are in the source.

```{eval-rst}
.. py:currentmodule:: hcd_analysis.emulator.data_likelihood

.. py:function:: load_eboss_leg(npz_path=None, *, z_lo=2.2, z_hi=4.6, k_min=0.0, k_max=None, metals_on=True, resolution_on=False, mf_floor_on=False, dla_forward_frac=EBOSS_DLA_FORWARD_FRAC, add_cv_floor=None, resolution_float=False, resolution_coherent=False, resolution_coh_amp=1.0)

   Load the eBOSS DR14 P1D (Chabanier et al. 2019) as a ``DataLeg``. The file is the runtime data product
   ``eboss_dr14_p1d``: ``npz_path`` when given, else ``$HCD_P1D_DATA_DIR/eboss_dr14_p1d/``, verified by sha256.

.. py:function:: load_desi_leg(npz_path=None, *, z_lo=2.2, z_hi=4.2, k_min=DESI_KMIN, metals_on=True, resolution_on=False, add_cov_diag_inflation=True, mf_floor_on=False, use_snr3=None, snr3_stat_inflate=None, add_cv_floor=None, resolution_float=False, resolution_coherent=False, resolution_coh_amp=1.0, dla_cov_reduce=None)

   Load the DESI DR1 P1D as a ``DataLeg``. The file is the runtime data product ``desi_dr1_p1d`` (SNR>1) or
   ``desi_dr1_p1d_snr3``: ``npz_path`` when given, else ``$HCD_P1D_DATA_DIR/desi_dr1_p1d/``, verified by sha256.
   ``use_snr3`` (or the variable ``HCD_DESI_SNR3``) selects the sample; a file of the other sample is refused.

.. py:function:: load_ks_leg(base=None, *, z_lo=2.4, z_hi=4.6, k_max=None, metals_on=False, resolution_on=False, mf_floor_on=True, resolution_float=False)

   Load the KODIAQ-SQUAD conservative-mode P1D as a ``DataLeg``. The files are the runtime data product
   ``ks_p1d_conservative``: the directory ``base`` when given, else ``$HCD_LYAEMU_PARENT/lyaemu/data/kodiaq_squad/``,
   verified by sha256. ``k_max`` must be given explicitly.

.. py:class:: DataLeg

   One measured P1D after the cuts, ordered by redshift and then by k: ``z`` (the kept redshifts) and, per row,
   ``z_row``, ``k``, ``P_data`` and the covariance ``C_data``, with the survey's flags (``metals_on``,
   ``dla_forward_frac`` and others). A leg returned by a loader also carries ``data_products``, one
   ``runtime_data.LoadRecord`` per verified file it read.
```

## Lineage records

```{eval-rst}
.. autoapimodule:: hcd_analysis.lineage
   :members:
```

## The accepted model versions

The trained models come in the companion package `hcd-priya-models`, which `hcd-priya` requires. `load_published`
(above) loads a model from that package only against the digest listed here. No model is published yet.

```{eval-rst}
.. autoapimodule:: hcd_analysis.published_models
   :members:
```

## Checking a release from a fresh install

`python -I -m hcd_analysis.check_release --published` (or `RELEASE_DIR RELEASE_SHA256` for a release directory)
loads a model in an environment that has only the installed packages, predicts, and compares with reference
predictions published beside the release.

```{eval-rst}
.. autoapimodule:: hcd_analysis.check_release
   :members:
```

## The model registry

A provisional module that fetches, verifies and caches model files by URL. Nothing in the package uses it: the
models come in the model package.

```{eval-rst}
.. autoapimodule:: hcd_analysis.registry
   :members:
```

## Run-settings records

```{eval-rst}
.. autoapimodule:: hcd_analysis.run_settings
   :members:
```

## The `hcd` command

```{eval-rst}
.. autoapimodule:: hcd_analysis.cli.run
```
