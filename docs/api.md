# API reference

The modules a user of a released model needs. The package's other modules (the simulation pipeline, training,
inference and the validation tools) are documented in their docstrings and are not part of this reference.

## The predictor

```{eval-rst}
.. autoapimodule:: hcd_analysis.predictor
   :members:
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

## Lineage records

```{eval-rst}
.. autoapimodule:: hcd_analysis.lineage
   :members:
```

## The model registry

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
