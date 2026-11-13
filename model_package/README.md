# hcd-priya-models

The trained models of [hcd_priya](https://github.com/jibanCat/hcd_priya), the Lyman-alpha forest P1D emulator with
high-column-density absorber classes. This package holds data only. `hcd-priya` depends on it, so installing
`hcd-priya` installs the models too:

```python
import hcd_analysis.predictor as hp

emu = hp.load_published()          # the LF+MF model
# the nine PRIYA parameters; these values are the midpoint of each parameter's training range,
# which is not a fiducial cosmology: put your own here
params = {"ns": 0.925, "Ap": 1.9e-9, "herei": 4.0, "heref": 2.7, "alphaq": 2.15,
          "hub": 0.70, "omegamh2": 0.143, "hireionz": 7.25, "bhfeedback": 0.05}
pred = emu.predict(params, z=3.0, tau0_amp=1.0, dtau0=0.0)
print("shape of pred.P_class, the P1D of clean, LLS, subDLA, DLA:", pred.P_class.shape)
```

```text
shape of pred.P_class, the P1D of clean, LLS, subDLA, DLA: (4, 172)
```

- **What it holds.** The LF+MF release of the emulator (its name in this package is `lfmf`; `load_published()`
  loads it without a name): the low-fidelity backbone bundle, the
  multi-fidelity record and `release.json`, in the pickle-free format described in the hcd_priya documentation.
- **How it is checked.** `hcd-priya` pins the sha256 of the release it accepts. `load_published` loads the files of
  this package only if they verify against that digest, file by file. This package loads nothing itself.
- **How it is built.** The model files are not in git. `python model_package/prepare.py --release RELEASE_DIR
  --release-sha256 SHA256` copies an accepted release in after checking its digest and verifies every file;
  `python -m build --outdir <dir> model_package` then builds the wheel and the source archive. For tests,
  `prepare.py --synthetic` makes a stand-in with untrained networks (version `0+synthetic`, never published), which
  `load_published` loads only when `HCD_PRIYA_ALLOW_SYNTHETIC=1` is set. The docstring of `prepare.py` has the
  details.
- **Versions.** The version of this package is the version of the model bundle it holds. A version of `hcd-priya`
  accepts the model versions it pins; a model package it does not know is refused with a message that names both.
- **Licence.** The model files are released under the MIT licence (see `LICENSE`), like the code.
- **No model is published yet.** The package is built and tested in the hcd_priya repository
  (`model_package/`); the model files are copied in at build time and are not in git.
