"""Integration checks of the release MF on the real data: NOT run in CI. They run only when the environment variable
``HCD_MF_INTEGRATION_CONFIG`` names a JSON configuration kept with the private release records, outside this
repository; no real path, simulation name or digest is written here. Run locally, one file per process, under the
shared lock:

    HCD_MF_INTEGRATION_CONFIG=/path/to/mf_integration_config.json python -B -m pytest -q tests/integration/<this file>

Configuration keys (absolute paths):
  ``gate_d_dir``          the historical gate D directory (its arrays and its all-pairs product);
  ``lf_cache_gate_d``     the LF cache gate D matched rows on; ``hr_cache`` the HR cache;
  ``composition``         {"id", "excluded_sims", "ruling_ids"} of the release composition (one excluded simulation
                          for the fold identity below);
  ``backbone_dir``, ``backbone_manifest_sha256``  the release LF backbone bundle;
  ``mf_record_dir``, ``mf_record_sha256``         the built release MF record (optional: absent until it is built).

1. **Fold identity.** The composition's tables, fitted through ``mf_release.select_train_rows`` on gate D's measured
   targets, are bit for bit gate D's leave-one-simulation-out fold of the excluded simulation; the same reconstruction
   of the fitting inputs reproduces the historical all-pairs product bit for bit. Only TABLES are compared: no MF fit
   is evaluated against real HR truth here (no residual or accuracy statistic is computed, read or printed: the
   validation thresholds are not frozen yet, PI ruling 2026-11-06b D1).
2. **Reload.** The built release record reloads bound to the release backbone and the predictor applies it.
"""

import json
import os

import numpy as np
import pytest

ENV = "HCD_MF_INTEGRATION_CONFIG"


def _cfg():
    p = os.environ.get(ENV)
    if not p:
        return None
    if not os.path.isfile(p):
        raise pytest.UsageError(f"{ENV} names {p!r}, which is not a file (unset it to skip)")
    with open(p) as fh:
        return json.load(fh)


CFG = _cfg()
pytestmark = pytest.mark.skipif(
    CFG is None, reason=f"MF integration data not configured ({ENV} unset)"
)


def _rows_light(path):
    """The row keys and encoder inputs of a cache read dataset by dataset (no spectra in memory): params, z_grid,
    alpha_idx, sim_name, x = [normalize_params(params), z_unit] and tau0 = -log(target_F), computed exactly as
    ``data._assemble`` computes them."""
    import h5py
    from hcd_analysis.emulator.data import Z_LIMITS, normalize_params

    with h5py.File(path, "r") as h:
        params = h["params"][:].astype(np.float64)
        z = h["z_grid"][:].astype(np.float64)
        a = h["alpha_idx"][:].astype(np.int32)
        sim = np.array([s.decode() if isinstance(s, bytes) else s for s in h["sim_name"][:]])
        tf = h["target_F"][:].astype(np.float64) if "target_F" in h else None
    zu = (z - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])
    out = dict(
        params=params,
        z_grid=z,
        alpha_idx=a,
        sim_name=sim,
        x=np.concatenate([normalize_params(params), zu[:, None]], axis=1),
    )
    if tf is not None:
        out["tau0"] = -np.log(tf)
    return out


def test_composition_tables_are_gate_d_fold_bit_for_bit():
    import hcd_analysis.emulator  # noqa: F401
    from hcd_analysis import lineage as L
    from hcd_analysis.emulator import mf_modes as MM
    from hcd_analysis.emulator import mf_release as MR

    gd = CFG["gate_d_dir"]
    tables6, _, prov6 = MM.load_mode_mf(os.path.join(gd, "mf_modes_all6.npz"))
    assert L.sha256_file(CFG["lf_cache_gate_d"]) == prov6["lf_cache_sha256"]
    assert L.sha256_file(CFG["hr_cache"]) == prov6["hr_cache_sha256"]
    with np.load(os.path.join(gd, "gate_d_arrays.npz"), allow_pickle=False) as f:
        arr = {
            k: np.asarray(f[k]) for k in ("g_measured", "sim", "z", "alpha_idx")
        }  # never the residual arrays
    lf, hr = _rows_light(CFG["lf_cache_gate_d"]), _rows_light(CFG["hr_cache"])
    pairs = MM.match_pairs(lf, MR.hr_view(hr))
    h_rows, l_rows = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    assert len(pairs) == arr["g_measured"].shape[0]
    t = dict(
        x=lf["x"][l_rows],
        tau0=lf["tau0"][l_rows],
        alpha_idx=lf["alpha_idx"][l_rows],
        sim=lf["sim_name"][l_rows].astype(str),
        g=arr["g_measured"],
    )
    assert np.array_equal(t["sim"], arr["sim"].astype(str)) and np.array_equal(
        t["alpha_idx"], arr["alpha_idx"]
    )
    assert np.array_equal(hr["z_grid"][h_rows], arr["z"])
    # the reconstruction of gate D's fitting inputs reproduces its all-pairs product bit for bit
    all6 = MM.fit_mode_mf(t)
    for k in MM.TABLE_KEYS:
        assert np.asarray(all6[k]).tobytes() == np.asarray(tables6[k]).tobytes(), k
    comp = CFG["composition"]
    (s,) = comp["excluded_sims"]
    rows, train = MR.select_train_rows(t, exclude_sims=comp["excluded_sims"])
    new = MM.fit_mode_mf(t, train_rows=rows)
    fold = MM.fit_mode_mf(
        t, train_rows=np.where(t["sim"] != s)[0]
    )  # scripts/run_gate_d.py, the fold of s
    for k in MM.TABLE_KEYS:
        assert np.asarray(new[k]).tobytes() == np.asarray(fold[k]).tobytes(), k
    assert s not in train


def test_release_record_reloads_bound_to_the_release_backbone():
    if not CFG.get("mf_record_dir"):
        pytest.skip("the release MF record is not built yet (X1)")
    import hcd_analysis.predictor as hp
    from hcd_analysis.emulator import mf_release as MR

    rec = MR.load_mf_record(
        CFG["mf_record_dir"],
        expected_record_sha256=CFG["mf_record_sha256"],
        backbone_manifest_sha256=CFG["backbone_manifest_sha256"],
        composition=CFG["composition"],
    )
    assert rec.kind == "release" and rec.record["fit_code"]["dirty"] is False
    emu = hp.load_bundle(
        CFG["backbone_dir"],
        CFG["backbone_manifest_sha256"],
        mf_dir=CFG["mf_record_dir"],
        mf_record_sha256=CFG["mf_record_sha256"],
    )
    assert emu.provenance["mf"]["record_sha256"] == CFG["mf_record_sha256"]
