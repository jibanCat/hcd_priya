import h5py, numpy as np
from tests.emulator._fixture import write_synthetic_cache

def test_fixture_is_valid_v33(tmp_path):
    path = tmp_path / "obs.h5"
    write_synthetic_cache(path, n_sims=3, snaps_per_sim=2, n_alpha=4, n_k=8)
    with h5py.File(path, "r") as h:
        assert h.attrs["cache_version"] == "3.3"
        R = h["P_tier_p"].shape[0]
        assert R == 3 * 2 * 4
        cnt = h["tier_c_counts"][:].astype(float)          # (R,15)
        Pf = h["P_tier_c_filtered"][:]                      # (R,15,n_k)
        Pp = h["P_tier_p"][:]                               # (R,n_k)
        w = cnt / cnt.sum(1, keepdims=True)
        recon = np.einsum("rc,rck->rk", w, Pf)
        ok = np.isfinite(Pp)
        assert np.allclose(recon[ok], Pp[ok], rtol=0, atol=1e-12)
        assert np.isnan(Pp).any()

from hcd_analysis.emulator.data import load_cache, COARSE_SLICES, _collapse_p1d

def test_empty_class_collapses_to_zero_not_nan():
    # Only the clean class (fine bin 0) is populated; LLS/subDLA/DLA are empty.
    counts15 = np.zeros((1, 15), dtype=np.int64)
    counts15[0, 0] = 100
    p15 = np.zeros((1, 15, 4))                     # finite zeros (cache convention)
    out = _collapse_p1d(p15, counts15)             # (1, 4, 4)
    # Empty LLS class -> 0.0 and finite, NOT NaN (would poison structural sum).
    assert np.isfinite(out[0, 1]).all()
    assert np.allclose(out[0, 1], 0.0)
    # Genuine all-NaN segment (above native Nyquist) -> NaN, for masking.
    p15b = p15.copy()
    p15b[0, 0, :] = np.nan                          # clean class entirely NaN
    outb = _collapse_p1d(p15b, counts15)
    assert np.isnan(outb[0, 0]).all()

def test_load_collapses_15_to_4_and_builds_tau0_masks_weights(tmp_path):
    path = tmp_path / "obs.h5"; write_synthetic_cache(path, n_k=8)
    d = load_cache(path)
    R = d["P_tier_p"].shape[0]
    assert d["P_filt"].shape == (R, 4, 8)
    assert d["delta"].shape == (R, 3, 8)
    assert np.allclose(d["tau0"], -np.log(d["target_F"]))
    assert COARSE_SLICES == (slice(0,1), slice(1,8), slice(8,13), slice(13,15))
    assert d["mask"].shape == (R, 8)
    assert d["mask"].dtype == bool
    assert (d["mask"] == np.isfinite(d["P_tier_p"])).all()
    assert d["inv_nc"].shape == (R, 4)
    assert np.all(np.isfinite(d["inv_nc"]))
