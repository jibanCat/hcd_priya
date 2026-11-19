"""T25 (integration, real data): the h5/2 bundles on the data store under P2 = A3a. The accepted production bundle
loads through every release loader; the three synthetic bundles written before D5 (recipe without all_sims) are
historical: every release loader refuses them by the production rule, the research loader refuses them, and the
development path loads them with a warning."""

import warnings

import pytest

from tests.integration._config import CFG, needs_data, path

pytestmark = needs_data


def test_t25_the_production_bundle_loads():
    import hcd_analysis.predictor as hp
    from hcd_analysis.emulator import safe_h5 as H

    p = CFG["production"]
    rb = H.load_release(path(p["bundle"]), expected_manifest_sha256=p["manifest_sha256"])
    assert (
        rb.kind == "release"
        and rb.release_verified
        and rb.metas.release_verified
        and len(rb.ensemble.members) == 5
    )
    assert hp.load_bundle(path(p["bundle"]), p["manifest_sha256"]).n_members == 5


@pytest.mark.parametrize("i", [0, 1, 2])
def test_t25_a_pre_d5_synthetic_bundle_is_historical(i):
    import hcd_analysis.predictor as hp
    from hcd_analysis.emulator import safe_h5 as H

    b = CFG["historical_synthetic_bundles"][i]
    d, sha = path(b["bundle"]), b["manifest_sha256"]
    for load in (
        lambda: H.load_release(d, expected_manifest_sha256=sha),
        lambda: H.load_ensemble(d, expected_manifest_sha256=sha),
        lambda: H.load_bundle(d, expected_manifest_sha256=sha),
        lambda: hp.load_bundle(d, sha),
    ):
        with pytest.raises(H.SafeFormatError, match="members are not the ones.*ALL_SIMS_NOT_TRUE"):
            load()
    with pytest.raises(H.SafeFormatError, match="not a research bundle"):
        H.load_research(d, expected_manifest_sha256=sha, purpose="integration check")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ens, metas, _ = H.load_ensemble(d, expected_manifest_sha256=None)
    assert (
        [x.category for x in w] == [H.UnanchoredBundleWarning]
        and metas.kind == "dev"
        and len(ens.members) >= 2
    )
