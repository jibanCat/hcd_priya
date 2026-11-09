"""Every caller-given digest below the predictor and release entry points is taken only as an EXACT str of 64
lowercase hex digits, as the predictor and release entry points take theirs. A str subclass can override == and != and
so pass any digest comparison; it is refused, not compared. ``None`` keeps its meaning where a function allows it (an
unanchored read). Synthetic values only."""
import hashlib
import os

import numpy as np
import pytest

from hcd_analysis.emulator import eligibility as E
from hcd_analysis.emulator import loo_matched as LM
from hcd_analysis.emulator import product_manifest as PM
from hcd_analysis.emulator import products as PR
from hcd_analysis.emulator import release_testing as RT
from hcd_analysis.emulator import safe_h5 as H
from hcd_analysis.emulator import safe_io as S
from hcd_analysis.emulator import wc_map as W

EXACT = "an exact str"
WRONG = "e" * 64
GOOD = "0" * 64


class Liar(str):
    """A str subclass that compares equal to everything (and unequal to nothing)."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    __hash__ = str.__hash__


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _forged(right):
    """The values refused in place of a digest: a lying subclass of a WRONG digest, the right digest as np.str_ and
    as a plain subclass (neither is an exact str), and an exact str that is not 64 lowercase hex digits."""
    return [Liar(WRONG), np.str_(right), type("Sub", (str,), {})(right), "Z" * 64]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return RT.write_synthetic_bundle(str(tmp_path_factory.mktemp("b") / "bundle"), n_members=1)


def test_safe_h5_member_load_takes_exact_digests(bundle):
    prefix = bundle.prefixes[0]
    h5, js = _sha(prefix + H.EXT), _sha(prefix + ".json")
    _, meta, _ = H.load_member(prefix, expected_sha256=h5, expected_json_sha256=js)          # exact str: loads
    assert meta["seed"] == 0
    for bad in _forged(h5):
        with pytest.raises(H.SafeFormatError, match=EXACT):
            H.load_member(prefix, expected_sha256=bad, expected_json_sha256=js)
    for bad in _forged(js):
        with pytest.raises(H.SafeFormatError, match=EXACT):
            H.load_member(prefix, expected_sha256=h5, expected_json_sha256=bad)


def test_safe_h5_manifest_and_historical_loaders_take_exact_digests(bundle):
    bd, sha = bundle.bundle_dir, bundle.manifest_sha256
    assert len(H.verify_manifest(bd, expected_sha256=sha)) == 1                              # exact str: verifies
    for bad in _forged(sha):
        with pytest.raises(H.SafeFormatError, match=EXACT):
            H.verify_manifest(bd, expected_sha256=bad)
        with pytest.raises(H.SafeFormatError, match=EXACT):
            H.verify_manifest_historical(bd, expected_sha256=bad)
        with pytest.raises(H.SafeFormatError, match=EXACT):
            H.load_bundle_historical(bd, expected_manifest_sha256=bad, reason="a test")
        with pytest.raises(H.SafeFormatError, match=EXACT):
            H.load_ensemble_historical(bd, expected_manifest_sha256=bad, reason="a test")


def test_safe_io_digests_are_exact(tmp_path):
    p = str(tmp_path / "t.safetensors")
    S.write_safetensors(p, {"a": np.zeros(2)})
    right = _sha(p)
    assert S.read_safetensors(p, right)[0]["a"].shape == (2,)
    for bad in _forged(right):
        with pytest.raises(S.SafeFormatError, match=EXACT):
            S.read_safetensors(p, bad)
    where = str(tmp_path / "absent")
    for bad in _forged(GOOD):
        with pytest.raises(S.SafeFormatError, match=EXACT):
            S.load_member(where, expected_sha256=bad)
        with pytest.raises(S.SafeFormatError, match=EXACT):
            S.load_member(where, expected_json_sha256=bad)
        with pytest.raises(S.SafeFormatError, match=EXACT):
            S.read_manifest(where, bad)
        with pytest.raises(S.SafeFormatError, match=EXACT):
            S.verify_manifest(where, expected_sha256=bad)
        with pytest.raises(S.SafeFormatError, match=EXACT):
            S.load_bundle_historical(where, expected_manifest_sha256=bad, reason="a test")


def test_wc_map_digests_are_exact(bundle, tmp_path):
    path = os.path.join(bundle.bundle_dir, H.WC_MAP_FILE)
    raw, right = open(path, "rb").read(), _sha(path)
    assert W.read_wc_map(path, expected_sha256=right).sha256 == right                      # exact str: reads
    for bad in _forged(right):
        with pytest.raises(W.WcMapError, match=EXACT):
            W.read_wc_map(path, expected_sha256=bad)
        with pytest.raises(W.WcMapError, match=EXACT):
            W.parse_wc_map(raw, expected_sha256=bad)
        with pytest.raises(W.WcMapError, match=EXACT):
            W.build(str(tmp_path / "absent.h5"), str(tmp_path / "m.json"), expected_cache_sha256=bad)
    assert not os.path.lexists(tmp_path / "m.json")


def test_product_loaders_take_exact_digests(tmp_path):
    where = str(tmp_path / "absent.npz")
    for bad in _forged(GOOD):
        with pytest.raises(PR.ProductRefused, match=EXACT):
            PR.load_product(where, "dla_core", cache_sha256=bad)
        with pytest.raises(PR.ProductRefused, match=EXACT):
            PR.load_product(where, "dla_core", cache_sha256=GOOD, inputs={"x": bad})
        for kw in ({"lf_ckpt_sha256": [GOOD, bad]}, {"lf_cache_sha256": bad}, {"hr_cache_sha256": bad}):
            args = dict({"lf_ckpt_sha256": [GOOD], "lf_cache_sha256": GOOD, "hr_cache_sha256": GOOD}, **kw)
            with pytest.raises(PR.ProductRefused, match=EXACT):
                PR.load_mf_product(where, composition="c", **args)
        with pytest.raises(PM.ProductManifestError, match=EXACT):
            PM.verify(where, cache_sha256=bad, ensemble_manifest_sha256=GOOD)
        with pytest.raises(PM.ProductManifestError, match=EXACT):
            PM.verify(where, cache_sha256=GOOD, ensemble_manifest_sha256=bad)


def test_content_checks_and_fold_assembly_take_exact_cache_digests(tmp_path):
    for bad in _forged(GOOD):
        with pytest.raises(E.EligibilityError, match=EXACT):
            E.production_content_check([], [], None, cache_path=str(tmp_path / "absent.h5"),
                                       expected_cache_sha256=bad)
        with pytest.raises(E.EligibilityError, match=EXACT):
            E.research_content_check([], [], None, {}, cache_path=str(tmp_path / "absent.h5"),
                                     expected_cache_sha256=bad)
        with pytest.raises(ValueError, match=EXACT):
            LM.assemble_fold({}, "synthsim0", [], {}, cache_sha256=bad)
