"""scripts/derive_hcd_dndx_corrected.py is a historical derivation: its strict tier reads the cache through the strict
loader and stops when that loader refuses the cache (a pre-lineage one included), never falling back to a pinned
value. Synthetic cache only."""

import importlib.util
import os

import pytest

from hcd_analysis import tau0_cache as TC
from tests import tau0_fixtures as tf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_strict_tier_fails_loudly_on_a_cache_the_release_loader_refuses(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "derive_hcd_dndx_corrected", os.path.join(REPO, "scripts", "derive_hcd_dndx_corrected.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.__doc__.startswith("HISTORICAL")
    old = tf.write_v33(tmp_path / "old.h5", [("simA", 16, 3.0, 3, 1)])
    monkeypatch.setattr(mod.LK, "CACHE_PATH", str(old))
    monkeypatch.setenv("DERIVE_ALPHA_FULL", "1")
    monkeypatch.delenv("DERIVE_ALPHA_FALLBACK", raising=False)
    with pytest.raises(TC.CacheLineageError, match="3.3"):
        mod.deployed_alpha_fn()
