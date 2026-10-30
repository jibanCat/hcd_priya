"""The dN/dX-to-w_c map as a stamped product (hcd_analysis/emulator/wc_map.py) and the explicit map argument of
dndx_wc. Synthetic caches only (tests/emulator/_fixture.py), made-up values."""
import hashlib
import importlib.util
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis import lineage as L
from hcd_analysis import tau0_cache as T
from hcd_analysis.catalog import FINDER_ID
from hcd_analysis.emulator import dndx_wc as D
from hcd_analysis.emulator import wc_map as W
from tests import tau0_fixtures as tf
from tests.emulator._fixture import write_synthetic_cache_v34

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def cache(tmp_path, monkeypatch):
    tf.clean_code(monkeypatch)
    p = tmp_path / "cache_v34.h5"
    write_synthetic_cache_v34(p, n_sims=3, snaps_per_sim=4)
    return p


def _load(p):
    from hcd_analysis.emulator.data import load_cache
    return load_cache(p)


# ------------------------------------------------------------------------------------- the estimators are the originals
def test_the_delta_c_estimator_is_calibrate_delta_c(cache, monkeypatch):
    d = _load(cache)
    spec = importlib.util.spec_from_file_location("calibrate_delta_c", os.path.join(REPO, "scripts",
                                                                                     "calibrate_delta_c.py"))
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    monkeypatch.setattr(cal, "load_cache", lambda path: d)
    monkeypatch.setattr(cal.glob, "glob", lambda pattern: ["one_shard"])
    z0, wc0, wm0, sims0, _ = cal.collect("one_shard")
    c0, delta0, std0, mx0 = cal.fit(z0, wc0, wm0, deg=2)
    z1, wc1, wm1, sims1 = W.delta_c_blocks(d)
    c1, delta1, std1, mx1 = W.fit_delta_c(z1, wc1, wm1)
    for a, b in ((z0, z1), (wc0, wc1), (wm0, wm1), (c0, c1), (delta0, delta1), (std0, std1), (mx0, mx1)):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    assert list(sims0) == list(sims1)
    doc = W.refit(d)
    assert np.asarray(doc["delta_c"]["coeffs"]).tobytes() == c0.tobytes()
    assert doc["delta_c"]["n_blocks"] == z0.size and doc["delta_c"]["n_sims"] == 3


def test_the_xbar_estimator_is_hcd_xbar_polyfit(cache):
    from hcd_analysis.emulator.closure_legb import hcd_xbar_polyfit
    d = _load(cache)
    ref = hcd_xbar_polyfit(d)
    got, n = W.fit_xbar(d)
    assert got.dtype == ref.dtype and got.tobytes() == ref.tobytes()
    zg = np.array([d["z_grid"][d["snap_group_idx"] == g][0] for g in range(d["snap_dNdX"].shape[0])])
    assert n == int(np.sum((zg >= 2.1) & (zg <= 4.7)))
    assert np.asarray(W.refit(d)["xbar"]["coeffs"]).tobytes() == ref.tobytes()


# ------------------------------------------------------------------------------------------------ build, read, use
def test_build_writes_a_stamped_map_bound_to_the_cache(cache, tmp_path):
    out = tmp_path / "wc_map.json"
    rec, sha = W.build(cache, out, expected_cache_sha256=L.sha256_file(cache))
    assert sha == L.sha256_file(out)
    m = W.read_wc_map(out, expected_sha256=sha)
    cache_rec = _load(cache)["lineage"]
    inp = L.find_input(m.record, "cache")
    assert inp["sha256"] == L.sha256_file(cache) == m.cache_sha256
    assert inp["finder_ids"] == [FINDER_ID] and inp["release_eligible"] is True
    assert inp["info"] == {"file": "cache_v34.h5", "lineage_record_sha256": L.record_sha256(cache_rec)}
    assert m.record["release_eligible"] and m.record["finder_ids"] == [FINDER_ID]
    assert m.record["code"] == tf.CLEAN and rec == m.record
    # the map's w_c is dndx_wc.w_c_corrected with the map's delta_c and Xbar(z)
    dn, z = np.array([[0.3, 0.08, 0.04], [0.2, 0.05, 0.02]]), np.array([2.6, 4.0])
    ref = D.w_c_corrected(jnp.asarray(dn), jnp.asarray(np.polyval(m.xbar_coeffs, z)), jnp.asarray(z),
                          D.DeltaCMap(m.delta_c_coeffs, *m.delta_c_z_clamp, "x"))
    assert np.asarray(ref).tobytes() == m.w_c(dn, z).tobytes()
    assert np.allclose(m.w_c(dn, z).sum(-1), 1.0, atol=1e-12)


def test_build_refusals(cache, tmp_path, monkeypatch):
    out = tmp_path / "m.json"
    with pytest.raises(W.WcMapError, match="is not the expected"):
        W.build(cache, out, expected_cache_sha256="0" * 64)
    tf.clean_code(monkeypatch, {"commit": "c" * 40, "dirty": True})
    with pytest.raises(W.WcMapError, match="would not be release-eligible.*dirty"):
        W.build(cache, out)
    tf.clean_code(monkeypatch, {"commit": None, "dirty": None})
    with pytest.raises(W.WcMapError, match="commit is unknown"):
        W.build(cache, out)
    assert not out.exists()
    tf.clean_code(monkeypatch)
    W.build(cache, out)
    with pytest.raises(W.WcMapError, match="never overwritten"):
        W.build(cache, out)
    old = tf.write_v33(tmp_path / "v33.h5", [("simA", 16, 3.2, 3, 1)])
    with pytest.raises(T.CacheLineageError, match="cache version '3.3'"):
        W.build(old, tmp_path / "n.json")


def test_the_refit_command(cache, tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("refit_wc_map", os.path.join(REPO, "scripts", "refit_wc_map.py"))
    cmd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmd)
    sha = L.sha256_file(cache)
    assert cmd.main(["--cache", str(cache), "--cache-sha256", sha, "--out", str(tmp_path / "a.json")]) == 0
    assert "release_eligible" in capsys.readouterr().out
    tf.clean_code(monkeypatch, {"commit": "c" * 40, "dirty": True})
    assert cmd.main(["--cache", str(cache), "--cache-sha256", sha, "--out", str(tmp_path / "b.json")]) == 2
    assert not (tmp_path / "b.json").exists()
    with pytest.raises(SystemExit):
        cmd.main(["--cache", str(cache), "--out", str(tmp_path / "c.json")])          # the digest is required


# ------------------------------------------------------------------------------------------------- strict reading
@pytest.fixture
def good(cache, tmp_path):
    out = tmp_path / "good.json"
    W.build(cache, out)
    return json.loads(out.read_bytes())


def _bytes(doc, *, restamp=False):
    if restamp:                                                         # keep the payload digest consistent
        doc = dict(doc, lineage=dict(doc["lineage"], product=dict(doc["lineage"]["product"],
                                                                   payload_sha256=W.payload_sha256(doc))))
    return L.canonical_json(doc).encode("ascii")


def _edit(doc, fn):
    doc = json.loads(json.dumps(doc))
    fn(doc)
    return doc


@pytest.mark.parametrize("fn, restamp, match", [
    (lambda d: d["delta_c"]["coeffs"][1].__setitem__(0, 0.5), False, "payload digest"),
    (lambda d: d["xbar"]["coeffs"].__setitem__(2, 9.0), False, "payload digest"),
    (lambda d: d.__setitem__("format", "hcd_priya.wc_map/0"), True, "format"),
    (lambda d: d.__setitem__("extra", 1), True, "keys"),
    (lambda d: d.pop("lineage"), False, "no lineage record"),
    (lambda d: d["delta_c"].__setitem__("deg", 3), True, "degree"),
    (lambda d: d["delta_c"]["coeffs"].pop(), True, "shape"),
    (lambda d: d["xbar"]["coeffs"].__setitem__(0, True), True, "numbers"),
    (lambda d: d["delta_c"].__setitem__("z_clamp", [5.4, 2.0]), True, "empty"),
    (lambda d: d.__setitem__("class_order", ["clean", "subDLA", "LLS", "DLA"]), True, "class_order"),
    (lambda d: d["lineage"].__setitem__("kind", "tau0_cache"), True, "not wc_map"),
    (lambda d: d["lineage"]["inputs"].append(dict(d["lineage"]["inputs"][0], role="other")), True,
     "exactly the training cache"),
    (lambda d: d["lineage"]["inputs"][0].__setitem__("finder_ids", ["hcd_priya.finder/1"]), True,
     "claims release eligibility"),
    (lambda d: d["lineage"]["code"].__setitem__("dirty", True), True, "claims release eligibility"),
])
def test_edited_or_malformed_maps_are_refused(good, fn, restamp, match):
    with pytest.raises(W.WcMapError, match=match):
        W.parse_wc_map(_bytes(_edit(good, fn), restamp=restamp))


def test_ineligible_digest_json_and_file_refusals(good, tmp_path):
    raw = _bytes(good)
    m = W.parse_wc_map(raw)
    assert m.sha256 == hashlib.sha256(raw).hexdigest()
    with pytest.raises(W.WcMapError, match="is not the expected"):
        W.parse_wc_map(raw, expected_sha256="f" * 64)
    with pytest.raises(W.WcMapError, match="not finite|constant"):
        W.parse_wc_map(raw.replace(b'"deg":2', b'"deg":NaN', 1))
    with pytest.raises(W.WcMapError, match="duplicate key"):
        W.parse_wc_map(raw[:-1] + b',"format":"x"}')
    inel = _edit(good, lambda d: d["lineage"].update(release_eligible=False, ineligible_reasons=["the code tree was "
                                                                                                   "dirty"]))
    with pytest.raises(W.WcMapError, match="not release-eligible"):
        W.parse_wc_map(_bytes(inel))
    assert W.parse_wc_map(_bytes(inel), require_release_eligible=False).record["release_eligible"] is False
    p = tmp_path / "m.json"
    p.write_bytes(raw)
    os.symlink(p, tmp_path / "link.json")
    with pytest.raises(W.WcMapError, match="symlink"):
        W.read_wc_map(tmp_path / "link.json")
    big = tmp_path / "big.json"
    big.write_bytes(b" " * (W.MAX_BYTES + 1))
    with pytest.raises(W.WcMapError, match="exceed"):
        W.read_wc_map(big)


# ------------------------------------------------------------------------------------- dndx_wc's explicit map argument
def test_the_historical_default_is_unchanged_and_labelled():
    old = np.array([[-0.0044601756018632, 0.02941370276821661, -0.04147804040644924],
                    [-7.391610723256435e-06, 0.008526578464137916, -0.03592195195833433],
                    [-0.0006304124952415818, 0.009001921148434061, -0.03672596701410354],
                    [-0.0002204519439413697, 0.0033098470327118175, -0.0245078526302848]])
    z = jnp.array([1.5, 2.0, 3.1, 5.4, 6.0])
    zc = jnp.clip(z, 2.0, 5.4)
    ref = jnp.stack([zc ** 2, zc, jnp.ones_like(zc)], axis=-1) @ jnp.asarray(old).T
    assert np.asarray(D.delta_c(z)).tobytes() == np.asarray(ref).tobytes()
    assert np.asarray(D.delta_c(z, D.HISTORICAL_DELTA_C)).tobytes() == np.asarray(ref).tobytes()
    assert np.asarray(D._DELTA_C_COEFFS).tobytes() == old.tobytes()
    assert "historical" in D.HISTORICAL_DELTA_C.label and (D.Z_FIT_LO, D.Z_FIT_HI) == (2.0, 5.4)


def test_an_explicit_map_is_used_clamped_and_traceable():
    m = D.DeltaCMap(np.array([[0.0, 0.01, -0.02], [0.001, 0.0, 0.01], [0.0, -0.002, 0.0], [0.0, 0.0, 0.03]]),
                    2.2, 4.6, "synthetic")
    z = jnp.array([2.0, 2.2, 3.0, 4.6, 5.4])
    zc = np.clip(np.asarray(z), 2.2, 4.6)
    want = np.stack([np.polyval(m.coeffs[c], zc) for c in range(4)], axis=-1)
    got = np.asarray(D.delta_c(z, m))
    assert np.allclose(got, want, rtol=1e-14, atol=1e-16) and not np.allclose(got, np.asarray(D.delta_c(z)))
    assert np.array_equal(got[0], got[1]) and np.array_equal(got[-1], got[-2])        # clamped at the map's range
    dn, xb = jnp.array([[0.3, 0.1, 0.05]] * 5), jnp.full((5,), 0.4)
    eager = D.w_c_corrected(dn, xb, z, m)
    jitted = jax.jit(lambda a, b, c: D.w_c_corrected(a, b, c, m))(dn, xb, z)         # the map closed over
    assert np.allclose(np.asarray(jitted), np.asarray(eager), rtol=1e-14, atol=0)
    per = jax.vmap(lambda a, b, c: D.w_c_corrected(a, b, c, m))(dn, xb, z)
    assert np.allclose(np.asarray(per), np.asarray(eager), rtol=1e-14, atol=0)
    g = jax.grad(lambda a: D.w_c_corrected(a, xb[0], z[2], m)[3])(dn[0])
    assert np.all(np.isfinite(np.asarray(g))) and np.any(np.asarray(g) != 0)
    a = D.alpha_from_dndx_law(jnp.array([0.02, 0.01, 0.005]), jnp.array([2.0, 2.0, 2.0]), xb[0], 3.0, m)
    back = D.alpha_to_dndx_exact(np.asarray(a), np.asarray(xb[0]), 3.0, delta_map=m)
    assert np.allclose(back, 0.02 * 4.0 ** 2.0 * np.array([1.0, 0.5, 0.25]), rtol=1e-10)
    with pytest.raises(ValueError, match=r"shape \(4, 3\)"):
        D.delta_c(3.0, D.DeltaCMap(np.zeros((4, 4)), 2.0, 5.4, "bad"))
