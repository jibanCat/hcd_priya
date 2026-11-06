"""The dN/dX-to-w_c map as a stamped product (hcd_analysis/emulator/wc_map.py) and the explicit map argument of
dndx_wc. Synthetic caches only (tests/emulator/_fixture.py), made-up values."""
import copy
import hashlib
import importlib.util
import json
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis import lineage as L
from hcd_analysis import tau0_cache as T
from hcd_analysis.catalog import FINDER_ID
from hcd_analysis.emulator import dndx_wc as D
from hcd_analysis.emulator import wc_map as W
from tests import portable_numerics as PN
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
    tf.clean_code(monkeypatch)
    old = tf.write_v33(tmp_path / "v33.h5", [("simA", 16, 3.2, 3, 1)])             # the strict loader refuses it:
    assert cmd.main(["--cache", str(old), "--cache-sha256", L.sha256_file(old),    # refused, exit 2, nothing written
                     "--out", str(tmp_path / "d.json")]) == 2
    assert "refused" in capsys.readouterr().err and not (tmp_path / "d.json").exists()


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


# ---------------------------------------------------------------- the training-set refit (wcmap_refit_subset/1, T33)
@pytest.fixture
def cache5(tmp_path, monkeypatch):
    tf.clean_code(monkeypatch)
    p = tmp_path / "cache5_v34.h5"
    write_synthetic_cache_v34(p, n_sims=5, snaps_per_sim=4)
    return p


# payload sha256 on the 5-simulation synthetic cache (seed 0), computed with the pre-D5 refit and the W-1 reference view
GOLDEN_WCMAP_REFIT = "d3d1100320904150cdf7693bdc2162b36c47d8178cc44b1c892c05c3ade19e57"
GOLDEN_WCMAP_SUBSET = "ceef303edcaf4f39f6cefc8044d33079b2a3dff49e637b3a3c1e3dfdf95f62b6"       # sim2 held out
# the same two payloads at full precision (the pinned stack writes exactly these: their payload sha256 are the two
# digests above, test_the_stored_golden_payloads_are_the_golden_digests); synthetic values, they pin the rules
GOLDEN_PAYLOAD_REFIT = {
    "format": "hcd_priya.wc_map/1", "class_order": ["clean", "LLS", "subDLA", "DLA"],
    "delta_c": {
        "coeffs": [[-0.2558140142630737, 1.086232151951372, -0.8339652484747228],
                   [-3.371583723084258, 18.87252282190424, -23.383002642375782],
                   [0.1943593534564159, -0.9943498618156089, 0.9387341019851997],
                   [0.652787623343305, -2.926487979024196, 2.345262745419201]],
        "resid_std": [0.25508745913109915, 3.204096481046615, 0.3364998879918978, 0.3110236086633414],
        "resid_max": [0.5816143414321119, 11.59961699306672, 0.674562139811513, 1.0622273775829256],
        "mean": [0.20977280732785228, 2.219333981908799, -0.2938344386787291, -0.7202041415743079],
        "deg": 2, "z_clamp": [2.0, 5.4], "n_blocks": 20, "n_sims": 5, "z_min": 2.0,
        "z_max": 3.2},
    "xbar": {"coeffs": [-0.8406544359243013, 4.973241918435401, -6.438738972618132], "deg": 2, "z_fit": [2.1, 4.7],
             "n_groups": 15}}
GOLDEN_PAYLOAD_SUBSET = {
    "format": "hcd_priya.wc_map/1", "class_order": ["clean", "LLS", "subDLA", "DLA"],
    "delta_c": {
        "coeffs": [[-0.2501597781683499, 1.0648171033485048, -0.8428675491290122],
                   [-4.189298425081595, 23.725116478868625, -29.95923699423536],
                   [0.2500145360768185, -1.1937680079433415, 1.0787854707412903],
                   [0.26057575578392755, -1.0962792018178298, 0.2514987355858057]],
        "resid_std": [0.24809156943553037, 3.431741946699929, 0.3401519670599727, 0.18250530775505683],
        "resid_max": [0.6093443588276736, 10.999472369085149, 0.6492028302550328, 0.5034005538877963],
        "mean": [0.18454486352538607, 2.5685488122551874, -0.2849101788167422, -0.785219928884416],
        "deg": 2, "z_clamp": [2.0, 5.4], "n_blocks": 16, "n_sims": 4, "z_min": 2.0,
        "z_max": 3.2},
    "xbar": {"coeffs": [-0.21489274949232198, 1.4614637851215497, -1.5494406419981959], "deg": 2, "z_fit": [2.1, 4.7],
             "n_groups": 12}}

# T31 off the exact replay: per-field tolerances of the payload's computed floats (test_golden_digests_of_the_map_rules
# says why each). The pin parsing and the exact-replay rule live in tests/portable_numerics.py.
WC_MAP_TOLS = {                           # (section, key): (rtol, atol)
    ("delta_c", "coeffs"): (1e-9, 0.0),
    ("xbar", "coeffs"): (1e-9, 0.0),
    ("delta_c", "resid_std"): (1e-12, 0.0),
    ("delta_c", "resid_max"): (1e-10, 0.0),
    ("delta_c", "mean"): (1e-12, 0.0),
}
TOLERANT_FIELDS = tuple(WC_MAP_TOLS)
CONSTRAINTS_FILE = PN.CONSTRAINTS_FILE
PINNED_STACK_ENV = PN.PINNED_STACK_ENV
_requirement_strict = PN._requirement_strict
numpy_pins = PN.numpy_pins
pinned_numpy_versions = PN.pinned_numpy_versions
pinned_stack_declared = PN.pinned_stack_declared
exact_replay = PN.exact_replay


def payload_mismatches(got, want, tols=None):
    """The dotted paths where the w_c map payload ``got`` departs from ``want`` (empty: they agree). The key sets must
    be equal; the computed floats (``WC_MAP_TOLS``: the least-squares outputs and the block means) must have the same
    shape, be finite and agree element by element to their own tolerance (``|got - want| <= atol + rtol |want|``,
    ``portable_numerics.field_mismatches``); every other field (counts, degrees, the clamp and fit ranges, the redshift
    range, the format and class order) must be exactly equal (canonical JSON)."""
    tols = WC_MAP_TOLS if tols is None else tols
    out = []
    if set(got) != set(want):
        out.append("keys")
    for sec in ("delta_c", "xbar"):
        g, w = got.get(sec), want[sec]
        if not isinstance(g, dict) or set(g) != set(w):
            out.append(f"{sec} keys")
            continue
        for k in sorted(w):
            if (sec, k) in tols:
                a, b = np.asarray(g[k], dtype=np.float64), np.asarray(w[k], dtype=np.float64)
                if not np.all(np.isfinite(a)) or PN.field_mismatches({k: a}, {k: b}, {k: tols[(sec, k)]}):
                    out.append(f"{sec}.{k}")
            elif L.canonical_json(g[k]) != L.canonical_json(w[k]):
                out.append(f"{sec}.{k}")
    for k in sorted(set(want) - {"delta_c", "xbar"}):
        if L.canonical_json(got.get(k)) != L.canonical_json(want[k]):
            out.append(k)
    return out


def _w1_reference_fold_map(d, train_sims):
    """W-1's fold_view + fit_fold_map (scripts/w1/common.py at 5067bb4), copied as the reference: (delta_c coeffs,
    xbar coeffs, n_blocks, n_xbar_groups, n_sims)."""
    rk = ("sim_name", "z_grid", "params", "x", "coarse_counts", "w_c_cache", "alpha_idx", "tau0")
    gk = ("snap_dNdX", "snap_total_path_dX", "snap_f_nhi", "snap_n_absorbers")
    sim = np.asarray(d["sim_name"]).astype(str)
    rows = np.flatnonzero(np.isin(sim, [str(s) for s in train_sims]))
    old_g = np.asarray(d["snap_group_idx"])[rows]
    groups = np.unique(old_g)
    view = {k: np.asarray(d[k])[rows] for k in rk if k in d}
    view["sim_name"] = sim[rows]
    view["snap_group_idx"] = np.searchsorted(groups, old_g).astype(np.int32)
    for k in gk:
        if k in d:
            view[k] = np.asarray(d[k])[groups]
    z, wc, wm, sims = W.delta_c_blocks(view)
    coeffs, _delta, _rs, _rm = W.fit_delta_c(z, wc, wm, W.DEG)
    xcf, n_groups = W.fit_xbar(view)
    return np.asarray(coeffs, np.float64), np.asarray(xcf, np.float64), int(z.size), int(n_groups), len(set(sims))


def test_the_full_set_refit_is_the_refit_byte_for_byte(cache5):
    d = _load(cache5)
    sims = sorted(set(d["sim_name"].tolist()))
    assert L.canonical_json(W.refit_subset(d, sims)) == L.canonical_json(W.refit(d))
    assert L.canonical_json(W.refit_subset(d, list(reversed(sims)) + sims[:1])) == L.canonical_json(W.refit(d))


@pytest.mark.parametrize("held", [["sim0"], ["sim1", "sim3"], ["sim4", "sim2", "sim0"]])
def test_the_training_set_refit_is_the_w1_fold_map(cache5, held):
    d = _load(cache5)
    train = sorted(set(d["sim_name"].tolist()) - set(held))
    doc = W.refit_subset(d, train)
    dc, xb, nb, ng, ns = _w1_reference_fold_map(d, train)
    assert np.asarray(doc["delta_c"]["coeffs"]).tobytes() == dc.tobytes()
    assert np.asarray(doc["xbar"]["coeffs"]).tobytes() == xb.tobytes()
    assert (doc["delta_c"]["n_blocks"], doc["xbar"]["n_groups"], doc["delta_c"]["n_sims"]) == (nb, ng, ns)
    assert ns == len(train) and doc["delta_c"]["coeffs"] != W.refit(d)["delta_c"]["coeffs"]


def test_no_held_out_simulation_reaches_the_training_set_refit(cache5):
    d = _load(cache5)
    train = ["sim0", "sim2", "sim3"]
    ref = W.refit_subset(d, train)
    poisoned = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in d.items()}
    held_rows = np.flatnonzero(~np.isin(d["sim_name"], train))
    held_groups = np.unique(d["snap_group_idx"][held_rows])
    poisoned["coarse_counts"][held_rows] = poisoned["coarse_counts"][held_rows] * 7 + 3
    poisoned["w_c_cache"][held_rows] = 0.5
    poisoned["snap_dNdX"][held_groups] *= 11.0
    poisoned["snap_total_path_dX"][held_groups] *= 0.1
    assert L.canonical_json(W.refit_subset(poisoned, train)) == L.canonical_json(ref)
    assert L.canonical_json(W.refit(poisoned)) != L.canonical_json(W.refit(d))
    view = W.training_view(d, train)
    assert set(view["sim_name"].tolist()) == set(train) and np.array_equal(view["source_rows"],
                                                                            np.flatnonzero(np.isin(d["sim_name"],
                                                                                                   train)))


def test_the_training_set_refit_refuses_an_empty_or_unknown_set(cache5):
    d = _load(cache5)
    for bad in ([], ["sim9"], ["sim0", "simX"]):
        with pytest.raises(W.WcMapError, match="non-empty subset"):
            W.refit_subset(d, bad)


def test_golden_digests_of_the_map_rules(cache5):                                 # T31
    """The map rules on cache5, against the goldens.

    Always: the payload is compared field by field with the stored golden payloads (``payload_mismatches``): every
    count, degree, clamp and fit range, the redshift range, the format and the class order exactly; the computed
    floats to their own tolerance (``WC_MAP_TOLS``). Only in an exact replay (``exact_replay()``: HCD_EXACT_REPLAY=1
    with HCD_PINNED_STACK=1 and numpy, jax and jaxlib at their pins, never on a GitHub runner) also the payload sha256
    against the golden digests, bit level. The numpy version alone no longer selects the digests: the GitHub job
    "Python 3.11, newest dependencies" ran the pinned numpy on another CPU and failed them (run 42143147836). Its
    digest is reproduced here by OpenBLAS's Haswell or Zen kernels (OPENBLAS_CORETYPE), the ones a CPU without
    AVX-512 gets.

    The tolerances. The w_c map enters the forward as w_c = w_M0 (1 + delta_c(z)) and through Xbar(z), the mean path
    per sightline; the scientific scale of delta_c(z) is its fit-residual scatter (resid_std, the per-class prior
    width), of order 0.2 to 3 on this fixture and percent level on the production map. Measured discrepancies (relative,
    element by element): OpenBLAS kernels for other CPUs (Haswell, Zen, SandyBridge, SkylakeX), one thread, numpy and
    XLA instruction-set variants, both stacks (A6 CI portability report); Session B r18, numpy 2.5.4 against the pinned
    2.4.6; and an 8-ulp perturbation of every libm, XLA and least-squares input (backward stability: any LAPACK).

    - ``delta_c.coeffs`` and ``xbar.coeffs``: 1e-9. Measured at most 2.5e-14 (CPU) and 5.0e-13 (numpy 2.5.4,
      ``xbar.coeffs`` of the subset fit, 3598 ulps: the degree-2 fit's conditioning amplifies the last-bit rounding of
      another linear-algebra path), 3.0e-13 under the 8-ulp bound; 1e-9 is 2000 times the largest and above the
      normwise worst case of these fits' conditioning (cond(V)^2 eps, about 3e-10). A coefficient change of 1e-9
      moves delta_c(z) by at most 3e-7 over the clamp range (sum of |c_i| z^i), about 1e-7 of that class's
      resid_std, and Xbar(z) by at most 5e-8 (values of order 1).
    - ``delta_c.resid_std``: 1e-12. Measured 1 ulp (2.2e-16) on every path, 3.3e-15 under the 8-ulp bound: the
      residual norm is stationary in the coefficients at the least-squares optimum. 300 times the bound.
    - ``delta_c.resid_max``: 1e-10. Measured at most 4.6e-15 (CPU) and 3.0e-15 (numpy 2.5.4), 7.9e-15 under the
      8-ulp bound; a diagnostic (not used by the forward), so the tolerance is loose: 1e4 times the bound.
    - ``delta_c.mean``: 1e-12. Bitwise on every measured path; 2.3e-15 under the 8-ulp bound (it goes through XLA's
      exp in w_M0); 400 times the bound. The means are of order 0.2 to 3 here, never near zero, so atol is 0.

    A change of the rules (another simulation set, estimator, degree or clamp) moves the fitted outputs by orders of
    magnitude more and fails (test_the_t31_comparison_refuses_a_change_above_the_tolerance,
    test_the_t31_comparison_refuses_a_meaningful_change)."""
    d = _load(cache5)
    full, sub = W.refit(d), W.refit_subset(d, ["sim0", "sim1", "sim3", "sim4"])
    assert payload_mismatches(full, GOLDEN_PAYLOAD_REFIT) == []
    assert payload_mismatches(sub, GOLDEN_PAYLOAD_SUBSET) == []
    if exact_replay():                           # HCD_EXACT_REPLAY=1: a same-machine replay on the pinned stack
        assert W.payload_sha256(full) == GOLDEN_WCMAP_REFIT
        assert W.payload_sha256(sub) == GOLDEN_WCMAP_SUBSET


def test_the_stored_golden_payloads_are_the_golden_digests():
    assert W.payload_sha256(GOLDEN_PAYLOAD_REFIT) == GOLDEN_WCMAP_REFIT
    assert W.payload_sha256(GOLDEN_PAYLOAD_SUBSET) == GOLDEN_WCMAP_SUBSET


def test_the_t31_comparison_refuses_a_change_above_the_tolerance():
    want = GOLDEN_PAYLOAD_SUBSET
    assert payload_mismatches(copy.deepcopy(want), want) == []
    measured = copy.deepcopy(want)                 # the largest measured cross-stack difference, 5.0e-13, passes
    measured["xbar"]["coeffs"] = [c * (1 + 5e-13) for c in want["xbar"]["coeffs"]]
    assert payload_mismatches(measured, want) == []
    for sec, k in TOLERANT_FIELDS:                  # ten times its own tolerance, on one element of each, fails
        bad = copy.deepcopy(want)
        a = np.asarray(bad[sec][k], dtype=np.float64)
        a.flat[0] *= 1 + 10 * WC_MAP_TOLS[(sec, k)][0]
        bad[sec][k] = a.tolist()
        assert payload_mismatches(bad, want) == [f"{sec}.{k}"]
        ok = copy.deepcopy(want)                    # a tenth of it passes
        a = np.asarray(ok[sec][k], dtype=np.float64)
        a.flat[0] *= 1 + 0.1 * WC_MAP_TOLS[(sec, k)][0]
        ok[sec][k] = a.tolist()
        assert payload_mismatches(ok, want) == []
    for sec, k in (("delta_c", "z_clamp"), ("xbar", "z_fit")):     # one ulp on an exact field
        bad = copy.deepcopy(want)
        bad[sec][k][0] = float(np.nextafter(bad[sec][k][0], np.inf))
        assert payload_mismatches(bad, want) == [f"{sec}.{k}"]
    for k in ("z_min", "z_max"):
        bad = copy.deepcopy(want)
        bad["delta_c"][k] = float(np.nextafter(bad["delta_c"][k], np.inf))
        assert payload_mismatches(bad, want) == [f"delta_c.{k}"]
    for sec, k in (("delta_c", "n_blocks"), ("delta_c", "n_sims"), ("delta_c", "deg"), ("xbar", "n_groups")):
        bad = copy.deepcopy(want)
        bad[sec][k] += 1
        assert payload_mismatches(bad, want) == [f"{sec}.{k}"]
    bad = copy.deepcopy(want)
    bad["delta_c"]["coeffs"][0][0] = float("nan")
    assert payload_mismatches(bad, want) == ["delta_c.coeffs"]
    bad = copy.deepcopy(want)                       # a changed shape fails, even one that broadcasts (review m1)
    bad["delta_c"]["coeffs"] = bad["delta_c"]["coeffs"][:1]
    assert np.asarray(bad["delta_c"]["coeffs"]).shape[0] == 1 < np.asarray(want["delta_c"]["coeffs"]).shape[0]
    assert payload_mismatches(bad, want) == ["delta_c.coeffs"]
    bad = copy.deepcopy(want)                       # the same values under an extra axis: broadcasting alone passes it
    bad["delta_c"]["coeffs"] = [copy.deepcopy(want["delta_c"]["coeffs"])]
    assert payload_mismatches(bad, want) == ["delta_c.coeffs"]
    bad = copy.deepcopy(want)
    del bad["xbar"]["n_groups"]
    assert payload_mismatches(bad, want) == ["xbar keys"]
    assert payload_mismatches(dict(want, class_order=["clean", "LLS", "DLA", "subDLA"]), want) == ["class_order"]
    assert payload_mismatches(dict(want, format="hcd_priya.wc_map/2"), want) == ["format"]
    assert payload_mismatches(dict(want, extra=1), want) == ["keys"]
    assert "delta_c.coeffs" in payload_mismatches(GOLDEN_PAYLOAD_REFIT, want)      # another simulation set fails


def test_the_t31_comparison_refuses_a_meaningful_change():
    want = GOLDEN_PAYLOAD_REFIT
    bad = copy.deepcopy(want)                       # the sign of one class's correction
    bad["delta_c"]["coeffs"][1] = [-c for c in want["delta_c"]["coeffs"][1]]
    assert payload_mismatches(bad, want) == ["delta_c.coeffs"]
    bad = copy.deepcopy(want)                       # two classes exchanged (the class order of the rows)
    bad["delta_c"]["coeffs"][2], bad["delta_c"]["coeffs"][3] = want["delta_c"]["coeffs"][3], want["delta_c"]["coeffs"][2]
    assert payload_mismatches(bad, want) == ["delta_c.coeffs"]
    bad = copy.deepcopy(want)                       # the polynomial order reversed
    bad["xbar"]["coeffs"] = want["xbar"]["coeffs"][::-1]
    assert payload_mismatches(bad, want) == ["xbar.coeffs"]
    bad = copy.deepcopy(want)                       # delta_c(z) shifted by 1e-6 of the smallest resid_std
    bad["delta_c"]["coeffs"][0][2] += 1e-6 * min(want["delta_c"]["resid_std"])
    assert payload_mismatches(bad, want) == ["delta_c.coeffs"]
    bad = copy.deepcopy(want)                       # resid_std and resid_max exchanged
    bad["delta_c"]["resid_std"], bad["delta_c"]["resid_max"] = want["delta_c"]["resid_max"], want["delta_c"]["resid_std"]
    assert payload_mismatches(bad, want) == ["delta_c.resid_max", "delta_c.resid_std"]
    bad = copy.deepcopy(want)                       # a truncated mean
    bad["delta_c"]["mean"] = want["delta_c"]["mean"][:3]
    assert payload_mismatches(bad, want) == ["delta_c.mean"]
    bad = copy.deepcopy(want)
    bad["delta_c"]["mean"][0] = float("inf")
    assert payload_mismatches(bad, want) == ["delta_c.mean"]


def test_the_pinned_numpy_versions():
    assert numpy_pins("jax==0.10.1\nnumpy==2.4.6\nnumpyro==0.21.0\nscipy==1.17.1\n") == ("2.4.6",)
    assert numpy_pins("# numpy==1.0 (a comment)\nnumpy==2.4.6  # the pin\n") == ("2.4.6",)
    assert pinned_numpy_versions() and all(re.fullmatch(r"\d+\.\d+\.\d+", v) for v in pinned_numpy_versions())


# T31 review t1 (fail closed): a pin is read in any valid spelling, and a constraints file without one fails the test
@pytest.mark.parametrize("parse", [None, "strict"], ids=["requirement-grammar", "strict-parser"])
@pytest.mark.parametrize("line", ["numpy==2.5.4", "numpy == 2.5.4", "  numpy ==2.5.4", "NumPy==2.5.4", "NUMPY==2.5.4",
                                  "numpy==2.5.4 ; python_version >= '3.11'", "numpy==2.5.4;python_version>='3.11'",
                                  "numpy==2.5.4  # the pin", "numpy===2.5.4", "numpy==2.5.4 --hash=sha256:" + "0" * 64])
def test_a_numpy_pin_in_any_valid_spelling_is_read(line, parse):
    text = f"# the supported stack\njax==0.10.1\n{line}\nnumpyro==0.21.0\n-c other.txt\n"
    assert numpy_pins(text, parse=_requirement_strict if parse else None) == ("2.5.4",)


@pytest.mark.parametrize("parse", [None, "strict"], ids=["requirement-grammar", "strict-parser"])
@pytest.mark.parametrize("line", ["numpy>=2.4", "numpy==2.*", "numpy==2.4.6,<3", "numpy", "numpy==", "numpy=2.4.6",
                                  "numpy~=2.4"])
def test_a_numpy_line_that_is_not_an_exact_pin_fails(line, parse):
    with pytest.raises(ValueError, match="numpy"):
        numpy_pins(f"jax==0.10.1\n{line}\n", parse=_requirement_strict if parse else None)


@pytest.mark.parametrize("text", ["", "# only comments\n\n", "jax==0.10.1\nscipy==1.17.1\n", "numpyro==0.21.0\n"])
def test_a_constraints_file_that_pins_no_numpy_fails(tmp_path, text):
    path = tmp_path / "constraints-devenv.txt"
    path.write_text(text)
    with pytest.raises(ValueError, match="pins no numpy version"):
        pinned_numpy_versions(str(path))


# T31 review t2: a run that declares the supported pinned stack (HCD_PINNED_STACK exactly "1") must be on a pinned numpy.
# CI portability (PI decision 2026-11-05n): the exact digests need HCD_EXACT_REPLAY=1 as well, never on a GitHub runner.
PINS = {"numpy": ("2.4.6",), "jax": ("0.10.1",), "jaxlib": ("0.10.1",)}
ON_PINS = {"numpy": "2.4.6", "jax": "0.10.1", "jaxlib": "0.10.1"}
REPLAY = {"HCD_PINNED_STACK": "1", "HCD_EXACT_REPLAY": "1"}


@pytest.mark.parametrize("environ", [REPLAY], ids=["declared"])
def test_a_declared_pinned_stack_on_a_pinned_numpy_takes_the_exact_branch(environ):
    assert exact_replay(environ, versions=ON_PINS, pins=PINS) is True
    assert exact_replay({"HCD_PINNED_STACK": "1"}, versions=ON_PINS, pins=PINS) is False     # not without the opt-in


def test_a_declared_pinned_stack_on_another_numpy_fails():
    for environ in ({"HCD_PINNED_STACK": "1"}, REPLAY):
        with pytest.raises(ValueError, match="HCD_PINNED_STACK=1 declares the supported pinned stack, but numpy 2.5.4"):
            exact_replay(environ, versions=dict(ON_PINS, numpy="2.5.4"), pins=PINS)


@pytest.mark.parametrize("environ", [{}, {"HCD_PINNED_STACK": ""}, {"HCD_PINNED_STACK": "true"},
                                     {"HCD_PINNED_STACK": " 1"}, {"HCD_PINNED_STACK": "1 "}, {"HCD_PINNED_STACK": "01"},
                                     {"HCD_PINNED_STACK": "yes"}, {"OTHER": "1"}],
                         ids=["unset", "empty", "true", "space-1", "1-space", "01", "yes", "other-variable"])
def test_only_exactly_1_declares_the_pinned_stack(environ):
    assert not pinned_stack_declared(environ)
    assert exact_replay(environ, versions=dict(ON_PINS, numpy="2.5.4"), pins=PINS) is False     # tolerance path
    assert exact_replay(environ, versions=ON_PINS, pins=PINS) is False      # the numpy version alone: tolerance path
    with pytest.raises(ValueError, match="needs HCD_PINNED_STACK=1"):
        exact_replay(dict(environ, HCD_EXACT_REPLAY="1"), versions=ON_PINS, pins=PINS)
    assert pinned_stack_declared({"HCD_PINNED_STACK": "1"})


def test_a_declared_pinned_stack_uses_the_constraints_files_pins(tmp_path):
    path = tmp_path / "constraints-devenv.txt"
    path.write_text("numpy == 2.5.4 ; python_version >= '3.11'\njax==0.10.1\njaxlib==0.10.1\n")
    pins = {n: PN.pinned_versions(n, str(path)) for n in PN.EXACT_REPLAY_PACKAGES}
    assert exact_replay(REPLAY, versions=dict(ON_PINS, numpy="2.5.4"), pins=pins) is True
    with pytest.raises(ValueError, match="not one of the pinned versions"):
        exact_replay(REPLAY, versions=ON_PINS, pins=pins)


def test_the_recorded_pin_applies_only_when_the_constraints_file_is_absent(tmp_path):
    assert pinned_numpy_versions(str(tmp_path / "absent.txt")) == PN.PINNED_RECORDED["numpy"]
    for n in PN.EXACT_REPLAY_PACKAGES:
        assert PN.pinned_versions(n, str(tmp_path / "absent.txt")) == PN.PINNED_RECORDED[n]
    path = tmp_path / "constraints-devenv.txt"
    path.write_text("numpy == 9.9.9 ; python_version >= '3.11'\n")
    assert pinned_numpy_versions(str(path)) == ("9.9.9",)
    with pytest.raises(ValueError, match="pins no jax version"):
        PN.pinned_versions("jax", str(path))


def test_the_recorded_pins_are_the_constraints_files():
    """The wheel tier (tests/ alone, no constraints file) uses PINNED_RECORDED: it must be the file's pins."""
    assert os.path.exists(CONSTRAINTS_FILE)
    for n in PN.EXACT_REPLAY_PACKAGES:
        assert PN.pinned_versions(n) == PN.PINNED_RECORDED[n], n


@pytest.mark.parametrize("gha", ["true"], ids=["github-actions"])
def test_a_github_runner_never_takes_the_exact_branch(gha):
    runner = {"GITHUB_ACTIONS": gha, "HCD_PINNED_STACK": "1"}          # the supported job: pinned, declared
    assert exact_replay(runner, versions=ON_PINS, pins=PINS) is False
    assert exact_replay({"GITHUB_ACTIONS": gha, "HCD_PINNED_STACK": ""}, versions=ON_PINS, pins=PINS) is False
    with pytest.raises(ValueError, match="on a GitHub runner"):
        exact_replay(dict(runner, HCD_EXACT_REPLAY="1"), versions=ON_PINS, pins=PINS)
    assert exact_replay(dict(REPLAY, GITHUB_ACTIONS=""), versions=ON_PINS, pins=PINS) is True    # not a runner


@pytest.mark.parametrize("value", ["", "0", "true", "yes", " 1", "1 ", "01"])
def test_only_exactly_1_asks_for_the_exact_replay(value):
    assert exact_replay(dict(REPLAY, HCD_EXACT_REPLAY=value), versions=ON_PINS, pins=PINS) is False


@pytest.mark.parametrize("package", ["jax", "jaxlib"])
def test_the_exact_replay_needs_jax_and_jaxlib_at_their_pins(package):
    off = dict(ON_PINS, **{package: "0.10.2"})
    assert exact_replay({"HCD_PINNED_STACK": "1"}, versions=off, pins=PINS) is False      # tolerance path: allowed
    with pytest.raises(ValueError, match=f"not on the pinned stack: {package} 0.10.2"):
        exact_replay(REPLAY, versions=off, pins=PINS)


def test_the_field_comparison_rules():
    want = {"f": np.array([1.0, -2.0, 0.0, np.nan]), "i": np.array([1, 2, 3])}
    tols = {"f": (1e-12, 1e-15), "i": (0, 0)}
    assert PN.field_mismatches({k: np.array(v, copy=True) for k, v in want.items()}, want, tols) == []
    assert PN.field_mismatches(dict(want, f=np.array([1.0 + 5e-13, -2.0, 5e-16, np.nan])), want, tols) == []
    assert PN.field_mismatches(dict(want, f=np.array([1.0 + 2e-12, -2.0, 0.0, np.nan])), want, tols) == ["f"]
    assert PN.field_mismatches(dict(want, f=np.array([1.0, -2.0, 2e-15, np.nan])), want, tols) == ["f"]   # atol
    assert PN.field_mismatches(dict(want, f=np.array([1.0, -2.0, 0.0, 0.0])), want, tols) == ["f"]   # a lost NaN
    assert PN.field_mismatches(dict(want, f=np.array([1.0, -2.0, np.nan, np.nan])), want, tols) == ["f"]
    assert PN.field_mismatches(dict(want, f=np.array([1.0, -2.0, 0.0, np.inf])), want, tols) == ["f"]
    assert PN.field_mismatches(dict(want, f=np.array([[1.0, -2.0, 0.0, np.nan]])), want, tols) == ["f"]   # shape
    assert PN.field_mismatches(dict(want, i=np.array([1, 2, 4])), want, tols) == ["i"]           # integers: exact
    assert PN.field_mismatches(dict(want, i=np.array([1.0, 2.0, 3.0])), want, tols) == ["i"]     # dtype
    assert PN.field_mismatches(dict(want, extra=np.zeros(1)), want, tols) == ["keys"]
    with pytest.raises(ValueError, match="compared exactly"):
        PN.field_mismatches(want, want, dict(tols, i=(1e-12, 0)))
    with pytest.raises(ValueError, match="non-negative"):
        PN.field_mismatches(want, want, dict(tols, f=(-1e-12, 0)))
    with pytest.raises(ValueError, match="its own tolerance"):
        PN.field_mismatches(want, want, {"f": tols["f"]})
