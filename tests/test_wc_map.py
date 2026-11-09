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

# T31 off the pinned stack (Session B r18): the numpy of the supported, pinned environment, and the tolerance of the
# least-squares outputs elsewhere (test_golden_digests_of_the_map_rules says why 1e-9)
PINNED_NUMPY_RECORDED = ("2.4.6",)        # numpy==2.4.6 in Session B's tools/ci/constraints-devenv.txt (ef23c83)
WC_MAP_RTOL = 1e-9
TOLERANT_FIELDS = (("delta_c", "coeffs"), ("delta_c", "resid_std"), ("delta_c", "resid_max"), ("xbar", "coeffs"))


CONSTRAINTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "ci",
                                "constraints-devenv.txt")
_STRICT_REQUIREMENT = re.compile(r"(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)\s*(?:\[[^\]]*\])?\s*"
                                 r"(?P<spec>[^;]*?)\s*(?:;(?P<marker>.*))?")
_EXACT_SPEC = re.compile(r"===?\s*(?P<version>[0-9][0-9A-Za-z.+!_-]*)")


def _requirement_strict(line):
    """``(canonical name, exact version or None)`` of one requirement line, without ``packaging``: a name (``-``,
    ``_`` and ``.`` runs equal, any case), optional extras, the specifiers and an optional environment marker. The
    version is returned only for one ``==`` or ``===`` specifier without a wildcard. Unparsable: ValueError."""
    m = _STRICT_REQUIREMENT.fullmatch(line)
    if m is None or (m.group("marker") is not None and not m.group("marker").strip()):
        raise ValueError(f"unparsable requirement line {line!r}")
    e = _EXACT_SPEC.fullmatch(m.group("spec").strip())
    return re.sub(r"[-_.]+", "-", m.group("name")).lower(), e.group("version") if e and "*" not in e.group(0) else None


def _requirement(line):
    """``(canonical name, exact version or None)`` of one requirement line, with the standard requirement grammar
    (``packaging.requirements``, a pytest dependency) when it is importable, else ``_requirement_strict``."""
    try:
        from packaging.requirements import InvalidRequirement, Requirement
        from packaging.utils import canonicalize_name
    except ImportError:
        return _requirement_strict(line)
    try:
        req = Requirement(line)
    except InvalidRequirement as e:
        raise ValueError(f"unparsable requirement line {line!r}: {e}") from None
    specs = list(req.specifier)
    exact = len(specs) == 1 and specs[0].operator in ("==", "===") and "*" not in specs[0].version
    return canonicalize_name(req.name), specs[0].version if exact else None


def numpy_pins(text, parse=None):
    """The exact numpy pins of a pip constraints text, in any valid spelling (spaces, case, an environment marker, an
    inline comment, per-requirement options such as ``--hash``); comment, blank and option (``-c``, ``-r``) lines are
    skipped. Fail closed: a numpy line that is not one exact pin (a range, a wildcard, several specifiers) or any
    unparsable line raises ValueError. ``parse``: the line parser (default ``_requirement``)."""
    parse = parse or _requirement
    pins = []
    for raw in text.splitlines():
        line = re.sub(r"\s+--\S.*$", "", re.sub(r"(^|\s)#.*$", "", raw)).strip()
        if not line or line.startswith("-"):
            continue
        name, version = parse(line)
        if name == "numpy":
            if version is None:
                raise ValueError(f"the numpy line {raw.strip()!r} is not an exact pin (numpy==X)")
            pins.append(version)
    return tuple(pins)


def pinned_numpy_versions(path=CONSTRAINTS_FILE):
    """The numpy version(s) of the supported, pinned environment (docs/compatibility.md; the PI's rule B-D3).

    - When the constraints file (Session B's ``tools/ci/constraints-devenv.txt``) exists, its exact numpy pins, read
      by ``numpy_pins``. Fail closed: a file that pins no numpy version, or whose numpy line is not one exact pin,
      raises ValueError, so the test fails instead of silently taking the recorded version.
    - Only when the file is absent (a tests-only copy, or a tree without it): ``PINNED_NUMPY_RECORDED``."""
    if not os.path.exists(path):
        return PINNED_NUMPY_RECORDED
    with open(path) as fh:
        pins = numpy_pins(fh.read())
    if not pins:
        raise ValueError(f"{path} exists but pins no numpy version (numpy==X): the pinned stack cannot be told, so "
                         "the bit-level T31 digests cannot be placed; fix the constraints file or this parser")
    return pins


PINNED_STACK_ENV = "HCD_PINNED_STACK"


def pinned_stack_declared(environ=None):
    """Whether this run declares itself the supported pinned stack: ``HCD_PINNED_STACK`` exactly ``"1"`` (Session B's
    supported CI job and its pinned replays set it; the newer-dependency jobs set it empty). Unset, empty or any other
    value (``"true"``, ``" 1"``) is not a declaration."""
    return (os.environ if environ is None else environ).get(PINNED_STACK_ENV) == "1"


def on_pinned_numpy(version=None, environ=None, pins=None):
    """Whether T31 takes the exact-digest branch: ``version`` (default ``np.__version__``) is one of ``pins`` (default
    ``pinned_numpy_versions()``: the constraints file's pins, fail closed; the recorded pin only when that file is
    absent). A run that declares the pinned stack (``pinned_stack_declared``) MUST be on a pinned numpy: otherwise
    ValueError, so a supported job that drifted to another numpy fails instead of quietly taking the tolerance path."""
    version = np.__version__ if version is None else version
    pins = pinned_numpy_versions() if pins is None else tuple(pins)
    if version in pins:
        return True
    if pinned_stack_declared(environ):
        raise ValueError(f"{PINNED_STACK_ENV}=1 declares the supported pinned stack, but numpy {version} is not one of "
                         f"the pinned versions {pins}: the bit-level T31 digests would not be checked; fix the "
                         "environment or the pin")
    return False


def payload_mismatches(got, want, rtol=WC_MAP_RTOL):
    """The dotted paths where the w_c map payload ``got`` departs from ``want`` (empty: they agree). The key sets must
    be equal; the least-squares outputs (``TOLERANT_FIELDS``) must have the same shape, be finite and agree element by
    element to ``rtol`` relative (``|got - want| <= rtol |want|``); every other field (counts, degrees, the clamp and
    fit ranges, the redshift range, the means, the format and class order) must be exactly equal (canonical JSON)."""
    out = []
    if set(got) != set(want):
        out.append("keys")
    for sec in ("delta_c", "xbar"):
        g, w = got.get(sec), want[sec]
        if not isinstance(g, dict) or set(g) != set(w):
            out.append(f"{sec} keys")
            continue
        for k in sorted(w):
            if (sec, k) in TOLERANT_FIELDS:
                a, b = np.asarray(g[k], dtype=np.float64), np.asarray(w[k], dtype=np.float64)
                if a.shape != b.shape or not np.all(np.isfinite(a)) or np.any(np.abs(a - b) > rtol * np.abs(b)):
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

    On the pinned numpy (``on_pinned_numpy()``: ``np.__version__`` in ``pinned_numpy_versions()``, the supported
    environment) the payload sha256 must be the golden digests, bit level, as before; a run that declares the pinned
    stack (``HCD_PINNED_STACK=1``) on another numpy fails. On any other numpy the least-squares outputs may move by
    linear-algebra (BLAS/LAPACK) rounding, so the payload is compared field by field with the stored golden payloads
    (``payload_mismatches``, which also runs on the pinned stack): every count, degree, clamp and fit range, the
    redshift range, the means, the format and the class order exactly; ``delta_c.coeffs``, ``delta_c.resid_std``,
    ``delta_c.resid_max`` and ``xbar.coeffs`` to ``WC_MAP_RTOL`` = 1e-9 relative, element by element.

    Why 1e-9. Measured (Session B r18: numpy 2.5.4 against the pinned 2.4.6, the same code and cache): the largest
    difference is 5.0e-13 relative (``xbar.coeffs`` of the subset fit, 3598 ulps: the degree-2 fit's conditioning
    amplifies another linear-algebra path's last-bit rounding), then 6.4e-14 (``delta_c.coeffs``), 3.0e-15
    (``resid_max``, 18 ulps) and 2.2e-16 (``resid_std``, 1 ulp); the means, ranges and counts are identical. 1e-9 is
    2000 times the largest, so a linear-algebra path that rounds several times worse still passes, and it is above the
    normwise worst case of these fits' conditioning (cond(V)^2 eps, at most about 3e-10 here); it is orders of
    magnitude below any scientific use of these numbers. A change of the rules (another simulation set, estimator,
    degree or clamp) moves the fitted outputs by orders of magnitude more than 1e-9 and fails
    (test_the_t31_comparison_refuses_a_change_above_the_tolerance).

    The pinned branch is keyed on the numpy version string, not on its linear-algebra build: the same numpy version
    built on another BLAS would fail the bit-level digests, loudly (fail closed, outside the supported stack). The
    means are compared exactly; on some other, unsupported numpy a reduction could move one by an ulp and fail
    (fail closed by design)."""
    d = _load(cache5)
    full, sub = W.refit(d), W.refit_subset(d, ["sim0", "sim1", "sim3", "sim4"])
    if on_pinned_numpy():                        # a declared pinned stack (HCD_PINNED_STACK=1) must take this branch
        assert W.payload_sha256(full) == GOLDEN_WCMAP_REFIT
        assert W.payload_sha256(sub) == GOLDEN_WCMAP_SUBSET
    assert payload_mismatches(full, GOLDEN_PAYLOAD_REFIT) == []
    assert payload_mismatches(sub, GOLDEN_PAYLOAD_SUBSET) == []


def test_the_stored_golden_payloads_are_the_golden_digests():
    assert W.payload_sha256(GOLDEN_PAYLOAD_REFIT) == GOLDEN_WCMAP_REFIT
    assert W.payload_sha256(GOLDEN_PAYLOAD_SUBSET) == GOLDEN_WCMAP_SUBSET


def test_the_t31_comparison_refuses_a_change_above_the_tolerance():
    want = GOLDEN_PAYLOAD_SUBSET
    assert payload_mismatches(copy.deepcopy(want), want) == []
    measured = copy.deepcopy(want)                 # the largest measured cross-stack difference, 5.0e-13, passes
    measured["xbar"]["coeffs"] = [c * (1 + 5e-13) for c in want["xbar"]["coeffs"]]
    assert payload_mismatches(measured, want) == []
    for sec, k in TOLERANT_FIELDS:                  # ten times the tolerance, on one element of each, fails
        bad = copy.deepcopy(want)
        a = np.asarray(bad[sec][k], dtype=np.float64)
        a.flat[0] *= 1 + 10 * WC_MAP_RTOL
        bad[sec][k] = a.tolist()
        assert payload_mismatches(bad, want) == [f"{sec}.{k}"]
    for sec, k in (("delta_c", "mean"), ("delta_c", "z_clamp"), ("xbar", "z_fit")):     # one ulp on an exact field
        bad = copy.deepcopy(want)
        bad[sec][k][0] = float(np.nextafter(bad[sec][k][0], np.inf))
        assert payload_mismatches(bad, want) == [f"{sec}.{k}"]
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


# T31 review t2: a run that declares the supported pinned stack (HCD_PINNED_STACK exactly "1") must be on a pinned numpy
@pytest.mark.parametrize("environ", [{"HCD_PINNED_STACK": "1"}], ids=["declared"])
def test_a_declared_pinned_stack_on_a_pinned_numpy_takes_the_exact_branch(environ):
    assert on_pinned_numpy(version="2.4.6", environ=environ, pins=("2.4.6",)) is True


def test_a_declared_pinned_stack_on_another_numpy_fails():
    with pytest.raises(ValueError, match="HCD_PINNED_STACK=1 declares the supported pinned stack, but numpy 2.5.4"):
        on_pinned_numpy(version="2.5.4", environ={"HCD_PINNED_STACK": "1"}, pins=("2.4.6",))


@pytest.mark.parametrize("environ", [{}, {"HCD_PINNED_STACK": ""}, {"HCD_PINNED_STACK": "true"},
                                     {"HCD_PINNED_STACK": " 1"}, {"HCD_PINNED_STACK": "1 "}, {"HCD_PINNED_STACK": "01"},
                                     {"HCD_PINNED_STACK": "yes"}, {"OTHER": "1"}],
                         ids=["unset", "empty", "true", "space-1", "1-space", "01", "yes", "other-variable"])
def test_only_exactly_1_declares_the_pinned_stack(environ):
    assert not pinned_stack_declared(environ)
    assert on_pinned_numpy(version="2.5.4", environ=environ, pins=("2.4.6",)) is False      # as today: tolerance path
    assert on_pinned_numpy(version="2.4.6", environ=environ, pins=("2.4.6",)) is True       # as today: exact digests
    assert pinned_stack_declared({"HCD_PINNED_STACK": "1"})


def test_a_declared_pinned_stack_uses_the_constraints_files_pins(tmp_path):
    path = tmp_path / "constraints-devenv.txt"
    path.write_text("numpy == 2.5.4 ; python_version >= '3.11'\n")
    pins = pinned_numpy_versions(str(path))
    assert on_pinned_numpy(version="2.5.4", environ={"HCD_PINNED_STACK": "1"}, pins=pins) is True
    with pytest.raises(ValueError, match="not one of the pinned versions"):
        on_pinned_numpy(version="2.4.6", environ={"HCD_PINNED_STACK": "1"}, pins=pins)


def test_the_recorded_pin_applies_only_when_the_constraints_file_is_absent(tmp_path):
    assert pinned_numpy_versions(str(tmp_path / "absent.txt")) == PINNED_NUMPY_RECORDED
    path = tmp_path / "constraints-devenv.txt"
    path.write_text("numpy == 9.9.9 ; python_version >= '3.11'\n")
    assert pinned_numpy_versions(str(path)) == ("9.9.9",)
