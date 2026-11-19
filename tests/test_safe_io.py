"""The pickle-free prototype format (hcd_analysis/emulator/safe_io.py, docs/release_format.md; kept as evidence).

Synthetic only: a small Emulator built from a fixed PRNG key and a synthetic normalizer with the production key layout
(nested dicts, integer keys). The real-member pilot is scripts/export_safe_bundle.py, run outside the test suite.
"""

import json
import os
import struct
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator import safe_io as S
from hcd_analysis.emulator.model import Emulator
from hcd_analysis.emulator.schema import CHECKPOINT_SCHEMA_VERSION

ARCH = {"in_dim": 10, "n_k": 8, "n_basis": 4, "baseline_n_layers": 1, "baseline_width": 16}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _meta(**over):
    m = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "arch_cfg": dict(ARCH),
        "seed": 3,
        "k_com_hmpc": list(np.arange(1, ARCH["n_k"] + 1) * 0.05),
        "L_box_hmpc": 120.0,
        "k_convention": "synthetic",
        "cache_sha256": "0" * 64,
        "param_limits": [[0.0, 1.0]] * 9,
        "cache_path": "synthetic.h5",
        "git_sha": "synthetic",
        "n_k": ARCH["n_k"],
        "recipe": {"w": 1.0},
    }
    m.update(over)
    return m


def _norm():
    r = np.random.default_rng(0)
    return {
        "f_nhi": {"mean": r.normal(size=30), "std": r.uniform(1, 2, 30)},
        "dndx": {"mean": r.normal(size=3), "std": r.uniform(1, 2, 3)},
        "P_filt": {
            "mu_marg": r.normal(size=(4, 8)),
            "sig_marg": r.uniform(1, 2, (4, 8)),
            "sig_cosmo": r.uniform(1, 2, (4, 8)),
            "cell_mean": {i: r.normal(size=(4, 8)) for i in range(5)},
        },
        "delta": {"mean": r.normal(size=(3, 8)), "std": r.uniform(1, 2, (3, 8))},
    }


@pytest.fixture
def member(tmp_path):
    model = Emulator(**ARCH, key=jax.random.PRNGKey(3))
    prefix = str(tmp_path / "m0")
    S.export_member(prefix, model, _meta(), _norm())
    return prefix, model


def _tree_equal(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(_tree_equal(a[k], b[k]) for k in a)
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# --------------------------------------------------------------------------------------------- round trip
def test_member_round_trip_is_exact(member):
    prefix, model = member
    m2, meta2, norm2 = S.load_member(prefix)
    l1, t1 = jax.tree_util.tree_flatten_with_path(model)
    l2, t2 = jax.tree_util.tree_flatten_with_path(m2)
    assert t1 == t2
    for (p1, x1), (p2, x2) in zip(l1, l2):
        assert jax.tree_util.keystr(p1) == jax.tree_util.keystr(p2)
        assert np.asarray(x1).dtype == np.asarray(x2).dtype == np.float64
        assert np.asarray(x1).tobytes() == np.asarray(x2).tobytes()
    assert meta2 == json.loads(json.dumps(_meta()))
    assert _tree_equal(_norm(), norm2)
    assert set(norm2["P_filt"]["cell_mean"]) == set(range(5))  # integer keys come back as ints


def test_predictions_and_gradients_are_identical(member):
    prefix, model = member
    m2, _, _ = S.load_member(prefix)
    x = jnp.linspace(0.1, 0.9, 10)
    for tau0 in (0.2, 0.5):
        o1, o2 = model(x, tau0), m2(x, tau0)
        assert set(o1) == set(o2)
        for k in o1:
            assert np.asarray(o1[k]).tobytes() == np.asarray(o2[k]).tobytes()

    def f(mod, x, t):
        o = mod(x, t)
        return jnp.sum(o["P_filt_resid"] ** 2) + jnp.sum(o["P_filt_base"])

    g1 = jax.grad(f, argnums=(1, 2))(model, x, 0.3)
    g2 = jax.grad(f, argnums=(1, 2))(m2, x, 0.3)
    for a, b in zip(g1, g2):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_tensor_names_are_stable_tree_paths(member):
    prefix, _ = member
    t, meta = S.read_safetensors(prefix + ".safetensors")
    assert meta["format"] == S.FORMAT
    assert "model.enc.layers[0].weight" in t
    assert "norm/P_filt/cell_mean/#0" in t and "norm/dndx/mean" in t
    with open(prefix + ".json") as f:
        doc = json.load(f)
    assert set(doc["tensors"]) == set(t)
    for n, a in t.items():
        assert doc["tensors"][n] == {"shape": list(a.shape), "dtype": "F64"}
    assert doc["tensors"]["model.enc.layers[0].weight"]["shape"][1] == ARCH["in_dim"]
    assert doc["safetensors_file"] == "m0.safetensors"


def test_dtypes_and_byte_order(tmp_path):
    p = str(tmp_path / "t.safetensors")
    arrs = {
        "f64": np.arange(6, dtype=np.float64).reshape(2, 3),
        "f32": np.ones(3, np.float32),
        "i64": np.array([1, -2], np.int64),
        "be": np.arange(4, dtype=">f8"),
        "scalar": np.float64(2.5),
        "empty": np.zeros((0, 3)),
        "b": np.array([True, False]),
        "negzero": np.array([-0.0, 0.0]),
    }
    S.write_safetensors(p, arrs)
    out, _ = S.read_safetensors(p)
    assert (
        out["f64"].dtype == np.float64
        and out["f32"].dtype == np.float32
        and out["i64"].dtype == np.int64
    )
    assert out["be"].dtype == np.dtype("<f8") and np.array_equal(out["be"], np.arange(4.0))
    assert out["scalar"].shape == () and out["scalar"] == 2.5 and out["empty"].shape == (0, 3)
    assert np.array_equal(out["b"], [True, False]) and out["b"].dtype == np.bool_
    assert out["negzero"].tobytes() == arrs["negzero"].tobytes()  # bits, not values
    for k in ("f64", "f32", "i64"):
        assert out[k].tobytes() == arrs[k].tobytes()


def test_module_never_uses_pickle():
    src = open(S.__file__).read()
    assert "import pickle" not in src and "pickle." not in src and "allow_pickle=True" not in src


# --------------------------------------------------------------------------------------------- header refusals
def _rewrite_header(path, mutate):
    raw = open(path, "rb").read()
    (n,) = struct.unpack("<Q", raw[:8])
    h = json.loads(raw[8 : 8 + n])
    h = mutate(h)
    hb = json.dumps(h).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + raw[8 + n :])


def _raw_file(tmp_path, header_bytes, body=b""):
    p = str(tmp_path / "raw.safetensors")
    open(p, "wb").write(struct.pack("<Q", len(header_bytes)) + header_bytes + body)
    return p


def test_refuses_corrupt_files(tmp_path):
    p = str(tmp_path / "t.safetensors")
    S.write_safetensors(p, {"a": np.arange(4.0), "b": np.ones((2, 2))})
    raw = open(p, "rb").read()
    cases = {
        "short": raw[:5],
        "badlen": struct.pack("<Q", 10**9) + raw[8:],
        "truncated": raw[:-8],
        "trailing": raw + b"\0" * 8,
        "notjson": raw[:8] + b"\xff" * (len(raw) - 8),
    }
    for name, blob in cases.items():
        q = str(tmp_path / f"{name}.safetensors")
        open(q, "wb").write(blob)
        with pytest.raises(S.SafeFormatError):
            S.read_safetensors(q)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda h: {**h, "a": {**h["a"], "dtype": "F16"}},  # unsupported dtype
        lambda h: {**h, "a": {**h["a"], "dtype": ["F64"]}},  # dtype not a string
        lambda h: {**h, "a": {**h["a"], "shape": [5]}},  # shape vs bytes
        lambda h: {**h, "a": {**h["a"], "data_offsets": [0, 16]}},  # size mismatch
        lambda h: {**h, "b": {**h["b"], "data_offsets": [0, 32]}},  # overlap
        lambda h: {**h, "a": {**h["a"], "extra": 1}},  # malformed entry
        lambda h: {**h, "a": {**h["a"], "shape": [True, 4]}},  # bool is not an int
        lambda h: {**h, "a": {**h["a"], "shape": [2**62 + 1, 4]}},  # int64 wrap
        lambda h: {**h, "a": {**h["a"], "shape": [2**70]}},  # beyond int64
        lambda h: {**h, "a": {**h["a"], "shape": [1] * 9 + [4]}},  # too many dims
        lambda h: {
            **h,
            "a": {**h["a"], "shape": [2**62, 0], "data_offsets": [32, 32]},
            "b": {**h["b"], "data_offsets": [0, 32]},
        },  # huge zero-size dims
        lambda h: {
            **h,
            "a": {**h["a"], "shape": [2**39, 2**39, 0], "data_offsets": [32, 32]},
            "b": {**h["b"], "data_offsets": [0, 32]},
        },  # reshape refused in pass 2
        lambda h: {**h, "__metadata__": {"x": 1}},  # non-string metadata
        lambda h: {**h, "": h["a"]},  # empty name
    ],
)
def test_refuses_malformed_headers(tmp_path, mutate):
    p = str(tmp_path / "t.safetensors")
    S.write_safetensors(p, {"a": np.arange(4.0), "b": np.ones((2, 2))})
    _rewrite_header(p, mutate)
    with pytest.raises(S.SafeFormatError):
        S.read_safetensors(p)


def test_refuses_a_true_gap_and_odd_headers(tmp_path):
    gap = json.dumps(
        {
            "a": {"dtype": "F64", "shape": [1], "data_offsets": [0, 8]},
            "b": {"dtype": "F64", "shape": [1], "data_offsets": [16, 24]},
        }
    ).encode()
    gap += b" " * (-len(gap) % 8)
    with pytest.raises(S.SafeFormatError, match="gap"):
        S.read_safetensors(_raw_file(tmp_path, gap, b"\0" * 24))
    with pytest.raises(S.SafeFormatError, match="start"):
        S.read_safetensors(_raw_file(tmp_path, b' {"a":1}', b""))
    deep = b'{"a":' + b"[" * 200000 + b"]" * 200000 + b"}"
    with pytest.raises(S.SafeFormatError, match="RecursionError"):
        S.read_safetensors(_raw_file(tmp_path, deep + b" " * (-len(deep) % 8)))
    dup = b'{"a":{"dtype":"F64","shape":[1],"data_offsets":[0,8]},"a":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}}'
    dup += b" " * (-len(dup) % 8)
    with pytest.raises(S.SafeFormatError, match="duplicate"):
        S.read_safetensors(_raw_file(tmp_path, dup, np.float64(1).tobytes()))


def test_expected_digest_is_checked_on_the_parsed_bytes(tmp_path):
    p = str(tmp_path / "t.safetensors")
    S.write_safetensors(p, {"a": np.arange(4.0)})
    S.read_safetensors(p, expected_sha256=S.sha256_file(p))
    with pytest.raises(S.SafeFormatError, match="sha256"):
        S.read_safetensors(p, expected_sha256="0" * 64)


def test_refuses_non_numeric_arrays(tmp_path):
    for bad in (
        {"a": np.array(["x"])},
        {"a": np.array([object()], dtype=object)},
        {"a": np.zeros(2, np.float16)},
    ):
        with pytest.raises(S.SafeFormatError):
            S.write_safetensors(str(tmp_path / "t.safetensors"), bad)
    with pytest.raises(S.SafeFormatError):
        S.norm_tensors({"a": np.array([object()], dtype=object)})
    with pytest.raises(S.SafeFormatError):
        S.norm_tensors({"a/b": np.zeros(1)})  # a key the names cannot carry


def test_writes_never_overwrite(member, tmp_path):
    prefix, model = member
    before = open(prefix + ".safetensors", "rb").read()
    with pytest.raises(S.SafeFormatError, match="never overwrites"):
        S.export_member(prefix, model, _meta(), _norm())
    with pytest.raises(S.SafeFormatError, match="never overwritten"):
        S.write_safetensors(
            prefix + ".safetensors", {"a": np.zeros(1)}
        )  # the exclusive publish itself
    assert open(prefix + ".safetensors", "rb").read() == before
    assert not [
        f for f in os.listdir(tmp_path) if f.startswith(".tmp_")
    ]  # no temporary file left behind


# --------------------------------------------------------------------------------------------- normalizer names
@pytest.mark.parametrize(
    "names,match",
    [
        ({"norm/a/#1": 1, "norm/a/#01": 2}, "canonical"),
        ({"norm/a/# 1": 1}, "canonical"),
        ({"norm/a/#1_0": 1}, "canonical"),
        ({"norm/a/#abc": 1}, "canonical"),
        ({"norm/a/#" + "1" * 25: 1}, "canonical"),
        ({"norm/a//b": 1}, "empty segment"),
        ({"norm/a": 1, "norm/a/b": 2}, "nests under a leaf|collides"),
    ],
)
def test_normalizer_names_must_be_canonical(names, match):
    with pytest.raises(S.SafeFormatError, match=match):
        S.norm_from_tensors({k: np.zeros(1) + v for k, v in names.items()})


def _replace_tensors(prefix, fn):
    t, meta = S.read_safetensors(prefix + ".safetensors")
    fn(t)
    os.remove(prefix + ".safetensors")
    S.write_safetensors(prefix + ".safetensors", t, metadata=meta)
    _edit_sidecar(
        prefix,
        lambda d: d.update(
            safetensors_sha256=S.sha256_file(prefix + ".safetensors"),
            tensors={n: {"shape": list(a.shape), "dtype": S._code(a.dtype)} for n, a in t.items()},
        ),
    )


def _edit_sidecar(prefix, fn):
    with open(prefix + ".json") as f:
        doc = json.load(f)
    fn(doc)
    with open(prefix + ".json", "w") as f:
        json.dump(doc, f)


@pytest.mark.parametrize(
    "fn,match",
    [
        (lambda t: t.pop("model.enc.layers[0].bias"), "differ from the network"),
        (lambda t: t.update({"model.extra": np.zeros(2)}), "differ from the network"),
        (lambda t: t.update({"model.enc.layers[0].bias": np.zeros(3)}), "expects"),
        (
            lambda t: t.update(
                {"model.enc.layers[0].bias": t["model.enc.layers[0].bias"].astype(np.float32)}
            ),
            "never down-cast",
        ),
        (lambda t: t.update({"zzz": np.zeros(1)}), "outside model/norm"),
        (lambda t: t.update({"normX/a": np.zeros(1)}), "outside model/norm"),
        (lambda t: t.update({"norm/dndx/mean": t["norm/dndx/mean"].astype(np.float32)}), "float64"),
        (lambda t: t.update({"norm/dndx/mean": np.zeros(4)}), "float64"),
        (lambda t: t.pop("norm/delta/std"), "keys"),
        (lambda t: t.update({"norm/extra/mean": np.zeros(3)}), "keys"),
        (
            lambda t: [t.pop(k) for k in [k for k in t if k.startswith("norm/P_filt/cell_mean/")]],
            "keys|cell_mean",
        ),
    ],
)
def test_load_refuses_tensor_mismatch(member, fn, match):
    prefix, _ = member
    _replace_tensors(prefix, fn)
    with pytest.raises(S.SafeFormatError, match=match):
        S.load_member(prefix)


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda d: d.update(format="hcd_priya.safe/0"), "format"),
        (lambda d: d["meta"].update(schema_version="1.0"), "schema"),
        (lambda d: d["meta"]["arch_cfg"].update(activation="relu"), "allowlist"),
        (lambda d: d["meta"]["arch_cfg"].update(n_k=8.0), "int in"),
        (lambda d: d["meta"]["arch_cfg"].update(baseline_width=10**9), "int in"),
        (lambda d: d["meta"].update(seed="3"), "seed"),
        (lambda d: d["meta"].update(seed=2**40), "seed"),
        (lambda d: d["meta"].update(cache_sha256="abc"), "cache_sha256"),
        (lambda d: d["meta"]["recipe"].update(kfkms=[1.0]), "velocity"),
        (lambda d: d.update(safetensors_sha256="0" * 64), "sha256"),
        (lambda d: d["tensors"].pop("norm/dndx/mean"), "index"),
        (lambda d: d.update(safetensors_file="../m0.safetensors"), "safetensors_file"),
        (lambda d: d.update(safetensors_file="/abs/m0.safetensors"), "safetensors_file"),
        (lambda d: d.pop("tensors"), "lacks"),
    ],
)
def test_load_refuses_bad_sidecar(member, edit, match):
    prefix, _ = member
    _edit_sidecar(prefix, edit)
    with pytest.raises(S.SafeFormatError, match=match):
        S.load_member(prefix)


def test_corrupted_bytes_fail_the_digest(member):
    prefix, _ = member
    raw = bytearray(open(prefix + ".safetensors", "rb").read())
    raw[-3] ^= 0xFF
    open(prefix + ".safetensors", "wb").write(bytes(raw))
    with pytest.raises(S.SafeFormatError, match="sha256"):
        S.load_member(prefix)


# --------------------------------------------------------------------------------------------- manifest, bundle
@pytest.fixture
def bundle(tmp_path):
    bd = tmp_path / "bundle"
    bd.mkdir()
    pres = []
    for i in range(2):
        pre = str(bd / f"m{i}")
        S.export_member(pre, Emulator(**ARCH, key=jax.random.PRNGKey(i)), _meta(seed=i), _norm())
        pres.append(pre)
    S.write_manifest(str(bd), pres, {"source": "synthetic"})
    return str(bd), pres


def _edit_manifest(bd, fn):
    p = os.path.join(bd, "manifest.json")
    man = json.load(open(p))
    fn(man)
    os.remove(p)
    open(p, "w").write(json.dumps(man))


def test_bundle_round_trip_and_manifest_digest(bundle):
    bd, pres = bundle
    assert S.verify_manifest(bd) == pres
    sha = S.sha256_file(os.path.join(bd, "manifest.json"))
    loaded = S.load_bundle_historical(bd, expected_manifest_sha256=sha, reason="test")
    with pytest.raises(TypeError):
        S.load_bundle_historical(bd, reason="test")  # the anchor must be passed explicitly
    for bad in ("", "  ", None):
        with pytest.raises(ValueError, match="non-empty reason"):
            S.load_bundle_historical(bd, expected_manifest_sha256=sha, reason=bad)
    assert not hasattr(S, "load_bundle")  # no lineage-free release-sounding name
    assert [m["seed"] for _, m, _ in loaded] == [0, 1]
    with pytest.raises(S.SafeFormatError, match="expected digest"):
        S.verify_manifest(bd, expected_sha256="0" * 64)
    with pytest.raises(FileExistsError):
        open(os.path.join(bd, "manifest.json"), "x")


@pytest.mark.parametrize(
    "fn,match",
    [
        (lambda m: m.update(files=[]), "do not match its members"),
        (lambda m: m["files"].pop(), "do not match its members"),
        (
            lambda m: m["files"].append({"path": "../outside.bin", "sha256": "0" * 64, "bytes": 1}),
            "plain file name",
        ),
        (
            lambda m: m["files"].append({"path": "/abs/x", "sha256": "0" * 64, "bytes": 1}),
            "plain file name",
        ),
        (lambda m: m.update(members=["../../etc/x"]), "plain file name"),
        (lambda m: m.update(members=m["members"] + m["members"][:1]), "duplicate"),
        (lambda m: m.update(format="hcd_priya.safe/0"), "format"),
    ],
)
def test_manifest_must_tie_members_to_files(bundle, fn, match):
    bd, _ = bundle
    _edit_manifest(bd, fn)
    with pytest.raises(S.SafeFormatError, match=match):
        S.verify_manifest(bd)


def test_manifest_refuses_tampering_extras_and_symlinks(bundle, tmp_path):
    bd, pres = bundle
    open(os.path.join(bd, "evil.pkl"), "wb").write(b"\x80\x04.")
    with pytest.raises(S.SafeFormatError, match="does not list"):
        S.verify_manifest(bd)
    assert S.verify_manifest(bd, allow_extra=True) == pres  # explicit opt-in only
    os.remove(os.path.join(bd, "evil.pkl"))
    with open(pres[0] + ".json", "a") as f:
        f.write(" ")
    with pytest.raises(S.SafeFormatError, match="does not match the manifest"):
        S.verify_manifest(bd)
    with pytest.raises(S.SafeFormatError, match="does not match the manifest"):
        S.load_bundle_historical(bd, expected_manifest_sha256=None, reason="test")
    os.remove(pres[1] + ".safetensors")
    os.symlink(str(tmp_path / "elsewhere"), pres[1] + ".safetensors")
    with pytest.raises(S.SafeFormatError):
        S.verify_manifest(bd, allow_extra=True)
    with pytest.raises(S.SafeFormatError, match="not directly inside"):
        S.write_manifest(str(tmp_path / "other"), pres, {})


def test_clean_process_load(bundle, tmp_path):
    bd, _ = bundle
    out = str(tmp_path / "clean.npy")
    code = (
        "import sys, numpy as np, jax; jax.config.update('jax_enable_x64', True); sys.path.insert(0, sys.argv[1]);"
        "from hcd_analysis.emulator import safe_io as S;"
        "(m, _, _), _ = S.load_bundle_historical(sys.argv[2], expected_manifest_sha256=sys.argv[4], reason='t');"
        "import jax.numpy as jnp;"
        "o = m(jnp.linspace(0.1, 0.9, 10), 0.4); np.save(sys.argv[3], np.asarray(o['P_filt_resid']))"
    )
    sha = S.sha256_file(os.path.join(bd, "manifest.json"))
    subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, REPO, bd, out, sha], check=True, cwd=str(tmp_path)
    )
    ref = np.asarray(
        Emulator(**ARCH, key=jax.random.PRNGKey(0))(jnp.linspace(0.1, 0.9, 10), 0.4)["P_filt_resid"]
    )
    assert np.load(out).tobytes() == ref.tobytes()


def test_float64_member_refuses_a_float32_network(member, tmp_path):
    """The package enables x64 on import; a process that turns it off again gets float32 networks and must be refused,
    never handed a silently down-cast one."""
    prefix, _ = member
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); from hcd_analysis.emulator import safe_io as S; import jax;"
        "jax.config.update('jax_enable_x64', False)\n"
        "try:\n    S.load_member(sys.argv[2])\nexcept S.SafeFormatError as e:\n"
        "    assert 'never down-cast' in str(e), e; print('REFUSED')\n"
    )
    r = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, REPO, prefix],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().endswith("REFUSED")


# --------------------------------------------------------------------------------------------- HDF5 tables
def test_table_round_trip_and_refusals(tmp_path):
    p = str(tmp_path / "mf.h5")
    r = np.random.default_rng(1)
    arrays = {
        "ratio": r.normal(size=(3, 13, 8)),
        "z": np.linspace(2.2, 4.6, 13),
        "mask": np.ones((3, 8), bool),
    }
    S.write_table_h5(p, arrays, {"kind": "synthetic MF table", "n_sims": 3})
    req = {"ratio": ("float64", (3, 13, None)), "z": ("float64", (13,)), "mask": ("bool", (3, 8))}
    out, attrs = S.read_table_h5(p, req)
    assert (
        all(out[k].tobytes() == arrays[k].tobytes() for k in arrays)
        and out["ratio"].dtype == np.float64
    )
    assert attrs["kind"] == "synthetic MF table" and attrs["format"] == S.FORMAT
    with pytest.raises(S.SafeFormatError, match="exists"):
        S.write_table_h5(p, arrays, {})
    with pytest.raises(S.SafeFormatError, match="expected float32"):
        S.read_table_h5(p, {"ratio": ("float32", (3, 13, 8))})
    with pytest.raises(S.SafeFormatError, match="expected"):
        S.read_table_h5(p, {"ratio": ("float64", (3, 12, 8))})
    with pytest.raises(S.SafeFormatError, match="missing"):
        S.read_table_h5(p, {"nope": ("float64", (1,))})
    with pytest.raises(S.SafeFormatError, match="not numeric"):
        S.write_table_h5(str(tmp_path / "s.h5"), {"s": np.array(["a"])}, {})
    with pytest.raises(S.SafeFormatError, match="attribute"):
        S.write_table_h5(str(tmp_path / "t.h5"), {}, {"bad": [1, 2]})
    with pytest.raises(S.SafeFormatError, match="reserved"):
        S.write_table_h5(str(tmp_path / "u.h5"), {}, {"format": "other"})
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".tmp_")]
    with pytest.raises(S.SafeFormatError, match="plain top-level"):
        S.read_table_h5(p, {"g/ratio": ("float64", (3, 13, 8))})


def test_table_reader_refuses_links_vlen_and_array_attributes(tmp_path):
    import h5py

    other = str(tmp_path / "other.h5")
    with h5py.File(other, "w") as f:
        f["x"] = np.arange(3.0)
    p = str(tmp_path / "t.h5")
    S.write_table_h5(p, {"a": np.arange(3.0)}, {})
    with h5py.File(p, "a") as f:
        f["ext"] = h5py.ExternalLink(other, "x")
        f["soft"] = h5py.SoftLink("/a")
        f.create_dataset("vlen", (2,), dtype=h5py.vlen_dtype(np.float64))
    for name, match in (("ext", "ExternalLink"), ("soft", "SoftLink"), ("vlen", "expected")):
        with pytest.raises(S.SafeFormatError, match=match):
            S.read_table_h5(p, {name: ("float64", (None,))})
    with h5py.File(p, "a") as f:
        f.attrs["arr"] = np.arange(3)
    with pytest.raises(S.SafeFormatError, match="attribute"):
        S.read_table_h5(p, {"a": ("float64", (3,))})


def test_publish_refuses_symlink_targets_and_accepts_only_our_own_inode(tmp_path, monkeypatch):
    target = tmp_path / "t.safetensors"
    os.symlink(str(tmp_path / "nowhere"), target)  # a dangling symlink at the target
    with pytest.raises(S.SafeFormatError, match="never overwritten"):
        S.write_safetensors(str(target), {"a": np.zeros(1)})
    assert os.path.islink(target) and not os.path.exists(tmp_path / "nowhere")
    real_link = os.link

    def link_then_eexist(src, dst, **kw):  # an NFS retransmit: done, then EEXIST
        real_link(src, dst, **kw)
        raise FileExistsError(dst)

    monkeypatch.setattr(os, "link", link_then_eexist)
    p = str(tmp_path / "ok.safetensors")
    S.write_safetensors(p, {"a": np.arange(3.0)})
    assert np.array_equal(S.read_safetensors(p)[0]["a"], np.arange(3.0))
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".tmp_")]


def test_publish_refuses_a_temporary_file_replaced_before_linking(tmp_path):
    """The temporary file is written through its own descriptor; if its name is swapped (a symlink, or another file)
    before the hard link, nothing is published and the swapped-in target is left untouched."""
    victim = tmp_path / "victim"
    victim.write_bytes(b"keep me")
    for swap in ("symlink", "file"):
        path = str(tmp_path / f"out_{swap}.safetensors")
        tmp, held, _ = S._write_tmp(path, "", lambda fh: fh.write(b"payload"))
        os.remove(tmp)
        if swap == "symlink":
            os.symlink(str(victim), tmp)
        else:
            open(tmp, "wb").write(b"other")  # may reuse a freed inode number: held is open
        with pytest.raises(S.SafeFormatError, match="replaced before publishing"):
            S._publish(tmp, path, held)
        assert held.closed and not os.path.lexists(path) and not os.path.lexists(tmp)
    assert victim.read_bytes() == b"keep me"


def test_hdf5_writer_never_reopens_the_temporary_name(tmp_path, monkeypatch):
    import h5py

    real = h5py.File

    def only_file_objects(name, *a, **k):
        assert not isinstance(name, (str, bytes, os.PathLike)), f"reopened by name: {name}"
        return real(name, *a, **k)

    monkeypatch.setattr(h5py, "File", only_file_objects)
    S.write_table_h5(str(tmp_path / "t.h5"), {"x": np.arange(3.0)}, {"a": 1})
    monkeypatch.setattr(h5py, "File", real)
    assert np.array_equal(
        S.read_table_h5(str(tmp_path / "t.h5"), {"x": ("float64", (3,))})[0]["x"], np.arange(3.0)
    )


def test_json_refuses_nan_and_infinity_and_loader_refuses_non_finite_arrays(member):
    for c in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(S.SafeFormatError, match="not allowed"):
            S._json_loads(f'{{"a": {c}}}'.encode(), "x.json")
    for c in ("1e999", "-1e400"):  # numbers that overflow to infinity
        with pytest.raises(S.SafeFormatError, match="not finite"):
            S._json_loads(f'{{"a": {c}}}'.encode(), "x.json")
    assert S._json_loads(b'{"a": 1e300}', "x.json") == {"a": 1e300}
    with pytest.raises(S.SafeFormatError, match="non-finite"):
        S._require_finite({"model.w": np.array([1.0, np.nan])}, "m")
    S._require_finite({"model.w": np.array([1.0, 2.0]), "i": np.array([1, 2])}, "m")
    prefix, _ = member  # the safetensors loader itself refuses a NaN
    st, js = prefix + ".safetensors", prefix + ".json"
    tensors, smeta = S.read_safetensors(st)
    name = sorted(n for n in tensors if n.startswith("model"))[0]
    tensors[name] = np.full_like(tensors[name], np.nan)
    os.remove(st)
    S.write_safetensors(st, tensors, metadata=smeta)
    doc = json.load(open(js))
    doc["safetensors_sha256"] = S.sha256_file(st)
    os.remove(js)
    json.dump(doc, open(js, "w"))
    with pytest.raises(S.SafeFormatError, match="non-finite"):
        S.load_member(prefix)


def test_published_files_get_the_umask_mode(tmp_path):
    old = os.umask(0o022)
    try:
        S.write_safetensors(str(tmp_path / "a.safetensors"), {"a": np.zeros(1)})
        S.write_table_h5(str(tmp_path / "a.h5"), {"x": np.zeros(2)}, {})
    finally:
        os.umask(old)
    for f in ("a.safetensors", "a.h5"):
        assert os.stat(tmp_path / f).st_mode & 0o777 == 0o644


def test_normalizer_keys_beyond_int64_digits_are_refused_at_write():
    with pytest.raises(S.SafeFormatError, match="18 digits"):
        S.norm_tensors({"a": {10**18: np.zeros(1)}})


def test_table_reader_refuses_invalid_utf8_string_attributes_and_dot_names(tmp_path):
    import h5py

    p = str(tmp_path / "u.h5")
    S.write_table_h5(p, {"a": np.arange(3.0)}, {})
    with pytest.raises(S.SafeFormatError, match="plain top-level"):
        S.read_table_h5(p, {".": ("float64", (3,))})
    with h5py.File(p, "a") as f:
        f.attrs.create("bad", b"\xff\xfe", dtype=h5py.string_dtype("utf-8"))
    with pytest.raises(S.SafeFormatError, match="UTF-8"):
        S.read_table_h5(p, {"a": ("float64", (3,))})


def test_table_writer_refuses_names_the_reader_refuses(tmp_path):
    for bad in ("-a", "a b", ".hidden", "k(1)", "g/a"):
        with pytest.raises(S.SafeFormatError, match="not plain"):
            S.write_table_h5(str(tmp_path / f"n{abs(hash(bad))}.h5"), {bad: np.zeros(1)}, {})
    S.write_table_h5(str(tmp_path / "ok.h5"), {"a.b": np.zeros(1)}, {})
    assert S.read_table_h5(str(tmp_path / "ok.h5"), {"a.b": ("float64", (1,))})[0]["a.b"].shape == (
        1,
    )
