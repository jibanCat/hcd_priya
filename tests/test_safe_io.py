"""The pickle-free release format (hcd_analysis/emulator/safe_io.py, docs/release_format.md).

Synthetic only: a small Emulator built from a fixed PRNG key and a synthetic normalizer with the production key layout
(nested dicts, integer keys). The real-member pilot is scripts/export_safe_bundle.py, run outside the test suite."""
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
    m = {"schema_version": CHECKPOINT_SCHEMA_VERSION, "arch_cfg": dict(ARCH), "seed": 3,
         "k_com_hmpc": list(np.arange(1, ARCH["n_k"] + 1) * 0.05), "L_box_hmpc": 120.0,
         "k_convention": "synthetic", "cache_sha256": "0" * 64, "param_limits": [[0.0, 1.0]] * 9,
         "cache_path": "synthetic.h5", "git_sha": "synthetic", "n_k": ARCH["n_k"], "recipe": {"w": 1.0}}
    m.update(over)
    return m


def _norm():
    r = np.random.default_rng(0)
    return {"f_nhi": {"mean": r.normal(size=30), "std": r.uniform(1, 2, 30)},
            "dndx": {"mean": r.normal(size=3), "std": r.uniform(1, 2, 3)},
            "P_filt": {"mu_marg": r.normal(size=(4, 8)), "sig_marg": r.uniform(1, 2, (4, 8)),
                       "sig_cosmo": r.uniform(1, 2, (4, 8)),
                       "cell_mean": {i: r.normal(size=(4, 8)) for i in range(5)}},
            "delta": {"mean": r.normal(size=(3, 8)), "std": r.uniform(1, 2, (3, 8))}}


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
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


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
        assert np.array_equal(np.asarray(x1), np.asarray(x2))
    assert meta2 == json.loads(json.dumps(_meta()))
    assert _tree_equal(_norm(), norm2)
    assert set(norm2["P_filt"]["cell_mean"]) == set(range(5))              # integer keys come back as ints


def test_predictions_and_gradients_are_identical(member):
    prefix, model = member
    m2, _, _ = S.load_member(prefix)
    x = jnp.linspace(0.1, 0.9, 10)
    for tau0 in (0.2, 0.5):
        o1, o2 = model(x, tau0), m2(x, tau0)
        assert set(o1) == set(o2)
        for k in o1:
            assert np.array_equal(np.asarray(o1[k]), np.asarray(o2[k]))

    def f(mod, x, t):
        o = mod(x, t)
        return jnp.sum(o["P_filt_resid"] ** 2) + jnp.sum(o["P_filt_base"])
    g1 = jax.grad(f, argnums=(1, 2))(model, x, 0.3)
    g2 = jax.grad(f, argnums=(1, 2))(m2, x, 0.3)
    for a, b in zip(g1, g2):
        assert np.array_equal(np.asarray(a), np.asarray(b))


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


def test_dtypes_and_byte_order(tmp_path):
    p = str(tmp_path / "t.safetensors")
    arrs = {"f64": np.arange(6, dtype=np.float64).reshape(2, 3), "f32": np.ones(3, np.float32),
            "i64": np.array([1, -2], np.int64), "be": np.arange(4, dtype=">f8"), "scalar": np.float64(2.5),
            "empty": np.zeros((0, 3)), "b": np.array([True, False])}
    S.write_safetensors(p, arrs)
    out, _ = S.read_safetensors(p)
    assert out["f64"].dtype == np.float64 and out["f32"].dtype == np.float32 and out["i64"].dtype == np.int64
    assert out["be"].dtype == np.dtype("<f8") and np.array_equal(out["be"], np.arange(4.0))
    assert out["scalar"].shape == () and out["scalar"] == 2.5 and out["empty"].shape == (0, 3)
    assert np.array_equal(out["b"], [True, False]) and out["b"].dtype == np.bool_
    for k in ("f64", "f32", "i64"):
        assert np.array_equal(out[k], arrs[k])


def test_module_never_uses_pickle():
    src = open(S.__file__).read()
    assert "import pickle" not in src and "pickle." not in src and "allow_pickle=True" not in src


# --------------------------------------------------------------------------------------------- refusals
def _rewrite_header(path, mutate):
    raw = open(path, "rb").read()
    (n,) = struct.unpack("<Q", raw[:8])
    h = json.loads(raw[8:8 + n])
    h = mutate(h)
    hb = json.dumps(h).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + raw[8 + n:])


def test_refuses_corrupt_files(tmp_path):
    p = str(tmp_path / "t.safetensors")
    S.write_safetensors(p, {"a": np.arange(4.0), "b": np.ones((2, 2))})
    raw = open(p, "rb").read()
    cases = {"short": raw[:5], "badlen": struct.pack("<Q", 10 ** 9) + raw[8:], "truncated": raw[:-8],
             "trailing": raw + b"\0" * 8, "notjson": raw[:8] + b"\xff" * (len(raw) - 8)}
    for name, blob in cases.items():
        q = str(tmp_path / f"{name}.safetensors")
        open(q, "wb").write(blob)
        with pytest.raises(S.SafeFormatError):
            S.read_safetensors(q)


@pytest.mark.parametrize("mutate", [
    lambda h: {**h, "a": {**h["a"], "dtype": "F16"}},                                    # unsupported dtype
    lambda h: {**h, "a": {**h["a"], "shape": [5]}},                                       # shape vs bytes
    lambda h: {**h, "a": {**h["a"], "data_offsets": [0, 16]}},                            # gap / size
    lambda h: {**h, "b": {**h["b"], "data_offsets": [0, 32]}},                            # overlap
    lambda h: {**h, "a": {**h["a"], "extra": 1}},                                         # malformed entry
    lambda h: {**h, "a": {**h["a"], "shape": [True, 4]}},                                 # bool is not an int
    lambda h: {**h, "__metadata__": {"x": 1}},                                            # non-string metadata
])
def test_refuses_malformed_headers(tmp_path, mutate):
    p = str(tmp_path / "t.safetensors")
    S.write_safetensors(p, {"a": np.arange(4.0), "b": np.ones((2, 2))})
    _rewrite_header(p, mutate)
    with pytest.raises(S.SafeFormatError):
        S.read_safetensors(p)


def test_refuses_duplicate_header_keys(tmp_path):
    p = str(tmp_path / "t.safetensors")
    hb = b'{"a":{"dtype":"F64","shape":[1],"data_offsets":[0,8]},"a":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}}'
    hb += b" " * (-len(hb) % 8)
    open(p, "wb").write(struct.pack("<Q", len(hb)) + hb + np.float64(1).tobytes())
    with pytest.raises(S.SafeFormatError, match="duplicate"):
        S.read_safetensors(p)


def test_refuses_non_numeric_arrays(tmp_path):
    for bad in ({"a": np.array(["x"])}, {"a": np.array([object()], dtype=object)}, {"a": np.zeros(2, np.float16)}):
        with pytest.raises(S.SafeFormatError):
            S.write_safetensors(str(tmp_path / "t.safetensors"), bad)
    with pytest.raises(S.SafeFormatError):
        S.norm_tensors({"a": np.array([object()], dtype=object)})
    with pytest.raises(S.SafeFormatError):
        S.norm_tensors({"a/b": np.zeros(1)})                                     # a key the names cannot carry


def test_export_never_overwrites(member):
    prefix, model = member
    before = open(prefix + ".safetensors", "rb").read()
    with pytest.raises(S.SafeFormatError, match="never overwrites"):
        S.export_member(prefix, model, _meta(), _norm())
    assert open(prefix + ".safetensors", "rb").read() == before


def _edit_sidecar(prefix, fn):
    with open(prefix + ".json") as f:
        doc = json.load(f)
    fn(doc)
    with open(prefix + ".json", "w") as f:
        json.dump(doc, f)


@pytest.mark.parametrize("edit,match", [
    (lambda d: d.update(format="hcd_priya.safe/0"), "format"),
    (lambda d: d["meta"].update(schema_version="1.0"), "schema"),
    (lambda d: d["meta"]["arch_cfg"].update(activation="relu"), "allowlist"),
    (lambda d: d["meta"]["arch_cfg"].update(n_k=8.0), "positive int"),
    (lambda d: d["meta"].update(seed="3"), "seed"),
    (lambda d: d["meta"].update(cache_sha256="abc"), "cache_sha256"),
    (lambda d: d["meta"]["recipe"].update(kfkms=[1.0]), "velocity"),
    (lambda d: d.update(safetensors_sha256="0" * 64), "sha256"),
    (lambda d: d["tensors"].pop("norm/dndx/mean"), "index"),
])
def test_load_refuses_bad_sidecar(member, edit, match):
    prefix, _ = member
    _edit_sidecar(prefix, edit)
    with pytest.raises(S.SafeFormatError, match=match):
        S.load_member(prefix)


def _replace_tensors(prefix, fn):
    t, meta = S.read_safetensors(prefix + ".safetensors")
    fn(t)
    os.remove(prefix + ".safetensors")
    S.write_safetensors(prefix + ".safetensors", t, metadata=meta)
    _edit_sidecar(prefix, lambda d: d.update(
        safetensors_sha256=S.sha256_file(prefix + ".safetensors"),
        tensors={n: {"shape": list(a.shape), "dtype": S._code(a.dtype)} for n, a in t.items()}))


@pytest.mark.parametrize("fn,match", [
    (lambda t: t.pop("model.enc.layers[0].bias"), "differ from the rebuilt network"),
    (lambda t: t.update({"model.extra": np.zeros(2)}), "differ from the rebuilt network"),
    (lambda t: t.update({"model.enc.layers[0].bias": np.zeros(3)}), "expects"),
    (lambda t: t.update({"model.enc.layers[0].bias": t["model.enc.layers[0].bias"].astype(np.float32)}),
     "never down-cast"),
])
def test_load_refuses_tensor_mismatch(member, fn, match):
    prefix, _ = member
    _replace_tensors(prefix, fn)
    with pytest.raises(S.SafeFormatError, match=match):
        S.load_member(prefix)


def test_corrupted_bytes_fail_the_digest(member):
    prefix, _ = member
    raw = bytearray(open(prefix + ".safetensors", "rb").read())
    raw[-3] ^= 0xFF
    open(prefix + ".safetensors", "wb").write(bytes(raw))
    with pytest.raises(S.SafeFormatError, match="sha256"):
        S.load_member(prefix)


# --------------------------------------------------------------------------------------------- manifest, clean process
def test_manifest_round_trip_and_tamper(member, tmp_path):
    prefix, _ = member
    S.write_manifest(str(tmp_path), [prefix], {"source": "synthetic"})
    assert S.verify_manifest(str(tmp_path)) == [prefix]
    with pytest.raises(FileExistsError):
        S.write_manifest(str(tmp_path), [prefix], {})
    with open(prefix + ".json", "a") as f:
        f.write(" ")
    with pytest.raises(S.SafeFormatError, match="does not match the manifest"):
        S.verify_manifest(str(tmp_path))


def test_clean_process_load(member, tmp_path):
    prefix, model = member
    out = str(tmp_path / "clean.npy")
    code = ("import sys, numpy as np, jax; jax.config.update('jax_enable_x64', True); sys.path.insert(0, sys.argv[1]);"
            "from hcd_analysis.emulator import safe_io as S;"
            "m, _, _ = S.load_member(sys.argv[2]); import jax.numpy as jnp;"
            "o = m(jnp.linspace(0.1, 0.9, 10), 0.4); np.save(sys.argv[3], np.asarray(o['P_filt_resid']))")
    subprocess.run([sys.executable, "-I", "-c", code, REPO, prefix, out], check=True, cwd=str(tmp_path))
    ref = np.asarray(model(jnp.linspace(0.1, 0.9, 10), 0.4)["P_filt_resid"])
    assert np.array_equal(np.load(out), ref)


# --------------------------------------------------------------------------------------------- HDF5 tables
def test_table_round_trip_and_refusals(tmp_path):
    p = str(tmp_path / "mf.h5")
    r = np.random.default_rng(1)
    arrays = {"ratio": r.normal(size=(3, 13, 8)), "z": np.linspace(2.2, 4.6, 13), "mask": np.ones((3, 8), bool)}
    S.write_table_h5(p, arrays, {"kind": "synthetic MF table", "n_sims": 3})
    out, attrs = S.read_table_h5(p, {"ratio": ("float64", 3), "z": ("float64", 1), "mask": ("bool", 2)})
    assert all(np.array_equal(out[k], arrays[k]) for k in arrays) and out["ratio"].dtype == np.float64
    assert attrs["kind"] == "synthetic MF table" and attrs["format"] == S.FORMAT
    with pytest.raises(S.SafeFormatError, match="exists"):
        S.write_table_h5(p, arrays, {})
    with pytest.raises(S.SafeFormatError, match="expected float32"):
        S.read_table_h5(p, {"ratio": ("float32", 3)})
    with pytest.raises(S.SafeFormatError, match="missing"):
        S.read_table_h5(p, {"nope": ("float64", 1)})
    with pytest.raises(S.SafeFormatError, match="not numeric"):
        S.write_table_h5(str(tmp_path / "s.h5"), {"s": np.array(["a"])}, {})
    with pytest.raises(S.SafeFormatError, match="attribute"):
        S.write_table_h5(str(tmp_path / "t.h5"), {}, {"bad": [1, 2]})


def test_float64_member_refuses_a_float32_skeleton(member, tmp_path):
    """The package enables x64 on import; a process that turns it off again gets float32 skeletons and must be refused,
    never handed a silently down-cast network."""
    prefix, _ = member
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from hcd_analysis.emulator import safe_io as S; import jax;"
            "jax.config.update('jax_enable_x64', False)\n"
            "try:\n    S.load_member(sys.argv[2])\nexcept S.SafeFormatError as e:\n"
            "    assert 'never down-cast' in str(e), e; print('REFUSED')\n")
    r = subprocess.run([sys.executable, "-I", "-c", code, REPO, prefix], capture_output=True, text=True,
                       cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().endswith("REFUSED")
