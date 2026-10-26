"""The HDF5 + JSON release-format prototype (hcd_analysis/emulator/safe_h5.py).

Synthetic only: a small Emulator from a fixed PRNG key and a synthetic normalizer with the production key layout.
Malformed artifacts are built by editing a valid file and re-anchoring its sha256 in the sidecar, so each refusal is
exercised by the structural checks, not by the digest."""
import json
import os
import subprocess
import sys

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator import safe_h5 as H
from hcd_analysis.emulator.model import Emulator
from hcd_analysis.emulator.schema import CHECKPOINT_SCHEMA_VERSION

ARCH = {"in_dim": 10, "n_k": 8, "n_basis": 4, "baseline_n_layers": 1, "baseline_width": 16}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEAF = ".enc.layers[0].weight"


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
    H.export_member(prefix, model, _meta(), _norm())
    return prefix, model


def _tree_equal(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(_tree_equal(a[k], b[k]) for k in a)
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _reanchor(prefix, index_from_file=True):
    """Re-anchor the sidecar to the (edited) .h5: sha256, and the tensor index if requested."""
    with open(prefix + ".json") as f:
        doc = json.load(f)
    doc["h5_sha256"] = H.sha256_file(prefix + ".h5")
    if index_from_file:
        idx = {}
        with h5py.File(prefix + ".h5", "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    n = ("model" + name[len("model/"):]) if name.startswith("model/") else name
                    idx[n] = {"shape": list(obj.shape), "dtype": "F64"}
            f.visititems(visit)
        doc["tensors"] = idx
    os.remove(prefix + ".json")
    with open(prefix + ".json", "w") as f:
        json.dump(doc, f)


def _edit(prefix, fn, **kw):
    with h5py.File(prefix + ".h5", "a") as f:
        fn(f)
    _reanchor(prefix, **kw)


# --------------------------------------------------------------------------------------------- round trip
def test_round_trip_is_exact_and_names_are_tree_paths(member):
    prefix, model = member
    m2, meta2, norm2 = H.load_member(prefix)
    l1, t1 = jax.tree_util.tree_flatten_with_path(model)
    l2, t2 = jax.tree_util.tree_flatten_with_path(m2)
    assert t1 == t2
    for (p1, x1), (p2, x2) in zip(l1, l2):
        assert jax.tree_util.keystr(p1) == jax.tree_util.keystr(p2)
        assert np.asarray(x1).dtype == np.asarray(x2).dtype == np.float64
        assert np.asarray(x1).tobytes() == np.asarray(x2).tobytes()
    assert meta2 == json.loads(json.dumps(_meta())) and _tree_equal(_norm(), norm2)
    assert set(norm2["P_filt"]["cell_mean"]) == set(range(5))
    with h5py.File(prefix + ".h5", "r") as f:
        assert set(f.keys()) == {"model", "norm"} and LEAF in f["model"]
        assert "#0" in f["norm/P_filt/cell_mean"]
        d = f["model"][LEAF]
        assert d.chunks is None and d.compression is None and d.dtype == np.dtype("<f8")


def test_predictions_and_gradients_are_identical(member):
    prefix, model = member
    m2, _, _ = H.load_member(prefix)
    x = jnp.linspace(0.1, 0.9, 10)
    for tau0 in (0.2, 0.5):
        o1, o2 = model(x, tau0), m2(x, tau0)
        assert set(o1) == set(o2) and all(np.asarray(o1[k]).tobytes() == np.asarray(o2[k]).tobytes() for k in o1)

    def f(mod, x, t):
        o = mod(x, t)
        return jnp.sum(o["P_filt_resid"] ** 2) + jnp.sum(o["P_filt_base"])
    g1 = jax.grad(f, argnums=(1, 2))(model, x, 0.3)
    g2 = jax.grad(f, argnums=(1, 2))(m2, x, 0.3)
    assert all(np.asarray(a).tobytes() == np.asarray(b).tobytes() for a, b in zip(g1, g2))


def test_module_never_uses_pickle_and_export_never_overwrites(member):
    src = open(H.__file__).read()
    assert "import pickle" not in src and "pickle." not in src and "allow_pickle" not in src
    prefix, model = member
    before = open(prefix + ".h5", "rb").read()
    with pytest.raises(H.SafeFormatError, match="never overwrites"):
        H.export_member(prefix, model, _meta(), _norm())
    assert open(prefix + ".h5", "rb").read() == before


# --------------------------------------------------------------------------------------------- refusals
def _other(tmp_path):
    p = str(tmp_path / "other.h5")
    with h5py.File(p, "w") as f:
        f["x"] = np.zeros(4)
    return p


@pytest.mark.parametrize("edit,match", [
    (lambda f, o: f["model"].__setitem__("alias", h5py.SoftLink(f"/model/{LEAF}")), "only hard links"),
    (lambda f, o: f["norm"].__setitem__("ext", h5py.ExternalLink(o, "/x")), "only hard links"),
    (lambda f, o: f["norm"].__setitem__("alias", f["model"][LEAF]), "several links"),
    (lambda f, o: f["model"][LEAF].attrs.__setitem__("note", 1), "carries attributes"),
    (lambda f, o: f.attrs.__setitem__("extra", 1), "root attributes"),
    (lambda f, o: f.create_group("extra"), "exactly /model and /norm"),
    (lambda f, o: f["norm"].__setitem__("dt", np.dtype("f8")), "not a group or dataset"),
    (lambda f, o: f["norm"].create_dataset("ext", shape=(4,), dtype="<f8", external=[(o + ".raw", 0, 32)]),
     "is external;"),
])
def test_structural_refusals(member, tmp_path, edit, match):
    prefix, _ = member
    o = _other(tmp_path)
    open(o + ".raw", "wb").write(b"\0" * 32)
    _edit(prefix, lambda f: edit(f, o))
    with pytest.raises(H.SafeFormatError, match=match):
        H.load_member(prefix)


@pytest.mark.parametrize("create_kw,match", [
    (lambda a: dict(data=a, chunks=True), "is chunked;"),
    (lambda a: dict(data=a, compression="gzip"), "chunked and filtered"),
    (lambda a: dict(data=a, chunks=True, fletcher32=True), "chunked and filtered"),
    (lambda a: dict(data=a, chunks=True, shuffle=True), "chunked and filtered"),
    (lambda a: dict(data=a.astype(">f8")), "little-endian float64"),
    (lambda a: dict(data=a.astype(np.float32)), "little-endian float64"),
    (lambda a: dict(data=a[:, :3]), "network expects"),
])
def test_dataset_layout_and_dtype_refusals(member, create_kw, match):
    prefix, _ = member

    def fn(f):
        a = f["model"][LEAF][...]
        del f["model"][LEAF]
        f["model"].create_dataset(LEAF, **create_kw(a))
    _edit(prefix, fn)
    with pytest.raises(H.SafeFormatError, match=match):
        H.load_member(prefix)


def test_reshaped_normalizer_and_extra_model_tensor_are_refused(member, tmp_path):
    prefix, _ = member
    _edit(prefix, lambda f: f["model"].create_dataset(".extra", data=np.zeros(3)))
    with pytest.raises(H.SafeFormatError, match="differ from the network"):
        H.load_member(prefix)
    pre2 = str(tmp_path / "m1")
    H.export_member(pre2, Emulator(**ARCH, key=jax.random.PRNGKey(3)), _meta(), _norm())

    def fn(f):
        del f["norm/dndx/mean"]
        f["norm/dndx"].create_dataset("mean", data=np.zeros(4))
    _edit(pre2, fn)
    with pytest.raises(H.SafeFormatError, match=r"norm/dndx/mean: .*expected float64 \(3,\)"):
        H.load_member(pre2)


def test_virtual_dataset_is_refused(member):
    prefix, _ = member

    def fn(f):
        layout = h5py.VirtualLayout(shape=(3,), dtype="<f8")
        layout[:] = h5py.VirtualSource(".", "norm/dndx/mean", shape=(3,))
        f["norm"].create_virtual_dataset("vds", layout)
    _edit(prefix, fn)
    with pytest.raises(H.SafeFormatError, match="is virtual;"):
        H.load_member(prefix)


def test_missing_extra_and_noncanonical_tensors_are_refused(member):
    prefix, _ = member
    _edit(prefix, lambda f: f["model"].__delitem__(LEAF))
    with pytest.raises(H.SafeFormatError, match="differ from the network"):
        H.load_member(prefix)


def test_noncanonical_normalizer_key_is_refused(member):
    prefix, _ = member
    _edit(prefix, lambda f: f["norm/P_filt/cell_mean"].create_dataset("#01", data=np.zeros((4, 8))))
    with pytest.raises(H.SafeFormatError, match="canonical"):
        H.load_member(prefix)


def test_user_block_is_refused(member, tmp_path):
    prefix, _ = member
    src = prefix + ".h5"
    dst = str(tmp_path / "ub.h5")
    with h5py.File(src, "r") as a, h5py.File(dst, "w", userblock_size=512) as b:
        b.attrs["format"] = a.attrs["format"]
        for k in a:
            a.copy(a[k], b, name=k)
    os.remove(src)
    os.rename(dst, src)
    _reanchor(prefix)
    with pytest.raises(H.SafeFormatError, match="user block"):
        H.load_member(prefix)


def test_symlinked_member_is_refused(member, tmp_path):
    prefix, _ = member
    real = str(tmp_path / "real.h5")
    os.rename(prefix + ".h5", real)
    os.symlink(real, prefix + ".h5")
    with pytest.raises(H.SafeFormatError, match="is a symlink"):
        H.load_member(prefix)


def test_unanchored_bytes_are_never_parsed(member, monkeypatch):
    prefix, _ = member
    raw = bytearray(open(prefix + ".h5", "rb").read())
    raw[-5] ^= 0xFF
    open(prefix + ".h5", "wb").write(bytes(raw))

    def boom(*a, **k):
        raise AssertionError("libhdf5 opened bytes that failed the digest")
    monkeypatch.setattr(h5py, "File", boom)
    with pytest.raises(H.SafeFormatError, match="sha256"):
        H.load_member(prefix)


@pytest.mark.parametrize("edit,match", [
    (lambda d: d.update(format="hcd_priya.h5/0"), "format"),
    (lambda d: d.update(h5_file="../m0.h5"), "h5_file"),
    (lambda d: d["tensors"].pop("norm/dndx/mean"), "index"),
    (lambda d: d["meta"]["arch_cfg"].update(n_k=10 ** 9), "int in"),
    (lambda d: d.pop("tensors"), "lacks"),
])
def test_sidecar_refusals(member, edit, match):
    prefix, _ = member
    with open(prefix + ".json") as f:
        doc = json.load(f)
    edit(doc)
    os.remove(prefix + ".json")
    with open(prefix + ".json", "w") as f:
        json.dump(doc, f)
    with pytest.raises(H.SafeFormatError, match=match):
        H.load_member(prefix)


def test_float32_networks_are_refused_in_a_non_x64_process(member, tmp_path):
    prefix, _ = member
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from hcd_analysis.emulator import safe_h5 as H; import jax;"
            "jax.config.update('jax_enable_x64', False)\n"
            "try:\n    H.load_member(sys.argv[2])\nexcept H.SafeFormatError as e:\n"
            "    assert 'never down-cast' in str(e), e; print('REFUSED')\n")
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code, REPO, prefix], capture_output=True, text=True,
                       cwd=str(tmp_path))
    assert r.returncode == 0 and r.stdout.strip().endswith("REFUSED"), r.stderr[-2000:]


# --------------------------------------------------------------------------------------------- bundles, clean process
@pytest.fixture
def bundle(tmp_path):
    bd = tmp_path / "bundle"
    bd.mkdir()
    pres = []
    for i in range(2):
        pre = str(bd / f"m{i}")
        H.export_member(pre, Emulator(**ARCH, key=jax.random.PRNGKey(i)), _meta(seed=i), _norm())
        pres.append(pre)
    H.write_manifest(str(bd), pres, {"source": "synthetic"})
    return str(bd), pres


def test_bundle_round_trip_manifest_and_unlisted_files(bundle):
    bd, pres = bundle
    assert H.verify_manifest(bd) == pres
    sha = H.sha256_file(os.path.join(bd, "manifest.json"))
    assert [m["seed"] for _, m, _ in H.load_bundle(bd, expected_manifest_sha256=sha)] == [0, 1]
    with pytest.raises(TypeError):
        H.load_bundle(bd)
    with pytest.raises(H.SafeFormatError, match="manifest"):
        H.load_bundle(bd, expected_manifest_sha256="f" * 64)
    open(os.path.join(bd, "evil.pkl"), "wb").write(b"\x80\x04.")
    with pytest.raises(H.SafeFormatError, match="does not list"):
        H.verify_manifest(bd)


def test_clean_process_load(bundle, tmp_path):
    bd, _ = bundle
    out = str(tmp_path / "clean.npy")
    sha = H.sha256_file(os.path.join(bd, "manifest.json"))
    code = ("import sys, numpy as np, jax; jax.config.update('jax_enable_x64', True); sys.path.insert(0, sys.argv[1]);"
            "from hcd_analysis.emulator import safe_h5 as H;"
            "(m, _, _), _ = H.load_bundle(sys.argv[2], expected_manifest_sha256=sys.argv[4]); import jax.numpy as jnp;"
            "o = m(jnp.linspace(0.1, 0.9, 10), 0.4); np.save(sys.argv[3], np.asarray(o['P_filt_resid']))")
    subprocess.run([sys.executable, "-I", "-B", "-c", code, REPO, bd, out, sha], check=True, cwd=str(tmp_path))
    ref = np.asarray(Emulator(**ARCH, key=jax.random.PRNGKey(0))(jnp.linspace(0.1, 0.9, 10), 0.4)["P_filt_resid"])
    assert np.load(out).tobytes() == ref.tobytes()
