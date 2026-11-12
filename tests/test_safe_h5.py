"""The HDF5 + JSON release format (hcd_analysis/emulator/safe_h5.py): members and h5/2 bundles; the lineage checks of
the release loader are in tests/test_bundle_lineage.py, the historical h5/1 loader at the end of this file.

Synthetic only: a small Emulator from a fixed PRNG key and a synthetic normalizer with the production key layout;
bundles from hcd_analysis.emulator.release_testing. Malformed artifacts are built by editing a valid file and
re-anchoring its sha256 in the sidecar, so each refusal is exercised by the structural checks, not by the digest.
"""

import json
import os
import subprocess
import sys
import types

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator import release_testing as RT
from hcd_analysis.emulator import safe_h5 as H
from hcd_analysis.emulator.model import Emulator
from hcd_analysis.emulator.schema import CHECKPOINT_SCHEMA_VERSION

ARCH = {"in_dim": 10, "n_k": 8, "n_basis": 4, "baseline_n_layers": 1, "baseline_width": 16}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEAF = ".enc.layers[0].weight"


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
                    n = ("model" + name[len("model/") :]) if name.startswith("model/") else name
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
        assert set(o1) == set(o2) and all(
            np.asarray(o1[k]).tobytes() == np.asarray(o2[k]).tobytes() for k in o1
        )

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


@pytest.mark.parametrize(
    "edit,match",
    [
        (
            lambda f, o: f["model"].__setitem__("alias", h5py.SoftLink(f"/model/{LEAF}")),
            "only hard links",
        ),
        (lambda f, o: f["norm"].__setitem__("ext", h5py.ExternalLink(o, "/x")), "only hard links"),
        (lambda f, o: f["norm"].__setitem__("alias", f["model"][LEAF]), "several links"),
        (lambda f, o: f["model"][LEAF].attrs.__setitem__("note", 1), "carries attributes"),
        (lambda f, o: f.attrs.__setitem__("extra", 1), "root attributes"),
        (lambda f, o: f.create_group("extra"), "exactly /model and /norm"),
        (lambda f, o: f["norm"].__setitem__("dt", np.dtype("f8")), "not a group or dataset"),
        (
            lambda f, o: f["norm"].create_dataset(
                "ext", shape=(4,), dtype="<f8", external=[(o + ".raw", 0, 32)]
            ),
            "is external;",
        ),
    ],
)
def test_structural_refusals(member, tmp_path, edit, match):
    prefix, _ = member
    o = _other(tmp_path)
    open(o + ".raw", "wb").write(b"\0" * 32)
    _edit(prefix, lambda f: edit(f, o))
    with pytest.raises(H.SafeFormatError, match=match):
        H.load_member(prefix)


@pytest.mark.parametrize(
    "create_kw,match",
    [
        (lambda a: dict(data=a, chunks=True), "is chunked;"),
        (lambda a: dict(data=a, compression="gzip"), "chunked and filtered"),
        (lambda a: dict(data=a, chunks=True, fletcher32=True), "chunked and filtered"),
        (lambda a: dict(data=a, chunks=True, shuffle=True), "chunked and filtered"),
        (lambda a: dict(data=a.astype(">f8")), "little-endian float64"),
        (lambda a: dict(data=a.astype(np.float32)), "little-endian float64"),
        (lambda a: dict(data=a[:, :3]), "network expects"),
    ],
)
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


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda d: d.update(format="hcd_priya.h5/0"), "format"),
        (lambda d: d.update(h5_file="../m0.h5"), "h5_file"),
        (lambda d: d["tensors"].pop("norm/dndx/mean"), "index"),
        (lambda d: d["meta"]["arch_cfg"].update(n_k=10**9), "int in"),
        (lambda d: d.pop("tensors"), "lacks"),
    ],
)
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
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); from hcd_analysis.emulator import safe_h5 as H; import jax;"
        "jax.config.update('jax_enable_x64', False)\n"
        "try:\n    H.load_member(sys.argv[2])\nexcept H.SafeFormatError as e:\n"
        "    assert 'never down-cast' in str(e), e; print('REFUSED')\n"
    )
    r = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, REPO, prefix],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert r.returncode == 0 and r.stdout.strip().endswith("REFUSED"), r.stderr[-2000:]


# --------------------------------------------------------------------------------------------- bundles, clean process
@pytest.fixture
def bundle(tmp_path):
    info = RT.write_synthetic_bundle(str(tmp_path / "bundle"), arch=ARCH, norm=_norm())
    return info.bundle_dir, info.prefixes


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
    code = (
        "import sys, numpy as np, jax; jax.config.update('jax_enable_x64', True); sys.path.insert(0, sys.argv[1]);"
        "from hcd_analysis.emulator import safe_h5 as H;"
        "(m, _, _), _ = H.load_bundle(sys.argv[2], expected_manifest_sha256=sys.argv[4]); import jax.numpy as jnp;"
        "o = m(jnp.linspace(0.1, 0.9, 10), 0.4); np.save(sys.argv[3], np.asarray(o['P_filt_resid']))"
    )
    subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, REPO, bd, out, sha], check=True, cwd=str(tmp_path)
    )
    ref = np.asarray(
        Emulator(**ARCH, key=jax.random.PRNGKey(0))(jnp.linspace(0.1, 0.9, 10), 0.4)["P_filt_resid"]
    )
    assert np.load(out).tobytes() == ref.tobytes()


# --------------------------------------------------------------------------------------------- hardening
def _dcpl(layout):
    p = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    p.set_layout(layout)
    return p


def test_unallocated_compact_and_named_type_storage_are_refused(member, tmp_path):
    prefix, _ = member

    def unallocated(f):
        shape = f["norm/dndx/mean"].shape
        del f["norm/dndx/mean"]
        f["norm/dndx"].create_dataset("mean", shape=shape, dtype="<f8")  # declared, never written

    _edit(prefix, unallocated)
    with pytest.raises(H.SafeFormatError, match="no allocated storage"):
        H.load_member(prefix)
    for name, fn, match in (
        (
            "c",
            lambda f: _replace(
                f, "norm/dndx/mean", lambda a: dict(data=a, dcpl=_dcpl(h5py.h5d.COMPACT))
            ),
            "is compact; only contiguous",
        ),
        ("t", _named_type, "named datatype"),
    ):
        pre = str(tmp_path / name)
        H.export_member(pre, Emulator(**ARCH, key=jax.random.PRNGKey(3)), _meta(), _norm())
        _edit(pre, fn)
        with pytest.raises(H.SafeFormatError, match=match):
            H.load_member(pre)


def _replace(f, path, kw):
    a = f[path][...]
    grp, leaf = path.rsplit("/", 1)
    del f[path]
    f[grp].create_dataset(leaf, **kw(a))


def _named_type(f):
    f["norm"]["T"] = np.dtype("<f8")
    a = f["norm/dndx/mean"][...]
    del f["norm/dndx/mean"]
    f["norm/dndx"].create_dataset("mean", data=a, dtype=f["norm"]["T"])
    del f["norm"]["T"]  # the type stays committed, now unlinked


def test_normalizer_shapes_are_refused_before_any_data_is_read(member, monkeypatch):
    prefix, _ = member
    _edit(prefix, lambda f: _replace(f, "norm/dndx/mean", lambda a: dict(data=np.zeros(5))))
    reads = []
    real = h5py.Dataset.__getitem__

    def counting(self, *a, **k):
        reads.append(self.name)
        return real(self, *a, **k)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", counting)
    with pytest.raises(H.SafeFormatError, match="expected float64"):
        H.load_member(prefix)
    assert reads == []


@pytest.mark.parametrize(
    "bound,value,match",
    [
        ("_MAX_DATASET_BYTES", 64, "exceeds the size bound"),
        ("_MAX_TOTAL_BYTES", 4096, "size bound of a member"),
        ("_MAX_OBJECTS", 5, "more than 5 objects"),
        ("_MAX_DEPTH", 1, "nested deeper"),
    ],
)
def test_resource_bounds_are_enforced_from_metadata(member, monkeypatch, bound, value, match):
    prefix, _ = member
    monkeypatch.setattr(H, bound, value)
    with pytest.raises(H.SafeFormatError, match=match):
        H.load_member(prefix)


def test_top_level_links_and_array_format_attribute_are_refused(member, tmp_path, monkeypatch):
    prefix, _ = member

    def soft_norm(f):  # the top-level keys stay exactly {model, norm}
        f.move("norm", "model/norm_real")
        f["norm"] = h5py.SoftLink("/model/norm_real")

    _edit(prefix, soft_norm, index_from_file=False)
    with pytest.raises(H.SafeFormatError, match="plain hard-linked group"):
        H.load_member(prefix)
    pre = str(tmp_path / "fa")
    H.export_member(pre, Emulator(**ARCH, key=jax.random.PRNGKey(3)), _meta(), _norm())
    _edit(
        pre,
        lambda f: f.attrs.__setitem__(
            "format", np.array([H.FORMAT] * 2000, dtype=h5py.string_dtype())
        ),
    )
    read = []
    real_getitem = h5py.AttributeManager.__getitem__
    monkeypatch.setattr(
        h5py.AttributeManager,
        "__getitem__",
        lambda self, k: read.append(k) or real_getitem(self, k),
    )
    with pytest.raises(H.SafeFormatError, match="scalar string"):
        H.load_member(pre)
    assert read == []  # refused from metadata, before any value is read


def test_index_entry_json_digest_nan_and_truncation_are_refused(bundle, tmp_path):
    bd, pres = bundle
    sha = H.sha256_file(os.path.join(bd, "manifest.json"))
    js = pres[0] + ".json"
    json_sha = H.sha256_file(js)
    doc = json.load(open(js))
    doc["tensors"]["norm/dndx/mean"]["shape"] = [4]
    os.remove(js)
    json.dump(doc, open(js, "w"))
    with pytest.raises(H.SafeFormatError, match="member0.json does not match the manifest"):
        H.load_bundle(
            bd, expected_manifest_sha256=sha
        )  # the manifest's file digests refuse it first
    with pytest.raises(H.SafeFormatError, match="sha256 does not match the manifest"):
        H.load_member(
            pres[0], expected_json_sha256=json_sha
        )  # and so does the member's own digest check
    with pytest.raises(H.SafeFormatError, match="JSON index entry"):
        H.load_member(pres[0])
    raw = open(js).read().replace('"L_box_hmpc": 120.0', '"L_box_hmpc": NaN')
    os.remove(js)
    open(js, "w").write(raw)
    with pytest.raises(H.SafeFormatError, match="NaN"):
        H.load_member(pres[0])
    hp = pres[1] + ".h5"
    b = open(hp, "rb").read()[:2048]
    os.remove(hp)
    open(hp, "wb").write(b)
    _reanchor(pres[1], index_from_file=False)
    with pytest.raises(H.SafeFormatError, match="unreadable HDF5"):
        H.load_member(pres[1])


def test_non_finite_values_are_refused_at_export_and_load(member, tmp_path):
    prefix, _ = member
    _edit(prefix, lambda f: f["norm/dndx/mean"].__setitem__(0, np.nan))
    with pytest.raises(H.SafeFormatError, match="non-finite"):
        H.load_member(prefix)
    bad = _norm()
    bad["dndx"]["mean"][0] = np.inf
    with pytest.raises(H.SafeFormatError, match="non-finite"):
        H.export_member(
            str(tmp_path / "inf"), Emulator(**ARCH, key=jax.random.PRNGKey(3)), _meta(), bad
        )
    assert not os.path.lexists(str(tmp_path / "inf.h5"))


def test_export_writes_through_the_descriptor_not_the_name(tmp_path, monkeypatch):
    real = h5py.File

    def only_file_objects(name, *a, **k):
        assert not isinstance(name, (str, bytes, os.PathLike)), f"reopened by name: {name}"
        return real(name, *a, **k)

    monkeypatch.setattr(h5py, "File", only_file_objects)
    pre = str(tmp_path / "fd")
    H.export_member(pre, Emulator(**ARCH, key=jax.random.PRNGKey(3)), _meta(), _norm())
    monkeypatch.setattr(h5py, "File", real)
    H.load_member(pre)


# ---------------------------------------------- bounded reads, JSON numbers, meta, write digest, groups, versions
def test_files_are_read_bounded_regular_and_without_following_symlinks(
    member, tmp_path, monkeypatch
):
    prefix, _ = member
    js = prefix + ".json"
    raw = open(js, "rb").read()
    monkeypatch.setattr(H.S, "MAX_JSON_BYTES", len(raw) - 1)
    with pytest.raises(H.SafeFormatError, match="exceed the bound"):
        H.load_member(prefix)
    monkeypatch.undo()
    real = str(tmp_path / "real.json")
    os.rename(js, real)
    os.symlink(real, js)
    with pytest.raises(H.SafeFormatError, match="symlink"):
        H.load_member(prefix)
    os.remove(js)
    os.mkfifo(js)  # would block an open() for reading
    with pytest.raises(H.SafeFormatError, match="not a regular file"):
        H.load_member(prefix)


def test_an_overflowing_json_number_is_refused(member):
    prefix, _ = member
    raw = open(prefix + ".json").read().replace('"L_box_hmpc": 120.0', '"L_box_hmpc": 1e999')
    assert "1e999" in raw
    os.remove(prefix + ".json")
    open(prefix + ".json", "w").write(raw)
    with pytest.raises(H.SafeFormatError, match="not finite"):
        H.load_member(prefix)


@pytest.mark.parametrize("bad", [float("nan"), np.int64(3)])
def test_a_meta_that_is_not_plain_finite_json_is_refused_before_anything_is_written(tmp_path, bad):
    meta = _meta()
    meta["recipe"] = {"x": bad}
    pre = str(tmp_path / "m")
    with pytest.raises(H.SafeFormatError, match="plain finite JSON"):
        H.export_member(pre, Emulator(**ARCH, key=jax.random.PRNGKey(3)), meta, _norm())
    assert os.listdir(tmp_path) == []


def test_the_sidecar_digest_is_of_the_bytes_written_not_of_the_name_afterwards(
    tmp_path, monkeypatch
):
    real_publish = H.S._publish

    def publish_then_swap(tmp, path, held):
        real_publish(tmp, path, held)
        if path.endswith(".h5"):  # a co-writer replaces the published file
            os.remove(path)
            open(path, "wb").write(b"not the exported member")

    monkeypatch.setattr(H.S, "_publish", publish_then_swap)
    pre = str(tmp_path / "m")
    H.export_member(pre, Emulator(**ARCH, key=jax.random.PRNGKey(3)), _meta(), _norm())
    monkeypatch.undo()
    with pytest.raises(H.SafeFormatError, match="sha256 does not match"):
        H.load_member(pre)


def test_groups_outside_the_declared_structure_are_refused(member, tmp_path):
    prefix, _ = member
    _edit(prefix, lambda f: f["norm"].create_group("spare"), index_from_file=False)
    with pytest.raises(H.SafeFormatError, match="holds no dataset"):
        H.load_member(prefix)
    pre = str(tmp_path / "g")
    H.export_member(pre, Emulator(**ARCH, key=jax.random.PRNGKey(3)), _meta(), _norm())
    _edit(pre, lambda f: f["model"].create_group("spare"), index_from_file=False)
    with pytest.raises(H.SafeFormatError, match="must be flat"):
        H.load_member(pre)


def test_the_writer_limits_object_versions_to_hdf5_1_10(tmp_path, monkeypatch):
    seen = []
    real_file = h5py.File

    def recording_file(*a, **k):
        seen.append(k.get("libver"))
        return real_file(*a, **k)

    monkeypatch.setattr(h5py, "File", recording_file)
    H.export_member(
        str(tmp_path / "m"), Emulator(**ARCH, key=jax.random.PRNGKey(3)), _meta(), _norm()
    )
    assert seen == [("earliest", "v110")]


# ---------------------------------------------------------------------- name bound, file types, ensembles
def test_a_long_link_name_is_refused_before_any_path_is_built(member):
    prefix, _ = member

    def long_chain(f):
        g = f["norm"]
        for _ in range(3):
            g = g.create_group("x" * 300)
        g.create_dataset("leaf", data=np.zeros(1))

    _edit(prefix, long_chain, index_from_file=False)
    with pytest.raises(H.SafeFormatError, match="longer than 255 characters"):
        H.load_member(prefix)


def _open_fds():
    return len(os.listdir("/proc/self/fd"))


def test_the_h5_and_the_manifest_reads_are_bounded_and_regular(bundle, monkeypatch):
    bd, pres = bundle
    sha = H.sha256_file(os.path.join(bd, "manifest.json"))
    hp = pres[0] + ".h5"
    monkeypatch.setattr(H, "_MAX_FILE_BYTES", os.path.getsize(hp) - 1)
    with pytest.raises(H.SafeFormatError, match="exceed the bound"):
        H.load_member(pres[0])
    monkeypatch.undo()
    man_size = os.path.getsize(os.path.join(bd, "manifest.json"))
    monkeypatch.setattr(
        H.S, "MAX_JSON_BYTES", man_size - 1
    )  # the manifest is the first file load_bundle reads
    with pytest.raises(H.SafeFormatError, match="manifest.json: .* exceed the bound"):
        H.load_bundle(bd, expected_manifest_sha256=sha)
    monkeypatch.undo()
    os.remove(hp)
    os.mkfifo(hp)
    with pytest.raises(H.SafeFormatError, match="not a regular file"):
        H.load_member(pres[0])


@pytest.mark.parametrize("which", [".json", ".h5", "manifest"])
def test_a_directory_in_place_of_a_file_is_refused_without_leaking_a_descriptor(bundle, which):
    bd, pres = bundle
    path = os.path.join(bd, "manifest.json") if which == "manifest" else pres[0] + which
    os.remove(path)
    os.mkdir(path)
    before = _open_fds()
    for _ in range(5):
        with pytest.raises(
            H.SafeFormatError, match="not a regular file|missing, not a regular file"
        ):
            if which == "manifest":
                H.load_bundle(bd, expected_manifest_sha256=None)
            else:
                H.load_member(pres[0])
    assert _open_fds() == before


def test_a_non_finite_network_weight_is_refused_at_load(member):
    prefix, _ = member

    def nan_weight(f):
        name = sorted(f["model"].keys())[0]
        f["model"][name][...] = np.nan

    _edit(prefix, nan_weight)
    with pytest.raises(H.SafeFormatError, match="non-finite"):
        H.load_member(prefix)


def test_a_scalar_format_attribute_that_is_not_a_string_is_refused_before_it_is_read(
    member, monkeypatch
):
    prefix, _ = member
    compound = np.dtype([("a", h5py.string_dtype()), ("b", h5py.string_dtype())])

    def scalar_compound(f):
        del f.attrs["format"]
        f.attrs.create("format", np.array((H.FORMAT, H.FORMAT), dtype=compound), dtype=compound)

    _edit(prefix, scalar_compound)
    read = []
    real_getitem = h5py.AttributeManager.__getitem__
    monkeypatch.setattr(
        h5py.AttributeManager,
        "__getitem__",
        lambda self, k: read.append(k) or real_getitem(self, k),
    )
    with pytest.raises(H.SafeFormatError, match="scalar string"):
        H.load_member(prefix)
    assert read == []


def test_a_meta_that_does_not_round_trip_is_refused_before_anything_is_written(tmp_path):
    meta = _meta()
    meta["recipe"] = {1: "a", "b": 2}  # mixed key types: the sorted dump would fail
    with pytest.raises(H.SafeFormatError, match="plain finite JSON"):
        H.export_member(
            str(tmp_path / "m"), Emulator(**ARCH, key=jax.random.PRNGKey(3)), meta, _norm()
        )
    meta["recipe"] = {"t": (1, 2)}  # a tuple reloads as a list
    with pytest.raises(H.SafeFormatError, match="round-trip"):
        H.export_member(
            str(tmp_path / "m"), Emulator(**ARCH, key=jax.random.PRNGKey(3)), meta, _norm()
        )
    assert os.listdir(tmp_path) == []


def test_load_ensemble_requires_one_normalizer_and_one_k_grid(tmp_path):
    from hcd_analysis.emulator.ensemble import EnsembleEmulator

    same = RT.write_synthetic_bundle(str(tmp_path / "same"), arch=ARCH, norm=_norm())
    ens, metas, norm = H.load_ensemble(
        same.bundle_dir, expected_manifest_sha256=same.manifest_sha256
    )
    assert (
        isinstance(ens, EnsembleEmulator)
        and len(metas) == 2
        and metas.anchored
        and metas.lineage_verified
    )

    def other_norm(i, model, meta, nm):
        if i == 1:
            nm = _norm()
            nm["dndx"]["mean"] = np.asarray(nm["dndx"]["mean"]) + 1.0
        return model, meta, nm

    diff = RT.write_synthetic_bundle(
        str(tmp_path / "diff"), arch=ARCH, norm=_norm(), member_fn=other_norm
    )
    with pytest.raises(H.SafeFormatError, match="does not share member 0's normalizer"):
        H.load_ensemble(diff.bundle_dir, expected_manifest_sha256=diff.manifest_sha256)


def test_a_size_mismatch_in_the_manifest_check_is_refused_before_hashing(tmp_path, monkeypatch):
    p = tmp_path / "f.bin"
    p.write_bytes(b"0123456789")

    def no_read(*a):
        raise AssertionError("read before the size check")

    monkeypatch.setattr(H.S.os, "read", no_read)
    with pytest.raises(H.SafeFormatError, match="the manifest lists 3"):
        H.S._sha256_regular(str(p), expected_bytes=3)


# --------------------------------------------------------------------------------- name order, counts, manifest, ensembles
def test_a_long_name_is_refused_before_its_subtree_is_walked(member):
    prefix, _ = member

    def long_then_soft(f):
        g = f["norm"].create_group("x" * 300)
        g["s"] = h5py.SoftLink("/model")  # would be refused later in the walk

    _edit(prefix, long_then_soft, index_from_file=False)
    with pytest.raises(H.SafeFormatError, match="longer than 255 characters"):
        H.load_member(prefix)


def test_a_name_at_the_bound_is_not_refused_for_its_length(member):
    prefix, _ = member
    _edit(
        prefix,
        lambda f: f["norm"].create_group("y" * 255).create_dataset("leaf", data=np.zeros(1)),
        index_from_file=False,
    )
    with pytest.raises(H.SafeFormatError) as e:  # refused, but by the normalizer structure
        H.load_member(prefix)
    assert "longer than" not in str(e.value)


def test_link_counts_are_checked_before_any_name_is_read(member, monkeypatch):
    prefix, _ = member
    _edit(
        prefix,
        lambda f: [
            f["norm"].require_group("many").create_dataset(f"d{i}", data=np.zeros(1))
            for i in range(40)
        ],
        index_from_file=False,
    )
    monkeypatch.setattr(H, "_MAX_OBJECTS", 30)
    listed, real_iter = [], h5py.Group.__iter__  # every spelling of a listing goes through __iter__
    monkeypatch.setattr(
        h5py.Group, "__iter__", lambda self: listed.append(self.name) or real_iter(self)
    )
    with pytest.raises(H.SafeFormatError, match="more than 30 objects"):
        H.load_member(prefix)
    assert (
        "/norm" in listed and "/norm/many" not in listed
    )  # /norm was listed, never the 40-link group


@pytest.mark.parametrize("what", ["k", "arch", "norm_keys", "param_limits"])
def test_load_ensemble_refuses_another_k_grid_architecture_or_normalizer_structure(tmp_path, what):
    def differ(i, model, meta, nm):
        if i != 1:
            return model, meta, nm
        meta = dict(meta)
        if what == "k":
            meta["k_com_hmpc"] = [1.2 * k for k in meta["k_com_hmpc"]]
        if what == "arch":
            arch = dict(ARCH, baseline_width=ARCH["baseline_width"] // 2)
            meta["arch_cfg"] = arch
            model = Emulator(**arch, key=jax.random.PRNGKey(i))
        if what == "norm_keys":
            nm = _norm()
            nm["P_filt"]["cell_mean"][5] = np.zeros((4, 8))
        if what == "param_limits":
            meta["param_limits"] = [[lo, hi * 1.01] for lo, hi in meta["param_limits"]]
        return model, meta, nm

    info = RT.write_synthetic_bundle(
        str(tmp_path / what), arch=ARCH, norm=_norm(), member_fn=differ
    )
    match = {
        "norm_keys": "share member 0's normalizer",
        "param_limits": "another parameter box",
    }.get(what, "another architecture or k grid")
    with pytest.raises(H.SafeFormatError, match=match):
        H.load_ensemble(info.bundle_dir, expected_manifest_sha256=info.manifest_sha256)


def test_the_manifest_pass_refuses_a_size_mismatch_before_hashing_it(bundle, monkeypatch):
    import hashlib

    bd, pres = bundle
    hp = pres[0] + ".h5"
    raw = open(hp, "rb").read()
    os.remove(hp)
    open(hp, "wb").write(raw + b"\0" * 10)
    fed = []

    class Counting:
        def __init__(self, *a):
            self.h = hashlib.sha256(*a)

        def update(self, b):
            fed.append(len(b))
            self.h.update(b)

        def hexdigest(self):
            return self.h.hexdigest()

    monkeypatch.setattr(H.S, "hashlib", types.SimpleNamespace(sha256=Counting))
    with pytest.raises(H.SafeFormatError, match="member0.h5 does not match the manifest"):
        H.verify_manifest(bd)
    assert sum(fed) < len(raw)  # the oversized member was never hashed


@pytest.mark.parametrize("bad", [True, 12.0, -1])
def test_a_manifest_byte_count_must_be_a_non_negative_int(bundle, bad):
    bd, _ = bundle
    mp = os.path.join(bd, "manifest.json")
    man = json.load(open(mp))
    man["files"][0]["bytes"] = bad
    os.remove(mp)
    json.dump(man, open(mp, "w"))
    with pytest.raises(H.SafeFormatError, match="is not a non-negative int"):
        H.verify_manifest(bd)


def test_root_counts_are_checked_before_any_root_name_is_read(member, monkeypatch):
    prefix, _ = member
    _edit(prefix, lambda f: f.attrs.__setitem__("second", 1.0), index_from_file=False)
    attr_lists, link_lists = [], []
    real_attr_iter, real_iter = h5py.AttributeManager.__iter__, h5py.Group.__iter__
    monkeypatch.setattr(
        h5py.AttributeManager,
        "__iter__",
        lambda self: attr_lists.append(len(self)) or real_attr_iter(self),
    )
    with pytest.raises(H.SafeFormatError, match="scalar string"):
        H.load_member(prefix)
    assert attr_lists == []  # no attribute name listed, by any spelling
    with h5py.File(prefix + ".h5", "r") as f:  # control: the spy sees both spellings
        list(f.attrs.keys()), set(f.attrs)
    assert attr_lists == [2, 2]
    monkeypatch.undo()
    _edit(prefix, lambda f: f.attrs.__delitem__("second"), index_from_file=False)
    _edit(prefix, lambda f: f.create_group("extra"), index_from_file=False)
    monkeypatch.setattr(
        h5py.Group, "__iter__", lambda self: link_lists.append(self.name) or real_iter(self)
    )
    with pytest.raises(H.SafeFormatError, match="exactly /model and /norm"):
        H.load_member(prefix)
    assert "/" not in link_lists  # no root link name listed, by any spelling
    with h5py.File(prefix + ".h5", "r") as f:
        set(f.keys()), list(f)
    assert link_lists.count("/") == 2


def test_a_name_one_past_the_bound_is_refused_for_its_length(member):
    prefix, _ = member
    _edit(
        prefix,
        lambda f: f["norm"].create_group("y" * 256).create_dataset("leaf", data=np.zeros(1)),
        index_from_file=False,
    )
    with pytest.raises(H.SafeFormatError, match="longer than 255 characters"):
        H.load_member(prefix)


# ------------------------------------------------------------------------------------- historical h5/1 bundles
def _v1_bundle(tmp_path, norms=None):
    bd = tmp_path / "v1"
    bd.mkdir()
    pres = []
    for i, nm in enumerate(norms or (_norm(), _norm())):
        pre = str(bd / f"m{i}")
        H.export_member(
            pre,
            Emulator(**ARCH, key=jax.random.PRNGKey(i)),
            _meta(seed=i),
            nm,
            fmt=H.FORMAT_HISTORICAL,
        )
        pres.append(pre)
    H.write_manifest_historical(str(bd), pres, {"source": "synthetic"})
    return str(bd), pres, H.sha256_file(str(bd / "manifest.json"))


def test_an_h5_1_bundle_loads_only_through_the_historical_loader_with_a_reason(tmp_path, caplog):
    from hcd_analysis.emulator.ensemble import EnsembleEmulator

    bd, pres, sha = _v1_bundle(tmp_path)
    with pytest.raises(
        H.SafeFormatError, match="historical: it loads only through load_ensemble_historical"
    ):
        H.load_ensemble(bd, expected_manifest_sha256=sha)
    with pytest.raises(
        H.SafeFormatError, match="historical: it loads only through load_ensemble_historical"
    ):
        H.load_bundle(bd, expected_manifest_sha256=sha)
    with pytest.raises(H.SafeFormatError, match="manifest format"):
        H.load_ensemble(bd, expected_manifest_sha256=None)
    with pytest.raises(H.SafeFormatError, match="historical member"):
        H.load_member(pres[0])  # an h5/1 member is not read as h5/2
    for bad in ("", "  ", None):
        with pytest.raises(ValueError, match="non-empty reason"):
            H.load_ensemble_historical(bd, expected_manifest_sha256=sha, reason=bad)
    ens, metas, norm = H.load_ensemble_historical(
        bd, expected_manifest_sha256=sha, reason="a reproduction"
    )
    assert isinstance(ens, EnsembleEmulator) and [m["seed"] for m in metas] == [0, 1]
    assert (
        metas.historical_reason == "a reproduction"
        and not metas.lineage_verified
        and metas.anchored
    )
    assert metas.format == H.FORMAT_HISTORICAL and "a reproduction" in caplog.text
    assert metas.kind == "historical" and metas.release_verified is False  # T24
    assert H.verify_manifest_historical(bd, expected_sha256=sha) == pres
    with pytest.raises(H.SafeFormatError, match="manifest"):
        H.load_ensemble_historical(bd, expected_manifest_sha256="f" * 64, reason="r")
    assert [
        m["seed"]
        for _, m, _ in H.load_bundle_historical(bd, expected_manifest_sha256=sha, reason="r")
    ] == [0, 1]


def test_the_historical_loader_keeps_the_h5_1_checks(tmp_path):
    nm = _norm()
    nm["dndx"]["mean"] = np.asarray(nm["dndx"]["mean"]) + 1.0
    bd, pres, sha = _v1_bundle(tmp_path, norms=(_norm(), nm))
    with pytest.raises(H.SafeFormatError, match="does not share member 0's normalizer"):
        H.load_ensemble_historical(bd, expected_manifest_sha256=sha, reason="r")
    open(os.path.join(bd, "extra.json"), "w").write("{}")
    with pytest.raises(H.SafeFormatError, match="does not list"):
        H.load_bundle_historical(bd, expected_manifest_sha256=sha, reason="r")
    os.remove(os.path.join(bd, "extra.json"))
    with h5py.File(pres[1] + ".h5", "r") as f:
        assert f.attrs["format"] == H.FORMAT_HISTORICAL


def test_a_research_bundle_loads_through_the_research_loader_only(tmp_path):
    """The research path on this HDF5 stack too (B's older-libhdf5 job runs this file): the release loaders refuse a
    research bundle and load_research reads it, anchored, as a ResearchBundle."""
    res = RT.write_synthetic_research_bundle(str(tmp_path / "research"), arch=ARCH, norm=_norm())
    for load in (H.load_release, H.load_ensemble, H.load_bundle):
        with pytest.raises(H.SafeFormatError, match="never a production model"):
            load(res.bundle_dir, expected_manifest_sha256=res.manifest_sha256)
    rb = H.load_research(
        res.bundle_dir, expected_manifest_sha256=res.manifest_sha256, purpose="format test"
    )
    assert isinstance(rb, H.ResearchBundle) and rb.kind == "research" and not rb.release_verified
    assert (
        len(rb.ensemble.members) == 2 and rb.metas.format == H.FORMAT and rb.split["held_out_sims"]
    )
    rel = RT.write_synthetic_bundle(str(tmp_path / "release"), arch=ARCH, norm=_norm())
    rr = H.load_release(rel.bundle_dir, expected_manifest_sha256=rel.manifest_sha256)
    assert (
        rr.kind == "release"
        and rr.release_verified
        and rr.metas.kind == "release"
        and rr.metas.release_verified
    )
