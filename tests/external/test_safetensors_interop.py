"""OPTIONAL external test: files written by safe_io are read by the official ``safetensors`` package and vice versa.

Runs only in an interpreter where ``safetensors`` is importable (an overlay venv; the core environment does not carry
it). Without it the test is SKIPPED, which is not a pass: the interop claim then stays unverified.
"""

import numpy as np
import pytest

from hcd_analysis.emulator import safe_io as S

st_numpy = pytest.importorskip(
    "safetensors.numpy", reason="official safetensors package not installed (overlay venv)"
)


def _arrays():
    r = np.random.default_rng(7)
    return {
        "model.enc.layers[0].weight": r.normal(size=(5, 3)),
        "norm/P_filt/cell_mean/#0": r.normal(size=(4, 2)),
        "f32": r.normal(size=3).astype(np.float32),
        "i64": np.arange(4, dtype=np.int64),
        "i32": np.arange(3, dtype=np.int32),
        "u8": np.arange(5, dtype=np.uint8),
        "bool": np.array([True, False]),
        "scalar": np.float64(1.5),
        "empty": np.zeros((0, 2)),
    }


def test_big_endian_input_is_stored_little_endian_for_the_official_reader(tmp_path):
    p = str(tmp_path / "be.safetensors")
    S.write_safetensors(p, {"be": np.arange(4, dtype=">f8")})
    out = st_numpy.load_file(p)
    assert out["be"].dtype == np.dtype("<f8") and np.array_equal(out["be"], np.arange(4.0))


def test_official_reader_reads_safe_io_files(tmp_path):
    p = str(tmp_path / "a.safetensors")
    S.write_safetensors(p, _arrays(), metadata={"format": S.FORMAT})
    out = st_numpy.load_file(p)
    for k, a in _arrays().items():
        assert (
            out[k].dtype == np.asarray(a).dtype
            and out[k].shape == np.shape(a)
            and out[k].tobytes() == np.asarray(a).tobytes()
        )


def test_safe_io_reads_official_files(tmp_path):
    p = str(tmp_path / "b.safetensors")
    st_numpy.save_file(_arrays(), p, metadata={"format": S.FORMAT})
    out, meta = S.read_safetensors(p)
    assert meta == {"format": S.FORMAT}
    for k, a in _arrays().items():
        assert (
            out[k].dtype == np.asarray(a).dtype
            and out[k].shape == np.shape(a)
            and out[k].tobytes() == np.asarray(a).tobytes()
        )
