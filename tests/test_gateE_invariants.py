"""Gate E invariants (GATE_E_SPEC v1 section 1, PU-0061/0062): the forward rewiring onto the canonical coordinate must
leave the data side byte-identical. Digests pinned at worktree 80bbc5d (before any gate E change): the three leg
loaders under their production arguments (``closure_legb.build_legb_ctx`` + ``PROD_FORWARD_BY_LEG``), and the metal
and resolution factors, which act on data k only.

The leg digests hash stored data products through the loaders (data-gated: they skip without the products, so they
never run on a GitHub runner) and stay exact. The factors are computed floating-point arrays: numpy's geomspace (the
k grid) and XLA's cos/exp give last-bit differences between CPUs, so they are compared with stored synthetic
reference arrays under per-field tolerances, and their bit-level digest only in an exact replay
(tests/portable_numerics.py)."""
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from hcd_analysis.emulator import data_likelihood as DL

from tests.gate_helpers import external_path  # noqa: E402
from tests import portable_numerics as PN  # noqa: E402
DATA = {"DESI": external_path("HCD_P1D_DATA_DIR", "desi_dr1_p1d", "desi_dr1_p1d.npz"),
        "eBOSS": external_path("HCD_P1D_DATA_DIR", "eboss_dr14_p1d", "eboss_dr14_p1d.npz"),
        "KS": external_path("HCD_LYAEMU_PARENT", "lyaemu", "data", "kodiaq_squad", "")}

# Production leg arguments: build_legb_ctx(sample_res=True, coherent_res=False) with PROD_FORWARD_BY_LEG
# (DESI and eBOSS metals on, KS resolution_float=True, k_max=0.065).
PROD_LEG_ARGS = {
    "DESI": lambda: DL.load_desi_leg(metals_on=True, resolution_float=True, resolution_coherent=False),
    "KS": lambda: DL.load_ks_leg(resolution_float=True, k_max=0.065),
    "eBOSS": lambda: DL.load_eboss_leg(resolution_float=True, resolution_coherent=False),
}

PINNED = {   # sha256 of leg_digest() at 80bbc5d
    "DESI": "392c218e8a14ebad6c7964d753d2a51e55979d6af96224d9da2407d9542db407",
    "KS": "4fae133fb02719b881cc89e3018d7b170c6423e46401aca916d984594d20d05d",
    "eBOSS": "514415bf46162394db38a1dc3703017076186049175cb5257f8e2a5ab25ba62f",
    "factors": "1b0a1ccffc40dfdad583b9dcbcd3462444c9cb7a0d0bb6d05e0c5914b345205d",
}


def _feed(h, name, v):
    h.update(name.encode())
    if isinstance(v, (np.ndarray,)) or hasattr(v, "__array__") and not isinstance(v, (str, bytes)):
        a = np.ascontiguousarray(np.asarray(v))
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    else:
        h.update(repr(v).encode())


def leg_digest(leg):
    h = hashlib.sha256()
    for name in leg._fields:
        _feed(h, name, getattr(leg, name))
    return h.hexdigest()


METAL_CASES = (dict(a_SiIII=0.01), dict(a_SiII=0.004), dict(a_SiIII=0.012, a_SiII=-0.003))
RES_CASES = ((10.0, 0.1), (3.2, -0.05), (25.0, 0.02))
FACTOR_FIELDS = tuple(f"metal{sorted(kw.items())}" for kw in METAL_CASES) + tuple(f"res{R},{b}" for R, b in RES_CASES)


def factor_arrays(metal_cases=METAL_CASES, res_cases=RES_CASES):
    """``{field: factor on the k grid}`` in the digest order (``FACTOR_FIELDS``, the names the digest feeds)."""
    k = np.geomspace(5e-4, 0.1, 257)
    out = {f"metal{sorted(kw.items())}": np.asarray(DL._metal_factor(k, **kw)) for kw in metal_cases}
    out.update({f"res{R},{b}": np.asarray(DL._resolution_factor(k, R, b_res=b)) for R, b in res_cases})
    return out


def factors_digest(arrays=None):
    """sha256 of the factors (``factor_arrays()`` by default), fed in ``FACTOR_FIELDS`` order."""
    arrays = factor_arrays() if arrays is None else arrays
    h = hashlib.sha256()
    for name in FACTOR_FIELDS:
        _feed(h, name, arrays[name])
    return h.hexdigest()


# The factors at 80bbc5d, stored at full precision (synthetic: fixed made-up amplitudes, radii and k grid; they pin the
# functional forms). Their digest is PINNED["factors"] (test_the_stored_factors_are_the_pinned_digest).
FACTORS_REF = Path(__file__).parent / "golden" / "ci_portability" / "gateE_factors_80bbc5d.npz"
# Per-field tolerances (rtol, atol). Both factors multiply the model P1D, so a relative change of a factor is the
# same relative change of the model P1D. Scientific scale: the P1D measurement and emulator errors, about a percent
# and never below 1e-3 relative in any bin. Measured cross-CPU discrepancy (numpy with and without AVX-512, OpenBLAS
# and XLA instruction-set variants, both stacks): at most 3 ulps (3.4e-16 relative, metal) and 1 ulp (1.8e-16,
# resolution); an 8-ulp perturbation of every libm, XLA and k-grid output, which bounds an implementation not
# reachable on our machine, moves them by less than 1e-14. Each tolerance is 1e-12 relative: about 3000 times the
# measured discrepancy and 100 times the 8-ulp bound, and 1e-9 of the scientific scale. The factors are of order 1
# (0.97 to 1.3) and never near zero, so atol is 0.
METAL_TOL = (1e-12, 0.0)      # P -> P (1 + f_metal): oscillatory cos(k dv) terms of amplitude 2a
RES_TOL = (1e-12, 0.0)        # P -> P exp(2 b_res k^2 R^2)
FACTOR_TOLS = {name: METAL_TOL if name.startswith("metal") else RES_TOL for name in FACTOR_FIELDS}


def _unavailable(msg):
    if os.environ.get("HCD_GATE_RUN") == "1":
        pytest.fail(msg + " (HCD_GATE_RUN=1)")
    pytest.skip(msg)


@pytest.mark.parametrize("leg", ["DESI", "KS", "eBOSS"])
def test_production_leg_loaders_unchanged(leg):
    if not os.path.exists(DATA[leg]):
        _unavailable(f"data product absent: {DATA[leg]}")
    assert leg_digest(PROD_LEG_ARGS[leg]()) == PINNED[leg]


def test_metal_and_resolution_factors_unchanged():
    assert PN.field_mismatches(factor_arrays(), PN.load_reference(FACTORS_REF), FACTOR_TOLS) == []


def test_metal_and_resolution_factors_bitwise_in_an_exact_replay():
    """Byte identity of the computed factors: only in an exact replay; otherwise a SKIP that says so."""
    PN.require_exact_replay()
    assert factors_digest(factor_arrays()) == PINNED["factors"]


def test_the_exact_replay_hashes_the_computed_factors(monkeypatch):
    """The exact branch forced, the computed factors moved by one ulp in one element: the digest test must fail. So it
    hashes the computed result, never the stored reference (which hashes to the pin everywhere)."""
    monkeypatch.setattr(PN, "exact_replay", lambda *a, **k: True)
    real = DL._metal_factor

    def one_ulp(k, **kw):
        out = np.array(real(k, **kw), dtype=np.float64)
        out[100] = np.nextafter(out[100], np.inf)
        return out
    monkeypatch.setattr(DL, "_metal_factor", one_ulp)
    with pytest.raises(AssertionError):
        PN.call_unskipped(test_metal_and_resolution_factors_bitwise_in_an_exact_replay)


def test_the_stored_factors_are_the_pinned_digest():
    """The stored reference arrays are the 80bbc5d factors, bit for bit (a stored file: exact everywhere)."""
    ref = PN.load_reference(FACTORS_REF)
    assert list(ref) == list(FACTOR_FIELDS) and factors_digest(ref) == PINNED["factors"]
    flipped = {k: np.array(v, copy=True) for k, v in ref.items()}
    flipped[FACTOR_FIELDS[0]].view(np.uint64)[100] ^= np.uint64(1)          # one bit of one element
    assert factors_digest(flipped) != PINNED["factors"]


def _ulps(arrays, n, seed=0):
    rng = np.random.default_rng(seed)
    return {k: v + rng.integers(-n, n + 1, v.shape) * np.spacing(np.abs(v)) for k, v in arrays.items()}


def test_the_factor_comparison_passes_rounding_and_refuses_a_meaningful_change():
    ref = PN.load_reference(FACTORS_REF)
    assert PN.field_mismatches(_ulps(ref, 8), ref, FACTOR_TOLS) == []          # 8 ulps everywhere: rounding, passes
    a_SiIII = dict(METAL_CASES[0])
    for change, field in ((dict(a_SiIII=-a_SiIII["a_SiIII"]), FACTOR_FIELDS[0]),              # the sign of a_SiIII
                          (dict(a_SiIII=a_SiIII["a_SiIII"] * 1.01), FACTOR_FIELDS[0])):          # a 1% amplitude
        got = dict(ref, **{field: np.asarray(DL._metal_factor(np.geomspace(5e-4, 0.1, 257), **change))})
        assert PN.field_mismatches(got, ref, FACTOR_TOLS) == [field]
    got = dict(ref, **{FACTOR_FIELDS[3]: np.asarray(DL._resolution_factor(np.geomspace(5e-4, 0.1, 257), 10.0,
                                                                            b_res=-0.1))})      # b_res sign
    assert PN.field_mismatches(got, ref, FACTOR_TOLS) == [FACTOR_FIELDS[3]]
    for name in FACTOR_FIELDS:      # 1e-9 relative on one element of each field (1e-6 of the scientific scale) fails
        bad = {k: np.array(v, copy=True) for k, v in ref.items()}
        bad[name][128] *= 1 + 1e-9
        assert PN.field_mismatches(bad, ref, FACTOR_TOLS) == [name]
    swapped = dict(ref, **{FACTOR_FIELDS[0]: ref[FACTOR_FIELDS[3]], FACTOR_FIELDS[3]: ref[FACTOR_FIELDS[0]]})
    assert PN.field_mismatches(swapped, ref, FACTOR_TOLS) == sorted([FACTOR_FIELDS[0], FACTOR_FIELDS[3]])
    reversed_k = dict(ref, **{FACTOR_FIELDS[1]: ref[FACTOR_FIELDS[1]][::-1]})
    assert PN.field_mismatches(reversed_k, ref, FACTOR_TOLS) == [FACTOR_FIELDS[1]]
    nan = {k: np.array(v, copy=True) for k, v in ref.items()}
    nan[FACTOR_FIELDS[2]][5] = np.nan
    assert PN.field_mismatches(nan, ref, FACTOR_TOLS) == [FACTOR_FIELDS[2]]
    assert PN.field_mismatches(dict(ref, **{FACTOR_FIELDS[4]: ref[FACTOR_FIELDS[4]][:-1]}), ref,
                               FACTOR_TOLS) == [FACTOR_FIELDS[4]]                               # truncated
    assert PN.field_mismatches(dict(ref, **{FACTOR_FIELDS[5]: ref[FACTOR_FIELDS[5]].astype(np.float32)}), ref,
                               FACTOR_TOLS) == [FACTOR_FIELDS[5]]                               # another dtype
    assert PN.field_mismatches({k: v for k, v in ref.items() if k != FACTOR_FIELDS[5]}, ref, FACTOR_TOLS) == ["keys"]
    with pytest.raises(ValueError, match="its own tolerance"):
        PN.field_mismatches(ref, ref, {k: v for k, v in FACTOR_TOLS.items() if k != FACTOR_FIELDS[0]})
    with pytest.raises(ValueError, match="unknown"):
        PN.field_mismatches(ref, ref, dict(FACTOR_TOLS, extra=(1e-12, 0.0)))
