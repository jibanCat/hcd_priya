"""Product identity for the gate E forward (GATE_E_SPEC v1 section 2; emulator-debug campaign 2026-10).

Every product the forward loads carries its kind, the box-mode axis k_com = 2 pi n / L (never a velocity grid) and
provenance with digests (code commit, the cache sha256, the sha256 of every input). Loaders take the expected kind and
digests and refuse anything else, so an old product, an unlabelled product (e.g. the gate C error vector marked NOT
FOR USE) or a product built from other checkpoints cannot enter the forward ("do not mix old and new products")."""
from __future__ import annotations

import json
import os

import numpy as np

from .schema import CHECKPOINT_SCHEMA_VERSION, L_BOX_HMPC

PRODUCT_KINDS = ("cemu_t1", "cemu_t2", "cemu_t3", "dla_core")
REQUIRED_PROVENANCE = ("code_commit", "cache_sha256", "inputs", "row_rule")
_VELOCITY_KEYS = ("kfkms", "k_skm", "k_kms", "kgrid", "k", "k_eval")
_RESERVED = ("product_kind", "schema_version", "k_com_hmpc", "provenance")


class ProductRefused(ValueError):
    """A product that may not enter the gate E forward."""


def _check_kcom(k_com_hmpc):
    k = np.asarray(k_com_hmpc, float)
    if k.ndim != 1 or not np.allclose(k, 2 * np.pi * np.arange(1, k.size + 1) / L_BOX_HMPC, rtol=1e-12, atol=0):
        raise ProductRefused("k_com_hmpc must be the box modes 2 pi n / L, n = 1..K")
    return k


def _check_names(names):
    bad = [n for n in names if n in _VELOCITY_KEYS]
    if bad:
        raise ProductRefused(f"velocity-labelled keys refused (products live on the mode axis): {bad}")


def save_product(path, kind, *, k_com_hmpc, provenance, **arrays):
    """Write a schema-2.0 product once (refuses to overwrite)."""
    if kind not in PRODUCT_KINDS:
        raise ProductRefused(f"unknown product kind {kind!r}; expected one of {PRODUCT_KINDS}")
    _check_names(arrays)
    clash = [n for n in arrays if n in _RESERVED]
    if clash:
        raise ProductRefused(f"reserved keys: {clash}")
    missing = [k for k in REQUIRED_PROVENANCE if k not in provenance]
    if missing:
        raise ProductRefused(f"provenance lacks {missing}")
    k = _check_kcom(k_com_hmpc)
    if os.path.exists(path):
        raise ProductRefused(f"{path} exists; products are written once")
    np.savez(path, product_kind=kind, schema_version=CHECKPOINT_SCHEMA_VERSION, k_com_hmpc=k,
             provenance=json.dumps(provenance, sort_keys=True), **{n: np.asarray(v) for n, v in arrays.items()})


def load_product(path, kind, *, cache_sha256, inputs=None):
    """(arrays, k_com_hmpc, provenance) of a product of ``kind`` built from the cache ``cache_sha256`` and, if given,
    from exactly the ``inputs`` digests named there; refuses everything else."""
    with np.load(path, allow_pickle=False) as f:
        files = set(f.files)
        if not {"product_kind", "schema_version", "k_com_hmpc", "provenance"} <= files:
            raise ProductRefused(f"{path}: unlabelled or pre-2026-10 product (no kind/schema/mode axis/provenance)")
        got = str(f["product_kind"])
        if got != kind:
            raise ProductRefused(f"{path}: product kind {got!r} is not {kind!r}")
        if str(f["schema_version"]) != CHECKPOINT_SCHEMA_VERSION:
            raise ProductRefused(f"{path}: schema {f['schema_version']} != {CHECKPOINT_SCHEMA_VERSION}")
        _check_names(files)
        k = _check_kcom(f["k_com_hmpc"])
        prov = json.loads(str(f["provenance"]))
        arrays = {n: np.asarray(f[n]) for n in files if n not in _RESERVED}
    missing = [x for x in REQUIRED_PROVENANCE if x not in prov]
    if missing:
        raise ProductRefused(f"{path}: provenance lacks {missing}")
    if prov["cache_sha256"] != cache_sha256:
        raise ProductRefused(f"{path}: built from cache {prov['cache_sha256'][:12]}, expected {cache_sha256[:12]}")
    for name, digest in (inputs or {}).items():
        if prov["inputs"].get(name) != digest:
            raise ProductRefused(f"{path}: inputs[{name!r}] is {prov['inputs'].get(name)}, expected {digest}")
    return arrays, k, prov


def load_mf_product(path, *, lf_ckpt_sha256, lf_cache_sha256):
    """The gate D mode-axis MF product as a ``ModeMF``, refused unless it was fitted against exactly the LF ensemble
    ``lf_ckpt_sha256`` (ordered .eqx digests) and the LF cache ``lf_cache_sha256``."""
    from .mf_modes import ModeMF, load_mode_mf
    try:
        tables, k_com, prov = load_mode_mf(path)
    except (KeyError, ValueError) as e:
        raise ProductRefused(f"{path}: not a schema-2.0 mode-axis MF product ({e})") from e
    _check_kcom(k_com)
    if list(prov.get("lf_ckpt_sha256", [])) != list(lf_ckpt_sha256):
        raise ProductRefused(f"{path}: fitted against another LF ensemble than the forward's")
    if prov.get("lf_cache_sha256") != lf_cache_sha256:
        raise ProductRefused(f"{path}: built from LF cache {str(prov.get('lf_cache_sha256'))[:12]}, "
                             f"expected {lf_cache_sha256[:12]}")
    if prov.get("res_corr") != "off (NORC)":
        raise ProductRefused(f"{path}: res_corr is retired in the gate E forward (provenance {prov.get('res_corr')!r})")
    return ModeMF.from_tables(tables), k_com, prov
