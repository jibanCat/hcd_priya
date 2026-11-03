"""Synthetic release bundles (format ``hcd_priya.h5/2``) for tests and CI. Synthetic values only; never a real model.

A release bundle carries a lineage block that names a training cache, its catalogues, the code commits and the w_c map,
and the release loader (``safe_h5.load_ensemble`` with a manifest digest) refuses a bundle whose block does not
establish a clean lineage. Tests and CI jobs that only need a loadable bundle (for example the cross-version check of
the HDF5 format) build one here: a made-up cache record (eligible, the current finder id, made-up digests), a w_c map
fitted to nothing (made-up coefficients, its record naming that cache), small untrained networks, and metas that tie
every member to the cache and to one made-up clean commit, with a recipe that declares the full training set
(``all_sims`` true: the production rule of ``eligibility``). Everything passes the release loader's checks because it
is self-consistent, not because it describes real data; the trust of a real bundle comes from its published manifest
digest.

Synthetic h5/2 bundles written before D5 (recipe ``{"synthetic": true}`` without ``all_sims``) are historical
artefacts: the release loaders refuse them and they load through the development path only.

    from hcd_analysis.emulator import release_testing as RT
    info = RT.write_synthetic_bundle(bundle_dir)                       # two members by default
    ens, metas, norm = safe_h5.load_ensemble(bundle_dir, expected_manifest_sha256=info.manifest_sha256)
"""
from __future__ import annotations

import hashlib
import os
from typing import NamedTuple

import numpy as np

SYNTHETIC_ARCH = {"in_dim": 10, "n_k": 8, "n_basis": 4, "baseline_n_layers": 1, "baseline_width": 16}
SYNTHETIC_COMMIT = "5" * 40                     # a made-up clean commit (40 hex digits)
SYNTHETIC_CODE = {"commit": SYNTHETIC_COMMIT, "dirty": False}
SYNTHETIC_CACHE_NAME = "synthetic_cache.h5"


def _h(*parts):
    return hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).hexdigest()


def synthetic_norm(n_k=SYNTHETIC_ARCH["n_k"], seed=0):
    """A normalizer with the production structure, made-up values (P_filt of order 0.1, dN/dX of order 0.1)."""
    r = np.random.default_rng(seed)
    return {"f_nhi": {"mean": r.normal(size=30), "std": r.uniform(1.0, 2.0, 30)},
            "dndx": {"mean": np.log(np.array([0.3, 0.1, 0.05])), "std": np.full(3, 0.2)},
            "P_filt": {"mu_marg": -2.0 + 0.1 * r.normal(size=(4, n_k)), "sig_marg": r.uniform(0.05, 0.1, (4, n_k)),
                       "sig_cosmo": r.uniform(0.05, 0.1, (4, n_k)),
                       "cell_mean": {i: r.normal(size=(4, n_k)) for i in range(3)}},
            "delta": {"mean": r.normal(size=(3, n_k)), "std": r.uniform(1.0, 2.0, (3, n_k))}}


def synthetic_cache_record(n_groups=2, *, carried=False, code=None):
    """``(record, identity)``: an eligible ``tau0_cache`` lineage record of ``n_groups`` made-up groups (each with its
    snapshot, catalogue, CDDF and raw inputs) and a made-up file identity {sha256, size}. ``carried``: the groups are
    carried and promoted, with the carry's source as the one validated input."""
    from .. import lineage as L
    from .. import tau0_cache as T
    from ..catalog import FINDER_ID, RELEASE_FINDER_IDS
    inputs, groups = [], []
    for g in range(n_groups):
        sim, snap = f"synthsim{g}", 10 + g
        for kind in ("snapshot", "catalog", "cddf"):
            inputs.append({"role": T.group_role(kind, sim, snap), "kind": kind, "sha256": _h(kind, sim, snap),
                           "size": 1, "finder_ids": [FINDER_ID], "release_eligible": True})
        inputs.append({"role": T.group_role("raw_spectra", sim, snap), "kind": "raw_spectra",
                       "sha256": _h("raw", sim, snap), "size": 1, "mtime_ns": 1})
        groups.append({"sim": sim, "snap": snap, "z_grid": 3.0, "rows": T.ROWS_CARRIED if carried else T.ROWS_BUILT})
    product = {"payload_sha256": _h("payload"), "n_rows": 20 * n_groups, "n_snaps": n_groups, "groups": groups}
    validation = None
    if carried:
        src = {"role": "historical_source", "kind": T.KIND, "sha256": _h("source"), "size": 1}
        inputs.append(src)
        product["carried"] = {"source_sha256": src["sha256"], "source_size": 1, "fields": list(T.CARRIED_FIELDS),
                              "promoted": {"passed": True, "comparison_sha256": _h("comparison"),
                                           "staged_sha256": _h("staged")}}
        validation = {"all_passed": True, "validated_inputs": ["historical_source"], "summary_sha256": _h("summary")}
    rec = L.build_record(T.KIND, T.KIND_VERSION, inputs=inputs, finder_ids=None,
                         settings={"cache_version": T.CACHE_VERSION, "fidelity": "lf", "n_k": 8,
                                   "alpha_range": [0.66, 1.33]},
                         code=code or SYNTHETIC_CODE, allowed_finder_ids=RELEASE_FINDER_IDS, product=product,
                         validation=validation)
    return rec, {"sha256": _h("synthetic training cache", L.record_sha256(rec)), "size": 1 << 20}


def synthetic_wc_map_doc():
    """A w_c map payload with made-up coefficients (no fit behind it)."""
    from . import wc_map as W
    return {"format": W.FORMAT, "class_order": list(W.CLASS_ORDER),
            "delta_c": {"coeffs": [[0.0, 0.002, -0.01], [0.0, 0.001, -0.005], [0.0, 0.0, -0.003],
                                   [0.0, 0.0, -0.002]], "deg": 2, "z_clamp": list(W.DELTA_C_Z_CLAMP),
                        "resid_std": [0.01, 0.005, 0.003, 0.002], "resid_max": [0.03, 0.02, 0.01, 0.005],
                        "mean": [-0.004, -0.002, -0.003, -0.002], "n_blocks": 4, "n_sims": 2, "z_min": 2.0,
                        "z_max": 5.4},
            "xbar": {"coeffs": [0.0, 0.02, 0.3], "deg": 2, "z_fit": list(W.XBAR_Z_FIT), "n_groups": 4}}


def synthetic_meta(seed, cache_record, cache_identity, *, arch=None, commit=SYNTHETIC_COMMIT):
    """A schema-2.0 member meta tied to the synthetic cache and commit: the box modes, the production parameter box."""
    from .. import lineage as L
    from .data import PARAM_LIMITS
    from .schema import CHECKPOINT_SCHEMA_VERSION, K_CONVENTION, L_BOX_HMPC
    arch = dict(arch or SYNTHETIC_ARCH)
    n_k = arch["n_k"]
    return {"schema_version": CHECKPOINT_SCHEMA_VERSION, "arch_cfg": arch, "seed": int(seed),
            "k_com_hmpc": [2.0 * np.pi * n / L_BOX_HMPC for n in range(1, n_k + 1)], "n_k": n_k,
            "L_box_hmpc": float(L_BOX_HMPC), "k_convention": K_CONVENTION,
            "param_limits": np.asarray(PARAM_LIMITS, float).tolist(), "git_sha": commit, "code_dirty": False,
            "cache_sha256": cache_identity["sha256"], "cache_lineage_sha256": L.record_sha256(cache_record),
            "cache_path": SYNTHETIC_CACHE_NAME, "recipe": {"synthetic": True, "all_sims": True}}


class SyntheticBundle(NamedTuple):
    bundle_dir: str
    manifest_sha256: str
    prefixes: list
    models: list
    norm: dict
    lineage: dict
    cache_record: dict
    cache_identity: dict


def write_synthetic_bundle(bundle_dir, *, n_members=2, seeds=None, arch=None, norm=None, carried=False,
                           export_code=None, member_fn=None, cache=None, wc_map_path=None):
    """Write an h5/2 bundle of ``n_members`` untrained networks (``Emulator(**arch, key=PRNGKey(seed))``, seeds
    0, 1, ... by default) into the NEW or empty directory ``bundle_dir``. ``member_fn(i, model, meta, norm)``, when
    given, returns the ``(model, meta, norm)`` to export as member ``i`` (a test that needs one member to differ).
    ``cache`` = ``(record, {sha256, size})`` of another (for example a synthetic test) cache file, and ``wc_map_path``
    a w_c map fitted to it, replace the made-up ones.
    Returns a ``SyntheticBundle`` with the manifest's sha256 (the anchor to load it with) and the models and
    normalizer that were written."""
    import jax
    from . import safe_h5 as H
    from . import wc_map as W
    from .model import Emulator
    os.makedirs(bundle_dir, exist_ok=True)
    if os.listdir(bundle_dir):
        raise FileExistsError(f"{bundle_dir} is not empty")
    arch = dict(arch or SYNTHETIC_ARCH)
    seeds = list(range(n_members)) if seeds is None else list(seeds)
    norm = norm if norm is not None else synthetic_norm(arch["n_k"])
    rec, ident = synthetic_cache_record(carried=carried) if cache is None else cache
    if wc_map_path is None:
        doc = synthetic_wc_map_doc()
        wrec = W.make_record(doc, rec, ident, code=SYNTHETIC_CODE, cache_name=SYNTHETIC_CACHE_NAME)
        W.write_wc_map(os.path.join(bundle_dir, H.WC_MAP_FILE), doc, wrec)
        wcm = W.read_wc_map(os.path.join(bundle_dir, H.WC_MAP_FILE))
    else:
        wcm = H.export_wc_map(bundle_dir, wc_map_path)
    prefixes, models, metas = [], [], []
    for i, s in enumerate(seeds):
        model, meta, mnorm = Emulator(**arch, key=jax.random.PRNGKey(s)), synthetic_meta(s, rec, ident, arch=arch), norm
        if member_fn is not None:
            model, meta, mnorm = member_fn(i, model, meta, mnorm)
        prefix = os.path.join(bundle_dir, f"member{s}")
        H.export_member(prefix, model, meta, mnorm)
        prefixes.append(prefix)
        models.append(model)
        metas.append(meta)
    block = H.make_lineage_block(cache_record=rec, cache_identity=ident, wc_map=wcm, member_metas=metas,
                                 export_code=export_code or SYNTHETIC_CODE)
    path = H.write_manifest(bundle_dir, prefixes, {"source": "synthetic (release_testing)"}, lineage=block)
    return SyntheticBundle(os.path.abspath(bundle_dir), H.sha256_file(path), prefixes, models, norm, block, rec, ident)
