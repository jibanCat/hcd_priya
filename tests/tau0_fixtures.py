"""Synthetic tau0 caches for the tests (production key layout, made-up values only).

``datasets(groups)`` builds the datasets of a cache; ``write_v33`` writes them as a pre-lineage 3.3 file with plain
h5py; ``write_v34`` writes them through ``tau0_cache.write_cache`` with release-eligible placeholder inputs (call it
with the code identity patched to a clean commit, ``clean_code(monkeypatch)``)."""

import hashlib

import h5py
import numpy as np

from hcd_analysis import lineage as L
from hcd_analysis import tau0_cache as T
from hcd_analysis.catalog import FINDER_ID

K, NB, NH = 6, 15, 30
CLEAN = {"commit": "c" * 40, "dirty": False}
ATTRS = dict(
    created_utc="t0",
    git_sha="abc",
    cache_version="3.3",
    n_k=K,
    tau_thresh=1e6,
    tier_c_recipe="uniform",
    alpha_range=np.array([0.6, 0.9]),
    k_convention="angular",
)


def clean_code(monkeypatch, code=CLEAN):
    monkeypatch.setattr(L, "code_identity", lambda repo_dir=None: dict(code))


def datasets(groups, *, edges_shift=0.0):
    """groups: [(sim, snap, z_grid, n_alpha, seed)], rows grouped by group in the given order."""
    rows, blocks = [], []
    for gi, (sim, snap, zg, na, seed) in enumerate(groups):
        rng = np.random.default_rng(seed)
        blocks.append(
            dict(
                snap_sim_name=sim,
                snap_snap=snap,
                snap_total_path_dX=rng.random(),
                snap_dNdX_LLS=rng.random(),
                snap_dNdX_subDLA=rng.random(),
                snap_dNdX_DLA=rng.random(),
                snap_f_nhi=rng.random(NH),
                snap_n_absorbers=rng.integers(0, 9, NH),
            )
        )
        for a in range(na):
            rows.append(
                dict(
                    sim_name=sim,
                    params=rng.random(9),
                    alpha_slope=0.6 + 0.1 * a,
                    target_F=rng.random(),
                    scale=rng.random(),
                    z_meta=zg + 1e-6,
                    z_grid=zg,
                    dv_kms=10.0 + rng.random(),
                    snap=snap,
                    alpha_idx=a,
                    nbins_native=1400,
                    snap_group_idx=gi,
                    kfkms=rng.random(K),
                    P_tier_p=rng.random(K),
                    P_tier_c=rng.random((NB, K)),
                    P_tier_c_filtered=rng.random((NB, K)),
                    tier_c_counts=rng.integers(0, 99, NB),
                    mean_F_by_bin=rng.random(NB),
                )
            )
    d = {
        "tier_c_labels": np.array([f"c{i}" for i in range(NB)], dtype=object),
        "tier_c_nhi_edges": np.linspace(17, 22, NB - 1),
        "param_names": np.array([f"p{i}" for i in range(9)], dtype=object),
        "log_nhi_centres": np.linspace(17.1, 22.9, NH) + edges_shift,
        "log_nhi_edges": np.linspace(17, 23, NH + 1),
        "sim_name": np.array([r["sim_name"] for r in rows], dtype=object),
    }
    for k in ("params",) + T.ROW_P1D_KEYS + T.ROW_TIERC_KEYS + T.ROW_COUNT_KEYS + T.ROW_MEANF_KEYS:
        d[k] = np.stack([r[k] for r in rows])
    for k in T.ROW_FLOAT_KEYS:
        d[k] = np.array([r[k] for r in rows], dtype=np.float64)
    for k in T.ROW_INT_KEYS:
        d[k] = np.array([r[k] for r in rows], dtype=np.int32)
    d["snap_sim_name"] = np.array([b["snap_sim_name"] for b in blocks], dtype=object)
    d["snap_snap"] = np.array([b["snap_snap"] for b in blocks], dtype=np.int32)
    for k in T.SNAP_KEYS[2:]:
        d[k] = (
            np.array([b[k] for b in blocks])
            if np.ndim(blocks[0][k]) == 0
            else np.stack([b[k] for b in blocks])
        )
    return d


def write_v33(path, groups, *, edges_shift=0.0, attrs=None):
    d = datasets(groups, edges_shift=edges_shift)
    with h5py.File(path, "w") as f:
        for k, v in dict(
            ATTRS, n_rows=len(d["sim_name"]), n_snaps=len(d["snap_snap"]), **(attrs or {})
        ).items():
            f.attrs[k] = v
        for k, v in d.items():
            if k in T.STR_KEYS:
                f.create_dataset(k, data=v, dtype=h5py.string_dtype())
            else:
                f.create_dataset(k, data=v)
    return path


def _h(*parts):
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


def group_entries(sim, snap, *, eligible=True, tag=""):
    """Placeholder input entries of one group, as load_snapshot_inputs lists them."""
    out = []
    for kind in ("snapshot", "catalog", "cddf"):
        out.append(
            {
                "role": T.group_role(kind, sim, snap),
                "kind": kind,
                "sha256": _h(kind, sim, snap, tag),
                "size": 1,
                "finder_ids": [FINDER_ID],
                "release_eligible": bool(eligible),
            }
        )
    out.append(
        {
            "role": T.group_role("raw_spectra", sim, snap),
            "kind": "raw_spectra",
            "sha256": _h("raw", sim, snap),
            "size": 1,
            "mtime_ns": 1,
        }
    )
    return out


SOURCE_ROLE = "historical_source"


def source_entry(sha256="0" * 64, size=1, kind=T.KIND):
    """A carry's historical source as a cache input (no record of its own)."""
    return {"role": SOURCE_ROLE, "kind": kind, "sha256": sha256, "size": size}


def carry_kw(source=None, *, promoted=True):
    """The write_v34 keywords of a carried cache whose source is ``source`` (``source_entry()`` by default)."""
    src = source or source_entry()
    car = {
        "source_sha256": src["sha256"],
        "source_size": src["size"],
        "fields": list(T.CARRIED_FIELDS),
    }
    if promoted:
        car["promoted"] = {"passed": True, "comparison_sha256": "1" * 64, "staged_sha256": "2" * 64}
    return {
        "rows": T.ROWS_CARRIED,
        "carried": car,
        "extra_inputs": [src],
        "validation": {"all_passed": True, "validated_inputs": [SOURCE_ROLE]},
    }


def write_v34(
    path,
    groups,
    *,
    edges_shift=0.0,
    attrs=None,
    eligible=True,
    settings=None,
    rows=T.ROWS_BUILT,
    extra_inputs=(),
    **kw,
):
    d = datasets(groups, edges_shift=edges_shift)
    entries = [
        e
        for (sim, snap, *_r) in groups
        for e in group_entries(sim, snap, eligible=eligible, tag=str(path).rsplit("/", 1)[-1])
    ]
    entries += [dict(e) for e in extra_inputs]
    gl = [
        {"sim": sim, "snap": int(snap), "z_grid": float(zg), "rows": rows}
        for (sim, snap, zg, *_r) in groups
    ]
    a = dict(ATTRS, n_rows=len(d["sim_name"]), n_snaps=len(d["snap_snap"]), **(attrs or {}))
    return T.write_cache(
        path,
        d,
        a,
        inputs=entries,
        settings=settings or {"cache_version": T.CACHE_VERSION},
        groups=gl,
        **kw,
    )
