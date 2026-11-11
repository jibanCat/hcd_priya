#!/usr/bin/env python3
"""Export trained emulator members to the pickle-free release format and prove the conversion exact.

Release export (format ``hcd_priya.h5/2``, docs/release_format_h5.md): the members named with ``--member`` (training
checkpoints ``.eqx`` + ``.norm.pkl`` + ``.meta.json``), the training cache and the w_c map, each pinned by a sha256 given
on the command line, become a NEW bundle directory ``<out>/bundle`` with the lineage block and the w_c map. Before
anything is written the export refuses: a dirty tree or an unknown commit; a SHA256SUMS file that does not hash to the
given digest, or that does not list every member file with its digest (checked before any file is unpickled); a cache
that does not hash to its digest or that the release reader refuses (format 3.4, eligible, promoted if carried); a w_c
map that does not hash to its digest, is not eligible or was fitted to another cache; members that were not all trained
on that cache and its lineage record at one clean, known commit, that record a cache path instead of a file name, or
that differ in architecture or recipe; members that are not production members by the metadata rule
(``eligibility.production_recipe_reasons``: the recipe declares ``all_sims`` true, carries no reserved held-out or fold
key anywhere and no research marking); members whose content does not match the declared full training set
(``eligibility.production_content_check``, every component PASS: the split recomputed from the cache with the recipe's
``n_train`` / ``n_val``, every normaliser tensor refitted on the training rows bitwise, the w_c map refitted on the
cache); and a lineage block that is not release-eligible.

Write order: the members and the map first, then the content checks again on the files READ BACK
(``safe_h5.load_member`` with the digests just written, the map from the bundle), and only then the manifest, whose
provenance records the outcomes (``eligibility_checks``). A failed read-back check leaves no manifest, hence nothing
loadable (exit 1). The map's record names the cache's FILE NAME and the map check compares it: ``--cache`` must carry
that name (a renamed copy of the right cache is refused).

    python scripts/export_safe_bundle.py --out NEW_DIR --member DIR/NAME_seed0 [--member ...] \\
        --sha256sums DIR/SHA256SUMS --sha256sums-sha256 HEX --cache CACHE.h5 --cache-sha256 HEX \\
        --wc-map WC_MAP.json --wc-map-sha256 HEX

Historical reproduction (the earlier conversion of the pinned production ensemble, format ``hcd_priya.h5/1`` or the
safetensors prototype ``hcd_priya.safe/1``; never a release bundle):

    python scripts/export_safe_bundle.py --out NEW_DIR --historical-reproduction REASON --manifest-sha256 HEX \\
        [--manifest PATH] [--container h5|safetensors]

Checks after writing (all must hold; any failure exits non-zero), comparing BYTES, not values: every leaf, the
normalizer tree and the meta identical after the round trip; every head's output and the ensemble P_filt, with its
gradients in theta9 and tau0, identical on fixed inputs; a clean subprocess that only sees the bundle (the release
loader with the manifest digest; the historical loader with the reason) reproduces the ensemble P_filt; the bundle
verifies with no unlisted file; the originals are unchanged; for a release bundle also the public predictor
(hcd_analysis.predictor.load_bundle) reproduces the ensemble P_filt. Evidence (inputs, predictions,
``pilot_summary.json``) goes to ``<out>/evidence``. Exit status: 0 PASS, 1 FAIL, 2 refused (nothing written).
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from hcd_analysis import lineage as L  # noqa: E402
from hcd_analysis.emulator import eligibility as E  # noqa: E402
from hcd_analysis.emulator import prod_ensemble as PE  # noqa: E402
from hcd_analysis.emulator import safe_h5 as H  # noqa: E402
from hcd_analysis.emulator import safe_io as S  # noqa: E402
from hcd_analysis.emulator import train as T  # noqa: E402
from hcd_analysis.emulator.data import Z_LIMITS  # noqa: E402
from hcd_analysis.emulator.ensemble import EnsembleEmulator  # noqa: E402
from hcd_analysis.emulator.predict import predict_P_filt  # noqa: E402

N_INPUTS, SEED = 16, 20261022
CONTAINERS = {"safetensors": S, "h5": H}
EXTS = (".eqx", ".norm.pkl", ".meta.json")

CHILD = """
import sys, numpy as np, jax
jax.config.update('jax_enable_x64', True)
sys.path.insert(0, sys.argv[1])
from hcd_analysis.emulator import safe_h5 as H, safe_io as S
from hcd_analysis.emulator.ensemble import EnsembleEmulator
from hcd_analysis.emulator.predict import predict_P_filt
mode, bundle, sha = sys.argv[6], sys.argv[2], sys.argv[3]
if mode == 'release':
    ens, metas, norm = H.load_ensemble(bundle, expected_manifest_sha256=sha)
    assert metas.release_verified and metas.kind == 'release'
    pf = norm['P_filt']
else:
    loaded = (H.load_bundle_historical(bundle, expected_manifest_sha256=sha, reason=sys.argv[7]) if mode == 'h5'
              else S.load_bundle_historical(bundle, expected_manifest_sha256=sha, reason=sys.argv[7]))
    ens = EnsembleEmulator([m for m, _, _ in loaded])
    pf = loaded[0][2]['P_filt']
inp = np.load(sys.argv[4])
out = np.stack([np.asarray(predict_P_filt(ens, x[:9], x[9], t, pf)) for x, t in zip(inp['x'], inp['tau0'])])
np.save(sys.argv[5], out)
"""


class Refused(RuntimeError):
    """An input the export refuses; nothing is written."""


def _inputs():
    r = np.random.default_rng(SEED)
    x = r.uniform(size=(N_INPUTS, 10))
    z = Z_LIMITS[0] + x[:, 9] * (Z_LIMITS[1] - Z_LIMITS[0])
    tau0 = (
        r.uniform(0.7, 1.3, N_INPUTS) * 0.0023 * (1.0 + z) ** 3.65
    )  # alpha x Kim07, inside the cache ladder
    return x, tau0


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _tree_diff(a, b, where="norm"):
    """Paths where two normalizer trees differ in structure, dtype, shape or bytes."""
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or set(a) != set(b):
            return [where]
        return [p for k in a for p in _tree_diff(a[k], b[k], f"{where}/{k}")]
    return [] if _same(a, b) else [where]


def _maxabs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) if np.size(a) else 0.0


def _versions(container):
    import equinox

    v = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "equinox": equinox.__version__,
        "numpy": np.__version__,
    }
    if container == "h5":  # the bytes of an h5 bundle depend on the HDF5 stack
        import h5py

        v.update(h5py=h5py.__version__, libhdf5=h5py.version.hdf5_version)
    return v


def _git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------------------------------------- release inputs
def read_sums(path):
    """{file name: sha256} of a ``sha256sum`` listing (``<hex>  <name>`` or ``<hex> *<name>``)."""
    out = {}
    for i, line in enumerate(open(path, encoding="utf-8").read().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise Refused(f"{path}:{i}: not a sha256sum line")
        name = os.path.basename(parts[1].strip().lstrip("*"))
        if name in out:
            raise Refused(f"{path}: {name} is listed twice")
        out[name] = parts[0].lower()
    return out


def check_release_inputs(a):
    """Every metadata check before anything is written. Returns (prefixes, metas, cache record, cache identity, w_c
    map, code, the member files' sha256, the loaded cache)."""
    from hcd_analysis import tau0_cache as TC
    from hcd_analysis.emulator import wc_map as W
    from hcd_analysis.emulator.data import load_cache

    code = L.code_identity()
    if (
        not (isinstance(code.get("commit"), str) and len(code["commit"]) == 40)
        or code.get("dirty") is not False
    ):
        raise Refused(
            f"the release export runs from a clean, committed checkout only (code identity {code})"
        )
    if L.sha256_file(a.sha256sums) != a.sha256sums_sha256:
        raise Refused(f"{a.sha256sums} does not hash to {a.sha256sums_sha256}; nothing was read")
    sums = read_sums(a.sha256sums)
    sums_dir = os.path.dirname(os.path.abspath(a.sha256sums))
    prefixes = [os.path.abspath(p) for p in a.member]
    if len(set(map(os.path.basename, prefixes))) != len(prefixes):
        raise Refused("two members share a name")
    files = {}
    for p in prefixes:
        if os.path.dirname(p) != sums_dir:
            raise Refused(f"member {p} is not beside {a.sha256sums}")
        for e in EXTS:
            name = os.path.basename(p) + e
            if name not in sums:
                raise Refused(f"{name} is not listed in {a.sha256sums}")
            got = L.sha256_file(p + e)
            if got != sums[name]:
                raise Refused(f"{p + e} does not match {a.sha256sums} (nothing unpickled)")
            files[name] = got
    metas = [json.load(open(p + ".meta.json")) for p in prefixes]
    ident = L.file_identity(a.cache)
    if ident["sha256"] != a.cache_sha256:
        raise Refused(f"{a.cache} has sha256 {ident['sha256']}, not {a.cache_sha256}")
    try:
        d = load_cache(
            a.cache
        )  # strict (tau0_cache.read_cache: 3.4, eligible, promoted if carried)
    except TC.CacheLineageError as e:
        raise Refused(str(e)) from None
    rec = d["lineage"]
    if d["cache_sha256"] != ident["sha256"] or L.file_identity(a.cache) != ident:
        raise Refused(f"{a.cache} changed while it was read")
    if L.sha256_file(a.wc_map) != a.wc_map_sha256:
        raise Refused(f"{a.wc_map} does not hash to {a.wc_map_sha256}")
    try:
        wcm = W.read_wc_map(a.wc_map, expected_sha256=a.wc_map_sha256)
    except W.WcMapError as e:
        raise Refused(str(e)) from None
    if wcm.cache_sha256 != ident["sha256"]:
        raise Refused(
            f"the w_c map was fitted to the cache {wcm.cache_sha256}, not {ident['sha256']}"
        )
    m0 = metas[0]
    sims = sorted(
        set(np.asarray(d["sim_name"]).astype(str).tolist())
    )  # simulation_sets/1 with the cache's names
    for i, m in enumerate(metas):
        if m.get("arch_cfg") != m0.get("arch_cfg") or m.get("recipe") != m0.get("recipe"):
            raise Refused(f"member {i} has another architecture or recipe than member 0")
        why = E.production_recipe_reasons(m, i, sims=sims)
        if why:
            raise Refused(
                f"member {i} is not a production member (it was not trained on every simulation, or it is "
                "marked research): " + "; ".join(why)
            )
    if len({m.get("seed") for m in metas}) != len(metas):
        raise Refused("two members share a seed")
    return (
        prefixes,
        metas,
        rec,
        {"sha256": ident["sha256"], "size": ident["size"]},
        wcm,
        code,
        files,
        d,
    )


def content_checks(metas, norms, wcm, d, a):
    """The production content rules on the given objects (every component must PASS)."""
    return E.production_content_check(
        metas,
        norms,
        wcm,
        cache=d,
        cache_path=a.cache,
        expected_cache_sha256=a.cache_sha256,
        cache_size=os.path.getsize(a.cache),
    )


def _failed(check):
    return "; ".join(
        f"{c['component']} {c['outcome']} ({c['sub_reason']}"
        + (f", max_rel {c['max_rel']:.3g}" if c.get("max_rel") else "")
        + (f": {c['detail'].get('why')}" if c["detail"].get("why") else "")
        + ")"
        for c in check["components"]
        if c["outcome"] != E.PASS
    )


# -------------------------------------------------------------------------------------------------- verification
def verify(legacy, loaded, bundle, bundle_sha, evidence, child_args, checks, dmax):
    """The byte-level round-trip checks shared by both modes (``checks``/``dmax`` filled in place)."""
    x, tau0 = _inputs()
    for i, ((m0, meta0, n0), (m1, meta1, n1)) in enumerate(zip(legacy, loaded)):
        l0, t0 = jax.tree_util.tree_flatten_with_path(m0)
        l1, t1 = jax.tree_util.tree_flatten_with_path(m1)
        checks[f"m{i}_treedef"] = t0 == t1
        checks[f"m{i}_leaves"] = all(
            jax.tree_util.keystr(pa) == jax.tree_util.keystr(pb) and _same(u, v)
            for (pa, u), (pb, v) in zip(l0, l1)
        )
        checks[f"m{i}_norm"] = not _tree_diff(n0, n1)
        checks[f"m{i}_meta"] = json.loads(json.dumps(meta0)) == meta1
        o0 = jax.vmap(m0)(jnp.asarray(x), jnp.asarray(tau0))
        o1 = jax.vmap(m1)(jnp.asarray(x), jnp.asarray(tau0))
        checks[f"m{i}_heads"] = set(o0) == set(o1) and all(_same(o0[k], o1[k]) for k in o0)
        dmax[f"m{i}_heads"] = max(_maxabs(o0[k], o1[k]) for k in o0)
    ens0 = EnsembleEmulator([m for m, _, _ in legacy])
    ens1 = EnsembleEmulator([m for m, _, _ in loaded])
    pf0, pf1 = legacy[0][2]["P_filt"], loaded[0][2]["P_filt"]

    def f(ens, pf, th, zu, t):
        return jnp.sum(jnp.log(predict_P_filt(ens, th, zu, t, pf)))

    g = jax.grad(f, argnums=(2, 4))
    P0, P1, G0, G1 = [], [], [], []
    for xi, ti in zip(x, tau0):
        P0.append(np.asarray(predict_P_filt(ens0, xi[:9], xi[9], ti, pf0)))
        P1.append(np.asarray(predict_P_filt(ens1, xi[:9], xi[9], ti, pf1)))
        G0.append([np.asarray(v) for v in g(ens0, pf0, jnp.asarray(xi[:9]), xi[9], ti)])
        G1.append([np.asarray(v) for v in g(ens1, pf1, jnp.asarray(xi[:9]), xi[9], ti)])
    P0, P1 = np.stack(P0), np.stack(P1)
    checks["ensemble_P_filt"] = _same(P0, P1)
    checks["ensemble_grad_theta9_tau0"] = all(
        _same(u, v) for a0, a1 in zip(G0, G1) for u, v in zip(a0, a1)
    )
    dmax["ensemble_P_filt"] = _maxabs(P0, P1)
    dmax["ensemble_grad"] = max(_maxabs(u, v) for a0, a1 in zip(G0, G1) for u, v in zip(a0, a1))
    inp = os.path.join(evidence, "pilot_inputs.npz")
    np.savez(inp, x=x, tau0=tau0)
    np.save(os.path.join(evidence, "pilot_P_filt_legacy.npy"), P0)
    child_out = os.path.join(evidence, "pilot_P_filt_clean_process.npy")
    r = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            CHILD,
            REPO,
            bundle,
            bundle_sha,
            inp,
            child_out,
            *child_args,
        ],
        capture_output=True,
        text=True,
        cwd=evidence,
    )
    checks["clean_process_ran"] = r.returncode == 0
    checks["clean_process_P_filt"] = r.returncode == 0 and _same(np.load(child_out), P0)
    return x, tau0, P0, r


def predictor_check(bundle, bundle_sha, metas, x, tau0, P0, checks, dmax):
    """The public predictor on the new bundle reproduces the ensemble P_filt at the pilot inputs (physical
    parameters and the snapshot grid: only the inputs on a snapshot redshift are used)."""
    import hcd_analysis.predictor as hp
    from hcd_analysis.emulator.data import PARAM_LIMITS

    emu = hp.load_bundle(bundle, bundle_sha)
    lim = np.asarray(PARAM_LIMITS, float)
    worst, n = 0.0, 0
    for xi, ti, p0 in zip(x, tau0, P0):
        z = Z_LIMITS[0] + xi[9] * (Z_LIMITS[1] - Z_LIMITS[0])
        zg = round(2.0 + 0.2 * round((z - 2.0) / 0.2), 10)
        th = xi[:9]
        params = dict(zip(hp.PARAM_NAMES, (lim[:, 0] + th * (lim[:, 1] - lim[:, 0])).tolist()))
        zu = (zg - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])
        ref = np.asarray(
            predict_P_filt(
                EnsembleEmulator(emu._rb.ensemble.members),
                jnp.asarray(th),
                zu,
                ti,
                emu._rb.norm["P_filt"],
            )
        )
        got = emu.predict(params, zg, tau_eff=float(ti)).P_class
        worst = max(worst, float(np.max(np.abs(got / ref - 1.0))))
        n += 1
    checks["predictor_P_class"] = n > 0 and worst <= 1e-12
    dmax["predictor_P_class_rel"] = worst


# --------------------------------------------------------------------------------------------------------- modes
def run_release(a):
    try:
        prefixes, metas, rec, ident, wcm, code, files, d = check_release_inputs(a)
        legacy = [T.load_checkpoint(p) for p in prefixes]  # digests verified above
        if [json.loads(json.dumps(m)) for _, m, _ in legacy] != metas:
            raise Refused("a member's meta changed between the two reads")
        block = H.make_lineage_block(
            cache_record=rec, cache_identity=ident, wc_map=wcm, member_metas=metas, export_code=code
        )
        if not block["release_eligible"]:
            raise Refused(
                "the bundle would not be release-eligible: "
                + "; ".join(block["ineligible_reasons"])
            )
        before = content_checks(metas, [n for _, _, n in legacy], wcm, d, a)
        if not before.passed:
            raise Refused(
                "the members' content is not the declared full training set: " + _failed(before)
            )
    except Refused as e:
        print(f"refusing: {e}")
        return 2
    bundle, evidence = os.path.join(a.out, "bundle"), os.path.join(a.out, "evidence")
    os.makedirs(bundle)
    os.makedirs(evidence)
    out_prefixes = []
    for p, (model, meta, norm) in zip(prefixes, legacy):
        q = os.path.join(bundle, os.path.basename(p))
        H.export_member(q, model, meta, norm)
        out_prefixes.append(q)
    H.export_wc_map(bundle, a.wc_map)
    back = [
        H.load_member(
            q,
            expected_sha256=S.sha256_file(q + H.EXT),
            expected_json_sha256=S.sha256_file(q + ".json"),
        )
        for q in out_prefixes
    ]  # the objects actually written
    from hcd_analysis.emulator import wc_map as W

    after = content_checks(
        [m for _, m, _ in back],
        [n for _, _, n in back],
        W.read_wc_map(os.path.join(bundle, H.WC_MAP_FILE)),
        d,
        a,
    )
    if not after.passed:
        print(
            "FAIL: the content checks on the files read back do not pass (no manifest written, nothing loadable): "
            + _failed(after)
        )
        return 1
    del d
    man_path = H.write_manifest(
        bundle,
        out_prefixes,
        {
            "source_sha256sums": os.path.basename(a.sha256sums),
            "source_sha256sums_sha256": a.sha256sums_sha256,
            "source_files_sha256": files,
            "cache_file": os.path.basename(a.cache),
            "export_code": code,
            "versions": _versions("h5"),
            "conversion": "training checkpoints read by train.load_checkpoint after their sha256 matched the pinned "
            "SHA256SUMS; the cache read by tau0_cache.read_cache; the w_c map by wc_map.read_wc_map",
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "eligibility_checks": dict(after, read_back=True, before_writing_passed=before.passed),
        },
        lineage=block,
    )
    bundle_sha = S.sha256_file(man_path)
    checks, dmax = {}, {}
    ens, loaded_metas, _norm = H.load_ensemble(
        bundle, expected_manifest_sha256=bundle_sha
    )  # the lineage enforced
    checks["bundle_lineage_verified"] = (
        loaded_metas.release_verified and loaded_metas.kind == "release"
    )
    checks["content_checks_passed"] = after.passed
    loaded = H.load_bundle(bundle, expected_manifest_sha256=bundle_sha)
    x, tau0, P0, r = verify(legacy, loaded, bundle, bundle_sha, evidence, ["release"], checks, dmax)
    predictor_check(bundle, bundle_sha, metas, x, tau0, P0, checks, dmax)
    after = {os.path.basename(p) + e: L.sha256_file(p + e) for p in prefixes for e in EXTS}
    checks["originals_unchanged"] = after == files
    checks["cache_unchanged"] = L.sha256_file(a.cache) == a.cache_sha256
    checks["wc_map_unchanged"] = L.sha256_file(a.wc_map) == a.wc_map_sha256
    return _finish(
        evidence,
        checks,
        dmax,
        r,
        {
            "format": H.FORMAT,
            "container": "h5",
            "n_members": len(prefixes),
            "bundle_manifest_sha256": bundle_sha,
            "lineage": block,
        },
    )


def run_historical(a):
    F = CONTAINERS[a.container]
    man_src = a.manifest or PE.DEFAULT_MANIFEST_PATH
    src_sha = S.sha256_file(man_src)
    if src_sha != a.manifest_sha256:
        print(
            f"refusing: {man_src} has sha256 {src_sha}, not the expected {a.manifest_sha256}; nothing was read"
        )
        return 2
    manifest, prefixes = PE.verify_manifest(manifest_path=man_src)  # digests, pairing, count, order
    before = {p + e: S.sha256_file(p + e) for p in prefixes for e in EXTS}
    bundle, evidence = os.path.join(a.out, "bundle"), os.path.join(a.out, "evidence")
    os.makedirs(bundle)
    os.makedirs(evidence)
    legacy = [T.load_checkpoint(p) for p in prefixes]  # the provenance-verified legacy read
    out_prefixes = []
    for p, (model, meta, norm) in zip(prefixes, legacy):
        q = os.path.join(bundle, os.path.basename(p))
        if F is H:
            H.export_member(q, model, meta, norm, fmt=H.FORMAT_HISTORICAL)
        else:
            S.export_member(q, model, meta, norm)
        out_prefixes.append(q)
    prov = {
        "source_manifest": os.path.basename(man_src),
        "source_manifest_sha256": src_sha,
        "source_files_sha256": {os.path.basename(k): v for k, v in before.items()},
        "code_commit": _git("rev-parse", "HEAD"),
        "code_dirty": bool(_git("status", "--porcelain")),
        "versions": _versions(a.container),
        "historical_reproduction": a.historical_reproduction,
        "conversion": "legacy .eqx/.norm.pkl/.meta.json read by train.load_checkpoint after verify_manifest",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    man_path = (
        H.write_manifest_historical(bundle, out_prefixes, prov)
        if F is H
        else S.write_manifest(bundle, out_prefixes, prov)
    )
    bundle_sha = S.sha256_file(man_path)
    checks, dmax = {}, {}
    loaded = (
        H.load_bundle_historical(
            bundle, expected_manifest_sha256=bundle_sha, reason=a.historical_reproduction
        )
        if F is H
        else S.load_bundle_historical(
            bundle, expected_manifest_sha256=bundle_sha, reason=a.historical_reproduction
        )
    )
    checks["bundle_verifies"] = True
    _x, _t, _p, r = verify(
        legacy,
        loaded,
        bundle,
        bundle_sha,
        evidence,
        ["h5" if F is H else "safetensors", a.historical_reproduction],
        checks,
        dmax,
    )
    after = {k: S.sha256_file(k) for k in before}
    checks["originals_unchanged"] = after == before
    checks["source_manifest_unchanged"] = S.sha256_file(man_src) == src_sha
    PE.verify_manifest(manifest_path=man_src)
    checks["source_manifest_verifies_after"] = True
    fmt = H.FORMAT_HISTORICAL if F is H else S.FORMAT
    return _finish(
        evidence,
        checks,
        dmax,
        r,
        {
            "format": fmt,
            "container": a.container,
            "n_members": len(prefixes),
            "bundle_manifest_sha256": bundle_sha,
            "historical_reproduction": a.historical_reproduction,
        },
    )


def _finish(evidence, checks, dmax, r, info):
    ok = all(checks.values())
    summary = dict(
        info,
        verdict="PASS" if ok else "FAIL",
        checks=checks,
        max_abs_diff=dmax,
        n_inputs=N_INPUTS,
        seed=SEED,
        child_stderr_tail=r.stderr[-2000:] if r.returncode else "",
    )
    with open(os.path.join(evidence, "pilot_summary.json"), "x") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    for k, v in checks.items():
        print(f"{'ok  ' if v else 'FAIL'} {k}")
    print(f"verdict: {summary['verdict']}; bundle manifest sha256 {info['bundle_manifest_sha256']}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="NEW directory (refused if it exists)")
    rel = ap.add_argument_group("release export (hcd_priya.h5/2)")
    rel.add_argument(
        "--member",
        action="append",
        help="a member's checkpoint prefix (repeatable, in member order)",
    )
    rel.add_argument("--sha256sums", help="the SHA256SUMS listing every member file")
    rel.add_argument("--sha256sums-sha256", help="its expected sha256")
    rel.add_argument("--cache", help="the release training cache (format 3.4)")
    rel.add_argument("--cache-sha256", help="its expected sha256")
    rel.add_argument("--wc-map", help="the w_c map (scripts/refit_wc_map.py)")
    rel.add_argument("--wc-map-sha256", help="its expected sha256")
    his = ap.add_argument_group(
        "historical reproduction (hcd_priya.h5/1 or the safetensors prototype)"
    )
    his.add_argument(
        "--historical-reproduction", metavar="REASON", help="convert the pinned production ensemble"
    )
    his.add_argument(
        "--manifest", default=None, help="production manifest (default: the pinned one)"
    )
    his.add_argument("--manifest-sha256", help="expected sha256 of the production manifest")
    his.add_argument("--container", choices=sorted(CONTAINERS), default="h5")
    a = ap.parse_args(argv)
    release_args = (
        "member",
        "sha256sums",
        "sha256sums_sha256",
        "cache",
        "cache_sha256",
        "wc_map",
        "wc_map_sha256",
    )
    if a.historical_reproduction is not None:
        if not a.historical_reproduction.strip() or a.manifest_sha256 is None:
            ap.error("--historical-reproduction needs a non-empty reason and --manifest-sha256")
        if any(getattr(a, k) is not None for k in release_args):
            ap.error("the release arguments do not go with --historical-reproduction")
    else:
        missing = [k for k in release_args if getattr(a, k) is None]
        if missing:
            ap.error(
                "a release export needs "
                + ", ".join("--" + k.replace("_", "-") for k in missing)
                + " (or --historical-reproduction REASON for the earlier conversion)"
            )
        if a.manifest is not None or a.manifest_sha256 is not None or a.container != "h5":
            ap.error(
                "--manifest, --manifest-sha256 and --container belong to --historical-reproduction"
            )
    if os.path.exists(a.out):
        print(f"refusing: {a.out} exists (the export never writes into an existing directory)")
        return 2
    return run_historical(a) if a.historical_reproduction is not None else run_release(a)


if __name__ == "__main__":
    sys.exit(main())
