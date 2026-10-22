#!/usr/bin/env python3
"""Convert the pinned production LF ensemble to the pickle-free release format and prove the conversion exact.

This is the deliberate legacy-conversion step of docs/release_format.md: the members are read only after
``prod_ensemble.verify_manifest`` has checked their digests, through the existing ``train.load_checkpoint`` (which
unpickles this project's own ``.norm.pkl``), and written into a NEW directory. Nothing is overwritten; the originals'
sha256 are checked again at the end.

Checks (all must hold; any failure exits non-zero): every leaf, the normalizer tree and the meta bit-identical after the
round trip; every head's output and the ensemble P_filt, with its gradients in theta9 and tau0, bit-identical on fixed
inputs; a clean subprocess that only sees the bundle reproduces the ensemble P_filt; the bundle manifest verifies; the
originals are unchanged. The summary (booleans and maximum absolute differences) goes to ``<out>/pilot_summary.json``;
prediction values stay in the output directory.

  python scripts/export_safe_bundle.py --out /path/to/new_dir [--manifest checkpoints/production_ensemble_manifest.json]
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from hcd_analysis.emulator import prod_ensemble as PE  # noqa: E402
from hcd_analysis.emulator import safe_io as S  # noqa: E402
from hcd_analysis.emulator import train as T  # noqa: E402
from hcd_analysis.emulator.data import Z_LIMITS  # noqa: E402
from hcd_analysis.emulator.ensemble import EnsembleEmulator  # noqa: E402
from hcd_analysis.emulator.predict import predict_P_filt  # noqa: E402

N_INPUTS, SEED = 16, 20261022
EXTS = (".eqx", ".norm.pkl", ".meta.json")

CHILD = """
import sys, numpy as np, jax
jax.config.update('jax_enable_x64', True)
sys.path.insert(0, sys.argv[1])
from hcd_analysis.emulator import safe_io as S
from hcd_analysis.emulator.ensemble import EnsembleEmulator
from hcd_analysis.emulator.predict import predict_P_filt
prefixes = S.verify_manifest(sys.argv[2])
loaded = [S.load_member(p) for p in prefixes]
ens = EnsembleEmulator([m for m, _, _ in loaded])
pf = loaded[0][2]['P_filt']
inp = np.load(sys.argv[3])
out = np.stack([np.asarray(predict_P_filt(ens, x[:9], x[9], t, pf)) for x, t in zip(inp['x'], inp['tau0'])])
np.save(sys.argv[4], out)
"""


def _inputs():
    r = np.random.default_rng(SEED)
    x = r.uniform(size=(N_INPUTS, 10))
    z = Z_LIMITS[0] + x[:, 9] * (Z_LIMITS[1] - Z_LIMITS[0])
    tau0 = r.uniform(0.7, 1.3, N_INPUTS) * 0.0023 * (1.0 + z) ** 3.65      # alpha x Kim07, inside the cache ladder
    return x, tau0


def _tree_diff(a, b, where="norm"):
    """Paths where two normalizer trees differ in structure, dtype, shape or value."""
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or set(a) != set(b):
            return [where]
        return [p for k in a for p in _tree_diff(a[k], b[k], f"{where}/{k}")]
    a, b = np.asarray(a), np.asarray(b)
    return [] if (a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)) else [where]


def _maxabs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) if np.size(a) else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="NEW directory for the bundle (refused if it exists)")
    ap.add_argument("--manifest", default=None, help="production manifest (default: the pinned one)")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"refusing: {a.out} exists (the export never writes into an existing directory)")

    manifest, prefixes = PE.verify_manifest(manifest_path=a.manifest)          # digests, pairing, count, order
    before = {p + e: S.sha256_file(p + e) for p in prefixes for e in EXTS}
    os.makedirs(a.out)
    checks, dmax = {}, {}

    legacy = [T.load_checkpoint(p) for p in prefixes]                          # the provenance-verified legacy read
    out_prefixes = []
    for p, (model, meta, norm) in zip(prefixes, legacy):
        q = os.path.join(a.out, os.path.basename(p))
        S.export_member(q, model, meta, norm)
        out_prefixes.append(q)
    man_path = S.write_manifest(a.out, out_prefixes, {
        "source_manifest": os.path.abspath(a.manifest or PE.DEFAULT_MANIFEST_PATH),
        "source_manifest_sha256": S.sha256_file(a.manifest or PE.DEFAULT_MANIFEST_PATH),
        "source_files_sha256": {os.path.basename(k): v for k, v in before.items()},
        "code_commit": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                                      text=True).stdout.strip(),
        "conversion": "legacy .eqx/.norm.pkl/.meta.json read by train.load_checkpoint after verify_manifest",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})

    loaded = [S.load_member(q) for q in S.verify_manifest(a.out)]
    checks["manifest_verifies"] = True
    x, tau0 = _inputs()
    for i, ((m0, meta0, n0), (m1, meta1, n1)) in enumerate(zip(legacy, loaded)):
        l0, t0 = jax.tree_util.tree_flatten_with_path(m0)
        l1, t1 = jax.tree_util.tree_flatten_with_path(m1)
        checks[f"m{i}_treedef"] = t0 == t1
        checks[f"m{i}_leaves"] = all(
            jax.tree_util.keystr(pa) == jax.tree_util.keystr(pb) and np.asarray(u).dtype == np.asarray(v).dtype
            and np.array_equal(np.asarray(u), np.asarray(v)) for (pa, u), (pb, v) in zip(l0, l1))
        checks[f"m{i}_norm"] = not _tree_diff(n0, n1)
        checks[f"m{i}_meta"] = json.loads(json.dumps(meta0)) == meta1
        o0 = jax.vmap(m0)(jnp.asarray(x), jnp.asarray(tau0))
        o1 = jax.vmap(m1)(jnp.asarray(x), jnp.asarray(tau0))
        checks[f"m{i}_heads"] = set(o0) == set(o1) and all(np.array_equal(np.asarray(o0[k]), np.asarray(o1[k]))
                                                           for k in o0)
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
    checks["ensemble_P_filt"] = np.array_equal(P0, P1)
    checks["ensemble_grad_theta9_tau0"] = all(np.array_equal(u, v) for a0, a1 in zip(G0, G1) for u, v in zip(a0, a1))
    dmax["ensemble_P_filt"] = _maxabs(P0, P1)
    dmax["ensemble_grad"] = max(_maxabs(u, v) for a0, a1 in zip(G0, G1) for u, v in zip(a0, a1))

    inp = os.path.join(a.out, "pilot_inputs.npz")
    np.savez(inp, x=x, tau0=tau0)
    np.save(os.path.join(a.out, "pilot_P_filt_legacy.npy"), P0)
    child_out = os.path.join(a.out, "pilot_P_filt_clean_process.npy")
    r = subprocess.run([sys.executable, "-I", "-c", CHILD, REPO, a.out, inp, child_out], capture_output=True,
                       text=True, cwd=a.out)
    checks["clean_process_ran"] = r.returncode == 0
    checks["clean_process_P_filt"] = r.returncode == 0 and np.array_equal(np.load(child_out), P0)

    after = {k: S.sha256_file(k) for k in before}
    checks["originals_unchanged"] = after == before
    PE.verify_manifest(manifest_path=a.manifest)
    checks["source_manifest_verifies_after"] = True

    ok = all(checks.values())
    summary = {"format": S.FORMAT, "verdict": "PASS" if ok else "FAIL", "checks": checks, "max_abs_diff": dmax,
               "n_members": len(prefixes), "n_inputs": N_INPUTS, "seed": SEED, "bundle_manifest": man_path,
               "child_stderr_tail": r.stderr[-2000:] if r.returncode else ""}
    with open(os.path.join(a.out, "pilot_summary.json"), "x") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    for k, v in checks.items():
        print(f"{'ok  ' if v else 'FAIL'} {k}")
    print(f"verdict: {summary['verdict']}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
