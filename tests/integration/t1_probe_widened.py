"""D5 implementation step 0 (AI implementer, 2026-11-02): the WIDENED T1 probe (plan v2 T1(a), D5P_R3 M3 / c4).
Usage: t1_widened_probe.py REPO BUNDLE_DIR MANIFEST_SHA OUT_NPZ OUT_JSON.

It uses only public calls that exist at a5e4054 AND after D5 (hcd_analysis.predictor.load_bundle, Bundle.predict,
Bundle.predict_many, add_hcd, the Bundle attributes; safe_h5.load_ensemble / load_bundle / load_release), so the same
file runs at the pre-change and the post-change revision. Beyond D5-R2's parity.py (kept unchanged beside it) it
records:
  * the four parity.py calls plus a per-z tau_eff array call and a 2-D k call;
  * per call: every output array, flags, labels, provenance (canonical JSON digests), params, stacked, z;
  * the public attribute set of the Bundle object, its manifest_sha256, n_members, lineage, provenance, labels(),
    the w_c map sha256 and document;
  * safe_h5.load_ensemble: every member leaf (tree path, dtype, shape, bytes), the normaliser tensors, the metas'
    JSON and their load flags (anchored, lineage_verified, format, manifest_sha256, lineage);
  * safe_h5.load_bundle: the members' leaves and normalisers; safe_h5.load_release: its pre-D5 fields.
Hash rule (as parity.py): sorted names; per array name, dtype string, shape string, bytes. Writes OUT_NPZ and
OUT_JSON (exclusive) only."""

import hashlib
import json
import os
import subprocess
import sys

repo, bundle, sha, out_npz, out_json = sys.argv[1:6]
sys.path.insert(0, repo)
import numpy as np  # noqa: E402
import jax  # noqa: E402
import hcd_analysis.predictor as hp  # noqa: E402
from hcd_analysis.emulator import safe_h5 as H  # noqa: E402
from hcd_analysis.emulator import safe_io as S  # noqa: E402
from hcd_analysis.emulator.data import PARAM_LIMITS  # noqa: E402

assert os.path.realpath(os.path.dirname(os.path.dirname(hp.__file__))) == os.path.realpath(
    repo
), hp.__file__
head = subprocess.run(
    ["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True
).stdout.strip()
dirty = subprocess.run(
    ["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
    capture_output=True,
    text=True,
).stdout.strip()


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), default=str)


def csha(x):
    return hashlib.sha256(canon(x).encode()).hexdigest()


summary = {
    "repo": repo,
    "head": head,
    "dirty": bool(dirty),
    "predictor_file": hp.__file__,
    "calls": {},
}
arrays = {}

emu = hp.load_bundle(bundle, sha)
summary["bundle"] = {
    "public_attributes": sorted(n for n in dir(emu) if not n.startswith("_")),
    "type": f"{type(emu).__module__}.{type(emu).__qualname__}",
    "manifest_sha256": emu.manifest_sha256,
    "n_members": emu.n_members,
    "lineage_canonical_sha256": csha(emu.lineage),
    "provenance_canonical_sha256": csha(emu.provenance),
    "labels_canonical_sha256": csha(emu.labels()),
    "labels": emu.labels(),
    "wc_map_sha256": emu.wc_map.sha256,
    "wc_map_doc_canonical_sha256": csha(emu.wc_map.doc),
    "wc_map_xbar_z_fit": list(emu.wc_map.xbar_z_fit),
}

lim = np.asarray(PARAM_LIMITS, float)
P_mid = dict(zip(hp.PARAM_NAMES, (0.5 * (lim[:, 0] + lim[:, 1])).tolist()))
P_off = dict(
    zip(
        hp.PARAM_NAMES,
        (
            lim[:, 0]
            + np.array([0.3, 0.7, 0.4, 0.6, 0.25, 0.55, 0.45, 0.35, 0.65]) * (lim[:, 1] - lim[:, 0])
        ).tolist(),
    )
)


def dump(tag, pred):
    for f in ("k", "P_class", "template", "w_c", "dndx", "member_std", "tau_eff", "z"):
        v = getattr(pred, f)
        if v is not None:
            arrays[f"{tag}/{f}"] = np.asarray(v)
    arrays[f"{tag}/P_tot_wc"] = np.asarray(hp.add_hcd(pred, pred.w_c))
    summary["calls"][tag] = {
        "flags": json.loads(canon(pred.flags)),
        "flags_sha256": csha(pred.flags),
        "labels_sha256": csha(pred.labels),
        "provenance_sha256": csha(pred.provenance),
        "params": pred.params,
        "stacked": pred.stacked,
        "z": pred.z if not pred.stacked else list(np.asarray(pred.z).tolist()),
        "fields": sorted(f.name for f in __import__("dataclasses").fields(pred)),
    }


dump("mid_z3.0_taueff0.36", emu.predict(P_mid, 3.0, tau_eff=0.36))
dump("off_z2.6_taueff0.25", emu.predict(P_off, 2.6, tau_eff=0.25))
dump(
    "off_many_tau0amp", emu.predict_many(P_off, [2.6, 2.8, 3.0, 3.4, 4.2], tau0_amp=1.0, dtau0=0.0)
)
dump(
    "mid_many_tau0amp_k",
    emu.predict_many(P_mid, [3.0, 3.6], tau0_amp=0.95, dtau0=0.1, k=np.linspace(0.003, 0.018, 12)),
)
dump(
    "off_many_taueff_per_z",
    emu.predict_many(P_off, [2.6, 3.0, 3.4], tau_eff=np.array([0.25, 0.36, 0.5])),
)
dump(
    "mid_many_k2d",
    emu.predict_many(
        P_mid,
        [3.0, 3.6],
        tau0_amp=1.0,
        dtau0=0.0,
        k=np.stack([np.linspace(0.003, 0.018, 12), np.linspace(0.004, 0.017, 12)]),
    ),
)


def leaves(prefix, model):
    for path, leaf in jax.tree_util.tree_flatten_with_path(model)[0]:
        if hasattr(leaf, "dtype"):
            arrays[f"{prefix}{jax.tree_util.keystr(path)}"] = np.asarray(leaf)


ens, metas, norm = H.load_ensemble(bundle, expected_manifest_sha256=sha)
for i, m in enumerate(ens.members):
    leaves(f"load_ensemble/member{i}/model", m)
for k, v in S.norm_tensors(norm).items():
    arrays[f"load_ensemble/norm/{k}"] = np.asarray(v)
summary["load_ensemble"] = {
    "n_members": len(ens.members),
    "metas_canonical_sha256": csha(list(metas)),
    "anchored": metas.anchored,
    "lineage_verified": metas.lineage_verified,
    "format": metas.format,
    "manifest_sha256": metas.manifest_sha256,
    "lineage_canonical_sha256": csha(metas.lineage),
    "historical_reason": metas.historical_reason,
    "is_list": isinstance(metas, list),
}
members = H.load_bundle(bundle, expected_manifest_sha256=sha)
for i, (m, meta, nm) in enumerate(members):
    leaves(f"load_bundle/member{i}/model", m)
    for k, v in S.norm_tensors(nm).items():
        arrays[f"load_bundle/member{i}/norm/{k}"] = np.asarray(v)
summary["load_bundle"] = {
    "n_members": len(members),
    "metas_canonical_sha256": csha([m for _, m, _ in members]),
}
rb = H.load_release(bundle, expected_manifest_sha256=sha)
summary["load_release"] = {
    "anchored": rb.anchored,
    "manifest_sha256": rb.manifest_sha256,
    "manifest_canonical_sha256": csha(rb.manifest),
    "lineage_canonical_sha256": csha(rb.lineage),
    "wc_map_sha256": rb.wc_map.sha256,
    "n_members": len(rb.ensemble.members),
    "metas_canonical_sha256": csha(list(rb.metas)),
    "norm_tensors_sha256": None,
    "bundle_dir_basename": os.path.basename(rb.bundle_dir),
}
h = hashlib.sha256()
for k, v in sorted(S.norm_tensors(rb.norm).items()):
    a = np.ascontiguousarray(v)
    h.update(k.encode())
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
summary["load_release"]["norm_tensors_sha256"] = h.hexdigest()

np.savez(out_npz, **arrays)
h = hashlib.sha256()
for k in sorted(arrays):
    a = np.ascontiguousarray(arrays[k])
    h.update(k.encode())
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
summary["n_arrays"] = len(arrays)
summary["arrays_sha256"] = h.hexdigest()
summary["per_array_sha256"] = {
    k: hashlib.sha256(np.ascontiguousarray(arrays[k]).tobytes()).hexdigest() for k in sorted(arrays)
}
# the digest that must not change across D5 (everything but the checkout's identity)
stable = {k: v for k, v in summary.items() if k not in ("repo", "head", "dirty", "predictor_file")}
summary["stable_canonical_sha256"] = csha(stable)
with open(out_json, "x") as fh:
    json.dump(summary, fh, indent=1, sort_keys=True, default=str)
print(
    json.dumps(
        {
            k: summary[k]
            for k in ("head", "dirty", "n_arrays", "arrays_sha256", "stable_canonical_sha256")
        },
        indent=1,
    )
)
