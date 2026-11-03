"""D5-R2 prediction parity probe (AI reviewer). Usage: parity.py REPO BUNDLE_DIR MANIFEST_SHA OUT_NPZ OUT_JSON.
Loads the bundle through the public predictor of the checkout REPO, predicts at fixed points, writes every output
array to OUT_NPZ and a summary (refusals, hashes) to OUT_JSON. Writes nothing else."""
import hashlib
import json
import os
import subprocess
import sys

repo, bundle, sha, out_npz, out_json = sys.argv[1:6]
sys.path.insert(0, repo)
import numpy as np  # noqa: E402
import hcd_analysis.predictor as hp  # noqa: E402
from hcd_analysis.emulator import safe_h5 as H  # noqa: E402

assert os.path.realpath(os.path.dirname(os.path.dirname(hp.__file__))) == os.path.realpath(repo), hp.__file__
head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()

summary = {"repo": repo, "head": head, "dirty": bool(dirty), "predictor_file": hp.__file__}

# refusals
for label, digest in (("none", None), ("wrong", "0" * 64), ("wrong_one_char", sha[:-1] + ("0" if sha[-1] != "0" else "1"))):
    try:
        hp.load_bundle(bundle, digest)
        summary[f"refuse_{label}"] = "LOADED (not refused)"
    except Exception as e:  # noqa: BLE001
        summary[f"refuse_{label}"] = f"{type(e).__module__}.{type(e).__name__}: {str(e)[:200]}"

emu = hp.load_bundle(bundle, sha)
summary.update(manifest_sha256=emu.manifest_sha256, n_members=emu.n_members,
               lineage_release_eligible=emu.lineage["release_eligible"], lineage_code_commit=emu.lineage["code_commit"],
               lineage_cache_sha256=emu.lineage["cache"]["sha256"], finder_ids=emu.lineage["finder_ids"],
               wc_map_sha256=emu.provenance["wc_map_sha256"],
               lineage_canonical_sha256=hashlib.sha256(json.dumps(emu.lineage, sort_keys=True,
                                                                  separators=(",", ":")).encode()).hexdigest())

lim = np.asarray(hp.PARAM_LIMITS if hasattr(hp, "PARAM_LIMITS") else __import__(
    "hcd_analysis.emulator.data", fromlist=["PARAM_LIMITS"]).PARAM_LIMITS, float)
P_mid = dict(zip(hp.PARAM_NAMES, (0.5 * (lim[:, 0] + lim[:, 1])).tolist()))
P_off = dict(zip(hp.PARAM_NAMES, (lim[:, 0] + np.array([0.3, 0.7, 0.4, 0.6, 0.25, 0.55, 0.45, 0.35, 0.65])
                                  * (lim[:, 1] - lim[:, 0])).tolist()))
arrays = {}


def dump(tag, pred):
    for f in ("k", "P_class", "template", "w_c", "dndx", "member_std", "tau_eff", "z"):
        v = getattr(pred, f)
        if v is not None:
            arrays[f"{tag}/{f}"] = np.asarray(v)
    tot = hp.add_hcd(pred, pred.w_c)
    arrays[f"{tag}/P_tot_wc"] = np.asarray(tot)
    summary[f"{tag}_flags"] = json.loads(json.dumps(pred.flags, default=str))


dump("mid_z3.0_taueff0.36", emu.predict(P_mid, 3.0, tau_eff=0.36))
dump("off_z2.6_taueff0.25", emu.predict(P_off, 2.6, tau_eff=0.25))
dump("off_many_tau0amp", emu.predict_many(P_off, [2.6, 2.8, 3.0, 3.4, 4.2], tau0_amp=1.0, dtau0=0.0))
dump("mid_many_tau0amp_k", emu.predict_many(P_mid, [3.0, 3.6], tau0_amp=0.95, dtau0=0.1,
                                            k=np.linspace(0.003, 0.018, 12)))

np.savez(out_npz, **arrays)
h = hashlib.sha256()
for k in sorted(arrays):
    a = np.ascontiguousarray(arrays[k])
    h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
summary["n_arrays"] = len(arrays)
summary["arrays_sha256"] = h.hexdigest()
summary["per_array_sha256"] = {k: hashlib.sha256(np.ascontiguousarray(arrays[k]).tobytes()).hexdigest()
                               for k in sorted(arrays)}
with open(out_json, "x") as fh:
    json.dump(summary, fh, indent=1, sort_keys=True)
print(json.dumps({k: v for k, v in summary.items() if k != "per_array_sha256"}, indent=1, sort_keys=True))
