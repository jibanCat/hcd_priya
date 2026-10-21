#!/usr/bin/env python3
"""(Re)build or --check the pinned production-ensemble manifest (freeze decision 6).

GENERATE (default): scan a checkpoints dir for the five ``prod_repaired_seed{0..4}`` members (PE.PROD_BASENAME),
compute SHA256 for every .eqx/.norm.pkl/.meta.json, and write the machine-readable manifest
(``checkpoints/production_ensemble_manifest.json``). The manifest is NEVER hand-edited: any
intentional replacement of the production ensemble regenerates it HERE, then goes through PI
sign-off and a new analysis.lock (see checkpoints/README_production_ensemble.md).

CHECK (--check): verify the on-disk files against the COMMITTED manifest (digests + exact
pairing + count + order + the stray-member tripwire) and exit nonzero on ANY mismatch. This is
the same battery every deployed driver runs at load time
(hcd_analysis.emulator.prod_ensemble.verify_manifest); diagnostic scripts that keep their own
member plumbing can call ``gen_ensemble_manifest.verify(...)`` (subprocess-free) instead.

ENV: PYTHONNOUSERSITE=1 PYTHONPATH=<repo> JAX_PLATFORMS=cpu \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/gen_ensemble_manifest.py [--check]
"""
from __future__ import annotations
import os as _os_rr  # this checkout's root (emulator-debug 2026-10; never an absolute literal)
_REPO_ROOT = _os_rr.path.dirname(_os_rr.path.dirname(_os_rr.path.abspath(__file__)))

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

from hcd_analysis.emulator import prod_ensemble as PE

# Gate E (PU-0067): the production members are the repaired-cache gate C checkpoints on Turbo; the manifest (committed with
# the code) names that directory. The pre-2026-10 schema-1 manifest is kept as checkpoints/production_ensemble_manifest_v1_pre2026-10.json.
DEFAULT_CHECKPOINTS_DIR = "/nfs/turbo/umor-yueyingn/mfho/hcd/emulator_v2/checkpoints/gateC"

DESCRIPTION = (
    "The deployed N=5 production emulator ensemble (prod_repaired_seed0..4, gate C, S4-repaired cache, schema 2.0): "
    "all-sims-trained "
    "Equinox checkpoints + paired normalizers, pinned by SHA256. The deployed forward is the "
    "ensemble MEAN over members of the reconstructed post-exp P_filt (ensemble.load_ensemble)."
)
NORM_NOTE = (
    "The five .norm.pkl files are byte-identical by construction: every member is trained on "
    "the same data with the same recipe (only the network init seed differs), so they share "
    "one normalization; checkpoint<->normalizer pairing is therefore pinned STRUCTURALLY "
    "(basename per index), not by digest."
)
PROVENANCE_NOTE = (
    "The .eqx/.norm.pkl/.meta.json/.hist.json files live in checkpoints_dir (not in git) and are pinned here by "
    "SHA256 and cross-checked against that directory's SHA256SUMS. Per-member training config: "
    f"{PE.PROD_BASENAME}<i>.meta.json (digest-pinned here; schema 2.0, cache_sha256, k_com_hmpc); history: "
    f"{PE.PROD_BASENAME}<i>.hist.json beside it. Trained by gate C (PU-0056/PU-0057) from a git-archive export of "
    "commit 01ad2ff by the notes batch gateC/batch_gateC_train_01ad2ff.sbatch (metas carry git_sha null; see checkpoints_dir/MANIFEST_README.txt)."
)
REPLACEMENT_PROCEDURE = (
    "Intentional replacement ONLY: retrain via scripts/train_production_emulator.py (once per "
    "seed), regenerate this manifest via scripts/gen_ensemble_manifest.py (NEVER hand-edit), "
    "obtain PI sign-off, and regenerate analysis.lock. See "
    "checkpoints/README_production_ensemble.md."
)


def build_manifest(checkpoints_dir, n_members=PE.N_PROD_MEMBERS):
    """Build the manifest dict from a checkpoints dir. Refuses to build over a stray
    production-prefix file (the manifest must pin EVERYTHING the prefix can match)."""
    members = []
    for i in range(int(n_members)):
        name = f"{PE.PROD_BASENAME}{i}"
        eqx = os.path.join(checkpoints_dir, name + ".eqx")
        norm = os.path.join(checkpoints_dir, name + ".norm.pkl")
        meta = os.path.join(checkpoints_dir, name + ".meta.json")
        for p in (eqx, norm, meta):
            if not os.path.exists(p):
                raise PE.ProductionEnsembleError(
                    f"cannot build manifest: missing production ensemble file {p}")
        members.append(dict(
            index=i, name=name,
            eqx=name + ".eqx", eqx_sha256=PE.sha256_file(eqx),
            norm=name + ".norm.pkl", norm_sha256=PE.sha256_file(norm),
            meta=name + ".meta.json", meta_sha256=PE.sha256_file(meta),
            hist=name + ".hist.json"))
    pinned = {m["eqx"] for m in members}
    on_disk = sorted(os.path.basename(p) for p in
                     glob.glob(os.path.join(checkpoints_dir, PE.PROD_BASENAME + "*.eqx")))
    strays = [b for b in on_disk if b not in pinned]
    if strays:
        raise PE.ProductionEnsembleError(
            f"cannot build manifest: stray production-prefix checkpoint(s) {strays} in "
            f"{checkpoints_dir} beyond the {n_members} pinned members -- remove them (or "
            f"deliberately change the member count) before regenerating")
    norm_identical = len({m["norm_sha256"] for m in members}) == 1
    caches, k_ref = set(), None
    for m in members:
        meta = json.load(open(os.path.join(checkpoints_dir, m["meta"])))
        if meta.get("schema_version") != PE.MEMBER_SCHEMA_VERSION:
            raise PE.ProductionEnsembleError(f"cannot build manifest: {m['name']} has checkpoint schema "
                                             f"{meta.get('schema_version')!r}, expected {PE.MEMBER_SCHEMA_VERSION}")
        caches.add(meta.get("cache_sha256"))
        k_ref = meta.get("k_com_hmpc") if k_ref is None else k_ref
        if meta.get("k_com_hmpc") != k_ref:
            raise PE.ProductionEnsembleError(f"cannot build manifest: {m['name']} k_com_hmpc differs from member 0")
        m["schema_version"] = meta.get("schema_version")
        m["cache_sha256"] = meta.get("cache_sha256")
    if len(caches) != 1:
        raise PE.ProductionEnsembleError(f"cannot build manifest: members were trained on different caches {sorted(caches)}")
    sums = os.path.join(checkpoints_dir, "SHA256SUMS")
    return {
        "schema_version": PE.MANIFEST_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": DESCRIPTION,
        "checkpoints_dir": os.path.abspath(checkpoints_dir),
        "basename": PE.PROD_BASENAME,
        "member_schema_version": PE.MEMBER_SCHEMA_VERSION,
        "cache_sha256": caches.pop(),
        "sha256sums_sha256": PE.sha256_file(sums) if os.path.exists(sums) else None,
        "n_members": int(n_members),
        "normalizers_byte_identical": bool(norm_identical),
        "normalizers_byte_identical_note": NORM_NOTE,
        "members": members,
        "provenance": {
            "training_script": "scripts/train_production_emulator.py",
            "note": PROVENANCE_NOTE,
            "replacement_procedure": REPLACEMENT_PROCEDURE,
        },
    }


def verify(checkpoints_dir=None, manifest_path=None):
    """Subprocess-free verification hook for diagnostic scripts: raises
    ``prod_ensemble.ProductionEnsembleError`` on any drift from the committed manifest.
    Returns ``(manifest, member_prefixes)``."""
    return PE.verify_manifest(checkpoints_dir=checkpoints_dir, manifest_path=manifest_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints-dir", default=DEFAULT_CHECKPOINTS_DIR,
                    help="dir holding the prod_repaired_seed{0..4} binaries "
                         f"(default {DEFAULT_CHECKPOINTS_DIR})")
    ap.add_argument("--manifest", default=PE.DEFAULT_MANIFEST_PATH,
                    help=f"manifest path (default {PE.DEFAULT_MANIFEST_PATH})")
    ap.add_argument("--check", action="store_true",
                    help="verify the on-disk files against the COMMITTED manifest; exit "
                         "nonzero on any mismatch (write nothing)")
    a = ap.parse_args(argv)

    if a.check:
        try:
            manifest, prefixes = PE.verify_manifest(checkpoints_dir=a.checkpoints_dir,
                                                    manifest_path=a.manifest)
        except PE.ProductionEnsembleError as e:
            print(f"[gen_ensemble_manifest --check] FAIL: {e}", file=sys.stderr)
            return 1
        print(f"[gen_ensemble_manifest --check] OK: {len(prefixes)} members verified "
              f"({a.checkpoints_dir} vs {a.manifest})")
        return 0

    manifest = build_manifest(a.checkpoints_dir)
    os.makedirs(os.path.dirname(os.path.abspath(a.manifest)), exist_ok=True)
    with open(a.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"[gen_ensemble_manifest] wrote {a.manifest} "
          f"({manifest['n_members']} members, normalizers_byte_identical="
          f"{manifest['normalizers_byte_identical']})")
    for m in manifest["members"]:
        print(f"  {m['name']}: eqx {m['eqx_sha256'][:16]}...  norm {m['norm_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
