#!/usr/bin/env python3
"""verify_checkpoint.py -- read-only verifier for a DESI follow-up recoverability checkpoint (PI DECISIONS #23).

Resolves every path in PROVENANCE_MANIFEST.json and SHAS.json, recomputes every sha256, checks both repositories are
clean and pushed at the recorded HEADs, and exits 0 only if everything matches. The tags are cut only after exit 0.
Usage: python3 scripts/verify_checkpoint.py <checkpoint_dir>
"""
import hashlib
import json
import os
import subprocess
import sys


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(repo, *a):
    r = subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def main(cp):
    cp = os.path.abspath(cp)
    fails = []
    man = json.load(open(os.path.join(cp, "PROVENANCE_MANIFEST.json")))
    shas = json.load(open(os.path.join(cp, "SHAS.json")))
    n = 0
    for entry in man["files"]:
        p = entry["path"] if os.path.isabs(entry["path"]) else os.path.join(cp, entry["path"])
        if not os.path.exists(p):
            fails.append(f"missing: {entry['path']}"); continue
        got = sha256(p); n += 1
        if got != entry["sha256"]:
            fails.append(f"sha mismatch: {entry['path']}")
    for name, want in man.get("directory_manifests", {}).items():
        mp = want["manifest_path"]
        if not os.path.exists(mp) or sha256(mp) != want["manifest_sha256"]:
            fails.append(f"directory manifest missing or changed: {name}")
    for repo_key in ("code", "notes"):
        repo = shas[repo_key]["path"]
        rc, head = git(repo, "rev-parse", "HEAD")
        if rc != 0 or head != shas[repo_key]["head"]:
            fails.append(f"{repo_key} HEAD {head} != recorded {shas[repo_key]['head']}")
        rc, st = git(repo, "status", "--porcelain")
        if st.strip():
            fails.append(f"{repo_key} tree dirty")
        rc, remote = git(repo, "ls-remote", "origin", shas[repo_key]["branch"])
        if rc != 0 or not remote.startswith(shas[repo_key]["head"]):
            fails.append(f"{repo_key} HEAD not at origin/{shas[repo_key]['branch']}")
        for path, blob in shas[repo_key].get("blobs", {}).items():
            rc, b = git(repo, "rev-parse", f"HEAD:{path}")
            if rc != 0 or b != blob:
                fails.append(f"{repo_key} blob {path} != recorded")
    print(f"verified {n} files, {len(man.get('directory_manifests', {}))} directory manifests, 2 repositories")
    if fails:
        print("FAIL:"); [print("  " + f) for f in fails]; return 1
    print("CHECKPOINT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
