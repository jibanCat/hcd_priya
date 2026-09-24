#!/usr/bin/env python3
"""verify_checkpoint.py -- read-only verifier for a DESI follow-up recoverability checkpoint (PI DECISIONS #23).

Two modes, chosen automatically:
  PRE-TAG (the tags in SHAS.json do not exist yet): every file in PROVENANCE_MANIFEST.json resolves and re-hashes; both
    working trees are clean and pushed; the code HEAD and blobs equal the recorded ones. Prints the notes HEAD to tag.
  TAG (the tags exist): the same file checks on the working tree PLUS: every checkpoint file AT THE NOTES TAG hashes to the
    manifest (immutability), the code tag points at the recorded code HEAD with the recorded blobs, and both tags exist on
    origin. HEAD may have moved on since the tag; that is allowed.
Exit 0 only if everything matches. Usage: python3 scripts/verify_checkpoint.py <checkpoint_dir>
"""
import hashlib
import json
import os
import subprocess
import sys


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(repo, *a):
    r = subprocess.run(["git", "-C", repo, *a], capture_output=True)
    return r.returncode, r.stdout


def main(cp):
    cp = os.path.abspath(cp)
    fails = []
    man = json.load(open(os.path.join(cp, "PROVENANCE_MANIFEST.json")))
    shas = json.load(open(os.path.join(cp, "SHAS.json")))
    code, notes = shas["code"]["path"], shas["notes"]["path"]
    n = 0
    for entry in man["files"]:
        p = entry["path"] if os.path.isabs(entry["path"]) else os.path.join(cp, entry["path"])
        if not os.path.exists(p):
            fails.append(f"missing: {entry['path']}"); continue
        n += 1
        if sha256(p) != entry["sha256"]:
            fails.append(f"sha mismatch: {entry['path']}")
    for name, want in man.get("directory_manifests", {}).items():
        mp = want["manifest_path"]
        if not os.path.exists(mp) or sha256(mp) != want["manifest_sha256"]:
            fails.append(f"directory manifest missing or changed: {name}")
    code_tag, notes_tag = shas["code"]["tag"], shas["notes"]["tag"]
    rc_c, _ = git(code, "rev-parse", "-q", "--verify", f"refs/tags/{code_tag}")
    rc_n, _ = git(notes, "rev-parse", "-q", "--verify", f"refs/tags/{notes_tag}")
    tag_mode = (rc_c == 0 and rc_n == 0)
    # code identity
    rc, out = git(code, "rev-parse", "HEAD" if not tag_mode else f"{code_tag}^{{commit}}")
    code_commit = out.decode().strip()
    if rc != 0 or code_commit != shas["code"]["head"]:
        fails.append(f"code {'tag' if tag_mode else 'HEAD'} {code_commit} != recorded {shas['code']['head']}")
    for path, blob in shas["code"].get("blobs", {}).items():
        rc, b = git(code, "rev-parse", f"{shas['code']['head']}:{path}")
        if rc != 0 or b.decode().strip() != blob:
            fails.append(f"code blob {path} != recorded")
    if tag_mode:
        rel = os.path.relpath(cp, notes)
        for entry in man["files"]:
            if os.path.isabs(entry["path"]):
                continue
            rc, b = git(notes, "cat-file", "-p", f"{notes_tag}:{rel}/{entry['path']}")
            if rc != 0 or sha256_bytes(b) != entry["sha256"]:
                fails.append(f"at tag {notes_tag}: {entry['path']} missing or changed")
        for repo, tag in ((code, code_tag), (notes, notes_tag)):
            rc, out = git(repo, "ls-remote", "--tags", "origin", tag)
            if rc != 0 or not out.strip():
                fails.append(f"tag {tag} not on origin")
        mode = "TAG"
    else:
        for repo_key, branch in (("code", shas["code"]["branch"]), ("notes", shas["notes"]["branch"])):
            repo = shas[repo_key]["path"]
            rc, st = git(repo, "status", "--porcelain")
            if st.strip():
                fails.append(f"{repo_key} tree dirty")
            rc, head = git(repo, "rev-parse", "HEAD"); head = head.decode().strip()
            rc, remote = git(repo, "ls-remote", "origin", branch)
            if rc != 0 or not remote.decode().startswith(head):
                fails.append(f"{repo_key} HEAD not at origin/{branch}")
        rc, nh = git(notes, "rev-parse", "HEAD")
        mode = f"PRE-TAG (notes HEAD to tag: {nh.decode().strip()})"
    print(f"mode {mode}; verified {n} files, {len(man.get('directory_manifests', {}))} directory manifests")
    if fails:
        print("FAIL:"); [print("  " + f) for f in fails]; return 1
    print("CHECKPOINT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
