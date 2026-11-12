"""Check that a built sdist and wheel contain exactly the committed package: the code and the two package-data files
of the checkout's HEAD, byte for byte, plus the license and the build's own metadata, and nothing else (no cache,
checkpoint, test, script, linked, uncommitted or stale file).

usage: python tools/ci/check_dist.py DIST_DIR

DIST_DIR must hold exactly one wheel and one sdist (the output of `python -m build`), and this script must run from
a git checkout: the expected files are the .py files under hcd_analysis/ and the package-data files in the HEAD
commit, each compared by its git blob id. Exit status 1 with the offending names on any violation.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
import subprocess
import sys
import tarfile
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
PACKAGES = ("hcd_analysis",)
DIST = (
    "hcd_priya"  # the distribution name hcd-priya as wheel, sdist and metadata directories spell it
)
PACKAGE_DATA = (
    "hcd_analysis/comparison/phase0_v1.json",
    "hcd_analysis/emulator/hcd_lit_dndx_corrected.json",
)
DIST_INFO_FILES = {
    "METADATA",
    "WHEEL",
    "RECORD",
    "entry_points.txt",
    "top_level.txt",
    "licenses/LICENSE",
}
SDIST_COMMITTED = ("pyproject.toml", "LICENSE", "README.md", "MANIFEST.in")
SDIST_GENERATED = {"PKG-INFO", "setup.cfg"}
EGG_INFO_FILES = {
    "PKG-INFO",
    "SOURCES.txt",
    "dependency_links.txt",
    "entry_points.txt",
    "requires.txt",
    "top_level.txt",
}


def git_blob_id(data, oid_length=40):
    """The git object id of a blob holding `data`: sha1, or sha256 in a sha256 repository (64-character ids)."""
    h = hashlib.new("sha256" if oid_length == 64 else "sha1")
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def committed_files(repo=REPO):
    """{path: blob id} at HEAD for the package files (every .py under PACKAGES, and PACKAGE_DATA) and for the
    committed files the sdist carries (SDIST_COMMITTED). A tracked symlink among them is refused."""
    out = subprocess.run(
        ["git", "-C", repo, "ls-tree", "-r", "-z", "HEAD", "--", *PACKAGES, *SDIST_COMMITTED],
        capture_output=True,
        check=True,
    ).stdout.decode()
    package, top = {}, {}
    for rec in filter(None, out.split("\0")):
        meta, path = rec.split("\t", 1)
        mode, _, oid = meta.split()
        if mode == "120000":
            raise SystemExit(f"tracked symlink in the package: {path}")
        if path in SDIST_COMMITTED:
            top[path] = oid
        elif path.endswith(".py") or path in PACKAGE_DATA:
            package[path] = oid
    missing = set(PACKAGE_DATA) - set(package)
    if missing:
        raise SystemExit(f"package data not committed: {sorted(missing)}")
    return package, top


def _compare(prefix, found, expected):
    """Problems for a {path: bytes} found in a distribution against {path: blob id} expected."""
    problems = [f"{prefix}: unexpected file {p}" for p in sorted(set(found) - set(expected))]
    problems += [f"{prefix}: missing {p}" for p in sorted(set(expected) - set(found))]
    problems += [
        f"{prefix}: {p} differs from HEAD"
        for p in sorted(set(found) & set(expected))
        if git_blob_id(found[p], len(expected[p])) != expected[p]
    ]
    return problems


def check_wheel(path, package):
    problems, found, top = [], {}, None
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            m = re.fullmatch(DIST + r"-[^/]+\.dist-info/(.+)", name)
            if m:
                if m.group(1) not in DIST_INFO_FILES:
                    problems.append(f"wheel: unexpected metadata file {name}")
                if m.group(1) == "top_level.txt":
                    top = set(zf.read(name).decode().split())
            else:
                found[name] = zf.read(name)
    problems += _compare("wheel", found, package)
    if top != set(PACKAGES):
        problems.append(
            f"wheel: top_level.txt names {sorted(top or ())}, expected {sorted(PACKAGES)}"
        )
    return problems


def check_sdist(path, package, top):
    problems, found, found_top = [], {}, {}
    with tarfile.open(path) as tf:
        base = None
        for m in tf.getmembers():
            parts = m.name.split("/")
            base = base or parts[0]
            if m.issym() or m.islnk():
                problems.append(f"sdist: link {m.name}")
                continue
            if parts[0] != base or ".." in parts or m.name.startswith("/"):
                problems.append(f"sdist: path outside {base}/: {m.name}")
                continue
            if m.isdir():
                continue
            rel = "/".join(parts[1:])
            if rel in SDIST_GENERATED or (
                parts[1] == DIST + ".egg-info" and "/".join(parts[2:]) in EGG_INFO_FILES
            ):
                continue
            data = tf.extractfile(m).read()
            (found_top if rel in SDIST_COMMITTED else found)[rel] = data
    problems += _compare("sdist", found, package)
    problems += _compare("sdist", found_top, top)
    return problems


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.exit(__doc__)
    wheels = glob.glob(os.path.join(argv[0], "*.whl"))
    sdists = glob.glob(os.path.join(argv[0], "*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        sys.exit(f"expected one wheel and one sdist in {argv[0]}, found {wheels + sdists}")
    package, top = committed_files()
    problems = check_wheel(wheels[0], package) + check_sdist(sdists[0], package, top)
    for p in problems:
        print(p)
    head = subprocess.run(
        ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print(
        f"{os.path.basename(wheels[0])}, {os.path.basename(sdists[0])} against HEAD {head}: {len(package)} package "
        f"files ({len(PACKAGE_DATA)} package data), compared by git blob id; {len(problems)} problems"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
