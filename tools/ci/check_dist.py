"""Check that a built sdist and wheel contain exactly the package: its code, its two package-data files and the
license, and nothing else (no cache, checkpoint, test, script or linked file).

usage: python tools/ci/check_dist.py DIST_DIR

DIST_DIR must hold exactly one wheel and one sdist (the output of `python -m build`). Every module under the
checkout's hcd_analysis/ and cli/ must be in both. Exit status 1 with the offending names on any violation.
"""
from __future__ import annotations

import glob
import os
import re
import sys
import tarfile
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
PACKAGES = ("hcd_analysis", "cli")
PACKAGE_DATA = ("hcd_analysis/comparison/phase0_v1.json", "hcd_analysis/emulator/hcd_lit_dndx_corrected.json")
DIST_INFO_FILES = {"METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt", "licenses/LICENSE"}
SDIST_TOP_FILES = {"PKG-INFO", "pyproject.toml", "setup.cfg", "LICENSE", "README.md", "MANIFEST.in"}
EGG_INFO_FILES = {"PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt", "requires.txt",
                  "top_level.txt"}


def source_modules(repo=REPO):
    """Every .py file of the packages in the checkout, as a posix path relative to the checkout."""
    out = set()
    for pkg in PACKAGES:
        for dp, dns, fs in os.walk(os.path.join(repo, pkg)):
            dns[:] = [d for d in dns if d != "__pycache__" and os.path.isfile(os.path.join(dp, d, "__init__.py"))]
            out |= {os.path.relpath(os.path.join(dp, f), repo).replace(os.sep, "/") for f in fs if f.endswith(".py")}
    return out


def is_package_file(name):
    return name.split("/")[0] in PACKAGES and (name.endswith(".py") or name in PACKAGE_DATA)


def check_wheel(path, modules):
    problems = []
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        top = None
        for name in sorted(names):
            m = re.fullmatch(r"hcd_analysis-[^/]+\.dist-info/(.+)", name)
            if m:
                if m.group(1) not in DIST_INFO_FILES:
                    problems.append(f"wheel: unexpected metadata file {name}")
                if m.group(1) == "top_level.txt":
                    top = set(zf.read(name).decode().split())
            elif not is_package_file(name):
                problems.append(f"wheel: unexpected file {name}")
    for name in sorted((modules | set(PACKAGE_DATA)) - names):
        problems.append(f"wheel: missing {name}")
    if top is not None and top != set(PACKAGES):
        problems.append(f"wheel: top_level.txt names {sorted(top)}, expected {sorted(PACKAGES)}")
    return problems


def check_sdist(path, modules):
    problems = []
    with tarfile.open(path) as tf:
        members = tf.getmembers()
    base = None
    names = set()
    for m in members:
        if m.issym() or m.islnk():
            problems.append(f"sdist: link {m.name}")
        parts = m.name.split("/")
        if base is None:
            base = parts[0]
        if parts[0] != base or ".." in parts or m.name.startswith("/"):
            problems.append(f"sdist: path outside {base}/: {m.name}")
            continue
        if m.isdir():
            continue
        rel = "/".join(parts[1:])
        names.add(rel)
        if rel in SDIST_TOP_FILES or is_package_file(rel):
            continue
        if parts[1] == "hcd_analysis.egg-info" and "/".join(parts[2:]) in EGG_INFO_FILES:
            continue
        problems.append(f"sdist: unexpected file {m.name}")
    for name in sorted((modules | set(PACKAGE_DATA) | {"pyproject.toml", "LICENSE"}) - names):
        problems.append(f"sdist: missing {name}")
    return problems


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.exit(__doc__)
    wheels = glob.glob(os.path.join(argv[0], "*.whl"))
    sdists = glob.glob(os.path.join(argv[0], "*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        sys.exit(f"expected one wheel and one sdist in {argv[0]}, found {wheels + sdists}")
    modules = source_modules()
    problems = check_wheel(wheels[0], modules) + check_sdist(sdists[0], modules)
    for p in problems:
        print(p)
    print(f"{os.path.basename(wheels[0])}, {os.path.basename(sdists[0])}: {len(modules)} modules, "
          f"{len(PACKAGE_DATA)} package-data files; {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
