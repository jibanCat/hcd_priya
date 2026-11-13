"""Put the model files into this package before it is built. They are never committed to git.

    python model_package/prepare.py --release RELEASE_DIR --release-sha256 SHA256 [--allow-extra NAME ...]
    python model_package/prepare.py --synthetic            # a stand-in with untrained networks, for tests and CI
    python model_package/prepare.py --clean

``--release``: RELEASE_DIR is an accepted LF+MF release (the directory that holds ``release.json``) and SHA256 the
sha256 of its ``release.json``. In this order:

1. the sha256 of ``release.json`` is compared with SHA256; nothing is copied unless they are equal;
2. the release's version must be a plain public version (for example ``1.2.0``): it becomes the version of this
   package, and a version the package index would rewrite or refuse is stopped here;
3. the files are copied: regular files only, at most one directory level, no symbolic link, no hidden name. At the
   top of the release only these are taken: ``release.json``, the two directories it names, and the documentation
   files ``README.md``, ``LICENSE``, ``ATTRIBUTION.md`` and ``SHA256SUMS``. Anything else stops the script, unless it
   is named with ``--allow-extra`` (a file or a directory at the top of the release), so that no stray file reaches a
   published wheel unnoticed;
4. the copy is loaded with ``hcd_analysis.predictor.load_release``, which verifies every model file against the
   digest and refuses a directory that holds a file its manifest does not list;
5. if the release has a ``SHA256SUMS`` file, every copied file must be listed in it with its sha256, and it must
   list nothing else.

Only then are ``data/models.json`` and ``_version.py`` written. The script prints the entry to add to
``hcd_analysis/published_models.py`` so that hcd-priya accepts this model. On any failure everything it wrote is
removed. It needs ``hcd_analysis`` installed.

``--synthetic``: writes a synthetic release (untrained networks; no real number) with the test suite's builder. Run it
from a repository checkout, in an environment that has ``hcd_analysis``'s dependencies and ``pytest`` (the ``test``
extra). The package version is ``0+synthetic``: a version with a local label cannot be uploaded to the package index,
so a synthetic build cannot be published by mistake.

Then build: ``python -m build --outdir <dir> model_package`` (the wheel and the source archive). The script removes
``model_package/build`` and the ``egg-info`` directory before it writes anything and with ``--clean``: files left
there by an earlier build would otherwise be packed into the next wheel.

Adding another model (for example the cross-validation fold of the closure example) is one more entry in
``models.json``: a name, a format, a sub-directory and an anchor digest.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "src", "hcd_priya_models")
DATA = os.path.join(PKG, "data")
VERSION = os.path.join(PKG, "_version.py")
FORMAT = "hcd_priya.models/1"
SYNTHETIC_VERSION = "0+synthetic"
DOC_FILES = ("README.md", "LICENSE", "ATTRIBUTION.md", "SHA256SUMS")
# A canonical public version (PEP 440): what the package index keeps unchanged and accepts.
PUBLIC_VERSION = re.compile(
    r"(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?\Z"
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_build() -> None:
    """Remove what an earlier build left in the package directory (it would be packed into the next wheel)."""
    shutil.rmtree(os.path.join(HERE, "build"), ignore_errors=True)
    for path in glob.glob(os.path.join(HERE, "src", "*.egg-info")) + glob.glob(
        os.path.join(HERE, "*.egg-info")
    ):
        shutil.rmtree(path, ignore_errors=True)


def clean() -> None:
    shutil.rmtree(DATA, ignore_errors=True)
    if os.path.exists(VERSION):
        os.unlink(VERSION)
    clean_build()


def release_parts(src: str) -> tuple[str, list[str], str]:
    """The release's version, the names of the directories that release.json names (the backbone, the MF record
    and, in a release of format 2, the incidence MF record), and the release's format."""
    with open(os.path.join(src, "release.json"), encoding="utf-8") as fh:
        doc = json.load(fh)
    parts = [str(doc[p]["dir"]) for p in ("backbone", "mf", "incidence") if p in doc]
    return str(doc["release_version"]), parts, str(doc["format"])


def copy_release(src: str, dst: str, parts: list[str], extra: list[str]) -> list[str]:
    """Copy the release: regular files only, at most one directory level, no symlink, no hidden name; at the top
    only release.json, the part directories, the documentation files and the names in ``extra``. Returns the
    copied paths, relative to the release."""
    allowed = {"release.json", *parts, *DOC_FILES, *extra}
    stray = sorted(set(os.listdir(src)) - allowed)
    if stray:
        sys.exit(
            f"prepare: the release directory holds {stray}, which release.json does not name and which are not "
            f"the documentation files {list(DOC_FILES)}: remove them, or name each with --allow-extra"
        )
    copied = []
    for root, dirs, files in os.walk(src, followlinks=False):
        rel = os.path.relpath(root, src)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        for name in dirs + files:
            path = os.path.join(root, name)
            if os.path.islink(path) or name.startswith("."):
                sys.exit(
                    f"prepare: {path} is a symbolic link or a hidden name: refusing to copy the release"
                )
        if depth > 1 or (depth == 1 and dirs):
            sys.exit(
                f"prepare: {root} holds a directory more than one level below the release directory"
            )
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for name in files:
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                sys.exit(f"prepare: {path} is not a regular file")
            shutil.copyfile(path, os.path.join(dst, rel, name))
            copied.append(os.path.normpath(os.path.join(rel, name)).replace(os.sep, "/"))
    return sorted(copied)


def check_sums(dst: str, copied: list[str]) -> str:
    """If the release has SHA256SUMS: every copied file is listed there with its sha256, and nothing else is."""
    path = os.path.join(dst, "SHA256SUMS")
    if not os.path.exists(path):
        return "the release has no SHA256SUMS file; its documentation files are copied as they are"
    listed = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                digest, _, name = line.strip().partition("  ")
                listed[name.lstrip("*").removeprefix("./")] = digest
    want = [c for c in copied if c != "SHA256SUMS"]
    if sorted(listed) != want:
        sys.exit(
            f"prepare: SHA256SUMS and the copied files differ: only in SHA256SUMS {sorted(set(listed) - set(want))}, "
            f"only copied {sorted(set(want) - set(listed))}"
        )
    bad = [c for c in want if sha256_file(os.path.join(dst, c)) != listed[c]]
    if bad:
        sys.exit(f"prepare: these files do not have the sha256 that SHA256SUMS lists: {bad}")
    return f"SHA256SUMS lists exactly the {len(want)} other files, each with its sha256"


def write_package(version: str, synthetic: bool, models: list[dict]) -> None:
    with open(os.path.join(DATA, "models.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {"format": FORMAT, "version": version, "synthetic": synthetic, "models": models},
            fh,
            indent=1,
        )
        fh.write("\n")
    with open(VERSION, "w", encoding="utf-8") as fh:
        fh.write(
            '"""Written by prepare.py: the version of the model bundle this package holds."""\n\n'
        )
        fh.write(f"__version__ = {json.dumps(version)}\n")


def check_loaded(loaded: dict, version: str, fmt: str) -> None:
    """Stop unless the release that was loaded back from the copy is the one whose manifest was read: the same
    version and the same format. An explicit check, so that it also runs under ``python -O``."""
    for key, want in (("release_version", version), ("format", fmt)):
        got = loaded.get(key)
        if got != want:
            sys.exit(
                f"prepare: the copied release loads with {key} {got!r}, not {want!r} as its manifest says: no "
                "package entry was written"
            )


def add_release(
    src: str, anchor: str, name: str = "lfmf", extra: tuple[str, ...] = ()
) -> tuple[dict, str]:
    """Check, copy and verify one LF+MF release; return its entry of models.json and its release version."""
    got = sha256_file(os.path.join(src, "release.json"))
    if got != anchor:
        sys.exit(
            f"prepare: release.json has sha256 {got}, not the digest given ({anchor}): nothing was copied"
        )
    version, parts, fmt = release_parts(src)
    dst = os.path.join(DATA, name)
    copied = copy_release(src, dst, parts, list(extra))
    try:
        import hcd_analysis.predictor as hp
    except ImportError as exc:
        sys.exit(
            f"prepare: hcd_analysis is needed to verify the release ({exc}); install hcd-priya first"
        )
    emu = hp.load_release(
        dst, anchor
    )  # verifies every model file; refuses a file its manifests do not list
    check_loaded(emu.provenance["release"], version, fmt)
    sums = check_sums(dst, copied)
    print(
        f"prepare: {name}: {len(copied)} files copied and verified; {emu.n_members} members; {sums}"
    )
    entry = {"name": name, "format": fmt, "path": name, "anchor_sha256": anchor}
    return entry, version


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Put the model files into the hcd-priya-models package before a build."
    )
    ap.add_argument("--release", metavar="RELEASE_DIR", help="an accepted LF+MF release directory")
    ap.add_argument("--release-sha256", metavar="SHA256", help="the sha256 of its release.json")
    ap.add_argument(
        "--allow-extra",
        metavar="NAME",
        action="append",
        default=[],
        help="also copy this file or directory at the top of the release (repeatable)",
    )
    ap.add_argument(
        "--synthetic", action="store_true", help="a synthetic stand-in (untrained networks)"
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="remove the copied files, the version file and build leftovers",
    )
    args = ap.parse_args(argv)
    modes = [bool(args.release), args.synthetic, args.clean]
    if sum(modes) != 1 or (args.release and not args.release_sha256):
        ap.error("give --release DIR --release-sha256 SHA256, or --synthetic, or --clean")
    if args.clean:
        clean()
        print("prepare: cleaned")
        return 0
    if os.path.exists(DATA) or os.path.exists(VERSION):
        sys.exit(f"prepare: {DATA} or {VERSION} exists: run with --clean first")
    clean_build()
    os.makedirs(DATA)
    try:
        if args.synthetic:
            import tempfile

            sys.path.insert(
                0, os.path.dirname(HERE)
            )  # the repository root: the test suite's builder
            try:
                from tests.synthetic_release import make_release
            except ImportError as exc:
                sys.exit(
                    f"prepare: --synthetic needs a repository checkout and the test extra (pytest): {exc}"
                )
            with tempfile.TemporaryDirectory() as tmp:
                rel = make_release(os.path.join(tmp, "release"))
                entry, _version = add_release(str(rel["root"]), rel["sha"])
            write_package(SYNTHETIC_VERSION, True, [entry])
            print(
                f"prepare: synthetic package ready, version {SYNTHETIC_VERSION} (untrained networks; never published)"
            )
        else:
            src = os.path.abspath(args.release)
            version, _parts, _fmt = release_parts(src)
            if not PUBLIC_VERSION.match(version):
                sys.exit(
                    f"prepare: the release's version {version!r} is not a plain public version (for example "
                    "1.2.0): it cannot be the version of a published package. Nothing was copied."
                )
            entry, version = add_release(src, args.release_sha256, extra=tuple(args.allow_extra))
            write_package(version, False, [entry])
            pin = {"name": entry["name"], "version": version, "format": entry["format"]}
            pin["anchor_sha256"] = entry["anchor_sha256"]
            print(f"prepare: package ready, version {version}")
            print("prepare: the entry for hcd_analysis/published_models.py (MODELS):")
            print("    " + json.dumps(pin))
    except BaseException:
        clean()
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
