"""Scan a built sdist and wheel for content that must not be distributed. check_dist.py proves that the archives hold
exactly the committed package; a distribution can match the repository exactly and still carry private information,
so this scan reads what the archives actually contain.

usage: python tools/ci/check_dist_content.py DIST_DIR --release
       python tools/ci/check_dist_content.py DIST_DIR --known FILE

DIST_DIR must hold exactly one wheel and one sdist (the output of `python -m build`). The scan reads:
- every member of both archives as text (UTF-8, undecodable bytes replaced), line by line, including every copy of a
  duplicated zip member;
- the name of every member, directories and links included, and every link target;
- the zip archive and member comments;
- the owner fields of the sdist's tar headers.

It checks them against RULES (private filesystem paths, pointers into private notes, batch-system account
directives, credentials, confidentiality markers) and finds e-mail addresses other than the declared authors' (read
from the header of the archive's own top-level metadata, never from a member deeper in the archive). Internal codes
such as ledger ids or spec names are not findings. A tar header that names an account other than a generic build
account is a finding (archive-owner).

A finding names the archive, the member, the line number, the rule and the sha256 of the stripped line. The matched
text is never printed: a member name or tar owner that matches a rule is shown only as its sha256. So a public CI log
does not republish it.

--release   The release gate: any finding fails (exit 1). docs/release_checklist.md requires it to pass.
--known     The CI mode, against the known findings still to be cleaned. FILE holds tab-separated
            `rule  member  sha256` lines and `#` comments, never the text itself. One line stands for one occurrence.
            Each archive must hold exactly as many occurrences of each (rule, member, digest) as FILE lists: one more
            is a new finding, one fewer is a stale entry. Both fail (exit 1). tests/test_dist_content.py pins the
            entries FILE may hold, so the list cannot grow without a reviewed change to that test. It must be empty
            before a release.

Limits: a regular-expression scan cannot recognise an unpublished number or result, and it reads text line by line,
so a phrase wrapped over two lines, or text inside a compressed or UTF-16 member, is not seen. Section 4 of the
release checklist (a human read of the content) still applies; check_dist.py allows only Python files and two JSON
package-data files. Exit status 2 on a usage error or an unreadable archive.
"""

from __future__ import annotations

import argparse
import collections
import functools
import glob
import hashlib
import os
import re
import sys
import tarfile
import zipfile
import zlib

# (rule id, what it finds, pattern). Literal account or user names never appear here: the path rules find them.
_ROOTS = (
    r"home|nfs|scratch|pscratch|gpfs|lustre|turbo|global|project|projects|work|Users|Volumes|mnt"
)
_SECRET_NAME = r"\w*(?:token|secret|passw(?:or)?d|api[_-]?key|access[_-]?key)\w*"
RULES = (
    (
        "private-path",
        "an absolute path under a home, cluster, scratch or mounted filesystem",
        re.compile(rf"(?<![\w.~-])/(?:{_ROOTS})/[\w.~-]"),
    ),
    (
        "private-path",
        "such a filesystem root as a quoted string, from which a path is assembled",
        re.compile(rf"[\"']/(?:{_ROOTS})/?[\"']"),
    ),
    ("private-path", "a session scratch directory", re.compile(r"/tmp/claude-")),
    ("private-path", "a conda environment path", re.compile(r"(?<![\w-])\.conda[/\\]envs[/\\]")),
    ("private-path", "a Windows user directory", re.compile(r"\b[A-Za-z]:\\+(?:Users|home)\\")),
    (
        "internal-pointer",
        "an agent worktree path (private note names are matched by digest, POINTER_DIGESTS)",
        re.compile(r"\.claude[/\\]"),
    ),
    (
        "cluster-directive",
        "a batch-system account or mail directive, or a submission with an account",
        re.compile(
            r"#SBATCH\s+(?:--account\b|--mail-user\b|-A\s?\S)|#PBS\s+(?:-A|-M)\b|#BSUB\s+(?:-P|-u)\b"
            r"|#\$\s+(?:-P|-M)\b|\b(?:sbatch|salloc|srun|qsub|bsub)\b[^\n]*(?:--account\b|\s-A\s?\S)"
        ),
    ),
    (
        "credential",
        "a private key, an access token, a key-like assigned secret",
        re.compile(
            r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"
            r"|\bgh[pousr]_[A-Za-z0-9]{36}\b|\bgithub_pat_\w{22,}|\bglpat-[\w-]{20,}|\bhf_[A-Za-z0-9]{30,}"
            r"|\bAIza[\w-]{35}\b|\bsk-(?:ant-)?[\w-]{20,}|\bAKIA[0-9A-Z]{16}\b|\bxox[abprs]-[A-Za-z0-9-]{10,}"
            rf"|(?i:\b{_SECRET_NAME}\b[\"']?\s*[:=]\s*[\"'][^\"'\s]{{8,}}[\"'])"
            rf"|(?i:\b{_SECRET_NAME}\b\s*[:=]\s*(?=[A-Za-z0-9+/_-]{{16,}}\b)(?=[^\s]*\d)(?=[^\s]*[A-Za-z])"
            r"[A-Za-z0-9+/_-]{16,}\b)"
            r"|://[^/\s:@'\"]+:[^/\s@'\"]+@"
        ),
    ),
    (
        "confidential-marker",
        "a confidentiality or embargo marker",
        re.compile(
            r"(?i)\b(?:confidential(?:ity)?|do not (?:distribute|circulate|cite)"
            r"|not for (?:public )?distribution|under embargo|embargoed)\b"
        ),
    ),
)
EMAIL = re.compile(
    r"(?<![\w.%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"
)
# Addresses that identify no one: the reserved example domains (RFC 2606) and an ssh git host (git@host:owner/repo).
EMAIL_EXEMPT = re.compile(
    r"@(?:[\w-]+\.)*example\.(?:com|org|net|edu)$|\.(?:example|invalid|test|localhost)$"
    r"|^git@[\w.-]+$"
)
GENERIC_OWNERS = {
    "",
    "root",
    "runner",
    "docker",
    "nobody",
}  # the build accounts that identify no one
AUTHOR_FIELDS = ("Author-email:", "Maintainer-email:")
RULE_IDS = tuple(dict.fromkeys(r[0] for r in RULES)) + ("email", "archive-owner")

# The names of the private notes repository, its folders and a marker of withheld material are matched by the sha256
# of a word, so this public file never spells them out. A word is a run of letters, digits, '_' and '-'; it matches when
# one, two or three of its consecutive '_'- or '-'-separated parts, joined by '_', have a listed digest:
# - POINTER_DIGESTS: compared in lower case, so case and '-' or '_' spellings do not matter;
# - EXACT_POINTER_DIGESTS: compared as written (an upper-case marker, not the ordinary word in lower case).
# tests/test_dist_content.py pins both sets; a private check outside the repository confirms which names they are.
POINTER_DIGESTS = frozenset(
    {
        "ba717dd9d4e3e0f52e338e04534394ebc5c5f45c948814476932d4a3a36ca713",
        "b225390a8984c8de4206c746772ab5ef2abb0a02bcb043da10f6d41c274d1a02",
        "9678f4c7207f3da2d59556c00fa5ceb8a3b107d1f539ba303e3ba91320d7aed1",
        "d3a4646482ceba965bbe9bcfc9fe9fb10eb0c8df829fa20ed9baff5b247964f2",
        "4e2f58503a2a022a1d09530042b39c555c797387916626f85e4d251e31f27989",
    }
)
EXACT_POINTER_DIGESTS = frozenset(
    {"d8e283d959d759036209e868ca4b18fb0306bbe9f04c135267e01131ada566cc"}
)
_WORD = re.compile(r"[A-Za-z0-9_-]+")
_MAX_PARTS = 3


def _runs(word):
    """The words made of one to _MAX_PARTS consecutive parts of `word`, joined by '_'."""
    parts = [x for x in re.split(r"[_-]", word) if x]
    return {
        "_".join(parts[i:j])
        for i in range(len(parts))
        for j in range(i + 1, min(i + _MAX_PARTS, len(parts)) + 1)
    }


@functools.lru_cache(maxsize=1 << 16)
def _word_is_private(word):
    return any(
        hashlib.sha256(run.lower().encode()).hexdigest() in POINTER_DIGESTS
        or hashlib.sha256(run.encode()).hexdigest() in EXACT_POINTER_DIGESTS
        for run in _runs(word)
    )


def has_private_name(line):
    return any(_word_is_private(word) for word in _WORD.findall(line))


def sha256(text):
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


line_sha256 = sha256


def _lines(text):
    """Lines as an editor numbers them: split at newline only, with a carriage return before it dropped."""
    return [ln[:-1] if ln.endswith("\r") else ln for ln in text.split("\n")]


def read_archive(path):
    """(kind, [(member name relative to the archive root, text)], [names and link targets], [comments], owners,
    top-level metadata text)."""
    texts, names, comments, owners, meta = [], [], [], set(), ""
    if path.endswith(".whl"):
        with zipfile.ZipFile(path) as zf:
            comments.append(zf.comment.decode("utf-8", "replace"))
            for info in zf.infolist():
                names.append(info.filename)
                comments.append(info.comment.decode("utf-8", "replace"))
                if info.is_dir():
                    continue
                text = zf.open(info).read().decode("utf-8", "replace")
                texts.append((info.filename, text))
                if re.fullmatch(r"[^/]+\.dist-info/METADATA", info.filename):
                    meta = text
        return "wheel", texts, names, comments, owners, meta
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            rel = m.name.split("/", 1)[1] if "/" in m.name else ""
            names.append(m.name)
            if m.issym() or m.islnk():
                names.append(m.linkname)
            owners.add((m.uname or "", m.gname or ""))
            if not m.isfile():
                continue
            text = tf.extractfile(m).read().decode("utf-8", "replace")
            texts.append((rel, text))
            if rel == "PKG-INFO":
                meta = text
    return "sdist", texts, names, comments, owners, meta


def declared_emails(metadata):
    """The author and maintainer addresses in the header block (up to the first blank line) of the top-level
    metadata; the README embedded after it does not count."""
    found = set()
    for line in _lines(metadata):
        if not line.strip():
            break
        if line.startswith(AUTHOR_FIELDS):
            found |= {m.group(0).lower() for m in EMAIL.finditer(line)}
    return found


def line_rules(line, allowed_emails=frozenset()):
    rules = [rid for rid, _, rx in RULES if rx.search(line)]
    if has_private_name(line):
        rules.append("internal-pointer")
    if any(
        m.group(0).lower() not in allowed_emails and not EMAIL_EXEMPT.search(m.group(0).lower())
        for m in EMAIL.finditer(line)
    ):
        rules.append("email")
    return list(dict.fromkeys(rules))


def scan_text(text, allowed_emails=frozenset()):
    """[(line number, rule id, line sha256)] for one text."""
    return [
        (no, rid, sha256(line))
        for no, line in enumerate(_lines(text), 1)
        for rid in line_rules(line, allowed_emails)
    ]


def shown(name):
    """A member name as printed and listed: the name itself, or its digest when the name matches a rule."""
    return f"name-sha256:{sha256(name)}" if line_rules(name) else name


def scan_archive(path):
    """[(kind, member as shown, line number, rule id, digest)]; line 0 is the member's name, a link target or a
    comment; the member '(tar headers)' carries the owner findings."""
    kind, texts, names, comments, owners, meta = read_archive(path)
    allowed = declared_emails(meta)
    findings = []
    for name in names:
        findings += [(kind, shown(name), 0, rid, sha256(name)) for rid in line_rules(name, allowed)]
    for c in comments:
        findings += [(kind, "(zip comment)", no, rid, h) for no, rid, h in scan_text(c, allowed)]
    for uname, gname in sorted(owners):
        if uname not in GENERIC_OWNERS or gname not in GENERIC_OWNERS:
            findings.append((kind, "(tar headers)", 0, "archive-owner", sha256(f"{uname}:{gname}")))
    for name, text in texts:
        findings += [(kind, shown(name), no, rid, h) for no, rid, h in scan_text(text, allowed)]
    return findings


def read_known(path):
    """{(rule, member, digest): number of listed occurrences}."""
    known = collections.Counter()
    with open(path, encoding="utf-8") as f:
        for no, line in enumerate(f, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if (
                len(parts) != 3
                or parts[0] not in RULE_IDS
                or not re.fullmatch(r"[0-9a-f]{64}", parts[2])
            ):
                raise ValueError(
                    f"{path}:{no}: expected 'rule<TAB>member<TAB>sha256' with a known rule"
                )
            known[tuple(parts)] += 1
    return known


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dist_dir")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--release", action="store_true", help="any finding fails")
    mode.add_argument(
        "--known", metavar="FILE", help="exactly the occurrences listed in FILE do not fail"
    )
    args = ap.parse_args(argv)
    wheels = glob.glob(os.path.join(args.dist_dir, "*.whl"))
    sdists = glob.glob(os.path.join(args.dist_dir, "*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        print(
            f"expected one wheel and one sdist in {args.dist_dir}, found {sorted(wheels + sdists)}"
        )
        return 2
    try:
        known = collections.Counter() if args.release else read_known(args.known)
        per_archive = [scan_archive(wheels[0]), scan_archive(sdists[0])]
    except (OSError, ValueError, EOFError, zlib.error, zipfile.BadZipFile, tarfile.TarError) as e:
        print(f"cannot scan: {type(e).__name__}: {e}")
        return 2
    new = stale = total = 0
    for kind, findings in zip(("wheel", "sdist"), per_archive):
        counts = collections.Counter()
        for _, name, no, rid, h in findings:
            key = (rid, name, h)
            counts[key] += 1
            status = "known" if counts[key] <= known[key] else "NEW"
            new += status == "NEW"
            total += 1
            print(f"{kind}: {name}:{no}: {rid} [sha256 {h}] {status}")
        for key in sorted(known):
            if counts[key] < known[key]:
                stale += 1
                print(
                    f"{kind}: listed {known[key]} time(s) in {args.known} but found {counts[key]} "
                    f"(remove the entry): {key[0]}\t{key[1]}\t{key[2]}"
                )
    mode_name = "release gate" if args.release else f"CI mode, known list {args.known}"
    print(
        f"{os.path.basename(wheels[0])}, {os.path.basename(sdists[0])}: {total} findings ({new} not in the known "
        f"list, {stale} stale list entries) [{mode_name}]"
    )
    return 1 if new or stale else 0


if __name__ == "__main__":
    sys.exit(main())
