"""Print the requirements of a wheel, one per line, optionally with one extra and without some distributions.

    python tools/ci/wheel_requires.py dist/hcd_priya-*.whl --extra test --without hcd-priya-models > requirements.txt

Why: hcd-priya requires hcd-priya-models, and the synthetic model package used in CI is built with hcd_analysis
itself. So the environment first gets every other requirement of the wheel (this list), then the model package is
built, then both wheels are installed. Standard library only; it reads the wheel's METADATA and installs nothing.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from email.parser import Parser


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirements(wheel: str, extra: str | None, without: set[str]) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        (meta,) = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        parsed = Parser().parsestr(zf.read(meta).decode())
        requires, own = parsed.get_all("Requires-Dist") or [], norm(parsed.get("Name", ""))
    out = []
    for req in requires:
        spec, _, marker = req.partition(";")
        name = norm(re.split(r"[\s<>=!~\[(]", spec.strip(), maxsplit=1)[0])
        if name in without or name == own:  # own: an extra that refers to the package itself
            continue
        m = re.search(r"""extra\s*==\s*['"]([^'"]+)['"]""", marker)
        if m and (extra is None or norm(m.group(1)) != norm(extra)):
            continue
        if m and marker.strip() != m.group(0).strip():
            sys.exit(
                f"wheel_requires: a marker with an extra and another condition is not handled: {req!r}"
            )
        out.append(spec.strip() if m else req.strip())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Print a wheel's requirements, one per line.")
    ap.add_argument("wheel")
    ap.add_argument("--extra", help="also the requirements of this extra")
    ap.add_argument(
        "--without", action="append", default=[], help="leave this distribution out (repeatable)"
    )
    args = ap.parse_args(argv)
    if args.extra and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.extra):
        ap.error("--extra takes the name of one extra")
    for line in requirements(args.wheel, args.extra, {norm(w) for w in args.without}):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
