"""
Restore unchanged figure re-renders to avoid git blob bloat.

Matplotlib writes PNGs with small run-to-run byte variation
(creation-timestamp metadata, minor raster jitter).  Over many
regeneration cycles this would pollute the repo with effectively
identical blobs — the fix is to detect when a re-render's *pixel
content* is the same as the committed version and, if so, restore
the committed bytes so git sees no change.

Usage
-----
    python3 tools/freshen_figures.py           # dry-run report only
    python3 tools/freshen_figures.py --apply   # `git checkout` unchanged PNGs

Algorithm
---------
For each PNG in `git status --porcelain` that shows as modified:
  1. Decode both the working-tree and HEAD version with Pillow.
  2. Compare the RGBA pixel arrays exactly (.tobytes()).
  3. If identical → restore the committed bytes via `git checkout`.
  4. If not → report as "real" change; user inspects manually.

Only modifies the working tree; never rewrites history.  Safe to run
any time before `git add`.

Dependencies: `Pillow` (already a matplotlib dependency).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git"] + args, cwd=REPO, text=True)


def _git_status_modified_pngs() -> list[Path]:
    """Return paths of tracked PNGs whose working-tree bytes differ from HEAD."""
    out = _git(["status", "--porcelain", "--", "*.png"])
    paths = []
    for line in out.splitlines():
        if not line:
            continue
        # Porcelain format: XY  path   (X = index, Y = working tree)
        if len(line) < 3:
            continue
        status = line[:2]
        path = line[3:].strip().strip('"')
        # Only care about modified (not untracked/deleted)
        if "M" in status and path.endswith(".png"):
            paths.append(REPO / path)
    return paths


def _pixel_bytes(path: Path) -> bytes:
    """Return an RGBA bytes representation of the PNG's pixel content.
    Ignores PNG text chunks (creation time, software version, etc.).
    """
    with Image.open(path) as im:
        im = im.convert("RGBA")
        return im.tobytes() + bytes(im.size)  # include size for strictness


def _head_pixel_bytes(rel_path: Path) -> bytes | None:
    """Pixel bytes of the version at HEAD; None if not in HEAD."""
    try:
        data = subprocess.check_output(
            ["git", "show", f"HEAD:{rel_path.as_posix()}"],
            cwd=REPO,
        )
    except subprocess.CalledProcessError:
        return None
    from io import BytesIO
    with Image.open(BytesIO(data)) as im:
        im = im.convert("RGBA")
        return im.tobytes() + bytes(im.size)


def check(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """
    Split modified PNGs into (bytewise-identical-pixels, real-changes).
    """
    identical, real = [], []
    for p in paths:
        rel = p.relative_to(REPO)
        head = _head_pixel_bytes(rel)
        if head is None:
            real.append(p)  # new file or renamed — treat as real change
            continue
        try:
            wt = _pixel_bytes(p)
        except Exception as exc:
            print(f"  skip (unreadable): {rel}  ({exc})")
            real.append(p)
            continue
        if wt == head:
            identical.append(p)
        else:
            real.append(p)
    return identical, real


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply", action="store_true",
        help="git-checkout the unchanged-pixel files; without this flag, "
             "only report what would be done.",
    )
    args = ap.parse_args()

    pngs = _git_status_modified_pngs()
    if not pngs:
        print("No modified tracked PNGs.  Nothing to do.")
        return 0

    print(f"Inspecting {len(pngs)} modified PNG(s)…")
    identical, real = check(pngs)

    print()
    print(f"Bytewise-identical pixel content vs HEAD:  {len(identical)}")
    for p in identical:
        print(f"  {p.relative_to(REPO)}")

    print()
    print(f"Real content changes (keep for review):    {len(real)}")
    for p in real:
        print(f"  {p.relative_to(REPO)}")

    if args.apply and identical:
        print()
        print(f"Restoring {len(identical)} identical file(s) with `git checkout`…")
        rel_paths = [str(p.relative_to(REPO)) for p in identical]
        subprocess.check_call(["git", "checkout", "--", *rel_paths], cwd=REPO)
        print("  done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
