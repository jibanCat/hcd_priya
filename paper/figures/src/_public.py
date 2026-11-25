"""Copy one figure of the public code repository into the paper, unchanged, after checking its sha256.

Used by the figure scripts that show a figure of the code repository. Nothing is redrawn: the
file is copied byte for byte, so the paper shows exactly the image of the public tree.

Where the file is read from:
  - a local clone, if HCD_PRIYA_REPO points to one (`git show COMMIT:PATH`; this only reads the clone);
  - otherwise GitHub, at the same commit (cached under .build/cache/, like the code facts).
If neither works and the image is already in the paper folder, the script keeps it and says so ("KEPT").
Nothing is ever written to the code repository.
"""

import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPO = "jibanCat/hcd_priya"


def fetch(commit, path, repo=REPO):
    local = os.environ.get("HCD_PRIYA_REPO") if repo == REPO else None
    if local:
        r = subprocess.run(["git", "-C", local, "show", f"{commit}:{path}"], capture_output=True)
        if r.returncode == 0:
            return r.stdout
    cached = ROOT / ".build" / "cache" / commit / path
    if cached.exists():
        return cached.read_bytes()
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "paperkit"})
    try:
        data = urllib.request.urlopen(req, timeout=30).read()
    except Exception:  # noqa: BLE001
        return None
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)
    return data


def copy(commit, path, sha256, out):
    out = Path(out)
    data = fetch(commit, path)
    if data is None:
        if out.exists():
            print("KEPT", out, f"(cannot read {path} at {commit[:7]})")
            return 0
        print(f"cannot read {path} at {commit[:7]}, and there is no image to keep", file=sys.stderr)
        return 1
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        print(f"{path} at {commit[:7]} has sha256 {got}, expected {sha256}: not copied", file=sys.stderr)
        return 1
    if not out.exists() or out.read_bytes() != data:
        out.write_bytes(data)
    print("copied", path, "at", commit[:7], "->", out.relative_to(ROOT))
    return 0
