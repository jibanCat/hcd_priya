"""Repository root of THIS checkout. Package code must build every in-repo path from here, never from an absolute
literal, so that a second checkout (e.g. a git worktree) loads its own code and products only.
See docs/kgrid_incident_2026-10.md."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_STR = str(REPO_ROOT)


def repo_relative(path) -> str:
    """``path`` relative to the repository root, with '/' separators: how a provenance record names a repository file
    without this machine's absolute layout. The path is made absolute but symlinks are not followed, so a linked data
    file inside the checkout keeps its in-repository name. A path outside the repository is refused (ValueError).
    """
    rel = os.path.relpath(os.path.abspath(os.fspath(path)), REPO_ROOT_STR)
    if rel == os.pardir or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel):
        raise ValueError(
            f"{path} is not inside the repository; a provenance record names repository files only"
        )
    return Path(rel).as_posix()
