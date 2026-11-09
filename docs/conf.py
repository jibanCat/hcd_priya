"""Sphinx configuration of the hcd_priya documentation site (built by Read the Docs from .readthedocs.yaml).

Build locally, from the repository root, with the pinned documentation requirements installed:

    python -m sphinx -W --keep-going -b html docs docs/_build/html

The site holds only the pages listed in ``include_patterns``; the other files under docs/ are research records and
are not published on the site. The API reference is parsed statically by sphinx-autoapi (the package is never
imported, so the build needs neither jax nor any other dependency of the package).

Pages that include a file from the repository root (README.md, CONTRIBUTING.md, CHANGELOG.md, CODE_OF_CONDUCT.md)
get their links rewritten by ``rewrite_included_links``: a link to a file that is also a page of the site points to
that page, and a link to any other repository file points to that file on GitHub (an image to its raw bytes, so it
displays). A link to a file that does not exist stops the build.
"""
from __future__ import annotations

import os
import re

DOCS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(DOCS)
REPO_URL = "https://github.com/jibanCat/hcd_priya"


def _version():
    """The package version, read from hcd_analysis/__init__.py without importing the package."""
    with open(os.path.join(REPO, "hcd_analysis", "__init__.py"), encoding="utf-8") as f:
        found = re.findall(r'^__version__ = "([^"]+)"$', f.read(), flags=re.M)
    if len(found) != 1:
        raise RuntimeError("hcd_analysis/__init__.py must define __version__ exactly once")
    return found[0]


project = "hcd_priya"
author = "Ming-Feng Ho"
copyright = "Ming-Feng Ho"
release = version = _version()

extensions = ["myst_parser", "autoapi.extension", "sphinxcontrib.mermaid"]
source_suffix = {".md": "markdown"}
myst_fence_as_directive = ["mermaid"]  # a ```mermaid block is drawn as a diagram, as GitHub draws it
root_doc = "index"
include_patterns = [
    "index.md",
    "install.md",
    "data.md",
    "compatibility.md",
    "release_format_h5.md",
    "tutorials/production_and_research_bundles.md",
    "api.md",
    "contributing.md",
    "code_of_conduct.md",
    "changelog.md",
    "release_checklist.md",
]
exclude_patterns = ["_build"]

myst_heading_anchors = 3

autoapi_type = "python"
autoapi_dirs = [os.path.join(REPO, "hcd_analysis")]
autoapi_generate_api_docs = False      # only the modules named in api.md are documented
autoapi_add_toctree_entry = False
autoapi_member_order = "bysource"
autoapi_options = ["members", "show-inheritance", "show-module-summary"]
autosummary_generate = False           # autoapi loads autosummary; the site generates no autosummary pages
# autoapi parses every module, also those not on the site, and its static parser cannot follow a few names bound by
# tuple unpacking (``A, B = ...``) that other modules import. Only that warning category is silenced.
suppress_warnings = ["autoapi.python_import_resolution"]

html_theme = "sphinx_rtd_theme"
html_title = f"hcd_priya {release}"
html_show_sourcelink = False

# Repository files that are also pages of the site: {path from the repository root: page name}.
SITE_PAGES = {
    "README.md": "index",
    "CONTRIBUTING.md": "contributing",
    "CODE_OF_CONDUCT.md": "code_of_conduct",
    "CHANGELOG.md": "changelog",
    "docs/compatibility.md": "compatibility",
    "docs/release_format_h5.md": "release_format_h5",
    "docs/release_checklist.md": "release_checklist",
    "docs/tutorials/production_and_research_bundles.md": "tutorials/production_and_research_bundles",
}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")
_LOCAL_LINK = re.compile(r"\]\((?!https?://|mailto:|#)([^)\s#]+)(#[^)\s]*)?\)")


def _git_ref():
    """The commit Read the Docs builds (so a repository link shows the documented version), else main."""
    return os.environ.get("READTHEDOCS_GIT_COMMIT_HASH") or "main"


def rewrite_included_links(app, relative_path, parent_docname, content):
    """Rewrite the local links of a repository-root file included into a page (the Sphinx ``include-read`` event)."""
    path = os.path.normpath(os.path.join(app.srcdir, str(relative_path)))
    if os.path.dirname(path) != REPO:
        return
    depth = parent_docname.count("/")

    def repl(m):
        target, anchor = m.group(1), m.group(2) or ""
        rel = os.path.normpath(target)
        if not os.path.exists(os.path.join(REPO, rel)):
            raise RuntimeError(f"{os.path.basename(path)} links to {target}, which is not in the repository")
        if rel in SITE_PAGES:
            return f"]({'../' * depth}{SITE_PAGES[rel]}.md{anchor})"
        kind = "raw" if rel.lower().endswith(IMAGE_SUFFIXES) else "blob"
        return f"]({REPO_URL}/{kind}/{_git_ref()}/{rel}{anchor})"

    content[0] = _LOCAL_LINK.sub(repl, content[0])


def setup(app):
    app.connect("include-read", rewrite_included_links)
