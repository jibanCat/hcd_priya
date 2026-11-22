"""Documentation lint: links, names of removed files, and a few sentences that were corrected.

This file establishes no scientific fact. It keeps corrected text from coming back: a broken link, a pointer to a
document that was removed, an instruction that loads the published model in a superseded way, a sentence that says
the model is not published. The scientific definitions are tested against the code in
tests/test_science_definitions.py. Needs the repository: it reads the tracked text files. No network.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

from hcd_analysis.emulator import mf_release as MR

ROOT = pathlib.Path(__file__).resolve().parents[1]
THIS_FILE = "tests/test_docs_integrity.py"
TEXT_SUFFIXES = (".md", ".py", ".toml", ".yaml", ".yml", ".txt", ".cfg", ".in", ".cff", ".sh")
# a generator of a record, not documentation; its text is not read here
NOT_READ = ("scripts/gen_analysis_lock.py",)

# The documents removed from the repository after the release of 1.0.0, each by the shortest name that is its own.
REMOVED = (
    "p1d_definition.md",
    "docs/analysis.md",
    "hcd_mf_analysis.md",
    "figure_guide.md",
    "clustering_definitions.md",
    "multipole_jacobian_explained.md",
    "dla_truth_unmatched_analysis.md",
    "docs/release_format.md",
    "hcd_analysis/emulator/README.md",
    "06_performance_walkthrough/README.md",
    "STUDENT_PROJECTS.md",
    "_build_notebooks.py",
)
LEGACY_NOTEBOOKS = "notebooks/tutorials"

# Files that still name a removed document, with the reason. The list is exact: a file that stops naming one, or a
# new file that names one, fails the test, so the list empties as the remaining pointers are settled.
README_WAITS = "the README waits for the project lead's own rewrite"
DOCSTRING_WAITS = "a docstring or comment of package code; package files are not changed by a documentation change"
PAGE_WAITS = "the page itself waits for the project lead's decision to keep, reduce or remove it"
PENDING = {
    "README.md": (
        ("docs/analysis.md", "hcd_analysis/emulator/README.md", "p1d_definition.md"),
        README_WAITS,
    ),
    "docs/bugs_found.md": (("p1d_definition.md",), PAGE_WAITS),
    "docs/dla_truth_validation.md": (
        ("dla_truth_unmatched_analysis.md", "docs/analysis.md"),
        PAGE_WAITS,
    ),
    "docs/fast_mode_physics.md": (("docs/analysis.md",), PAGE_WAITS),
    "hcd_analysis/clustering.py": (("clustering_definitions.md",), DOCSTRING_WAITS),
    "hcd_analysis/dla_truth.py": (("dla_truth_unmatched_analysis.md",), DOCSTRING_WAITS),
    "hcd_analysis/emulator/safe_io.py": (("docs/release_format.md",), DOCSTRING_WAITS),
    "hcd_analysis/report.py": (
        ("p1d_definition.md",),
        "the catalogue pipeline's generator of that page; package code, reported to the project lead",
    ),
}
# A file that names a removed document for a reason that does not pass.
STANDING = {
    "tests/test_ci_tools.py": (
        ("hcd_analysis/emulator/README.md",),
        "a file name inside a made-up wheel, to test the content gate; it names no document",
    ),
}
PENDING_LEGACY_NOTEBOOKS = {"README.md": README_WAITS}

# Sentences that said the model or its package was not published, each with the page that held it.
NOT_PUBLISHED = (
    ("docs/compatibility.md", "<path to the hcd_priya_models wheel>"),
    ("docs/compatibility.md", "which is not on the package index yet"),
    ("docs/release_format_h5.md", "The trained models themselves are not distributed yet"),
)
# The pages of the supported interface: one convention per quantity.
PREDICTOR_PAGES = (
    "README.md",
    "docs/likelihood.md",
    "docs/api.md",
    "docs/model_figures.md",
    "docs/release_format_h5.md",
    "docs/rogers_comparison.md",
)


def _tracked_files() -> list[str]:
    """The files of the repository, as paths relative to its root.

    Returns
    -------
    list of str
        What ``git ls-files`` lists. Where the tree is not a Git checkout or Git is absent: every file under the
        root outside hidden folders and build folders.
    """
    try:
        listed = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return sorted(line for line in listed.stdout.split("\n") if line)
    except (OSError, subprocess.CalledProcessError):
        skipped = {"build", "dist", "node_modules", "__pycache__", "_build"}
        found = []
        for path in ROOT.rglob("*"):
            parts = path.relative_to(ROOT).parts
            hidden = any(part.startswith(".") or part in skipped for part in parts)
            if path.is_file() and not hidden and not any(p.endswith(".egg-info") for p in parts):
                found.append(path.relative_to(ROOT).as_posix())
        return sorted(found)


def _text_files() -> dict[str, str]:
    """The text files that the name rules read.

    Returns
    -------
    dict of str to str
        Path to text, for every tracked file with a text suffix, except this file (it holds the names it looks
        for), ``CHANGELOG.md`` (it records what was removed) and the files of ``NOT_READ``.
    """
    texts = {}
    for path in _tracked_files():
        if path.endswith(TEXT_SUFFIXES) and path not in (THIS_FILE, "CHANGELOG.md", *NOT_READ):
            texts[path] = (ROOT / path).read_text(encoding="utf-8")
    return texts


def _pages() -> dict[str, str]:
    """The Markdown pages of the repository.

    Returns
    -------
    dict of str to str
        Path to text, for every tracked ``.md`` file.
    """
    return {
        path: (ROOT / path).read_text(encoding="utf-8")
        for path in _tracked_files()
        if path.endswith(".md")
    }


def _without_code(text: str) -> str:
    """A Markdown text with its fenced code blocks and its inline code removed.

    Parameters
    ----------
    text : str
        The text of a page.

    Returns
    -------
    str
        The same text without what stands between code fences or between backticks, so that a link pattern does
        not match code.
    """
    text = re.sub(r"^(```|~~~).*?^\1[^\n]*$", "", text, flags=re.S | re.M)
    return re.sub(r"`[^`\n]*`", "", text)


def _code_blocks(text: str) -> list[str]:
    """The fenced code blocks of a Markdown text.

    Parameters
    ----------
    text : str
        The text of a page.

    Returns
    -------
    list of str
        The content of each block between a pair of code fences.
    """
    return [body for _, body in re.findall(r"^(```|~~~)[^\n]*\n(.*?)^\1", text, flags=re.S | re.M)]


def _anchors(text: str) -> set[str]:
    """The anchors that the headings of a Markdown page give, as GitHub and the documentation site form them.

    Parameters
    ----------
    text : str
        The text of a page.

    Returns
    -------
    set of str
        One anchor per heading: lower case, formatting and punctuation dropped, spaces as hyphens; a repeated
        heading also gives the anchor with ``-1``, ``-2``, ... appended.
    """
    anchors: set[str] = set()
    fenced = re.sub(r"^(```|~~~).*?^\1[^\n]*$", "", text, flags=re.S | re.M)
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", fenced, flags=re.M):
        plain = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
        plain = re.sub(r"[`*]", "", plain).strip().lower()
        slug = re.sub(r"[^\w\- ]", "", plain).replace(" ", "-")
        count = sum(
            1
            for known in anchors
            if known == slug or re.fullmatch(re.escape(slug) + r"-\d+", known)
        )
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def _names_found(texts: dict[str, str], names: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Which files name which of a set of names.

    Parameters
    ----------
    texts : dict of str to str
        Path to text.
    names : tuple of str
        The names looked for, as plain strings.

    Returns
    -------
    dict of str to tuple of str
        For each file that holds at least one of the names: the names it holds, sorted.
    """
    found = {}
    for path, text in texts.items():
        held = tuple(sorted(name for name in names if name in text))
        if held:
            found[path] = held
    return found


# ------------------------------------------------------------------------------------------- links
def test_no_internal_link_is_broken() -> None:
    """Every relative link and image of every page leads to a tracked file, and its anchor to a heading there.

    Absolute links (to the repository at a tag, or to other sites) are counted and not fetched.
    """
    tracked = set(_tracked_files())
    pages = _pages()
    broken, relative, absolute = [], 0, 0
    for page, text in pages.items():
        prose = _without_code(text)
        targets = re.findall(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)", prose)
        targets += re.findall(r"^\[[^\]]+\]:\s*(\S+)", prose, flags=re.M)
        for target in targets:
            if re.match(r"[a-z][a-z0-9+.-]*:", target):
                absolute += 1
                continue
            relative += 1
            path, _, anchor = target.partition("#")
            if path:
                resolved = (ROOT / page).parent.joinpath(path).resolve()
                inside = resolved == ROOT or ROOT in resolved.parents
                name = resolved.relative_to(ROOT).as_posix() if inside else None
                is_folder = name is not None and any(t.startswith(name + "/") for t in tracked)
                if name not in tracked and not is_folder and name != ".":
                    broken.append(f"{page}: {target} (no such tracked file)")
                    continue
            else:
                name = page
            if anchor and name in pages and anchor.lower() not in _anchors(pages[name]):
                broken.append(f"{page}: {target} (no heading gives that anchor in {name})")
    assert not broken, "\n".join(broken)
    assert relative >= 50 and absolute >= 50, (
        relative,
        absolute,
    )  # the pattern does find the links


# ------------------------------------------------------------------------------------------- removed documents
def test_no_page_names_a_deleted_document() -> None:
    """No tracked text file names a document that was removed, outside the changelog and an exact list of files
    that wait, each with its reason."""
    expected = {path: tuple(sorted(names)) for path, (names, _) in {**PENDING, **STANDING}.items()}
    assert all(reason for _, reason in {**PENDING, **STANDING}.values())
    found = _names_found(_text_files(), REMOVED)
    new = {path: names for path, names in found.items() if expected.get(path) != names}
    settled = sorted(path for path in expected if path not in found)
    assert not new, f"a removed document is named (or the list above is no longer exact): {new}"
    assert (
        not settled
    ), f"these files no longer name a removed document; take them off the list: {settled}"


def test_no_page_points_to_the_legacy_notebooks() -> None:
    """The folder of the older notebooks is named nowhere, outside the changelog and the files that wait."""
    found = _names_found(_text_files(), (LEGACY_NOTEBOOKS,))
    assert sorted(found) == sorted(PENDING_LEGACY_NOTEBOOKS), sorted(found)
    assert not (ROOT / LEGACY_NOTEBOOKS).exists()


# ------------------------------------------------------------------------------------------- corrected sentences
def test_the_published_model_is_loaded_the_supported_way() -> None:
    """No code block of a page calls ``load_bundle`` on the published release: the published model is loaded whole
    (``load_published``, ``load_release``); ``load_bundle`` reads a bundle, such as the backbone part of a release.
    """
    calls, wrong = 0, []
    for page, text in _pages().items():
        for block in _code_blocks(text):
            for first in re.findall(r"load_bundle\(\s*[\"']([^\"']*)[\"']", block):
                calls += 1
                if "release" in first and not first.endswith("/backbone"):
                    wrong.append(f"{page}: load_bundle({first!r}, ...)")
    assert not wrong, "\n".join(wrong)
    assert calls >= 3, calls  # the pattern does find the calls


def test_no_page_says_the_model_is_not_published() -> None:
    """The sentences that said the model or its package was not published do not come back."""
    pages = {path: " ".join(text.split()) for path, text in _pages().items()}
    for page, phrase in NOT_PUBLISHED:
        assert phrase not in pages[page], f"{page}: {phrase!r}"
    absent = r"hcd-priya-models`?[^.]{0,120}\bnot (?:yet )?on the package index"
    for page, text in pages.items():
        if page != "CHANGELOG.md":
            assert not re.search(absent, text), page


def test_one_convention_per_quantity_in_the_predictor_pages() -> None:
    """The pages of the supported interface describe no 35-bin grid of the model, and name the normalisation of a
    class by its own mean flux only to say that it is not the convention."""
    for page in PREDICTOR_PAGES:
        text = " ".join((ROOT / page).read_text(encoding="utf-8").split())
        assert not re.search(r"\b35[- ]bins?\b", text), f"{page}: a 35-bin grid"
        for sentence in re.split(r"(?<=[.:])\s", text):
            if re.search(r"own (?:subset's )?mean flux|own subset", sentence):
                qualified = "diagnostic" in sentence or re.search(r"\bnot\b", sentence)
                assert qualified, f"{page}: {sentence!r}"


def test_the_mean_flux_convention_cites_the_line_that_holds_it() -> None:
    """The source line that the constant ``MEAN_FLUX_CONVENTION`` cites holds the expression it quotes."""
    cited = re.match(r"(\S+\.py):(\d+) \((.+?)\)", MR.MEAN_FLUX_CONVENTION["source"])
    assert cited, MR.MEAN_FLUX_CONVENTION["source"]
    path, line, expression = cited.group(1), int(cited.group(2)), cited.group(3)
    lines = (ROOT / path).read_text(encoding="utf-8").split("\n")
    assert expression in lines[line - 1], f"{path}:{line} is {lines[line - 1].strip()!r}"
