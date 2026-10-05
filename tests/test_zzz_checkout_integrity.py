"""Gate A, task 11 and blocking test BT8: the checkout under test is self-contained.

A test run in one checkout must import and load only that checkout's code and products. Absolute paths naming a
checkout of this repository (historically /home/mfho/hcd_priya, also any sibling such as a git worktree) make a
second checkout silently test or load the first one's code and products. Lines carrying '# historical-artifact path'
are exempt: they name a frozen historical product on purpose. The separate notes repository may be named.
The module-origin check also runs at session end (tests/conftest.py, pytest_sessionfinish)."""
import io
import os
import re
import sys
import tokenize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHECKOUT_RX = re.compile(r"/home/mfho/hcd_priya(?!_notes)(?:_[A-Za-z0-9]+)?(?=[/\"'\s:;,)]|$)")  # historical-artifact path
_EXEMPT_MARK = "# historical-artifact path"
PROJECT_TOP = ("hcd_analysis", "scripts", "tests", "cli", "config")


def _code_files():
    out = []
    for sub in ("hcd_analysis", "tests", "scripts"):
        for dp, dns, fs in os.walk(os.path.join(REPO_ROOT, sub)):
            dns[:] = [d for d in dns if d != "__pycache__"]
            out += [os.path.join(dp, f) for f in fs if f.endswith(".py")]
    return sorted(out)


def _absolute_checkout_literals(path):
    src = open(path, encoding="utf-8").read()
    lines = src.splitlines()
    hits = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type != tokenize.STRING or not _CHECKOUT_RX.search(tok.string):
            continue
        span = lines[tok.start[0] - 1: tok.end[0]]
        if any(_EXEMPT_MARK in ln for ln in span):
            continue
        hits.append(f"{os.path.relpath(path, REPO_ROOT)}:{tok.start[0]}")
    return hits


def test_guard_pattern_catches_every_checkout_and_spares_the_notes_repo():
    base = "/home/mfho/" + "hcd_priya"          # assembled so that no literal in this file names a checkout
    assert _CHECKOUT_RX.search(f'"cd {base} && pytest"')
    assert _CHECKOUT_RX.search(f'"{base}_emudebug/scripts/x.py"')
    assert _CHECKOUT_RX.search(f"'PYTHONPATH={base} python3'")
    assert not _CHECKOUT_RX.search(f'"{base}_notes/docs/x.md"')


def test_no_absolute_checkout_literals_in_code():
    hits = [h for p in _code_files() for h in _absolute_checkout_literals(p)]
    assert not hits, f"{len(hits)} string literals name a checkout by absolute path: {hits[:20]}"


def _shell_files():
    out = [os.path.join(REPO_ROOT, f) for f in os.listdir(REPO_ROOT) if f.endswith((".sh", ".sbatch", ".slurm"))]
    for dp, dns, fs in os.walk(os.path.join(REPO_ROOT, "scripts")):
        out += [os.path.join(dp, f) for f in fs if f.endswith((".sh", ".sbatch", ".slurm"))]
    return sorted(out)


def test_no_absolute_checkout_paths_in_batch_scripts():
    """A batch script that cds into, or puts on PYTHONPATH, a fixed checkout runs that checkout's code whichever
    checkout it is submitted from. Batch scripts resolve the repository from the submission directory."""
    hits = []
    for path in _shell_files():
        for i, line in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
            if _CHECKOUT_RX.search(line) and _EXEMPT_MARK not in line:
                hits.append(f"{os.path.relpath(path, REPO_ROOT)}:{i}")
    assert not hits, f"{len(hits)} batch-script lines name a checkout by absolute path: {hits[:20]}"


def foreign_project_modules(modules=None):
    from tests.gate_helpers import foreign_project_modules as _f
    return _f(modules if modules is not None else sys.modules)


def test_loaded_project_modules_come_from_this_checkout():
    import hcd_analysis
    assert os.path.dirname(os.path.dirname(os.path.abspath(hcd_analysis.__file__))) == REPO_ROOT
    foreign = foreign_project_modules()
    assert not foreign, f"project modules loaded from another checkout: {foreign[:10]}"


def test_foreign_module_detection_is_not_path_substring_based():
    class _M:
        __file__ = "/some/other/place/hcd_analysis/emulator/data.py"
    assert foreign_project_modules({"hcd_analysis.emulator.data": _M()})
