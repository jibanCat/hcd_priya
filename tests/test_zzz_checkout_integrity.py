"""Gate A, task 11: the checkout under test is self-contained.

A test run in one checkout must import and load only that checkout's code and products; absolute literals into a
specific checkout (historically /home/mfho/hcd_priya) make a second checkout (e.g. the emulator-debug worktree)
silently test or load the first one's code and products. Lines carrying '# historical-artifact path' are exempt:
they name a frozen historical product on purpose."""
import io
import os
import sys
import tokenize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ABS = "/home/mfho/hcd_priya"  # historical-artifact path (the literal this guard searches for)
_EXEMPT_MARK = "# historical-artifact path"


def _code_files():
    out = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(REPO_ROOT, "hcd_analysis")) for f in fs if f.endswith(".py")]
    for sub in ("tests", "scripts"):
        d = os.path.join(REPO_ROOT, sub)
        out += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".py")]
    return sorted(out)


def _absolute_repo_literals(path):
    src = open(path, encoding="utf-8").read()
    lines = src.splitlines()
    hits = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type != tokenize.STRING:
            continue
        body = tok.string.lstrip("rbuRBUfF")[1:]
        for q in ('"""', "'''"):
            if tok.string.lstrip("rbuRBUfF").startswith(q):
                body = tok.string.lstrip("rbuRBUfF")[3:]
        if body.startswith(_ABS) and not body.startswith((_ABS + "_",)):
            line = lines[tok.start[0] - 1]
            if _EXEMPT_MARK not in line:
                hits.append(f"{os.path.relpath(path, REPO_ROOT)}:{tok.start[0]}")
    return hits


def test_no_absolute_repo_literals_in_code():
    hits = [h for p in _code_files() for h in _absolute_repo_literals(p)]
    assert not hits, f"{len(hits)} absolute repo literals (use paths relative to this checkout): {hits[:15]}"


def test_loaded_project_modules_come_from_this_checkout():
    """Run late in the session (file name sorts after most tests); any project module imported from another
    checkout (via a sys.path insertion or an absolute spec path) is a failure."""
    import hcd_analysis
    assert os.path.dirname(os.path.dirname(os.path.abspath(hcd_analysis.__file__))) == REPO_ROOT
    foreign = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None) or ""
        if "/hcd_priya" in f and not os.path.abspath(f).startswith(REPO_ROOT + os.sep):
            foreign.append(f"{name}: {f}")
    assert not foreign, f"modules loaded from another checkout: {foreign[:10]}"
