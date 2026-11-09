"""Gate A blocking test BT7 (gate A review section 11).

(a) Every package function that still consumes the single pre-2026-10 velocity grid (a ``cache_k`` argument or
attribute, or a checkpoint ``meta["kfkms"]``) carries the marker 'PRE-2026-10 INTERFACE' in its docstring together
with the later gate that replaces it, so that no reader mistakes it for the canonical coordinate.
(b) No comment or docstring in hcd_analysis/, scripts/ (recursive) or tests/ (recursive) claims that per-row k grids
are shared, the same across rows, or that row 0 is canonical. The isolated incident reproduction is exempt."""
import ast
import io
import os
import re
import tokenize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKER = "PRE-2026-10 INTERFACE"
GATE_RX = re.compile(r"gate [A-H]\b")


def _py_files(*subdirs):
    out = []
    for sub in subdirs:
        for dp, dns, fs in os.walk(os.path.join(REPO_ROOT, sub)):
            dns[:] = [d for d in dns if d != "__pycache__"]
            out += [os.path.join(dp, f) for f in fs if f.endswith(".py")]
    return sorted(out)


def _consumes_single_grid(fn_node, src):
    seg = ast.get_source_segment(src, fn_node) or ""
    args = {a.arg for a in fn_node.args.args + fn_node.args.kwonlyargs}
    return "cache_k" in args or "ctx.cache_k" in seg or re.search(r"meta\[\s*['\"]kfkms['\"]\s*\]", seg) is not None


def test_bt7a_single_grid_consumers_are_marked():
    unmarked = []
    for path in _py_files("hcd_analysis"):
        src = open(path, encoding="utf-8").read()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _consumes_single_grid(node, src):
                doc = ast.get_docstring(node) or ""
                if MARKER not in doc or not GATE_RX.search(doc):
                    unmarked.append(f"{os.path.relpath(path, REPO_ROOT)}:{node.lineno} {node.name}")
    assert not unmarked, f"single-grid consumers without '{MARKER} ... gate X' docstring: {unmarked}"


def test_bt7a_named_consumers_are_marked():
    from hcd_analysis.emulator import closure_legb, multifidelity
    for fn in (closure_legb.make_legb_mock, multifidelity.load_lf_backbone):
        assert MARKER in (fn.__doc__ or ""), fn.__name__


_SHARED_RX = re.compile(r"same across rows|rows share the (cache )?(angular-?)?k|per-row, shared grid|"
                        r"\(row-0\) grid|canonical row-0|row 0 (is )?canonical|representative k-grid", re.IGNORECASE)
_EXEMPT = {os.path.join("tests", "regression", "test_incident_2026_kgrid.py"), os.path.join("tests", "test_interface_markers.py")}


def _comments_and_strings(path):
    src = open(path, encoding="utf-8").read()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            yield tok.start[0], tok.string


def test_bt7b_no_shared_grid_wording():
    hits = []
    for path in _py_files("hcd_analysis", "scripts", "tests"):
        rel = os.path.relpath(path, REPO_ROOT)
        if rel in _EXEMPT:
            continue
        for line, text in _comments_and_strings(path):
            if _SHARED_RX.search(text):
                hits.append(f"{rel}:{line}")
    assert not hits, f"comments/docstrings asserting shared or row-0 k grids: {hits}"
