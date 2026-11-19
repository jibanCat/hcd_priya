"""The main likelihood entry points show every verified data load by default (C4a review MAJOR-1): each calls
``runtime_data.show_loads()`` in ``main()`` before it parses its arguments, so the product id, version, digest and
path of each data file it reads reach stderr. The library itself configures no logging (tests/test_runtime_data.py).
"""

import ast
import logging
import os
import sys

import pytest

from hcd_analysis import runtime_data as RD

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY_POINTS = [
    "hcd_analysis/emulator/closure_legb.py",
    "scripts/run_real_fit.py",
    "scripts/run_stepA.py",
    "scripts/run_prod_sbc_shard.py",
    "scripts/run_dnuis_bias_shard.py",
]


def _calls(node):
    return {
        c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", None)
        for c in ast.walk(node)
        if isinstance(c, ast.Call)
    }


@pytest.mark.parametrize("path", ENTRY_POINTS)
def test_each_entry_point_shows_data_loads_before_it_does_anything_else(path):
    tree = ast.parse(open(os.path.join(REPO, path), encoding="utf-8").read())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    shown = [
        i for i, s in enumerate(main.body) if isinstance(s, ast.Expr) and "show_loads" in _calls(s)
    ]
    parsed = [i for i, s in enumerate(main.body) if "parse_args" in _calls(s)]
    assert shown and parsed and shown[0] < parsed[0]


def test_the_package_entry_point_attaches_the_display(monkeypatch):
    from hcd_analysis.emulator import closure_legb as CL

    lg = logging.getLogger("hcd_analysis.runtime_data")
    handlers, level = list(lg.handlers), lg.level
    monkeypatch.setattr(sys, "argv", ["closure_legb"])  # no mode: prints its help and loads nothing
    try:
        CL.main()
        assert any(h.name == RD.SHOW_HANDLER_NAME for h in lg.handlers)
        assert lg.getEffectiveLevel() <= logging.INFO
    finally:
        lg.handlers[:] = handlers
        lg.setLevel(level)
