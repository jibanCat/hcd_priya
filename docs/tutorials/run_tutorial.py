#!/usr/bin/env python3
"""Run the executable examples of docs/tutorials/production_and_research_bundles.md and check what they print.

Every example in the tutorial is a ```python block preceded by the line ``<!-- example: NAME -->``; the output it must
print is the ```text block after it, preceded by ``<!-- expected: NAME -->``. This module runs the examples in order,
in one shared namespace, inside a new temporary directory, and compares each example's printed output with the
expected one. Nothing outside the temporary directory is written.

    python docs/tutorials/run_tutorial.py            # run every example; exit 0 when every output matches
    python docs/tutorials/run_tutorial.py --show     # also print each example's output

Run it from a repository checkout with the package's environment (JAX with float64; a few minutes on one CPU). The
test ``tests/test_tutorial_production_research.py`` runs the same function, so the tutorial cannot go stale.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import sys
import tempfile
from typing import NamedTuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TUTORIAL = os.path.join(HERE, "production_and_research_bundles.md")
_BLOCK = re.compile(
    r"<!-- (example|expected): ([A-Za-z0-9_]+) -->\n```(python|text)\n(.*?)```", re.S
)


class Example(NamedTuple):
    name: str
    code: str
    expected: str


class Result(NamedTuple):
    name: str
    output: str
    expected: str

    @property
    def ok(self):
        return _norm(self.output) == _norm(self.expected)


def _norm(text):
    return "\n".join(line.rstrip() for line in text.strip("\n").splitlines())


def parse(text):
    """The examples of the tutorial text, in order; refused unless every example has exactly one expected output and
    the two kinds of block alternate."""
    code, out = {}, {}
    order = []
    for kind, name, lang, body in _BLOCK.findall(text):
        if (kind == "example") != (lang == "python"):
            raise ValueError(f"block {name}: an example is python, an expected output is text")
        table = code if kind == "example" else out
        if name in table:
            raise ValueError(f"block {name} appears twice as {kind}")
        table[name] = body
        if kind == "example":
            order.append(name)
    if set(code) != set(out):
        raise ValueError(
            f"examples without an expected output or the reverse: {sorted(set(code) ^ set(out))}"
        )
    return [Example(n, code[n], out[n]) for n in order]


def run(path=TUTORIAL, workdir=None):
    """Run every example of the tutorial at ``path`` in a new temporary directory (inside ``workdir`` when given).
    Returns the results in order. The process's working directory, ``sys.path`` and the code-identity function the
    examples replace are restored afterwards."""
    examples = parse(open(path, encoding="utf-8").read())
    saved_cwd, saved_path = os.getcwd(), list(sys.path)
    if REPO not in sys.path:
        sys.path.insert(0, REPO)  # the package and the test suite's synthetic cache writer
    from hcd_analysis import lineage

    saved_identity = lineage.code_identity
    results = []
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        os.chdir(tmp)
        ns = {"__name__": "tutorial"}  # the tutorial defines everything it uses
        try:
            for ex in examples:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    exec(compile(ex.code, f"<tutorial example {ex.name}>", "exec"), ns)
                results.append(Result(ex.name, buf.getvalue(), ex.expected))
        finally:
            os.chdir(saved_cwd)
            sys.path[:] = saved_path
            lineage.code_identity = saved_identity
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--show", action="store_true", help="print each example's output")
    ap.add_argument("--tutorial", default=TUTORIAL)
    a = ap.parse_args(argv)
    results = run(a.tutorial)
    for r in results:
        print(f"{'ok  ' if r.ok else 'DIFF'} {r.name}")
        if a.show or not r.ok:
            print(
                "  output:\n" + "\n".join("    " + line for line in r.output.rstrip().splitlines())
            )
            if not r.ok:
                print(
                    "  expected:\n"
                    + "\n".join("    " + line for line in r.expected.rstrip().splitlines())
                )
    bad = [r.name for r in results if not r.ok]
    print(f"{len(results) - len(bad)} of {len(results)} examples print their documented output")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
