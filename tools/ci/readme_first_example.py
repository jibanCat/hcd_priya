"""Run the first example of the README exactly as it is written, in the current environment.

    python -I tools/ci/readme_first_example.py README.md

It takes the first Python block of the README's Quickstart and the output block printed under it, and checks:

- the example writes the nine parameters out: it holds one ``params = {...}`` of plain numbers whose names are the
  model's nine parameter names, and it does not call ``example_params`` (a first example never hides its values);
- it loads with ``hp.load_published`` and predicts;
- it runs, and prints the lines the README shows. Each printed line is ``label: value``. The labels must be the
  README's, in order. The values are compared only where they do not depend on the model: a model with another
  number of wavenumbers, or a synthetic test model, prints other shapes and flags;
- ``pred.P_class`` is an array with four rows (clean, LLS, subDLA, DLA) and at least one column, of finite numbers.

Run it with ``-I`` in an environment that has only the installed wheels: the script fails if ``hcd_analysis`` was
imported from the directory that holds the README (a source tree), so the README is tested against the installation.
Exit status 0 on success.
"""

from __future__ import annotations

import ast
import contextlib
import io
import os
import sys

# the labels whose value is the same for every model
SAME_FOR_EVERY_MODEL = ("classes, in the order of P_class",)


def blocks(text: str) -> tuple[str, str]:
    """The first Python block of the Quickstart and the first text block after it."""
    start = text.index("## Quickstart\n")
    a = text.index("```python\n", start) + len("```python\n")
    b = text.index("```\n", a)
    c = text.index("```text\n", b) + len("```text\n")
    return text[a:b], text[c : text.index("```\n", c)]


def written_out_params(code: str) -> dict | None:
    """The dict literal assigned to ``params`` in the example, if its keys are strings and its values numbers."""
    for node in ast.walk(ast.parse(code)):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "params"
            and isinstance(node.value, ast.Dict)
        ):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                return None
            ok = all(isinstance(k, str) for k in value) and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in value.values()
            )
            return value if ok else None
    return None


def main(argv=None) -> int:
    (readme,) = argv if argv is not None else sys.argv[1:]
    code, shown = blocks(open(readme, encoding="utf-8").read())
    print(code.rstrip())
    params = written_out_params(code)
    if "example_params" in code or params is None:
        print(
            "FAILED: the first example must write the nine parameters out as params = {name: number, ...}"
        )
        return 1
    if "hp.load_published(" not in code or ".predict(" not in code:
        print("FAILED: the first example must load with hp.load_published and predict")
        return 1
    out = io.StringIO()
    scope: dict = {}
    with contextlib.redirect_stdout(out):
        exec(compile(code, readme, "exec"), scope)
    printed = out.getvalue().splitlines()
    import numpy as np
    import hcd_analysis
    import hcd_analysis.predictor as hp

    print("-> printed:")
    for line in printed:
        print("   " + line)
    if tuple(params) != tuple(hp.PARAM_NAMES):
        print(
            f"FAILED: the example's parameter names {tuple(params)} are not the model's {tuple(hp.PARAM_NAMES)}"
        )
        return 1
    want = shown.splitlines()
    label = lambda line: line.partition(": ")[0]
    if [label(x) for x in printed] != [label(x) for x in want] or not all(": " in x for x in want):
        print("FAILED: the example does not print the lines that the README shows under it:")
        print(shown.rstrip())
        return 1
    for got, exp in zip(printed, want):
        if label(exp) in SAME_FOR_EVERY_MODEL and got != exp:
            print(f"FAILED: the README shows {exp!r}, the example printed {got!r}")
            return 1
    value = scope["pred"].P_class
    shape = getattr(value, "shape", None)
    ok = shape is not None and len(shape) == 2 and shape[0] == 4 and shape[1] > 0
    ok = ok and bool(np.all(np.isfinite(value)))
    where = os.path.realpath(hcd_analysis.__file__)
    tree = os.path.realpath(os.path.dirname(os.path.abspath(readme)))
    print(f"-> pred.P_class has shape {shape}; finite: {ok}")
    print(f"-> hcd_analysis {hcd_analysis.__version__} from {where}")
    if os.path.commonpath([where, tree]) == tree:
        print(
            "FAILED: hcd_analysis was imported from the directory of the README, not from an installation"
        )
        return 1
    print("PASSED" if ok else "FAILED: the result is not a finite (4, K) array with K > 0")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
