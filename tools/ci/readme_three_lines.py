"""Run the three-line example of the README exactly as it is written, in the current environment.

    python -I tools/ci/readme_three_lines.py README.md

It takes the first Python block of the README's Quickstart, checks that it is the three-line path (load_published,
predict, P_class), runs it, and checks what the last line gives: an array with four rows (clean, LLS, subDLA, DLA)
and at least one column, of finite numbers. Run it with ``-I`` in an environment that has only the installed wheels:
the script fails if ``hcd_analysis`` was imported from the directory that holds the README (a source tree), so the
README is tested against the installation. Exit status 0 on success.
"""

from __future__ import annotations

import os
import sys


def main(argv=None) -> int:
    (readme,) = argv if argv is not None else sys.argv[1:]
    text = open(readme, encoding="utf-8").read()
    start = text.index("## Quickstart\n")
    a = text.index("```python\n", start) + len("```python\n")
    block = text[a : text.index("```\n", a)]
    lines = [ln for ln in block.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    wanted = (
        "import hcd_analysis.predictor as hp",
        "emu = hp.load_published(",
        "pred = emu.predict(",
        "pred.P_class",
    )
    if len(lines) != 4 or not all(ln.startswith(w) for ln, w in zip(lines, wanted)):
        print("FAILED: the first Python block of the Quickstart is not the three-line example:")
        print(block)
        return 1
    print(block.rstrip())
    scope: dict = {}
    exec(compile("\n".join(lines[:3]), readme, "exec"), scope)  # the import and the first two lines
    value = eval(compile(lines[3].split("#")[0].strip(), readme, "eval"), scope)  # the third line
    import numpy as np
    import hcd_analysis

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
