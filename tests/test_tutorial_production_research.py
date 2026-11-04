"""The production and research bundle tutorial (docs/tutorials/production_and_research_bundles.md) stays correct:
every example in it runs, on synthetic inputs in a temporary directory, and prints exactly the output the tutorial
documents under it (docs/tutorials/run_tutorial.py runs them). The tutorial covers the six topics the PI required and
carries a workflow diagram. No data store is needed."""
import importlib.util
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUTORIAL = os.path.join(REPO, "docs", "tutorials", "production_and_research_bundles.md")


def _runner():
    spec = importlib.util.spec_from_file_location("run_tutorial", os.path.join(REPO, "docs", "tutorials",
                                                                                 "run_tutorial.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return _runner().run(TUTORIAL, workdir=str(tmp_path_factory.mktemp("tutorial")))


def test_every_example_prints_its_documented_output(results):
    bad = [(r.name, r.output, r.expected) for r in results if not r.ok]
    assert not bad, bad[0]
    assert len(results) >= 15


def test_the_tutorial_covers_the_required_topics_and_has_a_workflow_diagram():
    text = open(TUTORIAL, encoding="utf-8").read()
    for heading in ("## 1. Loading and verifying a production bundle",
                    "## 2. Creating, exporting, loading and evaluating a research cross-validation bundle",
                    "## 3. A fold-trained model passed to the production loader",
                    "## 4. Training and held-out splits, and fold-specific w_c maps",
                    "## 5. Avoiding leakage in training, validation and export",
                    "## 6. A future, larger simulation set"):
        assert heading in text, heading
    assert "```mermaid" in text and "flowchart" in text
    names = [n for n, _ in re.findall(r"<!-- (?:example): ([A-Za-z0-9_]+) -->()", text)]
    assert len(names) == len(set(names)) and "PENDING" not in text


def test_the_runner_refuses_a_malformed_tutorial(tmp_path):
    R = _runner()
    bad = tmp_path / "t.md"
    bad.write_text("<!-- example: a -->\n```python\nprint(1)\n```\n")
    with pytest.raises(ValueError, match="without an expected output"):
        R.run(str(bad))
    bad.write_text("<!-- example: a -->\n```python\nprint(1)\n```\n<!-- expected: a -->\n```text\n2\n```\n")
    assert [r.ok for r in R.run(str(bad))] == [False]
    bad.write_text("<!-- example: a -->\n```python\nprint(1)\n```\n<!-- expected: a -->\n```text\n1\n```\n")
    assert [r.ok for r in R.run(str(bad))] == [True]


# ---------------------------- the library version that carries the D5 rules (PU-0522; R3 delta d1, pre-merge P-M1)
# It reads pyproject.toml and docs/, so it lives in this source-tier file (Session B's wheel tier copies only tests/).
D5_VERSION = "0.2.0"


def _pyproject_version_source(text):
    """``("static", version)`` or ``("dynamic", attr)`` of the ``[project]`` version in a pyproject.toml text."""
    project = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S).group(1)
    static = re.search(r'^version\s*=\s*"([^"]+)"\s*$', project, re.M)
    if static:
        return "static", static.group(1)
    assert re.search(r'^dynamic\s*=\s*\[[^\]]*"version"', project, re.M), "no [project] version, static or dynamic"
    attr = re.search(r'^version\s*=\s*\{\s*attr\s*=\s*"([^"]+)"\s*\}', text, re.M)
    return "dynamic", attr.group(1) if attr else None


def test_the_library_version_is_the_one_the_docs_name_for_the_d5_rules():
    import hcd_analysis
    assert hcd_analysis.__version__ == D5_VERSION
    kind, value = _pyproject_version_source(open(os.path.join(REPO, "pyproject.toml")).read())
    assert (kind, value) in (("static", D5_VERSION), ("dynamic", "hcd_analysis.__version__")), (kind, value)
    doc = re.sub(r"\s+", " ", open(os.path.join(REPO, "docs", "release_format_h5.md")).read())
    rules = re.findall(r"The production and research rules apply from library version (\d+\.\d+\.\d+) ", doc)
    a3a = re.findall(r"Synthetic `h5/2` bundles written by a library before version (\d+\.\d+\.\d+) ", doc)
    written_by = re.findall(r"bundles written by (\d+\.\d+\.\d+) declare", doc, re.I)
    earlier = re.findall(r"Every earlier version, (\d+\.\d+\.\d+) included", doc)
    assert rules == [hcd_analysis.__version__] and a3a == [hcd_analysis.__version__], (rules, a3a)
    assert len(written_by) == 2 and set(written_by) == {hcd_analysis.__version__}, written_by
    assert earlier and all(e != hcd_analysis.__version__ for e in earlier)
    dynamic = ('[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.setuptools.dynamic]\n'
               'version = {attr = "hcd_analysis.__version__"}\n')                 # Session B's packaging form
    assert _pyproject_version_source(dynamic) == ("dynamic", "hcd_analysis.__version__")
