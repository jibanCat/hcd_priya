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
