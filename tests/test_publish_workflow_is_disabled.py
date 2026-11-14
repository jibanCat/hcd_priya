"""The publishing workflow cannot publish: it starts only by hand, and every job is switched off. This test changes
only with the maintainer's release approval. It reads the workflow file from the repository."""

import os

import yaml

from hcd_analysis.paths import REPO_ROOT

PATH = os.path.join(REPO_ROOT, ".github", "workflows", "publish.yml")


def _workflow():
    doc = yaml.safe_load(open(PATH, encoding="utf-8"))
    return doc, doc.get("on", doc.get(True))  # YAML reads the key `on` as the boolean true


def test_it_starts_only_by_hand():
    _doc, on = _workflow()
    assert list(on) == ["workflow_dispatch"]


def test_every_job_is_switched_off_and_nothing_has_a_token_by_default():
    doc, _on = _workflow()
    assert doc["permissions"] == {}
    assert set(doc["jobs"]) == {"build", "publish"}
    for name, job in doc["jobs"].items():
        assert job.get("if") is False, f"job {name} is not disabled"
    assert doc["jobs"]["publish"]["permissions"] == {"id-token": "write"}
    assert doc["jobs"]["build"]["permissions"] == {"contents": "read"}


def test_no_other_workflow_can_publish():
    folder = os.path.dirname(PATH)
    for name in sorted(os.listdir(folder)):
        text = open(os.path.join(folder, name), encoding="utf-8").read()
        if name != "publish.yml":
            for word in ("pypi", "id-token", "twine", "secrets."):
                assert word not in text.lower(), f"{name} mentions {word}"
