"""The publishing workflow publishes only on the maintainer's approval: it starts only by hand, it holds no
credential, only its upload job may ask for an identity, and no other workflow can publish. It was switched on for
the release of 1.0.0 with the maintainer's approval (until then every job carried `if: false`, and this file was
named test_publish_workflow_is_disabled.py). What the workflow does once started (one protected environment per
package and index, the upload action pinned to a commit, the check of the tag and of the release date) is tested
in tests/test_ci_tools.py. It reads the workflow file from the repository."""

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


def test_no_job_is_switched_off_and_nothing_has_a_token_by_default():
    doc, _on = _workflow()
    assert doc["permissions"] == {}
    assert set(doc["jobs"]) == {"build", "publish"}
    for name, job in doc["jobs"].items():
        assert "if" not in job, f"job {name} carries a condition of its own"
    # the upload job runs in a protected environment (a required reviewer), never without one
    assert doc["jobs"]["publish"][
        "environment"
    ], "the upload job must run in a protected environment"
    assert doc["jobs"]["publish"]["needs"] == "build"
    assert doc["jobs"]["publish"]["permissions"] == {"id-token": "write"}
    assert doc["jobs"]["build"]["permissions"] == {"contents": "read"}


def test_no_other_workflow_can_publish():
    folder = os.path.dirname(PATH)
    for name in sorted(os.listdir(folder)):
        text = open(os.path.join(folder, name), encoding="utf-8").read()
        if name != "publish.yml":
            for word in ("pypi", "id-token", "twine", "secrets."):
                assert word not in text.lower(), f"{name} mentions {word}"
