"""model_package/prepare.py: the check between loading the copied release and writing the package entry.

The script itself writes into the checkout, so it is not run here; its check is a function and is tested alone, on
synthetic values.
"""

import importlib.util
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPARE = os.path.join(REPO, "model_package", "prepare.py")


@pytest.fixture(scope="module")
def prepare():
    spec = importlib.util.spec_from_file_location("model_package_prepare", PREPARE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_release_that_loads_as_its_manifest_says_passes(prepare):
    loaded = {"release_version": "synthetic-release-1", "format": "hcd_priya.release/2", "more": 1}
    assert prepare.check_loaded(loaded, "synthetic-release-1", "hcd_priya.release/2") is None


@pytest.mark.parametrize(
    "loaded, names",
    [
        (
            {"release_version": "another", "format": "hcd_priya.release/2"},
            "release_version 'another'",
        ),
        (
            {"release_version": "synthetic-release-1", "format": "hcd_priya.release/1"},
            "format 'hcd_priya.release/1'",
        ),
        ({"release_version": "synthetic-release-1"}, "format None"),
    ],
)
def test_another_version_or_format_stops_before_an_entry_is_written(prepare, loaded, names):
    with pytest.raises(SystemExit) as stop:
        prepare.check_loaded(loaded, "synthetic-release-1", "hcd_priya.release/2")
    assert names in str(stop.value.code) and "no package entry was written" in str(stop.value.code)


def test_the_check_is_not_an_assert_that_python_O_would_skip():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('p', {PREPARE!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "m.check_loaded({'release_version': 'a', 'format': 'b'}, 'a', 'c')\n"
    )
    run = subprocess.run(
        [sys.executable, "-O", "-B", "-c", code],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(REPO),
    )
    assert run.returncode != 0 and "no package entry was written" in run.stderr
    with open(PREPARE, encoding="utf-8") as fh:
        body = fh.read().split("def add_release(")[1].split("\ndef ")[0]
    assert "assert " not in body, "add_release checks with explicit refusals, not with assert"
