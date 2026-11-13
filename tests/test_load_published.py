"""hp.load_published: a model is found in the installed model package (hcd-priya-models) and loaded only against a
digest that hcd-priya pins. Synthetic releases (untrained networks) in stand-in model packages: no real model.
"""

import json
import os
import sys
import warnings

import numpy as np
import pytest

import hcd_analysis.predictor as hp
from hcd_analysis import published_models as PM
from hcd_analysis.emulator.mf_release import MFRecordError
from hcd_analysis.emulator.release_manifest import ReleaseManifestError
from tests.synthetic_release import make_release, model_package, use_model_package

SWITCH = "HCD_PRIYA_ALLOW_SYNTHETIC"
# the nine parameters, written out: the midpoint of each parameter's training range
PARAMS = {
    "ns": 0.925,
    "Ap": 1.9e-9,
    "herei": 4.0,
    "heref": 2.7,
    "alphaq": 2.15,
    "hub": 0.70,
    "omegamh2": 0.143,
    "hireionz": 7.25,
    "bhfeedback": 0.05,
}


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    return make_release(tmp_path_factory.mktemp("rel") / "release")


@pytest.fixture(autouse=True)
def no_switch(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)


def _pin(release, version, **change):
    pin = {
        "name": "lfmf",
        "version": version,
        "format": "hcd_priya.release/1",
        "anchor_sha256": release["sha"],
    }
    pin.update(change)
    return pin


# ------------------------------------------------------------------------------------- a published (pinned) model
def test_a_pinned_model_loads_and_is_the_release_it_was_built_from(release, tmp_path, monkeypatch):
    model_package(tmp_path, release, version="7.0.0", synthetic=False)
    use_model_package(monkeypatch, tmp_path)
    monkeypatch.setattr(PM, "MODELS", [_pin(release, "7.0.0")])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a published model loads without any warning
        emu = hp.load_published("lfmf")
        assert type(hp.load_published()) is hp.Bundle  # no name: the LF+MF model
    ref = hp.load_release(str(release["root"]), release["sha"])
    p = emu.predict(PARAMS, 3.0, tau0_amp=1.0, dtau0=0.0)
    q = ref.predict(PARAMS, 3.0, tau0_amp=1.0, dtau0=0.0)
    assert type(emu) is hp.Bundle and p.labels["product"] == "LF+MF"
    np.testing.assert_array_equal(p.P_class, q.P_class)
    assert emu.provenance["release"]["release_sha256"] == release["sha"]
    assert hp.published_models() == [("lfmf", "7.0.0")]


def test_the_digest_comes_from_the_code_package_not_from_the_model_package(
    release, tmp_path, monkeypatch
):
    """The model package states the right digest for its files, but the code pins another: refused. And the reverse:
    the model package states a wrong digest for itself, the code pins the right one: loaded."""
    model_package(tmp_path, release, version="7.0.0", synthetic=False, anchor="0" * 64)
    use_model_package(monkeypatch, tmp_path)
    monkeypatch.setattr(PM, "MODELS", [_pin(release, "7.0.0")])
    assert type(hp.load_published("lfmf")) is hp.Bundle
    monkeypatch.setattr(PM, "MODELS", [_pin(release, "7.0.0", anchor_sha256="1" * 64)])
    with pytest.raises(ReleaseManifestError):
        hp.load_published("lfmf")


def test_changed_model_files_are_refused(release, tmp_path, monkeypatch):
    pkg = model_package(tmp_path, release, version="7.0.0", synthetic=False)
    use_model_package(monkeypatch, tmp_path)
    monkeypatch.setattr(PM, "MODELS", [_pin(release, "7.0.0")])
    target = pkg / "data" / "lfmf" / "mf" / "mf_record.json"
    os.chmod(target, 0o644)
    target.write_bytes(target.read_bytes().replace(b"synthetic", b"SYNTHETIC", 1))
    with pytest.raises(MFRecordError):
        hp.load_published("lfmf")


def test_another_release_under_a_pinned_version_is_refused(release, tmp_path, monkeypatch):
    other = make_release(tmp_path / "other" / "release", n_members=3)
    model_package(tmp_path / "site", other, version="7.0.0", synthetic=False)
    use_model_package(monkeypatch, tmp_path / "site")
    monkeypatch.setattr(PM, "MODELS", [_pin(release, "7.0.0")])
    with pytest.raises(ReleaseManifestError):
        hp.load_published("lfmf")


def test_a_model_version_that_is_not_pinned_is_refused_and_both_versions_are_named(
    release, tmp_path, monkeypatch
):
    model_package(tmp_path, release, version="8.0.0", synthetic=False)
    use_model_package(monkeypatch, tmp_path)
    monkeypatch.setattr(PM, "MODELS", [_pin(release, "7.0.0"), _pin(release, "7.1.0")])
    with pytest.raises(
        hp.ModelPackageError, match=r"hcd-priya-models 8\.0\.0 is installed.*7\.0\.0, 7\.1\.0"
    ):
        hp.load_published("lfmf")
    monkeypatch.setattr(PM, "MODELS", [])
    with pytest.raises(hp.ModelPackageError, match="none yet"):
        hp.load_published("lfmf")


def test_a_name_the_model_package_does_not_hold_lists_what_it_holds(release, tmp_path, monkeypatch):
    model_package(tmp_path, release, version="7.0.0", synthetic=False)
    use_model_package(monkeypatch, tmp_path)
    monkeypatch.setattr(PM, "MODELS", [_pin(release, "7.0.0")])
    with pytest.raises(
        hp.NoPublishedModelError, match=r"holds no model 'fold-0' \(it holds: lfmf\)"
    ):
        hp.load_published("fold-0")


# ------------------------------------------------------------------------------------- the model package itself
def test_a_missing_model_package_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "hcd_priya_models", None)  # import raises ImportError
    with pytest.raises(
        hp.ModelPackageError,
        match="hcd-priya-models is not installed.*install section of the README",
    ):
        hp.load_published("lfmf")


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda d: d.update(version="7.0.1"), "names another version"),
        (lambda d: d.update(synthetic="no"), "malformed"),
        (lambda d: d["models"][0].update(url="https://x.example.org"), "malformed"),
        (lambda d: d["models"][0].pop("anchor_sha256"), "malformed"),
        (lambda d: d["models"].append(dict(d["models"][0])), "malformed"),
        (lambda d: d.update(format="other/1"), "cannot be read"),
        (lambda d: d["models"][0].update(path="../data/lfmf"), "are missing"),
        (lambda d: d["models"][0].update(path="lfmf/backbone"), "are missing"),
        (lambda d: d["models"][0].update(path="ABSOLUTE"), "are missing"),
        (lambda d: d["models"][0].update(path="nowhere"), "are missing"),
    ],
)
def test_a_malformed_model_package_is_refused(release, tmp_path, monkeypatch, change, match):
    pkg = model_package(tmp_path, release, version="7.0.0", synthetic=False)
    use_model_package(monkeypatch, tmp_path)
    monkeypatch.setattr(PM, "MODELS", [_pin(release, "7.0.0")])
    doc = json.loads((pkg / "data" / "models.json").read_text())
    change(doc)
    if (
        doc["models"] and doc["models"][0].get("path") == "ABSOLUTE"
    ):  # an existing directory, given absolutely
        doc["models"][0]["path"] = str(pkg / "data" / "lfmf")
    (pkg / "data" / "models.json").write_text(json.dumps(doc))
    with pytest.raises(hp.ModelPackageError, match=match):
        hp.load_published("lfmf")


def test_a_format_that_is_not_a_release_is_refused(release, tmp_path, monkeypatch):
    model_package(tmp_path, release, version="7.0.0", synthetic=False)
    use_model_package(monkeypatch, tmp_path)
    monkeypatch.setattr(PM, "MODELS", [_pin(release, "7.0.0", format="hcd_priya.h5/2")])
    with pytest.raises(hp.ModelPackageError, match="not one load_published loads"):
        hp.load_published("lfmf")


# ------------------------------------------------------------------------------------- synthetic test builds
def test_a_synthetic_build_loads_only_with_the_test_switch_and_warns(
    release, tmp_path, monkeypatch
):
    model_package(tmp_path, release)  # version 0+synthetic, synthetic true, not pinned
    use_model_package(monkeypatch, tmp_path)
    with pytest.raises(hp.ModelPackageError, match="synthetic test build"):
        hp.load_published("lfmf")
    monkeypatch.setenv(SWITCH, "0")
    with pytest.raises(hp.ModelPackageError, match="synthetic test build"):
        hp.load_published("lfmf")
    monkeypatch.setenv(SWITCH, "1")
    with pytest.warns(UserWarning, match="untrained networks and its numbers are not science"):
        emu = hp.load_published("lfmf")
    assert type(emu) is hp.Bundle


def test_the_test_switch_does_not_open_a_build_that_is_not_marked_synthetic(
    release, tmp_path, monkeypatch
):
    model_package(tmp_path, release, version="8.0.0", synthetic=False)
    use_model_package(monkeypatch, tmp_path)
    monkeypatch.setenv(SWITCH, "1")
    with pytest.raises(hp.ModelPackageError, match="none yet"):
        hp.load_published("lfmf")


def test_a_pin_wins_over_the_synthetic_mark_with_the_switch_on_or_off(
    release, tmp_path, monkeypatch
):
    """A package that calls itself synthetic under a pinned version is loaded against the pin, whatever it states
    about itself and whatever the switch: the mark can only restrict."""
    other = make_release(tmp_path / "other" / "release", n_members=3)
    # the package holds ANOTHER release, states that release's own digest (self-consistent), and says "synthetic"
    model_package(tmp_path / "bad", other, version="7.0.0", synthetic=True)
    use_model_package(monkeypatch, tmp_path / "bad")
    monkeypatch.setattr(PM, "MODELS", [_pin(release, "7.0.0")])
    for switch in ("0", "1"):
        monkeypatch.setenv(SWITCH, switch)
        with pytest.raises(ReleaseManifestError):
            hp.load_published("lfmf")


def test_a_pinned_package_that_says_synthetic_loads_as_published_without_a_warning(
    release, tmp_path, monkeypatch
):
    model_package(tmp_path, release, version="7.0.0", synthetic=True)
    use_model_package(monkeypatch, tmp_path)
    monkeypatch.setattr(PM, "MODELS", [_pin(release, "7.0.0")])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        emu = hp.load_published("lfmf")
    assert "synthetic_build" not in emu.provenance["release"]


def test_the_synthetic_route_needs_the_local_version_label(release, tmp_path, monkeypatch):
    """A version that a package index could serve never takes the synthetic route, even with the switch on."""
    model_package(tmp_path, release, version="0.0.1", synthetic=True)
    use_model_package(monkeypatch, tmp_path)
    monkeypatch.setenv(SWITCH, "1")
    with pytest.raises(hp.ModelPackageError, match="none yet"):
        hp.load_published("lfmf")


def test_the_synthetic_route_still_checks_the_files_and_marks_the_model(
    release, tmp_path, monkeypatch
):
    monkeypatch.setenv(SWITCH, "1")
    pkg = model_package(tmp_path / "ok", release)
    use_model_package(monkeypatch, tmp_path / "ok")
    with pytest.warns(UserWarning, match="synthetic test build"):
        emu = hp.load_published("lfmf")
    pred = emu.predict(PARAMS, 3.0, tau0_amp=1.0, dtau0=0.0)
    assert emu.provenance["release"]["synthetic_build"] is True
    assert pred.provenance["release"]["synthetic_build"] is True
    # and where a user looks first: a flag that is True, and a label
    assert pred.flags["synthetic_build"] is True
    assert pred.labels["synthetic_build"].startswith(
        "SYNTHETIC TEST BUILD: the networks are untrained"
    )
    stack = emu.predict_many(PARAMS, [2.6, 3.0], tau0_amp=1.0, dtau0=0.0)
    assert stack.flags["synthetic_build"] is True and "synthetic_build" in stack.labels
    ref = hp.load_release(str(release["root"]), release["sha"])
    assert "synthetic_build" not in ref.provenance["release"]
    plain = ref.predict(
        PARAMS, 3.0, tau0_amp=1.0, dtau0=0.0
    )  # the same files loaded as a release: no mark
    assert "synthetic_build" not in plain.flags and "synthetic_build" not in plain.labels
    # a changed file, and a wrong stated digest, are refused on this route too
    target = pkg / "data" / "lfmf" / "mf" / "mf_record.json"
    os.chmod(target, 0o644)
    target.write_bytes(target.read_bytes().replace(b"synthetic", b"SYNTHETIC", 1))
    with pytest.raises(MFRecordError), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hp.load_published("lfmf")
    model_package(tmp_path / "wrong", release, anchor="0" * 64)
    use_model_package(monkeypatch, tmp_path / "wrong")
    with pytest.raises(ReleaseManifestError), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hp.load_published("lfmf")


def test_two_pins_for_one_model_version_are_refused(release, tmp_path, monkeypatch):
    model_package(tmp_path, release, version="7.0.0", synthetic=False)
    use_model_package(monkeypatch, tmp_path)
    monkeypatch.setattr(
        PM, "MODELS", [_pin(release, "7.0.0"), _pin(release, "7.0.0", anchor_sha256="1" * 64)]
    )
    with pytest.raises(hp.ModelPackageError):
        hp.load_published("lfmf")


# ------------------------------------------------------------------------------------- what ships
def test_the_installed_model_package_loads(monkeypatch):
    """Whatever hcd-priya-models is installed here: a synthetic build (CI, a development environment) loads with the
    test switch; a published one loads against its pin."""
    import hcd_priya_models

    doc = hcd_priya_models.manifest()
    assert doc["version"] == hcd_priya_models.__version__
    assert (doc["version"].endswith("+synthetic")) == doc["synthetic"]
    if doc["synthetic"]:
        monkeypatch.setenv(SWITCH, "1")
        with pytest.warns(UserWarning, match="synthetic test build"):
            emu = hp.load_published("lfmf")
    else:
        emu = hp.load_published("lfmf")
    pred = emu.predict(PARAMS, z=3.0, tau0_amp=1.0, dtau0=0.0)
    assert pred.P_class.shape[0] == 4 and np.all(np.isfinite(pred.P_class))


def test_the_list_of_accepted_models_is_well_formed():
    """Holds for the empty list and for every later one: four keys, a known format, a 64-hex digest, a plain public
    version (never a synthetic one), and each (name, version) once."""
    import re

    seen = set()
    for m in PM.MODELS:
        assert set(m) == {"name", "version", "format", "anchor_sha256"}
        assert m["format"] == "hcd_priya.release/1"
        assert re.fullmatch(r"[0-9a-f]{64}", m["anchor_sha256"])
        assert (
            re.fullmatch(r"(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*", m["version"])
            and "+" not in m["version"]
        )
        assert (m["name"], m["version"]) not in seen
        seen.add((m["name"], m["version"]))


def test_no_model_is_pinned_yet():
    """No model is published: the list of accepted models is empty. This test changes with the first publication."""
    assert PM.MODELS == [] and hp.published_models() == []


def test_example_params_is_the_midpoint_and_says_so_with_its_values(capsys):
    p = hp.example_params()
    out = capsys.readouterr().out
    assert list(p) == list(hp.PARAM_NAMES) and all(type(v) is float for v in p.values())
    np.testing.assert_allclose(
        hp.params_vector(p), np.mean(np.asarray(hp.PARAM_LIMITS), axis=1), rtol=1e-15
    )
    # the values are never hidden: the call prints what the point is, and every value
    assert "the midpoint of the training range" in out
    assert "not a fiducial cosmology, not a best fit and not a simulation" in out
    assert [ln.strip() for ln in out.splitlines()[2:]] == [f"{n} = {v:g}" for n, v in p.items()]
    quiet = hp.example_params(verbose=False)
    assert capsys.readouterr().out == "" and quiet == p
    with pytest.raises(TypeError):
        hp.example_params(False)  # keyword only: a bare False would read as a value
    # the values the README and the docstrings write out are this point
    np.testing.assert_allclose(hp.params_vector(PARAMS), hp.params_vector(p), rtol=1e-12)


def test_the_model_loads_without_a_name_and_under_any_spelling_of_it(
    release, tmp_path, monkeypatch
):
    model_package(tmp_path, release, version="7.0.0", synthetic=False)
    use_model_package(monkeypatch, tmp_path)
    monkeypatch.setattr(PM, "MODELS", [_pin(release, "7.0.0")])
    assert hp.DEFAULT_MODEL == "LF+MF"
    ref = hp.load_published().manifest_sha256  # no name: the LF+MF model
    for name in ("LF+MF", "lf+mf", "LF-MF", "lf_mf", "LFMF", "lfmf", " LF + MF "):
        assert hp.load_published(name).manifest_sha256 == ref, name
    for name in ("LF", "MF", "lfmf2", "", "fold-0"):
        with pytest.raises(hp.NoPublishedModelError, match=r"\(it holds: lfmf\)"):
            hp.load_published(name)


def test_two_names_that_read_alike_in_one_package_are_refused_as_a_malformed_package(
    release, tmp_path, monkeypatch
):
    pkg = model_package(tmp_path, release, version="7.0.0", synthetic=False)
    doc = json.loads((pkg / "data" / "models.json").read_text())
    doc["models"].append(dict(doc["models"][0], name="LF+MF"))
    (pkg / "data" / "models.json").write_text(json.dumps(doc))
    use_model_package(monkeypatch, tmp_path)
    monkeypatch.setattr(PM, "MODELS", [_pin(release, "7.0.0")])
    for name in ("lfmf", "LF+MF", "fold-0"):  # whatever is asked for: the package's list is refused
        with pytest.raises(hp.ModelPackageError, match="its list of models is malformed"):
            hp.load_published(name)


def test_the_codes_entry_may_spell_the_name_like_a_user(release, tmp_path, monkeypatch):
    """The entry in the code's list is compared without case and punctuation too: an entry written "LF+MF" accepts
    the package's "lfmf". The digest still comes from that entry."""
    model_package(tmp_path, release, version="7.0.0", synthetic=False)
    use_model_package(monkeypatch, tmp_path)
    monkeypatch.setattr(PM, "MODELS", [_pin(release, "7.0.0", name="LF+MF")])
    assert hp.load_published().provenance["release"]["release_sha256"] == release["sha"]
    assert hp.published_models() == [("LF+MF", "7.0.0")]
    monkeypatch.setattr(
        PM, "MODELS", [_pin(release, "7.0.0", name="LF+MF", anchor_sha256="0" * 64)]
    )
    with pytest.raises(ReleaseManifestError):
        hp.load_published()
    monkeypatch.setattr(PM, "MODELS", [_pin(release, "6.0.0", name="LF+MF")])
    with pytest.raises(hp.ModelPackageError, match="accepts for 'lfmf' the model versions: 6.0.0"):
        hp.load_published()
    monkeypatch.setattr(
        PM, "MODELS", [_pin(release, "7.0.0"), _pin(release, "7.0.0", name="LF+MF")]
    )
    with pytest.raises(
        hp.ModelPackageError, match="accepts for 'lfmf'"
    ):  # two entries for one version
        hp.load_published()
