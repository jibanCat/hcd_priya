"""Packaging checks of the package as installed (CI runs this file against the wheel in a fresh environment).

- Every module imports with only the declared dependencies; the modules that need fake_spectra (installed separately,
  docs/fake_spectra_integration.md) are named here and must fail on exactly that import when it is absent.
- The two package-data files ship and their loaders read them from the installed package.
- The console script resolves and runs.
Synthetic expectations only: no value read from a package-data file is asserted."""
import importlib
import importlib.metadata as md
import json
import os
import pkgutil
from importlib import resources

import pytest

import cli
import hcd_analysis

NEEDS_FAKE_SPECTRA = {"hcd_analysis.priya_p1d"}
PACKAGE_DATA = (("hcd_analysis.comparison", "phase0_v1.json"),
                ("hcd_analysis.emulator", "hcd_lit_dndx_corrected.json"))


def _have_fake_spectra():
    try:
        importlib.import_module("fake_spectra")
    except ModuleNotFoundError:
        return False
    return True


def _module_names():
    names = []
    for pkg in (hcd_analysis, cli):
        names.append(pkg.__name__)
        names += [m.name for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + ".")]
    return sorted(names)


@pytest.mark.parametrize("name", _module_names())
def test_every_module_imports_with_the_declared_dependencies(name):
    if name in NEEDS_FAKE_SPECTRA and not _have_fake_spectra():
        with pytest.raises(ModuleNotFoundError) as exc:
            importlib.import_module(name)
        assert exc.value.name.split(".")[0] == "fake_spectra"
    else:
        importlib.import_module(name)


def test_the_fake_spectra_list_names_existing_modules():
    assert NEEDS_FAKE_SPECTRA <= set(_module_names())


@pytest.mark.parametrize("package, name", PACKAGE_DATA)
def test_package_data_ships_inside_the_package(package, name):
    res = resources.files(package).joinpath(name)
    assert res.is_file()
    assert isinstance(json.loads(res.read_text(encoding="utf-8")), dict)
    pkg_dir = os.path.dirname(importlib.import_module(package).__file__)
    assert os.path.isfile(os.path.join(pkg_dir, name))


def test_package_data_loaders_read_the_installed_files():
    from hcd_analysis.comparison import cases
    from hcd_analysis.emulator import lit_dndx
    assert os.path.dirname(os.path.abspath(cases.__file__)) == cases.CONFIG_DIR
    assert isinstance(cases.load_config(), dict) and cases.load_config()
    assert os.path.isfile(lit_dndx._DERIVATION_JSON)
    assert isinstance(lit_dndx.lit_points_for_display(), dict)


def test_distribution_metadata_matches_the_package():
    dist = md.distribution("hcd_analysis")
    assert dist.version == hcd_analysis.__version__
    assert dist.metadata["Requires-Python"] == ">=3.11"
    assert dist.metadata["License-Expression"] == "MIT"
    assert sorted((dist.read_text("top_level.txt") or "").split()) == ["cli", "hcd_analysis"]


def test_console_script_runs():
    from click.testing import CliRunner
    (ep,) = [e for e in md.entry_points(group="console_scripts") if e.name == "hcd"]
    assert ep.value == "cli.run:cli"
    result = CliRunner().invoke(ep.load(), ["--help"])
    assert result.exit_code == 0, result.output
    assert "run-one" in result.output
