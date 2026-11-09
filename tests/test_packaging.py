"""Packaging checks of the package as installed (CI runs this file against the wheel in a fresh environment).

- Every third-party import (module-level or inside a function) maps to a declared requirement, and every declared
  requirement is imported; the undeclared exceptions are named here with their reason.
- Every module imports with only the declared dependencies; the modules that need fake_spectra (installed separately,
  docs/fake_spectra_integration.md) are named here and must fail on exactly that import when it is absent.
- The two package-data files ship and their loaders read them from the installed package.
- The console script resolves and runs, with the same subcommands, and no generic top-level `cli` package ships.
Synthetic expectations only: no value read from a package-data file is asserted."""
import ast
import glob
import importlib
import importlib.metadata as md
import json
import os
import pkgutil
import re
import subprocess
import sys
from importlib import resources

import pytest

import hcd_analysis

NEEDS_FAKE_SPECTRA = {"hcd_analysis.priya_p1d"}
UNDECLARED = {   # imported but deliberately not a requirement
    "fake_spectra": "the simulation reader, installed separately (docs/fake_spectra_integration.md)",
    "GPy": "reference-emulator adapter; runs in the reference's own environment",
    "paramz": "reference-emulator adapter; runs in the reference's own environment",
    "lyaemu": "reference-emulator adapter; runs in the reference's own environment",
}
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
    names = [hcd_analysis.__name__]
    names += [m.name for m in pkgutil.walk_packages(hcd_analysis.__path__, hcd_analysis.__name__ + ".")]
    return sorted(names)


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _third_party_imports():
    """{import root: {module file}} over every .py of the installed packages, imports at any depth."""
    roots = {}
    for pkg in (hcd_analysis,):
        base = os.path.dirname(os.path.dirname(pkg.__file__))
        for path in glob.glob(os.path.join(os.path.dirname(pkg.__file__), "**", "*.py"), recursive=True):
            for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                else:
                    continue
                for n in names:
                    root = n.split(".")[0]
                    if root not in sys.stdlib_module_names and root not in ("hcd_analysis", "__future__"):
                        roots.setdefault(root, set()).add(os.path.relpath(path, base))
    return roots


def test_every_import_is_a_declared_requirement_and_every_requirement_is_imported():
    """Static, so function-local imports count too, and a dependency that only arrives with pytest or as another
    package's dependency (tqdm through numpyro) is not mistaken for a declared one."""
    required = {_norm(re.split(r"[\s;<>=!~\[(]", r, maxsplit=1)[0])
                for r in md.requires("hcd-priya") or () if "extra ==" not in r}
    providers = md.packages_distributions()
    imports = _third_party_imports()
    undeclared, used = {}, set()
    for root, files in imports.items():
        dists = {_norm(d) for d in providers.get(root, ())} & required
        used |= dists
        if not dists and root not in UNDECLARED:
            undeclared[root] = sorted(files)
    assert not undeclared, f"imported but not declared in pyproject.toml: {undeclared}"
    assert not required - used, f"declared but imported nowhere: {sorted(required - used)}"
    assert set(UNDECLARED) <= set(imports), f"stale exceptions: {sorted(set(UNDECLARED) - set(imports))}"


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
    assert os.path.samefile(cases.CONFIG_DIR, os.path.join(os.path.dirname(hcd_analysis.__file__), "comparison"))
    assert isinstance(cases.load_config(), dict) and cases.load_config()
    assert os.path.isfile(lit_dndx._DERIVATION_JSON)
    assert isinstance(lit_dndx.lit_points_for_display(), dict)


def test_distribution_metadata_matches_the_package():
    dist = md.distribution("hcd-priya")
    assert dist.metadata["Name"] == "hcd-priya"
    assert dist.version == hcd_analysis.__version__
    assert dist.metadata["Requires-Python"] == ">=3.11"
    assert dist.metadata["License-Expression"] == "MIT"
    assert sorted((dist.read_text("top_level.txt") or "").split()) == ["hcd_analysis"]


def test_console_script_runs():
    """The installed `hcd` executable itself (not only its entry point object) runs, from a neutral directory."""
    (ep,) = [e for e in md.entry_points(group="console_scripts") if e.name == "hcd"]
    assert ep.value == "hcd_analysis.cli.run:cli"
    exe = os.path.join(os.path.dirname(sys.executable), "hcd")
    assert os.path.isfile(exe), f"no installed console script next to {sys.executable}"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run([exe, "--help"], capture_output=True, text=True, env=env, cwd=os.path.dirname(exe),
                            timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "run-one" in result.stdout


def test_cli_keeps_its_subcommands():
    """Moving the CLI under hcd_analysis.cli kept the `hcd` command set."""
    from hcd_analysis.cli.run import cli
    assert sorted(cli.commands) == ["benchmark", "convergence", "coverage", "report", "run-all", "run-hires",
                                    "run-one", "run-sim"]
