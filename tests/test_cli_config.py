"""The `hcd` command never falls back to built-in settings: every subcommand needs an existing --config file.

The built-in values of hcd_analysis.config and config/default.yaml differ, and one approved set of defaults is not
chosen yet, so a run without a config file must stop with a clear message before any work starts. Synthetic config
values only."""
import pytest
from click.testing import CliRunner

from hcd_analysis.cli import run


def _required_args(command):
    """Placeholder values for a subcommand's required options, so the only thing missing is --config."""
    args = []
    for p in command.params:
        if getattr(p, "required", False) and p.opts:
            args += [p.opts[0], "1" if getattr(p.type, "name", "") == "integer" else "x"]
    return args


@pytest.fixture
def no_config_load(monkeypatch):
    """Fail the test if any subcommand reaches load_config."""
    def _refuse(*a, **k):
        raise AssertionError("load_config was reached")
    monkeypatch.setattr(run, "load_config", _refuse)


@pytest.mark.parametrize("name", sorted(run.cli.commands))
def test_every_subcommand_refuses_to_run_without_a_config(name, no_config_load):
    result = CliRunner().invoke(run.cli, [name, *_required_args(run.cli.commands[name])])
    assert result.exit_code == 2, result.output
    assert "--config is required" in result.output


@pytest.mark.parametrize("name", sorted(run.cli.commands))
def test_every_subcommand_refuses_a_missing_config_file(name, no_config_load, tmp_path):
    missing = tmp_path / "absent.yaml"
    result = CliRunner().invoke(run.cli, [name, "--config", str(missing), *_required_args(run.cli.commands[name])])
    assert result.exit_code == 2, result.output
    assert "config file not found" in result.output


def test_an_explicit_config_file_is_the_one_used(tmp_path):
    path = tmp_path / "synthetic.yaml"
    path.write_text("z_min: 2.5\nz_max: 3.5\nabsorber:\n  tau_threshold: 7.0\n")
    cfg = run.make_config(str(path), None, ("n_workers=3",), False, False)
    assert (cfg.z_min, cfg.z_max, cfg.absorber.tau_threshold, cfg.n_workers) == (2.5, 3.5, 7.0, 3)
