"""The `hcd` command does not run without a config file: every subcommand needs an existing --config file.

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
    result = CliRunner().invoke(
        run.cli, [name, "--config", str(missing), *_required_args(run.cli.commands[name])]
    )
    assert result.exit_code == 2, result.output
    assert "config file not found" in result.output


def test_an_explicit_config_file_is_the_one_used(tmp_path):
    path = tmp_path / "synthetic.yaml"
    path.write_text("z_min: 2.5\nz_max: 3.5\nabsorber:\n  tau_threshold: 7.0\n")
    cfg = run.make_config(str(path), None, ("n_workers=3",), False, False)
    assert (cfg.z_min, cfg.z_max, cfg.absorber.tau_threshold, cfg.n_workers) == (2.5, 3.5, 7.0, 3)


def _benchmark_config(tmp_path, root):
    path = tmp_path / "synthetic.yaml"
    path.write_text(f"output_root: {root}\ndata_root: {tmp_path / 'no_data'}\n")
    return str(path)


def test_benchmark_records_its_settings_before_writing_into_the_output_root(tmp_path, monkeypatch):
    """benchmarking.md goes into the output root, so the benchmark's settings are recorded there first: the root
    never holds files of unrecorded settings (hcd_analysis/run_settings.py)."""
    import hcd_analysis.pipeline as pipeline
    import hcd_analysis.report as report
    from hcd_analysis.run_settings import RUN_SETTINGS_FILE

    root = tmp_path / "out"
    seen = {}

    def fake_benchmark(cfg, n_sims=1, n_snaps=1):
        seen["record_first"] = (root / RUN_SETTINGS_FILE).is_file()
        return dict(
            n_total_sims=1,
            n_total_snaps=1,
            avg_time_per_snap_s=1.0,
            campaign_serial_hr=1.0,
            n_workers_assumed=1,
            campaign_parallel_hr=1.0,
        )

    monkeypatch.setattr(pipeline, "run_benchmark", fake_benchmark)
    monkeypatch.setattr(
        report, "generate_benchmarking_md", lambda d, r: (d / "benchmarking.md").write_text("x")
    )
    result = CliRunner().invoke(
        run.cli, ["benchmark", "--config", _benchmark_config(tmp_path, root)]
    )
    assert result.exit_code == 0, result.output
    assert seen["record_first"] and (root / "docs" / "benchmarking.md").is_file()


def test_benchmark_is_refused_in_a_root_of_unrecorded_settings_before_anything_runs(
    tmp_path, monkeypatch
):
    import hcd_analysis.pipeline as pipeline
    from hcd_analysis.run_settings import RunSettingsMismatch

    root = tmp_path / "out"
    root.mkdir()
    (root / "results_of_unknown_settings.npz").write_bytes(b"")
    monkeypatch.setattr(pipeline, "run_benchmark", lambda *a, **k: pytest.fail("the benchmark ran"))
    result = CliRunner().invoke(
        run.cli, ["benchmark", "--config", _benchmark_config(tmp_path, root)]
    )
    assert isinstance(result.exception, RunSettingsMismatch)
    assert sorted(p.name for p in root.iterdir()) == ["results_of_unknown_settings.npz"]
