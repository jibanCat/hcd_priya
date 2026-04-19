"""
CLI entry points for hcd_analysis.

Usage:
  hcd run-one  --sim <sim_fragment> --snap <snap_number> [options]
  hcd run-sim  --sim <sim_fragment> [options]
  hcd run-all  [options]
  hcd benchmark [options]
  hcd report   [options]
  hcd coverage [options]

Global options (all subcommands):
  --config PATH          YAML config file (default: config/default.yaml)
  --output-root PATH     Override output directory
  --set KEY=VALUE        Override any config key (dot notation, repeatable)
                         Example: --set n_workers=8 --set cddf.A=1.2
  --debug                Enable debug mode (2 sims, 3 snaps each)
  --fast                 Use fast NHI estimator (no Voigt fitting)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from hcd_analysis.config import load_config, save_config
from hcd_analysis.snapshot_map import build_snapshot_map, print_coverage_table


def _parse_set_overrides(set_values):
    """Parse 'key.subkey=value' strings into {key.subkey: value} dict."""
    overrides = {}
    for item in set_values:
        if "=" not in item:
            click.echo(f"ERROR: --set value must be KEY=VALUE, got: {item!r}", err=True)
            sys.exit(1)
        k, v = item.split("=", 1)
        # Auto-convert value type
        for converter in (int, float):
            try:
                v = converter(v)
                break
            except ValueError:
                pass
        if v in ("true", "True"):
            v = True
        elif v in ("false", "False"):
            v = False
        elif v in ("null", "None"):
            v = None
        overrides[k] = v
    return overrides


# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

_common_options = [
    click.option("--config", "config_path", default="config/default.yaml",
                 show_default=True, help="YAML config file"),
    click.option("--output-root", default=None, help="Override output root directory"),
    click.option("--set", "set_values", multiple=True, metavar="KEY=VALUE",
                 help="Override config key (dot notation). Repeatable."),
    click.option("--debug", is_flag=True, default=False, help="Debug mode"),
    click.option("--fast", is_flag=True, default=False, help="Fast NHI (no Voigt fitting)"),
    click.option("-v", "--verbose", is_flag=True, default=False),
]


def add_common_options(func):
    for option in reversed(_common_options):
        func = option(func)
    return func


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def make_config(config_path, output_root, set_values, debug, fast):
    overrides = _parse_set_overrides(set_values)
    if output_root:
        overrides["output_root"] = output_root
    if debug:
        overrides["debug"] = True
    if fast:
        overrides["benchmark"] = True
    cfg_file = config_path if Path(config_path).exists() else None
    return load_config(cfg_file, overrides)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """hcd_analysis – HCD absorber pipeline for fake_spectra emulator outputs."""
    pass


# ---------------------------------------------------------------------------
# coverage: show which (sim, snap) pairs exist
# ---------------------------------------------------------------------------

@cli.command()
@add_common_options
@click.option("--z-min", default=2.0, show_default=True)
@click.option("--z-max", default=6.0, show_default=True)
def coverage(config_path, output_root, set_values, debug, fast, verbose, z_min, z_max):
    """Print a coverage table of available (sim, snap) pairs."""
    setup_logging(verbose)
    cfg = make_config(config_path, output_root, set_values, debug, fast)
    snap_map = build_snapshot_map(cfg.data_root, z_min=z_min, z_max=z_max,
                                  prefer_grid=cfg.prefer_grid)
    print_coverage_table(snap_map)
    click.echo(f"\n{len(snap_map)} simulations found.")
    total = sum(len(ss.entries) for ss in snap_map)
    click.echo(f"{total} total (sim, snap) pairs in z=[{z_min}, {z_max}].")


# ---------------------------------------------------------------------------
# run-one: single (sim, snap)
# ---------------------------------------------------------------------------

@cli.command("run-one")
@add_common_options
@click.option("--sim", required=True, help="Simulation folder name (or fragment)")
@click.option("--snap", required=True, type=int, help="Snapshot number (e.g. 17)")
def run_one(config_path, output_root, set_values, debug, fast, verbose, sim, snap):
    """Run full analysis for one simulation and one snapshot."""
    setup_logging(verbose)
    cfg = make_config(config_path, output_root, set_values, debug, fast)

    from hcd_analysis.snapshot_map import build_snapshot_map
    from hcd_analysis.pipeline import run_one_snap

    snap_map = build_snapshot_map(cfg.data_root, z_min=cfg.z_min, z_max=cfg.z_max,
                                  sim_filter=[sim], prefer_grid=cfg.prefer_grid)
    if not snap_map:
        click.echo(f"ERROR: No simulation matching '{sim}' found.", err=True)
        sys.exit(1)

    ss = snap_map[0]
    entries = [e for e in ss.entries if e.snap == snap]
    if not entries:
        click.echo(f"ERROR: Snap {snap:03d} not available for sim '{sim}'.", err=True)
        available = [e.snap for e in ss.entries]
        click.echo(f"  Available snaps: {available}")
        sys.exit(1)

    result = run_one_snap(ss, entries[0], cfg)
    if result is None:
        click.echo("FAILED. Check error.txt in output directory.", err=True)
        sys.exit(1)

    click.echo(f"Done. Absorbers: {result.catalog.summary()}")
    click.echo(f"Timing: {result.timing}")


# ---------------------------------------------------------------------------
# run-sim: one simulation, all redshifts
# ---------------------------------------------------------------------------

@cli.command("run-sim")
@add_common_options
@click.option("--sim", required=True, help="Simulation folder name (or fragment)")
def run_sim(config_path, output_root, set_values, debug, fast, verbose, sim):
    """Run full analysis for one simulation, all redshifts."""
    setup_logging(verbose)
    cfg = make_config(config_path, output_root, set_values, debug, fast)

    from hcd_analysis.snapshot_map import build_snapshot_map
    from hcd_analysis.pipeline import run_sim_all_z

    snap_map = build_snapshot_map(cfg.data_root, z_min=cfg.z_min, z_max=cfg.z_max,
                                  sim_filter=[sim], prefer_grid=cfg.prefer_grid)
    if not snap_map:
        click.echo(f"ERROR: No simulation matching '{sim}' found.", err=True)
        sys.exit(1)

    results = run_sim_all_z(snap_map[0], cfg)
    n_ok = sum(1 for r in results if r is not None)
    click.echo(f"Done. {n_ok}/{len(results)} snapshots succeeded.")


# ---------------------------------------------------------------------------
# run-all: all simulations, all redshifts
# ---------------------------------------------------------------------------

@cli.command("run-all")
@add_common_options
@click.option("--n-workers", default=None, type=int, help="Override n_workers")
def run_all(config_path, output_root, set_values, debug, fast, verbose, n_workers):
    """Run full analysis for all simulations and all redshifts."""
    setup_logging(verbose)
    set_values = list(set_values)
    if n_workers is not None:
        set_values.append(f"n_workers={n_workers}")
    cfg = make_config(config_path, output_root, set_values, debug, fast)

    from hcd_analysis.pipeline import run_all as _run_all

    all_results = _run_all(cfg)
    total = sum(len(v) for v in all_results.values())
    n_ok = sum(1 for v in all_results.values() for r in v if r is not None)
    click.echo(f"Done. {n_ok}/{total} (sim, snap) pairs succeeded.")


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

@cli.command()
@add_common_options
@click.option("--n-sims", default=1, show_default=True, type=int)
@click.option("--n-snaps", default=2, show_default=True, type=int)
@click.option("--out", default=None, help="Write benchmark JSON to this file")
def benchmark(config_path, output_root, set_values, debug, fast, verbose, n_sims, n_snaps, out):
    """Run a timed benchmark and extrapolate to full campaign."""
    setup_logging(verbose)
    cfg = make_config(config_path, output_root, set_values, debug=False, fast=True)

    from hcd_analysis.pipeline import run_benchmark
    from hcd_analysis.report import generate_benchmarking_md

    result = run_benchmark(cfg, n_sims=n_sims, n_snaps=n_snaps)

    click.echo("\n=== Benchmark Results ===")
    click.echo(f"Total sims: {result['n_total_sims']}")
    click.echo(f"Total snaps: {result['n_total_snaps']}")
    click.echo(f"Avg time / snap: {result['avg_time_per_snap_s']} s")
    click.echo(f"Campaign serial: {result['campaign_serial_hr']} hr")
    click.echo(f"Campaign ({result['n_workers_assumed']} workers): {result['campaign_parallel_hr']} hr")

    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        click.echo(f"Wrote {out}")

    docs_dir = Path(cfg.output_root) / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    generate_benchmarking_md(docs_dir, result)
    click.echo(f"Wrote {docs_dir}/benchmarking.md")


# ---------------------------------------------------------------------------
# report: generate all figures + markdown docs
# ---------------------------------------------------------------------------

@cli.command()
@add_common_options
@click.option("--figures-dir", default="figures", show_default=True)
@click.option("--docs-dir", default="docs", show_default=True)
def report(config_path, output_root, set_values, debug, fast, verbose, figures_dir, docs_dir):
    """Generate all figures and markdown documentation."""
    setup_logging(verbose)
    cfg = make_config(config_path, output_root, set_values, debug, fast)

    from hcd_analysis.report import (
        generate_all_markdown,
        plot_discovery_summary,
    )
    from hcd_analysis.snapshot_map import build_snapshot_map

    figs = Path(figures_dir)
    docs = Path(docs_dir)
    figs.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)

    snap_map = build_snapshot_map(cfg.data_root, z_min=cfg.z_min, z_max=cfg.z_max,
                                  prefer_grid=cfg.prefer_grid)
    plot_discovery_summary(snap_map, figs)
    generate_all_markdown(docs)
    click.echo(f"Figures in {figs}/")
    click.echo(f"Docs in {docs}/")


# ---------------------------------------------------------------------------
# run-hires: HiRes simulations
# ---------------------------------------------------------------------------

@cli.command("run-hires")
@add_common_options
@click.option("--n-workers", default=None, type=int, help="Override n_workers")
def run_hires(config_path, output_root, set_values, debug, fast, verbose, n_workers):
    """Run full analysis for HiRes simulations (hires_data_root → outputs/hires/)."""
    setup_logging(verbose)
    set_values = list(set_values)
    if n_workers is not None:
        set_values.append(f"n_workers={n_workers}")
    cfg = make_config(config_path, output_root, set_values, debug, fast)

    from hcd_analysis.pipeline import run_hires as _run_hires

    all_results = _run_hires(cfg)
    total = sum(len(v) for v in all_results.values())
    n_ok = sum(1 for v in all_results.values() for r in v if r is not None)
    click.echo(f"HiRes done. {n_ok}/{total} (sim, snap) pairs succeeded.")


# ---------------------------------------------------------------------------
# convergence: compute HiRes/LF P1D ratios
# ---------------------------------------------------------------------------

@cli.command("convergence")
@add_common_options
def convergence(config_path, output_root, set_values, debug, fast, verbose):
    """Compute HiRes/LowRes P1D convergence ratios for matching sims."""
    setup_logging(verbose)
    cfg = make_config(config_path, output_root, set_values, debug, fast)

    from hcd_analysis.pipeline import compute_convergence_ratios

    ratios = compute_convergence_ratios(cfg)
    n_sims = len(ratios)
    n_z = sum(len(v) for v in ratios.values())
    click.echo(f"Convergence ratios computed for {n_sims} sims, {n_z} (sim, z) entries.")
    click.echo("Saved to outputs/hires/<sim>/convergence_ratios.npz")


if __name__ == "__main__":
    cli()
