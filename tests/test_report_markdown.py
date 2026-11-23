"""The Markdown pages that the report of the catalogue pipeline writes (hcd_analysis/report.py).

The page p1d_definition.md was retired from the documentation, so the report must not write it again. The test
writes into a temporary folder and reads nothing of a simulation.
"""

from __future__ import annotations

import pathlib

from hcd_analysis import report

WRITTEN = (
    "assumptions.md",
    "benchmarking.md",
    "cddf_model.md",
    "data_layout.md",
    "fake_spectra_integration.md",
    "pipeline_overview.md",
)


def test_the_report_writes_its_six_pages_and_not_the_retired_one(tmp_path: pathlib.Path) -> None:
    """``generate_all_markdown`` writes the six pages it wrote before, each with text, and no
    ``p1d_definition.md``; the function that wrote that page is gone."""
    report.generate_all_markdown(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(WRITTEN)
    assert all((tmp_path / name).read_text(encoding="utf-8").strip() for name in WRITTEN)
    assert not hasattr(report, "generate_p1d_definition_md")
