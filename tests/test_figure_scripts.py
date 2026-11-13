"""The figure scripts of scripts/figures and the data files they read (docs/figures/data).

The scripts are run here on a synthetic model, so these tests check that each script runs, reads only files whose
digest is recorded, reports the design points the predictor flags, and writes its image with the side file that
says what it was drawn from. They do not check what a figure of the trained model shows.
"""

import hashlib
import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import pytest

import hcd_analysis.predictor as hp
from tests import synthetic_release as SR

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts", "figures")
DATA = os.path.join(REPO, "docs", "figures", "data")
NOT_DATA = {"SHA256SUMS", "README.md", "ho21/PROVENANCE.md"}  # the files that describe the data
PNG = b"\x89PNG\r\n\x1a\n"
# The synthetic model serves LF+MF from z = 2.6 only, so the tests that run a model figure move the script's
# redshifts there. The redshifts the scripts are written with are checked on their own.
SYNTHETIC_Z = (2.6, 2.8, 3.0)


def _script(name):
    sys.path.insert(0, SCRIPTS)
    try:
        spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(SCRIPTS)
    return module


C = _script("figures_common")


def _sums():
    with open(os.path.join(DATA, "SHA256SUMS"), encoding="utf-8") as fh:
        return {name: digest for digest, name in (line.split() for line in fh)}


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    return SR.make_release(tmp_path_factory.mktemp("figures") / "release", incidence=True)


@pytest.fixture()
def design(tmp_path):
    """Three design points: the midpoint, and two points moved inside the range the synthetic model was built on."""
    path = tmp_path / "design.csv"
    rows = [dict(C.MIDPOINT), dict(C.MIDPOINT, ns=0.93), dict(C.MIDPOINT, hub=0.69)]
    lines = ["# a synthetic design table", ",".join(hp.PARAM_NAMES)]
    lines += [",".join(repr(float(r[n])) for n in hp.PARAM_NAMES) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _args(release):
    return ["--release", str(release["root"]), release["sha"]]


def _facts(out):
    with open(out, "rb") as fh:
        assert fh.read(8) == PNG
    with open(os.path.splitext(out)[0] + ".json", encoding="utf-8") as fh:
        return json.load(fh)


# ---- the committed data


def test_every_data_file_is_listed_with_its_digest():
    found = set()
    for folder, _, names in os.walk(DATA):
        for name in names:
            found.add(os.path.relpath(os.path.join(folder, name), DATA).replace(os.sep, "/"))
    sums = _sums()
    assert found - NOT_DATA == set(
        sums
    ), "a data file without a recorded digest, or a digest without its file"
    for name, digest in sums.items():
        assert _sha(os.path.join(DATA, name)) == digest, name


def test_the_ho21_provenance_names_each_file_with_its_digest_and_the_commit():
    with open(os.path.join(DATA, "ho21", "PROVENANCE.md"), encoding="utf-8") as fh:
        text = fh.read()
    ho21 = {n: d for n, d in _sums().items() if n.startswith("ho21/")}
    assert sorted(ho21) == [
        "ho21/cddf_z225.txt",
        "ho21/cddf_z253.txt",
        "ho21/cddf_z34.txt",
        "ho21/dndx_all.txt",
    ]
    assert "confirmed the rows on 2026-11-13" in text
    for name, digest in ho21.items():
        assert os.path.basename(name) in text and digest in text, name
    for needed in (
        "github.com/sbird/DLA_data",  # where the tables originate
        "147c616ba4f5779bea0ef3e23255f4a9c2021cea",
        "github.com/jibanCat/DLA_data",  # the fork they were fetched from
        "daa595c692254c93b86009318a6e2b029db67112",
        "10.1093/mnras/stab2169",
        "no licence file",
    ):
        assert needed in text, needed
    assert (
        len(
            [
                w
                for w in text.replace("`", " ").split()
                if len(w) == 40 and set(w) <= set("0123456789abcdef")
            ]
        )
        >= 1
    )


def test_a_data_file_whose_bytes_changed_is_refused(tmp_path, monkeypatch):
    copy = tmp_path / "data"
    shutil.copytree(DATA, copy)
    monkeypatch.setattr(C, "DATA", str(copy))
    assert C.data_file("ho21", "cddf_z225.txt") == str(copy / "ho21" / "cddf_z225.txt")
    with open(copy / "ho21" / "cddf_z225.txt", "ab") as fh:
        fh.write(b"\n")
    with pytest.raises(SystemExit, match="is not the one recorded"):
        C.data_file("ho21", "cddf_z225.txt")
    (copy / "ho21" / "another.txt").write_text("1 2 3\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="is not the one recorded"):
        C.data_file("ho21", "another.txt")


def test_the_parameters_of_the_figures_are_the_midpoint_the_package_states():
    assert list(C.MIDPOINT) == list(hp.PARAM_NAMES)
    assert C.MIDPOINT == pytest.approx(hp.example_params(verbose=False), rel=1e-12)
    for name, value in C.MIDPOINT.items():
        assert f"{name} = {value:g}" in C.MIDPOINT_TEXT


def test_the_ho21_tables_read_as_centre_value_and_68_per_cent_limits():
    cddf = _script("fig_cddf_vs_ho21")
    assert [name for _, name, _ in cddf.PANELS] == [
        "cddf_z225.txt",
        "cddf_z253.txt",
        "cddf_z34.txt",
    ]
    for _, name, z in cddf.PANELS:
        logn, f, lo, hi = cddf.read_ho21_cddf(os.path.join(DATA, "ho21", name))
        assert logn.size > 0 and logn.shape == f.shape == lo.shape == hi.shape
        assert np.all(np.diff(logn) > 0) and logn[0] >= 20.0 and logn[-1] < 23.0
        assert np.all(f > 0) and np.all(lo <= f) and np.all(f <= hi)
        table = np.loadtxt(os.path.join(DATA, "ho21", name))
        drawn = table[1] > 0
        assert np.array_equal(logn, table[0][drawn]) and np.array_equal(f, table[1][drawn])
        assert np.array_equal(
            lo, table[2][drawn]
        ), "row 2: the lower 68 per cent limit, not a 95 per cent row"
        assert np.array_equal(
            hi, table[3][drawn]
        ), "row 3: the upper 68 per cent limit, not a 95 per cent row"
        low, high = (float(v) for v in name[len("cddf_z") : -len(".txt")].replace("225", "2 2.5").replace("253", "2.5 3").replace("34", "3 4").split())  # fmt: skip
        assert low <= z < high, "the model's redshift lies inside the measurement's bin"


def test_the_cddf_reader_maps_the_columns_as_the_collections_own_reader_does():
    """The mapping of ``ho21_cddf`` in the DLA data collection (PROVENANCE.md compares the two step by step),
    restated here in plain numpy: the table transposed, the column density as ten to the power of column 0, f in
    column 1, and with its ``onesigma`` option the limits in columns 2 and 3 (columns 4 and 5 otherwise).
    """
    cddf = _script("fig_cddf_vs_ho21")
    for _, name, _ in cddf.PANELS:
        path = os.path.join(DATA, "ho21", name)
        data = np.loadtxt(path).T
        assert data.shape == (30, 6)
        nhi, f_theirs, lower68, upper68 = 10 ** data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        lower95, upper95 = data[:, 4], data[:, 5]
        logn, f, lo, hi = cddf.read_ho21_cddf(path)
        drawn = f_theirs > 0  # the one difference in drawing: a bin with f = 0 is not drawn here
        assert np.array_equal(10**logn, nhi[drawn]) and np.array_equal(f, f_theirs[drawn])
        assert np.array_equal(lo, lower68[drawn]) and np.array_equal(hi, upper68[drawn])
        assert np.array_equal(f - lo, (f_theirs - lower68)[drawn])  # the lower error bar
        assert np.array_equal(hi - f, (upper68 - f_theirs)[drawn])  # the upper error bar
        if not np.array_equal(
            lower68, lower95
        ):  # and it is the 68 per cent pair that is read, not the 95
            assert not np.array_equal(lo, lower95[drawn])
        assert np.all(lower95[drawn] <= lo) and np.all(hi <= upper95[drawn])
    # the other difference: the bin 3 to 4 takes the file named for it, whose numbers are not those of the
    # 2.5 to 3 file that the collection's reader opens for that range
    files = {label: name for label, name, _ in cddf.PANELS}
    assert files["3 < z < 4"] == "cddf_z34.txt" and files["2.5 < z < 3"] == "cddf_z253.txt"
    assert not np.array_equal(
        np.loadtxt(os.path.join(DATA, "ho21", "cddf_z34.txt")),
        np.loadtxt(os.path.join(DATA, "ho21", "cddf_z253.txt")),
    )


def test_the_class_ratio_table_has_the_columns_the_figure_reads():
    ratio = _script("fig_class_ratio_vs_rogers18")
    table = ratio.read_table(os.path.join(DATA, "class_ratio_z3p0_lf60.csv"))
    k = table["k_s_per_km"]
    assert np.all(np.diff(k) > 0) and k[-1] < 0.03
    for _, stem, name in ratio.PANELS:
        mean, sd = table[f"R_{stem}_addback_mean"], table[f"R_{stem}_addback_sd_over_sims"]
        assert mean.shape == sd.shape == k.shape and np.all(mean > 0) and np.all(sd >= 0)
    assert [name for _, _, name in ratio.PANELS] == ["LLS", "Sub-DLA", "Small-DLA", "Large-DLA"]
    assert ratio.ROGERS_KMIN == pytest.approx(2 * np.pi / 8000.0)


# ---- the scripts


def test_the_class_ratio_figure_needs_no_model(tmp_path):
    ratio = _script("fig_class_ratio_vs_rogers18")
    out = str(tmp_path / "ratio.png")
    assert ratio.main(["--out", out]) == 0
    facts = _facts(out)
    assert facts["z"] == 3.0 and facts["template_box_km_per_s"] == 8000.0
    assert facts["template_extrapolated_below_k_s_per_km"] == pytest.approx(7.85e-4, rel=1e-3)
    assert facts["model"].startswith("none")


def test_the_cddf_figure_runs_on_a_model_with_an_incidence_record(
    tmp_path, release, design, capsys, monkeypatch
):
    cddf = _script("fig_cddf_vs_ho21")
    assert [z for _, _, z in cddf.PANELS] == [2.2, 2.8, 3.4]
    moved = tuple((label, name, z) for (label, name, _), z in zip(cddf.PANELS, SYNTHETIC_Z))
    monkeypatch.setattr(cddf, "PANELS", moved)
    out = str(tmp_path / "cddf.png")
    assert cddf.main(_args(release) + ["--design", design, "--out", out]) == 0
    facts = _facts(out)
    assert facts["parameters"] == C.MIDPOINT and facts["incidence_record"] is True
    assert "github.com/sbird/DLA_data" in facts["measurement_source"]
    assert [p["model_z"] for p in facts["panels"]] == list(SYNTHETIC_Z)
    for panel in facts["panels"]:
        assert panel["design_points"] == 3 and panel["design_points_refused"] == 0
        assert panel["design_points_in_band"] == 3
        assert panel["model_bins_absent"] >= 1, "the first bin is never supported"
        assert (
            "incidence_mf_applied" in panel["model_flags_set"]
        ), "the flags of the drawn line are recorded"
        assert panel["design_points_flagged"] == sum(panel["design_point_flags"].values()) == 0
    assert "3 design points" in capsys.readouterr().out
    assert facts["design_table"] == "a table supplied with --design, not digest-checked"
    assert "composition" not in json.dumps(facts)


def test_the_cddf_figure_refuses_a_model_without_an_incidence_record(tmp_path):
    cddf = _script("fig_cddf_vs_ho21")
    plain = SR.make_release(tmp_path / "plain")
    with pytest.raises(SystemExit, match="no incidence MF record"):
        cddf.main(_args(plain) + ["--out", str(tmp_path / "cddf.png")])
    assert not (tmp_path / "cddf.png").exists()


def test_the_ho21_dndx_table_reads_as_centre_value_and_68_per_cent_limits():
    dndx = _script("fig_dndx_vs_ho21")
    path = os.path.join(DATA, "ho21", "dndx_all.txt")
    z, value, lo, hi = dndx.read_ho21_dndx(path)
    table = np.loadtxt(path)
    assert table.shape == (6, 18)
    assert np.array_equal(z, table[0]) and np.array_equal(value, table[1])
    assert np.array_equal(lo, table[2]), "row 2: the lower 68 per cent limit, not a 95 per cent row"
    assert np.array_equal(hi, table[3]), "row 3: the upper 68 per cent limit, not a 95 per cent row"
    assert np.all(np.diff(z) > 0) and 2.0 < z[0] and z[-1] < 5.0
    # the order the rows were confirmed in: the 95 per cent interval holds the 68 per cent one, which holds the value
    assert (
        np.all(table[4] <= lo)
        and np.all(lo <= value)
        and np.all(value <= hi)
        and np.all(hi <= table[5])
    )


def test_a_dndx_table_that_is_not_six_rows_is_refused(tmp_path):
    dndx = _script("fig_dndx_vs_ho21")
    short = tmp_path / "short.txt"
    np.savetxt(short, [[2.25, 2.75], [0.05, 0.06], [0.04, 0.05]])
    with pytest.raises(SystemExit, match="expected six rows"):
        dndx.read_ho21_dndx(str(short))


def test_the_dndx_figure_runs_on_a_model_with_an_incidence_record(
    tmp_path, release, design, monkeypatch
):
    dndx = _script("fig_dndx_vs_ho21")
    assert dndx.REDSHIFTS[0] == 2.2 and dndx.REDSHIFTS[-1] == 4.6 and len(dndx.REDSHIFTS) == 13
    assert [(name, column) for name, column, _, _ in dndx.CLASSES] == [
        ("DLA", 2),
        ("subDLA", 1),
        ("LLS", 0),
    ]
    assert hp.HCD_CLASSES == ("LLS", "subDLA", "DLA"), "the columns of pred.dndx the script indexes"
    monkeypatch.setattr(dndx, "REDSHIFTS", SYNTHETIC_Z)
    out = str(tmp_path / "dndx.png")
    assert dndx.main(_args(release) + ["--design", design, "--out", out]) == 0
    facts = _facts(out)
    assert (
        "68 per cent interval" in facts["measurement"]
        and "18 redshift bins" in facts["measurement"]
    )
    assert facts["design_points"] == 3 and facts["design_points_refused"] == 0
    assert facts["redshifts"] == list(dndx.REDSHIFTS) and facts["parameters"] == C.MIDPOINT
    assert facts["heldout_error"].startswith(
        "not drawn"
    ), "this synthetic record carries no held-out error"
    with_error = SR.make_release(tmp_path / "with_error", incidence=True, heldout=True)
    out2 = str(tmp_path / "dndx_with_error.png")
    monkeypatch.setattr(
        dndx, "REDSHIFTS", (2.6, 3.0, 5.0)
    )  # 3.0 and 5.0 have a synthetic estimate, 2.6 has none
    assert dndx.main(_args(with_error) + ["--no-band", "--out", out2]) == 0
    assert _facts(out2)["heldout_error"].startswith(
        "drawn on the model lines: pred.dndx_err times pred.dndx"
    )


def test_the_dndx_figure_refuses_a_model_without_an_incidence_record(tmp_path):
    dndx = _script("fig_dndx_vs_ho21")
    plain = SR.make_release(tmp_path / "plain")
    with pytest.raises(SystemExit, match="no incidence MF record"):
        dndx.main(_args(plain) + ["--out", str(tmp_path / "dndx.png")])
    assert not (tmp_path / "dndx.png").exists()


def test_an_absent_bin_breaks_the_line_and_the_band():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    x = np.arange(5.0)
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])  # the bin in the middle is absent
    (drawn,) = C.line(ax, x, y)
    assert len(drawn.get_ydata()) == 5 and np.isnan(
        drawn.get_ydata()[2]
    ), "the gap is kept, not bridged"
    assert len(ax.lines) == 1, "no bin stands alone here, so no marker is added"
    alone = np.array(
        [1.0, 2.0, np.nan, 4.0, np.nan]
    )  # the bin at index 3 has no supported neighbour
    C.line(ax, x, alone)
    marker = ax.lines[-1]
    assert marker.get_marker() == "o" and marker.get_linestyle() == "None"
    assert list(marker.get_xdata()) == [3.0] and list(marker.get_ydata()) == [4.0]
    pieces = C.band(ax, x, 0.5 * y, 2.0 * y).get_paths()
    assert len(pieces) == 2, "the band has one piece on each side of the absent bin"
    for piece in pieces:
        assert not np.any((piece.vertices[:, 0] > 1.0) & (piece.vertices[:, 0] < 3.0))
    plt.close(fig)


def test_the_band_is_drawn_over_the_committed_design_points_unless_told_otherwise(tmp_path):
    import argparse

    ap = argparse.ArgumentParser()
    C.add_design_arguments(ap)
    committed = C.design_points(ap.parse_args([]), hp.PARAM_NAMES)
    assert len(committed) == 60 and all(list(point) == list(hp.PARAM_NAMES) for point in committed)
    assert len({tuple(point.values()) for point in committed}) == 60, "sixty different points"
    assert C.design_points(ap.parse_args(["--no-band"]), hp.PARAM_NAMES) is None
    other = tmp_path / "other.csv"
    other.write_text(
        ",".join(hp.PARAM_NAMES)
        + "\n"
        + ",".join(repr(float(v)) for v in C.MIDPOINT.values())
        + "\n",
        encoding="utf-8",
    )
    assert C.design_points(ap.parse_args(["--design", str(other)]), hp.PARAM_NAMES) == [C.MIDPOINT]
    with pytest.raises(SystemExit, match="contradict"):
        C.design_points(ap.parse_args(["--no-band", "--design", str(other)]), hp.PARAM_NAMES)


def test_no_band_draws_the_line_and_the_points_only(tmp_path, release, monkeypatch):
    cddf = _script("fig_cddf_vs_ho21")
    moved = tuple((label, name, z) for (label, name, _), z in zip(cddf.PANELS, SYNTHETIC_Z))
    monkeypatch.setattr(cddf, "PANELS", moved)
    out = str(tmp_path / "cddf.png")
    assert cddf.main(_args(release) + ["--no-band", "--out", out]) == 0
    facts = _facts(out)
    assert all("design_points" not in panel for panel in facts["panels"])
    assert facts["design_table"].startswith("none")


def test_with_no_option_the_band_is_drawn_over_the_sixty_committed_points(
    tmp_path, release, monkeypatch
):
    cddf = _script("fig_cddf_vs_ho21")
    monkeypatch.setattr(cddf, "PANELS", cddf.PANELS[:1])  # one panel is enough, and faster
    moved = tuple((label, name, SYNTHETIC_Z[0]) for (label, name, _) in cddf.PANELS)
    monkeypatch.setattr(cddf, "PANELS", moved)
    out = str(tmp_path / "cddf.png")
    assert cddf.main(_args(release) + ["--out", out]) == 0
    facts = _facts(out)
    (panel,) = facts["panels"]
    assert panel["design_points"] == 60
    assert panel["design_points_in_band"] + panel["design_points_refused"] == 60
    digest = _sums()["priya_lf_design_60.csv"]
    assert facts["design_table"] == f"docs/figures/data/priya_lf_design_60.csv, sha256 {digest}"


def test_a_design_table_with_another_header_is_refused(tmp_path):
    path = tmp_path / "design.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="the header must be"):
        C.read_design(str(path), hp.PARAM_NAMES)


def test_the_band_reports_flagged_points_and_drops_none(release):
    emu = hp.load_release(str(release["root"]), release["sha"])
    # far outside any training range: the predictor flags it or refuses it
    far = dict(C.MIDPOINT, ns=5.0)
    drawn = emu.predict(C.MIDPOINT, z=3.0, tau0_amp=1.0, dtau0=0.0)
    lo, hi, flagged, refused = C.envelope(
        emu, [dict(C.MIDPOINT), far], 3.0, "dndx", C.flags_set(drawn)
    )
    assert [i for i, _ in flagged] + [i for i, _ in refused] == [1]
    assert np.all(lo <= drawn.dndx) and np.all(drawn.dndx <= hi)
    # every flag the drawn line does not have is reported, whatever its name: with no baseline, the flags that
    # every prediction of this model carries are reported too
    _, _, everything, _ = C.envelope(emu, [dict(C.MIDPOINT)], 3.0, "dndx")
    assert everything == [(0, C.flags_set(drawn))] and "incidence_mf_applied" in everything[0][1]
    assert C.flag_counts([(0, ["a", "b"]), (3, ["b"])]) == {"a": 1, "b": 2}


def test_a_band_with_every_point_refused_stops_with_a_message(release):
    emu = hp.load_release(str(release["root"]), release["sha"])
    not_a_point = dict(C.MIDPOINT, ns=float("nan"))
    with pytest.raises(SystemExit, match="refused every design point"):
        C.envelope(emu, [not_a_point], 3.0, "dndx")
