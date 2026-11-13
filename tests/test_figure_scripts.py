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
    assert sorted(ho21) == ["ho21/cddf_z225.txt", "ho21/cddf_z253.txt", "ho21/cddf_z34.txt"]
    for name, digest in ho21.items():
        assert os.path.basename(name) in text and digest in text, name
    for needed in ("github.com/jibanCat/DLA_data", "10.1093/mnras/stab2169", "no licence file"):
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
        low, high = (float(v) for v in name[len("cddf_z") : -len(".txt")].replace("225", "2 2.5").replace("253", "2.5 3").replace("34", "3 4").split())  # fmt: skip
        assert low <= z < high, "the model's redshift lies inside the measurement's bin"


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
    assert [p["model_z"] for p in facts["panels"]] == list(SYNTHETIC_Z)
    for panel in facts["panels"]:
        assert panel["design_points"] == 3 and panel["design_points_refused"] == 0
        assert panel["model_bins_absent"] >= 1, "the first bin is never supported"
    assert "3 design points" in capsys.readouterr().out
    assert "composition" not in json.dumps(facts)


def test_the_cddf_figure_refuses_a_model_without_an_incidence_record(tmp_path):
    cddf = _script("fig_cddf_vs_ho21")
    plain = SR.make_release(tmp_path / "plain")
    with pytest.raises(SystemExit, match="no incidence MF record"):
        cddf.main(_args(plain) + ["--out", str(tmp_path / "cddf.png")])
    assert not (tmp_path / "cddf.png").exists()


def test_the_dndx_figure_draws_no_measurement_unless_told_what_its_rows_are(
    tmp_path, release, design, capsys, monkeypatch
):
    dndx = _script("fig_dndx_vs_ho21")
    assert dndx.REDSHIFTS[0] == 2.2 and dndx.REDSHIFTS[-1] == 4.6 and len(dndx.REDSHIFTS) == 13
    monkeypatch.setattr(dndx, "REDSHIFTS", SYNTHETIC_Z)
    out = str(tmp_path / "dndx.png")
    assert dndx.main(_args(release) + ["--design", design, "--out", out]) == 0
    facts = _facts(out)
    assert facts["measurement"].startswith("not drawn")
    assert facts["design_points"] == 3 and facts["design_points_refused"] == 0
    assert facts["redshifts"] == list(dndx.REDSHIFTS) and facts["parameters"] == C.MIDPOINT
    capsys.readouterr()

    table = tmp_path / "dndx_table.txt"  # synthetic: six rows of four numbers
    centre = np.array([0.05, 0.06, 0.07, 0.08])
    np.savetxt(
        table,
        [[2.25, 2.75, 3.25, 3.75], centre, 0.9 * centre, 1.1 * centre, 0.8 * centre, 1.2 * centre],
    )
    with pytest.raises(SystemExit) as refused:
        dndx.main(_args(release) + ["--ho21-dndx", str(table), "--out", out])
    assert refused.value.code == 2
    assert "the meaning of its rows is not recorded" in capsys.readouterr().err
    told = ["--ho21-dndx", str(table), "--ho21-dndx-rows-are-centre-and-intervals"]
    assert dndx.main(_args(release) + told + ["--out", out]) == 0
    assert "rows read as centre and intervals" in _facts(out)["measurement"]


def test_a_design_table_with_another_header_is_refused(tmp_path):
    path = tmp_path / "design.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="the header must be"):
        C.read_design(str(path), hp.PARAM_NAMES)


def test_the_band_reports_flagged_points_and_drops_none(release):
    emu = hp.load_release(str(release["root"]), release["sha"])
    far = dict(
        C.MIDPOINT, ns=5.0
    )  # far outside any training range: the predictor flags it or refuses it
    lo, hi, flagged, refused = C.envelope(emu, [dict(C.MIDPOINT), far], 3.0, "dndx")
    assert [i for i, _ in flagged] + [i for i, _ in refused] == [1]
    one = emu.predict(C.MIDPOINT, z=3.0, tau0_amp=1.0, dtau0=0.0).dndx
    assert np.all(lo <= one) and np.all(one <= hi)
