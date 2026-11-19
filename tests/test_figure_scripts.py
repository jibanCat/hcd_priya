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
NOT_DATA = {
    "SHA256SUMS",
    "README.md",
    "ho21/PROVENANCE.md",
    "lls_subdla/PROVENANCE.md",
    "lls_incidence/PROVENANCE.md",
}  # describe the data
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


def test_the_lls_subdla_provenance_names_each_file_with_its_digest_and_its_source():
    """PROVENANCE.md names each file with its digest; the collection file with its commits; the two built files with
    their papers' arXiv sources (the .tex digests and table lines); and the two files they replace.
    """
    with open(os.path.join(DATA, "lls_subdla", "PROVENANCE.md"), encoding="utf-8") as fh:
        text = fh.read()
    files = {n: d for n, d in _sums().items() if n.startswith("lls_subdla/")}
    assert sorted(files) == [
        "lls_subdla/omeara2007_fn_all19p0.dat",
        "lls_subdla/prochaska2010_lls_region.dat",
        "lls_subdla/zafar_2013.txt",
    ]
    for name, digest in files.items():
        assert os.path.basename(name) in text and digest in text, name
    for needed in (
        "github.com/sbird/DLA_data",  # where the subDLA table originates
        "90104dd49364f0478e7dccd52675a41794af4033",  # the commit that added it there
        "github.com/jibanCat/DLA_data",  # the fork it was fetched from
        "daa595c692254c93b86009318a6e2b029db67112",
        "no licence file",
        "10.1051/0004-6361/201321154",
        "10.1086/510711",
        "10.1088/0004-637X/718/1/392",
        "astro-ph/0610726v2",  # the source of the O'Meara table, and its .tex digest
        "5dd7e2e9891b2d5c16ffda8620ddeae74c5916a8c005fb303c4999d11ed84bb8",
        "0912.0292v1",  # the source of the region, and its .tex digest
        "f236288f014668c136b1f4a5b4e5d4c19b741f0538d3c090edeabffe2bd574b9",
        "summary.dat",  # the two files these replace, and why
        "prochaska_lls.dat",
        "3498694e59701b72c738e5eef950c62b73683c1e",
        "Omega_m = 0.3",  # the cosmology of each, and the conversion
        "Delta X(own Omega_m) / Delta X(line Omega_m)",
        "1307.0602v2",  # the Zafar file compared with the paper's table, and the digest of the source read
        "ad3866f8002f9826702477775e01cb4faf7d0553a6450833b083b411ff79ba35",
    ):
        assert needed in text, needed


def test_the_zafar_file_is_the_papers_table():
    """zafar_2013.txt against the table of the combined sub-DLA sample of Zafar et al. (2013) (arXiv 1307.0602v2,
    lines 267 to 272), restated from the paper: per log N_HI range, log f with its printed plus and minus errors. The
    file gives the middle of each range and its half width, then log f, the plus error and the minus error.
    """
    table = [  # (log N_HI range, log f, plus, minus)
        ((19.00, 19.30), -20.43, 0.09, 0.10),
        ((19.30, 19.60), -20.75, 0.09, 0.10),
        ((19.60, 19.90), -21.15, 0.09, 0.13),
        ((19.90, 20.10), -21.30, 0.09, 0.12),
        ((20.10, 20.30), -21.61, 0.11, 0.16),
    ]
    data = np.loadtxt(os.path.join(DATA, "lls_subdla", "zafar_2013.txt"))
    assert data.shape == (5, 5)
    for row, ((a, b), log_f, plus, minus) in zip(data, table):
        np.testing.assert_allclose(
            row, [(a + b) / 2, (b - a) / 2, log_f, plus, minus], rtol=0, atol=1e-12
        )


def test_the_two_built_tables_are_the_papers_published_values():
    """omeara2007_fn_all19p0.dat holds row ALL-19.0 of the f(N) table of O'Meara et al. (2007) (their Table 5): log f
    and its errors per bin, the bin edges, and the bin centre. prochaska2010_lls_region.dat holds the corners of the
    region between the four extreme power laws of the "Conservative Range" of the f(N) table of Prochaska, O'Meara
    and Worseck (2010): each row's line log f = log f19 + beta (log N - 19), beta = (log f19 - log k) / 19, for log N
    from 17.5 to 19. The published numbers are restated here from the papers."""
    om = np.loadtxt(os.path.join(DATA, "lls_subdla", "omeara2007_fn_all19p0.dat"), ndmin=2)
    edges = np.array([19.0, 19.3, 19.6, 19.9, 20.3])
    logf = np.array([-20.17, -20.70, -21.04, -21.48])
    err_up, err_down = np.array([0.10, 0.13, 0.14, 0.13]), np.array([0.10, 0.14, 0.15, 0.14])
    assert om.shape == (4, 6)
    assert np.allclose(om[:, 2], edges[:-1], rtol=0, atol=1e-12)
    assert np.allclose(om[:, 3], edges[1:], rtol=0, atol=1e-12)
    assert np.allclose(om[:, 0], (edges[:-1] + edges[1:]) / 2, rtol=0, atol=1e-12)
    assert np.allclose(om[:, 1], logf, rtol=0, atol=1e-12)
    assert np.allclose(om[:, 4], logf - err_down, rtol=0, atol=1e-12)
    assert np.allclose(om[:, 5], logf + err_up, rtol=0, atol=1e-12)
    rows = [(-20.25, -4.1), (-20.25, 4.7), (-19.85, -18.0), (-19.85, -4.6)]  # (log f19, log k)

    def lines(log_n):
        return np.array([f19 + (f19 - k) / 19.0 * (log_n - 19.0) for f19, k in rows])

    corners = np.loadtxt(os.path.join(DATA, "lls_subdla", "prochaska2010_lls_region.dat"), ndmin=2)
    assert corners.shape == (6, 2)
    # the corners in the file's order: at log N 19 the lowest and the highest line, the upper kink, at 17.5 the
    # highest and the lowest, the lower kink
    envelope = ["min", "max", "max", "max", "min", "min"]
    for (log_n, log_f), side in zip(corners, envelope):
        values = lines(log_n)
        edge = values.max() if side == "max" else values.min()
        assert abs(log_f - edge) < 1e-6, (log_n, log_f, side, edge)
    assert list(corners[:, 0][[0, 1, 3, 4]]) == [19.0, 19.0, 17.5, 17.5]
    for i in (2, 5):  # each kink is where two of the lines cross
        values = lines(corners[i, 0])
        edge = values.max() if envelope[i] == "max" else values.min()
        assert np.sum(np.abs(values - edge) < 1e-5) == 2, corners[i]


def test_the_conversion_to_the_line_cosmology():
    """Delta X against two closed forms (Omega_m = 1: the integral of (1 + z)^(1/2); Omega_m = 0: of (1 + z)^2), the
    line's Omega_m from the midpoint, a factor of 1 for a measurement already in the line's cosmology, and the sign
    of the factor: a larger Omega_m gives a shorter Delta X."""
    z1, z2 = 2.0, 4.0
    assert C.absorption_distance(z1, z2, 1.0) == pytest.approx(
        2.0 / 3.0 * ((1 + z2) ** 1.5 - (1 + z1) ** 1.5), rel=1e-13
    )
    assert C.absorption_distance(z1, z2, 0.0) == pytest.approx(
        ((1 + z2) ** 3 - (1 + z1) ** 3) / 3.0, rel=1e-13
    )
    assert C.OMEGA_M_LINE == C.MIDPOINT["omegamh2"] / C.MIDPOINT["hub"] ** 2
    assert C.to_line_cosmology(z1, z2, C.OMEGA_M_LINE) == 1.0
    assert (
        C.to_line_cosmology(z1, z2, C.OMEGA_M_LINE + 0.05)
        < 1.0
        < C.to_line_cosmology(z1, z2, C.OMEGA_M_LINE - 0.05)
    )
    # each measurement's Omega_m and redshift range as its PROVENANCE.md gives them, written out here so that a
    # change of either constant is also a change of this test
    assert C.OMEGA_M_OF == {"ho21": 0.279, "zafar13": 0.3, "omeara07": 0.3, "prochaska10": 0.3}
    assert C.Z_RANGE_OF == {
        "zafar13": (1.51, 5.00),
        "omeara07": (1.68, 4.47),
        "prochaska10": (3.3, 4.4),
    }


def test_the_lls_incidence_provenance_names_the_file_its_sources_and_what_is_drawn():
    """PROVENANCE.md names the file with its digest, each paper with its DOI and the arXiv source read, the rows the
    figure draws and why the others are left out, the Poisson orientation, and the published table that the
    catalogue check reproduces (a check, not a source of values)."""
    with open(os.path.join(DATA, "lls_incidence", "PROVENANCE.md"), encoding="utf-8") as fh:
        text = fh.read()
    files = {n: d for n, d in _sums().items() if n.startswith("lls_incidence/")}
    assert sorted(files) == ["lls_incidence/lls_lz_literature.dat"]
    for name, digest in files.items():
        assert os.path.basename(name) in text and digest in text, name
    for needed in (
        "10.1088/0004-637X/718/1/392",  # Prochaska, O'Meara and Worseck 2010: the journal's Table 4
        "Table 4",
        "10.1088/0004-637X/765/2/137",  # O'Meara et al. 2013
        "1204.3093v2",
        "10.1088/0004-637X/736/1/42",  # Ribaudo, Lehner and Howk 2011
        "1105.0659v1",
        "10.1088/0004-637X/775/1/78",  # Fumagalli et al. 2013
        "1308.1101v2",
        "10.1093/mnras/sty2762",  # Crighton et al. 2019
        "1810.10040v1",
        "10.1093/mnras/staa2388",  # Fumagalli, Fotopoulou and Thomson 2020
        "2009.03322v1",
        "tau_912 >= 2",
        "N_HI >= 10^17.5",
        "What the figure draws",
        "Poisson interval",
        "no drawn row contains another",
        "344 quasars",
        "drawn as printed (symmetric)",
        "may include some just below tau = 2",
        "How to reproduce the table",
        "no value in the file comes from anywhere else",
    ):
        assert needed in " ".join(text.split()), needed  # line breaks read as spaces


def test_the_lls_incidence_rows_drawn_are_the_papers_published_values():
    """The twelve rows the figure draws, restated from the papers in their printed notation: (z_lo, z_hi, l(z),
    the printed minus error, the printed plus error). The file's err_lo is the printed minus error and err_hi the
    printed plus one, as printed."""
    fig = _script("fig_dndx_vs_ho21")
    printed = {
        "prochaska10_t4": [  # l(z)^{+a}_{-b}: the journal prints the smaller error with the plus sign
            (3.30, 3.40, 1.31, 0.60, 0.43),
            (3.40, 3.50, 2.18, 0.49, 0.41),
            (3.50, 3.65, 1.60, 0.27, 0.23),
            (3.65, 3.90, 2.04, 0.31, 0.27),
            (3.90, 4.10, 2.84, 0.63, 0.52),
            (4.10, 4.40, 3.22, 0.84, 0.68),
        ],
        "omeara13_t5_comb": [(2.00, 2.60, 0.92, 0.18, 0.18)],
        "fumagalli13_nc": [(2.6, 3.0, 1.21, 0.28, 0.28)],
        "crighton19_corr": [
            (3.75, 4.40, 2.21, 0.34, 0.40),
            (4.40, 4.70, 2.20, 0.39, 0.47),
            (4.70, 5.40, 2.95, 0.49, 0.58),
        ],
        "fumagalli20_dr16": [(3.3, 3.6, 2.32, 0.08, 0.08)],
    }
    rows = fig.read_lls(os.path.join(DATA, *fig.LLS_FILE))
    assert len(rows) == 18
    for source, want in printed.items():
        got = [row[1:] for row in rows if row[0] == source]
        assert got == want, source
    assert set(printed) == set(fig.LLS_DRAWN)


def test_the_lls_incidence_rows_left_out_share_absorbers_with_rows_drawn():
    """No drawn row contains another: the rows not drawn are the WFC3 and ACS rows of O'Meara et al. (2013) and the
    row of Ribaudo et al. (2011) below z 2.6 (inside the combined O'Meara row, whose bin covers theirs), and the rows
    of Ribaudo et al. (2011) above z 3.5 (the SDSS-DR7 absorbers of Prochaska et al. 2010). The combined O'Meara row
    is labelled with Ribaudo et al. too, and only the rows of Prochaska et al. (2010) have their printed errors
    swapped."""
    fig = _script("fig_dndx_vs_ho21")
    rows = fig.read_lls(os.path.join(DATA, *fig.LLS_FILE))
    left_out = sorted({row[0] for row in rows} - set(fig.LLS_DRAWN))
    assert left_out == [
        "omeara13_t5_acs",
        "omeara13_t5_wfc3",
        "ribaudo11_t5_r2",
        "ribaudo11_t7_rp10",
    ]
    assert [fig.LLS_DRAWN[s][2] for s in fig.LLS_DRAWN] == [True, False, False, False, False]
    assert fig.LLS_DRAWN["omeara13_t5_comb"][0].endswith("with Ribaudo, Lehner and Howk (2011)")
    assert fig.LLS_HOLLOW == {"fumagalli20_dr16"}
    markers = [fig.LLS_DRAWN[s][1] for s in fig.LLS_DRAWN]
    assert len(set(markers)) == len(markers) and "o" not in markers[:-1]
    # each legend label begins with what the sample counts, never with the class name LLS
    for source, (paper, _, _) in fig.LLS_DRAWN.items():
        label = fig.lls_label(source)
        assert label.endswith(", " + paper) and not label.startswith("LLS")
        own = source == "fumagalli20_dr16"
        assert label.startswith(r"$\tau \gtrsim 2$" if own else r"$N_\mathrm{HI} \geq 10^{17.5}$")


def test_the_lls_incidence_rows_are_drawn_per_unit_x_with_the_larger_error_upward():
    """l(X) = l(z) Delta z / Delta X(line) over each row's bin, the errors times the same factor (Delta X checked
    against a trapezoid sum on a fine grid); in the rows of Prochaska et al. (2010) the printed errors swapped, the
    larger upward; in the others as printed (the combined O'Meara row has two equal errors)."""
    fig = _script("fig_dndx_vs_ho21")
    rows = fig.read_lls(os.path.join(DATA, *fig.LLS_FILE))
    points = fig.lls_points(rows)
    assert {s: len(p) for s, p in points.items()} == {
        "prochaska10_t4": 6,
        "omeara13_t5_comb": 1,
        "fumagalli13_nc": 1,
        "crighton19_corr": 3,
        "fumagalli20_dr16": 1,
    }
    om = C.OMEGA_M_LINE
    for source, z_lo, z_hi, value, err_lo, err_hi in rows:
        if source not in fig.LLS_DRAWN:
            continue
        (point,) = [p for p in points[source] if p[0] == z_lo and p[1] == z_hi]
        grid = np.linspace(z_lo, z_hi, 20001)
        delta_x = np.trapezoid((1 + grid) ** 2 / np.sqrt(om * (1 + grid) ** 3 + 1 - om), grid)
        factor = (z_hi - z_lo) / delta_x
        assert point[5] == pytest.approx(factor, rel=1e-8) and 0.2 < factor < 0.4
        np.testing.assert_allclose(point[2], point[5] * value, rtol=1e-14, atol=0)
        down, up = (err_hi, err_lo) if fig.LLS_DRAWN[source][2] else (err_lo, err_hi)
        np.testing.assert_allclose(point[3:5], [point[5] * down, point[5] * up], rtol=1e-14, atol=0)
        assert point[4] >= point[3], "in every drawn row the larger error is upward"
    # the swap moved something: in every row of Prochaska et al. (2010) the error drawn upward is the larger
    p = points["prochaska10_t4"]
    assert np.all(p[:, 4] > p[:, 3])
    # and the combined O'Meara row is drawn as printed: two equal errors
    (p,) = points["omeara13_t5_comb"]
    assert p[3] == p[4] == p[5] * 0.18


def test_the_lls_reader_refuses_what_it_cannot_draw(tmp_path):
    fig = _script("fig_dndx_vs_ho21")
    per_x = tmp_path / "per_x.dat"
    per_x.write_text("# header\n3.0\t3.5\t0.5\t0.1\t0.1\tlX\t0.3\tsynthetic\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="reads l\\(z\\) rows only"):
        fig.read_lls(str(per_x))
    short = tmp_path / "short.dat"
    short.write_text("3.0\t3.5\t0.5\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="eight tab-separated fields"):
        fig.read_lls(str(short))
    # a row of a sample whose printed errors are swapped, with the printed plus error already the larger
    rows = [("prochaska10_t4", 3.3, 3.4, 1.0, 0.2, 0.3)]
    with pytest.raises(SystemExit, match="z 3.3 to 3.4: .* nothing to swap"):
        fig.lls_points(rows)
    with pytest.raises(SystemExit, match="no row of"):
        fig.lls_points([("prochaska10_t4", 3.3, 3.4, 1.0, 0.3, 0.2)])


def test_the_count_above_the_threshold_from_a_synthetic_distribution():
    """The count above 10^17.5 from a CDDF of bins 0.2 wide centred on 17.1, 17.3, ..., 22.9 (the bin 17.4 to 17.6
    holds the threshold): for a constant f it is f (10^23 - 10^17.5); a bin with no value is left out; the
    threshold on a bin edge; one value per redshift for a (n_z, Nbin) array; and the cross-check from class lines
    built from the same distribution gives the same number."""
    fig = _script("fig_dndx_vs_ho21")
    log_nhi = np.round(np.arange(17.1, 23.0, 0.2), 10)
    assert log_nhi.size == 30
    f = np.full(30, 1e-17)  # synthetic
    np.testing.assert_allclose(
        fig.cumulative_dndx(f, log_nhi), 1e-17 * (10**23.0 - 10**17.5), rtol=1e-12, atol=0
    )
    only_partial = np.where(log_nhi == 17.5, 1e-17, 0.0)
    np.testing.assert_allclose(
        fig.cumulative_dndx(only_partial, log_nhi),
        1e-17 * (10**17.6 - 10**17.5),
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        fig.cumulative_dndx(f, log_nhi, threshold=17.6),
        1e-17 * (10**23.0 - 10**17.6),
        rtol=1e-12,
        atol=0,
    )
    gap = f.copy()
    gap[-1] = np.nan  # the bin 22.8 to 23.0 has no value
    np.testing.assert_allclose(
        fig.cumulative_dndx(gap, log_nhi), 1e-17 * (10**22.8 - 10**17.5), rtol=1e-12, atol=0
    )
    below = np.where(log_nhi < 17.4, np.nan, f)  # NaN below the threshold changes nothing
    assert fig.cumulative_dndx(below, log_nhi) == fig.cumulative_dndx(f, log_nhi)
    assert np.isnan(fig.cumulative_dndx(np.full(30, np.nan), log_nhi))
    two = np.vstack([f, 2 * f])
    np.testing.assert_allclose(
        fig.cumulative_dndx(two, log_nhi),
        [1e-17 * (10**23 - 10**17.5), 2e-17 * (10**23 - 10**17.5)],
        rtol=1e-12,
        atol=0,
    )
    # class lines from the same distribution: LLS 17.2 to 19, subDLA 19 to 20.3, DLA from 20.3
    rising = 1e-17 * (1.0 + np.arange(30))  # synthetic, different in every bin
    lls, sub, dla = (
        fig.sum_over(rising, fig.widths_inside(log_nhi, a, b))
        for a, b in ((17.2, 19.0), (19.0, 20.3), (20.3, np.inf))
    )
    check = fig.cross_check(np.array([lls, sub, dla]), rising, log_nhi, 17.2)
    np.testing.assert_allclose(check, fig.cumulative_dndx(rising, log_nhi), rtol=1e-12, atol=0)
    with pytest.raises(SystemExit, match="not evenly spaced"):
        fig.widths_inside(np.array([17.1, 17.3, 17.6]), 17.5)


def test_the_class_spans_in_the_labels_are_the_catalogues():
    from hcd_analysis import catalog

    fig = _script("fig_dndx_vs_ho21")
    spans = {name: span for name, _, _, _, span in fig.CLASSES}
    assert f"10^{{{catalog.LOG_NHI_DLA_MIN:g}}}" in spans["DLA"]
    assert f"10^{{{catalog.LOG_NHI_SUBDLA_MIN:g}}}" in spans["subDLA"]
    assert f"10^{{{catalog.LOG_NHI_DLA_MIN:g}}}" in spans["subDLA"]
    assert f"10^{{{catalog.LOG_NHI_LLS_MIN:g}}}" in spans["LLS"]
    assert f"10^{{{catalog.LOG_NHI_SUBDLA_MIN:g}}}" in spans["LLS"]
    assert fig.LLS_THRESHOLD == 17.5


def test_the_lls_subdla_tables_read_as_the_collections_own_readers_read_them():
    """The mappings of ``zafar_data``, ``omeara_data`` (with ``format_error_bars``) and ``prochaska_10_data`` of the DLA
    data collection (PROVENANCE.md compares them step by step), restated here in plain numpy, against the readers of the
    CDDF figure, on the three files (the two built from the papers' tables keep the collection's formats): the same
    points, the same error bars, the same polygon. The values of f are near 1e-20, so every comparison is relative,
    with no absolute tolerance."""
    cddf = _script("fig_cddf_vs_ho21")

    def same(a, b):
        return np.allclose(a, b, rtol=1e-12, atol=0.0)

    # zafar_data: N = 10**c0; f = 10**c2; x errors 10**(c0 +- c1) about N; y errors from 10**(c2 + c3) and 10**(c2 - c4)
    path = os.path.join(DATA, "lls_subdla", "zafar_2013.txt")
    data = np.loadtxt(path)
    assert data.shape == (5, 5)
    nhi, f_theirs = 10 ** data[:, 0], 10 ** data[:, 2]
    upper_x, lower_x = 10 ** (data[:, 1] + data[:, 0]) - nhi, nhi - 10 ** (-data[:, 1] + data[:, 0])
    upper_y, lower_y = 10 ** (data[:, 3] + data[:, 2]) - f_theirs, f_theirs - 10 ** (
        -data[:, 4] + data[:, 2]
    )
    logn, half, f, lo, hi = cddf.read_zafar13(path)
    assert np.array_equal(10**logn, nhi) and np.array_equal(f, f_theirs)
    assert same(10 ** (logn + half) - 10**logn, upper_x) and same(
        10**logn - 10 ** (logn - half), lower_x
    )
    assert same(hi - f, upper_y) and same(f - lo, lower_y)
    # column 3 is the upper error and column 4 the lower one, and the file has rows where they differ
    assert np.any(data[:, 3] != data[:, 4])
    assert same(np.log10(hi) - np.log10(f), data[:, 3]) and same(
        np.log10(f) - np.log10(lo), data[:, 4]
    )
    # omeara_data: format_error_bars on 10**data: x, y, then the bin edges and the limits of f as distances
    path = os.path.join(DATA, "lls_subdla", "omeara2007_fn_all19p0.dat")
    data = 10 ** np.loadtxt(path)
    assert data.shape == (4, 6)
    logn, lo_n, hi_n, f, f_lo, f_hi = cddf.read_omeara07(path)
    assert same(10**logn, data[:, 0]) and same(f, data[:, 1])
    assert same(10**logn - 10**lo_n, data[:, 0] - data[:, 2])
    assert same(10**hi_n - 10**logn, data[:, 3] - data[:, 0])
    assert same(f - f_lo, data[:, 1] - data[:, 4]) and same(f_hi - f, data[:, 5] - data[:, 1])
    assert np.all(lo_n < logn) and np.all(logn < hi_n) and np.all(f_lo < f) and np.all(f < f_hi)
    # prochaska_10_data: ax.fill(10**c0, 10**c1), the corners in the file's order
    path = os.path.join(DATA, "lls_subdla", "prochaska2010_lls_region.dat")
    data = np.loadtxt(path)
    assert data.shape == (6, 2)
    logn, f = cddf.read_prochaska10(path)
    assert np.array_equal(10**logn, 10 ** data[:, 0]) and np.array_equal(f, 10 ** data[:, 1])


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
    table = ratio.read_table(os.path.join(DATA, ratio.RANGES_FILE))
    k = table["k_s_per_km"]
    assert np.all(np.diff(k) > 0) and k[-1] < 0.03
    assert [s[0] for s in ratio.RANGE_SETS] == [
        "sims60_allrungs",
        "sims60_rung1",
    ], "the wider set is drawn first"
    for _, stem, _ in ratio.PANELS:
        cols = {
            (s, q): table[f"R_{stem}_{s}_{q}"]
            for s, _, _ in ratio.RANGE_SETS
            for q in ("min", "p16", "p50", "p84", "max")
        }
        for (
            s,
            _,
            _,
        ) in ratio.RANGE_SETS:  # each set is ordered: min <= p16 <= p50 <= p84 <= max, all positive
            ordered = [cols[(s, q)] for q in ("min", "p16", "p50", "p84", "max")]
            assert all(np.all(a <= b) for a, b in zip(ordered, ordered[1:])) and np.all(
                ordered[0] > 0
            )
        # the 60 simulations at one rung are among the 60 simulations at all rungs: the wide range holds the narrow
        assert np.all(cols[("sims60_allrungs", "min")] <= cols[("sims60_rung1", "min")])
        assert np.all(cols[("sims60_rung1", "max")] <= cols[("sims60_allrungs", "max")])
    assert [name for _, _, name in ratio.PANELS] == ["LLS", "Sub-DLA", "Small-DLA", "Large-DLA"]
    assert ratio.ROGERS_KMIN == pytest.approx(2 * np.pi / 8000.0)


def test_the_template_band_is_the_published_line_with_its_amplitude_scaled():
    from hcd_analysis.hcd_template import class_ratio

    ratio = _script("fig_class_ratio_vs_rogers18")
    assert ratio.ALPHA_RANGE == (0.5, 2.0)
    k = np.geomspace(5e-4, 0.03, 40)
    published = class_ratio(k, ratio.Z)
    constant = class_ratio(np.array([1.0]), ratio.Z)
    for name, (line, lo, hi) in ratio.template_with_band(k, ratio.Z).items():
        c = float(constant[name][0])
        np.testing.assert_array_equal(
            line, published[name]
        )  # the line is Eq. 6 as published, untouched
        term = published[name] - c
        np.testing.assert_allclose(lo, c + 0.5 * term, rtol=1e-12)
        np.testing.assert_allclose(hi, c + 2.0 * term, rtol=1e-12)
        assert np.all(lo <= line) and np.all(line <= hi) and np.all(term > 0)
        # far above the figure's range the ratio is the constant: the band closes onto it
        far = ratio.template_with_band(np.array([0.5]), ratio.Z)[name]
        assert far[1][0] == pytest.approx(c, rel=1e-6) and far[2][0] == pytest.approx(c, rel=1e-6)


# ---- the scripts


def test_the_class_ratio_figure_needs_no_model(tmp_path):
    ratio = _script("fig_class_ratio_vs_rogers18")
    out = str(tmp_path / "ratio.png")
    assert ratio.main(["--out", out]) == 0
    facts = _facts(out)
    assert facts["z"] == 3.0 and facts["template_box_km_per_s"] == 8000.0
    assert facts["template_extrapolated_below_k_s_per_km"] == pytest.approx(7.85e-4, rel=1e-3)
    assert facts["model"].startswith("none")
    assert facts["simulations"].startswith("docs/figures/data/class_ratio_ranges_z3p0_lf60.csv")
    assert "this figure's choice" in facts["template_band"] and "0.5 to 2" in facts["template_band"]


def test_the_cddf_figure_runs_on_a_model_with_an_incidence_record(
    tmp_path, release, design, capsys, monkeypatch
):
    cddf = _script("fig_cddf_vs_ho21")
    assert [z for _, _, z in cddf.PANELS] == [2.2, 2.8, 3.4]
    moved = tuple((label, name, z) for (label, name, _), z in zip(cddf.PANELS, SYNTHETIC_Z))
    monkeypatch.setattr(cddf, "PANELS", moved)
    # record what is drawn as points: the measurements must be drawn times their cosmology factors
    from matplotlib.axes import Axes

    drawn_points, drawn_regions, real_errorbar, real_fill = {}, [], Axes.errorbar, Axes.fill

    def spy(self, x, y, *a, **k):
        drawn_points.setdefault(k.get("label"), []).append(
            (np.asarray(y), np.asarray(k.get("yerr")))
        )
        return real_errorbar(self, x, y, *a, **k)

    def spy_fill(self, x, y, *a, **k):
        drawn_regions.append((np.asarray(x), np.asarray(y), k.get("label")))
        return real_fill(self, x, y, *a, **k)

    monkeypatch.setattr(Axes, "errorbar", spy)
    monkeypatch.setattr(Axes, "fill", spy_fill)
    out = str(tmp_path / "cddf.png")
    assert cddf.main(_args(release) + ["--design", design, "--out", out]) == 0
    facts = _facts(out)
    # the conversion to the line's cosmology: the factors in the side file, and the drawn values and limits
    cosmo = facts["cosmology"]
    assert cosmo["line_omega_m"] == C.OMEGA_M_LINE and cosmo["own_omega_m"] == C.OMEGA_M_OF
    for panel, (_, name, _) in zip(facts["panels"], moved):
        assert panel["ho21_factor"] == C.to_line_cosmology(*cddf.HO21_Z_RANGE[name], 0.279)
    for name, label in cddf.LLS_SUBDLA.items():
        assert cosmo["factors_below_dla"][label]["factor"] == cddf.factor_of(name)
    # every drawn measurement is its file times its factor: Ho et al. and Zafar et al. in each panel, the O'Meara
    # points and the Prochaska region in the panel 3 < z < 4
    ho21 = drawn_points["Ho, Bird and Garnett (2021), 68 per cent interval"]
    assert len(ho21) == 3
    for (y, yerr), panel, (_, name, _) in zip(ho21, facts["panels"], moved):
        k = panel["ho21_factor"]
        logn, f, f_lo, f_hi = cddf.read_ho21_cddf(os.path.join(DATA, "ho21", name))
        assert np.allclose(y, k * f, rtol=1e-12, atol=0.0)
        assert np.allclose(yerr, [k * (f - f_lo), k * (f_hi - f)], rtol=1e-12, atol=0.0)
    name = "zafar_2013.txt"
    _, _, f, f_lo, f_hi = cddf.read_zafar13(os.path.join(DATA, "lls_subdla", name))
    k = cddf.factor_of(name)
    assert len(drawn_points[cddf.LLS_SUBDLA[name]]) == 3 and k != 1.0
    for y, yerr in drawn_points[cddf.LLS_SUBDLA[name]]:
        assert np.allclose(y, k * f, rtol=1e-12, atol=0.0)
        assert np.allclose(yerr, [k * (f - f_lo), k * (f_hi - f)], rtol=1e-12, atol=0.0)
    name = "prochaska2010_lls_region.dat"
    logn, f = cddf.read_prochaska10(os.path.join(DATA, "lls_subdla", name))
    ((x, y, label),) = drawn_regions
    assert label == cddf.LLS_SUBDLA[name] and np.array_equal(x, logn)
    assert (
        np.allclose(y, cddf.factor_of(name) * f, rtol=1e-12, atol=0.0)
        and cddf.factor_of(name) != 1.0
    )
    name = "omeara2007_fn_all19p0.dat"
    _, _, _, f, f_lo, f_hi = cddf.read_omeara07(os.path.join(DATA, "lls_subdla", name))
    y, yerr = drawn_points[cddf.LLS_SUBDLA[name]][0]
    k = cddf.factor_of(name)
    assert np.allclose(y, k * f, rtol=1e-12, atol=0.0)
    assert np.allclose(yerr, [k * (f - f_lo), k * (f_hi - f)], rtol=1e-12, atol=0.0)
    assert (
        k != 1.0 and facts["panels"][0]["ho21_factor"] != 1.0
    ), "the conversion is not a no-op here"
    assert facts["parameters"] == C.MIDPOINT and facts["incidence_record"] is True
    assert "github.com/sbird/DLA_data" in facts["measurement_source"]
    assert "lls_subdla/PROVENANCE.md" in facts["measurement_source_below_dla"]
    below = [p["measurements_below_dla"] for p in facts["panels"]]
    zafar, omeara, prochaska = (
        cddf.LLS_SUBDLA[n]
        for n in ("zafar_2013.txt", "omeara2007_fn_all19p0.dat", "prochaska2010_lls_region.dat")
    )
    assert below == [
        [zafar],
        [zafar],
        [zafar, omeara, prochaska],
    ], "the subDLA points in every panel, the others at 3 to 4"
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
    assert [(name, column) for name, column, *_ in dndx.CLASSES] == [
        ("DLA", 2),
        ("subDLA", 1),
        ("LLS", 0),
    ]
    assert hp.HCD_CLASSES == ("LLS", "subDLA", "DLA"), "the columns of pred.dndx the script indexes"
    monkeypatch.setattr(dndx, "REDSHIFTS", SYNTHETIC_Z)
    # record the points and the dashed line as drawn
    from matplotlib.axes import Axes

    drawn_points, real_errorbar, real_plot = {}, Axes.errorbar, Axes.plot

    def spy_errorbar(self, x, y, *a, **k):
        drawn_points[k.get("label")] = (np.asarray(x), np.asarray(y), k.get("xerr"), k.get("yerr"))
        return real_errorbar(self, x, y, *a, **k)

    drawn_lines = {}

    def spy_plot(self, *a, **k):
        drawn_lines[k.get("label")] = (k, a)
        return real_plot(self, *a, **k)

    monkeypatch.setattr(Axes, "errorbar", spy_errorbar)
    monkeypatch.setattr(Axes, "plot", spy_plot)
    out = str(tmp_path / "dndx.png")
    assert dndx.main(_args(release) + ["--design", design, "--out", out]) == 0
    monkeypatch.setattr(Axes, "errorbar", real_errorbar)
    monkeypatch.setattr(Axes, "plot", real_plot)
    facts = _facts(out)
    # the count above 10^17.5: the drawn dashed line, its cross-check, and what the side file says
    emu = hp.load_release(str(release["root"]), release["sha"])
    stack = emu.predict_many(C.MIDPOINT, z=list(SYNTHETIC_Z), tau0_amp=1.0, dtau0=0.0)
    want = dndx.cumulative_dndx(stack.cddf, stack.log_nhi)
    assert np.all(np.isfinite(want))
    (label,) = [k for k in drawn_lines if k and "10^{17.5}" in k]
    kw, (x, y) = drawn_lines[label]
    assert list(x) == list(SYNTHETIC_Z) and kw["color"] == C.YELLOW and kw["ls"] != "-"
    np.testing.assert_allclose(y, want, rtol=1e-12, atol=0)
    ratio = want / dndx.cross_check(stack.dndx, stack.cddf, stack.log_nhi, 17.2)
    cum = facts["cumulative"]
    assert cum["threshold_log_nhi"] == 17.5 and cum["bin_holding_the_threshold"] == [17.5]
    assert cum["cross_check_ratio_range"] == [float(ratio.min()), float(ratio.max())]
    assert set(cum["bins_without_value_above_the_threshold"]) == {f"{z:g}" for z in SYNTHETIC_Z}
    assert cum["label"] == "model (LF+MF): all absorbers with N_HI >= 10^17.5"
    # the LLS points: l(X) with the factors of the side file, the Poisson interval where it is drawn
    lls = facts["lls_measurements"]
    assert lls["drawn"] == {source: v[0] for source, v in dndx.LLS_DRAWN.items()}
    assert lls["swapped_printed_errors"] == ["prochaska10_t4"]
    assert "344 quasars" in lls["not_independent"] and "fumagalli20_dr16" in lls["own_marker"]
    # the drawn DLA points of Ho et al.: the table times the factor of each bin
    zm, value, lo68, hi68 = dndx.read_ho21_dndx(os.path.join(DATA, "ho21", "dndx_all.txt"))
    k = np.array([row["factor"] for row in facts["cosmology"]["factors_ho21"]])
    x, y, _, yerr = drawn_points["DLA, Ho, Bird and Garnett (2021), 68 per cent interval"]
    assert np.array_equal(x, zm) and np.all(k != 1.0)
    np.testing.assert_allclose(y, k * value, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        np.asarray(yerr), [k * (value - lo68), k * (hi68 - value)], rtol=1e-12, atol=0
    )
    factors = facts["cosmology"]["factors_lls"]
    assert len(factors) == 12
    points = dndx.lls_points(dndx.read_lls(os.path.join(DATA, *dndx.LLS_FILE)))
    for source in dndx.LLS_DRAWN:
        x, y, xerr, yerr = drawn_points[dndx.lls_label(source)]
        p = points[source]
        np.testing.assert_allclose(x, (p[:, 0] + p[:, 1]) / 2, rtol=1e-14, atol=0)
        np.testing.assert_allclose(y, p[:, 2], rtol=1e-14, atol=0)
        np.testing.assert_allclose(np.asarray(xerr), [x - p[:, 0], p[:, 1] - x], rtol=1e-12, atol=0)
        np.testing.assert_allclose(np.asarray(yerr), [p[:, 3], p[:, 4]], rtol=1e-14, atol=0)
        mine = [f["factor"] for f in factors if f["source"] == source]
        assert mine == list(p[:, 5])
    assert (
        "68 per cent interval" in facts["measurement"]
        and "18 redshift bins" in facts["measurement"]
    )
    # the conversion to the line's cosmology: one factor per bin, the bins of 1/6 from 2 to 5 about their centres
    factors = facts["cosmology"]["factors_ho21"]
    assert len(factors) == 18
    assert factors[0]["z_range"][0] == pytest.approx(2.0) and factors[-1]["z_range"][
        1
    ] == pytest.approx(5.0)
    for a, b in zip(factors, factors[1:]):
        assert a["z_range"][1] == pytest.approx(b["z_range"][0])
    for row in factors:
        assert row["factor"] == C.to_line_cosmology(*row["z_range"], 0.279)
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
    drawn = []
    from matplotlib.axes import Axes

    real_errorbar = Axes.errorbar

    def errorbar(self, x, y, yerr=None, **kw):
        drawn.append((np.asarray(x), np.asarray(y), yerr, kw.get("fmt")))
        return real_errorbar(self, x, y, yerr=yerr, **kw)

    monkeypatch.setattr(Axes, "errorbar", errorbar)
    assert dndx.main(_args(with_error) + ["--no-band", "--out", out2]) == 0
    # the bars themselves: per class, at the redshifts with an estimate, the held-out range of the model value v,
    # from v * exp(-e) to v * exp(+e) with e the held-out rms of ln(prediction / truth), each bound on its own.
    # The synthetic LF+MF table of the block, in the column order LLS, subDLA, DLA:
    table = SR.heldout_error()["dndx"]["lfmf"]
    bars = [d for d in drawn if d[3] == "none"]
    assert len(bars) == len(dndx.CLASSES) == 3
    for (x, y, yerr, _), (name, column, *_) in zip(bars, dndx.CLASSES):
        want = {z: row[column] for z, row in zip((3.0, 5.0), table) if row[column] is not None}
        assert list(x) == sorted(want), name  # no bar at 2.6, and none where the cell is null
        e = np.array([want[z] for z in x])
        below, above = np.asarray(yerr)  # the bar's extent under and over the line
        np.testing.assert_allclose(y - below, y * np.exp(-e), rtol=1e-12, atol=0)
        np.testing.assert_allclose(y + above, y * np.exp(e), rtol=1e-12, atol=0)
        assert np.all(above > below) and np.all(y > 0)  # asymmetric: exp(+e) - 1 > 1 - exp(-e)
    assert _facts(out2)["heldout_error"].startswith(
        "drawn on the model lines (LF+MF): the held-out range from pred.dndx_lower to pred.dndx_upper "
        "(pred.dndx * exp(-pred.dndx_err) and pred.dndx * exp(+pred.dndx_err), each bound computed on its own"
    )


# the PI's rule (2026-11-15): a held-out error is stated on the quantity, never propagated from the logarithm by an
# approximation, and its range is never called symmetric
FORBIDDEN = (
    r"to first order|first-order|for small values|fractional error|multiply by d|\bsymmetric|gaussian"
    r"|1[- ]?sigma|one sigma|1σ|±|lineari[sz]|approximat|propagat"
)


def test_no_public_text_on_the_held_out_error_uses_an_approximation_or_calls_it_symmetric():
    import re

    held_out = re.compile(
        r"dndx_err|cddf_err|dndx_lower|dndx_upper|cddf_lower|cddf_upper|held-out (error|range|rms)",
        re.I,
    )
    files = (
        "README.md",
        "CHANGELOG.md",
        "docs/release_format_h5.md",
        "docs/model_figures.md",
        "docs/models_and_data.md",
        "scripts/figures/fig_dndx_vs_ho21.py",
        "hcd_analysis/predictor.py",
        "hcd_analysis/emulator/incidence_mf.py",
        "docs/figures/dndx_vs_ho21.json",
    )
    for name in files:
        checked = 0
        text = open(os.path.join(REPO, name), encoding="utf-8").read()
        # a unit is a paragraph, and in a list each item on its own (a changelog lists one change per item)
        for paragraph in re.split(r"\n\s*\n|\n(?=\s*- )", text):
            if held_out.search(paragraph):
                checked += 1
                hit = re.search(FORBIDDEN, paragraph, re.I)
                assert hit is None, (name, hit and hit.group(0), paragraph[:300])
        assert checked >= 1, f"{name}: no paragraph on the held-out error was found to check"
    # and the figure takes its bars from the two bounds the prediction carries: it transforms nothing itself, and
    # never forms the error times the value
    src = open(os.path.join(REPO, "scripts/figures/fig_dndx_vs_ho21.py"), encoding="utf-8").read()
    assert not re.search(r"dndx_err[^\n]*\*\s*[\w.]*dndx\b|dndx\[[^\n]*\*\s*[\w.]*dndx_err", src)
    assert "stack.dndx_lower" in src and "stack.dndx_upper" in src and "np.exp" not in src


def test_the_dndx_figure_refuses_a_model_without_an_incidence_record(tmp_path):
    dndx = _script("fig_dndx_vs_ho21")
    plain = SR.make_release(tmp_path / "plain")
    with pytest.raises(SystemExit, match="no incidence MF record"):
        dndx.main(_args(plain) + ["--out", str(tmp_path / "dndx.png")])
    assert not (tmp_path / "dndx.png").exists()


# ---- the held-out prediction residual of the class P1D (every number below is synthetic)


@pytest.fixture(scope="module")
def release_p1d(tmp_path_factory):
    return SR.make_release(tmp_path_factory.mktemp("figures_p1d") / "release", p1d_heldout=True)


def _spy_on_plot(monkeypatch):
    """Record every Axes.plot call: (the axes, x, y, the keywords)."""
    from matplotlib.axes import Axes

    calls, real = [], Axes.plot

    def spy(self, x, y, *a, **k):
        calls.append((self, np.asarray(x, dtype=float), np.asarray(y, dtype=float), dict(k)))
        return real(self, x, y, *a, **k)

    monkeypatch.setattr(Axes, "plot", spy)
    return calls


def _figures(calls):
    out = []
    for ax, *_ in calls:
        if not any(ax.figure is f for f in out):
            out.append(ax.figure)
    return out


def test_the_p1d_validation_tables_are_drawn_as_the_record_holds_them(
    tmp_path, release_p1d, monkeypatch
):
    from hcd_analysis.emulator import p1d_heldout as PH

    val = _script("fig_p1d_validation")
    assert val.Z_BY_SIMULATION == (2.2, 3.0, 4.0, 5.4) and val.ROW_CHOICES == PH.ROWS
    held, _ = hp.p1d_heldout_record(hp.load_release(str(release_p1d["root"]), release_p1d["sha"]))
    calls = _spy_on_plot(monkeypatch)
    assert val.main(_args(release_p1d) + ["--out-dir", str(tmp_path)]) == 0
    figures = _figures(calls)
    assert len(figures) == 3
    for fig, fid, label in zip(figures, PH.FIDELITIES, ("LF", "LF+MF")):
        z, e = np.asarray(held.z[fid]), np.asarray(held.e[fid])
        mine = [c for c in calls if c[0].figure is fig]
        for row, title in enumerate(val.ROW_TITLES):
            panel = [c for c in mine if c[0].get_title(loc="left") == title]
            lines = [c for c in panel if c[3].get("ls") in ("-", "--")]
            assert len(lines) == len(z), (fid, title)
            for (_, x, y, kw), iz in zip(lines, range(len(z))):
                # the line of a redshift is the record's e times 100, at the modes of the box at the midpoint
                assert np.array_equal(y, 100.0 * e[iz, row])
                assert np.array_equal(x, val.mode_wavenumbers(held.n_modes, z[iz]))
                assert kw["ls"] == ("--" if z[iz] > 4.6 else "-")
            marks = [c for c in panel if c[3].get("marker") == "o"]
            assert len(marks) == len(z) and all(len(c[1]) == PH.N_LARGEST_MODES for c in marks)
            dotted = [c for c in panel if c[3].get("ls") == ":"]
            if fid == held.record["sensitivity"]["fidelity"]:
                # the sensitivity table at the highest redshift, and nowhere else
                assert len(dotted) == 1
                assert np.array_equal(dotted[0][2], 100.0 * np.asarray(held.e_sens)[-1, row])
            else:
                assert dotted == []
        facts = _facts(str(tmp_path / f"p1d_validation_{fid}.png"))
        assert facts["fidelity"] == label and facts["redshifts"] == [float(v) for v in z]
        assert (
            facts["what_it_is"]
            == "an empirical validation diagnostic, not a calibrated uncertainty"
        )
        assert facts["record_sha256"] == release_p1d["p1d_heldout_sha"] == held.record_sha256
        assert facts["n_held_out_simulations"] == len(held.record[fid]["sims"])
        assert facts["model"]["synthetic_build"] is False and facts["midpoint"] == {
            "hub": C.MIDPOINT["hub"],
            "omegamh2": C.MIDPOINT["omegamh2"],
        }
        # the numbers of the side file: per row, the record's own summary of that row alone
        assert set(facts["summary_per_cent"]) == set(PH.ROWS)
        for row, name in enumerate(PH.ROWS):
            want = PH.summarise(e[:, row : row + 1, :], z)
            got = facts["summary_per_cent"][name]
            assert set(got) == set(want)
            for regime in want:
                for q in ("median", "max"):
                    assert got[regime][q] == pytest.approx(100.0 * want[regime][q], abs=6e-4)
        assert ("sensitivity" in facts) == (fid == held.record["sensitivity"]["fidelity"])
    lf = _facts(str(tmp_path / "p1d_validation_lf.png"))
    assert lf["sensitivity"]["statement"] == held.record["sensitivity"]["statement"]
    assert PH.design_point(held.record["sensitivity"]["without_sim"]) in lf["sensitivity"]["dotted"]
    # The dotted line is the sensitivity table and not the registered one. The two are equal in the synthetic
    # record (every simulation has the same size of residual), so draw once more with the sensitivity table doubled.
    import matplotlib.pyplot as plt

    doubled = held._replace(e_sens=2.0 * np.asarray(held.e_sens))
    calls.clear()
    val.draw_table(plt, doubled, "lf", str(tmp_path / "doubled.png"), {})
    dotted = [c for c in calls if c[3].get("ls") == ":"]
    assert len(dotted) == len(val.ROW_TITLES)
    for ax, _, y, _ in dotted:
        row = val.ROW_TITLES.index(ax.get_title(loc="left"))
        assert np.array_equal(y, 200.0 * np.asarray(held.e_sens)[-1, row])
        assert not np.array_equal(y, 100.0 * np.asarray(held.e["lf"])[-1, row])


@pytest.mark.parametrize("row", ["mixture", "DLA"])
def test_the_p1d_validation_figure_per_simulation_draws_the_residual_vectors(
    tmp_path, release_p1d, monkeypatch, row
):
    from hcd_analysis.emulator import p1d_heldout as PH
    from hcd_analysis.emulator import safe_io as S

    val = _script("fig_p1d_validation")
    # the synthetic record holds LF at z 2, 3 and 5 and LF+MF at z 3 and 5: three of its redshifts and one it lacks
    monkeypatch.setattr(val, "Z_BY_SIMULATION", (2.0, 3.0, 5.0, 5.4))
    emu = hp.load_release(str(release_p1d["root"]), release_p1d["sha"])
    held, record_dir = hp.p1d_heldout_record(emu)
    rec = held.record
    arrays, _ = S.read_table_h5(os.path.join(record_dir, PH.TABLES_FILE), PH.table_datasets(rec))
    calls = _spy_on_plot(monkeypatch)
    assert val.main(_args(release_p1d) + ["--out-dir", str(tmp_path), "--row", row]) == 0
    name = "p1d_validation_by_simulation" + ("" if row == "mixture" else f"_{row}")
    assert sorted(f for f in os.listdir(tmp_path) if f.endswith(".png")) == sorted(
        [name + ".png", "p1d_validation_lf.png", "p1d_validation_lfmf.png"]
    )
    fig = _figures(calls)[2]
    mine = [c for c in calls if c[0].figure is fig]
    i_row = PH.ROWS.index(row)
    for fid, label in zip(PH.FIDELITIES, ("LF", "LF+MF")):
        z = list(rec[fid]["z"])
        d = arrays[f"{fid}_d"].astype(np.float64)
        for zz in val.Z_BY_SIMULATION:
            title = f"{label}, z = {zz:g}"
            panel = [c for c in mine if c[0].get_title(loc="left") == title]
            ax = [a for a in fig.axes if a.get_title(loc="left") == title][0]
            if zz not in z:
                assert panel == [] and [t.get_text() for t in ax.texts] == ["no estimate"]
                continue
            iz = z.index(zz)
            # one line per simulation that has a residual there, each the evidence file's vector times 100
            want = [100.0 * d[i, iz, 0, i_row] for i in range(len(rec[fid]["sims"]))]
            want = [y for y in want if np.isfinite(y).any()]
            assert len(panel) == len(want) >= 1
            for (_, x, y, _), w in zip(panel, want):
                assert np.array_equal(y, w)
                assert np.array_equal(x, val.mode_wavenumbers(held.n_modes, zz))
    # the synthetic record has one simulation without a value at its first redshift: one line fewer there
    first = [c for c in mine if c[0].get_title(loc="left") == "LF, z = 2"]
    assert len(first) == len(rec["lf"]["sims"]) - 1
    facts = _facts(str(tmp_path / (name + ".png")))
    assert facts["row"] == row and facts["rows"] == ["LF", "LF+MF"]
    assert facts["what_it_is"] == "an empirical validation diagnostic, not a calibrated uncertainty"
    assert facts["record_sha256"] == held.record_sha256
    assert facts["evidence_file_sha256"] == rec["tables"]["sha256"]
    assert facts["simulations"] == sorted({PH.design_point(s) for s in rec["lf"]["sims"]})
    assert facts["in_the_first_hue"]["design_point"] == PH.design_point(
        rec["sensitivity"]["without_sim"]
    )
    assert facts["in_the_second_hue"]["design_point"] in facts["simulations"]
    assert set(facts["rung_indices_drawn"]) == {"LF", "LF+MF"}


def test_the_rung_drawn_per_simulation_is_the_one_nearest_the_kim_mean_flux():
    from hcd_analysis.emulator.mean_flux_model import kim_tau0

    val = _script("fig_p1d_validation")
    for z in (2.2, 4.0):
        tau = float(kim_tau0(z)) * np.array([0.6, 0.8, 1.03, 1.3, np.nan])
        assert val.central_rung(np.exp(-tau), z) == 2
        assert val.central_rung(np.exp(-tau[::-1]), z) == 2
        assert val.central_rung(np.exp(-tau[:2]), z) == 1


def test_the_wavenumbers_of_the_p1d_validation_figures_are_the_modes_at_the_midpoint():
    from hcd_analysis.emulator import kcoord
    from hcd_analysis.emulator.schema import L_BOX_HMPC

    val = _script("fig_p1d_validation")
    k = val.mode_wavenumbers(7, 3.0)
    want = kcoord.k_skm_from_kcom(
        2.0 * np.pi * np.arange(1, 8) / L_BOX_HMPC, 3.0, C.MIDPOINT["hub"], C.MIDPOINT["omegamh2"]
    )
    assert k.shape == (7,) and np.array_equal(k, np.asarray(want, dtype=np.float64))
    assert np.all(np.diff(k) > 0) and k[1] == pytest.approx(2.0 * k[0])
    assert not np.array_equal(k, val.mode_wavenumbers(7, 4.0))


def test_the_p1d_validation_figures_refuse_a_model_without_the_record(tmp_path):
    val = _script("fig_p1d_validation")
    plain = SR.make_release(tmp_path / "plain")
    with pytest.raises(SystemExit, match="no P1D held-out record"):
        val.main(_args(plain) + ["--out-dir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_a_record_shipped_without_its_evidence_file_draws_the_two_tables_only(tmp_path, capsys):
    val = _script("fig_p1d_validation")
    alone = SR.make_release(tmp_path / "alone", p1d_heldout=True, p1d_evidence=False)
    out = tmp_path / "out"
    assert val.main(_args(alone) + ["--out-dir", str(out)]) == 0
    assert sorted(f for f in os.listdir(out) if f.endswith(".png")) == [
        "p1d_validation_lf.png",
        "p1d_validation_lfmf.png",
    ]
    assert "without its evidence file" in capsys.readouterr().out


def test_the_lines_of_the_docs_that_draw_the_range_on_a_prediction_run(
    tmp_path, release_p1d, monkeypatch
):
    """The code block of docs/model_figures.md that draws a class spectrum with its validation range, run as
    written on a prediction of the synthetic model."""
    import re

    import matplotlib.pyplot as plt

    page = open(os.path.join(REPO, "docs", "model_figures.md"), encoding="utf-8").read()
    blocks = re.findall(r"```python\n(.*?)```", page, flags=re.S)
    assert len(blocks) == 1 and "P_class_validation_lower" in blocks[0]
    assert len(blocks[0].strip().splitlines()) <= 8
    emu = hp.load_release(str(release_p1d["root"]), release_p1d["sha"])
    pred = emu.predict(C.MIDPOINT, z=3.0, tau0_amp=1.0, dtau0=0.0)
    assert pred.flags["p1d_validation_available"] is True
    monkeypatch.chdir(tmp_path)
    scope = {"pred": pred}
    exec(compile(blocks[0], "docs/model_figures.md", "exec"), scope)  # noqa: S102
    with open(tmp_path / "my_prediction.png", "rb") as fh:
        assert fh.read(8) == PNG
    band = scope["ax"].collections[0].get_paths()[0].vertices[:, 1]
    assert band.min() == pytest.approx(float(np.min(pred.P_class_validation_lower[scope["c"]])))
    assert band.max() == pytest.approx(float(np.max(pred.P_class_validation_upper[scope["c"]])))
    plt.close(scope["fig"])


def test_the_committed_p1d_validation_figures_are_of_one_record_and_say_what_they_are():
    """The three committed images and their side files: drawn from one record, each saying that it shows an
    empirical validation diagnostic; the docs page shows all three and gives the command. No value is pinned.
    """
    figures = os.path.join(REPO, "docs", "figures")
    names = ("p1d_validation_lf", "p1d_validation_lfmf", "p1d_validation_by_simulation")
    facts = [_facts(os.path.join(figures, n + ".png")) for n in names]
    assert len({f["record_sha256"] for f in facts}) == 1
    assert all(
        f["what_it_is"] == "an empirical validation diagnostic, not a calibrated uncertainty"
        for f in facts
    )
    assert [f.get("fidelity") for f in facts] == ["LF", "LF+MF", None] and facts[2][
        "row"
    ] == "mixture"
    assert all(f["model"]["synthetic_build"] is False for f in facts)
    page = open(os.path.join(REPO, "docs", "model_figures.md"), encoding="utf-8").read()
    for n in names:
        assert f"](figures/{n}.png)" in page, n
    assert "python scripts/figures/fig_p1d_validation.py --published" in page
    assert "empirical validation diagnostic" in page and "not a calibrated uncertainty" in page
    assert "P_class_validation_lower" in page and "P_class_validation_upper" in page


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


def test_the_example_the_figure_scripts_and_the_report_module_force_no_matplotlib_backend():
    """The choice of backend is the user's (the PI's review of 2026-11-16). The example, the figure scripts and the
    report module only save files, which works under any backend, so none of them calls matplotlib.use. A session
    without a display sets MPLBACKEND=Agg, and the example's help and the docs say so. (Older research scripts
    still set a backend; they are not in this list.)"""
    import glob
    import re

    files = ["examples/mock_recovery.py", "hcd_analysis/report.py"]
    files += sorted(
        os.path.relpath(f, REPO)
        for f in glob.glob(os.path.join(REPO, "scripts", "figures", "*.py"))
    )
    assert len(files) >= 6
    for name in files:
        src = open(os.path.join(REPO, name), encoding="utf-8").read()
        assert not re.search(r"\b(matplotlib|mpl)\.use\(|switch_backend\(", src), name
    for name in ("examples/mock_recovery.py", "docs/likelihood.md", "docs/model_figures.md"):
        assert "MPLBACKEND=Agg" in open(os.path.join(REPO, name), encoding="utf-8").read(), name


def test_a_figure_script_draws_under_the_backend_the_user_chose(tmp_path):
    """Under a backend that is not Agg the figure is still written as a PNG file, and the backend is left as the
    user chose it."""
    import matplotlib
    import matplotlib.pyplot as plt

    ratio = _script("fig_class_ratio_vs_rogers18")
    before = matplotlib.get_backend()
    plt.switch_backend("svg")
    try:
        out = str(tmp_path / "ratio.png")
        assert ratio.main(["--out", out]) == 0
        assert matplotlib.get_backend().lower() == "svg"
        assert open(out, "rb").read(8) == b"\x89PNG\r\n\x1a\n"
    finally:
        plt.switch_backend(before)
