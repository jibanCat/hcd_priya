"""The MF correction of the incidence (hcd_analysis/emulator/incidence_mf.py): the fit is the historical functions of
multifidelity.py on the matched pairs; the record round-trips and is strict; a release names it as an optional part; and an
LF+MF prediction's dN/dX is multiplied by the record's factor before the w_c map. Synthetic values only: the ratios
are round numbers that no simulation gives."""

import hashlib
import json
import os
import shutil

import jax.numpy as jnp
import numpy as np
import pytest

import hcd_analysis.predictor as hp
from hcd_analysis.emulator import incidence_mf as IM
from hcd_analysis.emulator import multifidelity as MF
from hcd_analysis.emulator import release_manifest as RM
from hcd_analysis.emulator.data import PARAM_LIMITS, normalize_params
from tests.synthetic_release import (
    INCIDENCE_TABLES,
    heldout_error,
    make_release,
    write_incidence,
)
from tests.test_predictor_mf import PARAMS

E = IM.MFRecordError
SIMS = ("synthsimA", "synthsimB", "synthsimC")
ZS = (2.0, 3.0, 4.0)
RATIO = np.array(
    [[2.0, 3.0, 0.5], [4.0, 3.0, 0.5], [4.0, 5.0, 0.25]]
)  # (z, class): the synthetic HR/LF
LOG_NHI = np.array([17.5, 20.5])
CDDF_RATIO = np.array([[2.0, 0.5], [4.0, 0.5], [4.0, 0.25]])


def snaps(sims=SIMS, zs=ZS):
    """Synthetic LF and HR snapshot dicts (the keys of multifidelity.load_snap_cache): one LF and one HR snapshot per
    simulation and redshift, HR = LF times RATIO exactly, plus one LF simulation with no HR run."""
    lim = np.asarray(PARAM_LIMITS, float)
    rows = {"lf": [], "hr": []}
    for i, s in enumerate(list(sims) + ["synthsim_lf_only"]):
        params = lim[:, 0] + (0.1 + 0.2 * i) * (lim[:, 1] - lim[:, 0])
        for j, z in enumerate(zs):
            dndx = np.array([0.25, 0.125, 0.0625]) * (1 + i) * (1 + j)
            f = np.array([8.0, 2.0]) * (1 + i) * (1 + j)
            rows["lf"].append((s, z, params, dndx, f))
            if s in sims:
                rows["hr"].append((s, z, params, dndx * RATIO[j], f * CDDF_RATIO[j]))

    def pack(r):
        params = np.array([x[2] for x in r])
        return dict(
            sim=np.array([x[0] for x in r]),
            z=np.round(np.array([x[1] for x in r]), 4),
            params=params,
            pu=np.round(normalize_params(params), 6),
            dndx=np.array([x[3] for x in r]),
            f_nhi=np.array([x[4] for x in r]),
            log_nhi=LOG_NHI.copy(),
        )

    return pack(rows["lf"]), pack(rows["hr"])


# ----------------------------------------------------------------------------------------------- the fit
def test_the_fit_is_the_historical_functions_on_the_matched_pairs():
    lf, hr = snaps()
    t, train, n = IM.fit_tables(lf, hr, exclude_sims=[])
    assert train == sorted(SIMS) and n == 9 and t["n_sims"].tolist() == [3, 3, 3]
    np.testing.assert_allclose(t["dndx_ratio"], RATIO, rtol=1e-14)
    np.testing.assert_allclose(t["cddf_ratio"], CDDF_RATIO, rtol=1e-14)
    assert t["z"].tolist() == list(ZS) and t["log_nhi_centres"].tolist() == LOG_NHI.tolist()
    assert t["cddf_n_pairs"].tolist() == [[3, 3]] * 3  # every pair has absorbers in both bins
    # the same numbers as the historical builders called by hand on the pairs
    pairs = MF.match_snap_hr_to_lf(lf, hr)
    h, l = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    old = MF.build_dndx_res_corr(lf["dndx"][l], lf["z"][l], hr["dndx"][h], hr["z"][h])
    oldc = MF.build_cddf_res_corr(lf["f_nhi"][l], lf["z"][l], hr["f_nhi"][h], hr["z"][h], LOG_NHI)
    assert t["dndx_ratio"].tobytes() == old["ratio"].tobytes()
    assert t["cddf_ratio"].tobytes() == oldc["ratio"].tobytes()


def test_the_composition_leaves_simulations_out_and_says_which_remain():
    lf, hr = snaps()
    hr["dndx"][hr["sim"] == "synthsimC"] *= 3.0  # one run with another ratio
    all3, _, _ = IM.fit_tables(lf, hr, exclude_sims=[])
    t, train, n = IM.fit_tables(lf, hr, exclude_sims=["synthsimC"])
    assert train == ["synthsimA", "synthsimB"] and n == 6 and t["n_sims"].tolist() == [2, 2, 2]
    np.testing.assert_allclose(t["dndx_ratio"], RATIO, rtol=1e-14)
    np.testing.assert_allclose(all3["dndx_ratio"], RATIO * 5.0 / 3.0, rtol=1e-14)
    with pytest.raises(E, match="unknown simulations"):
        IM.fit_tables(lf, hr, exclude_sims=["no_such_sim"])
    with pytest.raises(E, match="no training pairs left"):
        IM.fit_tables(lf, hr, exclude_sims=list(SIMS))
    with pytest.raises(TypeError, match="exclude_sims"):  # the composition has no default
        IM.fit_tables(lf, hr)


def test_an_hr_snapshot_without_its_lf_partner_is_refused_not_dropped():
    lf, hr = snaps()
    hr["z"][0] = 2.5  # no LF snapshot of that simulation there
    with pytest.raises(E, match="without an LF partner.*synthsimA z=2.5"):
        IM.fit_tables(lf, hr, exclude_sims=[])
    for bins in (LOG_NHI + 0.1, LOG_NHI[:1]):  # other centres; another number of bins
        lf, hr = snaps()
        hr["log_nhi"] = bins
        with pytest.raises(E, match="column-density bins differ"):
            IM.fit_tables(lf, hr, exclude_sims=[])


def test_a_snapshot_listed_twice_is_refused():
    lf, hr = snaps()
    twice = {k: (v if k == "log_nhi" else np.concatenate([v, v[:1]])) for k, v in hr.items()}
    with pytest.raises(E, match="listed more than once.*synthsimA z=2"):
        IM.fit_tables(lf, twice, exclude_sims=[])
    lf2 = {k: (v if k == "log_nhi" else np.concatenate([v, v[:1]])) for k, v in lf.items()}
    with pytest.raises(E, match="LF partner is not unique.*synthsimA z=2"):
        IM.fit_tables(lf2, hr, exclude_sims=[])
    # an LF snapshot listed twice that no HR snapshot pairs with does not matter
    only = lf["sim"] == "synthsim_lf_only"
    lf3 = {k: (v if k == "log_nhi" else np.concatenate([v, v[only][:1]])) for k, v in lf.items()}
    assert IM.fit_tables(lf3, hr, exclude_sims=[])[2] == 9


@pytest.mark.parametrize("side", ["lf", "hr"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_a_pair_without_a_finite_positive_dndx_is_refused_not_averaged_around(side, value):
    lf, hr = snaps()
    d = lf if side == "lf" else hr
    i = int(np.where((d["sim"] == "synthsimB") & (d["z"] == 3.0))[0][0])
    d["dndx"][i, 2] = value
    with pytest.raises(E, match="dN/dX is not finite and positive.*synthsimB z=3"):
        IM.fit_tables(lf, hr, exclude_sims=[])


def test_the_cddf_takes_empty_bins_and_refuses_bad_ones():
    lf, hr = snaps()
    hr["f_nhi"][
        :, 1
    ] = 0.0  # a bin that is empty in every HR snapshot: ratio 1, by the historical rule
    t, _, _ = IM.fit_tables(lf, hr, exclude_sims=[])
    assert t["cddf_ratio"][:, 1].tolist() == [1.0, 1.0, 1.0]
    assert (
        t["cddf_n_pairs"].tolist() == [[3, 0]] * 3
    )  # the count says: no correction was measured in bin 1
    lf, hr = snaps()
    hr["f_nhi"][
        (hr["sim"] == "synthsimA") & (hr["z"] == 3.0), 1
    ] = 0.0  # one pair of one redshift only
    assert IM.fit_tables(lf, hr, exclude_sims=[])[0]["cddf_n_pairs"].tolist() == [
        [3, 3],
        [3, 2],
        [3, 3],
    ]
    lf, hr = snaps()
    np.testing.assert_allclose(t["cddf_ratio"][:, 0], CDDF_RATIO[:, 0], rtol=1e-14)
    np.testing.assert_allclose(
        t["dndx_ratio"], RATIO, rtol=1e-14
    )  # the dN/dX table does not see it
    for value in (-1.0, float("nan")):
        lf, hr = snaps()
        hr["f_nhi"][0, 0] = value
        with pytest.raises(E, match="f\\(N_HI\\) is not finite and non-negative.*synthsimA z=2"):
            IM.fit_tables(lf, hr, exclude_sims=[])


def test_the_hr_product_is_read_as_plain_arrays(tmp_path):
    _, hr = snaps()
    arrays = dict(
        sim_name=hr["sim"],
        z=hr["z"],
        params=hr["params"],
        dndx=hr["dndx"],
        cddf=hr["f_nhi"],
        log_nhi_centres=hr["log_nhi"],
        class_names=np.array(IM.CLASS_ORDER),
    )
    p = tmp_path / "hr.npz"
    np.savez(p, **arrays)
    got = IM.hr_snap_from_product(p)
    assert set(got) == set(hr) and all(np.array_equal(got[k], hr[k]) for k in hr)
    for name, change, why in (
        ("missing", {"dndx": None}, "the HR incidence product lacks the arrays \\['dndx'\\]"),
        ("order", {"class_names": np.array(["DLA", "subDLA", "LLS"])}, "class_names must be"),
        ("shape", {"dndx": hr["dndx"][:, :2]}, "inconsistent shapes"),
        ("scalar", {"sim_name": np.array("synthsimA")}, "inconsistent shapes"),
        ("pickle", {"sim_name": np.array(list(hr["sim"]), dtype=object)}, "unreadable"),
        (
            "nan",
            {"dndx": np.where(np.arange(hr["dndx"].size).reshape(-1, 3) == 4, np.nan, hr["dndx"])},
            "finite",
        ),
        ("text", {"z": np.array(["two"] * len(hr["z"]))}, "not numeric"),
        # a name column out of step with the parameters: one name with two points, or two names on one point
        ("two_points", {"sim_name": np.roll(hr["sim"], 1)}, "more than one parameter point"),
        (
            "one_point",
            {"sim_name": np.array([f"{s}{i}" for i, s in enumerate(hr["sim"])])},
            "share one",
        ),
    ):
        q = tmp_path / f"{name}.npz"
        np.savez(q, **{k: v for k, v in dict(arrays, **change).items() if v is not None})
        with pytest.raises(E, match=why):
            IM.hr_snap_from_product(q)
    (tmp_path / "text.npz").write_text("not an archive")
    with pytest.raises(E, match="unreadable"):
        IM.hr_snap_from_product(tmp_path / "text.npz")


# ----------------------------------------------------------------------------------------------- the record
CACHE = "cd" * 32


def test_a_record_round_trips_and_its_factor_is_the_historical_lookup(tmp_path):
    lf, hr = snaps()
    t, train, n = IM.fit_tables(lf, hr, exclude_sims=[])
    sha = write_incidence(tmp_path / "inc", cache=CACHE, tables=t, train_sims=train, n_pairs=n)
    inc = IM.load_record(tmp_path / "inc", expected_record_sha256=sha, lf_cache_sha256=CACHE)
    assert inc.anchored is True and inc.record_sha256 == sha
    assert inc.cddf_n_pairs.tolist() == [[3, 3]] * 3 and not inc.cddf_n_pairs.flags.writeable
    for k in ("z", "dndx_ratio", "log_nhi_centres", "cddf_ratio"):
        assert getattr(inc, k).tobytes() == np.asarray(t[k], float).tobytes()  # bit for bit
        assert not getattr(inc, k).flags.writeable
    assert inc.n_sims.tolist() == [3, 3, 3]
    old = {"z": t["z"], "ratio": t["dndx_ratio"]}
    for z in (1.0, 2.0, 2.5, 3.0, 3.75, 4.0, 6.0):
        want = np.asarray(MF.dndx_res_factor(old, jnp.asarray(z)))
        assert inc.dndx_factor(z).tobytes() == want.tobytes()
    assert inc.dndx_factor(2.5).tolist() == [3.0, 3.0, 0.5]  # linear between the rows
    assert inc.dndx_factor(9.0).tolist() == RATIO[-1].tolist()  # held beyond the table
    assert inc.dndx_factor(0.0).tolist() == RATIO[0].tolist()
    assert inc.cddf_factor(2.5).tolist() == [3.0, 0.5] and inc.cddf_factor(7.0).tolist() == [
        4.0,
        0.25,
    ]
    old_c = {"z": t["z"], "log_nhi": LOG_NHI, "ratio": t["cddf_ratio"]}
    assert (
        inc.cddf_factor(3.5).tobytes()
        == np.asarray(MF.cddf_res_factor(old_c, jnp.asarray(3.5))).tobytes()
    )
    # the record states its inputs, its runs and the groups its HR product does not hold
    p = inc.provenance()
    assert p["lf_cache"]["sha256"] == CACHE and p["lf_cache"]["source"] == IM.LF_SOURCE
    assert p["hr_incidence"]["sha256"] == "ab" * 32 and p["hr_incidence"]["source"] == IM.HR_SOURCE
    assert p["hr_incidence"]["groups_not_in_product"] == ["synthsim9 snapshot 3"]
    assert p["train_sims"] == sorted(SIMS) and p["composition"]["excluded_sims"] == ["synthsim3"]
    assert p["n_sims"] == [3, 3, 3] and p["z"] == list(ZS)


def test_a_record_is_written_once_and_only_from_a_clean_tree(tmp_path):
    write_incidence(tmp_path / "inc", cache=CACHE)
    with pytest.raises(FileExistsError, match="written once"):
        write_incidence(tmp_path / "inc", cache=CACHE)
    with pytest.raises(E, match="clean tree"):
        write_incidence(
            tmp_path / "dirty", cache=CACHE, fit_code={"commit": "c" * 40, "dirty": True}
        )
    assert not (tmp_path / "dirty").exists()  # refused before anything is written
    bad = dict(INCIDENCE_TABLES, dndx_ratio=[[2.0, 3.0, float("nan")]] * 3)
    with pytest.raises(E, match="finite"):
        write_incidence(tmp_path / "nan", cache=CACHE, tables=bad)
    with pytest.raises(E, match="file, sha256"):
        write_incidence(tmp_path / "keys", cache=CACHE, lf_cache={"file": "x.h5"})


def _rewrite(tmp_path, name, change):
    """A record directory whose JSON was changed by ``change(doc)`` (or replaced by the bytes it returns), with the
    sha256 of the changed file."""
    d = tmp_path / name
    write_incidence(d, cache=CACHE)
    p = d / IM.RECORD_FILE
    doc = json.loads(p.read_text())
    out = change(doc)
    raw = out if isinstance(out, bytes) else json.dumps(doc).encode()
    os.chmod(p, 0o644)
    p.write_bytes(raw)
    return d, hashlib.sha256(raw).hexdigest()


def _set(path, value):
    def change(doc):
        d = doc
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value

    return change


@pytest.mark.parametrize(
    "name, change, why",
    [
        ("extra_key", _set(["note"], "anything"), "unknown \\['note'\\]"),
        ("format", _set(["format"], "hcd_priya.incidence_mf_record/0"), "format"),
        ("form", _set(["model_form"], "another form"), "model_form"),
        ("classes", _set(["class_order"], ["DLA", "subDLA", "LLS"]), "class_order"),
        ("z_order", _set(["tables", "z"], [3.0, 2.0, 5.0]), "must ascend"),
        ("one_z", _set(["tables", "z"], [3.0]), "at least two redshifts"),
        ("negative", _set(["tables", "dndx_ratio"], [[2.0, 3.0, -0.5]] * 3), "must be positive"),
        ("ragged", _set(["tables", "dndx_ratio"], [[2.0, 3.0], [1.0], [2.0]]), "dndx_ratio"),
        ("text", _set(["tables", "cddf_ratio"], [["2.0", 0.5]] * 3), "cddf_ratio"),
        ("bool", _set(["tables", "dndx_ratio"], [[True, 3.0, 0.5]] * 3), "dndx_ratio"),
        ("no_pairs", _set(["tables", "n_sims"], [0, 2, 3]), "positive integers"),
        ("pairs_sum", _set(["n_pairs"], 6), "sum of tables.n_sims"),
        ("too_many", _set(["train_sims"], ["synthsim1"]), "more pairs than"),
        ("excluded", _set(["train_sims"], ["synthsim1", "synthsim3"]), "excluded"),
        ("dirty", _set(["fit_code", "dirty"], True), "clean tree"),
        ("lf_source", _set(["lf_cache", "source"], "the HR cache"), "lf_cache.source"),
        ("hr_source", _set(["hr_incidence", "source"], "the HR cache"), "hr_incidence.source"),
        ("path", _set(["hr_incidence", "file"], "../hr.npz"), "plain file name"),
        (
            "groups",
            _set(["hr_incidence", "groups_not_in_product"], "none"),
            "groups_not_in_product",
        ),
        ("ruling", _set(["composition", "ruling_ids"], []), "ruling"),
        ("missing_key", lambda d: d.pop("composition") and None, "missing \\['composition'\\]"),
        ("no_tables_key", lambda d: d["tables"].pop("n_sims") and None, "missing \\['n_sims'\\]"),
        ("cddf_shape", _set(["tables", "cddf_ratio"], [[2.0, 0.5, 1.0]] * 3), "cddf_ratio"),
        ("pairs_shape", _set(["tables", "cddf_n_pairs"], [[1, 1]] * 3), "cddf_n_pairs"),
        ("pairs_float", _set(["tables", "cddf_n_pairs"], [[1.0] * 30] * 3), "cddf_n_pairs"),
        (
            "pairs_many",
            lambda d: d["tables"]["cddf_n_pairs"][0].__setitem__(3, 2),
            "more pairs than",
        ),
        (
            "no_pair_ratio",
            lambda d: d["tables"]["cddf_ratio"][0].__setitem__(0, 2.0),
            "must hold the ratio 1",
        ),
        ("cddf_rows", _set(["tables", "cddf_ratio"], [[2.0, 0.5]] * 2), "cddf_ratio"),
        ("huge", _set(["tables", "z"], [2.0, 3.0, 10**400]), "tables.z"),
        ("nan", lambda d: json.dumps(dict(d, version=float("nan"))).encode(), "NaN|constant"),
        (
            "duplicate",
            lambda d: json.dumps(d).encode()[:-1] + b', "version": "second"}',
            "duplicate",
        ),
        ("not_json", lambda d: b"\x00\x01", "JSON|json|decode"),
    ],
)
def test_a_changed_record_is_refused(tmp_path, name, change, why):
    d, sha = _rewrite(tmp_path, name, change)
    with pytest.raises(E, match=why):
        IM.load_record(d, expected_record_sha256=sha)


def test_a_record_loads_only_under_its_published_digest_from_its_own_directory(tmp_path):
    sha = write_incidence(tmp_path / "inc", cache=CACHE)
    for bad in (None, "ab" * 32, sha.upper(), sha[:-1]):
        with pytest.raises(E, match="sha256"):
            IM.load_record(tmp_path / "inc", expected_record_sha256=bad)
    with pytest.raises(E, match="built from the LF cache"):
        IM.load_record(tmp_path / "inc", expected_record_sha256=sha, lf_cache_sha256="ef" * 32)
    os.symlink(tmp_path / "inc", tmp_path / "link")
    with pytest.raises(E, match="not an incidence MF record directory"):
        IM.load_record(tmp_path / "link", expected_record_sha256=sha)
    (tmp_path / "inc" / "notes.txt").write_text("x")
    with pytest.raises(E, match="unexpected files"):
        IM.load_record(tmp_path / "inc", expected_record_sha256=sha)
    with pytest.raises(E, match="no readable"):
        IM.load_record(tmp_path, expected_record_sha256=sha)


def test_only_a_sealed_unchanged_record_is_applied(tmp_path):
    sha = write_incidence(tmp_path / "inc", cache=CACHE)
    inc = IM.load_record(tmp_path / "inc", expected_record_sha256=sha)
    assert IM.reverify(inc).dndx_ratio.tobytes() == inc.dndx_ratio.tobytes()
    raw = (tmp_path / "inc" / IM.RECORD_FILE).read_bytes()
    with pytest.raises(E, match="loaded by incidence_mf.load_record"):
        IM.reverify(IM.parse_record(raw))  # parsed without its published digest: not anchored
    with pytest.raises(E, match="changed in memory"):
        IM.reverify(inc._replace(dndx_ratio=inc.dndx_ratio * 2.0))
    with pytest.raises(E, match="changed in memory"):
        IM.reverify(inc._replace(record=dict(inc.record, version="another")))
    for name in (
        "n_sims",
        "cddf_n_pairs",
        "cddf_ratio",
    ):  # the support counts decide which bins are NaN
        with pytest.raises(E, match="changed in memory"):
            IM.reverify(inc._replace(**{name: getattr(inc, name) + 1}))
    with pytest.raises(E, match="sha256"):
        IM.reverify(inc._replace(record_bytes=raw + b" "))
    with pytest.raises(E, match="loaded by"):
        IM.reverify({"z": [2.0]})


# ----------------------------------------------------------------------------------------------- the manifest
def test_one_format_with_a_closed_list_of_optional_parts():
    plain = {
        "format": RM.FORMAT,
        "release_version": "v",
        "backbone": {"dir": "backbone", "manifest_sha256": "a" * 64},
        "mf": {"dir": "mf", "record_sha256": "b" * 64},
    }
    inc = {"dir": "incidence", "record_sha256": "c" * 64}
    full = dict(plain, incidence=inc)
    assert RM.check_release_manifest(dict(plain)) == plain
    assert RM.check_release_manifest(dict(full)) == full and full["format"] == RM.FORMAT
    assert set(RM.OPTIONAL_PARTS) == {"incidence"} and not hasattr(RM, "FORMAT_2")
    for bad, why in (
        (dict(full, error={"dir": "error", "record_sha256": "d" * 64}), "unknown \\['error'\\]"),
        (dict(full, incidence=dict(inc, dir="mf")), "two different directories"),
        (dict(full, incidence=dict(inc, dir="../incidence")), "plain path segment"),
        (dict(full, incidence=dict(inc, record_sha256="C" * 64)), "64 lowercase hex"),
        (dict(full, incidence=dict(inc, note="x")), "incidence: keys"),
        (dict(full, incidence=None), "incidence: keys"),
        (
            dict(full, format="hcd_priya.release/2"),
            "format 'hcd_priya.release/2', expected hcd_priya.release/1",
        ),
        (dict(plain, format="hcd_priya.release/2"), "expected hcd_priya.release/1"),
    ):
        with pytest.raises(RM.ReleaseManifestError, match=why):
            RM.check_release_manifest(bad)


def test_a_reader_from_before_optional_parts_refuses_a_manifest_that_names_one():
    """The check of the reader as it was before optional parts existed, kept here word for word in its two
    conditions: exactly four keys, then the format. A manifest with the incidence part fails the first, so such a
    reader refuses the release and never ignores the record."""
    old_keys = {"format", "release_version", "backbone", "mf"}

    def old_reader_accepts(m):
        return isinstance(m, dict) and set(m) == old_keys and m["format"] == "hcd_priya.release/1"

    plain = {
        "format": RM.FORMAT,
        "release_version": "v",
        "backbone": {"dir": "backbone", "manifest_sha256": "a" * 64},
        "mf": {"dir": "mf", "record_sha256": "b" * 64},
    }
    assert RM.KEYS == old_keys and old_reader_accepts(plain)
    assert not old_reader_accepts(
        dict(plain, incidence={"dir": "incidence", "record_sha256": "c" * 64})
    )


# ----------------------------------------------------------------------------------------------- the predictor
@pytest.fixture(scope="module")
def rel(tmp_path_factory):
    """One synthetic backbone and MF record, as a release with the incidence record and, from copies of the same
    two parts, as a release without it."""
    root = tmp_path_factory.mktemp("incidence")
    two = make_release(root / "two", incidence=True)
    one = root / "one"
    one.mkdir()
    for part in ("backbone", "mf"):
        shutil.copytree(two["root"] / part, one / part)
    _, sha1 = RM.write_release_manifest(
        str(one),
        release_version="synthetic-release-1",
        backbone_dir="backbone",
        backbone_manifest_sha256=two["bundle"].manifest_sha256,
        mf_dir="mf",
        mf_record_sha256=two["mf_sha"],
    )
    return {
        "two": hp.load_release(str(two["root"]), two["sha"]),
        "one": hp.load_release(str(one), sha1),
        "info": two,
        "root": root,
    }


def test_lf_mf_dndx_is_the_lf_dndx_times_the_factor_and_w_c_follows(rel):
    two, one = rel["two"], rel["one"]
    for z, factor, clamped, between in (
        (3.0, [4.0, 3.0, 0.5], False, False),  # a tabulated redshift
        (
            2.6,
            [3.2, 3.0, 0.5],
            False,
            True,
        ),  # not in the table: linear between its neighbours, flagged
        (5.2, [4.0, 5.0, 0.25], True, False),  # above the table: held at the last row, and flagged
    ):
        p = two.predict(PARAMS, z, tau_eff=0.4)
        q = one.predict(
            PARAMS, z, tau_eff=0.4
        )  # the same backbone and MF record, no incidence record
        lf = two.predict(PARAMS, z, tau_eff=0.4, fidelity="LF")
        np.testing.assert_allclose(p.dndx / q.dndx, factor, rtol=1e-13)
        np.testing.assert_allclose(p.dndx, np.asarray(q.dndx) * two._inc.dndx_factor(z), rtol=1e-15)
        assert (
            q.dndx.tobytes() == lf.dndx.tobytes()
        )  # without the record, LF+MF dN/dX is the LF dN/dX
        assert (
            p.w_c.tobytes() == two.wc_map.w_c(p.dndx, z)[1:].tobytes()
        )  # the shipped map of the corrected dN/dX
        assert not np.array_equal(p.w_c, q.w_c)
        # the class spectra are not touched: a mixed spectrum changes only through the fractions
        assert (
            p.P_class.tobytes() == q.P_class.tobytes()
            and p.template.tobytes() == q.template.tobytes()
        )
        assert hp.add_hcd(p, q.w_c).tobytes() == hp.add_hcd(q, q.w_c).tobytes()
        assert not np.array_equal(hp.add_hcd(p, p.w_c), hp.add_hcd(q, q.w_c))
        assert p.flags["incidence_mf_applied"] is True and p.flags["mf_applied"] is True
        assert p.flags["incidence_mf_z_clamped"] is clamped
        assert p.flags["incidence_mf_z_interpolated"] is between
        assert "incidence_mf_applied" not in q.flags and "incidence_mf" not in q.labels


def test_an_lf_prediction_is_not_corrected(rel):
    two, one = rel["two"], rel["one"]
    a = two.predict(PARAMS, 3.0, tau_eff=0.4, fidelity="LF")
    b = one.predict(PARAMS, 3.0, tau_eff=0.4, fidelity="LF")
    for k in ("dndx", "w_c", "P_class"):
        assert getattr(a, k).tobytes() == getattr(b, k).tobytes()
    assert a.flags["incidence_mf_applied"] is False and "dndx" not in a.labels
    # nothing was applied, so no flag says how the table was looked up: not at a tabulated redshift, not between
    # two rows (2.6), not beyond the table (5.2), where the same LF+MF prediction is flagged
    for z, flag in (
        (3.0, None),
        (2.6, "incidence_mf_z_interpolated"),
        (5.2, "incidence_mf_z_clamped"),
    ):
        lf = two.predict(PARAMS, z, tau_eff=0.4, fidelity="LF")
        for name in (
            "incidence_mf_z_clamped",
            "incidence_mf_z_interpolated",
            "incidence_mf_single_pair",
        ):
            assert lf.flags[name] is False, (z, name)
        if flag:
            assert two.predict(PARAMS, z, tau_eff=0.4).flags[flag] is True
    lf_stack = two.predict_many(PARAMS, [2.6, 3.0, 5.2], tau_eff=0.4, fidelity="LF")
    assert not any(lf_stack.flags[n] for n in lf_stack.flags if n.startswith("incidence_mf_"))
    assert "R_c(z)" in a.labels["incidence_mf"]  # the record is named, and said not to apply to LF


def test_the_stack_equals_the_single_predictions(rel):
    two = rel["two"]
    zs = [2.6, 3.0, 5.2]
    s = two.predict_many(PARAMS, zs, tau_eff=0.4)
    for i, z in enumerate(zs):
        p = two.predict(PARAMS, z, tau_eff=0.4)
        np.testing.assert_allclose(s.dndx[i], p.dndx, rtol=1e-12)
        np.testing.assert_allclose(s.w_c[i], p.w_c, rtol=1e-12)
    assert s.flags["incidence_mf_applied"] is True and s.flags["incidence_mf_z_clamped"] is True


def test_labels_and_provenance_say_what_was_applied(rel):
    two, info = rel["two"], rel["info"]
    p = two.predict(PARAMS, 3.0, tau_eff=0.4)
    assert "multiplied by the incidence MF factor" in p.labels["dndx"]
    assert "before the map" in p.labels["w_c"] and "fitted on LF counts" in p.labels["w_c"]
    assert "w_c stays the LF backbone's" not in p.labels["mf"] and "incidence MF" in p.labels["mf"]
    # no held-out error is shipped; the text that says how one is computed comes with the prediction
    # and it says how many HR simulations this record lists (the synthetic record lists two)
    label = p.labels["incidence_mf_error"]
    assert label == IM.heldout_error_label(2) and label.startswith(IM.HELDOUT_ERROR_STATEMENT)
    assert "No held-out error" in label and "six of sixty" in label
    assert "listed in provenance incidence_mf train_sims, 2 in number" in label
    assert "may rest on fewer" in label and "5 in number" in IM.heldout_error_label(5)
    assert len(p.provenance["incidence_mf"]["train_sims"]) == 2
    # the number is the length of the record's list, not the largest number of pairs behind a redshift (also 2)
    record = dict(two._inc.record, train_sims=["synthsim1", "synthsim2", "synthsim3"])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(two, "_inc", two._inc._replace(record=record))
        relabelled = two.predict(PARAMS, 3.0, tau_eff=0.4).labels["incidence_mf_error"]
    assert relabelled == IM.heldout_error_label(3) and int(two._inc.n_sims.max()) == 2
    assert p.provenance["incidence_mf"]["heldout_error"] is None
    prov = p.provenance["incidence_mf"]
    assert prov["record_sha256"] == info["incidence_sha"]
    assert prov["lf_cache"]["sha256"] == info["bundle"].lineage["cache"]["sha256"]
    assert prov["composition"]["id"] == "synthetic-composition" and prov["n_sims"] == [1, 2, 2]
    assert p.provenance["release"]["format"] == RM.FORMAT  # one format, with or without the part
    q = rel["one"].predict(PARAMS, 3.0, tau_eff=0.4)
    assert "incidence_mf" not in q.provenance and q.provenance["release"]["format"] == RM.FORMAT
    assert "w_c stays the LF backbone's" in q.labels["mf"]


def _release_with(rel, tmp_path, name, write):
    """A release from copies of the fixture's backbone and MF record, with the incidence record that
    ``write(path)`` writes (it returns the record's sha256)."""
    src, d = rel["info"]["root"], tmp_path / name
    d.mkdir()
    for part in ("backbone", "mf"):
        shutil.copytree(src / part, d / part)
    sha = write(d / "incidence")
    _, rs = RM.write_release_manifest(
        str(d),
        release_version="synthetic-release-1",
        backbone_dir="backbone",
        backbone_manifest_sha256=rel["info"]["bundle"].manifest_sha256,
        mf_dir="mf",
        mf_record_sha256=rel["info"]["mf_sha"],
        incidence_dir="incidence",
        incidence_record_sha256=sha,
    )
    return str(d), rs, sha


def test_a_record_fitted_on_another_lf_cache_is_refused_at_load(rel, tmp_path):
    d, rs, _ = _release_with(rel, tmp_path, "other", lambda p: write_incidence(p, cache="ef" * 32))
    with pytest.raises(E, match="not on this bundle's training cache"):
        hp.load_release(d, rs)


def test_a_release_names_its_incidence_record_by_digest(rel, tmp_path):
    cache = rel["info"]["bundle"].lineage["cache"]["sha256"]
    d, rs, sha = _release_with(rel, tmp_path, "good", lambda p: write_incidence(p, cache=cache))
    assert hp.load_release(d, rs).provenance["incidence_mf"]["record_sha256"] == sha
    p = os.path.join(d, "incidence", IM.RECORD_FILE)  # one changed number in the table
    os.chmod(p, 0o644)
    raw = open(p, "rb").read()
    assert raw.count(b"0.25") >= 1
    open(p, "wb").write(raw.replace(b"0.25", b"0.26", 1))  # one number of one table
    with pytest.raises(E, match="sha256 does not match"):
        hp.load_release(d, rs)


def test_the_record_is_applied_only_with_the_p1d_mf_record(rel, tmp_path):
    b = rel["info"]["bundle"]
    sha = write_incidence(tmp_path / "inc", cache=b.lineage["cache"]["sha256"])
    inc = IM.load_record(tmp_path / "inc", expected_record_sha256=sha)
    lf_only = hp.load_bundle(b.bundle_dir, b.manifest_sha256)
    with pytest.raises(E, match="applied with the P1D MF record"):
        lf_only._attach_incidence(inc)
    with pytest.raises(E, match="changed in memory"):
        rel["one"]._attach_incidence(inc._replace(dndx_ratio=inc.dndx_ratio * 2.0))
    assert rel["one"]._inc is None  # a refused record is not attached


def _member_mean_cddf(emu, z, tau):
    """The linear mean over the members of their CDDF output, computed here from the members themselves."""
    from tests.test_predictor_mf import _x

    norm = emu._rb.norm["f_nhi"]
    mean, std = np.asarray(norm["mean"], float), np.asarray(norm["std"], float)
    outs = [m(jnp.asarray(_x(z)), jnp.asarray(tau)) for m in emu._rb.ensemble.members]
    return np.mean([np.exp(np.asarray(o["f_nhi"]) * std + mean) for o in outs], axis=0)


def test_the_cddf_is_the_member_mean_times_the_factor_and_nan_where_nothing_supports_it(rel):
    """The synthetic table has one of each case: bin 0 has no pair at any redshift; bin 28 has none at z = 3.0;
    bin 29 has fewer pairs than its redshift; the other bins have them all."""
    two = rel["two"]
    assert two._inc.log_nhi_centres.size == 30
    for z, factor, f28, f29, gap, single_pair in (
        (3.0, 4.0, None, 0.25, True, False),  # a tabulated redshift whose bin 28 has no pair
        (
            2.6,
            3.2,
            None,
            0.25,
            True,
            True,
        ),  # between two rows: either without a pair leaves the bin out
        (5.2, 4.0, 0.5, 0.25, False, False),  # beyond the table: the edge row, which has bin 28
    ):
        mean = _member_mean_cddf(two, z, 0.4)
        assert np.all(np.isfinite(mean)) and np.all(mean > 0)
        lf = two.predict(PARAMS, z, tau_eff=0.4, fidelity="LF")
        mf = two.predict(PARAMS, z, tau_eff=0.4)
        for p in (lf, mf):
            assert p.cddf.shape == (30,) and p.log_nhi.tolist() == two._inc.log_nhi_centres.tolist()
            assert not p.cddf.flags.writeable and not p.log_nhi.flags.writeable
            assert np.isnan(p.cddf[0])  # never a number where the model was never trained
        # LF: the member mean, in every bin but the never-supported one
        np.testing.assert_allclose(lf.cddf[1:], mean[1:], rtol=1e-12)
        assert lf.flags["cddf_bins_without_support"] is False
        assert lf.flags["incidence_mf_single_pair"] is False
        # LF+MF: the member mean times the record's factor
        np.testing.assert_allclose(mf.cddf[1:28], factor * mean[1:28], rtol=1e-12)
        np.testing.assert_allclose(
            mf.cddf[29], f29 * mean[29], rtol=1e-12
        )  # fewer pairs: kept, not masked
        if f28 is None:
            assert np.isnan(
                mf.cddf[28]
            )  # no correction measured at this redshift: not the LF value
        else:
            np.testing.assert_allclose(mf.cddf[28], f28 * mean[28], rtol=1e-12)
        assert mf.flags["cddf_bins_without_support"] is gap
        assert mf.flags["incidence_mf_single_pair"] is single_pair
        assert "linear mean over the ensemble members" in mf.labels["cddf"]
        assert ("cddf_bins_without_support" in mf.labels["cddf"]) and "R(z, N_HI)" in mf.labels[
            "cddf"
        ]
        assert "R(z, N_HI)" not in lf.labels["cddf"]
    pairs = mf.provenance["incidence_mf"]["cddf_n_pairs"]
    assert pairs[1][0] == 0 and pairs[1][28] == 0 and pairs[1][29] == 1 and pairs[1][5] == 2


def test_the_cddf_of_a_stack_and_of_a_release_without_the_record(rel):
    two, one = rel["two"], rel["one"]
    zs = [2.6, 3.0, 5.2]
    s = two.predict_many(PARAMS, zs, tau_eff=0.4)
    assert s.cddf.shape == (3, 30) and s.log_nhi.shape == (30,)
    for i, z in enumerate(zs):
        one_z = two.predict(PARAMS, z, tau_eff=0.4).cddf
        assert np.array_equal(np.isnan(s.cddf[i]), np.isnan(one_z))  # the same bins are left out
        np.testing.assert_allclose(s.cddf[i], one_z, rtol=1e-12, equal_nan=True)
    assert (
        s.flags["cddf_bins_without_support"] is True and s.flags["incidence_mf_single_pair"] is True
    )
    # a release without the part has no record, so nothing says what the bins are: no CDDF, and no CDDF flag or label
    q = one.predict(PARAMS, 3.0, tau_eff=0.4)
    assert q.cddf is None and q.log_nhi is None
    assert "cddf_bins_without_support" not in q.flags and "cddf" not in q.labels
    # the class spectra and the incidence are not touched by the second evaluation of the members
    p = two.predict(PARAMS, 3.0, tau_eff=0.4)
    assert p.P_class.tobytes() == q.P_class.tobytes()


def test_no_cddf_element_can_be_the_exp_of_an_untrained_output(rel, tmp_path):
    """A bin that no training simulation populated keeps the placeholder normalisation (mean 0, std 1), so the
    exp of the network's output there is a number of order one, many orders above its neighbours. Such a bin is
    NaN; and a record that claims simulation pairs in such a bin is not of this model's training data.
    """
    b, info = rel["info"]["bundle"], rel["info"]
    cache = b.lineage["cache"]["sha256"]

    def fresh(untrained_bin):
        emu = hp.load_bundle(
            b.bundle_dir,
            b.manifest_sha256,
            mf_dir=str(info["root"] / "mf"),
            mf_record_sha256=info["mf_sha"],
        )
        m, s = (np.array(v) for v in emu._fn)
        m[untrained_bin], s[untrained_bin] = 0.0, 1.0
        emu._fn, emu._fn_untrained = (jnp.asarray(m), jnp.asarray(s)), (m == 0.0) & (s == 1.0)
        return emu

    sha = write_incidence(tmp_path / "inc", cache=cache)
    inc = IM.load_record(tmp_path / "inc", expected_record_sha256=sha)
    emu = fresh(
        0
    )  # the real case: the first bin lies below the catalogue's column-density threshold
    emu._attach_incidence(inc)
    for fidelity in ("LF", "LF+MF"):
        p = emu.predict(PARAMS, 3.0, tau_eff=0.4, fidelity=fidelity)
        assert np.isnan(p.cddf[0]) and np.all(np.isfinite(p.cddf[1:28]))
    with pytest.raises(E, match="never trained on"):  # the record has pairs in bin 5
        fresh(5)._attach_incidence(inc)
    # a record with another number of bins than the model's CDDF head
    lf, hr = snaps()
    t, train, n = IM.fit_tables(lf, hr, exclude_sims=[])
    sha2 = write_incidence(
        tmp_path / "two_bins", cache=cache, tables=t, train_sims=train, n_pairs=n
    )
    with pytest.raises(E, match="2 column-density bins, this model's CDDF head has 30"):
        fresh(0)._attach_incidence(
            IM.load_record(tmp_path / "two_bins", expected_record_sha256=sha2)
        )


def test_what_is_applied_is_a_copy_of_the_table(rel):
    two = rel["two"]
    assert isinstance(two._inc_tab["ratio"], jnp.ndarray) and isinstance(two._inc_z, tuple)
    assert not two._inc.dndx_ratio.flags.writeable


def test_a_manifest_without_an_optional_part_is_written_byte_for_byte_as_before(tmp_path):
    for part in ("backbone", "mf"):
        (tmp_path / part).mkdir()
    path, sha = RM.write_release_manifest(
        str(tmp_path),
        release_version="v",
        backbone_dir="backbone",
        backbone_manifest_sha256="a" * 64,
        mf_dir="mf",
        mf_record_sha256="b" * 64,
    )
    golden = (
        '{\n "backbone": {\n  "dir": "backbone",\n  "manifest_sha256": "' + "a" * 64 + '"\n },\n'
        ' "format": "hcd_priya.release/1",\n "mf": {\n  "dir": "mf",\n  "record_sha256": "'
        + "b" * 64
        + '"\n },\n'
        ' "release_version": "v"\n}'
    )
    assert (
        open(path, "rb").read() == golden.encode()
        and sha == hashlib.sha256(golden.encode()).hexdigest()
    )
    with pytest.raises(
        RM.ReleaseManifestError, match="both incidence_dir and incidence_record_sha256"
    ):
        RM.write_release_manifest(
            str(tmp_path / "x"),
            release_version="v",
            backbone_dir="backbone",
            backbone_manifest_sha256="a" * 64,
            mf_dir="mf",
            mf_record_sha256="b" * 64,
            incidence_dir="incidence",
        )


# --------------------------------------------------------------------------------------- the held-out error
def _with(block, path, value):
    """A deep copy of the synthetic held-out block with one entry replaced; ``path`` is a tuple of keys/indices."""
    out = json.loads(json.dumps(block))
    node = out
    for key in path[:-1]:
        node = node[key]
    if value is KeyError:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return out


def test_a_record_without_a_held_out_error_is_byte_for_byte_what_it_was(tmp_path):
    a = write_incidence(tmp_path / "a", cache=CACHE)
    b = write_incidence(tmp_path / "b", cache=CACHE, heldout_error=None)
    assert (
        a == b
    )  # the same bytes, so the same digest: the optional key adds nothing when it is absent
    inc = IM.load_record(tmp_path / "a", expected_record_sha256=a)
    assert inc.heldout is None and "heldout_error" not in inc.record
    assert inc.provenance()["heldout_error"] is None
    assert np.all(np.isnan(inc.heldout_dndx(3.0, True))) and inc.heldout_cddf(3.0, False).shape == (
        30,
    )


def test_a_record_carries_its_held_out_error_and_gives_it_only_at_a_tabulated_redshift(tmp_path):
    block = heldout_error()
    sha = write_incidence(tmp_path / "inc", cache=CACHE, heldout_error=block)
    inc = IM.load_record(tmp_path / "inc", expected_record_sha256=sha)
    assert inc.record["heldout_error"] == block and sha != write_incidence(
        tmp_path / "plain", cache=CACHE
    )
    np.testing.assert_array_equal(inc.heldout_dndx(3.0, False), [0.11, 0.12, 0.13])
    np.testing.assert_array_equal(inc.heldout_dndx(3.0, True), [0.05, 0.06, 0.07])
    np.testing.assert_array_equal(
        inc.heldout_dndx(5.0, True), [0.15, np.nan, 0.17]
    )  # null: no estimate
    for z in (
        2.0,
        2.6,
        4.0,
        5.2,
    ):  # not a redshift of the error table: never interpolated, never the nearest
        assert np.all(np.isnan(inc.heldout_dndx(z, True))) and np.all(
            np.isnan(inc.heldout_cddf(z, False))
        )
    c = inc.heldout_cddf(5.0, True)
    assert np.isnan(c[0]) and np.all(c[1:] == 0.21) and c.shape == (30,)
    with pytest.raises(ValueError):  # read-only, like every table of the record
        inc.heldout["dndx_lf"][0, 0] = 9.0
    with pytest.raises(
        TypeError
    ):  # and so is the container: a table cannot be swapped after the load
        inc.heldout["dndx_lfmf"] = inc.heldout["dndx_lf"]
    prov = inc.provenance()["heldout_error"]
    assert prov["kind"] == "rms_log_error_heldout" and prov["z"] == [3.0, 5.0]
    assert (
        prov["dndx_lfmf_n"] == [[5, 5, 5], [5, 0, 5]] and "cddf" not in prov and "dndx" not in prov
    )
    # sealed with the record: a changed error table is not applied
    changed = dict(inc.heldout, dndx_lf=inc.heldout["dndx_lf"] * 2.0)
    with pytest.raises(E, match="changed in memory"):
        IM.reverify(inc._replace(heldout=changed))
    with pytest.raises(E, match="changed in memory"):
        IM.reverify(inc._replace(heldout=None))


@pytest.mark.parametrize(
    "path, value, why",
    [
        (("kind",), "fractional_1sigma", "this code reads 'rms_log_error_heldout' only"),
        (("note",), "x", "unknown \\['note'\\]"),
        (("definition",), " ", "definition must be a non-empty text"),
        (("n_hr_simulations",), 0, "n_hr_simulations must be an integer >= 1"),
        (("z",), [5.0, 3.0], "strictly ascending"),
        (("z",), [3.0], "must have shape \\(1, 3\\)"),
        (("z",), [3.0, 4.0], "not a row of the correction table"),
        (("dndx", "lfmf", 0, 0), None, "null where its count is 5"),
        (("dndx", "lfmf", 1, 1), 0.2, "or null with a count of 0"),
        (("dndx", "lf", 0, 0), -0.1, "finite number >= 0"),
        (("dndx", "lf", 0, 0), True, "finite number >= 0"),
        (("dndx", "lf", 0, 0), 10**400, "finite number >= 0"),  # an integer too large for a float
        (("dndx", "lf_n", 0, 0), 6.0, "not an integer >= 0"),
        (("cddf", "lf", 0), [0.3] * 29, "must have shape \\(2, 30\\)"),
        (("cddf", "extra"), [], "unknown \\['extra'\\]"),
        (("source", "file"), "../x.npz", "plain file name"),
        (("source", "sha256"), "ef" * 31, "sha256"),
        (("source",), KeyError, "missing \\['source'\\]"),
        (("copied_from", "file"), "../t.json", "plain file name"),
        (("copied_from", "sha256"), "cd" * 31, "sha256"),
        (("copied_from",), KeyError, "missing \\['copied_from'\\]"),
    ],
)
def test_a_malformed_held_out_error_is_refused_before_anything_is_written(
    tmp_path, path, value, why
):
    with pytest.raises(E, match=why):
        write_incidence(
            tmp_path / "inc", cache=CACHE, heldout_error=_with(heldout_error(), path, value)
        )
    assert not (tmp_path / "inc").exists()


def test_a_table_file_becomes_the_block_only_if_it_fits_the_record():
    centres = INCIDENCE_TABLES["log_nhi_centres"]
    stated = heldout_error()["copied_from"]  # what the builder states: not a key of the file
    doc = dict(
        {k: v for k, v in heldout_error().items() if k != "copied_from"},
        class_names=["LLS", "subDLA", "DLA"],
        log_nhi_centres=list(centres),
        pooled_dndx_z2p2_to_4p6={"lf": [0.1, 0.1, 0.1]},  # for documentation: not copied
    )
    kw = dict(
        log_nhi_centres=centres,
        n_train_sims=2,
        table_file=stated["file"],
        table_sha256=stated["sha256"],
    )
    block = IM.heldout_block_from_table(doc, **kw)
    assert block == heldout_error() and "pooled_dndx_z2p2_to_4p6" not in block
    for bad, why in (
        (dict(doc, class_names=["DLA", "subDLA", "LLS"]), "class_names must be"),
        (dict(doc, log_nhi_centres=[c + 0.1 for c in centres]), "bins are not the record's"),
        (dict(doc, something="else"), "unknown \\['something'\\]"),
        ({k: v for k, v in doc.items() if k != "class_names"}, "missing \\['class_names'\\]"),
        (dict(doc, kind="another"), "this code reads"),
        (
            dict(doc, copied_from=stated),
            "unknown \\['copied_from'\\]",
        ),  # the file cannot state it itself
    ):
        with pytest.raises(E, match=why):
            IM.heldout_block_from_table(bad, **kw)
    # measured for another number of HR simulations than the record uses: refused unless it is said to be intended
    with pytest.raises(E, match="measured for a correction of 2 HR simulations"):
        IM.heldout_block_from_table(doc, **dict(kw, n_train_sims=3))
    same = IM.heldout_block_from_table(doc, **dict(kw, n_train_sims=3), other_composition=True)
    assert same == block


@pytest.fixture()
def rel_err(tmp_path, monkeypatch):
    """A release whose incidence record carries the synthetic held-out error; the once-per-process memory of the
    warning is emptied for the test."""
    monkeypatch.setattr(hp, "_HELDOUT_WARNED", set())
    info = make_release(tmp_path / "err", incidence=True, heldout=True)
    return hp.load_release(str(info["root"]), info["sha"])


def _heldout_warnings(caught):
    return [w for w in caught if issubclass(w.category, hp.HeldOutErrorWarning)]


def test_a_prediction_carries_the_held_out_error_of_its_fidelity(rel_err):
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mf = rel_err.predict(PARAMS, 3.0, tau_eff=0.4)
        lf = rel_err.predict(PARAMS, 3.0, tau_eff=0.4, fidelity="LF")
    np.testing.assert_array_equal(mf.dndx_err, [0.05, 0.06, 0.07])  # the corrected value's error
    np.testing.assert_array_equal(lf.dndx_err, [0.11, 0.12, 0.13])  # the uncorrected value's error
    assert mf.cddf_err.shape == mf.cddf.shape == (30,) and not mf.dndx_err.flags.writeable
    assert np.array_equal(
        np.isnan(mf.cddf_err), np.isnan(mf.cddf)
    )  # an estimate beside every number, here
    assert np.all(mf.cddf_err[np.isfinite(mf.cddf)] == 0.2) and np.all(lf.cddf_err[1:] == 0.3)
    assert mf.flags["incidence_heldout_error_available"] is True
    assert lf.flags["incidence_heldout_error_available"] is True
    assert not _heldout_warnings(
        caught
    ), "a finite estimate beside every corrected value: nothing to warn about"
    label = mf.labels["incidence_mf_error"]
    assert label.startswith(
        "The held-out error of this prediction's dndx and cddf is in dndx_err and cddf_err"
    )
    for text in (
        "A synthetic definition of the synthetic numbers.",
        "rms_log_error_heldout",
        "2 in number",
    ):
        assert text in label
    assert "No held-out error" not in label
    prov = mf.provenance["incidence_mf"]["heldout_error"]
    assert prov["definition"] == "A synthetic definition of the synthetic numbers." and prov[
        "z"
    ] == [3.0, 5.0]


def test_no_estimate_is_nan_and_is_said_once_per_record_and_process(rel_err):
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lf = rel_err.predict(
            PARAMS, 5.0, tau_eff=1.5, fidelity="LF"
        )  # an LF prediction never warns
        assert not _heldout_warnings(caught)
        mf = rel_err.predict(
            PARAMS, 5.0, tau_eff=1.5
        )  # the subDLA class has no LF+MF estimate at z 5.0
        again = rel_err.predict(PARAMS, 5.0, tau_eff=1.5)
        between = rel_err.predict(
            PARAMS, 2.6, tau_eff=0.4
        )  # not a redshift of the error table at all
    np.testing.assert_array_equal(mf.dndx_err, [0.15, np.nan, 0.17])
    assert mf.flags["incidence_heldout_error_available"] is False
    assert lf.flags["incidence_heldout_error_available"] is True
    assert np.all(np.isnan(between.dndx_err)) and np.all(np.isnan(between.cddf_err))
    assert np.array_equal(again.dndx_err, mf.dndx_err, equal_nan=True)
    said = _heldout_warnings(caught)
    assert len(said) == 1, "once per process for this record, not once per call"
    assert "has no held-out estimate for some class, bin or redshift" in str(said[0].message)
    assert said[0].filename == __file__, "the warning points at the caller's line"
    stack = rel_err.predict_many(PARAMS, [3.0, 5.0], tau_eff=[0.4, 1.5])
    np.testing.assert_array_equal(stack.dndx_err, [[0.05, 0.06, 0.07], [0.15, np.nan, 0.17]])
    assert (
        stack.cddf_err.shape == (2, 30)
        and stack.flags["incidence_heldout_error_available"] is False
    )


def test_a_record_without_a_held_out_error_gives_nan_and_warns_once(rel, monkeypatch):
    import warnings

    monkeypatch.setattr(hp, "_HELDOUT_WARNED", set())
    two, one = rel["two"], rel["one"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = two.predict(PARAMS, 3.0, tau_eff=0.4)
        two.predict(PARAMS, 3.0, tau_eff=0.4)
        q = one.predict(
            PARAMS, 3.0, tau_eff=0.4
        )  # no incidence record: nothing corrected, nothing to say
    assert (
        p.dndx_err.shape == (3,) and np.all(np.isnan(p.dndx_err)) and np.all(np.isnan(p.cddf_err))
    )
    assert p.flags["incidence_heldout_error_available"] is False
    assert "No held-out error is available for this record" in p.labels["incidence_mf_error"]
    assert q.dndx_err is None and q.cddf_err is None
    assert "incidence_heldout_error_available" not in q.flags
    said = _heldout_warnings(caught)
    assert len(said) == 1 and "carries no held-out error" in str(said[0].message)
    with warnings.catch_warnings():  # the standard filter turns it into an error
        warnings.simplefilter("error", hp.HeldOutErrorWarning)
        monkeypatch.setattr(hp, "_HELDOUT_WARNED", set())
        for _ in range(
            2
        ):  # an error every time: a caller that catches the first is not left without a word
            with pytest.raises(hp.HeldOutErrorWarning):
                two.predict(PARAMS, 3.0, tau_eff=0.4)


# ------------------------------------------------------------------------------------------------- the log
def test_a_call_that_applies_a_correction_says_so_once_in_the_log(rel, caplog):
    import logging

    two, one, info = rel["two"], rel["one"], rel["info"]
    name = "hcd_analysis.predictor"
    # the library configures no logging: its logger has no level and no handler of its own
    assert logging.getLogger(name).level == logging.NOTSET and not logging.getLogger(name).handlers
    with caplog.at_level(logging.INFO, logger=name):
        two.predict(PARAMS, 3.0, tau_eff=0.4)
    (record,) = [r for r in caplog.records if r.name == name]
    text = record.getMessage()
    assert record.levelno == logging.INFO and "LF+MF" in text and "at 1 redshift(s)" in text
    assert f"the P1D MF correction (record sha256 {info['mf_sha']})" in text
    assert f"dN/dX and CDDF (record sha256 {info['incidence_sha']})" in text
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=name):
        two.predict_many(
            PARAMS, [2.6, 3.0, 5.2], tau_eff=0.4
        )  # one record for the call, not one per redshift
        one.predict(
            PARAMS, 3.0, tau_eff=0.4
        )  # no incidence record: only the P1D correction is named
        two.predict(PARAMS, 3.0, tau_eff=0.4, fidelity="LF")  # nothing applied: nothing logged
    texts = [r.getMessage() for r in caplog.records if r.name == name]
    assert len(texts) == 2 and "at 3 redshift(s)" in texts[0]
    assert "P1D MF correction" in texts[1] and "incidence" not in texts[1]
    # so at the default level (WARNING, from the root logger) nothing is recorded
    caplog.clear()
    assert logging.getLogger(name).level == logging.NOTSET  # caplog put back what it found
    with caplog.at_level(logging.WARNING):
        two.predict(PARAMS, 3.0, tau_eff=0.4)
    assert not [r for r in caplog.records if r.name == name]
    assert not logging.getLogger(name).handlers


# ------------------------------------------------------------------------- a fold model with an incidence record
def _fold(tmp_path):
    """A synthetic research bundle that holds out synthsim2, with its fold MF record: (bundle, the loader keywords,
    the cache sha256)."""
    from hcd_analysis.emulator import release_testing as RT
    from tests.test_predictor_mf import write_mf

    res = RT.write_synthetic_research_bundle(
        str(tmp_path / "res"), held_out_sims=("synthsim2",), n_groups=3
    )
    cache = res.lineage["cache"]["sha256"]
    s = write_mf(
        tmp_path / "fold",
        manifest=res.manifest_sha256,
        cache=cache,
        kind="fold",
        backbone_kind="research",
        held_out=["synthsim2"],
        fold={"fold_id": "f0", "holdout_sims": ["synthsim2"]},
    )
    return res, dict(purpose="a test", mf_dir=str(tmp_path / "fold"), mf_record_sha256=s), cache


def _fold_incidence(path, cache, *, train, excluded):
    return write_incidence(
        path,
        cache=cache,
        train_sims=train,
        composition={
            "id": "synthetic-fold-composition",
            "excluded_sims": excluded,
            "ruling_ids": ["synthetic ruling"],
        },
    )


def test_a_fold_model_takes_an_incidence_record_built_without_its_held_out_simulations(
    tmp_path, monkeypatch
):
    """``load_research_bundle(..., incidence_dir=..., incidence_record_sha256=...)``: the fold model's LF+MF dN/dX
    is multiplied by the record's factor, the class spectra are untouched, an LF prediction is untouched, and without
    the two keywords the loader does what it did."""
    import warnings

    monkeypatch.setattr(hp, "_HELDOUT_WARNED", set())
    res, kw, cache = _fold(tmp_path)
    sha = _fold_incidence(
        tmp_path / "inc", cache, train=["synthsim0", "synthsim1"], excluded=["synthsim2"]
    )
    plain = hp.load_research_bundle(res.bundle_dir, res.manifest_sha256, **kw)
    rp = hp.load_research_bundle(
        res.bundle_dir,
        res.manifest_sha256,
        **kw,
        incidence_dir=str(tmp_path / "inc"),
        incidence_record_sha256=sha,
    )
    assert type(rp) is hp.ResearchPredictor and plain._inc is None
    a = plain.predict(PARAMS, 3.0, tau_eff=0.4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = rp.predict(PARAMS, 3.0, tau_eff=0.4)
    assert len(_heldout_warnings(caught)) == 1  # the synthetic record carries no held-out error
    assert p.flags["research_model"] is True and p.flags["mf_applied"] is True
    assert p.flags["incidence_mf_applied"] is True and "incidence_mf_applied" not in a.flags
    np.testing.assert_allclose(
        p.dndx, a.dndx * np.array(INCIDENCE_TABLES["dndx_ratio"][1]), rtol=1e-13, atol=0
    )
    assert p.P_class.tobytes() == a.P_class.tobytes()  # the class spectra are not touched
    assert not np.array_equal(p.w_c, a.w_c)  # the fractions follow the corrected incidence
    assert p.provenance["incidence_mf"]["record_sha256"] == sha
    assert p.provenance["model_kind"] == "research" and "incidence_mf" not in a.provenance
    lf, lf0 = (m.predict(PARAMS, 3.0, tau_eff=0.4, fidelity="LF") for m in (rp, plain))
    assert lf.flags["incidence_mf_applied"] is False
    assert lf.dndx.tobytes() == lf0.dndx.tobytes() and lf.w_c.tobytes() == lf0.w_c.tobytes()


def test_a_fold_model_refuses_an_incidence_record_that_is_not_held_out(tmp_path):
    """Refused: one keyword without the other; a record whose table was built with a held-out simulation; a record
    that does not state that it excludes every held-out simulation; a record without the fold MF record.
    """
    res, kw, cache = _fold(tmp_path)
    good = _fold_incidence(
        tmp_path / "good", cache, train=["synthsim0", "synthsim1"], excluded=["synthsim2"]
    )
    leak = _fold_incidence(
        tmp_path / "leak", cache, train=["synthsim1", "synthsim2"], excluded=["synthsim0"]
    )
    silent = _fold_incidence(
        tmp_path / "silent", cache, train=["synthsim0", "synthsim1"], excluded=[]
    )

    def load(name, sha, **over):
        return hp.load_research_bundle(
            res.bundle_dir,
            res.manifest_sha256,
            **dict(kw, **over),
            incidence_dir=None if name is None else str(tmp_path / name),
            incidence_record_sha256=sha,
        )

    for name, sha in (("good", None), (None, good)):
        with pytest.raises(E, match="both incidence_dir and incidence_record_sha256"):
            load(name, sha)
    with pytest.raises(E, match="not a held-out prediction"):
        load("leak", leak)
    with pytest.raises(E, match="does not state that it excludes"):
        load("silent", silent)
    with pytest.raises(E, match="applied with the P1D MF record"):
        load("good", good, mf_dir=None, mf_record_sha256=None)
    with pytest.raises(E, match="sha256 does not match"):
        load("good", leak)
    assert load("good", good).provenance["incidence_mf"]["record_sha256"] == good
