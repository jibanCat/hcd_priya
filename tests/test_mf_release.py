"""The release multi-fidelity (MF) record (hcd_analysis/emulator/mf_release.py; docs/release_format_h5.md, section
"The multi-fidelity record"). Synthetic values only: made-up simulations, made-up correction tables, made-up digests.

Predicates checked here: write-once; the composition and its ruling ids recorded and checked at load; the cache and
backbone digests recorded and checked at load; the record anchored by its own sha256 and the tables by theirs; a strict
HDF5 structure; the HR catalogue-derived columns excluded (a tested refusal); the mean-flux convention recorded as
the settled global convention; a release record refused from a dirty tree; held-out (fold) fits that exclude
simulation groups."""
import hashlib
import json
import os

import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401  (float64)
from hcd_analysis.emulator import mf_modes as MM
from hcd_analysis.emulator import mf_release as MR
from hcd_analysis.emulator.data import Z_LIMITS
from hcd_analysis.emulator.schema import L_BOX_HMPC

K = 8
SIMS = ("synthA", "synthB", "synthC", "synthD")
ZS = (2.6, 3.0, 3.4)
RUNGS = (0, 1, 2)
H64 = lambda s: hashlib.sha256(s.encode()).hexdigest()          # noqa: E731  (made-up digests)
BACKBONE = H64("synthetic backbone manifest")
LF_SHA, HR_SHA = H64("synthetic lf cache"), H64("synthetic hr cache")
COMMIT = "a" * 40
KCOM = 2 * np.pi * np.arange(1, K + 1) / L_BOX_HMPC


def synthetic_pairs(seed=0, sims=SIMS):
    """Matched synthetic LF and HR caches (dicts as data.load_cache returns them), one row per (sim, z, rung) in
    both, with catalogue-derived HR columns present."""
    rng = np.random.default_rng(seed)
    rows = [(s, z, r) for s in sims for z in ZS for r in RUNGS]
    n = len(rows)
    sim_par = {s: rng.uniform(0.2, 0.8, 9) for s in sims}
    from hcd_analysis.emulator.data import PARAM_LIMITS
    lim = np.asarray(PARAM_LIMITS, float)
    pu = np.stack([sim_par[s] for s, _, _ in rows])
    params = lim[:, 0] + pu * (lim[:, 1] - lim[:, 0])
    z = np.array([z for _, z, _ in rows])
    zu = (z - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])
    a = np.array([r for _, _, r in rows], dtype=np.int32)
    tau0 = 0.2 + 0.1 * a + 0.05 * (z - 2.6)
    kf = np.tile(np.linspace(1e-3, 5e-2, K + 4), (n, 1))
    base = dict(params=params, z_grid=z, alpha_idx=a, sim_name=np.array([s for s, _, _ in rows]), kfkms=kf,
                x=np.concatenate([pu, zu[:, None]], axis=1), tau0=tau0, k_com_hmpc=KCOM)
    lf = dict(base, P_filt=rng.uniform(0.5, 2.0, (n, 4, K + 4)))
    hr = dict(base, P_filt=rng.uniform(0.5, 2.0, (n, 4, K + 4)),
              snap_dNdX=rng.uniform(size=(n, 3)), snap_f_nhi=rng.uniform(size=(n, 5)),
              snap_n_absorbers=rng.uniform(size=n), snap_total_path_dX=rng.uniform(size=n))
    return lf, hr


def synthetic_targets(seed=0, sims=SIMS):
    lf, hr = synthetic_pairs(seed, sims)
    pairs = MM.match_pairs(lf, hr)
    l_rows = np.array([p[1] for p in pairs])
    P_lf = np.asarray(lf["P_filt"])[l_rows][:, :, :K] * 1.1
    return MM.measure_mode_targets(lf, MR.hr_view(hr), pairs, P_lf, x=lf["x"], n_modes=K)


def record_fields(**over):
    out = dict(kind="release", mf_version="synthetic-mf-1",
               backbone={"manifest_sha256": BACKBONE, "kind": "release", "cache_sha256": LF_SHA,
                         "held_out_sims": []},
               caches={"lf": {"file": "lf_cache.h5", "sha256": LF_SHA},
                       "hr": {"file": "hr_cache.h5", "sha256": HR_SHA}},
               composition={"id": "synthetic-xD", "excluded_sims": ["synthD"],
                            "ruling_ids": ["synthetic ruling 1"]},
               fold=None, fit_code={"commit": COMMIT, "dirty": False})
    out.update(over)
    return out


def build(tmp_path, name="mf", targets=None, **over):
    t = synthetic_targets() if targets is None else targets
    f = record_fields(**over)
    rows, train = MR.select_train_rows(t, exclude_sims=f["composition"]["excluded_sims"],
                                       holdout_sims=(f["fold"] or {}).get("holdout_sims", ()))
    tables = MM.fit_mode_mf(t, train_rows=rows)
    out = str(tmp_path / name)
    path, sha = MR.write_mf_record(out, tables, k_com_hmpc=KCOM, train_sims=train, n_pairs_train=len(rows), **f)
    return out, sha, tables


def load(d, sha, **kw):
    kw.setdefault("backbone_manifest_sha256", BACKBONE)
    return MR.load_mf_record(d, expected_record_sha256=sha, **kw)


def rewrite(d, fn):
    """Tamper with the record JSON (fn(dict) -> dict) and return the new sha256 (an attacker re-anchors)."""
    p = os.path.join(d, MR.RECORD_FILE)
    rec = json.load(open(p))
    raw = json.dumps(fn(rec), indent=1, sort_keys=True).encode()
    os.chmod(p, 0o644)
    with open(p, "wb") as fh:
        fh.write(raw)
    return hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------------------------- write once (X0.1)
def test_record_is_written_once_and_refuses_an_existing_directory(tmp_path):
    d, sha, _ = build(tmp_path)
    before = {f: open(os.path.join(d, f), "rb").read() for f in os.listdir(d)}
    with pytest.raises(FileExistsError, match="exists"):
        build(tmp_path)                                            # same directory again
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileExistsError, match="exists"):
        build(tmp_path, name="empty")                              # an existing, even empty, directory
    os.symlink(str(tmp_path / "nowhere"), str(tmp_path / "link"))
    with pytest.raises(FileExistsError, match="exists"):
        build(tmp_path, name="link")                               # a dangling symlink
    assert {f: open(os.path.join(d, f), "rb").read() for f in os.listdir(d)} == before
    assert sorted(os.listdir(d)) == sorted([MR.RECORD_FILE, MR.TABLES_FILE])


# --------------------------------------------------------------------------------------------- round trip
def test_round_trip_reproduces_the_tables_and_the_correction(tmp_path):
    t = synthetic_targets()
    d, sha, tables = build(tmp_path, targets=t)
    rec = load(d, sha)
    assert rec.kind == "release" and rec.record_sha256 == sha and rec.anchored is True
    for k in MM.TABLE_KEYS:
        np.testing.assert_array_equal(rec.tables[k], np.asarray(tables[k], float))
    np.testing.assert_array_equal(rec.k_com_hmpc, KCOM)
    for i in (0, 7, 20):
        np.testing.assert_array_equal(rec.g(t["x"][i], t["tau0"][i]), MM.apply_mode_mf(tables, t["x"][i], t["tau0"][i]))
    assert rec.record["train_sims"] == ["synthA", "synthB", "synthC"]
    assert rec.record["composition"]["ruling_ids"] == ["synthetic ruling 1"]


def test_record_carries_the_required_provenance_fields(tmp_path):
    d, sha, _ = build(tmp_path)
    r = load(d, sha).record
    assert r["format"] == MR.RECORD_FORMAT and r["mf_version"] == "synthetic-mf-1"
    assert r["backbone"]["manifest_sha256"] == BACKBONE
    assert r["caches"]["lf"]["sha256"] == LF_SHA and r["caches"]["hr"]["sha256"] == HR_SHA
    assert r["fit_code"] == {"commit": COMMIT, "dirty": False}
    assert r["mean_flux_convention"]["status"] == "SETTLED"
    assert "priya_p1d.py" in r["mean_flux_convention"]["source"]
    assert r["hr_inputs"] == MR.HR_INPUTS
    assert r["tables"]["sha256"] == hashlib.sha256(open(os.path.join(d, MR.TABLES_FILE), "rb").read()).hexdigest()


# --------------------------------------------------------------------------------------------- anchoring
def test_record_and_tables_are_anchored(tmp_path):
    d, sha, _ = build(tmp_path)
    with pytest.raises(MR.MFRecordError, match="sha256"):
        load(d, "0" * 64)
    with pytest.raises(MR.MFRecordError, match="published"):
        load(d, None)
    tp = os.path.join(d, MR.TABLES_FILE)
    import h5py
    size = os.path.getsize(tp)
    os.chmod(tp, 0o644)
    with h5py.File(tp, "r+") as f:                       # a changed table VALUE, same layout and size
        f["mf"]["log_rho"][0] = f["mf"]["log_rho"][0] + 1.0
    assert os.path.getsize(tp) == size
    with pytest.raises(MR.MFRecordError, match="tables"):
        load(d, sha)


def test_strict_tables_file_refuses_an_extra_dataset_or_attribute(tmp_path):
    import h5py
    d, sha, _ = build(tmp_path)
    for mutate in (lambda f: f["mf"].create_dataset("extra", data=np.zeros(2)),
                   lambda f: f.attrs.__setitem__("note", "x"),
                   lambda f: f["mf"]["log_rho"].attrs.__setitem__("unit", "x")):
        e = tmp_path / f"e{id(mutate)}"
        os.makedirs(e)
        for n in os.listdir(d):
            open(e / n, "wb").write(open(os.path.join(d, n), "rb").read())
        with h5py.File(e / MR.TABLES_FILE, "a") as f:
            mutate(f)
        new = hashlib.sha256(open(e / MR.TABLES_FILE, "rb").read()).hexdigest()
        s2 = rewrite(str(e), lambda r: dict(r, tables=dict(r["tables"], sha256=new,
                                                             bytes=os.path.getsize(e / MR.TABLES_FILE))))
        with pytest.raises(MR.MFRecordError):
            load(str(e), s2)


def test_record_refuses_unknown_and_missing_keys(tmp_path):
    d, sha, _ = build(tmp_path)
    s2 = rewrite(d, lambda r: dict(r, extra_claim="anything"))
    with pytest.raises(MR.MFRecordError, match="keys"):
        load(d, s2)
    s3 = rewrite(d, lambda r: {k: v for k, v in r.items() if k not in ("extra_claim", "fit_code")})
    with pytest.raises(MR.MFRecordError, match="keys"):
        load(d, s3)


def test_a_legacy_npz_product_is_not_a_release_record(tmp_path):
    t = synthetic_targets()
    tables = MM.fit_mode_mf(t)
    MM.save_mode_mf(str(tmp_path / "legacy.npz"), tables, k_com_hmpc=KCOM, provenance={"v": 1})
    with pytest.raises(MR.MFRecordError):
        load(str(tmp_path / "legacy.npz"), H64("x"))
    with pytest.raises(MR.MFRecordError):
        load(str(tmp_path), H64("x"))                               # a directory without a record


# --------------------------------------------------------------------------------------------- composition (X0.2)
def test_composition_excludes_its_simulations_and_is_checked_at_load(tmp_path):
    t = synthetic_targets()
    d, sha, tables = build(tmp_path, targets=t)
    rows, train = MR.select_train_rows(t, exclude_sims=["synthD"])
    assert "synthD" not in set(t["sim"][rows]) and set(t["sim"][rows]) == {"synthA", "synthB", "synthC"}
    ref = MM.fit_mode_mf(t, train_rows=np.where(t["sim"] != "synthD")[0])
    for k in MM.TABLE_KEYS:
        np.testing.assert_array_equal(np.asarray(tables[k]), np.asarray(ref[k]))
    comp = {"id": "synthetic-xD", "excluded_sims": ["synthD"]}
    assert load(d, sha, composition=comp).record["composition"]["id"] == "synthetic-xD"
    with pytest.raises(MR.MFRecordError, match="composition"):
        load(d, sha, composition={"id": "synthetic-all", "excluded_sims": []})
    with pytest.raises(MR.MFRecordError, match="composition"):
        load(d, sha, composition={"id": "synthetic-xD", "excluded_sims": ["synthC"]})


def test_composition_refusals_at_write(tmp_path):
    with pytest.raises(MR.MFRecordError, match="ruling"):
        build(tmp_path, name="a", composition={"id": "x", "excluded_sims": ["synthD"], "ruling_ids": []})
    with pytest.raises(MR.MFRecordError, match="unknown"):
        MR.select_train_rows(synthetic_targets(), exclude_sims=["no_such_sim"])
    with pytest.raises(MR.MFRecordError, match="id"):
        build(tmp_path, name="b", composition={"id": "", "excluded_sims": [], "ruling_ids": ["r"]})
    t = synthetic_targets()
    rows, train = MR.select_train_rows(t, exclude_sims=["synthD"])
    tables = MM.fit_mode_mf(t, train_rows=rows)
    with pytest.raises(MR.MFRecordError, match="train"):                # an excluded simulation among the training set
        MR.write_mf_record(str(tmp_path / "c"), tables, k_com_hmpc=KCOM, train_sims=list(SIMS),
                           n_pairs_train=len(rows), **record_fields())


# --------------------------------------------------------------------------------------------- digests (X0.3, X0.4)
def test_cache_and_backbone_digests_are_checked_at_load(tmp_path):
    d, sha, _ = build(tmp_path)
    assert load(d, sha, lf_cache_sha256=LF_SHA, hr_cache_sha256=HR_SHA).kind == "release"
    with pytest.raises(MR.MFRecordError, match="LF cache"):
        load(d, sha, lf_cache_sha256=H64("other"))
    with pytest.raises(MR.MFRecordError, match="HR cache"):
        load(d, sha, hr_cache_sha256=H64("other"))
    with pytest.raises(MR.MFRecordError, match="backbone"):
        load(d, sha, backbone_manifest_sha256=H64("another backbone"))
    with pytest.raises(MR.MFRecordError, match="backbone"):
        MR.load_mf_record(d, expected_record_sha256=sha, backbone_manifest_sha256=None)


def test_release_record_needs_its_lf_cache_to_be_the_backbone_cache(tmp_path):
    with pytest.raises(MR.MFRecordError, match="cache"):
        build(tmp_path, caches={"lf": {"file": "lf.h5", "sha256": H64("another lf")},
                                "hr": {"file": "hr.h5", "sha256": HR_SHA}})
    with pytest.raises(MR.MFRecordError, match="file name"):
        build(tmp_path, name="p", caches={"lf": {"file": "/private/dir/lf.h5", "sha256": LF_SHA},
                                          "hr": {"file": "hr.h5", "sha256": HR_SHA}})


def test_every_kind_needs_its_lf_cache_to_be_the_backbone_cache(tmp_path):
    """A fold (or historical) record's targets were formed with its backbone, so its LF cache is that backbone's
    training cache: refused at write and at load (check_record), not only when attached to a predictor."""
    other = {"lf": {"file": "lf.h5", "sha256": H64("another lf")}, "hr": {"file": "hr.h5", "sha256": HR_SHA}}
    fold = {"fold_id": "synthetic-fold-0", "holdout_sims": ["synthA"]}
    rb = {"manifest_sha256": BACKBONE, "kind": "research", "cache_sha256": LF_SHA, "held_out_sims": ["synthA"]}
    with pytest.raises(MR.MFRecordError, match="fold MF's LF cache must be its backbone's training cache"):
        build(tmp_path, name="f", kind="fold", fold=fold, backbone=rb, caches=other)
    with pytest.raises(MR.MFRecordError, match="historical MF's LF cache must be its backbone's training cache"):
        build(tmp_path, name="h", kind="historical", caches=other)
    d, sha, _ = build(tmp_path, name="ok", kind="fold", fold=fold, backbone=rb)
    s2 = rewrite(d, lambda r: dict(r, caches=dict(r["caches"], lf=dict(r["caches"]["lf"], sha256=H64("another lf")))))
    with pytest.raises(MR.MFRecordError, match="fold MF's LF cache must be its backbone's training cache"):
        load(d, s2, kind="fold")


# --------------------------------------------------------------------------------------------- clean tree
def test_release_record_from_a_dirty_tree_is_refused(tmp_path):
    with pytest.raises(MR.MFRecordError, match="dirty"):
        build(tmp_path, fit_code={"commit": COMMIT, "dirty": True})
    d, sha, _ = build(tmp_path, name="ok")
    s2 = rewrite(d, lambda r: dict(r, fit_code={"commit": COMMIT, "dirty": True}))
    with pytest.raises(MR.MFRecordError, match="dirty"):
        load(d, s2)


# ------------------------------------------------------------------------------------- HR catalogue columns (06b A1)
def test_hr_view_refuses_catalogue_derived_columns():
    _, hr = synthetic_pairs()
    v = MR.hr_view(hr)
    for name in ("snap_dNdX", "snap_f_nhi", "snap_n_absorbers", "snap_total_path_dX"):
        with pytest.raises(MR.MFRecordError, match="catalogue-derived"):
            v[name]
    with pytest.raises(MR.MFRecordError, match="not an MF input"):
        v["w_c_cache"]
    assert set(v) <= set(MR.HR_VIEW_KEYS)


def test_targets_do_not_depend_on_the_hr_catalogue_columns():
    lf, hr = synthetic_pairs()
    pairs = MM.match_pairs(lf, hr)
    P_lf = np.asarray(lf["P_filt"])[[p[1] for p in pairs]][:, :, :K]
    a = MM.measure_mode_targets(lf, MR.hr_view(hr), pairs, P_lf, x=lf["x"], n_modes=K)
    hr2 = dict(hr, **{k: np.full_like(hr[k], np.nan) for k in ("snap_dNdX", "snap_f_nhi", "snap_n_absorbers",
                                                              "snap_total_path_dX")})
    b = MM.measure_mode_targets(lf, MR.hr_view(hr2), pairs, P_lf, x=lf["x"], n_modes=K)
    np.testing.assert_array_equal(a["g"], b["g"])


def test_record_claiming_hr_catalogue_columns_or_a_verified_status_is_refused(tmp_path):
    d, sha, _ = build(tmp_path)
    s2 = rewrite(d, lambda r: dict(r, hr_inputs=dict(r["hr_inputs"], used=r["hr_inputs"]["used"] + ["snap_dNdX_DLA"])))
    with pytest.raises(MR.MFRecordError, match="catalogue"):
        load(d, s2)
    s3 = rewrite(d, lambda r: dict(r, hr_inputs=dict(MR.HR_INPUTS, catalogue_derived_status="VERIFIED")))
    with pytest.raises(MR.MFRecordError, match="catalogue"):
        load(d, s3)


# --------------------------------------------------------------------------------------------- mean-flux convention
def test_mean_flux_convention_is_recorded_as_settled_and_cannot_be_altered(tmp_path):
    """The global convention is the settled production convention: the record carries the code's statement, and a
    record that calls it proposed, names another convention or drops the production statement is refused."""
    d, sha, _ = build(tmp_path)
    assert load(d, sha).record["mean_flux_convention"] == MR.MEAN_FLUX_CONVENTION
    assert MR.MEAN_FLUX_CONVENTION["status"] == "SETTLED"
    for bad in (dict(MR.MEAN_FLUX_CONVENTION, status="PROPOSED"), dict(MR.MEAN_FLUX_CONVENTION, name="per-subset"),
                {k: v for k, v in MR.MEAN_FLUX_CONVENTION.items() if k != "production"}):
        s2 = rewrite(d, lambda r: dict(r, mean_flux_convention=bad))
        with pytest.raises(MR.MFRecordError, match="mean-flux"):
            load(d, s2)


def test_mean_flux_convention_text_is_public_safe():
    """The settled convention is stated without private references (no ruling ids, no audit wording)."""
    text = " ".join(str(v) for v in MR.MEAN_FLUX_CONVENTION.values())
    for word in ("PI ruling", "ruling", "under audit", "PROPOSED", "D2", "H6", "2026-"):
        assert word not in text, word
    assert "production convention" in MR.MEAN_FLUX_CONVENTION["production"]
    assert "diagnostic" in MR.MEAN_FLUX_CONVENTION["production"]


# --------------------------------------------------------------------------------------------- kinds
def test_historical_record_loads_only_when_asked_for_as_historical(tmp_path):
    d, sha, _ = build(tmp_path, kind="historical", composition={"id": "synthetic-all", "excluded_sims": [],
                                                                "ruling_ids": ["synthetic pilot"]})
    with pytest.raises(MR.MFRecordError, match="historical"):
        load(d, sha)
    assert load(d, sha, kind="historical").kind == "historical"


# --------------------------------------------------------------------------------------------- folds (06b D1)
def test_select_train_rows_holds_out_simulation_groups():
    t = synthetic_targets()
    rows, train = MR.select_train_rows(t, exclude_sims=["synthD"], holdout_sims=["synthA", "synthB"])
    assert train == ["synthC"] and set(t["sim"][rows]) == {"synthC"}
    with pytest.raises(MR.MFRecordError, match="both"):
        MR.select_train_rows(t, exclude_sims=["synthD"], holdout_sims=["synthD"])
    with pytest.raises(MR.MFRecordError, match="no training"):
        MR.select_train_rows(t, exclude_sims=["synthD"], holdout_sims=["synthA", "synthB", "synthC"])
    with pytest.raises(MR.MFRecordError, match="unknown"):
        MR.select_train_rows(t, holdout_sims=["nope"])
    assert MR.heldout_rows(t, ["synthA"]).tolist() == np.where(t["sim"] == "synthA")[0].tolist()


def test_fold_record_states_whether_its_backbone_excludes_the_held_out_group(tmp_path):
    fold = {"fold_id": "synthetic-fold-0", "holdout_sims": ["synthA"]}
    rb = {"manifest_sha256": BACKBONE, "kind": "research", "cache_sha256": LF_SHA, "held_out_sims": ["synthA"]}
    d, sha, _ = build(tmp_path, kind="fold", fold=fold, backbone=rb)
    r = load(d, sha, kind="fold").record
    assert r["fold"]["backbone_excludes_holdout"] is True and "synthA" not in r["train_sims"]
    rb2 = dict(rb, kind="release", held_out_sims=[])
    d2, sha2, _ = build(tmp_path, name="insample", kind="fold", fold=fold, backbone=rb2)
    assert load(d2, sha2, kind="fold").record["fold"]["backbone_excludes_holdout"] is False
    s3 = rewrite(d2, lambda r: dict(r, fold=dict(r["fold"], backbone_excludes_holdout=True)))
    with pytest.raises(MR.MFRecordError, match="backbone_excludes_holdout"):
        load(d2, s3, kind="fold")
    with pytest.raises(MR.MFRecordError, match="fold"):
        load(d, sha)                                               # a fold record is not a release record
    with pytest.raises(MR.MFRecordError, match="fold"):
        build(tmp_path, name="nofold", kind="fold", fold=None)
    with pytest.raises(MR.MFRecordError, match="release"):
        build(tmp_path, name="relfold", kind="release", fold=fold)
    with pytest.raises(MR.MFRecordError, match="research"):        # a release MF never sits on a research backbone
        build(tmp_path, name="relres", backbone=rb)


# ------------------------------------------------------------------------- cache provenance (C1 review MAJOR-1)
def test_cache_provenance_says_hr_is_digest_anchored_only_and_lf_lineage_verified(tmp_path):
    """The two cache digests are not symmetric: the LF cache is the bound backbone's lineage-verified training cache,
    the HR cache is digest-anchored only and never lineage-verified (PI ruling 2026-11-06b A1). Both statements are
    code constants written into every record and closed-checked at load."""
    assert MR.CACHE_PROVENANCE["hr"].startswith("digest-anchored only, never lineage-verified:")
    assert MR.CACHE_PROVENANCE["lf"].startswith("lineage-verified through the bound backbone:")
    assert set(MR.CACHE_PROVENANCE) == {"lf", "hr"}
    d, sha, _ = build(tmp_path)
    r = load(d, sha).record
    for side in ("lf", "hr"):
        assert r["caches"][side]["provenance"] == MR.CACHE_PROVENANCE[side]
        assert set(r["caches"][side]) == {"file", "sha256", "provenance"}
    upgraded = MR.CACHE_PROVENANCE["lf"]                        # the HR side may not claim the LF side's lineage
    s2 = rewrite(d, lambda r: dict(r, caches=dict(r["caches"], hr=dict(r["caches"]["hr"], provenance=upgraded))))
    with pytest.raises(MR.MFRecordError, match="caches.hr.provenance .*digest-anchored only"):
        load(d, s2)
    s3 = rewrite(d, lambda r: dict(r, caches=dict(r["caches"], lf=dict(r["caches"]["lf"], provenance="unverified"))))
    with pytest.raises(MR.MFRecordError, match="caches.lf.provenance"):
        load(d, s3)
    s4 = rewrite(d, lambda r: dict(r, caches={s: {"file": c["file"], "sha256": c["sha256"]}
                                              for s, c in r["caches"].items()}))
    with pytest.raises(MR.MFRecordError, match="keys"):          # a record without the statements (the old form)
        load(d, s4)


def test_the_writer_takes_no_cache_provenance_from_the_caller(tmp_path):
    claim = {"lf": {"file": "lf_cache.h5", "sha256": LF_SHA},
             "hr": {"file": "hr_cache.h5", "sha256": HR_SHA, "provenance": MR.CACHE_PROVENANCE["lf"]}}
    with pytest.raises(MR.MFRecordError, match="caches must be"):
        build(tmp_path, caches=claim)
    assert not os.path.lexists(tmp_path / "mf")


# The two cache statements, written out in full (delta review R1-1): changing either is a schema change of the
# write-once record and a deliberate edit of these literals.
LF_STATEMENT = ("lineage-verified through the bound backbone: the backbone bundle's training cache (caches.lf.sha256 "
                "equals backbone.cache_sha256), which the bundle's lineage block and every member meta name; the "
                "builder reads it with data.load_cache, which checks its lineage record")
HR_STATEMENT = ("digest-anchored only, never lineage-verified: the builder pins its sha256 and reads it with "
                "data.load_cache_historical, which checks no lineage (a pre-lineage format 3.3 cache has no lineage "
                "record); only its per-class P1D rows and row keys are read (hr_inputs)")


def test_cache_provenance_statements_are_pinned_exactly(tmp_path):
    """Delta review R1-1 (mutants D1d, D1e): both statements are these exact strings, in the code constant and in a
    written and reloaded record; a clause appended to the HR statement or a changed LF clause fails here."""
    assert MR.CACHE_PROVENANCE == {"lf": LF_STATEMENT, "hr": HR_STATEMENT}
    d, sha, _ = build(tmp_path)
    with open(os.path.join(d, MR.RECORD_FILE)) as fh:
        on_disk = json.load(fh)["caches"]
    assert on_disk["lf"]["provenance"] == LF_STATEMENT and on_disk["hr"]["provenance"] == HR_STATEMENT
    caches = load(d, sha).record["caches"]
    assert caches["lf"]["provenance"] == LF_STATEMENT and caches["hr"]["provenance"] == HR_STATEMENT


@pytest.mark.parametrize("side, altered", [
    ("hr", HR_STATEMENT + "; lineage-verified by a later audit"),           # delta review Q2c
    ("hr", HR_STATEMENT + " "),                                             # Q2b
    ("hr", HR_STATEMENT.replace("never lineage-verified", "lineage-verified")),   # Q2d
    ("hr", "note: " + HR_STATEMENT),
    ("lf", LF_STATEMENT + "; and the HR cache too"),
    ("lf", LF_STATEMENT[:-1]),
])
def test_a_cache_statement_that_differs_by_any_text_is_refused(tmp_path, side, altered):
    """Delta review R1-1 (mutant D1b): the statements are compared by exact equality, not containment, so a record
    carrying the code's statement plus anything else, or less than it, is refused at load."""
    d, _, _ = build(tmp_path)
    s = rewrite(d, lambda r: dict(r, caches=dict(r["caches"], **{side: dict(r["caches"][side], provenance=altered)})))
    with pytest.raises(MR.MFRecordError, match=f"caches.{side}.provenance is not the code's statement"):
        load(d, s)


# ------------------------------------------------------------------- pinned guards (C1 review MAJOR-2, mutants R*)
def test_hr_inputs_are_pinned_exactly():
    """R10c: the HR inputs of record are this exact constant (06b A1); dropping a catalogue-derived name from the
    excluded list, or changing the status, is a deliberate edit of this test."""
    assert MR.HR_INPUTS == {
        "used": ["P_tier_c_filtered", "tier_c_counts", "params", "z_grid", "alpha_idx", "sim_name", "kfkms"],
        "used_as": "the per-class P1D rows (P_filt, the count-weighted collapse of P_tier_c_filtered over the fine "
                   "N_HI bins) and the row keys that match them to LF rows",
        "excluded_catalogue_derived": ["snap_dNdX_LLS", "snap_dNdX_subDLA", "snap_dNdX_DLA", "snap_f_nhi",
                                       "snap_n_absorbers", "snap_total_path_dX"],
        "catalogue_derived_status": "PRE-FIX: not read by the MF; no verified or corrected provenance is claimed for "
                                    "them"}
    for prefix in MR.HR_CATALOGUE_DERIVED:                       # every refused prefix is named as excluded
        assert any(n.startswith(prefix) for n in MR.HR_INPUTS["excluded_catalogue_derived"]), prefix
    assert not set(MR.HR_INPUTS["used"]) & set(MR.HR_INPUTS["excluded_catalogue_derived"])


def test_mean_flux_convention_is_global_and_its_source_line_holds_the_stated_expression():
    """R12a, R12b: the convention is "global" (settled: the production convention), and the cited file:line holds the expression the
    record quotes, inside the named function, so the citation cannot drift silently."""
    import importlib.util
    import re
    mfc = MR.MEAN_FLUX_CONVENTION
    assert mfc["name"] == "global" and mfc["status"] == "SETTLED"
    m = re.match(r"(hcd_analysis/[\w/]+\.py):(\d+) \((.+?)\),", mfc["source"])
    assert m, mfc["source"]
    rel, line, expr = m.group(1), int(m.group(2)), m.group(3)
    assert rel == mfc["function"].rsplit(".", 1)[0].replace(".", "/") + ".py"
    path = importlib.util.find_spec(mfc["function"].rsplit(".", 1)[0]).origin    # works from a wheel install too
    assert path.replace(os.sep, "/").endswith(rel)
    lines = open(path).read().splitlines()
    assert lines[line - 1].strip() == expr == "dflux = np.exp(-tau_eff) / target_F - 1.0"
    defs = [ln for ln in lines[:line] if re.match(r"def \w+\(", ln)]
    assert defs[-1].startswith(f"def {mfc['function'].rsplit('.', 1)[1]}(")


def test_fit_code_dirty_must_be_a_boolean(tmp_path):
    """R13c: a clean-tree flag that is not a boolean (0, "false") is refused, not read as falsy."""
    d, sha, _ = build(tmp_path)
    for bad in (0, "false", None):
        s2 = rewrite(d, lambda r: dict(r, fit_code={"commit": COMMIT, "dirty": bad}))
        with pytest.raises(MR.MFRecordError, match="boolean"):
            load(d, s2)


def test_composition_id_alone_is_checked_at_load(tmp_path):
    """R2b: the same excluded simulations under another composition id is another composition."""
    d, sha, _ = build(tmp_path)
    with pytest.raises(MR.MFRecordError, match="composition"):
        load(d, sha, composition={"id": "synthetic-other", "excluded_sims": ["synthD"]})
    with pytest.raises(MR.MFRecordError, match="composition"):
        load(d, sha, composition={"id": "synthetic-xD", "excluded_sims": ["synthD"], "ruling_ids": ["another"]})


def test_fold_holdout_must_lie_outside_the_training_and_excluded_sets(tmp_path):
    """R11a: a re-anchored fold record that trained on its held-out group, or holds out an excluded simulation, is
    refused at load (D1: such a fold is not a held-out fit)."""
    fold = {"fold_id": "synthetic-fold-0", "holdout_sims": ["synthA"]}
    rb = {"manifest_sha256": BACKBONE, "kind": "research", "cache_sha256": LF_SHA, "held_out_sims": ["synthA"]}
    d, sha, _ = build(tmp_path, kind="fold", fold=fold, backbone=rb)
    s2 = rewrite(d, lambda r: dict(r, train_sims=sorted(r["train_sims"] + ["synthA"])))
    with pytest.raises(MR.MFRecordError, match="outside the training and excluded sets"):
        load(d, s2, kind="fold")
    s3 = rewrite(d, lambda r: dict(r, train_sims=["synthB", "synthC"],
                                   composition=dict(r["composition"], excluded_sims=["synthA", "synthD"])))
    with pytest.raises(MR.MFRecordError, match="outside the training and excluded sets"):
        load(d, s3, kind="fold")
    s4 = rewrite(d, lambda r: dict(r, fold=dict(r["fold"], holdout_sims=[]), train_sims=["synthB", "synthC"]))
    with pytest.raises(MR.MFRecordError, match="non-empty"):
        load(d, s4, kind="fold")


def _copy_dir(src, dst):
    os.makedirs(dst)
    for n in os.listdir(src):
        with open(os.path.join(dst, n), "wb") as fh:
            fh.write(open(os.path.join(src, n), "rb").read())
    return str(dst)


def _retable(tmp_path, d, name, write):
    """A copy of record ``d`` whose tables file is rewritten by ``write(path, arrays)`` and re-anchored (the attacker
    controls both files): returns (dir, the new record sha256)."""
    import h5py
    e = _copy_dir(d, tmp_path / name)
    tp = os.path.join(e, MR.TABLES_FILE)
    with h5py.File(tp, "r") as f:
        arrays = {k: f["mf"][k][...] for k in f["mf"]}
    os.chmod(tp, 0o644)
    os.remove(tp)
    write(tp, arrays)
    new = hashlib.sha256(open(tp, "rb").read()).hexdigest()
    return e, rewrite(e, lambda r: dict(r, tables=dict(r["tables"], sha256=new, bytes=os.path.getsize(tp))))


def _tables_file(fmt=MR.TABLES_FORMAT, userblock=0, extra_top=False):
    def write(path, arrays):
        import h5py
        kw = {"userblock_size": userblock} if userblock else {}
        with h5py.File(path, "w", libver=("earliest", "v110"), **kw) as f:
            f.attrs["format"] = fmt
            g = f.create_group("mf")
            for n in sorted(arrays):
                g.create_dataset(n, data=np.ascontiguousarray(arrays[n], dtype="<f8"))
            if extra_top:
                f.create_group("notes")
    return write


def test_strict_tables_file_refuses_a_user_block_a_second_top_level_entry_and_another_format(tmp_path):
    """R5a, R5b, R5e: a re-anchored tables file with a user block, a top level holding more than /mf, or another
    format value is refused; the same rewrite with none of them loads (the control)."""
    d, sha, _ = build(tmp_path)
    e, s = _retable(tmp_path, d, "control", _tables_file())
    assert load(e, s).kind == "release"
    for name, write, why in (("ub", _tables_file(userblock=512), "user block"),
                             ("top", _tables_file(extra_top=True), "exactly /mf"),
                             ("fmt", _tables_file(fmt="hcd_priya.mf/0"), "format")):
        e, s = _retable(tmp_path, d, name, write)
        with pytest.raises(MR.MFRecordError, match=why):
            load(e, s)


def test_record_directory_refuses_an_extra_file_and_a_symlink(tmp_path):
    """R5c, R5d (the reviewer's forgeries F09 and F07b): an extra file in the record directory, and a record directory
    reached through a symlink, are refused."""
    d, sha, _ = build(tmp_path)
    e = _copy_dir(d, tmp_path / "extra")
    assert load(e, sha).kind == "release"                       # the copy itself loads
    (tmp_path / "extra" / "README.txt").write_text("anything")
    with pytest.raises(MR.MFRecordError, match="unexpected files"):
        load(e, sha)
    os.symlink(d, str(tmp_path / "link"))
    with pytest.raises(MR.MFRecordError, match="not an MF record directory"):
        load(str(tmp_path / "link"), sha)


# --------------------------------------------------------------- exact-type digests (delta review R1-2, Q3g and Q3h)
class LyingStr(str):
    """A str subclass that compares equal to anything: a forged digest that passes `!=` against any hex digest."""
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    __hash__ = str.__hash__


class PlainStr(str):
    """A str subclass with no overrides: its value is a genuine digest, but its type is not str."""


@pytest.mark.parametrize("make", [lambda sha: LyingStr(H64("another record")), lambda sha: PlainStr(sha)],
                         ids=["lying-str-other-digest", "plain-str-subclass-own-digest"])
def test_the_published_record_digest_must_be_an_exact_str(tmp_path, make):
    """A str subclass as the published digest is refused before the file is read: a lying one would otherwise pass
    the digest comparison for any record and be carried as the record's sha256 into provenance."""
    d, sha, _ = build(tmp_path)
    with pytest.raises(MR.MFRecordError, match="published sha256"):
        load(d, make(sha))
    assert type(load(d, sha).record_sha256) is str


@pytest.mark.parametrize("arg", ["backbone_manifest_sha256", "lf_cache_sha256", "hr_cache_sha256"])
@pytest.mark.parametrize("make", [lambda v: LyingStr(H64("another")), lambda v: PlainStr(v)],
                         ids=["lying-str", "plain-str-subclass"])
def test_the_backbone_and_cache_digests_must_be_exact_str(tmp_path, arg, make):
    """The digests a record is checked against at load are exact str: a lying subclass would pass `!=` against any
    record's value, so a record of another backbone or cache would load."""
    d, sha, _ = build(tmp_path)
    genuine = {"backbone_manifest_sha256": BACKBONE, "lf_cache_sha256": LF_SHA, "hr_cache_sha256": HR_SHA}[arg]
    with pytest.raises(MR.MFRecordError, match="exact str"):
        load(d, sha, **{arg: make(genuine)})
    assert load(d, sha, **{arg: genuine}).kind == "release"
