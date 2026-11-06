"""The public predictor's MF path (hcd_analysis/predictor.py with an mf_release record) on synthetic h5/2 bundles and
synthetic MF tables: untrained networks, made-up corrections, made-up digests; the values mean nothing. Checked: the
correction is applied (P_MF = P_LF exp(g), per member, before any k placement), LF-only prediction is untouched,
the MF provenance is reported, and an absent, mismatched, historical or fold MF is refused clearly on a production
bundle."""
import numpy as np
import pytest

import hcd_analysis.predictor as hp
from hcd_analysis.emulator import mf_modes as MM
from hcd_analysis.emulator import mf_release as MR
from hcd_analysis.emulator import release_testing as RT
from hcd_analysis.emulator.data import Z_LIMITS, normalize_params
from hcd_analysis.emulator.schema import L_BOX_HMPC
from tests.test_mf_release import COMMIT, H64, synthetic_targets

K = RT.SYNTHETIC_ARCH["n_k"]
KCOM = 2 * np.pi * np.arange(1, K + 1) / L_BOX_HMPC
PARAMS = {"ns": 0.95, "Ap": 2.0e-9, "herei": 3.8, "heref": 2.9, "alphaq": 1.8, "hub": 0.7, "omegamh2": 0.143,
          "hireionz": 7.0, "bhfeedback": 0.05}
SIMS = ("synthsim0", "synthsim1", "synthsim2", "synthsim3")


def _tables(sims=SIMS, exclude=(), holdout=(), k=K):
    t = synthetic_targets(sims=sims)
    t = dict(t, g=t["g"][..., :k])
    rows, train = MR.select_train_rows(t, exclude_sims=exclude, holdout_sims=holdout)
    return MM.fit_mode_mf(t, train_rows=rows), train, len(rows)


def write_mf(path, *, manifest, cache, kind="release", backbone_kind="release", held_out=(), fold=None,
             exclude=("synthsim3",), k=K):
    tables, train, n = _tables(exclude=exclude, holdout=(fold or {}).get("holdout_sims", ()), k=k)
    kcom = 2 * np.pi * np.arange(1, k + 1) / L_BOX_HMPC
    return MR.write_mf_record(
        str(path), tables, k_com_hmpc=kcom, kind=kind, mf_version="synthetic-mf-1",
        backbone={"manifest_sha256": manifest, "kind": backbone_kind, "cache_sha256": cache,
                  "held_out_sims": list(held_out)},
        caches={"lf": {"file": RT.SYNTHETIC_CACHE_NAME, "sha256": cache},
                "hr": {"file": "synthetic_hr.h5", "sha256": H64("synthetic hr")}},
        composition={"id": "synthetic-composition", "excluded_sims": list(exclude),
                     "ruling_ids": ["synthetic ruling"]},
        fold=fold, fit_code={"commit": COMMIT, "dirty": False}, train_sims=train, n_pairs_train=n)[1]


@pytest.fixture(scope="module")
def info(tmp_path_factory):
    return RT.write_synthetic_bundle(str(tmp_path_factory.mktemp("b") / "bundle"), n_members=3)


@pytest.fixture(scope="module")
def mf(info, tmp_path_factory):
    d = tmp_path_factory.mktemp("m") / "mf"
    return str(d), write_mf(d, manifest=info.manifest_sha256, cache=info.lineage["cache"]["sha256"])


@pytest.fixture(scope="module")
def emu_lf(info):
    return hp.load_bundle(info.bundle_dir, info.manifest_sha256)


@pytest.fixture(scope="module")
def emu_mf(info, mf):
    return hp.load_bundle(info.bundle_dir, info.manifest_sha256, mf_dir=mf[0], mf_record_sha256=mf[1])


def _x(z):
    th = np.asarray(normalize_params(np.array([PARAMS[n] for n in hp.PARAM_NAMES])), dtype=np.float64)
    return np.concatenate([th, [(z - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])]])


# ------------------------------------------------------------------------------------------------ the correction
def test_mf_is_applied_to_the_lf_prediction(emu_lf, emu_mf, mf):
    rec = MR.load_mf_record(mf[0], expected_record_sha256=mf[1], backbone_manifest_sha256=emu_lf.manifest_sha256)
    for z, tau in ((3.0, 0.4), (2.6, 0.25)):
        a, b = emu_lf.predict(PARAMS, z, tau_eff=tau), emu_mf.predict(PARAMS, z, tau_eff=tau)
        g = rec.g(_x(z), tau)
        assert np.max(np.abs(g)) > 1e-3                              # a non-trivial correction
        np.testing.assert_allclose(b.P_class, a.P_class * np.exp(g), rtol=1e-13, atol=0)
        np.testing.assert_allclose(b.template, b.P_class[1:] / b.P_class[:1] - 1.0, rtol=1e-13, atol=0)
        np.testing.assert_array_equal(b.w_c, a.w_c)                  # the class fractions stay the LF backbone's
        np.testing.assert_array_equal(b.k, a.k)
        assert b.flags["mf_applied"] is True and b.labels["product"].endswith("+MF")


def test_mf_is_applied_per_member_before_the_k_placement(emu_lf, emu_mf, mf):
    rec = MR.load_mf_record(mf[0], expected_record_sha256=mf[1], backbone_manifest_sha256=emu_lf.manifest_sha256)
    modes = emu_mf.predict(PARAMS, 3.0, tau_eff=0.4)
    a = emu_lf.predict(PARAMS, 3.0, tau_eff=0.4)
    np.testing.assert_allclose(modes.member_std, a.member_std * np.exp(rec.g(_x(3.0), 0.4)), rtol=1e-12, atol=0)
    kq = 0.5 * (modes.k[1:4] + modes.k[2:5])
    at_k = emu_mf.predict(PARAMS, 3.0, tau_eff=0.4, k=kq)
    want = np.stack([np.interp(kq, modes.k, modes.P_class[c]) for c in range(4)])
    np.testing.assert_allclose(at_k.P_class, want, rtol=1e-12, atol=0)


def test_stacked_mf_prediction_equals_the_single_ones(emu_mf):
    st = emu_mf.predict_many(PARAMS, [2.6, 3.0], tau_eff=[0.25, 0.4])
    for i, (z, t) in enumerate(((2.6, 0.25), (3.0, 0.4))):
        np.testing.assert_allclose(st.P_class[i], emu_mf.predict(PARAMS, z, tau_eff=t).P_class, rtol=1e-13, atol=0)
    assert st.flags["mf_applied"] is True


# ------------------------------------------------------------------------------------------------ LF unchanged
def test_lf_prediction_is_bit_for_bit_unchanged(emu_lf, emu_mf, info):
    fresh = hp.load_bundle(info.bundle_dir, info.manifest_sha256)
    for e in (emu_lf, fresh):
        a = e.predict(PARAMS, 3.0, tau_eff=0.4)
        assert "mf_applied" not in a.flags and "mf" not in a.provenance and "mf" not in a.labels
    a = emu_lf.predict(PARAMS, 3.0, tau_eff=0.4)
    for f in ("LF", None):
        b = (emu_mf if f else emu_lf).predict(PARAMS, 3.0, tau_eff=0.4, fidelity=f)
        for n in ("k", "P_class", "template", "w_c", "dndx", "member_std"):
            assert np.asarray(getattr(a, n)).tobytes() == np.asarray(getattr(b, n)).tobytes(), n
    assert emu_mf.predict(PARAMS, 3.0, tau_eff=0.4, fidelity="LF").flags["mf_applied"] is False


# ------------------------------------------------------------------------------------------------ provenance
def test_mf_provenance_is_reported(emu_mf, mf, info):
    p = emu_mf.predict(PARAMS, 3.0, tau_eff=0.4)
    m = p.provenance["mf"]
    assert m["record_sha256"] == mf[1] and m["backbone_manifest_sha256"] == info.manifest_sha256
    assert m["kind"] == "release" and m["mf_version"] == "synthetic-mf-1"
    assert m["composition"]["ruling_ids"] == ["synthetic ruling"]
    assert m["mean_flux_convention"]["status"] == "PROPOSED"
    assert m["hr_inputs"] == MR.HR_INPUTS
    assert m["caches"] == {"lf": {"file": RT.SYNTHETIC_CACHE_NAME, "sha256": info.lineage["cache"]["sha256"],
                                  "provenance": MR.CACHE_PROVENANCE["lf"]},
                           "hr": {"file": "synthetic_hr.h5", "sha256": H64("synthetic hr"),
                                  "provenance": MR.CACHE_PROVENANCE["hr"]}}
    assert m["caches"]["hr"]["provenance"].startswith("digest-anchored only, never lineage-verified")
    assert "mf" in emu_mf.provenance and emu_mf.provenance["mf"] == m
    assert "mf" in p.labels


def test_mf_domain_flags(emu_mf):
    inside = emu_mf.predict(PARAMS, 3.0, tau_eff=0.4)
    assert inside.flags["mf_z_clamped"] is False and inside.flags["mf_tau0_clamped"] is False
    assert emu_mf.predict(PARAMS, 4.2, tau_eff=0.4).flags["mf_z_clamped"] is True
    assert emu_mf.predict(PARAMS, 3.0, tau_eff=3.0).flags["mf_tau0_clamped"] is True


# ------------------------------------------------------------------------------------------------ refusals
def test_absent_mf_is_refused_when_asked_for(emu_lf, info, mf):
    with pytest.raises(hp.MFUnavailableError, match="no MF record"):
        emu_lf.predict(PARAMS, 3.0, tau_eff=0.4, fidelity="LF+MF")
    with pytest.raises(hp.PredictorInputError, match="fidelity"):
        emu_lf.predict(PARAMS, 3.0, tau_eff=0.4, fidelity="HR")
    with pytest.raises(MR.MFRecordError, match="both"):
        hp.load_bundle(info.bundle_dir, info.manifest_sha256, mf_dir=mf[0])
    with pytest.raises(MR.MFRecordError, match="both"):
        hp.load_bundle(info.bundle_dir, info.manifest_sha256, mf_record_sha256=mf[1])


def test_mismatched_mf_is_refused(info, tmp_path):
    other = RT.write_synthetic_bundle(str(tmp_path / "other"), n_members=2, seeds=[7, 8])
    d = tmp_path / "mf_other"
    s = write_mf(d, manifest=other.manifest_sha256, cache=other.lineage["cache"]["sha256"])
    with pytest.raises(MR.MFRecordError, match="backbone"):
        hp.load_bundle(info.bundle_dir, info.manifest_sha256, mf_dir=str(d), mf_record_sha256=s)
    with pytest.raises(MR.MFRecordError, match="sha256"):
        hp.load_bundle(other.bundle_dir, other.manifest_sha256, mf_dir=str(d), mf_record_sha256=H64("x"))
    d2 = tmp_path / "mf_cache"
    s2 = write_mf(d2, manifest=info.manifest_sha256, cache=H64("another training cache"))
    with pytest.raises(MR.MFRecordError, match="cache"):
        hp.load_bundle(info.bundle_dir, info.manifest_sha256, mf_dir=str(d2), mf_record_sha256=s2)
    d3 = tmp_path / "mf_modes"
    s3 = write_mf(d3, manifest=info.manifest_sha256, cache=info.lineage["cache"]["sha256"], k=K - 2)
    with pytest.raises(MR.MFRecordError, match="modes"):
        hp.load_bundle(info.bundle_dir, info.manifest_sha256, mf_dir=str(d3), mf_record_sha256=s3)


def test_historical_and_fold_mf_are_refused_on_a_production_bundle(info, tmp_path):
    c = info.lineage["cache"]["sha256"]
    d = tmp_path / "hist"
    s = write_mf(d, manifest=info.manifest_sha256, cache=c, kind="historical", exclude=())
    with pytest.raises(MR.MFRecordError, match="historical"):
        hp.load_bundle(info.bundle_dir, info.manifest_sha256, mf_dir=str(d), mf_record_sha256=s)
    d2 = tmp_path / "fold"
    s2 = write_mf(d2, manifest=info.manifest_sha256, cache=c, kind="fold",
                  fold={"fold_id": "f0", "holdout_sims": ["synthsim2"]})
    with pytest.raises(MR.MFRecordError, match="fold"):
        hp.load_bundle(info.bundle_dir, info.manifest_sha256, mf_dir=str(d2), mf_record_sha256=s2)


def test_a_hand_built_or_unverified_mf_is_refused(info, mf):
    from hcd_analysis.emulator import safe_h5 as H
    rb = H.load_release(info.bundle_dir, expected_manifest_sha256=info.manifest_sha256)
    rec = MR.load_mf_record(mf[0], expected_record_sha256=mf[1], backbone_manifest_sha256=info.manifest_sha256)
    with pytest.raises(MR.MFRecordError, match="load_mf_record"):
        hp.Bundle(rb, mf=rec._replace(anchored=False))
    with pytest.raises(MR.MFRecordError, match="load_mf_record"):
        hp.Bundle(rb, mf={"tables": rec.tables})
    assert "mf" in hp.Bundle(rb, mf=rec).provenance


def test_bundle_rechecks_the_binding_of_a_loaded_record(info, tmp_path):
    """Bundle itself (not only load_mf_record) refuses a record of another backbone or another kind."""
    from hcd_analysis.emulator import safe_h5 as H
    rb = H.load_release(info.bundle_dir, expected_manifest_sha256=info.manifest_sha256)
    other = RT.write_synthetic_bundle(str(tmp_path / "other"), n_members=2, seeds=[7, 8])
    d = tmp_path / "mf_other"
    s = write_mf(d, manifest=other.manifest_sha256, cache=info.lineage["cache"]["sha256"])
    rec = MR.load_mf_record(str(d), expected_record_sha256=s, backbone_manifest_sha256=other.manifest_sha256)
    with pytest.raises(MR.MFRecordError, match="backbone"):
        hp.Bundle(rb, mf=rec)
    d2 = tmp_path / "hist"
    s2 = write_mf(d2, manifest=info.manifest_sha256, cache=info.lineage["cache"]["sha256"], kind="historical",
                  exclude=())
    hist = MR.load_mf_record(str(d2), expected_record_sha256=s2, backbone_manifest_sha256=info.manifest_sha256,
                             kind="historical")
    with pytest.raises(MR.MFRecordError, match="historical"):
        hp.Bundle(rb, mf=hist)


# ------------------------------------------------------------------------------------------------ held-out folds
def test_fold_mf_on_its_research_backbone_gives_held_out_predictions(tmp_path):
    res = RT.write_synthetic_research_bundle(str(tmp_path / "res"), held_out_sims=("synthsim2",), n_groups=3)
    c = res.lineage["cache"]["sha256"]
    d = tmp_path / "fold"
    s = write_mf(d, manifest=res.manifest_sha256, cache=c, kind="fold", backbone_kind="research",
                 held_out=["synthsim2"], fold={"fold_id": "f0", "holdout_sims": ["synthsim2"]})
    lf = hp.load_research_bundle(res.bundle_dir, res.manifest_sha256, purpose="a test")
    rp = hp.load_research_bundle(res.bundle_dir, res.manifest_sha256, purpose="a test", mf_dir=str(d),
                                 mf_record_sha256=s)
    rec = MR.load_mf_record(str(d), expected_record_sha256=s, backbone_manifest_sha256=res.manifest_sha256,
                            kind="fold")
    p, a = rp.predict(PARAMS, 3.0, tau_eff=0.4), lf.predict(PARAMS, 3.0, tau_eff=0.4)
    np.testing.assert_allclose(p.P_class, a.P_class * np.exp(rec.g(_x(3.0), 0.4)), rtol=1e-13, atol=0)
    assert p.flags["research_model"] is True and p.flags["mf_applied"] is True
    assert p.provenance["mf"]["fold"] == {"fold_id": "f0", "holdout_sims": ["synthsim2"],
                                          "backbone_excludes_holdout": True}
    rel = tmp_path / "rel"
    s2 = write_mf(rel, manifest=res.manifest_sha256, cache=c)
    with pytest.raises(MR.MFRecordError):                            # a release record on a research backbone
        hp.load_research_bundle(res.bundle_dir, res.manifest_sha256, purpose="a test", mf_dir=str(rel),
                                mf_record_sha256=s2)


# ------------------------------------------------------------------------------- fold backbones (D1 panel M1, M2)
@pytest.fixture(scope="module")
def res(tmp_path_factory):
    return RT.write_synthetic_research_bundle(str(tmp_path_factory.mktemp("r") / "res"), held_out_sims=("synthsim2",),
                                              n_groups=3)


def _rb(res):
    from hcd_analysis.emulator import safe_h5 as H
    return H.load_research(res.bundle_dir, expected_manifest_sha256=res.manifest_sha256, purpose="a test")


def _metas(rb, fn):
    from hcd_analysis.emulator import safe_h5 as H
    m = rb.metas
    return rb._replace(metas=H.MemberMetas([fn(i, dict(x)) for i, x in enumerate(m)], anchored=m.anchored,
                                           lineage_verified=m.lineage_verified, manifest_sha256=m.manifest_sha256,
                                           fmt=m.format, kind=m.kind, lineage=m.lineage))


def test_backbone_identity_is_derived_from_the_member_metas_not_the_declared_split(res):
    rb = _rb(res)
    assert hp.backbone_identity(hp.ResearchPredictor(rb)) == {
        "manifest_sha256": res.manifest_sha256, "kind": "research", "cache_sha256": res.lineage["cache"]["sha256"],
        "held_out_sims": ["synthsim2"]}
    lie = dict(rb.split, held_out_sims=["synthsim1"])
    rb2 = rb._replace(split=lie, research=dict(rb.research, split=dict(rb.research["split"],
                                                                         held_out_sims=["synthsim1"])))
    assert hp.backbone_identity(hp.ResearchPredictor(rb2))["held_out_sims"] == ["synthsim2"]


def test_backbone_identity_refuses_disagreeing_members_or_another_cache(res):
    rb = _rb(res)
    other = _metas(rb, lambda i, m: dict(m, recipe=dict(m["recipe"], holdout_sims=["synthsim1"])) if i else m)
    with pytest.raises(MR.MFRecordError, match="different simulations"):
        hp.backbone_identity(hp.ResearchPredictor(other))
    cache = _metas(rb, lambda i, m: dict(m, cache_sha256=H64("another cache")) if i else m)
    with pytest.raises(MR.MFRecordError, match="training cache"):
        hp.backbone_identity(hp.ResearchPredictor(cache))


def test_backbone_identity_refuses_metas_whose_excluded_set_cannot_be_derived(res):
    """R14a (panel M1): a member naming simulations under a key that says neither held out nor trained on has no
    derivable excluded set; the identity is refused, not guessed from the other keys."""
    rb = _rb(res)
    vague = _metas(rb, lambda i, m: dict(m, recipe=dict(m["recipe"], compared_with=["synthsim2"])) if i else m)
    with pytest.raises(MR.MFRecordError, match="cannot be derived from their metas"):
        hp.backbone_identity(hp.ResearchPredictor(vague))


def test_fold_whose_backbone_saw_the_held_out_group_is_refused(res, tmp_path):
    c = res.lineage["cache"]["sha256"]
    d = tmp_path / "saw"                                    # the backbone holds out synthsim2, the fold synthsim1
    s = write_mf(d, manifest=res.manifest_sha256, cache=c, kind="fold", backbone_kind="research",
                 held_out=["synthsim2"], fold={"fold_id": "f1", "holdout_sims": ["synthsim1"]})
    with pytest.raises(MR.MFRecordError, match="trained on held-out"):
        hp.load_research_bundle(res.bundle_dir, res.manifest_sha256, purpose="a test", mf_dir=str(d),
                                mf_record_sha256=s)
    d2 = tmp_path / "claims"                                # the record claims a held-out set the metas do not have
    s2 = write_mf(d2, manifest=res.manifest_sha256, cache=c, kind="fold", backbone_kind="research",
                  held_out=["synthsim1"], fold={"fold_id": "f1", "holdout_sims": ["synthsim1"]})
    with pytest.raises(MR.MFRecordError, match="member metas say"):
        hp.load_research_bundle(res.bundle_dir, res.manifest_sha256, purpose="a test", mf_dir=str(d2),
                                mf_record_sha256=s2)


def test_one_fold_through_the_predictor_reproduces_the_cache_input_path(res, tmp_path):
    c = res.lineage["cache"]["sha256"]
    d = tmp_path / "fold"
    s = write_mf(d, manifest=res.manifest_sha256, cache=c, kind="fold", backbone_kind="research",
                 held_out=["synthsim2"], fold={"fold_id": "f2", "holdout_sims": ["synthsim2"]})
    rp = hp.load_research_bundle(res.bundle_dir, res.manifest_sha256, purpose="a test", mf_dir=str(d),
                                 mf_record_sha256=s)
    zs, taus = [2.6, 3.0, 3.4], [0.25, 0.4, 0.33]
    x = np.stack([_x(z) for z in zs])
    cache_path = hp.heldout_P_filt_at(rp, x, np.array(taus))
    for i, (z, t) in enumerate(zip(zs, taus)):
        np.testing.assert_allclose(rp.predict(PARAMS, z, tau_eff=t).P_class, cache_path["mf"][i], rtol=1e-12, atol=0)
        np.testing.assert_allclose(rp.predict(PARAMS, z, tau_eff=t, fidelity="LF").P_class, cache_path["lf"][i],
                                   rtol=1e-12, atol=0)
    np.testing.assert_allclose(cache_path["lf"], hp.lf_P_filt_at(rp, x, np.array(taus)), rtol=1e-12, atol=0)


def test_held_out_path_needs_a_fold_record(emu_lf, emu_mf):
    for e in (emu_lf, emu_mf):
        with pytest.raises(MR.MFRecordError, match="fold MF record"):
            hp.heldout_P_filt_at(e, _x(3.0)[None], np.array([0.4]))
