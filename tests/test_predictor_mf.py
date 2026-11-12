"""The public predictor's MF path (hcd_analysis/predictor.py with an mf_release record) on synthetic h5/2 bundles and
synthetic MF tables: untrained networks, made-up corrections, made-up digests; the values mean nothing. Checked: the
correction is applied (P_MF = P_LF exp(g), per member, before any k placement), LF-only prediction is untouched,
the MF provenance is reported, and an absent, mismatched, historical or fold MF is refused clearly on a production
bundle."""

import os

import numpy as np
import pytest

import hcd_analysis.predictor as hp
from hcd_analysis.emulator import mf_modes as MM
from hcd_analysis.emulator import mf_release as MR
from hcd_analysis.emulator import release_testing as RT
from hcd_analysis.emulator.data import Z_LIMITS, normalize_params
from hcd_analysis.emulator.schema import L_BOX_HMPC
from tests.test_mf_release import (
    COMMIT,
    H64,
    HR_STATEMENT,
    LF_STATEMENT,
    LyingStr,
    PlainStr,
    synthetic_targets,
)

K = RT.SYNTHETIC_ARCH["n_k"]
KCOM = 2 * np.pi * np.arange(1, K + 1) / L_BOX_HMPC
PARAMS = {
    "ns": 0.95,
    "Ap": 2.0e-9,
    "herei": 3.8,
    "heref": 2.9,
    "alphaq": 1.8,
    "hub": 0.7,
    "omegamh2": 0.143,
    "hireionz": 7.0,
    "bhfeedback": 0.05,
}
SIMS = ("synthsim0", "synthsim1", "synthsim2", "synthsim3")


def _tables(sims=SIMS, exclude=(), holdout=(), k=K):
    t = synthetic_targets(sims=sims)
    t = dict(t, g=t["g"][..., :k])
    rows, train = MR.select_train_rows(t, exclude_sims=exclude, holdout_sims=holdout)
    return MM.fit_mode_mf(t, train_rows=rows), train, len(rows)


def write_mf(
    path,
    *,
    manifest,
    cache,
    kind="release",
    backbone_kind="release",
    held_out=(),
    fold=None,
    exclude=("synthsim3",),
    k=K,
    z_shift=0.0,
    validated_z_min=MR.VALIDATED_Z_MIN,
):
    tables, train, n = _tables(exclude=exclude, holdout=(fold or {}).get("holdout_sims", ()), k=k)
    if z_shift:  # synthetic tables labelled at lower redshifts
        tables = dict(tables, z_tab=np.asarray(tables["z_tab"], float) - z_shift)
    kcom = 2 * np.pi * np.arange(1, k + 1) / L_BOX_HMPC
    return MR.write_mf_record(
        str(path),
        tables,
        k_com_hmpc=kcom,
        kind=kind,
        mf_version="synthetic-mf-1",
        backbone={
            "manifest_sha256": manifest,
            "kind": backbone_kind,
            "cache_sha256": cache,
            "held_out_sims": list(held_out),
        },
        caches={
            "lf": {"file": RT.SYNTHETIC_CACHE_NAME, "sha256": cache},
            "hr": {"file": "synthetic_hr.h5", "sha256": H64("synthetic hr")},
        },
        composition={
            "id": "synthetic-composition",
            "excluded_sims": list(exclude),
            "ruling_ids": ["synthetic ruling"],
        },
        fold=fold,
        fit_code={"commit": COMMIT, "dirty": False},
        train_sims=train,
        n_pairs_train=n,
        mode_mapping={"tol": MR.D1_TOL, "canonical_max": 0.0, "stored_max": 0.0, "exception": None},
        validated_z_min=validated_z_min,
    )[1]


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
    return hp.load_bundle(
        info.bundle_dir, info.manifest_sha256, mf_dir=mf[0], mf_record_sha256=mf[1]
    )


def _x(z):
    th = np.asarray(
        normalize_params(np.array([PARAMS[n] for n in hp.PARAM_NAMES])), dtype=np.float64
    )
    return np.concatenate([th, [(z - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])]])


# ------------------------------------------------------------------------------------------------ the correction
def test_mf_is_applied_to_the_lf_prediction(emu_lf, emu_mf, mf):
    rec = MR.load_mf_record(
        mf[0], expected_record_sha256=mf[1], backbone_manifest_sha256=emu_lf.manifest_sha256
    )
    for z, tau in ((3.0, 0.4), (2.6, 0.25)):
        a, b = emu_lf.predict(PARAMS, z, tau_eff=tau), emu_mf.predict(PARAMS, z, tau_eff=tau)
        g = rec.g(_x(z), tau)
        assert np.max(np.abs(g)) > 1e-3  # a non-trivial correction
        np.testing.assert_allclose(b.P_class, a.P_class * np.exp(g), rtol=1e-13, atol=0)
        np.testing.assert_allclose(
            b.template, b.P_class[1:] / b.P_class[:1] - 1.0, rtol=1e-13, atol=0
        )
        np.testing.assert_array_equal(b.w_c, a.w_c)  # the class fractions stay the LF backbone's
        np.testing.assert_array_equal(b.k, a.k)
        assert b.flags["mf_applied"] is True and b.labels["product"].endswith("+MF")


def test_mf_is_applied_per_member_before_the_k_placement(emu_lf, emu_mf, mf):
    rec = MR.load_mf_record(
        mf[0], expected_record_sha256=mf[1], backbone_manifest_sha256=emu_lf.manifest_sha256
    )
    modes = emu_mf.predict(PARAMS, 3.0, tau_eff=0.4)
    a = emu_lf.predict(PARAMS, 3.0, tau_eff=0.4)
    np.testing.assert_allclose(
        modes.member_std, a.member_std * np.exp(rec.g(_x(3.0), 0.4)), rtol=1e-12, atol=0
    )
    kq = 0.5 * (modes.k[1:4] + modes.k[2:5])
    at_k = emu_mf.predict(PARAMS, 3.0, tau_eff=0.4, k=kq)
    want = np.stack([np.interp(kq, modes.k, modes.P_class[c]) for c in range(4)])
    np.testing.assert_allclose(at_k.P_class, want, rtol=1e-12, atol=0)


def test_stacked_mf_prediction_equals_the_single_ones(emu_mf):
    st = emu_mf.predict_many(PARAMS, [2.6, 3.0], tau_eff=[0.25, 0.4])
    for i, (z, t) in enumerate(((2.6, 0.25), (3.0, 0.4))):
        np.testing.assert_allclose(
            st.P_class[i], emu_mf.predict(PARAMS, z, tau_eff=t).P_class, rtol=1e-13, atol=0
        )
    assert st.flags["mf_applied"] is True


def test_the_mf_correction_label_says_whether_the_mf_is_applied(emu_lf, emu_mf):
    """PR #19's single-fidelity label (``mf_correction`` "none ...") stays true of every LF output; an LF+MF output
    says the record's correction is applied (merge of pr19/round1-fixes)."""
    assert emu_mf.predict(PARAMS, 3.0, tau_eff=0.4).labels["mf_correction"].startswith("applied")
    assert (
        emu_mf.predict_many(PARAMS, [2.6, 3.0], tau_eff=0.4)
        .labels["mf_correction"]
        .startswith("applied")
    )
    for e, f in ((emu_mf, "LF"), (emu_lf, None), (emu_lf, "LF")):
        lab = e.predict(PARAMS, 3.0, tau_eff=0.4, fidelity=f).labels["mf_correction"]
        assert lab.startswith("none") and "single-fidelity" in lab


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
    assert (
        m["mean_flux_convention"]["status"] == "SETTLED"
        and m["mean_flux_convention"]["name"] == "global"
    )
    assert m["hr_inputs"] == MR.HR_INPUTS
    assert m["caches"] == {
        "lf": {
            "file": RT.SYNTHETIC_CACHE_NAME,
            "sha256": info.lineage["cache"]["sha256"],
            "provenance": MR.CACHE_PROVENANCE["lf"],
        },
        "hr": {
            "file": "synthetic_hr.h5",
            "sha256": H64("synthetic hr"),
            "provenance": MR.CACHE_PROVENANCE["hr"],
        },
    }
    assert m["caches"]["hr"]["provenance"] == HR_STATEMENT  # the literals (delta review R1-1)
    assert m["caches"]["lf"]["provenance"] == LF_STATEMENT
    assert "mf" in emu_mf.provenance and emu_mf.provenance["mf"] == m
    assert "mf" in p.labels


def test_mf_domain_flags(emu_mf):
    inside = emu_mf.predict(PARAMS, 3.0, tau_eff=0.4)
    assert inside.flags["mf_z_clamped"] is False and inside.flags["mf_tau0_clamped"] is False
    assert inside.flags["mf_z_above_validated"] is False
    assert emu_mf.predict(PARAMS, 4.2, tau_eff=0.4).flags["mf_z_clamped"] is True
    assert emu_mf.predict(PARAMS, 3.0, tau_eff=3.0).flags["mf_tau0_clamped"] is True


def test_mf_above_the_validated_range_is_served_unchanged_but_flagged(info, emu_lf, tmp_path):
    """The frozen grouped-validation domain of the MF ends at mf_release.VALIDATED_Z_MAX (4.6). An LF+MF prediction
    above it, even inside the record's tables (where nothing else flags it), is served with the same values but
    flagged mf_z_above_validated; the edge itself (to the redshift tolerance) is inside; an LF prediction from the same
    object is not flagged. The synthetic tables (z 2.6, 3.0, 3.4) are relabelled 1.6 up (4.2, 4.6, 5.0), so that 4.8 and
    5.0 are inside them."""
    assert MR.VALIDATED_Z_MAX == 4.6
    d = tmp_path / "hi"
    s = write_mf(
        d, manifest=info.manifest_sha256, cache=info.lineage["cache"]["sha256"], z_shift=-1.6
    )
    e = hp.load_bundle(info.bundle_dir, info.manifest_sha256, mf_dir=str(d), mf_record_sha256=s)
    for z, above in ((4.4, False), (4.6, False), (4.6 + 5e-10, False), (4.8, True), (5.0, True)):
        p = e.predict(PARAMS, z, tau_eff=0.4)
        assert p.flags["mf_applied"] is True and p.flags["mf_z_clamped"] is False
        assert p.flags["mf_z_above_validated"] is above, z
    hi = e.predict(PARAMS, 4.8, tau_eff=0.4)
    again = e.predict(PARAMS, 4.8, tau_eff=0.4, fidelity="LF+MF")
    assert hi.P_class.tobytes() == again.P_class.tobytes()  # flagged, not changed or refused
    assert "validated" in hi.labels["mf"] and "4.6" in hi.labels["mf"]
    lf = e.predict(PARAMS, 4.8, tau_eff=0.4, fidelity="LF")
    assert lf.flags["mf_applied"] is False and lf.flags["mf_z_above_validated"] is False
    assert lf.P_class.tobytes() == emu_lf.predict(PARAMS, 4.8, tau_eff=0.4).P_class.tobytes()
    assert e.predict_many(PARAMS, [4.4, 4.8], tau_eff=0.4).flags["mf_z_above_validated"] is True
    assert e.predict_many(PARAMS, [4.4, 4.6], tau_eff=0.4).flags["mf_z_above_validated"] is False
    assert (
        emu_lf.predict(PARAMS, 4.8, tau_eff=0.4).flags.get("mf_z_above_validated") is None
    )  # no MF record


def test_mf_below_the_tables_lowest_redshift_is_refused_not_clamped(emu_mf, emu_lf, res, tmp_path):
    """C1 review m4 (D1 panel, physics lens): the synthetic tables start at z 2.6. Below that the MF path refuses,
    single, stacked and on the held-out path; the LF prediction there is still available, and equals the LF-only
    bundle's. Above the tables (z 4.2) the MF is still held constant and flagged."""
    for f in (None, "LF+MF"):
        with pytest.raises(
            hp.MFRangeError, match=r"refused below z = 2.6 \(its tables' lowest redshift 2.6"
        ):
            emu_mf.predict(PARAMS, 2.4, tau_eff=0.3, fidelity=f)
        with pytest.raises(hp.MFRangeError, match=r"z = \[2.2, 2.4\]"):
            emu_mf.predict_many(PARAMS, [2.2, 2.4, 3.0], tau_eff=0.3, fidelity=f)
    assert issubclass(hp.MFRangeError, hp.PredictorInputError)
    lf = emu_mf.predict(PARAMS, 2.4, tau_eff=0.3, fidelity="LF")
    assert lf.flags["mf_applied"] is False
    assert lf.P_class.tobytes() == emu_lf.predict(PARAMS, 2.4, tau_eff=0.3).P_class.tobytes()
    assert (
        emu_mf.predict(PARAMS, 2.6, tau_eff=0.3).flags["mf_z_clamped"] is False
    )  # the lowest table z itself
    hi = emu_mf.predict(PARAMS, 4.2, tau_eff=0.4)
    assert hi.flags["mf_applied"] is True and hi.flags["mf_z_clamped"] is True
    c = res.lineage["cache"]["sha256"]
    d = tmp_path / "fold"
    s = write_mf(
        d,
        manifest=res.manifest_sha256,
        cache=c,
        kind="fold",
        backbone_kind="research",
        held_out=["synthsim2"],
        fold={"fold_id": "f3", "holdout_sims": ["synthsim2"]},
    )
    rp = hp.load_research_bundle(
        res.bundle_dir, res.manifest_sha256, purpose="a test", mf_dir=str(d), mf_record_sha256=s
    )
    with pytest.raises(hp.MFRangeError, match="refused below"):
        hp.heldout_P_filt_at(rp, np.stack([_x(3.0), _x(2.4)]), np.array([0.4, 0.3]))
    assert hp.heldout_P_filt_at(rp, _x(2.6)[None], np.array([0.3]))["mf"].shape[0] == 1


def test_mf_below_validated_z_min_is_refused_even_inside_the_tables(info, emu_lf, res, tmp_path):
    """The record's validated_z_min (2.2 by default) bounds the MF from below together with the tables' lowest
    redshift: tables starting at z 2.0 still refuse an LF+MF prediction at z 2.0 (single, stacked, held-out); the LF
    prediction there is unchanged; a record that raises validated_z_min to 2.6 refuses below 2.6."""
    c = info.lineage["cache"]["sha256"]
    s = write_mf(
        tmp_path / "lo", manifest=info.manifest_sha256, cache=c, z_shift=0.6
    )  # z_tab 2.0, 2.4, 2.8
    e = hp.load_bundle(
        info.bundle_dir, info.manifest_sha256, mf_dir=str(tmp_path / "lo"), mf_record_sha256=s
    )
    assert float(e._mf_dom[0][0]) == pytest.approx(2.0)
    assert e.provenance["mf"]["validated_z_min"] == 2.2
    with pytest.raises(hp.MFRangeError, match=r"refused below z = 2.2 .*validated_z_min 2.2"):
        e.predict(PARAMS, 2.0, tau_eff=0.3)
    with pytest.raises(hp.MFRangeError, match=r"z = \[2.0\]"):
        e.predict_many(PARAMS, [2.0, 2.2, 3.0], tau_eff=0.3, fidelity="LF+MF")
    assert e.predict(PARAMS, 2.2, tau_eff=0.3).flags["mf_applied"] is True
    lf = e.predict(PARAMS, 2.0, tau_eff=0.3, fidelity="LF")
    assert lf.P_class.tobytes() == emu_lf.predict(PARAMS, 2.0, tau_eff=0.3).P_class.tobytes()
    s2 = write_mf(
        tmp_path / "hi", manifest=info.manifest_sha256, cache=c, z_shift=0.6, validated_z_min=2.6
    )
    e2 = hp.load_bundle(
        info.bundle_dir, info.manifest_sha256, mf_dir=str(tmp_path / "hi"), mf_record_sha256=s2
    )
    with pytest.raises(hp.MFRangeError, match=r"refused below z = 2.6"):
        e2.predict(PARAMS, 2.4, tau_eff=0.3)
    assert e2.predict(PARAMS, 2.6, tau_eff=0.3).flags["mf_applied"] is True
    rc = res.lineage["cache"]["sha256"]
    s3 = write_mf(
        tmp_path / "fold",
        manifest=res.manifest_sha256,
        cache=rc,
        kind="fold",
        backbone_kind="research",
        held_out=["synthsim2"],
        fold={"fold_id": "f3", "holdout_sims": ["synthsim2"]},
        z_shift=0.6,
    )
    rp = hp.load_research_bundle(
        res.bundle_dir,
        res.manifest_sha256,
        purpose="a test",
        mf_dir=str(tmp_path / "fold"),
        mf_record_sha256=s3,
    )
    with pytest.raises(hp.MFRangeError, match="validated_z_min"):
        hp.heldout_P_filt_at(rp, _x(2.0)[None], np.array([0.3]))
    assert hp.heldout_P_filt_at(rp, _x(2.2)[None], np.array([0.3]))["mf"].shape[0] == 1


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
        hp.load_bundle(
            other.bundle_dir, other.manifest_sha256, mf_dir=str(d), mf_record_sha256=H64("x")
        )
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
    s2 = write_mf(
        d2,
        manifest=info.manifest_sha256,
        cache=c,
        kind="fold",
        fold={"fold_id": "f0", "holdout_sims": ["synthsim2"]},
    )
    with pytest.raises(MR.MFRecordError, match="fold"):
        hp.load_bundle(info.bundle_dir, info.manifest_sha256, mf_dir=str(d2), mf_record_sha256=s2)


def test_a_hand_built_or_unverified_mf_is_refused(info, mf):
    from hcd_analysis.emulator import safe_h5 as H

    rb = H.load_release(info.bundle_dir, expected_manifest_sha256=info.manifest_sha256)
    rec = MR.load_mf_record(
        mf[0], expected_record_sha256=mf[1], backbone_manifest_sha256=info.manifest_sha256
    )
    with pytest.raises(MR.MFRecordError, match="load_mf_record"):
        hp.Bundle(rb, mf=rec._replace(anchored=False))
    with pytest.raises(MR.MFRecordError, match="load_mf_record"):
        hp.Bundle(rb, mf={"tables": rec.tables})
    with pytest.raises(
        MR.MFRecordError, match="load_mf_record"
    ):  # a hand-built tuple has no sealed bytes
        hp.Bundle(
            rb,
            mf=MR.MFRecord(
                rec.record,
                rec.record_sha256,
                rec.tables,
                rec.k_com_hmpc,
                rec.kind,
                rec.mf_dir,
                True,
            ),
        )
    assert "mf" in hp.Bundle(rb, mf=rec).provenance


def test_a_record_changed_in_memory_is_refused_and_provenance_describes_the_tables_applied(
    info, mf, tmp_path
):
    """C1 review m1 (forgeries F11, F10d): the predictor applies only what it re-derives from the record's sealed
    bytes. Another record's tables, a relabelled kind, a changed record dict or bytes that do not match the digest are
    refused; an accepted record's provenance names the sha256 of the tables actually applied."""
    import hashlib
    from hcd_analysis.emulator import safe_h5 as H

    rb = H.load_release(info.bundle_dir, expected_manifest_sha256=info.manifest_sha256)
    c = info.lineage["cache"]["sha256"]
    a = MR.load_mf_record(
        mf[0], expected_record_sha256=mf[1], backbone_manifest_sha256=info.manifest_sha256
    )
    d = tmp_path / "b"
    sb = write_mf(
        d, manifest=info.manifest_sha256, cache=c, exclude=("synthsim2",)
    )  # another fit, same backbone
    b = MR.load_mf_record(
        str(d), expected_record_sha256=sb, backbone_manifest_sha256=info.manifest_sha256
    )
    assert any(not np.array_equal(a.tables[n], b.tables[n]) for n in MM.TABLE_KEYS)
    forged = {
        "F11: A's record with B's tables": a._replace(tables=b.tables),
        "A's digest with B's bytes": a._replace(
            record_bytes=b.record_bytes, tables_bytes=b.tables_bytes
        ),
        "A's bytes with B's tables bytes": a._replace(tables_bytes=b.tables_bytes),
        "a changed record dict": a._replace(record=dict(a.record, mf_version="another")),
        "a relabelled kind": a._replace(kind="historical"),
        "B's modes": a._replace(k_com_hmpc=np.asarray(a.k_com_hmpc) * 1.5),
        "B entirely under A's digest": b._replace(
            record_sha256=mf[1]
        ),  # provenance would name A, apply B
        "A without its record bytes": a._replace(record_bytes=None),
    }
    for what, f in forged.items():
        with pytest.raises(
            MR.MFRecordError, match="changed in memory|do not match|sha256|load_mf_record"
        ):
            hp.Bundle(rb, mf=f)
    hd = tmp_path / "hist"
    sh = write_mf(hd, manifest=info.manifest_sha256, cache=c, kind="historical", exclude=())
    hist = MR.load_mf_record(
        str(hd),
        expected_record_sha256=sh,
        backbone_manifest_sha256=info.manifest_sha256,
        kind="historical",
    )
    relabelled = hist._replace(kind="release", record=dict(hist.record, kind="release"))  # F10d
    with pytest.raises(MR.MFRecordError, match="changed in memory"):
        hp.Bundle(rb, mf=relabelled)
    for rec, sha in ((a, mf[1]), (b, sb)):  # genuine records: provenance = the tables applied
        emu = hp.Bundle(rb, mf=rec)
        p = emu.predict(PARAMS, 3.0, tau_eff=0.4)
        assert p.provenance["mf"]["record_sha256"] == sha
        assert p.provenance["mf"]["tables_sha256"] == hashlib.sha256(rec.tables_bytes).hexdigest()
        lf = emu.predict(PARAMS, 3.0, tau_eff=0.4, fidelity="LF").P_class
        np.testing.assert_allclose(p.P_class, lf * np.exp(rec.g(_x(3.0), 0.4)), rtol=1e-13, atol=0)


def test_the_predictor_keeps_the_fresh_object_reverify_returns(info, mf, tmp_path):
    """Delta review R1-3a (mutants D3b, D3e): the bundle stores the FRESH MFRecord that reverify re-derives from the
    sealed bytes, never the caller's object. Kept the caller's object, a subclass overriding provenance() and g()
    (Q3e), the caller editing its own record after attach (Q3f) and an always-equal record dict claiming B (Q3i)
    would make provenance["mf"] name something other than the correction applied (F11 reopened)."""
    import copy
    import hashlib
    from hcd_analysis.emulator import safe_h5 as H

    rb = H.load_release(info.bundle_dir, expected_manifest_sha256=info.manifest_sha256)
    a = MR.load_mf_record(
        mf[0], expected_record_sha256=mf[1], backbone_manifest_sha256=info.manifest_sha256
    )
    d = tmp_path / "b"
    sb = write_mf(
        d,
        manifest=info.manifest_sha256,
        cache=info.lineage["cache"]["sha256"],
        exclude=("synthsim2",),
    )
    b = MR.load_mf_record(
        str(d), expected_record_sha256=sb, backbone_manifest_sha256=info.manifest_sha256
    )
    g_a, g_b = a.g(_x(3.0), 0.4), b.g(_x(3.0), 0.4)
    assert not np.allclose(g_a, g_b)
    tables_a = hashlib.sha256(a.tables_bytes).hexdigest()

    def assert_applies_and_names_a(emu, caller):
        assert (
            type(emu._mf) is MR.MFRecord and emu._mf is not caller
        )  # the fresh object, not the caller's
        p = emu.predict(PARAMS, 3.0, tau_eff=0.4)
        lf = emu.predict(PARAMS, 3.0, tau_eff=0.4, fidelity="LF").P_class
        np.testing.assert_allclose(
            p.P_class, lf * np.exp(g_a), rtol=1e-13, atol=0
        )  # A is applied ...
        pm = p.provenance["mf"]
        assert (
            pm["record_sha256"] == mf[1] and pm["tables_sha256"] == tables_a
        )  # ... and A is named
        assert pm["train_sims"] == a.record["train_sims"] and pm["mf_version"] == "synthetic-mf-1"

    class SubRec(MR.MFRecord):  # Q3e
        def provenance(self):
            return {"record_sha256": "forged"}

        def g(self, x, tau0):
            return g_b

    sub = SubRec(*a)
    assert_applies_and_names_a(hp.Bundle(rb, mf=sub), sub)

    own = MR.load_mf_record(
        mf[0], expected_record_sha256=mf[1], backbone_manifest_sha256=info.manifest_sha256
    )
    emu = hp.Bundle(rb, mf=own)  # Q3f: the caller edits its own object after attach
    own.record["mf_version"] = "changed after attach"
    own.record["train_sims"] = ["synthsim9"]
    assert_applies_and_names_a(emu, own)

    class EqDict(dict):  # Q3i: equal to anything, claiming B's record
        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

    eq = a._replace(record=EqDict(copy.deepcopy(b.record)))
    assert_applies_and_names_a(hp.Bundle(rb, mf=eq), eq)


class _TwoFacedBytes(bytes):
    """Hashes (its buffer) as one record's bytes while bytes(x) gives another's (delta review Q3h)."""

    other = b""

    def __bytes__(self):
        return self.other


class _PlainBytes(bytes):
    """A bytes subclass with no overrides: genuine content, but not exactly bytes."""


def test_sealed_digest_and_bytes_must_be_exact_str_and_bytes(info, mf, tmp_path):
    """Delta review R1-2 (Q3g, Q3h): reverify takes the digest and the sealed bytes only as exact str and bytes. A str
    subclass that equals anything, or a bytes subclass whose buffer is A's record while bytes() gives B's, would apply
    B while provenance names A's record digest; plain subclasses carrying the genuine values are refused too.
    """
    from hcd_analysis.emulator import safe_h5 as H

    rb = H.load_release(info.bundle_dir, expected_manifest_sha256=info.manifest_sha256)
    a = MR.load_mf_record(
        mf[0], expected_record_sha256=mf[1], backbone_manifest_sha256=info.manifest_sha256
    )
    d = tmp_path / "b"
    sb = write_mf(
        d,
        manifest=info.manifest_sha256,
        cache=info.lineage["cache"]["sha256"],
        exclude=("synthsim2",),
    )
    b = MR.load_mf_record(
        str(d), expected_record_sha256=sb, backbone_manifest_sha256=info.manifest_sha256
    )
    two_faced = type("TF", (_TwoFacedBytes,), {"other": b.record_bytes})(a.record_bytes)
    two_faced_tables = type("TFT", (_TwoFacedBytes,), {"other": b.tables_bytes})(a.tables_bytes)
    forged = {
        "Q3g: B under a str subclass equal to anything, printing A's digest": b._replace(
            record_sha256=LyingStr(mf[1])
        ),
        "Q3h: A's buffer, B's bytes(), B's record and tables, A's digest": b._replace(
            record_sha256=mf[1], record_bytes=two_faced
        ),
        "A's tables buffer whose bytes() is B's tables": a._replace(tables_bytes=two_faced_tables),
        "A's own digest as a plain str subclass": a._replace(record_sha256=PlainStr(mf[1])),
        "A's own record bytes as a plain bytes subclass": a._replace(
            record_bytes=_PlainBytes(a.record_bytes)
        ),
        "A's own tables bytes as a plain bytes subclass": a._replace(
            tables_bytes=_PlainBytes(a.tables_bytes)
        ),
    }
    for what, f in forged.items():
        with pytest.raises(MR.MFRecordError, match="load_mf_record"):
            hp.Bundle(rb, mf=f)
        with pytest.raises(MR.MFRecordError, match="load_mf_record"):
            MR.reverify(f)
    fresh = MR.reverify(a)  # the genuine record: exact types throughout
    assert (type(fresh.record_sha256), type(fresh.record_bytes), type(fresh.tables_bytes)) == (
        str,
        bytes,
        bytes,
    )
    assert hp.Bundle(rb, mf=a).provenance["mf"]["record_sha256"] == mf[1]


def test_bundle_rechecks_the_binding_of_a_loaded_record(info, tmp_path):
    """Bundle itself (not only load_mf_record) refuses a record of another backbone or another kind."""
    from hcd_analysis.emulator import safe_h5 as H

    rb = H.load_release(info.bundle_dir, expected_manifest_sha256=info.manifest_sha256)
    other = RT.write_synthetic_bundle(str(tmp_path / "other"), n_members=2, seeds=[7, 8])
    d = tmp_path / "mf_other"
    s = write_mf(d, manifest=other.manifest_sha256, cache=info.lineage["cache"]["sha256"])
    rec = MR.load_mf_record(
        str(d), expected_record_sha256=s, backbone_manifest_sha256=other.manifest_sha256
    )
    with pytest.raises(MR.MFRecordError, match="backbone"):
        hp.Bundle(rb, mf=rec)
    d2 = tmp_path / "hist"
    s2 = write_mf(
        d2,
        manifest=info.manifest_sha256,
        cache=info.lineage["cache"]["sha256"],
        kind="historical",
        exclude=(),
    )
    hist = MR.load_mf_record(
        str(d2),
        expected_record_sha256=s2,
        backbone_manifest_sha256=info.manifest_sha256,
        kind="historical",
    )
    with pytest.raises(MR.MFRecordError, match="historical"):
        hp.Bundle(rb, mf=hist)


# ------------------------------------------------------------------------------------------------ held-out folds
def test_fold_mf_on_its_research_backbone_gives_held_out_predictions(tmp_path):
    res = RT.write_synthetic_research_bundle(
        str(tmp_path / "res"), held_out_sims=("synthsim2",), n_groups=3
    )
    c = res.lineage["cache"]["sha256"]
    d = tmp_path / "fold"
    s = write_mf(
        d,
        manifest=res.manifest_sha256,
        cache=c,
        kind="fold",
        backbone_kind="research",
        held_out=["synthsim2"],
        fold={"fold_id": "f0", "holdout_sims": ["synthsim2"]},
    )
    lf = hp.load_research_bundle(res.bundle_dir, res.manifest_sha256, purpose="a test")
    rp = hp.load_research_bundle(
        res.bundle_dir, res.manifest_sha256, purpose="a test", mf_dir=str(d), mf_record_sha256=s
    )
    rec = MR.load_mf_record(
        str(d), expected_record_sha256=s, backbone_manifest_sha256=res.manifest_sha256, kind="fold"
    )
    p, a = rp.predict(PARAMS, 3.0, tau_eff=0.4), lf.predict(PARAMS, 3.0, tau_eff=0.4)
    np.testing.assert_allclose(
        p.P_class, a.P_class * np.exp(rec.g(_x(3.0), 0.4)), rtol=1e-13, atol=0
    )
    assert p.flags["research_model"] is True and p.flags["mf_applied"] is True
    assert p.provenance["mf"]["fold"] == {
        "fold_id": "f0",
        "holdout_sims": ["synthsim2"],
        "backbone_excludes_holdout": True,
    }
    rel = tmp_path / "rel"
    s2 = write_mf(rel, manifest=res.manifest_sha256, cache=c)
    with pytest.raises(MR.MFRecordError):  # a release record on a research backbone
        hp.load_research_bundle(
            res.bundle_dir,
            res.manifest_sha256,
            purpose="a test",
            mf_dir=str(rel),
            mf_record_sha256=s2,
        )


# ------------------------------------------------------------------------------- fold backbones (D1 panel M1, M2)
@pytest.fixture(scope="module")
def res(tmp_path_factory):
    return RT.write_synthetic_research_bundle(
        str(tmp_path_factory.mktemp("r") / "res"), held_out_sims=("synthsim2",), n_groups=3
    )


def _rb(res):
    from hcd_analysis.emulator import safe_h5 as H

    return H.load_research(
        res.bundle_dir, expected_manifest_sha256=res.manifest_sha256, purpose="a test"
    )


def _metas(rb, fn):
    from hcd_analysis.emulator import safe_h5 as H

    m = rb.metas
    return rb._replace(
        metas=H.MemberMetas(
            [fn(i, dict(x)) for i, x in enumerate(m)],
            anchored=m.anchored,
            lineage_verified=m.lineage_verified,
            manifest_sha256=m.manifest_sha256,
            fmt=m.format,
            kind=m.kind,
            lineage=m.lineage,
        )
    )


def test_backbone_identity_is_derived_from_the_member_metas_not_the_declared_split(res):
    rb = _rb(res)
    assert hp.backbone_identity(hp.ResearchPredictor(rb)) == {
        "manifest_sha256": res.manifest_sha256,
        "kind": "research",
        "cache_sha256": res.lineage["cache"]["sha256"],
        "held_out_sims": ["synthsim2"],
    }
    lie = dict(rb.split, held_out_sims=["synthsim1"])
    rb2 = rb._replace(
        split=lie,
        research=dict(rb.research, split=dict(rb.research["split"], held_out_sims=["synthsim1"])),
    )
    assert hp.backbone_identity(hp.ResearchPredictor(rb2))["held_out_sims"] == ["synthsim2"]


def test_backbone_identity_refuses_disagreeing_members_or_another_cache(res):
    rb = _rb(res)
    other = _metas(
        rb, lambda i, m: dict(m, recipe=dict(m["recipe"], holdout_sims=["synthsim1"])) if i else m
    )
    with pytest.raises(MR.MFRecordError, match="different simulations"):
        hp.backbone_identity(hp.ResearchPredictor(other))
    cache = _metas(rb, lambda i, m: dict(m, cache_sha256=H64("another cache")) if i else m)
    with pytest.raises(MR.MFRecordError, match="training cache"):
        hp.backbone_identity(hp.ResearchPredictor(cache))


def test_backbone_identity_refuses_metas_whose_excluded_set_cannot_be_derived(res):
    """R14a (panel M1): a member naming simulations under a key that says neither held out nor trained on has no
    derivable excluded set; the identity is refused, not guessed from the other keys."""
    rb = _rb(res)
    vague = _metas(
        rb, lambda i, m: dict(m, recipe=dict(m["recipe"], compared_with=["synthsim2"])) if i else m
    )
    with pytest.raises(MR.MFRecordError, match="cannot be derived from their metas"):
        hp.backbone_identity(hp.ResearchPredictor(vague))


def test_fold_whose_backbone_saw_the_held_out_group_is_refused(res, tmp_path):
    c = res.lineage["cache"]["sha256"]
    d = tmp_path / "saw"  # the backbone holds out synthsim2, the fold synthsim1
    s = write_mf(
        d,
        manifest=res.manifest_sha256,
        cache=c,
        kind="fold",
        backbone_kind="research",
        held_out=["synthsim2"],
        fold={"fold_id": "f1", "holdout_sims": ["synthsim1"]},
    )
    with pytest.raises(MR.MFRecordError, match="trained on held-out"):
        hp.load_research_bundle(
            res.bundle_dir, res.manifest_sha256, purpose="a test", mf_dir=str(d), mf_record_sha256=s
        )
    d2 = tmp_path / "claims"  # the record claims a held-out set the metas do not have
    s2 = write_mf(
        d2,
        manifest=res.manifest_sha256,
        cache=c,
        kind="fold",
        backbone_kind="research",
        held_out=["synthsim1"],
        fold={"fold_id": "f1", "holdout_sims": ["synthsim1"]},
    )
    with pytest.raises(MR.MFRecordError, match="member metas say"):
        hp.load_research_bundle(
            res.bundle_dir,
            res.manifest_sha256,
            purpose="a test",
            mf_dir=str(d2),
            mf_record_sha256=s2,
        )


def test_one_fold_through_the_predictor_reproduces_the_cache_input_path(res, tmp_path):
    c = res.lineage["cache"]["sha256"]
    d = tmp_path / "fold"
    s = write_mf(
        d,
        manifest=res.manifest_sha256,
        cache=c,
        kind="fold",
        backbone_kind="research",
        held_out=["synthsim2"],
        fold={"fold_id": "f2", "holdout_sims": ["synthsim2"]},
    )
    rp = hp.load_research_bundle(
        res.bundle_dir, res.manifest_sha256, purpose="a test", mf_dir=str(d), mf_record_sha256=s
    )
    zs, taus = [2.6, 3.0, 3.4], [0.25, 0.4, 0.33]
    x = np.stack([_x(z) for z in zs])
    cache_path = hp.heldout_P_filt_at(rp, x, np.array(taus))
    for i, (z, t) in enumerate(zip(zs, taus)):
        np.testing.assert_allclose(
            rp.predict(PARAMS, z, tau_eff=t).P_class, cache_path["mf"][i], rtol=1e-12, atol=0
        )
        np.testing.assert_allclose(
            rp.predict(PARAMS, z, tau_eff=t, fidelity="LF").P_class,
            cache_path["lf"][i],
            rtol=1e-12,
            atol=0,
        )
    np.testing.assert_allclose(
        cache_path["lf"], hp.lf_P_filt_at(rp, x, np.array(taus)), rtol=1e-12, atol=0
    )


def test_held_out_path_needs_a_fold_record(emu_lf, emu_mf):
    for e in (emu_lf, emu_mf):
        with pytest.raises(MR.MFRecordError, match="fold MF record"):
            hp.heldout_P_filt_at(e, _x(3.0)[None], np.array([0.4]))


# ------------------------------------------------------------------------- the release manifest (C1 review m3)
@pytest.fixture(scope="module")
def release(tmp_path_factory):
    """A synthetic release directory: backbone/ (h5/2 bundle), mf/ (its MF record), other_mf/ (a record of another
    backbone) and release.json naming backbone/ and mf/."""
    from hcd_analysis.emulator import release_manifest as RM

    root = tmp_path_factory.mktemp("rel") / "release"
    root.mkdir()
    b = RT.write_synthetic_bundle(str(root / "backbone"), n_members=2)
    s = write_mf(root / "mf", manifest=b.manifest_sha256, cache=b.lineage["cache"]["sha256"])
    o = RT.write_synthetic_bundle(
        str(tmp_path_factory.mktemp("o") / "other"), n_members=2, seeds=[7, 8]
    )
    so = write_mf(root / "other_mf", manifest=o.manifest_sha256, cache=o.lineage["cache"]["sha256"])
    _, rs = RM.write_release_manifest(
        str(root),
        release_version="synthetic-release-1",
        backbone_dir="backbone",
        backbone_manifest_sha256=b.manifest_sha256,
        mf_dir="mf",
        mf_record_sha256=s,
    )
    return {"root": root, "sha": rs, "bundle": b, "mf_sha": s, "other_mf_sha": so}


def _manifest_copy(release, tmp_path, fn):
    """A copy of the release whose release.json is rewritten by fn(dict) and re-anchored: (dir, its sha256)."""
    import hashlib
    import json
    import shutil

    d = tmp_path / "copy"
    shutil.copytree(release["root"], d, symlinks=True)
    m = json.load(open(d / "release.json"))
    raw = json.dumps(fn(m), indent=1, sort_keys=True).encode()
    os.chmod(d / "release.json", 0o644)
    (d / "release.json").write_bytes(raw)
    return str(d), hashlib.sha256(raw).hexdigest()


def test_release_round_trip_equals_the_two_digest_path(release):
    r, b = release["root"], release["bundle"]
    emu = hp.load_release(str(r), release["sha"])
    two = hp.load_bundle(
        b.bundle_dir, b.manifest_sha256, mf_dir=str(r / "mf"), mf_record_sha256=release["mf_sha"]
    )
    p, q = emu.predict(PARAMS, 3.0, tau_eff=0.4), two.predict(PARAMS, 3.0, tau_eff=0.4)
    assert p.P_class.tobytes() == q.P_class.tobytes() and p.flags["mf_applied"] is True
    assert p.provenance["release"] == {
        "format": "hcd_priya.release/1",
        "release_version": "synthetic-release-1",
        "release_sha256": release["sha"],
    }
    assert p.provenance["mf"] == q.provenance["mf"] and "release" not in q.provenance
    assert (
        "release" not in hp.load_bundle(b.bundle_dir, b.manifest_sha256).provenance
    )  # LF-only unchanged
    from hcd_analysis.emulator import release_manifest as RM

    with pytest.raises(FileExistsError, match="written once"):
        RM.write_release_manifest(
            str(r),
            release_version="again",
            backbone_dir="backbone",
            backbone_manifest_sha256=b.manifest_sha256,
            mf_dir="mf",
            mf_record_sha256=release["mf_sha"],
        )


def test_release_is_anchored_on_its_published_sha256(release):
    from hcd_analysis.emulator import release_manifest as RM

    for bad in (None, H64("another release"), release["sha"].upper()):
        with pytest.raises(RM.ReleaseManifestError, match="sha256"):
            hp.load_release(str(release["root"]), bad)


def test_release_without_an_mf_part_is_refused(release, tmp_path):
    from hcd_analysis.emulator import release_manifest as RM

    d, s = _manifest_copy(release, tmp_path, lambda m: dict(m, mf=None))
    with pytest.raises(RM.ReleaseManifestError, match="names no MF record"):
        hp.load_release(d, s)
    d2, s2 = _manifest_copy(
        release, tmp_path / "b", lambda m: {k: v for k, v in m.items() if k != "mf"}
    )
    with pytest.raises(RM.ReleaseManifestError, match="missing \\['mf'\\]"):
        hp.load_release(d2, s2)


def test_release_manifest_refuses_an_extra_key_and_another_format(release, tmp_path):
    from hcd_analysis.emulator import release_manifest as RM

    for i, (fn, why) in enumerate(
        (
            (lambda m: dict(m, note="anything"), "unknown \\['note'\\]"),
            (lambda m: dict(m, mf=dict(m["mf"], tables_sha256=H64("x"))), "mf: keys"),
            (lambda m: dict(m, format="hcd_priya.release/0"), "format"),
            (lambda m: dict(m, release_version=" "), "release_version"),
        )
    ):
        d, s = _manifest_copy(release, tmp_path / str(i), fn)
        with pytest.raises(RM.ReleaseManifestError, match=why):
            hp.load_release(d, s)


def test_release_part_directories_are_plain_names_inside_the_release(release, tmp_path):
    from hcd_analysis.emulator import release_manifest as RM

    cases = [
        (lambda m: dict(m, backbone=dict(m["backbone"], dir="../backbone")), "plain path segment"),
        (lambda m: dict(m, mf=dict(m["mf"], dir="sub/mf")), "plain path segment"),
        (lambda m: dict(m, mf=dict(m["mf"], dir="..")), "plain path segment"),
        (
            lambda m: dict(m, mf=dict(m["mf"], dir=str(release["root"] / "mf"))),
            "plain path segment",
        ),
        (lambda m: dict(m, mf=dict(m["mf"], dir="backbone")), "two different directories"),
        (lambda m: dict(m, mf=dict(m["mf"], dir="absent")), "not a directory inside the release"),
    ]
    for i, (fn, why) in enumerate(cases):
        d, s = _manifest_copy(release, tmp_path / str(i), fn)
        with pytest.raises(RM.ReleaseManifestError, match=why):
            hp.load_release(d, s)
    d, s = _manifest_copy(
        release, tmp_path / "link", lambda m: dict(m, mf=dict(m["mf"], dir="mf_link"))
    )
    os.symlink(os.path.join(d, "mf"), os.path.join(d, "mf_link"))
    with pytest.raises(RM.ReleaseManifestError, match="symlink"):
        hp.load_release(d, s)


@pytest.mark.parametrize("target", ["inside", "outside"])
def test_a_symlinked_backbone_part_is_refused(release, tmp_path, target):
    """Delta review R1-3b (mutant D5c, forgery Q4g): the backbone part, like the MF part, is a plain directory, never a
    symlink, whether the link points at the release's own backbone or at a bundle outside the release. load_bundle
    itself accepts a symlinked bundle directory, so this check is the only one that catches it."""
    import shutil
    from hcd_analysis.emulator import release_manifest as RM

    d, s = _manifest_copy(
        release,
        tmp_path / target,
        lambda m: dict(m, backbone=dict(m["backbone"], dir="backbone_link")),
    )
    if target == "inside":
        link_to = os.path.join(d, "backbone")
    else:
        link_to = str(tmp_path / "elsewhere")
        shutil.copytree(os.path.join(d, "backbone"), link_to, symlinks=True)
    os.symlink(link_to, os.path.join(d, "backbone_link"))
    assert hp.load_bundle(
        link_to, release["bundle"].manifest_sha256
    ).manifest_sha256  # the target itself loads
    with pytest.raises(
        RM.ReleaseManifestError, match="backbone: 'backbone_link' is not a directory .*symlink"
    ):
        hp.load_release(d, s)


def test_release_naming_a_mismatched_pair_is_refused(release, tmp_path):
    """The manifest names another backbone's MF record (with its correct digest): the record's backbone pin refuses
    it; and a correct record named with a wrong digest is refused by its anchor."""
    d, s = _manifest_copy(
        release,
        tmp_path / "a",
        lambda m: dict(m, mf={"dir": "other_mf", "record_sha256": release["other_mf_sha"]}),
    )
    with pytest.raises(MR.MFRecordError, match="backbone"):
        hp.load_release(d, s)
    d2, s2 = _manifest_copy(
        release,
        tmp_path / "b",
        lambda m: dict(m, mf=dict(m["mf"], record_sha256=release["other_mf_sha"])),
    )
    with pytest.raises(MR.MFRecordError, match="sha256"):
        hp.load_release(d2, s2)
    d3, s3 = _manifest_copy(
        release,
        tmp_path / "c",
        lambda m: dict(m, backbone=dict(m["backbone"], manifest_sha256=H64("other"))),
    )
    from hcd_analysis.emulator import safe_h5 as H

    with pytest.raises(H.SafeFormatError):  # the backbone's own anchor
        hp.load_release(d3, s3)


@pytest.mark.parametrize(
    "make",
    [lambda s: LyingStr(H64("x")), lambda s: PlainStr(s)],
    ids=["lying-str-other-digest", "plain-str-subclass-own-digest"],
)
def test_release_digest_must_be_an_exact_str(release, make):
    """Delta review R1-2: the published release sha256 is exact str; a lying subclass would pass the comparison for
    any release.json."""
    from hcd_analysis.emulator import release_manifest as RM

    with pytest.raises(RM.ReleaseManifestError, match="published sha256"):
        hp.load_release(str(release["root"]), make(release["sha"]))
    assert (
        hp.load_release(str(release["root"]), release["sha"]).provenance["release"][
            "release_sha256"
        ]
        == release["sha"]
    )


@pytest.mark.parametrize(
    "field", ["backbone_dir", "mf_dir", "backbone_manifest_sha256", "mf_record_sha256"]
)
def test_release_manifest_parts_must_be_exact_str(release, tmp_path, field):
    """Delta review R1-2: the part names and digests written into release.json are exact str (a parsed manifest
    always has them); a subclass is refused before anything is written."""
    from hcd_analysis.emulator import release_manifest as RM
    import shutil

    d = tmp_path / "rel"
    shutil.copytree(release["root"], d, symlinks=True)
    os.chmod(d / "release.json", 0o644)
    os.remove(d / "release.json")
    kw = dict(
        release_version="v",
        backbone_dir="backbone",
        mf_dir="mf",
        backbone_manifest_sha256=release["bundle"].manifest_sha256,
        mf_record_sha256=release["mf_sha"],
    )
    with pytest.raises(RM.ReleaseManifestError, match="exact str"):
        RM.write_release_manifest(str(d), **dict(kw, **{field: PlainStr(kw[field])}))
    assert not os.path.lexists(d / "release.json")
    assert RM.write_release_manifest(str(d), **kw)[1]


# ------------------------------------------------------------------------------- the backbone digest (R2-1)
def test_backbone_digest_is_taken_only_as_an_exact_str(info, res):
    """R2-1: the backbone manifest digest given to safe_h5.load_release / load_research (and through them to
    predictor.load_bundle / load_research_bundle) must be an exact str, like the MF and release digests. A str
    subclass that compares equal to anything (a forged digest, here of the wrong value) is refused before any
    comparison; so are a plain subclass and np.str_ carrying the right value; the plain str of that value loads.
    """
    from hcd_analysis.emulator import safe_h5 as H

    wrong = LyingStr("0" * 64)
    with pytest.raises(H.SafeFormatError, match="load_release needs .*an exact str"):
        H.load_release(info.bundle_dir, expected_manifest_sha256=wrong)
    with pytest.raises(H.SafeFormatError, match="research load needs .*an exact str"):
        H.load_research(
            res.bundle_dir, expected_manifest_sha256=LyingStr("0" * 64), purpose="a test"
        )
    with pytest.raises(H.SafeFormatError, match="exact str"):
        hp.load_research_bundle(res.bundle_dir, LyingStr("0" * 64), purpose="a test")
    for v in (wrong, PlainStr(info.manifest_sha256), np.str_(info.manifest_sha256)):
        with pytest.raises(H.SafeFormatError, match="exact str"):
            hp.load_bundle(info.bundle_dir, v)
    for v in (PlainStr(res.manifest_sha256), np.str_(res.manifest_sha256)):
        with pytest.raises(H.SafeFormatError, match="exact str"):
            hp.load_research_bundle(res.bundle_dir, v, purpose="a test")
    assert (
        hp.load_bundle(info.bundle_dir, str(np.str_(info.manifest_sha256))).manifest_sha256
        == info.manifest_sha256
    )
    assert (
        hp.load_research_bundle(
            res.bundle_dir, str(PlainStr(res.manifest_sha256)), purpose="a test"
        ).manifest_sha256
        == res.manifest_sha256
    )
