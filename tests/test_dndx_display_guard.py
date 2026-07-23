"""Display-path estimand guard (paper-agent request 2026-07-20, defence-in-depth for the
artifact-only sourcing rule): the retired tau_LL >= 2 LLS display object must fail LOUDLY
if requested by name, and display labels/estimand ids must be assertable so corrected
points can never render under the old, false legends (the silent-substitution hazard:
the corrected LIT preserved attribute name + 4-tuple arity, so old hard-coded legends
would not error).

Complements the existing LAW-boundary guard (inference.assert_dndx_law_estimand,
tests/test_lit_dndx_corrected.py); this file guards the DISPLAY boundary.
"""
import pytest

from hcd_analysis.emulator import lit_dndx
from hcd_analysis.emulator.inference import HCD_LIT_DNDX_ESTIMAND


CLS = ("LLS", "subDLA", "DLA")


# ---------------------------------------------------------------- retired names
@pytest.mark.parametrize("name", ["LIT_TAU2", "LLS_TAU2_COMPILATION",
                                  "LLS_TAU2_DISPLAY", "lit_points_tau2",
                                  "lls_tau2_points_for_display"])
def test_retired_name_access_raises_tombstone(name):
    with pytest.raises(AttributeError, match="RETIRED"):
        getattr(lit_dndx, name)


def test_unknown_name_still_plain_attributeerror():
    with pytest.raises(AttributeError):
        lit_dndx.no_such_attribute_xyz
    # the documented defect-record constant stays accessible (it is a tombstone DOC,
    # not a retired display object)
    assert isinstance(lit_dndx.LLS_TAU2_OLD_DEFECTS, dict)


# ------------------------------------------------------------- estimand identity
def test_estimand_ids_match_law_boundary():
    assert lit_dndx.ESTIMAND_ID == dict(HCD_LIT_DNDX_ESTIMAND)


def test_display_payload_carries_agreeing_ids_and_labels():
    d = lit_dndx.lit_points_for_display()
    assert d["estimand_id"] == lit_dndx.ESTIMAND_ID
    assert d["estimand_label"] == lit_dndx.ESTIMAND_LABEL
    # labels pass the guard (id <-> label agreement is what the paper acceptance
    # check verifies)
    lit_dndx.assert_display_labels(d["estimand_label"], where="test payload")
    # per-class source strings carry no retired token
    for c in CLS:
        src = d[c][3]
        for tok in lit_dndx.RETIRED_DISPLAY_TOKENS:
            assert tok not in src, (c, tok, src)


def test_labels_carry_the_binned_ranges():
    lab = lit_dndx.ESTIMAND_LABEL
    assert "17.2" in lab["LLS"] and "19.0" in lab["LLS"] and "K1a" in lab["LLS"]
    assert "19.0" in lab["subDLA"] and "20.3" in lab["subDLA"]
    assert "20.3" in lab["DLA"]


# ------------------------------------------------------------------ guard firing
def test_assert_display_estimand_accepts_deployed():
    for c in CLS:
        lit_dndx.assert_display_estimand(c, lit_dndx.ESTIMAND_ID[c], where="test ok")


def test_assert_display_estimand_rejects_wrong_and_retired():
    with pytest.raises(AssertionError, match="estimand"):
        lit_dndx.assert_display_estimand("LLS", "binned_19.0_20.3", where="test wrong")
    with pytest.raises(AssertionError, match="RETIRED"):
        lit_dndx.assert_display_estimand("LLS", "cumulative_tau2_ge17.5",
                                         where="test retired")


@pytest.mark.parametrize("bad", [r"$\tau_{\rm LL}\geq 2$ systems",
                                 "tau_LL >= 2 (cumulative)",
                                 "tau912>=2 compilation"])
def test_assert_display_labels_rejects_retired_tokens(bad):
    lab = dict(lit_dndx.ESTIMAND_LABEL)
    lab["LLS"] = bad
    with pytest.raises(AssertionError, match="RETIRED|required"):
        lit_dndx.assert_display_labels(lab, where="test bad label")


def test_assert_display_labels_rejects_missing_range_token():
    lab = dict(lit_dndx.ESTIMAND_LABEL)
    lab["subDLA"] = "sub-DLA incidence"
    with pytest.raises(AssertionError, match="required"):
        lit_dndx.assert_display_labels(lab, where="test missing token")
