"""OOS instrument-resolution arm selector (Task 2C): --b-res-oos-member wires the Task-2A basis
npz (res_instr_injection_basis.npz, members bres1/bres2/bres_real/bstar) into the resolution arm
INSTEAD of the scalar --b-res. Golden-safe: with oos_member=None (the default), arm_inject_spec /
build_arm_ctx / the write-time out_arm are BYTE-IDENTICAL to the pre-Task-2C scalar-only arm."""
import pytest

import hcd_analysis.emulator  # x64 before jax
from scripts.run_dnuis_bias_shard import RES_INSTR_BASIS, _out_arm, _resinj_oos_spec, arm_inject_spec


def test_resinj_oos_spec_none_is_noop():
    """No member selected -> None (the runner keeps the scalar b_res arm)."""
    assert _resinj_oos_spec(None, 1.0) is None
    assert _resinj_oos_spec(None, -1.0) is None       # strength irrelevant when member is None


def test_resinj_oos_spec_member_builds_basis_dict():
    assert _resinj_oos_spec("bstar", -1.0) == {"path": RES_INSTR_BASIS, "member": "bstar", "strength": -1.0}
    assert _resinj_oos_spec("bres1", 1.0) == {"path": RES_INSTR_BASIS, "member": "bres1", "strength": 1.0}


def test_arm_inject_spec_scalar_unchanged_when_oos_member_none():
    """The default (oos_member=None) path is BYTE-IDENTICAL to the pre-Task-2C scalar arm."""
    assert arm_inject_spec("resolution", "desi", b_res=0.02, oos_member=None) == {"resolution": {"b_res": 0.02}}
    assert arm_inject_spec("resolution", "desi", b_res=0.02) == {"resolution": {"b_res": 0.02}}  # kwarg omitted too


def test_arm_inject_spec_oos_member_selects_basis():
    got = arm_inject_spec("resolution", "desi", b_res=0.02, oos_member="bstar", oos_strength=1.0)
    assert got == {"resolution": {"path": RES_INSTR_BASIS, "member": "bstar", "strength": 1.0}}


def test_arm_inject_spec_oos_ks_not_ready_still_raises():
    """The KS-not-ready guard (echelle R_z + diag surgery required before ANY resolution injection)
    applies to the OOS arm too -- it must not silently bypass the guard."""
    with pytest.raises(SystemExit, match="[Kk][Ss]"):
        arm_inject_spec("resolution", "ks", b_res=0.02, oos_member="bstar")


def test_arm_inject_spec_oos_ks_ready_allows():
    got = arm_inject_spec("resolution", "ks", b_res=0.02, oos_member="bres_real",
                          ks_resolution_ready=True, oos_strength=-1.0)
    assert got == {"resolution": {"path": RES_INSTR_BASIS, "member": "bres_real", "strength": -1.0}}


def test_out_arm_write_time_mapping():
    """The write-time output tag: an OOS member set -> "resolution_oos" (so it never collides with the
    scalar in-span "resolution" arm's pkls); member None -> unchanged (byte-identical, any arm)."""
    assert _out_arm("resolution", "bstar") == "resolution_oos"
    assert _out_arm("resolution", None) == "resolution"
    assert _out_arm("metal_misspec", None) == "metal_misspec"
