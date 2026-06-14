"""Phase-5a EMUCOH closure-validation arms (run_stepA.build_config).

The emucoh closure gate compares each "_EC1" mock (mf_emucoh=1.0, term ON) against its "_EC0"
control (the same mock with mf_emucoh=0), BOTH run fresh with current code. For that comparison to
be CLEAN — only the likelihood covariance differs, the mock data + noise are byte-identical — every
_EC1 chain must match its _EC0 chain in EVERY field except ``mf_emucoh`` (and the id/mock_id). This
test pins that invariant so a future edit can't silently de-pair the ON/OFF arms.

It also pins that the validation mocks ISOLATE emucoh: no MF resolution floor (mf=False),
no shape floor (mf_shape=0), no diagonal-floor-on-DESI (desi_floor=False) — so the ON arm adds
ONLY the 60-sim emucoh term.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path("/home/mfho/hcd_priya")
sys.path.insert(0, str(ROOT / "scripts"))

import run_stepA  # noqa: E402


# the 8 baselines mirrored as a matched (OFF control, ON) pair by the EMUCOH validation block.
BASES = ["D_f3", "D_f4", "D_f6", "D_f7", "D_lmed3_15", "D_lmed5_15", "D_lmed7_15", "D_llsmed"]
PAIRS = [(f"{b}_EC0", f"{b}_EC1") for b in BASES]
# EC2 = the per-term diagonal-allocation re-validation arm (EC1 + mf_emucoh_offdiag_only) on a subset.
ODA_BASES = ["D_f3", "D_f6", "D_lmed7_15", "D_llsmed"]

# fields that ARE allowed to differ between an OFF baseline chain and its ON copy.
ALLOWED_DIFF = {"id", "mock_id", "mf_emucoh"}


@pytest.fixture(scope="module")
def by_mock():
    cfg = run_stepA.build_config()
    d = {}
    for c in cfg:
        d.setdefault(c["mock_id"], []).append(c)
    return d


def test_each_EC_arm_exists_with_4_chains(by_mock):
    for off, on in PAIRS:
        assert off in by_mock, f"OFF baseline {off} missing from build_config"
        assert on in by_mock, f"ON copy {on} missing from build_config"
        assert len(by_mock[on]) == 4, f"{on} should have 4 chains, has {len(by_mock[on])}"
        assert len(by_mock[off]) == 4, f"{off} should have 4 chains, has {len(by_mock[off])}"


def test_EC_arms_turn_emucoh_on_baseline_off(by_mock):
    for off, on in PAIRS:
        for c in by_mock[off]:
            assert float(c["mf_emucoh"]) == 0.0, f"OFF {off} must have mf_emucoh=0"
        for c in by_mock[on]:
            assert float(c["mf_emucoh"]) == 1.0, f"ON {on} must have mf_emucoh=1.0"


def test_EC_arms_isolate_emucoh_no_other_floor(by_mock):
    # mf=False (no MF resolution correction/floor), mf_shape=0 (no shape floor), desi_floor=False
    # (no diagonal-floor-on-DESI) — so the ON arm adds ONLY the emucoh term.
    for _off, on in PAIRS:
        for c in by_mock[on]:
            assert c["mf"] is False, f"{on} must be LF (mf=False) to isolate emucoh"
            assert float(c["mf_shape"]) == 0.0, f"{on} must have mf_shape=0"
            assert c["desi_floor"] is False, f"{on} must have desi_floor=False"
            assert c["survey"] == "DESI", f"{on} must be the DESI leg"


def test_ON_copy_is_byte_identical_to_OFF_except_emucoh(by_mock):
    # pair by chain_id; assert every field except {id, mock_id, mf_emucoh} matches EXACTLY. This is
    # what guarantees the mock data + noise (seed/fold/sim) are identical ⇒ a clean ON/OFF compare.
    for off, on in PAIRS:
        off_by_cid = {c["chain_id"]: c for c in by_mock[off]}
        on_by_cid = {c["chain_id"]: c for c in by_mock[on]}
        assert set(off_by_cid) == set(on_by_cid) == {0, 1, 2, 3}
        for cid in (0, 1, 2, 3):
            co, cn = off_by_cid[cid], on_by_cid[cid]
            keys = set(co) | set(cn)
            for key in keys:
                if key in ALLOWED_DIFF:
                    continue
                assert co.get(key) == cn.get(key), (
                    f"{on} chain {cid} field {key!r}={cn.get(key)!r} != OFF {off} {co.get(key)!r} "
                    f"(ON/OFF must be byte-identical except emucoh)")
            # and the mock-identifying fields that fix the noise draw are the ones we just checked:
            assert co["fold"] == cn["fold"] and co["sim"] == cn["sim"] and co["seed"] == cn["seed"]


def test_total_config_grew_by_80_chains(by_mock):
    # 8 baselines × (EC0 + EC1) + 4 EC2 = (16 + 4) mocks × 4 chains. Sanity vs drop/dup.
    ec_chains = (sum(len(by_mock[off]) + len(by_mock[on]) for off, on in PAIRS)
                 + sum(len(by_mock[f"{b}_EC2"]) for b in ODA_BASES))
    assert ec_chains == 80


def test_EC2_is_EC1_plus_offdiag_only(by_mock):
    # EC2 must be byte-identical to EC1 except mf_emucoh_offdiag_only (True vs False); same fold/sim/
    # seed ⇒ identical mock, only the diagonal allocation differs. EC1/EC0 carry offdiag_only=False.
    for b in ODA_BASES:
        ec1 = {c["chain_id"]: c for c in by_mock[f"{b}_EC1"]}
        ec2 = {c["chain_id"]: c for c in by_mock[f"{b}_EC2"]}
        assert set(ec1) == set(ec2) == {0, 1, 2, 3}
        for cid in (0, 1, 2, 3):
            c1, c2 = ec1[cid], ec2[cid]
            assert c1.get("mf_emucoh_offdiag_only") is False
            assert c2.get("mf_emucoh_offdiag_only") is True
            assert float(c1["mf_emucoh"]) == float(c2["mf_emucoh"]) == 1.0
            for key in (set(c1) | set(c2)):
                if key in {"id", "mock_id", "mf_emucoh_offdiag_only"}:
                    continue
                assert c1.get(key) == c2.get(key), f"{b}_EC2 chain {cid} field {key!r} differs from EC1"
