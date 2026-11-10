"""Manifest-pinned production ensemble (freeze decision 6) -- the fail-loud battery.

The deployed drivers used to build the production member list with the permissive glob
``final_prod_seed*.eqx``; a stray ``final_prod_seed5.eqx`` would have silently JOINED the
deployed ensemble. This suite pins the replacement contract
(``hcd_analysis.emulator.prod_ensemble`` + ``scripts/gen_ensemble_manifest.py`` +
``checkpoints/production_ensemble_manifest.json``):

  * modified checkpoint / modified normalizer / manifest-digest edit  -> DIGEST MISMATCH
  * missing member (.eqx or .norm.pkl)                                -> MISSING
  * unexpected extra member (stray final_prod_seed5.eqx)              -> TRIPWIRE raise
  * incorrect member count (n_members edit, dropped entry)            -> COUNT raise
  * wrong checkpoint<->normalizer pairing (manifest or on-disk swap)  -> PAIRING/DIGEST raise
  * ordering mismatch (reordered manifest members)                    -> ORDER raise
  * alternate ensemble prefix or glob-shaped stray                    -> TRIPWIRE/MISSING raise
  * gen_ensemble_manifest --check                                     -> exit 0 green / 1 red
  * REAL checkpoints dir happy path (skip if the binaries are absent)
  * doc agreement: README digest table + analysis.lock member list == manifest

Fake fixtures use tiny files: every red path fires in ``verify_manifest`` BEFORE any
equinox/jax deserialization, so no real checkpoints are needed for the red battery.
"""
import os as _os_rr  # this checkout's root (emulator-debug 2026-10; never an absolute literal)
_REPO_ROOT = _os_rr.path.dirname(_os_rr.path.dirname(_os_rr.path.abspath(__file__)))
import hashlib
import json
import os
from pathlib import Path

import pytest

from hcd_analysis.emulator import prod_ensemble as PE
from scripts import gen_ensemble_manifest as GEN

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MANIFEST = ROOT / "checkpoints" / PE.MANIFEST_BASENAME
README = ROOT / "checkpoints" / "README_production_ensemble.md"
ANALYSIS_LOCK = ROOT / "analysis.lock"
from tests.gate_helpers import external_path  # noqa: E402
REAL_CKPT_DIR = Path(external_path("HCD_CHECKPOINTS_DIR"))   # gate C members
BASE = PE.PROD_BASENAME


# --------------------------------------------------------------------------------------------- #
#  fixtures: a tiny fake 5-member ensemble + its (honestly generated) manifest
# --------------------------------------------------------------------------------------------- #
FAKE_CACHE = "d9" * 32


def _meta(i, schema="2.0", cache=FAKE_CACHE):
    import math
    return {"seed": i, "schema_version": schema, "cache_sha256": cache,
            "k_com_hmpc": [2 * math.pi * n / 120.0 for n in range(1, 5)]}


def _make_fake_members(d):
    norm_bytes = b"NORM-IDENTICAL-BYTES-v1" * 8       # byte-identical across members (as in prod)
    for i in range(5):
        (d / f"{BASE}{i}.eqx").write_bytes(f"EQX-MEMBER-{i}-".encode() * 16)
        (d / f"{BASE}{i}.norm.pkl").write_bytes(norm_bytes)
        (d / f"{BASE}{i}.meta.json").write_text(json.dumps(_meta(i)) + "\n")
        (d / f"{BASE}{i}.hist.json").write_text("{}\n")


@pytest.fixture
def fake(tmp_path):
    """(checkpoints_dir, manifest_path) for a well-formed fake ensemble."""
    d = tmp_path / "checkpoints"
    d.mkdir()
    _make_fake_members(d)
    mpath = d / PE.MANIFEST_BASENAME
    rc = GEN.main(["--checkpoints-dir", str(d), "--manifest", str(mpath), "--record-checkpoints-dir"])
    assert rc == 0
    return d, mpath


def _flip_last_byte(path):
    data = bytearray(Path(path).read_bytes())
    data[-1] ^= 0xFF
    Path(path).write_bytes(bytes(data))


def _edit_manifest(mpath, mutate):
    m = json.loads(Path(mpath).read_text())
    mutate(m)
    Path(mpath).write_text(json.dumps(m, indent=2) + "\n")


def _verify(fake_pair):
    d, mpath = fake_pair
    return PE.verify_manifest(checkpoints_dir=str(d), manifest_path=str(mpath))


# --------------------------------------------------------------------------------------------- #
#  happy path (fake): structure, order, prefixes
# --------------------------------------------------------------------------------------------- #
def test_fake_happy_path_verify_and_paths(fake):
    d, mpath = fake
    manifest, prefixes = _verify(fake)
    assert manifest["n_members"] == 5 and len(prefixes) == 5
    assert [os.path.basename(p) for p in prefixes] == [f"{BASE}{i}" for i in range(5)]
    assert prefixes == PE.production_member_paths(checkpoints_dir=str(d),
                                                  manifest_path=str(mpath))
    # generated manifest records the byte-identical normalizers truthfully
    assert manifest["normalizers_byte_identical"] is True


# --------------------------------------------------------------------------------------------- #
#  modified checkpoint / modified normalizer / manifest digest edit -> digest mismatch
# --------------------------------------------------------------------------------------------- #
def test_modified_checkpoint_rejected(fake):
    d, _ = fake
    _flip_last_byte(d / f"{BASE}3.eqx")
    with pytest.raises(PE.ProductionEnsembleError, match="DIGEST MISMATCH") as e:
        _verify(fake)
    assert f"{BASE}3.eqx" in str(e.value)
    assert "manifest expects sha256" in str(e.value) and "on-disk file is" in str(e.value)


def test_modified_normalizer_rejected(fake):
    d, _ = fake
    _flip_last_byte(d / f"{BASE}1.norm.pkl")
    with pytest.raises(PE.ProductionEnsembleError, match="DIGEST MISMATCH") as e:
        _verify(fake)
    assert f"{BASE}1.norm.pkl" in str(e.value)


def test_manifest_digest_edit_rejected(fake):
    _, mpath = fake

    def mutate(m):
        m["members"][2]["eqx_sha256"] = "0" * 64

    _edit_manifest(mpath, mutate)
    with pytest.raises(PE.ProductionEnsembleError, match="DIGEST MISMATCH") as e:
        _verify(fake)
    assert "0" * 64 in str(e.value)          # expected digest quoted
    assert f"{BASE}2.eqx" in str(e.value)


# --------------------------------------------------------------------------------------------- #
#  missing member
# --------------------------------------------------------------------------------------------- #
def test_missing_checkpoint_rejected(fake):
    d, _ = fake
    (d / f"{BASE}2.eqx").unlink()
    with pytest.raises(PE.ProductionEnsembleError, match="MISSING") as e:
        _verify(fake)
    assert f"{BASE}2.eqx" in str(e.value)


def test_missing_normalizer_rejected(fake):
    d, _ = fake
    (d / f"{BASE}4.norm.pkl").unlink()
    with pytest.raises(PE.ProductionEnsembleError, match="MISSING") as e:
        _verify(fake)
    assert f"{BASE}4.norm.pkl" in str(e.value)


def test_missing_manifest_rejected(tmp_path):
    with pytest.raises(PE.ProductionEnsembleError, match="manifest not found"):
        PE.verify_manifest(checkpoints_dir=str(tmp_path),
                           manifest_path=str(tmp_path / "nope.json"))


# --------------------------------------------------------------------------------------------- #
#  unexpected extra member: the stray final_prod_seed5.eqx tripwire (the bug this closes)
# --------------------------------------------------------------------------------------------- #
def test_stray_extra_member_rejected(fake):
    d, _ = fake
    (d / f"{BASE}5.eqx").write_bytes(b"EQX-STRAY")
    with pytest.raises(PE.ProductionEnsembleError, match="UNEXPECTED") as e:
        _verify(fake)
    assert f"{BASE}5.eqx" in str(e.value)


def test_stray_blocks_load_before_deserialization(fake):
    """load_production_ensemble must raise on the stray BEFORE any equinox deserialization
    (the fakes are not real .eqx files, so reaching the loader would raise something else)."""
    d, mpath = fake
    (d / f"{BASE}5.eqx").write_bytes(b"EQX-STRAY")
    with pytest.raises(PE.ProductionEnsembleError, match="UNEXPECTED"):
        PE.load_production_ensemble(checkpoints_dir=str(d), manifest_path=str(mpath))


def test_glob_shaped_stray_rejected(fake):
    """Anything matching the production prefix that is not pinned -- even a name that would
    sort inside the old permissive glob (final_prod_seed01.eqx) -- must raise."""
    d, _ = fake
    (d / f"{BASE}01.eqx").write_bytes(b"EQX-GLOB-SHAPED")
    with pytest.raises(PE.ProductionEnsembleError, match="UNEXPECTED") as e:
        _verify(fake)
    assert f"{BASE}01.eqx" in str(e.value)


def test_generator_refuses_stray(fake):
    d, _ = fake
    (d / f"{BASE}7.eqx").write_bytes(b"EQX-STRAY")
    with pytest.raises(PE.ProductionEnsembleError, match="stray"):
        GEN.build_manifest(str(d))


# --------------------------------------------------------------------------------------------- #
#  incorrect member count
# --------------------------------------------------------------------------------------------- #
def test_wrong_n_members_rejected(fake):
    _, mpath = fake
    _edit_manifest(mpath, lambda m: m.update(n_members=4))
    with pytest.raises(PE.ProductionEnsembleError, match="n_members=4"):
        _verify(fake)


def test_dropped_member_entry_rejected(fake):
    _, mpath = fake
    _edit_manifest(mpath, lambda m: m["members"].pop())      # 4 entries, n_members still 5
    with pytest.raises(PE.ProductionEnsembleError, match="count mismatch"):
        _verify(fake)


# --------------------------------------------------------------------------------------------- #
#  wrong checkpoint<->normalizer pairing (structural: the norms are byte-identical, so a
#  digest can NOT catch a mis-pair -- the basename-per-index contract must)
# --------------------------------------------------------------------------------------------- #
def test_mispaired_normalizer_in_manifest_rejected(fake):
    _, mpath = fake

    def mutate(m):
        m["members"][0]["norm"] = f"{BASE}1.norm.pkl"

    _edit_manifest(mpath, mutate)
    with pytest.raises(PE.ProductionEnsembleError, match="MIS-PAIRED") as e:
        _verify(fake)
    assert f"{BASE}0.norm.pkl" in str(e.value)       # expected
    assert f"{BASE}1.norm.pkl" in str(e.value)       # got


def test_on_disk_checkpoint_swap_rejected(fake):
    """Rename files so basenames mis-pair on disk (seed0.eqx <-> seed1.eqx contents swapped):
    the per-file digests then disagree with the manifest."""
    d, _ = fake
    a, b, t = d / f"{BASE}0.eqx", d / f"{BASE}1.eqx", d / "tmp.swap"
    a.rename(t); b.rename(a); t.rename(b)
    with pytest.raises(PE.ProductionEnsembleError, match="DIGEST MISMATCH"):
        _verify(fake)


# --------------------------------------------------------------------------------------------- #
#  ordering mismatch
# --------------------------------------------------------------------------------------------- #
def test_reordered_manifest_rejected(fake):
    _, mpath = fake

    def mutate(m):
        m["members"][0], m["members"][1] = m["members"][1], m["members"][0]

    _edit_manifest(mpath, mutate)
    with pytest.raises(PE.ProductionEnsembleError, match="OUT OF ORDER"):
        _verify(fake)


def test_renamed_index_rejected(fake):
    """Indexes forced back to 0..4 after a swap: the name-per-index contract still fires."""
    _, mpath = fake

    def mutate(m):
        m["members"][0], m["members"][1] = m["members"][1], m["members"][0]
        for i, mem in enumerate(m["members"]):
            mem["index"] = i                                  # hide the swap from the index check

    _edit_manifest(mpath, mutate)
    with pytest.raises(PE.ProductionEnsembleError, match="name mismatch"):
        _verify(fake)


# --------------------------------------------------------------------------------------------- #
#  alternate ensemble prefix: there is NO glob/prefix input to redirect the loader
# --------------------------------------------------------------------------------------------- #
def test_alternate_prefix_ensemble_rejected(fake):
    """A directory whose members carry an alternate prefix can not satisfy the manifest:
    the pinned final_prod_seed<i> files are missing there."""
    d, mpath = fake
    for i in range(5):
        for ext in (".eqx", ".norm.pkl", ".meta.json"):
            (d / f"{BASE}{i}{ext}").rename(d / f"alt_prod_seed{i}{ext}")
    with pytest.raises(PE.ProductionEnsembleError, match="MISSING"):
        _verify(fake)


# --------------------------------------------------------------------------------------------- #
#  gen_ensemble_manifest --check: green and red exit codes
# --------------------------------------------------------------------------------------------- #
def test_gen_check_green_then_red(fake):
    d, mpath = fake
    argv = ["--check", "--checkpoints-dir", str(d), "--manifest", str(mpath)]
    assert GEN.main(argv) == 0
    _flip_last_byte(d / f"{BASE}0.eqx")
    assert GEN.main(argv) != 0


def test_gen_verify_helper_is_subprocess_free(fake):
    d, mpath = fake
    manifest, prefixes = GEN.verify(checkpoints_dir=str(d), manifest_path=str(mpath))
    assert len(prefixes) == 5
    _flip_last_byte(d / f"{BASE}2.norm.pkl")
    with pytest.raises(PE.ProductionEnsembleError, match="DIGEST MISMATCH"):
        GEN.verify(checkpoints_dir=str(d), manifest_path=str(mpath))


# --------------------------------------------------------------------------------------------- #
#  REAL checkpoints happy path (the deployed object) -- skip gracefully when absent
# --------------------------------------------------------------------------------------------- #
def _real_files_present():
    if not COMMITTED_MANIFEST.exists() or not REAL_CKPT_DIR.is_dir():
        return False
    return all((REAL_CKPT_DIR / f"{BASE}{i}{ext}").exists()
               for i in range(5) for ext in (".eqx", ".norm.pkl", ".meta.json"))


def _unavailable(msg):
    """Skip normally; FAIL in a gate run (HCD_GATE_RUN=1), so a gate can not pass without the real-member checks."""
    if os.environ.get("HCD_GATE_RUN") == "1":
        pytest.fail(msg + " (HCD_GATE_RUN=1: gate runs require the real production checkpoints)")
    pytest.skip(msg)


def needs_real():
    """Called first in each real-member test (a call, not a fixture, so a gate run reports FAILED, not ERROR): the
    real members (gitignored binaries) in HCD_CHECKPOINTS_DIR. The red battery above still pins the contract."""
    if not _real_files_present():
        _unavailable(f"real production checkpoints not present at {REAL_CKPT_DIR}: set HCD_CHECKPOINTS_DIR to the "
                     f"directory holding {BASE}0..4 (.eqx, .norm.pkl, .meta.json)")


def test_real_checkpoints_verify_against_committed_manifest():
    needs_real()
    manifest, prefixes = PE.verify_manifest(checkpoints_dir=str(REAL_CKPT_DIR),
                                            manifest_path=str(COMMITTED_MANIFEST))
    assert [os.path.basename(p) for p in prefixes] == \
        [f"{BASE}{i}" for i in range(5)]
    # the five real normalizers are byte-identical (single sha) and the manifest says so
    assert manifest["normalizers_byte_identical"] is True
    assert len({m["norm_sha256"] for m in manifest["members"]}) == 1


def test_real_load_production_ensemble():
    needs_real()
    ens, meta0, norm0, manifest = PE.load_production_ensemble(
        checkpoints_dir=str(REAL_CKPT_DIR), manifest_path=str(COMMITTED_MANIFEST))
    assert len(ens.members) == 5
    assert "P_filt" in norm0 and "arch_cfg" in meta0
    assert manifest["n_members"] == 5


def test_gen_check_green_on_real_dir():
    needs_real()
    rc = GEN.main(["--check", "--checkpoints-dir", str(REAL_CKPT_DIR),
                   "--manifest", str(COMMITTED_MANIFEST)])
    assert rc == 0


# --------------------------------------------------------------------------------------------- #
#  documentation agreement: README digest table and analysis.lock member list vs the manifest
# --------------------------------------------------------------------------------------------- #
def test_readme_digest_table_matches_manifest():
    assert COMMITTED_MANIFEST.exists(), "committed manifest missing"
    assert README.exists(), "checkpoints/README_production_ensemble.md missing"
    manifest = json.loads(COMMITTED_MANIFEST.read_text())
    text = README.read_text()
    for mem in manifest["members"]:
        assert mem["eqx_sha256"] in text, \
            f"README digest table drifted: {mem['name']} eqx sha absent"
        assert mem["norm_sha256"] in text, \
            f"README digest table drifted: {mem['name']} norm sha absent"
        assert mem["meta_sha256"] in text, \
            f"README digest table drifted: {mem['name']} meta sha absent"


# PI ruling 2026-10-21 (D2 = a): the schema-2 manifest is canonical; the July analysis.lock and the schema-1 manifest
# (byte-preserved as production_ensemble_manifest_v1_pre2026-10.json) are historical records. The lock certifies the
# historical ensemble only, and current code must refuse what it pins. When analysis.lock is regenerated (PI
# authorization required) these three tests are replaced by the agreement check of the new lock.
V1_MANIFEST = ROOT / "checkpoints" / "production_ensemble_manifest_v1_pre2026-10.json"


def _july_lock_emulator():
    if not ANALYSIS_LOCK.exists() and not (ROOT / ".git").exists():
        pytest.skip("analysis.lock is export-ignored from source archives; run this test from a git checkout")
    return json.loads(ANALYSIS_LOCK.read_text())["emulator"]


def test_july_lock_is_the_historical_v1_record():
    """analysis.lock pins the byte-preserved schema-1 manifest (by sha256) and names its five members."""
    emu = _july_lock_emulator()
    v1 = json.loads(V1_MANIFEST.read_text())
    assert emu["manifest_schema_version"] == 1 == v1["schema_version"]
    assert emu["manifest_sha256"] == hashlib.sha256(V1_MANIFEST.read_bytes()).hexdigest()
    assert emu["ensemble_members"] == [m["name"] for m in v1["members"]]
    assert emu["n_members"] == v1["n_members"]


def test_july_lock_does_not_certify_the_current_manifest():
    """The lock names neither the canonical manifest's bytes nor any of its members: it says nothing about the
    current model."""
    emu = _july_lock_emulator()
    current = json.loads(COMMITTED_MANIFEST.read_text())
    assert current["schema_version"] == PE.MANIFEST_SCHEMA_VERSION == 2
    assert emu["manifest_schema_version"] != current["schema_version"]
    assert emu["manifest_sha256"] != hashlib.sha256(COMMITTED_MANIFEST.read_bytes()).hexdigest()
    assert not set(emu["ensemble_members"]) & {m["name"] for m in current["members"]}


def test_current_loader_refuses_the_manifest_the_july_lock_pins():
    """Current code rejects the artifact the July lock certifies, loudly and before touching any file."""
    with pytest.raises(PE.ProductionEnsembleError, match=r"schema_version=1, .*superseded at gate E"):
        PE.verify_manifest(manifest_path=str(V1_MANIFEST))


# --------------------------------------------------------------------------------------------- #
#  schema v2 (gate E, PU-0067): members are schema-2.0 checkpoints of one training cache; the
#  manifest names its directory; schema-1 (pre-2026-10) manifests are refused
# --------------------------------------------------------------------------------------------- #
def test_v2_manifest_names_its_directory_and_the_cache(fake, monkeypatch):
    """A recorded directory beats HCD_CHECKPOINTS_DIR (set here to an absent directory)."""
    d, mpath = fake
    m = json.loads(Path(mpath).read_text())
    assert m["schema_version"] == 2 and m["checkpoints_dir"] == str(d) and m["cache_sha256"] == FAKE_CACHE
    monkeypatch.setenv("HCD_CHECKPOINTS_DIR", str(d / "absent"))
    _, prefixes = PE.verify_manifest(manifest_path=str(mpath))          # directory taken from the manifest
    assert prefixes == [os.path.join(str(d), f"{BASE}{i}") for i in range(5)]


def test_argument_beats_a_recorded_directory(fake, monkeypatch):
    """Loader order, first rule: an explicit checkpoints_dir wins over the manifest's directory and the environment."""
    d, mpath = fake
    _edit_manifest(mpath, lambda m: m.update(checkpoints_dir=str(d / "absent_recorded")))
    monkeypatch.setenv("HCD_CHECKPOINTS_DIR", str(d / "absent_env"))
    _, prefixes = PE.verify_manifest(checkpoints_dir=str(d), manifest_path=str(mpath))
    assert prefixes == [os.path.join(str(d), f"{BASE}{i}") for i in range(5)]


def test_null_manifest_without_the_environment_uses_its_own_directory(fake, monkeypatch, tmp_path):
    """Loader order, last rule: no argument, checkpoints_dir null, HCD_CHECKPOINTS_DIR unset -> the manifest's own
    directory (not the working directory)."""
    d, mpath = fake
    _edit_manifest(mpath, lambda m: m.update(checkpoints_dir=None))
    monkeypatch.delenv("HCD_CHECKPOINTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    _, prefixes = PE.verify_manifest(manifest_path=str(mpath))
    assert prefixes == [os.path.join(str(d), f"{BASE}{i}") for i in range(5)]


def test_v2_manifest_without_a_directory_is_located_by_the_environment(fake, monkeypatch):
    """A manifest whose checkpoints_dir is null (the committed form) takes HCD_CHECKPOINTS_DIR; an explicit argument wins."""
    d, mpath = fake
    _edit_manifest(mpath, lambda m: m.update(checkpoints_dir=None))
    monkeypatch.setenv("HCD_CHECKPOINTS_DIR", str(d))
    _, prefixes = PE.verify_manifest(manifest_path=str(mpath))
    assert prefixes == [os.path.join(str(d), f"{BASE}{i}") for i in range(5)]
    monkeypatch.setenv("HCD_CHECKPOINTS_DIR", str(d / "absent"))
    assert PE.verify_manifest(checkpoints_dir=str(d), manifest_path=str(mpath))[1] == prefixes
    with pytest.raises(PE.ProductionEnsembleError, match="MISSING"):
        PE.verify_manifest(manifest_path=str(mpath))


def test_generator_writes_no_directory_unless_asked(tmp_path):
    d = tmp_path / "checkpoints"
    d.mkdir()
    _make_fake_members(d)
    mpath = tmp_path / PE.MANIFEST_BASENAME
    assert GEN.main(["--checkpoints-dir", str(d), "--manifest", str(mpath)]) == 0
    assert json.loads(mpath.read_text())["checkpoints_dir"] is None


def test_committed_manifest_names_no_directory():
    assert json.loads(COMMITTED_MANIFEST.read_text())["checkpoints_dir"] is None


def test_v2_schema1_manifest_refused(fake):
    _, mpath = fake
    _edit_manifest(mpath, lambda m: m.update(schema_version=1))
    with pytest.raises(PE.ProductionEnsembleError, match="schema_version"):
        _verify(fake)


@pytest.mark.parametrize("field,value,msg", [("schema_version", "1.0", "schema"),
                                             ("cache_sha256", "ee" * 32, "cache")])
def test_v2_member_meta_must_be_schema2_on_the_manifest_cache(tmp_path, field, value, msg):
    d = tmp_path / "checkpoints"
    d.mkdir()
    _make_fake_members(d)
    meta = _meta(3)
    meta[field] = value
    (d / f"{BASE}3.meta.json").write_text(json.dumps(meta) + "\n")
    mpath = d / PE.MANIFEST_BASENAME
    with pytest.raises(PE.ProductionEnsembleError, match=msg):
        GEN.build_manifest(str(d))


def test_v2_sha256sums_cross_check(fake):
    d, mpath = fake
    lines = [f"{PE.sha256_file(d / f'{BASE}{i}.eqx')}  {BASE}{i}.eqx" for i in range(5)]
    lines[2] = "00" * 32 + f"  {BASE}2.eqx"
    (d / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    with pytest.raises(PE.ProductionEnsembleError, match="SHA256SUMS"):
        _verify(fake)


# --------------------------------------------------------------------------------------------- #
#  the comparison config pins the committed manifest by sha256 (data-free; scrub delta review m2)
# --------------------------------------------------------------------------------------------- #
def _c1_pins():
    from hcd_analysis.comparison import cases as CS
    return CS.load_config("phase0_v1")["comparators"]["C1"]


def test_phase0_pin_is_the_committed_manifest():
    """phase0_v1.json's C1 entry names the committed manifest and pins its exact bytes: a regenerated manifest must
    land together with the new pin."""
    c1 = _c1_pins()
    assert ROOT / c1["manifest"] == COMMITTED_MANIFEST
    assert c1["manifest_sha256"] == hashlib.sha256(COMMITTED_MANIFEST.read_bytes()).hexdigest()


def test_adapter_refuses_a_manifest_that_is_not_the_pinned_one(tmp_path, monkeypatch):
    """hcd_priya.predict checks the manifest's sha256 against the pin before it loads anything."""
    from hcd_analysis.comparison.adapters import hcd_priya as H
    monkeypatch.setenv("HCD_CHECKPOINTS_DIR", str(tmp_path / "absent"))
    edited = tmp_path / PE.MANIFEST_BASENAME
    edited.write_bytes(COMMITTED_MANIFEST.read_bytes() + b"\n")          # same content, other bytes
    pins = _c1_pins()
    with pytest.raises(RuntimeError, match="does not match the pinned sha256"):
        H.predict([], cache_path=str(tmp_path / "absent.h5"), pins=pins, manifest_path=str(edited))
    # control: pinning the edited bytes passes the check, and the loader then stops at the absent members
    pins = dict(pins, manifest_sha256=hashlib.sha256(edited.read_bytes()).hexdigest())
    with pytest.raises(PE.ProductionEnsembleError, match="MISSING"):
        H.predict([], cache_path=str(tmp_path / "absent.h5"), pins=pins, manifest_path=str(edited))
