"""W4 -- the analysis.lock GENERATOR (scripts/gen_analysis_lock.py) + lock-vs-live verification.

FREEZE-STEP CONTRACT (the reason several tests below are shaped the way they are):
  * Regeneration of the committed analysis.lock happens ONLY at the freeze cut, AFTER the
    independent review panel, on the MERGED tree (--i-am-the-freeze-step). Until then the
    committed lock stays the stale June one ON PURPOSE (the KS prior reparameterization is
    in flight), so `--check` is EXPECTED to exit 1 today -- loudly and informatively -- and
    the default suite must stay green while saying so.
  * The generator is the ONLY writer of analysis.lock; hand-edits are prohibited.
  * blind.lock is OUT OF SCOPE: a read-only input that must never be written by anyone.

Leg summaries are the expensive step (a full production ctx build per survey), so the default
tests feed the generator a RECORDED fixture (tests/golden/analysis_lock_legs_fixture.json,
produced by the generator itself via --emit-leg-summary at the W4 base). Because the fixture
embeds the forward/prior signatures of ITS build tree, fixture-driven generation passes
allow_stale_legs=True: these tests exercise the MACHINERY and stay green when in-flight work
moves the live signatures. The opt-in real-leg test (HCD_LOCK_REAL_LEGS=1) builds a leg live
and is strict. At the freeze step the fixture is REBUILT from the merged tree.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=<tree> JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_analysis_lock_generator.py -q
"""
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import pytest

# THIS tree (worktree-safe): test the generator that lives next to this test file.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_PATH = os.path.join(REPO, "scripts", "gen_analysis_lock.py")
FIXTURE = os.path.join(REPO, "tests", "golden", "analysis_lock_legs_fixture.json")
BLIND_LOCK = os.path.join(REPO, "blind.lock")
COMMITTED_LOCK = os.path.join(REPO, "analysis.lock")
MANIFEST = os.path.join(REPO, "checkpoints", "production_ensemble_manifest.json")

REQUIRED_SECTIONS = (
    "analysis", "blinding", "covariance", "created_utc", "decisions", "emulator",
    "git_commit", "legs", "nuts", "outputs", "prior_constants", "priors", "provenance",
    "signatures", "surveys", "uncovered_constants",
)

# The retired June-lock poison values (must never re-enter a generated lock as prior centers).
POISON_LLS_ALLZ_MEDIAN = 0.2908612702546296
POISON_RATIO_ZSLOPE = [0.95, 0.15, 0.4]


def _load_gen():
    spec = importlib.util.spec_from_file_location("gen_analysis_lock", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def GEN():
    return _load_gen()


@pytest.fixture(scope="module")
def leg_summaries():
    with open(FIXTURE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def lock(GEN, leg_summaries):
    """One generated lock shared by the read-only assertions below."""
    return GEN.generate_lock(leg_summaries, allow_stale_legs=True)


def _run_script(args, timeout=900):
    """Run the generator script as a subprocess (the CLI surface under test)."""
    env = dict(os.environ)
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("JAX_PLATFORMS", "cpu")
    env["PYTHONPATH"] = REPO
    return subprocess.run([sys.executable, GEN_PATH] + args, capture_output=True,
                          text=True, timeout=timeout, env=env, cwd=REPO)


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ------------------------------------------------------------------------------------------- #
# (a) every REQUIRED section is emitted
# ------------------------------------------------------------------------------------------- #
def test_required_sections_present(lock):
    missing = [s for s in REQUIRED_SECTIONS if s not in lock]
    assert not missing, f"generated lock is missing required sections: {missing}"
    # legs x3 with the exact leg set
    assert set(lock["legs"]) == {"DESI", "KS", "eBOSS"}
    for name, leg in lock["legs"].items():
        for f in ("name", "z", "k_min", "k_max", "n_rows", "n_z", "dla_forward_frac",
                  "metals_on", "mf_floor_on", "forward", "prior"):
            assert f in leg, f"legs.{name} missing field {f!r}"
        # everything forward_stamp carries must ride along per leg
        for f in ("res_corr_on", "fix_alpha_res", "sample_res", "f_res_amp_sigma",
                  "metal_prior", "metal_node_z", "dla_cov_reduced", "dla_forward_frac",
                  "use_snr3", "cv_floor_on", "cv_floor_rank1", "alpha_mode",
                  "forward_signature", "hcd_prior_signature"):
            assert f in leg["forward"], f"legs.{name}.forward missing stamp field {f!r}"
    # both signatures, hex sha256
    for k in ("forward_signature", "hcd_prior_signature"):
        assert isinstance(lock["signatures"][k], str) and len(lock["signatures"][k]) == 64
    # prior_constants is the WHOLE live payload (spot-check the derivation provenance keys)
    for k in ("HCD_LIT_DNDX_LAW", "HCD_LLS_SURVEY_BOOST", "adopted_kernel",
              "derivation_json_sha256"):
        assert k in lock["prior_constants"]
    # uncovered-constants registry resolved (the forward_signature 'NOT covered' list)
    vals = lock["uncovered_constants"]["values"]
    for k in ("closure_legb.HCD_INCIDENCE_SLOPE", "closure_legb.ZSLOPE_PRIOR_SIGMA",
              "closure_legb.SIGMA_A0", "closure_legb.SIGMA_S",
              "closure_legb.F_RES_AMP_SIGMA", "closure_legb.F_RES_SLOPE_SIGMA",
              "closure_legb.PROD_FORWARD_BY_LEG", "closure_legb.PROD_RES_CORR_ON",
              "data_likelihood.DESI_DLA_COV_REDUCE",
              "closure_legb.build_legb_ctx.KS_NORC_KMAX_CAP",
              "closure_legb.build_legb_ctx.mf_anchor_mult_default"):
        assert k in vals, f"uncovered_constants.values missing {k!r}"
    # nuts block pins the deployed sampler settings
    for k in ("n_chains", "n_warmup", "n_samples", "max_tree_depth", "target_accept",
              "dense_mass", "init", "seed_derivation"):
        assert k in lock["nuts"], f"nuts missing {k!r}"
    assert "TODO_AT_FREEZE" not in json.dumps(lock["nuts"]), \
        f"nuts block has unresolved TODOs at this base: {lock['nuts']}"
    # decisions: PI records + ks_zlo
    assert lock["decisions"]["pi_decision_records"], "no PI decision records"
    for rec in lock["decisions"]["pi_decision_records"]:
        assert rec["path"].startswith("/home/mfho/hcd_priya_notes/")
        assert len(rec["sha256"]) == 64, f"unresolved decision record: {rec}"
    assert lock["decisions"]["ks_zlo"]["baseline"] == 2.4
    assert lock["decisions"]["ks_zlo"]["diagnostic"] == 2.8


# ------------------------------------------------------------------------------------------- #
# (b) self-consistency: two consecutive generations identical modulo created_utc
# ------------------------------------------------------------------------------------------- #
def test_two_generations_identical_modulo_created_utc(GEN, leg_summaries, tmp_path):
    a = GEN.generate_lock(leg_summaries, allow_stale_legs=True)
    b = GEN.generate_lock(leg_summaries, allow_stale_legs=True)
    a2, b2 = copy.deepcopy(a), copy.deepcopy(b)
    a2.pop("created_utc"), b2.pop("created_utc")
    assert GEN.canonical_dumps(a2) == GEN.canonical_dumps(b2), \
        "two consecutive generations differ beyond created_utc (nondeterminism)"
    # and the on-disk serialization round-trips (the file IS the lock)
    p = tmp_path / "analysis.lock.candidate.json"
    p.write_text(GEN.canonical_dumps(a))
    assert json.loads(p.read_text()) == a


# ------------------------------------------------------------------------------------------- #
# (c) blinding block == blind.lock fields; blind.lock untouched by generation
# ------------------------------------------------------------------------------------------- #
def test_blinding_verbatim_and_blind_lock_untouched(GEN, leg_summaries):
    before_sha = _sha(BLIND_LOCK)
    before_mtime = os.stat(BLIND_LOCK).st_mtime_ns
    lock = GEN.generate_lock(leg_summaries, allow_stale_legs=True)
    assert _sha(BLIND_LOCK) == before_sha, "generation MODIFIED blind.lock (forbidden)"
    assert os.stat(BLIND_LOCK).st_mtime_ns == before_mtime, \
        "generation touched blind.lock's mtime (forbidden -- read-only input)"
    with open(BLIND_LOCK) as f:
        bl = json.load(f)
    b = lock["blinding"]
    # VERBATIM copies; a prior_sigma desync between the two locks = blinding bug.
    assert b["params"] == bl["blind_params"]
    assert b["offset_sigma_multiple"] == bl["offset_sigma_multiple"]
    assert b["prior_sigma"] == bl["prior_sigma"]
    assert b["blind_lock_git_commit"] == bl["git_commit"]
    assert b["blind_lock_sha256"] == before_sha
    # seed_str stays REFERENCE ONLY -- never duplicated into analysis.lock
    assert bl["seed_str"] not in json.dumps(lock)


# ------------------------------------------------------------------------------------------- #
# (d) emulator block == the committed manifest (all TEN eqx+norm digests + count)
# ------------------------------------------------------------------------------------------- #
def test_emulator_block_matches_committed_manifest(lock):
    with open(MANIFEST) as f:
        man = json.load(f)
    emu = lock["emulator"]
    assert emu["n_members"] == man["n_members"] == 5
    assert emu["manifest_schema_version"] == man["schema_version"]
    assert emu["normalizers_byte_identical"] == man["normalizers_byte_identical"]
    assert emu["members"] == man["members"], "embedded members differ from the manifest"
    assert emu["ensemble_members"] == [m["name"] for m in man["members"]]
    assert emu["manifest_sha256"] == _sha(MANIFEST)
    # ALL TEN digests: 5 eqx + 5 norm (meta digests ride along as well)
    digests = [m[k] for m in emu["members"] for k in ("eqx_sha256", "norm_sha256")]
    assert len(digests) == 10 and all(len(d) == 64 for d in digests)
    assert all(len(m["meta_sha256"]) == 64 for m in emu["members"])
    # the 5 eqx digests are distinct; the 5 norm digests are byte-identical (per manifest)
    assert len({m["eqx_sha256"] for m in emu["members"]}) == 5
    assert len({m["norm_sha256"] for m in emu["members"]}) == 1


# ------------------------------------------------------------------------------------------- #
# (e) poisoned-value tripwires (regression against carrying the June lock forward)
# ------------------------------------------------------------------------------------------- #
def test_no_retired_values_in_generated_lock(lock):
    # priors.zslope.mu must NOT be the retired wrong-object ratio slope
    assert lock["priors"]["zslope"]["mu"] != POISON_RATIO_ZSLOPE, \
        "priors.zslope.mu carries the RETIRED lit/sim ratio slope (hcd-dndx-zslope-bug)"
    # no alpha prior center anywhere may sit at the retired all-z-median centre (x boost)
    boosts = [1.0] + [float(v) for v in
                      lock["prior_constants"]["HCD_LLS_SURVEY_BOOST"].values()]

    def walk(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{path}.{k}" if path else k
                if k == "alpha_hcd_mu" and isinstance(v, list) and v:
                    for b in boosts:
                        assert abs(v[0] - POISON_LLS_ALLZ_MEDIAN * b) > 1e-4 * max(
                            POISON_LLS_ALLZ_MEDIAN * b, 1e-30), (
                            f"{p}[0]={v[0]} is the RETIRED all-z-median LLS centre x boost "
                            f"{b} (CENTER-construction bug)")
                walk(v, p)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")

    walk(lock)
    # the deployed centers are the corrected ones (sanity, not just not-poisoned):
    assert lock["priors"]["zslope"]["mu"][0] > 1.5, \
        "z-slope LLS center regressed below the incidence-slope floor"


def test_generator_tripwire_raises_on_poisoned_lock(GEN, lock):
    """assert_not_poisoned (the generation-time defense) must FIRE on a poisoned lock."""
    bad = copy.deepcopy(lock)
    bad["priors"]["zslope"]["mu"] = list(POISON_RATIO_ZSLOPE)
    with pytest.raises(AssertionError, match="ratio"):
        GEN.assert_not_poisoned(bad)
    bad2 = copy.deepcopy(lock)
    bad2["legs"]["DESI"]["prior"]["alpha_hcd_mu"][0] = POISON_LLS_ALLZ_MEDIAN
    with pytest.raises(AssertionError, match="all-z-median"):
        GEN.assert_not_poisoned(bad2)
    # and the clean lock passes (it was generated through this tripwire already)
    GEN.assert_not_poisoned(lock)


def test_check_is_self_consistent_against_fresh_lock(GEN, leg_summaries, tmp_path):
    """The eventual-green path: a freshly generated lock file, compared against a fresh live
    generation, must verify CLEAN (exit 0 / n_diffs 0, volatile fields ignored). This is the
    exact comparison the freeze step relies on -- proven here on a tmp file so the committed
    stale lock stays untouched."""
    fresh = tmp_path / "analysis.lock.fresh.json"
    fresh.write_text(GEN.canonical_dumps(
        GEN.generate_lock(leg_summaries, allow_stale_legs=True)))
    rc, lines = GEN.check_lock(leg_summaries, allow_stale_legs=True,
                               committed_path=str(fresh))
    assert rc == 0, "check_lock is not self-consistent against a fresh generation:\n" + \
        "\n".join(lines)
    assert any("OK" in ln for ln in lines)


# ------------------------------------------------------------------------------------------- #
# (f) --check against the committed stale lock exits 1 TODAY with a readable diff
# ------------------------------------------------------------------------------------------- #
def test_check_mode_reports_stale_committed_lock():
    """TODAY the committed analysis.lock is the stale June one (freeze gated on the review
    panel + the in-flight KS reparameterization), so --check MUST exit 1 and name the drifted
    fields, including the known poisoned rows. EVENTUAL-GREEN EXPECTATION: at the freeze cut,
    after --i-am-the-freeze-step regenerates the lock on the merged tree, this same command
    exits 0 and the skipped test below flips to enforced."""
    r = _run_script(["--check", "--legs-cache", FIXTURE, "--allow-stale-legs-cache"])
    assert r.returncode == 1, (
        f"--check must exit 1 against the stale June lock (got {r.returncode}); "
        f"stdout tail: {r.stdout[-800:]} stderr tail: {r.stderr[-400:]}")
    out = r.stdout
    assert "STALE" in out
    assert "priors.zslope.mu" in out, "diff does not name the drifted zslope prior"
    assert "POISONED" in out, "diff does not call out the known poisoned rows"
    assert "committed" in out and "live" in out, "diff is not a readable two-sided report"


@pytest.mark.skip(reason=(
    "FREEZE-STEP CONTRACT: the committed analysis.lock is EXPECTED to be the stale June one "
    "until the freeze cut (regeneration is gated on the independent review panel, the merged "
    "tree, and PI sign-off; the KS prior reparameterization is in flight). On-demand status: "
    "scripts/gen_analysis_lock.py --check (exit 1 today, by design). AT THE FREEZE STEP: "
    "regenerate via --i-am-the-freeze-step on the merged tree, then UNSKIP this test -- it "
    "then enforces committed == live forever after."))
def test_committed_lock_matches_live_AT_FREEZE():
    r = _run_script(["--check"])
    assert r.returncode == 0, f"committed analysis.lock is stale vs live:\n{r.stdout}"


# ------------------------------------------------------------------------------------------- #
# (g) the refuse-to-overwrite guard
# ------------------------------------------------------------------------------------------- #
def test_refuses_to_overwrite_committed_lock_without_freeze_flag():
    before = _sha(COMMITTED_LOCK)
    r = _run_script(["--legs-cache", FIXTURE, "--allow-stale-legs-cache",
                     "--out", COMMITTED_LOCK])
    assert r.returncode != 0, "writing the committed analysis.lock without the freeze flag " \
                              "must be REFUSED"
    assert "REFUSED" in (r.stdout + r.stderr)
    assert _sha(COMMITTED_LOCK) == before, "the committed lock was modified by a refused run"


def test_refuses_blind_lock_target_even_with_freeze_flag(tmp_path):
    before = _sha(BLIND_LOCK)
    for extra in ([], ["--i-am-the-freeze-step"]):
        r = _run_script(["--legs-cache", FIXTURE, "--allow-stale-legs-cache",
                         "--out", BLIND_LOCK] + extra)
        assert r.returncode != 0, "blind.lock as --out must ALWAYS be refused"
        assert "blind.lock" in (r.stdout + r.stderr)
    # any path NAMED blind.lock is refused too (there is no legitimate writer)
    r = _run_script(["--legs-cache", FIXTURE, "--allow-stale-legs-cache",
                     "--out", str(tmp_path / "blind.lock")])
    assert r.returncode != 0
    assert _sha(BLIND_LOCK) == before


def test_writes_normal_out_path_and_is_deterministic(GEN, tmp_path):
    out1 = tmp_path / "lock1.json"
    out2 = tmp_path / "lock2.json"
    for out in (out1, out2):
        r = _run_script(["--legs-cache", FIXTURE, "--allow-stale-legs-cache",
                         "--out", str(out)])
        assert r.returncode == 0, f"generation failed: {r.stdout[-500:]} {r.stderr[-500:]}"
        assert out.exists()
    a, b = json.loads(out1.read_text()), json.loads(out2.read_text())
    for volatile in ("created_utc",):
        a.pop(volatile), b.pop(volatile)
    assert a == b, "two CLI generations differ beyond created_utc"


# ------------------------------------------------------------------------------------------- #
# stale-legs-cache guard (the freeze step must never consume a drifted fixture silently)
# ------------------------------------------------------------------------------------------- #
def test_stale_legs_cache_is_refused_by_default(GEN, leg_summaries):
    stale = copy.deepcopy(leg_summaries)
    for sv in stale:
        stale[sv]["forward"]["forward_signature"] = "0" * 64
    with pytest.raises(SystemExit, match="STALE LEGS CACHE"):
        GEN.generate_lock(stale, allow_stale_legs=False)


# ------------------------------------------------------------------------------------------- #
# opt-in REAL leg build (slow; a full production ctx build). HCD_LOCK_REAL_LEGS=1 enables.
# ------------------------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.environ.get("HCD_LOCK_REAL_LEGS"),
                    reason="opt-in slow test: set HCD_LOCK_REAL_LEGS=1 to build a real leg "
                           "(full production ctx build, several minutes)")
def test_real_leg_build_matches_fixture_schema(GEN, leg_summaries):
    live = GEN.collect_leg_summaries(in_process=False)   # subprocess per survey, strict
    # the fixture and a live build must agree on the summary SCHEMA per survey; at the same
    # tree state they must agree on the VALUES too (this is the fixture-refresh check the
    # freeze step runs).
    for sv in ("desi", "eboss", "ks"):
        assert set(live[sv]) == set(leg_summaries[sv]), f"leg summary schema drift on {sv}"
    lock = GEN.generate_lock(live, allow_stale_legs=False)   # strict: signatures must be live
    GEN.assert_not_poisoned(lock)
