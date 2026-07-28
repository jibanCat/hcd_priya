"""A1c PRE-LAUNCH FIXES (2026-07-28), from the unanimous NO-GO of the mandated pre-launch panel.

Three defects, all in PROVENANCE and READOUT, none in the mock protocol itself (the panel
verified truth source, kwarg-identical propagation, truth recording, RNG neutrality, default
byte-identity and signatures as correct):

  1. `truth_site_semantics` was stamped from the STATIC, flag-unaware tuple
     `closure_legb.SELF_DRAWN_EXTRA_SITES`, so an A1c pkl -- whose six repaired truths are
     finite and drawn -- recorded all six under `not_self_drawn` with a note asserting
     "f_res truth = prior center 0; metal f/k node truth at/below the flat-log support edge".
     A1's own pkl carries a BYTE-IDENTICAL block, so the defective and corrected arms were
     indistinguishable on their own provenance while their `sites_extra` truths differ
     (nan vs drawn). The certification arm of record would have shipped 48 pkls attesting to
     the defect they exist to repair.

  2. The MANDATORY pre-readout population banner never printed `fres_selfdraw`, so the
     pre-registration's binding check (section 5c) was unperformable, and the superseded
     metals-only build read identically to the arm of record.

  3. The readout computed NO pull and NO rank for either repaired sector (`f_res` was excluded
     outright by a `f_Si`/`k_Si` filter), so the statistic demonstrating the repair would have
     been chosen AFTER the gate result was known.

The default (flags-off) path must stay byte-identical throughout: the preservation and
resumability argument for the 96 A1 pkls rests on it.

Run: /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_a1c_prelaunch_fixes.py -q
"""
import importlib
import sys

import numpy as np
import pytest

sys.path.insert(0, "/home/mfho/hcd_priya")

CL = importlib.import_module("hcd_analysis.emulator.closure_legb")
runner = importlib.import_module("scripts.run_prod_sbc_shard")

METAL = ["f_SiIII_eBOSS_z0", "f_SiIII_eBOSS_z1", "k_SiIII_eBOSS_z0", "k_SiIII_eBOSS_z1"]
FRES = ["f_res_amp", "f_res_slope"]
CORE = ["tau0_amp", "dtau0", "s_lls", "s_subdla", "s_dla"]
PRESENT = CORE + METAL + FRES

# the EXACT string the historical (flags-off) stamp carried; byte-identity is the contract
HISTORICAL_NOTE = ("not_self_drawn: f_res truth = prior center 0; metal f/k node truth at/below "
                   "the flat-log support edge (no center exists)")


def _cfg(**kw):
    return dict(runner.RUN_CFG_DEFAULTS, survey="eBOSS", **kw)


# --------------------------------------------------------------------------- 1. semantics stamp

def test_default_path_stamp_is_byte_identical_to_the_historical_one():
    """THE PRESERVATION CONTRACT. With both flags off the stamp must be character-for-character
    what the 96 preserved A1 pkls carry -- set membership AND the note string."""
    s = runner._truth_site_semantics(PRESENT, _cfg())
    assert s["self_draw"] == sorted(CORE)
    assert s["not_self_drawn"] == sorted(METAL + FRES)
    assert s["note"] == HISTORICAL_NOTE


def test_metal_selfdraw_moves_the_four_metal_nodes_into_self_draw():
    s = runner._truth_site_semantics(PRESENT, _cfg(metal_selfdraw=True))
    assert s["self_draw"] == sorted(CORE + METAL)
    assert s["not_self_drawn"] == sorted(FRES)
    assert "flat-log support edge" not in s["note"], "must not still assert the metal defect"
    assert "f_res truth = prior center 0" in s["note"], "f_res IS still pinned in this config"


def test_fres_selfdraw_moves_the_two_fres_sites_into_self_draw():
    s = runner._truth_site_semantics(PRESENT, _cfg(fres_selfdraw=True))
    assert s["self_draw"] == sorted(CORE + FRES)
    assert s["not_self_drawn"] == sorted(METAL)
    assert "f_res truth = prior center 0" not in s["note"]
    assert "flat-log support edge" in s["note"], "metal IS still pinned in this config"


def test_both_flags_leave_nothing_pinned_and_say_so():
    """THE A1c CONFIGURATION. Nothing may remain in not_self_drawn, and the note must not
    assert either defect."""
    s = runner._truth_site_semantics(PRESENT, _cfg(metal_selfdraw=True, fres_selfdraw=True))
    assert s["self_draw"] == sorted(PRESENT)
    assert s["not_self_drawn"] == []
    assert "prior center 0" not in s["note"]
    assert "flat-log support edge" not in s["note"]
    assert "DRAWN" in s["note"]


def test_stamp_never_invents_sites_that_are_absent():
    """Only sites actually present in sites_extra may be reported, or the stamp would claim
    truths that do not exist (e.g. a leg with no metal nodes)."""
    s = runner._truth_site_semantics(CORE + FRES, _cfg(metal_selfdraw=True, fres_selfdraw=True))
    assert s["self_draw"] == sorted(CORE + FRES)
    assert s["not_self_drawn"] == []


def test_stamp_reads_the_static_tuple_as_its_floor():
    """The runner must still defer to closure_legb.SELF_DRAWN_EXTRA_SITES for the core sector;
    the flags only ADD to it. Guards against the wrong fix (mutating the module tuple), which
    would mis-stamp default-path pkls and break byte-identity."""
    assert set(CORE) <= set(CL.SELF_DRAWN_EXTRA_SITES)
    s = runner._truth_site_semantics(PRESENT, _cfg())
    assert set(s["self_draw"]) == set(PRESENT) & set(CL.SELF_DRAWN_EXTRA_SITES)


def test_a_selfdraw_run_can_never_stamp_the_defect_it_repairs():
    """The regression in one line: the A1c stamp must differ from the A1 stamp."""
    a1 = runner._truth_site_semantics(PRESENT, _cfg())
    a1c = runner._truth_site_semantics(PRESENT, _cfg(metal_selfdraw=True, fres_selfdraw=True))
    assert a1 != a1c
    assert a1["not_self_drawn"] and not a1c["not_self_drawn"]


# --------------------------------------------------------------------------- 2. readout banner

def test_population_banner_prints_fres_selfdraw():
    """Pre-registration 5c makes confirming fres_selfdraw=True a BINDING pre-readout check.
    It cannot be satisfied if the analyzer never prints it."""
    src = open("/home/mfho/hcd_priya/scripts/analyze_sbc_perleg.py").read()
    banner = src.split("[population]", 1)[1].split("\n\n", 1)[0]
    for field in ("metal_selfdraw", "fres_selfdraw", "diag_no_sample_metals",
                  "metal_prior", "prior_sig"):
        assert field in banner, f"population banner must print {field}"


# --------------------------------------------------------------------------- 3. repaired sectors

def _sites_extra(truths, n_draws=200, seed=0):
    """Draws are ALWAYS finite; only the truth may be nan. That is exactly the A1 shape: the fit
    sampled those sites perfectly well, it was the mock TRUTH that was never drawn."""
    rng = np.random.default_rng(seed)
    out = {}
    for nm, t in truths.items():
        centre = t if np.isfinite(t) else 0.005
        out[nm] = {"draws": rng.normal(centre, abs(centre) * 0.1 + 1e-3, n_draws), "truth": t}
    return out


_ANALYZER_NS = None


def _analyzer():
    """`analyze_sbc_perleg.py` is a SCRIPT with no `__main__` guard, and since the 2026-07-27
    footgun fix its positional ROOT is mandatory, so importing it raises SystemExit. Rather than
    restructure a launch-critical script hours before a production array, load only its
    definitions: imports, the two repaired-site constants, and the function bodies. Everything
    under test here is pure."""
    global _ANALYZER_NS
    if _ANALYZER_NS is not None:
        return _ANALYZER_NS
    import ast
    path = "/home/mfho/hcd_priya/scripts/analyze_sbc_perleg.py"
    tree = ast.parse(open(path).read(), filename=path)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id.startswith("REPAIRED_SITE")
                for t in node.targets):
            keep.append(node)
    ns = {"__name__": "analyze_sbc_perleg_defs"}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    _ANALYZER_NS = type("NS", (), ns)
    return _ANALYZER_NS


def test_repaired_sector_summary_covers_metal_AND_fres():
    """The f_Si/k_Si filter excluded f_res entirely; both sectors were repaired, so both must
    be scored."""
    an = _analyzer()
    truths = {nm: 0.01 for nm in METAL}
    truths.update({"f_res_amp": 0.005, "f_res_slope": 0.3})
    acc = {}
    for m in range(12):
        an._accumulate_repaired_sites(acc, _sites_extra(truths, seed=m))
    out = an.repaired_sector_summary(acc)
    assert set(out["sites"]) == set(METAL + FRES)
    for nm in METAL + FRES:
        rec = out["sites"][nm]
        assert np.isfinite(rec["pull_mean"]) and np.isfinite(rec["pull_sd"])
        assert np.isfinite(rec["rank_ks_p"])
        assert rec["n"] == 12


def test_repaired_sector_summary_is_not_a_gate():
    """PI #9: adding a gate would be a gate-definition change. The summary must carry no
    pass/fail verdict for these sites."""
    an = _analyzer()
    acc = {}
    for m in range(8):
        an._accumulate_repaired_sites(acc, _sites_extra({"f_res_amp": 0.005}, seed=m))
    out = an.repaired_sector_summary(acc)
    blob = repr(out).lower()
    assert "pass" not in blob and "fail" not in blob


def test_repaired_sector_summary_skips_pinned_nan_truths():
    """On A1 (and on any flags-off arm) the truths are nan; the summary must report the sector
    as unscoreable rather than emit nan pulls that look like measurements."""
    an = _analyzer()
    acc = {}
    for m in range(8):
        an._accumulate_repaired_sites(acc, _sites_extra({"f_res_amp": np.nan}, seed=m))
    out = an.repaired_sector_summary(acc)
    assert out["sites"]["f_res_amp"]["truth_present"] is False
    assert not np.isfinite(out["sites"]["f_res_amp"]["pull_mean"])


def test_summary_is_none_when_no_repaired_sites_present():
    an = _analyzer()
    assert an.repaired_sector_summary({}) is None


# ------------------------------------------------------- 4. the corrected path's DEFINING property

def _fnsrc(name):
    """The AST of a named closure_legb function."""
    import ast
    import inspect
    import textwrap
    return ast.parse(textwrap.dedent(inspect.getsource(getattr(CL, name))))


def test_selfdraw_extractors_consume_no_randomness():
    """THE LOAD-BEARING ASSUMPTION OF THE ENTIRE PAIRED READOUT.

    The pre-registered NO-REPEAT test is a PAIRED t on the 48 common mocks, valid only because
    A1c mock m and A1 mock m share a bit-identical cosmology/tau0/HCD truth. That holds because
    both extractors READ values already present in the prior trace rather than drawing new ones,
    so neither consumes RNG and neither shifts the stream.

    Verified STRUCTURALLY rather than by example: an AST scan proves the property for every
    input, where a numeric spot-check would only cover the inputs tried. Any future edit that
    introduces a sample/PRNG call into these functions fails here."""
    import ast
    banned_attr = {"sample", "normal", "uniform", "split", "PRNGKey", "fold_in", "randint",
                   "choice", "permutation", "gamma", "beta", "exponential"}
    for fn in ("_selfdraw_metal_nodes", "_selfdraw_fres_bres"):
        tree = _fnsrc(fn)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                nm = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                assert nm not in banned_attr, f"{fn} must not consume randomness (found {nm}())"
                if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                    assert f.value.id not in ("numpyro", "random"), \
                        f"{fn} must not call {f.value.id}.{f.attr}"


def test_fres_selfdraw_builds_bres_with_the_DEPLOYED_builder():
    """The mock's b_res(z) curve must be the fit's curve BY CONSTRUCTION, not a re-derivation.
    The fit builds it as _bres_of_z(z_global, amp, slope) (closure_legb, _legb_model); the
    self-draw must produce a bit-identical array from the same raw values."""
    from types import SimpleNamespace
    zg = np.array([2.2, 2.6, 3.0, 3.4, 3.8, 4.2])
    raw = {"f_res_amp": 0.0123, "f_res_slope": 0.456}
    ctx = SimpleNamespace(fres_selfdraw_truth=True, sample_res=True, z_global=zg)
    got = CL._selfdraw_fres_bres(raw, ctx)
    want = np.asarray(CL._bres_of_z(zg, raw["f_res_amp"], raw["f_res_slope"]), float)
    assert np.array_equal(got, want), "self-draw b_res must equal the deployed builder exactly"


def test_fres_selfdraw_is_off_by_default():
    """Default OFF must return None, i.e. the historical call shape with no b_res kwarg."""
    from types import SimpleNamespace
    ctx = SimpleNamespace(fres_selfdraw_truth=False, sample_res=True,
                          z_global=np.array([2.2, 3.0]))
    assert CL._selfdraw_fres_bres({"f_res_amp": 0.01, "f_res_slope": 0.4}, ctx) is None
    # and it must also refuse where the fit does not float f_res at all (would be a no-op truth)
    off = SimpleNamespace(fres_selfdraw_truth=True, sample_res=False,
                          z_global=np.array([2.2, 3.0]))
    assert CL._selfdraw_fres_bres({"f_res_amp": 0.01, "f_res_slope": 0.4}, off) is None


def test_mock_and_fit_slice_bres_with_the_SAME_nearest_z_map():
    """The mock (make_leg_a_legmock) and the fit (_data_loglik_legcore) each slice the global
    b_res curve down to a leg's z grid. If those two nearest-z maps ever diverge, the mock would
    carry a resolution distortion the fit does not model -- a phantom systematic of exactly the
    class this campaign has been burned by. Pin that both spell it identically."""
    import inspect
    import re
    mock_src = inspect.getsource(CL.make_leg_a_legmock)
    fit_src = inspect.getsource(CL._data_loglik_legcore)
    mock_sel = [s for s in mock_src.splitlines() if "sel" in s and "argmin" in s]
    fit_sel = [s for s in fit_src.splitlines() if "sel" in s and "argmin" in s]
    assert mock_sel and fit_sel, "both paths must build an explicit nearest-z sel"

    def index_expr(line):
        """The INDEX comprehension, stripped of the surrounding array constructor. The mock
        wraps it in jnp.asarray and the fit in np.array; that difference is immaterial (both
        materialise the same integer indices), so compare the comprehension itself."""
        s = re.sub(r"\s+", "", line).split("=", 1)[1]
        m = re.search(r"\[(int\(np\.argmin.*)\]", s)
        assert m, f"unrecognised nearest-z construction: {line.strip()}"
        return m.group(1)

    assert index_expr(mock_sel[0]) == index_expr(fit_sel[0]), (
        f"nearest-z maps diverged:\n  mock: {mock_sel[0].strip()}\n  fit : {fit_sel[0].strip()}")
    # and the fit must slice the global curve with exactly that map
    assert "b_res_global[jnp.asarray(sel)]" in re.sub(r"\s+", "", fit_src)
