"""scripts/tau_threshold_sensitivity.py: the pure parts (class collapse, templates, comparison, summary, reading rule)
against the emulator cache loader and hand-made cases, and the command line's own settings record and refusals.
Synthetic arrays only; the run itself needs fake_spectra."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from hcd_analysis import run_settings as RS

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "tau_threshold_sensitivity.py"
_spec = importlib.util.spec_from_file_location("tau_threshold_sensitivity", _PATH)
TS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TS)


def test_the_class_ranges_and_the_collapse_are_the_emulator_loaders():
    from hcd_analysis.emulator import data as D
    assert tuple((s.start, s.stop) for s in D.COARSE_SLICES) == TS.COARSE_RANGES
    assert tuple(D.COARSE_NAMES) == TS.CLASSES
    rng = np.random.default_rng(1)
    p15 = rng.uniform(0.5, 2.0, size=(15, 9))
    counts = rng.integers(0, 50, size=15)
    counts[8:13] = 0                                              # an empty class stays 0, as in the loader
    np.testing.assert_array_equal(TS.collapse(p15, counts), D._collapse_p1d(p15[None], counts[None])[0])


def test_templates_are_the_forward_models_choice_of_pieces():
    pf = np.array([[2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]])
    pu = pf * 1.5
    t = TS.templates(pf, pu)
    np.testing.assert_allclose(t["LLS"], [0.5, 0.5])
    np.testing.assert_allclose(t["subDLA"], [1.0, 1.0])
    np.testing.assert_allclose(t["DLA"], [2.75, 2.75])               # unfiltered DLA over filtered clean


def test_compare_separates_an_amplitude_change_from_a_shape_change():
    ref = np.array([1.0, 0.5, 0.25, 0.1])
    assert TS.compare(ref, ref) == (1.0, 0.0)
    a, s = TS.compare(1.2 * ref, ref)
    assert a == pytest.approx(1.2) and s == pytest.approx(0.0, abs=1e-15)
    a, s = TS.compare(0.5 * ref, ref)
    assert a == pytest.approx(0.5)
    t = ref + np.array([0.0, 0.0, 0.0, 0.1])
    a, s = TS.compare(t, ref)
    a_exp = float(np.sum(t * ref) / np.sum(ref * ref))
    s_exp = float(np.sqrt(np.mean((t - a_exp * ref) ** 2)) / np.sqrt(np.mean(ref ** 2)))   # rms, relative to rms(ref)
    assert (a, s) == (pytest.approx(a_exp), pytest.approx(s_exp))
    assert np.isnan(TS.compare(ref, np.zeros(4))[0])


def test_lowk_change_is_relative_to_the_templates_largest_value_there():
    kf = np.array([0.001, 0.003, 0.005, 0.02])
    ref = np.array([0.1, 0.0004, -0.05, 9.0])                         # crosses zero; the k = 0.02 value is excluded
    t = ref + np.array([0.0, 0.0004, 0.0, 5.0])
    assert TS.lowk_change(t, ref, kf, 0.01) == pytest.approx(0.0004 / 0.1)


def test_class_counts_use_the_production_edges():
    assert TS.class_counts([17.19, 17.2, 18.99, 19.0, 20.29, 20.3, 21.0]) == {"LLS": 2, "subDLA": 2, "DLA": 2}


K = 8


def _result(lls=1.0, sub=1.0, dla=1.0, shape=None, clean=None, n_lls=100, kf=None):
    """Synthetic fine-bin results: clean power 1, class excesses 0.2/0.5/1.0 with a falling shape, scaled per class;
    ``shape`` adds to the LLS excess, ``clean`` multiplies the clean power."""
    base = np.linspace(1.0, 0.2, K)
    p15 = np.ones((15, K))
    p15[0] = clean if clean is not None else 1.0
    excess = 0.2 * lls * base + (shape if shape is not None else 0.0)
    p15[1:8] = p15[0] * (1.0 + excess)
    p15[8:13] = p15[0] * (1.0 + 0.5 * sub * base)
    pu = p15.copy()
    pu[13:15] = p15[0] * (1.0 + 1.0 * dla * base)
    p15[13:15] = p15[0] * 1.3
    return {"kf": np.linspace(0.001, 0.02, K) if kf is None else kf, "pf15": p15, "pu15": pu,
            "counts15": np.full(15, 10), "abs_counts": {"LLS": n_lls, "subDLA": 50, "DLA": 20}}


def _summary(**rows):
    res = {("z", 100.0): _result()}
    res.update({("z", float(k[1:])): v for k, v in rows.items()})
    return TS.summarize(res, 100.0, n_k=K)


def test_summary_rows_carry_the_numbers_the_rule_reads():
    s = _summary(T50=_result(lls=1.02, n_lls=103), T200=_result(clean=np.full(K, 1.002)))
    r = s["z|T50"]
    assert r["threshold"] == 50.0 and r["tag"] == "z" and r["n_k"] == K
    assert r["LLS"]["count_ratio"] == pytest.approx(1.03)
    assert r["LLS"]["amplitude_ratio"] == pytest.approx(1.02, rel=1e-12)
    assert r["sightlines"] == {"clean": 10, "LLS": 70, "subDLA": 50, "DLA": 20}
    assert s["z|T200"]["clean_max_frac_change"] == pytest.approx(0.002)
    assert s["z|T100"]["LLS"]["shape_residual"] == 0.0


def test_a_change_confined_to_the_first_stored_k_bin_registers():
    """The stored arrays start at the fundamental mode (k = 0 is never stored), so bin 0 must be compared."""
    shape = np.zeros(K)
    shape[0] = 0.05
    r = _summary(T50=_result(shape=shape))["z|T50"]
    assert r["LLS"]["shape_residual"] > 0.01 and r["LLS"]["lowk_change"] > 0.1
    clean = np.ones(K)
    clean[0] = 1.02
    assert _summary(T50=_result(clean=clean))["z|T50"]["clean_max_frac_change"] == pytest.approx(0.02)


@pytest.mark.parametrize("case, reading", [
    ({"lls": 1.04}, "negligible"),
    ({"lls": 0.96}, "negligible"),
    ({"lls": 1.20}, "modest"),
    ({"lls": 0.80}, "modest"),                                 # a decrease is read by |a - 1|
    ({"lls": 1.40}, "important"),
    ({"lls": 0.60}, "important"),
    ({"clean": np.full(K, 1.02)}, "modest"),                   # templates unchanged, clean changed by 2%
    ({"clean": np.full(K, 0.98)}, "modest"),                   # a clean decrease counts too
    ({"shape": 0.012 * np.cos(np.linspace(0, 3, K))}, "modest"),
    ({"shape": 0.05 * np.cos(np.linspace(0, 3, K))}, "important"),
])
def test_the_reading_rule(case, reading):
    s = _summary(T50=_result(**case), T200=_result())
    assert TS.verdict(s, {50.0, 200.0}, 100.0)["reading"] == reading


def test_the_rule_boundaries():
    assert TS._reading(0.0499, 0.0499, 0.0099) == "negligible"
    assert TS._reading(0.05, 0.0, 0.0) == "modest"
    assert TS._reading(0.0, 0.05, 0.0) == "modest"
    assert TS._reading(0.0, 0.0, 0.01) == "modest"
    assert TS._reading(0.29, 0.10, 0.5) == "modest"
    assert TS._reading(0.2901, 0.0, 0.0) == "important"
    assert TS._reading(0.0, 0.1001, 0.0) == "important"


def test_per_class_readings_use_a_and_s_of_that_class_only():
    shape = 0.05 * np.cos(np.linspace(0, 3, K))
    v = TS.verdict(_summary(T50=_result(sub=0.7, clean=np.full(K, 1.02)), T200=_result(shape=shape)),
                   {50.0, 200.0}, 100.0)
    assert v["per_class"]["subDLA"]["reading"] == "important"          # a decrease of 30%
    assert v["per_class"]["LLS"]["reading"] == "important"             # shape only
    assert v["per_class"]["DLA"]["reading"] == "negligible"            # the clean change is not a class's


def test_lowk_change_cut_and_normalisation():
    kf = np.array([0.0005, 0.002, 0.006, 0.01, 0.02])
    ref = np.array([9.0, 0.2, -0.3, 50.0, 50.0])
    t = np.array([0.0, 0.1, -0.3, 0.0, 0.0])
    # 1e-3 <= k < 0.01: the first bin (below the data range) and k = 0.01 are excluded; normalised by max |t_ref|
    assert TS.lowk_change(t, ref, kf, 0.01, 1e-3) == pytest.approx(0.1 / 0.3)
    assert TS.lowk_change(t, ref, kf, 0.01) == pytest.approx(9.0 / 9.0)     # kmin 0 includes the first bin
    assert np.isnan(TS.lowk_change(t, ref, kf, 0.001, 0.0006))              # no bin in the range


@pytest.mark.parametrize("n", [5, 6, 9, 12, 13, 1003, 20000, 40000, 691200])
def test_the_self_check_blocks_always_end_with_a_shorter_block(n):
    b = TS._self_check_block(n)
    assert 1 <= b < n and n % b != 0                                       # the last block is shorter
    if n >= 100:
        assert n // b in (3, 4)                                            # and there are about four blocks


def test_per_class_readings_differ_and_only_headline_rows_count():
    s = _summary(T50=_result(sub=1.2), T200=_result(), T30=_result(lls=2.0))
    v = TS.verdict(s, {50.0, 200.0}, 100.0)
    assert v["reading"] == "modest"
    assert {c: v["per_class"][c]["reading"] for c in ("LLS", "subDLA", "DLA")} == \
        {"LLS": "negligible", "subDLA": "modest", "DLA": "negligible"}
    assert TS.verdict(s, {30.0}, 100.0)["reading"] == "important"
    with pytest.raises(ValueError, match="no summary rows"):
        TS.verdict(s, {75.0}, 100.0)


def test_the_summary_needs_the_reference_threshold():
    with pytest.raises(KeyError, match="no reference"):
        TS.summarize({("z", 50.0): _result()}, 100.0, n_k=K)


# ------------------------------------------------------------------------------------------------ the command line
def _main_env(tmp_path, monkeypatch, calls):
    """main() with a tiny synthetic input, a JSON config (valid YAML), PyYAML stubbed if absent and the heavy steps
    replaced by recorders."""
    import h5py
    src = tmp_path / "in.hdf5"
    with h5py.File(src, "w") as f:
        g = f.create_group("Header")
        for k, v in dict(redshift=3.0, nbins=8).items():
            g.attrs[k] = v
        f.create_dataset("tau/H/1/1215", data=np.zeros((4, 8), dtype=np.float32))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(json.dumps({"absorber": {"tau_threshold": 100.0, "fast_mode": True, "min_log_nhi": 17.2}}))
    if "yaml" not in sys.modules:
        try:
            import yaml  # noqa: F401
        except ImportError:
            stub = types.ModuleType("yaml")
            stub.safe_load = lambda b: json.loads(b)
            monkeypatch.setitem(sys.modules, "yaml", stub)
    monkeypatch.setattr(TS, "run_input", lambda tag, *a, **k: calls.append(("run", tag)))
    monkeypatch.setattr(TS, "self_check", lambda tag, *a, **k: calls.append(("check", tag)))
    return ["--sim", "synth", "--input", f"a={src}", "--input", f"b={src}", "--alpha", "1.0", "--config", str(cfg)]


def test_the_command_line_records_its_settings_and_refuses_a_rerun_with_others(tmp_path, monkeypatch):
    calls = []
    args = _main_env(tmp_path, monkeypatch, calls)
    out = tmp_path / "out"
    assert TS.main(args + ["--out", str(out)]) == 0                 # no results yet: the summary is deferred
    assert calls == [("run", "a"), ("run", "b")]
    rec = json.loads((out / RS.RUN_SETTINGS_FILE).read_text())["settings"]
    assert rec["thresholds"] == [30.0, 50.0, 100.0, 200.0, 300.0] and set(rec["inputs"]) == {"a", "b"}
    assert "chunk" not in rec and "headline" not in rec
    before = sorted((p.name, p.stat().st_mtime_ns) for p in out.iterdir())
    with pytest.raises(RS.RunSettingsMismatch, match="thresholds"):
        TS.main(args + ["--out", str(out), "--thresholds", "50", "100", "200"])
    assert sorted((p.name, p.stat().st_mtime_ns) for p in out.iterdir()) == before
    TS.main(args + ["--out", str(out), "--chunk", "7", "--headline", "50", "--self-check", "2"])   # not compared


@pytest.mark.parametrize("extra, why", [(["--only", "c"], "not among the inputs"),
                                        (["--thresholds", "100", "33.3333333"], "6 significant digits"),
                                        (["--thresholds", "100", "-5", "50", "200"], "positive"),
                                        (["--thresholds", "100", "inf", "50", "200"], "positive"),
                                        (["--headline", "75"], "must be among"),
                                        (["--reference", "75"], "must be among"),
                                        (["--anchor-cache", "c.h5", "--n-skewers", "2"], "cannot be combined"),
                                        (["--input", "a=x"], "unique TAG=PATH")])
def test_bad_arguments_are_refused_before_anything_is_created(tmp_path, monkeypatch, capsys, extra, why):
    args = _main_env(tmp_path, monkeypatch, [])
    with pytest.raises(SystemExit):
        TS.main(args + ["--out", str(tmp_path / "out")] + extra)
    assert why in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def _check(passed=True, **over):
    doc = {"catalogue_identical": True, "counts_identical": True, "max_rel_diff_filtered": 1e-15,
           "max_rel_diff_unfiltered": 1e-15, "tau_eff_rel_diff": 1e-15, "scale_solved_rel_diff": 1e-12}
    if not passed:
        doc["counts_identical"] = False
    doc.update(over)
    doc["passed"] = TS.self_check_passed(doc)
    return doc


@pytest.mark.parametrize("over, ok", [
    ({}, True), ({"catalogue_identical": False}, False), ({"counts_identical": False}, False),
    ({"max_rel_diff_filtered": 1e-9}, False), ({"max_rel_diff_unfiltered": float("nan")}, False),
    ({"tau_eff_rel_diff": 2e-10}, False), ({"scale_solved_rel_diff": 2e-4}, False),
    ({"scale_solved_rel_diff": 9e-5}, True), ({"tau_eff_rel_diff": None}, False)])
def test_the_self_check_criterion(over, ok):
    doc = _check(**over)
    assert TS.self_check_passed(doc) is ok
    doc.pop("max_rel_diff_filtered")
    assert TS.self_check_passed(doc) is False                          # a missing entry fails


def _write_results(out, tags=("a", "b"), thrs=(50, 100, 200), **over):
    for tag in tags:
        for thr in thrs:
            r = _result(**over.get((tag, thr), {}))
            np.savez(out / f"result_{tag}_T{thr}.npz", kf=r["kf"], pf15=r["pf15"], pu15=r["pu15"],
                     counts15=r["counts15"], log_nhi=np.array([17.5, 19.5, 20.5]), scale=np.array(0.8),
                     target_F=np.array(0.7), tau_eff=np.array(0.4), z_grid=np.array(3.0))


def test_a_failed_or_missing_self_check_makes_the_reading_invalid(tmp_path, monkeypatch):
    args = _main_env(tmp_path, monkeypatch, []) + ["--thresholds", "50", "100", "200"]
    out = tmp_path / "out"
    TS.main(args + ["--out", str(out)])
    _write_results(out)
    (out / "self_check_a.json").write_text(json.dumps(dict(_check(), threshold=100.0)))
    (out / "self_check_b.json").write_text(json.dumps(dict(_check(passed=False), threshold=100.0)))
    assert TS.main(args + ["--out", str(out), "--self-check", "4"]) == 2
    doc = json.loads((out / "summary.json").read_text())
    assert doc["verdict"]["reading"].startswith("invalid") and "'b'" in doc["verdict"]["reading"]
    assert doc["verdict"]["valid"] is False
    assert {c["reading"] for c in doc["verdict"]["per_class"].values()} == {"invalid"}
    (out / "self_check_b.json").unlink()                              # missing, and this call has no --self-check
    assert TS.main(args + ["--out", str(out)]) == 2
    assert "'b'" in json.loads((out / "summary.json").read_text())["verdict"]["reading"]
    (out / "self_check_b.json").write_text(json.dumps(dict(_check(), threshold=100.0)))
    assert TS.main(args + ["--out", str(out)]) == 0
    doc = json.loads((out / "summary.json").read_text())
    assert doc["verdict"]["reading"] == "negligible" and doc["verdict"]["valid"] is True
    assert doc["summary_settings"]["n_k"] == 172 and doc["summary_settings"]["lowk_kmax"] == 0.01


def test_the_summary_is_strict_json_when_a_ratio_is_undefined(tmp_path, monkeypatch):
    args = _main_env(tmp_path, monkeypatch, []) + ["--thresholds", "50", "100", "200"]
    out = tmp_path / "out"
    TS.main(args + ["--out", str(out)])
    _write_results(out)
    for tag in ("a", "b"):                                            # the reference has no LLS absorber
        f = out / f"result_{tag}_T100.npz"
        d = dict(np.load(f))
        d["log_nhi"] = np.array([19.5, 20.5])
        f.unlink()
        np.savez(f, **d)
    TS.main(args + ["--out", str(out)])

    def no_constant(c):
        raise ValueError(c)
    doc = json.loads((out / "summary.json").read_text(), parse_constant=no_constant)
    assert doc["rows"]["a|T50"]["LLS"]["count_ratio"] is None


@pytest.mark.parametrize("key, value", [("scale", np.array(0.81)), ("target_F", np.array(0.71)),
                                        ("tau_eff", np.array(0.41)), ("z_grid", np.array(3.2)),
                                        ("kf", np.linspace(0.001, 0.03, K))])
def test_results_of_one_input_from_different_passes_are_refused(tmp_path, key, value):
    _write_results(tmp_path, tags=("a",))
    TS.load_results(tmp_path, ["a"], [50.0, 100.0, 200.0])
    d = dict(np.load(tmp_path / "result_a_T200.npz"))
    d[key] = value
    np.savez(tmp_path / "result_a_T200.npz", **d)
    with pytest.raises(ValueError, match=f"disagree on {key}"):
        TS.load_results(tmp_path, ["a"], [50.0, 100.0, 200.0])


def test_results_are_written_once(tmp_path):
    p = tmp_path / "r.npz"
    TS._save_once(p, x=np.arange(3))
    before = p.read_bytes()
    with pytest.raises(FileExistsError):
        TS._save_once(p, x=np.arange(4))
    assert p.read_bytes() == before and [f.name for f in tmp_path.iterdir()] == ["r.npz"]
    assert p.stat().st_mode & 0o777 == 0o644


def test_the_blocked_mean_optical_depth_and_scale_with_an_uneven_last_block(monkeypatch):
    tau = np.random.default_rng(3).exponential(0.5, size=(103, 16))
    monkeypatch.setattr(TS, "_read_rows", lambda path, s, e: tau[s:e].copy())
    monkeypatch.setattr(TS, "_filter_rows", lambda t, tau_eff: t)
    whole = float(-np.log(np.mean(np.exp(-tau))))
    assert TS._global_tau_eff("x", 103, 25) == pytest.approx(whole, rel=1e-13)     # blocks 25 x 4 + 3
    s_val = TS._solve_scale("x", 103, 25, whole, 0.55)
    assert float(np.mean(np.exp(-s_val * tau))) == pytest.approx(0.55, rel=1e-12)


def test_the_anchor_compares_every_stored_bin_with_the_matching_cache_row(tmp_path):
    import h5py
    r = _result()
    r["z_grid"], r["scale"] = np.array(3.0), np.array(0.8)
    cache = tmp_path / "cache.h5"
    with h5py.File(cache, "w") as h:
        h["sim_name"] = np.array([b"other", b"synth"])
        h["z_grid"], h["alpha_slope"], h["scale"] = [3.0, 3.0], [1.0, 1.0], [0.5, 0.8]
        h["tier_c_counts"] = np.stack([r["counts15"], r["counts15"]])
        h["P_tier_c_filtered"] = np.stack([r["pf15"], r["pf15"]])
        pu = r["pu15"].copy()
        pu[3, 0] *= 1.001                                         # differs only in the first stored k bin
        h["P_tier_c"] = np.stack([r["pu15"], pu])
    out = TS.anchor({("z", 100.0): r, ("z", 50.0): _result(lls=3.0)}, cache, "synth", 1.0, 100.0, n_k=K)
    assert set(out) == {"z"}                                         # only the reference threshold is anchored
    out = out["z"]
    assert out["cache_row"] == 1 and out["count_diff_by_bin"] == [0] * 15 and out["scale_rel_diff"] == 0.0
    assert out["max_rel_diff_filtered"] == 0.0
    assert out["max_rel_diff_unfiltered"] == pytest.approx(abs(1 / 1.001 - 1))          # mine / cache - 1, bin 0


def test_the_anchor_needs_exactly_one_row_of_that_simulation_z_and_alpha(tmp_path):
    import h5py
    r = _result()
    r["z_grid"], r["scale"] = np.array(3.0), np.array(0.8)
    for name, alphas in (("two_rows", [1.0, 1.0]), ("other_alpha", [1.1, 1.2])):
        cache = tmp_path / f"{name}.h5"
        with h5py.File(cache, "w") as h:
            h["sim_name"] = np.array([b"synth", b"synth"])
            h["z_grid"], h["alpha_slope"], h["scale"] = [3.0, 3.0], alphas, [0.8, 0.8]
            h["tier_c_counts"] = np.stack([r["counts15"]] * 2)
            h["P_tier_c_filtered"] = np.stack([r["pf15"]] * 2)
            h["P_tier_c"] = np.stack([r["pu15"]] * 2)
        out = TS.anchor({("z", 100.0): r}, cache, "synth", 1.0, 100.0, n_k=K)["z"]
        assert out == {"error": f"{2 if name == 'two_rows' else 0} matching cache rows"}


def _replace_arg(args, flag, value):
    out = list(args)
    out[out.index(flag) + 1] = value
    return out


@pytest.mark.parametrize("change, key", [
    (lambda a: _replace_arg(a, "--alpha", "1.1"), "alpha_slope"),
    (lambda a: _replace_arg(a, "--sim", "other"), "sim"),
    (lambda a: a + ["--n-skewers", "2"], "n_skewers"),
])
def test_the_command_line_refuses_each_result_setting(tmp_path, monkeypatch, change, key):
    args = _main_env(tmp_path, monkeypatch, [])
    out = tmp_path / "out"
    TS.main(args + ["--out", str(out)])
    rec = json.loads((out / RS.RUN_SETTINGS_FILE).read_text())["settings"]
    assert set(rec) == {"tool", "sim", "inputs", "thresholds", "alpha_slope", "absorber_except_threshold", "config",
                        "anchor_cache", "n_skewers", "scale"}
    with pytest.raises(RS.RunSettingsMismatch, match=key):
        TS.main(change(args) + ["--out", str(out)])


def test_the_command_line_refuses_another_config_or_a_rewritten_input(tmp_path, monkeypatch):
    import os
    args = _main_env(tmp_path, monkeypatch, [])
    out = tmp_path / "out"
    TS.main(args + ["--out", str(out)])
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(json.dumps({"absorber": {"tau_threshold": 100.0, "fast_mode": True, "min_log_nhi": 17.5}}))
    with pytest.raises(RS.RunSettingsMismatch, match="absorber_except_threshold.min_log_nhi"):
        TS.main(args + ["--out", str(out)])
    cfg.write_text(json.dumps({"absorber": {"tau_threshold": 100.0, "fast_mode": True, "min_log_nhi": 17.2}}))
    src = tmp_path / "in.hdf5"
    st = src.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    with pytest.raises(RS.RunSettingsMismatch, match="mtime_ns"):
        TS.main(args + ["--out", str(out)])


def test_the_command_line_refuses_a_rewritten_anchor_cache(tmp_path, monkeypatch):
    import os
    args = _main_env(tmp_path, monkeypatch, [])
    cache = tmp_path / "cache.h5"
    cache.write_bytes(b"not read here")
    out = tmp_path / "out"
    TS.main(args + ["--out", str(out), "--anchor-cache", str(cache)])
    rec = json.loads((out / RS.RUN_SETTINGS_FILE).read_text())["settings"]["anchor_cache"]
    assert rec["bytes"] == 13 and rec["path"] == str(cache.resolve())
    st = cache.stat()
    os.utime(cache, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    with pytest.raises(RS.RunSettingsMismatch, match="anchor_cache.mtime_ns"):
        TS.main(args + ["--out", str(out), "--anchor-cache", str(cache)])


def test_asking_for_a_self_check_with_none_written_is_invalid(tmp_path, monkeypatch):
    args = _main_env(tmp_path, monkeypatch, []) + ["--thresholds", "50", "100", "200"]
    out = tmp_path / "out"
    TS.main(args + ["--out", str(out)])
    _write_results(out)
    assert TS.main(args + ["--out", str(out)]) == 0                           # no self-check asked for, none present
    assert TS.main(args + ["--out", str(out), "--self-check", "4"]) == 2      # asked for, none written
    doc = json.loads((out / "summary.json").read_text())
    assert doc["verdict"]["reading"] == "invalid: self-check failed or missing for ['a', 'b']"
    assert doc["summary_settings"]["lowk_kmin"] == 1e-3


def test_code_dirty_is_null_when_git_cannot_tell(monkeypatch):
    monkeypatch.setattr(TS, "_git", lambda *a: None)
    assert TS._dirty() is None


@pytest.mark.parametrize("order", ["undefined_first", "undefined_last"])
def test_an_undefined_comparison_makes_the_reading_invalid_whatever_the_order(order):
    good = _summary(T50=_result(), T200=_result())
    bad = dict(good["z|T50"])
    bad.update(tag="a" if order == "undefined_first" else "zz",
               LLS=dict(good["z|T50"]["LLS"], amplitude_ratio=float("nan")))
    rows = {"x|T50": bad, **good} if order == "undefined_first" else {**good, "x|T50": bad}
    v = TS.verdict(rows, {50.0, 200.0}, 100.0)
    assert v["reading"].startswith("invalid: undefined comparison")
    assert {c["reading"] for c in v["per_class"].values()} == {"invalid"}


def test_a_self_check_at_another_threshold_does_not_count(tmp_path, monkeypatch):
    args = _main_env(tmp_path, monkeypatch, []) + ["--thresholds", "50", "100", "200"]
    out = tmp_path / "out"
    TS.main(args + ["--out", str(out)])
    _write_results(out)
    (out / "self_check_a.json").write_text(json.dumps(dict(_check(), threshold=100.0)))
    (out / "self_check_b.json").write_text(json.dumps(dict(_check(), threshold=200.0)))
    assert TS.main(args + ["--out", str(out)]) == 2
    assert "'b'" in json.loads((out / "summary.json").read_text())["verdict"]["reading"]


@pytest.mark.parametrize("extra, why", [(["--self-check", "-1"], "--self-check must be"),
                                        (["--self-check", "5"], "exceeds the skewers"),
                                        (["--chunk", "0"], "--chunk must be positive"),
                                        (["--n-k", "0"], "--n-k must be positive"),
                                        (["--n-skewers", "0"], "--n-skewers positive")])
def test_sizes_are_validated_before_anything_is_created(tmp_path, monkeypatch, capsys, extra, why):
    args = _main_env(tmp_path, monkeypatch, [])
    with pytest.raises(SystemExit):
        TS.main(args + ["--out", str(tmp_path / "out")] + extra)
    assert why in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_the_one_pass_check_tolerates_summation_order_only(tmp_path):
    _write_results(tmp_path, tags=("a",))
    for rel, ok in ((1e-15, True), (1e-9, False)):
        d = dict(np.load(tmp_path / "result_a_T200.npz"))
        d["tau_eff"] = np.array(0.4 * (1 + rel))
        (tmp_path / "result_a_T200.npz").unlink()
        np.savez(tmp_path / "result_a_T200.npz", **d)
        if ok:
            TS.load_results(tmp_path, ["a"], [50.0, 100.0, 200.0])
        else:
            with pytest.raises(ValueError, match="disagree on tau_eff"):
                TS.load_results(tmp_path, ["a"], [50.0, 100.0, 200.0])


def test_a_class_without_sightlines_is_an_undefined_comparison():
    empty = _result()
    empty["counts15"] = empty["counts15"].copy()
    empty["counts15"][13:15] = 0                                       # no DLA sightline at this threshold
    s = _summary(T50=empty, T200=_result())
    assert np.isnan(s["z|T50"]["DLA"]["amplitude_ratio"]) and np.isnan(s["z|T50"]["DLA"]["lowk_change"])
    v = TS.verdict(s, {50.0, 200.0}, 100.0)
    assert v["reading"].startswith("invalid: undefined comparison") and v["valid"] is False
    ref = _result()
    ref["counts15"] = ref["counts15"].copy()
    ref["counts15"][13:15] = 0                                         # empty at the reference only
    res = {("z", 100.0): ref, ("z", 50.0): _result(), ("z", 200.0): _result()}
    s = TS.summarize(res, 100.0, n_k=K)
    assert all(np.isnan(s[k]["DLA"]["amplitude_ratio"]) and np.isnan(s[k]["DLA"]["lowk_change"])
               for k in ("z|T50", "z|T200"))
    v = TS.verdict(s, {50.0, 200.0}, 100.0)
    assert v["reading"].startswith("invalid: undefined comparison") and v["valid"] is False


def test_the_low_k_maximum_is_undefined_if_any_row_is():
    s = _summary(T50=_result(), T200=_result())
    s["z|T200"]["LLS"]["lowk_change"] = float("nan")                    # a finite row first, a NaN one later
    assert np.isnan(TS.verdict(s, {50.0, 200.0}, 100.0)["per_class"]["LLS"]["max_lowk_change"])


def test_the_command_line_uses_the_data_range_low_k_window(tmp_path, monkeypatch):
    kf = np.array([0.0004, 0.002, 0.004, 0.006, 0.008, 0.02, 0.03, 0.04])
    args = _main_env(tmp_path, monkeypatch, []) + ["--thresholds", "50", "100", "200"]
    out = tmp_path / "out"
    TS.main(args + ["--out", str(out)])
    for tag in ("a", "b"):
        for thr in (50, 100, 200):
            r = _result(kf=kf)
            if thr == 50:
                r["pf15"] = r["pf15"].copy()
                r["pf15"][1:8, 0] *= 1.5                               # a change only in the bin below 1e-3
            np.savez(out / f"result_{tag}_T{thr}.npz", kf=kf, pf15=r["pf15"], pu15=r["pu15"], counts15=r["counts15"],
                     log_nhi=np.array([17.5, 19.5, 20.5]), scale=np.array(0.8), target_F=np.array(0.7),
                     tau_eff=np.array(0.4), z_grid=np.array(3.0))
    TS.main(args + ["--out", str(out)])
    rows = json.loads((out / "summary.json").read_text())["rows"]
    assert rows["a|T50"]["LLS"]["lowk_change"] == pytest.approx(0.0, abs=1e-15)
    TS.main(args + ["--out", str(out), "--lowk-kmin", "0"])
    assert json.loads((out / "summary.json").read_text())["rows"]["a|T50"]["LLS"]["lowk_change"] > 0.1


def test_an_empty_clean_set_at_the_reference_is_invalid_without_warnings():
    import warnings
    ref = _result()
    ref["counts15"] = ref["counts15"].copy()
    ref["pf15"] = ref["pf15"].copy()
    ref["counts15"][0] = 0
    ref["pf15"][0] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = TS.summarize({("z", 100.0): ref, ("z", 50.0): _result(), ("z", 200.0): _result()}, 100.0, n_k=K)
    assert TS.verdict(s, {50.0, 200.0}, 100.0)["valid"] is False


def test_an_undefined_comparison_exits_2_and_is_marked_invalid(tmp_path, monkeypatch):
    args = _main_env(tmp_path, monkeypatch, []) + ["--thresholds", "50", "100", "200"]
    out = tmp_path / "out"
    TS.main(args + ["--out", str(out)])
    _write_results(out)
    f = out / "result_a_T50.npz"
    d = dict(np.load(f))
    d["counts15"] = d["counts15"].copy()
    d["counts15"][0] = 0                                               # no clean sightline
    f.unlink()
    np.savez(f, **d)
    assert TS.main(args + ["--out", str(out)]) == 2
    doc = json.loads((out / "summary.json").read_text())
    assert doc["verdict"]["valid"] is False and doc["verdict"]["reading"].startswith("invalid: undefined")
    (out / "self_check_a.json").write_text(json.dumps(dict(_check(passed=False), threshold=100.0)))
    assert TS.main(args + ["--out", str(out)]) == 2                    # both reasons are reported
    reading = json.loads((out / "summary.json").read_text())["verdict"]["reading"]
    assert "undefined comparison" in reading and "self-check failed or missing" in reading


def test_existing_results_and_self_checks_are_reused_not_recomputed(tmp_path):
    _write_results(tmp_path, tags=("a",))
    TS.run_input("a", tmp_path / "no_such_input.hdf5", [50.0, 100.0, 200.0], 1.0, {}, "synth", tmp_path,
                 lambda m: None)                                       # nothing to do: the input is never opened
    doc = dict(_check(), threshold=100.0)
    (tmp_path / "self_check_a.json").write_text(json.dumps(doc))
    before = (tmp_path / "self_check_a.json").stat().st_mtime_ns
    got = TS.self_check("a", tmp_path / "no_such_input.hdf5", 10, 100.0, 1.0, {}, "synth", tmp_path, lambda m: None)
    assert got == doc and (tmp_path / "self_check_a.json").stat().st_mtime_ns == before


def test_the_low_k_window_is_validated(tmp_path, monkeypatch, capsys):
    args = _main_env(tmp_path, monkeypatch, [])
    with pytest.raises(SystemExit):
        TS.main(args + ["--out", str(tmp_path / "out"), "--lowk-kmin", "0.02"])
    assert "kmin < kmax" in capsys.readouterr().err and not (tmp_path / "out").exists()
    assert np.isnan(TS._nan_max([0.01, float("nan")])) and TS._nan_max([0.01, 0.02]) == 0.02


def test_the_summary_applies_the_data_range_low_k_window():
    kf = np.array([0.0004, 0.002, 0.004, 0.006, 0.008, 0.02, 0.03, 0.04])   # the first bin lies below the data range
    ref = _result(kf=kf)
    cur = _result(kf=kf)
    cur["pf15"] = cur["pf15"].copy()
    cur["pf15"][1:8, 0] *= 1.5                                         # a large LLS change in that first bin only
    res = {("z", 100.0): ref, ("z", 50.0): cur}
    inside = TS.summarize(res, 100.0, n_k=K)["z|T50"]["LLS"]["lowk_change"]
    with_first = TS.summarize(res, 100.0, n_k=K, lowk_kmin=0.0)["z|T50"]["LLS"]["lowk_change"]
    assert inside == pytest.approx(0.0, abs=1e-15) and with_first > 0.1


def test_a_stored_self_check_without_its_threshold_does_not_count(tmp_path, monkeypatch):
    args = _main_env(tmp_path, monkeypatch, []) + ["--thresholds", "50", "100", "200"]
    out = tmp_path / "out"
    TS.main(args + ["--out", str(out)])
    _write_results(out)
    (out / "self_check_a.json").write_text(json.dumps(dict(_check(), threshold=100.0)))
    (out / "self_check_b.json").write_text(json.dumps(_check()))     # no threshold recorded
    assert TS.main(args + ["--out", str(out)]) == 2


def test_the_scripts_dirty_flag(monkeypatch):
    for answer, dirty in (("", False), (" M x", True), (None, None)):
        monkeypatch.setattr(TS, "_git", lambda *a, _r=answer: _r)
        assert TS._dirty() is dirty
    seen = []
    monkeypatch.setattr(TS, "_git", lambda *a: seen.append(a) or "")
    TS._dirty()
    assert seen and seen[0][0] == "--no-optional-locks"
