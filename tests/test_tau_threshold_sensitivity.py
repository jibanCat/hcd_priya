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
                                        (["--headline", "75"], "must be among")])
def test_bad_arguments_are_refused_before_anything_is_created(tmp_path, monkeypatch, capsys, extra, why):
    args = _main_env(tmp_path, monkeypatch, [])
    with pytest.raises(SystemExit):
        TS.main(args + ["--out", str(tmp_path / "out")] + extra)
    assert why in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_a_failed_self_check_makes_the_reading_invalid(tmp_path, monkeypatch):
    args = _main_env(tmp_path, monkeypatch, [])
    out = tmp_path / "out"
    TS.main(args + ["--out", str(out), "--thresholds", "50", "100", "200"])
    for tag in ("a", "b"):
        for thr in (50, 100, 200):
            r = _result()
            np.savez(out / f"result_{tag}_T{thr}.npz", kf=r["kf"], pf15=r["pf15"], pu15=r["pu15"],
                     counts15=r["counts15"], log_nhi=np.array([17.5, 19.5, 20.5]))
    (out / "self_check_a.json").write_text(json.dumps({"passed": True}))
    (out / "self_check_b.json").write_text(json.dumps({"passed": False}))
    assert TS.main(args + ["--out", str(out), "--thresholds", "50", "100", "200", "--self-check", "4"]) == 2
    doc = json.loads((out / "summary.json").read_text())
    assert doc["verdict"]["reading"].startswith("invalid") and "'b'" in doc["verdict"]["reading"]
    (out / "self_check_b.json").unlink()
    (out / "self_check_b.json").write_text(json.dumps({"passed": True}))
    assert TS.main(args + ["--out", str(out), "--thresholds", "50", "100", "200", "--self-check", "4"]) == 0
    doc = json.loads((out / "summary.json").read_text())
    assert doc["verdict"]["reading"] == "negligible" and doc["summary_settings"]["n_k"] == 172


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
    out = TS.anchor({("z", 100.0): r}, cache, "synth", 1.0, 100.0, n_k=K)["z"]
    assert out["cache_row"] == 1 and out["count_diff_by_bin"] == [0] * 15 and out["scale_rel_diff"] == 0.0
    assert out["max_rel_diff_filtered"] == 0.0
    assert out["max_rel_diff_unfiltered"] == pytest.approx(abs(1 / 1.001 - 1))          # mine / cache - 1, bin 0
