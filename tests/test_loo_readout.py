"""The matched LOO paired read-out (hcd_analysis/comparison/loo_readout.py). Synthetic errors only."""

import numpy as np
import pytest

from hcd_analysis.comparison import loo_readout as R

K = np.logspace(np.log10(5e-4), np.log10(0.1), 40)


def _fold(name, rng, scale_c1=1.0, scale_gp=1.0, rows=3, z=3.0):
    k = np.tile(K, rows)
    base = rng.normal(0.0, 0.02, k.size)
    return {
        "fold": name,
        "status": "OK",
        "k_skm": k,
        "z": np.full(k.size, z),
        "e_c1": scale_c1 * base + rng.normal(0.0, 0.002, k.size),
        "e_gp": scale_gp * base + rng.normal(0.0, 0.002, k.size),
    }


def test_band_edges():
    k = np.array([9.99e-4, 1e-3, 2.999e-3, 3e-3, 0.0199, 0.02, 0.0499, 0.05, 0.065, 0.0651])
    assert R.band_index(k).tolist() == [-1, 0, 0, 1, 2, 3, 3, 4, 4, -1]


def test_weighted_median_is_the_median_of_the_repeated_sample():
    rng = np.random.default_rng(0)
    for _ in range(200):
        v = np.sort(rng.normal(size=rng.integers(1, 30)))
        w = rng.integers(0, 4, v.size)
        if w.sum() == 0:
            w[0] = 1
        assert R.weighted_median(v, w) == np.median(np.repeat(v, w))


def test_bootstrap_statistics_equal_the_explicitly_resampled_pool():
    rng = np.random.default_rng(1)
    folds = [_fold(f"s{i}", rng) for i in range(5)]
    R._check(folds, R.BANDS)
    pooled = R._pool(folds, R.BANDS)
    w = np.array([2.0, 0.0, 1.0, 1.0, 1.0])
    got = R._stats(pooled, w)
    for b in range(len(R.BANDS)):
        for j, m in enumerate(R.METHODS):
            e = np.concatenate(
                [np.repeat(f["e_" + m][f["_band"] == b], int(w[i])) for i, f in enumerate(folds)]
            )
            assert got[b, j, 0] == pytest.approx(np.median(np.abs(e)), rel=0, abs=0)
            assert got[b, j, 1] == pytest.approx(np.sqrt(np.mean(e**2)), rel=1e-12)


@pytest.mark.parametrize(
    "c1, gp, verdict",
    [(0.5, 1.0, "C1 better"), (1.0, 0.5, "GP better"), (1.0, 1.0, "mixed / inconclusive")],
)
def test_the_three_way_verdict_end_to_end(c1, gp, verdict):
    rng = np.random.default_rng(2)
    out = R.readout([_fold(f"s{i}", rng, c1, gp) for i in range(12)], n_boot=400)
    assert out["verdict"] == verdict and out["n_folds_compared"] == 12


@pytest.mark.parametrize(
    "ci, verdict",
    [
        ([[-0.3, -0.2], [-0.1, -0.05]], "C1 better"),  # both intervals below 0
        ([[0.05, 0.1], [0.2, 0.3]], "GP better"),  # both above 0
        ([[-0.3, -0.2], [-0.1, 0.05]], "mixed / inconclusive"),  # one below, one straddling 0
        ([[-0.3, 0.1], [-0.1, 0.2]], "mixed / inconclusive"),  # opposite signs
        ([[-0.3, 0.0], [-0.1, 0.0]], "mixed / inconclusive"),
    ],
)  # touching 0 is not entirely below
def test_the_verdict_needs_both_band_averaged_intervals_on_one_side_of_zero(ci, verdict):
    assert R.verdict(ci) == verdict


def test_the_bootstrap_is_paired_simulation_level_and_95_percent(monkeypatch):
    rng = np.random.default_rng(7)
    folds = [_fold(f"s{i}", rng, 0.9, 1.0) for i in range(6)]
    out = R.readout(folds, n_boot=50, seed=11)
    work = sorted((dict(f) for f in folds), key=lambda f: f["fold"])
    R._check(work, R.BANDS)
    pooled = R._pool(work, R.BANDS)
    draw = np.random.default_rng(11)
    boot = []
    for _ in range(50):  # one draw of folds, shared by both methods
        w = np.bincount(draw.integers(0, 6, 6), minlength=6).astype(float)
        st = R._stats(pooled, w)
        boot.append(st[:, 0, :] - st[:, 1, :])
    boot = np.array(boot)
    ci = np.percentile(boot, [2.5, 97.5], axis=0)
    avg_ci = np.percentile(boot.mean(axis=1), [2.5, 97.5], axis=0)
    for b in range(len(R.BANDS)):
        assert out["per_band"][b]["delta_median_ci95"] == ci[:, b, 0].tolist()
        assert out["per_band"][b]["delta_rms_ci95"] == ci[:, b, 1].tolist()
    assert out["band_averaged"]["delta_median_ci95"] == avg_ci[:, 0].tolist()
    assert out["band_averaged"]["delta_rms_ci95"] == avg_ci[:, 1].tolist()
    assert out["verdict"] == R.verdict(avg_ci)


def test_the_wilcoxon_tests_are_on_the_per_fold_deltas():
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(8)
    folds = [_fold(f"s{i}", rng, 0.8, 1.0) for i in range(8)]
    out = R.readout(folds, n_boot=20)
    work = sorted((dict(f) for f in folds), key=lambda f: f["fold"])
    R._check(work, R.BANDS)
    pooled = R._pool(work, R.BANDS)
    per = np.array(
        [(lambda st: st[:, 0, :] - st[:, 1, :])(R._stats(pooled, np.eye(8)[i])) for i in range(8)]
    )
    assert out["wilcoxon_p"]["median"] == float(wilcoxon(per[:, :, 0].mean(axis=1)).pvalue)
    assert out["wilcoxon_p"]["rms"] == float(wilcoxon(per[:, :, 1].mean(axis=1)).pvalue)
    for b in range(len(R.BANDS)):
        assert out["per_band"][b]["wilcoxon_p"]["median"] == float(wilcoxon(per[:, b, 0]).pvalue)
        assert out["per_band"][b]["wilcoxon_p"]["rms"] == float(wilcoxon(per[:, b, 1]).pvalue)


def test_missing_expected_folds_count_as_not_run_and_the_order_of_files_does_not_matter():
    rng = np.random.default_rng(9)
    folds = [_fold(f"s{i}", rng, 0.9, 1.0) for i in range(5)]
    a = R.readout(folds, expected=[f"s{i}" for i in range(7)], n_boot=60)
    b = R.readout(folds[::-1], expected=[f"s{i}" for i in range(7)], n_boot=60)
    assert a == b
    assert a["n_folds_expected"] == 7 and a["n_folds_compared"] == 5
    assert [x["fold"] for x in a["not_run"]] == ["s5", "s6"] and "no fold file" in a["not_run"][0][
        "reason"
    ]
    with pytest.raises(ValueError, match="outside the expected list"):
        R.readout(folds, expected=["s0", "s1"], n_boot=10)


def test_folds_from_two_training_arms_are_not_mixed():
    rng = np.random.default_rng(10)
    folds = [dict(_fold(f"s{i}", rng), arm="verbatim") for i in range(3)]
    R.readout(folds, n_boot=10)
    folds[0]["arm"] = "step_matched"
    with pytest.raises(ValueError, match="different training arms"):
        R.readout(folds, n_boot=10)


def test_point_estimates_are_the_pooled_statistics_and_the_bootstrap_is_seeded():
    rng = np.random.default_rng(3)
    folds = [_fold(f"s{i}", rng, 0.8, 1.0) for i in range(6)]
    a = R.readout(folds, n_boot=200)
    b = R.readout(folds, n_boot=200)
    c = R.readout(folds, n_boot=200, seed=1)
    assert a == b and a["band_averaged"]["delta_rms_ci95"] != c["band_averaged"]["delta_rms_ci95"]
    sel = np.concatenate([R.band_index(f["k_skm"]) == 2 for f in folds])
    e1 = np.concatenate([f["e_c1"] for f in folds])[sel]
    assert a["per_band"][2]["median_abs_e"]["c1"] == np.median(np.abs(e1))
    assert a["per_band"][2]["rms_e"]["c1"] == pytest.approx(np.sqrt(np.mean(e1**2)), rel=1e-12)


def test_not_run_folds_are_left_out_for_both_methods_and_listed():
    rng = np.random.default_rng(4)
    folds = [_fold(f"s{i}", rng) for i in range(4)] + [
        {"fold": "s9", "status": "NOT_RUN", "reason": "GP: LinAlgError"}
    ]
    out = R.readout(folds, n_boot=50)
    assert (
        out["not_run"] == [{"fold": "s9", "reason": "GP: LinAlgError"}]
        and out["n_folds_compared"] == 4
    )


def test_non_finite_errors_on_compared_points_are_refused_not_cut():
    rng = np.random.default_rng(5)
    folds = [_fold(f"s{i}", rng) for i in range(3)]
    outside = int(np.flatnonzero(folds[0]["k_skm"] > 0.065)[0])
    folds[0]["e_gp"][outside] = np.nan  # outside the stratum: not compared
    R.readout(folds, n_boot=20)
    inside = int(np.flatnonzero(R.band_index(folds[1]["k_skm"]) == 3)[0])
    folds[1]["e_c1"][inside] = np.inf
    with pytest.raises(ValueError, match="non-finite c1 error"):
        R.readout(folds, n_boot=20)


def test_methods_must_be_on_identical_points_and_redshifts_outside_the_range_are_not_compared():
    rng = np.random.default_rng(6)
    folds = [_fold(f"s{i}", rng) for i in range(3)]
    bad = dict(folds[0], e_gp=folds[0]["e_gp"][:-1])
    with pytest.raises(ValueError, match="identical points"):
        R.readout([bad] + folds[1:], n_boot=10)
    high = _fold("s3", rng, z=4.8)
    with pytest.raises(ValueError, match="no point in band"):
        R.readout(folds + [high], n_boot=10)


def test_the_verdict_is_taken_from_the_band_averaged_bootstrap_intervals(monkeypatch):
    rng = np.random.default_rng(12)
    folds = [_fold(f"s{i}", rng, 0.95, 1.0) for i in range(6)]
    seen = []
    real = R.verdict
    monkeypatch.setattr(R, "verdict", lambda ci: seen.append(np.array(ci)) or real(ci))
    out = R.readout(folds, n_boot=40)
    assert len(seen) == 1
    ci = np.array(
        [
            [
                out["band_averaged"]["delta_median_ci95"][0],
                out["band_averaged"]["delta_rms_ci95"][0],
            ],
            [
                out["band_averaged"]["delta_median_ci95"][1],
                out["band_averaged"]["delta_rms_ci95"][1],
            ],
        ]
    )
    assert np.array_equal(seen[0], ci)


def test_flagged_gp_fits_are_listed_and_a_duplicate_fold_is_refused():
    rng = np.random.default_rng(13)
    folds = [_fold(f"s{i}", rng) for i in range(3)]
    folds[1]["gp_unconverged_z"] = [2.4, 4.6]
    out = R.readout(folds, n_boot=10)
    assert out["gp_unconverged"] == {"s1": [2.4, 4.6]}
    with pytest.raises(ValueError, match="appears twice"):
        R.readout(folds + [dict(folds[0])], n_boot=10)


def test_the_read_out_script_refuses_context_folds_and_needs_the_expected_list(tmp_path):
    import json
    import os
    import subprocess
    import sys

    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
        "matched_loo_readout.py",
    )
    rng = np.random.default_rng(3)

    def write(name, weights):
        f = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in _fold(name, rng).items()
        }
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(dict(f, weights=weights)))
        return str(p)

    prim = [write(f"s{i}", "head_a_per_fold") for i in range(3)]
    ctx = write("s3", "truth (context only)")
    exp = tmp_path / "expected.txt"
    exp.write_text("s0\ns1\ns2\ns3\ns4\n")

    def run(*a):
        return subprocess.run(
            [sys.executable, "-I", "-B", script, *a, "--n-boot", "50"],
            capture_output=True,
            text=True,
        )

    r = run("--expected-folds", str(exp), "--folds", *prim, ctx, "--out", str(tmp_path / "a.json"))
    assert (
        r.returncode != 0
        and "expected 'head_a_per_fold'" in r.stderr
        and not (tmp_path / "a.json").exists()
    )
    r = run(
        "--expected-folds",
        str(exp),
        "--folds",
        *prim,
        ctx,
        "--truth-weights-context",
        "--out",
        str(tmp_path / "b.json"),
    )
    assert (
        r.returncode != 0
        and "expected 'truth (context only)'" in r.stderr
        and not (tmp_path / "b.json").exists()
    )
    r = run("--folds", *prim, "--out", str(tmp_path / "c.json"))
    assert r.returncode != 0 and "required: --expected-folds" in r.stderr
    r = run("--expected-folds", str(exp), "--folds", *prim, "--out", str(tmp_path / "d.json"))
    assert r.returncode == 0, r.stderr
    out = json.load(open(tmp_path / "d.json"))
    assert out["weights"] == "head_a_per_fold" and [m["fold"] for m in out["not_run"]] == [
        "s3",
        "s4",
    ]
    ctxs = [
        write(f"t{i}", "truth (context only)") for i in range(2)
    ]  # a context read-out, labelled as such
    (tmp_path / "expected_t.txt").write_text("t0\nt1\n")
    r = run(
        "--expected-folds",
        str(tmp_path / "expected_t.txt"),
        "--folds",
        *ctxs,
        "--truth-weights-context",
        "--out",
        str(tmp_path / "e.json"),
    )
    assert r.returncode == 0, r.stderr
    assert json.load(open(tmp_path / "e.json"))["weights"] == "truth (context only)"
