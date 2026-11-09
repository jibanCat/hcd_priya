"""The GP LOO refit adapter's logic (hcd_analysis/comparison/adapters/lyaemu_loo_refit.py) with a stand-in GP class:
seeding, status records, refusals and the NOT_RUN rules. Synthetic values only; the end-to-end refit with GPy is the
optional tests/external/test_lyaemu_loo_refit_external.py."""
import os
import types

import numpy as np
import pytest

from hcd_analysis.comparison.adapters import lyaemu_loo_refit as R


def _cube(X, limits):
    lim = np.asarray(limits, float)
    return (np.asarray(X, float) - lim[:, 0]) / (lim[:, 1] - lim[:, 0])


def _fold(nz=2, n=12, m=3, nk=4, rng=None):
    rng = rng or np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, (nz, n, 3))
    X[:, 0, :], X[:, 1, :] = 0.05, 0.95                             # every column spans the unit cube
    return X, rng.uniform(1.0, 2.0, (nz, n, nk)), rng.uniform(0.3, 0.7, (nz, m, 3)), np.array([2.2, 2.4])


LIMITS = np.array([[0.0, 1.0]] * 3)


def _fake(statuses, log=None, *, fail_at=None, extra_print=""):
    log = [] if log is None else log

    class FakeGP:
        def __init__(self, *, params, powers, param_limits, zbin, traindir):
            assert traindir is None
            if fail_at is not None and np.isclose(zbin, fail_at):
                raise np.linalg.LinAlgError("not positive definite")
            print(f"Optimizing GP for z:{zbin}{extra_print}")
            statuses.append("Converged")
            log.append((float(zbin), float(np.random.random())))     # what the seed fixes
            self.nk = np.shape(powers)[1]
            self.gp = self

        def parameter_names_flat(self):
            return np.array(["rbf.variance"])

        @property
        def param_array(self):
            return np.array([1.5])

        def log_likelihood(self):
            return -3.0

        def predict(self, Xh):
            return np.ones((len(Xh), self.nk)), np.full((len(Xh), self.nk), 0.1)
    return FakeGP, log


def test_each_redshift_is_seeded_reproducibly_and_its_statuses_recorded():
    X, Y, Xh, z = _fold()
    runs = []
    for _ in range(2):
        st = []
        gp, log = _fake(st)
        status, reason, per_z, mean, std = R.fit_fold(X, Y, Xh, z, LIMITS, 7, statuses=st, sklearn_gp=gp, to_cube=_cube)
        runs.append(log)
    assert status == "OK" and reason == "" and mean.shape == (2, 3, 4) and np.all(np.isfinite(std))
    assert [p["seed_z"] for p in per_z] == [700, 701] and all(p["statuses"] == ["Converged"] for p in per_z)
    assert runs[0] == runs[1] and runs[0][0][1] != runs[0][1][1]       # same seed, same draws; z differ


def test_a_column_that_does_not_span_the_cube_makes_the_fold_not_run_before_any_fit():
    X, Y, Xh, z = _fold()
    X[1, :, 2] = 0.5
    st = []
    gp, log = _fake(st)
    status, reason, per_z, mean, _ = R.fit_fold(X, Y, Xh, z, LIMITS, 1, statuses=st, sklearn_gp=gp, to_cube=_cube)
    assert status == "NOT_RUN" and "span" in reason and mean is None and log == []


def test_a_linalg_error_makes_the_fold_not_run():
    X, Y, Xh, z = _fold()
    st = []
    gp, _ = _fake(st, fail_at=2.4)
    status, reason, per_z, mean, _ = R.fit_fold(X, Y, Xh, z, LIMITS, 1, statuses=st, sklearn_gp=gp, to_cube=_cube)
    assert status == "NOT_RUN" and "LinAlgError at z=2.4" in reason and mean is None and len(per_z) == 1


@pytest.mark.parametrize("line", ["\nSaving GP to x", "\nLoading pre-trained GP for z:2.2"])
def test_a_fit_that_saves_or_loads_is_refused(line):
    X, Y, Xh, z = _fold()
    st = []
    gp, _ = _fake(st, extra_print=line)
    with pytest.raises(R.RefitRefused, match="unexpected fit log"):
        R.fit_fold(X, Y, Xh, z, LIMITS, 1, statuses=st, sklearn_gp=gp, to_cube=_cube)


def test_threads_must_be_one():
    R.check_threads({v: "1" for v in R.THREAD_VARS})
    with pytest.raises(R.RefitRefused, match="single-threaded"):
        R.check_threads({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"})
    with pytest.raises(R.RefitRefused, match="single-threaded"):
        R.check_threads({})


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file permissions")
def test_a_writable_reference_copy_is_refused(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "emulator_params.json").write_text("{}")
    with pytest.raises(R.RefitRefused, match="writable"):
        R.refuse_writable(str(base))
    os.chmod(base / "emulator_params.json", 0o444)
    os.chmod(base, 0o555)
    try:
        R.refuse_writable(str(base))
        with pytest.raises(PermissionError):
            open(base / "new.json", "x").close()
    finally:
        os.chmod(base, 0o755)


def _reference(rng=None):
    rng = rng or np.random.default_rng(3)
    zout = np.array([4.6, 2.4, 2.2])                                  # upstream's order is not ours
    params = rng.uniform(0.0, 1.0, (8, 10))
    fv = rng.uniform(1.0, 2.0, (8, 3 * 5))
    return params, fv, zout


def test_input_rows_must_be_upstream_training_rows_and_parity_is_recorded():
    params, fv, zout = _reference()
    z = np.array([2.2, 2.4])
    pick = np.array([[0, 3, 5], [0, 3, 5]])
    X = params[pick]
    Y = np.stack([fv[pick[0], 2 * 5:3 * 5], fv[pick[1], 1 * 5:2 * 5]])
    Y[1, 2, 0] *= 1.001                                               # one entry off by 1e-3 at z = 2.4
    par = R.match_reference_design(X, Y, z, params, fv, zout)
    assert par[0] == {"z": 2.2, "max_rel": 0.0, "n_rows_above_1e-4": 0}
    assert par[1]["n_rows_above_1e-4"] == 1 and par[1]["max_rel"] == pytest.approx(1e-3)
    off = X.copy()
    off[0, 1, 4] *= 1.0 + 1e-6
    with pytest.raises(R.RefitRefused, match="not exactly one"):
        R.match_reference_design(off, Y, z, params, fv, zout)
    twice = np.vstack([params, params[3:4]])
    with pytest.raises(R.RefitRefused, match="not exactly one"):
        R.match_reference_design(X, Y, z, twice, np.vstack([fv, fv[3:4]]), zout)
    with pytest.raises(R.RefitRefused, match="not on upstream"):
        R.match_reference_design(X, Y, np.array([2.2, 3.0]), params, fv, zout)
    with pytest.raises(R.RefitRefused, match="modes"):
        R.match_reference_design(X, Y[..., :4], z, params, fv, zout)


def test_a_limit_missed_by_a_few_ulp_moves_onto_the_input_and_a_real_excursion_is_refused():
    lim = np.array([[0.6, 1.5], [2.0, 3.0]])
    X = np.array([[np.nextafter(np.nextafter(0.6, 0.0), 0.0), 2.5], [1.0, 3.0]])   # 2 ulp below the low limit
    new, adj = R.ulp_adjust_limits(lim, X)
    assert new[0, 0] == X[0, 0] and new[0, 1] == 1.5 and np.array_equal(new[1], lim[1])
    assert adj == [{"column": 0, "side": "low", "limit": 0.6, "set_to": float(X[0, 0]), "ulp": 2.0}]
    assert R.ulp_adjust_limits(lim, np.array([[0.6, 3.0]]))[1] == []                 # on the limits: unchanged
    with pytest.raises(R.RefitRefused, match="outside upstream's limit"):
        R.ulp_adjust_limits(lim, np.array([[1.0, 3.0 + 1e-12]]))


def _ulps(x, n, toward):
    for _ in range(n):
        x = np.nextafter(x, toward)
    return x


@pytest.mark.parametrize("side, toward", [(0, 0.0), (1, 10.0)])
def test_the_limit_moves_at_most_4_ulp_on_either_side(side, toward):
    lim = np.array([[0.6, 1.5], [2.0, 3.0]])
    edge = lim[0, side]
    for n, ok in ((4, True), (5, False)):
        X = np.array([[1.0, 2.5], [1.0, 2.5]])
        X[0, 0] = _ulps(edge, n, toward)
        if ok:
            new, adj = R.ulp_adjust_limits(lim, X)
            assert new[0, side] == X[0, 0] and adj[0]["ulp"] == pytest.approx(n)
        else:
            with pytest.raises(R.RefitRefused, match="outside upstream's limit"):
                R.ulp_adjust_limits(lim, X)


def test_span_needs_both_ends_of_every_column():
    lim = np.array([[0.0, 1.0]] * 2)
    cube = np.array([[0.1, 0.1], [0.9, 0.9]])
    assert R.span_ok(_cube(cube, lim))
    assert not R.span_ok(_cube(np.array([[0.25, 0.1], [0.9, 0.9]]), lim))    # low end missing
    assert not R.span_ok(_cube(np.array([[0.1, 0.1], [0.75, 0.9]]), lim))    # high end missing


def test_a_redshift_whose_optimizer_never_converged_is_flagged():
    X, Y, Xh, z = _fold()
    st = []

    class Unconverged:
        def __init__(self, *, params, powers, param_limits, zbin, traindir):
            print(f"Optimizing GP for z:{zbin}\nRestarting optimization")
            st.extend(["Maximum number of f evaluations reached"] * 2)
            self.nk, self.gp = np.shape(powers)[1], self

        def parameter_names_flat(self):
            return np.array(["v"])

        param_array = np.array([1.0])

        def log_likelihood(self):
            return 0.0

        def predict(self, Xh):
            return np.ones((len(Xh), self.nk)), np.ones((len(Xh), self.nk))
    status, _, per_z, _, _ = R.fit_fold(X, Y, Xh, z, LIMITS, 1, statuses=st, sklearn_gp=Unconverged, to_cube=_cube)
    assert status == "OK" and all(p["converged_any"] is False and p["restarted"] for p in per_z)


def test_the_kept_run_decides_the_convergence_flag():
    runs = [(5.0, "Maximum number of iterations"), (2.0, "Converged"), (1.0, "Maximum number of iterations")]
    assert R.adopted_run(runs, restarted=True) == 2                 # the lowest objective among the restarts
    assert R.adopted_run(runs, restarted=False) == 0
    assert R.adopted_run([], restarted=True) is None
    many = [(9.0, "x")] + [(float(10 - i), "Converged") for i in range(12)]
    assert R.adopted_run(many, restarted=True) == len(many) - 1      # only the last N_RESTARTS runs compete
    X, Y, Xh, z = _fold()
    st = []

    class Kept:
        def __init__(self, *, params, powers, param_limits, zbin, traindir):
            print(f"Optimizing GP for z:{zbin}\nRestarting optimization")
            self.optimization_runs = [types.SimpleNamespace(f_opt=f, status=t) for f, t in runs]
            st.extend(t for _, t in runs)
            self.nk, self.gp = np.shape(powers)[1], self

        def parameter_names_flat(self):
            return np.array(["v"])

        param_array = np.array([1.0])

        def log_likelihood(self):
            return 0.0

        def predict(self, Xh):
            return np.ones((len(Xh), self.nk)), np.ones((len(Xh), self.nk))
    status, _, per_z, _, _ = R.fit_fold(X, Y, Xh, z, LIMITS, 1, statuses=st, sklearn_gp=Kept, to_cube=_cube)
    assert status == "OK"
    for p in per_z:                                                # some run converged, but not the kept one
        assert p["adopted_run"] == 2 and p["converged_any"] and not p["converged_final"]
        assert p["runs"][1] == {"f_opt": 2.0, "status": "Converged"}


def _kept_gp(runs, st):
    class Kept:
        def __init__(self, *, params, powers, param_limits, zbin, traindir):
            print(f"Optimizing GP for z:{zbin}\nRestarting optimization")
            self.optimization_runs = [types.SimpleNamespace(f_opt=f, status=t) for f, t in runs]
            st.extend(t for _, t in runs)
            self.nk, self.gp = np.shape(powers)[1], self

        def parameter_names_flat(self):
            return np.array(["v"])

        param_array = np.array([1.0])

        def log_likelihood(self):
            return 0.0

        def predict(self, Xh):
            return np.ones((len(Xh), self.nk)), np.ones((len(Xh), self.nk))
    return Kept


MAX = "Maximum number of iterations"


@pytest.mark.parametrize("runs, kept", [
    ([(0.5, MAX), (3.0, "Converged"), (1.0, "Converged")] + [(2.0, MAX)] * 8, 2),   # the first run never competes
    ([(1.0, MAX), (1.0, "Converged")] + [(2.0, MAX)] * 9, 1),                       # a tie with the first run
    ([(4.0, MAX), (3.0, "Converged"), (0.5, MAX)] + [(2.0, "Converged")] * 8, 2)])  # neither first nor last
def test_the_kept_run_is_the_best_restart_wherever_it_sits(runs, kept):
    assert len(runs) == 1 + R.N_RESTARTS and R.adopted_run(runs, restarted=True) == kept


def test_a_kept_run_that_converged_is_not_flagged():
    X, Y, Xh, z = _fold()
    st = []
    runs = [(5.0, MAX), (1.0, "Converged"), (2.0, MAX)]
    status, _, per_z, _, _ = R.fit_fold(X, Y, Xh, z, LIMITS, 1, statuses=st, sklearn_gp=_kept_gp(runs, st),
                                        to_cube=_cube)
    assert status == "OK" and all(p["adopted_run"] == 1 and p["converged_final"] for p in per_z)
