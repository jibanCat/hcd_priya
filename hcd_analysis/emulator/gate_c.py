"""Gate C validation helpers (emulator-debug campaign 2026-10; spec GATE_C_SPEC.md in the notes repository).

Truth is the held-out simulation's own source product (cache rows on their own header-exact grid); predictions come
through the schema-2.0 physical-coordinate API. These helpers keep the comparison honest: coordinate agreement is
measured explicitly (C1), interpolation onto a fixed physical grid never extrapolates, and held-out entries are paired
with upstream's leave-one-out product by parameters, mean-flux rung and redshift (C4)."""
from __future__ import annotations

import numpy as np


def coordinate_agreement(k_pred, k_row):
    """max over the row's finite modes of |k_pred / k_row - 1| (C1)."""
    k_pred, k_row = np.asarray(k_pred, float), np.asarray(k_row, float)
    m = np.isfinite(k_row)
    return float(np.max(np.abs(k_pred[m] / k_row[m] - 1.0)))


def interp_to_grid(k_target, k, P):
    """Linear interpolation of P(k) onto k_target (the forward's convention); refuses any target outside [k_min, k_max]
    of the finite modes instead of clamping."""
    k, P, k_target = np.asarray(k, float), np.asarray(P, float), np.asarray(k_target, float)
    m = np.isfinite(k) & np.isfinite(P)
    if k_target.min() < k[m].min() or k_target.max() > k[m].max():
        raise ValueError(f"target k [{k_target.min():.4g}, {k_target.max():.4g}] outside the sampled modes "
                         f"[{k[m].min():.4g}, {k[m].max():.4g}]")
    return np.interp(k_target, k[m], P[m])


def class_rms(res_cls, keep):
    """Per-class RMS of fractional residuals res_cls (rows, 4, K) over the kept (rows, K) finite modes: the historical
    LOSO/ensemble metric (run_loso_sweep.py, validate_production_ensemble.py)."""
    r = np.where(np.asarray(keep, bool)[:, None, :] & np.isfinite(res_cls), res_cls, np.nan)
    return np.sqrt(np.nanmean(r ** 2, axis=(0, 2)))


def ensemble_residual(member_residuals):
    """Residual of the ensemble MEAN prediction from members' fractional residuals against the same truth
    (mean of P_pred / P_true - 1 = mean of the residuals, because the truth is common)."""
    return np.mean(np.stack([np.asarray(r, float) for r in member_residuals]), axis=0)


def match_upstream_loo(entries, up_params, up_zout, *, rtol=1e-6, rung_atol=2e-6, z_atol=1e-6):
    """Pair our held-out entries with upstream LOO rows. ``entries``: dicts with ``params`` (the simulation parameters,
    upstream's order), ``alpha`` (mean-flux rung), ``z`` and ``key``; ``up_params[:, 0]`` is upstream's rung and the
    rest its parameters. Returns {key: (upstream_row, upstream_z_index)} for entries with exactly one match."""
    up_params = np.asarray(up_params, float)
    out = {}
    for e in entries:
        rows = np.where(np.all(np.isclose(up_params[:, 1:], np.asarray(e["params"], float)[None, :], rtol=rtol, atol=0),
                               axis=1) & np.isclose(up_params[:, 0], e["alpha"], rtol=0, atol=rung_atol))[0]
        zi = np.where(np.isclose(np.asarray(up_zout, float), e["z"], rtol=0, atol=z_atol))[0]
        if rows.size == 1 and zi.size == 1:
            out[e["key"]] = (int(rows[0]), int(zi[0]))
    return out


# Upstream defect F1 (gate B PARITY_TABLE): the z = 2.2 slot of this simulation in upstream's products holds the
# z = 2.39 snapshot. Pinned by upstream's own parameter values (ns, Ap, herei, heref, alphaq, hub, omegamh2, hireionz,
# bhfeedback), so the exclusion is made on the upstream side and cannot follow a renamed or re-ordered simulation.
F1_PARAMS = np.array([0.95925, 2.3433333333333333e-09, 3.81, 2.99, 1.765, 0.725, 0.1443, 6.825, 0.046666666666666669])
F1_Z = 2.2
C4_N_ENTRIES = 7790          # upstream's 60 x 10 x 13 = 7800 (loo row, z) entries minus the 10 F1 entries


def c4_collect(member_evals, sim_params):
    """C4/S7 entries from per-checkpoint evaluations. ``member_evals`` yields, per held-out simulation, the list of its
    member evaluations (mappings with rows, k_ks, res_ks, sim_name, alpha, z; members must agree on rows and k bins).
    Returns (entries, res, k_ks): one entry per held-out row, key (simulation index, row index), and res[key] the
    (members, n_k) residuals. No finiteness filter: ``c4_matched_set`` checks the compared entries."""
    entries, res, k_ks = [], {}, None
    for n, members in enumerate(member_evals):
        e0 = members[0]
        for e in members[1:]:
            if not (np.array_equal(e["rows"], e0["rows"]) and np.array_equal(e["k_ks"], e0["k_ks"])):
                raise ValueError(f"members of held-out simulation {n} disagree on rows or k bins")
        if k_ks is None:
            k_ks = np.asarray(e0["k_ks"], float)
        elif not np.array_equal(k_ks, e0["k_ks"]):
            raise ValueError(f"held-out simulation {n} disagrees on the k bins")
        r = np.stack([np.asarray(e["res_ks"], float) for e in members])
        for i in range(np.asarray(e0["rows"]).size):
            entries.append(dict(params=sim_params[str(e0["sim_name"][i])], alpha=float(e0["alpha"][i]),
                                z=float(e0["z"][i]), key=(n, i)))
            res[(n, i)] = r[:, i]
    return entries, res, k_ks


def c4_f1_entries(up_params, up_zout, *, f1_params=F1_PARAMS, f1_z=F1_Z, n_rungs=10):
    """Upstream (row, z index) entries of defect F1: the rows holding the pinned parameter vector (one per rung) at
    z = 2.2. Raises unless exactly ``n_rungs`` rows and one z match."""
    up_params = np.asarray(up_params, float)
    rows = np.where(np.all(np.isclose(up_params[:, 1:], np.asarray(f1_params, float)[None, :], rtol=1e-12, atol=0),
                           axis=1))[0]
    zi = np.where(np.isclose(np.asarray(up_zout, float), f1_z, rtol=0, atol=1e-6))[0]
    if rows.size != n_rungs or zi.size != 1:
        raise ValueError(f"F1 not found as pinned: {rows.size} upstream rows hold its parameters (expected {n_rungs}), "
                         f"{zi.size} z slots at {f1_z}")
    return {(int(r), int(zi[0])) for r in rows}


def c4_matched_set(entries, res, up_params, up_zout, *, n_expected=C4_N_ENTRIES, f1_params=F1_PARAMS, f1_z=F1_Z):
    """The C4/S7 comparison set, checked: every upstream (loo row, z) entry outside F1 is matched by exactly one of our
    entries, no compared entry has a non-finite residual in any member, and the count is ``n_expected``. Returns
    [(key, (upstream_row, upstream_z_index))] in the order of ``entries``."""
    up_params = np.asarray(up_params, float)
    rungs_per_sim = int(np.unique(np.round(up_params[:, 0], 9)).size)
    f1 = c4_f1_entries(up_params, up_zout, f1_params=f1_params, f1_z=f1_z, n_rungs=rungs_per_sim)
    match = match_upstream_loo(entries, up_params, up_zout)
    used = {}
    for key, ue in match.items():
        used.setdefault(ue, []).append(key)
    twice = {ue: ks for ue, ks in used.items() if len(ks) > 1}
    if twice:
        raise ValueError(f"{len(twice)} upstream entries matched more than once, e.g. {next(iter(twice.items()))}")
    every = {(r, z) for r in range(up_params.shape[0]) for z in range(np.asarray(up_zout).size)}
    missing = every - f1 - set(used)
    if missing:
        raise ValueError(f"{len(missing)} upstream entries unmatched, e.g. {sorted(missing)[:5]}")
    out = [(key, ue) for key, ue in match.items() if ue not in f1]
    bad = [key for key, _ in out if not np.all(np.isfinite(res[key]))]
    if bad:
        raise ValueError(f"{len(bad)} compared entries have a non-finite residual in some member, e.g. {bad[:5]}")
    if len(out) != n_expected:
        raise ValueError(f"C4 comparison set has {len(out)} entries, expected {n_expected}")
    return out
