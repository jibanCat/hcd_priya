"""Matched leave-one-simulation-out (LOO) validation, the hcd_priya side.

Both emulators of the matched comparison are trained on the SAME rows per fold: the PRIYA design (the 10 PRIYA
mean-flux rungs, i.e. the even ``alpha_idx`` of the 20-rung ladder, at the 13 redshifts 2.2-4.6) of every simulation
except the held-out one, with the cache table as the shared training values and truth. Everything learned from data is
fitted without the held-out simulation: the network and its normalizers (``train.train_fold`` on the fold's rows),
and the class-weight correction ``delta_c(z)`` of C1's total-P1D composition (refitted per fold on the training
simulations' snapshot groups, exactly as ``scripts/calibrate_delta_c.py`` fits it; by default on every redshift of
those simulations, as the deployed calibration does, with ``z_range`` as the sensitivity option). The absorption path
per sightline is analytic (box, cosmology, z), not learned.

The held-out prediction of the total P1D uses the Head-A class weights ``w_c = w_M0(dN/dX, Xbar) (1 + delta_c(z))``
with the GEOMETRIC path Xbar = X_tot / n_sightlines, the path ``delta_c`` is calibrated on (see ``w_c_with_coeffs``).
"""

from __future__ import annotations

import os

import numpy as np

from .data import production_row_split
from .dndx_wc import Z_FIT_HI, Z_FIT_LO, mu_from_dndx, w_c_from_mu
from ..comparison.conventions import PARAM_NAMES  # the cache's parameter order

PRIYA_ALPHA_IDX = tuple(range(0, 19, 2))  # the 10 PRIYA rungs among the 20 of the cache ladder
PRIYA_Z = tuple(round(2.2 + 0.2 * i, 1) for i in range(13))
VAL_SEED, VAL_FRAC = 12345, 0.10  # the production row-split constants
PRODUCTION_STEPS = 6840  # optimizer steps of a production-design LOO member (180 epochs)


def matched_mask(d):
    """Rows of the PRIYA design: an even alpha_idx of the PRIYA rungs and a PRIYA redshift."""
    z = np.asarray(d["z_grid"])
    on_z = np.any(np.abs(z[:, None] - np.asarray(PRIYA_Z)[None, :]) < 1e-6, axis=1)
    return np.isin(np.asarray(d["alpha_idx"]), PRIYA_ALPHA_IDX) & on_z


def fold_split(d, holdout_sim, *, val_seed=VAL_SEED, val_frac=VAL_FRAC):
    """(fit_rows, val_rows, held_rows) of one fold. All three are matched-design rows; the held-out simulation's rows
    are only in ``held_rows``; the early-stopping rows are ``data.production_row_split`` applied to the training
    simulations' matched rows (no held-out data)."""
    m = matched_mask(d)
    sims = np.asarray(d["sim_name"])
    rows = np.flatnonzero(m)
    held = rows[sims[rows] == holdout_sim]
    if held.size == 0:
        raise ValueError(f"holdout simulation {holdout_sim!r} has no matched rows")
    train_rows = rows[sims[rows] != holdout_sim]
    fit_local, val_local, none = production_row_split(
        sims[train_rows], val_seed=val_seed, val_frac=val_frac
    )
    assert none.size == 0
    return np.sort(train_rows[fit_local]), np.sort(train_rows[val_local]), np.sort(held)


def group_table(d):
    """Per snapshot group: z, simulation, counted class fractions (4,), dN/dX (3,), geometric Xbar."""
    gid = np.asarray(d["snap_group_idx"])
    G = np.asarray(d["snap_dNdX"]).shape[0]
    first = np.array([np.flatnonzero(gid == g)[0] for g in range(G)])
    cc = np.asarray(d["coarse_counts"], dtype=np.float64)[first]
    n_sl = cc.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        w_cnt = cc / n_sl[:, None]
        xbar = np.asarray(d["snap_total_path_dX"]) / n_sl
    return {
        "z": np.asarray(d["z_grid"])[first],
        "sim": np.asarray(d["sim_name"])[first],
        "w_counted": w_cnt,
        "dndx": np.asarray(d["snap_dNdX"]),
        "xbar_geom": xbar,
        "ok": n_sl > 0,
    }


def fit_delta_c(d, sims, *, deg=2, z_range=None):
    """delta_c(z) coefficients (4, deg+1; np.polyval order; clean, LLS, subDLA, DLA) and the per-class residual std,
    fitted on the snapshot groups of ``sims`` only: delta = w_counted / w_M0(dN/dX, X_tot / n_sl) - 1 per group, one
    unweighted degree-``deg`` polynomial in z per class (the method of scripts/calibrate_delta_c.py). ``z_range``
    optionally restricts the groups (a sensitivity choice; the default uses every z of those simulations).
    """
    t = group_table(d)
    use = t["ok"] & np.isin(t["sim"], np.asarray(list(sims)))
    if z_range is not None:
        use &= (t["z"] >= z_range[0] - 1e-6) & (t["z"] <= z_range[1] + 1e-6)
    if not use.any():
        raise ValueError("no snapshot group to fit delta_c on")
    wm = np.asarray(w_c_from_mu(mu_from_dndx(t["dndx"][use], t["xbar_geom"][use])))
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(wm > 0, t["w_counted"][use] / wm - 1.0, 0.0)
    z = t["z"][use]
    coeffs = np.stack([np.polyfit(z, delta[:, c], deg) for c in range(4)])
    resid = np.stack([delta[:, c] - np.polyval(coeffs[c], z) for c in range(4)], axis=1)
    return coeffs, resid.std(axis=0)


def w_c_with_coeffs(dndx, xbar, z, coeffs):
    """``dndx_wc.w_c_corrected`` with explicit delta_c coefficients (z clamped to the calibration range the same way).
    ``xbar`` must be the path the coefficients were fitted with (the geometric path for ``fit_delta_c``).
    """
    z = np.clip(np.asarray(z, dtype=np.float64), Z_FIT_LO, Z_FIT_HI)
    w0 = np.asarray(
        w_c_from_mu(
            mu_from_dndx(np.asarray(dndx, dtype=np.float64), np.asarray(xbar, dtype=np.float64))
        )
    )
    dc = np.stack([np.polyval(coeffs[c], z) for c in range(4)], axis=-1)
    w = w0 * (1.0 + dc)
    return w / np.sum(w, axis=-1, keepdims=True)


def member_outputs(model, norm_stats, x, tau0):
    """One member at encoder input ``x`` and physical ``tau0``: its reconstructed per-class P_filt (4,K) and its
    physical Head-A dN/dX (3,), each with the member's OWN normalizer (the formulas of
    ``comparison/adapters/hcd_priya.py``)."""
    from .data import untransform_prediction
    from .predict import reconstruct_P_filt_jax

    pred = model(x, tau0)
    pf = np.asarray(
        reconstruct_P_filt_jax(pred["P_filt_base"], pred["P_filt_resid"], norm_stats["P_filt"])
    )
    dn = untransform_prediction({"dndx": np.asarray(pred["dndx"])[None]}, norm_stats)["dndx"][0]
    return pf, np.asarray(dn, dtype=np.float64)


def predict_fold(members, d, held_rows, coeffs, *, truth_weights=False, outputs=member_outputs):
    """C1's held-out prediction for one fold. ``members``: [(model, norm_stats)] of the fold's trained members. Per
    held-out row, at the row's own (theta, z, tau0): P_filt = the mean over members of their reconstructed per-class
    P_filt (the mean after reconstruction, as deployed), dN/dX = the members' mean physical Head-A output, the class
    weights ``w_c_with_coeffs(dN/dX, Xbar, z, coeffs)`` with the per-fold ``coeffs`` (``fit_delta_c`` on the
    training simulations) and the row's geometric path Xbar = X_tot / n_sightlines (analytic in box, cosmology and
    z, so not held-out information), and P = sum_c w_c P_filt. ``truth_weights`` uses the row's counted class
    fractions instead (context only: it uses held-out truth). Returns P (rows, K) and the per-row ingredients.
    """
    from .data import Z_LIMITS, normalize_params
    from .model import structural_tier_p

    t = group_table(d)
    held_rows = np.asarray(held_rows)
    P, PF, W, DN = [], [], [], []
    for r in held_rows:
        z = float(d["z_grid"][r])
        th = np.asarray(
            normalize_params(np.asarray(d["params"][r], dtype=np.float64)), dtype=np.float64
        )
        x = np.concatenate([th, [(z - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])]])
        per = [outputs(m, ns, x, float(d["tau0"][r])) for m, ns in members]
        pf = np.mean([p[0] for p in per], axis=0)
        dn = np.mean([p[1] for p in per], axis=0)
        if truth_weights:
            w = np.asarray(d["w_c_cache"][r], dtype=np.float64)
        else:
            w = w_c_with_coeffs(dn, t["xbar_geom"][int(d["snap_group_idx"][r])], z, coeffs)
        P.append(np.asarray(structural_tier_p(w, pf)))
        PF.append(pf)
        W.append(w)
        DN.append(dn)
    return np.asarray(P), {
        "P_filt": np.asarray(PF),
        "w_c": np.asarray(W),
        "dndx": np.asarray(DN),
        "rows": held_rows,
        "weights": "truth" if truth_weights else "head_a_per_fold",
    }


def fold_errors(
    d, holdout_sim, held_rows, P_c1, gp_record, k_com_hmpc, *, member_ok=None, min_members=4
):
    """One fold's paired errors for ``comparison.loo_readout``: e = P / T - 1 for C1 and the GP on identical points
    (the held-out rows, modes 1..n with n the GP's mode count), T the cache's P_tier_p, k in s/km per row. NOT_RUN
    (for both methods) when the GP record is NOT_RUN or fewer than ``min_members`` C1 members are usable
    (``member_ok``: one bool per member). Refused when the two sides do not hold the same rows."""
    from .kcoord import k_skm_from_kcom

    held_rows = np.asarray(held_rows)
    if gp_record["fold_sim"] != holdout_sim:
        raise ValueError(f"GP record is for {gp_record['fold_sim']!r}, not {holdout_sim!r}")
    reason = None
    if gp_record["status"] != "OK":
        reason = f"GP: {gp_record['reason']}"
    elif member_ok is not None and int(np.sum(member_ok)) < min_members:
        reason = f"C1: {int(np.sum(member_ok))} usable members (< {min_members})"
    if reason:
        return {"fold": holdout_sim, "status": "NOT_RUN", "reason": reason}
    gp_rows = np.asarray(gp_record["rows_held"]).ravel()
    gp_mean = np.asarray(gp_record["mean"], dtype=np.float64)
    gp_mean = gp_mean.reshape(-1, gp_mean.shape[-1])
    if sorted(gp_rows.tolist()) != sorted(held_rows.tolist()):
        raise ValueError(f"{holdout_sim}: the GP record and C1 hold different rows")
    n = gp_mean.shape[1]
    pos = {int(r): i for i, r in enumerate(gp_rows)}
    gp = gp_mean[[pos[int(r)] for r in held_rows]]
    T = np.asarray(d["P_tier_p"], dtype=np.float64)[held_rows, :n]
    ih, io = PARAM_NAMES.index("hub"), PARAM_NAMES.index("omegamh2")
    k = np.stack(
        [
            np.asarray(
                k_skm_from_kcom(
                    np.asarray(k_com_hmpc, np.float64)[:n],
                    float(d["z_grid"][r]),
                    float(d["params"][r][ih]),
                    float(d["params"][r][io]),
                )
            )
            for r in held_rows
        ]
    )
    z = np.repeat(np.asarray(d["z_grid"], dtype=np.float64)[held_rows][:, None], n, axis=1)
    return {
        "fold": holdout_sim,
        "status": "OK",
        "e_c1": (np.asarray(P_c1, np.float64)[:, :n] / T - 1.0).ravel(),
        "e_gp": (gp / T - 1.0).ravel(),
        "k_skm": k.ravel(),
        "z": z.ravel(),
    }


def check_matched_design(d):
    """Refuse an incomplete matched design: every simulation must hold exactly one row per (PRIYA rung, PRIYA
    redshift). Returns the simulation names."""
    m = matched_mask(d)
    sims = np.asarray(d["sim_name"])[m]
    keys = list(zip(sims, np.asarray(d["alpha_idx"])[m], np.round(np.asarray(d["z_grid"])[m], 1)))
    if len(set(keys)) != len(keys):
        raise ValueError("a (simulation, rung, z) row of the matched design appears twice")
    want = len(PRIYA_ALPHA_IDX) * len(PRIYA_Z)
    names, counts = np.unique(sims, return_counts=True)
    short = [(str(n), int(c)) for n, c in zip(names, counts) if c != want]
    if short:
        raise ValueError(
            f"matched design incomplete: {len(short)} simulations without {want} rows, e.g. {short[:3]}"
        )
    return [str(n) for n in names]


def assemble_fold(
    d,
    holdout_sim,
    members,
    rec,
    *,
    cache_sha256,
    expected_members=5,
    truth_weights=False,
    predict=None,
):
    """One fold's paired errors under the ensemble and failure rules. ``members``: [(directory, fold_json, meta,
    (model, norm_stats))] of the fold's C1 members; ``rec``: the GP refit record, or ``{"status": "NOT_RUN", ...}``
    when its job left none. Refused (ValueError) when a directory is given twice, there are more members than
    expected, a member was trained on another cache or holds out another simulation, the members mix training arms,
    k grids or fold rows, or the GP's training rows are not C1's fit and early-stopping rows. A missing member, or one
    with a non-finite prediction on a compared mode, counts as dropped; fewer than 4 usable members or a GP NOT_RUN make the fold
    NOT_RUN for both methods. ``predict`` defaults to ``predict_fold``. ``cache_sha256``, when not None, is taken only
    as an exact str (``safe_io.exact_hex64``; a str subclass could pass the cache tie for any member).
    """
    from .safe_io import require_exact_hex64

    if cache_sha256 is not None:
        require_exact_hex64(cache_sha256, "assemble_fold: cache_sha256", ValueError)
    predict = predict or predict_fold
    real = [os.path.realpath(m[0]) for m in members]
    if len(set(real)) != len(real):
        raise ValueError("a member directory is given twice")
    if len(members) > expected_members:
        raise ValueError(
            f"{len(members)} member directories, more than the expected {expected_members}"
        )
    arms = set()
    for md, fj, meta, _ in members:
        if meta.get("cache_sha256") != cache_sha256:
            raise ValueError(f"{md}: trained on another cache")
        if fj["holdout_sim"] != holdout_sim:
            raise ValueError(f"{md}: holds out {fj['holdout_sim']!r}, not {holdout_sim!r}")
        arms.add("step_matched" if (meta.get("recipe") or {}).get("step_matched") else "verbatim")
    if len(arms) > 1:
        raise ValueError(f"the members mix training arms: {sorted(arms)}")
    for md, fj, meta, _ in members[1:]:
        f0, m0 = members[0][1], members[0][2]
        if meta["k_com_hmpc"] != m0["k_com_hmpc"] or any(
            fj[k] != f0[k] for k in ("held_rows", "fit_rows", "val_rows", "delta_c_coeffs")
        ):
            raise ValueError(
                f"{md}: not of the same fold as {members[0][0]} (rows, delta_c or k grid differ)"
            )
    ok = [False] * expected_members
    if members:
        f0, m0 = members[0][1], members[0][2]
        held, coeffs = np.asarray(f0["held_rows"]), np.asarray(f0["delta_c_coeffs"])
        if rec["status"] == "OK" and sorted(np.ravel(rec["rows_train"]).tolist()) != sorted(
            f0["fit_rows"] + f0["val_rows"]
        ):
            raise ValueError("the GP's training rows are not C1's fit and early-stopping rows")
        nets = [m[3] for m in members]
        n_cmp = (
            np.asarray(rec["mean"]).shape[-1] if rec["status"] == "OK" else None
        )  # the compared modes
        ok[: len(nets)] = [
            bool(
                np.all(
                    np.isfinite(
                        predict([n], d, held, coeffs, truth_weights=truth_weights)[0][:, :n_cmp]
                    )
                )
            )
            for n in nets
        ]
        use = [n for n, good in zip(nets, ok) if good]
        P = predict(use, d, held, coeffs, truth_weights=truth_weights)[0] if use else None
        out = fold_errors(d, holdout_sim, held, P, rec, m0["k_com_hmpc"], member_ok=ok)
    else:
        out = {
            "fold": holdout_sim,
            "status": "NOT_RUN",
            "reason": "C1: no member directory (0 usable members)",
        }
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in out.items()}
    out.update(
        member_ok=ok,
        expected_members=expected_members,
        arm=next(iter(arms)) if arms else None,
        gp_unconverged_z=[
            p["z"] for p in rec.get("per_z", []) if not p.get("converged_final", False)
        ],
        weights="truth (context only)" if truth_weights else "head_a_per_fold",
    )
    return out


GP_CACHE_KEYS = (
    "params",
    "param_names",
    "sim_name",
    "z_grid",
    "alpha_idx",
    "alpha_slope",
    "P_tier_p",
)


def read_cache_for_gp(path):
    """The cache arrays ``gp_fold_inputs`` needs, read with h5py (the module needs the emulator environment: the
    ``hcd_analysis.emulator`` package imports JAX)."""
    import h5py

    with h5py.File(path, "r") as h:
        c = {k: h[k][...] for k in GP_CACHE_KEYS}
    for k in ("param_names", "sim_name"):
        c[k] = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in c[k]])
    return c


def gp_fold_inputs(c, holdout_sim, n_modes):
    """The GP side of one fold (``adapters/lyaemu_loo_refit.py``): per PRIYA redshift (ascending), the matched rows of
    the training simulations as X = [rung, the 9 parameters] and Y = P_tier_p on modes 1..``n_modes``, and the held-out
    simulation's rows as X_held, Y_held (the truth) and their cache row indices. Every redshift holds the same
    (simulation, rung) rows in the same order, as upstream's per-redshift GPs do; refused otherwise, or when a used
    value is not finite."""
    m = matched_mask(c)
    sims, z, aidx = np.asarray(c["sim_name"]), np.asarray(c["z_grid"]), np.asarray(c["alpha_idx"])
    P = np.asarray(c["P_tier_p"], dtype=np.float64)[:, : int(n_modes)]
    theta = np.column_stack(
        [np.asarray(c["alpha_slope"], np.float64), np.asarray(c["params"], np.float64)]
    )
    X_tr, Y_tr, X_h, Y_h, rows_h, rows_t, key0 = [], [], [], [], [], [], None
    for zz in PRIYA_Z:
        rows = np.flatnonzero(m & (np.abs(z - zz) < 1e-6))
        rows = rows[np.lexsort((aidx[rows], sims[rows]))]  # by simulation, then rung
        held, train = rows[sims[rows] == holdout_sim], rows[sims[rows] != holdout_sim]
        if held.size != len(PRIYA_ALPHA_IDX):
            raise ValueError(
                f"z={zz}: {held.size} held-out rows of {holdout_sim!r}, expected {len(PRIYA_ALPHA_IDX)}"
            )
        key = list(zip(sims[train], aidx[train]))
        if key0 is None:
            key0 = key
        elif key != key0:
            raise ValueError(f"z={zz}: the training rows differ from z={PRIYA_Z[0]}")
        if not (np.all(np.isfinite(P[rows])) and np.all(np.isfinite(theta[rows]))):
            raise ValueError(f"z={zz}: non-finite training or truth values on modes 1..{n_modes}")
        rows_t.append(train)
        X_tr.append(theta[train])
        Y_tr.append(P[train])
        X_h.append(theta[held])
        Y_h.append(P[held])
        rows_h.append(held)
    return {
        "X_train": np.stack(X_tr),
        "Y_train": np.stack(Y_tr),
        "X_held": np.stack(X_h),
        "Y_held": np.stack(Y_h),
        "rows_held": np.stack(rows_h),
        "rows_train": np.stack(rows_t),
        "z": np.asarray(PRIYA_Z, dtype=np.float64),
        "param_names": np.asarray(c["param_names"]),
        "holdout_sim": np.asarray(holdout_sim),
        "train_sims": np.array([k[0] for k in key0]),
        "train_alpha_idx": np.array([k[1] for k in key0]),
    }


def write_gp_fold_inputs(path, inputs):
    """Write ``gp_fold_inputs`` to a new .npz (never overwrites; no object arrays, so it loads with
    allow_pickle=False)."""
    with open(path, "xb") as f:
        np.savez(f, **inputs)


def step_matched_epochs(n_fit_rows, *, batch_size=512, production_steps=PRODUCTION_STEPS):
    """Epochs of the pre-registered step-matched arm: the production epoch count scaled by the ratio of the production
    optimizer steps to a matched fold's steps at that epoch count, rounded down, i.e.
    floor(production_steps / ceil(n_fit_rows / batch_size))."""
    per_epoch = -(-int(n_fit_rows) // int(batch_size))
    return int(production_steps) // per_epoch
