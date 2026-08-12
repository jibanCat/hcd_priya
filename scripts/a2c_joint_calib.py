#!/usr/bin/env python3
"""Direction-resolved joint calibration test (PREREG v2, 2026-08-11; PI-authorized
execution). POPULATION-ONLY: reads the committed A2c DESI and A1c eBOSS N=48 pkls and
committed gate/diagnostic JSONs. NO sampling, NO likelihood evaluation, NO model code.

Frozen design: notes 2026-08-11-JOINT-CALIBRATION-PREREG-v2.md. This module freezes
two implementation interpretations DECIDED BEFORE ANY REAL-DATA CONTACT (documented
also in the review record):
  (I1) the sweep uses PER-PLANE 2-dim whitening (each plane is its own "space in
       use", prereg section 3): smaller Wishart noise than 7-dim whitening at L=75
       and no 7x7 inversions; the identity E[z z^T | y] = I_2 holds per plane.
  (I2) seed streams: numpy SeedSequence(20260811).spawn(4) assigned in the FROZEN
       order [null-DESI, null-eBOSS, boot-DESI, boot-eBOSS].

Statistics (P3 = (ns, tau0_amp, dtau0)):
  T_max/T_min: extreme eigenvalues of S = (1/N) sum z_i z_i^T, Holm m=2
  T_corr: max |off-diagonal| of corr(S), upper tail (rotation conjunct, not Holm)
  T_det: logdet(S), central 95% band (specificity discriminator)
  sweep: T_pmax/T_pmax over the 18 external planes of P7, Holm m=2 pair
Null: posterior-draw exchangeability, ONE shared draw index per mock per replicate,
LOO moment downdate from FULL-sample moments, B=20000, p=(r+1)/(B+1).
Bootstrap stability: mock-level resampling, B_boot=2000.
"""
import glob
import hashlib
import itertools
import json
import os
import pickle

import numpy as np

N_TOTAL = 48
RTOL = 1e-9
B_NULL = 20_000
B_BOOT = 2_000
SEED_ROOT = 20260811
ALPHA = 0.05

P3 = ("ns", "tau0_amp", "dtau0")
P7 = ("ns", "Ap", "tau0_amp", "dtau0", "alpha_dla", "alpha_lls", "alpha_subdla")
MAIN_CHANNELS = ("ns", "Ap", "alpha_dla", "alpha_lls", "alpha_subdla")
EXTRA_CHANNELS = ("tau0_amp", "dtau0")
P3_IDX = tuple(P7.index(c) for c in P3)
# the 18 external planes: all P7 pairs except the 3 internal to P3
ALL_PAIRS = tuple(itertools.combinations(range(len(P7)), 2))
INTERNAL = {tuple(sorted(p)) for p in itertools.combinations(P3_IDX, 2)}
EXternal_PLANES = tuple(p for p in ALL_PAIRS if tuple(sorted(p)) not in INTERNAL)
NEAR_BOUND_MOCKS = (29, 34, 47)   # frozen descriptive leave-out (prereg 5.5)


class JointCalibRefusal(RuntimeError):
    """Integrity/deployment-consistency failure: no output."""


# ------------------------------ loading + conjuncts ------------------------------

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_arm(outdir, sha_file, n_total=N_TOTAL):
    """Census + sha conjuncts; returns per-mock dicts with the P7 draw block."""
    found = sorted(os.path.basename(p) for p in glob.glob(os.path.join(outdir, "mock_*.pkl"))
                   if not p.endswith(".smoke.pkl"))
    expect = [f"mock_{m:04d}.pkl" for m in range(n_total)]
    if found != expect:
        raise JointCalibRefusal(f"census != mock_0000..{n_total-1:04d} (got {len(found)})")
    recorded = {}
    with open(sha_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise JointCalibRefusal(f"malformed sha line: {line!r}")
            recorded[os.path.basename(parts[1])] = parts[0]
    recs = []
    for m in range(n_total):
        p = os.path.join(outdir, f"mock_{m:04d}.pkl")
        base = os.path.basename(p)
        if base not in recorded:
            raise JointCalibRefusal(f"{base} absent from sha manifest")
        if _sha256(p) != recorded[base]:
            raise JointCalibRefusal(f"sha mismatch on {base}")
        with open(p, "rb") as f:
            z = pickle.load(f)
        names = list(z["names"])
        draws = np.asarray(z["draws"], float)
        truth = np.asarray(z["truth_vec"], float)
        L = int(z["L"])
        if draws.shape[0] != L:
            raise JointCalibRefusal(f"mock {m}: len(draws) {draws.shape[0]} != L {L}")
        X = np.empty((L, len(P7)))
        t = np.empty(len(P7))
        for c, ch in enumerate(P7):
            if ch in MAIN_CHANNELS:
                j = names.index(ch)
                X[:, c] = draws[:, j]
                t[c] = truth[j]
            else:
                se = z["sites_extra"][ch]
                col = np.asarray(se["draws"], float)
                if col.shape[0] != L:
                    raise JointCalibRefusal(
                        f"mock {m}: sites_extra[{ch}] rows {col.shape[0]} != L {L} "
                        "(alignment conjunct)")
                X[:, c] = col
                t[c] = float(se["truth"])
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(t))):
            raise JointCalibRefusal(f"mock {m}: non-finite draws/truth")
        recs.append(dict(m=m, X=X, t=t, L=L, truth_full=truth))
    return recs


def check_consistency(recs, gate_json, leg, esc_diag_json=None):
    """Deployment-consistency conjuncts at rtol 1e-9 (prereg section 8)."""
    with open(gate_json) as f:
        legrec = json.load(f)["legs"][leg]
    Ls = [r["L"] for r in recs]
    if list(map(int, legrec["L_all"])) != Ls:
        raise JointCalibRefusal("L census != gate JSON L_all")
    checks = []
    for ch, jkey in (("ns", "ns"), ("Ap", "Ap")):
        c = P7.index(ch)
        pulls = np.array([(r["X"][:, c].mean() - r["t"][c]) / r["X"][:, c].std(ddof=1)
                          for r in recs])
        checks.append((f"{jkey} pull mean", float(pulls.mean()),
                       float(legrec["pulls"][jkey]["mean"])))
        checks.append((f"{jkey} pull sd", float(pulls.std(ddof=1)),
                       float(legrec["pulls"][jkey]["std"])))
    if esc_diag_json is not None:
        with open(esc_diag_json) as f:
            esc = json.load(f)
        for ch, key in (("tau0_amp", "tau0_amp_sampled"), ("dtau0", "dtau0_sampled")):
            c = P7.index(ch)
            pulls = np.array([(r["X"][:, c].mean() - r["t"][c]) / r["X"][:, c].std(ddof=1)
                              for r in recs])
            checks.append((f"{ch} sampled pull sd", float(pulls.std(ddof=1)),
                           float(esc["P1"]["sampled_sites"][key]["pull_sd"])))
    for label, got, want in checks:
        if not np.isclose(got, want, rtol=RTOL, atol=0.0):
            raise JointCalibRefusal(
                f"deployment-consistency FAIL on {label}: {got!r} vs {want!r}")
    return {label: got for label, got, _ in checks}


def check_truth_pairing(recs_a, recs_b):
    """The two arms share identical truth vectors per mock (prereg section 8)."""
    for ra, rb in zip(recs_a, recs_b):
        if not np.array_equal(ra["truth_full"], rb["truth_full"]):
            raise JointCalibRefusal(f"truth pairing FAIL on mock {ra['m']}")


# ------------------------------ core linear algebra ------------------------------

def sym_inv_sqrt(S):
    w, V = np.linalg.eigh(S)
    if np.min(w) <= 0.0:
        raise JointCalibRefusal("non-positive-definite posterior covariance")
    return (V * (1.0 / np.sqrt(w))) @ V.T


def observed_z(X, t, cols):
    """Whitened truth displacement in the subspace `cols` (full-sample moments)."""
    Y = X[:, cols]
    mu = Y.mean(axis=0)
    S = np.cov(Y.T, ddof=1).reshape(len(cols), len(cols))
    return sym_inv_sqrt(S) @ (mu - t[cols])


def loo_z_table(X, t_cols_unused, cols):
    """(L, d) table: row j = whitened displacement using pseudo-truth = draw j and
    LOO moments excluding row j (closed-form rank-1 downdate of full moments)."""
    Y = X[:, cols]
    L, d = Y.shape
    mu = Y.mean(axis=0)
    Sig = np.cov(Y.T, ddof=1).reshape(d, d)
    A = (L - 1) * Sig                       # scatter matrix sum (x-mu)(x-mu)^T
    out = np.empty((L, d))
    for j in range(L):
        xj = Y[j]
        mu_j = (L * mu - xj) / (L - 1)
        dev = xj - mu
        A_j = A - (L / (L - 1)) * np.outer(dev, dev)
        Sig_j = A_j / (L - 2)
        out[j] = sym_inv_sqrt(Sig_j) @ (mu_j - xj)
    return out


def eig22_batch(S2):
    """Closed-form eigenvalues of stacked symmetric 2x2 matrices (B,2,2)."""
    a, b, c = S2[..., 0, 0], S2[..., 1, 1], S2[..., 0, 1]
    tr, disc = a + b, np.sqrt(((a - b) * 0.5) ** 2 + c ** 2)
    return tr * 0.5 - disc, tr * 0.5 + disc     # (lo, hi)


def stats_p3(S3):
    """T_max, T_min, T_det, T_corr for stacked (B,3,3) or single (3,3)."""
    single = S3.ndim == 2
    S = S3[None] if single else S3
    w = np.linalg.eigvalsh(S)
    tmax, tmin = w[:, -1], w[:, 0]
    tdet = np.sum(np.log(w), axis=1)
    dd = np.sqrt(np.diagonal(S, axis1=-2, axis2=-1))
    C = S / (dd[:, :, None] * dd[:, None, :])
    tcorr = np.max(np.abs(np.stack([C[:, 0, 1], C[:, 0, 2], C[:, 1, 2]], axis=1)), axis=1)
    if single:
        return float(tmax[0]), float(tmin[0]), float(tdet[0]), float(tcorr[0])
    return tmax, tmin, tdet, tcorr


def p_upper(null, obs):
    return float(((null >= obs).sum() + 1) / (null.size + 1))


def p_lower(null, obs):
    return float(((null <= obs).sum() + 1) / (null.size + 1))


def holm_pair(p_a, p_b, alpha=ALPHA):
    """Holm m=2: returns (fires_a, fires_b)."""
    if p_a <= p_b:
        fa = p_a < alpha / 2
        fb = fa and (p_b < alpha)
    else:
        fb = p_b < alpha / 2
        fa = fb and (p_a < alpha)
    return bool(fa), bool(fb)


# ------------------------------ per-arm analysis ------------------------------

def analyze_arm(recs, rng_null, rng_boot, B_null=None, B_boot=None):
    if B_null is None:
        B_null = B_NULL      # module attribute read at call time (testable)
    if B_boot is None:
        B_boot = B_BOOT
    N = len(recs)
    # observed
    z3 = np.stack([observed_z(r["X"], r["t"], list(P3_IDX)) for r in recs])   # (N,3)
    S3_obs = (z3.T @ z3) / N
    tmax_o, tmin_o, tdet_o, tcorr_o = stats_p3(S3_obs)
    w_obs, V_obs = np.linalg.eigh(S3_obs)      # ascending
    v_max, v_min = V_obs[:, -1].copy(), V_obs[:, 0].copy()
    for v in (v_max, v_min):                   # frozen sign convention
        if v[np.argmax(np.abs(v))] < 0:
            v *= -1.0
    zpl_obs = {}
    for p in EXternal_PLANES:
        zpl_obs[p] = np.stack([observed_z(r["X"], r["t"], list(p)) for r in recs])
    def sweep_stats(zpl_by_plane):
        los, his = [], []
        for p in EXternal_PLANES:
            Z = zpl_by_plane[p]
            S2 = np.einsum("ni,nj->ij", Z, Z) / N
            lo, hi = eig22_batch(S2[None])
            los.append(lo[0]); his.append(hi[0])
        return float(np.min(los)), float(np.max(his))
    tpmin_o, tpmax_o = sweep_stats(zpl_obs)

    # null: precompute LOO tables, then vectorized replicates with SHARED index
    Z3_loo = [loo_z_table(r["X"], None, list(P3_IDX)) for r in recs]
    Zpl_loo = {p: [loo_z_table(r["X"], None, list(p)) for r in recs]
               for p in EXternal_PLANES}
    Ls = np.array([r["L"] for r in recs])
    J = np.stack([rng_null.integers(0, L, size=B_null) for L in Ls], axis=1)  # (B,N)
    G3 = np.stack([Z3_loo[i][J[:, i]] for i in range(N)], axis=1)             # (B,N,3)
    S3_b = np.einsum("bni,bnj->bij", G3, G3) / N
    tmax_b, tmin_b, tdet_b, tcorr_b = stats_p3(S3_b)
    tpmax_b = np.full(B_null, -np.inf)
    tpmin_b = np.full(B_null, np.inf)
    for p in EXternal_PLANES:
        Gp = np.stack([Zpl_loo[p][i][J[:, i]] for i in range(N)], axis=1)     # (B,N,2)
        S2_b = np.einsum("bni,bnj->bij", Gp, Gp) / N
        lo, hi = eig22_batch(S2_b)
        tpmax_b = np.maximum(tpmax_b, hi)
        tpmin_b = np.minimum(tpmin_b, lo)

    p_tmax, p_tmin = p_upper(tmax_b, tmax_o), p_lower(tmin_b, tmin_o)
    p_tcorr = p_upper(tcorr_b, tcorr_o)
    det_band = (float(np.quantile(tdet_b, 0.025)), float(np.quantile(tdet_b, 0.975)))
    det_central = bool(det_band[0] <= tdet_o <= det_band[1])
    p_tpmax, p_tpmin = p_upper(tpmax_b, tpmax_o), p_lower(tpmin_b, tpmin_o)
    f_tmax, f_tmin = holm_pair(p_tmax, p_tmin)
    f_tpmax, f_tpmin = holm_pair(p_tpmax, p_tpmin)

    # bootstrap stability (observed z3, mock resampling)
    idx = rng_boot.integers(0, N, size=(B_boot, N))
    cos_max = np.empty(B_boot)
    cos_min = np.empty(B_boot)
    for b in range(B_boot):
        Zb = z3[idx[b]]
        Sb = (Zb.T @ Zb) / N
        wb, Vb = np.linalg.eigh(Sb)
        cos_max[b] = abs(float(Vb[:, -1] @ v_max))
        cos_min[b] = abs(float(Vb[:, 0] @ v_min))

    # descriptive: null-replicate eigenvector |cos| (NEVER a condition)
    Vn = np.linalg.eigh(S3_b)[1]
    null_cos_max = np.abs(np.einsum("bi,i->b", Vn[:, :, -1], v_max))

    # descriptive: near-bound leave-out (no p-values)
    keep = [i for i in range(N) if recs[i]["m"] not in NEAR_BOUND_MOCKS]
    z3_lo = z3[keep]
    S3_lo = (z3_lo.T @ z3_lo) / len(keep)
    lo_stats = stats_p3(S3_lo)

    arm = dict(
        n=N, L_min=int(Ls.min()), L_median=float(np.median(Ls)), L_max=int(Ls.max()),
        S3=S3_obs.tolist(),
        primary=dict(
            T_max=dict(observed=tmax_o, p=p_tmax, holm_fires=f_tmax,
                       null_q=[float(np.quantile(tmax_b, q)) for q in (0.05, 0.5, 0.95)]),
            T_min=dict(observed=tmin_o, p=p_tmin, holm_fires=f_tmin,
                       null_q=[float(np.quantile(tmin_b, q)) for q in (0.05, 0.5, 0.95)]),
            holm=dict(m=2, alpha=ALPHA)),
        T_corr=dict(observed=tcorr_o, p=p_tcorr, fires=bool(p_tcorr < ALPHA),
                    null_q=[float(np.quantile(tcorr_b, q)) for q in (0.05, 0.5, 0.95)]),
        T_det=dict(observed=tdet_o, central_band=det_band, inside=det_central),
        sweep=dict(planes=[[P7[a], P7[b]] for a, b in EXternal_PLANES],
                   T_pmax=dict(observed=tpmax_o, p=p_tpmax, holm_fires=f_tpmax),
                   T_pmin=dict(observed=tpmin_o, p=p_tpmin, holm_fires=f_tpmin)),
        directions=dict(
            coords=list(P3),
            v_max=v_max.tolist(), v_min=v_min.tolist(),
            eigenvalues_desc=[float(w_obs[-1]), float(w_obs[1]), float(w_obs[0])],
            boot_cos_max=dict(median=float(np.median(cos_max)),
                              q05=float(np.quantile(cos_max, 0.05)),
                              q95=float(np.quantile(cos_max, 0.95)),
                              frac_ge_0p7=float(np.mean(cos_max >= 0.7))),
            boot_cos_min=dict(median=float(np.median(cos_min)),
                              q05=float(np.quantile(cos_min, 0.05)),
                              q95=float(np.quantile(cos_min, 0.95)),
                              frac_ge_0p7=float(np.mean(cos_min >= 0.7))),
            null_replicate_cos_max_median_DESCRIPTIVE=float(np.median(null_cos_max))),
        descriptive=dict(
            S3_diag=[float(S3_obs[i, i]) for i in range(3)],
            proj_vmax=(z3 @ v_max).tolist(), proj_vmin=(z3 @ v_min).tolist(),
            near_bound_leave_out=dict(mocks_removed=list(NEAR_BOUND_MOCKS),
                                      T_max=lo_stats[0], T_min=lo_stats[1],
                                      T_det=lo_stats[2], T_corr=lo_stats[3],
                                      note="DESCRIPTIVE ONLY, no p-value")),
        B_null=B_null, B_boot=B_boot)
    arm["fires_any_primary_or_det"] = bool(f_tmax or f_tmin or not det_central)
    return arm, z3


def classify(desi, eboss):
    """Frozen outcome logic (prereg sections 11-12)."""
    eboss_fires = eboss["fires_any_primary_or_det"]
    d = desi
    rot = (d["primary"]["T_max"]["holm_fires"] and d["primary"]["T_min"]["holm_fires"]
           and d["T_det"]["inside"] and d["T_corr"]["fires"]
           and d["directions"]["boot_cos_max"]["median"] >= 0.7
           and d["directions"]["boot_cos_min"]["median"] >= 0.7
           and not eboss_fires)
    if eboss_fires:
        return "E_control_fires"
    if rot:
        return "A_rotation_established"
    if d["primary"]["T_max"]["holm_fires"] or d["primary"]["T_min"]["holm_fires"]:
        return "B_anisotropy_no_rotation_word"
    if d["sweep"]["T_pmax"]["holm_fires"] or d["sweep"]["T_pmin"]["holm_fires"]:
        return "C_sweep_only"
    return "D_nothing_fires"


def run(desi_dir, desi_sha, desi_gate, esc_diag_json,
        eboss_dir, eboss_sha, eboss_gate, out_json):
    ss = np.random.SeedSequence(SEED_ROOT)
    kids = ss.spawn(4)     # FROZEN order: null-DESI, null-eBOSS, boot-DESI, boot-eBOSS
    rngs = [np.random.default_rng(k) for k in kids]
    recs_d = load_arm(desi_dir, desi_sha)
    recs_e = load_arm(eboss_dir, eboss_sha)
    check_truth_pairing(recs_d, recs_e)
    cons_d = check_consistency(recs_d, desi_gate, "DESI", esc_diag_json=esc_diag_json)
    cons_e = check_consistency(recs_e, eboss_gate, "eBOSS")
    desi, z3_d = analyze_arm(recs_d, rngs[0], rngs[2])
    eboss, z3_e = analyze_arm(recs_e, rngs[1], rngs[3])
    # descriptive truth-paired per-mock contrast
    norm_ratio = np.linalg.norm(z3_d, axis=1) / np.linalg.norm(z3_e, axis=1)
    cosv = np.einsum("ni,ni->n", z3_d, z3_e) / (
        np.linalg.norm(z3_d, axis=1) * np.linalg.norm(z3_e, axis=1))
    out = dict(
        prereg="2026-08-11-JOINT-CALIBRATION-PREREG-v2", seed_root=SEED_ROOT,
        consistency=dict(DESI=cons_d, eBOSS=cons_e),
        DESI=desi, eBOSS=eboss,
        paired_contrast_DESCRIPTIVE=dict(
            norm_ratio_median=float(np.median(norm_ratio)),
            norm_ratio_q25=float(np.quantile(norm_ratio, 0.25)),
            norm_ratio_q75=float(np.quantile(norm_ratio, 0.75)),
            cos_median=float(np.median(cosv))),
        outcome=classify(desi, eboss))
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    return out
