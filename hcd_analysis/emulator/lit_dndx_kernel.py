"""PRIYA-CDDF cache kernel for the cumulative->binned LLS estimand correction
(corrected-law re-derivation, spec 2026-07-18 sec 3 / step 5).

The kernel corrects ONLY the LLS estimand: cumulative tau>=2 l(X)(>=17.5) -> binned
[17.2, 19.0). PINNED K1 RECIPE (the adjudication-of-record definition):
  b(z) = l_X([17.2,19.0)) / l_X(>=17.5), per snap-group:
    numerator   = ``snap_dNdX_LLS`` (exact class edges — verified IDENTICAL to the
                  f_nhi bin integral over [17.2,19.0) in this cache, median ratio 1.0000)
    denominator = ``snap_dNdX_LLS`` - l([17.2,17.5)) + ``snap_dNdX_subDLA`` +
                  ``snap_dNdX_DLA``
  with the floor piece l([17.2,17.5)) from integrating ``snap_f_nhi`` over [17.2,17.5)
  using LOCAL POWER-LAW-IN-N sub-bin interpolation (MANDATORY: 17.5 is a bin CENTRE of
  the 0.2-dex grid, edges 17.4/17.6 — never raw bin-centre slicing).
  Aggregate = per-sim ratio then suite MEDIAN at fixed z; r(z) interpolated linearly in
  ln(1+z) to each literature z_bar.

ADJUDICATION (the 0.88-vs-0.96 same-cache ambiguity, spec step 6): computed by
``adjudication()`` — the sibling's ``_lx_above`` construction (const-f partial bin) is
reproduced bit-for-bit (0.834@z2.4 / 0.877@z3.0), and the impact-team A1 ~0.96 family is
reproduced by the naive 17.5->17.6 EDGE-SNAP floor (over-removal of the whole
[17.4,17.6) bin from the denominator) — a bin-centre-trap artifact, not a valid recipe.

numpy + h5py only (no JAX); datasets are read lazily (a few MB of the ~1 GB cache).
"""
from __future__ import annotations

import hashlib

import numpy as np
import h5py

CACHE_PATH = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"

_KERNEL_DATASETS = ("snap_dNdX_LLS", "snap_dNdX_subDLA", "snap_dNdX_DLA",
                    "snap_f_nhi", "snap_total_path_dX", "log_nhi_edges")


def load_kernel_inputs(path=CACHE_PATH):
    """Read ONLY the kernel-relevant datasets (lazy; never the P1D arrays). Returns a dict
    with per-group z, sim name, class columns, f_nhi, edges, and cache provenance
    (dataset shapes + sha256 of the f_nhi bytes)."""
    with h5py.File(path, "r") as h:
        lls = h["snap_dNdX_LLS"][:].astype(float)
        sub = h["snap_dNdX_subDLA"][:].astype(float)
        dla = h["snap_dNdX_DLA"][:].astype(float)
        f_nhi = h["snap_f_nhi"][:].astype(float)
        edges = h["log_nhi_edges"][:].astype(float)
        total_dX = h["snap_total_path_dX"][:].astype(float)
        gid = h["snap_group_idx"][:]
        zrow = h["z_grid"][:].astype(float)
        sim = np.array([s.decode() if isinstance(s, bytes) else s
                        for s in h["snap_sim_name"][:]])
    ng = lls.shape[0]
    order = np.argsort(gid, kind="stable")
    firsts = order[np.searchsorted(gid[order], np.arange(ng))]
    zg = zrow[firsts]
    return dict(path=path, zg=zg, sim=sim, lls=lls, sub=sub, dla=dla, f_nhi=f_nhi,
                edges=edges, total_dX=total_dX,
                sha256_f_nhi=hashlib.sha256(np.ascontiguousarray(f_nhi).tobytes()).hexdigest(),
                shapes=dict(snap_f_nhi=tuple(f_nhi.shape), log_nhi_edges=tuple(edges.shape),
                            snap_dNdX_LLS=tuple(lls.shape)))


# --------------------------------------------------------------------------- #
#  Floor piece l([17.2,17.5)) — the 17.5-is-a-bin-CENTRE trap                  #
# --------------------------------------------------------------------------- #
def floor_piece_17p2_17p5(f_nhi, edges, method="subbin_powerlaw"):
    """Integrate the CDDF over [17.2, 17.5) per group. 17.2 IS an edge (index 1); 17.5 is
    the CENTRE of bin [17.4,17.6) (index 2) — the partial piece of that bin needs a
    sub-bin treatment.

    methods:
      subbin_powerlaw (PINNED): local power-law-in-N inside bin 2, slope from the central
        difference of ln f between the neighbour bins, normalized to preserve the bin's
        total; exact on a power-law CDDF.
      const_f: f constant inside the bin (the sibling's ``_lx_above`` construction).
      naive_edge_17p4 / naive_edge_17p6: the bin-centre-trap artifacts (snap 17.5 to the
        nearest edge below/above) — adjudication diagnostics ONLY.
    """
    f = np.asarray(f_nhi, float)
    edges = np.asarray(edges, float)
    assert abs(edges[1] - 17.2) < 1e-9 and abs(edges[2] - 17.4) < 1e-9, edges[:4]
    N = 10.0 ** edges
    full_b1 = f[:, 1] * (N[2] - N[1])                    # [17.2,17.4), exact edges
    N0, N1, Nm = 10.0 ** 17.4, 10.0 ** 17.6, 10.0 ** 17.5
    if method == "naive_edge_17p4":
        return full_b1
    if method == "naive_edge_17p6":
        return full_b1 + f[:, 2] * (N1 - N0)
    if method == "const_f":
        return full_b1 + f[:, 2] * (Nm - N0)
    if method != "subbin_powerlaw":
        raise ValueError(f"unknown method {method!r}")
    # local power-law slope in N: central difference of ln f between bins 1 and 3
    lnNc = np.log(10.0 ** (0.5 * (edges[:-1] + edges[1:])))
    with np.errstate(divide="ignore", invalid="ignore"):
        lnf1 = np.log(f[:, 1])
        lnf3 = np.log(f[:, 3])
        a = (lnf3 - lnf1) / (lnNc[3] - lnNc[1])
    a = np.where(np.isfinite(a), a, 0.0)                 # fallback: constant f
    ap1 = a + 1.0
    # guard a ~ -1 (log integral); |ap1| floor keeps it well-conditioned
    ap1 = np.where(np.abs(ap1) < 1e-6, 1e-6, ap1)
    denom = N1 ** ap1 - N0 ** ap1
    C = f[:, 2] * (N1 - N0) * ap1 / denom                # preserve the bin total
    partial = C * (Nm ** ap1 - N0 ** ap1) / ap1
    return full_b1 + partial


def _lx_fnhi_range(f_nhi, edges, i_lo, i_hi):
    """f_nhi integral over FULL bins [i_lo, i_hi) (both exact edges)."""
    f = np.asarray(f_nhi, float)
    N = 10.0 ** np.asarray(edges, float)
    dN = np.diff(N)
    return (f[:, i_lo:i_hi] * dN[None, i_lo:i_hi]).sum(axis=1)


def lx_above_17p5_sibling(f_nhi, edges):
    """The sibling's ``_lx_above`` construction: l(>=17.5) from f_nhi with the partial
    bin [17.5,17.6) integrated as const-f (f2*(10^17.6-10^17.5)) + full bins above."""
    f = np.asarray(f_nhi, float)
    part = f[:, 2] * (10.0 ** 17.6 - 10.0 ** 17.5)
    return part + _lx_fnhi_range(f_nhi, edges, 3, f.shape[1])


# --------------------------------------------------------------------------- #
#  Per-z aggregation + interpolation                                           #
# --------------------------------------------------------------------------- #
def _per_z(zg, vals, aggregate="median"):
    """Aggregate a per-group array at fixed z (per-sim ratio -> suite median/mean).
    Returns (z_vals, agg, p16, p84, n)."""
    z_vals = np.unique(np.round(zg, 6))
    agg = np.empty_like(z_vals)
    p16 = np.empty_like(z_vals)
    p84 = np.empty_like(z_vals)
    n = np.empty(len(z_vals), int)
    for i, zt in enumerate(z_vals):
        sel = np.isclose(zg, zt, atol=1e-6)
        v = vals[sel]
        v = v[np.isfinite(v)]
        agg[i] = np.median(v) if aggregate == "median" else np.mean(v)
        p16[i] = np.percentile(v, 16)
        p84[i] = np.percentile(v, 84)
        n[i] = len(v)
    return z_vals, agg, p16, p84, n


def interp_ln1pz(z_tab, y_tab, z):
    """Linear interpolation in ln(1+z) (the pinned r(z) interpolation)."""
    return np.interp(np.log1p(np.asarray(z, float)), np.log1p(np.asarray(z_tab, float)),
                     np.asarray(y_tab, float))


# --------------------------------------------------------------------------- #
#  Pinned K1                                                                   #
# --------------------------------------------------------------------------- #
def k1_per_group(inputs, floor_method="subbin_powerlaw"):
    """Per-group r = l([17.2,19.0)) / l(>=17.5): num = the LLS class column (exact class
    edges); den = LLS - floor([17.2,17.5)) + subDLA + DLA."""
    fl = floor_piece_17p2_17p5(inputs["f_nhi"], inputs["edges"], method=floor_method)
    num = inputs["lls"]
    den = inputs["lls"] - fl + inputs["sub"] + inputs["dla"]
    return num / den


def k1_table(inputs, floor_method="subbin_powerlaw", aggregate="median"):
    """The pinned K1 kernel table: per-sim ratio then suite MEDIAN at fixed z, with the
    16-84% suite scatter (the kernel-budget statistical part). NOTE: the suite median
    crosses r > 1 above z ~ 4.6 (binned exceeding cumulative there is a floor-share
    artifact of PRIYA's high-z CDDF, outside the literature fit range z <= 4.23) — the
    kernel is only ever EVALUATED at the lit z_bar, so the saturation is cosmetic."""
    r = k1_per_group(inputs, floor_method=floor_method)
    z_vals, agg, p16, p84, n = _per_z(inputs["zg"], r, aggregate=aggregate)
    return dict(z_vals=z_vals, r=agg, r16=p16, r84=p84, n=n,
                floor_method=floor_method, aggregate=aggregate)


def k1_r_at(inputs, z, floor_method="subbin_powerlaw", aggregate="median", _tab=None):
    tab = _tab or k1_table(inputs, floor_method=floor_method, aggregate=aggregate)
    return interp_ln1pz(tab["z_vals"], tab["r"], z)


def k1_smooth_fit(inputs, z_pivot=3.0, z_lo=2.0, z_hi=4.6, _tab=None):
    """K1b: fit ln r(z) linear in ln((1+z)/(1+z_pivot)) over the analysis window.
    Returns (r_pivot, eta) and a callable."""
    tab = _tab or k1_table(inputs)
    sel = (tab["z_vals"] >= z_lo) & (tab["z_vals"] <= z_hi)
    x = np.log((1.0 + tab["z_vals"][sel]) / (1.0 + z_pivot))
    y = np.log(tab["r"][sel])
    coef = np.polyfit(x, y, 1)                            # [eta, ln r_pivot]
    r_p = float(np.exp(coef[1]))
    eta = float(coef[0])

    def r_at(z):
        return r_p * np.exp(eta * np.log((1.0 + np.asarray(z, float)) / (1.0 + z_pivot)))

    return dict(r_pivot=r_p, eta=eta, r_at=r_at, z_window=(z_lo, z_hi))


def floor_factor_table(inputs, floor_method="subbin_powerlaw", aggregate="median"):
    """F_floor(z) = l([17.2,19.0)) / l([17.5,19.0)) per group -> suite median at fixed z
    (the K2/K3 floor borrow; PRIYA measures ~1.40-1.47 over z 2.4-4.2)."""
    fl = floor_piece_17p2_17p5(inputs["f_nhi"], inputs["edges"], method=floor_method)
    F = inputs["lls"] / (inputs["lls"] - fl)
    z_vals, agg, p16, p84, n = _per_z(inputs["zg"], F, aggregate=aggregate)
    return dict(z_vals=z_vals, F=agg, F16=p16, F84=p84, n=n, floor_method=floor_method)


def floor_factor_at(inputs, z, _tab=None):
    tab = _tab or floor_factor_table(inputs)
    return interp_ln1pz(tab["z_vals"], tab["F"], z)


# --------------------------------------------------------------------------- #
#  Kernel uncertainty budget (spec sec 3)                                      #
# --------------------------------------------------------------------------- #
def kernel_budget(inputs, r_k2_at, z_pivot=3.0, _tab=None):
    """sigma_r(3) = quadrature of (i) the K1 suite 16-84% half-spread at z=3 (fractional)
    and (ii) half the K1-K2 spread at z=3 (model-form part, expected dominant).
    sigma_eta from the K1-vs-K2 slope spread: eta_K = d ln r_K / d ln(1+z) at the pivot."""
    tab = _tab or k1_table(inputs)
    i3 = int(np.argmin(np.abs(tab["z_vals"] - z_pivot)))
    r3_k1 = float(tab["r"][i3])
    suite_half = 0.5 * (tab["r84"][i3] - tab["r16"][i3]) / r3_k1
    r3_k2 = float(np.asarray(r_k2_at(np.array([z_pivot])))[0])
    model_half = 0.5 * abs(r3_k1 - r3_k2) / r3_k1
    s_r3 = float(np.hypot(suite_half, model_half))
    # slopes eta = d ln r / d ln(1+z) at the pivot (finite difference for K2; fit for K1)
    eta_k1 = k1_smooth_fit(inputs, z_pivot=z_pivot, _tab=tab)["eta"]
    dz = 0.2
    r_lo, r_hi = (float(np.asarray(r_k2_at(np.array([zz])))[0])
                  for zz in (z_pivot - dz, z_pivot + dz))
    eta_k2 = (np.log(r_hi) - np.log(r_lo)) / (np.log(1 + z_pivot + dz) - np.log(1 + z_pivot - dz))
    sigma_eta = float(0.5 * abs(eta_k1 - eta_k2))
    return dict(s_r3=s_r3, suite_half=float(suite_half), model_half=float(model_half),
                sigma_eta=sigma_eta, eta_k1=float(eta_k1), eta_k2=float(eta_k2),
                r3_k1=r3_k1, r3_k2=r3_k2)


# --------------------------------------------------------------------------- #
#  Same-file consistency substitute (hcd_summary_lf.h5 absent here)            #
# --------------------------------------------------------------------------- #
def same_file_consistency(inputs):
    """Sum of the 3 class columns vs the CDDF-integrated l(>=17.2) from the SAME file.
    Weaker than the sibling's cross-file check; flagged. (Measured: the class columns are
    numerically identical to the f_nhi integrals in this cache — ratio 1.0000 — so this
    check guards transcription, not independent construction.)"""
    csum = inputs["lls"] + inputs["sub"] + inputs["dla"]
    l_int = _lx_fnhi_range(inputs["f_nhi"], inputs["edges"], 1, inputs["f_nhi"].shape[1])
    ratio = csum / l_int
    return dict(median_ratio=float(np.median(ratio)),
                p16=float(np.percentile(ratio, 16)), p84=float(np.percentile(ratio, 84)),
                flag="same-file check only (hcd_summary_lf.h5 absent)")


# --------------------------------------------------------------------------- #
#  Adjudication of the 0.88-vs-0.96 same-cache ambiguity (spec step 6)         #
# --------------------------------------------------------------------------- #
ADJUDICATION_Z = (2.4, 3.0, 3.6, 4.2)


def adjudication(inputs, z_eval=ADJUDICATION_Z):
    """Run BOTH prior recipes and the named variants under the pinned per-z aggregation,
    and attribute the differences (floor sub-bin treatment, mean-vs-median,
    f_nhi-integration-vs-class-columns). All r values at the standard z_eval grid."""
    z_eval = np.asarray(z_eval, float)
    zg = inputs["zg"]

    def per_z_at(vals, aggregate="median"):
        z_vals, agg, _, _, _ = _per_z(zg, vals, aggregate=aggregate)
        return interp_ln1pz(z_vals, agg, z_eval)

    # pinned K1 (class columns; power-law sub-bin floor; median)
    r_pin = k1_per_group(inputs, floor_method="subbin_powerlaw")
    # pinned with const-f floor (the sibling's sub-bin treatment on the class columns)
    r_cf = k1_per_group(inputs, floor_method="const_f")
    # naive edge snaps (the trap)
    r_n174 = k1_per_group(inputs, floor_method="naive_edge_17p4")
    r_n176 = k1_per_group(inputs, floor_method="naive_edge_17p6")
    # sibling _lx_above: pure f_nhi integration for BOTH num and den, const-f partial
    num_f = _lx_fnhi_range(inputs["f_nhi"], inputs["edges"], 1, 10)     # [17.2,19.0)
    den_f = lx_above_17p5_sibling(inputs["f_nhi"], inputs["edges"])
    r_sib = num_f / den_f
    # no-floor-removal candidate (den = l(>=17.2)): NOT any team's number, shown for span
    r_nofloor = inputs["lls"] / (inputs["lls"] + inputs["sub"] + inputs["dla"])

    recipes = {
        "pinned_k1": dict(r=per_z_at(r_pin), note="class columns + subbin powerlaw floor "
                                                  "+ suite median (THE pinned recipe)"),
        "pinned_k1_mean": dict(r=per_z_at(r_pin, "mean"), note="pinned but suite MEAN"),
        "pinned_k1_constf_floor": dict(r=per_z_at(r_cf), note="pinned but const-f partial "
                                                              "bin (sibling sub-bin treatment)"),
        "sibling_lx_above": dict(r=per_z_at(r_sib), note="all-f_nhi integration, const-f "
                                                         "partial (their 0.877@z3 family)"),
        "naive_edge_17p4": dict(r=per_z_at(r_n174), note="17.5 snapped DOWN to 17.4 edge "
                                                         "(under-removal; trap)"),
        "naive_edge_17p6": dict(r=per_z_at(r_n176), note="17.5 snapped UP to 17.6 edge "
                                                         "(over-removal; reproduces the "
                                                         "impact-team A1 ~0.96 family)"),
        "no_floor_removal": dict(r=per_z_at(r_nofloor), note="den = l(>=17.2) (floor never "
                                                             "removed; span reference only)"),
    }
    i3 = int(np.argmin(np.abs(z_eval - 3.0)))
    attribution = {
        "floor_subbin_treatment": dict(
            delta_r3_pinned_minus_constf=float(recipes["pinned_k1"]["r"][i3]
                                               - recipes["pinned_k1_constf_floor"]["r"][i3]),
            delta_r3_naive176_minus_pinned=float(recipes["naive_edge_17p6"]["r"][i3]
                                                 - recipes["pinned_k1"]["r"][i3]),
            note="const-f vs power-law sub-bin = the ENTIRE sibling-vs-pinned gap; the "
                 "17.6 edge-snap = the A1 ~0.96 artifact"),
        "mean_vs_median": dict(
            delta_r3=float(recipes["pinned_k1_mean"]["r"][i3]
                           - recipes["pinned_k1"]["r"][i3]),
            note="suite mean vs median: negligible"),
        "fnhi_vs_class_columns": dict(
            delta_r3=float(recipes["sibling_lx_above"]["r"][i3]
                           - recipes["pinned_k1_constf_floor"]["r"][i3]),
            note="class columns are numerically IDENTICAL to the f_nhi bin integrals in "
                 "this cache (same-file construction); zero contribution"),
    }
    return dict(z_eval=z_eval, recipes=recipes, attribution=attribution)
