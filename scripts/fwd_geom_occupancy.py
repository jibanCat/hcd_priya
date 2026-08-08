#!/usr/bin/env python
"""PI #18 forward-geometry scan -- ANALYTIC OCCUPANCY (zero forward evaluations).

Prereg 2026-08-08-FORWARD-GEOMETRY-SCAN-PREREG.md v5. Surfaces are the UNION of the 20-rung
MF ladder (family F, forward mean) and its nested 4-knot C_emu subset (family K), which are
EXACTLY rungs {0,5,14,19} of the same ladder -- so 20 knots per (leg,z), not 24.
"""
import json, pickle, sys
import numpy as np
sys.path.insert(0, "/home/mfho/hcd_priya/scripts")
import fwd_geom_core as G

R = "/scratch/cavestru_root/cavestru1/mfho/cert_2026-07"
ARM = {"DESI": ("armp_DESI_corrected_v1", 48), "eBOSS": ("armp_eBOSS_corrected_v2", 48)}
LADDER = 0.65555638 + np.arange(20) * 0.03556264      # verified: 20 rungs, uniform
KSUB = [0, 5, 14, 19]                                 # family K = C_emu knots, nested exactly
EPS_COMMON = 27 ** -0.25                              # prereg v4.7: COMMON eps on both legs

def run():
    out = {"_meta": dict(ladder=LADDER.tolist(), family_K_rungs=KSUB,
                         eps_common=EPS_COMMON, prereg="v5")}
    for leg, (d, n) in ARM.items():
        z = G.Z_GRID[leg]; c = G.c_of_z(z)
        # surfaces: (n_z * 20) lines  x + c_i y = ln alpha_j
        C = np.repeat(c, 20); A = np.tile(np.log(LADDER), len(z))
        ZI = np.repeat(z, 20); RJ = np.tile(np.arange(20), len(z))
        permock = []
        for m in range(n):
            zz = pickle.load(open(f"{R}/{d}/mock_{m:04d}.pkl", "rb"))
            se = zz["sites_extra"]
            ta = np.asarray(se["tau0_amp"]["draws"]); dt = np.asarray(se["dtau0"]["draws"])
            xy = G.xy_of(ta, dt)
            mu, L = G.plane_whitener(xy)
            dw = G.signed_distance(xy, C, A, L=L)                 # (L_draws, n_surf)
            near = np.abs(dw) < EPS_COMMON
            cross = G.crossings_between(dw)
            ns = np.asarray(zz["draws"])[:, zz["names"].index("ns")]
            # pre-declared COUPLED direction: n_s response along each surface normal
            nrm = np.stack([np.ones_like(C), C])                   # (2, n_surf)
            u = xy @ nrm / np.linalg.norm(L.T @ nrm, axis=0)       # (L_draws, n_surf)
            su = u.std(0); sn = ns.std()
            r = ((u - u.mean(0)) * (ns - ns.mean())[:, None]).mean(0) / np.maximum(su * sn, 1e-30)
            b = ((u - u.mean(0)) * (ns - ns.mean())[:, None]).mean(0) / np.maximum(su ** 2, 1e-30)
            permock.append(dict(
                mock=m, L=int(zz["L"]),
                min_abs_dw=float(np.abs(dw).min()),
                frac_mass_near_any=float(near.any(1).mean()),
                n_surf_ever_near=int(near.any(0).sum()),
                frac_transitions_crossing_any=float(cross.any(1).mean()) if cross.size else 0.0,
                mean_ns_r=float(np.nanmean(np.abs(r))), max_ns_slope=float(np.nanmax(np.abs(b))),
                min_abs_dw_K=float(np.abs(dw[:, np.isin(RJ, KSUB)]).min())))
        out[leg] = dict(
            n_surfaces=int(len(C)), n_z=int(len(z)), n_mocks=n,
            n_surfaces_K=int(np.isin(RJ, KSUB).sum()),
            per_mock=permock,
            # PER-MOCK FIRST, then equal-weight average (prereg v4.6 M6)
            frac_mass_near_any=float(np.mean([p["frac_mass_near_any"] for p in permock])),
            median_min_abs_dw=float(np.median([p["min_abs_dw"] for p in permock])),
            min_min_abs_dw=float(np.min([p["min_abs_dw"] for p in permock])),
            median_min_abs_dw_K=float(np.median([p["min_abs_dw_K"] for p in permock])),
            frac_transitions_crossing=float(np.mean([p["frac_transitions_crossing_any"] for p in permock])),
            mean_surf_ever_near=float(np.mean([p["n_surf_ever_near"] for p in permock])),
            mean_ns_r=float(np.mean([p["mean_ns_r"] for p in permock])),
            max_ns_slope=float(np.max([p["max_ns_slope"] for p in permock])),
            frac_mocks_with_mass_near=float(np.mean([p["frac_mass_near_any"] > 0 for p in permock])))
    return out

if __name__ == "__main__":
    r = run()
    json.dump(r, open(sys.argv[1], "w"), indent=1)
    for leg in ARM:
        v = r[leg]
        print(f"=== {leg}: {v['n_surfaces']} surfaces ({v['n_surfaces_K']} are family K) ===")
        print(f"  posterior mass within eps of ANY surface : {v['frac_mass_near_any']*100:.2f}%")
        print(f"  mocks with ANY mass near a surface       : {v['frac_mocks_with_mass_near']*100:.1f}%")
        print(f"  median (over mocks) min |d_w|            : {v['median_min_abs_dw']:.3f}  (min {v['min_min_abs_dw']:.3f})")
        print(f"  same, family-K subset only               : {v['median_min_abs_dw_K']:.3f}")
        print(f"  mean surfaces ever approached per mock   : {v['mean_surf_ever_near']:.2f}")
        print(f"  transitions crossing any surface         : {v['frac_transitions_crossing']*100:.2f}%")
        print(f"  |corr(n_s, normal coord)| mean           : {v['mean_ns_r']:.4f}   max |slope| {v['max_ns_slope']:.4f}")
