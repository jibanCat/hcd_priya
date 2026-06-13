"""CLEAN MF imprint at eBOSS k — the raw HR/LF cache ratio (no predict-path, no emulator).

The MF (LF->HR resolution) correction is the cache-level P_HR/P_LF ratio over the 6 overlap sims
(HR design points that are exact LF points, tau0-rung matched). The earlier predict-path forward
check gave a spurious +32% (artifact). This reuses diag_lf_vs_hr_highk's exact HR<->LF matching and
reports the correction OVER THE eBOSS k-band [0.0011,0.0195] s/km, per z, + the implied Delta n_s
(a multiplicative tilt in the ratio aliases into the spectral tilt: Dn_s ~ d ln(P_HR/P_LF)/d ln k).

Run (emu-jax): PYTHONPATH=/home/mfho/hcd_priya python3 scripts/diag_eboss_mf_imprint_clean.py
"""
import sys
import numpy as np

sys.path.insert(0, "/home/mfho/hcd_priya/scripts")
import hcd_analysis.emulator  # noqa: F401
import diag_lf_vs_hr_highk as D   # reuse load/match/_norm (importing does NOT run main)

EB_LO, EB_HI = 0.001084, 0.019512   # eBOSS k band [s/km]


def main():
    lf = D.load(D.LF_CACHE)
    hr = D.load(D.HR_CACHE)
    pairs = D.match(lf, hr)
    # accumulate the MF correction (P_HR/P_LF - 1) per (z, k) over the overlap sims, eBOSS band only.
    recs = []   # (z, k, corr)
    sims = set()
    for hrow, lrow in pairs:
        klf, Plf, z, sim = lf["kfkms"][lrow], lf["P_clean"][lrow], lf["z"][lrow], lf["sim"][lrow]
        khr, Phr = hr["kfkms"][hrow], hr["P_clean"][hrow]
        m = (np.isfinite(klf) & (klf >= EB_LO - 1e-9) & (klf <= EB_HI + 1e-9)
             & np.isfinite(Plf) & (Plf > 0))
        mh = np.isfinite(khr) & (khr > 0) & np.isfinite(Phr) & (Phr > 0)
        if m.sum() < 5 or mh.sum() < 5:
            continue
        Phr_on_lf = np.exp(np.interp(np.log(klf[m]), np.log(khr[mh]), np.log(Phr[mh]),
                                     left=np.nan, right=np.nan))
        corr = Phr_on_lf / Plf[m] - 1.0       # MF correction: how much HR exceeds LF (the boost)
        sims.add(sim)
        for kv, cv in zip(klf[m], corr):
            if np.isfinite(cv):
                recs.append((float(z), float(kv), float(cv)))
    R = np.array(recs)
    z_all, k_all, c_all = R[:, 0], R[:, 1], R[:, 2]
    print(f"=== CLEAN MF imprint at eBOSS k (raw P_HR/P_LF − 1; {len(sims)} overlap sims) ===")
    print(f"  eBOSS band [{EB_LO:.4f},{EB_HI:.4f}] s/km")
    print(f"  {'z':>4} {'corr@k_lo':>10} {'corr@k_hi':>10} {'mean':>8} {'tilt(Δn_s)':>11}")
    dns = []
    for z in sorted(set(np.round(z_all, 2))):
        mz = np.abs(z_all - z) < 0.05
        if mz.sum() < 6:
            continue
        ks, cs = k_all[mz], c_all[mz]
        # bin-average per k (the 6 sims share the k grid up to dv jitter)
        order = np.argsort(ks); ks, cs = ks[order], cs[order]
        lo = np.mean(cs[ks < np.percentile(ks, 15)])
        hi = np.mean(cs[ks > np.percentile(ks, 85)])
        slope = np.polyfit(np.log(ks), np.log1p(cs), 1)[0]   # d ln(1+corr)/d ln k ≈ Δn_s
        dns.append(slope)
        print(f"  {z:>4.1f} {100*lo:>9.2f}% {100*hi:>9.2f}% {100*np.mean(cs):>7.2f}% {slope:>+11.4f}")
    print(f"\n  ACROSS z: mean corr @low-k {100*np.mean([np.mean(c_all[(np.abs(z_all-z)<0.05)&(k_all<0.003)]) for z in sorted(set(np.round(z_all,2))) if ((np.abs(z_all-z)<0.05)&(k_all<0.003)).sum()>0]):+.2f}%, "
          f"mean tilt Δn_s ≈ {np.mean(dns):+.4f} (rms {np.std(dns):.4f})")
    print("  => the GENUINE MF correction at eBOSS k (vs the spurious predict-path +32%).")


if __name__ == "__main__":
    main()
