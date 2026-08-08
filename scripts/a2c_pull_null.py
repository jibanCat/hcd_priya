#!/usr/bin/env python
"""PI #19: A2c pull-dispersion null audit. Existing stored draws only. No new sampling."""
import json, pickle, sys
import numpy as np
from scipy import stats
R="/scratch/cavestru_root/cavestru1/mfho/cert_2026-07"
ARM={"A2c_DESI":("armp_DESI_corrected_v1",48),"A1c_eBOSS":("armp_eBOSS_corrected_v2",48)}
RNG=np.random.default_rng(20260808)

def load(arm):
    d,n=ARM[arm]; out=[]
    for m in range(n):
        z=pickle.load(open(f"{R}/{d}/mock_{m:04d}.pkl","rb"))
        i=z["names"].index("ns")
        out.append(dict(m=m,draws=np.asarray(z["draws"])[:,i],truth=float(z["truth_vec"][i]),L=int(z["L"])))
    return out

def gate(rows):
    """EXACT frozen gate: analyze_firstarm_selfdraw.py:169-170."""
    mu=np.array([r["draws"].mean() for r in rows])
    sd=np.array([r["draws"].std(ddof=1) for r in rows])
    th=np.array([r["truth"] for r in rows])
    p=(mu-th)/sd
    return p, float(p.mean()), float(p.std(ddof=1))

def pit_scores(rows,rng):
    """Randomized PIT: u = (#draws < truth + U*#ties)/(L+1) style, finite-L randomized."""
    z=[]
    for r in rows:
        d=r["draws"]; L=len(d)
        below=(d<r["truth"]).sum(); ties=(d==r["truth"]).sum()
        u=(below+rng.random()*(ties+1))/(L+1)
        z.append(stats.norm.ppf(min(max(u,1e-12),1-1e-12)))
    return np.array(z)

def split_null(rows,B,rng):
    """SHAPE-AWARE posterior-exchangeability null with DISJOINT split (PI sec-8).
    Subset A -> mu_hat,sd_hat ; subset B -> pseudo-truth. Preserves shape, truncation,
    heteroscedasticity and the exact gate formula."""
    stat=np.empty(B)
    A=[];Bd=[]
    for r in rows:
        d=r["draws"]; L=len(d); h=L//2
        A.append(d[:h]); Bd.append(d[h:])          # contiguous split preserves autocorrelation
    mu=np.array([a.mean() for a in A]); sd=np.array([a.std(ddof=1) for a in A])
    for b in range(B):
        th=np.array([bd[rng.integers(len(bd))] for bd in Bd])
        stat[b]=((mu-th)/sd).std(ddof=1)
    return stat, mu, sd

def h3_exact_moment_null(rows,B,rng):
    """H3: the SAME split construction but with pseudo-truths drawn from the SAME subset used
    for the moments -- isolates the self-normalization / random-denominator contribution."""
    stat=np.empty(B)
    for b in range(B):
        s=[]
        for r in rows:
            d=r["draws"]; mu=d.mean(); sd=d.std(ddof=1)
            s.append((mu-d[rng.integers(len(d))])/sd)
        stat[b]=np.std(s,ddof=1)
    return stat

def main():
    out={}
    for arm in ARM:
        rows=load(arm)
        pulls,pm,psd=gate(rows)
        L=np.array([r["L"] for r in rows])
        # --- gaussian reference (the DEPLOYED null): sd of N=48 unit normals
        g=np.array([RNG.standard_normal(48).std(ddof=1) for _ in range(200000)])
        p_gauss=float((g>=psd).mean())
        # --- shape-aware exchangeability null (disjoint split)
        se,mu_s,sd_s=split_null(rows,20000,RNG)
        p_shape=float((se>=psd).mean())
        # --- H3 self-normalized null
        h3=h3_exact_moment_null(rows,4000,RNG)
        p_h3=float((h3>=psd).mean())
        # --- PIT normal-score control
        zp=pit_scores(rows,RNG)
        gp=np.array([RNG.standard_normal(48).std(ddof=1) for _ in range(100000)])
        out[arm]=dict(
            n=48, pull_mean=pm, pull_sd=psd,
            p_gaussian_ref=p_gauss,
            shape_null_mean=float(se.mean()), shape_null_p=p_shape,
            shape_null_q=[float(np.quantile(se,q)) for q in (0.05,0.5,0.95,0.99)],
            h3_null_mean=float(h3.mean()), h3_null_p=p_h3,
            PIT_mean=float(zp.mean()), PIT_sd=float(zp.std(ddof=1)),
            PIT_p_sd=float((gp>=zp.std(ddof=1)).mean()),
            PIT_ks_p=float(stats.kstest(zp,"norm").pvalue),
            L_min=int(L.min()), L_med=float(np.median(L)),
            width_cv=float(np.std([r["draws"].std(ddof=1) for r in rows],ddof=1)/
                           np.mean([r["draws"].std(ddof=1) for r in rows])),
            skew_med=float(np.median([stats.skew(r["draws"]) for r in rows])),
            kurt_med=float(np.median([stats.kurtosis(r["draws"]) for r in rows])),
            frac_truth_within_0p05_of_bound=float(np.mean(
                [min(r["truth"],1-r["truth"])<0.05 for r in rows])))
    return out

if __name__=="__main__":
    r=main(); json.dump(r,open(sys.argv[1],"w"),indent=1)
    for a,v in r.items():
        print(f"=== {a} ===")
        print(f"  gate pull sd = {v['pull_sd']:.4f}  (mean {v['pull_mean']:+.4f})   L min {v['L_min']} med {v['L_med']:.0f}")
        print(f"  DEPLOYED Gaussian-unit-pull ref   : p = {v['p_gaussian_ref']:.4f}")
        print(f"  SHAPE-AWARE exchangeability null  : mean {v['shape_null_mean']:.4f}  p = {v['shape_null_p']:.4f}   q95 {v['shape_null_q'][2]:.4f} q99 {v['shape_null_q'][3]:.4f}")
        print(f"  H3 self-normalized null           : mean {v['h3_null_mean']:.4f}  p = {v['h3_null_p']:.4f}")
        print(f"  PIT normal-score control          : mean {v['PIT_mean']:+.4f} sd {v['PIT_sd']:.4f}  p(sd) = {v['PIT_p_sd']:.4f}  KS p = {v['PIT_ks_p']:.4f}")
        print(f"  posterior width CV {v['width_cv']:.3f}  median skew {v['skew_med']:+.3f} kurt {v['kurt_med']:+.3f}  truths near bound {v['frac_truth_within_0p05_of_bound']*100:.0f}%")
