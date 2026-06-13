#!/usr/bin/env python3
"""Convert the eBOSS DR14 Lyα-forest P1D (Chabanier+2019, arXiv:1812.03554) → a portable npz.

The raw ``boss_dr14_data/{Pk1D_data,Pk1D_syst,Pk1D_cor}.dat`` are plain TEXT, so this runs under
the emu-jax env (numpy only; NO astropy, unlike the DESI FITS converter). The P1D is on a
(NZ=13 × NK=35)=455 (z,k) grid, z-MAJOR, z∈[2.2,4.6], k ANGULAR [s/km] (the same convention as the
emulator cache — NO 2π factor). The covariance is BLOCK-DIAGONAL per z (no cross-z covariance):
  σ_z[k]   = sqrt( Σ_8 syst_s[z,k]²  +  stat[z,k]² )            # absolute P-units variance
  C_block_z = diag(σ_z) · corr_z · diag(σ_z)                    # 35×35 per z
  C_data    = block_diag(C_block_0, …, C_block_12)              # 455×455

Run (emu-jax python):
  /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/convert_eboss_dr14_p1d.py \
    --base /home/mfho/lya_emulator_full/lyaemu/data/boss_dr14_data \
    --out  /home/mfho/data/eboss_dr14_p1d/eboss_dr14_p1d.npz

Pk1D_cor.dat layout TRAP (the single biggest pitfall): the file has 3 comment lines + 26 blank
separators (2 after each of the 13 z-blocks). np.loadtxt skips BOTH → (455,35), and .reshape(13,35,35)
recovers the per-z blocks ONLY because the blanks come AFTER each complete 35-row block. We
ASSERT the shape and a per-block diagonal==1, and the loader-side golden test pins one off-diagonal
(cov[0,1] = σ0·σ1·corr[0,1]) — the only check that proves a block landed on the right z with the
right σ pairing (the diag round-trip alone does NOT, since diag(corr)≡1).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.linalg import block_diag

NZ, NK = 13, 35
SYST_COLS = ("continuum", "noise", "resolution", "sb", "linemask",
             "dlamask", "dlacompleteness", "balcompleteness")   # Pk1D_syst.dat column order


def build(base):
    data = np.loadtxt(os.path.join(base, "Pk1D_data.dat"))      # (455,6): z k PLya stat Pnoise PSB
    syst = np.loadtxt(os.path.join(base, "Pk1D_syst.dat"))      # (455,8)
    cor = np.loadtxt(os.path.join(base, "Pk1D_cor.dat"))        # (455,35): comments+blanks auto-skipped
    assert data.shape == (NZ * NK, 6), data.shape
    assert syst.shape == (NZ * NK, len(SYST_COLS)), syst.shape
    assert cor.shape == (NZ * NK, NK), cor.shape                # the block-parse trap guard

    z, k = data[:, 0], data[:, 1]
    plya, stat, pnoise, psb = data[:, 2], data[:, 3], data[:, 4], data[:, 5]
    sigma = np.sqrt((syst ** 2).sum(axis=1) + stat ** 2)        # (455,) block-diag cov ingredient

    corr_blocks = cor.reshape(NZ, NK, NK)
    cov_blocks = []
    for iz in range(NZ):
        assert np.allclose(np.diag(corr_blocks[iz]), 1.0), f"corr block {iz} diag != 1"
        s = sigma[iz * NK:(iz + 1) * NK]
        cb = np.outer(s, s) * corr_blocks[iz]
        cb = 0.5 * (cb + cb.T)                                  # symmetrize text round-off
        cov_blocks.append(cb)
    cov = block_diag(*cov_blocks)                              # (455,455)

    out = dict(z=z, k=k, plya=plya, stat=stat, pnoise=pnoise, psb=psb,
               sigma=sigma, cov=cov, corr=cor, nz=NZ, nk=NK, z_unique=np.unique(z),
               row_is_zmajor=bool(np.all(z[:NK] == z[0])))
    for j, c in enumerate(SYST_COLS):
        out["syst_" + c] = syst[:, j]

    # self-checks (must all pass)
    diag_ok = bool(np.allclose(np.diag(cov), sigma ** 2))
    sym_ok = bool(np.allclose(cov, cov.T))
    block_pd = all(np.linalg.eigvalsh(b).min() > 0 for b in cov_blocks)
    crossz_zero = bool(np.all(cov[np.ix_(z != z[0], z == z[0])] == 0.0))   # no cross-z covariance
    assert diag_ok and sym_ok and block_pd and crossz_zero, (diag_ok, sym_ok, block_pd, crossz_zero)
    return out, (diag_ok, sym_ok, block_pd, crossz_zero)


def main():
    ap = argparse.ArgumentParser(description="Convert eBOSS DR14 P1D → npz")
    ap.add_argument("--base", default="/home/mfho/lya_emulator_full/lyaemu/data/boss_dr14_data")
    ap.add_argument("--out", default="/home/mfho/data/eboss_dr14_p1d/eboss_dr14_p1d.npz")
    args = ap.parse_args()
    out, (diag_ok, sym_ok, block_pd, crossz) = build(args.base)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **out)
    z = out["z"]; k = out["k"]; cov = out["cov"]
    print(f"wrote {args.out}")
    print(f"  grid NZ={out['nz']} NK={out['nk']} ({NZ * NK} rows); z={np.round(out['z_unique'], 2)}")
    print(f"  k angular [{k.min():.6f},{k.max():.6f}] s/km; z-major={out['row_is_zmajor']}")
    print(f"  cov {cov.shape} sym={sym_ok} diag==σ²={diag_ok} per-block-PD={block_pd} cross-z-zero={crossz}")
    print(f"  golden: cov[0,1]={cov[0, 1]:.6e}  (= σ0·σ1·corr[0,1])")


if __name__ == "__main__":
    main()
