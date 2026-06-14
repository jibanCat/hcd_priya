"""Convert the DESI DR1 Lyα P1D FITS (Karaçaylı 2025, Zenodo 16943723) → a portable npz.

The emulator env (emu-jax) has no astropy; this one-time converter runs under SYSTEM
python3 (astropy 7.x) and writes a numpy-only npz the likelihood can read. The DESI
optimal-estimator (QMLE, SB1-subtracted = metal side-band removed, continuum-corrected)
P1D is on a (NZ=12 × NK=85)=1020 (z,k) grid with a FULL 1020×1020 covariance (strongly
correlated — adjacent-bin |corr|≈0.43). k is ANGULAR s/km (k_max≈π/60, the DESI Nyquist),
the SAME convention as the emulator cache `kfkms`.

Run (SYSTEM python, NOT emu-jax):
  python3 scripts/convert_desi_dr1_p1d.py \
    --fits /home/mfho/data/desi_dr1_p1d/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits \
    --inflation /home/mfho/data/desi_dr1_p1d/cov_diag_inflation.fits \
    --out /home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz
"""
from __future__ import annotations
import argparse
import numpy as np
from astropy.io import fits

P1D_COLS = ("Z", "K1", "K2", "K", "PLYA", "PRAW", "PNOISE", "PFID", "PSMOOTH",
            "E_PK", "E_STAT", "E_SYST")
SYST_COLS = ("E_DLA_COMPLETENESS", "E_BAL_COMPLETENESS", "E_RESOLUTION",
             "E_CONTINUUM", "E_CONTINUUM_ADD", "E_NOISE_SCALE", "E_NOISE_ADD", "E_SYST")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fits", required=True)
    ap.add_argument("--inflation", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {}
    with fits.open(args.fits) as h:
        d = h["P1D_BLIND"].data
        nz = int(h["P1D_BLIND"].header["NZ"]); nk = int(h["P1D_BLIND"].header["NK"])
        for c in P1D_COLS:
            out[c.lower()] = np.asarray(d[c], dtype=np.float64)
        syst = h["SYSTEMATICS"].data
        for c in SYST_COLS:
            out["syst_" + c.lower()] = np.asarray(syst[c], dtype=np.float64)
        out["cov"] = np.asarray(h["COVARIANCE"].data, dtype=np.float64)          # (1020,1020) total
        out["cov_stat"] = np.asarray(h["COVARIANCE_STAT"].data, dtype=np.float64)
        out["cov_syst"] = np.asarray(h["COVARIANCE_SYST"].data, dtype=np.float64)
    out["nz"] = nz; out["nk"] = nk
    out["z_unique"] = np.unique(out["z"])                                        # (NZ,)
    # the flat row order is z-major (NZ blocks of NK) or k-major? infer from Z column.
    out["row_is_zmajor"] = bool(np.all(out["z"][:nk] == out["z"][0]))            # first nk share one z?

    if args.inflation:
        with fits.open(args.inflation) as h:
            out["cov_diag_inflation"] = np.asarray(h[0].data, dtype=np.float64)  # (1020,)

    # sanity: diag(cov) == E_PK^2
    diag_ok = np.allclose(np.sqrt(np.diag(out["cov"])), out["e_pk"], rtol=1e-3)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}")
    print(f"  grid NZ={nz} NK={nk} ({nz*nk} rows); z_unique={np.round(out['z_unique'],2)}")
    print(f"  k angular [{out['k'].min():.5f},{out['k'].max():.5f}] s/km; "
          f"z-major rows={out['row_is_zmajor']}")
    print(f"  cov {out['cov'].shape} symmetric={np.allclose(out['cov'],out['cov'].T)} "
          f"diag==E_PK^2: {diag_ok}")
    print(f"  components: total/stat/syst cov + {len(SYST_COLS)} syst terms + diag_inflation"
          f"={'cov_diag_inflation' in out}")


if __name__ == "__main__":
    main()
