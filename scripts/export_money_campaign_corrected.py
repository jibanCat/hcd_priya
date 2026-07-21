"""Frozen export of the CORRECTED-prior money-figure campaign (paper request
2026-07-20). Reads the reran desi_hcd_prior_{on,off}{,_hi}_corr arms (corrected
closure prior center 0.18812, option-a pre-NORC forward), verifies the matched-pair
+ corrected-center contract, and writes a NEW dated immutable export with the same
acceptance-gate fields the paper verified last time.

Row-2 = the with/without-prior NUTS posteriors (closure-mock, truth = held-out sim).
Row-1 = the corrected 68% prior band (prior68_corrected_*), recomputed from the
corrected closure prior so it matches the reran arms at the source.

Blind status: BLIND-SAFE. 100% closure mocks (truth = held-out PRIYA sim, known); no
real-data n_s/A_p anywhere, including metadata.

Run (login node, after all 14 pkls land):
  cd /home/mfho/hcd_priya && PYTHONNOUSERSITE=1 PYTHONPATH=. JAX_PLATFORMS=cpu \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/export_money_campaign_corrected.py
"""
from __future__ import annotations
import argparse, hashlib, json, pickle, platform, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SC = Path("/scratch/cavestru_root/cavestru1/mfho")
ARMS = {"on": SC/"desi_hcd_prior_on_corr", "off": SC/"desi_hcd_prior_off_corr",
        "on_hi": SC/"desi_hcd_prior_on_hi_corr", "off_hi": SC/"desi_hcd_prior_off_hi_corr"}
N_PAIRS = 6
DEFAULT_OUT = Path("/home/mfho/hcd_priya_notes/artifacts/paper_exports/money_campaign_corrected_2026-07-20")
CORRECTED_LLS_CENTER = 0.18811862702546295   # survey=None closure, post-528ba89 (export of record)
RUN_CMD = ("cd /home/mfho/hcd_priya && PYTHONNOUSERSITE=1 PYTHONPATH=. JAX_PLATFORMS=cpu "
           "/home/mfho/.conda/envs/emu-jax/bin/python3 scripts/export_money_campaign_corrected.py")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()


def load_arm(d, mocks):
    out = []
    for m in mocks:
        p = d / f"mock_{m:04d}.pkl"
        if not p.exists():
            raise SystemExit(f"MISSING {p} — campaign not complete; do not export yet.")
        out.append((p, pickle.load(open(p, "rb"))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    from hcd_analysis.emulator.data import PARAM_LIMITS
    PL = np.asarray(PARAM_LIMITS)
    lo_ns, hi_ns = float(PL[0, 0]), float(PL[0, 1])
    lo_ap, hi_ap = float(PL[1, 0]), float(PL[1, 1])
    assert [ [lo_ns, hi_ns], [lo_ap, hi_ap] ] == [[0.8, 1.05], [1.2e-9, 2.6e-9]], "PARAM_LIMITS drifted"
    def phys(u_ns, u_ap):
        return lo_ns + np.asarray(u_ns)*(hi_ns-lo_ns), lo_ap + np.asarray(u_ap)*(hi_ap-lo_ap)

    on = load_arm(ARMS["on"], range(N_PAIRS))
    off = load_arm(ARMS["off"], range(N_PAIRS))
    on_hi = load_arm(ARMS["on_hi"], [0])[0][1]
    off_hi = load_arm(ARMS["off_hi"], [0])[0][1]

    # ---- CONTRACT ASSERTS (matched pairs + corrected center + widths) ------------------------
    for m in range(N_PAIRS):
        po, do = on[m][1], off[m][1]
        assert po["sim"] == do["sim"], f"pair {m} data mismatch: {po['sim']} vs {do['sim']}"
        assert np.allclose(po["dndx_truth"], do["dndx_truth"]), f"pair {m} truth mismatch (data must be prior-indep)"
        c_on = float(np.asarray(po["alpha_hcd_mu"])[0])
        assert abs(c_on - CORRECTED_LLS_CENTER) < 1e-4, (
            f"pair {m} 'on' LLS center {c_on:.5f} != corrected {CORRECTED_LLS_CENTER:.5f} "
            f"(the rerun did NOT pick up the corrected prior)")
        assert tuple(np.round(po["widths"], 3)) == (0.15, 0.40, 0.50), po["widths"]
        assert tuple(np.round(do["widths"], 1)) == (5.0, 5.0, 5.0), do["widths"]
    for hi, tag in ((on_hi, "on_hi"), (off_hi, "off_hi")):
        assert abs(float(np.asarray(hi["alpha_hcd_mu"])[0]) - CORRECTED_LLS_CENTER) < 1e-4, f"{tag} center wrong"
    assert on_hi["sim"] == off_hi["sim"], "hi pair data mismatch"

    # ---- ROW 2: with/without posteriors (physical) + matched-pair deltas ----------------------
    CLS = tuple(on_hi["dndx_class_order"])
    def pack(rec):
        ns, ap = phys(rec["ns_draws"], rec["Ap_draws"])
        return dict(ns=ns, Ap=ap, lls=np.asarray(rec["alpha_pivot_draws"])[:, 0],
                    sub=np.asarray(rec["alpha_pivot_draws"])[:, 1])
    d_ns, d_ap = [], []
    for m in range(N_PAIRS):
        o, f_ = pack(on[m][1]), pack(off[m][1])
        d_ns.append((f_["ns"].mean()-o["ns"].mean())/o["ns"].std())
        d_ap.append((f_["Ap"].mean()-o["Ap"].mean())/o["Ap"].std())
    d_ns, d_ap = np.asarray(d_ns), np.asarray(d_ap)

    ns_on_hi, ap_on_hi = phys(on_hi["ns_draws"], on_hi["Ap_draws"])
    ns_off_hi, ap_off_hi = phys(off_hi["ns_draws"], off_hi["Ap_draws"])
    ns_tr_hi, ap_tr_hi = phys(on_hi["ns_truth"], on_hi["Ap_truth"])

    np.savez(out/"money_row2_arms.npz",
             class_order=np.array(CLS), n_pairs=N_PAIRS,
             lls_center_corrected=np.float64(CORRECTED_LLS_CENTER),
             # hi-arm contours (physical)
             ns_on_hi=ns_on_hi, Ap_on_hi=ap_on_hi, ns_off_hi=ns_off_hi, Ap_off_hi=ap_off_hi,
             ns_truth=np.float64(ns_tr_hi), Ap_truth=np.float64(ap_tr_hi),
             lls_on_hi=np.asarray(on_hi["alpha_pivot_draws"])[:, 0],
             lls_off_hi=np.asarray(off_hi["alpha_pivot_draws"])[:, 0],
             sub_on_hi=np.asarray(on_hi["alpha_pivot_draws"])[:, 1],
             sub_off_hi=np.asarray(off_hi["alpha_pivot_draws"])[:, 1],
             lls_truth=np.float64(on_hi["alpha_pivot_truth"][0]),
             sub_truth=np.float64(on_hi["alpha_pivot_truth"][1]),
             dndx_truth=np.asarray(on_hi["dndx_truth"], float),
             dndx_z=np.asarray(on_hi["dndx_z"], float), dndx_Xbar=np.asarray(on_hi["dndx_Xbar"], float),
             dndx_draws_on_hi=np.asarray(on_hi["dndx_draws"]), dndx_draws_off_hi=np.asarray(off_hi["dndx_draws"]),
             # matched-pair shift inset (N=6)
             pair_d_ns=d_ns, pair_d_Ap=d_ap,
             pair_ns_shift=np.float64(d_ns.mean()), pair_ns_sem=np.float64(d_ns.std(ddof=1)/np.sqrt(N_PAIRS)),
             pair_Ap_shift=np.float64(d_ap.mean()), pair_Ap_sem=np.float64(d_ap.std(ddof=1)/np.sqrt(N_PAIRS)),
             alpha_hcd_mu=np.asarray(on_hi["alpha_hcd_mu"], float),
             alpha_hcd_sigma=np.asarray(on_hi["alpha_hcd_sigma"], float),
             PARAM_LIMITS_ns_Ap=np.array([[lo_ns, hi_ns], [lo_ap, hi_ap]]),
             note=np.array("closure-mock posteriors (truth=held-out PRIYA sim); with(on)/without(off) HCD prior; "
                           "matched-pair identical data; corrected closure center 0.18812; pre-NORC forward (option a)"))

    # ---- ROW 1: corrected 68% prior band (recomputed from the corrected prior) ---------------
    import hcd_analysis.emulator  # noqa
    import jax, jax.numpy as jnp
    from hcd_analysis.emulator.dndx_wc import alpha_to_dndx
    from hcd_analysis.emulator.closure_legb import HCD_INCIDENCE_SLOPE
    from hcd_analysis.emulator.inference import HCD_Z_PIVOT
    from hcd_analysis.emulator.sampler_numpyro import _dla_raw_mu
    zg = np.asarray(on_hi["dndx_z"], float); Xb = np.asarray(on_hi["dndx_Xbar"], float)
    mu = np.asarray(on_hi["alpha_hcd_mu"], float); sd = np.asarray(on_hi["alpha_hcd_sigma"], float)
    s_c = np.asarray(HCD_INCIDENCE_SLOPE, float); zp = float(HCD_Z_PIVOT)
    rng = np.random.default_rng(20260717)
    smp = rng.normal(mu, sd, size=(20000, 3))          # closure sampler: plain Normal LLS/subDLA
    for j in (0, 1):                                    # truncate <0 (deployed low=0)
        bad = smp[:, j] < 0
        while bad.any():
            smp[bad, j] = rng.normal(mu[j], sd[j], size=int(bad.sum())); bad = smp[:, j] < 0
    rmu = float(_dla_raw_mu(mu[2]))
    smp[:, 2] = np.log1p(np.exp(np.minimum(rmu + rng.standard_normal(20000), 30.0)))
    shape = ((1.0+zg)[:, None]/(1.0+zp))**s_c[None, :]
    lo = np.empty((len(zg), 3)); hi = np.empty((len(zg), 3))
    for j, z in enumerate(zg):
        dd = np.asarray(alpha_to_dndx(jnp.asarray(smp*shape[j][None, :]), jnp.asarray(float(Xb[j])), jnp.asarray(float(z))))
        lo[j], hi[j] = np.percentile(dd, [16, 84], axis=0)
    np.savez(out/"money_row1_layers.npz",
             class_order=np.array(CLS), zg=zg, xbar=Xb,
             prior68_corrected_lo=lo, prior68_corrected_hi=hi,
             prior68_corrected_alpha_mu=mu, prior68_corrected_alpha_sigma=sd,
             lls_center_corrected=np.float64(CORRECTED_LLS_CENTER),
             note=np.array("corrected closure prior band (center 0.18812), matches the reran row-2 arms at the source; "
                           "supersedes prior68_corrected_* in dndx_repin_2026-07-20 for the money figure"))

    # ---- provenance sidecar (acceptance gate) -------------------------------------------------
    import jax as _jax
    inputs = {}
    for tag, d in ARMS.items():
        for p in sorted(d.glob("mock_*.pkl")):
            inputs[str(p)] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    outputs = {n: {"sha256": sha256(out/n), "bytes": (out/n).stat().st_size}
               for n in ("money_row2_arms.npz", "money_row1_layers.npz")}
    prov = {
        "schema": "money_campaign_corrected_v1",
        "purpose": "corrected-prior money-figure campaign (paper request 2026-07-20); supersedes the "
                   "pre-correction row-2 arms so row-1 band and row-2 posteriors are consistent at the source",
        "forward_decision": "OPTION A: corrected closure prior (LLS center 0.18812) + PRE-NORC forward "
                            "(res_corr_on=True default; NOT the deployed NORC forward). Money-figure banner stays PRE-NORC.",
        "commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "scripts/desi_hcd_prior_sensitivity.py",
                          "scripts/export_money_campaign_corrected.py", "hcd_analysis/")),
        "run_command": RUN_CMD,
        "campaign_launch": {"seed": 20260621, "fold": 0, "n_mocks_fold": 8, "n_pairs": N_PAIRS,
                            "base_depth": "warmup 250 / samples 600", "hi_depth": "warmup 400 / samples 2000",
                            "arms": {k: str(v) for k, v in ARMS.items()},
                            "slurm_jobs": "54237701 on / 54237702 off / 54237703 on_hi / 54237704 off_hi"},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "jax": _jax.__version__,
                        "hostname": platform.node(), "conda_env": "emu-jax", "platform": platform.platform()},
        "inputs_sha256": inputs, "outputs_sha256": outputs,
        "PARAM_LIMITS_ns_Ap": [[lo_ns, hi_ns], [lo_ap, hi_ap]],
        "blind_status": "BLIND-SAFE: 100% closure mocks (truth = held-out PRIYA sim, known); no real-data n_s/A_p "
                        "in any array or field",
        "chain_of_record_status": "ARTIFACT OF RECORD for the corrected money-figure campaign (paper re-pin). "
                                  "Immutable: regenerate to a NEW dated directory, never edit in place. NOT a real-data chain.",
        "matched_pair_delta": {"n_s": [float(d_ns.mean()), float(d_ns.std(ddof=1)/np.sqrt(N_PAIRS))],
                               "A_p": [float(d_ap.mean()), float(d_ap.std(ddof=1)/np.sqrt(N_PAIRS))]},
    }
    (out/"PROVENANCE.json").write_text(json.dumps(prov, indent=1, sort_keys=True))
    print(f"[export] wrote {out}")
    for n, meta in outputs.items():
        print(f"  {n}  sha256={meta['sha256'][:16]}...  {meta['bytes']} B")
    print(f"  commit={prov['commit'][:12]} dirty={prov['dirty']}  matched-pair n_s "
          f"{prov['matched_pair_delta']['n_s'][0]:+.2f}+/-{prov['matched_pair_delta']['n_s'][1]:.2f}  "
          f"A_p {prov['matched_pair_delta']['A_p'][0]:+.2f}+/-{prov['matched_pair_delta']['A_p'][1]:.2f}")


if __name__ == "__main__":
    main()
