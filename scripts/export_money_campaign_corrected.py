"""Frozen export of the CORRECTED-prior money-figure campaign (paper request
2026-07-20). Reads the reran desi_hcd_prior_{on,off}{,_hi}_corr arms (corrected
closure prior center 0.18812, option-a pre-NORC forward), verifies the matched-pair
+ corrected-center contract, and writes a NEW dated immutable export with the same
acceptance-gate fields the paper verified last time.

Row-2 = the with/without-prior NUTS posteriors (closure-mock, truth = held-out sim).
Row-1 = the corrected 68% prior band (prior68_corrected_*), recomputed from the
corrected closure prior so it matches the reran arms at the source.

V2 RE-ISSUE (W7 2026-07-22, one-time re-issue under the corrected model): the campaign
pkls on scratch PREDATE the exact-inverse migration of the campaign runner -- their
stored `dndx_draws` were computed with the APPROXIMATE `alpha_to_dndx`, which silently
SATURATED out-of-simplex draws at dN/dX = 27.631021/Xbar (808 entries in the off_hi
arm; readout defect B). The v1 export (money_campaign_corrected_2026-07-20) passed
those arrays through verbatim into `money_row2_arms.npz`. v2 IGNORES the pkls' stored
dndx arrays and RE-DERIVES the row-2 per-draw dN/dX(z) (and dndx_truth) from
`alpha_pivot_draws` through `alpha_to_dndx_exact` in mask mode -- out-of-simplex draws
become NaN (excluded, counted per arm in PROVENANCE), never a silent clamp value.

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
DEFAULT_OUT = Path("/home/mfho/hcd_priya_notes/artifacts/paper_exports/money_campaign_corrected_2026-07-22")
SATURATION_MU = 27.631021    # -log(1e-12): the retired approximate-inverse clamp fingerprint
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


def rederive_dndx_draws(rec, tag):
    """Re-derive per-draw dN/dX(z) (L,nZ,3) from the pkl's alpha_pivot_draws through the EXACT
    occupancy-map inverse in mask mode (readout defect B / v2 re-issue, 2026-07-22).

    The pkls on scratch predate the campaign runner's exact-inverse migration: their STORED
    `dndx_draws` were computed with the APPROXIMATE alpha_to_dndx, which silently saturated
    out-of-simplex draws at 27.631021/Xbar -- those stored arrays are IGNORED and superseded.
    Same construction as desi_hcd_prior_sensitivity._dndx_from_pivot_draws (closure z-slope
    HCD_INCIDENCE_SLOPE, pivot HCD_Z_PIVOT), on the pkl's own dndx_z / dndx_Xbar grid.

    Returns (dndx_draws, n_invalid, n_total). Out-of-simplex (draw, z) entries are NaN."""
    from hcd_analysis.emulator.dndx_wc import alpha_to_dndx_exact
    from hcd_analysis.emulator.closure_legb import HCD_INCIDENCE_SLOPE
    from hcd_analysis.emulator.inference import HCD_Z_PIVOT
    a = np.asarray(rec["alpha_pivot_draws"], float)             # (L,3)
    zg = np.asarray(rec["dndx_z"], float)                       # (nZ,)
    Xb = np.asarray(rec["dndx_Xbar"], float)                    # (nZ,)
    s_c = np.asarray(HCD_INCIDENCE_SLOPE, float)                # (3,) closure/SBC slope
    shape = ((1.0 + zg)[:, None] / (1.0 + float(HCD_Z_PIVOT))) ** s_c[None, :]   # (nZ,3)
    out = np.empty((a.shape[0], len(zg), 3))
    n_invalid = 0
    for j in range(len(zg)):
        out[:, j, :], ok = alpha_to_dndx_exact(a * shape[j][None, :], float(Xb[j]),
                                               float(zg[j]), mode="mask")
        n_invalid += int(np.size(ok) - np.count_nonzero(ok))
    n_total = int(a.shape[0] * len(zg))
    # no silent clamp value may survive the re-derivation (the v1 artifact fingerprint)
    assert not np.any(np.isclose(out, (SATURATION_MU / Xb)[None, :, None], rtol=1e-5)), (
        f"{tag}: re-derived dndx draws contain the 27.631021/Xbar saturation fingerprint -- "
        f"the exact-inverse path is broken")
    stored = np.asarray(rec["dndx_draws"], float)
    both = np.isfinite(out) & ~np.isclose(stored, (SATURATION_MU / Xb)[None, :, None], rtol=1e-5)
    rel = np.abs(out[both] / stored[both] - 1.0) if both.any() else np.zeros(1)
    print(f"[row2 {tag}] re-derived dndx draws: {n_invalid}/{n_total} out-of-domain -> NaN "
          f"(pkl stored these silently saturated at 27.631021/Xbar); exact-vs-stored rel diff "
          f"on mutually-valid entries: median {np.median(rel):.3e} max {np.max(rel):.3e} "
          f"(renorm-level typical; near-edge tails expected)")
    return out, n_invalid, n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    out = Path(a.out_dir)
    # IMMUTABLE-ARTIFACT REFUSE GUARD (ported from export_deployed_centre_dndx.py, W1 review
    # required fix 2026-07-22): the default out-dir IS the delivered frozen artifact of record
    # (money_campaign_corrected_2026-07-20, sha-pinned by the paper). Post-exact-inverse a
    # default re-run would silently overwrite it with mask-mode-different arrays under the same
    # schema id. Enforce the sidecar's "regenerate to a NEW dated directory, never edit in place".
    if (out / "PROVENANCE.json").exists():
        raise SystemExit(
            f"REFUSING to write into {out}: it already contains a PROVENANCE.json, i.e. it is a "
            f"delivered immutable artifact that a downstream consumer may pin by sha256. "
            f"Regenerate to a NEW dated directory (--out-dir) instead of editing in place.")
    out.mkdir(parents=True, exist_ok=True)

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

    # v2: RE-DERIVE the hi-arm per-draw dN/dX(z) + truth through the EXACT inverse (mask mode);
    # the pkls' stored dndx arrays predate the migration and carry silent saturation (v1 defect).
    import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
    from hcd_analysis.emulator.dndx_wc import alpha_to_dndx_exact
    dndx_on_hi, ninv_on, ntot_on = rederive_dndx_draws(on_hi, "on_hi")
    dndx_off_hi, ninv_off, ntot_off = rederive_dndx_draws(off_hi, "off_hi")
    # the truth alpha triple is in-domain by construction: exact inverse in RAISE mode.
    from hcd_analysis.emulator.closure_legb import HCD_INCIDENCE_SLOPE as _S_C
    from hcd_analysis.emulator.inference import HCD_Z_PIVOT as _ZP
    _zg_t = np.asarray(on_hi["dndx_z"], float); _xb_t = np.asarray(on_hi["dndx_Xbar"], float)
    _shape_t = ((1.0 + _zg_t)[:, None] / (1.0 + float(_ZP))) ** np.asarray(_S_C, float)[None, :]
    dndx_truth_exact = np.stack(
        [alpha_to_dndx_exact(np.asarray(on_hi["alpha_pivot_truth"], float)[None, :] * _shape_t[j],
                             float(_xb_t[j]), float(_zg_t[j]))[0] for j in range(len(_zg_t))])

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
             dndx_truth=dndx_truth_exact,
             dndx_z=np.asarray(on_hi["dndx_z"], float), dndx_Xbar=np.asarray(on_hi["dndx_Xbar"], float),
             dndx_draws_on_hi=dndx_on_hi, dndx_draws_off_hi=dndx_off_hi,
             dndx_invalid_on_hi=np.int64(ninv_on), dndx_invalid_off_hi=np.int64(ninv_off),
             # matched-pair shift inset (N=6)
             pair_d_ns=d_ns, pair_d_Ap=d_ap,
             pair_ns_shift=np.float64(d_ns.mean()), pair_ns_sem=np.float64(d_ns.std(ddof=1)/np.sqrt(N_PAIRS)),
             pair_Ap_shift=np.float64(d_ap.mean()), pair_Ap_sem=np.float64(d_ap.std(ddof=1)/np.sqrt(N_PAIRS)),
             alpha_hcd_mu=np.asarray(on_hi["alpha_hcd_mu"], float),
             alpha_hcd_sigma=np.asarray(on_hi["alpha_hcd_sigma"], float),
             PARAM_LIMITS_ns_Ap=np.array([[lo_ns, hi_ns], [lo_ap, hi_ap]]),
             note=np.array("closure-mock posteriors (truth=held-out PRIYA sim); with(on)/without(off) HCD prior; "
                           "matched-pair identical data; corrected closure center 0.18812; pre-NORC forward (option a). "
                           "v2: dndx_draws_*/dndx_truth RE-DERIVED from alpha_pivot_draws through alpha_to_dndx_exact "
                           "(mask mode; out-of-simplex draws NaN, counted in dndx_invalid_*) -- the campaign pkls' "
                           "stored dndx arrays predate the exact-inverse migration and silently saturated at "
                           "27.631021/Xbar (v1 shipped 808 such entries in dndx_draws_off_hi)"))

    # ---- ROW 1: corrected 68% prior band (recomputed from the corrected prior) ---------------
    import hcd_analysis.emulator  # noqa  (x64 before jax)
    from hcd_analysis.emulator.dndx_wc import alpha_to_dndx_exact
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
    # EXACT inverse in mask mode (readout defect B, 2026-07-22): z-scaled draws can leave the
    # occupancy simplex at high z; such draws are EXCLUDED from the percentiles (NaN) and
    # counted into PROVENANCE, instead of the old silent saturation at 27.631021/Xbar.
    band_ninv = 0
    for j, z in enumerate(zg):
        dd, ok = alpha_to_dndx_exact(smp*shape[j][None, :], float(Xb[j]), float(z), mode="mask")
        band_ninv += int(np.size(ok) - np.count_nonzero(ok))
        lo[j], hi[j] = np.nanpercentile(dd, [16, 84], axis=0)
    band_ntot = int(len(zg) * smp.shape[0])
    print(f"[row1 band] out-of-domain draws excluded (not saturated): {band_ninv}/{band_ntot}")
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
    from hcd_analysis.emulator import inference as _INF_SIG
    prov = {
        "schema": "money_campaign_corrected_v2",
        # prior-geometry pin (2026-07-23, paper-agent Q4): proves which prior era produced this
        # export, independently of the commit. Module-attribute read (rebinding-trap safe).
        "hcd_prior_signature": _INF_SIG.hcd_prior_signature(),
        "purpose": "corrected-prior money-figure campaign (paper request 2026-07-20); supersedes the "
                   "pre-correction row-2 arms so row-1 band and row-2 posteriors are consistent at the source",
        "supersedes": [
            "money_campaign_corrected_2026-07-20 (v1: money_row2_arms.npz passed the campaign pkls' "
            "stored dndx arrays through verbatim -- 808 silently saturated dndx_draws_off_hi entries "
            "at 27.631021/Xbar from the pre-migration approximate alpha_to_dndx; superseded, do not "
            "read its dndx_draws_* / dndx_truth)"],
        "schema_changes_vs_v1": [
            "dndx_draws_on_hi / dndx_draws_off_hi / dndx_truth: RE-DERIVED from the pkls' "
            "alpha_pivot_draws (truth: alpha_pivot_truth) through alpha_to_dndx_exact -- draws in "
            "mask mode (out-of-simplex -> NaN, never a clamp value), truth in raise mode; the pkls' "
            "stored dndx arrays (approximate inverse, silent saturation) are IGNORED",
            "ADDED npz keys dndx_invalid_on_hi / dndx_invalid_off_hi (per-arm out-of-domain draw-z "
            "entry counts) + the row2_dndx_invalid_draws provenance block",
            "renorm-level differences vs v1 expected in every dndx_* array and in the row-1 "
            "percentile layers: the v1 ARTIFACT was produced at commit 3033ec4, BEFORE the "
            "exact-inverse migration (a9c4d44/W1) -- its row-1 band was approximate-map plain "
            "percentiles (no mask, no invalid counts) and its sidecar carries no inverse_map/"
            "row1_band_invalid_draws fields"],
        "forward_decision": "OPTION A: corrected closure prior (LLS center 0.18812) + PRE-NORC forward "
                            "(res_corr_on=True default; NOT the deployed NORC forward). Money-figure banner stays PRE-NORC.",
        "commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "git_status_porcelain": git("status", "--porcelain"),
        "dirty_tracked_files": bool(git("status", "--porcelain", "--untracked-files=no")),
        "producer": "scripts/export_money_campaign_corrected.py",
        "producer_present_at_commit": subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "-e",
             "HEAD:scripts/export_money_campaign_corrected.py"]).returncode == 0,
        "producer_unmodified_vs_commit": subprocess.run(
            ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--",
             "scripts/export_money_campaign_corrected.py"]).returncode == 0,
        "code_sha256": {
            "scripts/export_money_campaign_corrected.py": sha256(Path(__file__)),
            "hcd_analysis/emulator/dndx_wc.py": sha256(ROOT / "hcd_analysis/emulator/dndx_wc.py"),
            "hcd_analysis/emulator/closure_legb.py": sha256(ROOT / "hcd_analysis/emulator/closure_legb.py"),
            "hcd_analysis/emulator/inference.py": sha256(ROOT / "hcd_analysis/emulator/inference.py"),
        },
        "run_command": RUN_CMD,
        "campaign_launch": {"seed": 20260621, "fold": 0, "n_mocks_fold": 8, "n_pairs": N_PAIRS,
                            "base_depth": "warmup 250 / samples 600", "hi_depth": "warmup 400 / samples 2000",
                            "arms": {k: str(v) for k, v in ARMS.items()},
                            "slurm_jobs": "54237701 on / 54237702 off / 54237703 on_hi / 54237704 off_hi"},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "jax": _jax.__version__,
                        "hostname": platform.node(), "conda_env": "emu-jax", "platform": platform.platform()},
        "inputs_sha256": inputs, "outputs_sha256": outputs,
        "inverse_map": ("alpha_to_dndx_exact, mode='mask' (EXACT inverse incl. the 4-class "
                        "renormalisation; readout defect B, 2026-07-22). Pre-2026-07-22 runs "
                        "used the approximate alpha_to_dndx, which silently saturated "
                        "out-of-domain draws at 27.631021/Xbar."),
        "row1_band_invalid_draws": {
            "n_invalid": band_ninv, "n_total": band_ntot,
            "policy": ("out-of-simplex draws (sum(alpha)>=1 after z-scaling) are EXCLUDED "
                       "from the band percentiles (NaN) and counted here, instead of the old "
                       "silent saturation")},
        "row2_dndx_invalid_draws": {
            "on_hi": {"n_invalid": ninv_on, "n_total": ntot_on},
            "off_hi": {"n_invalid": ninv_off, "n_total": ntot_off},
            "policy": ("per-arm out-of-simplex (draw, z) entries in the re-derived "
                       "dndx_draws_{on,off}_hi are NaN (masked, alpha_to_dndx_exact "
                       "mode='mask') and counted here; the campaign pkls' STORED dndx_draws "
                       "(approximate inverse, silent saturation at 27.631021/Xbar) are "
                       "ignored -- v1 shipped 808 saturated off_hi entries")},
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
    print(f"  row2 out-of-domain (NaN-masked, was silent saturation): on_hi {ninv_on}/{ntot_on}  "
          f"off_hi {ninv_off}/{ntot_off}")


if __name__ == "__main__":
    main()
