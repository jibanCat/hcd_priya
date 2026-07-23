"""Export Leg-B NUTS posterior draws (from the shard .pkl bookkeeping) into
cobaya/GetDist-readable chains: per mock a <root>.1.txt + <root>.paramnames + <root>.yaml.

The .pkl is internal coverage bookkeeping; the CHAIN PRODUCT is this GetDist/cobaya format
(weight  minuslogpost  <params...>), so GetDist/cobaya tooling reads it directly. Unit-cube
theta is exported as-is for now (closure coverage is computed in sampled space); the
unit->physical map for the real-data fit is a flagged refinement.
"""
import argparse
import glob
import os
import pickle

import numpy as np

from hcd_analysis.emulator.inference import PARAM_NAMES

LABELS = {  # GetDist LaTeX labels
    "ns": "n_s", "Ap": "A_p", "herei": r"z_{\rm HeII,i}", "heref": r"z_{\rm HeII,f}",
    "alphaq": r"\alpha_q", "hub": "h", "omegamh2": r"\Omega_m h^2",
    "hireionz": r"z_{\rm HI,reion}", "bhfeedback": r"\epsilon_{\rm BH}",
    "alpha_lls": r"\alpha_{\rm LLS}", "alpha_subdla": r"\alpha_{\rm subDLA}",
    "alpha_dla": r"\alpha_{\rm DLA}",
}


def param_names(rec):
    """Packed draw order = [theta9, tau0(kept global z), alpha_lls, alpha_subdla, alpha_dla]."""
    n_tau0 = int(rec["draws"].shape[1]) - len(PARAM_NAMES) - 3
    names = list(PARAM_NAMES) + [f"tau0_{i}" for i in range(n_tau0)] + \
        ["alpha_lls", "alpha_subdla", "alpha_dla"]
    labels = [LABELS.get(n, rf"\tau_0[{n.split('_')[1]}]" if n.startswith("tau0_") else n)
              for n in names]
    return names, labels


def export_mock(rec, root):
    draws = np.asarray(rec["draws"])                       # (L, P)
    L = draws.shape[0]
    minuslogpost = -np.asarray(rec.get("ll_draws", np.zeros(L)))[:L]  # -loglik proxy
    weight = np.ones(L)
    table = np.column_stack([weight, minuslogpost, draws])
    names, labels = param_names(rec)
    np.savetxt(f"{root}.1.txt", table,
               fmt=["%.6g"] * table.shape[1],
               header="weight  minuslogpost  " + "  ".join(names))
    with open(f"{root}.paramnames", "w") as f:
        for n, lab in zip(names, labels):
            f.write(f"{n}\t{lab}\n")
    # minimal cobaya-style yaml (params + sampler + Leg-B truth, for provenance)
    truth = np.asarray(rec["truth_vec"])
    with open(f"{root}.yaml", "w") as f:
        f.write("# cobaya/GetDist-style metadata for a Leg-B mock NUTS chain\n")
        f.write(f"# sim: {rec.get('sim','?')}\n")
        f.write("sampler:\n  numpyro_nuts: {dense_mass: true}\n")
        f.write("params:\n")
        for n, lab, tv in zip(names, labels, truth):
            f.write(f"  {n}: {{latex: '{lab}', truth: {float(tv):.6g}}}\n")
    return f"{root}.1.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True, help="dir of shard_*.pkl")
    ap.add_argument("--out-dir", required=True, help="where to write the cobaya/GetDist chains")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    n = 0
    for fn in sorted(glob.glob(os.path.join(a.shard_dir, "shard_*.pkl"))):
        with open(fn, "rb") as f:
            dd = pickle.load(f)
        for rec, mi in zip(dd["per_mock"], dd["idxs"]):
            root = os.path.join(a.out_dir, f"legb_mock_{mi:03d}")
            export_mock(rec, root)
            n += 1
    print(f"exported {n} mock chains (cobaya/GetDist) -> {a.out_dir}")


if __name__ == "__main__":
    main()
