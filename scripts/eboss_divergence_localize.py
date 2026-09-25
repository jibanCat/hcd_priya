#!/usr/bin/env python3
"""Blind-safe localization readout of divergent NUTS draws (eBOSS prereg v1.2 amendment, section 4).

Runs ONLY on a chain directory whose health record reports divergences (a RED gate by construction:
the frozen gate needs 0 divergences for GREEN and AMBER). It never reads ``blind.lock``, never
unblinds, and never prints a value of the blinded columns (``ns``, ``Ap``): for those it reports only
the within-chain quantile, a rank statistic that is invariant under the additive blind offset.

For every divergent draw (chain c, row i from ``<root>.divergences.npz``):
  (a) the within-chain quantile of every chain column and of minusloglike (rank / N);
  (b) for every boxed NON-blinded parameter (the theta box minus ns/Ap, tau0_amp/dtau0 recovered
      exactly from the tau0 ladder, the LogUniform metal nodes from ``<root>.nuisance.json``) the
      distance to the nearer prior edge in units of the prior width (log10 width for the metal nodes);
  (c) whether the divergent rows cluster (consecutive rows) or are isolated.
Writes ``<out>.json`` and ``<out>.md``. Inputs: the chain directory and analysis.lock only.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLINDED = ("ns", "Ap")
EDGE_FLAG = 0.02        # "near edge" = within 2 percent of the prior width (the readout's rail convention)


def refuse(msg):
    print(f"REFUSE: {msg}", file=sys.stderr)
    sys.exit(3)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def within_chain_quantile(col, i):
    """rank statistic: fraction of the chain's draws strictly below draw i plus half the ties (0..1)."""
    x = np.asarray(col, float)
    return float((np.sum(x < x[i]) + 0.5 * np.sum(x == x[i])) / x.size)


def edge_distance(v, lo, hi, log10=False):
    """distance to the nearer prior edge in units of the prior width; None outside the box."""
    if log10:
        v, lo, hi = np.log10(v), np.log10(lo), np.log10(hi)
    w = hi - lo
    if not (lo - 1e-12 <= v <= hi + 1e-12):
        return None
    return float(min(v - lo, hi - v) / w)


def cluster_flags(rows):
    rows = sorted(int(r) for r in rows)
    return [dict(row=r, consecutive_with_previous=(k > 0 and r == rows[k - 1] + 1)) for k, r in enumerate(rows)]


def read_chain(path):
    with open(path) as f:
        cols = f.readline()[1:].split()
    if cols[:2] != ["weight", "minusloglike"]:
        refuse(f"bad header in {path}")
    return cols, np.loadtxt(path, ndmin=2)


def analyze(chain_dir, root, lock, box, recover, z, z_pivot):
    dv_p = os.path.join(chain_dir, f"{root}.divergences.npz")
    if not os.path.exists(dv_p):
        refuse(f"missing {dv_p} (the rerun driver writes it; the first run has none)")
    dv = np.load(dv_p)
    flags = np.asarray(dv["diverging"], bool)
    chain_files = [str(s) for s in dv["chain_files"]]
    health = json.load(open(os.path.join(chain_dir, f"{root}.health.json")))
    if int(health.get("n_divergent", 0)) <= 0 or int(flags.sum()) <= 0:
        refuse("no divergent draw recorded: the localization readout is preregistered for a RED gate only")
    if flags.sum(axis=1).astype(int).tolist() != [int(x) for x in health["per_chain_div"]]:
        refuse("divergence flags disagree with the health record")
    nuis_p = os.path.join(chain_dir, f"{root}.nuisance.npz")
    nuis = {k: np.asarray(v) for k, v in np.load(nuis_p).items()} if os.path.exists(nuis_p) else {}
    nuis_json = os.path.join(chain_dir, f"{root}.nuisance.json")
    nb = json.load(open(nuis_json)).get("bounds", {}) if os.path.exists(nuis_json) else {}
    out = []
    for c, fn in enumerate(chain_files):
        cols, tab = read_chain(os.path.join(chain_dir, fn))
        names = cols[2:]
        if tab.shape[0] != flags.shape[1]:
            refuse(f"{fn}: {tab.shape[0]} rows != {flags.shape[1]} divergence flags")
        tau_idx = [k + 2 for k, n in enumerate(names) if n.startswith("tau0_z")]
        amp, dt = recover(tab[:, tau_idx], z, z_pivot=z_pivot)
        rows = np.flatnonzero(flags[c])
        clus = {d["row"]: d["consecutive_with_previous"] for d in cluster_flags(rows)}
        for i in rows:
            q = {"minusloglike": within_chain_quantile(tab[:, 1], i)}
            for k, n in enumerate(names):
                q[n] = within_chain_quantile(tab[:, k + 2], i)
            q["tau0_amp"], q["dtau0"] = within_chain_quantile(amp, i), within_chain_quantile(dt, i)
            for s, arr in nuis.items():
                if arr.shape == flags.shape:
                    q[s] = within_chain_quantile(arr[c], i)
            edges = {}
            for n in names:
                if n in BLINDED or n not in box:
                    continue
                edges[n] = edge_distance(tab[i, names.index(n) + 2], *box[n])
            edges["tau0_amp"] = edge_distance(amp[i], *box["tau0_amp"])
            edges["dtau0"] = edge_distance(dt[i], *box["dtau0"])
            for s, arr in nuis.items():
                key = "f" if s.startswith(("f_SiIII_", "f_SiII_")) else ("k" if s.startswith(("k_SiIII_", "k_SiII_")) else None)
                if key and key in nb and arr.shape == flags.shape:
                    edges[s] = edge_distance(arr[c, i], *nb[key], log10=True)
            near = sorted(n for n, d in edges.items() if d is not None and d <= EDGE_FLAG)
            out.append(dict(chain=c, chain_file=fn, row=int(i), consecutive_with_previous=bool(clus[int(i)]),
                            quantiles=q, edge_distance=edges, near_edge=near,
                            verdict=("near edge " + ", ".join(near)) if near else "interior"))
    return dict(schema="eboss_divergence_localization.v1", root=root, chain_dir=os.path.abspath(chain_dir),
                n_divergent=int(flags.sum()), per_chain_div=[int(x) for x in flags.sum(axis=1)],
                n_draws=int(flags.shape[1]), blinded_columns_reported_as="within-chain quantile only",
                edge_flag_fraction=EDGE_FLAG, draws=out)


def to_markdown(res):
    lines = [f"# Divergence localization (blind-safe): {res['root']}", "",
             f"{res['n_divergent']} divergent draws of {res['n_draws']} per chain; per chain {res['per_chain_div']}. "
             "Blinded columns (ns, Ap) appear as within-chain quantiles only.", ""]
    for d in res["draws"]:
        lines.append(f"## chain {d['chain']} row {d['row']} ({'consecutive' if d['consecutive_with_previous'] else 'isolated'}): {d['verdict']}")
        lines.append("")
        lines.append("| column | within-chain quantile | edge distance (prior widths) |")
        lines.append("|---|---|---|")
        for n, q in d["quantiles"].items():
            e = d["edge_distance"].get(n)
            lines.append(f"| {n} | {q:.3f} | {'' if e is None else f'{e:.3f}'} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain-dir", required=True)
    ap.add_argument("--root", default="real_eboss")
    ap.add_argument("--analysis-lock", required=True)
    ap.add_argument("--leg", default="eBOSS")
    ap.add_argument("--out", required=True, help="output prefix (writes <out>.json and <out>.md)")
    a = ap.parse_args(argv)
    if os.path.exists(os.path.join(a.chain_dir, "UNBLINDED.stamp")):
        refuse("this directory was unblinded: the localization readout is for a RED (never unblinded) directory")
    cons = _load("eboss_priya_consistency")
    lock = json.load(open(a.analysis_lock))
    leg = lock["legs"][a.leg]
    z_lock = np.asarray(leg["z"] if "z" in leg else lock["surveys"][a.leg.lower()]["z"], float)
    health0 = json.load(open(os.path.join(a.chain_dir, f"{a.root}.health.json")))
    if health0.get("z_kept"):                       # a restricted product (PI #28): its own grid, a subset of the lock grid
        z = np.asarray(health0["z_kept"], float)
        if not all(np.any(np.isclose(zz, z_lock, atol=1e-6)) for zz in z):
            refuse("health.z_kept is not a subset of the lock's leg z grid")
    else:
        z = z_lock
    zp = float(leg.get("prior", {}).get("tau0_pivot_z", cons.TAU0_PIVOT_Z))
    res = analyze(a.chain_dir, a.root, lock, cons.BOX, cons.recover_tau0_amp_dtau0, z, zp)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
        f.write("\n")
    with open(a.out + ".md", "w") as f:
        f.write(to_markdown(res))
    print(f"wrote {a.out}.json and .md: {res['n_divergent']} divergent draws; verdicts "
          f"{[d['verdict'] for d in res['draws']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
