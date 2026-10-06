"""Gate E registered criteria E1 and E4 (GATE_E_SPEC v1 section 4, PU-0061). Real-data tests on the production
context: the gate C ensemble, the gate D MF product and the DESI / KS / eBOSS legs with production cuts. The emulator-
error products wait for the S10-S12 amendment; a synthetic T1 block exercises the T1 consumer of the binding."""
import ast
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401
import jax
import jax.numpy as jnp

from hcd_analysis.emulator import closure_legb as CL
from hcd_analysis.emulator import kcoord as KC
from hcd_analysis.emulator.data import PARAM_LIMITS, sampling_unit_bounds

ROOT = Path(__file__).resolve().parents[1]
GATEC = "/nfs/turbo/umor-yueyingn/mfho/hcd/emulator_v2/checkpoints/gateC"
MF = "/nfs/turbo/umor-yueyingn/mfho/hcd/emulator_v2/mf/gateD/mf_modes_all6.npz"
DLA_CORE = "/nfs/turbo/umor-yueyingn/mfho/hcd/emulator_v2/gateE/products/dla_core_gateE.npz"   # A1 rev 1 s4 (PU-0071)
L_BOX = 120.0


def _need(*paths):
    miss = [p for p in paths if not os.path.exists(p)]
    if miss:
        if os.environ.get("HCD_GATE_RUN") == "1":
            pytest.fail(f"inputs absent: {miss}")
        pytest.skip(f"inputs absent: {miss}")


@pytest.fixture(scope="module")
def prod_ctx():
    _need(f"{GATEC}/prod_repaired_seed0.eqx", MF, DLA_CORE)
    ctx, d = CL.build_legb_ctx(ensemble_ckpts=[f"{GATEC}/prod_repaired_seed{i}" for i in range(5)], mf_product=MF,
                               dla_core_product=DLA_CORE,
                               ks_kwargs=dict(resolution_float=True, k_max=0.065), with_eboss=True, metals_on=True,
                               sample_res=True)
    return ctx


def _theta_set():
    lo, hi = sampling_unit_bounds()
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = 0.5 * (lo + hi)
    out = [c]
    for h in (lo[5], hi[5]):
        for o in (lo[6], hi[6]):
            t = c.copy(); t[5], t[6] = h, o
            out.append(t)
    from scipy.stats import qmc
    s = qmc.Sobol(d=9, scramble=True, seed=20261006).random(32)
    out += list(lo + s * (hi - lo))
    return out


def _k_first_principles(z, theta_unit):
    """Independent of kcoord: k_n = (2 pi n / L) (1 + z) / (100 sqrt(Om (1+z)^3 + 1 - Om)), Om = omegamh2 / hub^2, with
    (hub, omegamh2) from the PARAM_LIMITS unit-cube map."""
    lim = np.asarray(PARAM_LIMITS, float)
    hub = lim[5, 0] + theta_unit[5] * (lim[5, 1] - lim[5, 0])
    omh2 = lim[6, 0] + theta_unit[6] * (lim[6, 1] - lim[6, 0])
    om = omh2 / hub ** 2
    n = np.arange(1, 173)
    return 2 * np.pi * n / L_BOX * (1 + z) / (100.0 * np.sqrt(om * (1 + z) ** 3 + 1 - om))


def _inputs(ctx):
    zg = np.asarray(ctx.z_global)
    tau0 = jnp.asarray(ctx.tau0_mu)
    alpha = jnp.tile(jnp.asarray(ctx.alpha_hcd_mu)[None, :], (zg.size, 1))
    cores = {l.name: ctx.dla_core_leg[l.name] for l in ctx.legs}            # the data-bin core (A1 rev 1 s4)
    return tau0, alpha, cores


def _synthetic_t1(ctx):
    rng = np.random.default_rng(0)
    out = {}
    for leg in ctx.legs:
        A = rng.normal(0, 0.005, (leg.n_z, 4, 4, 172, 4))
        out[leg.name] = jnp.asarray(np.einsum("zcekt,zdekt->zcdkt", A, A))
    return out, jnp.asarray([0.656, 0.833, 1.153, 1.331])


# ---------------------------------------------------------------------------------------------------------- E1 (a)
def test_e1a_one_binding_per_leg_z_feeds_every_consumer_and_equals_first_principles(prod_ctx, monkeypatch):
    rho, ac = _synthetic_t1(prod_ctx)
    ctx = prod_ctx._replace(rho_zb_per_leg=rho, alpha_centres=ac)
    tau0, alpha, cores = _inputs(ctx)
    calls = []
    orig = dict(kgrid=KC.kgrid, bind=KC.bind, at_data=KC.at_data)

    def spy_kgrid(k_com, z, theta):
        kg = orig["kgrid"](k_com, z, theta); calls.append(("kgrid", float(z), id(kg), kg)); return kg

    def spy_bind(kg, k):
        b = orig["bind"](kg, k); calls.append(("bind", id(kg), id(b), b)); return b

    def spy_at(b, f):
        calls.append(("at", id(b))); return orig["at_data"](b, f)
    monkeypatch.setattr(KC, "kgrid", spy_kgrid)
    monkeypatch.setattr(KC, "bind", spy_bind)
    monkeypatch.setattr(KC, "at_data", spy_at)
    worst = 0.0
    n_z_total = sum(leg.n_z for leg in ctx.legs)
    with jax.disable_jit():
        for th in _theta_set():
            calls.clear()
            ll = CL._data_loglik_legcore(ctx, jnp.asarray(th), tau0, alpha, ctx.legs, cores)
            assert np.isfinite(float(ll))
            kgs = [c for c in calls if c[0] == "kgrid"]
            binds = [c for c in calls if c[0] == "bind"]
            assert len(kgs) == n_z_total and len(binds) == n_z_total          # one per (leg, z) per evaluation
            for kg_call, b_call in zip(kgs, binds):
                assert b_call[1] == kg_call[2]                                 # the binding is of that kgrid
                worst = max(worst, float(np.max(np.abs(np.asarray(kg_call[3].k_skm) /
                                                       _k_first_principles(kg_call[1], th) - 1))))
            bind_ids = [c[2] for c in binds]
            at_ids = [c[1] for c in calls if c[0] == "at"]
            assert set(at_ids) == set(bind_ids)                                # every consumer used a binding
            for bid in bind_ids:                                               # mean (4 classes), core, T1
                assert at_ids.count(bid) == 3
    assert worst <= 1e-12, worst


# ---------------------------------------------------------------------------------------------------------- E1 (b)
def test_e1b_box_modes_identical_across_checkpoints_cache_and_mf(prod_ctx):
    from hcd_analysis.emulator import products as PR, train as T
    from hcd_analysis.emulator.data import load_cache
    k = np.asarray(prod_ctx.k_com_hmpc)
    np.testing.assert_array_equal(k, 2 * np.pi * np.arange(1, 173) / L_BOX * 0 + np.asarray(load_cache(CL.CACHE_PATH)["k_com_hmpc"]))
    _, mf_k, _ = PR.load_mf_product(MF, lf_ckpt_sha256=prod_ctx.product_digests["lf_ensemble_eqx_sha256"],
                                    lf_cache_sha256=T._sha256_or_none(CL.CACHE_PATH))
    np.testing.assert_allclose(mf_k, k, rtol=1e-14, atol=0)
    for i in range(5):
        _, meta, _ = T.load_checkpoint(f"{GATEC}/prod_repaired_seed{i}")
        np.testing.assert_array_equal(np.asarray(meta["k_com_hmpc"]), k)


# ---------------------------------------------------------------------------------------------------------- E1 (c)
DENY_NAMES = {"cache_k", "CACHE_KMAX", "build_mf_correction", "load_lf_backbone"}
DENY_ATTRS = {"cache_k", "kfkms"}
DENY_KEYS = {"kfkms", "cache_k"}
ALLOW = {("hcd_analysis.emulator.schema", None), ("hcd_analysis.emulator.data", "load_cache"),
         ("hcd_analysis.emulator.data", "datarange_mask"),
         ("hcd_analysis.emulator.mf_modes", "measure_mode_targets"), ("hcd_analysis.emulator.mf_modes", "mode_mapping_error"),
         ("hcd_analysis.emulator.mf_modes", "canonical_mapping_error"), ("hcd_analysis.emulator.mf_modes", "load_mode_mf"),
         ("hcd_analysis.emulator.products", None), ("hcd_analysis.emulator.error_vector_io", None),
         ("hcd_analysis.emulator.train", "load_checkpoint"), ("hcd_analysis.emulator.train", "save_checkpoint"),
         ("hcd_analysis.emulator.train", "_velocity_key_paths"),
         # np.load of the OBSERVATIONAL data files (not emulator products)
         ("hcd_analysis.emulator.data_likelihood", "load_desi_leg"), ("hcd_analysis.emulator.data_likelihood", "load_eboss_leg"),
         # gate-F mock-protocol injection resolvers (truth side; never called by the forward, see E1d)
         ("hcd_analysis.emulator.closure_legb", "_resolve_res_corr_inject"),
         ("hcd_analysis.emulator.closure_legb", "_resolve_res_instr_inject")}


def _reachable_modules():
    def mod_path(name):
        p = ROOT / (name.replace(".", "/") + ".py")
        if p.exists():
            return p
        p = ROOT / name.replace(".", "/") / "__init__.py"
        return p if p.exists() else None
    seen, todo = {}, [("scripts.run_real_fit", ROOT / "scripts/run_real_fit.py")]
    while todo:
        name, p = todo.pop()
        if name in seen:
            continue
        seen[name] = p
        pkg = name if p.name == "__init__.py" else name.rsplit(".", 1)[0]
        for n in ast.walk(ast.parse(p.read_text())):
            mods = []
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                base = (pkg.rsplit(".", n.level - 1)[0] if n.level > 1 else pkg) if n.level else ""
                mod = (f"{base}.{n.module}" if n.module else base) if n.level else n.module
                mods = [mod] + [f"{mod}.{a.name}" for a in n.names]
            for m in mods:
                if m and m.startswith("hcd_analysis"):
                    mp = mod_path(m)
                    if mp is not None and m not in seen:
                        todo.append((m, mp))
    return seen


def test_e1c_static_scan_of_the_production_import_graph():
    mods = _reachable_modules()
    assert "hcd_analysis.emulator.forward" in mods and "hcd_analysis.emulator.multifidelity" not in mods
    assert not any("legacy_forward" in m for m in mods)
    bad = []
    for name, p in mods.items():
        if (name, None) in ALLOW:
            continue
        tree = ast.parse(p.read_text())
        parent = {}
        for node in ast.walk(tree):
            for ch in ast.iter_child_nodes(node):
                parent[ch] = node

        def enclosing(node):
            while node in parent:
                node = parent[node]
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return node.name
            return None
        for node in ast.walk(tree):
            hit = None
            if isinstance(node, ast.Name) and node.id in DENY_NAMES:
                hit = node.id
            elif isinstance(node, ast.Attribute) and node.attr in DENY_ATTRS | DENY_NAMES:
                hit = node.attr
            elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value in DENY_KEYS:
                hit = f"[{node.slice.value!r}]"
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load"
                  and isinstance(node.func.value, ast.Name) and node.func.value.id in ("np", "numpy")):
                hit = "np.load"
            if hit and (name, enclosing(node)) not in ALLOW:
                bad.append((name, enclosing(node), hit, node.lineno))
    assert not bad, bad


# ---------------------------------------------------------------------------------------------------------- E1 (d)
def test_e1d_dynamic_call_graph_touches_no_retired_code():
    _need(f"{GATEC}/prod_repaired_seed0.eqx", MF, DLA_CORE)
    deny_files = ("multifidelity.py", "legacy_forward_pre2026_10.py", "closure_legb_figs.py")
    deny_names = {"predict_P_obs_on_leg", "build_mf_correction", "load_lf_backbone", "_predict_P_obs_mf",
                  "_emu_var_on_cache", "data_loglik", "load_mf_floor", "load_mf_emucoh", "load_mf_shape"}
    seen = set()

    def prof(frame, event, arg):
        if event == "call":
            co = frame.f_code
            seen.add((os.path.basename(co.co_filename), co.co_name))
    sys.setprofile(prof)
    try:
        ctx, _ = CL.build_legb_ctx(ensemble_ckpts=[f"{GATEC}/prod_repaired_seed{i}" for i in range(5)], mf_product=MF,
                                   dla_core_product=DLA_CORE,
                                   ks_kwargs=dict(resolution_float=True, k_max=0.065), with_eboss=True,
                                   metals_on=True, sample_res=True)
        tau0, alpha, cores = _inputs(ctx)
        with jax.disable_jit():
            CL._data_loglik_legcore(ctx, jnp.full(9, 0.5), tau0, alpha, ctx.legs, cores)
    finally:
        sys.setprofile(None)
    hits = sorted(s for s in seen if s[0] in deny_files or s[1] in deny_names)
    assert not hits, hits
    assert ("forward.py", "predict_leg") in seen and ("kcoord.py", "bind") in seen


# ---------------------------------------------------------------------------------------------------------- E4 (a)
def test_e4a_cosmology_response_of_k_and_zero_mf_derivative():
    K = np.arange(1, 173)
    kcom = 2 * np.pi * K / L_BOX
    lim = np.asarray(PARAM_LIMITS, float)
    th = jnp.asarray(np.full(9, 0.37))
    for z in (2.2, 3.0, 4.6):
        J = jax.jacfwd(lambda t: KC.kgrid(kcom, z, t).k_skm)(th)            # (172, 9)
        hub = lim[5, 0] + 0.37 * np.ptp(lim[5]); omh2 = lim[6, 0] + 0.37 * np.ptp(lim[6])
        om = omh2 / hub ** 2
        E2 = om * (1 + z) ** 3 + 1 - om
        k = np.asarray(KC.k_skm_from_kcom(kcom, z, hub, omh2))
        dk_dom = -0.5 * k * ((1 + z) ** 3 - 1) / E2
        expect_hub = dk_dom * (-2 * omh2 / hub ** 3) * np.ptp(lim[5])
        expect_omh2 = dk_dom * (1 / hub ** 2) * np.ptp(lim[6])
        np.testing.assert_allclose(np.asarray(J[:, 5]), expect_hub, rtol=1e-10)
        np.testing.assert_allclose(np.asarray(J[:, 6]), expect_omh2, rtol=1e-10)
        assert np.all(np.asarray(J[:, [0, 1, 2, 3, 4, 7, 8]]) == 0.0)
        for i, ex in ((5, expect_hub), (6, expect_omh2)):
            h = 1e-6 * float(th[i])
            fd = (np.asarray(KC.kgrid(kcom, z, th.at[i].add(h)).k_skm) - np.asarray(KC.kgrid(kcom, z, th.at[i].add(-h)).k_skm)) / (2 * h)
            np.testing.assert_allclose(fd, ex, rtol=1e-6)
    _need(MF)
    from hcd_analysis.emulator.mf_modes import ModeMF, load_mode_mf
    tables, _, _ = load_mode_mf(MF)
    mf = ModeMF.from_tables(tables)
    for z in (2.2, 3.0, 4.6):
        for tau0 in (0.2, 0.6, 1.2):
            x = jnp.concatenate([th, jnp.asarray([(z - 2.0) / 3.4])])
            Jg = jax.jacfwd(lambda xx: mf(xx, tau0))(x)
            assert np.all(np.asarray(Jg[..., :9]) == 0.0)


# ---------------------------------------------------------------------------------------------------------- E4 (b)
def test_e4b_one_fixed_grid_for_every_z_changes_the_forward_by_more_than_one_percent(prod_ctx, monkeypatch):
    tau0, alpha, cores = _inputs(prod_ctx)
    lo, hi = sampling_unit_bounds()
    th = np.asarray(0.5 * (np.asarray(lo) + np.asarray(hi)), float)
    th[5], th[6] = lo[5], hi[6]                                                # a box corner (largest Omega_m)
    from hcd_analysis.emulator import forward as FW
    good = {l.name: FW.predict_leg(prod_ctx.model, jnp.asarray(th), tau0[: l.n_z], alpha[: l.n_z], leg=l,
                                   k_com=prod_ctx.k_com_hmpc, pf_stats=prod_ctx.pf_stats, dla_core=cores[l.name],
                                   mf=prod_ctx.mf).P_model for l in prod_ctx.legs}
    orig = KC.kgrid

    def row0_grid(k_com, z, theta):
        kg = orig(k_com, z, theta)
        return KC.KGrid(4.0366e-4 * jnp.arange(1, 173, dtype=float), kg.z, kg.hub, kg.omegamh2, kg.k_com_hmpc,
                        kg.schema_version)
    monkeypatch.setattr(KC, "kgrid", row0_grid)
    worst = 0.0
    for l in prod_ctx.legs:
        bad = FW.predict_leg(prod_ctx.model, jnp.asarray(th), tau0[: l.n_z], alpha[: l.n_z], leg=l,
                             k_com=prod_ctx.k_com_hmpc, pf_stats=prod_ctx.pf_stats, dla_core=cores[l.name],
                             mf=prod_ctx.mf).P_model
        keep = np.isfinite(np.asarray(l.P_data))
        worst = max(worst, float(np.max(np.abs(np.asarray(bad)[keep] / np.asarray(good[l.name])[keep] - 1))))
    assert worst > 0.01, worst
