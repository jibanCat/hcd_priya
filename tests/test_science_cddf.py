"""Science checks of the incidence and the CDDF of hcd_priya that need no trained model.

- The decode of the incidence outputs (dN/dX and f(N_HI)) in the predictor is the inverse of the training encoding;
  so is the decode of the difference between the unfiltered and the filtered class spectra (the ``delta`` output).
- The CDDF routine of the pipeline has the documented normalisation: absorbers per unit N_HI (linear, cm^-2) per
  unit absorption distance X, with dX = (1 + z)^2 L_com H_0 / c per sightline.
- On absorber counts, the CDDF summed over a class's column-density range IS that class's dN/dX for LLS and for
  subDLA + DLA; for subDLA and for DLA alone it lies in the interval the bin across 20.3 allows.

Every number here is synthetic or a published constant. The decode tests call private functions of the predictor
(_predict_many, _cddf_many, _predict_one, _cddf_one), the ones every prediction goes through: a rename
breaks these tests, not the package. ``_forward_many``, which the builders of the multi-fidelity targets use, is
checked to give the same spectra and dN/dX as ``_predict_many``, value for value.
"""

import math
from types import SimpleNamespace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import hcd_analysis.predictor as hp
from tests import science_cddf_sums as GC
from hcd_analysis import catalog as CAT
from hcd_analysis import cddf as CD
from hcd_analysis.emulator import data as D

C_KMS = 299792.458  # the speed of light, km/s (exact, SI definition)
H0_PER_H = 100.0  # H_0 / h in km/s/Mpc


# ===================================================================== the decode of the incidence
class _Member(eqx.Module):
    """A stand-in network member: for the input whose first entry is r, it returns row r of its stored outputs (the
    five outputs the predictor reads: P_filt_base, P_filt_resid, dndx, f_nhi, delta). Synthetic values only.
    """

    out: dict

    def __call__(self, x, tau):
        r = jnp.asarray(x[0]).astype(jnp.int32)
        return {k: v[r] for k, v in self.out.items()}


class _Ensemble(eqx.Module):
    members: list


N_BINS, K_SYN = 30, 6
SPREAD = 0.3  # the two members encode value x (1 + SPREAD) and value x (1 - SPREAD): their linear mean is the value


def synthetic_cache(scale=1.0, seed=11):
    """A training cache in the layout ``data.make_batch`` reads: 6 snapshot groups (2 simulations x 3 redshifts), 3
    mean-flux rows each. f(N_HI) of the size of a real CDDF (2e-18 to 1e-27 per cm^-2), bin 0 a structural zero, one
    empty bin elsewhere; dN/dX of order 0.03 to 0.5. ``scale`` multiplies dN/dX, f(N_HI) and P_filt (not the
    zeros)."""
    rng = np.random.default_rng(seed)
    zs, n_alpha = (2.2, 3.0, 4.2), 3
    n_groups = 2 * len(zs)
    n_rows = n_groups * n_alpha
    centres = 17.1 + 0.2 * np.arange(N_BINS)
    f = 10.0 ** (-17.5 - 1.6 * (centres - 17.0))[None, :] * rng.uniform(
        0.5, 2.0, (n_groups, N_BINS)
    )
    f[:, 0] = 0.0
    f[3, 12] = 0.0
    dndx = np.array([0.3, 0.06, 0.03])[None, :] * rng.uniform(0.6, 1.6, (n_groups, 3))
    grp = np.repeat(np.arange(n_groups), n_alpha)
    p = rng.uniform(0.05, 0.2, (n_rows, 4, K_SYN))
    return {
        "snap_group_idx": grp,
        "snap_f_nhi": f * scale,
        "snap_dNdX": dndx * scale,
        "z_grid": np.array([zs[g % len(zs)] for g in grp]),
        "alpha_idx": np.tile(np.arange(n_alpha), n_groups),
        "P_filt": p * scale,
        "delta": rng.normal(scale=1e-3, size=(n_rows, 3, K_SYN)),
        "x": rng.uniform(size=(n_rows, 10)),
        "tau0": rng.uniform(0.2, 1.0, n_rows),
        "mask": np.ones((n_rows, 4, K_SYN), dtype=bool),
        "inv_nc": np.ones((n_rows, 4)),
    }


def _norm_for_predictor(norm):
    pf = {
        k: jnp.asarray(np.asarray(norm["P_filt"][k], dtype=np.float64))
        for k in ("mu_marg", "sig_marg", "sig_cosmo")
    }
    dn = (jnp.asarray(norm["dndx"]["mean"]), jnp.asarray(norm["dndx"]["std"]))
    fn = (jnp.asarray(norm["f_nhi"]["mean"]), jnp.asarray(norm["f_nhi"]["std"]))
    dl = (jnp.asarray(norm["delta"]["mean"]), jnp.asarray(norm["delta"]["std"]))
    return pf, dn, fn, dl


def test_the_predictor_decodes_the_incidence_with_the_inverse_of_the_training_encoding():
    """Training encodes dN/dX and f(N_HI) as (ln max(x, 1e-30) - mu) / sigma (``data.make_batch`` with the
    normaliser of ``data.fit_target_norm``, ``data.TARGET_TRANSFORMS``); the predictor decodes each member's output o
    as exp(o sigma + mu) (``predictor._predict_many`` and ``predictor._cddf_many``, the jitted functions every
    stacked prediction uses) and then takes the linear mean over the members (that mean is done here as the
    predictor does it; the predictor's own mean is checked by test_predictor.py). Encode a synthetic cache with the
    training functions (two members: the values x 1.3 and x 0.7), feed the encodings to the predictor's decoders as
    the outputs of two stand-in members, and get the cache's values back to 1e-12 relative from the linear mean:
    dN/dX, f(N_HI) at every non-empty bin, and (in passing) P_filt.
    The same for the ``delta`` output, the difference between the unfiltered and the filtered class spectra, which
    training encodes as (arcsinh(x) - mu) / sigma and the predictor decodes as sinh(o sigma + mu): here the two
    members hold the same values, of either sign, and the decode returns them.
    An empty bin (structural zero) is encoded at the floor and decodes to the floor 1e-30, not to zero (the
    predictor sets such bins NaN elsewhere); the never-populated bin keeps the placeholder normaliser mean 0, std 1.
    The training-side inverse ``data.untransform_prediction`` gives the same values. No trained model.
    """
    d = synthetic_cache()
    rows = np.arange(d["x"].shape[0])
    norm = D.fit_target_norm(d, rows)
    assert (
        norm["f_nhi"]["mean"][0] == 0.0 and norm["f_nhi"]["std"][0] == 1.0
    )  # the placeholder of bin 0
    batches = [D.make_batch(synthetic_cache(scale=1.0 + s), rows, norm) for s in (SPREAD, -SPREAD)]
    for b in batches:  # the encoding marks exactly the zeros as not to be learned
        assert np.array_equal(b["t_f_nhi_mask"], d["snap_f_nhi"][d["snap_group_idx"]] > 0)
        assert b["t_dndx_mask"].all()
    members = [
        _Member(
            {
                "P_filt_base": jnp.asarray(b["t_p_base"]),
                "P_filt_resid": jnp.asarray(b["t_p_resid"]),
                "dndx": jnp.asarray(b["t_dndx"]),
                "f_nhi": jnp.asarray(b["t_f_nhi"]),
                "delta": jnp.asarray(b["t_delta"]),
            }
        )
        for b in batches
    ]
    ens = _Ensemble(members)
    pf, dn, fn, dl = _norm_for_predictor(norm)
    x = np.zeros((rows.size, 10))
    x[:, 0] = rows
    tau = jnp.asarray(d["tau0"])
    P_m, dn_m, D_m = hp._predict_many(
        ens, pf, *dn, *dl, jnp.asarray(x), tau
    )  # (rows, members, 4, K), (rows, members, 3), (rows, members, 3, K)
    # the function of the builders of the multi-fidelity targets gives the same spectra and dN/dX, value for value
    P_b, dn_b = hp._forward_many(ens, pf, *dn, jnp.asarray(x), tau)
    assert np.array_equal(np.asarray(P_b), np.asarray(P_m))
    assert np.array_equal(np.asarray(dn_b), np.asarray(dn_m))
    f_m = hp._cddf_many(ens, *fn, jnp.asarray(x), tau)  # (rows, members, Nbin)
    dndx, cddf, P = (np.asarray(a, dtype=np.float64).mean(axis=1) for a in (dn_m, f_m, P_m))

    want_dndx = d["snap_dNdX"][d["snap_group_idx"]]
    want_f = d["snap_f_nhi"][d["snap_group_idx"]]
    full = want_f > 0
    np.testing.assert_allclose(dndx, want_dndx, rtol=1e-12, atol=0)
    np.testing.assert_allclose(cddf[full], want_f[full], rtol=1e-12, atol=0)
    np.testing.assert_allclose(cddf[~full], 1e-30, rtol=1e-12, atol=0)  # the floor of the encoding
    np.testing.assert_allclose(P, d["P_filt"], rtol=1e-12, atol=0)
    delta = np.asarray(D_m, dtype=np.float64).mean(axis=1)
    assert np.any(d["delta"] < 0) and np.any(d["delta"] > 0)
    # values of order 1e-3 and of either sign: a relative comparison with no absolute tolerance
    np.testing.assert_allclose(delta, d["delta"], rtol=1e-9, atol=0)
    # the training-side inverse agrees (per member, then the same linear mean)
    back = [
        D.untransform_prediction(
            {
                "dndx": b["t_dndx"],
                "f_nhi": b["t_f_nhi"],
                "P_filt_base": b["t_p_base"],
                "P_filt_resid": b["t_p_resid"],
                "delta": b["t_delta"],
            },
            norm,
        )
        for b in batches
    ]
    np.testing.assert_allclose(
        np.mean([b["delta"] for b in back], axis=0), d["delta"], rtol=1e-9, atol=0
    )
    np.testing.assert_allclose(
        np.mean([b["dndx"] for b in back], axis=0), want_dndx, rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        np.mean([b["f_nhi"] for b in back], axis=0)[full], want_f[full], rtol=1e-12, atol=0
    )


def test_the_incidence_decode_gives_a_known_answer_written_by_hand():
    """mu = ln(0.25, 0.05, 0.02), sigma = (0.5, 1, 2) and network output o = (0, 1, -0.5) decode to dN/dX =
    (0.25, 0.05 e, 0.02 / e) = (0.25, 0.1359140914229523, 0.007357588823428847); mu = ln 1e-20, sigma = 2, o = -1 to
    f = 1e-20 e^-2 = 1.353352832366127e-21 in every bin. For the difference between the unfiltered and the filtered
    class spectra: mu = (0.5, 0.5, -1), sigma = (2, 1, 0.5) and o = (0.25, -1, 6) decode to sinh(1, -0.5, 2) =
    (1.1752011936438014, -0.5210953054937474, 3.626860407847019) at every mode. Through the predictor's single-input
    decoders (``_predict_one``, ``_cddf_one``) with two identical stand-in members; and the training encoding of
    those values gives o back."""
    mu_d, sd_d = np.log([0.25, 0.05, 0.02]), np.array([0.5, 1.0, 2.0])
    o_d = np.array([0.0, 1.0, -0.5])
    mu_f, sd_f = np.full(N_BINS, math.log(1e-20)), np.full(N_BINS, 2.0)
    o_f = np.full(N_BINS, -1.0)
    mu_l = np.tile(np.array([0.5, 0.5, -1.0])[:, None], (1, K_SYN))
    sd_l = np.tile(np.array([2.0, 1.0, 0.5])[:, None], (1, K_SYN))
    o_l = np.tile(np.array([0.25, -1.0, 6.0])[:, None], (1, K_SYN))
    want_l = np.tile(
        np.array([1.1752011936438014, -0.5210953054937474, 3.626860407847019])[:, None], (1, K_SYN)
    )
    out = {
        "P_filt_base": jnp.zeros((1, 4, K_SYN)),
        "P_filt_resid": jnp.zeros((1, 4, K_SYN)),
        "dndx": jnp.asarray(o_d[None]),
        "f_nhi": jnp.asarray(o_f[None]),
        "delta": jnp.asarray(o_l[None]),
    }
    ens = _Ensemble([_Member(out), _Member(out)])
    pf = {k: jnp.zeros((4, K_SYN)) for k in ("mu_marg", "sig_marg", "sig_cosmo")}
    x, tau = jnp.zeros(10), jnp.asarray(0.5)
    _, dn, dl = hp._predict_one(
        ens, pf, jnp.asarray(mu_d), jnp.asarray(sd_d), jnp.asarray(mu_l), jnp.asarray(sd_l), x, tau
    )
    f = hp._cddf_one(ens, jnp.asarray(mu_f), jnp.asarray(sd_f), x, tau)
    np.testing.assert_allclose(
        np.asarray(dn), [[0.25, 0.1359140914229523, 0.007357588823428847]] * 2, rtol=1e-14, atol=0
    )
    np.testing.assert_allclose(np.asarray(f), 1.353352832366127e-21, rtol=1e-14, atol=0)
    np.testing.assert_allclose(np.asarray(dl), [want_l] * 2, rtol=1e-14, atol=0)
    enc_l = D.apply_norm(D.TARGET_TRANSFORMS["delta"][0](want_l), {"mean": mu_l, "std": sd_l})
    np.testing.assert_allclose(enc_l, o_l, rtol=0, atol=1e-14)
    enc = D.apply_norm(
        D.TARGET_TRANSFORMS["dndx"][0](np.array([0.25, 0.1359140914229523, 0.007357588823428847])),
        {"mean": mu_d, "std": sd_d},
    )
    np.testing.assert_allclose(enc, o_d, rtol=0, atol=1e-14)
    enc_f = D.apply_norm(
        D.safe_log(np.full(N_BINS, 1.353352832366127e-21)), {"mean": mu_f, "std": sd_f}
    )
    np.testing.assert_allclose(enc_f, o_f, rtol=0, atol=1e-14)


# =================================================================== the normalisation of the CDDF
def _absorber(i, log_n):
    nhi = 10.0**log_n
    return CAT.Absorber(
        skewer_idx=i,
        pix_start=0,
        pix_end=1,
        v_center=0.0,
        NHI=nhi,
        b_kms=float("nan"),
        log_NHI=float(log_n),
        absorber_class=CAT.classify_system(nhi),
        fit_success=True,
        fast_mode=True,
    )


def catalogue(log_values, z):
    return CAT.AbsorberCatalog(
        sim_name="synthetic",
        snap=0,
        z=z,
        dv_kms=10.0,
        absorbers=[_absorber(i, v) for i, v in enumerate(log_values)],
    )


def header(z, hubble=0.7, omegam=0.3, n_skewers=1000, box_kpc_h=120000.0):
    return SimpleNamespace(
        box=box_kpc_h,
        hubble=hubble,
        omegam=omegam,
        omegal=1.0 - omegam,
        redshift=z,
        n_skewers=n_skewers,
    )


def dx_per_sightline(z, box_kpc_h=120000.0):
    """The absorption distance of one sightline through a comoving box of L Mpc/h: dX = (1 + z)^2 L H_0 / c with L
    in Mpc and H_0 = 100 h km/s/Mpc, so h cancels: (1 + z)^2 (L in Mpc/h) 100 / c. Written out here.
    """
    return (1.0 + z) ** 2 * (box_kpc_h / 1000.0) * H0_PER_H / C_KMS


def test_the_cddf_is_absorbers_per_unit_column_density_per_unit_absorption_distance():
    """``cddf.measure_cddf`` (the routine the pipeline writes cddf.npz with) and ``measure_cddf_from_dataframe``,
    on a catalogue with a known number of absorbers in every bin of the default 30 bins over log10 N_HI 17 to 23:
    f_i = n_i / ((10^upper_i - 10^lower_i) n_sightlines dX) to 1e-12, with dX = (1 + z)^2 L H_0 / c (0.6404430627804519
    at z = 3 for 120 Mpc/h, by hand); the same f for h 0.65 and 0.75 and Omega_m 0.25 and 0.35 at a fixed box in
    Mpc/h; and the sum of f dN over the bins times the total path is the number of absorbers (the integral of f over
    N is the absorbers per unit X). At z 2.2, 3.0, 4.6."""
    assert dx_per_sightline(3.0) == pytest.approx(0.6404430627804519, rel=1e-15, abs=0)
    edges = np.linspace(17.0, 23.0, 31)
    n_i = 1 + (7 * np.arange(30)) % 13
    values = np.concatenate([edges[i] + 0.2 * (np.arange(n) + 0.5) / n for i, n in enumerate(n_i)])
    for z in (2.2, 3.0, 4.6):
        cat = catalogue(values, z)
        path = 1000 * dx_per_sightline(z)
        want = n_i / ((10.0 ** edges[1:] - 10.0 ** edges[:-1]) * path)
        results = [CD.measure_cddf(cat, header(z))]
        results.append(
            CD.measure_cddf_from_dataframe({"log_nhi": values}, z, 120000.0, 0.7, 0.3, 0.7, 1000)
        )
        for h, om in ((0.65, 0.25), (0.75, 0.35)):
            results.append(CD.measure_cddf(cat, header(z, hubble=h, omegam=om)))
        for r in results:
            assert np.array_equal(r["n_absorbers"], n_i)
            assert r["dX_per_sightline"] == pytest.approx(dx_per_sightline(z), rel=1e-14, abs=0)
            assert r["total_path"] == pytest.approx(path, rel=1e-14, abs=0)
            np.testing.assert_allclose(r["log_nhi_edges"], edges, rtol=0, atol=1e-12)
            np.testing.assert_allclose(r["f_nhi"], want, rtol=1e-12, atol=0)
            assert float(
                np.sum(GC.per_bin_per_x(r["f_nhi"], r["log_nhi_edges"]))
            ) * path == pytest.approx(values.size, rel=1e-12, abs=0)


def test_the_cddf_recovers_a_known_power_law_within_poisson_errors():
    """A catalogue drawn from f(N) = A N^-1.7 on [1e17, 1e23] cm^-2 with 30 absorbers per unit X (seeded): the
    expected count in a bin is A X_total (N_lo^-0.7 - N_hi^-0.7) / 0.7, so the expected f_i is the bin average of
    A N^-1.7 (measure_cddf reports the bin average at the bin centre; for 0.2 dex bins and slope -1.7 it is 0.35 per
    cent below f at the centre). Each bin with at least 25 expected absorbers within 5 Poisson standard deviations;
    chi2 over the bins with at least 5 within 5 standard deviations of its degrees of freedom."""
    beta, per_x, z = 1.7, 30.0, 3.0
    n_lo, n_hi = 1e17, 1e23
    path = 1000 * dx_per_sightline(z)
    rng = np.random.default_rng(20261117)
    n = rng.poisson(per_x * path)
    g = 1.0 - beta
    u = rng.random(n)
    log_n = np.log10((n_lo**g - u * (n_lo**g - n_hi**g)) ** (1.0 / g))
    r = CD.measure_cddf(catalogue(log_n, z), header(z))
    e = r["log_nhi_edges"]
    expect = per_x * path * (10.0 ** (g * e[:-1]) - 10.0 ** (g * e[1:])) / (n_lo**g - n_hi**g)
    dn = 10.0 ** e[1:] - 10.0 ** e[:-1]
    f_expect = expect / (dn * path)
    big = expect >= 25
    assert big.sum() >= 10
    sd = np.sqrt(expect[big]) / (dn[big] * path)
    assert np.all(np.abs(r["f_nhi"][big] - f_expect[big]) <= 5.0 * sd), (
        r["f_nhi"][big] / f_expect[big]
    )
    ok = expect >= 5
    chi2 = float(np.sum((r["n_absorbers"][ok] - expect[ok]) ** 2 / expect[ok]))
    dof = int(ok.sum())
    assert abs(chi2 - dof) <= 5.0 * math.sqrt(2.0 * dof), (chi2, dof)
    assert float(np.sum(r["f_nhi"] * dn)) * path == pytest.approx(n, rel=1e-12, abs=0)


# ================================================================ the CDDF against dN/dX on counts
def test_on_counts_the_cddf_of_a_class_is_its_dndx_up_to_the_bin_across_20_3():
    """The pipeline's two products of one snapshot, on a synthetic catalogue: f(N_HI) from ``cddf.measure_cddf``
    and the class counts from ``catalog.reconcile_counts`` (written to meta.json), from which the cache builder
    takes dN/dX_c = count_c / total path (``scripts/build_emulator_cache.py::compute_dndx_per_class``, written out
    here). The class edges 17.2 and 19.0 are bin edges, 20.3 the centre of the bin [20.2, 20.4). Asserted:
    - LLS: the sum of f dN over its whole bins equals dN/dX_LLS (1e-12);
    - subDLA + DLA: the sum from 19.0 up equals their dN/dX, less the absorbers above the last edge (log N 23),
      which dN/dX counts as DLAs and the CDDF does not (1e-12; the difference times the path is that integer);
    - subDLA and DLA alone: dN/dX lies between the whole-bin sum and the whole-bin sum plus the bin across 20.3,
      and equals the whole-bin sum plus the absorbers of that bin on the class's side (1e-12);
    - splitting that bin with a share s for subDLA errs by at most max(s, 1 - s) of its content, and by exactly
      half of it for s = 1/2 when all its absorbers lie above 20.3."""
    edges = np.linspace(17.0, 23.0, 31)
    for x in (CAT.LOG_NHI_LLS_MIN, CAT.LOG_NHI_SUBDLA_MIN):
        assert np.min(np.abs(edges - x)) < GC.EDGE_TOL
    centres = 0.5 * (edges[:-1] + edges[1:])
    assert np.min(np.abs(centres - CAT.LOG_NHI_DLA_MIN)) < GC.EDGE_TOL
    z = 3.0
    path = 1000 * dx_per_sightline(z)
    rng = np.random.default_rng(3)
    base = np.concatenate(
        [
            rng.uniform(17.0, 17.2 - 1e-6, 40),  # forest: in the CDDF's first bin, in no class
            rng.uniform(17.2 + 1e-6, 19.0 - 1e-6, 400),
            rng.uniform(19.0 + 1e-6, 20.2 - 1e-6, 120),
            rng.uniform(20.4 + 1e-6, 22.9, 60),
        ]
    )
    for n_below, n_above, n_over in ((7, 5, 0), (7, 5, 2), (0, 9, 0)):
        vals = np.concatenate(
            [
                base,
                rng.uniform(20.2 + 1e-6, 20.3 - 1e-6, n_below),  # the bin across 20.3, subDLA side
                rng.uniform(20.3 + 1e-6, 20.4 - 1e-6, n_above),  # its DLA side
                rng.uniform(23.1, 23.6, n_over),  # above the CDDF's last edge: DLAs
            ]
        )
        cat = catalogue(vals, z)
        r = CD.measure_cddf(cat, header(z))
        counts = CAT.reconcile_counts(cat, r["n_absorbers"], r["log_nhi_edges"])
        dndx = {c: counts[c] / r["total_path"] for c in GC.CLASS_RANGES}
        s = GC.class_sums(r["f_nhi"], r["log_nhi_edges"])
        assert s["LLS"]["straddle_bins"].size == 0
        assert [list(s[c]["straddle_bins"]) for c in ("subDLA", "DLA")] == [[16], [16]]
        straddle = s["subDLA"]["straddle"]
        assert straddle * path == pytest.approx(n_below + n_above, rel=1e-12)
        assert s["LLS"]["whole"] == pytest.approx(dndx["LLS"], rel=1e-12, abs=0)
        both = s["subDLA"]["whole"] + straddle + s["DLA"]["whole"]
        assert both + n_over / path == pytest.approx(dndx["subDLA"] + dndx["DLA"], rel=1e-12, abs=0)
        assert (dndx["subDLA"] + dndx["DLA"] - both) * path == pytest.approx(n_over, abs=1e-9)
        assert s["subDLA"]["whole"] + n_below / path == pytest.approx(
            dndx["subDLA"], rel=1e-12, abs=0
        )
        assert s["DLA"]["whole"] + (n_above + n_over) / path == pytest.approx(
            dndx["DLA"], rel=1e-12, abs=0
        )
        for c in ("subDLA", "DLA"):
            extra = n_over / path if c == "DLA" else 0.0
            assert (
                s[c]["whole"]
                <= dndx[c] * (1 + 1e-12)
                <= (s[c]["whole"] + straddle + extra) * (1 + 2e-12)
            )
        for share in (0.0, 0.25, 0.5, 0.8, 1.0):
            err = s["subDLA"]["whole"] + share * straddle - dndx["subDLA"]
            assert abs(err) <= max(share, 1.0 - share) * straddle * (1 + 1e-12) + 1e-15
        if n_below == 0:
            err = s["subDLA"]["whole"] + 0.5 * straddle - dndx["subDLA"]
            assert err == pytest.approx(0.5 * straddle, rel=1e-12)
