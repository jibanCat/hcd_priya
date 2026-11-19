"""A synthetic world for the cache-lineage tests: raw spectra files, catalogue-only snapshots rebuilt by the pipeline,
the matching pre-fix ("old") snapshots with their box-edge duplicates, and a historical 3.3 cache built from the old
snapshots by the cache builder. Made-up optical depths only.

The cache builder calls fake_spectra (the mean-flux solve, the power spectrum, the PRIYA filter). Where fake_spectra is
not installed (the test environment), ``use_fake_spectra_stand_in`` installs a small numpy stand-in for the duration
of a test, so the builder's own code runs end to end; its values are not PRIYA's. Also a PyYAML stand-in for the
pipeline's config module, as tests/test_catalog_systems.py does."""

import contextlib
import json
import os
import shutil
import sys
import types
from pathlib import Path

import h5py
import numpy as np

from hcd_analysis import lineage as L

SIMS = (
    "ns0.91Ap2e-09herei4heref3alphaq2hub0.7omegamh20.1421hireionz7bhfeedback0.05",
    "ns0.97Ap1.8e-09herei3.8heref2.9alphaq1.9hub0.7omegamh20.1421hireionz7.2bhfeedback0.04",
)
GROUPS = ((SIMS[0], 3, 3.0), (SIMS[0], 5, 2.6), (SIMS[1], 3, 3.0))
NBINS = {
    (SIMS[0], 3): 400,
    (SIMS[0], 5): 420,
    (SIMS[1], 3): 400,
}  # pixels per skewer (varies with z, as in PRIYA)
NSK = 8
CLEAN = {"commit": "c" * 40, "dirty": False}


def _stand_in_modules():
    fs = types.ModuleType("fake_spectra")
    flux = types.ModuleType("fake_spectra.fluxstatistics")
    spec = types.ModuleType("fake_spectra.spectra")

    def obs_mean_tau(z):
        return 0.0023 * (1.0 + z) ** 3.65

    def mean_flux(tau, mean_flux_desired):
        lo, hi = 0.0, 1.0
        while np.mean(np.exp(-hi * tau)) > mean_flux_desired:
            hi *= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if np.mean(np.exp(-mid * tau)) > mean_flux_desired else (lo, mid)
        return 0.5 * (lo + hi)

    def _powerspectrum(inarray, axis=-1):
        return (np.abs(np.fft.rfft(inarray, axis=axis)) / inarray.shape[axis]) ** 2

    def _flux_power_bins(vmax, npix):
        return 2.0 * np.pi * np.fft.rfftfreq(npix, d=vmax / npix)

    def flux_power(*a, **k):
        raise NotImplementedError("fake_spectra stand-in")

    class Spectra:
        def _filter_single_tau_complex(self, tau, tau_eff, tau_thresh=1.0e6, thresh2=0.25):
            return np.where(tau > tau_thresh, tau_eff, tau), None

    flux.obs_mean_tau, flux.mean_flux, flux._powerspectrum = obs_mean_tau, mean_flux, _powerspectrum
    flux._flux_power_bins, flux.flux_power = _flux_power_bins, flux_power
    spec.Spectra = Spectra
    fs.fluxstatistics, fs.spectra = flux, spec
    return {"fake_spectra": fs, "fake_spectra.fluxstatistics": flux, "fake_spectra.spectra": spec}


def use_fake_spectra_stand_in(mp):
    """Install the stand-in (unless fake_spectra is installed) and make hcd_analysis.priya_p1d import afresh; both
    are undone when ``mp`` (a pytest MonkeyPatch) is."""
    import hcd_analysis

    try:
        import fake_spectra  # noqa: F401

        return False
    except ImportError:
        pass
    for name, mod in _stand_in_modules().items():
        mp.setitem(sys.modules, name, mod)
    mp.delitem(sys.modules, "hcd_analysis.priya_p1d", raising=False)
    mp.delattr(hcd_analysis, "priya_p1d", raising=False)
    return True


def use_yaml_stand_in(mp):
    try:
        import yaml  # noqa: F401

        return
    except ImportError:
        stub = types.ModuleType("yaml")

        def _absent(name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            raise ImportError(
                f"PyYAML is not installed (yaml.{name} requested; stand-in from tests/tau0_world.py)"
            )

        stub.__getattr__ = _absent
        mp.setitem(sys.modules, "yaml", stub)


def write_raw(emu, sim, snap, z, seed):
    NB = NBINS[(sim, snap)]
    d = Path(emu) / sim / "output" / f"SPECTRA_{snap:03d}"
    d.mkdir(parents=True, exist_ok=True)
    ics = Path(emu) / sim / "SimulationICs.json"
    if not ics.exists():
        ns = float(sim[2:6])
        ics.write_text(
            json.dumps(
                {
                    "ns": ns,
                    "scalar_amp": 2.0e-9,
                    "here_i": 4.0,
                    "here_f": 3.0,
                    "alpha_q": 2.0,
                    "hubble": 0.7,
                    "omega0": 0.29,
                    "hireionz": 7.0,
                    "bhfeedback": 0.05,
                }
            )
        )
    rng = np.random.default_rng(seed)
    tau = rng.uniform(0.0, 0.3, size=(NSK, NB))
    tau[0, NB - 5 :] = 1e5
    tau[0, :4] = 1e5  # a system straddling the box edge (the old finder: plus its head)
    tau[1, 100:106] = 1e5
    tau[2, 200:230] = 3e6  # above the PRIYA filter threshold
    tau[5, 50:54] = 2e4
    tau[6, NB - 3 :] = 4e5
    tau[6, :2] = 4e5  # a second straddling system
    if seed % 2:
        tau[4, 150:160] = 2e5  # class counts differ between groups
    p = d / "lya_forest_spectra_grid_480.hdf5"
    with h5py.File(p, "w") as f:
        g = f.create_group("Header")
        for k, v in dict(
            redshift=z, hubble=0.7, omegam=0.29, omegal=0.71, box=120000.0, nbins=NB, omegab=0.05
        ).items():
            g.attrs[k] = v
        f.create_dataset("tau/H/1/1215", data=tau)
    return p


def rebuild_snapshot(root, emu, sim, snap, z, *, catalogue_only=True, **absorber):
    """A catalogue-only (or full) snapshot of the pipeline, with the release finder settings unless changed."""
    from hcd_analysis import pipeline as PL
    from hcd_analysis.config import PipelineConfig
    from hcd_analysis.io import SimInfo
    from hcd_analysis.snapshot_map import SimSnapshot, SnapEntry

    raw = Path(emu) / sim / "output" / f"SPECTRA_{snap:03d}" / "lya_forest_spectra_grid_480.hdf5"
    c = PipelineConfig(
        output_root=str(root), resume=True, n_workers=1, catalogue_only=catalogue_only
    )
    c.p1d.nhi_excl_thresholds = []
    c.absorber.tau_threshold = 100.0
    for k, v in absorber.items():
        setattr(c.absorber, k, v)
    entry = SnapEntry(snap=snap, a=1.0 / (1.0 + z), z=z, path=raw)
    res = PL.run_one_snap(
        SimSnapshot(sim=SimInfo(name=sim, path=Path(emu) / sim, params={}), entries=[entry]),
        entry,
        c,
    )
    assert res is not None
    return PL.snap_output_dir(c, sim, snap)


def write_old_snapshot(old_root, new_dir, raw, *, head_factor=0.5):
    """The pre-fix snapshot directory: the rebuilt catalogue plus, for every system straddling the box edge, its own
    head (N_HI = head_factor * the system's), as arrays without a record; meta.json counts and cddf.npz of that
    catalogue; the ``done`` marker."""
    from hcd_analysis import catalog as C
    from hcd_analysis.cddf import measure_cddf
    from hcd_analysis.io import read_header

    new = C.AbsorberCatalog.load_npz(new_dir / "catalog.npz")
    NB = NBINS[(new.sim_name, int(new.snap))]
    old = C.AbsorberCatalog(sim_name=new.sim_name, snap=new.snap, z=new.z, dv_kms=new.dv_kms)
    for a in new.absorbers:
        if a.pix_end >= NB:
            nhi = a.NHI * head_factor
            old.absorbers.append(
                C.Absorber(
                    skewer_idx=a.skewer_idx,
                    pix_start=0,
                    pix_end=a.pix_end - NB,
                    v_center=0.0,
                    NHI=nhi,
                    b_kms=np.nan,
                    log_NHI=float(np.log10(nhi)),
                    absorber_class=C.classify_system(nhi),
                    fit_success=True,
                    fast_mode=True,
                )
            )
        old.absorbers.append(a)
    d = Path(old_root) / new.sim_name / f"snap_{new.snap:03d}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(str(d / "catalog.npz"), **old._arrays())
    hdr = read_header(raw)
    cddf = measure_cddf(old, hdr)
    np.savez(str(d / "cddf.npz"), **{k: np.asarray(v) for k, v in cddf.items()})
    meta = json.loads((new_dir / "meta.json").read_text())
    counts = {c: 0 for c in ("LLS", "subDLA", "DLA", "forest")}
    for a in old.absorbers:
        counts[a.absorber_class] += 1
    old_meta = {
        k: meta[k]
        for k in ("sim_name", "snap", "z", "dv_kms", "nbins", "n_skewers", "box_kpc_h", "hubble")
    }
    old_meta["n_absorbers"] = counts
    (d / "meta.json").write_text(json.dumps(old_meta))
    (d / "done").touch()
    return d, old


def trace_rows(old_root):
    """The trace rows of the old snapshots (duplicates per class), as the catalogue wrap trace lists them."""
    from hcd_analysis import catalog as C

    rows = []
    for sim, snap, _z in GROUPS:
        cat = C.load_catalog_historical(
            Path(old_root) / sim / f"snap_{snap:03d}" / "catalog.npz", reason="trace"
        )
        a = cat._arrays()
        dup = C.find_wrap_duplicates(
            a["skewer_idx"], a["pix_start"], a["pix_end"], NBINS[(sim, snap)]
        )
        cls = [C.classify_system(x) for x in np.asarray(a["NHI"])[dup]]
        rows.append(
            {
                "tier": "LF",
                "sim": sim,
                "snap": snap,
                "dups": {c: cls.count(c) for c in ("LLS", "subDLA", "DLA")},
            }
        )
    return rows


BUILDER_COMMIT = "abc1234"  # the synthetic historical cache's builder commit of record


def pin_of_record(mp, cv, hist, commits=(BUILDER_COMMIT,)):
    """Make ``hist`` and ``commits`` the carry tool's (module ``cv``) source and builder commits of record."""
    mp.setattr(cv, "SOURCE_OF_RECORD_SHA256", L.sha256_file(hist))
    mp.setattr(cv, "COMMITS_OF_RECORD", {c: "synthetic" for c in commits})


def to_v33(src, dst, *, repair_added=None):
    """A copy of a cache as a pre-lineage 3.3 file: the same datasets and attributes, no record."""
    from hcd_analysis import tau0_cache as T

    d, attrs, _ = T.read_cache_historical(src, reason="test: write a 3.3 copy")
    with h5py.File(dst, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        f.attrs["cache_version"] = "3.3"
        if repair_added is not None:
            f.attrs["repair_record"] = json.dumps(
                {
                    "added": repair_added,  # as the real record:
                    "builder_commit": f"{BUILDER_COMMIT} (a free-text note)",
                }
            )
        for k, v in d.items():
            if k in T.STR_KEYS:
                f.create_dataset(k, data=np.asarray(v, dtype=object), dtype=h5py.string_dtype())
            else:
                f.create_dataset(k, data=v)
    return dst


def build(argv):
    """Run the cache builder's main."""
    from hcd_analysis.paths import REPO_ROOT

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import build_emulator_cache_tau0 as bt0

    bt0.main([str(a) for a in argv])


@contextlib.contextmanager
def clean_code(mp, code=CLEAN):
    mp.setattr(L, "code_identity", lambda repo_dir=None: dict(code))
    yield


def make_world(base, mp):
    """Build the world under ``base`` with ``mp`` patched (code identity clean, stand-ins). Returns a dict of paths."""
    mp.setattr(L, "code_identity", lambda repo_dir=None: dict(CLEAN))
    use_yaml_stand_in(mp)
    use_fake_spectra_stand_in(mp)
    base = Path(base)
    emu, rebuilt, old = base / "emu", base / "rebuilt", base / "old"
    raws = {}
    for i, (sim, snap, z) in enumerate(GROUPS):
        raws[(sim, snap)] = write_raw(emu, sim, snap, z, seed=i)
    past = 1_700_000_000_000_000_000  # the raw files are older than everything else
    for p in raws.values():
        os.utime(p, ns=(past, past))
    for sim, snap, z in GROUPS:
        nd = rebuild_snapshot(rebuilt, emu, sim, snap, z)
        write_old_snapshot(old, nd, raws[(sim, snap)])
    hist4 = base / "hist_v34.h5"
    build(
        [
            "--hcd-root",
            old,
            "--emu-root",
            emu,
            "--output",
            hist4,
            "--historical-reproduction",
            "a test world",
        ]
    )
    hist = to_v33(hist4, base / "hist_v33.h5", repair_added=[[SIMS[0], 5, 2.6]])
    (base / "trace_rows.json").write_text(json.dumps(trace_rows(old)))
    return {
        "base": base,
        "emu": emu,
        "rebuilt": rebuilt,
        "old": old,
        "hist": hist,
        "raws": raws,
        "trace": base / "trace_rows.json",
    }


def copy_tree(src, dst):
    shutil.copytree(src, dst)
    return Path(dst)
