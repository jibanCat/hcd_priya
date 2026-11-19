"""Runtime data products (hcd_analysis.runtime_data) and the loaders that use them: where a product is found (an explicit
argument, else its environment variable, never a built-in path), that its bytes are verified before use, that a
failure names what to do, and what a passing load reports. Synthetic files only: the tests swap the registry for one
whose digests are those of small files they write."""

import hashlib
import logging
import os

import pytest

from hcd_analysis import runtime_data as RD
from hcd_analysis import config as CFG
from hcd_analysis import io as HIO
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator import multifidelity as MF

VAR = "HCD_TEST_DATA_DIR"


def _sha(b):
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def registry(monkeypatch, tmp_path):
    """Two synthetic products under one variable, and a third whose bytes are another product's."""
    a, b = b"alpha product bytes\n", b"beta product bytes\n"
    root = tmp_path / "data"
    (root / "pa").mkdir(parents=True)
    (root / "pb").mkdir()
    (root / "pa" / "a.txt").write_bytes(a)
    (root / "pb" / "b1.txt").write_bytes(b)
    (root / "pb" / "b2.txt").write_bytes(a + b)
    products = {
        "prod_a": RD.DataProduct("prod_a", "alpha v1", VAR, "pa", {"a.txt": _sha(a)}),
        "prod_b": RD.DataProduct(
            "prod_b", "beta v2", VAR, "pb", {"b1.txt": _sha(b), "b2.txt": _sha(a + b)}
        ),
    }
    monkeypatch.setattr(RD, "PRODUCTS", products)
    monkeypatch.delenv(VAR, raising=False)
    return root


def test_the_environment_variable_locates_a_product_and_its_bytes_are_verified(
    registry, monkeypatch, caplog
):
    monkeypatch.setenv(VAR, str(registry))
    with caplog.at_level(logging.INFO, logger="hcd_analysis.runtime_data"):
        path = RD.product_file("prod_b", "b2.txt")
    assert path == str(registry / "pb" / "b2.txt")
    msg = caplog.text
    assert "prod_b" in msg and "beta v2" in msg and VAR in msg and "verified" in msg


def test_an_explicit_argument_takes_precedence_over_the_environment(
    registry, monkeypatch, tmp_path, caplog
):
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "a.txt").write_bytes((registry / "pa" / "a.txt").read_bytes())
    monkeypatch.setenv(VAR, str(registry))
    with caplog.at_level(logging.INFO, logger="hcd_analysis.runtime_data"):
        path = RD.product_file("prod_a", "a.txt", explicit=str(other / "a.txt"))
    assert path == str(other / "a.txt") and "argument" in caplog.text
    assert RD.product_dir("prod_a", explicit=str(other)) == str(
        other
    )  # a directory argument works the same


def test_with_neither_argument_nor_variable_the_error_names_the_variable_to_set(registry):
    with pytest.raises(RD.DataProductError) as e:
        RD.product_file("prod_a", "a.txt")
    text = str(e.value)
    assert VAR in text and "pa/a.txt" in text and "prod_a" in text


def test_an_absent_file_is_refused_with_its_product_and_variable(registry, monkeypatch, tmp_path):
    monkeypatch.setenv(VAR, str(tmp_path / "empty"))
    with pytest.raises(RD.DataProductError, match=f"prod_a.*{VAR}|{VAR}.*prod_a"):
        RD.product_file("prod_a", "a.txt")


def test_unknown_bytes_are_refused_as_an_unknown_version(registry, monkeypatch):
    (registry / "pa" / "a.txt").write_bytes(b"edited bytes\n")
    monkeypatch.setenv(VAR, str(registry))
    with pytest.raises(RD.DataProductError, match="unknown version") as e:
        RD.product_file("prod_a", "a.txt")
    assert _sha(b"edited bytes\n")[:12] in str(e.value)


def test_the_bytes_of_another_known_product_are_refused_by_name(registry, monkeypatch):
    (registry / "pa" / "a.txt").write_bytes((registry / "pb" / "b1.txt").read_bytes())
    monkeypatch.setenv(VAR, str(registry))
    with pytest.raises(RD.DataProductError, match=r"prod_b.*b1\.txt"):
        RD.product_file("prod_a", "a.txt")


def test_a_product_directory_is_returned_only_when_every_file_it_needs_is_right(
    registry, monkeypatch
):
    monkeypatch.setenv(VAR, str(registry))
    assert RD.product_dir("prod_b") == str(registry / "pb")
    (registry / "pb" / "b2.txt").write_bytes(b"edited\n")
    with pytest.raises(RD.DataProductError, match="unknown version"):
        RD.product_dir("prod_b")
    assert RD.product_dir("prod_b", names=["b1.txt"]) == str(
        registry / "pb"
    )  # only the files asked for are read
    (registry / "pb" / "b1.txt").unlink()
    with pytest.raises(RD.DataProductError, match="not found"):
        RD.product_dir("prod_b", names=["b1.txt"])


def test_an_unlisted_product_or_file_is_a_programming_error(registry, monkeypatch):
    monkeypatch.setenv(VAR, str(registry))
    with pytest.raises(KeyError):
        RD.product_file("no_such_product", "a.txt")
    with pytest.raises(KeyError):
        RD.product_file("prod_a", "no_such_file.txt")


def test_the_registry_is_complete_and_well_formed():
    expected = {
        "desi_dr1_p1d": "HCD_P1D_DATA_DIR",
        "desi_dr1_p1d_snr3": "HCD_P1D_DATA_DIR",
        "eboss_dr14_p1d": "HCD_P1D_DATA_DIR",
        "ks_p1d_conservative": "HCD_LYAEMU_PARENT",
        "mf_res_corr": "HCD_KS_BASEDIR",
    }
    assert {k: p.env for k, p in RD.PRODUCTS.items()} == expected
    for key, p in RD.PRODUCTS.items():
        assert p.product_id == key and p.version and p.files and not os.path.isabs(p.subdir)
        assert all(len(d) == 64 and int(d, 16) >= 0 for d in p.files.values())
    digests = [d for p in RD.PRODUCTS.values() for d in p.files.values()]
    assert len(digests) == len(set(digests))  # every file has its own bytes


@pytest.fixture
def no_data_env(monkeypatch):
    for var in (
        "HCD_P1D_DATA_DIR",
        "HCD_LYAEMU_PARENT",
        "HCD_KS_BASEDIR",
        "HCD_DESI_SNR3",
        "HCD_CV_FLOOR",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize(
    "load, var",
    [
        (lambda: DL.load_desi_leg(), "HCD_P1D_DATA_DIR"),
        (lambda: DL.load_desi_leg(use_snr3=True), "HCD_P1D_DATA_DIR"),
        (lambda: DL.load_eboss_leg(), "HCD_P1D_DATA_DIR"),
        (lambda: DL.load_ks_leg(k_max=0.065), "HCD_LYAEMU_PARENT"),
        (lambda: MF.load_res_corr(), "HCD_KS_BASEDIR"),
    ],
)
def test_every_loader_has_no_built_in_path_and_names_its_variable(no_data_env, load, var):
    with pytest.raises(RD.DataProductError, match=var):
        load()


@pytest.mark.parametrize(
    "load, product",
    [
        (lambda: DL.load_desi_leg(), "desi_dr1_p1d"),
        (lambda: DL.load_desi_leg(use_snr3=True), "desi_dr1_p1d_snr3"),
        (lambda: DL.load_eboss_leg(), "eboss_dr14_p1d"),
        (lambda: DL.load_ks_leg(k_max=0.065), "ks_p1d_conservative"),
        (lambda: MF.load_res_corr(), "mf_res_corr"),
    ],
)
def test_every_loader_reads_its_data_through_the_verified_resolver(monkeypatch, load, product):
    seen = []

    def stop(pid, *args, **kwargs):
        seen.append(pid)
        raise RD.DataProductError("stop")

    monkeypatch.setattr(RD, "resolve_file", stop)
    monkeypatch.setattr(RD, "resolve_dir", stop)
    with pytest.raises(RD.DataProductError):
        load()
    assert seen and seen[0] == product


@pytest.mark.parametrize("resolution_float, n_files", [(False, 2), (True, 3)])
def test_the_ks_loader_verifies_every_file_it_reads(monkeypatch, resolution_float, n_files):
    """The resolution-float KS leg also reads the detailed results file, so that file is verified as well."""
    seen = {}

    def stop(pid, explicit=None, names=None):
        seen["names"] = list(names)
        raise RD.DataProductError("stop")

    monkeypatch.setattr(RD, "resolve_dir", stop)
    with pytest.raises(RD.DataProductError):
        DL.load_ks_leg(k_max=0.065, resolution_float=resolution_float)
    assert len(seen["names"]) == n_files and set(seen["names"]) <= set(
        RD.PRODUCTS["ks_p1d_conservative"].files
    )


def test_an_explicit_loader_path_is_verified_too(monkeypatch, tmp_path, no_data_env):
    fake = tmp_path / "desi_dr1_p1d.npz"
    fake.write_bytes(b"not the release's file")
    with pytest.raises(RD.DataProductError, match="unknown version"):
        DL.load_desi_leg(str(fake))


def test_the_module_paths_name_no_developer_directory():
    """The informational module attributes kept for older callers resolve from the variables, never a built-in:
    imported in a fresh interpreter with the variables unset, each names its unset variable."""
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k not in ("HCD_P1D_DATA_DIR", "HCD_KS_BASEDIR")}
    code = (
        "from hcd_analysis.emulator import data_likelihood as DL, multifidelity as MF; "
        "print(DL.DESI_SNR3_NPZ); print(MF.RES_CORR_DIR)"
    )
    out = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env={**env, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")
    assert out[0] == os.path.join(
        "<HCD_P1D_DATA_DIR unset>", "desi_dr1_p1d", "desi_dr1_p1d_snr3.npz"
    )
    assert out[1] == os.path.join("<HCD_KS_BASEDIR unset>", "res_corr")


def test_the_simulation_roots_come_from_the_config_or_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("HCD_LF_RAW_DIR", raising=False)
    monkeypatch.delenv("HCD_HR_RAW_DIR", raising=False)
    assert CFG.PipelineConfig().data_root == "" and CFG.PipelineConfig().hires_data_root == ""
    monkeypatch.setenv("HCD_LF_RAW_DIR", str(tmp_path / "lf"))
    monkeypatch.setenv("HCD_HR_RAW_DIR", str(tmp_path / "hr"))
    assert CFG.PipelineConfig().data_root == str(tmp_path / "lf")
    assert CFG.PipelineConfig().hires_data_root == str(tmp_path / "hr")
    yml = tmp_path / "c.yaml"
    yml.write_text(f"data_root: {tmp_path / 'from_config'}\n")
    assert CFG.load_config(str(yml)).data_root == str(
        tmp_path / "from_config"
    )  # the config file wins


@pytest.mark.parametrize("root", ["", "missing"])
def test_simulation_discovery_refuses_an_unset_or_missing_root(tmp_path, root):
    arg = str(tmp_path / root) if root else root
    with pytest.raises(ValueError, match="HCD_LF_RAW_DIR"):
        HIO.discover_simulations(arg)


# --- provenance records name repository files by their repository-relative path (review m2) -------------------------
from hcd_analysis import paths as P  # noqa: E402


def test_repo_relative_names_a_repository_file_without_this_machines_layout():
    p = os.path.join(P.REPO_ROOT_STR, "hcd_analysis", "_emulator_data", "observables_tau0_lf.h5")
    assert P.repo_relative(p) == "hcd_analysis/_emulator_data/observables_tau0_lf.h5"
    assert P.repo_relative(os.path.join(P.REPO_ROOT_STR, "a", "..", "b.txt")) == "b.txt"


@pytest.mark.parametrize("outside", ["/" + "elsewhere/x.h5", "../outside/x.h5"])
def test_repo_relative_refuses_a_path_outside_the_repository(outside):
    path = outside if outside.startswith("/") else os.path.join(P.REPO_ROOT_STR, outside)
    with pytest.raises(ValueError, match="not inside the repository"):
        P.repo_relative(path)


def test_the_shipped_derivation_record_names_its_cache_as_the_writer_would():
    import json
    from hcd_analysis.emulator import lit_dndx_kernel as LK

    rec = json.load(
        open(
            os.path.join(os.path.dirname(LK.__file__), "hcd_lit_dndx_corrected.json"),
            encoding="utf-8",
        )
    )
    assert rec["cache_provenance"]["path"] == P.repo_relative(LK.CACHE_PATH)


# --- what a load establishes, carried by its result and shown at local load (C4a review MAJOR-1, MINOR-1, MINOR-2) ---
import dataclasses  # noqa: E402
import io as _io  # noqa: E402
import pickle  # noqa: E402

import numpy as np  # noqa: E402

LOGGER = "hcd_analysis.runtime_data"


@pytest.fixture
def shown():
    """Restore the runtime_data logger after a test that attaches the display."""
    lg = logging.getLogger(LOGGER)
    handlers, level = list(lg.handlers), lg.level
    yield lg
    lg.handlers[:] = handlers
    lg.setLevel(level)


def test_a_verified_load_returns_a_record_of_what_it_established(registry, monkeypatch):
    monkeypatch.setenv(VAR, str(registry))
    rec = RD.resolve_file("prod_b", "b2.txt")
    assert isinstance(rec, RD.LoadRecord)
    assert (rec.product_id, rec.version, rec.file) == ("prod_b", "beta v2", "b2.txt")
    assert rec.sha256 == RD.PRODUCTS["prod_b"].files["b2.txt"]
    assert rec.source == f"environment variable {VAR}"
    assert rec.path == str(registry / "pb" / "b2.txt") and rec.real_path == os.path.realpath(
        rec.path
    )
    directory, recs = RD.resolve_dir("prod_b")
    assert directory == str(registry / "pb") and [r.file for r in recs] == ["b1.txt", "b2.txt"]
    rec = RD.resolve_file(
        "prod_a", "a.txt", explicit=registry / "pa" / "a.txt"
    )  # a path object is accepted
    assert rec.source == "argument" and rec.path == str(registry / "pa" / "a.txt")


def test_the_record_and_the_line_give_the_path_as_given_and_its_real_path(
    registry, monkeypatch, tmp_path, caplog
):
    link = tmp_path / "link"
    link.symlink_to(registry, target_is_directory=True)
    monkeypatch.setenv(VAR, str(link))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        rec = RD.resolve_file("prod_a", "a.txt")
    assert rec.path == str(link / "pa" / "a.txt")
    assert rec.real_path == os.path.realpath(registry / "pa" / "a.txt") != rec.path
    assert rec.path in caplog.text and rec.real_path in caplog.text
    assert rec.sha256[:12] in caplog.text and "alpha v1" in caplog.text and "prod_a" in caplog.text
    monkeypatch.chdir(registry)
    monkeypatch.setenv(VAR, ".")
    rec = RD.resolve_file("prod_a", "a.txt")
    assert rec.path == os.path.join(".", "pa", "a.txt") and os.path.isabs(rec.real_path)


@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_or_blank_variable_is_refused_with_what_to_do(registry, monkeypatch, value):
    monkeypatch.chdir(registry)  # an empty variable taken as a location would find pa/a.txt here
    monkeypatch.setenv(VAR, value)
    with pytest.raises(RD.DataProductError, match=f"{VAR} is set but empty"):
        RD.product_file("prod_a", "a.txt")
    with pytest.raises(RD.DataProductError, match=f"{VAR} is set but empty"):
        RD.product_dir("prod_a")


@pytest.mark.parametrize("value", ["", "   "])
def test_an_explicit_empty_path_is_refused_not_ignored(registry, monkeypatch, value):
    monkeypatch.setenv(VAR, str(registry))  # falling back to the variable would succeed
    with pytest.raises(RD.DataProductError, match="the path passed to the loader is empty"):
        RD.product_file("prod_a", "a.txt", explicit=value)
    with pytest.raises(RD.DataProductError, match="the path passed to the loader is empty"):
        RD.product_dir("prod_a", explicit=value)


def test_show_loads_prints_every_verified_load_without_configuring_the_root_logger(
    registry, monkeypatch, shown
):
    root = logging.getLogger()
    before = (list(root.handlers), root.level)
    buf = _io.StringIO()
    RD.show_loads(stream=buf)
    RD.show_loads(stream=buf)  # idempotent: one display, one line per load
    assert sum(h.name == RD.SHOW_HANDLER_NAME for h in shown.handlers) == 1
    monkeypatch.setenv(VAR, str(registry))
    rec = RD.resolve_file("prod_a", "a.txt")
    lines = buf.getvalue().splitlines()
    assert len(lines) == 1
    for part in ("prod_a", "a.txt", "alpha v1", rec.sha256[:12], VAR, rec.path, rec.real_path):
        assert part in lines[0]
    assert (list(root.handlers), root.level) == before


def test_the_library_configures_no_logging_on_import():
    """Importing the loaders attaches no handler to the root logger or to runtime_data's own logger."""
    import subprocess
    import sys

    code = (
        "import logging; from hcd_analysis import runtime_data; "
        "from hcd_analysis.emulator import data_likelihood, multifidelity; "
        "print(len(logging.getLogger().handlers), len(logging.getLogger('hcd_analysis.runtime_data').handlers))"
    )
    out = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out == ["0", "0"]


# Synthetic P1D files in the loaders' formats, with the registry swapped for their digests.
def _npz(path, scale):
    z = np.repeat([2.2, 2.4], 4)
    k = np.tile([0.002, 0.004, 0.008, 0.016], 2)
    n = z.size
    np.savez(
        path,
        z=z,
        k=k,
        plya=scale * np.ones(n),
        cov=np.eye(n),
        cov_stat=0.5 * np.eye(n),
        cov_diag_inflation=np.zeros(n),
        syst_e_dla_completeness=np.full(n, 0.01),
    )
    return _sha(path.read_bytes())


@pytest.fixture
def p1d(monkeypatch, tmp_path):
    """$HCD_P1D_DATA_DIR-style tree (root) and a second copy elsewhere (other), with the three P1D products'
    registry entries replaced by the synthetic files' digests."""
    root, other = tmp_path / "p1d", tmp_path / "elsewhere"
    (root / "desi_dr1_p1d").mkdir(parents=True)
    (root / "eboss_dr14_p1d").mkdir()
    other.mkdir()
    digests = {}
    for pid, sub, name, scale in (
        ("desi_dr1_p1d", "desi_dr1_p1d", "desi_dr1_p1d.npz", 1.0),
        ("desi_dr1_p1d_snr3", "desi_dr1_p1d", "desi_dr1_p1d_snr3.npz", 2.0),
        ("eboss_dr14_p1d", "eboss_dr14_p1d", "eboss_dr14_p1d.npz", 3.0),
    ):
        digests[pid] = _npz(root / sub / name, scale)
        (other / name).write_bytes((root / sub / name).read_bytes())
    products = dict(RD.PRODUCTS)
    for pid, d in digests.items():
        p = products[pid]
        products[pid] = dataclasses.replace(p, files={next(iter(p.files)): d})
    monkeypatch.setattr(RD, "PRODUCTS", products)
    for var in ("HCD_P1D_DATA_DIR", "HCD_DESI_SNR3", "HCD_CV_FLOOR", "HCD_CV_FLOOR_RANK1"):
        monkeypatch.delenv(var, raising=False)
    return dataclasses.make_dataclass("P1D", ["root", "other", "digests"])(root, other, digests)


def test_a_loaded_leg_carries_the_record_of_its_verified_data(p1d, monkeypatch):
    monkeypatch.setenv("HCD_P1D_DATA_DIR", str(p1d.root))
    leg = DL.load_eboss_leg()
    (rec,) = leg.data_products
    assert (rec.product_id, rec.file, rec.sha256) == (
        "eboss_dr14_p1d",
        "eboss_dr14_p1d.npz",
        p1d.digests["eboss_dr14_p1d"],
    )
    assert rec.version == RD.PRODUCTS["eboss_dr14_p1d"].version
    assert rec.source == "environment variable HCD_P1D_DATA_DIR"
    assert rec.path == str(p1d.root / "eboss_dr14_p1d" / "eboss_dr14_p1d.npz")
    assert DL.load_desi_leg().data_products[0].product_id == "desi_dr1_p1d"
    # an attribute, not a field: the fields, and every digest taken over them, are those of a leg without it
    assert "data_products" not in DL.DataLeg._fields and len(DL.DataLeg._fields) == 21
    assert leg._replace(P_data=2 * leg.P_data).data_products == leg.data_products
    assert pickle.loads(pickle.dumps(leg)).data_products == leg.data_products
    assert DL.DataLeg(*leg).data_products == ()  # a leg built directly records no load


@pytest.mark.parametrize("via_env", [False, True])
def test_the_desi_snr3_switch_reads_an_explicit_file_as_the_snr3_product(p1d, monkeypatch, via_env):
    """An explicit path wins over the variable under the SNR>3 switch, from the argument or HCD_DESI_SNR3."""
    monkeypatch.setenv("HCD_P1D_DATA_DIR", str(p1d.root))
    explicit = p1d.other / "desi_dr1_p1d_snr3.npz"
    if via_env:
        monkeypatch.setenv("HCD_DESI_SNR3", "1")
        leg = DL.load_desi_leg(str(explicit))
    else:
        leg = DL.load_desi_leg(str(explicit), use_snr3=True)
    (rec,) = leg.data_products
    assert leg.use_snr3 and rec.product_id == "desi_dr1_p1d_snr3"
    assert rec.source == "argument" and rec.path == str(explicit)
    monkeypatch.delenv("HCD_P1D_DATA_DIR")
    assert DL.load_desi_leg(str(explicit), use_snr3=True).data_products[0].path == str(
        explicit
    )  # no variable


@pytest.mark.parametrize(
    "given, use_snr3", [("desi_dr1_p1d.npz", True), ("desi_dr1_p1d_snr3.npz", False)]
)
@pytest.mark.parametrize("via_env", [False, True])
def test_an_explicit_desi_file_of_the_other_sample_is_refused_never_swapped(
    p1d, monkeypatch, given, use_snr3, via_env
):
    monkeypatch.setenv("HCD_P1D_DATA_DIR", str(p1d.root))  # the selected sample's file is there too
    if via_env:
        monkeypatch.setenv("HCD_DESI_SNR3", "1" if use_snr3 else "0")
        call = lambda: DL.load_desi_leg(str(p1d.other / given))  # noqa: E731
    else:
        call = lambda: DL.load_desi_leg(str(p1d.other / given), use_snr3=use_snr3)  # noqa: E731
    with pytest.raises(RD.DataProductConflict, match="never swaps") as e:
        call()
    text = str(e.value)
    assert "desi_dr1_p1d_snr3" in text and str(p1d.other / given) in text
    assert ("(from HCD_DESI_SNR3)" in text) == via_env and (
        "(from the use_snr3 argument)" in text
    ) != via_env


def test_the_conflict_names_the_default_when_neither_argument_nor_variable_set_the_switch(
    p1d, monkeypatch
):
    monkeypatch.setenv("HCD_P1D_DATA_DIR", str(p1d.root))
    with pytest.raises(
        RD.DataProductConflict, match=r"use_snr3=False \(from the default: HCD_DESI_SNR3 is unset\)"
    ):
        DL.load_desi_leg(str(p1d.other / "desi_dr1_p1d_snr3.npz"))


def _ks_files(d):
    rows = [(z, k) for z in (2.4, 2.6) for k in (0.006, 0.01, 0.02)]
    (d / "final-conservative-p1d-karacayli_etal2021.txt").write_text(
        "| z | k | P | e |\n" + "".join(f"| {z} | {k} | 0.5 | 0.05 |\n" for z, k in rows)
    )
    np.savetxt(d / "final-conservative-covariance-karacayli_etal2021.txt", np.eye(len(rows)))


def test_the_ks_leg_and_the_res_corr_table_carry_their_records(monkeypatch, tmp_path):
    ks, rc = tmp_path / "ks", tmp_path / "rc"
    ks.mkdir()
    rc.mkdir()
    _ks_files(ks)
    np.savetxt(rc / "resolution_correction.txt", np.ones((2, 3)))
    np.savetxt(rc / "kfkms.txt", np.array([[0.01, 0.02, 0.03], [0.011, 0.021, 0.031]]))
    np.savetxt(rc / "zout.txt", np.array([3.0, 2.0]))
    products = dict(RD.PRODUCTS)
    for pid, d in (("ks_p1d_conservative", ks), ("mf_res_corr", rc)):
        p = products[pid]
        products[pid] = dataclasses.replace(
            p,
            files={
                n: (_sha((d / n).read_bytes()) if (d / n).exists() else h)
                for n, h in p.files.items()
            },
        )
    monkeypatch.setattr(RD, "PRODUCTS", products)
    leg = DL.load_ks_leg(str(ks), k_max=0.065)
    assert [r.file for r in leg.data_products] == [
        "final-conservative-p1d-karacayli_etal2021.txt",
        "final-conservative-covariance-karacayli_etal2021.txt",
    ]
    assert {r.product_id for r in leg.data_products} == {"ks_p1d_conservative"}
    table = MF.load_res_corr(str(rc))
    z_rc, logk_rc, rc_tab = table  # still the three arrays
    assert [r.file for r in table.data_products] == list(RD.PRODUCTS["mf_res_corr"].files)
    assert pickle.loads(pickle.dumps(table)).data_products == table.data_products


@pytest.mark.parametrize(
    "load",
    [
        lambda: DL.load_desi_leg(""),
        lambda: DL.load_desi_leg("", use_snr3=True),
        lambda: DL.load_eboss_leg(""),
        lambda: DL.load_ks_leg("", k_max=0.065),
        lambda: MF.load_res_corr(""),
    ],
)
def test_every_loader_refuses_an_explicit_empty_path(p1d, monkeypatch, tmp_path, load):
    for var in ("HCD_P1D_DATA_DIR", "HCD_LYAEMU_PARENT", "HCD_KS_BASEDIR"):
        monkeypatch.setenv(var, str(p1d.root))
    with pytest.raises(RD.DataProductError, match="the path passed to the loader is empty"):
        load()
