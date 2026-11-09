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
        "prod_b": RD.DataProduct("prod_b", "beta v2", VAR, "pb", {"b1.txt": _sha(b), "b2.txt": _sha(a + b)}),
    }
    monkeypatch.setattr(RD, "PRODUCTS", products)
    monkeypatch.delenv(VAR, raising=False)
    return root


def test_the_environment_variable_locates_a_product_and_its_bytes_are_verified(registry, monkeypatch, caplog):
    monkeypatch.setenv(VAR, str(registry))
    with caplog.at_level(logging.INFO, logger="hcd_analysis.runtime_data"):
        path = RD.product_file("prod_b", "b2.txt")
    assert path == str(registry / "pb" / "b2.txt")
    msg = caplog.text
    assert "prod_b" in msg and "beta v2" in msg and VAR in msg and "verified" in msg


def test_an_explicit_argument_takes_precedence_over_the_environment(registry, monkeypatch, tmp_path, caplog):
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "a.txt").write_bytes((registry / "pa" / "a.txt").read_bytes())
    monkeypatch.setenv(VAR, str(registry))
    with caplog.at_level(logging.INFO, logger="hcd_analysis.runtime_data"):
        path = RD.product_file("prod_a", "a.txt", explicit=str(other / "a.txt"))
    assert path == str(other / "a.txt") and "argument" in caplog.text
    assert RD.product_dir("prod_a", explicit=str(other)) == str(other)       # a directory argument works the same


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


def test_a_product_directory_is_returned_only_when_every_file_it_needs_is_right(registry, monkeypatch):
    monkeypatch.setenv(VAR, str(registry))
    assert RD.product_dir("prod_b") == str(registry / "pb")
    (registry / "pb" / "b2.txt").write_bytes(b"edited\n")
    with pytest.raises(RD.DataProductError, match="unknown version"):
        RD.product_dir("prod_b")
    assert RD.product_dir("prod_b", names=["b1.txt"]) == str(registry / "pb")   # only the files asked for are read
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
    expected = {"desi_dr1_p1d": "HCD_P1D_DATA_DIR", "desi_dr1_p1d_snr3": "HCD_P1D_DATA_DIR",
                "eboss_dr14_p1d": "HCD_P1D_DATA_DIR", "ks_p1d_conservative": "HCD_LYAEMU_PARENT",
                "mf_res_corr": "HCD_KS_BASEDIR"}
    assert {k: p.env for k, p in RD.PRODUCTS.items()} == expected
    for key, p in RD.PRODUCTS.items():
        assert p.product_id == key and p.version and p.files and not os.path.isabs(p.subdir)
        assert all(len(d) == 64 and int(d, 16) >= 0 for d in p.files.values())
    digests = [d for p in RD.PRODUCTS.values() for d in p.files.values()]
    assert len(digests) == len(set(digests))                                  # every file has its own bytes


@pytest.fixture
def no_data_env(monkeypatch):
    for var in ("HCD_P1D_DATA_DIR", "HCD_LYAEMU_PARENT", "HCD_KS_BASEDIR", "HCD_DESI_SNR3", "HCD_CV_FLOOR"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("load, var", [
    (lambda: DL.load_desi_leg(), "HCD_P1D_DATA_DIR"),
    (lambda: DL.load_desi_leg(use_snr3=True), "HCD_P1D_DATA_DIR"),
    (lambda: DL.load_eboss_leg(), "HCD_P1D_DATA_DIR"),
    (lambda: DL.load_ks_leg(k_max=0.065), "HCD_LYAEMU_PARENT"),
    (lambda: MF.load_res_corr(), "HCD_KS_BASEDIR"),
])
def test_every_loader_has_no_built_in_path_and_names_its_variable(no_data_env, load, var):
    with pytest.raises(RD.DataProductError, match=var):
        load()


@pytest.mark.parametrize("load, product", [
    (lambda: DL.load_desi_leg(), "desi_dr1_p1d"),
    (lambda: DL.load_desi_leg(use_snr3=True), "desi_dr1_p1d_snr3"),
    (lambda: DL.load_eboss_leg(), "eboss_dr14_p1d"),
    (lambda: DL.load_ks_leg(k_max=0.065), "ks_p1d_conservative"),
    (lambda: MF.load_res_corr(), "mf_res_corr"),
])
def test_every_loader_reads_its_data_through_the_verified_resolver(monkeypatch, load, product):
    seen = []

    def stop(pid, *args, **kwargs):
        seen.append(pid)
        raise RD.DataProductError("stop")

    monkeypatch.setattr(RD, "product_file", stop)
    monkeypatch.setattr(RD, "product_dir", stop)
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

    monkeypatch.setattr(RD, "product_dir", stop)
    with pytest.raises(RD.DataProductError):
        DL.load_ks_leg(k_max=0.065, resolution_float=resolution_float)
    assert len(seen["names"]) == n_files and set(seen["names"]) <= set(RD.PRODUCTS["ks_p1d_conservative"].files)


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
    code = ("from hcd_analysis.emulator import data_likelihood as DL, multifidelity as MF; "
            "print(DL.DESI_SNR3_NPZ); print(MF.RES_CORR_DIR)")
    out = subprocess.run([sys.executable, "-B", "-c", code], env={**env, "JAX_PLATFORMS": "cpu"},
                         capture_output=True, text=True, check=True).stdout.split("\n")
    assert out[0] == os.path.join("<HCD_P1D_DATA_DIR unset>", "desi_dr1_p1d", "desi_dr1_p1d_snr3.npz")
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
    assert CFG.load_config(str(yml)).data_root == str(tmp_path / "from_config")      # the config file wins


@pytest.mark.parametrize("root", ["", "missing"])
def test_simulation_discovery_refuses_an_unset_or_missing_root(tmp_path, root):
    arg = str(tmp_path / root) if root else root
    with pytest.raises(ValueError, match="HCD_LF_RAW_DIR"):
        HIO.discover_simulations(arg)
