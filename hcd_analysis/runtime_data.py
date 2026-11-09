"""Runtime data products: where the package finds them, which bytes they must be, and what a load reports.

The package ships no observational data, no correction table and no developer path. Each runtime data product (an
observed P1D measurement with its covariance, a correction table) is found, in this order of precedence:

1. an explicit argument: the path given to the loader (a file, or a directory for a product of several files);
2. the product's environment variable: a directory under which the product's files lie at a fixed relative path.

There is no built-in default. With neither, the loader stops and names the variable to set.

Every file is then verified against the sha256 recorded below before it is read, whichever way it was found. A file
whose bytes are those of another listed product (for example the SNR>3 DESI file where the SNR>1 one belongs) is
refused with that product's name; any other content is refused as an unknown version. A missing file is refused with
its product, variable and relative path. A passing load logs, at INFO on the logger ``hcd_analysis.runtime_data``, the
product id, its version, where it was found and the verified digest. Only that local log line shows the resolved path;
the product id and version are what a run should record.

| Product id | Variable | Files under ``$VARIABLE/<subdir>/`` |
|---|---|---|
| ``desi_dr1_p1d`` | ``HCD_P1D_DATA_DIR`` | ``desi_dr1_p1d/desi_dr1_p1d.npz`` |
| ``desi_dr1_p1d_snr3`` | ``HCD_P1D_DATA_DIR`` | ``desi_dr1_p1d/desi_dr1_p1d_snr3.npz`` |
| ``eboss_dr14_p1d`` | ``HCD_P1D_DATA_DIR`` | ``eboss_dr14_p1d/eboss_dr14_p1d.npz`` |
| ``ks_p1d_conservative`` | ``HCD_LYAEMU_PARENT`` | ``lyaemu/data/kodiaq_squad/`` (three files) |
| ``mf_res_corr`` | ``HCD_KS_BASEDIR`` | ``res_corr/`` (three files) |

The simulation roots of the pipeline (``PipelineConfig.data_root`` and ``hires_data_root``) follow the same
precedence, with the config file (or ``--set``) as the explicit input and ``HCD_LF_RAW_DIR`` / ``HCD_HR_RAW_DIR`` as
the variables (``hcd_analysis.config``). They are directories of simulation outputs, not digest-verified here: each
catalogue records the identity of the snapshot it was built from.
"""
from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
from typing import Mapping, Optional

_log = logging.getLogger(__name__)


class DataProductError(RuntimeError):
    """A runtime data product is not where it should be, or not the bytes it should be."""


@dataclasses.dataclass(frozen=True)
class DataProduct:
    product_id: str
    version: str
    env: str                       # the environment variable naming the directory
    subdir: str                    # the product's directory under it
    files: Mapping[str, str]       # file name -> sha256 of its bytes


PRODUCTS = {
    "desi_dr1_p1d": DataProduct(
        "desi_dr1_p1d", "DESI DR1 P1D, SNR>1 baseline sample, npz from scripts/convert_desi_dr1_p1d.py",
        "HCD_P1D_DATA_DIR", "desi_dr1_p1d",
        {"desi_dr1_p1d.npz": "8d5153d861f21d2ac0a1894f99dd2a9d633a39cc07fa564aac1cba78429c9296"}),
    "desi_dr1_p1d_snr3": DataProduct(
        "desi_dr1_p1d_snr3", "DESI DR1 P1D, SNR>3 sample (the DR1 cosmology baseline), npz",
        "HCD_P1D_DATA_DIR", "desi_dr1_p1d",
        {"desi_dr1_p1d_snr3.npz": "20c526b9d2be91ebb80ad44ed1cede23eb5a81e04903173e54e937039cce3b64"}),
    "eboss_dr14_p1d": DataProduct(
        "eboss_dr14_p1d", "eBOSS DR14 P1D (Chabanier et al. 2019), npz from scripts/convert_eboss_dr14_p1d.py",
        "HCD_P1D_DATA_DIR", "eboss_dr14_p1d",
        {"eboss_dr14_p1d.npz": "843f8f246a9fb3fb3d3a3c180ab58c19b0b2f823aa738222e514eb351e886e96"}),
    "ks_p1d_conservative": DataProduct(
        "ks_p1d_conservative", "KODIAQ-SQUAD P1D, conservative mode (Karacayli et al. 2021), as distributed with lyaemu",
        "HCD_LYAEMU_PARENT", "lyaemu/data/kodiaq_squad",
        {"final-conservative-p1d-karacayli_etal2021.txt":
             "299b11cbb394848a04787498ba64fd1ba958db598c0fbadc082724347c1e9dbf",
         "final-conservative-covariance-karacayli_etal2021.txt":
             "a3f317d812c9afa2557837e25fdf06e2afdc268df438cd4a412203623f603d98",
         "detailed-p1d-results-karacayli_etal2021.txt":
             "f443a491eead6730f65e9e19a96a247d92947aed307323c95cf7304879975517"}),
    "mf_res_corr": DataProduct(
        "mf_res_corr", "PRIYA L15n512/L15n384 resolution-correction table, the KS emulator directory's res_corr",
        "HCD_KS_BASEDIR", "res_corr",
        {"resolution_correction.txt": "7b3dded248ff050adac0313a675324dd736bcc8fb8d8893a9f2ec2dbb0710491",
         "kfkms.txt": "2901188e1eb0de12bce17ebde474ca5d476022391c3d249c10c4d611899b4e90",
         "zout.txt": "67715b9d068a0d26a631373e9248a3c3b90f12585cacacd180f4ea858b5aad87"}),
}


def env_path(product_id: str, name: Optional[str] = None) -> str:
    """The location a product (or one of its files) has under its variable, for display and existence checks only:
    ``<VARIABLE unset>/...`` (a path that never exists) when the variable is unset. Loading goes through
    :func:`product_file` / :func:`product_dir`, which verify."""
    p = PRODUCTS[product_id]
    root = os.environ.get(p.env) or f"<{p.env} unset>"
    return os.path.join(root, p.subdir, *([name] if name else []))


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _where(p: DataProduct, explicit: Optional[str], name: Optional[str] = None):
    """(directory, source) of a product: the explicit directory, else the variable's; refuses neither."""
    if explicit:
        return explicit, "argument"
    root = os.environ.get(p.env)
    if not root:
        what = f"{p.subdir}/{name}" if name else f"{p.subdir}/"
        raise DataProductError(
            f"{p.product_id} ({p.version}): no location given. Set {p.env} to the directory under which "
            f"{what} lies, or pass the path to the loader. There is no built-in default.")
    return os.path.join(root, p.subdir), f"environment variable {p.env}"


def _verify(p: DataProduct, name: str, path: str, source: str) -> str:
    if not os.path.isfile(path):
        raise DataProductError(
            f"{p.product_id} ({p.version}): {p.subdir}/{name} not found at {path} (from {source}). Point {p.env} at "
            f"the directory that holds {p.subdir}/, or pass the file's path to the loader.")
    digest = _sha256_file(path)
    if digest != p.files[name]:
        other = [(q.product_id, f) for q in PRODUCTS.values() for f, d in q.files.items() if d == digest]
        if other:
            raise DataProductError(
                f"{p.product_id}: the file at {path} (from {source}) is {other[0][0]}'s {other[0][1]}, not "
                f"{p.subdir}/{name}. Check {p.env} or the path passed to the loader.")
        raise DataProductError(
            f"{p.product_id}: the file at {path} (from {source}) is an unknown version of {p.subdir}/{name}: "
            f"sha256 {digest[:12]}..., expected {p.files[name][:12]}... ({p.version}). Use the release's file.")
    _log.info("%s %s: %s verified (sha256 %s...), found by %s at %s", p.product_id, name, p.version, digest[:12],
              source, path)
    return path


def product_file(product_id: str, name: str, explicit: Optional[str] = None) -> str:
    """The verified path of file ``name`` of product ``product_id``. ``explicit`` is a path to that file (it takes
    precedence over the product's variable). Raises :class:`DataProductError` when the file cannot be found or is not
    the recorded bytes, and ``KeyError`` for an unlisted product or file."""
    p = PRODUCTS[product_id]
    p.files[name]                                  # KeyError for a file the product does not list
    if explicit:
        return _verify(p, name, explicit, "argument")
    directory, source = _where(p, None, name)
    return _verify(p, name, os.path.join(directory, name), source)


def product_dir(product_id: str, explicit: Optional[str] = None, names=None) -> str:
    """The directory of a product of several files, after verifying ``names`` (default: every listed file).
    ``explicit`` is the directory itself (it takes precedence over the product's variable)."""
    p = PRODUCTS[product_id]
    directory, source = _where(p, explicit)
    for name in (p.files if names is None else names):
        p.files[name]
        _verify(p, name, os.path.join(directory, name), source)
    return directory
