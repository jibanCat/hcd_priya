"""Error-vector (C_emu sigma / rho) product I/O under checkpoint schema 2.0: labelled by comoving modes, never by
a velocity grid. See hcd_priya_notes docs/superpowers/emulator-paper-history/2026-10-05-INCIDENT-kgrid-representation-regression.md."""
import numpy as np

from .schema import CHECKPOINT_SCHEMA_VERSION, SchemaCollapseError


_VELOCITY_KEYS = ("kfkms", "k_skm", "k_kms", "kgrid")


def save_error_vector(path, *, sigma, k_com_hmpc, z_band_edges, tau0_band_centres, class_names, **extra):
    bad = [k for k in extra if k in _VELOCITY_KEYS]
    if bad:
        raise SchemaCollapseError(f"error vectors are labelled by k_com_hmpc, not by a velocity grid ({bad})")
    K = int(np.asarray(k_com_hmpc).shape[0])
    if np.asarray(sigma).shape[1] != K:
        raise SchemaCollapseError(f"sigma has {np.asarray(sigma).shape[1]} modes on axis 1, k_com_hmpc has {K}")
    if "rho" in extra and np.asarray(extra["rho"]).shape[2] != K:
        raise SchemaCollapseError(f"rho has {np.asarray(extra['rho']).shape[2]} modes on axis 2, k_com_hmpc has {K}")
    np.savez(path, sigma=sigma, k_com_hmpc=np.asarray(k_com_hmpc, float), z_band_edges=z_band_edges,
             tau0_band_centres=tau0_band_centres, class_names=class_names,
             schema_version=np.array(CHECKPOINT_SCHEMA_VERSION), **extra)


def load_error_vector(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if "kfkms" in d or str(d.get("schema_version", "")) != CHECKPOINT_SCHEMA_VERSION:
        raise SchemaCollapseError(
            f"{path}: pre-2026-10 error vector (kfkms label or no schema_version); rebuild it under schema 2.0")
    d["schema_version"] = str(d["schema_version"])
    return d
