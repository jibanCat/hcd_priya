"""A synthetic LF+MF release directory for tests: a synthetic backbone bundle (untrained networks), a synthetic MF
record bound to it, and release.json. No trained model, no real number."""

from pathlib import Path

from hcd_analysis.emulator import release_manifest as RM
from hcd_analysis.emulator import release_testing as RT


def make_release(root, *, version="synthetic-release-1", n_members=2):
    """Write the release under ``root`` (created). Returns {"root": Path, "sha": the sha256 of release.json,
    "bundle": the backbone's SyntheticBundle, "mf_sha": the MF record's sha256}."""
    from tests.test_predictor_mf import write_mf

    root = Path(root)
    root.mkdir(parents=True)
    b = RT.write_synthetic_bundle(str(root / "backbone"), n_members=n_members)
    s = write_mf(root / "mf", manifest=b.manifest_sha256, cache=b.lineage["cache"]["sha256"])
    _, rs = RM.write_release_manifest(
        str(root),
        release_version=version,
        backbone_dir="backbone",
        backbone_manifest_sha256=b.manifest_sha256,
        mf_dir="mf",
        mf_record_sha256=s,
    )
    return {"root": root, "sha": rs, "bundle": b, "mf_sha": s}
