"""The bundle loaders never import the trainer (``hcd_analysis.emulator.train``: pickle checkpoints and optax): a fresh
process that imports the loaders, writes synthetic bundles and loads them through every loader ends without the
trainer in ``sys.modules`` (D5 plan v3.1 E2, the isolation test). Synthetic bundles only."""
import os
import subprocess
import sys
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CODE = textwrap.dedent("""
    import sys, warnings
    sys.path.insert(0, sys.argv[1])
    import hcd_analysis.predictor as hp
    from hcd_analysis.emulator import release_testing as RT
    from hcd_analysis.emulator import safe_h5 as H
    info = RT.write_synthetic_bundle(sys.argv[2] + "/release")
    sha = info.manifest_sha256
    H.load_release(info.bundle_dir, expected_manifest_sha256=sha)
    H.load_ensemble(info.bundle_dir, expected_manifest_sha256=sha)
    H.load_bundle(info.bundle_dir, expected_manifest_sha256=sha)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        H.load_ensemble(info.bundle_dir, expected_manifest_sha256=None)
    hp.load_bundle(info.bundle_dir, sha).predict({n: 0.5 * (lo + hi) for n, (lo, hi) in
                                                  zip(hp.PARAM_NAMES, hp.PARAM_LIMITS)}, 3.0, tau_eff=0.4)
    if hasattr(RT, "write_synthetic_research_bundle"):
        res = RT.write_synthetic_research_bundle(sys.argv[2] + "/research")
        H.load_research(res.bundle_dir, expected_manifest_sha256=res.manifest_sha256, purpose="isolation test")
        hp.load_research_bundle(res.bundle_dir, res.manifest_sha256, purpose="isolation test")
    print(sorted(m for m in ("hcd_analysis.emulator.train", "optax") if m in sys.modules))
""")


def test_no_loader_imports_the_trainer(tmp_path):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", JAX_PLATFORMS="cpu")
    r = subprocess.run([sys.executable, "-B", "-c", CODE, REPO, str(tmp_path)], capture_output=True, text=True,
                       env=env, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-3000:]
    assert r.stdout.strip().splitlines()[-1] == "[]", r.stdout
