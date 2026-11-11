"""scripts/run_gate_d.py writes its products once: an existing --out-dir is refused before any input is read, so a rerun
can never overwrite a recorded MF product (for example mf/gateD/mf_modes_all6.npz)."""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_existing_out_dir_is_refused_and_left_unchanged(tmp_path):
    out = tmp_path / "gateD"
    out.mkdir()
    (out / "mf_modes_all6.npz").write_bytes(b"recorded product")
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(REPO, "scripts", "run_gate_d.py"),
            "--ckpt-dir",
            str(tmp_path / "no_ckpts"),
            "--out-dir",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
    )
    assert r.returncode != 0
    assert "exists" in (r.stderr + r.stdout)
    assert sorted(os.listdir(out)) == ["mf_modes_all6.npz"]
    assert (out / "mf_modes_all6.npz").read_bytes() == b"recorded product"
