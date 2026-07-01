"""arm -> inject_spec mapping is a pure function so the resolution b_res is configurable (the
option-a resolution certification injects b_res over a +/-1 sigma bracket {0.015, 0.02, 0.03})."""
import hcd_analysis.emulator  # x64 before jax
from scripts.run_dnuis_bias_shard import arm_inject_spec


def test_resolution_b_res_configurable():
    assert arm_inject_spec("resolution", "desi") == {"resolution": {"b_res": 0.02}}          # default
    assert arm_inject_spec("resolution", "desi", b_res=0.03) == {"resolution": {"b_res": 0.03}}
    assert arm_inject_spec("resolution", "desi", b_res=0.015) == {"resolution": {"b_res": 0.015}}


def test_other_arms_unaffected():
    assert arm_inject_spec("metal_misspec", "desi") == {"metal_misspec": {"form": "desi_full"}}
    assert arm_inject_spec("metal_misspec", "eboss") == {"metal_misspec": {"form": "eboss"}}
    assert arm_inject_spec("metal_matched", "desi") == {}
