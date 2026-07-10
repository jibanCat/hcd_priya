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


def test_treatment_flags_bracket():
    """The 4-arm resolution comparison bracket maps a treatment label -> the run flags: a=option-a (in cov,
    no float); b=option-b tight (float, prior 0.02); c=option-b wide/cup1d-faithful (float, prior c_sigma);
    d=arm-D (coherent cov, no float). Distinct treatments -> distinct output tags so they never collide."""
    import pytest
    from scripts.run_dnuis_bias_shard import treatment_flags
    assert treatment_flags("a") == dict(float_res=False, coherent_res=False, f_res_amp_sigma=None)
    assert treatment_flags("b") == dict(float_res=True, coherent_res=False, f_res_amp_sigma=None)
    assert treatment_flags("c", c_prior_sigma=0.05) == dict(float_res=True, coherent_res=False,
                                                            f_res_amp_sigma=0.05)
    assert treatment_flags("d") == dict(float_res=False, coherent_res=True, f_res_amp_sigma=None)
    with pytest.raises(SystemExit):
        treatment_flags("x")


def test_resolution_ks_raises():
    """KS's R_z is the DESI pixel proxy ~7-15x too large (KS is echelle, sigma~3.2 km/s), so a
    b_res injection is a ~70% distortion the forward cannot fit -> the old -21sigma ESS collapse
    (4-referee panel, unanimous). arm_inject_spec must RAISE on (resolution, ks) -- NOT silently
    return a valid spec -- mirroring the metal_misspec KS guard, until the echelle R_z lands."""
    import pytest
    with pytest.raises(SystemExit, match="[Kk][Ss]"):
        arm_inject_spec("resolution", "ks")
    # desi/eboss still return a valid spec (guard is KS-only)
    assert arm_inject_spec("resolution", "eboss") == {"resolution": {"b_res": 0.02}}
