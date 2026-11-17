"""tests/ci_tiers.toml classifies every test file exactly once, with a tier (how it runs) and a level (when it runs:
pr, nightly, release or not run), so no file can drop out of CI unnoticed, and the scientific invariants of the pr
level cannot leave it without this test failing.

CI (.github/workflows/tests.yml, tools/ci/run_test_tier.py) runs the "wheel" and "source" tiers and refuses any skip
there except of a test named in its file's asset_tests or skipped_by_design, which is reported as not run, and any
expected failure except a deferred_legacy deferral or one whose exact message is in the file's deferred_xfails; every
other tier is reported as not run too, with the reason recorded here."""

import ast
import glob
import os
import tomllib

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TIERS = {"wheel", "source", "assets", "deferred", "external", "no_tests"}
CI_TIERS = {"wheel", "source"}
LEVELS = {"pr", "nightly", "release", "not run"}
# the levels each tier may take: the CI tiers run at pr or nightly; the assets tier only with its assets (or never);
# the other tiers never, except an external file whose extra package the nightly workflow installs
TIER_LEVELS = {
    "wheel": {"pr", "nightly"},
    "source": {"pr", "nightly"},
    "assets": {"release", "not run"},
    "deferred": {"not run"},
    "no_tests": {"not run"},
    "external": {"nightly", "not run"},
}
PER_TEST = {"pr_tests": "nightly", "nightly_tests": "pr", "release_tests": None}

# The tests that check an independent scientific invariant at the pr level (a result against a closed form, an
# independent implementation, a known identity or a stored reference from another code path) in five areas (the
# likelihood, the covariance, normalisation, the predictor's reconstruction and the incidence model), from an audit
# of the pr level (2026-11-17), and two that the audit kept at pr (the exact inverse of the training encoding; the
# truth-recovery tests of the multi-fidelity correction, the whole file). A test listed here must run at the pr
# level: its file is at pr and does not move it to nightly_tests, or its file is at nightly and names it in pr_tests.
PR_INVARIANTS = {
    "the likelihood": [
        (
            "tests/test_likelihood.py",
            "test_the_log_likelihood_is_the_gaussian_density_with_its_normalisation",
        ),
        (
            "tests/test_likelihood.py",
            "test_the_order_of_the_redshifts_is_the_data_sets_and_a_coupled_covariance_is_used_in_full",
        ),
        (
            "tests/test_likelihood.py",
            "test_the_model_vector_is_the_mixed_spectrum_at_the_data_wavenumbers_times_the_factor",
        ),
        ("tests/test_likelihood.py", "test_absolute_hcd_fractions_replace_the_models_own"),
        (
            "tests/test_likelihood.py",
            "test_a_mock_made_at_a_point_has_zero_chi2_there_and_a_lower_likelihood_elsewhere",
        ),
        (
            "tests/test_likelihood.py",
            "test_gaussian_terms_are_stated_log_densities_added_by_logp_only",
        ),
        (
            "tests/test_likelihood.py",
            "test_interpolation_is_exact_for_a_line_and_never_extrapolates",
        ),
        (
            "tests/test_likelihood.py",
            "test_modes_per_bin_is_the_spacing_of_the_bin_centres_over_the_spacing_of_the_modes",
        ),
        (
            "tests/test_example_mock_recovery.py",
            "test_the_chain_is_read_with_its_weights_after_burn_in_and_the_predictive_is_drawn",
        ),
    ],
    "the covariance": [
        (
            "tests/test_likelihood.py",
            "test_a_covariance_that_is_symmetric_up_to_rounding_is_kept_as_given",
        ),
        (
            "tests/test_likelihood.py",
            "test_the_predictive_is_the_model_plus_noise_of_the_data_covariance",
        ),
        (
            "tests/test_likelihood.py",
            "test_a_data_set_from_blocks_is_ordered_redshift_by_redshift_and_read_only",
        ),
        (
            "tests/test_likelihood.py",
            "test_the_xq100_loader_reads_five_redshifts_of_thirteen_bins_from_the_users_files",
        ),
    ],
    "normalisation": [
        ("tests/test_likelihood.py", "test_hcd_fractions_relative_to_priya_scale_the_models_own"),
        ("tests/test_predictor.py", "test_add_hcd_is_the_mixture_with_one_amplitude_per_class"),
        ("tests/test_predictor.py", "test_amplitudes_per_redshift_on_a_stack"),
        ("tests/test_predictor.py", "test_the_mean_flux_conversions_are_the_historical_ones"),
        ("tests/test_predictor.py", "test_the_mean_flux_inputs"),
        ("tests/test_emulator_dndx_wc.py", "test_wc_sums_to_one_and_matches_paper_point"),
        ("tests/test_emulator_dndx_wc.py", "test_wc_corrected_batched_z_shape_and_sum_at_extremes"),
        ("tests/test_emulator_dndx_wc.py", "test_wc_from_mu_inputs_are_disjoint_binned_classes"),
        ("tests/test_wc_map.py", "test_build_writes_a_stamped_map_bound_to_the_cache"),
        ("tests/test_emulator_model.py", "test_structural_identity_through_transform_roundtrip"),
        (
            "tests/test_docstring_examples.py",
            "test_the_docstring_examples_run_and_print_what_they_say",
        ),
    ],
    "the predictor's reconstruction": [
        (
            "tests/test_predictor.py",
            "test_the_prediction_is_the_ensemble_mean_with_its_spread_template_and_w_c",
        ),
        ("tests/test_predictor.py", "test_native_modes_and_requested_k"),
        ("tests/test_predictor.py", "test_the_stack_equals_the_single_predictions"),
        ("tests/test_grad_fidelity.py", "test_predict_P_filt_matches_numpy_reconstruct"),
        ("tests/test_emulator_data.py", "test_normalize_params_maps_to_unit_cube"),
        ("tests/test_emulator_model.py", "test_baseline_head_is_theta_blind"),
        ("tests/test_kcoord.py", "test_first_principles_grid_matches_box_formula"),
        ("tests/test_kcoord.py", "test_h_cancels"),
        ("tests/test_kcoord.py", "test_cosmology_response_is_exactly_minus_dlnE"),
        ("tests/test_kcoord.py", "test_kgrid_object_carries_the_contract"),
        ("tests/test_kcoord_binding.py", "test_binding_equals_linear_interpolation_in_k"),
        (
            "tests/test_kcoord_binding.py",
            "test_binding_gradient_in_theta_equals_interp_with_moving_abscissa",
        ),
        (
            "tests/test_kcoord_binding.py",
            "test_binding_counts_bins_outside_the_modes_and_clamps_like_interp",
        ),
        ("tests/test_kcoord_binding.py", "test_box_bounds_equal_a_brute_force_scan"),
        ("tests/test_kcoord_binding.py", "test_box_bounds_follow_a_narrowed_sampling_box"),
        ("tests/test_schema.py", "test_validator_accepts_a_consistent_cache"),
        ("tests/test_predictor_mf.py", "test_mf_is_applied_to_the_lf_prediction"),
        ("tests/test_predictor_mf.py", "test_mf_is_applied_per_member_before_the_k_placement"),
        ("tests/test_predictor_mf.py", "test_lf_prediction_is_bit_for_bit_unchanged"),
        (
            "tests/test_predictor_mf.py",
            "test_fold_mf_on_its_research_backbone_gives_held_out_predictions",
        ),
        (
            "tests/test_predictor_mf.py",
            "test_one_fold_through_the_predictor_reproduces_the_cache_input_path",
        ),
        ("tests/test_mode_mf_module.py", "test_module_is_jittable_and_exactly_blind_to_cosmology"),
        ("tests/test_emulator_model.py", "test_headB_dense_unchanged"),
        (
            "tests/test_build_mf_release.py",
            "test_release_build_binds_to_the_backbone_and_uses_its_lf_prediction",
        ),
        (
            "tests/test_build_mf_release.py",
            "test_d1_exception_passes_the_recorded_case_and_is_recorded",
        ),
    ],
    "the incidence model": [
        (
            "tests/test_incidence_mf.py",
            "test_the_fit_is_the_historical_functions_on_the_matched_pairs",
        ),
        (
            "tests/test_incidence_mf.py",
            "test_the_composition_leaves_simulations_out_and_says_which_remain",
        ),
        ("tests/test_incidence_mf.py", "test_the_cddf_takes_empty_bins_and_refuses_bad_ones"),
        (
            "tests/test_incidence_mf.py",
            "test_a_record_round_trips_and_its_factor_is_the_historical_lookup",
        ),
        (
            "tests/test_incidence_mf.py",
            "test_lf_mf_dndx_is_the_lf_dndx_times_the_factor_and_w_c_follows",
        ),
        (
            "tests/test_incidence_mf.py",
            "test_the_cddf_is_the_member_mean_times_the_factor_and_nan_where_nothing_supports_it",
        ),
        (
            "tests/test_incidence_mf.py",
            "test_a_fold_model_takes_an_incidence_record_built_without_its_held_out_simulations",
        ),
        (
            "tests/test_incidence_mf.py",
            "test_a_prediction_carries_the_held_out_range_on_the_quantity_itself",
        ),
        (
            "tests/test_incidence_mf.py",
            "test_a_bound_is_nan_where_there_is_no_estimate_and_none_without_a_record",
        ),
        (
            "tests/test_figure_scripts.py",
            "test_the_dndx_figure_runs_on_a_model_with_an_incidence_record",
        ),
        (
            "tests/test_build_incidence_mf.py",
            "test_a_build_writes_the_record_of_the_stated_composition",
        ),
        ("tests/test_wc_map.py", "test_the_delta_c_estimator_is_calibrate_delta_c"),
        ("tests/test_wc_map.py", "test_the_xbar_estimator_is_hcd_xbar_polyfit"),
        ("tests/test_wc_map.py", "test_the_historical_default_is_unchanged_and_labelled"),
        ("tests/test_wc_map.py", "test_an_explicit_map_is_used_clamped_and_traceable"),
        ("tests/test_wc_map.py", "test_the_training_set_refit_is_the_w1_fold_map"),
        ("tests/test_wc_map.py", "test_no_held_out_simulation_reaches_the_training_set_refit"),
        ("tests/test_wc_map.py", "test_golden_digests_of_the_map_rules"),
        ("tests/test_emulator_model.py", "test_headA_outputs_are_tau0_invariant_by_gradient"),
        ("tests/test_emulator_dndx_wc.py", "test_alpha_to_dndx_exact_inverse_of_wc"),
        ("tests/test_emulator_dndx_wc.py", "test_forward_inverse_roundtrip_with_delta"),
        (
            "tests/test_emulator_dndx_wc.py",
            "test_alpha_roundtrip_at_corrected_laws_remeasured_tolerance",
        ),
    ],
    "kept at pr by the audit": [
        ("tests/test_emulator_data.py", "test_reconstruction_roundtrip"),
    ],
}
PR_INVARIANT_FILES = {"tests/test_mf_modes.py"}  # kept at pr by the audit, the whole file


def _doc():
    with open(os.path.join(TESTS_DIR, "ci_tiers.toml"), "rb") as f:
        return tomllib.load(f)


def _test_files():
    root = os.path.dirname(TESTS_DIR)
    pats = ("test_*.py", "regression/test_*.py", "integration/test_*.py", "external/test_*.py")
    found = (p for pat in pats for p in glob.glob(os.path.join(TESTS_DIR, pat)))
    return {os.path.relpath(p, root).replace(os.sep, "/") for p in found}


def test_tier_names_are_the_known_ones():
    assert set(_doc()["tiers"]) == TIERS


def test_level_names_are_the_known_ones_and_every_file_has_one_its_tier_allows():
    assert set(_doc()["levels"]) == LEVELS
    for path, entry in _doc()["files"].items():
        assert entry.get("level") in TIER_LEVELS[entry["tier"]], (
            path,
            entry["tier"],
            entry.get("level"),
        )


def test_every_test_file_is_classified_exactly_once():
    listed = set(_doc()["files"])
    present = _test_files()
    assert not present - listed, f"unclassified test files: {sorted(present - listed)}"
    assert not listed - present, f"classified files that do not exist: {sorted(listed - present)}"


def test_each_entry_has_a_known_tier_and_out_of_ci_entries_say_why():
    for path, entry in _doc()["files"].items():
        assert set(entry) <= {
            "tier",
            "level",
            "reason",
            "asset_tests",
            "skipped_by_design",
            "deferred_xfails",
            *PER_TEST,
        }, path
        assert entry["tier"] in TIERS, path
        if entry["tier"] not in CI_TIERS or set(entry) & {
            "asset_tests",
            "skipped_by_design",
            *PER_TEST,
        }:
            assert entry.get(
                "reason", ""
            ).strip(), f"{path}: a file or test outside CI needs its reason"


def test_declared_skips_are_test_functions_of_their_file():
    """asset_tests and skipped_by_design (allowed only in a CI tier) name test functions of the file, or single
    parameter cases of them ("test_x[a]"), each once, so a rename leaves no stale entry."""
    root = os.path.dirname(TESTS_DIR)
    for path, entry in _doc()["files"].items():
        names = entry.get("asset_tests", []) + entry.get("skipped_by_design", [])
        if not names:
            continue
        assert entry["tier"] in CI_TIERS, path
        assert len(set(names)) == len(names), path
        with open(os.path.join(root, path), encoding="utf-8") as f:
            defined = {
                n.name for n in ast.walk(ast.parse(f.read())) if isinstance(n, ast.FunctionDef)
            }
        bases = {n.split("[")[0] for n in names}
        assert bases <= defined, f"{path}: not test functions here: {sorted(bases - defined)}"


def test_declared_deferrals_are_exact_messages_in_a_ci_tier():
    """deferred_xfails lists full expected-failure messages (exact matches, never prefixes) of a CI-tier file."""
    for path, entry in _doc()["files"].items():
        if "deferred_xfails" in entry:
            assert entry["tier"] in CI_TIERS and entry.get("reason", "").strip(), path
            msgs = entry["deferred_xfails"]
            assert msgs and all(isinstance(m, str) and len(m) > 40 for m in msgs), path


def test_external_tier_is_exactly_tests_external():
    for path, entry in _doc()["files"].items():
        assert (entry["tier"] == "external") == path.startswith("tests/external/"), path


def _functions(path):
    root = os.path.dirname(TESTS_DIR)
    with open(os.path.join(root, path), encoding="utf-8") as f:
        return {n.name for n in ast.walk(ast.parse(f.read())) if isinstance(n, ast.FunctionDef)}


def test_tests_at_another_level_than_their_file_are_test_functions_of_it_and_no_prefix_of_another():
    """pr_tests (of a nightly file), nightly_tests (of a pr file) and release_tests (of a pr or nightly file) name
    test functions of their CI-tier file, each once and in one list only. None is the start of another function's
    name: the runner selects and deselects by pytest node-id prefix."""
    for path, entry in _doc()["files"].items():
        named = [(key, n) for key in PER_TEST for n in entry.get(key, [])]
        if not named:
            continue
        assert entry["tier"] in CI_TIERS, path
        for key in PER_TEST:
            if key in entry:
                assert entry[key], (path, key)
                assert PER_TEST[key] in (None, entry["level"]), (path, key, entry["level"])
        names = [n for _, n in named]
        assert len(set(names)) == len(names), path
        defined = _functions(path)
        assert (
            set(names) <= defined
        ), f"{path}: not test functions here: {sorted(set(names) - defined)}"
        for n in names:
            assert not [d for d in defined if d != n and d.startswith(n)], (path, n)


def _runs_at_pr(path, test):
    entry = _doc()["files"][path]
    if test in entry.get("release_tests", []):
        return False
    if entry["level"] == "pr":
        return test not in entry.get("nightly_tests", [])
    return entry["level"] == "nightly" and test in entry.get("pr_tests", [])


def test_the_scientific_invariants_of_the_pr_level_run_at_pr():
    """Every test of PR_INVARIANTS exists in its file and runs at the pr level, in a CI tier: none can move to
    nightly, release or out of CI, and none can be renamed, without this test failing first."""
    files = _doc()["files"]
    n = 0
    for area, rows in PR_INVARIANTS.items():
        for path, test in rows:
            assert files[path]["tier"] in CI_TIERS, (area, path)
            assert test in _functions(path), (area, path, test)
            assert _runs_at_pr(path, test), f"{area}: {path}::{test} does not run at the pr level"
            n += 1
    for path in PR_INVARIANT_FILES:
        entry = files[path]
        assert entry["tier"] in CI_TIERS and entry["level"] == "pr", path
        assert not entry.get("nightly_tests") and not entry.get("release_tests"), path
    assert n >= 72
