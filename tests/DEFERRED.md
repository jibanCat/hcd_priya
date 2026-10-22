# Deferred tests of the pre-2026-10 inference and closure stack

136 functions (154 test nodes, measured on the real data links) in 28 files.

Gate E (the 2026 k-grid fix, `docs/kgrid_incident_2026-10.md`) retired the pre-2026-10 closure and inference
interfaces. The tests below exercise those interfaces. Each carries `@pytest.mark.deferred_legacy(CAUSE)`:
`tests/conftest.py` treats it as an expected failure only when the error matches the cause's exact pattern
(`DEFERRED_CAUSES`). Any other failure fails the test, and a pass fails it too (strict). These tests cover the
retired stack only; they are not coverage of the current forward model. Rebuilding them against the current
interfaces is follow-up work.

The 7 strict `xfail` markers (9 test nodes) of the retired mock/truth builders and the 9 lock-generator tests
(`LockGeneratorDeferred`) are separate and documented at their definitions.

## CTX_SIGNATURE (106 functions, 124 test nodes)

the pre-2026-10 `build_legb_ctx` signature (`use_xclass`, `with_mf`, a single `ckpt`) is retired at gate E; the gate E context takes `ensemble_ckpts`.

- `tests/test_eboss_leg.py`:
  - `test_legb_model_priors_only_site_order_match` (4 parametrized nodes)
- `tests/test_hcd_pivot_center.py`:
  - `test_build_legb_ctx_real_fit_alpha_hcd_mu_is_lit_consistent`
  - `test_hier_ratios_rederived_at_pivot_z`
- `tests/test_ks_dndx_reparam.py`:
  - `test_forbidden_combinations_raise`
  - `test_induced_geometry_vs_deployed_prior`
  - `test_ks_legacy_override_reproduces_old_prior`
  - `test_non_ks_builds_have_no_mapped_fields`
  - `test_reconstruct_matches_model_deterministics`
  - `test_reference_curve_pins`
  - `test_reference_lls_component_is_boosted_lit_law`
  - `test_self_draw_truth_pack_on_mapped_ctx`
  - `test_structural_bound_extreme_draws`
  - `test_survey_alpha_prior_vectors_bit_unchanged`
  - `test_twin_parity_sites_and_values`
- `tests/test_legb_2d_tilt.py`:
  - `test_2d_draws_matrix_appends_A_B_r_columns`
  - `test_2d_fast_postprocess_yields_alpha_columns`
  - `test_2d_ignores_marginalize_zslope`
  - `test_2d_loglik_grad_in_A_B_r_finite`
  - `test_2d_pivot_alpha_equals_A_times_r`
  - `test_2d_positivity_prior_draws`
  - `test_2d_reconstruct_deterministics_uses_btilt`
  - `test_2d_requires_hierarchical`
  - `test_2d_slopes_match_incidence_slope_at_btilt_center`
  - `test_2d_z_shape_uses_btilt_plus_dslope`
  - `test_ctx_2d_defaults_are_closure_self_consistent`
  - `test_loglik_core_raises_on_zflat_by_default`
  - `test_model_priors_only_site_order_match_2d` (6 parametrized nodes)
  - `test_off_legacy_byte_exact_when_2d_off`
  - `test_off_optionB_byte_exact_when_2d_off`
- `tests/test_legb_convergence.py`:
  - `test_convergence_mode_disperses_and_returns_finite_battery`
- `tests/test_legb_dla_masking.py`:
  - `test_make_legb_mock_applies_per_leg_truth_frac`
  - `test_real_ctx_truth_forward_consistency_desi_ten_percent`
  - `test_truth_pack_alpha_dla_is_ten_percent_residual`
- `tests/test_legb_golden.py`:
  - `test_legb_legacy_forward_matches_golden`
  - `test_legb_mf_golden`
- `tests/test_legb_hier_hcd.py`:
  - `test_fast_postprocess_on_branch_yields_alpha_columns`
  - `test_loglik_grad_in_A_r_finite`
  - `test_model_priors_only_site_order_match` (8 parametrized nodes)
  - `test_off_branch_potential_byte_exact_vs_legacy`
  - `test_off_branch_site_list_is_the_current_code`
  - `test_on_branch_reparam_alpha_equals_A_times_r`
  - `test_ratio_centers_match_closure_truth`
  - `test_ratio_infl_scales_widths_not_centers`
  - `test_reconstruct_deterministics_on_branch_reinserts_alpha`
- `tests/test_legb_leg_a.py`:
  - `test_build_legb_ctx_ensemble`
- `tests/test_legb_metal_flatlog.py`:
  - `test_a_siII_threaded_into_rescore_loglik`
  - `test_amplitude_map_and_Fref_provenance`
  - `test_constrain_fn_mirror_parity` (2 parametrized nodes)
  - `test_flatlog_sites_are_loguniform_with_expected_bounds`
  - `test_uniform_default_site_construction_unchanged`
- `tests/test_legb_metal_modelcplus.py`:
  - `test_arm4_out_of_class_decreasing_gauss`
  - `test_arm_desi_in_class_carries_cross`
  - `test_arm_eboss_in_class_to_machine_precision`
  - `test_knode_constrain_fn_mirror_parity`
  - `test_knode_site_inventory_and_order`
  - `test_knodes_none_uses_scalar_decorr_default`
  - `test_metal_node_sites_extra_helper`
  - `test_per_z_decorrelation_interp_and_forward_golden`
- `tests/test_legb_metal_zevo.py`:
  - `test_arm3_ma2025_injection_finite_and_siII_off`
  - `test_arm_injection_de_double_count_and_desi_siII`
  - `test_arm_injection_eboss_is_siIII_only`
  - `test_cemu_variance_transform_uses_same_per_z_mfac`
  - `test_eboss_high_z_clamp_forward_finite`
  - `test_flatlog2node_constrain_fn_mirror_parity`
  - `test_flatlog2node_site_inventory_and_order`
  - `test_per_z_map_equals_metal_inject_and_forward_golden`
  - `test_predict_nodes_none_is_scalar_path_byte_exact`
- `tests/test_legb_per_leg_alpha.py`:
  - `test_dict_loglik_equals_per_leg_sum`
  - `test_guard_per_leg_zslope_requires_per_leg_alpha`
  - `test_joint_builder_fields_and_poison`
  - `test_joint_reconstruction_matches_replay` (2 parametrized nodes)
  - `test_joint_stamp_contents`
  - `test_joint_stamp_rejects_mixed_signatures`
  - `test_joint_trace_per_leg_zslope_switch`
  - `test_joint_trace_site_names_and_order`
  - `test_mirror_parity_per_leg` (2 parametrized nodes)
  - `test_one_leg_joint_bitwise_rename_equivalence`
  - `test_per_leg_priors_bitwise_match_single_leg_builds`
  - `test_reconstruct_rejects_shared_samples_on_per_leg_zslope_ctx`
  - `test_short_nuts_matches_golden`
  - `test_trace_matches_sites_golden`
- `tests/test_legb_postprocess_golden.py`:
  - `test_fast_postprocess_matches_replay_golden`
- `tests/test_legb_resolution.py`:
  - `test_bres_vec_forward_response_per_z_indexed`
  - `test_f_res_prior_sigma_configurable`
  - `test_fres_sites_in_both_models_same_order`
  - `test_fres_sites_present_ordered_and_golden_off`
- `tests/test_norc_forward.py`:
  - `test_default_reproduces_mf_golden`
  - `test_norc_forward_drops_log_rc`
  - `test_norc_ks_kmax_cap`
  - `test_res_corr_off_returns_ones`
- `tests/test_res_corr_alpha.py`:
  - `test_alpha_one_reproduces_golden`
  - `test_alpha_response_forward_only`
  - `test_alpha_sites_in_both_models_same_order`
  - `test_alpha_z_zslope`
  - `test_truth_unchanged_by_alpha`
- `tests/test_res_corr_inject.py`:
  - `test_inject_b1_perturbs_truth`
  - `test_inject_is_z_ge_2p8_localized`
  - `test_inject_none_is_noop`
  - `test_inject_on_truth_not_forward`
- `tests/test_res_corr_norc_injection.py`:
  - `test_bvec_ks_capped`
  - `test_bvec_matches_interp_res_corr`
  - `test_inject_multiplies_truth_by_exp_b`
  - `test_noop_default_byte_identical`
- `tests/test_sampling_box_restriction.py`:
  - `test_prior_predictive_inside_original_priya_box`
- `tests/test_tau0_informative_prior.py`:
  - `test_priors_only_site_order_identical_uniform_vs_gauss`
  - `test_priors_only_site_order_unchanged_default`

## REAL_FIT_CKPT_DIR (20 functions, 20 test nodes)

`scripts/run_real_fit.build_real_ctx` verifies the ensemble against `<repo>/checkpoints` (the schema-1 layout); the schema-2 members live in the manifest's `checkpoints_dir`.

- `tests/test_dla_completeness_forward_parity.py`:
  - `test_default_arm_ctx_byte_identical`
  - `test_dla_desi_norc_pins_fix_alpha_res`
  - `test_dla_desi_uses_deployed_forward`
- `tests/test_ks_dndx_reparam.py`:
  - `test_model_smoke_real_ks_ctx`
- `tests/test_norc_single_authority.py`:
  - `test_cross_driver_norc_fix_alpha_res`
  - `test_dnuis_oos_ks_literal_matches_map`
  - `test_flip_side_dnuis_fix_alpha_res`
  - `test_one_leg_flip_guard`
  - `test_restore_flip_does_not_trip_ks_cap`
  - `test_sbc_cli_restore_still_wins`
  - `test_sbc_default_tracks_authority`
  - `test_sbc_resolved_forward_and_runcfg_keyset`
- `tests/test_prod_forward_wiring.py`:
  - `test_build_real_ctx_desi_eboss_kskwargs_none`
  - `test_build_real_ctx_desi_forwards_certified_knobs`
  - `test_build_real_ctx_eboss_width_005`
  - `test_build_real_ctx_ks_forwards_fres_and_kskwargs`
  - `test_build_real_ctx_ks_merges_zlo_with_kskwargs`
- `tests/test_r6_pairing.py`:
  - `test_mapped_truth_in_simplex_and_site_set`
  - `test_override_flips_only_the_parameterization`
  - `test_shared_truth_gives_bit_identical_mocks`

## SCHEMA1_CHECKPOINT (9 functions, 9 test nodes)

the test loads a pre-2026-10 (schema-1) checkpoint, which `train.load_checkpoint` refuses by design.

- `tests/test_closure_sbc.py`:
  - `test_leg_a_mock_C_matches_likelihood`
  - `test_potential_equals_loglik_plus_logprior`
  - `test_tiny_nuts_runs_finite`
  - `test_vmap_sum_equals_loop_sum`
- `tests/test_data_likelihood.py`:
  - `test_mf_path_matches_gate_script_to_tight_tol`
- `tests/test_prod_sbc.py`:
  - `test_build_ctx_ensemble_uses_ensemble_model`
  - `test_build_ctx_single_model_path_unchanged`
  - `test_mock_for_index_deterministic_and_shard_invariant`
  - `test_run_leg_a_sbc_return_per_mock_shards`

## RETIRED_BUILDER (1 functions, 1 test nodes)

the pre-2026-10 mock/truth builder is retired at gate E (a `NotImplementedError` stub).

- `tests/test_legb_dla_masking.py`:
  - `test_make_truth_stores_masked_baseline_and_full_dla_excess`
