#!/bin/bash
# Excluded unit tests for FAST CI mode (Lfast).
# Source this file and append "${EXCLUDED_UNIT_TESTS[@]}" to pytest args.
# Supports: --ignore=<path>, --ignore-glob=<pattern>, --deselect=<node_id>
# All paths are relative to tests/ (run_unit.sh cwd).

EXCLUDED_UNIT_TESTS=(
    ###########################################################################
    # ALGORITHMS — exclude heavy GPU/Ray integration tests, keep core logic
    ###########################################################################

    # Sequence packing gradients — requires multi-GPU + Ray, 1 test
    --ignore=unit/algorithms/test_sequence_packing_gradients.py

    # test_grpo.py — keep ~9 core tests, exclude 14 heavy/redundant ones
    --deselect=unit/algorithms/test_grpo.py::test_calculate_rewards_multiple_tasks
    --deselect=unit/algorithms/test_grpo.py::test_calculate_rewards_missing_environment
    --deselect=unit/algorithms/test_grpo.py::test_dapo_dynamic_sampling_filters_zero_std
    --deselect=unit/algorithms/test_grpo.py::test_dapo_dynamic_sampling_batch_caching
    --deselect=unit/algorithms/test_grpo.py::test_dapo_dynamic_sampling_disabled
    --deselect=unit/algorithms/test_grpo.py::test_noncolocated_inference_requires_explicit_gpus_per_node_multi_node
    --deselect=unit/algorithms/test_grpo.py::test_refit_policy_generation_sglang_colocated_http
    --deselect=unit/algorithms/test_grpo.py::test_refit_policy_generation_sglang_non_colocated_raises
    --deselect=unit/algorithms/test_grpo.py::test_grpo_advantage_estimator_zero_std
    --deselect=unit/algorithms/test_grpo.py::test_grpo_advantage_estimator_negative_advantages
    --deselect=unit/algorithms/test_grpo.py::test_grpo_advantage_estimator_zero_std_and_zero_advantage
    --deselect=unit/algorithms/test_grpo.py::test_grpo_advantage_estimator_small_nonzero_std

    # test_loss_functions.py — keep 10 core losses, exclude 21 variants/edge cases
    --deselect=unit/algorithms/test_loss_functions.py::test_clipped_pg_loss_force_on_policy_ratio
    --deselect=unit/algorithms/test_loss_functions.py::test_clipped_pg_loss_zero_mask
    --deselect=unit/algorithms/test_loss_functions.py::test_clipped_pg_loss_on_policy_kl_importance_sampling
    --deselect=unit/algorithms/test_loss_functions.py::test_clipped_pg_loss_on_policy_truncated_importance_sampling
    --deselect=unit/algorithms/test_loss_functions.py::test_clipped_pg_loss_icepop_importance_sampling
    --deselect=unit/algorithms/test_loss_functions.py::test_clipped_pg_loss_seq_mask_tis
    --deselect=unit/algorithms/test_loss_functions.py::test_masked_mean_all_zeros
    --deselect=unit/algorithms/test_loss_functions.py::test_clipped_pg_loss_dual_clip
    --deselect=unit/algorithms/test_loss_functions.py::test_clipped_pg_loss_entropy
    --deselect=unit/algorithms/test_loss_functions.py::test_clipped_pg_loss_gspo
    --deselect=unit/algorithms/test_loss_functions.py::test_clipped_pg_loss_gspo_batch_size_2
    --deselect=unit/algorithms/test_loss_functions.py::test_clipped_pg_loss_gspo_importance_sampling_correction
    --deselect=unit/algorithms/test_loss_functions.py::test_distillation_loss_mixed_kl
    --deselect=unit/algorithms/test_loss_functions.py::test_distillation_loss_invalid_k_zero
    --deselect=unit/algorithms/test_loss_functions.py::test_distillation_loss_zero_outside_topk
    --deselect=unit/algorithms/test_loss_functions.py::test_distillation_loss_gradient_flow
    --deselect=unit/algorithms/test_loss_functions.py::test_distillation_loss_edge_cases
    --deselect=unit/algorithms/test_loss_functions.py::test_distillation_loss_fn_initialization
    --deselect=unit/algorithms/test_loss_functions.py::test_distillation_loss_fn_call

    # test_distillation.py — keep 3, exclude 7 error-path/edge-case tests
    --deselect=unit/algorithms/test_distillation.py::test_validate_function
    --deselect=unit/algorithms/test_distillation.py::test_check_vocab_equality_pass
    --deselect=unit/algorithms/test_distillation.py::test_check_vocab_equality_vocab_mismatch_raises
    --deselect=unit/algorithms/test_distillation.py::test_check_vocab_equality_length_mismatch_raises
    --deselect=unit/algorithms/test_distillation.py::test_check_vocab_equality_config_vocab_size_mismatch_raises
    --deselect=unit/algorithms/test_distillation.py::test_distillation_setup_non_colocated_smoke
    --deselect=unit/algorithms/test_distillation.py::test_noncolocated_inference_requires_explicit_gpus_per_node_multi_node

    # test_sft.py — keep 3 core loop tests, exclude 2 edge cases
    --deselect=unit/algorithms/test_sft.py::test_training_with_disabled_validation
    --deselect=unit/algorithms/test_sft.py::test_training_with_negative_val_period

    # test_utils.py — keep 5 core, exclude 11 (HF gated tokenizer tests + edge cases)
    --deselect=unit/algorithms/test_utils.py::test_get_tokenizer_no_chat_template
    --deselect=unit/algorithms/test_utils.py::test_get_tokenizer_default_chat_template
    --deselect=unit/algorithms/test_utils.py::test_get_tokenizer_null_chat_template
    --deselect=unit/algorithms/test_utils.py::test_get_tokenizer_custom_jinja_template
    --deselect=unit/algorithms/test_utils.py::test_async_non_colocated_idle_ratio_and_generation_time
    --deselect=unit/algorithms/test_utils.py::test_minimal_inputs_no_counts_no_flops
    --deselect=unit/algorithms/test_utils.py::test_calculate_baseline_and_std_per_prompt_single_generation_per_prompt
    --deselect=unit/algorithms/test_utils.py::test_calculate_baseline_and_std_per_prompt_identical_rewards
    --deselect=unit/algorithms/test_utils.py::test_calculate_baseline_and_std_per_prompt_empty_input
    --deselect=unit/algorithms/test_utils.py::test_calculate_baseline_and_std_per_prompt_nan_handling
    --deselect=unit/algorithms/test_utils.py::test_calculate_baseline_and_std_per_prompt_numerical_precision

    # test_reward_functions.py — keep 5 core, exclude 8 edge/error paths
    --deselect=unit/algorithms/test_reward_functions.py::test_reward_scaling_disabled
    --deselect=unit/algorithms/test_reward_functions.py::test_reward_shaping_disabled
    --deselect=unit/algorithms/test_reward_functions.py::test_reward_shaping_no_penalties
    --deselect=unit/algorithms/test_reward_functions.py::test_reward_shaping_missing_config_values
    --deselect=unit/algorithms/test_reward_functions.py::test_reward_shaping_missing_assistant_response
    --deselect=unit/algorithms/test_reward_functions.py::test_reward_shaping_mismatched_lengths
    --deselect=unit/algorithms/test_reward_functions.py::test_stop_properly_penalty_boundary_coefs
    --deselect=unit/algorithms/test_reward_functions.py::test_stop_properly_penalty_error_cases

    ###########################################################################
    # DISTRIBUTED — exclude 2D sharding variants, infrastructure tests
    ###########################################################################

    # test_virtual_cluster.py — all infrastructure/setup tests, 10 tests
    --ignore=unit/distributed/test_virtual_cluster.py

    # test_worker_groups.py — keep 8 core, exclude 14 (2D sharding variants, nsight, env)
    --deselect=unit/distributed/test_worker_groups.py::test_custom_environment_variables
    --deselect=unit/distributed/test_worker_groups.py::test_custom_environment_variables_override_existing
    --deselect=unit/distributed/test_worker_groups.py::test_run_all_workers_single_data_2d_sharding_no_filter
    --deselect=unit/distributed/test_worker_groups.py::test_run_all_workers_single_data_2d_sharding_filter_tp
    --deselect=unit/distributed/test_worker_groups.py::test_run_all_workers_single_data_2d_sharding_filter_dp_tp
    --deselect=unit/distributed/test_worker_groups.py::test_run_all_workers_multiple_data_fewer_data_than_workers
    --deselect=unit/distributed/test_worker_groups.py::test_run_all_workers_sharded_data_2d_shard_dp_replicate_tp
    --deselect=unit/distributed/test_worker_groups.py::test_run_all_workers_sharded_data_2d_free_axis_dp_shard_tp
    --deselect=unit/distributed/test_worker_groups.py::test_run_all_workers_sharded_data_2d_free_axis_dummy_calls
    --deselect=unit/distributed/test_worker_groups.py::test_run_all_workers_sharded_data_2d_output_replicated
    --deselect=unit/distributed/test_worker_groups.py::test_nsight_configuration_forwarding
    --deselect=unit/distributed/test_worker_groups.py::test_get_nsight_config_if_pattern_matches
    --deselect=unit/distributed/test_worker_groups.py::test_get_nsight_config_output_format
    --deselect=unit/distributed/test_worker_groups.py::test_environment_variable_precedence_full

    # test_batched_data_dict.py — keep 3 core, exclude 13 variants/edge cases
    --deselect=unit/distributed/test_batched_data_dict.py::test_shard_by_batch_size_list_data
    --deselect=unit/distributed/test_batched_data_dict.py::test_shard_by_batch_size_larger_example
    --deselect=unit/distributed/test_batched_data_dict.py::test_shard_by_batch_size_edge_cases
    --deselect=unit/distributed/test_batched_data_dict.py::test_shard_by_batch_size_validation
    --deselect=unit/distributed/test_batched_data_dict.py::test_shard_by_batch_size_matches_example
    --deselect=unit/distributed/test_batched_data_dict.py::test_shard_by_batch_size_dynamic
    --deselect=unit/distributed/test_batched_data_dict.py::test_sequence_packing_uniform_lengths
    --deselect=unit/distributed/test_batched_data_dict.py::test_sequence_packing_long_sequences
    --deselect=unit/distributed/test_batched_data_dict.py::test_sequence_packing_with_dynamic_batching_conflict
    --deselect=unit/distributed/test_batched_data_dict.py::test_shard_by_batch_size_with_packed_multimodal
    --deselect=unit/distributed/test_batched_data_dict.py::test_get_multimodal_dict_mixed_content_and_device_move
    --deselect=unit/distributed/test_batched_data_dict.py::test_from_batches_pads_3d_tensors_along_sequence_dim
    --deselect=unit/distributed/test_batched_data_dict.py::test_sequence_packing_microbatch_boundaries

    ###########################################################################
    # DATA — exclude HF gated model loading, heavy parametrization
    ###########################################################################

    # test_response_dataset.py — all HF loading tests, 7 tests
    --ignore=unit/data/datasets/test_response_dataset.py

    # test_data_shuffle_reproducity.py — parametrized model loading, 2 tests
    --ignore=unit/data/test_data_shuffle_reproducity.py

    # test_llm_message_utils.py — keep 8 core, exclude 17 edge/HF/multimodal
    --deselect=unit/data/test_llm_message_utils.py::test_message_log_to_flat_messages_empty
    --deselect=unit/data/test_llm_message_utils.py::test_message_log_to_flat_messages_missing_keys
    --deselect=unit/data/test_llm_message_utils.py::test_concatenate_messages_different_shapes
    --deselect=unit/data/test_llm_message_utils.py::test_get_keys_from_messages_empty
    --deselect=unit/data/test_llm_message_utils.py::test_get_keys_from_messages_empty_keys
    --deselect=unit/data/test_llm_message_utils.py::test_get_keys_from_messages_all_missing
    --deselect=unit/data/test_llm_message_utils.py::test_batch_pad_message_log_empty_batch
    --deselect=unit/data/test_llm_message_utils.py::test_batch_pad_message_log_no_tensors
    --deselect=unit/data/test_llm_message_utils.py::test_batch_pad_messages_mixed_dtypes
    --deselect=unit/data/test_llm_message_utils.py::test_batch_pad_message_log_divisible_by
    --deselect=unit/data/test_llm_message_utils.py::test_batch_pad_message_log_custom_pad_value
    --deselect=unit/data/test_llm_message_utils.py::test_get_formatted_message_log_models
    --deselect=unit/data/test_llm_message_utils.py::test_get_formatted_message_log_qwen3_enable_thinking
    --deselect=unit/data/test_llm_message_utils.py::test_formatted_message_log_empty_message
    --deselect=unit/data/test_llm_message_utils.py::test_message_log_to_flat_messages_with_packed_images
    --deselect=unit/data/test_llm_message_utils.py::test_batched_message_log_to_flat_message_with_packed_images
    --deselect=unit/data/test_llm_message_utils.py::test_get_formatted_message_log_multimodal_prompt_formatting

    # test_data_processor.py — keep 2 basic, exclude 3 heavy parametrized HF
    --deselect=unit/data/test_data_processor.py::test_math_hf_data_processor
    --deselect=unit/data/test_data_processor.py::test_math_hf_data_processor_without_prompt
    --deselect=unit/data/test_data_processor.py::test_eval_math_hf_data_processor

    # test_oai_format_dataset.py — keep 1, exclude 2
    --deselect=unit/data/datasets/test_oai_format_dataset.py::test_custom_keys
    --deselect=unit/data/datasets/test_oai_format_dataset.py::test_message_formatting

    ###########################################################################
    # MODELS — Generation (run in L0_Unit_Tests_Generation)
    ###########################################################################

    # Exclude all heavy GPU integration tests, keep config validation + utilities
    --ignore=unit/models/generation/test_vllm_large_model.py

    # test_vllm_generation.py — keep 3 lightweight, exclude 19 GPU integration
    --deselect=unit/models/generation/test_vllm_generation.py::test_input_data
    --deselect=unit/models/generation/test_vllm_generation.py::test_vllm_policy_generation
    --deselect=unit/models/generation/test_vllm_generation.py::test_vllm_worker_seed_behavior
    --deselect=unit/models/generation/test_vllm_generation.py::test_vllm_policy_tensor_parallel
    --deselect=unit/models/generation/test_vllm_generation.py::test_vllm_generate_text
    --deselect=unit/models/generation/test_vllm_generation.py::test_vllm_http_server
    --deselect=unit/models/generation/test_vllm_generation.py::test_replace_prefix_tokens_empty_model_prefix_returns_template
    --deselect=unit/models/generation/test_vllm_generation.py::test_replace_prefix_tokens_missing_eos_in_template_prefix_raises
    --deselect=unit/models/generation/test_vllm_generation.py::test_replace_prefix_tokens_tokenizer_without_eos_raises
    --deselect=unit/models/generation/test_vllm_generation.py::test_replace_prefix_tokens_uses_last_eos_in_template_prefix
    --deselect=unit/models/generation/test_vllm_generation.py::test_vllm_weight_update_and_prefix_cache_reset
    --deselect=unit/models/generation/test_vllm_generation.py::test_vllm_weight_update_memory
    --deselect=unit/models/generation/test_vllm_generation.py::test_vllm_generation_with_stop
    --deselect=unit/models/generation/test_vllm_generation.py::test_vllm_non_divisible_batch_handling
    --deselect=unit/models/generation/test_vllm_generation.py::test_vllm_generation_with_megatron_training
    --deselect=unit/models/generation/test_vllm_generation.py::test_vllm_generation_with_megatron_training_moe_model
    --deselect=unit/models/generation/test_vllm_generation.py::test_vllm_megatron_weight_update_memory
    --deselect=unit/models/generation/test_vllm_generation.py::test_vllm_megatron_pipeline_parallel
    --deselect=unit/models/generation/test_vllm_generation.py::test_vllm_megatron_weight_update_with_packing

    # test_sglang_generation.py — keep 2 config validation, exclude 10 GPU integration
    --deselect=unit/models/generation/test_sglang_generation.py::test_input_data
    --deselect=unit/models/generation/test_sglang_generation.py::test_sglang_policy_generation
    --deselect=unit/models/generation/test_sglang_generation.py::test_sglang_worker_seed_behavior
    --deselect=unit/models/generation/test_sglang_generation.py::test_sglang_policy_tensor_parallel
    --deselect=unit/models/generation/test_sglang_generation.py::test_sglang_generate_text
    --deselect=unit/models/generation/test_sglang_generation.py::test_sglang_http_server
    --deselect=unit/models/generation/test_sglang_generation.py::test_sglang_non_divisible_batch_handling
    --deselect=unit/models/generation/test_sglang_generation.py::test_sglang_generation_with_hf_training_colocated
    --deselect=unit/models/generation/test_sglang_generation.py::test_sglang_generation_with_hf_training_non_colocated
    --deselect=unit/models/generation/test_sglang_generation.py::test_sglang_weight_update_and_prefix_cache_reset

    # test_sglang_utils.py — keep 1, exclude 2
    --deselect=unit/models/generation/test_sglang_utils.py::test_async_loop_thread_run_when_stopped_raises

    # test_vllm_utils.py — keep 4 core, exclude 3
    --deselect=unit/models/generation/test_vllm_utils.py::test_vllm_utils_vlm_with_missing_images_fallback_to_tokens
    --deselect=unit/models/generation/test_vllm_utils.py::test_vllm_utils_vlm_with_none_content_fallback_to_tokens_and_sample_idx
    --deselect=unit/models/generation/test_vllm_utils.py::test_vllm_speculative_decoding_patch_still_needed

    # test_vllm_logprobs_mode.py — keep 1 core, exclude 1
    --deselect=unit/models/generation/test_vllm_logprobs_mode.py::test_apply_top_k_top_p_matches_vllm_upstream

    ###########################################################################
    # MODELS — Policy (run in L0_Unit_Tests_Policy)
    ###########################################################################

    # Whole-file excludes for all-heavy files
    --ignore=unit/models/policy/test_dtensor_worker_v2.py
    --ignore=unit/models/policy/test_patches.py

    # test_megatron_worker.py — keep 6 correctness checks, exclude 9 heavy integration
    --deselect=unit/models/policy/test_megatron_worker.py::test_megatron_policy_training
    --deselect=unit/models/policy/test_megatron_worker.py::test_megatron_policy_generation
    --deselect=unit/models/policy/test_megatron_worker.py::test_megatron_policy_logprobs
    --deselect=unit/models/policy/test_megatron_worker.py::test_megatron_reference_policy_functionality
    --deselect=unit/models/policy/test_megatron_worker.py::test_megatron_checkpoint_save_kill_and_restore
    --deselect=unit/models/policy/test_megatron_worker.py::test_megatron_dpo_training
    --deselect=unit/models/policy/test_megatron_worker.py::test_megatron_policy_topk_logits
    --deselect=unit/models/policy/test_megatron_worker.py::test_megatron_context_parallel_topk_agreement
    --deselect=unit/models/policy/test_megatron_worker.py::test_megatron_sft_training

    # test_policy_utils.py — keep 1, exclude 7
    --deselect=unit/models/policy/test_policy_utils.py::test_setup_ipc_gather_group_returns_none_when_dist_uninit
    --deselect=unit/models/policy/test_policy_utils.py::test_setup_ipc_gather_group_selects_matching_ranks
    --deselect=unit/models/policy/test_policy_utils.py::test_gather_ipc_handlers_returns_filtered_on_src
    --deselect=unit/models/policy/test_policy_utils.py::test_gather_ipc_handlers_non_src_returns_none
    --deselect=unit/models/policy/test_policy_utils.py::test_send_tensor_to_sglang_http_error
    --deselect=unit/models/policy/test_policy_utils.py::test_send_tensor_to_sglang_generic_error
    --deselect=unit/models/policy/test_policy_utils.py::test_stream_weights_via_http_impl_no_matching_url

    ###########################################################################
    # MODELS — Other subdirectories
    ###########################################################################

    # HuggingFace — all tests require gated HF model loading
    --ignore=unit/models/huggingface/

    # Megatron — GPU-heavy data tests
    --ignore=unit/models/megatron/test_megatron_data.py

    # DTensor — keep lora tests, exclude parallelize edge cases
    --deselect=unit/models/dtensor/test_parallelize.py::test_get_grad_norm_precision
    --deselect=unit/models/dtensor/test_lora.py::test_lora_patch_on_model_without_config
    --deselect=unit/models/dtensor/test_lora.py::test_backward_pass_without_config
    --deselect=unit/models/dtensor/test_lora.py::test_apply_lora_respects_wildcard
    --deselect=unit/models/dtensor/test_lora.py::test_no_patch_on_non_matching_module
    --deselect=unit/models/dtensor/test_lora.py::test_lora_patch_with_dtype_string
    --deselect=unit/models/dtensor/test_lora.py::test_dropout_pre_post_effects

    ###########################################################################
    # ENVIRONMENTS
    ###########################################################################

    # Retriever — GPU + Ray + model loading
    --ignore=unit/environments/test_retriever.py

    # Code environment — keep untrusted_code, exclude vllm integration
    --deselect=unit/environments/test_code_environment.py::test_vllm_execute_code

    ###########################################################################
    # UTILS
    ###########################################################################

    # Native checkpoint — keep 4 core, exclude heavy DCP-to-HF conversion
    --deselect=unit/utils/test_native_checkpoint.py::test_convert_dcp_to_hf

    # Packed tensor — keep 1 core roundtrip, exclude stress variants
    --deselect=unit/utils/test_packed_tensor.py::test_packed_broadcast_single_large_tensor
    --deselect=unit/utils/test_packed_tensor.py::test_packed_broadcast_multiple_batches
)
