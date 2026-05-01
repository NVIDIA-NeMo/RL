#!/bin/bash
# Excluded unit tests for FAST CI mode (Lfast).
# Source this file and append "${EXCLUDED_UNIT_TESTS[@]}" to pytest args.
# Supports: --ignore=<path>, --ignore-glob=<pattern>, --deselect=<node_id>
# All paths are relative to tests/ (run_unit.sh cwd).
#
# Principles:
#   - Run ALL cheap tests (<1s, pure math/mocks/tensor ops)
#   - Only exclude tests that are genuinely expensive (spin up WorkerGroups,
#     Policy, VllmGeneration, SglangGeneration, load HF models, multi-GPU)
#   - For expensive test suites, keep the most comprehensive correctness
#     checks (favor A+B over just A)
#   - Config/recipe tests always run (marked @run_first)

EXCLUDED_UNIT_TESTS=(
    ###########################################################################
    # ALGORITHMS
    ###########################################################################

    # Sequence packing gradients — requires multi-GPU + Ray (~73s)
    --ignore=unit/algorithms/test_sequence_packing_gradients.py

    # test_grpo.py — exclude only tests requiring Ray actors / SGLang integration
    --deselect=tests/unit/algorithms/test_grpo.py::test_calculate_rewards_multiple_tasks
    --deselect=tests/unit/algorithms/test_grpo.py::test_calculate_rewards_missing_environment
    --deselect=tests/unit/algorithms/test_grpo.py::test_noncolocated_inference_requires_explicit_gpus_per_node_multi_node
    --deselect=tests/unit/algorithms/test_grpo.py::test_refit_policy_generation_sglang_colocated_http
    --deselect=tests/unit/algorithms/test_grpo.py::test_refit_policy_generation_sglang_non_colocated_raises

    # test_utils.py — exclude only HF gated tokenizer tests
    --deselect=tests/unit/algorithms/test_utils.py::test_get_tokenizer_no_chat_template
    --deselect=tests/unit/algorithms/test_utils.py::test_get_tokenizer_default_chat_template
    --deselect=tests/unit/algorithms/test_utils.py::test_get_tokenizer_null_chat_template
    --deselect=tests/unit/algorithms/test_utils.py::test_get_tokenizer_custom_jinja_template

    ###########################################################################
    # DISTRIBUTED
    ###########################################################################

    # test_virtual_cluster.py — Ray cluster infrastructure tests (~58s each)
    --ignore=unit/distributed/test_virtual_cluster.py

    # test_worker_groups.py — exclude 2D sharding variants (require complex Ray setup)
    --deselect=tests/unit/distributed/test_worker_groups.py::test_run_all_workers_single_data_2d_sharding_no_filter
    --deselect=tests/unit/distributed/test_worker_groups.py::test_run_all_workers_single_data_2d_sharding_filter_tp
    --deselect=tests/unit/distributed/test_worker_groups.py::test_run_all_workers_single_data_2d_sharding_filter_dp_tp
    --deselect=tests/unit/distributed/test_worker_groups.py::test_run_all_workers_sharded_data_2d_shard_dp_replicate_tp
    --deselect=tests/unit/distributed/test_worker_groups.py::test_run_all_workers_sharded_data_2d_free_axis_dp_shard_tp
    --deselect=tests/unit/distributed/test_worker_groups.py::test_run_all_workers_sharded_data_2d_free_axis_dummy_calls
    --deselect=tests/unit/distributed/test_worker_groups.py::test_run_all_workers_sharded_data_2d_output_replicated

    ###########################################################################
    # DATA
    ###########################################################################

    # test_response_dataset.py — all require HF dataset loading
    --ignore=unit/data/datasets/test_response_dataset.py

    # test_data_shuffle_reproducity.py — requires HF model downloads
    --ignore=unit/data/test_data_shuffle_reproducity.py

    # test_llm_message_utils.py — exclude only HF tokenizer/multimodal tests
    --deselect=tests/unit/data/test_llm_message_utils.py::test_get_formatted_message_log_models
    --deselect=tests/unit/data/test_llm_message_utils.py::test_get_formatted_message_log_qwen3_enable_thinking
    --deselect=tests/unit/data/test_llm_message_utils.py::test_formatted_message_log_empty_message
    --deselect=tests/unit/data/test_llm_message_utils.py::test_message_log_to_flat_messages_with_packed_images
    --deselect=tests/unit/data/test_llm_message_utils.py::test_batched_message_log_to_flat_message_with_packed_images
    --deselect=tests/unit/data/test_llm_message_utils.py::test_get_formatted_message_log_multimodal_prompt_formatting

    # test_data_processor.py — exclude HF gated tests
    --deselect=tests/unit/data/test_data_processor.py::test_math_hf_data_processor
    --deselect=tests/unit/data/test_data_processor.py::test_math_hf_data_processor_without_prompt
    --deselect=tests/unit/data/test_data_processor.py::test_eval_math_hf_data_processor

    ###########################################################################
    # MODELS — Generation (run in L0_Unit_Tests_Generation)
    ###########################################################################

    --ignore=unit/models/generation/test_vllm_large_model.py

    # test_vllm_generation.py — keep 3 key expensive tests + 5 cheap replace_prefix_tokens tests
    # Kept: test_vllm_policy_generation (basic generation correctness),
    #        test_vllm_weight_update_and_prefix_cache_reset (weight sync),
    #        test_vllm_generation_with_megatron_training (generation + megatron A+B),
    #        test_VllmAsyncGenerationWorker_replace_prefix_tokens + 4 replace_prefix_tokens (cheap utility)
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_missing_required_config_key
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_top_p_top_k_validation
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_worker_seed_behavior
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_policy_tensor_parallel
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_generate_text
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_http_server
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_weight_update_memory
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_generation_with_stop
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_non_divisible_batch_handling
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_generation_with_megatron_training_moe_model
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_megatron_weight_update_memory
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_megatron_pipeline_parallel
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_megatron_weight_update_with_packing

    # vLLM — aggressive fast suite exclusions
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_policy_generation_async
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_generation_with_hf_training_non_colocated
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_refit_non_colocated_update_weights
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_http_server_correct_merged_tokens_matches_baseline

    # vLLM weight update — exclude fp8 variants
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_weight_update_and_prefix_cache_reset[fp8-1]
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_weight_update_and_prefix_cache_reset[fp8-2]

    # vLLM megatron training — keep only 1 variant (None-bfloat16-1)
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_generation_with_megatron_training[None-bfloat16-2]
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_generation_with_megatron_training[None-fp8-1]
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_generation_with_megatron_training[None-fp8-2]
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_generation_with_megatron_training[fp8-bfloat16-1]
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_generation_with_megatron_training[fp8-bfloat16-2]
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_generation_with_megatron_training[fp8-fp8-1]
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_generation_with_megatron_training[fp8-fp8-2]

    # vLLM colocated HF — keep only 1 variant for fast (True-False-bfloat16-False)
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_generation_with_hf_training_colocated[False-True-bfloat16-False]
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_generation_with_hf_training_colocated[True-False-fp8-False]
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_generation_with_hf_training_colocated[False-True-fp8-False]
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_generation_with_hf_training_colocated[False-False-bfloat16-True]
    --deselect=tests/unit/models/generation/test_vllm_generation.py::test_vllm_generation_with_hf_training_colocated[True-False-bfloat16-True]

    # test_sglang_generation.py — keep 2 key tests
    # Kept: test_sglang_policy_generation (basic generation correctness),
    #        test_sglang_generation_with_hf_training_colocated (generation + HF training A+B)
    --deselect=tests/unit/models/generation/test_sglang_generation.py::test_sglang_missing_required_config_key
    --deselect=tests/unit/models/generation/test_sglang_generation.py::test_sglang_top_p_top_k_validation
    --deselect=tests/unit/models/generation/test_sglang_generation.py::test_sglang_worker_seed_behavior
    --deselect=tests/unit/models/generation/test_sglang_generation.py::test_sglang_policy_tensor_parallel
    --deselect=tests/unit/models/generation/test_sglang_generation.py::test_sglang_generate_text
    --deselect=tests/unit/models/generation/test_sglang_generation.py::test_sglang_http_server
    --deselect=tests/unit/models/generation/test_sglang_generation.py::test_sglang_non_divisible_batch_handling
    --deselect=tests/unit/models/generation/test_sglang_generation.py::test_sglang_generation_with_hf_training_non_colocated
    --deselect=tests/unit/models/generation/test_sglang_generation.py::test_sglang_weight_update_and_prefix_cache_reset

    # test_vllm_utils.py — exclude only the @vllm-marked test (rest are cheap)
    --deselect=tests/unit/models/generation/test_vllm_utils.py::test_vllm_speculative_decoding_patch_still_needed

    # test_vllm_logprobs_mode.py — keep test_processed_logprobs_matches_manual_computation (66s, critical)
    --deselect=tests/unit/models/generation/test_vllm_logprobs_mode.py::test_apply_top_k_top_p_matches_vllm_upstream

    ###########################################################################
    # MODELS — Policy (run in L0_Unit_Tests_Policy)
    ###########################################################################

    # test_dtensor_worker_v2.py — all heavy GPU tests (~54s each)
    --ignore=unit/models/policy/test_dtensor_worker_v2.py

    # test_patches.py — requires model loading
    --ignore=unit/models/policy/test_patches.py

    # test_dtensor_worker.py — keep 3 correctness checks, exclude rest (~52-116s each)
    # Kept: test_dtensor_single_gpu_training[True], test_dtensor_loss_independent_of_microbatch_size_two_gpus,
    #        test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded (v2 only, no llama)
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestSingleGPUCluster::test_dtensor_single_gpu_logprob
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_lm_policy_init
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_training
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_training_with_lora
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_tp_and_tied_model_with_custom_parallel_plan
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_v1_policy_flops_range_check
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_with_lora

    # DTensor — exclude v1 (use_v2=False) variants of kept tests
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestSingleGPUCluster::test_dtensor_single_gpu_training[False]

    # DTensor logprob — exclude v1 variants (all 16 setups)
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[False-setup0]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[False-setup1]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[False-setup2]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[False-setup3]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[False-setup4]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[False-setup5]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[False-setup6]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[False-setup7]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[False-setup8]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[False-setup9]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[False-setup10]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[False-setup11]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[False-setup12]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[False-setup13]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[False-setup14]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[False-setup15]

    # DTensor logprob — exclude llama variants (v2 only): setup2-4 = llama TP=2, setup11-13 = llama CP=2
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[True-setup2]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[True-setup3]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[True-setup4]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[True-setup11]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[True-setup12]
    --deselect=tests/unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded[True-setup13]

    # test_megatron_worker.py — keep 2 correctness checks (~190s total), exclude rest (~77-114s each)
    # Kept: test_megatron_loss_independent_of_microbatch_size (loss correctness),
    #        test_megatron_context_parallel_logprob_agreement (logprob + CP correctness)
    --deselect=tests/unit/models/policy/test_megatron_worker.py::test_megatron_policy_training
    --deselect=tests/unit/models/policy/test_megatron_worker.py::test_megatron_policy_generation
    --deselect=tests/unit/models/policy/test_megatron_worker.py::test_megatron_policy_logprobs
    --deselect=tests/unit/models/policy/test_megatron_worker.py::test_megatron_reference_policy_functionality
    --deselect=tests/unit/models/policy/test_megatron_worker.py::test_megatron_checkpoint_save_kill_and_restore
    --deselect=tests/unit/models/policy/test_megatron_worker.py::test_megatron_dpo_training
    --deselect=tests/unit/models/policy/test_megatron_worker.py::test_megatron_policy_topk_logits
    --deselect=tests/unit/models/policy/test_megatron_worker.py::test_megatron_context_parallel_topk_agreement
    --deselect=tests/unit/models/policy/test_megatron_worker.py::test_megatron_sft_training
    --deselect=tests/unit/models/policy/test_megatron_worker.py::test_megatron_grad_norm_invariant_to_number_of_microbatches
    --deselect=tests/unit/models/policy/test_megatron_worker.py::test_megatron_gradient_norm_consistency_across_parallelism
    --deselect=tests/unit/models/policy/test_megatron_worker.py::test_megatron_context_parallel_training_agreement

    ###########################################################################
    # MODELS — Other subdirectories
    ###########################################################################

    # HuggingFace — all tests require gated HF model loading
    --ignore=unit/models/huggingface/

    # Megatron — GPU-heavy data tests
    --ignore=unit/models/megatron/test_megatron_data.py

    # DTensor — exclude parallelize_plan_keys (loads HF model configs, ~31-33s per param)
    --deselect=tests/unit/models/dtensor/test_parallelize.py::test_parallelize_plan_keys

    ###########################################################################
    # EXPERIENCE — rollout tests all require vLLM setup (~67-91s setup each)
    ###########################################################################

    --ignore=unit/experience/test_rollouts.py

    ###########################################################################
    # ENVIRONMENTS
    ###########################################################################

    # Retriever — GPU + Ray + model loading (~175s)
    --ignore=unit/environments/test_retriever.py

    # Reward model environment — requires model loading (~75s)
    --ignore=unit/environments/test_reward_model_environment.py

    # Code environment — exclude vllm integration (~82s)
    --deselect=tests/unit/environments/test_code_environment.py::test_vllm_execute_code

    ###########################################################################
    # UTILS
    ###########################################################################

    # Native checkpoint — exclude heavy DCP-to-HF conversion (~62-114s)
    --deselect=tests/unit/utils/test_native_checkpoint.py::test_convert_dcp_to_hf

    # Packed tensor — exclude stress variants
    --deselect=tests/unit/utils/test_packed_tensor.py::test_packed_broadcast_single_large_tensor
    --deselect=tests/unit/utils/test_packed_tensor.py::test_packed_broadcast_multiple_batches
)
