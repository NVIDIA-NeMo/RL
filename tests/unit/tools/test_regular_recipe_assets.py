# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.utils.config import register_omegaconf_resolvers


def test_local_deepseek_uses_native_reasoning_effort_name():
    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(
        root / "training_configs/super_rl/local_deepseek_v4_flash.yaml"
    )
    judge = config.deepseek_v4_flash_judge_model.responses_api_models.vllm_model
    assert judge.chat_template_kwargs == {
        "thinking": True,
        "reasoning_effort": "high",
    }


def test_equivalence_template_uses_gym_component_working_directory():
    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(
        root / "training_configs/super_rl/experiments/regular_s120_smoke.yaml"
    )
    resource = config.env.nemo_gym.equivalence_llm_judge.resources_servers.equivalence_llm_judge
    assert (
        resource.judge_prompt_template_fpath
        == "prompt_templates/equivalence_llm_judge.txt"
    )


def test_short_smoke_preserves_ten_update_warmup():
    register_omegaconf_resolvers()
    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(
        root / "training_configs/super_rl/experiments/regular_s120_smoke.yaml"
    )
    scheduler = config.policy.megatron_cfg.scheduler
    assert config.grpo.max_num_steps == 3
    assert scheduler.lr_warmup_iters == 10
    assert scheduler.lr_decay_iters == 11
    assert scheduler.lr_decay_style == "constant"


def test_regular_smoke_does_not_reset_failed_stream_budget_through_gap_fill():
    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(
        root / "training_configs/super_rl/experiments/regular_s120_smoke.yaml"
    )
    async_config = config.grpo.async_grpo
    assert async_config.nemo_gym_stream_retries == 1
    assert async_config.nemo_gym_fail_on_retry_exhaustion is True
    assert async_config.max_generation_failures == 0
    for resource in ("math_with_judge", "equivalence_llm_judge"):
        verifier = config.env.nemo_gym[resource].resources_servers[resource]
        assert verifier.fail_on_missing_judge_verdict is True
        assert verifier.judge_max_attempts == 4
        assert verifier.judge_responses_create_params.max_output_tokens == 8192


def test_regular_smoke_aligns_context_and_policy_serving_with_gold():
    register_omegaconf_resolvers()
    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(
        root / "training_configs/super_rl/experiments/regular_s120_smoke.yaml"
    )
    policy = config.policy
    assert policy.max_total_sequence_length == 131072
    assert policy.generation.vllm_cfg.max_model_len == 131072
    for batching in (policy.sequence_packing, policy.dynamic_batching):
        assert batching.train_mb_tokens == 131072
        assert batching.logprob_mb_tokens == 131072
    assert policy.generation.vllm_kwargs.max_num_seqs == 256
    assert policy.generation.vllm_kwargs.max_num_batched_tokens == 32768
    # Policy output and per-agent cumulative output budgets must not shrink.
    assert policy.generation.max_new_tokens == 102400
    components = OmegaConf.to_container(config.env.nemo_gym, resolve=False)
    for component in components.values():
        if isinstance(component, dict) and "responses_api_agents" in component:
            for agent in component["responses_api_agents"].values():
                assert agent["max_total_output_tokens"] == 102400
    assert (
        config.env.nemo_gym.policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking
        is True
    )
    # Changing policy throughput must not also increase judge token budgets.
    for resource in ("math_with_judge", "equivalence_llm_judge"):
        verifier = config.env.nemo_gym[resource].resources_servers[resource]
        assert verifier.judge_responses_create_params.max_output_tokens == 8192


def test_regular_smoke_enables_all_gold_output_penalties():
    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(
        root / "training_configs/super_rl/experiments/regular_s120_smoke.yaml"
    )
    penalties = config.reward_penalties
    assert penalties.penalize_duplicated_reasoning is True
    assert penalties.penalize_empty_final_answer is True
    assert penalties.penalize_unwanted_tokens is True
    assert penalties.penalize_malformed_think_tag is True
    assert penalties.token_ids.unwanted == [2]
    assert penalties.token_ids.think_open == 12
    assert penalties.token_ids.think_close == 13
