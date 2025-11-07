# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from datetime import datetime

import pytest
import torch

from nemo_rl.algorithms.utils import (
    calculate_baseline_and_std_per_prompt,
    get_tokenizer,
    maybe_pad_last_batch,
    print_performance_metrics,
)
from nemo_rl.data.chat_templates import COMMON_CHAT_TEMPLATES
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@pytest.fixture
def conversation_messages():
    """Fixture providing a multi-turn conversation for testing chat templates"""
    return [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What's the weather like today?"},
        {
            "role": "assistant",
            "content": "I don't have access to real-time weather data.",
        },
        {"role": "user", "content": "Can you help me with something else then?"},
        {"role": "assistant", "content": "Of course! What would you like help with?"},
    ]


def get_expected_llama_format(messages):
    """Generate the expected output format for Llama's chat template"""
    # Extract the date from the formatted output
    # Get current date
    current_date = datetime.now()
    formatted_date = current_date.strftime("%d %b %Y")

    # Extract system message if present
    if messages[0]["role"] == "system":
        system_message = messages[0]["content"].strip()
        messages = messages[1:]
    else:
        system_message = ""

    # Start with BOS token and system header
    expected = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    expected += "Cutting Knowledge Date: December 2023\n"
    expected += f"Today Date: {formatted_date}\n\n"
    expected += f"{system_message}<|eot_id|>"

    # Add each message
    for message in messages:
        if message["role"] not in ["ipython", "tool"]:
            expected += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
            expected += f"{message['content'].strip()}<|eot_id|>"

    return expected


def get_format_with_simple_role_header(messages):
    message = "<|begin_of_text|>"
    for msg in messages:
        message += (
            "<|start_header_id|>"
            + msg["role"]
            + "<|end_header_id|>\n\n"
            + msg["content"].strip()
            + "<|eot_id|>"
        )
    return message


@pytest.mark.hf_gated
def test_get_tokenizer_no_chat_template(conversation_messages):
    """Test get_tokenizer when no chat template is specified in config"""
    config = {"name": "meta-llama/Llama-3.2-1B-Instruct"}
    tokenizer = get_tokenizer(config)

    # Verify that the tokenizer's default template is used
    formatted = tokenizer.apply_chat_template(conversation_messages, tokenize=False)

    expected = get_expected_llama_format(conversation_messages)
    assert formatted == expected


@pytest.mark.hf_gated
def test_get_tokenizer_default_chat_template(conversation_messages):
    """Test get_tokenizer when chat_template is 'default' in config"""
    config = {"name": "meta-llama/Llama-3.2-1B-Instruct", "chat_template": "default"}
    tokenizer = get_tokenizer(config)

    # Verify that the tokenizer's default template is used
    formatted = tokenizer.apply_chat_template(conversation_messages, tokenize=False)
    expected = get_expected_llama_format(conversation_messages)
    assert formatted == expected


@pytest.mark.hf_gated
def test_get_tokenizer_null_chat_template(conversation_messages):
    """Test get_tokenizer when chat_template is None in config"""
    config = {"name": "meta-llama/Llama-3.2-1B-Instruct", "chat_template": None}
    tokenizer = get_tokenizer(config)

    # Verify that the passthrough template is used
    formatted = tokenizer.apply_chat_template(conversation_messages, tokenize=False)

    expected = "".join(msg["content"] for msg in conversation_messages)

    assert formatted == expected


@pytest.mark.hf_gated
def test_get_tokenizer_custom_jinja_template(conversation_messages):
    """Test get_tokenizer when a custom jinja template is specified"""
    custom_template = COMMON_CHAT_TEMPLATES.simple_role_header
    config = {
        "name": "meta-llama/Llama-3.2-1B-Instruct",
        "chat_template": custom_template,
    }
    tokenizer = get_tokenizer(config)

    # Verify that the custom template is used
    formatted = tokenizer.apply_chat_template(conversation_messages, tokenize=False)
    expected = get_format_with_simple_role_header(conversation_messages)
    assert formatted == expected


def test_maybe_pad_last_batch():
    """Test maybe_pad_last_batch function for various scenarios"""
    # Test case 1: No padding needed
    batch_size = 8
    dp_size = 2
    mbs = 2

    batch = BatchedDataDict(
        {
            "input_ids": torch.randn(batch_size, 10),
            "input_lengths": torch.randint(1, 10, (batch_size,)),
            "sample_mask": torch.ones(batch_size),
            "token_mask": torch.ones(batch_size, 10),
            "reference_policy_logprobs": torch.randn(batch_size, 10),
        }
    )

    result = maybe_pad_last_batch(batch, dp_size, mbs)

    # Should not be padded since 8 is divisible by (2 * 2) = 4
    assert result["input_ids"].shape[0] == batch_size
    assert result["input_lengths"].shape[0] == batch_size
    assert result["sample_mask"].shape[0] == batch_size
    assert result["token_mask"].shape[0] == batch_size
    assert result["reference_policy_logprobs"].shape[0] == batch_size

    # Test case 2: Padding needed
    batch_size = 7
    dp_size = 2
    mbs = 2

    batch = BatchedDataDict(
        {
            "input_ids": torch.randn(batch_size, 10),
            "input_lengths": torch.randint(1, 10, (batch_size,)),
            "sample_mask": torch.ones(batch_size),
            "token_mask": torch.ones(batch_size, 10),
            "reference_policy_logprobs": torch.randn(batch_size, 10),
        }
    )

    result = maybe_pad_last_batch(batch, dp_size, mbs)

    # Should be padded to 8 (next multiple of 4)
    expected_size = 8
    assert result["input_ids"].shape[0] == expected_size
    assert result["input_lengths"].shape[0] == expected_size
    assert result["sample_mask"].shape[0] == expected_size
    assert result["token_mask"].shape[0] == expected_size
    assert result["reference_policy_logprobs"].shape[0] == expected_size

    # Check that sample_mask padding is zeros
    assert torch.allclose(
        result["sample_mask"][-1], torch.zeros_like(batch["sample_mask"][-1])
    )

    # Test case 3: Batch without optional fields
    batch_size = 5
    dp_size = 3
    mbs = 2

    batch = BatchedDataDict(
        {
            "input_ids": torch.randn(batch_size, 10),
            "input_lengths": torch.randint(1, 10, (batch_size,)),
            "sample_mask": torch.ones(batch_size),
        }
    )

    result = maybe_pad_last_batch(batch, dp_size, mbs)

    # Should be padded to 6 (next multiple of 3 * 2 = 6)
    expected_size = 6
    assert result["input_ids"].shape[0] == expected_size
    assert result["input_lengths"].shape[0] == expected_size
    assert result["sample_mask"].shape[0] == expected_size
    assert "token_mask" not in result
    assert "reference_policy_logprobs" not in result


# Performance Metrics Tests


def _base_master_config(colocated: bool):
    return {
        "cluster": {"num_nodes": 2, "gpus_per_node": 8},
        "policy": {
            "generation": {
                "colocated": {
                    "enabled": colocated,
                    "resources": {"num_nodes": 1, "gpus_per_node": 8},
                }
            }
        },
        "grpo": {"num_prompts_per_step": 8, "num_generations_per_prompt": 10},
    }


def test_sync_colocated_throughput_flops_and_imbalance(capsys):
    master_config = _base_master_config(colocated=True)

    timing_metrics = {
        "policy_and_reference_logprobs": 2.0,
        "policy_training": 4.0,
        "total_step_time": 10.0,
        "generation": 5.0,
        "weight_sync": 1.0,
    }

    # total_num_gpus = 2 * 8 = 16
    # samples_per_step = 8 * 10 = 80
    metrics = {
        "total_num_tokens": 8000.0,
        "per_worker_token_counts": {0: 1000, 1: 2000, 2: 3000, 3: 4000},
    }

    # total_tflops = total_flops / policy_training / 1e12 = 1e15 / 4 / 1e12 = 250
    # per-rank TFLOPS message shows 31.25 TFLOPS per rank for 8 ranks
    train_results = {
        "total_flops": 1.0e15,
        "num_ranks": 8,
        "theoretical_tflops": 500.0,
    }

    perf = print_performance_metrics(
        train_results, metrics, timing_metrics, master_config
    )

    # Validate key throughput metrics
    assert math.isclose(perf["samples_per_sec_per_gpu"], 0.5, rel_tol=1e-6)
    assert math.isclose(perf["tokens_per_sec_per_gpu"], 50.0, rel_tol=1e-6)
    assert math.isclose(
        perf["policy_training_tokens_per_sec_per_gpu"], 125.0, rel_tol=1e-6
    )
    assert math.isclose(
        perf["policy_and_reference_logprobs_tokens_per_sec_per_gpu"],
        250.0,
        rel_tol=1e-6,
    )
    assert math.isclose(
        perf["training_worker_group_tokens_per_sec_per_gpu"],
        8000.0 / 6.0 / 16.0,
        rel_tol=1e-6,
    )
    assert math.isclose(
        perf["generation_tokens_per_sec_per_gpu"], 8000.0 / 5.0 / 16.0, rel_tol=1e-6
    )

    # Group totals
    assert math.isclose(perf["samples_per_sec"], 8.0, rel_tol=1e-6)
    assert math.isclose(perf["tokens_per_sec"], 800.0, rel_tol=1e-6)
    assert math.isclose(
        perf["training_worker_group_tokens_per_sec"], 8000.0 / 6.0, rel_tol=1e-6
    )

    # Imbalance metric from ratios [0.25, 0.5, 0.75, 1.0]
    assert math.isclose(perf["average_token_imbalance"], 0.375, rel_tol=1e-6)

    # Verify selected console output snippets
    out = capsys.readouterr().out
    assert "Performance Metrics" in out
    assert "Throughputs (per GPU)" in out
    assert "Average Token Imbalance" in out
    assert "Training FLOPS" in out
    assert "Floating Point Utilization" in out


def test_async_non_colocated_idle_ratio_and_generation_time(capsys):
    master_config = _base_master_config(colocated=False)
    master_config["async_grpo"] = {"enabled": True}

    timing_metrics = {
        "policy_and_reference_logprobs": 2.0,
        "policy_training": 4.0,
        "total_step_time": 10.0,
        "exposed_generation": 2.0,
        "prepare_for_generation/total": 1.0,
    }

    # total_num_gpus = 16, training_num_gpus = 8, generation_num_gpus = 8
    metrics = {
        "total_num_tokens": 6050.0,
        "per_worker_token_counts": [{0: 3000}, {1: 3050}],
    }

    train_results = {}

    perf = print_performance_metrics(
        train_results, metrics, timing_metrics, master_config
    )

    # Throughput checks
    assert math.isclose(perf["samples_per_sec_per_gpu"], 0.5, rel_tol=1e-6)
    assert math.isclose(
        perf["tokens_per_sec_per_gpu"], 6050.0 / 10.0 / 16.0, rel_tol=1e-6
    )
    assert math.isclose(
        perf["policy_training_tokens_per_sec_per_gpu"],
        6050.0 / 4.0 / 8.0,
        rel_tol=1e-6,
    )
    assert math.isclose(
        perf["policy_and_reference_logprobs_tokens_per_sec_per_gpu"],
        6050.0 / 2.0 / 8.0,
        rel_tol=1e-6,
    )
    assert math.isclose(
        perf["training_worker_group_tokens_per_sec_per_gpu"],
        6050.0 / (4.0 + 2.0) / 8.0,
        rel_tol=1e-6,
    )
    # generation_time = 2 + 2 + 4 = 8.0, per-gpu = 6050 / 8.0 / 8.0
    assert math.isclose(
        perf["generation_tokens_per_sec_per_gpu"], 6050.0 / 8.0 / 8.0, rel_tol=1e-6
    )

    # Aggregated worker counts: {0: 3000, 1: 3050} -> imbalance = 0.05
    imbalance = ((3050 - 3000) / 3050) / 2
    assert math.isclose(perf["average_token_imbalance"], imbalance, rel_tol=1e-6)


def test_minimal_inputs_no_counts_no_flops(capsys):
    master_config = _base_master_config(colocated=False)

    timing_metrics = {
        "policy_and_reference_logprobs": 1.0,
        "policy_training": 3.0,
        "total_step_time": 8.0,
        "exposed_generation": 0.2,
        "prepare_for_generation/total": 0.5,
    }

    metrics = {
        "total_num_tokens": 1600.0,
        # no per_worker_token_counts present
    }

    train_results = {}

    perf = print_performance_metrics(
        train_results, metrics, timing_metrics, master_config
    )

    # Core metrics exist
    for k in [
        "samples_per_sec",
        "tokens_per_sec",
        "samples_per_sec_per_gpu",
        "tokens_per_sec_per_gpu",
    ]:
        assert k in perf

    out = capsys.readouterr().out
    assert "Throughputs (per GPU)" in out


# ============================================================================
# Tests for calculate_baseline_and_std_per_prompt function
# ============================================================================


def test_calculate_baseline_and_std_per_prompt_basic():
    """Test basic functionality of calculate_baseline_and_std_per_prompt."""
    # Create rewards for 2 prompts, each with 3 generations
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    prompts = torch.tensor(
        [
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
            [4, 5, 6],  # prompt 1
            [4, 5, 6],  # prompt 1
            [4, 5, 6],  # prompt 1
        ]
    )
    valid_mask = torch.ones(6)

    baseline, std = calculate_baseline_and_std_per_prompt(prompts, rewards, valid_mask)

    expected_baseline = torch.tensor([2.5, 2.0, 1.5, 5.5, 5.0, 4.5])
    expected_std = torch.tensor(
        [0.707107, 1.414214, 0.707107, 0.707107, 1.414214, 0.707107]
    )

    assert torch.allclose(baseline, expected_baseline, rtol=1e-5)
    assert torch.allclose(std, expected_std, rtol=1e-5)


def test_calculate_baseline_and_std_per_prompt_single_generation_per_prompt():
    """Test calculate_baseline_and_std_per_prompt when num_valid < 2 (single generation per prompt)."""
    # Case where each prompt has only 1 generation (num_valid = 1 < 2)
    rewards = torch.tensor([2.5, 4.0])
    prompts = torch.tensor(
        [
            [1, 2, 3],  # prompt 0
            [4, 5, 6],  # prompt 1
        ]
    )
    valid_mask = torch.ones(2)

    baseline, std = calculate_baseline_and_std_per_prompt(prompts, rewards, valid_mask)

    # When num_valid <= 1 (single generation per prompt), baseline equals reward
    expected_baseline = torch.tensor([2.5, 4.0])
    expected_std = torch.tensor([0.0, 0.0])

    assert torch.allclose(baseline, expected_baseline, rtol=1e-5)
    assert torch.allclose(std, expected_std, rtol=1e-5)


def test_calculate_baseline_and_std_per_prompt_identical_rewards():
    """Test calculate_baseline_and_std_per_prompt when all rewards for a prompt are identical."""
    # All generations for both prompts have the same reward
    rewards = torch.tensor([3.0, 3.0, 3.0, 7.0, 7.0, 7.0])
    prompts = torch.tensor(
        [
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
            [4, 5, 6],  # prompt 1
            [4, 5, 6],  # prompt 1
            [4, 5, 6],  # prompt 1
        ]
    )
    valid_mask = torch.ones(6)

    baseline, std = calculate_baseline_and_std_per_prompt(prompts, rewards, valid_mask)

    expected_baseline = torch.tensor([3.0, 3.0, 3.0, 7.0, 7.0, 7.0])
    expected_std = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert torch.allclose(baseline, expected_baseline, rtol=1e-5)
    assert torch.allclose(std, expected_std, rtol=1e-5)


def test_calculate_baseline_and_std_per_prompt_mixed_prompt_sizes():
    """Test calculate_baseline_and_std_per_prompt with different number of generations per prompt."""
    # Prompt 0 has 2 generations, Prompt 1 has 3 generations
    rewards = torch.tensor([1.0, 2.0, 4.0, 5.0, 6.0])
    prompts = torch.tensor(
        [
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
            [4, 5, 6],  # prompt 1
            [4, 5, 6],  # prompt 1
            [4, 5, 6],  # prompt 1
        ]
    )
    valid_mask = torch.ones(5)

    baseline, std = calculate_baseline_and_std_per_prompt(prompts, rewards, valid_mask)

    expected_baseline = torch.tensor([2.0, 1.0, 5.5, 5.0, 4.5])
    expected_std = torch.tensor([0.0, 0.0, 0.707107, 1.414214, 0.707107])

    assert torch.allclose(baseline, expected_baseline, rtol=1e-5)
    assert torch.allclose(std, expected_std, rtol=1e-5)


def test_calculate_baseline_and_std_per_prompt_empty_input():
    """Test calculate_baseline_and_std_per_prompt with empty tensors."""
    rewards = torch.tensor([])
    prompts = torch.empty(0, 3, dtype=torch.long)
    valid_mask = torch.tensor([])

    baseline, std = calculate_baseline_and_std_per_prompt(prompts, rewards, valid_mask)

    assert baseline.shape == torch.Size([0])
    assert std.shape == torch.Size([0])
    assert torch.equal(baseline, torch.tensor([]))
    assert torch.equal(std, torch.tensor([]))


def test_calculate_baseline_and_std_per_prompt_nan_handling():
    """Test calculate_baseline_and_std_per_prompt handles valid_mask correctly with masked samples."""
    # Test that valid_mask properly excludes samples from baseline calculation
    # Note: The function doesn't handle actual NaN values; it uses valid_mask to exclude samples
    rewards = torch.tensor([1.0, 999.0, 3.0, 4.0, 5.0, 6.0])  # 999.0 should be ignored
    prompts = torch.tensor(
        [
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0 (invalid sample)
            [1, 2, 3],  # prompt 0
            [4, 5, 6],  # prompt 1
            [4, 5, 6],  # prompt 1
            [4, 5, 6],  # prompt 1
        ]
    )
    # Mark the second sample as invalid
    valid_mask = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    baseline, std = calculate_baseline_and_std_per_prompt(prompts, rewards, valid_mask)

    expected_baseline = torch.tensor([3.0, 4.0, 1.0, 5.5, 5.0, 4.5])
    expected_std = torch.tensor([0.0, 0.0, 0.0, 0.707107, 1.414214, 0.707107])

    assert torch.allclose(baseline, expected_baseline, rtol=1e-5)
    assert torch.allclose(std, expected_std, rtol=1e-5)


def test_calculate_baseline_and_std_per_prompt_cuda_compatibility():
    """Test calculate_baseline_and_std_per_prompt works with CUDA tensors if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0]).cuda()
    prompts = torch.tensor(
        [
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
            [4, 5, 6],  # prompt 1
            [4, 5, 6],  # prompt 1
        ]
    ).cuda()
    valid_mask = torch.ones(4).cuda()

    baseline, std = calculate_baseline_and_std_per_prompt(prompts, rewards, valid_mask)

    # Verify results are on CUDA and have expected values
    assert baseline.device.type == "cuda"
    assert std.device.type == "cuda"

    expected_baseline = torch.tensor([2.0, 1.0, 4.0, 3.0]).cuda()
    expected_std = torch.tensor([0.0, 0.0, 0.0, 0.0]).cuda()

    assert torch.allclose(baseline, expected_baseline, rtol=1e-5)
    assert torch.allclose(std, expected_std, rtol=1e-5)


def test_calculate_baseline_and_std_per_prompt_numerical_precision():
    """Test calculate_baseline_and_std_per_prompt with edge case numerical values."""
    # Use very small and very large values
    rewards = torch.tensor([1e-8, 2e-8, 3e-8, 1e8, 2e8, 3e8])
    prompts = torch.tensor(
        [
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
            [4, 5, 6],  # prompt 1
            [4, 5, 6],  # prompt 1
            [4, 5, 6],  # prompt 1
        ]
    )
    valid_mask = torch.ones(6)

    baseline, std = calculate_baseline_and_std_per_prompt(prompts, rewards, valid_mask)

    expected_baseline = torch.tensor([2.5e-8, 2e-8, 1.5e-8, 2.5e8, 2e8, 1.5e8])

    assert torch.allclose(baseline, expected_baseline, rtol=1e-5)
    # Std values should be finite and not NaN
    assert torch.isfinite(std).all()
    assert not torch.isnan(std).any()
