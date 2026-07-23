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

import asyncio
import gc
import json
import tempfile
from copy import deepcopy
from dataclasses import asdict

import pytest
import ray
import torch
from transformers import AutoTokenizer

import nemo_rl.experience.rollouts as rollouts_mod
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets.response_datasets import NemoGymDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.data.processors import nemo_gym_data_processor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.environments.games.sliding_puzzle import (
    SlidingPuzzleConfig,
    SlidingPuzzleEnv,
    SlidingPuzzleGameLogic,
    SlidingPuzzleMetadata,
)
from nemo_rl.experience.metric_utils import calculate_single_metric, pct
from nemo_rl.experience.rollout_manager import AsyncNemoGymRolloutImpl
from nemo_rl.experience.rollouts import (
    generate_responses_async,
    run_async_multi_turn_rollout,
    run_async_multi_turn_rollout_groups,
    run_async_nemo_gym_rollout,
    run_multi_turn_rollout,
    run_nemo_gym_rollout_sync,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration

# These are all fixtures
from tests.unit.environments.test_nemo_gym import (
    cluster,  # noqa: F401
    nemo_gym,  # noqa: F401
    nemo_gym_sanity_test_data,  # noqa: F401
    nemo_gym_tokenizer,  # noqa: F401
    nemo_gym_vllm_generation,  # noqa: F401
)

# Import the test environment definitions
from tests.unit.test_envs import (
    MultiStepCalcMetadata,
    MultiStepCalculatorEnv,
    _MultiStepCalculatorLogic,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


class TestCalculateSingleMetric:
    """Unit tests for calculate_single_metric function."""

    def test_single_value_returns_nan_for_stddev(self):
        """Test that stddev returns nan when given a single value (GitHub issue #1411)."""
        import math

        result = calculate_single_metric([42.0], batch_size=1, key_name="test")

        assert result["test/mean"] == 42.0
        assert result["test/max"] == 42.0
        assert result["test/min"] == 42.0
        assert result["test/median"] == 42.0
        assert math.isnan(result["test/stddev"]), (
            "stddev should be nan for single value"
        )

    def test_multiple_values_computes_stddev(self):
        """Test that stddev is computed correctly for multiple values."""
        result = calculate_single_metric([1.0, 2.0, 3.0], batch_size=3, key_name="test")

        assert result["test/mean"] == 2.0
        assert result["test/max"] == 3.0
        assert result["test/min"] == 1.0
        assert result["test/median"] == 2.0
        assert abs(result["test/stddev"] - 1.0) < 1e-9  # stdev of [1,2,3] is 1.0

    def test_two_identical_values_returns_zero_stddev(self):
        """Test that stddev is 0 when all values are identical."""
        result = calculate_single_metric([5.0, 5.0], batch_size=2, key_name="test")

        assert result["test/stddev"] == 0.0


class TestPct:
    """Unit tests for pct percentile helper."""

    def test_empty_returns_zero(self):
        """Test that an empty input short-circuits to 0.0."""
        assert pct([], 95) == 0.0

    def test_returns_float_for_int_input(self):
        """Test that integer input is coerced to float on return."""
        result = pct([3, 1, 2], 50)
        assert isinstance(result, float)
        assert result == 2.0

    def test_sorts_unsorted_input(self):
        """Test that pct sorts before indexing."""
        assert pct([5, 1, 3], 95) == 5.0

    def test_p95_small_list_clamps_to_max(self):
        """Test that p95 on a 5-element list clamps to the last index."""
        assert pct([1, 2, 3, 4, 5], 95) == 5.0

    def test_p99_clamps_to_last_index(self):
        """Test that p99 on a 5-element list clamps to the last index."""
        assert pct([1, 2, 3, 4, 5], 99) == 5.0

    def test_single_value(self):
        """Test that a single-element input returns that element."""
        assert pct([42], 95) == 42.0

    def test_median_like_p50(self):
        """Test that p50 lands on the upper-mid element (int truncation, no interpolation)."""
        assert pct([10, 20, 30, 40], 50) == 30.0


class _DummyTokenizer:
    pad_token_id = 0

    def batch_decode(self, generated_ids, skip_special_tokens=True):
        return ["ok" for _ in generated_ids]


class _DummySGLangGeneration:
    def __init__(self, use_async_rollouts=False):
        self.sglang_cfg = {
            "backend": "sglang",
            "use_async_rollouts": use_async_rollouts,
        }

    async def generate_async(self, data, greedy=False):
        yield (
            0,
            BatchedDataDict(
                {
                    "output_ids": torch.tensor([[1, 2]]),
                    "logprobs": torch.zeros((1, 2), dtype=torch.float32),
                    "generation_lengths": torch.tensor([1], dtype=torch.long),
                    "unpadded_sequence_lengths": torch.tensor([2], dtype=torch.long),
                    "truncated": torch.tensor([False], dtype=torch.bool),
                }
            ),
        )


def test_generate_responses_async_requires_sglang_opt_in():
    generation_input_data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1]]),
            "input_lengths": torch.tensor([1], dtype=torch.long),
        }
    )
    batch = BatchedDataDict({"message_log": [[]]})

    with pytest.raises(AssertionError, match="use_async_rollouts"):
        asyncio.run(
            generate_responses_async(
                _DummySGLangGeneration(use_async_rollouts=False),
                generation_input_data,
                batch,
                _DummyTokenizer(),
                input_lengths=generation_input_data["input_lengths"],
            )
        )


def test_generate_responses_async_allows_sglang_opt_in():
    generation_input_data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1]]),
            "input_lengths": torch.tensor([1], dtype=torch.long),
        }
    )
    batch = BatchedDataDict({"message_log": [[]]})

    updated_batch, generated_ids, gen_metrics = asyncio.run(
        generate_responses_async(
            _DummySGLangGeneration(use_async_rollouts=True),
            generation_input_data,
            batch,
            _DummyTokenizer(),
            input_lengths=generation_input_data["input_lengths"],
        )
    )

    assert updated_batch["message_log"][0][-1]["content"] == "ok"
    assert generated_ids[0].tolist() == [2]
    assert gen_metrics["total_generated_tokens"] == 1


@pytest.fixture(scope="function")
def rollout_tokenizer():
    """Loads the tokenizer for the tests."""
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(
        f"Tokenizer loaded. Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id}), EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})"
    )
    return tokenizer


# Separate fixture for cluster setup and teardown
@pytest.fixture(scope="function")
def rollout_cluster():
    cluster_instance = None
    cluster_name = f"test-rollout-cluster-{id(cluster_instance)}"  # Unique name
    print(f"\nCreating virtual cluster '{cluster_name}'...")
    try:
        # Use 1 GPU for simplicity
        cluster_instance = RayVirtualCluster(
            name=cluster_name,
            bundle_ct_per_node_list=[2],
            use_gpus=True,
            num_gpus_per_node=2,
            max_colocated_worker_groups=2,  # Allow policy and env
        )
        yield cluster_instance
    finally:
        print(f"\nCleaning up cluster '{cluster_name}'...")
        if cluster_instance:
            cluster_instance.shutdown()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Cluster '{cluster_name}' cleanup finished.")


# Fixture for the multi-step calculator environment actor
@pytest.fixture(scope="function")
def multi_step_calculator_environment(rollout_cluster):
    env_actor = None
    print("Creating MultiStepCalculatorEnv actor...")
    try:
        env_actor = MultiStepCalculatorEnv.remote()
        task_to_env = {"multi_step_calculator_game": env_actor}
        yield task_to_env, env_actor
    finally:
        print("Cleaning up multi_step_calculator_environment...")
        if env_actor:
            ray.kill(env_actor)
        print("multi_step_calculator_environment cleanup finished.")


# Fixture for the multi-step calculator initial batch data
@pytest.fixture(scope="function")
def initial_multi_step_calculator_batch(rollout_tokenizer):
    print("Creating initial multi-step calculator test batch...")

    problems = [
        {"problem": "(5 + 3) * 2", "answer": 16.0},
        {"problem": "(1 * 9) + 2", "answer": 11.0},
    ]
    batch_size = len(problems)
    max_steps = 5  # Allow a few steps

    batch_message_logs = []
    batch_extra_env_info = []
    batch_loss_multipliers = []
    batch_indices = []
    batch_task_names = []

    for i, p_info in enumerate(problems):
        problem_text = p_info["problem"]
        expected_answer = p_info["answer"]

        # tool_instructions = tool_instructions_template.format(problem=problem_text)
        tool_instructions = (
            "You have a calculator tool. To use it, respond with:\n"
            "'[operand1, operand2, operation_name]<call: calculator>'\n"
            "The valid 'operation_name' values are exactly: 'sum', 'diff', 'prod', 'div'.\n"
            "Example: [5, 3, sum]<call: calculator>\n"
            "You will receive the result of your calculation as <result>...</result>\n"
            "Use this result to make the next calculation if needed.\n"
            "IMPORTANT: Only perform one calculation step (one tool call) before waiting for a result and making a new tool call.\n"
            "IMPORTANT: Do not perform any other calculations or operations aside from the tool call and result. Doing so will result in failure.\n"
            "To give the final answer, just output the number. numbers inside of <result> don't count, so output just the final number yourself outside of this.\n"
            "Example full output: [2, 4, sum]<call: calculator>\n<result>6.0</result>\n[6, 6, diff]<call: calculator>\n<result>0.0</result> 0\n(note how you have to output the final 0 outside of the tags)"
            "------\n"
            f"Solve: {problem_text}"
        )

        # Apply chat template to the initial prompt
        initial_prompt_content = rollout_tokenizer.apply_chat_template(
            [{"role": "user", "content": tool_instructions}],
            tokenize=False,
            add_system_prompt=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )
        tokenized_prompt = rollout_tokenizer(
            initial_prompt_content, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        message_log = [
            {
                "role": "user",
                "content": initial_prompt_content,
                "token_ids": tokenized_prompt,
            }
        ]
        metadata = MultiStepCalcMetadata(
            problem=problem_text,
            expected_final_answer=expected_answer,
            max_steps=max_steps,
            current_step=0,
        )

        batch_message_logs.append(message_log)
        batch_extra_env_info.append(metadata)
        batch_loss_multipliers.append(1.0)
        batch_indices.append(i)
        batch_task_names.append("multi_step_calculator_game")

    initial_batch_dict = {
        "message_log": batch_message_logs,
        "extra_env_info": batch_extra_env_info,
        "loss_multiplier": batch_loss_multipliers,
        "idx": batch_indices,
        "task_name": batch_task_names,
        "stop_strings": [["<call: calculator>"]] * batch_size,
    }
    return BatchedDataDict(initial_batch_dict)


base_vllm_test_config: VllmConfig = {
    "backend": "vllm",
    "model_name": MODEL_NAME,
    "tokenizer_name": None,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.6,
    "max_new_tokens": 50,  # Increased for tool call format
    "temperature": 0.01,  # Near-greedy
    "top_p": 1.0,
    "top_k": None,
    "stop_token_ids": None,
    "stop_strings": None,
    "vllm_cfg": {
        "async_engine": False,
        "precision": "bfloat16",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 1,
        "max_model_len": 2048,
        "disable_log_stats": True,
        "disable_log_requests": True,
        "gpu_memory_utilization": 0.6,
        "enforce_eager": "False",
    },
    "colocated": {
        "enabled": True,
        "resources": {
            "gpus_per_node": None,
            "num_nodes": None,
        },
    },
}


@pytest.fixture(scope="function")
def multi_step_setup_vllm_sync(
    rollout_cluster,
    rollout_tokenizer,
    multi_step_calculator_environment,
    initial_multi_step_calculator_batch,
):
    """Sets up components for multi-step calculator tests using VllmGeneration with sync engine."""
    vllm_generation = None
    task_to_env, _ = multi_step_calculator_environment
    is_eval = True
    print("Creating VllmGeneration with sync engine for Multi-Step Calculator Test...")
    try:
        vllm_config = deepcopy(base_vllm_test_config)
        vllm_config["tokenizer_name"] = rollout_tokenizer.name_or_path
        if "gpt2" in rollout_tokenizer.name_or_path.lower():
            vllm_config["model_name"] = "gpt2"
        vllm_config = configure_generation_config(
            vllm_config, rollout_tokenizer, is_eval=is_eval
        )
        vllm_generation = VllmGeneration(rollout_cluster, vllm_config)
        vllm_generation.finish_generation()
        yield (
            vllm_generation,
            rollout_tokenizer,
            task_to_env,
            initial_multi_step_calculator_batch,
            rollout_cluster,
        )
    finally:
        print("Cleaning up VllmGeneration (sync engine, Multi-Step Calc Test)...")
        if vllm_generation:
            vllm_generation.shutdown()
        # Force garbage collection to help release resources
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        print("VllmGeneration cleanup finished (sync engine, Multi-Step Calc Test).")


@pytest.fixture(scope="function")
def multi_step_setup_vllm_async(
    rollout_cluster,
    rollout_tokenizer,
    multi_step_calculator_environment,
    initial_multi_step_calculator_batch,
):
    """Sets up components for multi-step calculator tests using VllmGeneration with async engine."""
    vllm_generation = None
    task_to_env, _ = multi_step_calculator_environment
    is_eval = True
    print("Creating VllmGeneration with async engine for Multi-Step Calculator Test...")
    try:
        vllm_config = deepcopy(base_vllm_test_config)
        vllm_config["vllm_cfg"]["async_engine"] = True
        vllm_config["tokenizer_name"] = rollout_tokenizer.name_or_path
        if "gpt2" in rollout_tokenizer.name_or_path.lower():
            vllm_config["model_name"] = "gpt2"
        vllm_config = configure_generation_config(
            vllm_config, rollout_tokenizer, is_eval=is_eval
        )
        vllm_generation = VllmGeneration(rollout_cluster, vllm_config)
        vllm_generation.finish_generation()
        yield (
            vllm_generation,
            rollout_tokenizer,
            task_to_env,
            initial_multi_step_calculator_batch,
            rollout_cluster,
        )
    finally:
        print("Cleaning up VllmGeneration (async engine, Multi-Step Calc Test)...")
        if vllm_generation:
            vllm_generation.shutdown()
        # Force garbage collection to help release resources
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        print("VllmGeneration cleanup finished (async engine, Multi-Step Calc Test).")


@pytest.mark.vllm
def test_run_multi_step_calculator_vllm_sync(multi_step_setup_vllm_sync):
    """Tests multi-step calculator rollout with VllmGeneration using sync generation and sync rollout."""
    vllm_generation, rollout_tokenizer, task_to_env, initial_batch, rollout_cluster = (
        multi_step_setup_vllm_sync
    )
    max_rollout_turns = initial_batch["extra_env_info"][0]["max_steps"] + 1
    max_seq_len = 1024

    print("\nRunning sync rollout with sync generation engine (VLLM)...")
    vllm_generation.prepare_for_generation()

    final_batch, rollout_metrics = run_multi_turn_rollout(
        policy_generation=vllm_generation,
        input_batch=initial_batch,
        tokenizer=rollout_tokenizer,
        task_to_env=task_to_env,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
    )

    vllm_generation.finish_generation()
    print("Sync rollout with sync generation engine complete (VLLM).")

    # --- Assertions ---
    assert isinstance(final_batch, BatchedDataDict)
    assert "message_log" in final_batch
    assert "total_reward" in final_batch
    assert len(final_batch["message_log"]) == len(initial_batch["message_log"])

    for i in range(len(final_batch["message_log"])):
        sample_log = final_batch["message_log"][i]
        expected_final_answer = initial_batch["extra_env_info"][i][
            "expected_final_answer"
        ]
        problem_text = initial_batch["extra_env_info"][i]["problem"]

        print(f"\n--- Verifying Sync Sample {i} (Problem: {problem_text}) ---")
        print(f"Expected Answer: {expected_final_answer}")

        tool_call_count = 0
        final_answer_msg = None
        for msg_idx, msg in enumerate(sample_log):
            print(f"  {msg_idx}: Role={msg['role']}, Content='{msg['content']}'")
            if msg["role"] == "assistant":
                if msg["content"].strip().endswith("<call: calculator>"):
                    tool_call_count += 1
                else:
                    final_answer_msg = msg["content"].strip()

        assert tool_call_count >= 1, f"Sync Sample {i}: Expected at least one tool call"
        print(
            f"✓ Sample {i}: Successfully made {tool_call_count} tool call(s) using sync rollout"
        )

        # Always require a valid final answer
        assert final_answer_msg is not None and final_answer_msg.strip(), (
            f"Sync Sample {i}: Expected a final answer message from assistant"
        )

        # Always require the final answer to be parseable and correct
        final_answer_logic = _MultiStepCalculatorLogic()
        extracted_final_answer = final_answer_logic._is_final_answer(final_answer_msg)
        assert extracted_final_answer is not None, (
            f"Sync Sample {i}: Could not parse final answer from: {final_answer_msg}"
        )
        assert abs(extracted_final_answer - expected_final_answer) < 1e-6, (
            f"Sync Sample {i}: Final answer incorrect. Expected {expected_final_answer}, Got {extracted_final_answer}"
        )

        # Check total reward (should be 1.0 if correct)
        assert final_batch["total_reward"][i] == 1.0, (
            f"Sync Sample {i}: Expected total reward 1.0 for correct answer, "
            f"got {final_batch['total_reward'][i]}"
        )
        print(f"✓ Sample {i}: Correct answer {extracted_final_answer}")
        print(f"--- Sync Sample {i}: Rollout verification PASSED ---")

    print("\nSync Multi-Step Calculator VLLM Test assertions passed for all samples.")


@pytest.mark.vllm
def test_run_multi_step_calculator_vllm_async(multi_step_setup_vllm_async):
    """Tests multi-step calculator rollout with VllmGeneration using async generation and async rollout."""
    vllm_generation, rollout_tokenizer, task_to_env, initial_batch, rollout_cluster = (
        multi_step_setup_vllm_async
    )
    max_rollout_turns = initial_batch["extra_env_info"][0]["max_steps"] + 1
    max_seq_len = 1024

    print("\nRunning async rollout with async generation engine (VLLM)...")
    vllm_generation.prepare_for_generation()

    final_batch, rollout_metrics = run_async_multi_turn_rollout(
        policy_generation=vllm_generation,
        input_batch=initial_batch,
        tokenizer=rollout_tokenizer,
        task_to_env=task_to_env,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
    )

    vllm_generation.finish_generation()
    print("Async rollout with async generation engine complete (VLLM).")

    # --- Assertions ---
    assert isinstance(final_batch, BatchedDataDict)
    assert "message_log" in final_batch
    assert "total_reward" in final_batch
    assert len(final_batch["message_log"]) == len(initial_batch["message_log"])

    for i in range(len(final_batch["message_log"])):
        sample_log = final_batch["message_log"][i]
        expected_final_answer = initial_batch["extra_env_info"][i][
            "expected_final_answer"
        ]
        problem_text = initial_batch["extra_env_info"][i]["problem"]

        print(f"\n--- Verifying Async Sample {i} (Problem: {problem_text}) ---")
        print(f"Expected Answer: {expected_final_answer}")

        tool_call_count = 0
        final_answer_msg = None
        for msg_idx, msg in enumerate(sample_log):
            print(f"  {msg_idx}: Role={msg['role']}, Content='{msg['content']}'")
            if msg["role"] == "assistant":
                if msg["content"].strip().endswith("<call: calculator>"):
                    tool_call_count += 1
                else:
                    final_answer_msg = msg["content"].strip()

        assert tool_call_count >= 1, (
            f"Async Sample {i}: Expected at least one tool call"
        )
        print(
            f"✓ Sample {i}: Successfully made {tool_call_count} tool call(s) using async rollout"
        )

        # Always require a valid final answer
        assert final_answer_msg is not None and final_answer_msg.strip(), (
            f"Async Sample {i}: Expected a final answer message from assistant"
        )

        # Always require the final answer to be parseable and correct
        final_answer_logic = _MultiStepCalculatorLogic()
        extracted_final_answer = final_answer_logic._is_final_answer(final_answer_msg)
        assert extracted_final_answer is not None, (
            f"Async Sample {i}: Could not parse final answer from: {final_answer_msg}"
        )
        assert abs(extracted_final_answer - expected_final_answer) < 1e-6, (
            f"Async Sample {i}: Final answer incorrect. Expected {expected_final_answer}, Got {extracted_final_answer}"
        )

        # Check total reward (should be 1.0 if correct)
        assert final_batch["total_reward"][i] == 1.0, (
            f"Async Sample {i}: Expected total reward 1.0 for correct answer, "
            f"got {final_batch['total_reward'][i]}"
        )
        print(f"✓ Sample {i}: Correct answer {extracted_final_answer}")
        print(f"--- Async Sample {i}: Rollout verification PASSED ---")

    print("\nAsync Multi-Step Calculator VLLM Test assertions passed for all samples.")


@pytest.mark.vllm
def test_max_seqlen_respected_sync(multi_step_setup_vllm_sync):
    """Tests multi-step calculator rollout with VllmGeneration (sync)."""
    vllm_generation, rollout_tokenizer, task_to_env, initial_batch, rollout_cluster = (
        multi_step_setup_vllm_sync
    )
    max_rollout_turns = initial_batch["extra_env_info"][0]["max_steps"] + 1
    max_seq_len = 290

    print("\nRunning multi-step calculator rollout (VLLM sync)...")
    vllm_generation.prepare_for_generation()
    final_batch, rollout_metrics = run_multi_turn_rollout(
        policy_generation=vllm_generation,
        input_batch=initial_batch,
        tokenizer=rollout_tokenizer,
        task_to_env=task_to_env,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
    )
    vllm_generation.finish_generation()
    print("Multi-step calculator rollout complete (VLLM sync).")

    # --- Assertions ---
    assert isinstance(final_batch, BatchedDataDict)
    assert "message_log" in final_batch
    assert "total_reward" in final_batch
    assert len(final_batch["message_log"]) == len(initial_batch["message_log"])
    flattened_message_log, _ = batched_message_log_to_flat_message(
        final_batch["message_log"]
    )
    # Check that the sequence length is respected by flattening the message log and checking the length
    assert len(flattened_message_log["token_ids"][0]) == max_seq_len, (
        f"Sequence length {len(flattened_message_log['token_ids'][0])} is not equal to max_seq_len {max_seq_len}"
    )


@pytest.mark.vllm
def test_max_seqlen_respected_async(multi_step_setup_vllm_async):
    """Tests multi-step calculator rollout with VllmGeneration (async)."""
    vllm_generation, rollout_tokenizer, task_to_env, initial_batch, rollout_cluster = (
        multi_step_setup_vllm_async
    )
    max_rollout_turns = initial_batch["extra_env_info"][0]["max_steps"] + 1
    max_seq_len = 290

    print("\nRunning multi-step calculator rollout (VLLM async)...")
    vllm_generation.prepare_for_generation()
    final_batch, rollout_metrics = run_async_multi_turn_rollout(
        policy_generation=vllm_generation,
        input_batch=initial_batch,
        tokenizer=rollout_tokenizer,
        task_to_env=task_to_env,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
    )
    vllm_generation.finish_generation()
    print("Multi-step calculator rollout complete (VLLM async).")

    # --- Assertions ---
    assert isinstance(final_batch, BatchedDataDict)
    assert "message_log" in final_batch
    assert "total_reward" in final_batch
    assert len(final_batch["message_log"]) == len(initial_batch["message_log"])
    flattened_message_log, _ = batched_message_log_to_flat_message(
        final_batch["message_log"]
    )
    # Check that the sequence length is respected by flattening the message log and checking the length
    assert len(flattened_message_log["token_ids"][0]) == max_seq_len, (
        f"Sequence length {len(flattened_message_log['token_ids'][0])} is not equal to max_seq_len {max_seq_len}"
    )


# --- Fixture for Sliding Puzzle Environment ---
@pytest.fixture(scope="function")
def sliding_puzzle_environment(rollout_cluster):
    env_actor = None
    print("Creating SlidingPuzzleEnv actor...")
    try:
        # Pass game config if needed, e.g., {"game_config": {"size": 3}}
        env_actor = SlidingPuzzleEnv.remote()
        task_to_env = {"sliding_puzzle_game": env_actor}
        yield task_to_env, env_actor
    finally:
        print("Cleaning up sliding_puzzle_environment...")
        if env_actor:
            env_actor.shutdown.remote()
            ray.kill(env_actor)
        print("sliding_puzzle_environment cleanup finished.")


# --- Fixture for Sliding Puzzle Initial Batch ---
@pytest.fixture(scope="function")
def initial_sliding_puzzle_batch(rollout_tokenizer):
    print("Creating initial sliding puzzle test batch...")
    batch_size = 1
    game_config: SlidingPuzzleConfig = {
        "size": 2,
        "shuffle_moves": 1,
    }
    max_moves = 10  # Set a limit for the test

    # Generate initial game state
    initial_game_state = SlidingPuzzleGameLogic.generate(game_config)
    initial_render = SlidingPuzzleGameLogic.render(initial_game_state)
    welcome_message = SlidingPuzzleGameLogic.init(initial_game_state)

    prompt_instructions = (
        f"{welcome_message}\n\n"
        f"Current Board State:\n{initial_render}\n\n"
        f"Reach the goal state where numbers are ordered 1 through {game_config['size'] ** 2 - 1} "
        f"with the empty space (0) at the bottom right.\n"
        f"Valid actions: 'up', 'down', 'left', 'right'\n"
        f"After thinking, output your chosen action on a new line starting with '<action></action>' like this:\n<action>your_action</action>"
        f"\nIf you just want to see the board, output <action>view</action>"
        f"\nThink carefully step-by-step before acting. If you get a 'cannot slide' error, try something different\n"
    )

    batch_message_logs = []
    batch_extra_env_info = []
    batch_loss_multipliers = []
    batch_indices = []
    batch_task_names = []

    for i in range(batch_size):
        # Apply chat template to the initial prompt
        initial_prompt_content = rollout_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_instructions}],
            tokenize=False,
            add_system_prompt=True,  # Include system prompt for Qwen
            add_generation_prompt=True,
            add_special_tokens=False,
        ).strip()
        tokenized_prompt = rollout_tokenizer(
            initial_prompt_content, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        message_log = [
            {
                "role": "user",
                "content": initial_prompt_content,
                "token_ids": tokenized_prompt,
            }
        ]

        metadata = SlidingPuzzleMetadata(
            game_state=initial_game_state, num_moves=0, max_moves=max_moves
        )

        batch_message_logs.append(message_log)
        batch_extra_env_info.append(metadata)
        batch_loss_multipliers.append(1.0)
        batch_indices.append(i)
        batch_task_names.append("sliding_puzzle_game")

    initial_batch_dict = {
        "message_log": batch_message_logs,
        "extra_env_info": batch_extra_env_info,
        "loss_multiplier": batch_loss_multipliers,
        "idx": batch_indices,
        "task_name": batch_task_names,
        "stop_strings": ["</action>"],
    }
    return BatchedDataDict(initial_batch_dict)


@pytest.fixture(scope="function")
def sliding_puzzle_setup_vllm(
    rollout_cluster,
    rollout_tokenizer,
    sliding_puzzle_environment,
    initial_sliding_puzzle_batch,
):
    """Sets up components for sliding puzzle tests using VllmGeneration."""
    vllm_generation = None
    task_to_env, _ = sliding_puzzle_environment
    is_eval = True
    print("Creating VllmGeneration for Sliding Puzzle Test...")
    try:
        vllm_config = deepcopy(base_vllm_test_config)
        # Qwen model name is already in base config
        vllm_config["tokenizer_name"] = rollout_tokenizer.name_or_path
        vllm_config = configure_generation_config(
            vllm_config, rollout_tokenizer, is_eval=is_eval
        )
        # Ensure max_new_tokens is sufficient
        vllm_config["max_new_tokens"] = 500
        vllm_generation = VllmGeneration(rollout_cluster, vllm_config)
        vllm_generation.finish_generation()
        yield (
            vllm_generation,
            rollout_tokenizer,
            task_to_env,
            initial_sliding_puzzle_batch,
            rollout_cluster,
        )
    finally:
        print("Cleaning up VllmGeneration (Sliding Puzzle Test)...")
        if vllm_generation:
            vllm_generation.shutdown()
        print("VllmGeneration cleanup finished (Sliding Puzzle Test).")


@pytest.mark.vllm
def test_run_sliding_puzzle_vllm(sliding_puzzle_setup_vllm):
    """Tests sliding puzzle rollout with VllmGeneration."""
    vllm_generation, rollout_tokenizer, task_to_env, initial_batch, rollout_cluster = (
        sliding_puzzle_setup_vllm
    )
    max_moves = initial_batch["extra_env_info"][0]["max_moves"]
    max_rollout_turns = max_moves + 1
    max_seq_len = 2048

    print("\nRunning sliding puzzle rollout (VLLM)...")
    vllm_generation.prepare_for_generation()
    final_batch, rollout_metrics = run_multi_turn_rollout(
        policy_generation=vllm_generation,
        input_batch=initial_batch,
        tokenizer=rollout_tokenizer,
        task_to_env=task_to_env,
        max_rollout_turns=max_rollout_turns,
        max_seq_len=max_seq_len,
        greedy=True,
    )
    print(rollout_metrics)
    vllm_generation.finish_generation()
    print("Sliding puzzle rollout complete (VLLM).")

    # --- Assertions ---
    assert isinstance(final_batch, BatchedDataDict)
    assert "message_log" in final_batch
    assert "total_reward" in final_batch
    assert len(final_batch["message_log"]) == len(initial_batch["message_log"])

    sample_log = final_batch["message_log"][0]
    print(f"Final Total Reward: {final_batch['total_reward'][0].item()}")

    # Count the number of <action> tags and environment messages
    action_tag_count = 0
    environment_message_count = 0

    for msg in sample_log:
        if msg["role"] == "assistant" and "<action>" in msg["content"]:
            action_tag_count += 1
        elif msg["role"] == "environment":
            environment_message_count += 1

    print(f"Found {action_tag_count} messages with <action> tags")
    print(f"Found {environment_message_count} environment messages")

    # Assert that we have multiple action tags and environment messages
    assert action_tag_count > 3, "Expected at least one message with <action> tag"
    assert environment_message_count > 3, "Expected at least one environment message"

    print("\nSliding Puzzle VLLM Test assertions passed.")


def test_run_async_nemo_gym_rollout_warns_when_max_seq_len_exceeds_engine():
    class _FakePolicyGeneration:
        cfg = {"vllm_cfg": {"max_model_len": 100}}

    # stop_strings is truthy so the function hits the next assert and exits
    # right after emitting the warning — keeps this test free of any rollout work.
    async def _consume_rollout():
        async for _ in run_async_nemo_gym_rollout(
            policy_generation=_FakePolicyGeneration(),
            input_batch={"extra_env_info": []},
            tokenizer=None,
            task_to_env={},
            generation_config={"stop_strings": "x", "max_new_tokens": 50},
            num_generations=1,
            log_full_result_tables=False,
            max_seq_len=200,
            max_rollout_turns=None,
        ):
            pass

    with pytest.warns(UserWarning, match="greater than the"):
        with pytest.raises(AssertionError, match="Stop strings"):
            asyncio.run(_consume_rollout())


def test_nemo_gym_context_limit_dispatches_on_active_backend():
    policy_generation = type(
        "_PolicyGeneration",
        (),
        {
            "cfg": {
                "backend": "sglang",
                "sglang_cfg": {"context_length": 256},
                "vllm_cfg": {"max_model_len": 1},
            }
        },
    )()

    assert (
        rollouts_mod._resolve_generation_context_limit(
            policy_generation,
            policy_max_seq_len=512,
        )
        == 256
    )


def test_nemo_gym_context_limit_uses_policy_cap_for_nullable_sglang_limit():
    policy_generation = type(
        "_PolicyGeneration",
        (),
        {
            "cfg": {
                "backend": "sglang",
                "sglang_cfg": {"context_length": None},
                "vllm_cfg": {"max_model_len": 1},
            }
        },
    )()

    assert (
        rollouts_mod._resolve_generation_context_limit(
            policy_generation,
            policy_max_seq_len=512,
        )
        == 512
    )

    with pytest.raises(ValueError, match="SGLang NeMo-Gym rollouts require"):
        rollouts_mod._resolve_generation_context_limit(
            policy_generation,
            policy_max_seq_len=None,
        )


def test_native_rollout_groups_match_whole_batch(monkeypatch):
    """One native batch can be split without changing data or metric semantics."""

    async def fake_sample_rollout(sample_idx, initial_sample_state, **kwargs):
        del kwargs
        final_state = {
            "message_log": initial_sample_state["message_log"]
            + [{"role": "assistant", "content": f"answer-{sample_idx}"}],
            "extra_env_info": initial_sample_state["extra_env_info"],
            "task_name": initial_sample_state["task_name"],
            "total_reward": torch.tensor(float(sample_idx - 1)),
            "idx": initial_sample_state["idx"],
            "reward/correctness": torch.tensor(float(sample_idx)),
            "reward/style": torch.tensor(float(sample_idx * 2)),
        }
        sample_metrics = {
            "turn_count": sample_idx + 1,
            "total_tokens": sample_idx + 10,
            "assistant_tokens": sample_idx + 2,
            "env_tokens": 8,
            "terminated": sample_idx % 2 == 0,
            "truncated": sample_idx == 3,
            "max_turns_reached": sample_idx == 3,
            "total_reward": float(sample_idx - 1),
            "turn_gen_tokens": [sample_idx + 2],
            "turn_input_tokens": [sample_idx + 10],
            "turn_total_tokens": [sample_idx + 12],
            "max_gen_tokens_per_turn": sample_idx + 2,
            "per_worker_token_counts": {f"worker-{sample_idx % 2}": sample_idx + 1},
        }
        return final_state, sample_metrics

    monkeypatch.setattr(
        rollouts_mod, "run_sample_multi_turn_rollout", fake_sample_rollout
    )
    input_batch = BatchedDataDict(
        {
            "message_log": [
                [{"role": "user", "content": f"prompt-{i}"}] for i in range(4)
            ],
            "extra_env_info": [{"sample": i} for i in range(4)],
            "task_name": ["test"] * 4,
            "idx": [100, 101, 102, 103],
            "loss_multiplier": torch.arange(4),
        }
    )
    rollout_kwargs = {
        "policy_generation": None,
        "input_batch": input_batch,
        "tokenizer": None,
        "task_to_env": {},
        "max_seq_len": 128,
        "max_rollout_turns": 2,
    }

    whole_batch, whole_metrics = run_async_multi_turn_rollout(**rollout_kwargs)

    async def collect_groups():
        return [
            group
            async for group in run_async_multi_turn_rollout_groups(
                **rollout_kwargs, num_generations=2
            )
        ]

    groups = asyncio.run(collect_groups())

    assert [group.group_index for group in groups] == [0, 1]
    assert [group.final_batch.size for group in groups] == [2, 2]
    assert [group.task_index for group in groups] == [None, None]
    for key in (
        "total_reward",
        "reward/correctness",
        "reward/style",
        "loss_multiplier",
    ):
        assert torch.equal(
            torch.cat([group.final_batch[key] for group in groups]), whole_batch[key]
        )
    assert [idx for group in groups for idx in group.final_batch["idx"]] == whole_batch[
        "idx"
    ]

    group_metrics = [group.rollout_metrics for group in groups]
    assert (
        sum(metrics["total_turns"] for metrics in group_metrics)
        == whole_metrics["total_turns"]
    )
    for key in (
        "avg_turns_per_sample",
        "natural_termination_rate",
        "truncation_rate",
        "max_turns_reached_rate",
        "mean_total_tokens_per_sample",
        "mean_gen_tokens_per_sample",
        "mean_env_tokens_per_sample",
        "mean_total_reward",
    ):
        assert (
            sum(metrics[key] for metrics in group_metrics) / len(groups)
            == (whole_metrics[key])
        )
    for key in (
        "histogram/gen_tokens_length",
        "histogram/input_tokens_length",
        "histogram/total_tokens_length",
    ):
        assert [value for metrics in group_metrics for value in metrics[key]] == (
            whole_metrics[key]
        )
    combined_worker_counts = {}
    for metrics in group_metrics:
        for worker, count in metrics["per_worker_token_counts"].items():
            combined_worker_counts[worker] = (
                combined_worker_counts.get(worker, 0) + count
            )
    assert combined_worker_counts == whole_metrics["per_worker_token_counts"]
    assert (
        min(metrics["min_total_reward"] for metrics in group_metrics)
        == (whole_metrics["min_total_reward"])
    )
    assert (
        max(metrics["max_total_reward"] for metrics in group_metrics)
        == (whole_metrics["max_total_reward"])
    )
    assert (
        max(metrics["max_turns_per_sample"] for metrics in group_metrics)
        == (whole_metrics["max_turns_per_sample"])
    )
    assert (
        max(metrics["max_gen_tokens_per_sample"] for metrics in group_metrics)
        == (whole_metrics["max_gen_tokens_per_sample"])
    )


def test_run_async_nemo_gym_rollout_streams_complete_prompt_groups(monkeypatch):
    """Prompt groups are yielded in completion order using async iteration."""

    clock = {"now": 0.0}

    class _FakeTimerContext:
        def __init__(self, timer, label):
            self.timer = timer
            self.label = label

        def __enter__(self):
            self.timer.start(self.label)

        def __exit__(self, exc_type, exc_value, traceback):
            self.timer.stop(self.label)

    class _FakeTimer:
        def __init__(self):
            self.start_times = {}
            self.elapsed = {}

        def start(self, label):
            self.start_times[label] = clock["now"]

        def stop(self, label):
            elapsed = clock["now"] - self.start_times.pop(label)
            self.elapsed.setdefault(label, []).append(elapsed)
            return elapsed

        def time(self, label):
            return _FakeTimerContext(self, label)

        def get_timing_metrics(self, reduction_op="mean"):
            assert reduction_op == "sum"
            return {label: sum(values) for label, values in self.elapsed.items()}

    class _ReadyRef:
        def __init__(self, value):
            self.value = value

        def __await__(self):
            async def _resolve():
                return self.value

            return _resolve().__await__()

    class _AsyncOnlyStream:
        def __init__(self, values):
            self.values = iter(values)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                value = next(self.values)
            except StopIteration as error:
                raise StopAsyncIteration from error
            clock["now"] += 1.0
            return value

    class _RunRolloutsRemote:
        def options(self, *, num_returns):
            assert num_returns == "streaming"
            return self

        def remote(self, rows, tokenizer, timer_prefix):
            del rows, tokenizer, timer_prefix
            # Both groups complete out of order internally and group 1 completes first.
            completion_order = [3, 1, 2, 0]
            values = []
            for position, rowidx in enumerate(completion_order):
                result = {
                    "rowidx": rowidx,
                    "input_message_log": [{"token_ids": [rowidx]}],
                    "message_log": [
                        {
                            "role": "assistant",
                            "token_ids": [rowidx],
                            "generation_logprobs": [0.0],
                        }
                    ],
                }
                timing = {"timing/remote": 1.0} if position == 3 else None
                values.append(_ReadyRef((rowidx, result, timing)))
            return _AsyncOnlyStream(values)

    class _PolicyGeneration:
        cfg = {"vllm_cfg": {"max_model_len": 128}}

    rows = []
    for task_index in (10, 10, 11, 11):
        rows.append(
            {
                "agent_ref": {"name": "agent"},
                "responses_create_params": {},
                "_ng_task_index": task_index,
            }
        )
    input_batch = BatchedDataDict(
        {
            "extra_env_info": rows,
            "message_log": [[{"role": "user", "content": "prompt"}]] * 4,
            "loss_multiplier": torch.ones(4),
        }
    )
    captured_groups = []

    def _postprocess_group(**kwargs):
        assert kwargs["log_full_result_tables"] is False
        task_index = int(kwargs["nemo_gym_rows"][0]["_ng_task_index"])
        captured_groups.append(
            (task_index, [result["rowidx"] for result in kwargs["results"]])
        )
        return rollouts_mod.NemoGymRolloutResult(
            input_ids=torch.empty(0),
            final_batch=kwargs["input_batch"],
            rollout_metrics={},
            task_index=task_index,
        )

    monkeypatch.setattr(
        rollouts_mod, "_postprocess_single_nemo_gym_group", _postprocess_group
    )
    monkeypatch.setattr(rollouts_mod, "Timer", _FakeTimer)

    async def _collect():
        results = []
        async for result in run_async_nemo_gym_rollout(
            policy_generation=_PolicyGeneration(),
            input_batch=input_batch,
            tokenizer=None,
            task_to_env={
                "nemo_gym": type(
                    "_Environment",
                    (),
                    {"run_rollouts": _RunRolloutsRemote()},
                )()
            },
            generation_config={
                "stop_strings": [],
                "stop_token_ids": [],
                "top_k": None,
                "temperature": 1.0,
                "top_p": 1.0,
                "max_new_tokens": 32,
            },
            num_generations=2,
            log_full_result_tables=False,
        ):
            results.append(result)
            if len(results) == 1:
                # Simulate work done by the consumer before requesting another group.
                clock["now"] += 100.0
        return results

    rollout_results = asyncio.run(_collect())

    assert [result.task_index for result in rollout_results] == [11, 10]
    assert captured_groups == [(11, [2, 3]), (10, [0, 1])]
    assert rollout_results[-1].rollout_metrics["timing/remote"] == 1.0
    assert rollout_results[-1].rollout_metrics["timing/rollout/run_rollouts"] == 4.0
    assert rollout_results[-1].rollout_metrics["timing/rollout/total"] == 4.0


def test_nemo_gym_stream_accumulator_validates_rows_and_completion():
    rows = [
        {"agent_ref": {"name": "agent"}},
        {"agent_ref": {"name": "agent"}},
    ]
    accumulator = rollouts_mod._NemoGymStreamAccumulator(
        rows=rows,
        num_generations=2,
        allow_mixed_agents=False,
    )

    assert accumulator.add(0, {"row": 0}) is None
    with pytest.raises(ValueError, match="duplicate row index 0"):
        accumulator.add(0, {"row": 0})
    with pytest.raises(RuntimeError, match=r"missing row indices \[1\]"):
        accumulator.finish()

    with pytest.raises(ValueError, match="outside the expected range"):
        rollouts_mod._NemoGymStreamAccumulator(
            rows=rows,
            num_generations=2,
            allow_mixed_agents=False,
        ).add(2, {"row": 2})


def test_nemo_gym_stream_accumulator_rejects_mixed_agent_group():
    accumulator = rollouts_mod._NemoGymStreamAccumulator(
        rows=[
            {"agent_ref": {"name": "agent-a"}},
            {"agent_ref": {"name": "agent-b"}},
        ],
        num_generations=2,
        allow_mixed_agents=False,
    )

    assert accumulator.add(0, {"row": 0}) is None
    with pytest.raises(ValueError, match="one NeMo-Gym agent"):
        accumulator.add(1, {"row": 1})


@pytest.mark.parametrize("log_full_result_tables", [False, True])
def test_postprocess_nemo_gym_group_returns_task_index(log_full_result_tables):
    rows = [
        {"agent_ref": {"name": "agent"}, "_ng_task_index": 42},
        {"agent_ref": {"name": "agent"}, "_ng_task_index": 42},
    ]
    results = []
    for reward in (1.0, 2.0):
        input_message = {
            "role": "user",
            "content": "prompt",
            "token_ids": torch.tensor([1]),
        }
        results.append(
            {
                "input_message_log": [input_message],
                "message_log": [
                    input_message,
                    {
                        "role": "assistant",
                        "content": "answer",
                        "token_ids": torch.tensor([2]),
                        "generation_logprobs": torch.tensor([-0.1]),
                    },
                ],
                "full_result": {"reward": reward},
            }
        )

    rollout_result = rollouts_mod._postprocess_single_nemo_gym_group(
        nemo_gym_rows=rows,
        results=results,
        timer=rollouts_mod.Timer(),
        timer_prefix="timing/rollout",
        input_batch=BatchedDataDict({"loss_multiplier": torch.ones(2)}),
        tokenizer=type("_Tokenizer", (), {"pad_token_id": 0})(),
        log_full_result_tables=log_full_result_tables,
        max_total_tokens_per_sample=128,
    )

    assert rollout_result.task_index == 42
    assert rollout_result.final_batch["total_reward"].tolist() == [1.0, 2.0]
    assert (
        "agent/full_result" in rollout_result.rollout_metrics
    ) is log_full_result_tables


def test_run_nemo_gym_rollout_sync_drains_entire_batch(monkeypatch):
    input_batch = BatchedDataDict({"loss_multiplier": torch.ones(3)})
    expected = rollouts_mod.NemoGymRolloutResult(
        input_ids=torch.empty(0),
        final_batch=input_batch,
        rollout_metrics={"metric": 1.0},
        task_index=None,
    )

    async def fake_stream(**kwargs):
        assert kwargs["num_generations"] == input_batch.size
        assert kwargs["returns_entire_batch"] is True
        assert kwargs["log_full_result_tables"] is False
        yield expected

    monkeypatch.setattr(rollouts_mod, "run_async_nemo_gym_rollout", fake_stream)

    actual = run_nemo_gym_rollout_sync(
        policy_generation=None,
        input_batch=input_batch,
        tokenizer=None,
        task_to_env={},
        generation_config={},
        log_full_result_tables=False,
    )

    assert actual is expected


def test_rollout_manager_consumes_stream_and_restores_input_order():
    class _ReadyRef:
        def __init__(self, value):
            self.value = value

        def __await__(self):
            async def _resolve():
                return self.value

            return _resolve().__await__()

    class _Stream:
        def __init__(self):
            self.values = iter(
                [
                    _ReadyRef(
                        (
                            1,
                            {
                                "value": "second",
                                "input_message_log": [
                                    {"role": "user", "token_ids": [1]}
                                ],
                            },
                            None,
                        )
                    ),
                    _ReadyRef(
                        (
                            0,
                            {
                                "value": "first",
                                "input_message_log": [
                                    {"role": "user", "token_ids": [1]}
                                ],
                            },
                            {"remote_time": 2.0},
                        )
                    ),
                ]
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.values)
            except StopIteration as error:
                raise StopAsyncIteration from error

    class _RunRolloutsRemote:
        def options(self, *, num_returns):
            assert num_returns == "streaming"
            return self

        def remote(self, inputs, tokenizer, timer_prefix):
            del inputs, tokenizer, timer_prefix
            return _Stream()

    manager = object.__new__(AsyncNemoGymRolloutImpl)
    manager._task_to_env = {
        "nemo_gym": type("_Environment", (), {"run_rollouts": _RunRolloutsRemote()})()
    }
    manager._tokenizer = None
    manager._result_to_completion = lambda result: result["value"]
    manager._compute_rollout_metrics = lambda completions, agent: {
        "completion_count": len(completions),
        "agent": agent,
    }

    completions, prompt_message_log, metrics = asyncio.run(
        manager._run_rollouts(
            inputs=[
                {"agent_ref": {"name": "agent"}},
                {"agent_ref": {"name": "agent"}},
            ],
            timer=rollouts_mod.Timer(),
            timer_prefix="timing/test",
        )
    )

    assert completions == ["first", "second"]
    assert prompt_message_log[0]["role"] == "user"
    torch.testing.assert_close(prompt_message_log[0]["token_ids"], torch.tensor([1]))
    assert metrics == {
        "completion_count": 2,
        "agent": "agent",
        "remote_time": 2.0,
    }


def test_rollout_manager_rejects_duplicate_stream_rows():
    class _ReadyRef:
        def __init__(self, value):
            self.value = value

        def __await__(self):
            async def _resolve():
                return self.value

            return _resolve().__await__()

    class _DuplicateStream:
        def __init__(self):
            self.values = iter(
                [
                    _ReadyRef((0, {"value": "first"}, None)),
                    _ReadyRef((0, {"value": "duplicate"}, None)),
                ]
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.values)
            except StopIteration as error:
                raise StopAsyncIteration from error

    class _RunRolloutsRemote:
        def options(self, *, num_returns):
            assert num_returns == "streaming"
            return self

        def remote(self, inputs, tokenizer, timer_prefix):
            del inputs, tokenizer, timer_prefix
            return _DuplicateStream()

    manager = object.__new__(AsyncNemoGymRolloutImpl)
    manager._task_to_env = {
        "nemo_gym": type("_Environment", (), {"run_rollouts": _RunRolloutsRemote()})()
    }
    manager._tokenizer = None

    with pytest.raises(ValueError, match="duplicate row index 0"):
        asyncio.run(
            manager._run_rollouts(
                inputs=[
                    {"agent_ref": {"name": "agent"}},
                    {"agent_ref": {"name": "agent"}},
                ],
                timer=rollouts_mod.Timer(),
                timer_prefix="timing/test",
            )
        )


@pytest.mark.nemo_gym
def test_run_async_nemo_gym_rollout(
    nemo_gym,  # noqa: F811
    nemo_gym_vllm_generation,  # noqa: F811
    nemo_gym_sanity_test_data,  # noqa: F811
    nemo_gym_tokenizer,  # noqa: F811
):
    # only keep the input part of the data for the test
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for data in nemo_gym_sanity_test_data["input"]:
            f.write(json.dumps(data) + "\n")
        data_path = f.name

    # load the dataset and convert to compatible format for Nemo RL
    nemo_gym_sanity_test_data = NemoGymDataset(data_path)
    nemo_rl_compatible_examples: list[DatumSpec] = [
        nemo_gym_data_processor(
            nemo_gym_sanity_test_data.dataset[idx], None, None, None, idx
        )
        for idx in range(len(nemo_gym_sanity_test_data.dataset))
    ]

    input_batch: BatchedDataDict[DatumSpec] = rl_collate_fn(nemo_rl_compatible_examples)
    rows = input_batch["extra_env_info"]
    assert len(rows) >= 2, "test expects the fixture to provide at least two rows"

    max_new_tokens = nemo_gym_vllm_generation.cfg["max_new_tokens"]
    # Row 0: per-agent override looser than the configured cap — min() should clamp down to max_new_tokens.
    # Row 1: no per-agent override — should fall back to max_new_tokens.
    rows[0]["responses_create_params"]["max_output_tokens"] = max_new_tokens + 1
    assert "max_output_tokens" not in rows[1]["responses_create_params"]

    actual_result = run_nemo_gym_rollout_sync(
        policy_generation=nemo_gym_vllm_generation,
        input_batch=input_batch,
        tokenizer=nemo_gym_tokenizer,
        task_to_env={"nemo_gym": nemo_gym},
        max_seq_len=nemo_gym_vllm_generation.cfg["vllm_cfg"]["max_model_len"],
        generation_config=nemo_gym_vllm_generation.cfg,
        log_full_result_tables=True,
        max_rollout_turns=None,
    )
    for row in rows:
        assert row["responses_create_params"]["max_output_tokens"] == max_new_tokens
    actual_result = asdict(actual_result)
    actual_result["final_batch"] = actual_result["final_batch"].get_dict()

    expected_result = {
        "final_batch": {
            "agent_ref": [
                {
                    "name": "example_multi_step_simple_agent",
                    "type": "responses_api_agents",
                },
                {
                    "name": "example_multi_step_simple_agent",
                    "type": "responses_api_agents",
                },
            ],
            "length": torch.tensor([3080, 3048]),
            "loss_multiplier": torch.tensor([1.0, 1.0]),
            "mask_sample": torch.tensor([False, False]),
            "total_reward": torch.tensor([0.0, 0.0]),
            "truncated": torch.tensor([False, False]),
        },
        "rollout_metrics": {
            # core metrics
            "timing/rollout/total": 0.0,
            "timing/rollout/run_rollouts": 0.0,
            "timing/rollout/await_results": 0.0,
            "timing/rollout/postprocess_results": 0.0,
            "timing/rollout/postprocess_results_pct": 0.0,
            "timing/rollout/prepare_for_metrics_calculation": 0.0,
            "timing/rollout/aggregate_metrics": 0.0,
            "timing/rollout/per_agent_misc_metrics": 0.0,
            "mean_gen_tokens_per_sample": None,
            "turns_per_sample/mean": 2.0,
            "turns_per_sample/max": 2,
            "turns_per_sample/min": 2,
            "turns_per_sample/median": 2.0,
            "turns_per_sample/stddev": 0.0,
            "turns_per_sample/histogram": None,
            "turns_per_sample/p95": None,
            "turns_per_sample/p99": None,
            "total_tokens_per_sample/mean": 3843.0,
            "total_tokens_per_sample/max": 3848,
            "total_tokens_per_sample/min": 3838,
            "total_tokens_per_sample/median": 3843.0,
            "total_tokens_per_sample/stddev": 7.0710678118654755,
            "total_tokens_per_sample/histogram": None,
            "gen_tokens_per_sample/mean": 732.5,
            "gen_tokens_per_sample/max": 748,
            "gen_tokens_per_sample/min": 717,
            "gen_tokens_per_sample/median": 732.5,
            "gen_tokens_per_sample/stddev": 21.920310216782973,
            "gen_tokens_per_sample/histogram": None,
            "max_gen_tokens_per_turn/mean": None,
            "max_gen_tokens_per_turn/max": None,
            "max_gen_tokens_per_turn/min": None,
            "max_gen_tokens_per_turn/median": None,
            "max_gen_tokens_per_turn/stddev": None,
            "max_gen_tokens_per_turn/histogram": None,
            "max_gen_tokens_per_turn/p95": None,
            "total_reward/mean": 0.0,
            "total_reward/max": 0.0,
            "total_reward/min": 0.0,
            "total_reward/median": 0.0,
            "total_reward/stddev": 0.0,
            "total_reward/histogram": None,
            "natural_termination_rate": None,
            "truncation_rate": None,
            # per agent metrics
            "example_multi_step_simple_agent/full_result": None,
            "example_multi_step_simple_agent/accuracy/histogram": None,
            "example_multi_step_simple_agent/accuracy/max": 0.0,
            "example_multi_step_simple_agent/accuracy/mean": 0.0,
            "example_multi_step_simple_agent/accuracy/median": 0.0,
            "example_multi_step_simple_agent/accuracy/min": 0.0,
            "example_multi_step_simple_agent/accuracy/stddev": 0.0,
            "example_multi_step_simple_agent/order_instruction_following_failure/histogram": None,
            "example_multi_step_simple_agent/order_instruction_following_failure/max": 0.0,
            "example_multi_step_simple_agent/order_instruction_following_failure/mean": 0.0,
            "example_multi_step_simple_agent/order_instruction_following_failure/median": 0.0,
            "example_multi_step_simple_agent/order_instruction_following_failure/min": 0.0,
            "example_multi_step_simple_agent/order_instruction_following_failure/stddev": 0.0,
            "example_multi_step_simple_agent/original_term_minefield_hit/histogram": None,
            "example_multi_step_simple_agent/original_term_minefield_hit/max": 0.0,
            "example_multi_step_simple_agent/original_term_minefield_hit/mean": 0.0,
            "example_multi_step_simple_agent/original_term_minefield_hit/median": 0.0,
            "example_multi_step_simple_agent/original_term_minefield_hit/min": 0.0,
            "example_multi_step_simple_agent/original_term_minefield_hit/stddev": 0.0,
            "example_multi_step_simple_agent/reward/histogram": None,
            "example_multi_step_simple_agent/reward/max": 0.0,
            "example_multi_step_simple_agent/reward/mean": 0.0,
            "example_multi_step_simple_agent/reward/median": 0.0,
            "example_multi_step_simple_agent/reward/min": 0.0,
            "example_multi_step_simple_agent/reward/stddev": 0.0,
            "example_multi_step_simple_agent/set_overlap/histogram": None,
            "example_multi_step_simple_agent/set_overlap/max": 0.0,
            "example_multi_step_simple_agent/set_overlap/mean": 0.0,
            "example_multi_step_simple_agent/set_overlap/median": 0.0,
            "example_multi_step_simple_agent/set_overlap/min": 0.0,
            "example_multi_step_simple_agent/set_overlap/stddev": 0.0,
        },
    }

    def _standardize(d: dict) -> dict:
        final_batch = d["final_batch"].copy()
        final_batch.pop("message_log", None)
        final_batch["total_reward"] = final_batch["total_reward"].tolist()
        final_batch["loss_multiplier"] = final_batch["loss_multiplier"].tolist()
        final_batch["length"] = final_batch["length"].tolist()
        # truncated depends on exact generation output which is not reproducible,
        # so just verify each value is a bool rather than checking exact values
        if "truncated" in final_batch:
            assert all(
                isinstance(v, (bool, int)) for v in final_batch["truncated"].tolist()
            )
            final_batch.pop("truncated")
        if "mask_sample" in final_batch:
            assert all(
                isinstance(v, (bool, int)) for v in final_batch["mask_sample"].tolist()
            )
            final_batch.pop("mask_sample")

        for key in d["rollout_metrics"]:
            # We remove these fields from comparison since we cannot guarantee exact generation reproducibility
            d["rollout_metrics"][key] = None

        return {
            "final_batch": final_batch,
            "rollout_metrics": d["rollout_metrics"],
        }

    assert _standardize(expected_result) == _standardize(actual_result)

    """
    If the result here does not match, please check the following:
    1. In nemo_rl/experience/rollouts.py::run_async_nemo_gym_rollout, the sampling params are passed appropriately
    2. In nemo_rl/models/generation/vllm/vllm_worker_async.py::VllmAsyncGenerationWorker::_setup_vllm_server::create_chat_completion, the sampling params (like top_k) are set as appropriate
    """
