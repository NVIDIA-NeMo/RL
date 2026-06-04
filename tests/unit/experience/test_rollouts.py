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
from nemo_rl.experience.interfaces import Completion, PromptGroupRecord
from nemo_rl.experience.rollout_manager import RolloutManager
from nemo_rl.experience.rollouts import (
    _calculate_single_metric,
    run_async_multi_turn_rollout,
    run_async_nemo_gym_rollout,
    run_multi_turn_rollout,
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
    """Unit tests for _calculate_single_metric function."""

    def test_single_value_returns_nan_for_stddev(self):
        """Test that stddev returns nan when given a single value (GitHub issue #1411)."""
        import math

        result = _calculate_single_metric([42.0], batch_size=1, key_name="test")

        assert result["test/mean"] == 42.0
        assert result["test/max"] == 42.0
        assert result["test/min"] == 42.0
        assert result["test/median"] == 42.0
        assert math.isnan(result["test/stddev"]), (
            "stddev should be nan for single value"
        )

    def test_multiple_values_computes_stddev(self):
        """Test that stddev is computed correctly for multiple values."""
        result = _calculate_single_metric(
            [1.0, 2.0, 3.0], batch_size=3, key_name="test"
        )

        assert result["test/mean"] == 2.0
        assert result["test/max"] == 3.0
        assert result["test/min"] == 1.0
        assert result["test/median"] == 2.0
        assert abs(result["test/stddev"] - 1.0) < 1e-9  # stdev of [1,2,3] is 1.0

    def test_two_identical_values_returns_zero_stddev(self):
        """Test that stddev is 0 when all values are identical."""
        result = _calculate_single_metric([5.0, 5.0], batch_size=2, key_name="test")

        assert result["test/stddev"] == 0.0


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
    with pytest.warns(UserWarning, match="greater than the"):
        with pytest.raises(AssertionError, match="Stop strings"):
            run_async_nemo_gym_rollout(
                policy_generation=_FakePolicyGeneration(),
                input_batch={"extra_env_info": []},
                tokenizer=None,
                task_to_env={},
                generation_config={"stop_strings": "x", "max_new_tokens": 50},
                max_seq_len=200,
                max_rollout_turns=None,
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

    actual_result = run_async_nemo_gym_rollout(
        policy_generation=nemo_gym_vllm_generation,
        input_batch=input_batch,
        tokenizer=nemo_gym_tokenizer,
        task_to_env={"nemo_gym": nemo_gym},
        max_seq_len=nemo_gym_vllm_generation.cfg["vllm_cfg"]["max_model_len"],
        generation_config=nemo_gym_vllm_generation.cfg,
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


# ---------------------------------------------------------------------------
# Tests for RolloutManager
# ---------------------------------------------------------------------------


def test_rollout_manager_raises_without_impl_params():
    """RolloutManager raises AssertionError when required params are missing."""
    common = {
        "tokenizer": None,
        "env_handles": {},
        "num_generations_per_prompt": 1,
        "max_seq_len": 1,
    }

    with pytest.raises(AssertionError, match="num_generations_per_prompt must be >= 1"):
        updated_common = common.copy()
        updated_common["num_generations_per_prompt"] = 0
        RolloutManager(**updated_common, use_nemo_gym=False)

    with pytest.raises(AssertionError, match="policy_generation is required"):
        RolloutManager(**common, use_nemo_gym=False)

    with pytest.raises(AssertionError, match="generation_config is required"):
        RolloutManager(**common, use_nemo_gym=True)


# ---------------------------------------------------------------------------
# Tests for AsyncRolloutManager (native async path)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def single_multi_step_calculator_input_sample(rollout_tokenizer):
    """Returns a single DatumSpec prompt dict (problem 0) for AsyncRolloutManager tests."""
    problem_text = "(5 + 3) * 2"
    expected_answer = 16.0
    max_steps = 5

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
    return {
        "message_log": message_log,
        "extra_env_info": metadata,
        "task_name": "multi_step_calculator_game",
        "stop_strings": ["<call: calculator>"],
        "idx": 0,
    }


@pytest.mark.vllm
def test_async_rollout_manager(
    multi_step_setup_vllm_async,
    single_multi_step_calculator_input_sample,
):
    """Standalone test for AsyncRolloutManager.

    Given 1 prompt with num_generations_per_prompt=N, asserts:
    - output is a PromptGroupRecord with N Completion objects
    - each Completion has a reward (float) and a non-empty message_log
    - rollout_metrics has the expected keys with correct types
    - completions hold independent (not aliased) message_log objects
    """
    vllm_generation, rollout_tokenizer, env_handles, _, _ = multi_step_setup_vllm_async
    input_sample = single_multi_step_calculator_input_sample
    num_generations = 2
    max_seq_len = 1024
    max_rollout_turns = input_sample["extra_env_info"]["max_steps"] + 1

    manager = RolloutManager(
        use_nemo_gym=False,
        tokenizer=rollout_tokenizer,
        env_handles=env_handles,
        num_generations_per_prompt=num_generations,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
        policy_generation=vllm_generation,
    )

    vllm_generation.prepare_for_generation()
    record = asyncio.run(manager.run_rollout(input_sample))
    vllm_generation.finish_generation()

    assert isinstance(record, PromptGroupRecord)
    assert len(record.completions) == num_generations, (
        f"Expected {num_generations} completions, got {len(record.completions)}"
    )
    assert record.prompt_idx == input_sample["idx"]

    for i, completion in enumerate(record.completions):
        assert isinstance(completion, Completion)

        # 1. message_log length
        assert len(completion.message_log) >= 4, (
            f"Completion {i}: expected >= 4 messages, got {len(completion.message_log)}"
        )

        # 2. last assistant content
        last_assistant = next(
            (m for m in reversed(completion.message_log) if m["role"] == "assistant"),
            None,
        )
        assert last_assistant is not None, f"Completion {i}: no assistant message found"
        assert last_assistant["content"].strip() == "16", (
            f"Completion {i}: last assistant content {last_assistant['content']!r} != '16'"
        )

        # 3. reward
        assert completion.reward == 1.0, (
            f"Completion {i}: reward {completion.reward} != 1.0"
        )

    # completions must be independent objects
    assert record.completions[0].message_log is not record.completions[1].message_log


@pytest.mark.vllm
def test_async_rollout_manager_truncation(
    multi_step_setup_vllm_async,
    single_multi_step_calculator_input_sample,
):
    """Small max_seq_len forces truncation and truncation_rate=1.0."""
    vllm_generation, rollout_tokenizer, task_to_env, _, _ = multi_step_setup_vllm_async
    input_sample = single_multi_step_calculator_input_sample
    num_generations = 2
    max_seq_len = 290
    max_rollout_turns = input_sample["extra_env_info"]["max_steps"] + 1

    manager = RolloutManager(
        use_nemo_gym=False,
        tokenizer=rollout_tokenizer,
        task_to_env=task_to_env,
        num_generations_per_prompt=num_generations,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
        policy_generation=vllm_generation,
    )
    vllm_generation.prepare_for_generation()
    record = asyncio.run(manager.run_rollout(input_sample))
    vllm_generation.finish_generation()

    assert len(record.completions) == num_generations
    assert all(c.truncated for c in record.completions)
    assert record.rollout_metrics["truncation_rate"] == 1.0
    assert record.rollout_metrics["natural_termination_rate"] == 0.0


@pytest.mark.vllm
def test_async_rollout_manager_matches_original(
    multi_step_setup_vllm_async,
    single_multi_step_calculator_input_sample,
):
    """Comparison test: AsyncRolloutManager output is structurally equivalent to the original.

    Calls run_async_multi_turn_rollout with a batch of N identical prompts,
    then calls AsyncRolloutManager with 1 prompt and N generations.
    Asserts that both produce N results with matching message-log depth, rewards,
    and rollout_metrics numeric values.

    TODO: remove this test together with run_async_multi_turn_rollout when the legacy path is deleted.
    """
    vllm_generation, rollout_tokenizer, env_handles, _, _ = multi_step_setup_vllm_async
    input_sample = single_multi_step_calculator_input_sample
    num_generations = 2
    max_seq_len = 1024
    max_rollout_turns = input_sample["extra_env_info"]["max_steps"] + 1

    # Build a batch of N identical prompts for the original function
    batch = BatchedDataDict(
        {
            "message_log": [
                deepcopy(input_sample["message_log"]) for _ in range(num_generations)
            ],
            "extra_env_info": [
                deepcopy(input_sample["extra_env_info"]) for _ in range(num_generations)
            ],
            "task_name": [input_sample["task_name"]] * num_generations,
            "stop_strings": [input_sample["stop_strings"]] * num_generations,
            "idx": list(range(num_generations)),
            "loss_multiplier": [1.0] * num_generations,
        }
    )

    vllm_generation.prepare_for_generation()
    original_batch, original_metrics = run_async_multi_turn_rollout(
        policy_generation=vllm_generation,
        input_batch=batch,
        tokenizer=rollout_tokenizer,
        task_to_env=env_handles,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
    )

    manager = RolloutManager(
        use_nemo_gym=False,
        tokenizer=rollout_tokenizer,
        env_handles=env_handles,
        num_generations_per_prompt=num_generations,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
        policy_generation=vllm_generation,
    )
    record = asyncio.run(manager.run_rollout(input_sample))
    vllm_generation.finish_generation()

    # Both should produce N results
    assert len(original_batch["message_log"]) == num_generations
    assert len(record.completions) == num_generations

    for i in range(num_generations):
        orig_msg_log = original_batch["message_log"][i]
        new_msg_log = record.completions[i].message_log

        # 1. message_log length matches
        assert len(orig_msg_log) == len(new_msg_log), (
            f"Completion {i}: message_log length {len(new_msg_log)} != original {len(orig_msg_log)}"
        )

        # 2. last assistant content matches
        def _last_assistant_content(msg_log):
            for m in reversed(msg_log):
                if m["role"] == "assistant":
                    return m.get("content", "")
            return ""

        orig_last = _last_assistant_content(orig_msg_log)
        new_last = _last_assistant_content(new_msg_log)
        assert orig_last == new_last, (
            f"Completion {i}: last assistant content mismatch\n"
            f"  original:  {orig_last!r}\n"
            f"  manager:   {new_last!r}"
        )

        # 3. reward matches
        orig_reward = original_batch["total_reward"][i].item()
        new_reward = record.completions[i].reward
        assert orig_reward == new_reward, (
            f"Completion {i}: reward mismatch — original {orig_reward}, manager {new_reward}"
        )

    # 4. rollout_metrics numeric values match (timing and histogram fields are excluded).
    # The new impl emits slash-style keys (X/mean, X/max, X/min) via _calculate_single_metric;
    # translate the legacy prefix-style keys before comparing.
    def _translate_legacy_key(key: str) -> str:
        if key == "avg_turns_per_sample":
            return "turns_per_sample/mean"
        if key == "max_turns_reached_rate":
            return key
        for prefix, suffix in (("mean_", "/mean"), ("max_", "/max"), ("min_", "/min")):
            if key.startswith(prefix):
                return f"{key[len(prefix) :]}{suffix}"
        return key

    new_metrics = record.rollout_metrics
    for key in original_metrics.keys():
        if key.startswith("timing/") or key.startswith("histogram/"):
            continue

        new_key = _translate_legacy_key(key)
        assert new_key in new_metrics, (
            f"rollout_metrics[{new_key!r}] missing from manager"
        )

        orig_val = original_metrics[key]
        new_val = new_metrics[new_key]

        assert type(orig_val) == type(new_val), (
            f"rollout_metrics[{key!r}] type mismatch: {type(orig_val)} != {type(new_val)}"
        )
        if not isinstance(orig_val, (bool, int, float)):
            continue

        assert orig_val == pytest.approx(new_val), (
            f"rollout_metrics[{key!r}] mismatch — original {orig_val}, manager {new_val}"
        )


# ---------------------------------------------------------------------------
# Tests for AsyncNemoGymRolloutManager
# ---------------------------------------------------------------------------


@pytest.mark.nemo_gym
def test_async_nemo_gym_rollout_manager(
    nemo_gym,  # noqa: F811
    nemo_gym_vllm_generation,  # noqa: F811
    nemo_gym_sanity_test_data,  # noqa: F811
    nemo_gym_tokenizer,  # noqa: F811
):
    """Standalone test for AsyncNemoGymRolloutManager.

    Given 1 prompt with num_generations_per_prompt=N, asserts:
    - output is a PromptGroupRecord with N Completion objects
    - each Completion has a reward (float) and a non-empty message_log
    - completions hold independent message_log objects

    If the result here does not match, please check the following:
    1. Test data changed: re-run test_nemo_gym_sanity (tests/unit/environments/test_nemo_gym.py)
       and use _write_actual_test_data output to refresh test_nemo_gym_sanity.json.
    2. Logic changed: inspect recent changes to AsyncNemoGymRolloutManager or the gym env.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for data in nemo_gym_sanity_test_data["input"]:
            f.write(json.dumps(data) + "\n")
        data_path = f.name

    dataset = NemoGymDataset(data_path)
    examples = [
        nemo_gym_data_processor(dataset.dataset[idx], None, None, None, idx)
        for idx in range(len(dataset.dataset))
    ]
    input_batch: BatchedDataDict[DatumSpec] = rl_collate_fn(examples)

    # Use only the first prompt
    single_prompt = {
        "message_log": input_batch["message_log"][0],
        "extra_env_info": input_batch["extra_env_info"][0],
        "task_name": "nemo_gym",
        "idx": 0,
        "loss_multiplier": float(input_batch["loss_multiplier"][0]),
    }
    num_generations = 2

    manager = RolloutManager(
        use_nemo_gym=True,
        tokenizer=nemo_gym_tokenizer,
        env_handles={"nemo_gym": nemo_gym},
        num_generations_per_prompt=num_generations,
        max_seq_len=nemo_gym_vllm_generation.cfg["vllm_cfg"]["max_model_len"],
        generation_config=nemo_gym_vllm_generation.cfg,
    )
    record = asyncio.run(manager.run_rollout(single_prompt))

    assert isinstance(record, PromptGroupRecord)
    assert len(record.completions) == num_generations, (
        f"Expected {num_generations} completions, got {len(record.completions)}"
    )
    assert record.prompt_idx == 0

    for i, completion in enumerate(record.completions):
        assert isinstance(completion, Completion)

        # 1. message_log length
        assert len(completion.message_log) == 2, (
            f"Completion {i}: expected 2 messages, got {len(completion.message_log)}"
        )

        # 2. last assistant token_ids
        last_assistant = next(
            (m for m in reversed(completion.message_log) if m["role"] == "assistant"),
            None,
        )
        assert last_assistant is not None, f"Completion {i}: no assistant message found"
        assert torch.equal(
            last_assistant["token_ids"],
            torch.tensor([151667, 198, 32313, 11, 1077]),
        ), (
            f"Completion {i}: last assistant token_ids {last_assistant['token_ids'].tolist()} "
            f"!= [151667, 198, 32313, 11, 1077]"
        )

        # 3. reward
        assert completion.reward == 0.0, (
            f"Completion {i}: reward {completion.reward} != 0.0"
        )

    # completions must be independent objects
    assert record.completions[0].message_log is not record.completions[1].message_log


@pytest.mark.nemo_gym
def test_async_nemo_gym_rollout_manager_matches_original(
    nemo_gym,  # noqa: F811
    nemo_gym_vllm_generation,  # noqa: F811
    nemo_gym_sanity_test_data,  # noqa: F811
    nemo_gym_tokenizer,  # noqa: F811
):
    """Comparison test: AsyncNemoGymRolloutManager output is structurally equivalent to the original.

    Calls run_async_nemo_gym_rollout with a batch of N identical rows,
    then calls AsyncNemoGymRolloutManager with 1 prompt, N generations.
    Asserts that both produce N results and rewards are in the same numeric domain.

    TODO: remove this test together with run_async_nemo_gym_rollout when the legacy path is deleted.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for data in nemo_gym_sanity_test_data["input"]:
            f.write(json.dumps(data) + "\n")
        data_path = f.name

    dataset = NemoGymDataset(data_path)
    examples = [
        nemo_gym_data_processor(dataset.dataset[idx], None, None, None, idx)
        for idx in range(len(dataset.dataset))
    ]
    input_batch: BatchedDataDict[DatumSpec] = rl_collate_fn(examples)

    num_generations = 2
    single_prompt = {
        "message_log": input_batch["message_log"][0],
        "extra_env_info": input_batch["extra_env_info"][0],
        "task_name": "nemo_gym",
        "idx": 0,
        "loss_multiplier": float(input_batch["loss_multiplier"][0]),
    }

    # Build a batch of N identical rows for the original function
    repeated_batch = BatchedDataDict(
        {
            "message_log": [
                deepcopy(input_batch["message_log"][0]) for _ in range(num_generations)
            ],
            "extra_env_info": [
                deepcopy(input_batch["extra_env_info"][0])
                for _ in range(num_generations)
            ],
            "loss_multiplier": input_batch["loss_multiplier"][0:1].repeat(
                num_generations
            ),
            "idx": list(range(num_generations)),
            "task_name": ["nemo_gym"] * num_generations,
        }
    )

    original_result = run_async_nemo_gym_rollout(
        policy_generation=nemo_gym_vllm_generation,
        input_batch=repeated_batch,
        tokenizer=nemo_gym_tokenizer,
        task_to_env={"nemo_gym": nemo_gym},
        generation_config=nemo_gym_vllm_generation.cfg,
        max_seq_len=nemo_gym_vllm_generation.cfg["vllm_cfg"]["max_model_len"],
        max_rollout_turns=None,
    )

    manager = RolloutManager(
        use_nemo_gym=True,
        tokenizer=nemo_gym_tokenizer,
        env_handles={"nemo_gym": nemo_gym},
        num_generations_per_prompt=num_generations,
        max_seq_len=nemo_gym_vllm_generation.cfg["vllm_cfg"]["max_model_len"],
        generation_config=nemo_gym_vllm_generation.cfg,
    )
    record = asyncio.run(manager.run_rollout(single_prompt))

    # Both should produce N completions
    assert len(original_result.final_batch["message_log"]) == num_generations
    assert len(record.completions) == num_generations

    for i in range(num_generations):
        orig_msg_log = original_result.final_batch["message_log"][i]
        new_msg_log = record.completions[i].message_log

        # 1. message_log length matches
        assert len(orig_msg_log) == len(new_msg_log), (
            f"Completion {i}: message_log length {len(new_msg_log)} != original {len(orig_msg_log)}"
        )

        # 2. last assistant token_ids match
        def _last_assistant_token_ids(msg_log):
            for m in reversed(msg_log):
                if m["role"] == "assistant":
                    return m.get("token_ids")
            return None

        orig_token_ids = _last_assistant_token_ids(orig_msg_log)
        new_token_ids = _last_assistant_token_ids(new_msg_log)
        assert orig_token_ids is not None, (
            f"Completion {i}: no assistant message in original"
        )
        assert new_token_ids is not None, (
            f"Completion {i}: no assistant message in manager"
        )
        assert torch.equal(orig_token_ids, new_token_ids), (
            f"Completion {i}: last assistant token_ids mismatch\n"
            f"  original:  {orig_token_ids.tolist()}\n"
            f"  manager:   {new_token_ids.tolist()}"
        )

        # 3. reward matches
        orig_reward = original_result.final_batch["total_reward"][i].item()
        new_reward = record.completions[i].reward
        assert orig_reward == new_reward, (
            f"Completion {i}: reward mismatch — original {orig_reward}, manager {new_reward}"
        )

    # 4. rollout_metrics numeric values match (timing and Table fields are excluded)
    orig_metrics = original_result.rollout_metrics
    new_metrics = record.rollout_metrics
    for key in orig_metrics.keys():
        # Skip timing and full_result fields
        if key.startswith("timing/") or key.endswith("/full_result"):
            continue

        # Check that the key is present in the new metrics
        assert key in new_metrics, f"rollout_metrics[{key!r}] missing from manager"

        orig_val = orig_metrics[key]
        new_val = new_metrics[key]

        # Skip non-numeric fields
        assert type(orig_val) == type(new_val), (
            f"rollout_metrics[{key!r}] type mismatch: {type(orig_val)} != {type(new_val)}"
        )
        if not isinstance(orig_val, (bool, int, float)):
            continue

        # Check equal
        assert orig_val == pytest.approx(new_val), (
            f"rollout_metrics[{key!r}] mismatch — original {orig_val}, manager {new_val}"
        )
