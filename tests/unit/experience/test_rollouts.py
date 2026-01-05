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

import gc
from copy import deepcopy
from dataclasses import asdict

import pytest
import ray
import torch
from transformers import AutoTokenizer

from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.environments.games.sliding_puzzle import (
    SlidingPuzzleConfig,
    SlidingPuzzleEnv,
    SlidingPuzzleGameLogic,
    SlidingPuzzleMetadata,
)
from nemo_rl.environments.penguin import penguin_example_to_nemo_rl_datum_spec
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_async_penguin_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration

# These are all fixtures
from tests.unit.environments.test_penguin import (
    PENGUIN_INSTALLED,
    cluster,  # noqa: F401
    penguin,  # noqa: F401
    penguin_sanity_test_data,  # noqa: F401
    penguin_tokenizer,  # noqa: F401
    penguin_vllm_generation,  # noqa: F401
)

# Import the test environment definitions
from tests.unit.test_envs import (
    MultiStepCalcMetadata,
    MultiStepCalculatorEnv,
    _MultiStepCalculatorLogic,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 1,
    reason="VLLM test requires at least 1 GPU",
)
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 1,
    reason="VLLM test requires at least 1 GPU",
)
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 1,
    reason="VLLM test requires at least 1 GPU",
)
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 1,
    reason="VLLM test requires at least 1 GPU",
)
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 1,
    reason="VLLM test requires at least 1 GPU",
)
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


@pytest.mark.skipif(
    not PENGUIN_INSTALLED,
    reason="Skipping Penguin test since Penguin is not installed!",
)
def test_run_async_penguin_rollout(
    penguin,  # noqa: F811
    penguin_vllm_generation,  # noqa: F811
    penguin_sanity_test_data,  # noqa: F811
    penguin_tokenizer,  # noqa: F811
):
    nemo_rl_compatible_examples: list[DatumSpec] = [
        penguin_example_to_nemo_rl_datum_spec(penguin_example, idx)
        for idx, penguin_example in enumerate(penguin_sanity_test_data["input"])
    ]
    input_batch: BatchedDataDict[DatumSpec] = rl_collate_fn(nemo_rl_compatible_examples)
    actual_result = run_async_penguin_rollout(
        policy_generation=penguin_vllm_generation,
        input_batch=input_batch,
        tokenizer=penguin_tokenizer,
        task_to_env={"penguin": penguin},
        max_seq_len=None,
        generation_config=penguin_vllm_generation.cfg,
        max_rollout_turns=None,
    )
    actual_result = asdict(actual_result)
    actual_result["final_batch"] = actual_result["final_batch"].get_dict()

    expected_result = {
        "final_batch": {
            "length": torch.tensor([3088, 3056]),
            "loss_multiplier": torch.tensor([1.0, 1.0]),
            "total_reward": torch.tensor([0.0, 0.0]),
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
            "multineedle_simple_agent/full_result": None,
            "multineedle_simple_agent/accuracy/histogram": None,
            "multineedle_simple_agent/accuracy/max": 0.0,
            "multineedle_simple_agent/accuracy/mean": 0.0,
            "multineedle_simple_agent/accuracy/median": 0.0,
            "multineedle_simple_agent/accuracy/min": 0.0,
            "multineedle_simple_agent/accuracy/stddev": 0.0,
            "multineedle_simple_agent/order_instruction_following_failure/histogram": None,
            "multineedle_simple_agent/order_instruction_following_failure/max": 0.0,
            "multineedle_simple_agent/order_instruction_following_failure/mean": 0.0,
            "multineedle_simple_agent/order_instruction_following_failure/median": 0.0,
            "multineedle_simple_agent/order_instruction_following_failure/min": 0.0,
            "multineedle_simple_agent/order_instruction_following_failure/stddev": 0.0,
            "multineedle_simple_agent/original_term_minefield_hit/histogram": None,
            "multineedle_simple_agent/original_term_minefield_hit/max": 0.0,
            "multineedle_simple_agent/original_term_minefield_hit/mean": 0.0,
            "multineedle_simple_agent/original_term_minefield_hit/median": 0.0,
            "multineedle_simple_agent/original_term_minefield_hit/min": 0.0,
            "multineedle_simple_agent/original_term_minefield_hit/stddev": 0.0,
            "multineedle_simple_agent/reward/histogram": None,
            "multineedle_simple_agent/reward/max": 0.0,
            "multineedle_simple_agent/reward/mean": 0.0,
            "multineedle_simple_agent/reward/median": 0.0,
            "multineedle_simple_agent/reward/min": 0.0,
            "multineedle_simple_agent/reward/stddev": 0.0,
            "multineedle_simple_agent/set_overlap/histogram": None,
            "multineedle_simple_agent/set_overlap/max": 0.0,
            "multineedle_simple_agent/set_overlap/mean": 0.0,
            "multineedle_simple_agent/set_overlap/median": 0.0,
            "multineedle_simple_agent/set_overlap/min": 0.0,
            "multineedle_simple_agent/set_overlap/stddev": 0.0,
        },
    }

    def _standardize(d: dict) -> dict:
        final_batch = d["final_batch"].copy()
        final_batch.pop("message_log", None)
        final_batch["total_reward"] = final_batch["total_reward"].tolist()
        final_batch["loss_multiplier"] = final_batch["loss_multiplier"].tolist()
        final_batch["length"] = final_batch["length"].tolist()

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
    1. In nemo_rl/experience/rollouts.py::run_async_penguin_rollout, the sampling params are passed appropriately
    2. In nemo_rl/models/generation/vllm/vllm_worker_async.py::VllmAsyncGenerationWorker::_setup_vllm_server::create_chat_completion, the sampling params (like top_k) are set as appropriate
    """
