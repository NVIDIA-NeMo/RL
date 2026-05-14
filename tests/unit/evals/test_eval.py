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


import pytest
import torch

from nemo_rl.data.collate_fn import eval_collate_fn, rl_collate_fn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.evals.eval import (
    NEMO_GYM_ROLLOUT_MODE,
    SINGLE_TURN_ROLLOUT_MODE,
    _collect_nemo_gym_evaluation_data,
    _get_eval_collate_fn,
    _make_json_serializable,
    _score_batch_rewards,
    _validate_eval_config,
    eval_cons_k,
    eval_pass_k,
)
from nemo_rl.experience.rollouts import AsyncNemoGymRolloutResult


def _base_master_config(rollout_mode=SINGLE_TURN_ROLLOUT_MODE):
    return {
        "eval": {
            "rollout_mode": rollout_mode,
            "max_rollout_turns": 1,
            "metric": "pass@k",
            "num_tests_per_prompt": 1,
            "seed": 42,
            "k_value": 1,
            "save_path": None,
            "save_full_gym_result": False,
        },
        "generation": {
            "backend": "vllm",
            "max_new_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "num_prompts_per_step": 1,
            "model_name": "test-model",
            "stop_token_ids": None,
            "stop_strings": None,
            "vllm_cfg": {
                "async_engine": False,
                "expose_http_server": False,
                "max_model_len": 128,
            },
        },
        "tokenizer": {"name": "test-model"},
        "data": {"max_input_seq_length": 128, "dataset_name": "aime2024"},
        "env": {"math": {"num_workers": 1}},
        "cluster": {"gpus_per_node": 1, "num_nodes": 1},
    }


def _nemo_gym_master_config():
    config = _base_master_config(NEMO_GYM_ROLLOUT_MODE)
    config["eval"].update(
        {
            "max_rollout_turns": None,
            "metric": "mean_reward",
        }
    )
    config["generation"].update(
        {
            "top_k": None,
            "vllm_cfg": {
                "async_engine": True,
                "expose_http_server": True,
                "max_model_len": 128,
            },
        }
    )
    config["data"] = {
        "max_input_seq_length": None,
        "dataset_name": "NemoGymDataset",
        "data_path": "/tmp/example.jsonl",
        "processor": "nemo_gym_data_processor",
        "env_name": "nemo_gym",
    }
    config["env"] = {"nemo_gym": {"config_paths": ["example.yaml"]}}
    return config


def test_eval_pass_k_basic():
    """Test basic pass@k evaluation."""
    # Test case: 3 samples, 2 correct, k=1
    rewards = torch.tensor([1.0, 0.0, 1.0])
    num_tests_per_prompt = 3
    score = eval_pass_k(rewards, num_tests_per_prompt=num_tests_per_prompt, k=1)
    group_size = len(rewards) / num_tests_per_prompt
    average_score = score / group_size
    expected = 2 / 3
    assert isinstance(average_score, float)
    assert average_score == pytest.approx(expected, rel=1e-6)


def test_validate_single_turn_eval_config():
    """Test validation for the existing single-turn eval mode."""
    _validate_eval_config(_base_master_config())


def test_validate_single_turn_eval_rejects_multi_turn_limit():
    """Test that single-turn eval does not silently ignore multi-turn config."""
    config = _base_master_config()
    config["eval"]["max_rollout_turns"] = 2

    with pytest.raises(AssertionError, match="max_rollout_turns"):
        _validate_eval_config(config)


def test_validate_nemo_gym_eval_config():
    """Test validation for NeMo-Gym eval mode."""
    _validate_eval_config(_nemo_gym_master_config())


def test_get_eval_collate_fn_selects_rollout_mode_collator():
    """Test that NeMo-Gym eval uses the RL collator for Gym metadata fields."""
    assert _get_eval_collate_fn(SINGLE_TURN_ROLLOUT_MODE) is eval_collate_fn
    assert _get_eval_collate_fn(NEMO_GYM_ROLLOUT_MODE) is rl_collate_fn


def test_validate_nemo_gym_eval_requires_no_prompt_repeats():
    """Test that NeMo-Gym eval rejects driver-level prompt repeats."""
    config = _nemo_gym_master_config()
    config["eval"]["num_tests_per_prompt"] = 2

    with pytest.raises(AssertionError, match="num_tests_per_prompt"):
        _validate_eval_config(config)


def test_validate_nemo_gym_eval_requires_openai_server():
    """Test that NeMo-Gym eval requires exposed vLLM HTTP servers."""
    config = _nemo_gym_master_config()
    config["generation"]["vllm_cfg"]["expose_http_server"] = False

    with pytest.raises(AssertionError, match="expose_http_server"):
        _validate_eval_config(config)


def test_validate_nemo_gym_eval_requires_top_k_null():
    """Test that NeMo-Gym eval rejects top-k sampling."""
    config = _nemo_gym_master_config()
    config["generation"]["top_k"] = -1

    with pytest.raises(AssertionError, match="top-k"):
        _validate_eval_config(config)


def test_score_batch_rewards_mean_reward():
    """Test mean_reward scoring sums batch rewards before dataset averaging."""
    rewards = torch.tensor([1.5, -0.5, 2.0])

    assert _score_batch_rewards(rewards, "mean_reward", 1, 1) == pytest.approx(3.0)


def test_collect_nemo_gym_evaluation_data_controls_full_result():
    """Test saved NeMo-Gym eval rows include full results only when requested."""
    batch = BatchedDataDict(
        {
            "extra_env_info": [
                {"responses_create_params": {"input": []}, "agent_ref": {"name": "a"}}
            ]
        }
    )
    rollout_result = AsyncNemoGymRolloutResult(
        input_ids=torch.tensor([[1, 2, 3]]),
        final_batch=BatchedDataDict(
            {
                "message_log": [
                    [{"role": "assistant", "token_ids": torch.tensor([3])}]
                ],
                "total_reward": torch.tensor([0.75]),
                "agent_ref": [{"name": "a"}],
            }
        ),
        rollout_metrics={},
        raw_results=[{"full_result": {"reward": 0.75, "details": {"ok": True}}}],
    )

    without_full_result = _collect_nemo_gym_evaluation_data(
        batch, rollout_result, include_full_gym_result=False, start_sample_index=10
    )
    with_full_result = _collect_nemo_gym_evaluation_data(
        batch, rollout_result, include_full_gym_result=True, start_sample_index=10
    )

    assert without_full_result[0]["sample_index"] == 10
    assert "full_result" not in without_full_result[0]
    assert with_full_result[0]["full_result"] == {
        "reward": 0.75,
        "details": {"ok": True},
    }


def test_make_json_serializable_preserves_structured_tensors():
    """Test that saved eval data keeps nested structure instead of stringifying it."""
    value = {
        "message_log": [{"role": "assistant", "token_ids": torch.tensor([1, 2])}],
        "reward": torch.tensor(1.25),
    }

    serializable = _make_json_serializable(value)

    assert serializable["message_log"] == [{"role": "assistant", "token_ids": [1, 2]}]
    assert serializable["reward"] == pytest.approx(1.25)


def test_eval_pass_k_all_correct():
    """Test pass@k when all samples are correct."""
    rewards = torch.tensor([1.0, 1.0, 1.0])
    num_tests_per_prompt = 3
    score = eval_pass_k(rewards, num_tests_per_prompt=num_tests_per_prompt, k=1)
    group_size = len(rewards) / num_tests_per_prompt
    average_score = score / group_size
    expected = 1.0
    assert isinstance(average_score, float)
    assert average_score == pytest.approx(expected, rel=1e-6)


def test_eval_pass_k_none_correct():
    """Test pass@k when no samples are correct."""
    rewards = torch.tensor([0.0, 0.0, 0.0])
    num_tests_per_prompt = 3
    score = eval_pass_k(rewards, num_tests_per_prompt=num_tests_per_prompt, k=1)
    average_score = score / (len(rewards) / num_tests_per_prompt)
    expected = 0.0
    assert isinstance(average_score, float)
    assert average_score == pytest.approx(expected, rel=1e-6)


def test_eval_pass_k_multiple_groups():
    """Test pass@k with multiple groups."""
    # Two groups: [1,0,1] and [0,1,0]
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    num_tests_per_prompt = 3
    score = eval_pass_k(rewards, num_tests_per_prompt=num_tests_per_prompt, k=1)
    average_score = score / (len(rewards) / num_tests_per_prompt)
    expected = 0.5
    assert isinstance(average_score, float)
    assert average_score == pytest.approx(expected, rel=1e-6)


def test_eval_cons_k_basic():
    """Test basic cons@k evaluation."""
    rewards = torch.tensor([1.0, 0.0, 1.0])
    extracted_answers = ["A", "B", "A"]
    num_tests_per_prompt = 3
    group_size = len(rewards) / num_tests_per_prompt
    score = eval_cons_k(
        rewards,
        num_tests_per_prompt=num_tests_per_prompt,
        k=1,
        extracted_answers=extracted_answers,
    )
    average_score = score / group_size
    expected = 2 / 3
    assert isinstance(average_score, float)
    assert average_score == pytest.approx(expected, rel=1e-6)


def test_eval_cons_k_multiple_groups():
    """Test cons@k with multiple groups."""
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    num_tests_per_prompt = 5
    extracted_answers = [
        "Correct",
        "Wrong1",
        "Correct",
        "Wrong2",
        "Correct",
        "Wrong3",
        "Correct",
        "Wrong4",
        "Correct",
        "Wrong4",
    ]
    group_size = len(rewards) / num_tests_per_prompt
    score = eval_cons_k(
        rewards,
        num_tests_per_prompt=num_tests_per_prompt,
        k=3,
        extracted_answers=extracted_answers,
    )
    average_score = score / group_size

    """
    For the first group, the extracted answers are [Correct, Wrong1, Correct, Wrong2, Correct]
    When calculating unbiased estimate of cons@3(k=3), we need to consider the majority vote of all Combination(5, 3) = 10 cases.
    The 10 cases are:
    - Correct, Wrong1, Correct      Majority: Correct
    - Correct, Wrong1, Wrong2       Majority: Correct(Choose the first one when there is a tie)
    - Correct, Wrong1, Correct      Majority: Correct
    - Correct, Correct, Wrong2      Majority: Correct
    - Correct, Correct, Correct     Majority: Correct
    - Correct, Wrong2, Correct      Majority: Correct
    - Wrong1, Correct, Wrong2       Majority: Wrong1 (Choose the first one when there is a tie)
    - Wrong1, Correct, Correct      Majority: Correct
    - Wrong1, Wrong2, Correct       Majority: Wrong1 (Choose the first one when there is a tie)
    - Correct, Wrong2, Correct      Majority: Correct
    The final result is 8/10.

    For the second group, the extracted answers are [Wrong3, Correct, Wrong4, Correct, Wrong4]
    When calculating unbiased estimate of cons@3(k=3), we need to consider the majority vote of all Combination(5, 3) = 10 cases.
    The 10 cases are:
    - Wrong3, Correct, Wrong4       Majority: Wrong3 (Choose the first one when there is a tie)
    - Wrong3, Correct, Correct      Majority: Correct
    - Wrong3, Correct, Wrong4       Majority: Wrong3 (Choose the first one when there is a tie)
    - Wrong3, Wrong4, Correct       Majority: Wrong3 (Choose the first one when there is a tie)
    - Wrong3, Wrong4, Wrong4        Majority: Wrong4
    - Wrong3, Correct, Wrong4       Majority: Wrong3 (Choose the first one when there is a tie)
    - Correct, Wrong4, Correct      Majority: Correct
    - Correct, Wrong4, Wrong4       Majority: Wrong4 (Choose the first one when there is a tie)
    - Correct, Correct, Wrong4      Majority: Correct
    - Wrong4, Correct, Wrong4       Majority: Wrong4
    The final result is 3/10.
    Since there len(rewards)/num_tests_per_prompt = 10/5 = 2 groups
    The final result is( 8/10 + 3/10 ) / 2 = 11/20 = 0.55
    """
    expected = 11 / 20
    assert isinstance(average_score, float)
    assert average_score == pytest.approx(expected, rel=1e-6)
