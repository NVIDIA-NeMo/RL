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


from types import SimpleNamespace

import pytest
import torch

from nemo_rl.evals.eval import (
    eval_cons_k,
    eval_pass_k,
    run_env_eval,
)


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


@pytest.mark.parametrize(
    ("generation_config", "expected_use_async"),
    [
        ({"backend": "sglang"}, False),
        ({"backend": "sglang", "use_async_rollouts": False}, False),
        ({"backend": "sglang", "use_async_rollouts": True}, True),
        ({"backend": "vllm", "vllm_cfg": {"async_engine": False}}, False),
        ({"backend": "vllm", "vllm_cfg": {"async_engine": True}}, True),
    ],
)
def test_run_env_eval_selects_backend_specific_async_path(
    monkeypatch, generation_config, expected_use_async
):
    captured = {}

    async def fake_run_env_eval_impl(
        vllm_generation, dataloader, env, master_config, use_async=False
    ):
        captured["use_async"] = use_async

    monkeypatch.setattr("nemo_rl.evals.eval._run_env_eval_impl", fake_run_env_eval_impl)

    run_env_eval(
        vllm_generation=object(),
        dataloader=object(),
        env=object(),
        master_config=SimpleNamespace(generation=generation_config),
    )

    assert captured["use_async"] is expected_use_async


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


def test_run_env_eval_impl_builds_vllm_multi_modal_prompts(monkeypatch):
    """The eval prompt loop must forward per-row ``multi_modal_data`` to vLLM.

    Row 0 is a VLM row: ``vlm_hf_data_processor`` leaves ``content`` as a list of
    typed parts, so anything that unconditionally joins it raises ``TypeError``.
    Row 1 is text-only and must fall back to the joined message_log text.
    """
    import asyncio

    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.evals.eval import _run_env_eval_impl

    image = object()
    batch = BatchedDataDict(
        {
            "message_log": [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "describe the image"},
                        ],
                    }
                ],
                [{"role": "user", "content": "plain"}],
            ],
            "extra_env_info": [{"ground_truth": "a"}, {"ground_truth": "b"}],
            "vllm_content": ["<image> describe the image", None],
            "vllm_multi_modal_data": [{"image": image}, {}],
        }
    )

    captured = {}

    async def fake_generate_texts(vllm_generation, inputs, use_async):
        captured["prompts"] = list(inputs["prompts"])
        return ["out-0", "out-1"]

    monkeypatch.setattr("nemo_rl.evals.eval._generate_texts", fake_generate_texts)
    monkeypatch.setattr("nemo_rl.evals.eval._print_results", lambda *a, **k: None)
    monkeypatch.setattr("nemo_rl.evals.eval.ray.get", lambda value: value)

    env = SimpleNamespace(
        step=SimpleNamespace(
            remote=lambda *args: SimpleNamespace(rewards=torch.tensor([1.0, 0.0]))
        ),
        shutdown=SimpleNamespace(remote=lambda: None),
    )
    master_config = SimpleNamespace(
        generation={"backend": "vllm"},
        eval={
            "metric": "pass@k",
            "num_tests_per_prompt": 1,
            "k_value": 1,
            "save_path": None,
        },
    )

    class _SingleBatchDataloader(list):
        dataset = [0, 1]

    asyncio.run(
        _run_env_eval_impl(
            vllm_generation=SimpleNamespace(shutdown=lambda: None),
            dataloader=_SingleBatchDataloader([batch]),
            env=env,
            master_config=master_config,
        )
    )

    prompts = captured["prompts"]
    assert prompts[0] == {
        "prompt": "<image> describe the image",
        "multi_modal_data": {"image": image},
    }
    assert prompts[1] == "plain"
