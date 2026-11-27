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

"""[CUSTOM-ENV-MASK-SAMPLE] Direct, fixture-light coverage of
AsyncNemoGymRolloutImpl._result_to_completion's `truncated` computation, which now folds
in any environment-reported `mask_sample` signal alongside the existing token-length
check -- see rollout_manager.py's own comment for the full rationale (some NeMo-Gym
environments, e.g. swe_agents, report failure modes `truncated`'s token-count check can't
see: agent-internal max-iteration caps, eval/agent timeouts, a subprocess crash that
produces zero messages).
"""

from nemo_rl.experience.rollout_manager import AsyncNemoGymRolloutImpl


def _make_impl(max_seq_len: int = 100) -> AsyncNemoGymRolloutImpl:
    """Construct a minimal, real AsyncNemoGymRolloutImpl -- tokenizer/task_to_env are
    unused by _result_to_completion itself, so plain placeholders are enough; the
    GenerationConfig fields _validate_init_params checks (stop_strings/stop_token_ids/
    top_k) must be falsy, and max_rollout_turns must be None, per its own asserts.
    """
    return AsyncNemoGymRolloutImpl(
        tokenizer=None,
        task_to_env={},
        num_generations_per_prompt=1,
        max_seq_len=max_seq_len,
        generation_config={
            "backend": "vllm",
            "max_new_tokens": 10,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "stop_token_ids": None,
            "stop_strings": None,
        },
        max_rollout_turns=None,
    )


def _make_result(token_counts: list[int], mask_sample: bool | None = None) -> dict:
    message_log = [{"role": "assistant", "token_ids": [0] * n} for n in token_counts]
    full_result = {"reward": 0.0}
    if mask_sample is not None:
        full_result["mask_sample"] = mask_sample
    return {
        "input_message_log": [],
        "message_log": message_log,
        "full_result": full_result,
    }


class TestResultToCompletionTruncated:
    def test_neither_length_nor_mask_sample_not_truncated(self) -> None:
        impl = _make_impl(max_seq_len=100)
        result = _make_result([10, 10], mask_sample=False)

        completion = impl._result_to_completion(result)

        assert completion.truncated is False

    def test_length_exactly_max_seq_len_is_truncated(self) -> None:
        impl = _make_impl(max_seq_len=20)
        result = _make_result([10, 10])  # sums to exactly max_seq_len

        completion = impl._result_to_completion(result)

        assert completion.truncated is True

    def test_env_mask_sample_true_is_truncated_even_under_length_budget(self) -> None:
        # The exact bug this fixes: swe_agents' own mask_sample (e.g. a lost-trajectory
        # crash) must exclude the sample from the loss even when nowhere near max_seq_len.
        impl = _make_impl(max_seq_len=10_000)
        result = _make_result([10, 10], mask_sample=True)

        completion = impl._result_to_completion(result)

        assert completion.truncated is True

    def test_missing_mask_sample_field_defaults_to_not_truncated_by_it(self) -> None:
        # Most environments (e.g. harbor_agent today) don't set this field at all --
        # must be a no-op for them, not an error or an accidental always-True.
        impl = _make_impl(max_seq_len=10_000)
        result = _make_result([10, 10])  # no "mask_sample" key at all

        completion = impl._result_to_completion(result)

        assert completion.truncated is False

    def test_reward_passed_through(self) -> None:
        impl = _make_impl(max_seq_len=10_000)
        result = _make_result([10, 10])
        result["full_result"]["reward"] = 0.75

        completion = impl._result_to_completion(result)

        assert completion.reward == 0.75
