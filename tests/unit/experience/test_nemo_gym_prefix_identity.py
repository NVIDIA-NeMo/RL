# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import uuid

from nemo_rl.experience.rollout_manager import AsyncNemoGymRolloutImpl


def _impl() -> AsyncNemoGymRolloutImpl:
    return AsyncNemoGymRolloutImpl(
        tokenizer=None,
        task_to_env={},
        num_generations_per_prompt=2,
        max_seq_len=100,
        max_rollout_turns=1,
        generation_config={
            "temperature": 1.0,
            "top_p": 1.0,
            "max_new_tokens": 100,
            "stop_strings": None,
            "stop_token_ids": None,
            "top_k": None,
        },
    )


def test_nemo_gym_inline_prefix_rollout_ids_are_unique_across_dispatches() -> None:
    impl = _impl()
    impl._num_generations_per_prompt = 2
    input_sample = {"extra_env_info": {"responses_create_params": {}}}

    rows = impl._build_inputs(input_sample) + impl._build_inputs(input_sample)

    rollout_ids = [row["_ng_rollout_id"] for row in rows]
    assert len(set(rollout_ids)) == 4
    assert all(uuid.UUID(rollout_id).hex == rollout_id for rollout_id in rollout_ids)
    assert "_ng_rollout_id" not in input_sample["extra_env_info"]


def test_nemo_gym_build_inputs_preserves_controller_capture_ids() -> None:
    impl = _impl()
    impl._num_generations_per_prompt = 2
    input_sample = {"extra_env_info": {"responses_create_params": {}}}

    rows = impl._build_inputs(input_sample, rollout_ids=["capture-0", "capture-1"])

    assert [row["_ng_rollout_id"] for row in rows] == ["capture-0", "capture-1"]
