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

from types import SimpleNamespace

import orjson
import pytest
import torch

import nemo_rl.experience.rollouts as rollouts_mod
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@pytest.mark.parametrize("existing_indices", [None, [7, 7, 2, 2]])
def test_prepare_legacy_gym_rows_supplies_task_indices(existing_indices):
    rows = [{"responses_create_params": {}} for _ in range(4)]
    if existing_indices is not None:
        for row, index in zip(rows, existing_indices):
            row["_ng_task_index"] = index
    rollouts_mod._prepare_nemo_gym_rows(
        rows,
        generation_config={"max_new_tokens": 64},
        sampling_params=SimpleNamespace(temperature=1.0, top_p=1.0),
        num_generations=2,
    )
    indices = [row["_ng_task_index"] for row in rows]
    if existing_indices is not None:
        assert indices == existing_indices
    else:
        assert orjson.loads(orjson.dumps(rows)) == rows
        assert all(type(index) is int and 0 <= index < (1 << 63) for index in indices)
        assert indices[0] == indices[1]
        assert indices[2] == indices[3]
        assert indices[0] != indices[2]
        next_rows = [{"responses_create_params": {}} for _ in range(2)]
        rollouts_mod._prepare_nemo_gym_rows(
            next_rows,
            generation_config={"max_new_tokens": 64},
            sampling_params=SimpleNamespace(temperature=1.0, top_p=1.0),
            num_generations=2,
        )
        assert next_rows[0]["_ng_task_index"] not in indices


@pytest.mark.parametrize("task_indices", [(7, 7, 2, 2), (7, 8, 2, 2)])
def test_postprocess_sync_nemo_gym_batch_validates_task_indices_per_prompt(
    task_indices,
):
    rows = [
        {"agent_ref": {"name": "agent"}, "_ng_task_index": index}
        for index in task_indices
    ]
    results = []
    for _ in rows:
        prompt = {"role": "user", "content": "prompt", "token_ids": torch.tensor([1])}
        results.append(
            {
                "input_message_log": [prompt],
                "message_log": [
                    prompt,
                    {
                        "role": "assistant",
                        "content": "answer",
                        "token_ids": torch.tensor([2]),
                        "generation_logprobs": torch.tensor([-0.1]),
                    },
                ],
                "full_result": {"reward": 1.0},
            }
        )

    def postprocess() -> rollouts_mod.NemoGymRolloutResult:
        return rollouts_mod._postprocess_single_nemo_gym_group(
            nemo_gym_rows=rows,
            results=results,
            timer=rollouts_mod.Timer(),
            timer_prefix="timing/rollout",
            policy_generation=SimpleNamespace(
                cfg={"backend": "coreweave", "coreweave_cfg": {"max_model_len": 128}}
            ),
            input_batch=BatchedDataDict({"loss_multiplier": torch.ones(4)}),
            tokenizer=SimpleNamespace(pad_token_id=0),
            log_full_result_tables=False,
            group_size=2,
        )

    if task_indices[0] != task_indices[1]:
        with pytest.raises(ValueError, match="Expected one _ng_task_index"):
            postprocess()
    else:
        result = postprocess()
        assert result.task_index is None
        assert result.final_batch["total_reward"].tolist() == [1.0] * 4
