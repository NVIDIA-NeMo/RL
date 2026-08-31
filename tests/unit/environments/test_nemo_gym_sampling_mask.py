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

import pytest
import torch

from nemo_rl.environments.nemo_gym import NemoGym


class _Tokenizer:
    def batch_decode(self, batch):
        return [" ".join(map(str, token_ids)) for token_ids in batch]


def _gym_result(*, include_mask: bool = True) -> dict:
    output = {
        "prompt_token_ids": [1, 2],
        "generation_token_ids": [3, 4],
        "generation_log_probs": [-0.1, -0.2],
    }
    if include_mask:
        output["sampling_mask"] = [[3, 9], [4]]
    return {
        "response": {"output": [output]},
        "responses_create_params": {"input": []},
    }


def test_nemo_gym_postprocess_tensorizes_sampling_mask_on_assistant():
    class _MockSelf:
        cfg = {"require_sampling_mask": True, "sampling_mask_top_k": 2}

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), {}, _gym_result(), _Tokenizer()
        )
    )

    user, assistant = result["message_log"]
    assert "sampling_mask_token_ids" not in user
    assert torch.equal(
        assistant["sampling_mask_token_ids"],
        torch.tensor([[3, 9], [4, 0]], dtype=torch.int32),
    )
    assert assistant["sampling_mask_sizes"].tolist() == [2, 1]


def test_nemo_gym_postprocess_requires_sampling_mask_when_enabled():
    class _MockSelf:
        cfg = {"require_sampling_mask": True, "sampling_mask_top_k": 2}

    with pytest.raises(ValueError, match="include sampling_mask"):
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), {}, _gym_result(include_mask=False), _Tokenizer()
        )
