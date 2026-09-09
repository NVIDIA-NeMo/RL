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
from PIL import Image

import nemo_rl.environments.nemo_gym as nemo_gym_module
from nemo_rl.data.multimodal_utils import ROLLOUT_MATCHED_MEDIA_KEY, PackedTensor
from nemo_rl.environments.nemo_gym import NemoGym
from nemo_rl.environments.nemotron_utils import (
    RolloutGeometryUnderdetermined,
    _process_single_image_at_num_tokens,
)


class _Tokenizer:
    def batch_decode(self, batch):
        return [" ".join(map(str, token_ids)) for token_ids in batch]


class _PinnedMismatchImageProcessor:
    """Image processor whose pinned run never reproduces the requested count."""

    max_model_len = 4096
    downsample_ratio = 0.5
    patch_size = 16


class _PinnedMismatchProcessor:
    image_token = "<image>"
    image_processor = _PinnedMismatchImageProcessor()

    def __call__(self, text, images, return_tensors):
        # Whatever the pinned budget asks for, produce one token too many, so
        # the count can never be reproduced without inferring a grid.
        return {"num_tokens": [self.image_processor.max_model_len - 4 + 1]}


def test_unreproducible_tiling_raises_instead_of_inferring_a_grid():
    processor = _PinnedMismatchProcessor()
    image = Image.new("RGB", (1152, 256))
    with pytest.raises(RolloutGeometryUnderdetermined, match="does not uniquely"):
        _process_single_image_at_num_tokens(processor, image, 288)
    # The pinned-budget mutation is rolled back even on failure.
    assert processor.image_processor.max_model_len == 4096


def test_geometry_failure_strips_media_masks_sample_and_marks_turns(monkeypatch):
    """A mid-sample geometry failure must not leave partial media behind.

    Turn 0 attaches media successfully; turn 1 fails. The sample must come
    back with no media on any turn (partial media would leave Megatron with
    fewer projected features than placeholder tokens), every user turn marked
    rollout-matched so the driver-side static reattach does not restore
    misaligned tensors, and the sample flagged for loss masking.
    """
    calls = {"n": 0}

    def fake_attach(user_message, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            user_message["pixel_values"] = PackedTensor(
                torch.ones(1, 1), dim_to_pack=0
            )
        else:
            raise RolloutGeometryUnderdetermined("count is ambiguous")

    monkeypatch.setattr(
        nemo_gym_module, "_attach_multimodal_data_to_user_message", fake_attach
    )

    nemo_gym_result = {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.1],
                },
                {
                    "prompt_token_ids": [1, 2, 3, 4, 5],
                    "generation_token_ids": [6, 7],
                    "generation_log_probs": [-0.2, -0.3],
                },
            ]
        },
        "responses_create_params": {"input": []},
    }

    class _MockSelf:
        cfg = {}
        _processor = object()

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), {}, nemo_gym_result, _Tokenizer()
        )
    )

    user_messages = [
        message for message in result["message_log"] if message["role"] == "user"
    ]
    assert len(user_messages) == 2
    for message in user_messages:
        assert not any(
            isinstance(value, PackedTensor) for value in message.values()
        ), "media must be stripped from every turn after a geometry failure"
        assert message[ROLLOUT_MATCHED_MEDIA_KEY] is True
    assert nemo_gym_result["instance_config"]["mask_sample"] is True
    # Only turns before the failure ever attached; the failing turn stopped it.
    assert calls["n"] == 2
