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

import numpy as np
import pytest
import torch
from PIL import Image

from nemo_rl.environments.nemotron_utils import (
    _resize_and_normalize_nemotron_video_frame,
    placeholder_runs_from_token_ids,
    verify_static_video_media_alignment,
)


def test_placeholder_runs_preserve_per_region_counts() -> None:
    assert placeholder_runs_from_token_ids(
        [10, 98, 99, 99, 97, 11, 98, 99, 99, 99, 97],
        (99, 98, 97),
    ) == [2, 3]


def test_static_video_alignment_rejects_different_rollout_grid() -> None:
    class _Tokenizer:
        unk_token_id = None

        @staticmethod
        def convert_tokens_to_ids(tokens):
            mapping = {"<image>": 99, "<img>": 98, "</img>": 97}
            return [mapping[token] for token in tokens]

    source = {
        "token_ids": torch.tensor([98, 99, 99, 99, 97]),
        "num_frames": SimpleNamespace(tensors=[torch.tensor([2])]),
        "imgs_sizes": SimpleNamespace(tensors=[torch.tensor([[2, 2]])]),
    }
    target = {"token_ids": torch.tensor([98, 99, 99, 99, 99, 97])}

    with pytest.raises(ValueError, match="3 placeholder tokens total.*4"):
        verify_static_video_media_alignment(source, target, _Tokenizer())


@pytest.mark.vllm
def test_policy_video_resize_matches_stock_vllm() -> None:
    """Keep policy pixels numerically aligned with unmodified vLLM 0.25.1."""
    stock_processor = pytest.importorskip(
        "vllm.transformers_utils.processors.nano_nemotron_vl"
    )
    stock_resize = stock_processor._bicubic_resize_and_normalize
    stock_resize = getattr(stock_resize, "_torchdynamo_orig_callable", stock_resize)

    frame_array = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    frame = Image.fromarray(frame_array, mode="RGB")
    norm_mean = torch.tensor([0.5, 0.4, 0.3], dtype=torch.float32).view(3, 1, 1)
    norm_std = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32).view(3, 1, 1)

    actual = _resize_and_normalize_nemotron_video_frame(
        frame,
        target_height=8,
        target_width=12,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    expected = stock_resize(
        torch.from_numpy(np.expand_dims(frame_array, axis=0)),
        size=(8, 12),
        norm_mean=norm_mean.unsqueeze(0),
        norm_std=norm_std.unsqueeze(0),
        dtype=torch.float32,
    ).squeeze(0)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
