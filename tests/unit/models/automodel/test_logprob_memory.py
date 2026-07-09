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

try:
    import nemo_automodel  # noqa: F401
except ImportError:
    pytest.skip("nemo_automodel not available", allow_module_level=True)

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.models.automodel.train import LogprobsPostProcessor


def _make_processor(
    *,
    chunk_size: int | None,
    sampling_params: TrainingSamplingParams | None = None,
) -> LogprobsPostProcessor:
    return LogprobsPostProcessor(
        cfg={"logprob_chunk_size": chunk_size},
        device_mesh=None,
        cp_mesh=None,
        tp_mesh=None,
        cp_size=1,
        sampling_params=sampling_params,
    )


def _full_selected_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    sampling_params: TrainingSamplingParams | None = None,
) -> torch.Tensor:
    processor = _make_processor(chunk_size=None, sampling_params=sampling_params)
    return processor._compute_local_logprobs(logits.clone(), input_ids)


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 16])
def test_local_logprobs_chunking_matches_full_selected_logprobs(chunk_size):
    torch.manual_seed(7)
    logits = torch.randn(2, 5, 11)
    input_ids = torch.tensor(
        [
            [0, 2, 4, 6, 8],
            [1, 3, 5, 7, 9],
        ]
    )

    processor = _make_processor(chunk_size=chunk_size)
    actual = processor._compute_local_logprobs(logits.clone(), input_ids)
    expected = _full_selected_logprobs(logits, input_ids)

    assert actual.shape == (2, 4)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("chunk_size", [None, 2])
def test_local_logprobs_accepts_cpu_input_ids_with_cuda_logits(chunk_size):
    torch.manual_seed(13)
    logits = torch.randn(2, 5, 11, device="cuda")
    input_ids = torch.tensor(
        [
            [0, 2, 4, 6, 8],
            [1, 3, 5, 7, 9],
        ]
    )

    processor = _make_processor(chunk_size=chunk_size)
    actual = processor._compute_local_logprobs(logits.clone(), input_ids)

    next_tokens = input_ids[:, 1:].to(logits.device)
    expected_logits = logits[:, : next_tokens.shape[1], :].to(torch.float32)
    expected = torch.nn.functional.log_softmax(expected_logits, dim=-1).gather(
        dim=-1,
        index=next_tokens.unsqueeze(-1),
    ).squeeze(-1)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_local_logprobs_empty_sequence_returns_empty_tensor(chunk_size):
    processor = _make_processor(chunk_size=chunk_size)
    logits = torch.randn(2, 1, 11)
    input_ids = torch.tensor([[0], [1]])

    actual = processor._compute_local_logprobs(logits, input_ids)

    assert actual.shape == (2, 0)
    assert actual.dtype == torch.float32


def test_local_logprobs_chunking_matches_full_selected_logprobs_bfloat16():
    torch.manual_seed(11)
    logits = torch.randn(2, 4, 9).to(torch.bfloat16)
    input_ids = torch.tensor(
        [
            [0, 2, 4, 6],
            [1, 3, 5, 7],
        ]
    )

    processor = _make_processor(chunk_size=2)
    actual = processor._compute_local_logprobs(logits.clone(), input_ids)
    expected = _full_selected_logprobs(logits, input_ids)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_local_logprobs_chunking_preserves_top_k_top_p_filtering():
    torch.manual_seed(23)
    logits = torch.randn(2, 5, 13)
    input_ids = torch.tensor(
        [
            [0, 2, 4, 6, 8],
            [1, 3, 5, 7, 9],
        ]
    )
    sampling_params = TrainingSamplingParams(
        temperature=1.0,
        top_k=4,
        top_p=0.75,
    )

    chunked_processor = _make_processor(
        chunk_size=2,
        sampling_params=sampling_params,
    )
    actual = chunked_processor._compute_local_logprobs(logits.clone(), input_ids)
    expected = _full_selected_logprobs(
        logits,
        input_ids,
        sampling_params=sampling_params,
    )

    torch.testing.assert_close(actual, expected)


def test_local_logprobs_chunking_limits_log_softmax_sequence_size(monkeypatch):
    torch.manual_seed(31)
    logits = torch.randn(2, 7, 17)
    input_ids = torch.tensor(
        [
            [0, 2, 4, 6, 8, 10, 12],
            [1, 3, 5, 7, 9, 11, 13],
        ]
    )
    seen_shapes = []
    original_log_softmax = torch.nn.functional.log_softmax

    def log_softmax(input_tensor, *args, **kwargs):
        seen_shapes.append(tuple(input_tensor.shape))
        return original_log_softmax(input_tensor, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "log_softmax", log_softmax)

    processor = _make_processor(chunk_size=2)
    actual = processor._compute_local_logprobs(logits, input_ids)

    assert actual.shape == (2, 6)
    assert seen_shapes == [(2, 2, 17), (2, 2, 17), (2, 2, 17)]
