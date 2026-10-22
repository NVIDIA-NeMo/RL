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

"""Dense padding must survive successive context and sequence parallel splits."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank
from nemo_rl.models.megatron import data as megatron_data


def dense_config(cp_size: int, tp_size: int, sequence_parallel: bool) -> dict:
    return {
        "make_sequence_length_divisible_by": 16,
        "dynamic_batching": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "megatron_cfg": {
            "context_parallel_size": cp_size,
            "tensor_model_parallel_size": tp_size,
            "sequence_parallel": sequence_parallel,
            "fp8_cfg": {"enabled": False},
        },
    }


@pytest.mark.mcore
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_dense_schedule_aligns_successive_parallel_splits(
    cp_size: int, sequence_parallel: bool
) -> None:
    """The pipeline schedule and actual microbatch use the same safe width."""
    data = MagicMock()
    data.size = 1
    data.make_microbatch_iterator.return_value = iter(())
    cfg = dense_config(cp_size, 8, sequence_parallel)
    with (
        patch.object(
            megatron_data, "get_and_validate_seqlen", return_value=(1, 173825)
        ),
        patch.object(
            megatron_data, "make_processed_microbatch_iterator"
        ) as make_iterator,
    ):
        *_, original_width, padded_width = megatron_data.get_microbatch_iterator(
            data, cfg, mbs=1, straggler_timer=MagicMock()
        )
    factor = make_iterator.call_args.kwargs["pad_individual_seqs_to_multiple_of"]
    assert original_width == 173825
    assert padded_width == ((original_width + factor - 1) // factor) * factor
    assert padded_width % cp_size == 0
    if sequence_parallel:
        # CP slicing happens before TP's sequence-parallel scatter.
        assert (padded_width // cp_size) % 8 == 0


@pytest.mark.mcore
@pytest.mark.parametrize("cp_size,tp_size", [(1, 8), (2, 8), (4, 8), (4, 2)])
@pytest.mark.parametrize("sequence_length", [17, 65, 173825])
def test_dense_padding_preserves_active_payload_and_cp_sp_shards(
    cp_size: int, tp_size: int, sequence_length: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Odd-length multimodal rows keep all original targets and router replay."""
    cfg = dense_config(cp_size, tp_size, True)
    factor = megatron_data._get_non_packed_sequence_pad_factor(cfg)
    tokens = torch.arange(sequence_length).unsqueeze(0)
    original = {
        "input_ids": tokens,
        "input_lengths": torch.tensor([sequence_length]),
        "token_mask": (tokens % 3 != 0).float(),
        "generation_logprobs": -tokens.float() / sequence_length,
        "routed_experts": torch.arange(4, dtype=torch.int32)
        .reshape(1, 1, 2, 2)
        .expand(1, sequence_length, 2, 2)
        .clone(),
        "media_token_validity_mask": tokens % 5 == 0,
        "pixel_values": torch.arange(24).reshape(1, 3, 2, 4),
    }
    batch = {name: tensor.clone() for name, tensor in original.items()}
    monkeypatch.setenv("NRL_MEGATRON_SKIP_DENSE_ATTENTION_MASK", "1")
    with (
        patch.object(megatron_data, "get_context_parallel_rank", return_value=0),
        patch.object(
            megatron_data, "get_context_parallel_world_size", return_value=cp_size
        ),
    ):
        processed = megatron_data.process_microbatch(
            batch,
            pad_individual_seqs_to_multiple_of=factor,
            model_slices_context_parallel_inputs=True,
        )
    assert processed.original_seq_length == sequence_length
    assert torch.equal(batch["input_lengths"], original["input_lengths"])
    assert torch.equal(batch["pixel_values"], original["pixel_values"])
    for name in (
        "input_ids",
        "token_mask",
        "generation_logprobs",
        "routed_experts",
        "media_token_validity_mask",
    ):
        assert torch.equal(batch[name][:, :sequence_length], original[name]), name
    assert torch.count_nonzero(batch["token_mask"][:, sequence_length:]) == 0
    assert (
        torch.count_nonzero(batch["media_token_validity_mask"][:, sequence_length:])
        == 0
    )
    # All mirrored CP chunks must remain divisible by TP, as in packed batches.
    width = processed.input_ids.shape[1]
    assert width % ((2 * cp_size if cp_size > 1 else 1) * tp_size) == 0
    indices = torch.arange(width).unsqueeze(0)
    all_indices = []
    for cp_rank in range(cp_size):
        local = _get_tokens_on_this_cp_rank(indices, cp_rank, cp_size)
        assert local.shape[1] % tp_size == 0
        assert all(
            part.shape[1] == local.shape[1] // tp_size
            for part in local.chunk(tp_size, 1)
        )
        all_indices.append(local.flatten())
    assert torch.equal(torch.cat(all_indices).sort().values, torch.arange(width))
