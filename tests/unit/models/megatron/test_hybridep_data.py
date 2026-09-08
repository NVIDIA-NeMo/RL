# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for HybridEP packed-sequence data processing."""

from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.mark.mcore
def test_hybridep_prepads_packed_inputs_before_model_forward():
    from megatron.core.packed_seq_params import PackedSeqParams

    from nemo_rl.models.megatron import hybridep

    def set_group_max(target, **_kwargs):
        target.fill_(14)

    input_ids = torch.tensor([[11, 12, 13, 0, 21, 22, 23, 24, 25, 0, 0, 0]])
    cu_seqlens_padded = torch.tensor([0, 4, 12], dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens_padded,
        cu_seqlens_kv=cu_seqlens_padded,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=8,
        max_seqlen_kv=8,
        qkv_format="thd",
        total_tokens=12,
    )

    with (
        patch.object(
            hybridep,
            "get_expert_tensor_and_model_parallel_group",
            return_value=MagicMock(),
        ) as mock_get_group,
        patch.object(
            hybridep.torch.distributed,
            "all_reduce",
            side_effect=set_group_max,
        ) as mock_all_reduce,
        patch(
            "nemo_rl.models.megatron.hybridep.torch.distributed.is_available",
            return_value=True,
        ),
        patch(
            "nemo_rl.models.megatron.hybridep.torch.distributed.is_initialized",
            return_value=True,
        ),
    ):
        (
            padded_input_ids,
            padded_local_input_ids,
            padded_params,
            padded_cu_seqlens,
        ) = hybridep.pad_packed_seq_for_hybridep(
            input_ids=input_ids,
            input_ids_cp_sharded=input_ids,
            packed_seq_params=packed_seq_params,
            cu_seqlens_padded=cu_seqlens_padded,
            pad_packed_seq_to_multiple_of=8,
            cp_rank=0,
            cp_size=1,
        )

    assert padded_input_ids.shape == (1, 16)
    assert padded_local_input_ids.shape == (1, 16)
    assert torch.equal(padded_input_ids[:, :12], input_ids)
    assert torch.count_nonzero(padded_input_ids[:, 12:]) == 0
    assert torch.equal(padded_cu_seqlens, torch.tensor([0, 4, 16]))
    assert padded_params.total_tokens == 16
    mock_get_group.assert_called_once_with(check_initialized=False)
    mock_all_reduce.assert_called_once()


@pytest.mark.mcore
def test_hybridep_prepadding_rejects_missing_alignment_group():
    from nemo_rl.models.megatron import hybridep

    with (
        patch.object(
            hybridep,
            "get_expert_tensor_and_model_parallel_group",
            return_value=None,
        ),
        patch.object(hybridep.torch.distributed, "all_reduce") as mock_all_reduce,
        patch(
            "nemo_rl.models.megatron.hybridep.torch.distributed.is_available",
            return_value=True,
        ),
        patch(
            "nemo_rl.models.megatron.hybridep.torch.distributed.is_initialized",
            return_value=True,
        ),
        pytest.raises(
            RuntimeError, match="HybridEP alignment group is not initialized"
        ),
    ):
        hybridep._get_hybridep_aligned_seq_len(
            local_seq_len=12,
            multiple=8,
            device=torch.device("cpu"),
        )

    mock_all_reduce.assert_not_called()


@pytest.mark.mcore
@pytest.mark.parametrize(
    ("cp_rank", "input_ids_cp_sharded", "expected_padded_local_input_ids"),
    [
        (
            0,
            torch.tensor([[1, 2, 3, 4, 13, 14, 15, 16]]),
            torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0]]),
        ),
        (
            1,
            torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12]]),
            torch.tensor([[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0]]),
        ),
    ],
)
def test_hybridep_prepadding_preserves_cp_zigzag_layout(
    cp_rank: int,
    input_ids_cp_sharded: torch.Tensor,
    expected_padded_local_input_ids: torch.Tensor,
) -> None:
    from megatron.core.packed_seq_params import PackedSeqParams

    from nemo_rl.models.megatron import hybridep

    def set_group_max(target, **_kwargs):
        target.fill_(10)

    input_ids = torch.arange(1, 17).view(1, 16)
    cu_seqlens_padded = torch.tensor([0, 16], dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens_padded,
        cu_seqlens_kv=cu_seqlens_padded,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=16,
        max_seqlen_kv=16,
        qkv_format="thd",
        total_tokens=8,
    )

    with (
        patch.object(
            hybridep,
            "get_expert_tensor_and_model_parallel_group",
            return_value=MagicMock(),
        ),
        patch.object(
            hybridep.torch.distributed,
            "all_reduce",
            side_effect=set_group_max,
        ),
        patch(
            "nemo_rl.models.megatron.hybridep.torch.distributed.is_available",
            return_value=True,
        ),
        patch(
            "nemo_rl.models.megatron.hybridep.torch.distributed.is_initialized",
            return_value=True,
        ),
    ):
        (
            padded_input_ids,
            padded_local_input_ids,
            padded_params,
            padded_cu_seqlens,
        ) = hybridep.pad_packed_seq_for_hybridep(
            input_ids=input_ids,
            input_ids_cp_sharded=input_ids_cp_sharded,
            packed_seq_params=packed_seq_params,
            cu_seqlens_padded=cu_seqlens_padded,
            pad_packed_seq_to_multiple_of=8,
            cp_rank=cp_rank,
            cp_size=2,
        )

    assert padded_input_ids.shape == (1, 24)
    assert torch.equal(padded_input_ids[:, :16], input_ids)
    assert torch.count_nonzero(padded_input_ids[:, 16:]) == 0
    assert torch.equal(padded_local_input_ids, expected_padded_local_input_ids)
    assert torch.equal(padded_cu_seqlens, torch.tensor([0, 24]))
    assert padded_params.max_seqlen_q == 24
    assert padded_params.total_tokens == 12


@pytest.mark.mcore
def test_hybridep_prepadding_returns_original_objects_when_already_aligned() -> None:
    from megatron.core.packed_seq_params import PackedSeqParams

    from nemo_rl.models.megatron import hybridep

    input_ids = torch.arange(1, 9).view(1, 8)
    cu_seqlens_padded = torch.tensor([0, 8], dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens_padded,
        cu_seqlens_kv=cu_seqlens_padded,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=8,
        max_seqlen_kv=8,
        qkv_format="thd",
        total_tokens=8,
    )

    with patch.object(hybridep, "_get_hybridep_aligned_seq_len", return_value=8):
        result = hybridep.pad_packed_seq_for_hybridep(
            input_ids=input_ids,
            input_ids_cp_sharded=input_ids,
            packed_seq_params=packed_seq_params,
            cu_seqlens_padded=cu_seqlens_padded,
            pad_packed_seq_to_multiple_of=8,
            cp_rank=0,
            cp_size=1,
        )

    assert result[0] is input_ids
    assert result[1] is input_ids
    assert result[2] is packed_seq_params
    assert result[3] is cu_seqlens_padded


@pytest.mark.mcore
@patch("nemo_rl.models.megatron.data.get_context_parallel_rank", return_value=0)
@patch("nemo_rl.models.megatron.data.get_context_parallel_world_size", return_value=2)
@patch(
    "nemo_rl.models.megatron.data.get_packed_seq_cp_partition_indices",
    return_value=torch.tensor([0, 3, 4, 5, 10, 11]),
)
@patch("nemo_rl.models.megatron.data._pack_sequences_for_megatron")
def test_hybridep_padding_mask_preserves_existing_cp_local_layout(
    mock_pack, mock_indices, mock_cp_world, mock_cp_rank
):
    """HybridEP masks fake tokens without adding NeMo-level dispatch padding."""
    from megatron.core.packed_seq_params import PackedSeqParams

    from nemo_rl.models.megatron.data import process_microbatch

    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)
    cu_seqlens_padded = torch.tensor([0, 4, 12], dtype=torch.int32)
    input_ids = torch.tensor([[11, 12, 13, 0, 21, 22, 23, 24, 25, 0, 0, 0]])
    input_ids_cp_sharded = input_ids[:, [0, 3, 4, 5, 10, 11]]
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens_padded,
        cu_seqlens_kv=cu_seqlens_padded,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=8,
        max_seqlen_kv=8,
        qkv_format="thd",
        total_tokens=input_ids_cp_sharded.shape[1],
    )
    mock_pack.return_value = (
        input_ids,
        input_ids_cp_sharded,
        packed_seq_params,
        cu_seqlens,
        cu_seqlens_padded,
    )
    data_dict = {
        "input_ids": torch.tensor([[11, 12, 13, 0, 0], [21, 22, 23, 24, 25]]),
        "input_lengths": torch.tensor([3, 5]),
    }

    result = process_microbatch(
        data_dict,
        seq_length_key="input_lengths",
        pack_sequences=True,
        create_packed_seq_padding_mask=True,
        straggler_timer=MagicMock(),
    )

    assert torch.equal(result.input_ids, input_ids)
    assert torch.equal(result.input_ids_cp_sharded, input_ids_cp_sharded)
    assert torch.equal(result.cu_seqlens_padded, cu_seqlens_padded)
    assert torch.equal(
        result.padding_mask,
        torch.tensor([[False, True, False, False, True, True]]),
    )
    mock_indices.assert_called_once_with(
        packed_seq_params,
        total_tokens=input_ids.shape[1],
        cp_size=2,
        cp_rank=0,
        device=input_ids.device,
    )


@pytest.mark.mcore
def test_hybridep_padding_mask_rejects_model_owned_cp_slicing():
    """Do not silently drop the mask in models that own CP input slicing."""
    from nemo_rl.models.megatron.data import process_microbatch

    with pytest.raises(
        NotImplementedError,
        match="context-parallel input slicing internally",
    ):
        process_microbatch(
            {},
            pack_sequences=True,
            model_slices_context_parallel_inputs=True,
            create_packed_seq_padding_mask=True,
        )
