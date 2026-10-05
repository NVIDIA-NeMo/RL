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

"""Exact layout tests for router statistics, independent of prediction loss."""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.megatron.router_mask import get_router_padding_mask


def test_dense_lengths_and_artificial_rows():
    lengths = torch.tensor([3, 1, 4])
    artificial = torch.tensor([False, True, False])
    mask = get_router_padding_mask(lengths, 8, artificial_inputs=artificial)
    assert mask.tolist() == [
        [False, False, False, True, True, True, True, True],
        [True] * 8,
        [False, False, False, False, True, True, True, True],
    ]
    # Neither token IDs nor policy loss masks are inputs to this decision.
    assert (~mask).sum().item() == 7


def test_packed_gaps_and_borrowed_row():
    mask = get_router_padding_mask(
        torch.tensor([3, 4, 1]),
        16,
        artificial_inputs=torch.tensor([False, True, False]),
        cu_seqlens_padded=torch.tensor([0, 4, 8, 16]),
    )
    assert mask.tolist() == [
        [False, False, False, True] + [True] * 4 + [False] + [True] * 7
    ]


@pytest.mark.parametrize("artificial", [None, torch.tensor([False, False])])
def test_no_artificial_rows_preserves_every_real_token(artificial):
    mask = get_router_padding_mask(
        torch.tensor([2, 2]), 2, artificial_inputs=artificial
    )
    assert not mask.any()


def test_dummy_only_microbatch_excludes_every_token():
    mask = get_router_padding_mask(
        torch.tensor([1, 3]), 4, artificial_inputs=torch.ones(2, dtype=torch.bool)
    )
    assert mask.all()


def test_row_count_mismatch_fails():
    with pytest.raises(ValueError, match="one boolean per input"):
        get_router_padding_mask(
            torch.tensor([3, 1]), 4, artificial_inputs=torch.tensor([True])
        )


@pytest.mark.mcore
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("alignment", [1, 8])
@pytest.mark.parametrize("model_slices", [False, True])
def test_shared_preparation_matches_actual_input_layout(
    monkeypatch, packed, alignment, model_slices
):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.megatron import data

    monkeypatch.setattr(data, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(data, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        data,
        "get_ltor_masks_and_position_ids",
        lambda **kwargs: (
            None,
            None,
            torch.arange(kwargs["data"].shape[1]).expand_as(kwargs["data"]),
        ),
    )
    batch = BatchedDataDict(
        {
            "input_ids": torch.tensor([[0, 7, 8], [9, 10, 0]]),
            "input_lengths": torch.tensor([3, 2]),
            "is_artificial_input": torch.tensor([False, True]),
            "sample_mask": torch.tensor([0.0, 0.0]),
            "token_mask": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
        }
    )
    processed = data.process_microbatch(
        batch,
        seq_length_key="input_lengths",
        pack_sequences=packed,
        pad_individual_seqs_to_multiple_of=alignment,
        create_router_padding_mask=True,
        model_slices_context_parallel_inputs=model_slices,
    )
    assert processed.padding_mask.shape == processed.input_ids_cp_sharded.shape
    assert processed.input_ids_cp_sharded[~processed.padding_mask].tolist() == [0, 7, 8]
    if model_slices:
        # Real and borrowed image positions both remain valid for media merge.
        assert processed.input_ids_cp_sharded[
            processed.media_token_validity_mask
        ].tolist() == [0, 7, 8, 9, 10]
    else:
        assert processed.media_token_validity_mask is None


@pytest.mark.mcore
@pytest.mark.parametrize(
    "delegate,packed,model_slices,cp",
    [(True, True, False, 1), (False, True, True, 2), (False, False, True, 2)],
)
def test_unsupported_layouts_remain_guarded(
    monkeypatch, delegate, packed, model_slices, cp
):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.megatron import data

    monkeypatch.setattr(data, "get_context_parallel_world_size", lambda: cp)
    with pytest.raises(NotImplementedError, match="Router exclusion"):
        data.process_microbatch(
            BatchedDataDict(),
            create_router_padding_mask=True,
            delegate_pack_to_model=delegate,
            pack_sequences=packed,
            model_slices_context_parallel_inputs=model_slices,
        )


@pytest.mark.mcore
def test_omni_uses_existing_model_owned_mask_sharding(monkeypatch):
    from nemo_rl.models.megatron import train

    monkeypatch.setattr(train, "unwrap_model", lambda model: model)
    monkeypatch.setattr(
        train, "get_model_config", lambda model: SimpleNamespace(sequence_parallel=True)
    )
    mask = torch.tensor([[False, True, False, True]])
    assert (
        train._prepare_padding_mask_for_model(
            SimpleNamespace(), mask, model_slices_context_parallel_inputs=True
        )
        is mask
    )


@pytest.mark.mcore
def test_no_artificial_inputs_preserve_dense_inputs_and_loss_masks(monkeypatch):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.megatron import data

    monkeypatch.setattr(data, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        data,
        "get_ltor_masks_and_position_ids",
        lambda **kwargs: (
            None,
            None,
            torch.arange(kwargs["data"].shape[1]).expand_as(kwargs["data"]),
        ),
    )
    fields = {
        "input_ids": torch.tensor([[0, 7, 8]]),
        "input_lengths": torch.tensor([3]),
        "sample_mask": torch.tensor([0.0]),
        "token_mask": torch.zeros(1, 3),
    }
    original = data.process_microbatch(
        BatchedDataDict({k: v.clone() for k, v in fields.items()})
    )
    batch = BatchedDataDict({k: v.clone() for k, v in fields.items()})
    candidate = data.process_microbatch(batch, create_router_padding_mask=True)
    assert torch.equal(original.input_ids, candidate.input_ids)
    assert not candidate.padding_mask.any()
    assert candidate.media_token_validity_mask is None
    for key in fields:
        assert torch.equal(batch[key], fields[key])


@pytest.mark.mcore
@pytest.mark.parametrize(
    "enabled,rate,expect_mask",
    [(False, 0.001, False), (True, 0.0, False), (True, 0.001, True)],
)
@pytest.mark.parametrize("recipe_enable", [None, False, True])
def test_shared_iterator_uses_resolved_bias_config(
    monkeypatch, enabled, rate, expect_mask, recipe_enable
):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.megatron import data

    monkeypatch.setattr(BatchedDataDict, "to", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(data, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        data, "get_ltor_masks_and_position_ids", lambda **kwargs: (None, None, None)
    )
    batch = BatchedDataDict(
        {
            "input_ids": torch.tensor([[7, 8], [0, 0]]),
            "input_lengths": torch.tensor([2, 1]),
            "is_artificial_input": torch.tensor([False, True]),
        }
    )
    cfg = {
        "sequence_packing": {"enabled": False},
        "dynamic_batching": {"enabled": False},
        "make_sequence_length_divisible_by": 1,
        "megatron_cfg": {
            "moe_router_bias_update_rate": 0.0,
            "freeze_moe_router": True,
            "tensor_model_parallel_size": 1,
            "context_parallel_size": 1,
            "sequence_parallel": False,
        },
    }
    if recipe_enable is not None:
        cfg["megatron_cfg"]["moe_router_enable_expert_bias"] = recipe_enable
    # Loaded provider state is authoritative even when the recipe omits the
    # enable flag or disagrees. Freezing router weights does not freeze bias.
    model_config = SimpleNamespace(
        moe_router_enable_expert_bias=enabled, moe_router_bias_update_rate=rate
    )
    iterator, *_ = data.get_microbatch_iterator(
        batch, cfg, 2, None, model_config=model_config
    )
    result = next(iterator)
    assert (result.padding_mask is not None) == expect_mask
    if expect_mask:
        assert result.padding_mask.tolist() == [[False, False], [True, True]]


@pytest.mark.mcore
@pytest.mark.parametrize("cp,rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("model_slices", [False, True])
@pytest.mark.parametrize("artificial", [None, False, True])
def test_prepacked_exclusion_preserves_source_gaps_and_media(
    monkeypatch, cp, rank, model_slices, artificial
):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.megatron import data

    monkeypatch.setattr(data, "get_context_parallel_world_size", lambda: cp)
    monkeypatch.setattr(data, "get_context_parallel_rank", lambda: rank)
    batch = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3, 0, 5, 6, 0, 0, 0, 0]]),
            "input_lengths": torch.tensor([8]),
            "token_mask": torch.zeros(1, 10),
            "cu_seqlens": [torch.tensor([0, 3, 5], dtype=torch.int32)],
            "cu_seqlens_padded": [torch.tensor([0, 4, 8], dtype=torch.int32)],
        }
    )
    if artificial is not None:
        batch["is_artificial_input"] = torch.tensor([artificial])
    result = data.process_microbatch(
        batch,
        seq_length_key="input_lengths",
        pack_sequences=True,
        create_router_padding_mask=True,
        model_slices_context_parallel_inputs=model_slices,
    )
    # Independent oracle: real lengths 3 and 2 in two four-position segments.
    physical = torch.tensor([[False, False, False, True, False, False, True, True]])
    expected_ids = torch.tensor([[1, 2, 3, 0, 5, 6, 0, 0]])
    if cp == 2 and not model_slices:
        indices = [0, 3, 4, 7] if rank == 0 else [1, 2, 5, 6]
        physical = physical[:, indices]
        expected_ids = expected_ids[:, indices]
    expected = torch.ones_like(physical) if artificial else physical
    assert torch.equal(result.padding_mask, expected)
    assert torch.equal(result.input_ids_cp_sharded, expected_ids)
    if model_slices:
        assert torch.equal(result.media_token_validity_mask, ~physical)


@pytest.mark.mcore
def test_prepacked_iterator_masks_provider_enabled_bias_without_recipe_flag(
    monkeypatch,
):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.megatron import data

    monkeypatch.setattr(BatchedDataDict, "to", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(data, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(data, "get_context_parallel_rank", lambda: 0)
    batch = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3, 0, 4, 0, 0, 0]]),
            "input_lengths": torch.tensor([8]),
            "cu_seqlens": [torch.tensor([0, 3, 4], dtype=torch.int32)],
            "cu_seqlens_padded": [torch.tensor([0, 4, 8], dtype=torch.int32)],
        }
    )
    cfg = {
        "sequence_packing": {"enabled": True, "fuse_loss": True},
        "dynamic_batching": {"enabled": False},
        "megatron_cfg": {},
    }
    iterator, *_ = data.get_microbatch_iterator(
        batch,
        cfg,
        1,
        None,
        model_config=SimpleNamespace(
            moe_router_enable_expert_bias=True, moe_router_bias_update_rate=0.001
        ),
    )
    assert next(iterator).padding_mask.tolist() == [
        [False, False, False, True, False, True, True, True]
    ]
