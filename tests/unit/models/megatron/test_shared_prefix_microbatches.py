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
"""CPU tests for the microbatches of shared-prefix packs.

A tree is one prompt followed by the completions of the rows that start with it.
A pack is one model forward that holds several trees and runs each prompt once.
A dense bin is a model forward of ordinary packed sequences.
"""

import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch

# Probe the concrete module: test_modelopt_worker_utils.py leaves stub
# ``megatron.bridge`` packages in sys.modules that a package probe accepts.
pytest.importorskip("megatron.bridge.training.utils.packed_seq_utils")

from nemo_rl.data.packing.shared_prefix.plan import (  # noqa: E402
    plan_shared_prefix_shards,
)
from nemo_rl.data.packing.shared_prefix.row_metadata import (  # noqa: E402
    SHARED_PREFIX_PROMPT_LENGTHS,
    SHARED_PREFIX_TREE_IDS,
    UNSHARED_ROW,
    prompt_hash,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from nemo_rl.models.megatron import data as megatron_data  # noqa: E402
from nemo_rl.models.megatron.shared_prefix import microbatches  # noqa: E402

pytestmark = pytest.mark.mcore

# Rows 0 and 1 share the prompt [1, 2, 3]. Rows 2 and 3 share the prompt [9, 8].
_PACK_TOKENS = [[1, 2, 3, 4, 5], [1, 2, 3, 6], [9, 8, 7], [9, 8, 6, 5, 4]]
_PACK_PROMPT_LENGTHS = [3, 3, 2, 2]
_PACK_TREE_IDS = [0, 0, 1, 1]


def shared_prefix_rows(
    tokens: list[list[int]], prompt_lengths: list[int], tree_ids: list[int]
) -> BatchedDataDict[Any]:
    """Rows of one microbatch with the two columns that the Megatron worker attaches.

    ``token_mask`` is 1 on every completion token.
    """
    lengths = [len(row) for row in tokens]
    width = max(lengths) + 1
    token_mask = torch.zeros(len(tokens), width)
    for row, (length, prompt_length) in enumerate(zip(lengths, prompt_lengths)):
        token_mask[row, prompt_length:length] = 1.0
    return BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [row + [0] * (width - len(row)) for row in tokens]
            ),
            "input_lengths": torch.tensor(lengths),
            "token_mask": token_mask,
            "sample_mask": torch.ones(len(tokens)),
            SHARED_PREFIX_TREE_IDS: torch.tensor(tree_ids),
            SHARED_PREFIX_PROMPT_LENGTHS: torch.tensor(prompt_lengths),
        }
    )


@contextmanager
def shared_prefix_topology(
    *, cp_rank: int = 0, cp_size: int = 1, tp_size: int = 1
) -> Iterator[None]:
    """Give the pack builder a TP/CP topology and keep batches on the host."""
    with (
        patch.object(microbatches, "get_context_parallel_rank", return_value=cp_rank),
        patch.object(
            microbatches, "get_context_parallel_world_size", return_value=cp_size
        ),
        patch.object(
            microbatches, "get_tensor_model_parallel_world_size", return_value=tp_size
        ),
        patch.object(BatchedDataDict, "to", lambda self, device: self),
    ):
        yield


def _pack_rows() -> BatchedDataDict[Any]:
    return shared_prefix_rows(_PACK_TOKENS, _PACK_PROMPT_LENGTHS, _PACK_TREE_IDS)


@pytest.mark.parametrize(
    ("tree_ids", "expected"),
    [
        (None, None),
        ([UNSHARED_ROW] * 4, None),
        (_PACK_TREE_IDS, (2, 2)),
        ([0, 1, 1, 1], (1, 3)),
        ([0, 0, 0, 0], (4,)),
    ],
    ids=["no_column", "dense_bin", "two_trees", "uneven_trees", "one_tree"],
)
def test_tree_sizes_tell_a_pack_from_a_dense_bin(
    tree_ids: Optional[list[int]], expected: Optional[tuple[int, ...]]
) -> None:
    data = _pack_rows()
    if tree_ids is None:
        del data[SHARED_PREFIX_TREE_IDS]
    else:
        data[SHARED_PREFIX_TREE_IDS] = torch.tensor(tree_ids)

    assert microbatches.shared_prefix_tree_sizes(data) == expected


def test_tree_sizes_reject_a_mix_of_tree_rows_and_unshared_rows() -> None:
    data = shared_prefix_rows(
        _PACK_TOKENS, _PACK_PROMPT_LENGTHS, [0, 0, UNSHARED_ROW, UNSHARED_ROW]
    )

    with pytest.raises(ValueError, match="a pack or a dense bin"):
        microbatches.shared_prefix_tree_sizes(data)


def test_tree_sizes_reject_loss_inside_a_shared_prompt() -> None:
    """Rows share only tokens without loss, so every row keeps its own loss."""
    data = _pack_rows()
    # The prompt length of row 0 now covers its first generated token.
    data[SHARED_PREFIX_PROMPT_LENGTHS][0] = 4

    with pytest.raises(
        ValueError, match="Row 0 .* column 3, inside its shared prompt of 4 tokens"
    ):
        microbatches.shared_prefix_tree_sizes(data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tree_sizes_read_fetched_columns_on_the_gpu() -> None:
    """A fetch can return its columns on the GPU; the plan columns are on the host."""
    data = _pack_rows()
    for column in ("input_ids", "input_lengths", "token_mask"):
        data[column] = data[column].cuda()

    assert microbatches.shared_prefix_tree_sizes(data) == (2, 2)

    data[SHARED_PREFIX_PROMPT_LENGTHS][0] = 4
    with pytest.raises(
        ValueError, match="Row 0 .* column 3, inside its shared prompt of 4 tokens"
    ):
        microbatches.shared_prefix_tree_sizes(data)


def test_pack_holds_each_prompt_once() -> None:
    data = _pack_rows()
    data["mtp_loss_mask"] = data["token_mask"] * data["sample_mask"].unsqueeze(-1)

    with shared_prefix_topology():
        microbatch = microbatches.build_shared_prefix_microbatch(
            data,
            tree_sizes=(2, 2),
            padding_multiple=2,
            exclude_sequence_padding_from_expert_bias=True,
        )

    # Each row is padded to an even length, then the pack to an even length.
    assert {
        "tokens": microbatch.input_ids_cp_sharded.tolist(),
        "positions": microbatch.position_ids.tolist(),
        "mtp_loss_mask": microbatch.mtp_loss_mask.tolist(),
    } == {
        "tokens": [[1, 2, 3, 4, 5, 0, 6, 9, 8, 7, 0, 6, 5, 4, 0, 0]],
        "positions": [[0, 1, 2, 3, 4, 5, 3, 0, 1, 2, 3, 2, 3, 4, 5, 0]],
        "mtp_loss_mask": [[0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0]],
    }
    assert microbatch.data_dict is data
    assert microbatch.input_ids is data["input_ids"]
    assert microbatch.original_seq_length == data["input_ids"].shape[1]
    assert (
        microbatch.attention_mask,
        microbatch.packed_seq_params,
        microbatch.cu_seqlens_padded,
    ) == (None, None, None)

    params = microbatch.shared_prefix_params
    layout = params.inputs.layout
    assert [
        (tree.prefix_len, tree.completion_lens, tree.logical_completion_lens)
        for _, tree in layout.iter_roots()
    ] == [(3, (3, 1), (2, 1)), (2, (2, 4), (1, 3))]
    # The pack is one MTP loss group, as a dense bin of the same rows is.
    assert layout.mtp_loss_group_root_counts == (2,)
    assert params.cp_shard.padded_total_length == 16
    assert params.sequence_length == data["input_ids"].shape[1]
    assert params.model_kwargs() == {
        "shared_prefix_layout": layout,
        "shared_prefix_exclude_sequence_padding_from_expert_bias": True,
    }


@pytest.mark.parametrize(
    ("cp_rank", "tokens", "positions"),
    [
        (0, [1, 2, 3, 4, 5, 4, 0, 0, 0, 0], [0, 1, 2, 3, 4, 4, 5, 6, 7, 0]),
        (1, [0, 0, 0, 6, 9, 8, 7, 0, 6, 5], [5, 6, 7, 3, 0, 1, 2, 3, 2, 3]),
    ],
)
def test_pack_is_sharded_for_context_parallel(
    cp_rank: int, tokens: list[int], positions: list[int]
) -> None:
    """A CP rank gets chunks ``r`` and ``2 * CP - 1 - r`` of the padded pack."""
    data = _pack_rows()

    with shared_prefix_topology(cp_rank=cp_rank, cp_size=2):
        microbatch = microbatches.build_shared_prefix_microbatch(
            data,
            tree_sizes=(2, 2),
            padding_multiple=4,
            exclude_sequence_padding_from_expert_bias=False,
        )

    # The 19 packed tokens are padded to 20, which gives four chunks of 5.
    cp_shard = microbatch.shared_prefix_params.cp_shard
    assert cp_shard.padded_total_length == 20
    assert microbatch.input_ids_cp_sharded.tolist() == [tokens]
    assert microbatch.position_ids.tolist() == [positions]
    assert microbatch.mtp_loss_mask is None


@patch("nemo_rl.models.megatron.data.process_microbatch")
def test_iterator_builds_packs_and_leaves_dense_bins_to_process_microbatch(
    mock_process: MagicMock,
) -> None:
    mock_process.return_value = megatron_data.ProcessedInputs(
        input_ids=MagicMock(),
        input_ids_cp_sharded=MagicMock(),
        attention_mask=None,
        position_ids=None,
        packed_seq_params=MagicMock(),
        cu_seqlens_padded=MagicMock(),
    )
    pack = _pack_rows()
    dense_bin = shared_prefix_rows(_PACK_TOKENS, [0] * 4, [UNSHARED_ROW] * 4)

    with shared_prefix_topology():
        pack_microbatch, dense_microbatch = (
            megatron_data.make_processed_microbatch_iterator(
                raw_iterator=iter([pack, dense_bin]),
                cfg={
                    "sequence_packing": {"enabled": True},
                    "make_sequence_length_divisible_by": 2,
                },
                seq_length_key="input_lengths",
                pad_individual_seqs_to_multiple_of=2,
                pad_packed_seq_to_multiple_of=1,
                straggler_timer=MagicMock(),
                pad_full_seq_to=None,
                create_packed_seq_padding_mask=True,
            )
        )

    params = pack_microbatch.shared_prefix_params
    assert params.cp_shard.padded_total_length == 16
    assert params.exclude_sequence_padding_from_expert_bias is True
    assert pack_microbatch.packed_seq_params is None
    # The dense bin is the only microbatch that reached main's packing.
    mock_process.assert_called_once()
    assert mock_process.call_args.kwargs["data_dict"] is dense_bin
    assert dense_microbatch.shared_prefix_params is None
    assert dense_microbatch.packed_seq_params is not None


def test_iterator_rejects_mixed_tree_ids_before_the_move_to_gpu() -> None:
    data = shared_prefix_rows(
        _PACK_TOKENS, _PACK_PROMPT_LENGTHS, [0, 0, UNSHARED_ROW, UNSHARED_ROW]
    )
    iterator = megatron_data.make_processed_microbatch_iterator(
        raw_iterator=iter([data]),
        cfg={"sequence_packing": {"enabled": True}},
        seq_length_key="input_lengths",
        pad_individual_seqs_to_multiple_of=1,
        pad_packed_seq_to_multiple_of=1,
        straggler_timer=MagicMock(),
        pad_full_seq_to=None,
    )

    with (
        patch.object(data, "to") as move_to_device,
        pytest.raises(ValueError, match="a pack or a dense bin"),
    ):
        next(iterator)
    move_to_device.assert_not_called()


@pytest.mark.parametrize("num_shards", [1, 2], ids=["one_rank", "two_ranks"])
@pytest.mark.parametrize(
    "bound_by_full_length", [False, True], ids=["packed_length", "full_length"]
)
def test_iterator_builds_the_forwards_that_the_driver_planned(
    bound_by_full_length: bool, num_shards: int
) -> None:
    """Every rank builds forwards of the kind and the length of the driver's plan.

    The batch has nine groups that share a prompt, which do not divide between two
    ranks, and two groups of one row, which have no row to share with.
    """
    rng = random.Random(0)
    tokens, prompt_lengths, group_ids = [], [], []
    for group in range(11):
        prompt = [rng.randrange(1, 50) for _ in range(rng.randint(1, 6))]
        for _ in range(rng.randint(2, 5) if group < 9 else 1):
            completion = [rng.randrange(1, 50) for _ in range(rng.randint(1, 9))]
            tokens.append(prompt + completion)
            prompt_lengths.append(len(prompt))
            group_ids.append(f"group-{group}")
    shards = plan_shared_prefix_shards(
        sequence_lengths=[len(row) for row in tokens],
        group_ids=group_ids,
        prompt_lengths=prompt_lengths,
        prompt_hashes=[
            prompt_hash(row[:length]) for row, length in zip(tokens, prompt_lengths)
        ],
        num_shards=num_shards,
        batch_size=None,
        max_tokens_per_microbatch=48,
        sequence_length_pad_multiple=4,
        algorithm="modified_first_fit_decreasing",
        bound_by_full_length=bound_by_full_length,
        largest_first=False,
    )
    cfg = {
        "sequence_packing": {"enabled": True},
        "dynamic_batching": {"enabled": False},
        "make_sequence_length_divisible_by": 4,
        "megatron_cfg": {
            "tensor_model_parallel_size": 1,
            "context_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "sequence_parallel": False,
        },
    }

    kinds_by_rank = []
    for shard in shards:
        rows = shard.row_indices
        data = shared_prefix_rows(
            [tokens[row] for row in rows],
            [prompt_lengths[row] for row in rows],
            shard.tree_ids,
        )
        data.micro_batch_indices = shard.micro_batch_indices
        data.micro_batch_lengths = shard.micro_batch_lengths
        data.elem_counts_per_gb = shard.elem_counts_per_gb
        with (
            shared_prefix_topology(),
            patch.object(megatron_data, "get_context_parallel_rank", return_value=0),
            patch.object(
                megatron_data, "get_context_parallel_world_size", return_value=1
            ),
        ):
            iterator, num_microbatches, *_ = megatron_data.get_microbatch_iterator(
                data, cfg, mbs=1, straggler_timer=MagicMock()
            )
            built = list(iterator)

        forwards = list(zip(shard.micro_batch_indices[0], shard.micro_batch_lengths[0]))
        assert num_microbatches == len(built) == len(forwards)
        planned_packs = [
            shard.tree_ids[start] != UNSHARED_ROW for (start, _), _ in forwards
        ]
        assert [
            (
                microbatch.shared_prefix_params is not None,
                microbatch.packed_seq_params is None,
                microbatch.input_ids_cp_sharded.shape[1],
            )
            for microbatch in built
        ] == [
            (pack, pack, planned_length)
            for pack, (_, planned_length) in zip(planned_packs, forwards)
        ]
        kinds_by_rank.append(
            [microbatch.shared_prefix_params is not None for microbatch in built]
        )

    assert sorted(row for shard in shards for row in shard.row_indices) == list(
        range(len(tokens))
    )
    # Ranks run the same kind at the same forward index, and both kinds occur.
    assert all(kinds == kinds_by_rank[0] for kinds in kinds_by_rank)
    assert True in kinds_by_rank[0] and False in kinds_by_rank[0]
