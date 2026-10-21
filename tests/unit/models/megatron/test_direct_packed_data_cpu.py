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

from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

pytest.importorskip("megatron.core")
pytest.importorskip("megatron.bridge")

from nemo_rl.models.megatron import data as megatron_data  # noqa: E402

pytestmark = pytest.mark.mcore


def _packed_row() -> BatchedDataDict:
    return BatchedDataDict(
        {
            "input_ids": torch.tensor([[10, 11, 12, 13, 20, 21, 22, 23]]),
            "target_ids": torch.tensor([[11, 12, 13, 14, 21, 22, 23, 24]]),
            "token_mask": torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
            "sample_mask": torch.tensor([0.5]),
            "packed_cu_seqlens": torch.tensor([[0, 4, 8]], dtype=torch.int32),
            "packed_cu_seqlens_lengths": torch.tensor([3]),
            "packed_max_seqlen": torch.tensor([4]),
        }
    )


def test_direct_packed_metadata_accepts_cp_aligned_row() -> None:
    row = _packed_row()
    row["mtp_loss_mask"] = row["token_mask"] * row["sample_mask"].unsqueeze(-1)

    metadata = megatron_data._validate_direct_packed_microbatch(
        row, context_parallel_size=2
    )

    assert metadata.cu_seqlens_length == 3
    assert metadata.max_seqlen == 4


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("input_ids", None, "missing required fields: input_ids"),
        ("input_ids", torch.zeros(2, 8), "exactly one row"),
        ("target_ids", torch.zeros(1, 7), "target_ids must have shape"),
        ("sample_mask", torch.ones(2), "sample_mask must have shape"),
        ("packed_max_seqlen", torch.tensor([4, 4]), r"shape \(1,\)"),
        ("packed_cu_seqlens_lengths", torch.tensor([4]), "lengths is invalid"),
        (
            "packed_cu_seqlens",
            torch.tensor([[0, 9, 8]], dtype=torch.int32),
            "increasing boundaries",
        ),
        ("packed_max_seqlen", torch.tensor([3]), "longest packed segment"),
        ("mtp_loss_mask", torch.zeros(1, 8), "mtp_loss_mask must match"),
    ],
)
def test_direct_packed_metadata_rejects_invalid_row(
    field: str, value: torch.Tensor | None, message: str
) -> None:
    row = _packed_row()
    if value is None:
        del row[field]
    else:
        row[field] = value

    with pytest.raises(ValueError, match=message):
        megatron_data._validate_direct_packed_microbatch(row, context_parallel_size=1)


def test_direct_packed_metadata_rejects_unshardable_segments() -> None:
    with pytest.raises(ValueError, match="segment lengths must be divisible"):
        megatron_data._validate_direct_packed_microbatch(
            _packed_row(), context_parallel_size=4
        )


def test_direct_packed_bundle_keeps_unsharded_tokens() -> None:
    params = MagicMock()
    with patch.object(
        megatron_data, "PackedSeqParams", return_value=params
    ) as constructor:
        batch, result_params = megatron_data._get_direct_packed_bundle_on_this_cp_rank(
            {"tokens": torch.tensor([[10, 11, 12]])},
            torch.tensor([0, 3], dtype=torch.int32),
            torch.tensor([3]),
            cp_size=1,
            cp_rank=0,
        )

    assert batch["tokens"].tolist() == [[10, 11, 12]]
    assert result_params is params
    assert constructor.call_args.kwargs["total_tokens"] == 3


def test_direct_packed_bundle_uses_mcore_per_document_cp_sharding() -> None:
    row = _packed_row()
    metadata = megatron_data._validate_direct_packed_microbatch(
        row, context_parallel_size=2
    )
    bundle = {
        "tokens": row["input_ids"],
        "labels": row["target_ids"],
        "loss_mask": row["token_mask"] * row["sample_mask"].unsqueeze(-1),
        "position_ids": row["position_ids"],
    }
    group = MagicMock()

    def shard(
        batch: dict[str, torch.Tensor], *, is_hybrid_cp: bool, cp_group: object
    ) -> dict[str, torch.Tensor]:
        assert is_hybrid_cp is False
        assert cp_group is group
        assert batch["cu_seqlens"].tolist() == [[0, 4, 8]]
        for name in bundle:
            batch[name] = batch[name][:, [1, 2, 5, 6]]
        return batch

    with (
        patch.object(megatron_data, "get_context_parallel_group", return_value=group),
        patch.object(megatron_data, "get_batch_on_this_cp_rank", side_effect=shard),
        patch.object(megatron_data, "PackedSeqParams") as params,
    ):
        local, _ = megatron_data._get_direct_packed_bundle_on_this_cp_rank(
            bundle,
            row["packed_cu_seqlens"][0],
            row["packed_max_seqlen"],
            cp_size=2,
            cp_rank=1,
            direct_packed_metadata=metadata,
        )

    assert local["tokens"].tolist() == [[11, 12, 21, 22]]
    assert local["labels"].tolist() == [[12, 13, 22, 23]]
    assert params.call_args.kwargs["total_tokens"] == 4


@pytest.mark.parametrize(
    ("tp", "cp", "sequence_parallel", "backend", "expected"),
    [
        (1, 1, False, None, (1, 1)),
        (8, 2, True, None, (32, 1)),
        (8, 2, True, "hybridep", (32, 128)),
    ],
)
def test_direct_packed_alignment_factors(
    tp: int,
    cp: int,
    sequence_parallel: bool,
    backend: str | None,
    expected: tuple[int, int],
) -> None:
    cfg = {
        "tensor_model_parallel_size": tp,
        "context_parallel_size": cp,
        "sequence_parallel": sequence_parallel,
        "moe_token_dispatcher_type": "flex" if backend else "alltoall",
        "moe_flex_dispatcher_backend": backend,
    }

    assert megatron_data._get_packed_sequence_alignment_factors(cfg) == expected
