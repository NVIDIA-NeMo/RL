# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixed-row reads through the real TransferQueue simple backend."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch

from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    encode_multimodal_for_wire,
    multimodal_row_tags,
)
from nemo_rl.data_plane.column_io import kv_first_write, read_columns, write_columns
from nemo_rl.data_plane.schema import GLOBAL_FORWARD_PAD_SEQLEN
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.interfaces import Completion, PromptGroupRecord
from nemo_rl.experience.payload import pack_payload, record_to_train_batch


def _write_rows(client, token_field="media_token_validity_mask", dense_tokens=True):
    """Row 0 text, rows 1-2 image (row 2 also audio), row 3 audio only.

    Producers write per-token media fields on every row (zeros without
    images); ``dense_tokens=False`` writes them on image rows only.
    """
    partition = f"sparse-media-{uuid4().hex}"
    metas, originals = [], []
    for i in range(4):
        length = i + 2
        fields = BatchedDataDict(
            input_ids=torch.full((1, length), i, dtype=torch.long),
            input_lengths=torch.tensor([length]),
            sample_mask=torch.ones(1),
        )
        media = {}
        if dense_tokens or i in (1, 2):
            fields[token_field] = torch.full(
                (1, length),
                i in (1, 2),
                dtype=torch.bool
                if token_field == "media_token_validity_mask"
                else torch.long,
            )
        if i in (1, 2):
            media["pixel_values"] = PackedTensor(
                [torch.full((i, 3, 2, 2), float(i))], dim_to_pack=0
            )
            media["imgs_sizes"] = PackedTensor(
                [torch.full((i, 2), 2, dtype=torch.long)], dim_to_pack=0
            )
        if i in (2, 3):
            media["input_features"] = PackedTensor(
                [torch.full((2, 3), float(i))], dim_to_pack=0
            )
        originals.append(media)
        for key, value in media.items():
            fields[key] = encode_multimodal_for_wire(key, value)
        metas.append(
            kv_first_write(
                fields,
                sample_ids=[f"row-{i}"],
                dp_client=client,
                partition_id=partition,
                tags=multimodal_row_tags(media, 1) or [{}],
            )
        )
    meta = metas[0].concat(*metas[1:])
    meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] = 8
    return meta, originals


def _read(client, meta, reader):
    if reader == "driver":
        return read_columns(client, meta, meta.fields)
    worker = SimpleNamespace(
        _require_dp_client=lambda: client,
        _pad_value_dict=lambda: {},
        _forward_pad_seqlen=lambda meta: int(
            meta.extra_info.get(GLOBAL_FORWARD_PAD_SEQLEN, 0)
        ),
        _maybe_assemble_routed_experts=lambda meta, data: data,
    )
    return TQWorkerMixin._fetch(worker, meta, fetch_policy="independent")


@pytest.mark.parametrize("reader", ["driver", "worker"])
@pytest.mark.parametrize(
    "indices", [[0, 1], [0, 2, 1, 3], [2, 1], [3, 0], [3, 1], [2, 0, 2]]
)
@pytest.mark.parametrize("pad", [False, True])
@pytest.mark.parametrize(
    "token_field", ["media_token_validity_mask", "mm_token_type_ids"]
)
def test_sparse_media_survives_tq_fetch(tq_client, reader, indices, pad, token_field):
    meta, originals = _write_rows(tq_client, token_field)
    meta = meta.subset(indices)
    if not pad:
        meta.extra_info.pop(GLOBAL_FORWARD_PAD_SEQLEN)
    try:
        result = _read(tq_client, meta, reader)
        width = 8 if pad else max(meta.sequence_lengths)
        assert result["input_ids"].shape == (len(indices), width)
        assert result[token_field].shape == result["input_ids"].shape
        for j, i in enumerate(indices):
            assert result["input_ids"][j, : i + 2].tolist() == [i] * (i + 2)
            mask = result[token_field][j]
            assert mask[: i + 2].tolist() == [i in (1, 2)] * (i + 2)
            assert not mask[i + 2 :].any()
            for key in ("pixel_values", "imgs_sizes", "input_features"):
                expected = originals[i].get(key)
                if expected is not None:
                    assert torch.equal(
                        result[key].slice([j]).as_tensor(), expected.as_tensor()
                    )
                elif key in result:
                    assert result[key].logical_segment_counts_by_row()[j] == 0
    finally:
        tq_client.clear_samples(None, meta.partition_id)


@pytest.mark.parametrize("reader", ["driver", "worker"])
def test_missing_expected_pixels_fail(tq_client, reader):
    meta, _ = _write_rows(tq_client)
    # Claim an image on a text row whose pixels were never written.
    meta.tags[0] = meta.tags[1]
    try:
        with pytest.raises(KeyError, match="pixel_values"):
            _read(tq_client, meta, reader)
    finally:
        tq_client.clear_samples(None, meta.partition_id)


@pytest.mark.parametrize("reader", ["driver", "worker"])
def test_sparse_per_token_media_fails(tq_client, reader):
    # A producer that writes the mask on image rows only breaks the contract.
    meta, _ = _write_rows(tq_client, dense_tokens=False)
    try:
        with pytest.raises(KeyError, match="media_token_validity_mask"):
            _read(tq_client, meta, reader)
    finally:
        tq_client.clear_samples(None, meta.partition_id)


@pytest.mark.parametrize("reader", ["driver", "worker"])
def test_dense_media_masks_preserve_text_row_values(tq_client, reader):
    meta, _ = _write_rows(tq_client)
    expected = torch.tensor(
        [
            [True, False, False, False, False],
            [False, True, False, False, False],
            [False, False, True, False, False],
            [True, False, True, False, True],
        ],
        dtype=torch.bool,
    )
    try:
        write_columns(tq_client, meta, {"media_token_validity_mask": expected})
        result = _read(tq_client, meta, reader)
        assert torch.equal(result["media_token_validity_mask"][:, :5], expected)
    finally:
        tq_client.clear_samples(None, meta.partition_id)


@pytest.mark.parametrize("reader", ["driver", "worker"])
@pytest.mark.parametrize("dense_producer", [False, True])
def test_prompt_group_producer_contract(tq_client, reader, dense_producer):
    """Real producer writes: the old sparse contract fails; the fixed one reads."""
    partition = f"producer-media-{uuid4().hex}"
    metas = []
    token_fields = ("media_token_validity_mask", "mm_token_type_ids")
    for has_media in (False, True):
        prompt = {
            "role": "user",
            "content": "prompt",
            "token_ids": torch.tensor([10, 11]),
        }
        if has_media:
            prompt.update(
                pixel_values=PackedTensor(torch.ones(1, 4), dim_to_pack=0),
                media_token_validity_mask=torch.tensor([True, False]),
                mm_token_type_ids=torch.tensor([2, 0]),
            )
        completion = Completion(
            message_log=[
                prompt,
                {
                    "role": "assistant",
                    "content": "answer",
                    "token_ids": torch.tensor([12]),
                    "generation_logprobs": torch.tensor([-0.1]),
                },
            ],
            reward=0.0,
            truncated=False,
            env_extras=None,
        )
        record = PromptGroupRecord(
            prompt_idx=int(has_media),
            prompt=[prompt],
            extra_env_info=None,
            metadata={},
            completions=[completion],
            rollout_metrics={},
        )
        batch = record_to_train_batch(
            record,
            pad_value_dict={"token_ids": 0, "input_ids": 0},
            include_message_violation_fields=False,
            per_token_multimodal_fields=token_fields if dense_producer else (),
        )
        ids, fields, tags = pack_payload(
            batch,
            weight_version=0,
            group_id=f"group-{int(has_media)}",
            prompt_idx=record.prompt_idx,
        )
        meta = tq_client.put_samples(ids, partition, fields, tags=tags)
        meta.sequence_lengths = [3]
        metas.append(meta)
    meta = metas[0].concat(metas[1])
    try:
        if not dense_producer:
            with pytest.raises(KeyError, match="media_token_validity_mask"):
                _read(tq_client, meta, reader)
        else:
            for indices in ([0, 1], [1, 0, 1], [0]):
                result = _read(tq_client, meta.subset(indices), reader)
                assert result["media_token_validity_mask"].tolist() == [
                    [bool(i), False, False] for i in indices
                ]
                assert result["mm_token_type_ids"].tolist() == [
                    [2 * i, 0, 0] for i in indices
                ]
    finally:
        tq_client.clear_samples(None, partition)
