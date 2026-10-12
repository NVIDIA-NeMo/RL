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
from nemo_rl.data_plane.column_io import kv_first_write, read_columns
from nemo_rl.data_plane.schema import GLOBAL_FORWARD_PAD_SEQLEN
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _write_rows(client):
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
        if i in (1, 2):
            media["pixel_values"] = PackedTensor(
                [torch.full((i, 3, 2, 2), float(i))], dim_to_pack=0
            )
            media["imgs_sizes"] = PackedTensor(
                [torch.full((i, 2), 2, dtype=torch.long)], dim_to_pack=0
            )
        if i == 2:
            media["input_features"] = PackedTensor([torch.ones(2, 3)], dim_to_pack=0)
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
        _forward_pad_seqlen=lambda meta: 8,
        _maybe_assemble_routed_experts=lambda meta, data: data,
    )
    return TQWorkerMixin._fetch(worker, meta, fetch_policy="independent")


@pytest.mark.parametrize("reader", ["driver", "worker"])
@pytest.mark.parametrize("indices", [[0, 2, 1, 3], [2, 1], [3, 0], [2, 0, 2]])
def test_sparse_media_survives_tq_fetch(tq_client, reader, indices):
    meta, originals = _write_rows(tq_client)
    meta = meta.subset(indices)
    try:
        result = _read(tq_client, meta, reader)
        assert result["input_ids"].shape == (len(indices), 8)
        for j, i in enumerate(indices):
            assert result["input_ids"][j, : i + 2].tolist() == [i] * (i + 2)
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
        with pytest.raises(ValueError, match="dropped requested fields.*pixel_values"):
            _read(tq_client, meta, reader)
    finally:
        tq_client.clear_samples(None, meta.partition_id)
