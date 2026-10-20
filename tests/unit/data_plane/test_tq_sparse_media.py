# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixed text/video KV reads must preserve the video's packed media bytes."""

from collections.abc import Callable

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    multimodal_row_tags,
    validate_requested_multimodal_fields,
)
from nemo_rl.data_plane.adapters import transfer_queue as adapter
from nemo_rl.data_plane.codec import materialize


def _client_with_common_field_store(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    adapter.TQDataPlaneClient, dict[str, dict[str, torch.Tensor]], list[list[str]]
]:
    """Model the installed controller's documented common-field projection."""
    store = {
        "video-a": {
            "input_ids": torch.tensor([11, 18, 18, 21]),
            "pixel_values": torch.arange(12, dtype=torch.bfloat16),
            "imgs_sizes": torch.tensor([2, 2], dtype=torch.int32),
            "num_frames": torch.ones(1, dtype=torch.int32),
        },
        "text": {"input_ids": torch.tensor([11, 21, 22, 23])},
        "video-b": {
            "input_ids": torch.tensor([11, 18, 18, 24]),
            "pixel_values": torch.arange(18, dtype=torch.bfloat16) + 20,
            "imgs_sizes": torch.tensor([2, 3], dtype=torch.int32),
            "num_frames": torch.ones(1, dtype=torch.int32),
        },
    }
    requests = []

    def get(
        *, keys: list[str], partition_id: str, select_fields: list[str]
    ) -> TensorDict:
        assert partition_id == "mixed"
        requests.append(list(keys))
        common = set.intersection(*(set(store[key]) for key in keys))
        return TensorDict(
            {
                name: torch.nested.as_nested_tensor(
                    [store[key][name] for key in keys], layout=torch.jagged
                )
                for name in select_fields
                if name in common
            },
            batch_size=[len(keys)],
        )

    monkeypatch.setattr(adapter.tq, "kv_batch_get", get)
    return object.__new__(adapter.TQDataPlaneClient), store, requests


@pytest.mark.parametrize(
    "keys",
    [
        ["video-a", "text", "video-b"],
        ["text", "video-b", "video-a"],
        ["video-b", "video-a", "text"],
    ],
)
def test_sparse_media_read_preserves_exact_rows(
    monkeypatch: pytest.MonkeyPatch, keys: list[str]
) -> None:
    client, store, _ = _client_with_common_field_store(monkeypatch)
    fields = ["input_ids", "pixel_values", "imgs_sizes", "num_frames"]
    restored = client.get_samples(keys, "mixed", fields)
    assert set(restored.keys()) == set(fields)
    tags = []
    for key in keys:
        if key == "text":
            tags.append({})
        else:
            mm = {
                name: PackedTensor(
                    [
                        store[key][name].reshape(-1, 3)
                        if name == "pixel_values"
                        else store[key][name].reshape(1, 2)
                        if name == "imgs_sizes"
                        else store[key][name]
                    ],
                    dim_to_pack=0,
                )
                for name in fields[1:]
            }
            tags.extend(multimodal_row_tags(mm, 1))
    data = materialize(restored, tags=tags)
    validate_requested_multimodal_fields(data, requested_fields=fields, tags=tags)
    for index, key in enumerate(keys):
        assert torch.equal(data["input_ids"][index], store[key]["input_ids"])
        media = data.slice(index, index + 1).get_multimodal_dict(as_tensors=True)
        for name in fields[1:]:
            if key == "text":
                assert media[name] is None
            else:
                assert media[name].dtype == store[key][name].dtype
                assert torch.equal(media[name].reshape(-1), store[key][name])


@pytest.mark.parametrize("keys", [["video-a", "video-b"], ["text"]])
def test_uniform_reads_keep_single_fetch(
    monkeypatch: pytest.MonkeyPatch, keys: list[str]
) -> None:
    client, _, requests = _client_with_common_field_store(monkeypatch)
    fields = ["input_ids"] if keys == ["text"] else ["input_ids", "pixel_values"]
    restored = client.get_samples(keys, "mixed", fields)
    assert set(restored.keys()) == set(fields)
    assert requests == [keys]


def test_storage_failure_is_not_converted_to_empty_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _ = _client_with_common_field_store(monkeypatch)
    original: Callable[..., TensorDict] = adapter.tq.kv_batch_get

    def fail_sparse_read(**kwargs) -> TensorDict:
        if kwargs["select_fields"] == ["pixel_values"]:
            raise RuntimeError("actual storage read failure")
        return original(**kwargs)

    monkeypatch.setattr(adapter.tq, "kv_batch_get", fail_sparse_read)
    with pytest.raises(RuntimeError, match="actual storage read failure"):
        client.get_samples(["video-a", "text"], "mixed", ["input_ids", "pixel_values"])


def test_model_fetch_rejects_missing_declared_media() -> None:
    tags = [{"pixel_values__row_shapes": {"shapes": [[4, 3]]}}]
    with pytest.raises(
        ValueError, match="missing although row geometry declares media"
    ):
        validate_requested_multimodal_fields(
            {}, requested_fields=["pixel_values"], tags=tags
        )
    with pytest.raises(ValueError, match="lost row segments"):
        validate_requested_multimodal_fields(
            {"pixel_values": PackedTensor([None], dim_to_pack=0)},
            requested_fields=["pixel_values"],
            tags=tags,
        )
    # Non-model reads intentionally select only scalar/token columns.
    validate_requested_multimodal_fields({}, requested_fields=["input_ids"], tags=tags)
