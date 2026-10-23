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
"""Custom packed inputs preserve native wire semantics and client isolation."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.data import multimodal_utils as mm
from nemo_rl.data_plane.factory import (
    build_data_plane_client,
    configure_multimodal_fields,
    declared_packed_fields,
)


@pytest.mark.parametrize(
    "names",
    [
        ["input_ids"],
        ["position_ids"],
        ["extra_block_kwargs"],
        ["mtp_input_mask"],
        ["output_processor"],
        ["output_processor_context"],
        ["is_spec_decode"],
        ["use_cache"],
        ["flash_attn_kwargs"],
        ["attention_mask"],
        ["labels"],
        ["decoder_input"],
        ["loss_mask"],
        ["cu_seqlens"],
        ["routed_experts"],
        ["token_ids_delta"],
        ["pixel_values"],
        ["token_type_ids"],
        ["token_ids"],
        ["teacher_topk_logits"],
        ["mtp_loss_mask"],
        ["group_id"],
        ["meta_idx"],
        ["multimodal_row_tags"],
        ["features__row_shapes"],
        ["bad-name"],
        [123],
        "features",
    ],
)
@pytest.mark.parametrize("entrypoint", ["direct", "config"])
def test_rejects_reserved_or_malformed_declarations(names, entrypoint):
    with pytest.raises(ValueError, match="packed_multimodal_fields"):
        if entrypoint == "direct":
            mm.register_packed_multimodal_fields(names)
        else:
            configure_multimodal_fields({"packed_multimodal_fields": names})
    assert mm.G_CUSTOM_PACKED_FIELDS is None


def test_local_config_object_rejects_declaration():
    from nemo_rl.data_plane.interfaces import LocalDataPlaneConfig

    cfg = LocalDataPlaneConfig(packed_multimodal_fields=["features"])
    for entrypoint in (configure_multimodal_fields, build_data_plane_client):
        with pytest.raises(ValueError, match="packed_multimodal_fields"):
            entrypoint(cfg)
    assert mm.G_CUSTOM_PACKED_FIELDS is None


def test_one_immutable_declaration_including_empty_clients():
    packed_reference = mm.PACKED_MULTIMODAL_FIELDS
    wire_reference = mm.WIRE_MULTIMODAL_FIELDS
    configure_multimodal_fields({"packed_multimodal_fields": ["features", "indices"]})
    configure_multimodal_fields(
        {"packed_multimodal_fields": ["indices", "features", "features"]}
    )
    assert {"features", "indices"} <= packed_reference & wire_reference
    for config in (
        {},
        {"packed_multimodal_fields": []},
        {"packed_multimodal_fields": ["other"]},
    ):
        with pytest.raises(ValueError, match="identical for every client"):
            configure_multimodal_fields(config)
    assert mm.G_CUSTOM_PACKED_FIELDS == frozenset({"features", "indices"})


def test_empty_declaration_cannot_be_widened():
    configure_multimodal_fields({})
    with pytest.raises(ValueError, match="identical for every client"):
        configure_multimodal_fields({"packed_multimodal_fields": ["features"]})


def test_failed_client_initialization_rolls_back_then_retry_succeeds(monkeypatch):
    from nemo_rl.data_plane.adapters import transfer_queue as adapter

    cfg = {
        "enabled": True,
        "impl": "transfer_queue",
        "packed_multimodal_fields": ["features"],
    }
    constructor = MagicMock(side_effect=RuntimeError("backend initialization failed"))
    monkeypatch.setattr(adapter, "TQDataPlaneClient", constructor)
    with pytest.raises(RuntimeError, match="backend initialization failed"):
        build_data_plane_client(cfg)
    assert "features" not in mm.PACKED_MULTIMODAL_FIELDS
    assert mm.G_CUSTOM_PACKED_FIELDS is None
    constructor.side_effect = None
    cfg["packed_multimodal_fields"] = ["corrected_features"]
    build_data_plane_client(cfg)
    assert mm.G_CUSTOM_PACKED_FIELDS == frozenset({"corrected_features"})
    before = constructor.call_count
    with pytest.raises(ValueError, match="identical for every client"):
        build_data_plane_client({"enabled": True, "impl": "transfer_queue"})
    assert constructor.call_count == before


def test_mixed_custom_native_and_text_rows_round_trip():
    from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
    from nemo_rl.data_plane.column_io import kv_first_write, read_columns
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    mm.register_packed_multimodal_fields(["features"])
    packed = {
        "features": mm.PackedTensor(
            [torch.ones(2, 3), None, torch.zeros(1, 3)], dim_to_pack=0
        ),
        "pixel_values": mm.PackedTensor([None, torch.ones(1, 4), None], dim_to_pack=0),
    }
    batch = BatchedDataDict(
        {
            "input_ids": torch.arange(9).reshape(3, 3),
            "input_lengths": torch.tensor([3, 3, 3]),
            "sample_mask": torch.ones(3),
            **{
                key: mm.encode_multimodal_for_wire(key, value)
                for key, value in packed.items()
            },
        }
    )
    client = NoOpDataPlaneClient()
    client.register_partition(
        "custom", fields=list(batch), num_samples=3, consumer_tasks=["train"]
    )
    meta = kv_first_write(
        batch,
        sample_ids=["a", "b", "c"],
        dp_client=client,
        partition_id="custom",
        tags=mm.multimodal_row_tags(packed, 3),
    )
    assert mm.present_multimodal_fields(meta) == ["features", "pixel_values"]
    fetched = read_columns(client, meta.subset([2, 1, 0]), list(batch), layout="padded")
    for key, original in packed.items():
        expected = original.slice([2, 1, 0])
        assert torch.equal(fetched[key].as_tensor(), expected.as_tensor())
        assert (
            fetched[key].logical_segment_counts_by_row()
            == expected.logical_segment_counts_by_row()
        )
    assert fetched.slice(1, 2).get_multimodal_dict(as_tensors=True)["features"] is None
    assert mm.present_multimodal_fields(replace(meta, fields=["input_ids"])) == []
    client.close()


def test_fresh_process_installs_deserialized_declaration():
    mm.register_packed_multimodal_fields(["parent_features"])
    code = """
import json
import pickle
from nemo_rl.data import multimodal_utils as mm
from nemo_rl.data_plane.factory import configure_multimodal_fields
assert mm.G_CUSTOM_PACKED_FIELDS is None
cfg = pickle.loads(pickle.dumps({"packed_multimodal_fields": ["child_features"]}))
configure_multimodal_fields(cfg)
print(json.dumps(sorted(mm.G_CUSTOM_PACKED_FIELDS)))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == ["child_features"]
    assert mm.G_CUSTOM_PACKED_FIELDS == frozenset({"parent_features"})


def test_single_controller_registers_fields_for_deserialized_clients(tmp_path):
    import ray.cloudpickle as cloudpickle

    from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
    from tests.unit.single_controller.test_checkpointing import (
        _ACTOR_CLS,
        _actor_master_config,
        _make_actor_args,
    )

    config = _actor_master_config(tmp_path)
    config.data_plane["packed_multimodal_fields"] = ["features"]
    config, args = cloudpickle.loads(cloudpickle.dumps((config, _make_actor_args())))
    assert mm.G_CUSTOM_PACKED_FIELDS is None

    async def run():
        actor = _ACTOR_CLS(config, args, SetupTimingMetrics())
        try:
            assert mm.G_CUSTOM_PACKED_FIELDS == frozenset({"features"})
            assert "features" in mm.WIRE_MULTIMODAL_FIELDS
        finally:
            actor._checkpointer.shutdown()

    asyncio.run(run())


def test_single_controller_partition_includes_custom_fields_without_processor():
    from nemo_rl.algorithms.single_controller_utils.setup import (
        _register_single_controller_partitions,
    )
    from tests.unit.single_controller.test_setup import _make_master_config

    configure_multimodal_fields({"packed_multimodal_fields": ["features"]})
    config = _make_master_config()
    config.data_plane["packed_multimodal_fields"] = ["features"]
    client = MagicMock()
    _register_single_controller_partitions(
        client,
        master_config=config,
        partition_id="custom",
        include_multimodal_fields=False,
    )
    fields = client.register_partition.call_args.kwargs["fields"]
    assert "features" in fields
    assert "input_ids" in fields
    assert len(fields) == len(set(fields))


@pytest.mark.parametrize("declared", [True, False])
def test_tq_from_wire_keeps_declared_custom_field_nested(declared: bool) -> None:
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import _from_wire

    if declared:
        mm.register_packed_multimodal_fields(["features"])
    # Uniform segment shapes must not cause packed row boundaries to be lost.
    features = mm.PackedTensor(
        [torch.ones(2, 3), torch.zeros(2, 3), torch.full((2, 3), 2.0)], dim_to_pack=0
    )
    nested, tags = features.to_wire()
    out = _from_wire(TensorDict({"features": nested}, batch_size=(3,)))
    assert out["features"].is_nested is declared
    if declared:
        rebuilt = mm.PackedTensor.from_wire(out["features"], tags)
        assert torch.equal(rebuilt.as_tensor(), features.as_tensor())


def test_failed_client_init_keeps_existing_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_rl.data_plane.adapters import transfer_queue as adapter

    mm.register_packed_multimodal_fields(["features"])
    monkeypatch.setattr(
        adapter, "TQDataPlaneClient", MagicMock(side_effect=RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError, match="boom"):
        build_data_plane_client(
            {
                "enabled": True,
                "impl": "transfer_queue",
                "packed_multimodal_fields": ["features"],
            }
        )
    assert mm.G_CUSTOM_PACKED_FIELDS == frozenset({"features"})
    assert "features" in mm.PACKED_MULTIMODAL_FIELDS
    assert "features" in mm.WIRE_MULTIMODAL_FIELDS


@pytest.mark.parametrize(
    "names,expected", [(None, []), ([], []), (["b", "a", "a"], ["a", "b"])]
)
def test_declared_packed_fields_normalizes(
    names: list[str] | None, expected: list[str]
) -> None:
    assert declared_packed_fields({"packed_multimodal_fields": names}) == expected
    assert declared_packed_fields({}) == []
    assert declared_packed_fields(None) == []
