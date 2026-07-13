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

import math
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.models.generation.vllm.vllm_sparse_delta import (
    VllmSparseDeltaApplier,
    _SparseLoadTracer,
)
from nemo_rl.utils.weight_transfer_sparse_codec import (
    SparseSourcePlan,
    SparseSourceRoute,
    decode_sparse_tensor_payload_for_staging,
    encode_sparse_infos,
    integer_dtype_for_element_size,
    iter_decoded_sparse_payload,
    partition_decoded_sparse_entries,
)


class _NativeLoaderModel:
    def __init__(self, **targets: torch.Tensor) -> None:
        self.targets = targets

    def parameters(self):
        return iter(self.targets.values())

    def buffers(self):
        return iter(())

    def load_weights(self, weights):
        for name, source in weights:
            target = self.targets
            if name == "weight":
                target["identity"].copy_(source)
            elif name == "weight_scale_inv":
                target["scale"].copy_(source)
            elif name.endswith("self_attn.k_proj.weight"):
                target["qkv"][4:6].copy_(source[2:4])
            elif name.endswith("mlp.gate_proj.weight"):
                target["merged"][:4].copy_(source[4:8])
            elif name.endswith("mlp.up_proj.weight"):
                target["merged"][4:8].copy_(source[4:8])
            elif name.endswith("experts.3.gate_proj.weight"):
                target["w13"][1, :4].copy_(source[4:8])
            elif name.endswith("experts.3.down_proj.weight"):
                target["w2"][1].copy_(source[:, 4:8])
            elif name.endswith("mixer.in_proj.weight"):
                target["mamba"][:2].copy_(source[2:4])
                target["mamba"][2:].copy_(source[6:10])
            elif name.endswith(("mixer.A", "mixer.A_log")):
                target["a"].copy_(source)
                target["a"].copy_(-torch.exp(target["a"]))
            elif name == "transformed":
                target["identity"].copy_(source + 1)
        return set()


def _applier(model: Any) -> VllmSparseDeltaApplier:
    return VllmSparseDeltaApplier(
        SimpleNamespace(
            model=model,
            vllm_config=SimpleNamespace(),
        ),
        torch.device("cpu"),
    )


def _bits(values: torch.Tensor) -> torch.Tensor:
    return values.contiguous().view(
        integer_dtype_for_element_size(values.element_size())
    )


def _decode_staged(payload: Any) -> list[Any]:
    return list(
        iter_decoded_sparse_payload(decode_sparse_tensor_payload_for_staging(payload))
    )


def _apply_payload(applier: VllmSparseDeltaApplier, payload: Any) -> None:
    decoded = _decode_staged(payload)
    plans = applier.sparse_delta_source_plans([item for item, _, _ in decoded])
    applier._apply_decoded_sparse_weight_deltas(
        partition_decoded_sparse_entries(decoded, plans)
    )


def test_sparse_plan_prewarm_uses_native_loader_without_applying_values() -> None:
    target = torch.zeros(8)
    applier = _applier(_NativeLoaderModel(identity=target))

    applier.prewarm({"weight": ((8,), torch.float32)})

    assert applier._plan_cache["weight"].identity
    assert torch.equal(target, torch.zeros_like(target))


def test_canonical_payload_is_partitioned_by_worker_source_plan() -> None:
    tensor = torch.empty((8, 2), dtype=torch.float32)
    locations = torch.tensor([1, 3, 8, 13])
    values = _bits(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    payload = encode_sparse_infos([("weight", tensor, locations, values, "overwrite")])
    payload[2][0].update(
        verification_locations=[1, 13],
        verification_values=[int(values[0]), int(values[3])],
    )
    decoded = _decode_staged(payload)

    first = partition_decoded_sparse_entries(
        decoded,
        {
            "weight": SparseSourcePlan(
                routes=(SparseSourceRoute(0, (2, 1), (4, 2), True),)
            )
        },
    )
    second = partition_decoded_sparse_entries(
        decoded,
        {
            "weight": SparseSourcePlan(
                routes=(SparseSourceRoute(8, (2, 1), (4, 2), True),)
            )
        },
    )

    first_item, first_locations, first_values = first[0]
    second_item, second_locations, second_values = second[0]
    assert first_locations.tolist() == [1, 3]
    assert first_values.tolist() == values[:2].tolist()
    assert first_item["verification_locations"] == [1]
    assert second_locations.tolist() == [8, 13]
    assert second_values.tolist() == values[2:].tolist()
    assert second_item["verification_locations"] == [13]


def test_canonical_payload_partition_handles_strided_source_view() -> None:
    values = torch.arange(8, dtype=torch.int32)
    payload = encode_sparse_infos(
        [
            (
                "weight",
                torch.empty((8,), dtype=torch.float32),
                torch.arange(8),
                values,
                "overwrite",
            )
        ]
    )
    partition = partition_decoded_sparse_entries(
        _decode_staged(payload),
        {
            "weight": SparseSourcePlan(
                routes=(SparseSourceRoute(1, (4, 1), (2, 2), False),)
            )
        },
    )

    _, locations, selected = partition[0]
    assert locations.tolist() == [1, 2, 5, 6]
    assert selected.tolist() == values[[1, 2, 5, 6]].tolist()


def test_canonical_payload_partitions_row_shards() -> None:
    values = torch.arange(16, dtype=torch.int32)
    payload = encode_sparse_infos(
        [
            (
                "weight",
                torch.empty((2, 8), dtype=torch.float32),
                torch.arange(16),
                values,
                "overwrite",
            )
        ]
    )
    left = SparseSourcePlan(routes=(SparseSourceRoute(0, (8, 1), (2, 4), False),))
    right = SparseSourcePlan(routes=(SparseSourceRoute(4, (8, 1), (2, 4), False),))
    expected = {0: [0, 1, 2, 3, 8, 9, 10, 11], 1: [4, 5, 6, 7, 12, 13, 14, 15]}
    for rank, plan in ((0, left), (1, right)):
        _, locations, selected = partition_decoded_sparse_entries(
            _decode_staged(payload), {"weight": plan}
        )[0]
        assert locations.tolist() == expected[rank]
        assert selected.tolist() == values[expected[rank]].tolist()


@pytest.mark.vllm
def test_backend_applies_decoded_sparse_payload_files() -> None:
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    applier = MagicMock()
    applier.update_weights_from_decoded_sparse_payload.return_value = {"ok": True}
    applier.update_weights_from_decoded_sparse_payload_files.return_value = {"ok": True}
    ext._get_sparse_delta_applier = MagicMock(return_value=applier)

    ext.prepare_sparse_delta_refit_info({"weight": ((8,), torch.float32)})

    assert ext.update_weights_from_decoded_sparse_payload(b"payload") == {"ok": True}
    assert ext.update_weights_from_decoded_sparse_payload_files("first", "second") == {
        "ok": True
    }
    applier.update_weights_from_decoded_sparse_payload.assert_called_once_with(
        b"payload"
    )
    applier.update_weights_from_decoded_sparse_payload_files.assert_called_once_with(
        "first", "second"
    )
    applier.prewarm.assert_called_once_with({"weight": ((8,), torch.float32)})


@pytest.mark.vllm
def test_sparse_payload_batches_preserve_order(tmp_path) -> None:
    applier = VllmSparseDeltaApplier(SimpleNamespace(), torch.device("cpu"))
    decoded_paths = [tmp_path / f"decoded-{index}.pt" for index in range(3)]
    decoded_payloads = [
        (
            (
                torch.tensor([index], dtype=torch.int32),
                torch.empty(0, dtype=torch.int64),
            ),
            (torch.tensor([index]),),
            [
                {
                    "name": "weight",
                    "index": index,
                    "decoded_location_group": 0,
                    "decoded_location_start": 0,
                    "decoded_location_end": 1,
                    "value_group": 0,
                    "value_start": 0,
                    "value_end": 1,
                }
            ],
        )
        for index in range(3)
    ]
    for path, payload in zip(decoded_paths, decoded_payloads, strict=True):
        torch.save(payload, path)
    decoded_applied: list[Any] = []
    applier.sparse_delta_source_plans = lambda _metadata: {
        "weight": SparseSourcePlan(identity=True)
    }
    applier._apply_decoded_sparse_weight_deltas = decoded_applied.append
    result = applier.update_weights_from_decoded_sparse_payload(
        *(path.read_bytes() for path in decoded_paths)
    )
    decoded_result = applier.update_weights_from_decoded_sparse_payload_files(
        *(str(path) for path in reversed(decoded_paths))
    )

    assert [payload[0][0]["index"] for payload in decoded_applied] == [
        0,
        1,
        2,
        2,
        1,
        0,
    ]
    assert result["receiver_deserialize_s"] >= 0.0
    assert result["receiver_plan_s"] >= 0.0
    assert result["receiver_sparse_apply_s"] >= 0.0
    assert decoded_result["receiver_deserialize_s"] >= 0.0
    assert decoded_result["receiver_partition_s"] >= 0.0


@pytest.mark.vllm
def test_decoded_sparse_payload_converts_compact_locations_for_apply(tmp_path) -> None:
    target = torch.zeros(8)
    payload = encode_sparse_infos(
        [
            (
                "weight",
                target,
                torch.tensor([1, 5]),
                _bits(torch.tensor([2.0, 6.0])),
                "overwrite",
            )
        ]
    )
    decoded = decode_sparse_tensor_payload_for_staging(payload)
    assert decoded[0][0].dtype == torch.int32
    path = tmp_path / "decoded.pt"
    torch.save(decoded, path)

    result = _applier(
        _NativeLoaderModel(identity=target)
    ).update_weights_from_decoded_sparse_payload_files(str(path))

    assert torch.equal(target, torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0]))
    assert result["receiver_partition_s"] >= 0.0


@pytest.mark.vllm
def test_native_loaders_compile_sparse_placement() -> None:
    targets = {
        "identity": torch.zeros(4),
        "qkv": torch.zeros(8, 2),
        "merged": torch.zeros(8, 2),
        "w13": torch.zeros(2, 4, 2),
        "w2": torch.zeros(2, 2, 4),
        "mamba": torch.zeros(6, 2),
        "a": torch.tensor([-2.0, -4.0]),
    }
    infos = [
        ("weight", (4,), [1, 2], [1.0, 2.0]),
        ("model.layers.0.self_attn.k_proj.weight", (4, 2), [0, 4, 5, 7], [1] * 4),
        ("model.layers.0.mlp.gate_proj.weight", (8, 2), [0, 8, 9, 15], [2] * 4),
        ("model.layers.0.mlp.up_proj.weight", (8, 2), [8, 15], [3] * 2),
        ("model.layers.0.mlp.experts.3.gate_proj.weight", (8, 2), [8, 15], [4] * 2),
        (
            "model.layers.0.mlp.experts.3.down_proj.weight",
            (2, 8),
            [3, 4, 7, 12, 15],
            [5] * 5,
        ),
        ("model.layers.0.mixer.in_proj.weight", (10, 2), [0, 4, 7, 12, 19], [6] * 5),
        (
            "backbone.layers.0.mixer.A_log",
            (2,),
            [0, 1],
            [math.log(3.0), math.log(2.0)],
        ),
        ("model.layers.0.mlp.experts.7.gate_proj.weight", (8, 2), [8], [9]),
    ]
    payload = encode_sparse_infos(
        [
            (
                name,
                torch.empty(shape),
                torch.tensor(locations),
                _bits(torch.tensor(values, dtype=torch.float32)),
                "overwrite",
            )
            for name, shape, locations, values in infos
        ]
    )
    payload[2][-1].update(
        verification_locations=[8],
        verification_values=[int(_bits(torch.tensor([9.0]))[0])],
    )
    payload[2][7].update(
        verification_locations=[0, 1],
        verification_values=[
            int(value) for value in _bits(torch.tensor([math.log(3.0), math.log(2.0)]))
        ],
    )

    applier = _applier(_NativeLoaderModel(**targets))
    _apply_payload(applier, payload)
    plans = applier.sparse_delta_source_plans(payload[2])
    verification = applier.finish_sparse_delta_refit()

    assert torch.equal(targets["identity"], torch.tensor([0.0, 1.0, 2.0, 0.0]))
    assert targets["qkv"].view(-1)[[8, 9, 11]].tolist() == [1.0, 1.0, 1.0]
    assert targets["merged"].view(-1)[[0, 1, 7, 8, 15]].tolist() == [2, 2, 2, 3, 3]
    assert targets["w13"].view(-1)[[8, 15]].tolist() == [4, 4]
    assert targets["w2"].view(-1)[[8, 11, 12, 15]].tolist() == [5, 5, 5, 5]
    assert targets["mamba"].view(-1)[[0, 3, 4, 11]].tolist() == [6, 6, 6, 6]
    assert torch.allclose(targets["a"], torch.tensor([-3.0, -2.0]))
    assert plans["model.layers.0.self_attn.k_proj.weight"].routes == (
        SparseSourceRoute(4, (2, 1), (2, 2), True),
    )
    assert verification["verification_candidates"] == 2
    assert verification["verification_samples"] == 2
    assert verification["verification_exact_mismatches"] == 0


@pytest.mark.vllm
def test_xor_applies_through_packed_native_loaders() -> None:
    targets = {
        "qkv": torch.zeros(8, 2),
        "merged": torch.zeros(8, 2),
        "w13": torch.zeros(2, 4, 2),
        "mamba": torch.zeros(6, 2),
    }
    infos = [
        (
            "model.layers.0.self_attn.k_proj.weight",
            (4, 2),
            [0, 4, 5, 7],
            [1.0] * 4,
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            (8, 2),
            [0, 8, 9, 15],
            [2.0] * 4,
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            (8, 2),
            [8, 15],
            [3.0] * 2,
        ),
        (
            "model.layers.0.mlp.experts.3.gate_proj.weight",
            (8, 2),
            [8, 15],
            [4.0] * 2,
        ),
        (
            "model.layers.0.mixer.in_proj.weight",
            (10, 2),
            [0, 4, 7, 12, 19],
            [6.0] * 5,
        ),
    ]
    payload = encode_sparse_infos(
        [
            (
                name,
                torch.empty(shape),
                torch.tensor(locations),
                _bits(torch.tensor(values)).bitwise_xor(
                    _bits(torch.zeros(len(values)))
                ),
                "xor",
            )
            for name, shape, locations, values in infos
        ]
    )

    _apply_payload(_applier(_NativeLoaderModel(**targets)), payload)

    assert targets["qkv"].view(-1)[[8, 9, 11]].tolist() == [1.0, 1.0, 1.0]
    assert targets["merged"].view(-1)[[0, 1, 7, 8, 15]].tolist() == [2, 2, 2, 3, 3]
    assert targets["w13"].view(-1)[[8, 15]].tolist() == [4.0, 4.0]
    assert targets["mamba"].view(-1)[[0, 3, 4, 11]].tolist() == [6, 6, 6, 6]


@pytest.mark.vllm
def test_vllm_native_loader_geometry() -> None:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
    )
    from vllm.model_executor.layers.mamba.mamba_mixer2 import (
        mamba_v2_sharded_weight_loader,
    )

    def trace(target, source_shape, load):
        source = torch.empty(source_shape, device="meta")
        tracer = _SparseLoadTracer([target], {"source": source})
        with tracer:
            load(source)
        return tracer.copies["source"]

    qkv = torch.nn.Parameter(torch.zeros(8, 2))
    qkv.output_dim = 0
    qkv_layer = SimpleNamespace(
        num_heads=4,
        num_kv_heads=2,
        num_kv_head_replicas=2,
        head_size=1,
        v_head_size=1,
        tp_rank=3,
    )
    qkv_layer.validate_shard_id = MethodType(
        QKVParallelLinear.validate_shard_id, qkv_layer
    )
    qkv_copies = trace(
        qkv,
        (4, 2),
        lambda source: QKVParallelLinear.weight_loader(qkv_layer, qkv, source, "k"),
    )

    merged = torch.nn.Parameter(torch.zeros(8, 2))
    merged.output_dim = 0
    merged_layer = SimpleNamespace(output_sizes=[8, 8], tp_size=2, tp_rank=1)
    merged_layer.validate_shard_id = MethodType(
        MergedColumnParallelLinear.validate_shard_id, merged_layer
    )
    merged_copies = trace(
        merged,
        (8, 2),
        lambda source: MergedColumnParallelLinear.weight_loader(
            merged_layer, merged, source, 1
        ),
    )

    expert = torch.nn.Parameter(torch.zeros(2, 8, 2))
    moe = SimpleNamespace(
        moe_config=SimpleNamespace(is_act_and_mul=True),
        _get_hidden_dim=FusedMoE._get_hidden_dim,
        _narrow_expert_data_for_padding=FusedMoE._narrow_expert_data_for_padding,
    )
    expert_copies = trace(
        expert,
        (8, 2),
        lambda source: FusedMoE._load_w13(moe, expert.data[1], 0, "w3", source, 1),
    )

    mamba = torch.nn.Parameter(torch.zeros(10, 1, 2))
    mamba_loader = mamba_v2_sharded_weight_loader(
        [(8, 0, False), (4, 2, True), (4, 2, True), (4, 0, False)], 2, 1
    )
    mamba_copies = trace(mamba, (16, 1, 2), lambda source: mamba_loader(mamba, source))

    cases = (
        (qkv_copies, [0, 4, 7], [8, 11]),
        (merged_copies, [0, 8, 15], [8, 15]),
        (expert_copies, [0, 8, 15], [24, 31]),
        (
            mamba_copies,
            [0, 8, 15, 16, 19, 20, 23, 28, 31],
            [0, 7, 8, 11, 12, 15, 16, 19],
        ),
    )
    for copies, source_locations, expected_rows in cases:
        mapped = [
            VllmSparseDeltaApplier._map_copy(
                torch.tensor(source_locations),
                torch.ones(len(source_locations)),
                copy,
            )[0]
            for copy in copies
        ]
        assert torch.cat(mapped).tolist() == expected_rows


@pytest.mark.vllm
def test_unknown_native_loader_fails_closed() -> None:
    model = _NativeLoaderModel(identity=torch.zeros(1))
    for name, error in (("unknown", "did not place"), ("transformed", "transform")):
        payload = encode_sparse_infos(
            [
                (
                    name,
                    torch.empty(1),
                    torch.tensor([0]),
                    _bits(torch.tensor([1.0])),
                    "overwrite",
                )
            ],
        )
        with pytest.raises(RuntimeError, match=error):
            _apply_payload(_applier(model), payload)


@pytest.mark.vllm
def test_unknown_sparse_operation_fails_closed() -> None:
    target = torch.zeros(1)
    payload = encode_sparse_infos(
        [
            (
                "weight",
                target,
                torch.tensor([0]),
                _bits(torch.tensor([1.0])),
                "overwrite",
            )
        ]
    )
    payload[2][0]["operation"] = "unknown"

    with pytest.raises(ValueError, match="Unsupported sparse-refit operation"):
        _apply_payload(_applier(_NativeLoaderModel(identity=target)), payload)


@pytest.mark.vllm
@pytest.mark.parametrize(
    ("initial", "verified_value", "exact_mismatches", "mismatches"),
    [(200.0, 4.0, 0, 0), (2.0, 4.0000005, 1, 0), (2.0, 5.0, 1, 1)],
)
def test_sparse_delta_verification_compares_replacement(
    initial: float,
    verified_value: float,
    exact_mismatches: int,
    mismatches: int,
) -> None:
    target = torch.tensor([1.0, initial, 3.0, initial])
    replacement = torch.tensor([initial + 4.0, initial + 4.0])
    payload = encode_sparse_infos(
        [
            (
                "weight",
                target,
                torch.tensor([1, 3]),
                _bits(replacement),
                "overwrite",
            )
        ],
    )
    payload[2][0].update(
        verification_locations=[1, 3],
        verification_values=[
            int(value)
            for value in _bits(
                torch.tensor([initial + verified_value, initial + verified_value])
            )
        ],
    )
    applier = _applier(_NativeLoaderModel(identity=target))

    _apply_payload(applier, payload)
    result = applier.finish_sparse_delta_refit()

    assert torch.equal(target, torch.tensor([1.0, initial + 4.0, 3.0, initial + 4.0]))
    assert result["verification_candidates"] == 2
    assert result["verification_samples"] == 2
    assert result["verification_exact_mismatches"] == 2 * exact_mismatches
    assert result["verification_mismatches"] == 2 * mismatches


@pytest.mark.vllm
def test_fp8_weight_and_scale_use_exact_bit_overwrite() -> None:
    target = torch.tensor([0x38, 0x40, 0x48], dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    scale = torch.tensor([1.0, 2.0])
    current = target.clone()
    current.view(torch.uint8)[1] = 0x41
    current.view(torch.uint8)[2] = 0x7F
    current_scale = scale.clone()
    current_scale[0] = 1.5
    payload = encode_sparse_infos(
        [
            (
                "weight",
                current,
                torch.tensor([1, 2]),
                current.view(torch.uint8)[1:3],
                "overwrite",
            ),
            (
                "weight_scale_inv",
                current_scale,
                torch.tensor([0]),
                current_scale.view(torch.int32)[:1],
                "overwrite",
            ),
        ]
    )
    payload[2][0].update(
        verification_locations=[1, 2], verification_values=[0x41, 0x7F]
    )
    payload[2][1].update(
        verification_locations=[0],
        verification_values=[int(current_scale.view(torch.int32)[0])],
    )
    applier = _applier(_NativeLoaderModel(identity=target, scale=scale))

    _apply_payload(applier, payload)
    _apply_payload(applier, payload)
    result = applier.finish_sparse_delta_refit()

    assert target.view(torch.uint8).tolist() == [0x38, 0x41, 0x7F]
    assert torch.equal(scale, current_scale)
    assert result["verification_samples"] == 6
    assert result["verification_exact_mismatches"] == 0
    assert result["verification_mismatches"] == 0
    assert result["verification_abs_sum"] == 0.0


@pytest.mark.vllm
def test_xor_applies_exact_bits_and_replay_reverts() -> None:
    baseline = torch.tensor([1.0, 2.0, 3.0])
    target = baseline.clone()
    current = torch.tensor([1.0, 5.0, -0.0])
    locations = torch.tensor([1, 2])
    xor_values = _bits(current)[locations].bitwise_xor(_bits(baseline)[locations])
    payload = encode_sparse_infos([("weight", current, locations, xor_values, "xor")])
    payload[2][0].update(
        verification_locations=locations.tolist(),
        verification_values=[int(value) for value in xor_values],
    )
    applier = _applier(_NativeLoaderModel(identity=target))

    _apply_payload(applier, payload)
    result = applier.finish_sparse_delta_refit()

    assert torch.equal(_bits(target), _bits(current))
    assert result["verification_exact_mismatches"] == 0
    assert result["verification_mismatches"] == 0

    _apply_payload(applier, payload)
    assert torch.equal(_bits(target), _bits(baseline))


@pytest.mark.vllm
def test_overwrite_casts_absolute_source_values() -> None:
    target = torch.zeros(2, dtype=torch.float16)
    source = torch.tensor([1.25, -2.5], dtype=torch.float32)
    payload = encode_sparse_infos(
        [
            (
                "weight",
                source,
                torch.tensor([0, 1]),
                _bits(source),
                "overwrite",
            )
        ]
    )

    _apply_payload(_applier(_NativeLoaderModel(identity=target)), payload)

    assert torch.equal(target, source.to(torch.float16))


@pytest.mark.vllm
@pytest.mark.parametrize(
    ("name", "source", "targets", "error"),
    [
        (
            "backbone.layers.0.mixer.A_log",
            torch.tensor([math.log(2.0)]),
            {"a": torch.tensor([-1.0])},
            "transformed weight",
        ),
        (
            "weight",
            torch.tensor([1.0], dtype=torch.float32),
            {"identity": torch.zeros(1, dtype=torch.float16)},
            "dtypes differ",
        ),
    ],
)
def test_xor_rejects_non_bitwise_compatible_targets(
    name: str,
    source: torch.Tensor,
    targets: dict[str, torch.Tensor],
    error: str,
) -> None:
    payload = encode_sparse_infos(
        [(name, source, torch.tensor([0]), _bits(source), "xor")]
    )

    with pytest.raises(RuntimeError, match=error):
        _apply_payload(_applier(_NativeLoaderModel(**targets)), payload)


@pytest.mark.vllm
def test_xor_rejects_overlapping_target_mappings() -> None:
    class RepeatedCopyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))

        def load_weights(self, weights) -> None:
            for _, source in weights:
                self.weight.copy_(source)
                self.weight.copy_(source)

    source = torch.tensor([1.0, 2.0])
    payload = encode_sparse_infos(
        [
            (
                "weight",
                source,
                torch.tensor([0, 1]),
                _bits(source).bitwise_xor(_bits(torch.zeros_like(source))),
                "xor",
            )
        ]
    )

    with pytest.raises(RuntimeError, match="target mappings overlap"):
        _apply_payload(_applier(RepeatedCopyModel()), payload)
