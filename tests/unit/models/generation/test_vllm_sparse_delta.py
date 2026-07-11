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
import sys
from types import MethodType, SimpleNamespace
from typing import Any

import pytest
import torch

from nemo_rl.models.generation.vllm.vllm_sparse_delta import (
    VllmSparseDeltaApplier,
    _SparseLoadTracer,
)
from nemo_rl.utils.weight_transfer_sparse_codec import encode_sparse_infos


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


def _stub_fp8(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "nemo_rl.models.generation.vllm.quantization.fp8",
        SimpleNamespace(is_fp8_model=lambda _config: False),
    )


@pytest.mark.vllm
def test_serialized_sparse_payload_batch_preserves_order(tmp_path) -> None:
    applier = VllmSparseDeltaApplier(SimpleNamespace(), torch.device("cpu"))
    payloads = [
        (torch.tensor([index]), torch.tensor([float(index)]), [{"index": index}])
        for index in range(3)
    ]
    paths = [tmp_path / f"{index}.pt" for index in range(3)]
    for path, payload in zip(paths, payloads, strict=True):
        torch.save(payload, path)
    applied: list[Any] = []
    compiled: list[Any] = []
    applier._compile_plans = compiled.append
    applier._apply_sparse_weight_deltas = lambda tensors, metadata: applied.append(
        (*tensors, metadata)
    )

    result = applier.update_weights_from_sparse_payload_files(
        *(str(path) for path in paths)
    )
    applier.update_weights_from_serialized_sparse_payload(
        *(path.read_bytes() for path in paths)
    )

    assert [[item["index"] for item in batch] for batch in compiled] == [[0, 1, 2]] * 2
    assert [item[2][0]["index"] for item in applied] == [0, 1, 2] * 2
    assert result["receiver_deserialize_s"] >= 0.0
    assert result["receiver_plan_s"] >= 0.0
    assert result["receiver_sparse_apply_s"] >= 0.0


@pytest.mark.vllm
def test_native_loaders_compile_sparse_placement(monkeypatch) -> None:
    _stub_fp8(monkeypatch)
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
            [math.log(1.5), math.log(0.5)],
        ),
        ("model.layers.0.mlp.experts.7.gate_proj.weight", (8, 2), [8], [9]),
    ]
    payload = encode_sparse_infos(
        [
            (
                name,
                torch.empty(shape),
                torch.tensor(locations),
                torch.tensor(values, dtype=torch.float32),
            )
            for name, shape, locations, values in infos
        ],
        empty_dtype=torch.float32,
    )
    payload[2][-1].update(verification_locations=[8], verification_deltas=[9.0])

    applier = _applier(_NativeLoaderModel(**targets))
    applier._apply_sparse_weight_deltas(payload[:2], payload[2])

    assert torch.equal(targets["identity"], torch.tensor([0.0, 1.0, 2.0, 0.0]))
    assert targets["qkv"].view(-1)[[8, 9, 11]].tolist() == [1.0, 1.0, 1.0]
    assert targets["merged"].view(-1)[[0, 1, 7, 8, 15]].tolist() == [2, 2, 2, 3, 3]
    assert targets["w13"].view(-1)[[8, 15]].tolist() == [4, 4]
    assert targets["w2"].view(-1)[[8, 11, 12, 15]].tolist() == [5, 5, 5, 5]
    assert targets["mamba"].view(-1)[[0, 3, 4, 11]].tolist() == [6, 6, 6, 6]
    assert torch.allclose(targets["a"], torch.tensor([-3.0, -2.0]))
    assert (applier._verification_candidates, applier._verification) == (1, [])


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
def test_unknown_native_loader_fails_closed(monkeypatch) -> None:
    _stub_fp8(monkeypatch)
    model = _NativeLoaderModel(identity=torch.zeros(1))
    for name, error in (("unknown", "did not place"), ("transformed", "transform")):
        payload = encode_sparse_infos(
            [(name, torch.empty(1), torch.tensor([0]), torch.tensor([1.0]))],
            empty_dtype=torch.float32,
        )
        with pytest.raises(RuntimeError, match=error):
            _applier(model)._apply_sparse_weight_deltas(payload[:2], payload[2])


@pytest.mark.vllm
@pytest.mark.parametrize(
    ("initial", "expected_delta", "exact_mismatches", "mismatches"),
    [(200.0, 4.0, 0, 0), (2.0, 4.0000005, 1, 0), (2.0, 5.0, 1, 1)],
)
def test_sparse_delta_verification_compares_applied_delta(
    monkeypatch,
    initial: float,
    expected_delta: float,
    exact_mismatches: int,
    mismatches: int,
) -> None:
    _stub_fp8(monkeypatch)
    target = torch.tensor([1.0, initial, 3.0, initial])
    payload = encode_sparse_infos(
        [("weight", target, torch.tensor([1, 3]), torch.tensor([4.0, 4.0]))],
        empty_dtype=target.dtype,
    )
    payload[2][0].update(
        verification_locations=[1, 3],
        verification_deltas=[expected_delta, expected_delta],
    )
    applier = _applier(_NativeLoaderModel(identity=target))

    applier._apply_sparse_weight_deltas(payload[:2], payload[2])
    result = applier.finish_sparse_delta_refit()

    assert torch.equal(target, torch.tensor([1.0, initial + 4.0, 3.0, initial + 4.0]))
    assert result["verification_candidates"] == 2
    assert result["verification_samples"] == 2
    assert result["verification_exact_mismatches"] == 2 * exact_mismatches
    assert result["verification_mismatches"] == 2 * mismatches
