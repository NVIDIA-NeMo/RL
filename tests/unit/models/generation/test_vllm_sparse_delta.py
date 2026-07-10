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

from types import MethodType, SimpleNamespace
from typing import Any

import pytest
import torch

from nemo_rl.models.generation.vllm.vllm_sparse_delta import VllmSparseDeltaApplier
from nemo_rl.utils.weight_transfer_sparse_codec import encode_sparse_infos


def _attach_tensor_attrs(tensor: torch.Tensor, **attrs: object) -> torch.Tensor:
    for name, value in attrs.items():
        setattr(tensor, name, value)
    return tensor


def _make_sparse_delta_extension(
    parameter_name: str,
    target: torch.Tensor,
    module: object,
) -> Any:
    model_runner = SimpleNamespace(
        model=SimpleNamespace(get_submodule=lambda _name: module)
    )
    ext = VllmSparseDeltaApplier(
        model_runner,
        torch.device("cpu"),
        rank=1,
    )
    ext._direct_sparse_delta_targets = {parameter_name: target}
    return ext


def _assert_sparse_plan(
    ext: Any,
    plan: Any,
    source_locations: list[int],
    expected_locations: list[int],
    expected_values: list[float],
) -> None:
    assert plan is not None
    values = torch.arange(len(source_locations), dtype=torch.float32)
    locations, values = ext._local_sparse_delta_update_inputs(
        torch.tensor(source_locations), values, plan
    )
    assert locations.tolist() == expected_locations
    assert values.tolist() == expected_values


@pytest.mark.vllm
def test_serialized_sparse_payload_batch_preserves_order(tmp_path) -> None:
    ext = VllmSparseDeltaApplier(SimpleNamespace(), torch.device("cpu"))
    payloads = [
        (torch.tensor([index]), torch.tensor([float(index)]), {"index": index})
        for index in range(3)
    ]
    paths = [tmp_path / f"{index}.pt" for index in range(3)]
    for path, payload in zip(paths, payloads, strict=True):
        torch.save(payload, path)
    applied: list[Any] = []

    def apply(payload: Any) -> dict[str, Any]:
        applied.append(payload)
        return {
            "ok": True,
            "receiver_sparse_apply_s": 2.0,
        }

    ext._apply_sparse_request = apply
    result = ext.update_weights_from_sparse_payload_files(
        *(str(path) for path in paths)
    )

    assert [item[2]["index"] for item in applied] == [0, 1, 2]
    assert all(
        torch.equal(item[1], payload[1])
        for item, payload in zip(applied, payloads, strict=True)
    )
    assert result["receiver_deserialize_s"] >= 0.0
    assert result["receiver_sparse_apply_s"] == 6.0


@pytest.mark.vllm
def test_direct_sparse_delta_placement() -> None:
    qkv_name = "model.layers.0.self_attn.qkv_proj.weight"
    qkv_target = _attach_tensor_attrs(torch.zeros(8, 2), output_dim=0)
    ext = _make_sparse_delta_extension(
        qkv_name,
        qkv_target,
        SimpleNamespace(
            tp_rank=1,
            num_kv_head_replicas=2,
            _get_shard_offset_mapping=lambda shard: {"q": 0, "k": 4, "v": 6}[shard],
            _get_shard_size_mapping=lambda shard: {"q": 4, "k": 2, "v": 2}[shard],
        ),
    )
    qkv_source = "model.layers.0.self_attn.k_proj.weight"
    plan = ext._direct_sparse_delta_qkv_plan(
        {"name": qkv_source, "shape": (2, 2)}, qkv_source, {qkv_name: qkv_target}
    )
    _assert_sparse_plan(ext, plan, [0, 1, 2, 3], [8, 9, 10, 11], [0.0, 1.0, 2.0, 3.0])

    merged_name = "model.layers.0.mlp.gate_up_proj.weight"
    merged_target = _attach_tensor_attrs(torch.zeros(8, 2), output_dim=0)
    ext = _make_sparse_delta_extension(
        merged_name,
        merged_target,
        SimpleNamespace(tp_rank=1, tp_size=2, output_sizes=(8, 8)),
    )
    for projection, expected_locations in (
        ("gate", [0, 1, 6, 7]),
        ("up", [8, 9, 14, 15]),
    ):
        source_name = f"model.layers.0.mlp.{projection}_proj.weight"
        plan = ext._direct_sparse_delta_target_plan(
            {"name": source_name, "shape": (8, 2)},
            {merged_name: merged_target},
        )
        _assert_sparse_plan(
            ext,
            plan,
            [6, 7, 8, 9, 14, 15],
            expected_locations,
            [2.0, 3.0, 4.0, 5.0],
        )

    expert_name = "model.layers.0.mlp.experts.w13_weight"
    expert_target = torch.zeros(2, 4, 2)
    expert_module = SimpleNamespace(
        tp_rank=1,
        moe_config=SimpleNamespace(is_act_and_mul=False),
        _map_global_expert_id_to_local_expert_id=lambda expert: (
            1 if expert == 3 else -1
        ),
    )
    ext = _make_sparse_delta_extension(
        expert_name,
        expert_target,
        expert_module,
    )
    for projection in ("gate_proj", "up_proj"):
        expert_source = f"model.layers.0.mlp.experts.3.{projection}.weight"
        plan = ext._direct_sparse_delta_expert_plan(
            {"name": expert_source, "shape": (8, 2)},
            expert_source,
            {expert_name: expert_target},
        )
        _assert_sparse_plan(
            ext,
            plan,
            [6, 7, 8, 9, 14, 15],
            [8, 9, 14, 15],
            [2.0, 3.0, 4.0, 5.0],
        )

    w2_target = torch.zeros(2, 2, 4)
    ext = _make_sparse_delta_extension(expert_name, w2_target, expert_module)
    expert_source = "model.layers.0.mlp.experts.3.down_proj.weight"
    plan = ext._direct_sparse_delta_expert_plan(
        {"name": expert_source, "shape": (2, 8)},
        expert_source,
        {"model.layers.0.mlp.experts.w2_weight": w2_target},
    )
    _assert_sparse_plan(ext, plan, [3, 4, 7, 11, 15], [8, 11, 15], [1.0, 2.0, 4.0])

    mamba_name = "model.layers.0.mixer.in_proj.weight"
    for target_shape, groups, source_locations, expected_locations, values in (
        ((16, 1, 2), 6, [0, 8, 25, 38, 40, 55], [0, 9, 22, 31], [1, 2, 4, 5]),
        (
            (14, 2),
            4,
            [0, 8, 24, 36, 44, 52, 55],
            [0, 8, 16, 20, 24, 27],
            [1, 2, 3, 4, 5, 6],
        ),
    ):
        target = _attach_tensor_attrs(
            torch.zeros(target_shape),
            weight_loader=MethodType(lambda _owner: None, SimpleNamespace()),
        )
        ext = _make_sparse_delta_extension(
            mamba_name,
            target,
            SimpleNamespace(
                tp_size=2,
                intermediate_size=8,
                groups_ssm_state_size=groups,
                num_heads=4,
            ),
        )
        plan = ext._direct_sparse_delta_mamba2_plan(
            {"name": mamba_name, "shape": (28, 2)},
            mamba_name,
            {mamba_name: target},
        )
        _assert_sparse_plan(ext, plan, source_locations, expected_locations, values)

    for attrs, source_shape, source_locations, expected_locations, values in (
        (
            {"output_dim": 0},
            (6, 2),
            [0, 1, 6, 7, 10, 11],
            [0, 1, 4, 5],
            [2, 3, 4, 5],
        ),
        (
            {"output_dim": 0, "input_dim": 1},
            (3, 4),
            [0, 1, 2, 3, 6, 7, 10, 11],
            [0, 1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6, 7],
        ),
    ):
        target = _attach_tensor_attrs(torch.zeros(3, 2), **attrs, tp_size=2, tp_rank=1)
        ext = _make_sparse_delta_extension("down_proj.weight", target, object())
        plan = ext._direct_sparse_delta_shard_plan(
            {"name": "down_proj.weight", "shape": source_shape}, target
        )
        _assert_sparse_plan(ext, plan, source_locations, expected_locations, values)


@pytest.mark.vllm
@pytest.mark.parametrize(
    ("initial", "expected_delta", "exact_mismatches", "mismatches"),
    [
        (200.0, 4.0, 0, 0),
        (2.0, 4.0000005, 1, 0),
        (2.0, 5.0, 1, 1),
    ],
)
def test_sparse_delta_sample_verification_only_compares_applied_delta(
    monkeypatch,
    initial: float,
    expected_delta: float,
    exact_mismatches: int,
    mismatches: int,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.quantization.fp8.is_fp8_model",
        lambda _config: False,
    )
    target = torch.tensor([1.0, initial, 3.0, initial])
    model_runner = SimpleNamespace(
        model=SimpleNamespace(),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(architectures=[]),
        ),
    )
    ext = VllmSparseDeltaApplier(model_runner, torch.device("cpu"))
    ext._direct_sparse_delta_targets = {"weight": target}
    ext._direct_sparse_delta_plan_cache = {
        "weight": ext._make_sparse_delta_target_plan(target, (4,))
    }
    payload = encode_sparse_infos(
        [("weight", target, torch.tensor([1, 3]), torch.tensor([4.0, 4.0]))],
        empty_dtype=target.dtype,
    )
    metadata = payload[2]
    metadata[0].update(
        verification_locations=[1, 3],
        verification_deltas=[expected_delta, expected_delta],
    )

    ext._apply_sparse_weight_deltas(payload[:2], metadata)
    result = ext.finish_sparse_delta_refit()

    assert torch.equal(target, torch.tensor([1.0, initial + 4.0, 3.0, initial + 4.0]))
    assert result["verification_candidates"] == 2
    assert result["verification_samples"] == 2
    assert result["verification_exact_mismatches"] == 2 * exact_mismatches
    assert result["verification_mismatches"] == 2 * mismatches
    rounded_difference = float((torch.tensor(expected_delta) - 4).abs())
    assert result["verification_max_abs"] == rounded_difference
