# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# NOTE: vllm_backend imports `vllm` eagerly at module top, so it is only imported
# inside the test bodies (which are marked @pytest.mark.vllm). This keeps the
# module collectable in the non-vllm unit lane, where these tests are deselected.

import contextlib
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file


def _write_sharded_checkpoint(model_dir, shards):
    """Write safetensors shards plus a model.safetensors.index.json.

    Args:
        model_dir: Directory (pathlib.Path) to write the checkpoint into.
        shards: Mapping of shard filename -> {weight_name: tensor}.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for shard_name, tensors in shards.items():
        save_file(tensors, str(model_dir / shard_name))
        for name in tensors:
            weight_map[name] = shard_name
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)


def _make_extension_with_drafter(mtp_start_layer_idx, num_mtp_layers):
    """Build a VllmInternalWorkerExtension with a mocked drafter and stubbed refit."""
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.device = torch.device("cpu")
    predictor = SimpleNamespace(
        mtp_start_layer_idx=mtp_start_layer_idx, num_mtp_layers=num_mtp_layers
    )
    ext.model_runner = MagicMock()
    ext.model_runner.drafter.model = SimpleNamespace(model=predictor)
    # Isolate this test from _load_draft_weights internals.
    ext._load_draft_weights = MagicMock()
    return ext


def _patch_vllm_postload(monkeypatch):
    """Stub the vLLM post-load helpers imported inside load_mtp_weights_from_disk."""
    monkeypatch.setattr(
        "vllm.config.set_current_vllm_config", lambda cfg: contextlib.nullcontext()
    )
    process_weights = MagicMock()
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.process_weights_after_loading",
        process_weights,
    )
    return process_weights


def _attach_tensor_attrs(tensor: torch.Tensor, **attrs: object) -> torch.Tensor:
    for name, value in attrs.items():
        setattr(tensor, name, value)
    return tensor


def _make_sparse_delta_extension(
    parameter_name: str,
    target: torch.Tensor,
    module: object,
    module_name: str | None = None,
) -> Any:
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.rank = 1
    ext._direct_sparse_delta_targets = {parameter_name: target}
    ext._direct_sparse_delta_modules = {
        module_name or parameter_name.rsplit(".", 1)[0]: module
    }
    ext._direct_sparse_delta_plan_cache = {}
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
    expert_source = "model.layers.0.mlp.experts.3.gate_proj.weight"
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

    expert_source = "model.layers.0.mlp.experts.3.up_proj.weight"
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
    mamba_target = torch.zeros(16, 1, 2)
    ext = _make_sparse_delta_extension(
        mamba_name,
        mamba_target,
        SimpleNamespace(
            tp_size=2,
            intermediate_size=8,
            groups_ssm_state_size=6,
            num_heads=4,
        ),
        "model.layers.0.mixer",
    )
    plan = ext._direct_sparse_delta_mamba2_plan(
        {"name": mamba_name, "shape": (28, 2)},
        mamba_name,
        {mamba_name: mamba_target},
    )
    _assert_sparse_plan(
        ext, plan, [0, 8, 25, 38, 40, 55], [0, 9, 22, 31], [1.0, 2.0, 4.0, 5.0]
    )

    mamba_target = torch.zeros(14, 2)
    ext = _make_sparse_delta_extension(
        mamba_name,
        mamba_target,
        SimpleNamespace(
            tp_size=2,
            intermediate_size=8,
            groups_ssm_state_size=4,
            num_heads=4,
        ),
        "model.layers.0.mixer",
    )
    plan = ext._direct_sparse_delta_mamba2_plan(
        {"name": mamba_name, "shape": (28, 2)},
        mamba_name,
        {mamba_name: mamba_target},
    )
    _assert_sparse_plan(
        ext,
        plan,
        [0, 8, 24, 36, 44, 52, 55],
        [0, 8, 16, 20, 24, 27],
        [1, 2, 3, 4, 5, 6],
    )

    shard_target = _attach_tensor_attrs(
        torch.zeros(3, 2), output_dim=0, tp_size=2, tp_rank=1
    )
    ext = _make_sparse_delta_extension("down_proj.weight", shard_target, object())
    plan = ext._direct_sparse_delta_shard_plan(
        {"name": "down_proj.weight", "shape": (6, 2)}, shard_target
    )
    _assert_sparse_plan(
        ext,
        plan,
        [0, 1, 6, 7, 10, 11],
        [0, 1, 4, 5],
        [2.0, 3.0, 4.0, 5.0],
    )

    shared_target = _attach_tensor_attrs(
        torch.zeros(3, 2), output_dim=0, input_dim=1, tp_size=2, tp_rank=1
    )
    ext = _make_sparse_delta_extension("down_proj.weight", shared_target, object())
    plan = ext._direct_sparse_delta_shard_plan(
        {"name": "down_proj.weight", "shape": (3, 4)}, shared_target
    )
    _assert_sparse_plan(
        ext,
        plan,
        [0, 1, 2, 3, 6, 7, 10, 11],
        [0, 1, 2, 3, 4, 5],
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    )


@pytest.mark.vllm
def test_read_mtp_layer_weights_from_checkpoint_filters_and_reads(tmp_path):
    """Only the requested MTP layer tensors are read, across the shards holding them."""
    from nemo_rl.models.generation.vllm.vllm_backend import (
        _read_mtp_layer_weights_from_checkpoint,
    )

    model_dir = tmp_path / "ckpt"
    mtp_block = torch.randn(4, 4)
    mtp_head = torch.randn(2, 4)
    other_layer = torch.randn(4, 4)
    embed = torch.randn(8, 4)
    # MTP layer index is 2; layer 0 and the top-level embed must be ignored.
    _write_sharded_checkpoint(
        model_dir,
        {
            "model-00001-of-00002.safetensors": {
                "model.layers.2.mlp.up_proj.weight": mtp_block,  # MTP, shard 1
                "model.layers.0.mlp.up_proj.weight": other_layer,  # non-MTP, same shard
            },
            "model-00002-of-00002.safetensors": {
                "model.layers.2.shared_head.head.weight": mtp_head,  # MTP, shard 2
                "model.embed_tokens.weight": embed,  # non-MTP, no layer index
            },
        },
    )

    weights = _read_mtp_layer_weights_from_checkpoint(str(model_dir), {2})

    by_name = dict(weights)
    assert set(by_name) == {
        "model.layers.2.mlp.up_proj.weight",
        "model.layers.2.shared_head.head.weight",
    }
    assert torch.equal(by_name["model.layers.2.mlp.up_proj.weight"], mtp_block)
    assert torch.equal(by_name["model.layers.2.shared_head.head.weight"], mtp_head)


@pytest.mark.vllm
def test_load_mtp_weights_from_disk_loads_only_mtp_layer(tmp_path, monkeypatch):
    """Success path: only MTP-layer weights are handed to the drafter, then post-loaded."""
    model_dir = tmp_path / "ckpt"
    _write_sharded_checkpoint(
        model_dir,
        {
            "model-00001-of-00001.safetensors": {
                "model.layers.2.mlp.up_proj.weight": torch.randn(4, 4),  # MTP
                "model.layers.2.embed_tokens.weight": torch.randn(8, 4),  # MTP
                "model.layers.0.mlp.up_proj.weight": torch.randn(4, 4),  # non-MTP
                "model.embed_tokens.weight": torch.randn(8, 4),  # non-MTP
            }
        },
    )
    ext = _make_extension_with_drafter(mtp_start_layer_idx=2, num_mtp_layers=1)
    process_weights = _patch_vllm_postload(monkeypatch)

    result = ext.load_mtp_weights_from_disk(str(model_dir))

    assert result is True
    ext._load_draft_weights.assert_called_once()
    loaded_names = {name for name, _ in ext._load_draft_weights.call_args[0][0]}
    assert loaded_names == {
        "model.layers.2.mlp.up_proj.weight",
        "model.layers.2.embed_tokens.weight",
    }
    process_weights.assert_called_once()


@pytest.mark.vllm
def test_load_mtp_weights_from_disk_returns_false_without_drafter(tmp_path):
    """When vLLM has not built a drafter, the load is skipped (no exception)."""
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.device = torch.device("cpu")
    ext.model_runner = MagicMock()
    ext.model_runner.drafter = None
    ext._load_draft_weights = MagicMock()

    assert ext.load_mtp_weights_from_disk(str(tmp_path)) is False
    ext._load_draft_weights.assert_not_called()


@pytest.mark.vllm
def test_load_mtp_weights_from_disk_raises_when_mtp_weights_missing(
    tmp_path, monkeypatch
):
    """A checkpoint without the MTP layer(s) fails loudly instead of silently."""
    model_dir = tmp_path / "ckpt"
    _write_sharded_checkpoint(
        model_dir,
        {
            "model-00001-of-00001.safetensors": {
                "model.layers.0.mlp.up_proj.weight": torch.randn(4, 4),
                "model.embed_tokens.weight": torch.randn(8, 4),
            }
        },
    )
    ext = _make_extension_with_drafter(mtp_start_layer_idx=2, num_mtp_layers=1)
    _patch_vllm_postload(monkeypatch)

    with pytest.raises(ValueError, match="No MTP layer weights"):
        ext.load_mtp_weights_from_disk(str(model_dir))
    ext._load_draft_weights.assert_not_called()
