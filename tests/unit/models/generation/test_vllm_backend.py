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


@pytest.mark.vllm
@pytest.mark.parametrize(
    ("rank_prefix", "rank", "group_world_size", "rollout_world_size", "expected"),
    [
        (4, 4, 8, 8, 4),
        (2, 1, 2, 4, 3),
    ],
)
def test_global_rollout_rank_handles_external_and_engine_local_dp(
    monkeypatch,
    rank_prefix,
    rank,
    group_world_size,
    rollout_world_size,
    expected,
):
    from nemo_rl.models.generation.vllm.vllm_backend import _global_rollout_rank

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: group_world_size)

    assert _global_rollout_rank(rank_prefix, rollout_world_size) == expected


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
def test_sharded_weight_load_context_is_scoped_to_current_tensor():
    """A mixed full/sharded refit stream should only mark sharded tensors."""
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    param = torch.nn.Parameter(torch.zeros(1, 8, 4), requires_grad=False)
    ext.model_runner = SimpleNamespace(
        model=SimpleNamespace(
            named_parameters=lambda: [("model.layers.0.mlp.experts.w13_weight", param)]
        )
    )
    expert_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    non_expert_name = "model.layers.0.self_attn.q_proj.weight"
    ext.state_dict_info = {
        expert_name: (torch.Size([8, 4]), torch.float32),
        non_expert_name: (torch.Size([8, 4]), torch.float32),
    }

    stream = ext._with_sharded_weight_load_contexts(
        [
            (expert_name, torch.zeros(4, 4)),
            (non_expert_name, torch.zeros(4, 4)),
        ]
    )

    assert next(stream)[0] == expert_name
    assert param.is_sharded_weight is True
    assert next(stream)[0] == non_expert_name
    assert not hasattr(param, "is_sharded_weight")

    with pytest.raises(StopIteration):
        next(stream)


class _FakeUnquantizedMethod:
    def __init__(self, backend_name="TRITON"):
        self.unquantized_backend = SimpleNamespace(name=backend_name)


class _FakeExpertOwner:
    def __init__(
        self,
        *,
        use_ep,
        expert_map=None,
        tp_rank=0,
        tp_size=1,
        backend_name="TRITON",
    ):
        self.use_ep = use_ep
        self._expert_map = expert_map
        self.logical_num_experts = 4
        self.global_num_experts = 4
        self.enable_eplb = False
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.quant_config = None
        self.base_quant_method = _FakeUnquantizedMethod(backend_name)

    def weight_loader(self, *args, **kwargs):
        raise AssertionError("The fake weight loader should not be called")

    def _map_global_expert_id_to_local_expert_id(self, expert_id):
        if self._expert_map is None:
            return expert_id
        return int(self._expert_map[expert_id])


def _expert_param(shape, owner):
    param = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
    param.weight_loader = owner.weight_loader
    return param


@pytest.mark.vllm
def test_checkpoint_engine_weight_layout_reports_ep_ownership():
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    owner = _FakeExpertOwner(
        use_ep=True, expert_map=torch.tensor([-1, 0, -1, 1], dtype=torch.int32)
    )
    w13 = _expert_param((2, 8, 4), owner)
    w2 = _expert_param((2, 4, 4), owner)
    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.model_runner = SimpleNamespace(
        model=SimpleNamespace(
            named_parameters=lambda: [
                ("model.layers.0.mlp.experts.w13_weight", w13),
                ("model.layers.0.mlp.experts.w2_weight", w2),
            ]
        )
    )

    layout = ext._checkpoint_engine_weight_layout()

    assert layout["model.layers.0.mlp.experts.w13_weight"] == {
        "tp_rank": 0,
        "tp_size": 1,
        "local_expert_ids": [1, 3],
    }


@pytest.mark.vllm
def test_checkpoint_engine_weight_layout_rejects_shuffled_backend():
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    owner = _FakeExpertOwner(use_ep=True, backend_name="FLASHINFER_TRTLLM")
    param = _expert_param((2, 8, 4), owner)
    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.model_runner = SimpleNamespace(
        model=SimpleNamespace(
            named_parameters=lambda: [("model.layers.0.mlp.experts.w13_weight", param)]
        )
    )

    with pytest.raises(ValueError, match="canonical unquantized Triton"):
        ext._checkpoint_engine_weight_layout()


@pytest.mark.vllm
def test_checkpoint_engine_weight_layout_rejects_transposed_experts():
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    owner = _FakeExpertOwner(use_ep=True)
    param = _expert_param((2, 8, 4), owner)
    param.is_transposed = True
    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.model_runner = SimpleNamespace(
        model=SimpleNamespace(
            named_parameters=lambda: [("model.layers.0.mlp.experts.w13_weight", param)]
        )
    )

    with pytest.raises(ValueError, match="canonical expert-weight orientation"):
        ext._checkpoint_engine_weight_layout()


@pytest.mark.vllm
def test_sharded_refit_directly_loads_full_ep_owned_experts():
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    owner = _FakeExpertOwner(
        use_ep=True, expert_map=torch.tensor([-1, 0, -1, 1], dtype=torch.int32)
    )
    param = _expert_param((2, 8, 4), owner)
    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.model_runner = SimpleNamespace(
        model=SimpleNamespace(
            named_parameters=lambda: [("model.layers.0.mlp.experts.w13_weight", param)]
        )
    )
    ext.state_dict_info = {
        "model.layers.0.mlp.experts.1.gate_proj.weight": (
            torch.Size([4, 4]),
            torch.float32,
        ),
        "model.layers.0.mlp.experts.3.gate_proj.weight": (
            torch.Size([4, 4]),
            torch.float32,
        ),
    }
    expert_1 = torch.full((4, 4), 1.0)
    expert_3 = torch.full((4, 4), 3.0)

    remaining = ext._load_sharded_expert_weight_groups(
        [
            ("model.layers.0.mlp.experts.1.gate_proj.weight", expert_1),
            ("model.layers.0.mlp.experts.3.gate_proj.weight", expert_3),
        ]
    )

    assert remaining == []
    torch.testing.assert_close(param[0, :4], expert_1)
    torch.testing.assert_close(param[1, :4], expert_3)
    torch.testing.assert_close(param[:, 4:], torch.zeros(2, 4, 4))


@pytest.mark.vllm
def test_sharded_refit_keeps_full_tp_weight_for_standard_loader():
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    owner = _FakeExpertOwner(use_ep=False, tp_rank=0, tp_size=2)
    param = _expert_param((4, 8, 4), owner)
    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.model_runner = SimpleNamespace(
        model=SimpleNamespace(
            named_parameters=lambda: [("model.layers.0.mlp.experts.w13_weight", param)]
        )
    )
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    ext.state_dict_info = {name: (torch.Size([8, 4]), torch.float32)}
    full_weight = torch.ones(8, 4)

    remaining = ext._load_sharded_expert_weight_groups([(name, full_weight)])

    assert len(remaining) == 1
    assert remaining[0][0] == name
    assert remaining[0][1] is full_weight
    torch.testing.assert_close(param, torch.zeros_like(param))


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
