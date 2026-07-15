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

"""Tests for destination-local vLLM refit loading."""

from types import SimpleNamespace

import pytest
import torch


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
def test_checkpoint_engine_weight_layout_reports_ep_and_pp_ownership(monkeypatch):
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
    monkeypatch.setattr(
        "vllm.model_executor.models.utils.get_pp_missing_layer_names",
        lambda model: ["model.layers.1."],
    )

    layout = ext._checkpoint_engine_weight_layout()

    assert layout["expert_params"]["model.layers.0.mlp.experts.w13_weight"] == {
        "tp_rank": 0,
        "tp_size": 1,
        "local_expert_ids": [1, 3],
    }
    assert layout["missing_weight_prefixes"] == ["model.layers.1."]


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
