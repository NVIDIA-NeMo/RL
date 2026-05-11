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

from types import SimpleNamespace

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.flextron import FrozenFlextronRouter


class _DummyFc1(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class _DummyMlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_fc1 = _DummyFc1()
        self.last_fc1_output: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ffn_states = torch.ones(
            hidden_states.shape[0],
            6,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self.last_fc1_output = self.linear_fc1(ffn_states)
        return hidden_states


class _DummyLayer(torch.nn.Module):
    def __init__(self, layer_number: int, *, has_mlp: bool) -> None:
        super().__init__()
        self.layer_number = layer_number
        self.mlp = _DummyMlp() if has_mlp else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.mlp is not None:
            hidden_states = self.mlp(hidden_states)
        return hidden_states + 1


class _DummyDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                _DummyLayer(1, has_mlp=True),
                _DummyLayer(2, has_mlp=False),
            ]
        )
        self.final_norm = torch.nn.Identity()


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = _DummyDecoder()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder.layers:
            hidden_states = layer(hidden_states)
        return self.decoder.final_norm(hidden_states)


def _router_config() -> SimpleNamespace:
    return SimpleNamespace(
        hybrid_layer_pattern="E*",
        hidden_size=4,
        ffn_hidden_size=6,
        flex_routers=[
            {
                "emb_int_list": [2, 3],
                "mlp_int_list": [4],
            }
        ],
        flextron_sampling_rates=[0.0, 1.0],
    )


def test_base_route_leaves_dummy_model_unmasked():
    model = _DummyModel()
    router = FrozenFlextronRouter(model=model, model_cfg=_router_config())
    hidden_states = torch.arange(4, dtype=torch.float32).unsqueeze(0)

    with router.use_router(0):
        output = model(hidden_states)

    assert torch.equal(output, hidden_states + 2)
    layer = model.decoder.layers[0]
    assert layer.mlp is not None
    assert torch.equal(layer.mlp.last_fc1_output, torch.ones(1, 6))
    assert router.active_router_id is None


def test_nested_route_masks_hidden_and_mlp_intermediate_dims():
    model = _DummyModel()
    router = FrozenFlextronRouter(model=model, model_cfg=_router_config())
    hidden_states = torch.arange(4, dtype=torch.float32).unsqueeze(0)

    with router.use_router(1):
        output = model(hidden_states)

    assert torch.equal(output[:, :2], torch.tensor([[2.0, 3.0]]))
    assert torch.equal(output[:, 2:], torch.zeros(1, 2))
    layer = model.decoder.layers[0]
    assert layer.mlp is not None
    assert torch.equal(layer.mlp.last_fc1_output[:, :4], torch.ones(1, 4))
    assert torch.equal(layer.mlp.last_fc1_output[:, 4:], torch.zeros(1, 2))
    assert router.active_router_id is None


def test_inference_mode_mask_does_not_break_grad_enabled_forward():
    model = _DummyModel()
    router = FrozenFlextronRouter(model=model, model_cfg=_router_config())
    hidden_states = torch.arange(4, dtype=torch.float32).unsqueeze(0)

    with torch.inference_mode(), router.use_router(1):
        model(hidden_states)

    grad_hidden_states = hidden_states.clone().requires_grad_()
    with router.use_router(1):
        output = model(grad_hidden_states)

    output.sum().backward()

    assert grad_hidden_states.grad is not None
    assert router.active_router_id is None


def test_router_ids_can_be_sampled_or_read_from_batch():
    model = _DummyModel()
    router = FrozenFlextronRouter(model=model, model_cfg=_router_config())

    sampled = router.sample_router_ids(batch_size=4)
    assert sampled.tolist() == [1, 1, 1, 1]

    data = BatchedDataDict(
        {
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "flex_router_ids": torch.tensor([0, 1]),
        }
    )
    assert router.get_router_ids(data).tolist() == [0, 1]
    assert router.grouped_indices(torch.tensor([1, 0, 1])) == {1: [0, 2], 0: [1]}


def test_router_sampling_uses_one_route_for_whole_batch(monkeypatch):
    model = _DummyModel()
    router = FrozenFlextronRouter(model=model, model_cfg=_router_config())
    sampled_num_samples = []

    def fake_multinomial(
        probs: torch.Tensor, num_samples: int, replacement: bool = False
    ) -> torch.Tensor:
        sampled_num_samples.append(num_samples)
        return torch.tensor([1], dtype=torch.long, device=probs.device)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)

    sampled = router.sample_router_ids(batch_size=4)

    assert sampled.tolist() == [1, 1, 1, 1]
    assert sampled_num_samples == [1]
