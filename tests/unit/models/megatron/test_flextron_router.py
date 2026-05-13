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


class _DummyFinalNorm(torch.nn.Module):
    def __init__(self, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.eps = eps
        self.seen_eps: list[float] = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.seen_eps.append(self.eps)
        return hidden_states


class _DummyInProj(torch.nn.Module):
    def __init__(self, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.eps = eps
        self.last_input: torch.Tensor | None = None
        self.seen_eps: list[float] = []

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.last_input = hidden_states
        self.seen_eps.append(self.eps)
        return hidden_states + 1, None


class MambaMixer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_proj = _DummyInProj()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected, _ = self.in_proj(hidden_states)
        return projected


class _DummyMambaLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_number = 1
        self.mixer = MambaMixer()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mixer(hidden_states)


class _DummyPreMlpLayerNorm(torch.nn.Module):
    def __init__(self, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.eps = eps
        self.last_input: torch.Tensor | None = None
        self.seen_eps: list[float] = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.last_input = hidden_states
        self.seen_eps.append(self.eps)
        return hidden_states + 1


class MoELayer(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        return hidden_states + 2, None


class _DummyGroupedFc1(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        ffn_states = torch.arange(
            6,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ).unsqueeze(0)
        return ffn_states, None


class TEGroupedMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_fc1 = _DummyGroupedFc1()
        self.last_input: torch.Tensor | None = None
        self.last_fc1_output: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.last_input = hidden_states
        ffn_states, _ = self.linear_fc1(hidden_states)
        self.last_fc1_output = ffn_states
        return hidden_states + 3, None


class TransformerLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_number = 1
        self.pre_mlp_layernorm = _DummyPreMlpLayerNorm()
        self.moe_layer = MoELayer()
        self.grouped_mlp = TEGroupedMLP()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        moe_output, _ = self.moe_layer(hidden_states)
        grouped_output, _ = self.grouped_mlp(hidden_states)
        return moe_output + grouped_output


class _DummyDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                _DummyLayer(1, has_mlp=True),
                _DummyLayer(2, has_mlp=False),
            ]
        )
        self.final_norm = _DummyFinalNorm()


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = _DummyDecoder()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder.layers:
            hidden_states = layer(hidden_states)
        return self.decoder.final_norm(hidden_states)


class _DummyMambaDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_DummyMambaLayer()])


class _DummyMambaModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = _DummyMambaDecoder()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class _DummyMoEDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([TransformerLayer()])


class _DummyMoEModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = _DummyMoEDecoder()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def _router_config() -> SimpleNamespace:
    return SimpleNamespace(
        hybrid_layer_pattern="E*",
        hidden_size=4,
        ffn_hidden_size=6,
        layernorm_epsilon=1.0e-5,
        flex_routers=[
            {
                "emb_int_list": [2, 3],
                "mlp_int_list": [4],
            }
        ],
        flextron_sampling_rates=[0.0, 1.0],
    )


def _mamba_router_config() -> SimpleNamespace:
    return SimpleNamespace(
        hybrid_layer_pattern="M",
        hidden_size=4,
        ffn_hidden_size=6,
        layernorm_epsilon=1.0e-5,
        flex_routers=[
            {
                "emb_int_list": [2],
                "mlp_int_list": [],
            }
        ],
        flextron_sampling_rates=[0.0, 1.0],
    )


def _moe_router_config() -> SimpleNamespace:
    return SimpleNamespace(
        hybrid_layer_pattern="E",
        hidden_size=4,
        ffn_hidden_size=6,
        layernorm_epsilon=1.0e-5,
        flex_routers=[
            {
                "emb_int_list": [2],
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

    expected_scale = (2 / 4) ** 0.5
    torch.testing.assert_close(
        output[:, :2],
        torch.tensor([[2.0, 3.0]]) * expected_scale,
    )
    assert torch.equal(output[:, 2:], torch.zeros(1, 2))
    assert model.decoder.final_norm.seen_eps == [5.0e-6]
    assert model.decoder.final_norm.eps == 1.0e-5
    layer = model.decoder.layers[0]
    assert layer.mlp is not None
    assert torch.equal(layer.mlp.last_fc1_output[:, :4], torch.ones(1, 4))
    assert torch.equal(layer.mlp.last_fc1_output[:, 4:], torch.zeros(1, 2))
    assert router.active_router_id is None


def test_mamba_route_hooks_in_proj_pre_and_post():
    model = _DummyMambaModel()
    router = FrozenFlextronRouter(model=model, model_cfg=_mamba_router_config())
    hidden_states = torch.arange(4, dtype=torch.float32).unsqueeze(0)

    with router.use_router(1):
        output = model(hidden_states)

    expected_scale = (2 / 4) ** 0.5
    expected_in_proj_input = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    expected_output = torch.tensor([[1.0, 2.0, 0.0, 0.0]]) * expected_scale
    in_proj = model.decoder.layers[0].mixer.in_proj
    assert len(router._handles) == 4
    assert in_proj.last_input is not None
    torch.testing.assert_close(in_proj.last_input, expected_in_proj_input)
    torch.testing.assert_close(output, expected_output)
    assert in_proj.seen_eps == [5.0e-6]
    assert in_proj.eps == 1.0e-5
    assert router.active_router_id is None


def test_moe_route_hooks_layernorm_moe_and_grouped_mlp():
    model = _DummyMoEModel()
    router = FrozenFlextronRouter(model=model, model_cfg=_moe_router_config())
    hidden_states = torch.arange(4, dtype=torch.float32).unsqueeze(0)

    with router.use_router(1):
        output = model(hidden_states)

    expected_scale = (2 / 4) ** 0.5
    expected_norm_input = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    expected_grouped_input = torch.tensor(
        [[expected_scale, 2 * expected_scale, 0.0, 0.0]]
    )
    expected_fc1_output = torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.0, 0.0]])
    expected_output = torch.tensor(
        [[5 + 2 * expected_scale, 5 + 4 * expected_scale, 0.0, 0.0]]
    )

    layer = model.decoder.layers[0]
    assert len(router._handles) == 6
    assert layer.pre_mlp_layernorm.last_input is not None
    torch.testing.assert_close(layer.pre_mlp_layernorm.last_input, expected_norm_input)
    assert layer.pre_mlp_layernorm.seen_eps == [5.0e-6]
    assert layer.pre_mlp_layernorm.eps == 1.0e-5
    assert layer.grouped_mlp.last_input is not None
    assert layer.grouped_mlp.last_fc1_output is not None
    torch.testing.assert_close(layer.grouped_mlp.last_input, expected_grouped_input)
    torch.testing.assert_close(
        layer.grouped_mlp.last_fc1_output,
        expected_fc1_output,
    )
    torch.testing.assert_close(output, expected_output)
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


def test_router_ids_read_from_batch_and_default_to_base():
    model = _DummyModel()
    router = FrozenFlextronRouter(model=model, model_cfg=_router_config())

    data = BatchedDataDict(
        {
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "flex_router_ids": torch.tensor([0, 1]),
        }
    )
    assert router.get_router_ids(data).tolist() == [0, 1]
    assert router.grouped_indices(torch.tensor([1, 0, 1])) == {1: [0, 2], 0: [1]}

    # No probs and no ids -> default to base route (id 0) for every sample.
    default_data = BatchedDataDict({"input_ids": torch.ones(3, 2, dtype=torch.long)})
    assert router.get_router_ids(default_data).tolist() == [0, 0, 0]


def _three_route_config() -> SimpleNamespace:
    return SimpleNamespace(
        hybrid_layer_pattern="E*",
        hidden_size=4,
        ffn_hidden_size=6,
        layernorm_epsilon=1.0e-5,
        flex_routers=[
            {"emb_int_list": [2, 3], "mlp_int_list": [4]},
            {"emb_int_list": [3, 4], "mlp_int_list": [5]},
            {"emb_int_list": [4, 4], "mlp_int_list": [6]},
        ],
        # base=0.3, route0=0.5, route1=0.15, route2=0.05 -> cdf [0.3, 0.8, 0.95, 1.0]
        flextron_sampling_rates=[0.3, 0.5, 0.15, 0.05],
    )


def test_resolve_router_ids_partitions_unit_interval_by_cdf():
    model = _DummyModel()
    router = FrozenFlextronRouter(model=model, model_cfg=_three_route_config())

    # cdf: [0.3, 0.8, 0.95, 1.0]; pick probs strictly interior to each partition
    # to avoid float32 rounding at the cumulative-sum boundaries.
    probs = torch.tensor([0.0, 0.2, 0.4, 0.7, 0.85, 0.9, 0.97, 0.99])
    ids = router.resolve_router_ids(probs)

    assert ids.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_resolve_router_ids_via_get_router_ids_prefers_probs():
    model = _DummyModel()
    router = FrozenFlextronRouter(model=model, model_cfg=_three_route_config())

    data = BatchedDataDict(
        {
            "input_ids": torch.ones(3, 2, dtype=torch.long),
            "flex_router_probs": torch.tensor([0.1, 0.5, 0.97]),
            # If probs took precedence, ids below must be ignored.
            "flex_router_ids": torch.tensor([7, 7, 7]),
        }
    )

    assert router.get_router_ids(data).tolist() == [0, 1, 3]


def test_resolve_router_ids_matches_sampling_distribution():
    model = _DummyModel()
    router = FrozenFlextronRouter(model=model, model_cfg=_three_route_config())

    torch.manual_seed(0)
    n = 200_000
    probs = torch.rand(n)
    ids = router.resolve_router_ids(probs)

    expected = torch.tensor([0.3, 0.5, 0.15, 0.05])
    empirical = torch.bincount(ids, minlength=4).float() / n
    torch.testing.assert_close(empirical, expected, atol=0.01, rtol=0.0)


