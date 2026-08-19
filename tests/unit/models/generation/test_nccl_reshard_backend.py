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

"""Unit tests for the vLLM-side nccl_reshard refit mapping (CPU, no GPU).

Covers the FFN-only bulk path in ``nemo_rl/models/generation/vllm/vllm_backend.py``
(``_build_hf_to_gen_backend_mapping`` + ``build_hf_to_local_param_map``), driven by
a synthetic ``refit_info`` and a fake ``named_parameters()`` (no real vLLM model,
no GPU).

``vllm_backend`` does ``import vllm`` at module top, so these are vllm-marked and
skipped where vllm is unavailable.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")  # module-top `import vllm` in vllm_backend

from nemo_rl.models.generation.vllm.vllm_backend import (  # noqa: E402
    VllmInternalWorkerExtension,
)
from nemo_rl.weight_sync.nccl_reshard_utils import (  # noqa: E402
    HFToLocalParamMap,
)

pytestmark = pytest.mark.vllm


# --------------------------------------------------------------------------
# _build_hf_to_gen_backend_mapping
# --------------------------------------------------------------------------
def _make_ext(vllm_params):
    """A VllmInternalWorkerExtension whose model exposes ``vllm_params``."""
    ext = VllmInternalWorkerExtension()  # no __init__
    # named_modules() is consulted to detect the FusedMoE backend (w13 layout);
    # an empty module map -> no match -> standard [gate; up] layout (the case
    # these tests assert).  See _build_hf_to_gen_backend_mapping.
    model = SimpleNamespace(
        named_parameters=lambda: list(vllm_params.items()),
        named_modules=lambda: [],
    )
    ext.model_runner = SimpleNamespace(model=model)
    return ext


def _param(*shape):
    return torch.empty(*shape)


def test_build_mapping_ffn_only():
    # Downsized bulk path: only FFN gate/up/down reach the resolver.
    H, E, Pl = 32, 2, 64
    refit_info = {
        "gen_tp_size": 4,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                # Dense MLP: gate/up -> gate_up_proj (merge), down -> direct.
                {
                    "name": "model.layers.0.mlp.gate_proj.weight",
                    "global_shape": [256, H],
                },
                {"name": "model.layers.0.mlp.up_proj.weight", "global_shape": [256, H]},
                {
                    "name": "model.layers.0.mlp.down_proj.weight",
                    "global_shape": [H, 256],
                },
                # MoE experts: gate/up -> w13 halves, down -> w2.
                {
                    "name": "model.layers.0.mlp.experts.gate_proj.weight",
                    "global_shape": [E, 128, H],
                    "grouped_expert_proj": "gate_proj",
                },
                {
                    "name": "model.layers.0.mlp.experts.up_proj.weight",
                    "global_shape": [E, 128, H],
                    "grouped_expert_proj": "up_proj",
                },
                {
                    "name": "model.layers.0.mlp.experts.down_proj.weight",
                    "global_shape": [E, H, 128],
                    "grouped_expert_proj": "down_proj",
                },
            ]
        },
    }
    gate_up = _param(128, H)  # 256*2/4
    down = _param(H, 64)  # 256/4 (row-parallel local)
    w13 = _param(E, 2 * Pl, H)  # gated: gate||up on intermediate axis (dim 1)
    w2 = _param(E, H, Pl)
    vllm_params = {
        "model.layers.0.mlp.gate_up_proj.weight": gate_up,
        "model.layers.0.mlp.down_proj.weight": down,
        "model.layers.0.mlp.experts.w13_weight": w13,
        "model.layers.0.mlp.experts.w2_weight": w2,
    }
    mapping = _make_ext(vllm_params)._build_hf_to_gen_backend_mapping(refit_info)

    # Dense gate/up -> gate_up_proj (dim-0 sub-slices)
    assert mapping["model.layers.0.mlp.gate_proj.weight"] == (gate_up, (slice(0, 64),))
    assert mapping["model.layers.0.mlp.up_proj.weight"] == (gate_up, (slice(64, 128),))
    # Dense down -> direct 1:1
    assert mapping["model.layers.0.mlp.down_proj.weight"] == (down, None)
    # Grouped expert gate/up -> w13 halves (dim-1 region); down -> w2 direct
    assert mapping["model.layers.0.mlp.experts.gate_proj.weight"] == (
        w13,
        (slice(None), slice(0, Pl), slice(None)),
    )
    assert mapping["model.layers.0.mlp.experts.up_proj.weight"] == (
        w13,
        (slice(None), slice(Pl, 2 * Pl), slice(None)),
    )
    assert mapping["model.layers.0.mlp.experts.down_proj.weight"] == (w2, None)


def test_build_mapping_non_gated_expert_up_is_direct():
    # Non-gated MoE (no gate_proj present): up_proj maps 1:1 to w13 (no slice).
    H, E = 16, 2
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.experts.up_proj.weight",
                    "global_shape": [E, 64, H],
                    "grouped_expert_proj": "up_proj",
                },
                {
                    "name": "model.layers.0.mlp.experts.down_proj.weight",
                    "global_shape": [E, H, 64],
                    "grouped_expert_proj": "down_proj",
                },
            ]
        },
    }
    w13 = _param(E, 64, H)
    w2 = _param(E, H, 64)
    vllm_params = {
        "model.layers.0.mlp.experts.w13_weight": w13,
        "model.layers.0.mlp.experts.w2_weight": w2,
    }
    mapping = _make_ext(vllm_params)._build_hf_to_gen_backend_mapping(refit_info)
    assert mapping["model.layers.0.mlp.experts.up_proj.weight"] == (w13, None)
    assert mapping["model.layers.0.mlp.experts.down_proj.weight"] == (w2, None)


def test_build_mapping_resolves_routed_experts_submodule():
    # vLLM 0.25 hangs the fused-MoE expert weights off a nested
    # ``routed_experts`` submodule (RoutedExperts is an nn.Module assigned as
    # MoERunner.routed_experts), so named_parameters() reports
    # ``...experts.routed_experts.w13_weight``.  The name built from the HF
    # side has no such segment, and an unresolved grouped expert is a hard
    # ValueError -- so without the flattened index this raises and every MoE
    # model fails to refit over nccl_reshard.
    H, E, Pl = 16, 2, 32
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.experts.gate_proj.weight",
                    "global_shape": [E, Pl, H],
                    "grouped_expert_proj": "gate_proj",
                },
                {
                    "name": "model.layers.0.mlp.experts.up_proj.weight",
                    "global_shape": [E, Pl, H],
                    "grouped_expert_proj": "up_proj",
                },
                {
                    "name": "model.layers.0.mlp.experts.down_proj.weight",
                    "global_shape": [E, H, Pl],
                    "grouped_expert_proj": "down_proj",
                },
            ]
        },
    }
    w13 = _param(E, 2 * Pl, H)
    w2 = _param(E, H, Pl)
    vllm_params = {
        "model.layers.0.mlp.experts.routed_experts.w13_weight": w13,
        "model.layers.0.mlp.experts.routed_experts.w2_weight": w2,
    }
    mapping = _make_ext(vllm_params)._build_hf_to_gen_backend_mapping(refit_info)

    assert mapping["model.layers.0.mlp.experts.gate_proj.weight"] == (
        w13,
        (slice(None), slice(0, Pl), slice(None)),
    )
    assert mapping["model.layers.0.mlp.experts.up_proj.weight"] == (
        w13,
        (slice(None), slice(Pl, 2 * Pl), slice(None)),
    )
    assert mapping["model.layers.0.mlp.experts.down_proj.weight"] == (w2, None)


def test_build_mapping_unmapped_param_raises():
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.some_unknown_module.weight",
                    "global_shape": [8, 8],
                },
            ]
        },
    }
    ext = _make_ext({"model.embed_tokens.weight": _param(8, 8)})
    with pytest.raises(ValueError):
        ext._build_hf_to_gen_backend_mapping(refit_info)


# --------------------------------------------------------------------------
# build_hf_to_local_param_map (the unified interface) + RefitCtx pre/post
# --------------------------------------------------------------------------
def test_build_hf_to_local_param_map_specs_and_roundtrip():
    # FFN-only: dense gate/up (merge) + down (direct), MoE experts (w13/w2).
    H, E, Pl = 32, 2, 64
    refit_info = {
        "gen_tp_size": 4,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.gate_proj.weight",
                    "global_shape": [256, H],
                    "dtype": "torch.float32",
                },
                {
                    "name": "model.layers.0.mlp.up_proj.weight",
                    "global_shape": [256, H],
                    "dtype": "torch.float32",
                },
                {
                    "name": "model.layers.0.mlp.down_proj.weight",
                    "global_shape": [H, 256],
                    "dtype": "torch.float32",
                },
                {
                    "name": "model.layers.0.mlp.experts.gate_proj.weight",
                    "global_shape": [E, 128, H],
                    "dtype": "torch.float32",
                    "grouped_expert_proj": "gate_proj",
                },
                {
                    "name": "model.layers.0.mlp.experts.up_proj.weight",
                    "global_shape": [E, 128, H],
                    "dtype": "torch.float32",
                    "grouped_expert_proj": "up_proj",
                },
                {
                    "name": "model.layers.0.mlp.experts.down_proj.weight",
                    "global_shape": [E, H, 128],
                    "dtype": "torch.float32",
                    "grouped_expert_proj": "down_proj",
                },
            ]
        },
    }
    gate_up = _param(128, H)  # dense gate||up, 256*2/4
    down = _param(H, 64)  # dense down (row-parallel local)
    w13 = _param(E, 2 * Pl, H)
    w2 = _param(E, H, Pl)
    ext = _make_ext(
        {
            "model.layers.0.mlp.gate_up_proj.weight": gate_up,
            "model.layers.0.mlp.down_proj.weight": down,
            "model.layers.0.mlp.experts.w13_weight": w13,
            "model.layers.0.mlp.experts.w2_weight": w2,
        }
    )

    pmap = ext.build_hf_to_local_param_map(refit_info)
    assert isinstance(pmap, HFToLocalParamMap)
    assert pmap.get("does.not.exist") is None

    # Direct param: base aliases the live vLLM tensor (.data is a distinct object
    # sharing storage, so compare data_ptr), no hooks (received in place).
    dn = pmap.get("model.layers.0.mlp.down_proj.weight")
    assert dn.base.data_ptr() == down.data_ptr()
    assert dn.pre is None and dn.post is None

    # Grouped expert down_proj -> w2 is also direct.
    edn = pmap.get("model.layers.0.mlp.experts.down_proj.weight")
    assert edn.base.data_ptr() == w2.data_ptr()
    assert edn.pre is None and edn.post is None

    # Merged dense gate_proj: pre allocates a recv buffer for gate's region of
    # gate_up_proj (rows [0:64] at TP=4); post scatters it back.
    g = pmap.get("model.layers.0.mlp.gate_proj.weight")
    assert g.pre is not None and g.post is not None
    ctx = g.pre(g.base)
    assert ctx.buf.shape == gate_up[0:64].shape
    assert ctx.extra["region"].shape == ctx.buf.shape
    ctx.buf.fill_(3.0)
    g.post(ctx)
    assert torch.equal(gate_up[0:64], torch.full_like(gate_up[0:64], 3.0))

    # Grouped expert gate_proj -> w13 gate half (dim-1 region); pre/post round-trip.
    eg = pmap.get("model.layers.0.mlp.experts.gate_proj.weight")
    assert eg.pre is not None and eg.post is not None
    egctx = eg.pre(eg.base)
    assert egctx.buf.shape == w13[:, 0:Pl, :].shape
    egctx.buf.fill_(5.0)
    eg.post(egctx)
    assert torch.equal(w13[:, 0:Pl, :], torch.full_like(w13[:, 0:Pl, :], 5.0))


def test_build_hf_to_local_param_map_quantizes_bf16_for_mxfp8(monkeypatch):
    H, E, Pl = 32, 2, 64
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.experts.gate_proj.weight",
                    "global_shape": [E, Pl, H],
                    "dtype": "torch.bfloat16",
                    "grouped_expert_proj": "gate_proj",
                },
                {
                    "name": "model.layers.0.mlp.experts.down_proj.weight",
                    "global_shape": [E, H, Pl],
                    "dtype": "torch.bfloat16",
                    "grouped_expert_proj": "down_proj",
                },
            ]
        },
    }
    w13 = torch.empty(E, 2 * Pl, H, dtype=torch.float8_e4m3fn)
    w13_scale = torch.empty(E, 2 * Pl, H // 32, dtype=torch.uint8)
    w2 = torch.empty(E, H, Pl, dtype=torch.float8_e4m3fn)
    w2_scale = torch.empty(E, H, Pl // 32, dtype=torch.uint8)
    ext = _make_ext(
        {
            "model.layers.0.mlp.experts.w13_weight": w13,
            "model.layers.0.mlp.experts.w13_weight_scale_from_checkpoint": w13_scale,
            "model.layers.0.mlp.experts.w2_weight": w2,
            "model.layers.0.mlp.experts.w2_weight_scale_from_checkpoint": w2_scale,
        }
    )

    def fake_quantize(weight):
        return (
            torch.full_like(weight, 3, dtype=torch.float8_e4m3fn),
            torch.full(
                (*weight.shape[:-1], weight.shape[-1] // 32), 7, dtype=torch.uint8
            ),
        )

    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.quantization.fp8.quantize_mxfp8_weight",
        fake_quantize,
    )

    pmap = ext.build_hf_to_local_param_map(refit_info)

    gate = pmap.get("model.layers.0.mlp.experts.gate_proj.weight")
    assert gate is not None and gate.pre is not None and gate.post is not None
    gate_ctx = gate.pre(gate.base)
    assert gate_ctx.buf.dtype == torch.bfloat16
    assert gate_ctx.buf.shape == w13[:, :Pl, :].shape
    gate.post(gate_ctx)
    assert torch.all(w13[:, :Pl, :].float() == 3)
    assert torch.all(w13_scale[:, :Pl, :] == 7)

    down = pmap.get("model.layers.0.mlp.experts.down_proj.weight")
    assert down is not None and down.pre is not None and down.post is not None
    down_ctx = down.pre(down.base)
    assert down_ctx.buf.dtype == torch.bfloat16
    assert down_ctx.buf.shape == w2.shape
    down.post(down_ctx)
    assert torch.all(w2.float() == 3)
    assert torch.all(w2_scale == 7)


def test_build_hf_to_local_param_map_uses_routed_expert_runtime_mxfp8_scale(
    monkeypatch,
):
    hidden_size, num_experts, intermediate_size = 32, 2, 64
    gate_name = "model.layers.0.mlp.experts.gate_proj.weight"
    up_name = "model.layers.0.mlp.experts.up_proj.weight"
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": gate_name,
                    "global_shape": [
                        num_experts,
                        intermediate_size,
                        hidden_size,
                    ],
                    "dtype": "torch.bfloat16",
                    "grouped_expert_proj": "gate_proj",
                },
                {
                    "name": up_name,
                    "global_shape": [
                        num_experts,
                        intermediate_size,
                        hidden_size,
                    ],
                    "dtype": "torch.bfloat16",
                    "grouped_expert_proj": "up_proj",
                },
            ]
        },
    }
    weight = torch.empty(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        dtype=torch.float8_e4m3fn,
    )
    runtime_scale = torch.empty(
        num_experts,
        2 * intermediate_size,
        hidden_size // 32,
        dtype=torch.uint8,
    )
    ext = _make_ext(
        {
            "model.layers.0.mlp.experts.routed_experts.w13_weight": weight,
            "model.layers.0.mlp.experts.routed_experts.w13_weight_scale": runtime_scale,
        }
    )

    def fake_quantize(value):
        return (
            torch.full_like(value, 3, dtype=torch.float8_e4m3fn),
            torch.full(
                (*value.shape[:-1], value.shape[-1] // 32),
                7,
                dtype=torch.uint8,
            ),
        )

    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.quantization.fp8.quantize_mxfp8_weight",
        fake_quantize,
    )

    spec = ext.build_hf_to_local_param_map(refit_info).get(gate_name)
    assert spec is not None and spec.pre is not None and spec.post is not None
    ctx = spec.pre(spec.base)
    spec.post(ctx)

    assert torch.all(weight[:, :intermediate_size, :].float() == 3)
    assert torch.all(runtime_scale[:, :intermediate_size, :] == 7)


def test_build_hf_to_local_param_map_quantizes_dense_gate_and_up_for_mxfp8(
    monkeypatch,
):
    hidden_size, intermediate_size = 32, 64
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.gate_proj.weight",
                    "global_shape": [intermediate_size, hidden_size],
                    "dtype": "torch.bfloat16",
                },
                {
                    "name": "model.layers.0.mlp.up_proj.weight",
                    "global_shape": [intermediate_size, hidden_size],
                    "dtype": "torch.bfloat16",
                },
            ]
        },
    }
    gate_up = torch.zeros(2 * intermediate_size, hidden_size, dtype=torch.float8_e4m3fn)
    gate_up_scale = torch.zeros(
        2 * intermediate_size, hidden_size // 32, dtype=torch.uint8
    )
    ext = _make_ext(
        {
            "model.layers.0.mlp.gate_up_proj.weight": gate_up,
            "model.layers.0.mlp.gate_up_proj.weight_scale_from_checkpoint": gate_up_scale,
        }
    )

    def fake_quantize(weight):
        fill_value = int(weight[0, 0].item())
        return (
            torch.full_like(weight, fill_value, dtype=torch.float8_e4m3fn),
            torch.full(
                (*weight.shape[:-1], weight.shape[-1] // 32),
                fill_value + 4,
                dtype=torch.uint8,
            ),
        )

    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.quantization.fp8.quantize_mxfp8_weight",
        fake_quantize,
    )

    pmap = ext.build_hf_to_local_param_map(refit_info)
    for name, fill_value in (("gate_proj", 1), ("up_proj", 2)):
        spec = pmap.get(f"model.layers.0.mlp.{name}.weight")
        assert spec is not None and spec.pre is not None and spec.post is not None
        ctx = spec.pre(spec.base)
        ctx.buf.fill_(fill_value)
        spec.post(ctx)

    assert torch.all(gate_up[:intermediate_size].float() == 1)
    assert torch.all(gate_up[intermediate_size:].float() == 2)
    assert torch.all(gate_up_scale[:intermediate_size] == 5)
    assert torch.all(gate_up_scale[intermediate_size:] == 6)


def test_build_hf_to_local_param_map_keeps_matching_blockwise_fp8_storage():
    H, E, P = 32, 2, 64
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.experts.down_proj.weight",
                    "global_shape": [E, H, P],
                    "dtype": "torch.float8_e4m3fn",
                    "grouped_expert_proj": "down_proj",
                }
            ]
        },
    }
    w2 = torch.empty(E, H, P, dtype=torch.float8_e4m3fn)
    ext = _make_ext({"model.layers.0.mlp.experts.w2_weight": w2})

    spec = ext.build_hf_to_local_param_map(refit_info).get(
        "model.layers.0.mlp.experts.down_proj.weight"
    )

    assert spec is not None
    assert spec.base.data_ptr() == w2.data_ptr()
    assert spec.pre is None and spec.post is None


def test_build_hf_to_local_param_map_rejects_wire_dtype_mismatch():
    hidden_size, intermediate_size = 32, 64
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.down_proj.weight",
                    "global_shape": [hidden_size, intermediate_size],
                    "dtype": "torch.float32",
                }
            ]
        },
    }
    down = torch.empty(hidden_size, intermediate_size, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="wire dtype torch.float32 does not match"):
        _make_ext(
            {"model.layers.0.mlp.down_proj.weight": down}
        ).build_hf_to_local_param_map(refit_info)


def test_build_hf_to_local_param_map_rejects_invalid_mxfp8_scale_shape():
    H, E, P = 32, 2, 64
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.experts.down_proj.weight",
                    "global_shape": [E, H, P],
                    "dtype": "torch.bfloat16",
                    "grouped_expert_proj": "down_proj",
                }
            ]
        },
    }
    w2 = torch.empty(E, H, P, dtype=torch.float8_e4m3fn)
    invalid_scale = torch.empty(E, H, 1, dtype=torch.uint8)
    ext = _make_ext(
        {
            "model.layers.0.mlp.experts.w2_weight": w2,
            "model.layers.0.mlp.experts.w2_weight_scale_from_checkpoint": invalid_scale,
        }
    )

    with pytest.raises(ValueError, match="has shape"):
        ext.build_hf_to_local_param_map(refit_info)


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("unknown_wire_dtype", "unsupported wire dtype"),
        ("missing_scale", "has no scale parameter"),
        ("invalid_scale_dtype", "expected torch.uint8"),
        ("invalid_k", "must have K divisible by 32"),
    ],
)
def test_build_hf_to_local_param_map_rejects_invalid_mxfp8_metadata(
    case: str, error: str
) -> None:
    H, E = 32, 2
    P = 63 if case == "invalid_k" else 64
    wire_dtype = "torch.unknown" if case == "unknown_wire_dtype" else "torch.bfloat16"
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.experts.down_proj.weight",
                    "global_shape": [E, H, P],
                    "dtype": wire_dtype,
                    "grouped_expert_proj": "down_proj",
                }
            ]
        },
    }
    w2 = torch.empty(E, H, P, dtype=torch.float8_e4m3fn)
    vllm_params = {"model.layers.0.mlp.experts.w2_weight": w2}
    if case != "missing_scale":
        scale_dtype = torch.float32 if case == "invalid_scale_dtype" else torch.uint8
        vllm_params["model.layers.0.mlp.experts.w2_weight_scale_from_checkpoint"] = (
            torch.empty(E, H, max(P // 32, 1), dtype=scale_dtype)
        )

    with pytest.raises(ValueError, match=error):
        _make_ext(vllm_params).build_hf_to_local_param_map(refit_info)
