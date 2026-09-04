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

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch.distributed._tensor import Shard

pytest.importorskip("vllm")  # module-top `import vllm` in vllm_backend

from nemo_rl.models.generation.vllm.vllm_backend import (  # noqa: E402
    VllmInternalWorkerExtension,
)
from nemo_rl.weight_sync.nccl_reshard_utils import (  # noqa: E402
    HFToLocalParamMap,
    MeshInfo,
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
    ext._unquantized_flashinfer_trtllm_param_ids = lambda: set()
    return ext


def _enable_trtllm_staging(ext):
    """Mark every synthetic model parameter as owned by a BF16 TRTLLM module."""
    ext._uses_unquantized_flashinfer_trtllm = lambda: True
    ext._unquantized_flashinfer_trtllm_param_ids = lambda: {
        id(param) for _, param in ext.model_runner.model.named_parameters()
    }


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
                # MoE experts: gate/up -> w13 halves, down -> w2.
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


def test_build_hf_to_local_param_map_stages_trtllm_local_experts():
    """Packed TRTLLM storage receives canonical EP-local weights via load_weights."""
    H, E, P = 16, 4, 32
    expert_name = "model.layers.0.mlp.experts.gate_proj.weight"
    refit_info = {
        "gen_tp_size": 2,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": expert_name,
                    "global_shape": [E, P, H],
                    "dtype": "torch.bfloat16",
                    "grouped_expert_proj": "gate_proj",
                    "dst_mesh_info": MeshInfo(torch.tensor([8, 9])),
                    "dst_placements": [Shard(0)],
                }
            ]
        },
    }
    packed_w13 = torch.full((128, 16, 24, 64), 7.0)
    ext = _make_ext(
        {
            "model.layers.0.mlp.experts.routed_experts.w13_weight": packed_w13,
        }
    )
    ext.device = torch.device("cpu")
    ext.pp_comm_groups = {0: SimpleNamespace(rank=9)}
    _enable_trtllm_staging(ext)
    ext._load_full_hf_weights = MagicMock(
        return_value={"model.layers.0.mlp.experts.routed_experts.w13_weight"}
    )

    spec = ext.build_hf_to_local_param_map(refit_info).get(expert_name)
    assert spec is not None and spec.pre is not None and spec.post is not None

    ctx = spec.pre(spec.base)
    assert ctx.buf.shape == (2, P, H)
    assert ctx.buf.dtype == torch.bfloat16
    ctx.buf[0].fill_(2.0)
    ctx.buf[1].fill_(3.0)
    spec.post(ctx)

    loaded_weights = ext._load_full_hf_weights.call_args.args[0]
    assert [name for name, _ in loaded_weights] == [
        "model.layers.0.mlp.experts.2.gate_proj.weight",
        "model.layers.0.mlp.experts.3.gate_proj.weight",
    ]
    torch.testing.assert_close(
        loaded_weights[0][1], torch.full((P, H), 2.0, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(
        loaded_weights[1][1], torch.full((P, H), 3.0, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(packed_w13, torch.full_like(packed_w13, 7.0))


def test_build_hf_to_local_param_map_stages_nemotron_lightning_padded_experts():
    """Receive the logical Nano/Lightning weight instead of its padded runtime form."""
    num_experts, intermediate_size, hidden_size = 128, 928, 2688
    expert_name = "backbone.layers.0.mlp.experts.up_proj.weight"
    runtime_name = "model.layers.0.mlp.experts.routed_experts.w13_weight"
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["backbone.layers.0"],
        "per_layer_params": {
            "backbone.layers.0": [
                {
                    "name": expert_name,
                    "global_shape": [
                        num_experts,
                        intermediate_size,
                        hidden_size,
                    ],
                    "dtype": "torch.bfloat16",
                    "grouped_expert_proj": "up_proj",
                    "dst_mesh_info": MeshInfo(torch.tensor([0])),
                    "dst_placements": [Shard(0)],
                }
            ]
        },
    }
    packed_runtime = torch.empty(
        num_experts,
        hidden_size // 64,
        1024,
        64,
        dtype=torch.bfloat16,
        device="meta",
    )
    ext = _make_ext({runtime_name: packed_runtime})
    ext.device = torch.device("meta")
    ext.pp_comm_groups = {0: SimpleNamespace(rank=0)}
    _enable_trtllm_staging(ext)
    ext._load_full_hf_weights = MagicMock(return_value={runtime_name})

    spec = ext.build_hf_to_local_param_map(refit_info).get(expert_name)
    assert spec is not None and spec.pre is not None and spec.post is not None

    ctx = spec.pre(spec.base)
    assert ctx.buf.shape == (num_experts, intermediate_size, hidden_size)
    assert ctx.buf.dtype == torch.bfloat16
    assert ctx.buf.numel() != packed_runtime.numel()
    spec.post(ctx)

    loaded_weights = ext._load_full_hf_weights.call_args.args[0]
    assert len(loaded_weights) == num_experts
    assert loaded_weights[0][0] == "backbone.layers.0.mlp.experts.0.up_proj.weight"
    assert loaded_weights[-1][0] == "backbone.layers.0.mlp.experts.127.up_proj.weight"
    assert loaded_weights[0][1].shape == (intermediate_size, hidden_size)


def test_build_hf_to_local_param_map_stages_only_bf16_trtllm_experts(
    monkeypatch,
):
    """Mixed MXFP8 models stage only first/last BF16 expert layers."""
    num_experts, intermediate_size, hidden_size = 2, 64, 32
    first_bf16_name = "backbone.layers.0.mlp.experts.up_proj.weight"
    mxfp8_name = "backbone.layers.1.mlp.experts.up_proj.weight"
    last_bf16_name = "backbone.layers.2.mlp.experts.up_proj.weight"
    first_bf16_runtime_name = "model.layers.0.mlp.experts.routed_experts.w13_weight"
    mxfp8_runtime_name = "model.layers.1.mlp.experts.routed_experts.w13_weight"
    last_bf16_runtime_name = "model.layers.2.mlp.experts.routed_experts.w13_weight"
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": [
            "backbone.layers.0",
            "backbone.layers.1",
            "backbone.layers.2",
        ],
        "per_layer_params": {
            "backbone.layers.0": [
                {
                    "name": first_bf16_name,
                    "global_shape": [num_experts, intermediate_size, hidden_size],
                    "dtype": "torch.bfloat16",
                    "grouped_expert_proj": "up_proj",
                    "dst_mesh_info": MeshInfo(torch.tensor([0])),
                    "dst_placements": [Shard(0)],
                }
            ],
            "backbone.layers.1": [
                {
                    "name": mxfp8_name,
                    "global_shape": [num_experts, intermediate_size, hidden_size],
                    "dtype": "torch.bfloat16",
                    "grouped_expert_proj": "up_proj",
                    "dst_mesh_info": MeshInfo(torch.tensor([0])),
                    "dst_placements": [Shard(0)],
                }
            ],
            "backbone.layers.2": [
                {
                    "name": last_bf16_name,
                    "global_shape": [num_experts, intermediate_size, hidden_size],
                    "dtype": "torch.bfloat16",
                    "grouped_expert_proj": "up_proj",
                    "dst_mesh_info": MeshInfo(torch.tensor([0])),
                    "dst_placements": [Shard(0)],
                }
            ],
        },
    }
    first_bf16_runtime = torch.empty(
        num_experts, hidden_size // 16, intermediate_size, 16, dtype=torch.bfloat16
    )
    last_bf16_runtime = torch.empty(
        num_experts, hidden_size // 16, intermediate_size, 16, dtype=torch.bfloat16
    )
    mxfp8_runtime = torch.empty(
        num_experts, intermediate_size, hidden_size, dtype=torch.float8_e4m3fn
    )
    mxfp8_scale = torch.empty(
        num_experts, intermediate_size, hidden_size // 32, dtype=torch.uint8
    )
    ext = _make_ext(
        {
            first_bf16_runtime_name: first_bf16_runtime,
            mxfp8_runtime_name: mxfp8_runtime,
            f"{mxfp8_runtime_name}_scale_from_checkpoint": mxfp8_scale,
            last_bf16_runtime_name: last_bf16_runtime,
        }
    )
    ext.device = torch.device("cpu")
    ext.pp_comm_groups = {0: SimpleNamespace(rank=0)}
    _enable_trtllm_staging(ext)
    ext._unquantized_flashinfer_trtllm_param_ids = lambda: {
        id(first_bf16_runtime),
        id(last_bf16_runtime),
    }
    ext._load_full_hf_weights = MagicMock(return_value=None)

    def fake_quantize(weight):
        return (
            torch.full_like(weight, 3, dtype=torch.float8_e4m3fn),
            torch.full(
                (*weight.shape[:-1], weight.shape[-1] // 32),
                7,
                dtype=torch.uint8,
            ),
        )

    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.quantization.fp8.quantize_mxfp8_weight",
        fake_quantize,
    )

    specs = ext.build_hf_to_local_param_map(refit_info)
    first_bf16_spec = specs.get(first_bf16_name)
    mxfp8_spec = specs.get(mxfp8_name)
    last_bf16_spec = specs.get(last_bf16_name)
    assert first_bf16_spec is not None and first_bf16_spec.pre is not None
    assert first_bf16_spec.post is not None and first_bf16_spec.base is None
    assert mxfp8_spec is not None and mxfp8_spec.pre is not None
    assert mxfp8_spec.post is not None and mxfp8_spec.base is not None
    assert last_bf16_spec is not None and last_bf16_spec.pre is not None
    assert last_bf16_spec.post is not None and last_bf16_spec.base is None

    first_bf16_spec.post(first_bf16_spec.pre(first_bf16_spec.base))
    mxfp8_spec.post(mxfp8_spec.pre(mxfp8_spec.base))
    last_bf16_spec.post(last_bf16_spec.pre(last_bf16_spec.base))

    assert ext._load_full_hf_weights.call_count == 2
    loaded_source_names = {
        weight_name
        for call in ext._load_full_hf_weights.call_args_list
        for weight_name, _ in call.args[0]
    }
    assert loaded_source_names == {
        f"backbone.layers.0.mlp.experts.{expert}.up_proj.weight"
        for expert in range(num_experts)
    } | {
        f"backbone.layers.2.mlp.experts.{expert}.up_proj.weight"
        for expert in range(num_experts)
    }
    assert torch.all(mxfp8_runtime.float() == 3)
    assert torch.all(mxfp8_scale == 7)


def test_build_hf_to_local_param_map_stages_qwen35_wrapped_experts():
    """Qwen3.5 wrapper prefixes and RoutedExperts names survive staged reload."""
    hidden_size, num_experts, intermediate_size = 16, 4, 32
    hf_prefix = "model.language_model.layers.0.mlp.experts"
    runtime_prefix = "language_model.model.layers.0.mlp.experts"
    expert_name = f"{hf_prefix}.down_proj.weight"
    runtime_name = f"{runtime_prefix}.routed_experts.w2_weight"
    refit_info = {
        "gen_tp_size": 2,
        "layer_names": ["model.language_model.layers.0"],
        "per_layer_params": {
            "model.language_model.layers.0": [
                {
                    "name": expert_name,
                    "global_shape": [
                        num_experts,
                        hidden_size,
                        intermediate_size,
                    ],
                    "dtype": "torch.bfloat16",
                    "grouped_expert_proj": "down_proj",
                    "dst_mesh_info": MeshInfo(torch.tensor([8, 9])),
                    "dst_placements": [Shard(0)],
                }
            ]
        },
    }
    packed_w2 = torch.full((128, 16, 24, 64), 7.0)
    ext = _make_ext({runtime_name: packed_w2})
    ext.device = torch.device("cpu")
    ext.pp_comm_groups = {0: SimpleNamespace(rank=9)}
    _enable_trtllm_staging(ext)
    ext._load_full_hf_weights = MagicMock(return_value={f"{runtime_prefix}.w2_weight"})

    spec = ext.build_hf_to_local_param_map(refit_info).get(expert_name)
    assert spec is not None and spec.pre is not None and spec.post is not None
    spec.post(spec.pre(spec.base))

    loaded_weights = ext._load_full_hf_weights.call_args.args[0]
    assert [name for name, _ in loaded_weights] == [
        f"{hf_prefix}.2.down_proj.weight",
        f"{hf_prefix}.3.down_proj.weight",
    ]


def test_build_hf_to_local_param_map_requires_pp_groups_for_trtllm_staging():
    """A missing NCCL communicator fails before staged expert placement."""
    expert_name = "model.layers.0.mlp.experts.down_proj.weight"
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": expert_name,
                    "global_shape": [2, 16, 32],
                    "dtype": "torch.bfloat16",
                    "grouped_expert_proj": "down_proj",
                    "dst_mesh_info": MeshInfo(torch.tensor([0])),
                    "dst_placements": [Shard(0)],
                }
            ]
        },
    }
    ext = _make_ext(
        {"model.layers.0.mlp.experts.routed_experts.w2_weight": torch.empty(2, 16, 32)}
    )
    ext.device = torch.device("cpu")
    ext.pp_comm_groups = None
    _enable_trtllm_staging(ext)

    with pytest.raises(RuntimeError, match="before.*per-PP-stage groups"):
        ext.build_hf_to_local_param_map(refit_info)


def test_build_hf_to_local_param_map_rejects_missing_trtllm_destination():
    """The staged load must report the fused destination parameter."""
    hidden_size, num_experts, intermediate_size = 16, 4, 32
    expert_name = "model.layers.0.mlp.experts.down_proj.weight"
    refit_info = {
        "gen_tp_size": 2,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": expert_name,
                    "global_shape": [
                        num_experts,
                        hidden_size,
                        intermediate_size,
                    ],
                    "dtype": "torch.bfloat16",
                    "grouped_expert_proj": "down_proj",
                    "dst_mesh_info": MeshInfo(torch.tensor([8, 9])),
                    "dst_placements": [Shard(0)],
                }
            ]
        },
    }
    ext = _make_ext(
        {
            "model.layers.0.mlp.experts.routed_experts.w2_weight": torch.empty(
                128, 16, 24, 64
            ),
        }
    )
    ext.device = torch.device("cpu")
    ext.pp_comm_groups = {0: SimpleNamespace(rank=9)}
    _enable_trtllm_staging(ext)
    ext._load_full_hf_weights = MagicMock(
        return_value={"model.layers.0.mlp.experts.w13_weight"}
    )

    spec = ext.build_hf_to_local_param_map(refit_info).get(expert_name)
    assert spec is not None and spec.pre is not None and spec.post is not None

    with pytest.raises(RuntimeError, match="w2_weight"):
        spec.post(spec.pre(spec.base))


def test_build_hf_to_local_param_map_rejects_trtllm_tensor_sharding():
    """TRTLLM expert staging supports expert-parallel destination shards only."""
    expert_name = "model.layers.0.mlp.experts.gate_proj.weight"
    refit_info = {
        "gen_tp_size": 2,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": expert_name,
                    "global_shape": [4, 32, 16],
                    "dtype": "torch.bfloat16",
                    "grouped_expert_proj": "gate_proj",
                    "dst_mesh_info": MeshInfo(torch.tensor([8, 9])),
                    "dst_placements": [Shard(1)],
                }
            ]
        },
    }
    ext = _make_ext(
        {
            "model.layers.0.mlp.experts.routed_experts.w13_weight": torch.empty(
                128, 16, 24, 64
            ),
        }
    )
    _enable_trtllm_staging(ext)

    with pytest.raises(ValueError, match="unsupported tensor shard dimensions"):
        ext.build_hf_to_local_param_map(refit_info)


def test_prepare_nccl_reshard_refit_info_validates_before_building_map(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext._validate_native_layerwise_refit = MagicMock(
        side_effect=RuntimeError("unsupported weight update")
    )
    ext.build_hf_to_local_param_map = MagicMock()
    restore_refit_info_placements = MagicMock()
    monkeypatch.setattr(
        "nemo_rl.weight_sync.nccl_reshard_utils.restore_refit_info_placements",
        restore_refit_info_placements,
    )

    with pytest.raises(RuntimeError, match="unsupported weight update"):
        ext.prepare_nccl_reshard_refit_info({"layer_names": []})

    ext._validate_native_layerwise_refit.assert_called_once_with("nccl_reshard")
    restore_refit_info_placements.assert_not_called()
    ext.build_hf_to_local_param_map.assert_not_called()
    assert not hasattr(ext, "nccl_reshard_refit_info")


def test_nccl_reshard_trtllm_refit_rejects_fp8_kv_cache(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.model_runner = SimpleNamespace(model=object())
    ext._uses_unquantized_flashinfer_trtllm = lambda: True
    ext._uses_fp8_kv_cache = lambda: True
    monkeypatch.setattr(
        vllm_backend,
        "_unquantized_flashinfer_trtllm_modules",
        lambda _model: [SimpleNamespace(expert_placement_strategy="linear")],
    )

    with pytest.raises(RuntimeError, match="FP8 KV cache"):
        ext._validate_native_layerwise_refit("nccl_reshard")


def test_legacy_refit_map_is_built_after_comm_groups_exist(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.device = torch.device("cpu")
    ext._validate_native_layerwise_refit = MagicMock()
    expected_map = HFToLocalParamMap()
    ext.build_hf_to_local_param_map = MagicMock(return_value=expected_map)
    refit_info = {"layer_names": [], "per_layer_params": {}}
    monkeypatch.setattr(
        "nemo_rl.weight_sync.nccl_reshard_utils.restore_refit_info_placements",
        lambda value: value,
    )

    class _FakeGroup:
        def __init__(self, *, rank, **_kwargs):
            self.rank = rank

        def init_nccl_communicator(self, *, device):
            assert device == torch.device("cpu")

    monkeypatch.setattr(
        "nemo_rl.distributed.stateless_process_group.StatelessProcessGroup",
        _FakeGroup,
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    ext.init_nccl_reshard_comm_group(
        rank_prefix=0,
        pp_ips=["127.0.0.1"],
        pp_ports=[29500],
        pp_size=1,
        train_ranks_per_stage=8,
        sub_world_size=10,
    )

    assert ext.pp_comm_groups[0].rank == 8
    assert ext.build_hf_to_local_param_map.call_count == 0

    ext.prepare_nccl_reshard_refit_info(refit_info)

    ext.build_hf_to_local_param_map.assert_called_once_with(refit_info)
    assert ext.hf_to_local_param_map is expected_map


def test_nccl_reshard_lifecycle_repeats_for_trtllm_moe_modules(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    model = torch.nn.Module()
    trtllm_moe = torch.nn.Module()
    model_config = object()
    vllm_config = object()
    call_order = []
    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.model_runner = SimpleNamespace(model=model, vllm_config=vllm_config)
    ext.model_config = model_config
    ext.device = torch.device("cpu")
    ext._uses_unquantized_flashinfer_trtllm = lambda: True
    ext._validate_native_layerwise_refit = lambda _transport=None: None
    ext._maybe_process_mtp_drafter_after_loading = lambda: call_order.append("mtp")

    monkeypatch.setattr(
        vllm_backend,
        "_unquantized_flashinfer_trtllm_modules",
        lambda _model: [trtllm_moe],
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.config.set_current_vllm_config", lambda _: contextlib.nullcontext()
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        vllm_backend,
        "_refresh_hpc_modules_after_layerwise_reload",
        lambda _model: None,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.initialize_layerwise_reload",
        lambda module: call_order.append(("initialize", module)),
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.finalize_layerwise_reload",
        lambda reload_model, config: call_order.append(
            ("finalize", reload_model, config)
        ),
    )

    for cycle in range(2):
        with ext._weight_update_lifecycle("nccl_reshard") as finalize:
            call_order.append(("transfer", cycle))
            finalize()

    assert call_order == [
        ("initialize", trtllm_moe),
        ("transfer", 0),
        ("finalize", model, model_config),
        "mtp",
        ("initialize", trtllm_moe),
        ("transfer", 1),
        ("finalize", model, model_config),
        "mtp",
    ]


def test_nccl_reshard_refit_runs_transport_lifecycle(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    ext.nccl_reshard_refit_info = {
        "layer_names": [],
        "per_layer_params": {},
        "misc_meta": {},
    }
    ext.pp_comm_groups = {}
    ext._receive_and_load_misc_params = MagicMock()
    ext._maybe_process_fp8_kv_cache = MagicMock()
    finalize = MagicMock()
    lifecycle_calls = []

    @contextlib.contextmanager
    def lifecycle(transport):
        lifecycle_calls.append(transport)
        yield finalize

    ext._weight_update_lifecycle = lifecycle
    monkeypatch.setattr(torch.cuda, "Stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.process_weights_after_loading",
        lambda *_args: pytest.fail("transport lifecycle must own finalization"),
    )

    assert ext.nccl_reshard_refit() is True
    assert lifecycle_calls == ["nccl_reshard"]
    finalize.assert_called_once_with()


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
