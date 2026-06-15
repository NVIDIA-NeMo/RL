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

"""Unit tests for the vLLM-side nccl_xfer refit mapping (CPU, no GPU).

Covers ``nemo_rl/models/generation/vllm/vllm_backend.py``:
- ``_fused_param_merge_slice`` — the dim-0 sub-slice math for vLLM's fused
  params (qkv_proj / gate_up_proj / fused_qkv_a_proj), all three branches.
- ``_build_hf_to_gen_backend_mapping`` — HF-name -> (vLLM param, slice) mapping,
  driven by a synthetic ``refit_info`` and a fake ``named_parameters()`` (no real
  vLLM model, no GPU).

``vllm_backend`` does ``import vllm`` at module top, so these are vllm-marked and
skipped where vllm is unavailable.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")  # module-top `import vllm` in vllm_backend

from nemo_rl.models.generation.vllm.vllm_backend import (  # noqa: E402
    VllmInternalWorkerExtension,
    _fused_param_merge_slice,
)

pytestmark = pytest.mark.vllm


# --------------------------------------------------------------------------
# _fused_param_merge_slice — the 3 branches
# --------------------------------------------------------------------------
def test_merge_slice_even_split_qkv():
    # q/k/v each global dim0=512; vLLM qkv_proj local dim0 = 512*3/4 = 384 (TP=4).
    prefix = "L.self_attn."
    suffixes = ["q_proj.weight", "k_proj.weight", "v_proj.weight"]
    hf_shapes = {prefix + s: (512, 32) for s in suffixes}
    assert _fused_param_merge_slice(hf_shapes, prefix, suffixes, 0, 384, 4) == (
        slice(0, 128),
    )
    assert _fused_param_merge_slice(hf_shapes, prefix, suffixes, 1, 384, 4) == (
        slice(128, 256),
    )
    assert _fused_param_merge_slice(hf_shapes, prefix, suffixes, 2, 384, 4) == (
        slice(256, 384),
    )


def test_merge_slice_fully_replicated():
    # DeepSeek MLA fused_qkv_a_proj (disable_tp): every rank holds the full concat.
    prefix = "L.self_attn."
    suffixes = ["q_a_proj.weight", "kv_a_proj_with_mqa.weight"]
    hf_shapes = {
        prefix + "q_a_proj.weight": (1536, 32),
        prefix + "kv_a_proj_with_mqa.weight": (512, 32),
    }
    local_dim0 = 1536 + 512  # == sum(global) -> replicated branch
    assert _fused_param_merge_slice(hf_shapes, prefix, suffixes, 0, local_dim0, 2) == (
        slice(0, 1536),
    )
    assert _fused_param_merge_slice(hf_shapes, prefix, suffixes, 1, local_dim0, 2) == (
        slice(1536, 2048),
    )


def test_merge_slice_kv_head_replication_fallback():
    # tp doesn't divide k/v evenly: q splits by tp, the rest share equally.
    prefix = "L.self_attn."
    suffixes = ["q_proj.weight", "k_proj.weight", "v_proj.weight"]
    hf_shapes = {
        prefix + "q_proj.weight": (512, 32),
        prefix + "k_proj.weight": (128, 32),
        prefix + "v_proj.weight": (128, 32),
    }
    # local_dim0=256: not sum(naive)=192, not sum(global)=768 -> fallback branch.
    # q -> 512/4=128; rest 128 split over {k,v} -> 64 each.
    assert _fused_param_merge_slice(hf_shapes, prefix, suffixes, 0, 256, 4) == (
        slice(0, 128),
    )
    assert _fused_param_merge_slice(hf_shapes, prefix, suffixes, 1, 256, 4) == (
        slice(128, 192),
    )
    assert _fused_param_merge_slice(hf_shapes, prefix, suffixes, 2, 256, 4) == (
        slice(192, 256),
    )


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


def test_build_mapping_all_cases():
    H, E, Pl, vocab = 32, 2, 64, 100
    refit_info = {
        "gen_tp_size": 4,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {"name": "model.layers.0.input_layernorm.weight", "global_shape": [H]},
                {
                    "name": "model.layers.0.self_attn.q_proj.weight",
                    "global_shape": [512, H],
                },
                {
                    "name": "model.layers.0.self_attn.k_proj.weight",
                    "global_shape": [512, H],
                },
                {
                    "name": "model.layers.0.self_attn.v_proj.weight",
                    "global_shape": [512, H],
                },
                {
                    "name": "model.layers.0.mlp.gate_proj.weight",
                    "global_shape": [256, H],
                },
                {"name": "model.layers.0.mlp.up_proj.weight", "global_shape": [256, H]},
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
                {"name": "lm_head.weight", "global_shape": [vocab, H]},
            ]
        },
    }
    layernorm = _param(H)
    qkv = _param(384, H)  # 512*3/4
    gate_up = _param(128, H)  # 256*2/4
    w13 = _param(E, 2 * Pl, H)  # gated: gate||up on intermediate axis (dim 1)
    w2 = _param(E, H, Pl)
    embed = _param(vocab, H)
    vllm_params = {
        "model.layers.0.input_layernorm.weight": layernorm,
        "model.layers.0.self_attn.qkv_proj.weight": qkv,
        "model.layers.0.mlp.gate_up_proj.weight": gate_up,
        "model.layers.0.mlp.experts.w13_weight": w13,
        "model.layers.0.mlp.experts.w2_weight": w2,
        "model.embed_tokens.weight": embed,
        # NOTE: no "lm_head.weight" -> exercises the tied-embedding fallback.
    }
    mapping = _make_ext(vllm_params)._build_hf_to_gen_backend_mapping(refit_info)

    # Direct 1:1
    assert mapping["model.layers.0.input_layernorm.weight"] == (layernorm, None)
    # QKV merge -> qkv_proj, dim-0 sub-slices
    assert mapping["model.layers.0.self_attn.q_proj.weight"] == (qkv, (slice(0, 128),))
    assert mapping["model.layers.0.self_attn.k_proj.weight"] == (
        qkv,
        (slice(128, 256),),
    )
    assert mapping["model.layers.0.self_attn.v_proj.weight"] == (
        qkv,
        (slice(256, 384),),
    )
    # Dense gate/up -> gate_up_proj (NOT collided with grouped experts)
    assert mapping["model.layers.0.mlp.gate_proj.weight"] == (gate_up, (slice(0, 64),))
    assert mapping["model.layers.0.mlp.up_proj.weight"] == (gate_up, (slice(64, 128),))
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
    # lm_head tied to embed_tokens
    assert mapping["lm_head.weight"] == (embed, None)


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
