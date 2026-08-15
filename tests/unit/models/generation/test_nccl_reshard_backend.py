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
                },
                {"name": "model.layers.0.mlp.up_proj.weight", "global_shape": [256, H]},
                {
                    "name": "model.layers.0.mlp.down_proj.weight",
                    "global_shape": [H, 256],
                },
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
    ext._uses_unquantized_flashinfer_trtllm = lambda: True
    ext._load_full_hf_weights = MagicMock()

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


def test_nccl_reshard_lifecycle_initializes_only_trtllm_moe_modules(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    model = SimpleNamespace()
    trtllm_moe = SimpleNamespace()
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
    ext._validate_weight_update_compatibility = lambda: None
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
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
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

    with ext._weight_update_lifecycle("nccl_reshard") as finalize:
        call_order.append("transfer")
        finalize()

    assert call_order == [
        ("initialize", trtllm_moe),
        "transfer",
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
