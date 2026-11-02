# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Direct expert load must leave the same bytes as TRT-LLM's own reload.

``NcclExtension._load_expert_weights_direct`` replays a copy of the loader's
MoE branch (``_load_weights_impl_v2``). This loads the same received expert
stacks into a real TRT-LLM model both ways and compares every expert parameter
bit for bit, so a TRT-LLM bump that changes the loader fails here instead of
silently loading different weights.
"""

import json
from functools import partial
from types import SimpleNamespace

import pytest
import torch

pytestmark = [
    pytest.mark.trtllm,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

_PREFIX = "model.language_model.layers.0.mlp.experts"
_NUM_EXPERTS, _HIDDEN, _INTER = 16, 2048, 512
_EP_SIZE = 2
_LOCAL_EXPERTS = _NUM_EXPERTS // _EP_SIZE


def _write_tiny_qwen35_moe_vl_config(path):
    """One-layer Qwen3.5-MoE VLM: the decoder sits under the wrapper's ``llm``."""
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "dtype": "bfloat16",
            "attention_bias": False,
            "full_attention_interval": 4,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": _HIDDEN,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 4096,
            "mlp_only_layers": [],
            "moe_intermediate_size": _INTER,
            "norm_topk_prob": True,
            "num_attention_heads": 32,
            "num_experts": _NUM_EXPERTS,
            "num_experts_per_tok": 4,
            "num_hidden_layers": 1,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "shared_expert_intermediate_size": _INTER,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 1e6,
                "rope_type": "default",
            },
            "vocab_size": 1024,
        },
        "vision_config": {
            "model_type": "qwen3_5_moe",
            "deepstack_visual_indexes": [],
            "depth": 1,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "in_channels": 3,
            "intermediate_size": 4304,
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": _HIDDEN,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    return path


def _extension_on_tiny_model(tmp_path, ep_rank):
    """An NcclExtension over one EP rank of a real BF16 TRT-LLM model, on one GPU.

    The rank's model holds only its own expert slots. Attention DP and
    all-gather MoE communication keep construction free of the peer buffers a
    TP allreduce or an NVLink all-to-all would exchange with the other rank.
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.checkpoints.auto_mapper import (
        AutoCheckpointMapper,
    )
    from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
    from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
    from tensorrt_llm.mapping import Mapping

    from nemo_rl.models.generation.trtllm.trtllm_backend import NcclExtension

    pretrained = load_pretrained_config(str(_write_tiny_qwen35_moe_vl_config(tmp_path)))
    model_config = ModelConfig(
        pretrained_config=pretrained,
        mapping=Mapping(
            world_size=_EP_SIZE,
            tp_size=_EP_SIZE,
            moe_ep_size=_EP_SIZE,
            moe_tp_size=1,
            rank=ep_rank,
            enable_attention_dp=True,
        ),
    )
    model = AutoModelForCausalLM.from_config(model_config).to("cuda")
    mapper = AutoCheckpointMapper.get("HF", "Qwen3_5MoeForConditionalGeneration")
    # Bound to the VLM wrapper, as the engine leaves it before the first reload.
    mapper.init_model_and_config(model, model_config)
    # ModelLoader.reload only reads these two attributes of the loader.
    model_loader = SimpleNamespace(
        weight_mapper=mapper,
        _call_load_weights=partial(ModelLoader._call_load_weights, None),
    )

    ext = NcclExtension.__new__(NcclExtension)
    ext.engine = SimpleNamespace(
        model_engine=SimpleNamespace(model=model, model_loader=model_loader)
    )
    ext._moe_weight_owner_map = None
    return ext, model, mapper, partial(ModelLoader.reload, model_loader, model)


def _received_stacks(generator):
    """One random BF16 ``[E, out, in]`` stack per projection, as the reshard delivers."""
    shapes = {
        "gate_proj": (_INTER, _HIDDEN),
        "up_proj": (_INTER, _HIDDEN),
        "down_proj": (_HIDDEN, _INTER),
    }
    return {
        projection: torch.randn(
            _LOCAL_EXPERTS,
            *shape,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        for projection, shape in shapes.items()
    }


@pytest.mark.parametrize(
    "split_by_projection",
    [False, True],
    ids=["one_batch", "one_call_per_projection"],
)
@pytest.mark.parametrize(
    "ep_rank", range(_EP_SIZE), ids=lambda r: f"ep{_EP_SIZE}_rank{r}"
)
def test_direct_expert_load_matches_model_loader_reload(
    tmp_path, monkeypatch, ep_rank, split_by_projection
):
    monkeypatch.setenv("TRTLLM_FORCE_COMM_METHOD", "ALLGATHER")
    ext, model, mapper, reload = _extension_on_tiny_model(tmp_path, ep_rank)
    # The reshard delivers exactly this rank's experts, in slot order.
    expert_ids = list(range(ep_rank * _LOCAL_EXPERTS, (ep_rank + 1) * _LOCAL_EXPERTS))
    generator = torch.Generator(device="cuda").manual_seed(0)

    def converted_batches(stacks):
        batches = [
            ext._convert_received_experts(_PREFIX, projection, expert_ids, stack)
            for projection, stack in stacks.items()
        ]
        if split_by_projection:
            return batches
        return [{name: t for batch in batches for name, t in batch.items()}]

    def load(batches, direct):
        # The refit runs every load inside one mapper update session.
        mapper.begin_update_weights()
        for batch in batches:
            if direct:
                assert ext._load_expert_weights_direct(dict(batch)) == {}
            else:
                reload(dict(batch), allow_partial_loading=True)
        mapper.finalize_update_weights()
        torch.cuda.synchronize()

    def expert_params():
        # Raw bytes, so even a -0.0/+0.0 or NaN-payload difference counts.
        return {
            name: p.detach().contiguous().view(torch.uint8).clone()
            for name, p in model.named_parameters()
            if ".mlp.experts." in name
        }

    weights = converted_batches(_received_stacks(generator))
    other_weights = converted_batches(_received_stacks(generator))
    initial = expert_params()
    assert initial
    assert all(p.shape[0] == _LOCAL_EXPERTS for p in initial.values())

    # Direct first, while the mapper is still bound to the wrapper.
    load(weights, direct=True)
    direct = expert_params()
    load(other_weights, direct=False)
    overwritten = expert_params()
    load(weights, direct=False)
    reloaded = expert_params()

    for name in initial:
        # Both loads must have written something, or the comparison is vacuous.
        assert not torch.equal(direct[name], initial[name]), name
        assert not torch.equal(overwritten[name], direct[name]), name
        assert torch.equal(direct[name], reloaded[name]), name
