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

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple, Union

import torch
from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from torch import Tensor

try:
    from megatron.core.extensions.transformer_engine import TENorm  # noqa: F401

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


class EagleSelfAttention(SelfAttention):
    """SelfAttention with configurable QKV input size."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        qkv_input_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            **kwargs,
        )

        if qkv_input_size is not None and qkv_input_size != config.hidden_size:
            self.linear_qkv = build_module(
                submodules.linear_qkv,
                qkv_input_size,
                self.linear_qkv_out_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="qkv",
                tp_group=self.pg_collection.tp,
            )


class EagleLayer(TransformerLayer):
    """Single Eagle layer that concatenates embeddings with hidden states."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        **kwargs,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            **kwargs,
        )

        if HAVE_TE:
            self.hidden_norm = TENorm(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        else:
            from megatron.core.transformer.torch_norm import WrappedTorchNorm

            self.hidden_norm = WrappedTorchNorm(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )

    def _forward_attention(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        embeds: Optional[Tensor] = None,
        **kwargs,
    ):
        assert embeds is not None, "EagleLayer requires `embeds` input."

        residual = hidden_states
        normalized_embeds = self.input_layernorm(embeds)
        normalized_hidden_states = self.hidden_norm(hidden_states)
        attention_input = torch.cat(
            [normalized_embeds, normalized_hidden_states], dim=-1
        )

        attention_output_with_bias = self.self_attention(
            attention_input,
            attention_mask=attention_mask,
            **{
                key: value
                for key, value in kwargs.items()
                if key not in ("context", "context_mask", "embeds")
            },
        )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.hidden_dropout)

        return hidden_states, None


def _make_eagle_layer_spec(
    base_spec: ModuleSpec,
    qkv_input_size: int,
) -> ModuleSpec:
    spec = copy.deepcopy(base_spec)
    spec.module = EagleLayer

    attention_spec = spec.submodules.self_attention
    attention_spec.module = EagleSelfAttention
    if attention_spec.params is None:
        attention_spec.params = {}
    attention_spec.params["qkv_input_size"] = qkv_input_size

    return spec


class EagleModel(MegatronModule):
    """Single-layer Eagle model with FC reduction + decoder + final norm."""

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        self.config = config
        self.hidden_size = config.hidden_size

        self.fc = ColumnParallelLinear(
            self.hidden_size * 3,
            self.hidden_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=True,
            skip_bias_add=True,
        )

        self.rotary_pos_emb = RotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_percent=getattr(config, "rotary_percent", 1.0),
            rotary_interleaved=getattr(config, "rotary_interleaved", False),
            seq_len_interpolation_factor=getattr(
                config, "seq_len_interpolation_factor", None
            ),
            rotary_base=getattr(config, "rotary_base", 10000),
        )

        base_layer_spec = get_gpt_layer_local_spec(
            qk_layernorm=getattr(config, "qk_layernorm", False),
            normalization=config.normalization,
        )
        eagle_layer_spec = _make_eagle_layer_spec(
            base_layer_spec, qkv_input_size=2 * self.hidden_size
        )
        self.layer = build_module(eagle_layer_spec, config=config, layer_number=1)

        if HAVE_TE:
            self.norm = TENorm(
                config=config,
                hidden_size=self.hidden_size,
                eps=config.layernorm_epsilon,
            )
        else:
            from megatron.core.transformer.torch_norm import WrappedTorchNorm

            self.norm = WrappedTorchNorm(
                config=config,
                hidden_size=self.hidden_size,
                eps=config.layernorm_epsilon,
            )

    def forward(
        self,
        hidden_states: Tensor,
        input_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        hidden_states, _ = self.fc(hidden_states)

        seq_length = hidden_states.shape[0] * self.config.context_parallel_size
        rotary_pos_emb = self.rotary_pos_emb(seq_length)
        rotary_pos_emb = (rotary_pos_emb, rotary_pos_emb)

        if self.config.sequence_parallel:
            from megatron.core.tensor_parallel import (
                scatter_to_sequence_parallel_region,
            )

            hidden_states = scatter_to_sequence_parallel_region(hidden_states)

        hidden_states, _ = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            embeds=input_embeds,
        )

        return self.norm(hidden_states)


class Eagle3ForCausalLM(MegatronModule):
    """Eagle3 draft model used for online speculative decoding training."""

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        self.config = config
        self.model = EagleModel(config=config)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
        )

    @classmethod
    def from_config(
        cls,
        config: TransformerConfig,
        fp16: bool = False,
        bf16: bool = True,
        *,
        num_layers: int | None = None,
    ) -> "Eagle3ForCausalLM":
        del num_layers  # Eagle3 currently uses a single draft layer.

        eagle_config = TransformerConfig(
            num_layers=1,
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.ffn_hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_query_groups=config.num_query_groups,
            kv_channels=config.kv_channels,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            layernorm_epsilon=config.layernorm_epsilon,
            add_bias_linear=config.add_bias_linear,
            gated_linear_unit=True,
            activation_func=torch.nn.functional.silu,
            normalization="RMSNorm",
            tensor_model_parallel_size=config.tensor_model_parallel_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=config.context_parallel_size,
            sequence_parallel=config.sequence_parallel,
            attention_softmax_in_fp32=False,
            fp16=fp16,
            bf16=bf16,
        )

        for attr_name in (
            "vocab_size",
            "rotary_base",
            "rotary_percent",
            "rotary_interleaved",
            "rope_scaling",
            "rope_scaling_factor",
        ):
            if hasattr(config, attr_name):
                setattr(eagle_config, attr_name, getattr(config, attr_name))

        for attr_name, attr_value in config.__dict__.items():
            if attr_name not in eagle_config.__dict__:
                setattr(eagle_config, attr_name, attr_value)

        eagle_config.num_layers_in_first_pipeline_stage = None
        eagle_config.num_layers_in_last_pipeline_stage = None
        return cls(config=eagle_config)

    def forward(
        self,
        hidden_states: Tensor,
        input_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        compute_logits: bool = True,
    ) -> Union[Tensor, Tensor]:
        hidden_states = self.model(
            hidden_states=hidden_states,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
        )

        if not compute_logits:
            return hidden_states

        logits, _ = self.lm_head(hidden_states)
        return logits.transpose(0, 1).contiguous()

    def combine_hidden_states(self, hidden_states: Tensor) -> Tensor:
        combined_hidden_states, _ = self.model.fc(hidden_states)
        return combined_hidden_states


def create_config_from_hf(
    hf_config,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
) -> TransformerConfig:
    """Create a Megatron TransformerConfig from a HF draft-model config."""
    head_dim = getattr(hf_config, "head_dim", None)
    if head_dim is None:
        head_dim = hf_config.hidden_size // hf_config.num_attention_heads

    config = TransformerConfig(
        num_layers=1,
        hidden_size=hf_config.hidden_size,
        ffn_hidden_size=hf_config.intermediate_size,
        num_attention_heads=hf_config.num_attention_heads,
        num_query_groups=getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads
        ),
        kv_channels=head_dim,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=getattr(hf_config, "rms_norm_eps", 1e-5),
        add_bias_linear=bool(getattr(hf_config, "mlp_bias", False)),
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        normalization="RMSNorm",
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        sequence_parallel=sequence_parallel,
    )
    config.vocab_size = hf_config.vocab_size
    config.rotary_base = getattr(hf_config, "rope_theta", 500000.0)
    config.rotary_percent = 1.0
    config.rotary_interleaved = False
    config.rope_scaling = getattr(hf_config, "rope_scaling", None) is not None
    if getattr(hf_config, "rope_scaling", None) is not None:
        config.rope_scaling_factor = hf_config.rope_scaling.get("factor", None)

    return config


def load_hf_weights_to_eagle(
    model: Eagle3ForCausalLM,
    hf_state_dict: Dict[str, Tensor],
    config: TransformerConfig,
) -> Tuple[list, list]:
    """Load HF Eagle weights into Megatron Eagle with TP-aware sharding."""
    del config

    new_state: Dict[str, Tensor] = {}
    model_state = model.state_dict()

    prefix = "model.layer"
    model_keys = set(model_state.keys())
    has_unfused_attn_ln = f"{prefix}.input_layernorm.weight" in model_keys
    has_fused_attn_ln = (
        f"{prefix}.self_attention.linear_qkv.layer_norm_weight" in model_keys
    )
    has_unfused_mlp_ln = f"{prefix}.pre_mlp_layernorm.weight" in model_keys
    has_fused_mlp_ln = f"{prefix}.mlp.linear_fc1.layer_norm_weight" in model_keys
    has_hidden_norm = f"{prefix}.hidden_norm.weight" in model_keys

    tp_rank = (
        parallel_state.get_tensor_model_parallel_rank()
        if parallel_state.model_parallel_is_initialized()
        else 0
    )

    split_axis_by_parameter = {
        "model.fc.weight": 0,
        f"{prefix}.self_attention.linear_qkv.weight": 0,
        f"{prefix}.self_attention.linear_proj.weight": 1,
        f"{prefix}.mlp.linear_fc1.weight": 0,
        f"{prefix}.mlp.linear_fc2.weight": 1,
        "lm_head.weight": 0,
    }

    def _shape(tensor: Tensor) -> tuple[int, ...]:
        return tuple(tensor.shape)

    def _shard_to_local_tp(parameter_name: str, tensor: Tensor) -> Tensor:
        target = model_state.get(parameter_name)
        if target is None:
            return tensor

        if _shape(tensor) == _shape(target):
            return tensor.to(dtype=target.dtype)

        split_axis = split_axis_by_parameter.get(parameter_name)
        if split_axis is None:
            raise RuntimeError(
                f"[specdec] Unexpected shape mismatch for non-TP key '{parameter_name}': "
                f"checkpoint={_shape(tensor)} model={_shape(target)}"
            )

        full_dim = tensor.shape[split_axis]
        local_dim = target.shape[split_axis]
        if local_dim <= 0 or full_dim % local_dim != 0:
            raise RuntimeError(
                f"[specdec] Cannot infer TP sharding for '{parameter_name}': "
                f"checkpoint={_shape(tensor)} model={_shape(target)}"
            )

        inferred_tp = full_dim // local_dim
        if tp_rank >= inferred_tp:
            raise RuntimeError(
                f"[specdec] tp_rank={tp_rank} out of range for key '{parameter_name}' "
                f"(inferred_tp={inferred_tp})"
            )

        local_shard = torch.chunk(tensor, inferred_tp, dim=split_axis)[tp_rank]
        local_shard = local_shard.contiguous()
        if _shape(local_shard) != _shape(target):
            raise RuntimeError(
                f"[specdec] Invalid TP shard shape for '{parameter_name}': "
                f"got={_shape(local_shard)} expected={_shape(target)}"
            )
        return local_shard.to(dtype=target.dtype)

    q_weight: Optional[Tensor] = None
    k_weight: Optional[Tensor] = None
    v_weight: Optional[Tensor] = None
    gate_weight: Optional[Tensor] = None
    up_weight: Optional[Tensor] = None
    qkv_weight: Optional[Tensor] = None
    fc1_weight: Optional[Tensor] = None

    for hf_key, hf_weight in hf_state_dict.items():
        if hf_key in ("fc.weight", "model.fc.weight"):
            new_state["model.fc.weight"] = hf_weight
            continue
        if hf_key in ("norm.weight", "model.norm.weight"):
            new_state["model.norm.weight"] = hf_weight
            continue
        if hf_key in ("lm_head.weight", "model.lm_head.weight"):
            new_state["lm_head.weight"] = hf_weight
            continue
        if not hf_key.startswith("midlayer."):
            continue

        if hf_key.endswith("self_attn.qkv_proj.weight"):
            qkv_weight = hf_weight
        elif hf_key.endswith("self_attn.q_proj.weight"):
            q_weight = hf_weight
        elif hf_key.endswith("self_attn.k_proj.weight"):
            k_weight = hf_weight
        elif hf_key.endswith("self_attn.v_proj.weight"):
            v_weight = hf_weight
        elif hf_key.endswith("self_attn.o_proj.weight"):
            new_state[f"{prefix}.self_attention.linear_proj.weight"] = hf_weight
        elif hf_key.endswith("mlp.gate_up_proj.weight"):
            fc1_weight = hf_weight
        elif hf_key.endswith("mlp.gate_proj.weight"):
            gate_weight = hf_weight
        elif hf_key.endswith("mlp.up_proj.weight"):
            up_weight = hf_weight
        elif hf_key.endswith("mlp.down_proj.weight"):
            new_state[f"{prefix}.mlp.linear_fc2.weight"] = hf_weight
        elif hf_key.endswith("hidden_norm.weight"):
            if has_hidden_norm:
                new_state[f"{prefix}.hidden_norm.weight"] = hf_weight
        elif hf_key.endswith("input_layernorm.weight"):
            if has_unfused_attn_ln:
                new_state[f"{prefix}.input_layernorm.weight"] = hf_weight
            elif has_fused_attn_ln:
                new_state[f"{prefix}.self_attention.linear_qkv.layer_norm_weight"] = (
                    hf_weight
                )
        elif hf_key.endswith("post_attention_layernorm.weight"):
            if has_unfused_mlp_ln:
                new_state[f"{prefix}.pre_mlp_layernorm.weight"] = hf_weight
            elif has_fused_mlp_ln:
                new_state[f"{prefix}.mlp.linear_fc1.layer_norm_weight"] = hf_weight

    if qkv_weight is not None:
        new_state[f"{prefix}.self_attention.linear_qkv.weight"] = qkv_weight
    else:
        any_qkv = any(t is not None for t in (q_weight, k_weight, v_weight))
        if any_qkv:
            if not (
                q_weight is not None and k_weight is not None and v_weight is not None
            ):
                raise RuntimeError(
                    "[specdec] Incomplete QKV tensors. Expected q_proj, k_proj, and v_proj."
                )
            new_state[f"{prefix}.self_attention.linear_qkv.weight"] = torch.cat(
                [q_weight, k_weight, v_weight], dim=0
            )

    if fc1_weight is not None:
        new_state[f"{prefix}.mlp.linear_fc1.weight"] = fc1_weight
    else:
        any_fc1 = any(t is not None for t in (gate_weight, up_weight))
        if any_fc1:
            if not (gate_weight is not None and up_weight is not None):
                raise RuntimeError(
                    "[specdec] Incomplete MLP tensors. Expected gate_proj and up_proj."
                )
            new_state[f"{prefix}.mlp.linear_fc1.weight"] = torch.cat(
                [gate_weight, up_weight], dim=0
            )

    for parameter_name in list(new_state.keys()):
        new_state[parameter_name] = _shard_to_local_tp(
            parameter_name, new_state[parameter_name]
        )

    return model.load_state_dict(new_state, strict=False)


def save_eagle_weights_to_hf(
    model: Eagle3ForCausalLM,
    config: TransformerConfig,
) -> Dict[str, Tensor]:
    """Export Megatron Eagle weights to HF naming."""
    raw_model = model
    while hasattr(raw_model, "module"):
        raw_model = raw_model.module

    source_state = raw_model.state_dict()
    hf_state: Dict[str, Tensor] = {}

    q_dim = config.num_attention_heads * config.kv_channels
    kv_dim = config.num_query_groups * config.kv_channels
    ffn_hidden_size = config.ffn_hidden_size

    for key, weight in source_state.items():
        if key == "model.fc.weight":
            hf_state["fc.weight"] = weight
        elif key == "model.norm.weight":
            hf_state["norm.weight"] = weight
        elif key == "lm_head.weight":
            hf_state["lm_head.weight"] = weight
        elif key == "model.layer.input_layernorm.weight":
            hf_state["midlayer.input_layernorm.weight"] = weight
        elif key == "model.layer.hidden_norm.weight":
            hf_state["midlayer.hidden_norm.weight"] = weight
        elif key == "model.layer.pre_mlp_layernorm.weight":
            hf_state["midlayer.post_attention_layernorm.weight"] = weight
        elif key == "model.layer.self_attention.linear_qkv.weight":
            if weight.shape[0] == q_dim + 2 * kv_dim:
                hf_state["midlayer.self_attn.qkv_proj.weight"] = weight
            else:
                q_proj, k_proj, v_proj = weight.split([q_dim, kv_dim, kv_dim], dim=0)
                hf_state["midlayer.self_attn.q_proj.weight"] = q_proj
                hf_state["midlayer.self_attn.k_proj.weight"] = k_proj
                hf_state["midlayer.self_attn.v_proj.weight"] = v_proj
        elif key == "model.layer.self_attention.linear_proj.weight":
            hf_state["midlayer.self_attn.o_proj.weight"] = weight
        elif key == "model.layer.mlp.linear_fc1.weight":
            gate_proj, up_proj = weight.split([ffn_hidden_size, ffn_hidden_size], dim=0)
            hf_state["midlayer.mlp.gate_proj.weight"] = gate_proj
            hf_state["midlayer.mlp.up_proj.weight"] = up_proj
        elif key == "model.layer.mlp.linear_fc2.weight":
            hf_state["midlayer.mlp.down_proj.weight"] = weight

    return hf_state
