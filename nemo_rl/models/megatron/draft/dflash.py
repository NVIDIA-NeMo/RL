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

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import (
    get_pg_size,
    get_tensor_model_parallel_group_if_none,
)
from torch import Tensor, nn
from torch.distributed import ProcessGroup

from nemo_rl.models.megatron.draft.block_attention import (
    dflash_block_only_attention,
)
from nemo_rl.models.megatron.draft.block_plan import DFlashBatchPlan

if TYPE_CHECKING:
    from megatron.core.dist_checkpointing.mapping import ShardedStateDict
    from nemo_rl.models.megatron.draft.sequence_layout import DraftSequenceLayout


_ShardedOffsets = tuple[tuple[int, int, int], ...]
_INPUT_ERROR_MESSAGES = (
    "target_taps shape does not match the DFlash plan",
    "block_embeddings shape does not match the DFlash plan",
    "DFlash plan and inputs must share a device",
    "DFlash inputs must share a device",
    "DFlash inputs must share a dtype",
    "DFlash inputs must use a floating dtype",
    "DFlash query positions must use torch.int64",
)
_INPUT_TYPE_ERROR_CODES = frozenset({4, 5, 6})


@dataclass(frozen=True, slots=True)
class DFlashBodyConfig:
    """Checkpoint-compatible DFlash body dimensions."""

    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    num_hidden_layers: int = 5
    num_target_taps: int = 5
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        integer_fields = (
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.num_hidden_layers,
            self.num_target_taps,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("DFlash dimensions must be positive")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        for field_name, value in (
            ("rope_theta", self.rope_theta),
            ("rms_norm_eps", self.rms_norm_eps),
            ("initializer_range", self.initializer_range),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")


class QwenRMSNorm(nn.Module):
    """Qwen RMS normalization with FP32 variance accumulation."""

    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float,
        params_dtype: torch.dtype,
        gradient_reduce_group: ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=params_dtype))
        self.eps = eps
        if get_pg_size(gradient_reduce_group) > 1:
            self.weight.register_hook(
                lambda gradient: self._sum_gradient(
                    gradient,
                    group=gradient_reduce_group,
                )
            )

    @staticmethod
    def _sum_gradient(gradient: Tensor, *, group: ProcessGroup | None) -> Tensor:
        torch.distributed.all_reduce(gradient, group=group)
        return gradient

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.to(torch.float32)
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


class _ColumnParallelProjection(ColumnParallelLinear):
    """MCore column projection with a tensor-only forward contract."""

    def __call__(self, hidden_states: Tensor) -> Tensor:
        return super().__call__(hidden_states)

    def _save_to_state_dict(
        self,
        destination: dict[str, Any],
        prefix: str,
        keep_vars: bool,
    ) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination.pop(f"{prefix}_extra_state", None)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.tp_group is None:
            return nn.functional.linear(hidden_states, self.weight, self.bias)
        output, _ = super().forward(hidden_states)
        return output


class _RowParallelProjection(RowParallelLinear):
    """MCore row projection with a tensor-only forward contract."""

    def __call__(self, hidden_states: Tensor) -> Tensor:
        return super().__call__(hidden_states)

    def _save_to_state_dict(
        self,
        destination: dict[str, Any],
        prefix: str,
        keep_vars: bool,
    ) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination.pop(f"{prefix}_extra_state", None)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.tp_group is None:
            return nn.functional.linear(hidden_states, self.weight, self.bias)
        output, _ = super().forward(hidden_states)
        return output


class _ShardedModule(nn.Module):
    """Small MCore-checkpoint-aware module used by the DFlash hierarchy."""

    def __init__(self, *, tp_group: ProcessGroup | None) -> None:
        super().__init__()
        self.tp_group = tp_group

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: _ShardedOffsets = (),
        metadata: dict[str, Any] | None = None,
    ) -> ShardedStateDict:
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        direct_state: dict[str, Any] = {}
        self._save_to_state_dict(direct_state, "", keep_vars=True)
        sharded = make_sharded_tensors_for_checkpoint(
            direct_state,
            prefix,
            sharded_offsets=sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )
        for name, module in self.named_children():
            sharded.update(
                sharded_state_dict_default(
                    module,
                    f"{prefix}{name}.",
                    sharded_offsets,
                    metadata,
                    tp_group=self.tp_group,
                )
            )
        return sharded


class _DFlashAttention(_ShardedModule):
    def __init__(
        self,
        config: DFlashBodyConfig,
        *,
        parallel_config: ModelParallelConfig,
        tp_group: ProcessGroup | None,
        init_method: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__(tp_group=tp_group)
        query_size = config.num_attention_heads * config.head_dim
        key_value_size = config.num_key_value_heads * config.head_dim
        self.q_proj = _ColumnParallelProjection(
            config.hidden_size,
            query_size,
            config=parallel_config,
            init_method=init_method,
            bias=False,
            gather_output=False,
            tp_group=tp_group,
        )
        self.k_proj = _ColumnParallelProjection(
            config.hidden_size,
            key_value_size,
            config=parallel_config,
            init_method=init_method,
            bias=False,
            gather_output=False,
            tp_group=tp_group,
        )
        self.v_proj = _ColumnParallelProjection(
            config.hidden_size,
            key_value_size,
            config=parallel_config,
            init_method=init_method,
            bias=False,
            gather_output=False,
            tp_group=tp_group,
        )
        self.o_proj = _RowParallelProjection(
            query_size,
            config.hidden_size,
            config=parallel_config,
            init_method=init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            tp_group=tp_group,
        )
        self.q_norm = QwenRMSNorm(
            config.head_dim,
            eps=config.rms_norm_eps,
            params_dtype=parallel_config.params_dtype,
            gradient_reduce_group=tp_group,
        )
        self.k_norm = QwenRMSNorm(
            config.head_dim,
            eps=config.rms_norm_eps,
            params_dtype=parallel_config.params_dtype,
            gradient_reduce_group=tp_group,
        )


class _DFlashMLP(_ShardedModule):
    def __init__(
        self,
        config: DFlashBodyConfig,
        *,
        parallel_config: ModelParallelConfig,
        tp_group: ProcessGroup | None,
        init_method: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__(tp_group=tp_group)
        self.gate_proj = _ColumnParallelProjection(
            config.hidden_size,
            config.intermediate_size,
            config=parallel_config,
            init_method=init_method,
            bias=False,
            gather_output=False,
            tp_group=tp_group,
        )
        self.up_proj = _ColumnParallelProjection(
            config.hidden_size,
            config.intermediate_size,
            config=parallel_config,
            init_method=init_method,
            bias=False,
            gather_output=False,
            tp_group=tp_group,
        )
        self.down_proj = _RowParallelProjection(
            config.intermediate_size,
            config.hidden_size,
            config=parallel_config,
            init_method=init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            tp_group=tp_group,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        gated = nn.functional.silu(self.gate_proj(hidden_states))
        return self.down_proj(gated * self.up_proj(hidden_states))


class _DFlashDecoderLayer(_ShardedModule):
    def __init__(
        self,
        config: DFlashBodyConfig,
        *,
        parallel_config: ModelParallelConfig,
        tp_group: ProcessGroup | None,
        init_method: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__(tp_group=tp_group)
        self.self_attn = _DFlashAttention(
            config,
            parallel_config=parallel_config,
            tp_group=tp_group,
            init_method=init_method,
        )
        self.mlp = _DFlashMLP(
            config,
            parallel_config=parallel_config,
            tp_group=tp_group,
            init_method=init_method,
        )
        self.input_layernorm = QwenRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            params_dtype=parallel_config.params_dtype,
        )
        self.post_attention_layernorm = QwenRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            params_dtype=parallel_config.params_dtype,
        )


def _rotate_half(hidden_states: Tensor) -> Tensor:
    first_half, second_half = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


def _build_rope_table(
    positions: Tensor,
    *,
    head_dim: int,
    theta: float,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(
                0,
                head_dim,
                2,
                dtype=torch.float32,
                device=positions.device,
            )
            / head_dim
        )
    )
    frequencies = positions.to(torch.float32).unsqueeze(-1) * inv_freq
    angles = torch.cat((frequencies, frequencies), dim=-1)
    cosine = angles.cos().to(dtype).unsqueeze(-2)
    sine = angles.sin().to(dtype).unsqueeze(-2)
    return cosine, sine


def _apply_rope(
    hidden_states: Tensor,
    *,
    cosine: Tensor,
    sine: Tensor,
) -> Tensor:
    return hidden_states * cosine + _rotate_half(hidden_states) * sine


class DFlashBody(_ShardedModule):
    """DFlash decoder body that shares target embeddings and output head."""

    def __init__(
        self,
        config: DFlashBodyConfig | None = None,
        *,
        tp_group: ProcessGroup | None = None,
        parallel_config: ModelParallelConfig | None = None,
    ) -> None:
        tp_group = get_tensor_model_parallel_group_if_none(tp_group)
        super().__init__(tp_group=tp_group)
        self.config = DFlashBodyConfig() if config is None else config
        self.tensor_parallel_size = get_pg_size(tp_group)
        if parallel_config is None:
            parallel_config = ModelParallelConfig(
                tensor_model_parallel_size=self.tensor_parallel_size,
                use_cpu_initialization=True,
                bf16=True,
                params_dtype=torch.bfloat16,
            )
        elif parallel_config.sequence_parallel:
            raise ValueError("DFlashBody does not support sequence_parallel=True")
        elif parallel_config.context_parallel_size > 1:
            raise ValueError("DFlashBody does not support context_parallel_size > 1")
        elif parallel_config.tensor_model_parallel_size != self.tensor_parallel_size:
            raise ValueError(
                "parallel_config.tensor_model_parallel_size must match the "
                "DFlash tensor parallel group size"
            )
        self.parallel_config = parallel_config
        partitioned_dimensions = (
            self.config.hidden_size,
            self.config.intermediate_size,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
        )
        if any(
            value % self.tensor_parallel_size != 0 for value in partitioned_dimensions
        ):
            raise ValueError(
                "DFlash dimensions must be divisible by tensor parallel size"
            )
        self.num_attention_heads_per_partition = (
            self.config.num_attention_heads // self.tensor_parallel_size
        )
        self.num_key_value_heads_per_partition = (
            self.config.num_key_value_heads // self.tensor_parallel_size
        )

        def init_method(weight: Tensor) -> Tensor:
            return nn.init.normal_(
                weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

        self.fc = _ColumnParallelProjection(
            self.config.num_target_taps * self.config.hidden_size,
            self.config.hidden_size,
            config=parallel_config,
            init_method=init_method,
            bias=False,
            gather_output=True,
            tp_group=tp_group,
        )
        self.hidden_norm = QwenRMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            params_dtype=parallel_config.params_dtype,
        )
        self.layers = nn.ModuleList(
            _DFlashDecoderLayer(
                self.config,
                parallel_config=parallel_config,
                tp_group=tp_group,
                init_method=init_method,
            )
            for _ in range(self.config.num_hidden_layers)
        )
        self.norm = QwenRMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            params_dtype=parallel_config.params_dtype,
        )

    def _validate_inputs(
        self,
        *,
        target_taps: Tensor,
        block_embeddings: Tensor,
        plan: DFlashBatchPlan,
        sequence_layout: DraftSequenceLayout | None,
    ) -> None:
        expected_target_shape = (
            (
                plan.batch_size,
                plan.sequence_length,
                self.config.num_target_taps,
                self.config.hidden_size,
            )
            if sequence_layout is None
            else (
                1,
                sequence_layout.cp_global_positions.numel(),
                self.config.num_target_taps,
                self.config.hidden_size,
            )
        )
        if target_taps.shape != expected_target_shape:
            error_code = 0
        else:
            num_blocks = plan.sample_rows.numel()
            expected_block_shape = (
                num_blocks,
                plan.block_size,
                self.config.hidden_size,
            )
            plan_tensors = (
                plan.token_valid_mask,
                plan.sample_rows,
                plan.anchor_positions,
                plan.query_positions,
                plan.slot_valid,
            )
            if sequence_layout is not None:
                plan_tensors = (*plan_tensors, sequence_layout.cp_global_positions)
            if block_embeddings.shape != expected_block_shape:
                error_code = 1
            elif any(tensor.device != target_taps.device for tensor in plan_tensors):
                error_code = 2
            elif block_embeddings.device != target_taps.device:
                error_code = 3
            elif block_embeddings.dtype != target_taps.dtype:
                error_code = 4
            elif not target_taps.dtype.is_floating_point:
                error_code = 5
            elif plan.query_positions.dtype != torch.int64:
                error_code = 6
            else:
                error_code = len(_INPUT_ERROR_MESSAGES)

        if self.tensor_parallel_size > 1:
            if self.tp_group is None:
                raise RuntimeError("DFlash tensor parallel group is unavailable")
            synchronized_error = torch.tensor(
                error_code,
                dtype=torch.int64,
                device=self.fc.weight.device,
            )
            torch.distributed.all_reduce(
                synchronized_error,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            error_code = int(synchronized_error.item())

        if error_code < len(_INPUT_ERROR_MESSAGES):
            message = _INPUT_ERROR_MESSAGES[error_code]
            if error_code in _INPUT_TYPE_ERROR_CODES:
                raise TypeError(message)
            raise ValueError(message)

    def forward(
        self,
        *,
        target_taps: Tensor,
        block_embeddings: Tensor,
        plan: DFlashBatchPlan,
        sequence_layout: DraftSequenceLayout | None = None,
        context_parallel_group: ProcessGroup | None = None,
    ) -> Tensor:
        """Update anchored block embeddings from target hidden-state taps."""
        self._validate_inputs(
            target_taps=target_taps,
            block_embeddings=block_embeddings,
            plan=plan,
            sequence_layout=sequence_layout,
        )
        config = self.config
        batch_size, sequence_length = target_taps.shape[:2]
        target_hidden = self.hidden_norm(self.fc(target_taps.flatten(start_dim=2)))
        hidden_states = torch.where(
            plan.slot_valid[..., None],
            block_embeddings,
            torch.zeros_like(block_embeddings),
        )
        if sequence_layout is None:
            trunk_positions = torch.arange(
                sequence_length,
                dtype=torch.int64,
                device=target_taps.device,
            ).expand(batch_size, -1)
            block_positions = plan.query_positions
        else:
            trunk_positions = sequence_layout.packed_logical_positions[
                sequence_layout.cp_global_positions
            ].clamp_min(0)[None, :]
            block_positions = plan.packed_rope_positions
        trunk_cosine, trunk_sine = _build_rope_table(
            trunk_positions,
            head_dim=config.head_dim,
            theta=config.rope_theta,
            dtype=target_taps.dtype,
        )
        block_cosine, block_sine = _build_rope_table(
            block_positions,
            head_dim=config.head_dim,
            theta=config.rope_theta,
            dtype=target_taps.dtype,
        )

        for layer_module in self.layers:
            layer = cast(_DFlashDecoderLayer, layer_module)
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)
            trunk_key = layer.self_attn.k_norm(
                layer.self_attn.k_proj(target_hidden).view(
                    batch_size,
                    sequence_length,
                    self.num_key_value_heads_per_partition,
                    config.head_dim,
                )
            )
            trunk_value = layer.self_attn.v_proj(target_hidden).view(
                batch_size,
                sequence_length,
                self.num_key_value_heads_per_partition,
                config.head_dim,
            )
            block_query = layer.self_attn.q_norm(
                layer.self_attn.q_proj(normalized).view(
                    *normalized.shape[:2],
                    self.num_attention_heads_per_partition,
                    config.head_dim,
                )
            )
            block_key = layer.self_attn.k_norm(
                layer.self_attn.k_proj(normalized).view(
                    *normalized.shape[:2],
                    self.num_key_value_heads_per_partition,
                    config.head_dim,
                )
            )
            block_value = layer.self_attn.v_proj(normalized).view(
                *normalized.shape[:2],
                self.num_key_value_heads_per_partition,
                config.head_dim,
            )
            trunk_key = _apply_rope(
                trunk_key,
                cosine=trunk_cosine,
                sine=trunk_sine,
            )
            block_query = _apply_rope(
                block_query,
                cosine=block_cosine,
                sine=block_sine,
            )
            block_key = _apply_rope(
                block_key,
                cosine=block_cosine,
                sine=block_sine,
            )
            attention_output = dflash_block_only_attention(
                plan=plan,
                trunk_k=trunk_key,
                trunk_v=trunk_value,
                block_q=block_query,
                block_k=block_key,
                block_v=block_value,
                sequence_layout=sequence_layout,
                context_parallel_group=context_parallel_group,
            )
            hidden_states = residual + layer.self_attn.o_proj(
                attention_output.flatten(start_dim=2)
            )
            hidden_states = hidden_states + layer.mlp(
                layer.post_attention_layernorm(hidden_states)
            )

        output = self.norm(hidden_states)
        return torch.where(
            plan.slot_valid[..., None],
            output,
            torch.zeros_like(output),
        )

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: _ShardedOffsets = (),
        metadata: dict[str, Any] | None = None,
    ) -> ShardedStateDict:
        """Return MCore shards with public names and global checkpoint shapes."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        sharded: ShardedStateDict = {}
        for name in ("fc", "hidden_norm", "norm"):
            sharded.update(
                sharded_state_dict_default(
                    getattr(self, name),
                    f"{prefix}{name}.",
                    sharded_offsets,
                    metadata,
                    tp_group=self.tp_group,
                )
            )
        for layer_index, layer in enumerate(self.layers):
            sharded.update(
                sharded_state_dict_default(
                    layer,
                    f"{prefix}layers.{layer_index}.",
                    sharded_offsets,
                    metadata,
                    tp_group=self.tp_group,
                )
            )
        return sharded


__all__ = ["DFlashBody", "DFlashBodyConfig", "QwenRMSNorm"]
