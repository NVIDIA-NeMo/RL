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
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Iterator

import torch
from torch import Tensor

_FLEX_BLOCK_SIZE = 128


class EagleTTTResourceLimitError(RuntimeError):
    """Raised when one TTT invocation retains more storage than its limit."""


StorageKey = tuple[str, int | None, int, int]


class EagleTTTResourceLedger:
    """Account unique additional storages owned or saved by one TTT invocation."""

    def __init__(self, *, limit_bytes: int) -> None:
        if limit_bytes < 1:
            raise ValueError(f"limit_bytes must be positive, got {limit_bytes}")
        self.limit_bytes = limit_bytes
        self._excluded: set[StorageKey] = set()
        self._owned: dict[StorageKey, tuple[int, str]] = {}
        self._autograd: dict[StorageKey, int] = {}

    @staticmethod
    def _storage(tensor: Tensor) -> tuple[StorageKey, int]:
        storage = tensor.untyped_storage()
        size = storage.nbytes()
        return (
            tensor.device.type,
            tensor.device.index,
            storage.data_ptr(),
            size,
        ), size

    def exclude(self, tensors: tuple[Tensor, ...]) -> None:
        """Exclude caller-owned storages that existed before the invocation."""
        for tensor in tensors:
            key, _ = self._storage(tensor)
            self._excluded.add(key)

    def track_owned(self, tensor: Tensor, *, category: str) -> None:
        """Classify one retained adapter storage as explicitly owned state."""
        key, size = self._storage(tensor)
        if key in self._excluded or key in self._owned:
            return
        self._autograd.pop(key, None)
        self._owned[key] = (size, category)
        self._check_limit()

    @property
    def owned_bytes(self) -> int:
        return sum(size for size, _ in self._owned.values())

    @property
    def autograd_bytes(self) -> int:
        return sum(self._autograd.values())

    @property
    def total_bytes(self) -> int:
        return self.owned_bytes + self.autograd_bytes

    def _check_limit(self) -> None:
        if self.total_bytes > self.limit_bytes:
            categories: dict[str, int] = {}
            for size, category in self._owned.values():
                categories[category] = categories.get(category, 0) + size
            category_text = ", ".join(
                f"{name}={size}" for name, size in sorted(categories.items())
            )
            if category_text:
                category_text = f", categories: {category_text}"
            raise EagleTTTResourceLimitError(
                "EAGLE TTT resource limit exceeded: "
                f"owned={self.owned_bytes}, autograd={self.autograd_bytes}, "
                f"total={self.total_bytes}, limit={self.limit_bytes} bytes"
                f"{category_text}"
            )

    @contextmanager
    def saved_tensors(self) -> Iterator[None]:
        """Account unique storages retained by autograd within this context."""

        def pack(tensor: Tensor) -> Tensor:
            key, size = self._storage(tensor)
            if (
                key not in self._excluded
                and key not in self._owned
                and key not in self._autograd
            ):
                self._autograd[key] = size
                self._check_limit()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            yield

    def reset(self) -> None:
        """Release all invocation accounting state."""
        self._excluded.clear()
        self._owned.clear()
        self._autograd.clear()


def _validate_pass_bounds(*, pass_count: int, max_passes: int) -> None:
    if max_passes < 1:
        raise ValueError(f"max_passes must be positive, got {max_passes}")
    if pass_count < 1:
        raise ValueError(f"pass_count must be positive, got {pass_count}")
    if pass_count > max_passes:
        raise ValueError(
            f"pass_count {pass_count} exceeds the configured maximum {max_passes}"
        )


@dataclass(frozen=True, slots=True)
class EagleTTTAttentionPlan:
    """Scalar-only metadata for one inference-aligned EAGLE training pass."""

    pass_index: int
    pass_count: int
    max_passes: int
    sequence_length: int

    def __post_init__(self) -> None:
        _validate_pass_bounds(
            pass_count=self.pass_count,
            max_passes=self.max_passes,
        )
        if not 0 <= self.pass_index < self.pass_count:
            raise ValueError(
                f"pass_index must be in [0, {self.pass_count}), got {self.pass_index}"
            )
        if self.sequence_length < 1:
            raise ValueError(
                f"sequence_length must be positive, got {self.sequence_length}"
            )

    @property
    def teacher_offset(self) -> int:
        """Return the target-logit offset predicted by this pass."""
        return self.pass_index + 1

    @property
    def student_slice(self) -> slice:
        """Return ModelOpt's student rows for this pass."""
        return slice(self.pass_index, -1)

    @property
    def teacher_slice(self) -> slice:
        """Return the target rows paired with :attr:`student_slice`."""
        return slice(self.pass_index + 1, None)

    def rope_positions(self, *, device: torch.device | None = None) -> Tensor:
        """Materialize the one base position vector shared by every pass."""
        return torch.arange(
            self.sequence_length,
            dtype=torch.int64,
            device=device,
        )

    def dense_visibility_mask(self, *, device: torch.device | None = None) -> Tensor:
        """Materialize a small dense oracle mask; never retain it in pass state."""
        query_positions = torch.arange(self.sequence_length, device=device)
        trunk_positions = torch.arange(self.sequence_length, device=device)
        trunk_visible = (
            trunk_positions[None, :] <= query_positions[:, None] - self.pass_index
        )
        if self.pass_index == 0:
            return trunk_visible
        branch_visible = tuple(
            trunk_positions[None, :]
            == query_positions[:, None] - (self.pass_index - branch_index - 1)
            for branch_index in range(self.pass_index)
        )
        return torch.cat((trunk_visible, *branch_visible), dim=1)


@dataclass(frozen=True, slots=True)
class EagleTTTSequenceLayout:
    """Compact padding and packed-document metadata for one token sequence."""

    valid_tokens: Tensor
    document_ids: Tensor

    def __post_init__(self) -> None:
        if self.valid_tokens.ndim != 2 or self.document_ids.ndim != 2:
            raise ValueError(
                "valid_tokens and document_ids must have [batch, sequence] shape"
            )
        if self.valid_tokens.shape != self.document_ids.shape:
            raise ValueError("valid_tokens and document_ids must have the same shape")
        if self.valid_tokens.device != self.document_ids.device:
            raise ValueError("valid_tokens and document_ids must share a device")
        if self.valid_tokens.dtype != torch.bool:
            raise ValueError("valid_tokens must have Boolean dtype")
        if self.document_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("document_ids must have an integer dtype")
        if self.valid_tokens.shape[0] < 1 or self.valid_tokens.shape[1] < 1:
            raise ValueError("sequence layout dimensions must be positive")
        if torch.any(self.document_ids[self.valid_tokens] < 0):
            raise ValueError("valid tokens must have a nonnegative document id")
        if torch.any(self.document_ids[~self.valid_tokens] != -1):
            raise ValueError("invalid tokens must use the -1 document sentinel")

    @property
    def batch_size(self) -> int:
        return self.valid_tokens.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.valid_tokens.shape[1]

    @classmethod
    def unpacked(
        cls,
        *,
        batch_size: int,
        sequence_length: int,
        valid_lengths: Tensor | None = None,
        device: torch.device | None = None,
    ) -> EagleTTTSequenceLayout:
        """Create a layout for an unpacked batch with optional right padding."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if valid_lengths is None:
            lengths = torch.full(
                (batch_size,),
                sequence_length,
                dtype=torch.int64,
                device=device,
            )
        else:
            if valid_lengths.shape != (batch_size,):
                raise ValueError(
                    f"valid_lengths must have shape ({batch_size},), "
                    f"got {valid_lengths.shape}"
                )
            if valid_lengths.dtype not in (torch.int32, torch.int64):
                raise ValueError("valid_lengths must have int32 or int64 dtype")
            lengths = valid_lengths.to(device=device, dtype=torch.int64)
            if torch.any(lengths < 0) or torch.any(lengths > sequence_length):
                raise ValueError("valid_lengths must lie within the sequence length")
        positions = torch.arange(sequence_length, device=lengths.device)
        valid_tokens = positions[None, :] < lengths[:, None]
        document_ids = torch.where(
            valid_tokens,
            torch.zeros((), dtype=torch.int64, device=lengths.device),
            torch.full((), -1, dtype=torch.int64, device=lengths.device),
        )
        return cls(valid_tokens=valid_tokens, document_ids=document_ids)

    @classmethod
    def from_cu_seqlens(
        cls,
        *,
        cu_seqlens: Tensor,
        sequence_length: int,
        device: torch.device | None = None,
    ) -> EagleTTTSequenceLayout:
        """Create a batch-one packed layout from MCore cumulative lengths."""
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError(
                "cu_seqlens must be a rank-1 tensor with at least two entries"
            )
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must have int32 or int64 dtype")
        cumulative = cu_seqlens.to(device=device, dtype=torch.int64)
        if cumulative[0].item() != 0:
            raise ValueError("cu_seqlens must start at zero")
        lengths = cumulative[1:] - cumulative[:-1]
        if torch.any(lengths < 0):
            raise ValueError("cu_seqlens must be monotonic")
        valid_length = cumulative[-1].item()
        if valid_length > sequence_length:
            raise ValueError("cu_seqlens may not exceed the sequence length")
        document_ids = torch.full(
            (sequence_length,),
            -1,
            dtype=torch.int64,
            device=cumulative.device,
        )
        document_ids[:valid_length] = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=cumulative.device),
            lengths,
        )
        valid_tokens = (
            torch.arange(
                sequence_length,
                device=cumulative.device,
            )
            < valid_length
        )
        return cls(
            valid_tokens=valid_tokens.unsqueeze(0),
            document_ids=document_ids.unsqueeze(0),
        )


def with_eagle_ttt_core_attention(spec: Any) -> Any:
    """Copy and adapt one MCore layer spec or ModelOpt transformer block spec."""
    adapted = deepcopy(spec)
    layer_specs = getattr(adapted, "layer_specs", None)
    candidates = (adapted,) if layer_specs is None else tuple(layer_specs)
    if not candidates:
        raise TypeError("block spec must expose at least one transformer layer spec")
    try:
        for layer_spec in candidates:
            self_attention = layer_spec.submodules.self_attention
            self_attention.submodules.core_attention = EagleTTTCoreAttention
    except AttributeError as error:
        raise TypeError(
            "each layer spec must expose self_attention core-attention submodules"
        ) from error
    return adapted


@dataclass(frozen=True, slots=True)
class EagleTTTStoragePlan:
    """Upper bound for all additional state retained by multi-pass training."""

    batch_size: int
    kv_heads: int
    sequence_length: int
    head_dim: int
    dtype: torch.dtype
    pass_count: int
    max_passes: int
    activation_budget_bytes: int
    layer_count: int = 1
    hidden_size: int = 0
    rope_dim: int = 0

    def __post_init__(self) -> None:
        _validate_pass_bounds(
            pass_count=self.pass_count,
            max_passes=self.max_passes,
        )
        dimensions = {
            "batch_size": self.batch_size,
            "kv_heads": self.kv_heads,
            "sequence_length": self.sequence_length,
            "head_dim": self.head_dim,
            "layer_count": self.layer_count,
        }
        for name, value in dimensions.items():
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        for name, value in {
            "hidden_size": self.hidden_size,
            "rope_dim": self.rope_dim,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative, got {value}")
        if self.activation_budget_bytes < 1:
            raise ValueError(
                "activation_budget_bytes must be positive, "
                f"got {self.activation_budget_bytes}"
            )
        if self.retained_bytes > self.activation_budget_bytes:
            raise ValueError(
                "packed EAGLE TTT state exceeds the activation budget: "
                f"requires {self.retained_bytes} bytes for {self.pass_count} passes, "
                f"budget is {self.activation_budget_bytes} bytes"
            )

    @property
    def retained_bytes(self) -> int:
        """Return the conservative total retained activation bound."""
        return (
            self.kv_bytes
            + self.hidden_bytes
            + self.rope_bytes
            + self.mask_bytes
            + self.loss_bytes
        )

    @property
    def kv_bytes(self) -> int:
        """Return per-layer trunk and prior-branch K/V bytes."""
        element_size = torch.empty((), dtype=self.dtype).element_size()
        elements_per_tensor = (
            self.batch_size * self.kv_heads * self.sequence_length * self.head_dim
        )
        return (
            self.layer_count * self.pass_count * 2 * elements_per_tensor * element_size
        )

    @property
    def hidden_bytes(self) -> int:
        """Return full-sequence branch outputs retained for pass backward."""
        element_size = torch.empty((), dtype=self.dtype).element_size()
        return (
            self.pass_count
            * self.batch_size
            * self.sequence_length
            * self.hidden_size
            * element_size
        )

    @property
    def rope_bytes(self) -> int:
        """Return the single shared base RoPE table bound."""
        element_size = torch.empty((), dtype=self.dtype).element_size()
        return self.sequence_length * self.rope_dim * element_size

    @property
    def mask_bytes(self) -> int:
        """Return cached Flex BlockMask metadata for every pass.

        PyTorch retains four int32 block-count vectors and four int32 block-index
        matrices (forward and transposed, partial and full blocks). This bound
        deliberately assumes every block-index matrix is full.
        """
        block_count = math.ceil(self.sequence_length / _FLEX_BLOCK_SIZE)
        return (
            self.batch_size
            * self.pass_count
            * 16
            * (block_count + block_count * block_count)
        )

    @property
    def loss_bytes(self) -> int:
        """Return projected-loss row indices, bins, and FP32 normalizer bytes."""
        if self.hidden_size == 0:
            return 0
        rows = sum(
            self.batch_size * max(self.sequence_length - pass_index - 1, 0)
            for pass_index in range(self.pass_count)
        )
        return rows * (2 * torch.int64.itemsize + 2 * torch.float32.itemsize)


@dataclass(frozen=True, slots=True)
class EagleTTTState:
    """Immutable references to one causal trunk and append-only prior branch K/V."""

    pass_count: int
    max_passes: int
    activation_budget_bytes: int
    trunk_key: Tensor
    trunk_value: Tensor
    branch_keys: tuple[Tensor, ...] = ()
    branch_values: tuple[Tensor, ...] = ()

    @classmethod
    def from_trunk(
        cls,
        *,
        trunk_key: Tensor,
        trunk_value: Tensor,
        pass_count: int,
        max_passes: int,
        activation_budget_bytes: int,
    ) -> EagleTTTState:
        """Validate the complete bound before retaining any supplied tensor."""
        _validate_kv_pair(trunk_key, trunk_value, name="trunk")
        EagleTTTStoragePlan(
            batch_size=trunk_key.shape[0],
            kv_heads=trunk_key.shape[1],
            sequence_length=trunk_key.shape[2],
            head_dim=trunk_key.shape[3],
            dtype=trunk_key.dtype,
            pass_count=pass_count,
            max_passes=max_passes,
            activation_budget_bytes=activation_budget_bytes,
        )
        return cls(
            pass_count=pass_count,
            max_passes=max_passes,
            activation_budget_bytes=activation_budget_bytes,
            trunk_key=trunk_key,
            trunk_value=trunk_value,
        )

    def append_branch(
        self,
        *,
        branch_key: Tensor,
        branch_value: Tensor,
    ) -> EagleTTTState:
        """Return a state retaining one additional same-anchor branch K/V pair."""
        if len(self.branch_keys) >= self.pass_count - 1:
            raise ValueError(
                f"pass_count {self.pass_count} retains at most "
                f"{self.pass_count - 1} prior branches"
            )
        _validate_kv_pair(branch_key, branch_value, name="branch")
        if branch_key.shape != self.trunk_key.shape:
            raise ValueError("branch K/V shape must match the causal trunk")
        if (
            branch_key.device != self.trunk_key.device
            or branch_key.dtype != self.trunk_key.dtype
        ):
            raise ValueError("branch K/V must share trunk device and dtype")
        return replace(
            self,
            branch_keys=(*self.branch_keys, branch_key),
            branch_values=(*self.branch_values, branch_value),
        )

    @property
    def retained_tensor_bytes(self) -> int:
        """Return the bytes represented by retained K/V tensor references."""
        tensors = (
            self.trunk_key,
            self.trunk_value,
            *self.branch_keys,
            *self.branch_values,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


class EagleTTTCoreAttention(torch.nn.Module):
    """MCore core-attention adapter for one EAGLE TTT decoder layer."""

    def __init__(
        self,
        *,
        config: Any,
        layer_number: int,
        attn_mask_type: Any,
        attention_type: str,
        cp_comm_type: str | None = None,
        softmax_scale: float | None = None,
        pg_collection: Any = None,
    ) -> None:
        super().__init__()
        del attn_mask_type, cp_comm_type
        if attention_type != "self":
            raise ValueError("EAGLE TTT only supports self attention")
        self.layer_number = layer_number
        self.context_parallel_size = int(getattr(config, "context_parallel_size", 1))
        self.sequence_parallel = bool(getattr(config, "sequence_parallel", False))
        self.softmax_scale = softmax_scale
        self.tp_group = getattr(pg_collection, "tp", None)
        self.layout: EagleTTTSequenceLayout | None = None
        self.plan: EagleTTTAttentionPlan | None = None
        self.state: EagleTTTState | None = None
        self._storage_plan: EagleTTTStoragePlan | None = None
        self._resource_ledger: EagleTTTResourceLedger | None = None
        self._packed_seq_params: object | None = None
        self._block_mask: Any = None
        self._called = False

    def arm(
        self,
        *,
        layout: EagleTTTSequenceLayout,
        storage_plan: EagleTTTStoragePlan,
        resource_ledger: EagleTTTResourceLedger,
        packed_seq_params: object | None,
    ) -> None:
        if self.layout is not None:
            raise RuntimeError(f"EAGLE TTT layer {self.layer_number} is already armed")
        if self.context_parallel_size != 1:
            raise ValueError("EAGLE TTT context parallel size must be one")
        self.layout = layout
        self._storage_plan = storage_plan
        self._resource_ledger = resource_ledger
        self._packed_seq_params = packed_seq_params

    def begin_pass(
        self,
        plan: EagleTTTAttentionPlan,
        *,
        block_mask: Any = None,
    ) -> None:
        if self.layout is None or self._storage_plan is None:
            raise RuntimeError(f"EAGLE TTT layer {self.layer_number} is not armed")
        if self.plan is not None:
            raise RuntimeError(
                f"EAGLE TTT layer {self.layer_number} has an unfinished pass"
            )
        expected_pass = 0 if self.state is None else len(self.state.branch_keys) + 1
        if plan.pass_index != expected_pass:
            raise ValueError(
                f"EAGLE TTT layer {self.layer_number} expected pass "
                f"{expected_pass}, got {plan.pass_index}"
            )
        if plan.pass_count != self._storage_plan.pass_count:
            raise ValueError("attention plan and session pass counts must match")
        if plan.sequence_length != self.layout.sequence_length:
            raise ValueError("attention plan and session sequence lengths must match")
        if block_mask is not None:
            resource_ledger = self._resource_ledger
            if resource_ledger is None:
                raise RuntimeError("EAGLE TTT resource ledger is not armed")
            for name in (
                "kv_num_blocks",
                "kv_indices",
                "full_kv_num_blocks",
                "full_kv_indices",
                "q_num_blocks",
                "q_indices",
                "full_q_num_blocks",
                "full_q_indices",
            ):
                tensor = getattr(block_mask, name, None)
                if isinstance(tensor, Tensor):
                    resource_ledger.track_owned(tensor, category="mask")
        self.plan = plan
        self._block_mask = block_mask
        self._called = False

    def finish_pass(self) -> None:
        if self.plan is None or not self._called:
            raise RuntimeError(
                f"EAGLE TTT layer {self.layer_number} did not execute its active pass"
            )
        self.plan = None
        self._block_mask = None
        self._called = False

    def reset(self) -> None:
        self.layout = None
        self.plan = None
        self.state = None
        self._storage_plan = None
        self._resource_ledger = None
        self._packed_seq_params = None
        self._block_mask = None
        self._called = False

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        *,
        attn_mask_type: Any,
        attention_bias: Tensor | None,
        packed_seq_params: object | None,
    ) -> Tensor:
        del attn_mask_type
        if self.layout is None or self.plan is None or self._storage_plan is None:
            raise RuntimeError(
                f"EAGLE TTT layer {self.layer_number} called outside an active pass"
            )
        if self._called:
            raise RuntimeError(
                f"EAGLE TTT layer {self.layer_number} executed twice in one pass"
            )
        if attention_mask is not None:
            raise ValueError(
                "dense attention_mask is unsupported; represent visibility in "
                "EagleTTTSequenceLayout"
            )
        if attention_bias is not None:
            raise ValueError("EAGLE TTT does not support an attention bias")
        if packed_seq_params is not self._packed_seq_params:
            raise ValueError("packed sequence parameters must match the active session")
        is_thd = (
            packed_seq_params is not None
            and getattr(packed_seq_params, "qkv_format", None) == "thd"
        )
        if is_thd and query.ndim == key.ndim == value.ndim == 3:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("MCore query, key, and value must be rank-4 tensors")
        if key.shape != value.shape:
            raise ValueError("MCore key and value shapes must match")
        if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
            raise ValueError(
                "MCore query and key sequence, batch, and head dims must match"
            )
        if (query.shape[1], query.shape[0]) != (
            self.layout.batch_size,
            self.layout.sequence_length,
        ):
            raise ValueError(
                "MCore attention shape must match the active sequence layout"
            )

        query_bhsd = query.permute(1, 2, 0, 3)
        key_bhsd = key.permute(1, 2, 0, 3)
        value_bhsd = value.permute(1, 2, 0, 3)
        if self.plan.pass_index == 0:
            state = EagleTTTState.from_trunk(
                trunk_key=key_bhsd,
                trunk_value=value_bhsd,
                pass_count=self.plan.pass_count,
                max_passes=self.plan.max_passes,
                activation_budget_bytes=self._storage_plan.activation_budget_bytes,
            )
        else:
            if self.state is None:
                raise RuntimeError(
                    "EAGLE TTT branch pass requires retained trunk state"
                )
            state = self.state.append_branch(
                branch_key=key_bhsd,
                branch_value=value_bhsd,
            )
        self.state = state
        if self._resource_ledger is None:
            raise RuntimeError("EAGLE TTT resource ledger is not armed")
        self._resource_ledger.track_owned(key_bhsd, category="kv")
        self._resource_ledger.track_owned(value_bhsd, category="kv")
        output = eagle_ttt_attention(
            query=query_bhsd,
            state=state,
            plan=self.plan,
            scale=self.softmax_scale,
            layout=self.layout,
            block_mask=self._block_mask,
        )
        self._called = True
        return output.permute(2, 0, 1, 3).contiguous().flatten(2)


class MCoreEagleTTTSession:
    """Explicit per-invocation lifecycle around construction-time MCore adapters."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.layers = tuple(
            module
            for module in model.modules()
            if isinstance(module, EagleTTTCoreAttention)
        )
        if not self.layers:
            raise ValueError("model contains no EagleTTTCoreAttention layers")
        tp_groups = {
            id(layer.tp_group): layer.tp_group
            for layer in self.layers
            if layer.tp_group is not None
        }
        if len(tp_groups) > 1:
            raise ValueError("EAGLE TTT layers must share one tensor-parallel group")
        sequence_parallel_modes = {layer.sequence_parallel for layer in self.layers}
        if len(sequence_parallel_modes) > 1:
            raise ValueError(
                "EAGLE TTT layers must share one sequence-parallel setting"
            )
        self.tp_group = next(iter(tp_groups.values()), None)
        self.sequence_parallel = next(iter(sequence_parallel_modes))
        self.layout: EagleTTTSequenceLayout | None = None
        self.local_sequence_length: int | None = None
        self.storage_plan: EagleTTTStoragePlan | None = None
        self.resource_ledger: EagleTTTResourceLedger | None = None
        self.packed_seq_params: object | None = None
        self.rotary_pos_emb: object | None = None
        self.block_masks: list[Any] = []
        self._next_pass = 0

    @staticmethod
    def _tensors(value: object) -> tuple[Tensor, ...]:
        if isinstance(value, Tensor):
            return (value,)
        if isinstance(value, (tuple, list)):
            tensors: list[Tensor] = []
            for item in value:
                tensors.extend(MCoreEagleTTTSession._tensors(item))
            return tuple(tensors)
        return ()

    @staticmethod
    def _validate_packed_layout(
        *,
        layout: EagleTTTSequenceLayout,
        packed_seq_params: object | None,
    ) -> None:
        if packed_seq_params is None:
            return
        cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q", None)
        if not isinstance(cu_seqlens, Tensor):
            raise ValueError("packed_seq_params must expose tensor cu_seqlens_q")
        expected = EagleTTTSequenceLayout.from_cu_seqlens(
            cu_seqlens=cu_seqlens,
            sequence_length=layout.sequence_length,
            device=layout.valid_tokens.device,
        )
        if not torch.equal(
            expected.valid_tokens, layout.valid_tokens
        ) or not torch.equal(expected.document_ids, layout.document_ids):
            raise ValueError(
                "packed sequence parameters do not match the sequence layout"
            )

    def _validate_tp_layout(
        self, layout: EagleTTTSequenceLayout
    ) -> EagleTTTSequenceLayout:
        if (
            self.tp_group is None
            or not torch.distributed.is_initialized()
            or torch.distributed.get_world_size(self.tp_group) == 1
        ):
            return layout
        world_size = torch.distributed.get_world_size(self.tp_group)
        shape = torch.tensor(
            [
                layout.batch_size,
                layout.sequence_length,
                int(self.sequence_parallel),
            ],
            dtype=torch.int64,
            device=layout.document_ids.device,
        )
        gathered_shapes = [torch.empty_like(shape) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_shapes, shape, group=self.tp_group)
        if any(
            other[0].item() != layout.batch_size
            or other[1].item() != layout.sequence_length
            for other in gathered_shapes
        ):
            raise ValueError("TP ranks must agree on EAGLE TTT layout shape")
        if any(
            other[2].item() != int(self.sequence_parallel) for other in gathered_shapes
        ):
            raise ValueError(
                "TP ranks must agree on EAGLE TTT sequence-parallel setting"
            )

        if self.sequence_parallel:
            local_documents = layout.document_ids.to(dtype=torch.int64)
            gathered_documents = [
                torch.empty_like(local_documents) for _ in range(world_size)
            ]
            torch.distributed.all_gather(
                gathered_documents,
                local_documents,
                group=self.tp_group,
            )
            global_documents = torch.cat(gathered_documents, dim=1)
            return EagleTTTSequenceLayout(
                valid_tokens=global_documents.ne(-1),
                document_ids=global_documents,
            )

        reference = layout.document_ids.clone()
        source = torch.distributed.get_global_rank(self.tp_group, 0)
        torch.distributed.broadcast(reference, src=source, group=self.tp_group)
        mismatch = torch.tensor(
            [not torch.equal(reference, layout.document_ids)],
            dtype=torch.int32,
            device=layout.document_ids.device,
        )
        torch.distributed.all_reduce(
            mismatch,
            op=torch.distributed.ReduceOp.MAX,
            group=self.tp_group,
        )
        if mismatch.item() != 0:
            raise ValueError("TP ranks must agree on EAGLE TTT document indices")
        return layout

    def _validate_tp_packed_seq_params(
        self,
        *,
        layout: EagleTTTSequenceLayout,
        packed_seq_params: object | None,
    ) -> None:
        if (
            self.tp_group is None
            or not torch.distributed.is_initialized()
            or torch.distributed.get_world_size(self.tp_group) == 1
        ):
            return
        world_size = torch.distributed.get_world_size(self.tp_group)
        cu_seqlens_names = (
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "cu_seqlens_q_padded",
            "cu_seqlens_kv_padded",
        )
        descriptor_values = [
            int(packed_seq_params is not None),
            int(
                packed_seq_params is not None
                and getattr(packed_seq_params, "qkv_format", None) == "thd"
            ),
        ]
        compact_metadata: list[Tensor] = []
        for name in cu_seqlens_names:
            value = (
                getattr(packed_seq_params, name, None)
                if packed_seq_params is not None
                else None
            )
            if value is None:
                descriptor_values.append(-1)
            elif (
                isinstance(value, Tensor)
                and value.ndim == 1
                and value.dtype in (torch.int32, torch.int64)
            ):
                descriptor_values.append(value.numel())
                compact_metadata.append(
                    value.to(device=layout.document_ids.device, dtype=torch.int64)
                )
            else:
                descriptor_values.append(-2)
        descriptor = torch.tensor(
            descriptor_values,
            dtype=torch.int64,
            device=layout.document_ids.device,
        )
        gathered_descriptors = [torch.empty_like(descriptor) for _ in range(world_size)]
        torch.distributed.all_gather(
            gathered_descriptors,
            descriptor,
            group=self.tp_group,
        )
        if any(
            not torch.equal(other, gathered_descriptors[0])
            for other in gathered_descriptors[1:]
        ):
            raise ValueError(
                "TP ranks must agree on EAGLE TTT packed sequence parameters"
            )
        if packed_seq_params is None:
            return
        if descriptor_values[1] != 1 or any(
            length < 2 for length in descriptor_values[2:4]
        ):
            raise ValueError(
                "packed_seq_params must expose THD query and key/value cu_seqlens"
            )
        if any(length == -2 for length in descriptor_values[2:]):
            raise ValueError("packed cu_seqlens must be rank-1 integer tensors")

        packed_metadata = torch.cat(compact_metadata)
        gathered_metadata = [
            torch.empty_like(packed_metadata) for _ in range(world_size)
        ]
        torch.distributed.all_gather(
            gathered_metadata,
            packed_metadata,
            group=self.tp_group,
        )
        if any(
            not torch.equal(other, gathered_metadata[0])
            for other in gathered_metadata[1:]
        ):
            raise ValueError(
                "TP ranks must agree on EAGLE TTT packed sequence parameters"
            )

    def begin(
        self,
        *,
        layout: EagleTTTSequenceLayout,
        storage_plan: EagleTTTStoragePlan,
        excluded_tensors: tuple[Tensor, ...],
        resource_ledger: EagleTTTResourceLedger,
        packed_seq_params: object | None = None,
    ) -> None:
        if self.layout is not None:
            raise RuntimeError("EAGLE TTT session is already active")
        if any(layer.context_parallel_size != 1 for layer in self.layers):
            raise ValueError("EAGLE TTT context parallel size must be one")
        if (layout.batch_size, layout.sequence_length) != (
            storage_plan.batch_size,
            storage_plan.sequence_length,
        ):
            raise ValueError("sequence layout and storage plan dimensions must match")
        session_layout = self._validate_tp_layout(layout)
        self._validate_tp_packed_seq_params(
            layout=session_layout,
            packed_seq_params=packed_seq_params,
        )
        session_storage_plan = (
            replace(storage_plan, sequence_length=session_layout.sequence_length)
            if session_layout is not layout
            else storage_plan
        )
        self._validate_packed_layout(
            layout=session_layout,
            packed_seq_params=packed_seq_params,
        )
        resource_ledger.exclude(tuple(self.model.parameters()))
        resource_ledger.exclude(excluded_tensors)
        if session_layout is not layout:
            resource_ledger.track_owned(
                session_layout.valid_tokens,
                category="layout",
            )
            resource_ledger.track_owned(
                session_layout.document_ids,
                category="layout",
            )
        armed: list[EagleTTTCoreAttention] = []
        try:
            for layer in self.layers:
                layer.arm(
                    layout=session_layout,
                    storage_plan=session_storage_plan,
                    resource_ledger=resource_ledger,
                    packed_seq_params=packed_seq_params,
                )
                armed.append(layer)
            rotary_module = getattr(self.model, "rotary_pos_emb", None)
            rotary_pos_emb = (
                rotary_module(session_storage_plan.sequence_length)
                if callable(rotary_module)
                else None
            )
            for tensor in self._tensors(rotary_pos_emb):
                resource_ledger.track_owned(tensor, category="rope")
        except BaseException:
            for layer in armed:
                layer.reset()
            raise
        self.layout = session_layout
        self.local_sequence_length = layout.sequence_length
        self.storage_plan = session_storage_plan
        self.resource_ledger = resource_ledger
        self.packed_seq_params = packed_seq_params
        self.rotary_pos_emb = rotary_pos_emb
        self._next_pass = 0

    def __call__(
        self,
        *,
        embeddings: Tensor,
        hidden_states: Tensor,
        plan: EagleTTTAttentionPlan,
        rope_positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if (
            self.layout is None
            or self.local_sequence_length is None
            or self.storage_plan is None
        ):
            raise RuntimeError("EAGLE TTT session is not active")
        if plan.pass_index != self._next_pass:
            raise ValueError(
                f"EAGLE TTT session expected pass {self._next_pass}, "
                f"got {plan.pass_index}"
            )
        expected_positions = torch.arange(
            self.local_sequence_length,
            dtype=torch.int64,
            device=rope_positions.device,
        )
        if not torch.equal(rope_positions, expected_positions):
            raise ValueError("rope_positions must be the shared base position vector")
        if plan.sequence_length != self.local_sequence_length:
            raise ValueError("attention plan and local sequence lengths must match")
        attention_plan = (
            replace(plan, sequence_length=self.layout.sequence_length)
            if plan.sequence_length != self.layout.sequence_length
            else plan
        )
        block_mask = None
        if self.layout.valid_tokens.is_cuda:
            block_mask = _layout_block_mask(
                layout=self.layout,
                pass_index=plan.pass_index,
            )
            self.block_masks.append(block_mask)
        for layer in self.layers:
            layer.begin_pass(attention_plan, block_mask=block_mask)
        output = self.model(
            embeddings=embeddings,
            hidden_states=hidden_states,
            attention_mask=None,
            rotary_pos_emb=self.rotary_pos_emb,
            packed_seq_params=self.packed_seq_params,
        )
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError("EAGLE module must return hidden and next-hidden tensors")
        for layer in self.layers:
            layer.finish_pass()
        self._next_pass += 1
        hidden, next_hidden = output
        if not isinstance(hidden, Tensor) or not isinstance(next_hidden, Tensor):
            raise TypeError("EAGLE module outputs must be tensors")
        return hidden, next_hidden

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()
        self.layout = None
        self.local_sequence_length = None
        self.storage_plan = None
        self.resource_ledger = None
        self.packed_seq_params = None
        self.rotary_pos_emb = None
        self.block_masks.clear()
        self._next_pass = 0


def _validate_kv_pair(key: Tensor, value: Tensor, *, name: str) -> None:
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"{name} key and value must be rank-4 tensors")
    if key.shape != value.shape:
        raise ValueError(f"{name} key and value shapes must match")
    if key.device != value.device:
        raise ValueError(f"{name} key and value must share a device")
    if key.dtype != value.dtype:
        raise ValueError(f"{name} key and value must share a dtype")


def _expand_gqa(tensor: Tensor, *, query_heads: int) -> Tensor:
    if query_heads % tensor.shape[1] != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    return tensor.repeat_interleave(query_heads // tensor.shape[1], dim=1)


def _dense_causal_trunk(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float,
    pass_index: int,
    layout: EagleTTTSequenceLayout | None,
) -> tuple[Tensor, Tensor]:
    compute_dtype = (
        torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    )
    expanded_key = _expand_gqa(key, query_heads=query.shape[1])
    expanded_value = _expand_gqa(value, query_heads=query.shape[1])
    scores = torch.einsum(
        "bhqd,bhkd->bhqk",
        query.to(compute_dtype),
        expanded_key.to(compute_dtype),
    ).mul_(scale)
    sequence_length = query.shape[2]
    query_positions = torch.arange(sequence_length, device=query.device)[:, None]
    key_positions = torch.arange(sequence_length, device=query.device)[None, :]
    causal = key_positions <= query_positions - pass_index
    if layout is None:
        visible = causal[None].expand(query.shape[0], -1, -1)
    else:
        same_document = (
            layout.document_ids[:, :, None] == layout.document_ids[:, None, :]
        )
        valid_pair = layout.valid_tokens[:, :, None] & layout.valid_tokens[:, None, :]
        visible = causal[None] & same_document & valid_pair
    scores.masked_fill_(~visible[:, None], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    safe_rows = visible.any(dim=-1)
    safe_scores = torch.where(safe_rows[:, None, :, None], scores, 0.0)
    output = torch.einsum(
        "bhqk,bhkd->bhqd",
        torch.softmax(safe_scores, dim=-1),
        expanded_value.to(compute_dtype),
    )
    output.masked_fill_(~safe_rows[:, None, :, None], 0.0)
    return output, lse


def _causal_mask(
    _batch: Tensor,
    _head: Tensor,
    query_index: Tensor,
    key_index: Tensor,
    *,
    pass_index: int,
) -> Tensor:
    return key_index <= query_index - pass_index


@lru_cache(maxsize=64)
def _causal_block_mask(
    sequence_length: int,
    pass_index: int,
    device: torch.device,
) -> Any:
    from torch.nn.attention.flex_attention import create_block_mask

    return create_block_mask(
        lambda batch, head, query, key: _causal_mask(
            batch,
            head,
            query,
            key,
            pass_index=pass_index,
        ),
        B=None,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=device,
        BLOCK_SIZE=_FLEX_BLOCK_SIZE,
    )


def _layout_block_mask(
    *,
    layout: EagleTTTSequenceLayout,
    pass_index: int,
) -> Any:
    from torch.nn.attention.flex_attention import BlockMask

    def packed_causal_mask(
        batch: Tensor,
        _head: Tensor,
        query: Tensor,
        key: Tensor,
    ) -> Tensor:
        return (
            layout.valid_tokens[batch, query]
            & layout.valid_tokens[batch, key]
            & (layout.document_ids[batch, query] == layout.document_ids[batch, key])
            & (key <= query - pass_index)
        )

    sequence_length = layout.sequence_length
    block_count = (sequence_length + _FLEX_BLOCK_SIZE - 1) // _FLEX_BLOCK_SIZE
    padded_length = block_count * _FLEX_BLOCK_SIZE
    valid_blocks = torch.nn.functional.pad(
        layout.valid_tokens,
        (0, padded_length - sequence_length),
        value=False,
    ).reshape(layout.batch_size, block_count, _FLEX_BLOCK_SIZE)
    document_blocks = torch.nn.functional.pad(
        layout.document_ids,
        (0, padded_length - sequence_length),
        value=-1,
    ).reshape(layout.batch_size, block_count, _FLEX_BLOCK_SIZE)

    invalid_min = torch.iinfo(layout.document_ids.dtype).max
    document_min = torch.where(
        valid_blocks,
        document_blocks,
        invalid_min,
    ).amin(dim=-1)
    document_max = torch.where(
        valid_blocks,
        document_blocks,
        -1,
    ).amax(dim=-1)
    any_valid = valid_blocks.any(dim=-1)
    all_valid = valid_blocks.all(dim=-1)
    same_document_possible = (document_max[:, :, None] >= document_min[:, None, :]) & (
        document_max[:, None, :] >= document_min[:, :, None]
    )

    block_indices = torch.arange(
        block_count,
        device=layout.valid_tokens.device,
        dtype=torch.int64,
    )
    query_start = block_indices * _FLEX_BLOCK_SIZE
    query_end = (
        torch.minimum(
            query_start + _FLEX_BLOCK_SIZE,
            query_start.new_tensor(sequence_length),
        )
        - 1
    )
    key_start = query_start
    key_end = query_end
    causal_possible = key_start[None, :] <= query_end[:, None] - pass_index
    candidate_blocks = (
        any_valid[:, :, None]
        & any_valid[:, None, :]
        & same_document_possible
        & causal_possible[None]
    )

    one_query_document = document_min == document_max
    one_key_document = one_query_document
    full_blocks = (
        all_valid[:, :, None]
        & all_valid[:, None, :]
        & one_query_document[:, :, None]
        & one_key_document[:, None, :]
        & (document_min[:, :, None] == document_min[:, None, :])
        & (key_end[None, :] <= query_start[:, None] - pass_index)[None]
    )
    partial_blocks = candidate_blocks & ~full_blocks

    def ordered(blocks: Tensor) -> tuple[Tensor, Tensor]:
        return (
            blocks.sum(dim=-1, dtype=torch.int32).unsqueeze(1),
            torch.argsort(
                blocks.to(torch.int8),
                dim=-1,
                descending=True,
                stable=True,
            )
            .to(torch.int32)
            .unsqueeze(1),
        )

    partial_num_blocks, partial_indices = ordered(partial_blocks)
    full_num_blocks, full_indices = ordered(full_blocks)
    return BlockMask.from_kv_blocks(
        partial_num_blocks,
        partial_indices,
        full_num_blocks,
        full_indices,
        BLOCK_SIZE=_FLEX_BLOCK_SIZE,
        mask_mod=packed_causal_mask,
        seq_lengths=(sequence_length, sequence_length),
    )


def reset_eagle_ttt_attention_state() -> None:
    """Release device-resident per-sequence mask state at session teardown."""
    _causal_block_mask.cache_clear()


@lru_cache(maxsize=1)
def _compiled_flex_attention() -> Any:
    """Compile FlexAttention once so CUDA never uses its eager score fallback."""
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, dynamic=True, fullgraph=True)


def _flex_causal_trunk(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float,
    pass_index: int,
    layout: EagleTTTSequenceLayout | None,
    block_mask: Any = None,
) -> tuple[Tensor, Tensor]:
    from torch.nn.attention.flex_attention import (
        AuxRequest,  # pyrefly: ignore[missing-module-attribute]
    )

    flex_attention_call = _compiled_flex_attention()
    output, auxiliary = flex_attention_call(
        query,
        key,
        value,
        block_mask=(
            block_mask
            if block_mask is not None
            else (
                _causal_block_mask(query.shape[2], pass_index, query.device)
                if layout is None
                else _layout_block_mask(layout=layout, pass_index=pass_index)
            )
        ),
        scale=scale,
        enable_gqa=query.shape[1] != key.shape[1],
        return_aux=AuxRequest(lse=True),
        kernel_options={
            # Let Flex guard padded rows in dynamic partial query blocks.
            "ROWS_GUARANTEED_SAFE": False,
            "BLOCKS_ARE_CONTIGUOUS": True,
        },
    )
    lse = auxiliary.lse
    if not isinstance(lse, Tensor):
        raise RuntimeError("FlexAttention did not return the requested LSE")
    return output, lse


def _merge_branch(
    *,
    query: Tensor,
    output: Tensor,
    lse: Tensor,
    branch_key: Tensor,
    branch_value: Tensor,
    position_offset: int,
    scale: float,
    layout: EagleTTTSequenceLayout | None,
) -> tuple[Tensor, Tensor]:
    batch, query_heads, sequence_length, head_dim = query.shape
    kv_heads = branch_key.shape[1]
    groups = query_heads // kv_heads
    compute_dtype = lse.dtype
    grouped_query = query.to(compute_dtype).reshape(
        batch,
        kv_heads,
        groups,
        sequence_length,
        head_dim,
    )
    if position_offset == 0:
        aligned_key = branch_key
        aligned_value = branch_value
    elif position_offset >= sequence_length:
        aligned_key = branch_key * 0
        aligned_value = branch_value * 0
    else:
        aligned_key = torch.nn.functional.pad(
            branch_key[:, :, : sequence_length - position_offset],
            (0, 0, position_offset, 0),
        )
        aligned_value = torch.nn.functional.pad(
            branch_value[:, :, : sequence_length - position_offset],
            (0, 0, position_offset, 0),
        )
    branch_scores = torch.einsum(
        "bhgsd,bhsd->bhgs",
        grouped_query,
        aligned_key.to(compute_dtype),
    ).reshape(batch, query_heads, sequence_length)
    branch_scores.mul_(scale)
    positions = torch.arange(sequence_length, device=query.device)
    valid_position = positions >= position_offset
    if layout is None:
        valid = valid_position[None].expand(batch, -1)
    else:
        source_positions = (positions - position_offset).clamp_min(0)
        valid = (
            valid_position[None]
            & layout.valid_tokens
            & layout.valid_tokens[:, source_positions]
            & (layout.document_ids == layout.document_ids[:, source_positions])
        )
    branch_scores.masked_fill_(~valid[:, None, :], float("-inf"))
    merged_lse = torch.logaddexp(lse, branch_scores)
    grouped_output = output.to(compute_dtype).reshape(
        batch,
        kv_heads,
        groups,
        sequence_length,
        head_dim,
    )
    trunk_weight = torch.where(
        torch.isfinite(lse),
        torch.exp(lse - merged_lse),
        0.0,
    ).reshape(
        batch,
        kv_heads,
        groups,
        sequence_length,
        1,
    )
    branch_weight = torch.where(
        valid[:, None, :],
        torch.exp(branch_scores - merged_lse),
        0.0,
    ).reshape(
        batch,
        kv_heads,
        groups,
        sequence_length,
        1,
    )
    merged_output = grouped_output * trunk_weight
    merged_output.add_(aligned_value.to(compute_dtype).unsqueeze(2) * branch_weight)
    return (
        merged_output.reshape(batch, query_heads, sequence_length, head_dim),
        merged_lse,
    )


def eagle_ttt_attention(
    *,
    query: Tensor,
    state: EagleTTTState,
    plan: EagleTTTAttentionPlan,
    scale: float | None = None,
    layout: EagleTTTSequenceLayout | None = None,
    block_mask: Any = None,
) -> Tensor:
    """Attend to one causal trunk and pointwise prior same-anchor branches."""
    if state.pass_count != plan.pass_count or state.max_passes != plan.max_passes:
        raise ValueError("state and attention plan pass bounds must match")
    if len(state.branch_keys) != plan.pass_index:
        raise ValueError(
            f"pass {plan.pass_index} requires {plan.pass_index} prior branches, "
            f"got {len(state.branch_keys)}"
        )
    if query.ndim != 4:
        raise ValueError("query must be a rank-4 tensor")
    if query.shape[0] != state.trunk_key.shape[0]:
        raise ValueError("query and trunk batch sizes must match")
    if query.shape[2:] != state.trunk_key.shape[2:]:
        raise ValueError("query and trunk sequence/head dimensions must match")
    if query.shape[2] != plan.sequence_length:
        raise ValueError("query sequence length must match the attention plan")
    if query.device != state.trunk_key.device or query.dtype != state.trunk_key.dtype:
        raise ValueError("query and retained state must share device and dtype")
    if layout is not None:
        if layout.valid_tokens.shape != (query.shape[0], query.shape[2]):
            raise ValueError("sequence layout must match query batch and sequence axes")
        if layout.valid_tokens.device != query.device:
            raise ValueError("sequence layout and query must share a device")
    _expand_gqa(state.trunk_key, query_heads=query.shape[1])

    attention_scale = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
    if query.is_cuda and query.dtype != torch.float64:
        output, lse = _flex_causal_trunk(
            query,
            state.trunk_key,
            state.trunk_value,
            scale=attention_scale,
            pass_index=plan.pass_index,
            layout=layout,
            block_mask=block_mask,
        )
    else:
        output, lse = _dense_causal_trunk(
            query,
            state.trunk_key,
            state.trunk_value,
            scale=attention_scale,
            pass_index=plan.pass_index,
            layout=layout,
        )
    for branch_index, (branch_key, branch_value) in enumerate(
        zip(state.branch_keys, state.branch_values, strict=True)
    ):
        output, lse = _merge_branch(
            query=query,
            output=output,
            lse=lse,
            branch_key=branch_key,
            branch_value=branch_value,
            position_offset=plan.pass_index - branch_index - 1,
            scale=attention_scale,
            layout=layout,
        )
    return output.to(dtype=query.dtype)
