# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, Mapping, MutableMapping, Sequence

import torch

DataPipelineMode = Literal[
    "legacy",
    "stream_teacher",
    "stream_rollout",
    "sparse_loss",
]
DPPlannerMode = Literal[
    "preserve_update_groups",
    "balance_loss_tokens",
    "preserve_source_shards",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _is_integral_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }


def _require_1d_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    integral: bool = False,
) -> None:
    _require(tensor.ndim == 1, f"{name} must be 1D, got shape {tuple(tensor.shape)}")
    if integral:
        _require(_is_integral_tensor(tensor), f"{name} must have an integral dtype")


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _tensor_to_int_list(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in tensor.tolist()]


STREAM_METADATA_KEYS = (
    "sample_ids",
    "sample_order",
    "update_group",
    "global_batch_slot",
    "prompt_ids",
    "generation_ids",
)


def validate_sample_ids_unique(
    sample_ids: torch.Tensor, *, context: str = "sample_ids"
) -> None:
    _require_1d_tensor(sample_ids, name=f"{context}.sample_ids", integral=True)
    values = _tensor_to_int_list(sample_ids)
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    _require(
        not duplicates,
        f"{context}.sample_ids must be unique; duplicates={sorted(duplicates)}",
    )


def validate_manifest_metadata(
    sample_order: torch.Tensor,
    update_group: torch.Tensor,
    global_batch_slot: torch.Tensor,
    *,
    context: str,
) -> None:
    for name, tensor in (
        ("sample_order", sample_order),
        ("update_group", update_group),
        ("global_batch_slot", global_batch_slot),
    ):
        _require_1d_tensor(tensor, name=f"{context}.{name}", integral=True)

    batch_size = sample_order.numel()
    _require(
        update_group.numel() == batch_size,
        f"{context}.update_group must match batch size {batch_size}",
    )
    _require(
        global_batch_slot.numel() == batch_size,
        f"{context}.global_batch_slot must match batch size {batch_size}",
    )
    validate_sample_ids_unique(sample_order, context=f"{context}.sample_order")


def validate_loss_span_bounds(
    loss_spans: "SpanTable",
    input_lengths: torch.Tensor,
    *,
    context: str = "loss_spans",
) -> None:
    _require_1d_tensor(input_lengths, name=f"{context}.input_lengths", integral=True)
    _require(
        input_lengths.numel() == loss_spans.sample_count,
        (
            f"{context}.input_lengths length {input_lengths.numel()} "
            f"must match span sample count {loss_spans.sample_count}"
        ),
    )
    _require(
        torch.all(input_lengths >= 1).item(),
        f"{context}.input_lengths must contain only positive lengths",
    )

    offsets = _tensor_to_int_list(loss_spans.offsets)
    starts = _tensor_to_int_list(loss_spans.starts)
    ends = _tensor_to_int_list(loss_spans.ends)
    lengths = _tensor_to_int_list(input_lengths)

    for sample_index, input_length in enumerate(lengths):
        max_valid_end = max(input_length - 1, 0)
        start_offset = offsets[sample_index]
        end_offset = offsets[sample_index + 1]
        for span_index in range(start_offset, end_offset):
            start = starts[span_index]
            end = ends[span_index]
            _require(
                0 <= start < end <= max_valid_end,
                (
                    f"{context} span {span_index} for sample {sample_index} "
                    f"must satisfy 0 <= start < end <= {max_valid_end}; "
                    f"got start={start}, end={end}, input_length={input_length}"
                ),
            )


def estimate_active_loss_positions(loss_spans: "SpanTable") -> int:
    total = 0
    for start, end in zip(
        _tensor_to_int_list(loss_spans.starts),
        _tensor_to_int_list(loss_spans.ends),
        strict=True,
    ):
        total += end - start
    return total


def build_repeated_sample_metadata(
    *,
    step: int,
    prompt_count: int,
    num_generations_per_prompt: int,
    train_global_batch_size: int,
    step_sample_stride: int | None = None,
) -> dict[str, torch.Tensor]:
    _require(step >= 0, "step must be non-negative")
    _require(prompt_count > 0, "prompt_count must be positive")
    _require(
        num_generations_per_prompt > 0,
        "num_generations_per_prompt must be positive",
    )
    _require(train_global_batch_size > 0, "train_global_batch_size must be positive")

    repeated_size = prompt_count * num_generations_per_prompt
    if step_sample_stride is None:
        step_sample_stride = repeated_size
    _require(step_sample_stride >= repeated_size, "step_sample_stride is too small")

    repeated_index = torch.arange(repeated_size, dtype=torch.int64)
    prompt_ids = torch.arange(prompt_count, dtype=torch.int64).repeat_interleave(
        num_generations_per_prompt
    )
    generation_ids = torch.arange(
        num_generations_per_prompt, dtype=torch.int64
    ).repeat(prompt_count)
    sample_ids = step * step_sample_stride + (
        prompt_ids * num_generations_per_prompt + generation_ids
    )

    return {
        "sample_ids": sample_ids,
        "sample_order": repeated_index.clone(),
        "update_group": repeated_index // train_global_batch_size,
        "global_batch_slot": repeated_index % train_global_batch_size,
        "prompt_ids": prompt_ids,
        "generation_ids": generation_ids,
    }


def attach_step_metadata(
    batch: MutableMapping[str, Any],
    step_manifest: "StepManifest",
) -> None:
    for key in STREAM_METADATA_KEYS:
        batch[key] = getattr(step_manifest, key).clone()


def build_step_manifest(
    *,
    batch_id: str,
    step: int,
    prompt_count: int,
    num_generations_per_prompt: int,
    train_global_batch_size: int,
    max_sequence_length: int,
    tokenizer_name_or_path: str,
    tokenizer_config: dict[str, Any],
    processor_config: dict[str, Any] | None = None,
    sampling_config: dict[str, Any] | None = None,
    configured_prompts_per_step: int | None = None,
) -> "StepManifest":
    step_sample_stride = None
    if configured_prompts_per_step is not None:
        step_sample_stride = configured_prompts_per_step * num_generations_per_prompt

    metadata = build_repeated_sample_metadata(
        step=step,
        prompt_count=prompt_count,
        num_generations_per_prompt=num_generations_per_prompt,
        train_global_batch_size=train_global_batch_size,
        step_sample_stride=step_sample_stride,
    )
    return StepManifest(
        batch_id=batch_id,
        step=step,
        sample_ids=metadata["sample_ids"],
        sample_order=metadata["sample_order"],
        update_group=metadata["update_group"],
        global_batch_slot=metadata["global_batch_slot"],
        prompt_ids=metadata["prompt_ids"],
        generation_ids=metadata["generation_ids"],
        max_sequence_length=max_sequence_length,
        tokenizer_name_or_path=tokenizer_name_or_path,
        tokenizer_config=tokenizer_config,
        processor_config=processor_config,
        sampling_config=sampling_config,
    )


def loss_spans_from_token_mask(
    token_mask: torch.Tensor,
    input_lengths: torch.Tensor,
    sample_mask: torch.Tensor | None = None,
) -> "SpanTable":
    _require(token_mask.ndim == 2, "token_mask must be rank-2")
    _require_1d_tensor(input_lengths, name="input_lengths", integral=True)
    _require(
        token_mask.shape[0] == input_lengths.numel(),
        "token_mask batch dimension must match input_lengths",
    )
    if sample_mask is not None:
        _require_1d_tensor(sample_mask, name="sample_mask")
        _require(
            sample_mask.numel() == input_lengths.numel(),
            "sample_mask length must match input_lengths",
        )

    offsets = [0]
    starts: list[int] = []
    ends: list[int] = []

    for sample_idx, input_length in enumerate(_tensor_to_int_list(input_lengths)):
        _require(input_length >= 1, "input_lengths must be positive")
        _require(
            input_length <= token_mask.shape[1],
            "input_lengths cannot exceed token_mask width",
        )
        if sample_mask is not None and not bool(sample_mask[sample_idx].item()):
            offsets.append(len(starts))
            continue

        active_positions = (
            token_mask[sample_idx, 1:input_length].to(dtype=torch.bool).nonzero()
        )
        positions = [int(position.item()) for position in active_positions.flatten()]

        if positions:
            span_start = positions[0]
            prev = positions[0]
            for position in positions[1:]:
                if position == prev + 1:
                    prev = position
                    continue
                starts.append(span_start)
                ends.append(prev + 1)
                span_start = position
                prev = position
            starts.append(span_start)
            ends.append(prev + 1)
        offsets.append(len(starts))

    return SpanTable(
        offsets=torch.tensor(offsets, dtype=torch.int64),
        starts=torch.tensor(starts, dtype=torch.int32),
        ends=torch.tensor(ends, dtype=torch.int32),
    )


def build_batch_manifest_from_train_data(
    *,
    batch_id: str,
    step: int,
    train_data: Mapping[str, Any],
    metadata: Mapping[str, Any],
    max_sequence_length: int,
    tokenizer_name_or_path: str,
    tokenizer_config: dict[str, Any],
    processor_config: dict[str, Any] | None = None,
    multimodal_manifest_ref: object | None = None,
) -> "BatchManifest":
    loss_spans = loss_spans_from_token_mask(
        train_data["token_mask"],
        train_data["input_lengths"],
        train_data["sample_mask"],
    )
    return BatchManifest(
        batch_id=batch_id,
        step=step,
        sample_ids=metadata["sample_ids"],
        sample_order=metadata["sample_order"],
        update_group=metadata["update_group"],
        global_batch_slot=metadata["global_batch_slot"],
        prompt_ids=metadata["prompt_ids"],
        generation_ids=metadata["generation_ids"],
        input_lengths=train_data["input_lengths"],
        sample_mask=train_data["sample_mask"],
        loss_spans=loss_spans,
        max_sequence_length=max_sequence_length,
        tokenizer_name_or_path=tokenizer_name_or_path,
        tokenizer_config=tokenizer_config,
        processor_config=processor_config,
        multimodal_manifest_ref=multimodal_manifest_ref,
    )


@dataclass(frozen=True)
class DataPipelineConfig:
    mode: DataPipelineMode = "legacy"
    max_chunk_tokens: int = 262144
    max_chunk_bytes: int | None = None
    max_chunk_loss_positions: int | None = None
    log_memory_metrics: bool = True

    def __post_init__(self) -> None:
        _require(
            self.mode in {"legacy", "stream_teacher", "stream_rollout", "sparse_loss"},
            f"unsupported data pipeline mode: {self.mode}",
        )
        _require(
            self.max_chunk_tokens > 0,
            "max_chunk_tokens must be a positive integer",
        )
        for name, value in (
            ("max_chunk_bytes", self.max_chunk_bytes),
            ("max_chunk_loss_positions", self.max_chunk_loss_positions),
        ):
            if value is not None:
                _require(value > 0, f"{name} must be positive when provided")


def default_data_pipeline_config() -> DataPipelineConfig:
    return DataPipelineConfig()


def parse_data_pipeline_config(
    raw_config: Mapping[str, object] | DataPipelineConfig | None,
) -> DataPipelineConfig:
    if raw_config is None:
        return default_data_pipeline_config()
    if isinstance(raw_config, DataPipelineConfig):
        return raw_config

    defaults = default_data_pipeline_config()
    values: dict[str, object] = {
        "mode": defaults.mode,
        "max_chunk_tokens": defaults.max_chunk_tokens,
        "max_chunk_bytes": defaults.max_chunk_bytes,
        "max_chunk_loss_positions": defaults.max_chunk_loss_positions,
        "log_memory_metrics": defaults.log_memory_metrics,
    }
    unknown_keys = sorted(set(raw_config) - set(values))
    _require(
        not unknown_keys,
        f"unknown distillation.data_pipeline keys: {unknown_keys}",
    )
    values.update(raw_config)
    return DataPipelineConfig(**values)


def validate_streaming_capabilities(
    policy_config: Mapping[str, Any],
    teacher_config: Mapping[str, Any],
    data_pipeline_config: DataPipelineConfig,
    *,
    processor_present: bool = False,
) -> None:
    if data_pipeline_config.mode == "legacy":
        return

    for role, config in (("policy", policy_config), ("teacher", teacher_config)):
        _validate_single_policy_streaming_capability(
            role,
            config,
            mode=data_pipeline_config.mode,
            processor_present=processor_present,
        )


def _validate_single_policy_streaming_capability(
    role: str,
    config: Mapping[str, Any],
    *,
    mode: DataPipelineMode,
    processor_present: bool = False,
) -> None:
    tokenizer_config = config.get("tokenizer", {})
    use_processor = isinstance(tokenizer_config, Mapping) and tokenizer_config.get(
        "use_processor", False
    )
    if processor_present or use_processor:
        raise AssertionError(
            f"{role} multimodal OPD streaming is not supported in v1"
        )

    dynamic_batching = config.get("dynamic_batching", {})
    sequence_packing = config.get("sequence_packing", {})
    dynamic_enabled = (
        isinstance(dynamic_batching, Mapping)
        and dynamic_batching.get("enabled", False)
    )
    sequence_packing_enabled = (
        isinstance(sequence_packing, Mapping)
        and sequence_packing.get("enabled", False)
    )
    if dynamic_enabled and sequence_packing_enabled:
        raise AssertionError(
            f"{role} dynamic batching and sequence packing are mutually exclusive"
        )
    if role == "teacher" and mode in {"stream_teacher", "sparse_loss"}:
        if dynamic_enabled:
            raise AssertionError(
                f"{role} dynamic batching is unsupported for {mode} v1"
            )
        if sequence_packing_enabled:
            raise AssertionError(
                f"{role} sequence packing is unsupported for {mode} v1"
            )
    if mode in {"stream_rollout", "sparse_loss"}:
        if dynamic_enabled:
            raise AssertionError(
                f"{role} dynamic batching is unsupported for {mode} v1"
            )
        if sequence_packing_enabled:
            raise AssertionError(
                f"{role} sequence packing is unsupported for {mode} v1"
            )

    megatron_config = config.get("megatron_cfg", {})
    dtensor_config = config.get("dtensor_cfg", {})
    if isinstance(dtensor_config, Mapping) and dtensor_config.get("enabled", False):
        cp_size = int(dtensor_config.get("context_parallel_size", 1))
        if mode == "sparse_loss" and cp_size > 1:
            raise AssertionError(f"{role} context parallel is unsupported for sparse_loss v1")
        if cp_size > 1 and sequence_packing_enabled:
            raise AssertionError(
                f"{role} DTensor context parallel with sequence packing is unsupported"
            )

    if isinstance(megatron_config, Mapping) and megatron_config.get("enabled", False):
        pp_size = int(megatron_config.get("pipeline_model_parallel_size", 1))
        cp_size = int(megatron_config.get("context_parallel_size", 1))
        if dynamic_enabled and pp_size > 1:
            raise AssertionError(
                f"{role} dynamic batching with pipeline parallelism is unsupported"
            )
        if cp_size > 1 and not sequence_packing_enabled:
            raise AssertionError(
                f"{role} Megatron context parallel requires sequence packing"
            )


@dataclass(frozen=True)
class SpanTable:
    offsets: torch.Tensor
    starts: torch.Tensor
    ends: torch.Tensor

    def __post_init__(self) -> None:
        _require_1d_tensor(self.offsets, name="SpanTable.offsets", integral=True)
        _require_1d_tensor(self.starts, name="SpanTable.starts", integral=True)
        _require_1d_tensor(self.ends, name="SpanTable.ends", integral=True)
        _require(
            self.starts.numel() == self.ends.numel(),
            "SpanTable.starts and SpanTable.ends must have the same length",
        )
        _require(
            self.offsets.numel() >= 1,
            "SpanTable.offsets must contain at least the leading zero offset",
        )

        offsets = _tensor_to_int_list(self.offsets)
        starts = _tensor_to_int_list(self.starts)
        ends = _tensor_to_int_list(self.ends)
        _require(offsets[0] == 0, "SpanTable.offsets must start at zero")
        _require(
            offsets[-1] == self.starts.numel(),
            (
                f"SpanTable.offsets must terminate at num_spans={self.starts.numel()}, "
                f"got {offsets[-1]}"
            ),
        )
        _require(
            all(curr <= nxt for curr, nxt in zip(offsets, offsets[1:])),
            "SpanTable.offsets must be nondecreasing",
        )
        for span_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
            _require(
                0 <= start < end,
                (
                    f"SpanTable span {span_index} must satisfy 0 <= start < end; "
                    f"got start={start}, end={end}"
                ),
            )

    @property
    def sample_count(self) -> int:
        return self.offsets.numel() - 1

    @property
    def span_count(self) -> int:
        return self.starts.numel()


@dataclass(frozen=True)
class StepManifest:
    batch_id: str
    step: int
    sample_ids: torch.Tensor
    sample_order: torch.Tensor
    update_group: torch.Tensor
    global_batch_slot: torch.Tensor
    prompt_ids: torch.Tensor
    generation_ids: torch.Tensor
    max_sequence_length: int
    tokenizer_name_or_path: str
    tokenizer_config: dict[str, Any]
    processor_config: dict[str, Any] | None = None
    sampling_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _require(bool(self.batch_id), "StepManifest.batch_id must be non-empty")
        _require(self.step >= 0, "StepManifest.step must be non-negative")
        _require(
            self.max_sequence_length > 0,
            "StepManifest.max_sequence_length must be positive",
        )
        validate_sample_ids_unique(self.sample_ids, context="StepManifest")
        validate_manifest_metadata(
            self.sample_order,
            self.update_group,
            self.global_batch_slot,
            context="StepManifest",
        )
        for name, tensor in (
            ("prompt_ids", self.prompt_ids),
            ("generation_ids", self.generation_ids),
        ):
            _require_1d_tensor(tensor, name=f"StepManifest.{name}", integral=True)
            _require(
                tensor.numel() == self.sample_ids.numel(),
                (
                    f"StepManifest.{name} length {tensor.numel()} must match "
                    f"batch size {self.sample_ids.numel()}"
                ),
            )
        _require(
            self.sample_order.numel() == self.sample_ids.numel(),
            "StepManifest.sample_order must match batch size",
        )

    @property
    def batch_size(self) -> int:
        return self.sample_ids.numel()


@dataclass(frozen=True)
class BatchManifest:
    batch_id: str
    step: int
    sample_ids: torch.Tensor
    sample_order: torch.Tensor
    update_group: torch.Tensor
    global_batch_slot: torch.Tensor
    prompt_ids: torch.Tensor
    generation_ids: torch.Tensor
    input_lengths: torch.Tensor
    sample_mask: torch.Tensor
    loss_spans: SpanTable
    max_sequence_length: int
    tokenizer_name_or_path: str
    tokenizer_config: dict[str, Any]
    processor_config: dict[str, Any] | None = None
    multimodal_manifest_ref: object | None = None

    def __post_init__(self) -> None:
        _require(bool(self.batch_id), "BatchManifest.batch_id must be non-empty")
        _require(self.step >= 0, "BatchManifest.step must be non-negative")
        _require(
            self.max_sequence_length > 0,
            "BatchManifest.max_sequence_length must be positive",
        )
        validate_sample_ids_unique(self.sample_ids, context="BatchManifest")
        validate_manifest_metadata(
            self.sample_order,
            self.update_group,
            self.global_batch_slot,
            context="BatchManifest",
        )
        batch_size = self.sample_ids.numel()
        _require(
            self.sample_order.numel() == batch_size,
            "BatchManifest.sample_order must match batch size",
        )
        for name, tensor, integral in (
            ("prompt_ids", self.prompt_ids, True),
            ("generation_ids", self.generation_ids, True),
            ("input_lengths", self.input_lengths, True),
            ("sample_mask", self.sample_mask, False),
        ):
            _require_1d_tensor(
                tensor,
                name=f"BatchManifest.{name}",
                integral=integral,
            )
            _require(
                tensor.numel() == batch_size,
                f"BatchManifest.{name} length {tensor.numel()} must match batch size {batch_size}",
            )
        _require(
            self.loss_spans.sample_count == batch_size,
            (
                "BatchManifest.loss_spans sample count "
                f"{self.loss_spans.sample_count} must match batch size {batch_size}"
            ),
        )
        _require(
            torch.all(self.input_lengths <= self.max_sequence_length).item(),
            "BatchManifest.input_lengths cannot exceed max_sequence_length",
        )
        validate_loss_span_bounds(
            self.loss_spans,
            self.input_lengths,
            context="BatchManifest.loss_spans",
        )
        sample_mask_values = self.sample_mask.to(dtype=torch.bool).tolist()
        offsets = _tensor_to_int_list(self.loss_spans.offsets)
        for sample_index, is_active in enumerate(sample_mask_values):
            if not is_active:
                _require(
                    offsets[sample_index] == offsets[sample_index + 1],
                    (
                        "BatchManifest.loss_spans must not contain active spans "
                        f"for masked sample {sample_index}"
                    ),
                )

    @property
    def batch_size(self) -> int:
        return self.sample_ids.numel()


@dataclass(frozen=True)
class StageLayout:
    dp: int
    cp: int
    tp: int
    pp: int
    pad_multiple: int
    supports_sequence_packing: bool
    supports_dynamic_batching: bool
    supports_multimodal: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("dp", self.dp),
            ("cp", self.cp),
            ("tp", self.tp),
            ("pp", self.pp),
            ("pad_multiple", self.pad_multiple),
        ):
            _require(value > 0, f"StageLayout.{name} must be positive")


@dataclass(frozen=True)
class TokenChunk:
    batch_id: str
    chunk_id: int
    sample_ids: torch.Tensor
    sample_order: torch.Tensor
    update_group: torch.Tensor
    global_batch_slot: torch.Tensor
    input_ids: torch.Tensor
    input_lengths: torch.Tensor
    sample_mask: torch.Tensor
    loss_spans: SpanTable
    multimodal_ref: object | None = None

    def __post_init__(self) -> None:
        _require(bool(self.batch_id), "TokenChunk.batch_id must be non-empty")
        _require(self.chunk_id >= 0, "TokenChunk.chunk_id must be non-negative")
        _require(self.input_ids.ndim == 2, "TokenChunk.input_ids must be rank-2")
        _require(
            _is_integral_tensor(self.input_ids),
            "TokenChunk.input_ids must have an integral dtype",
        )
        validate_sample_ids_unique(self.sample_ids, context="TokenChunk")
        validate_manifest_metadata(
            self.sample_order,
            self.update_group,
            self.global_batch_slot,
            context="TokenChunk",
        )
        batch_size = self.sample_ids.numel()
        _require_1d_tensor(
            self.input_lengths,
            name="TokenChunk.input_lengths",
            integral=True,
        )
        _require_1d_tensor(self.sample_mask, name="TokenChunk.sample_mask")
        _require(
            self.input_lengths.numel() == batch_size == self.input_ids.shape[0],
            "TokenChunk batch metadata must match input_ids batch dimension",
        )
        _require(
            self.sample_mask.numel() == batch_size,
            "TokenChunk.sample_mask must match batch size",
        )
        _require(
            self.loss_spans.sample_count == batch_size,
            "TokenChunk.loss_spans sample count must match batch size",
        )
        _require(
            torch.all(self.input_lengths >= 1).item(),
            "TokenChunk.input_lengths must contain only positive lengths",
        )
        _require(
            torch.all(self.input_lengths <= self.input_ids.shape[1]).item(),
            (
                "TokenChunk.input_lengths cannot exceed chunk transport width "
                f"{self.input_ids.shape[1]}"
            ),
        )
        validate_loss_span_bounds(
            self.loss_spans,
            self.input_lengths,
            context="TokenChunk.loss_spans",
        )
        sample_mask_values = self.sample_mask.to(dtype=torch.bool).tolist()
        offsets = _tensor_to_int_list(self.loss_spans.offsets)
        for sample_index, is_active in enumerate(sample_mask_values):
            if not is_active:
                _require(
                    offsets[sample_index] == offsets[sample_index + 1],
                    (
                        "TokenChunk.loss_spans must not contain active spans "
                        f"for masked sample {sample_index}"
                    ),
                )


@dataclass(frozen=True)
class TeacherTopKChunk:
    batch_id: str
    chunk_id: int
    sample_ids: torch.Tensor
    sample_order: torch.Tensor
    update_group: torch.Tensor
    global_batch_slot: torch.Tensor
    position_offsets: torch.Tensor
    positions: torch.Tensor
    topk_logits: torch.Tensor
    topk_indices: torch.Tensor

    def __post_init__(self) -> None:
        _require(bool(self.batch_id), "TeacherTopKChunk.batch_id must be non-empty")
        _require(
            self.chunk_id >= 0,
            "TeacherTopKChunk.chunk_id must be non-negative",
        )
        validate_sample_ids_unique(self.sample_ids, context="TeacherTopKChunk")
        validate_manifest_metadata(
            self.sample_order,
            self.update_group,
            self.global_batch_slot,
            context="TeacherTopKChunk",
        )
        batch_size = self.sample_ids.numel()
        _require_1d_tensor(
            self.position_offsets,
            name="TeacherTopKChunk.position_offsets",
            integral=True,
        )
        _require_1d_tensor(
            self.positions,
            name="TeacherTopKChunk.positions",
            integral=True,
        )
        _require(
            self.position_offsets.numel() == batch_size + 1,
            "TeacherTopKChunk.position_offsets must have shape [B + 1]",
        )
        _require(
            self.topk_logits.ndim == 2,
            "TeacherTopKChunk.topk_logits must be rank-2",
        )
        _require(
            self.topk_indices.ndim == 2,
            "TeacherTopKChunk.topk_indices must be rank-2",
        )
        _require(
            self.topk_logits.shape == self.topk_indices.shape,
            "TeacherTopKChunk top-k tensors must have identical shapes",
        )
        _require(
            self.topk_logits.shape[0] == self.positions.numel(),
            "TeacherTopKChunk row count must match the number of positions",
        )
        offsets = _tensor_to_int_list(self.position_offsets)
        _require(offsets[0] == 0, "TeacherTopKChunk.position_offsets must start at zero")
        _require(
            offsets[-1] == self.positions.numel(),
            (
                "TeacherTopKChunk.position_offsets must terminate at num_positions="
                f"{self.positions.numel()}"
            ),
        )
        _require(
            all(curr <= nxt for curr, nxt in zip(offsets, offsets[1:])),
            "TeacherTopKChunk.position_offsets must be nondecreasing",
        )
        _require(
            torch.all(self.positions >= 0).item(),
            "TeacherTopKChunk.positions must be non-negative",
        )


@dataclass(frozen=True)
class AnnotatedTokenChunk:
    batch_id: str
    chunk_id: int
    token_chunk_ref: object | None
    teacher_topk_ref: object | None
    end_of_stream: bool = False
    sample_ids: torch.Tensor | None = None
    sample_order: torch.Tensor | None = None
    update_group: torch.Tensor | None = None
    global_batch_slot: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _require(bool(self.batch_id), "AnnotatedTokenChunk.batch_id must be non-empty")
        _require(
            self.chunk_id >= 0,
            "AnnotatedTokenChunk.chunk_id must be non-negative",
        )
        validate_eos_marker(self)


def validate_eos_marker(
    chunk: AnnotatedTokenChunk, *, context: str = "AnnotatedTokenChunk"
) -> None:
    if chunk.end_of_stream:
        _require(
            chunk.token_chunk_ref is None and chunk.teacher_topk_ref is None,
            (
                f"{context} EOS markers must not carry payload references; got "
                f"token_chunk_ref={chunk.token_chunk_ref!r}, "
                f"teacher_topk_ref={chunk.teacher_topk_ref!r}"
            ),
        )
        return

    _require(
        chunk.token_chunk_ref is not None,
        f"{context} non-EOS chunks must provide token_chunk_ref",
    )
    _require(
        chunk.teacher_topk_ref is not None,
        f"{context} non-EOS chunks must provide teacher_topk_ref",
    )
    for name, tensor in (
        ("sample_ids", chunk.sample_ids),
        ("sample_order", chunk.sample_order),
        ("update_group", chunk.update_group),
        ("global_batch_slot", chunk.global_batch_slot),
    ):
        _require(tensor is not None, f"{context} non-EOS chunks must provide {name}")
    validate_sample_ids_unique(chunk.sample_ids, context=context)
    validate_manifest_metadata(
        chunk.sample_order,
        chunk.update_group,
        chunk.global_batch_slot,
        context=context,
    )
    _require(
        chunk.sample_ids.numel() == chunk.sample_order.numel(),
        f"{context}.sample_ids must match metadata batch size",
    )


@dataclass(frozen=True)
class ShardedBatchStream:
    batch_id: str
    layout: Literal["dp_fullseq"]
    dp_size: int
    manifest: BatchManifest
    shard_streams: Sequence[object]

    def __post_init__(self) -> None:
        _require(bool(self.batch_id), "ShardedBatchStream.batch_id must be non-empty")
        _require(
            self.layout == "dp_fullseq",
            f"unsupported stream layout: {self.layout}",
        )
        _require(self.dp_size > 0, "ShardedBatchStream.dp_size must be positive")
        _require(
            self.manifest.batch_id == self.batch_id,
            "ShardedBatchStream.batch_id must match manifest.batch_id",
        )
        _require(
            len(self.shard_streams) == self.dp_size,
            (
                "ShardedBatchStream.shard_streams length "
                f"{len(self.shard_streams)} must match dp_size={self.dp_size}"
            ),
        )


def estimate_chunk_tokens(chunk: TokenChunk | TeacherTopKChunk) -> int:
    if isinstance(chunk, TokenChunk):
        return int(chunk.input_lengths.sum().item())
    return int(chunk.positions.numel())


def estimate_chunk_bytes(
    chunk: TokenChunk | TeacherTopKChunk | AnnotatedTokenChunk,
) -> int:
    if isinstance(chunk, TokenChunk):
        return sum(
            _tensor_bytes(tensor)
            for tensor in (
                chunk.sample_ids,
                chunk.sample_order,
                chunk.update_group,
                chunk.global_batch_slot,
                chunk.input_ids,
                chunk.input_lengths,
                chunk.sample_mask,
                chunk.loss_spans.offsets,
                chunk.loss_spans.starts,
                chunk.loss_spans.ends,
            )
        )
    if isinstance(chunk, TeacherTopKChunk):
        return sum(
            _tensor_bytes(tensor)
            for tensor in (
                chunk.sample_ids,
                chunk.sample_order,
                chunk.update_group,
                chunk.global_batch_slot,
                chunk.position_offsets,
                chunk.positions,
                chunk.topk_logits,
                chunk.topk_indices,
            )
        )
    return 0


def validate_chunk_budgets(
    chunk: TokenChunk | TeacherTopKChunk | AnnotatedTokenChunk,
    config: DataPipelineConfig,
    *,
    context: str = "chunk",
) -> None:
    bytes_used = estimate_chunk_bytes(chunk)
    if config.max_chunk_bytes is not None:
        _require(
            bytes_used <= config.max_chunk_bytes,
            (
                f"{context} exceeds max_chunk_bytes={config.max_chunk_bytes}; "
                f"got {bytes_used}"
            ),
        )

    if isinstance(chunk, AnnotatedTokenChunk):
        return

    token_count = estimate_chunk_tokens(chunk)
    _require(
        token_count <= config.max_chunk_tokens,
        (
            f"{context} exceeds max_chunk_tokens={config.max_chunk_tokens}; "
            f"got {token_count}"
        ),
    )
    if config.max_chunk_loss_positions is not None:
        loss_positions = (
            estimate_active_loss_positions(chunk.loss_spans)
            if isinstance(chunk, TokenChunk)
            else int(chunk.positions.numel())
        )
        _require(
            loss_positions <= config.max_chunk_loss_positions,
            (
                f"{context} exceeds max_chunk_loss_positions="
                f"{config.max_chunk_loss_positions}; got {loss_positions}"
            ),
        )


def slice_span_table(loss_spans: SpanTable, sample_indices: torch.Tensor) -> SpanTable:
    _require_1d_tensor(sample_indices, name="sample_indices", integral=True)
    offsets = _tensor_to_int_list(loss_spans.offsets)
    starts = _tensor_to_int_list(loss_spans.starts)
    ends = _tensor_to_int_list(loss_spans.ends)

    out_offsets = [0]
    out_starts: list[int] = []
    out_ends: list[int] = []
    for sample_index in _tensor_to_int_list(sample_indices):
        _require(
            0 <= sample_index < loss_spans.sample_count,
            f"sample_index {sample_index} is out of bounds",
        )
        for span_index in range(offsets[sample_index], offsets[sample_index + 1]):
            out_starts.append(starts[span_index])
            out_ends.append(ends[span_index])
        out_offsets.append(len(out_starts))

    return SpanTable(
        offsets=torch.tensor(out_offsets, dtype=torch.int64),
        starts=torch.tensor(out_starts, dtype=torch.int32),
        ends=torch.tensor(out_ends, dtype=torch.int32),
    )


@dataclass(frozen=True)
class ChunkAssignment:
    src_shard: int
    dst_shard: int
    update_group: int
    sample_indices: torch.Tensor
    sample_ids: torch.Tensor
    sample_order: torch.Tensor
    global_batch_slot: torch.Tensor
    estimated_tokens: int
    estimated_loss_tokens: int

    def __post_init__(self) -> None:
        for name, value in (
            ("src_shard", self.src_shard),
            ("dst_shard", self.dst_shard),
            ("update_group", self.update_group),
            ("estimated_tokens", self.estimated_tokens),
            ("estimated_loss_tokens", self.estimated_loss_tokens),
        ):
            _require(value >= 0, f"ChunkAssignment.{name} must be non-negative")
        for name, tensor in (
            ("sample_indices", self.sample_indices),
            ("sample_ids", self.sample_ids),
            ("sample_order", self.sample_order),
            ("global_batch_slot", self.global_batch_slot),
        ):
            _require_1d_tensor(tensor, name=f"ChunkAssignment.{name}", integral=True)
        batch_size = self.sample_ids.numel()
        _require(
            self.sample_indices.numel()
            == self.sample_order.numel()
            == self.global_batch_slot.numel()
            == batch_size,
            "ChunkAssignment tensors must have the same length",
        )
        validate_sample_ids_unique(self.sample_ids, context="ChunkAssignment")
        sample_order = _tensor_to_int_list(self.sample_order)
        _require(
            sample_order == sorted(sample_order),
            "ChunkAssignment.sample_order must be sorted in canonical order",
        )


class DPReshardPlanner:
    def __init__(
        self,
        mode: DPPlannerMode = "balance_loss_tokens",
    ) -> None:
        _require(
            mode
            in {
                "preserve_update_groups",
                "balance_loss_tokens",
                "preserve_source_shards",
            },
            f"unsupported DPReshardPlanner mode: {mode}",
        )
        self.mode = mode

    def plan(
        self,
        manifest: BatchManifest,
        src_dp_size: int,
        dst_dp_size: int,
        max_chunk_tokens: int,
    ) -> list[list[ChunkAssignment]]:
        _require(src_dp_size > 0, "src_dp_size must be positive")
        _require(dst_dp_size > 0, "dst_dp_size must be positive")
        _require(max_chunk_tokens > 0, "max_chunk_tokens must be positive")

        per_group: dict[int, list[int]] = defaultdict(list)
        update_groups = _tensor_to_int_list(manifest.update_group)
        sample_orders = _tensor_to_int_list(manifest.sample_order)
        global_slots = _tensor_to_int_list(manifest.global_batch_slot)
        input_lengths = _tensor_to_int_list(manifest.input_lengths)
        loss_positions = self._loss_positions_per_sample(manifest.loss_spans)

        for index, update_group in enumerate(update_groups):
            per_group[update_group].append(index)

        ordered_groups = sorted(
            per_group,
            key=lambda group: min(sample_orders[index] for index in per_group[group]),
        )
        source_shards = self._source_shards_per_sample(
            update_groups,
            global_slots,
            src_dp_size,
        )
        assignments_by_dst: list[list[ChunkAssignment]] = [[] for _ in range(dst_dp_size)]
        dst_loads = [0 for _ in range(dst_dp_size)]

        for update_group in ordered_groups:
            indices = sorted(
                per_group[update_group],
                key=lambda index: (global_slots[index], sample_orders[index]),
            )
            indices_by_src_shard: dict[int, list[int]] = defaultdict(list)
            for index in indices:
                indices_by_src_shard[source_shards[index]].append(index)

            group_dst_shard = None
            if self.mode != "preserve_source_shards":
                group_dst_shard = self._select_dst_shard(
                    update_group,
                    indices,
                    dst_loads,
                    loss_positions,
                    dst_dp_size,
                )

            for src_shard in sorted(indices_by_src_shard):
                dst_shard = (
                    src_shard % dst_dp_size
                    if self.mode == "preserve_source_shards"
                    else group_dst_shard
                )
                _require(dst_shard is not None, "dst_shard must be selected")
                for chunk_indices in self._partition_group(
                    indices_by_src_shard[src_shard],
                    input_lengths,
                    max_chunk_tokens,
                ):
                    assignment = self._build_assignment(
                        manifest=manifest,
                        sample_indices=chunk_indices,
                        src_shard=src_shard,
                        dst_shard=dst_shard,
                        update_group=update_group,
                        input_lengths=input_lengths,
                        loss_positions=loss_positions,
                    )
                    assignments_by_dst[dst_shard].append(assignment)
                dst_loads[dst_shard] += sum(
                    loss_positions[index] for index in indices_by_src_shard[src_shard]
                )

        return assignments_by_dst

    def _source_shards_per_sample(
        self,
        update_groups: list[int],
        global_slots: list[int],
        src_dp_size: int,
    ) -> list[int]:
        group_widths: dict[int, int] = {}
        for update_group, global_slot in zip(update_groups, global_slots, strict=True):
            group_widths[update_group] = max(
                group_widths.get(update_group, 0),
                global_slot + 1,
            )

        group_shard_widths = {}
        for update_group, group_width in group_widths.items():
            _require(
                group_width % src_dp_size == 0,
                (
                    "update_group width must be divisible by src_dp_size to match "
                    "legacy shard_by_batch_size semantics; "
                    f"update_group={update_group}, width={group_width}, "
                    f"src_dp_size={src_dp_size}"
                ),
            )
            group_shard_widths[update_group] = group_width // src_dp_size

        source_shards = []
        for update_group, global_slot in zip(update_groups, global_slots, strict=True):
            shard_width = group_shard_widths[update_group]
            source_shards.append(global_slot // shard_width)
        return source_shards

    def _select_dst_shard(
        self,
        update_group: int,
        indices: list[int],
        dst_loads: list[int],
        loss_positions: list[int],
        dst_dp_size: int,
    ) -> int:
        if self.mode == "preserve_update_groups":
            return update_group % dst_dp_size

        estimated_group_loss_tokens = sum(loss_positions[index] for index in indices)
        min_load = min(dst_loads)
        candidates = [index for index, load in enumerate(dst_loads) if load == min_load]
        if len(candidates) == 1:
            return candidates[0]
        return min(
            candidates,
            key=lambda shard: (dst_loads[shard] + estimated_group_loss_tokens, shard),
        )

    def _partition_group(
        self,
        indices: list[int],
        input_lengths: list[int],
        max_chunk_tokens: int,
    ) -> list[list[int]]:
        partitions: list[list[int]] = []
        current: list[int] = []
        current_tokens = 0

        for index in indices:
            sample_tokens = input_lengths[index]
            _require(
                sample_tokens <= max_chunk_tokens,
                (
                    "A single sample exceeds max_chunk_tokens; "
                    f"sample_index={index}, input_length={sample_tokens}, "
                    f"max_chunk_tokens={max_chunk_tokens}"
                ),
            )
            if current and current_tokens + sample_tokens > max_chunk_tokens:
                partitions.append(current)
                current = []
                current_tokens = 0
            current.append(index)
            current_tokens += sample_tokens

        if current:
            partitions.append(current)
        return partitions

    def _build_assignment(
        self,
        *,
        manifest: BatchManifest,
        sample_indices: list[int],
        src_shard: int,
        dst_shard: int,
        update_group: int,
        input_lengths: list[int],
        loss_positions: list[int],
    ) -> ChunkAssignment:
        index_tensor = torch.tensor(sample_indices, dtype=torch.int64)
        sample_ids = manifest.sample_ids.index_select(0, index_tensor)
        sample_order = manifest.sample_order.index_select(0, index_tensor)
        global_batch_slot = manifest.global_batch_slot.index_select(0, index_tensor)
        return ChunkAssignment(
            src_shard=src_shard,
            dst_shard=dst_shard,
            update_group=update_group,
            sample_indices=index_tensor,
            sample_ids=sample_ids,
            sample_order=sample_order,
            global_batch_slot=global_batch_slot,
            estimated_tokens=sum(input_lengths[index] for index in sample_indices),
            estimated_loss_tokens=sum(loss_positions[index] for index in sample_indices),
        )

    def _loss_positions_per_sample(self, spans: SpanTable) -> list[int]:
        counts = [0 for _ in range(spans.sample_count)]
        offsets = _tensor_to_int_list(spans.offsets)
        starts = _tensor_to_int_list(spans.starts)
        ends = _tensor_to_int_list(spans.ends)
        for sample_index in range(spans.sample_count):
            for span_index in range(offsets[sample_index], offsets[sample_index + 1]):
                counts[sample_index] += ends[span_index] - starts[span_index]
        return counts


def build_token_chunk_from_assignment(
    *,
    manifest: BatchManifest,
    train_data: Mapping[str, Any],
    assignment: ChunkAssignment,
    chunk_id: int,
    truncate_to_input_lengths: bool = False,
) -> TokenChunk:
    sample_indices = assignment.sample_indices
    input_ids = train_data["input_ids"].index_select(0, sample_indices)
    input_lengths = manifest.input_lengths.index_select(0, sample_indices)
    if truncate_to_input_lengths:
        transport_width = int(input_lengths.max().item())
        input_ids = input_ids[:, :transport_width]

    chunk = TokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=chunk_id,
        sample_ids=manifest.sample_ids.index_select(0, sample_indices),
        sample_order=manifest.sample_order.index_select(0, sample_indices),
        update_group=manifest.update_group.index_select(0, sample_indices),
        global_batch_slot=manifest.global_batch_slot.index_select(0, sample_indices),
        input_ids=input_ids,
        input_lengths=input_lengths,
        sample_mask=manifest.sample_mask.index_select(0, sample_indices),
        loss_spans=slice_span_table(manifest.loss_spans, sample_indices),
        multimodal_ref=manifest.multimodal_manifest_ref,
    )
    return chunk


def build_token_stream_from_train_data(
    *,
    manifest: BatchManifest,
    train_data: Mapping[str, Any],
    data_pipeline_config: DataPipelineConfig,
    src_dp_size: int,
    dst_dp_size: int,
    planner_mode: DPPlannerMode = "preserve_update_groups",
    truncate_to_input_lengths: bool = False,
) -> ShardedBatchStream:
    planner = DPReshardPlanner(mode=planner_mode)
    plan = planner.plan(
        manifest=manifest,
        src_dp_size=src_dp_size,
        dst_dp_size=dst_dp_size,
        max_chunk_tokens=data_pipeline_config.max_chunk_tokens,
    )
    shard_streams: list[list[TokenChunk]] = [[] for _ in range(dst_dp_size)]
    chunk_id = 0
    for dst_shard, assignments in enumerate(plan):
        for assignment in assignments:
            chunk = build_token_chunk_from_assignment(
                manifest=manifest,
                train_data=train_data,
                assignment=assignment,
                chunk_id=chunk_id,
                truncate_to_input_lengths=truncate_to_input_lengths,
            )
            validate_chunk_budgets(
                chunk,
                data_pipeline_config,
                context=f"token_chunk[{chunk_id}]",
            )
            shard_streams[dst_shard].append(chunk)
            chunk_id += 1

    return ShardedBatchStream(
        batch_id=manifest.batch_id,
        layout="dp_fullseq",
        dp_size=dst_dp_size,
        manifest=manifest,
        shard_streams=shard_streams,
    )


def build_token_stream_from_rollout_batch(
    *,
    batch_id: str,
    step: int,
    rollout_batch: Mapping[str, Any],
    tokenizer_pad_token_id: int,
    make_sequence_length_divisible_by: int,
    max_sequence_length: int,
    tokenizer_name_or_path: str,
    tokenizer_config: dict[str, Any],
    data_pipeline_config: DataPipelineConfig,
    src_dp_size: int,
    dst_dp_size: int,
    planner_mode: DPPlannerMode = "preserve_update_groups",
) -> tuple[BatchManifest, ShardedBatchStream]:
    """Normalize a post-rollout message batch into OPD token chunks.

    This v1 boundary still receives rollout message logs on the driver. It avoids
    passing the resulting dense training batch into student training.
    """
    from nemo_rl.data.llm_message_utils import (
        add_loss_mask_to_message_log,
        batched_message_log_to_flat_message,
    )
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    add_loss_mask_to_message_log(rollout_batch["message_log"])
    flat_messages, input_lengths = batched_message_log_to_flat_message(
        rollout_batch["message_log"],
        pad_value_dict={"token_ids": tokenizer_pad_token_id},
        make_sequence_length_divisible_by=make_sequence_length_divisible_by,
    )
    multimodal_data = flat_messages.get_multimodal_dict(as_tensors=False)
    _require(
        not multimodal_data,
        "stream_rollout v1 does not support multimodal rollout batches",
    )

    train_data = BatchedDataDict(
        {
            "input_ids": flat_messages["token_ids"],
            "input_lengths": input_lengths,
            "token_mask": flat_messages["token_loss_mask"],
            "sample_mask": rollout_batch["loss_multiplier"],
        }
    )
    for key in STREAM_METADATA_KEYS:
        train_data[key] = rollout_batch[key]
    train_data.to("cpu")

    manifest = build_batch_manifest_from_train_data(
        batch_id=batch_id,
        step=step,
        train_data=train_data,
        metadata=rollout_batch,
        max_sequence_length=max_sequence_length,
        tokenizer_name_or_path=tokenizer_name_or_path,
        tokenizer_config=tokenizer_config,
    )
    token_stream = build_token_stream_from_train_data(
        manifest=manifest,
        train_data=train_data,
        data_pipeline_config=data_pipeline_config,
        src_dp_size=src_dp_size,
        dst_dp_size=dst_dp_size,
        planner_mode=planner_mode,
    )
    return manifest, token_stream


def token_chunk_to_batched_data(chunk: TokenChunk) -> Any:
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    return BatchedDataDict(
        {
            "input_ids": chunk.input_ids,
            "input_lengths": chunk.input_lengths,
        }
    )


def _resolve_chunk_micro_batch_size(
    *,
    chunk_batch_size: int,
    configured_micro_batch_size: int | None,
) -> int:
    _require(chunk_batch_size > 0, "chunk_batch_size must be positive")
    if configured_micro_batch_size is None:
        return chunk_batch_size
    _require(
        configured_micro_batch_size > 0,
        "configured_micro_batch_size must be positive",
    )
    candidate = min(configured_micro_batch_size, chunk_batch_size)
    while chunk_batch_size % candidate != 0:
        candidate -= 1
    return candidate


def _positions_from_spans(loss_spans: SpanTable) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = _tensor_to_int_list(loss_spans.offsets)
    starts = _tensor_to_int_list(loss_spans.starts)
    ends = _tensor_to_int_list(loss_spans.ends)
    out_offsets = [0]
    positions: list[int] = []
    for sample_index in range(loss_spans.sample_count):
        for span_index in range(offsets[sample_index], offsets[sample_index + 1]):
            positions.extend(range(starts[span_index], ends[span_index]))
        out_offsets.append(len(positions))
    return (
        torch.tensor(out_offsets, dtype=torch.int64),
        torch.tensor(positions, dtype=torch.int32),
    )


def _dense_positions_for_chunk(chunk: TokenChunk) -> tuple[torch.Tensor, torch.Tensor]:
    width = int(chunk.input_ids.shape[1])
    offsets = [sample_index * width for sample_index in range(chunk.sample_ids.numel() + 1)]
    positions = list(range(width)) * int(chunk.sample_ids.numel())
    return (
        torch.tensor(offsets, dtype=torch.int64),
        torch.tensor(positions, dtype=torch.int32),
    )


def build_teacher_topk_chunk_from_dense(
    *,
    token_chunk: TokenChunk,
    topk_logits: torch.Tensor,
    topk_indices: torch.Tensor,
    position_mode: Literal["dense", "loss"] = "loss",
) -> TeacherTopKChunk:
    _require(topk_logits.ndim == 3, "topk_logits must be rank-3")
    _require(topk_indices.ndim == 3, "topk_indices must be rank-3")
    _require(
        topk_logits.shape == topk_indices.shape,
        "top-k logits and indices must have identical shape",
    )
    _require(
        topk_logits.shape[0] == token_chunk.sample_ids.numel(),
        "top-k batch dimension must match token_chunk batch size",
    )
    _require(
        topk_logits.shape[1] >= token_chunk.input_ids.shape[1],
        "top-k sequence dimension must cover token_chunk transport width",
    )

    if position_mode == "dense":
        position_offsets, positions = _dense_positions_for_chunk(token_chunk)
    elif position_mode == "loss":
        position_offsets, positions = _positions_from_spans(token_chunk.loss_spans)
    else:
        raise AssertionError(f"unsupported teacher chunk position_mode={position_mode}")

    topk_logits_rows = []
    topk_indices_rows = []
    offsets = _tensor_to_int_list(position_offsets)
    for sample_index in range(token_chunk.sample_ids.numel()):
        sample_positions = positions[offsets[sample_index] : offsets[sample_index + 1]]
        if sample_positions.numel() == 0:
            continue
        sample_positions = sample_positions.to(
            dtype=torch.long,
            device=topk_logits.device,
        )
        topk_logits_rows.append(topk_logits[sample_index, sample_positions, :])
        topk_indices_rows.append(topk_indices[sample_index, sample_positions, :])

    topk_width = int(topk_logits.shape[-1])
    if topk_logits_rows:
        flat_topk_logits = torch.cat(topk_logits_rows, dim=0).cpu()
        flat_topk_indices = torch.cat(topk_indices_rows, dim=0).cpu()
    else:
        flat_topk_logits = torch.empty(
            (0, topk_width),
            dtype=topk_logits.dtype,
            device="cpu",
        )
        flat_topk_indices = torch.empty(
            (0, topk_width),
            dtype=topk_indices.dtype,
            device="cpu",
        )

    return TeacherTopKChunk(
        batch_id=token_chunk.batch_id,
        chunk_id=token_chunk.chunk_id,
        sample_ids=token_chunk.sample_ids.cpu(),
        sample_order=token_chunk.sample_order.cpu(),
        update_group=token_chunk.update_group.cpu(),
        global_batch_slot=token_chunk.global_batch_slot.cpu(),
        position_offsets=position_offsets,
        positions=positions,
        topk_logits=flat_topk_logits,
        topk_indices=flat_topk_indices,
    )


def iter_annotated_teacher_topk_chunks(
    *,
    token_chunks: Sequence[TokenChunk],
    get_topk_logits: Callable[..., Mapping[str, torch.Tensor] | None],
    k: int,
    micro_batch_size: int | None = None,
    batch_id: str | None = None,
    put_fn: Callable[[object], object] | None = None,
) -> Iterator[AnnotatedTokenChunk]:
    put = put_fn or (lambda value: value)
    for token_chunk in token_chunks:
        topk_batch = get_topk_logits(
            data=token_chunk_to_batched_data(token_chunk),
            k=k,
            micro_batch_size=_resolve_chunk_micro_batch_size(
                chunk_batch_size=int(token_chunk.sample_ids.numel()),
                configured_micro_batch_size=micro_batch_size,
            ),
        )
        if topk_batch is None:
            continue
        teacher_chunk = build_teacher_topk_chunk_from_dense(
            token_chunk=token_chunk,
            topk_logits=topk_batch["topk_logits"],
            topk_indices=topk_batch["topk_indices"],
            position_mode="loss",
        )
        yield AnnotatedTokenChunk(
            batch_id=token_chunk.batch_id,
            chunk_id=token_chunk.chunk_id,
            token_chunk_ref=put(token_chunk),
            teacher_topk_ref=put(teacher_chunk),
            sample_ids=token_chunk.sample_ids.cpu(),
            sample_order=token_chunk.sample_order.cpu(),
            update_group=token_chunk.update_group.cpu(),
            global_batch_slot=token_chunk.global_batch_slot.cpu(),
        )

    if batch_id is None:
        batch_id = token_chunks[0].batch_id if token_chunks else "empty-stream"
    next_chunk_id = (max(chunk.chunk_id for chunk in token_chunks) + 1) if token_chunks else 0
    yield AnnotatedTokenChunk(
        batch_id=batch_id,
        chunk_id=next_chunk_id,
        token_chunk_ref=None,
        teacher_topk_ref=None,
        end_of_stream=True,
    )


def collect_teacher_topk_chunks_to_dense(
    *,
    manifest: BatchManifest,
    teacher_chunks: Sequence[TeacherTopKChunk],
    sequence_length: int,
    topk_width: int | None = None,
) -> dict[str, torch.Tensor]:
    _require(sequence_length > 0, "sequence_length must be positive")
    if topk_width is None:
        non_empty_chunks = [
            chunk for chunk in teacher_chunks if chunk.topk_logits.shape[0] > 0
        ]
        _require(non_empty_chunks, "topk_width is required for an empty teacher stream")
        topk_width = int(non_empty_chunks[0].topk_logits.shape[-1])

    sample_id_to_row = {
        sample_id: row
        for row, sample_id in enumerate(_tensor_to_int_list(manifest.sample_ids))
    }
    dtype = (
        teacher_chunks[0].topk_logits.dtype
        if teacher_chunks
        else torch.float32
    )
    index_dtype = (
        teacher_chunks[0].topk_indices.dtype
        if teacher_chunks
        else torch.int64
    )
    dense_logits = torch.zeros(
        (manifest.batch_size, sequence_length, topk_width),
        dtype=dtype,
    )
    dense_indices = torch.zeros(
        (manifest.batch_size, sequence_length, topk_width),
        dtype=index_dtype,
    )

    for chunk in teacher_chunks:
        offsets = _tensor_to_int_list(chunk.position_offsets)
        positions = _tensor_to_int_list(chunk.positions)
        sample_ids = _tensor_to_int_list(chunk.sample_ids)
        for sample_index, sample_id in enumerate(sample_ids):
            row = sample_id_to_row[sample_id]
            for position_offset in range(offsets[sample_index], offsets[sample_index + 1]):
                position = positions[position_offset]
                _require(
                    position < sequence_length,
                    f"teacher position {position} exceeds sequence_length={sequence_length}",
                )
                dense_logits[row, position, :] = chunk.topk_logits[position_offset]
                dense_indices[row, position, :] = chunk.topk_indices[position_offset]

    return {
        "teacher_topk_logits": dense_logits,
        "teacher_topk_indices": dense_indices,
    }


def drain_annotated_stream(
    annotated_stream: ShardedBatchStream,
    *,
    get_fn: Callable[[object], object] | None = None,
) -> ShardedBatchStream:
    if get_fn is None:
        import ray

        get_fn = ray.get

    def handle_chunk(
        chunk: object,
        *,
        shard_chunks: list[AnnotatedTokenChunk],
        seen_eos: bool,
    ) -> bool:
        _require(
            isinstance(chunk, AnnotatedTokenChunk),
            f"annotated stream yielded {type(chunk).__name__}, expected AnnotatedTokenChunk",
        )
        _require(
            chunk.batch_id == annotated_stream.batch_id,
            "AnnotatedTokenChunk batch_id must match stream batch_id",
        )
        if chunk.end_of_stream:
            _require(
                not seen_eos,
                "annotated stream shard produced more than one EOS marker",
            )
            return True
        _require(
            not seen_eos,
            "annotated stream shard produced payload after EOS marker",
        )
        shard_chunks.append(chunk)
        return seen_eos

    def drain_shard_stream(shard_stream: object) -> list[AnnotatedTokenChunk]:
        shard_chunks: list[AnnotatedTokenChunk] = []
        seen_eos = False
        for chunk_ref in shard_stream:
            chunk = (
                chunk_ref
                if isinstance(chunk_ref, AnnotatedTokenChunk)
                else get_fn(chunk_ref)
            )
            seen_eos = handle_chunk(
                chunk,
                shard_chunks=shard_chunks,
                seen_eos=seen_eos,
            )
        _require(
            seen_eos,
            "annotated stream shard ended without an EOS marker",
        )
        return shard_chunks

    def drain_ray_generator_streams(
        shard_streams: list[object],
    ) -> list[list[AnnotatedTokenChunk]]:
        import ray

        drained = [[] for _ in shard_streams]
        seen_eos = [False for _ in shard_streams]
        pending_refs = {
            shard_idx: shard_stream._get_next_ref()
            for shard_idx, shard_stream in enumerate(shard_streams)
        }
        while pending_refs:
            ready_refs, _ = ray.wait(
                list(pending_refs.values()),
                num_returns=1,
                timeout=0.1,
                fetch_local=False,
            )
            if not ready_refs:
                continue
            ready_ref_set = set(ready_refs)
            for shard_idx, expected_ref in list(pending_refs.items()):
                if expected_ref not in ready_ref_set:
                    continue
                shard_stream = shard_streams[shard_idx]
                try:
                    chunk_ref = shard_stream._next_sync(timeout_s=0)
                except StopIteration:
                    _require(
                        seen_eos[shard_idx],
                        "annotated stream shard ended without an EOS marker",
                    )
                    del pending_refs[shard_idx]
                    continue
                if chunk_ref.is_nil():
                    continue
                chunk = get_fn(chunk_ref)
                seen_eos[shard_idx] = handle_chunk(
                    chunk,
                    shard_chunks=drained[shard_idx],
                    seen_eos=seen_eos[shard_idx],
                )
                if seen_eos[shard_idx]:
                    del pending_refs[shard_idx]
                else:
                    pending_refs[shard_idx] = shard_stream._get_next_ref()
        return drained

    shard_streams = list(annotated_stream.shard_streams)
    if all(
        hasattr(stream, "_next_sync") and hasattr(stream, "_get_next_ref")
        for stream in shard_streams
    ):
        drained_shards = drain_ray_generator_streams(shard_streams)
    elif len(shard_streams) == 1:
        drained_shards = [drain_shard_stream(shard_streams[0])]
    else:
        # Non-Ray generator-like streams may still be backed by collective
        # workers, so advance shards concurrently instead of shard-by-shard.
        with ThreadPoolExecutor(max_workers=len(shard_streams)) as executor:
            drained_shards = list(executor.map(drain_shard_stream, shard_streams))

    return ShardedBatchStream(
        batch_id=annotated_stream.batch_id,
        layout=annotated_stream.layout,
        dp_size=annotated_stream.dp_size,
        manifest=annotated_stream.manifest,
        shard_streams=drained_shards,
    )


def _student_shard_for_global_slot(
    global_slot: int,
    train_global_batch_size: int,
    student_dp_size: int,
) -> int:
    _require(student_dp_size > 0, "student_dp_size must be positive")
    _require(
        train_global_batch_size % student_dp_size == 0,
        (
            "train_global_batch_size must be divisible by student_dp_size; "
            f"got train_global_batch_size={train_global_batch_size}, "
            f"student_dp_size={student_dp_size}"
        ),
    )
    shard_width = train_global_batch_size // student_dp_size
    _require(
        0 <= global_slot < train_global_batch_size,
        (
            f"global_batch_slot={global_slot} is outside train_global_batch_size="
            f"{train_global_batch_size}"
        ),
    )
    return global_slot // shard_width


def route_annotated_chunks_to_student_shards(
    annotated_chunks: Sequence[AnnotatedTokenChunk],
    *,
    train_global_batch_size: int,
    student_dp_size: int,
) -> list[list[AnnotatedTokenChunk]]:
    routed: list[list[AnnotatedTokenChunk]] = [[] for _ in range(student_dp_size)]
    for chunk in annotated_chunks:
        _require(not chunk.end_of_stream, "EOS chunks should not be routed to students")
        _require(
            chunk.global_batch_slot is not None,
            "AnnotatedTokenChunk.global_batch_slot is required for student routing",
        )
        dst_shards = {
            _student_shard_for_global_slot(
                int(global_slot),
                train_global_batch_size,
                student_dp_size,
            )
            for global_slot in _tensor_to_int_list(chunk.global_batch_slot)
        }
        for dst_shard in sorted(dst_shards):
            routed[dst_shard].append(chunk)
    return routed


def route_drained_stream_to_student_shards(
    drained_stream: ShardedBatchStream,
    *,
    train_global_batch_size: int,
    student_dp_size: int,
) -> ShardedBatchStream:
    all_chunks = [
        chunk
        for shard_chunks in drained_stream.shard_streams
        for chunk in shard_chunks
    ]
    return ShardedBatchStream(
        batch_id=drained_stream.batch_id,
        layout=drained_stream.layout,
        dp_size=student_dp_size,
        manifest=drained_stream.manifest,
        shard_streams=route_annotated_chunks_to_student_shards(
            all_chunks,
            train_global_batch_size=train_global_batch_size,
            student_dp_size=student_dp_size,
        ),
    )


def route_drained_stream_to_student_shards_by_sample_ids(
    drained_stream: ShardedBatchStream,
    *,
    sharded_sample_ids: Sequence[torch.Tensor],
) -> ShardedBatchStream:
    sample_id_to_shard: dict[int, int] = {}
    for shard_idx, sample_ids in enumerate(sharded_sample_ids):
        _require_1d_tensor(sample_ids, name=f"sharded_sample_ids[{shard_idx}]", integral=True)
        for sample_id in _tensor_to_int_list(sample_ids):
            _require(
                sample_id not in sample_id_to_shard,
                f"sample_id {sample_id} appears in multiple student shards",
            )
            sample_id_to_shard[sample_id] = shard_idx

    routed: list[list[AnnotatedTokenChunk]] = [
        [] for _ in range(len(sharded_sample_ids))
    ]
    for chunk in [
        chunk
        for shard_chunks in drained_stream.shard_streams
        for chunk in shard_chunks
    ]:
        _require(not chunk.end_of_stream, "EOS chunks should not be routed to students")
        _require(
            chunk.sample_ids is not None,
            "AnnotatedTokenChunk.sample_ids is required for sample-id routing",
        )
        chunk_sample_ids = _tensor_to_int_list(chunk.sample_ids)
        missing_sample_ids = [
            sample_id
            for sample_id in chunk_sample_ids
            if sample_id not in sample_id_to_shard
        ]
        _require(
            not missing_sample_ids,
            (
                "annotated chunk sample_ids are missing from student shards: "
                f"{sorted(missing_sample_ids)}"
            ),
        )
        dst_shards = {sample_id_to_shard[sample_id] for sample_id in chunk_sample_ids}
        for dst_shard in sorted(dst_shards):
            routed[dst_shard].append(chunk)

    return ShardedBatchStream(
        batch_id=drained_stream.batch_id,
        layout=drained_stream.layout,
        dp_size=len(sharded_sample_ids),
        manifest=drained_stream.manifest,
        shard_streams=routed,
    )


def _select_rows_for_student_shard(
    global_batch_slot: torch.Tensor,
    *,
    train_global_batch_size: int,
    student_dp_size: int,
    student_dp_rank: int,
) -> list[int]:
    selected = []
    for row, global_slot in enumerate(_tensor_to_int_list(global_batch_slot)):
        if (
            _student_shard_for_global_slot(
                global_slot,
                train_global_batch_size,
                student_dp_size,
            )
            == student_dp_rank
        ):
            selected.append(row)
    return selected


def _loss_position_mask(
    *,
    sequence_length: int,
    position_offsets: torch.Tensor,
    positions: torch.Tensor,
    row_index: int,
) -> torch.Tensor:
    mask = torch.zeros(sequence_length, dtype=torch.bool)
    offsets = _tensor_to_int_list(position_offsets)
    for position_offset in range(offsets[row_index], offsets[row_index + 1]):
        position = int(positions[position_offset])
        _require(
            position + 1 < sequence_length,
            f"loss position {position} is outside sequence_length={sequence_length}",
        )
        mask[position + 1] = True
    return mask


def _require_same_tensor_metadata(
    actual: torch.Tensor,
    expected: torch.Tensor | None,
    *,
    name: str,
    context: str,
) -> None:
    _require(expected is not None, f"{context} envelope missing {name}")
    _require(
        torch.equal(actual.cpu(), expected.cpu()),
        f"{context} {name} must match envelope {name}",
    )


def _validate_chunk_matches_envelope(
    chunk: TokenChunk | TeacherTopKChunk,
    envelope: AnnotatedTokenChunk,
    *,
    context: str,
) -> None:
    _require(
        chunk.batch_id == envelope.batch_id,
        f"{context} batch_id must match envelope batch_id",
    )
    _require(
        chunk.chunk_id == envelope.chunk_id,
        f"{context} chunk_id must match envelope chunk_id",
    )
    for name in (
        "sample_ids",
        "sample_order",
        "update_group",
        "global_batch_slot",
    ):
        _require_same_tensor_metadata(
            getattr(chunk, name),
            getattr(envelope, name),
            name=name,
            context=context,
        )


def assemble_dense_distillation_batch_from_annotated_chunks(
    annotated_chunks: Sequence[AnnotatedTokenChunk],
    *,
    sequence_length: int | None = None,
    train_global_batch_size: int,
    student_dp_size: int,
    student_dp_rank: int,
    get_fn: Callable[[object], object] | None = None,
) -> Any:
    if get_fn is None:
        import ray

        get_fn = ray.get
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    resolved_chunks: list[tuple[AnnotatedTokenChunk, TokenChunk, TeacherTopKChunk]] = []
    for envelope in annotated_chunks:
        _require(not envelope.end_of_stream, "student assembly cannot consume EOS chunks")
        token_chunk = get_fn(envelope.token_chunk_ref)
        teacher_chunk = get_fn(envelope.teacher_topk_ref)
        _require(
            isinstance(token_chunk, TokenChunk),
            "token_chunk_ref did not resolve to TokenChunk",
        )
        _require(
            isinstance(teacher_chunk, TeacherTopKChunk),
            "teacher_topk_ref did not resolve to TeacherTopKChunk",
        )
        _validate_chunk_matches_envelope(
            token_chunk,
            envelope,
            context="resolved token_chunk",
        )
        _validate_chunk_matches_envelope(
            teacher_chunk,
            envelope,
            context="resolved teacher_topk chunk",
        )
        resolved_chunks.append((envelope, token_chunk, teacher_chunk))

    _require(resolved_chunks, "student stream shard did not receive any annotated chunks")
    if sequence_length is None:
        sequence_length = max(
            int(token_chunk.input_ids.shape[1])
            for _, token_chunk, _ in resolved_chunks
        )
    _require(sequence_length > 0, "sequence_length must be positive")

    rows: list[dict[str, torch.Tensor | int]] = []
    topk_width: int | None = None
    for _envelope, token_chunk, teacher_chunk in resolved_chunks:
        selected_rows = _select_rows_for_student_shard(
            token_chunk.global_batch_slot,
            train_global_batch_size=train_global_batch_size,
            student_dp_size=student_dp_size,
            student_dp_rank=student_dp_rank,
        )
        teacher_offsets = _tensor_to_int_list(teacher_chunk.position_offsets)
        for row_index in selected_rows:
            topk_width = int(teacher_chunk.topk_logits.shape[-1])
            teacher_logits = torch.zeros(
                (sequence_length, topk_width),
                dtype=teacher_chunk.topk_logits.dtype,
            )
            teacher_indices = torch.zeros(
                (sequence_length, topk_width),
                dtype=teacher_chunk.topk_indices.dtype,
            )
            for position_offset in range(
                teacher_offsets[row_index],
                teacher_offsets[row_index + 1],
            ):
                position = int(teacher_chunk.positions[position_offset])
                teacher_logits[position, :] = teacher_chunk.topk_logits[position_offset]
                teacher_indices[position, :] = teacher_chunk.topk_indices[position_offset]

            input_ids = token_chunk.input_ids[row_index]
            if input_ids.shape[0] < sequence_length:
                input_ids = torch.nn.functional.pad(
                    input_ids,
                    (0, sequence_length - input_ids.shape[0]),
                    value=0,
                )
            else:
                input_ids = input_ids[:sequence_length]

            rows.append(
                {
                    "sample_order": int(token_chunk.sample_order[row_index]),
                    "input_ids": input_ids.cpu(),
                    "input_lengths": token_chunk.input_lengths[row_index].cpu(),
                    "sample_mask": token_chunk.sample_mask[row_index].cpu(),
                    "token_mask": _loss_position_mask(
                        sequence_length=sequence_length,
                        position_offsets=teacher_chunk.position_offsets,
                        positions=teacher_chunk.positions,
                        row_index=row_index,
                    ),
                    "teacher_topk_logits": teacher_logits.cpu(),
                    "teacher_topk_indices": teacher_indices.cpu(),
                }
            )

    rows.sort(key=lambda row: int(row["sample_order"]))
    _require(rows, "student stream shard did not receive any annotated chunks")
    return BatchedDataDict(
        {
            "input_ids": torch.stack([row["input_ids"] for row in rows]),
            "input_lengths": torch.stack([row["input_lengths"] for row in rows]),
            "token_mask": torch.stack([row["token_mask"] for row in rows]),
            "sample_mask": torch.stack([row["sample_mask"] for row in rows]),
            "teacher_topk_logits": torch.stack(
                [row["teacher_topk_logits"] for row in rows]
            ),
            "teacher_topk_indices": torch.stack(
                [row["teacher_topk_indices"] for row in rows]
            ),
        }
    )


def attach_teacher_topk_to_local_batch_from_annotated_chunks(
    data: Any,
    annotated_chunks: Sequence[AnnotatedTokenChunk],
    *,
    get_fn: Callable[[object], object] | None = None,
) -> Any:
    if get_fn is None:
        import ray

        get_fn = ray.get

    _require("sample_ids" in data, "stream training data must include sample_ids")
    sample_ids = _tensor_to_int_list(data["sample_ids"])
    sample_id_to_row = {sample_id: row for row, sample_id in enumerate(sample_ids)}
    sequence_length = int(data["input_ids"].shape[1])
    _require(sequence_length > 0, "input_ids sequence length must be positive")

    topk_width: int | None = None
    dtype = torch.float32
    index_dtype = torch.int64
    pending_rows = set(sample_ids)
    teacher_rows: dict[int, list[tuple[int, torch.Tensor, torch.Tensor]]] = defaultdict(list)

    for envelope in annotated_chunks:
        _require(not envelope.end_of_stream, "local batch cannot consume EOS chunks")
        teacher_chunk = get_fn(envelope.teacher_topk_ref)
        _require(
            isinstance(teacher_chunk, TeacherTopKChunk),
            "teacher_topk_ref did not resolve to TeacherTopKChunk",
        )
        _validate_chunk_matches_envelope(
            teacher_chunk,
            envelope,
            context="resolved teacher_topk chunk",
        )
        topk_width = int(teacher_chunk.topk_logits.shape[-1])
        dtype = teacher_chunk.topk_logits.dtype
        index_dtype = teacher_chunk.topk_indices.dtype
        offsets = _tensor_to_int_list(teacher_chunk.position_offsets)
        positions = _tensor_to_int_list(teacher_chunk.positions)
        for sample_index, sample_id in enumerate(_tensor_to_int_list(teacher_chunk.sample_ids)):
            if sample_id not in sample_id_to_row:
                continue
            pending_rows.discard(sample_id)
            for position_offset in range(offsets[sample_index], offsets[sample_index + 1]):
                teacher_rows[sample_id].append(
                    (
                        positions[position_offset],
                        teacher_chunk.topk_logits[position_offset],
                        teacher_chunk.topk_indices[position_offset],
                    )
                )

    _require(
        topk_width is not None,
        "no teacher top-k chunks were available for local batch",
    )
    _require(
        not pending_rows,
        f"missing teacher annotations for sample_ids={sorted(pending_rows)}",
    )

    teacher_topk_logits = torch.zeros(
        (len(sample_ids), sequence_length, topk_width),
        dtype=dtype,
    )
    teacher_topk_indices = torch.zeros(
        (len(sample_ids), sequence_length, topk_width),
        dtype=index_dtype,
    )
    for sample_id, position_rows in teacher_rows.items():
        row = sample_id_to_row[sample_id]
        for position, logits, indices in position_rows:
            _require(
                0 <= position < sequence_length,
                f"teacher position {position} outside sequence_length={sequence_length}",
            )
            teacher_topk_logits[row, position] = logits
            teacher_topk_indices[row, position] = indices

    data["teacher_topk_logits"] = teacher_topk_logits
    data["teacher_topk_indices"] = teacher_topk_indices
    return data


def attach_sparse_teacher_topk_to_local_batch_from_annotated_chunks(
    data: Any,
    annotated_chunks: Sequence[AnnotatedTokenChunk],
    *,
    get_fn: Callable[[object], object] | None = None,
) -> Any:
    if get_fn is None:
        import ray

        get_fn = ray.get

    _require("sample_ids" in data, "sparse stream training data must include sample_ids")
    sample_ids = _tensor_to_int_list(data["sample_ids"])
    sample_id_to_row = {sample_id: row for row, sample_id in enumerate(sample_ids)}
    sequence_length = int(data["input_ids"].shape[1])
    _require(sequence_length > 0, "input_ids sequence length must be positive")

    topk_width: int | None = None
    dtype = torch.float32
    index_dtype = torch.int64
    pending_rows = set(sample_ids)
    teacher_rows: dict[int, list[tuple[int, torch.Tensor, torch.Tensor]]] = defaultdict(list)

    for envelope in annotated_chunks:
        _require(not envelope.end_of_stream, "local sparse batch cannot consume EOS chunks")
        teacher_chunk = get_fn(envelope.teacher_topk_ref)
        _require(
            isinstance(teacher_chunk, TeacherTopKChunk),
            "teacher_topk_ref did not resolve to TeacherTopKChunk",
        )
        _validate_chunk_matches_envelope(
            teacher_chunk,
            envelope,
            context="resolved teacher_topk chunk",
        )
        topk_width = int(teacher_chunk.topk_logits.shape[-1])
        dtype = teacher_chunk.topk_logits.dtype
        index_dtype = teacher_chunk.topk_indices.dtype
        offsets = _tensor_to_int_list(teacher_chunk.position_offsets)
        positions = _tensor_to_int_list(teacher_chunk.positions)
        for sample_index, sample_id in enumerate(_tensor_to_int_list(teacher_chunk.sample_ids)):
            if sample_id not in sample_id_to_row:
                continue
            pending_rows.discard(sample_id)
            for position_offset in range(offsets[sample_index], offsets[sample_index + 1]):
                position = positions[position_offset]
                _require(
                    0 <= position < sequence_length,
                    f"teacher position {position} outside sequence_length={sequence_length}",
                )
                teacher_rows[sample_id].append(
                    (
                        position,
                        teacher_chunk.topk_logits[position_offset],
                        teacher_chunk.topk_indices[position_offset],
                    )
                )

    _require(
        topk_width is not None,
        "no teacher top-k chunks were available for local sparse batch",
    )
    _require(
        not pending_rows,
        f"missing teacher annotations for sample_ids={sorted(pending_rows)}",
    )

    max_positions = max((len(rows) for rows in teacher_rows.values()), default=0)
    teacher_topk_logits = torch.zeros(
        (len(sample_ids), max_positions, topk_width),
        dtype=dtype,
    )
    teacher_topk_indices = torch.zeros(
        (len(sample_ids), max_positions, topk_width),
        dtype=index_dtype,
    )
    teacher_topk_positions = torch.zeros(
        (len(sample_ids), max_positions),
        dtype=torch.int64,
    )
    teacher_topk_mask = torch.zeros(
        (len(sample_ids), max_positions),
        dtype=torch.bool,
    )

    for sample_id, position_rows in teacher_rows.items():
        row = sample_id_to_row[sample_id]
        for sparse_index, (position, logits, indices) in enumerate(
            sorted(position_rows, key=lambda item: item[0])
        ):
            teacher_topk_positions[row, sparse_index] = position
            teacher_topk_logits[row, sparse_index] = logits
            teacher_topk_indices[row, sparse_index] = indices
            teacher_topk_mask[row, sparse_index] = True

    data["teacher_topk_sparse_logits"] = teacher_topk_logits
    data["teacher_topk_sparse_indices"] = teacher_topk_indices
    data["teacher_topk_sparse_positions"] = teacher_topk_positions
    data["teacher_topk_sparse_mask"] = teacher_topk_mask
    return data


@dataclass(frozen=True)
class ConservationOracle:
    manifest: BatchManifest

    def validate_token_chunks(self, chunks: Sequence[TokenChunk]) -> None:
        self._validate_unique_chunk_ids(chunks, context="token_chunks")
        actual_sample_ids = Counter[int]()
        actual_metadata = {}
        for chunk in chunks:
            _require(
                chunk.batch_id == self.manifest.batch_id,
                "Token chunk batch_id must match manifest.batch_id",
            )
            for sample_id, sample_order, update_group, global_batch_slot in zip(
                _tensor_to_int_list(chunk.sample_ids),
                _tensor_to_int_list(chunk.sample_order),
                _tensor_to_int_list(chunk.update_group),
                _tensor_to_int_list(chunk.global_batch_slot),
                strict=True,
            ):
                actual_sample_ids[sample_id] += 1
                actual_metadata[sample_id] = (
                    sample_order,
                    update_group,
                    global_batch_slot,
                )
        self._assert_expected_sample_ids(actual_sample_ids, context="token_chunks")
        self._assert_expected_metadata(actual_metadata, context="token_chunks")

    def validate_teacher_chunks(self, chunks: Sequence[TeacherTopKChunk]) -> None:
        self._validate_unique_chunk_ids(chunks, context="teacher_chunks")
        actual_sample_ids = Counter[int]()
        actual_metadata = {}
        actual_positions = Counter[tuple[int, int]]()
        for chunk in chunks:
            _require(
                chunk.batch_id == self.manifest.batch_id,
                "Teacher chunk batch_id must match manifest.batch_id",
            )
            sample_ids = _tensor_to_int_list(chunk.sample_ids)
            sample_orders = _tensor_to_int_list(chunk.sample_order)
            update_groups = _tensor_to_int_list(chunk.update_group)
            global_batch_slots = _tensor_to_int_list(chunk.global_batch_slot)
            offsets = _tensor_to_int_list(chunk.position_offsets)
            positions = _tensor_to_int_list(chunk.positions)
            for sample_id, sample_order, update_group, global_batch_slot in zip(
                sample_ids,
                sample_orders,
                update_groups,
                global_batch_slots,
                strict=True,
            ):
                actual_sample_ids[sample_id] += 1
                actual_metadata[sample_id] = (
                    sample_order,
                    update_group,
                    global_batch_slot,
                )
            for sample_index, sample_id in enumerate(sample_ids):
                for position_index in range(offsets[sample_index], offsets[sample_index + 1]):
                    actual_positions[(sample_id, positions[position_index])] += 1
        self._assert_expected_sample_ids(actual_sample_ids, context="teacher_chunks")
        self._assert_expected_metadata(actual_metadata, context="teacher_chunks")
        self._assert_expected_positions(actual_positions, context="teacher_chunks")

    def validate_student_boundary(
        self,
        sample_ids: torch.Tensor,
        sample_order: torch.Tensor,
        update_group: torch.Tensor,
        global_batch_slot: torch.Tensor,
    ) -> None:
        _require_1d_tensor(sample_ids, name="student.sample_ids", integral=True)
        _require_1d_tensor(sample_order, name="student.sample_order", integral=True)
        _require_1d_tensor(update_group, name="student.update_group", integral=True)
        _require_1d_tensor(
            global_batch_slot,
            name="student.global_batch_slot",
            integral=True,
        )
        _require(
            sample_ids.numel()
            == sample_order.numel()
            == update_group.numel()
            == global_batch_slot.numel(),
            "student boundary metadata tensors must have the same length",
        )
        actual_sample_ids = Counter(_tensor_to_int_list(sample_ids))
        actual_rows = list(
            zip(
                _tensor_to_int_list(sample_ids),
                _tensor_to_int_list(sample_order),
                _tensor_to_int_list(update_group),
                _tensor_to_int_list(global_batch_slot),
                strict=True,
            )
        )
        actual_metadata = {
            sample_id: (order, group, slot)
            for sample_id, order, group, slot in actual_rows
        }
        self._assert_expected_sample_ids(actual_sample_ids, context="student_boundary")
        expected_rows = sorted(
            zip(
                _tensor_to_int_list(self.manifest.sample_ids),
                _tensor_to_int_list(self.manifest.sample_order),
                _tensor_to_int_list(self.manifest.update_group),
                _tensor_to_int_list(self.manifest.global_batch_slot),
                strict=True,
            ),
            key=lambda row: row[1],
        )
        actual_order_rows = [
            (sample_id, sample_order)
            for sample_id, sample_order, _, _ in actual_rows
        ]
        expected_order_rows = [
            (sample_id, sample_order)
            for sample_id, sample_order, _, _ in expected_rows
        ]
        _require(
            actual_order_rows == expected_order_rows,
            (
                "student_boundary must be restored to canonical sample_order; "
                f"expected {expected_order_rows}, got {actual_order_rows}"
            ),
        )
        self._assert_expected_metadata(
            actual_metadata,
            context="student_boundary",
        )

    def expected_sample_ids(self) -> Counter[int]:
        return Counter(_tensor_to_int_list(self.manifest.sample_ids))

    def expected_active_positions(self) -> Counter[tuple[int, int]]:
        offsets = _tensor_to_int_list(self.manifest.loss_spans.offsets)
        starts = _tensor_to_int_list(self.manifest.loss_spans.starts)
        ends = _tensor_to_int_list(self.manifest.loss_spans.ends)
        sample_ids = _tensor_to_int_list(self.manifest.sample_ids)
        positions = Counter[tuple[int, int]]()
        for sample_index, sample_id in enumerate(sample_ids):
            for span_index in range(offsets[sample_index], offsets[sample_index + 1]):
                for position in range(starts[span_index], ends[span_index]):
                    positions[(sample_id, position)] += 1
        return positions

    def _validate_unique_chunk_ids(
        self,
        chunks: Sequence[TokenChunk] | Sequence[TeacherTopKChunk],
        *,
        context: str,
    ) -> None:
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        duplicates = [chunk_id for chunk_id, count in Counter(chunk_ids).items() if count > 1]
        _require(
            not duplicates,
            f"{context} contain duplicate chunk_id values: {sorted(duplicates)}",
        )

    def _assert_expected_sample_ids(
        self,
        actual: Counter[int],
        *,
        context: str,
    ) -> None:
        expected = self.expected_sample_ids()
        _require(actual == expected, self._format_counter_mismatch(context, expected, actual))

    def _assert_expected_positions(
        self,
        actual: Counter[tuple[int, int]],
        *,
        context: str,
    ) -> None:
        expected = self.expected_active_positions()
        _require(actual == expected, self._format_counter_mismatch(context, expected, actual))

    def _assert_expected_sample_order(
        self,
        actual_order: Mapping[int, int],
        *,
        context: str,
    ) -> None:
        expected_order = {
            sample_id: sample_order
            for sample_id, sample_order in zip(
                _tensor_to_int_list(self.manifest.sample_ids),
                _tensor_to_int_list(self.manifest.sample_order),
                strict=True,
            )
        }
        _require(
            dict(actual_order) == expected_order,
            (
                f"{context} sample_order mismatch: expected {expected_order}, "
                f"got {dict(actual_order)}"
            ),
        )

    def _assert_expected_metadata(
        self,
        actual_metadata: Mapping[int, tuple[int, int, int]],
        *,
        context: str,
    ) -> None:
        expected_metadata = {
            sample_id: (sample_order, update_group, global_batch_slot)
            for sample_id, sample_order, update_group, global_batch_slot in zip(
                _tensor_to_int_list(self.manifest.sample_ids),
                _tensor_to_int_list(self.manifest.sample_order),
                _tensor_to_int_list(self.manifest.update_group),
                _tensor_to_int_list(self.manifest.global_batch_slot),
                strict=True,
            )
        }
        _require(
            dict(actual_metadata) == expected_metadata,
            (
                f"{context} metadata mismatch: expected {expected_metadata}, "
                f"got {dict(actual_metadata)}"
            ),
        )

    def _format_counter_mismatch(
        self,
        context: str,
        expected: Counter[Any],
        actual: Counter[Any],
    ) -> str:
        missing = sorted(item for item, count in (expected - actual).items() for _ in range(count))
        duplicate = sorted(item for item, count in (actual - expected).items() for _ in range(count))
        return f"{context} conservation mismatch: missing={missing}, duplicate={duplicate}"


def build_conservation_oracle(manifest: BatchManifest) -> ConservationOracle:
    return ConservationOracle(manifest)


__all__ = [
    "AnnotatedTokenChunk",
    "BatchManifest",
    "ChunkAssignment",
    "ConservationOracle",
    "DPReshardPlanner",
    "DataPipelineConfig",
    "ShardedBatchStream",
    "SpanTable",
    "StageLayout",
    "StepManifest",
    "STREAM_METADATA_KEYS",
    "TeacherTopKChunk",
    "TokenChunk",
    "assemble_dense_distillation_batch_from_annotated_chunks",
    "attach_sparse_teacher_topk_to_local_batch_from_annotated_chunks",
    "attach_teacher_topk_to_local_batch_from_annotated_chunks",
    "attach_step_metadata",
    "build_teacher_topk_chunk_from_dense",
    "build_conservation_oracle",
    "build_batch_manifest_from_train_data",
    "build_repeated_sample_metadata",
    "build_step_manifest",
    "build_token_chunk_from_assignment",
    "build_token_stream_from_train_data",
    "build_token_stream_from_rollout_batch",
    "collect_teacher_topk_chunks_to_dense",
    "default_data_pipeline_config",
    "drain_annotated_stream",
    "estimate_active_loss_positions",
    "estimate_chunk_bytes",
    "estimate_chunk_tokens",
    "iter_annotated_teacher_topk_chunks",
    "loss_spans_from_token_mask",
    "parse_data_pipeline_config",
    "route_annotated_chunks_to_student_shards",
    "route_drained_stream_to_student_shards",
    "route_drained_stream_to_student_shards_by_sample_ids",
    "slice_span_table",
    "token_chunk_to_batched_data",
    "validate_chunk_budgets",
    "validate_eos_marker",
    "validate_loss_span_bounds",
    "validate_manifest_metadata",
    "validate_sample_ids_unique",
    "validate_streaming_capabilities",
]
