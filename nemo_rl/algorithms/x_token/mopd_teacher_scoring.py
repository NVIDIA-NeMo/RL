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

"""Teacher-native transcript scoring for online cross-tokenizer MOPD."""

from __future__ import annotations

import math
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Literal

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.algorithms.x_token.mopd import (
    _normalize_leading_instruction_messages,
    align_token_ids,
    apply_teacher_chat_template,
    build_char_mapping,
    classify_thinking_state_before_generated_span,
    compute_offsets_manual,
    first_teacher_prefix_chunk_to_mask,
    generated_think_tool_marker_error,
    map_generated_turns_to_parsed_messages,
    nearest_token_position,
    normalize_alignment_method,
    parse_qwen_chat_token_stream,
    proven_template_only_teacher_token_indices,
    qwen3_assistant_content_transform,
    qwen3_leading_think_prefix_len,
    qwen_chat_boundaries_present,
    render_teacher_open_thinking_prefix,
    sampled_im_end_token_id,
    split_sampled_assistant_eot,
    teacher_message_search_floor,
    teacher_prefix_before_span,
    teacher_template_trims_assistant_content,
    trim_leading_text_tokens_for_alignment,
    trim_outer_whitespace_with_offsets_for_alignment,
)
from nemo_rl.algorithms.x_token.token_aligner import AlignmentPair, TokenAligner
from nemo_rl.data.chat_templates import COMMON_CHAT_TEMPLATES
from nemo_rl.data.deepseek_v4_tokenizer import (
    get_deepseek_v4_tokenizer,
    should_use_deepseek_v4_chat_template,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

_PLAIN_TOKENIZER_LOAD_LOCK = threading.Lock()


@dataclass(frozen=True)
class MOPDTeacherScoreResult:
    """Cross-token teacher scores projected onto the student token grid.

    Attributes:
        logprobs: Floating ``[B, S]`` projected teacher log probabilities.
        valid_mask: Boolean ``[B, S]`` mask proving which values are valid.
        elapsed_seconds: End-to-end preparation, inference, and projection time.
        metrics: Scalar diagnostics suitable for collector aggregation.
    """

    logprobs: torch.Tensor
    valid_mask: torch.Tensor
    elapsed_seconds: float
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.logprobs.ndim != 2:
            raise ValueError(
                f"MOPD teacher logprobs must be [B, S], got {self.logprobs.shape}"
            )
        if not self.logprobs.is_floating_point():
            raise TypeError("MOPD teacher logprobs must have a floating dtype")
        if self.valid_mask.ndim != 2:
            raise ValueError(
                f"MOPD validity mask must be [B, S], got {self.valid_mask.shape}"
            )
        if self.valid_mask.dtype is not torch.bool:
            raise TypeError("MOPD validity mask must have dtype torch.bool")
        if self.logprobs.shape != self.valid_mask.shape:
            raise ValueError(
                "MOPD teacher score/mask shapes differ: "
                f"{self.logprobs.shape} vs {self.valid_mask.shape}"
            )
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("MOPD elapsed_seconds must be finite and nonnegative")
        if any(not math.isfinite(float(value)) for value in self.metrics.values()):
            raise ValueError("MOPD metrics must contain only finite scalars")


@dataclass(frozen=True)
class AssistantTurnStructure:
    """Structural decision for one sampled assistant span."""

    message_log_index: int
    student_start: int
    student_end: int
    initial_thinking_state: str
    render_mode: Literal["normal", "preserve_open", "masked"]
    structural_error: str | None
    assistant_text: str
    teacher_message_index: int | None
    has_student_im_end: bool


@dataclass(frozen=True)
class _PreparedTurn:
    student_global_start: int
    teacher_global_start: int
    pairs: tuple[AlignmentPair, ...]
    masked_pair_indices: frozenset[int]
    template_only_teacher_indices: frozenset[int]


@dataclass(frozen=True)
class _PreparedSample:
    teacher_ids: tuple[int, ...]
    turns: tuple[_PreparedTurn, ...]


@dataclass(frozen=True)
class MOPDTeacherScoringContext:
    """Immutable per-physical-teacher alignment state cached by a collector."""

    student_tokenizer: PreTrainedTokenizerBase
    student_alignment_tokenizer: PreTrainedTokenizerBase
    teacher_tokenizer: PreTrainedTokenizerBase
    aligner: TokenAligner
    alignment_method: str
    mask_first_teacher_prefix_chunk: bool
    exclude_proven_template_only_teacher_tokens: bool
    missing_think_close_policy: Literal["mask", "preserve_open_if_proven"]
    teacher_max_length: int


def _config_mapping(config: Any, *, field_name: str) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    model_dump = getattr(config, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    raise TypeError(f"{field_name} must be a mapping or Pydantic model")


def _required_bool(config: Mapping[str, Any], key: str) -> bool:
    value = config[key]
    if not isinstance(value, bool):
        raise TypeError(f"cross-tokenizer MOPD {key} must be bool")
    return value


def _load_plain_auto_tokenizer(
    name: str, tokenizer_kwargs: Mapping[str, Any]
) -> PreTrainedTokenizerBase:
    """Load with vanilla Transformers even after Fastokens patched the process.

    Fastokens replaces ``TokenizersBackend.from_pretrained`` globally, so merely
    calling ``AutoTokenizer`` does not preserve the independent preprocessing
    tokenizer required by cross-tokenizer MOPD. Both callers run during serial
    startup (driver preflight or collector construction); the lock also prevents
    overlapping loads if that assumption changes.
    """
    with _PLAIN_TOKENIZER_LOAD_LOCK:
        try:
            import fastokens
        except ImportError:
            return AutoTokenizer.from_pretrained(name, **dict(tokenizer_kwargs))

        was_patched = bool(getattr(fastokens, "_patched", False))
        if not was_patched:
            return AutoTokenizer.from_pretrained(name, **dict(tokenizer_kwargs))

        unpatch = getattr(fastokens, "unpatch_transformers", None)
        repatch = getattr(fastokens, "patch_transformers", None)
        if not callable(unpatch) or not callable(repatch):
            raise RuntimeError(
                "Fastokens is active but cannot be temporarily unpatched; "
                "cross-tokenizer MOPD requires a plain Transformers alignment "
                "tokenizer"
            )

        unpatch()
        try:
            return AutoTokenizer.from_pretrained(name, **dict(tokenizer_kwargs))
        finally:
            repatch()


def load_mopd_alignment_tokenizer(
    cross_tokenizer_config: Any,
) -> PreTrainedTokenizerBase:
    """Load the separate preprocessing tokenizer with plain Transformers.

    This object is never passed to the model worker. It exists only to render
    the teacher-native transcript, obtain offsets, and align sampled spans.
    """
    config = _config_mapping(
        cross_tokenizer_config,
        field_name="cross_tokenizer_config",
    )
    tokenizer_config = _config_mapping(
        config["tokenizer"],
        field_name="cross_tokenizer_config.tokenizer",
    )
    name = tokenizer_config["name"]
    if not isinstance(name, str) or not name:
        raise ValueError("cross-tokenizer tokenizer.name must be a nonempty string")
    tokenizer_kwargs = tokenizer_config["tokenizer_kwargs"]
    if not isinstance(tokenizer_kwargs, Mapping):
        raise TypeError("cross-tokenizer tokenizer.tokenizer_kwargs must be a mapping")

    resolved_tokenizer_kwargs = dict(tokenizer_kwargs)
    # Match the independent model-worker tokenizer default. Users can still
    # opt out explicitly with ``trust_remote_code: false``.
    resolved_tokenizer_kwargs.setdefault("trust_remote_code", True)
    tokenizer = _load_plain_auto_tokenizer(name, resolved_tokenizer_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    chat_template = tokenizer_config["chat_template"]
    if chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        )
    elif not isinstance(chat_template, str):
        raise TypeError("cross-tokenizer tokenizer.chat_template must be str or None")
    elif chat_template.lower() == "default":
        pass
    elif chat_template.endswith(".jinja"):
        tokenizer.chat_template = Path(chat_template).read_text(encoding="utf-8")
    else:
        tokenizer.chat_template = chat_template

    chat_template_kwargs = tokenizer_config["chat_template_kwargs"]
    if not isinstance(chat_template_kwargs, Mapping):
        raise TypeError(
            "cross-tokenizer tokenizer.chat_template_kwargs must be a mapping"
        )
    if chat_template_kwargs:
        tokenizer.apply_chat_template = partial(
            tokenizer.apply_chat_template,
            **dict(chat_template_kwargs),
        )
    return tokenizer


def load_mopd_student_alignment_tokenizer(
    student_tokenizer_config: Any,
) -> PreTrainedTokenizerBase:
    """Load a plain-HF copy of the student tokenizer for character offsets.

    The runtime tokenizer remains authoritative for sampled IDs and transcript
    recovery. This copy supplies canonical offsets even when Fastokens is
    enabled process-wide and intentionally omits offset tracking.
    """
    config = _config_mapping(
        student_tokenizer_config,
        field_name="policy.tokenizer",
    )
    name = config.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("policy.tokenizer.name must be a nonempty string")
    raw_tokenizer_kwargs = config.get("tokenizer_kwargs") or {}
    if not isinstance(raw_tokenizer_kwargs, Mapping):
        raise TypeError("policy.tokenizer.tokenizer_kwargs must be a mapping or null")
    tokenizer_kwargs = dict(raw_tokenizer_kwargs)
    tokenizer_kwargs.setdefault("trust_remote_code", True)
    tokenizer = _load_plain_auto_tokenizer(name, tokenizer_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    chat_template_kwargs = config.get("chat_template_kwargs")
    if chat_template_kwargs is not None and not isinstance(
        chat_template_kwargs, Mapping
    ):
        raise TypeError("policy.tokenizer.chat_template_kwargs must be a mapping")
    resolved_chat_template_kwargs = dict(chat_template_kwargs or {})
    if should_use_deepseek_v4_chat_template(config):
        return get_deepseek_v4_tokenizer(tokenizer, resolved_chat_template_kwargs)

    if "chat_template" in config:
        chat_template = config["chat_template"]
        if chat_template is None:
            tokenizer.chat_template = COMMON_CHAT_TEMPLATES.passthrough_prompt_response
        elif not isinstance(chat_template, str):
            raise TypeError("policy.tokenizer.chat_template must be str or null")
        elif chat_template.lower() == "default":
            pass
        elif chat_template.endswith(".jinja"):
            tokenizer.chat_template = Path(chat_template).read_text(encoding="utf-8")
        else:
            tokenizer.chat_template = chat_template
    if chat_template_kwargs is not None:
        tokenizer.apply_chat_template = partial(
            tokenizer.apply_chat_template,
            **resolved_chat_template_kwargs,
        )
    return tokenizer


def _validate_tokenizer_offsets(
    tokenizer: PreTrainedTokenizerBase,
    rendered: str,
    *,
    label: str,
) -> None:
    if not rendered:
        raise ValueError(f"{label} chat template rendered empty text")
    encoding = tokenizer(
        rendered,
        return_tensors=None,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = _to_int_list(encoding["input_ids"])
    if not token_ids:
        raise ValueError(f"{label} tokenizer encoded the preflight as empty")
    offsets = [(int(start), int(end)) for start, end in encoding["offset_mapping"]]
    if len(offsets) != len(token_ids):
        raise ValueError(f"{label} tokenizer returned different token/offset lengths")
    if any(start < 0 or end < start or end > len(rendered) for start, end in offsets):
        raise ValueError(f"{label} tokenizer returned invalid character offsets")
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    if not isinstance(decoded, str) or not decoded:
        raise TypeError(f"{label} tokenizer decode must return nonempty text")


def validate_mopd_student_alignment_preflight(
    student_tokenizer_config: Any,
) -> None:
    """Prove that the plain-HF student alignment tokenizer exposes offsets."""
    tokenizer = load_mopd_student_alignment_tokenizer(student_tokenizer_config)
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "MOPD tokenizer preflight input"},
            {"role": "assistant", "content": "MOPD tokenizer preflight output"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("student chat template preflight must render text")
    _validate_tokenizer_offsets(tokenizer, rendered, label="student alignment")


def validate_cross_tokenizer_preflight(cross_tokenizer_config: Any) -> None:
    """Validate alignment-tokenizer capabilities before allocating a teacher.

    The tokenizer is loaded with the same plain-HF path used by the collector,
    exercised, and then discarded. Model-worker construction is intentionally
    outside this preflight so a bad tokenizer/template cannot reserve GPUs.

    Args:
        cross_tokenizer_config: Nested ``CrossTokenizerMOPDConfig`` model or
            its complete mapping representation.

    Raises:
        TypeError: A tokenizer API returns an unsupported value.
        ValueError: Rendering, encoding, decoding, or offsets cannot be proven.
    """
    config = _config_mapping(
        cross_tokenizer_config,
        field_name="cross_tokenizer_config",
    )
    normalize_alignment_method(config["alignment_method"])
    _required_bool(config, "mask_first_teacher_prefix_chunk")
    _required_bool(config, "exclude_proven_template_only_teacher_tokens")
    missing_policy = config["missing_think_close_policy"]
    if missing_policy not in {"mask", "preserve_open_if_proven"}:
        raise ValueError(
            "cross-tokenizer MOPD missing_think_close_policy must be "
            "'mask' or 'preserve_open_if_proven'"
        )

    tokenizer = load_mopd_alignment_tokenizer(config)
    messages = [
        {"role": "user", "content": "MOPD tokenizer preflight input"},
        {"role": "assistant", "content": "MOPD tokenizer preflight output"},
    ]
    rendered = apply_teacher_chat_template(tokenizer, messages)
    _validate_tokenizer_offsets(tokenizer, rendered, label="cross-tokenizer")


def _classify_assistant_turn_structures(
    *,
    tokenizer: PreTrainedTokenizerBase,
    student_row_ids: list[int],
    student_length: int,
    message_log: Sequence[dict[str, Any]],
    recovered_messages: Sequence[dict[str, str]],
    teacher_message_index_by_log_index: Mapping[int, int],
    expects_thinking_state: bool,
    student_im_end_id: int | None,
    missing_think_close_policy: Literal["mask", "preserve_open_if_proven"],
) -> dict[int, AssistantTurnStructure]:
    """Classify generated turns and fail closed after an unsafe boundary."""
    structures: dict[int, AssistantTurnStructure] = {}
    student_cursor = 0
    causal_suffix_masked = False
    for message_index, message in enumerate(message_log):
        raw_token_ids = message.get("token_ids")
        if raw_token_ids is None:
            continue
        token_ids = _to_int_list(raw_token_ids)
        message_start = student_cursor
        message_end = message_start + len(token_ids)
        student_cursor = message_end
        if not (
            message.get("role") == "assistant"
            and "generation_logprobs" in message
            and token_ids
            and message_end <= student_length
        ):
            continue

        assistant_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        has_student_im_end = bool(
            student_im_end_id is not None and token_ids[-1] == student_im_end_id
        )
        teacher_message_index = teacher_message_index_by_log_index.get(message_index)
        render_mode: Literal["normal", "preserve_open", "masked"] = "normal"
        structural_error = None

        initial_state = classify_thinking_state_before_generated_span(
            tokenizer,
            student_row_ids,
            generated_start=message_start,
        )
        if initial_state == "unknown" and not expects_thinking_state:
            initial_state = "closed"
        if causal_suffix_masked:
            render_mode = "masked"
            structural_error = "causal_suffix"
        elif teacher_message_index is None or not (
            0 <= teacher_message_index < len(recovered_messages)
        ):
            render_mode = "masked"
            structural_error = "teacher_message_mapping"
            causal_suffix_masked = True
        else:
            structural_error = generated_think_tool_marker_error(
                assistant_text,
                initial_state=initial_state,
            )
            if structural_error is not None:
                render_mode = "masked"
                causal_suffix_masked = True
            elif initial_state == "open" and "</think>" not in assistant_text:
                structural_error = "missing_think_close"
                recovered_content = str(
                    recovered_messages[teacher_message_index].get("content") or ""
                )
                student_prefix_proven = recovered_content == (
                    "<think>\n" + assistant_text
                )
                if (
                    missing_think_close_policy == "preserve_open_if_proven"
                    and student_prefix_proven
                ):
                    render_mode = "preserve_open"
                else:
                    render_mode = "masked"
                    if not student_prefix_proven:
                        structural_error = "missing_think_close_unproven"
                causal_suffix_masked = True

        structures[message_index] = AssistantTurnStructure(
            message_log_index=message_index,
            student_start=message_start,
            student_end=message_end,
            initial_thinking_state=initial_state,
            render_mode=render_mode,
            structural_error=structural_error,
            assistant_text=assistant_text,
            teacher_message_index=teacher_message_index,
            has_student_im_end=has_student_im_end,
        )
    return structures


def _to_int_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) for item in value]


def _tokenize_text(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> list[int]:
    return _to_int_list(
        tokenizer(
            text,
            return_tensors=None,
            add_special_tokens=False,
        )["input_ids"]
    )


class MOPDTeacherScorer:
    """Cached scorer for one physical cross-tokenizer teacher group.

    A collector should construct one instance per physical teacher group and
    continue using its existing per-group lock around :meth:`score`. Distinct
    physical teachers may still score concurrently. One nonempty call performs
    exactly one batched ``teacher_group.get_logprobs`` invocation.
    """

    def __init__(
        self,
        *,
        teacher_group: Any,
        context: MOPDTeacherScoringContext,
    ) -> None:
        self.teacher_group = teacher_group
        self.context = context
        self._role_markers: dict[str, tuple[str, str]] = {}

    @classmethod
    def from_config(
        cls,
        *,
        student_tokenizer: PreTrainedTokenizerBase,
        teacher_group: Any,
        cross_tokenizer_config: Any,
        student_tokenizer_config: Any | None = None,
        student_alignment_tokenizer: PreTrainedTokenizerBase | None = None,
        teacher_tokenizer: PreTrainedTokenizerBase | None = None,
        aligner: TokenAligner | None = None,
    ) -> MOPDTeacherScorer:
        """Build and cache the preprocessing tokenizer and aligner once."""
        config = _config_mapping(
            cross_tokenizer_config,
            field_name="cross_tokenizer_config",
        )
        method = normalize_alignment_method(config["alignment_method"])
        missing_policy = config["missing_think_close_policy"]
        if missing_policy not in {"mask", "preserve_open_if_proven"}:
            raise ValueError(
                "cross-tokenizer MOPD missing_think_close_policy must be "
                "'mask' or 'preserve_open_if_proven'"
            )
        alignment_student_tokenizer = student_alignment_tokenizer
        if alignment_student_tokenizer is None:
            alignment_student_tokenizer = (
                load_mopd_student_alignment_tokenizer(student_tokenizer_config)
                if student_tokenizer_config is not None
                else student_tokenizer
            )
        alignment_tokenizer = teacher_tokenizer or load_mopd_alignment_tokenizer(config)
        cached_aligner = aligner or TokenAligner(
            alignment_student_tokenizer,
            alignment_tokenizer,
            projection_matrix_path=None,
        )
        if cached_aligner.student_tokenizer is not alignment_student_tokenizer:
            raise ValueError(
                "Injected MOPD aligner uses a different student alignment tokenizer"
            )
        if cached_aligner.teacher_tokenizer is not alignment_tokenizer:
            raise ValueError("Injected MOPD aligner uses a different teacher tokenizer")

        worker_config = getattr(teacher_group, "cfg", None)
        if not isinstance(worker_config, Mapping):
            raise TypeError("teacher_group.cfg must be a mapping")
        teacher_max_length = int(worker_config["max_total_sequence_length"])
        if teacher_max_length < 1:
            raise ValueError("teacher max_total_sequence_length must be positive")

        context = MOPDTeacherScoringContext(
            student_tokenizer=student_tokenizer,
            student_alignment_tokenizer=alignment_student_tokenizer,
            teacher_tokenizer=alignment_tokenizer,
            aligner=cached_aligner,
            alignment_method=method,
            mask_first_teacher_prefix_chunk=_required_bool(
                config,
                "mask_first_teacher_prefix_chunk",
            ),
            exclude_proven_template_only_teacher_tokens=_required_bool(
                config,
                "exclude_proven_template_only_teacher_tokens",
            ),
            missing_think_close_policy=missing_policy,
            teacher_max_length=teacher_max_length,
        )
        return cls(teacher_group=teacher_group, context=context)

    def _get_role_markers(self, role: str) -> tuple[str, str]:
        if role in self._role_markers:
            return self._role_markers[role]
        sentinel = "NRL_SENT_X9Z_2026"
        try:
            rendered = self.context.student_tokenizer.apply_chat_template(
                [{"role": role, "content": sentinel}],
                tokenize=False,
                add_generation_prompt=False,
            )
            token_ids = _tokenize_text(self.context.student_tokenizer, rendered)
            decoded = self.context.student_tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            )
        except (KeyError, TypeError, ValueError, RuntimeError):
            markers = ("", "")
        else:
            if sentinel not in decoded:
                markers = ("", "")
            else:
                markers = decoded.split(sentinel, 1)
        self._role_markers[role] = markers
        return markers

    def _message_content(self, message: Mapping[str, Any]) -> str:
        content = message.get("content")
        if content:
            return str(content)
        raw_token_ids = message.get("token_ids")
        if raw_token_ids is None:
            return ""
        role = str(message.get("role", ""))
        decoded = self.context.student_tokenizer.decode(
            _to_int_list(raw_token_ids),
            skip_special_tokens=True,
        )
        if role == "assistant":
            return decoded
        prefix, suffix = self._get_role_markers(role)
        if prefix and decoded.startswith(prefix):
            decoded = decoded[len(prefix) :]
        if suffix and decoded.endswith(suffix):
            decoded = decoded[: -len(suffix)]
        return decoded

    def _teacher_tokenize_with_offsets(
        self,
        text: str,
        metrics: Counter[str],
    ) -> tuple[list[int], list[tuple[int, int]] | None]:
        tokenizer = self.context.teacher_tokenizer
        try:
            encoding = tokenizer(
                text,
                return_tensors=None,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            token_ids = _to_int_list(encoding["input_ids"])
            offsets = [
                (int(start), int(end)) for start, end in encoding["offset_mapping"]
            ]
            return token_ids, offsets
        except (KeyError, NotImplementedError, TypeError, ValueError, RuntimeError):
            metrics["teacher_offset_fallbacks"] += 1
            token_ids = _tokenize_text(tokenizer, text)
            offsets = compute_offsets_manual(tokenizer, token_ids, text)
            return token_ids, offsets

    def _student_offsets(
        self,
        text: str,
        expected_ids: list[int],
    ) -> list[tuple[int, int]] | None:
        tokenizer = self.context.student_alignment_tokenizer
        try:
            try:
                encoding = tokenizer(
                    text,
                    return_tensors=None,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                token_ids = _to_int_list(encoding["input_ids"])
                offsets = [
                    (int(start), int(end)) for start, end in encoding["offset_mapping"]
                ]
            except (KeyError, NotImplementedError, TypeError, ValueError):
                token_ids = _tokenize_text(tokenizer, text)
                offsets = compute_offsets_manual(tokenizer, token_ids, text)
        except RuntimeError:
            return None
        if token_ids == expected_ids:
            return offsets
        try:
            decoded = tokenizer.decode(expected_ids, skip_special_tokens=True)
        except (TypeError, ValueError, RuntimeError):
            return None
        if decoded != text:
            return None
        return compute_offsets_manual(
            tokenizer,
            expected_ids,
            text,
            skip_special_tokens=True,
        )

    def _fallback_messages_and_mapping(
        self,
        message_log: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, str]], dict[int, int]]:
        messages = [
            {
                "role": str(message["role"]),
                "content": self._message_content(message),
            }
            for message in message_log
        ]
        normalized, instruction_prefix_length = _normalize_leading_instruction_messages(
            messages
        )
        if instruction_prefix_length == 0:
            return normalized, {
                message_index: message_index
                for message_index in range(len(message_log))
            }
        shift = instruction_prefix_length - 1
        return normalized, {
            message_index: message_index - shift
            for message_index in range(instruction_prefix_length, len(message_log))
        }

    def _recover_transcript(
        self,
        *,
        student_row_ids: list[int],
        student_length: int,
        message_log: Sequence[dict[str, Any]],
        metrics: Counter[str],
    ) -> tuple[list[dict[str, str]], dict[int, int]]:
        messages, mapping = self._fallback_messages_and_mapping(message_log)
        message_stream: list[int] = []
        for message in message_log:
            raw_token_ids = message.get("token_ids")
            if raw_token_ids is not None:
                message_stream.extend(_to_int_list(raw_token_ids))

        structured = qwen_chat_boundaries_present(
            self.context.student_tokenizer,
            student_row_ids,
        )
        exact_stream = (
            len(message_stream) >= student_length
            and message_stream[:student_length] == student_row_ids
        )
        parsed = (
            parse_qwen_chat_token_stream(
                self.context.student_tokenizer,
                student_row_ids,
            )
            if exact_stream
            else None
        )
        if parsed is not None:
            parsed_messages, parsed_bounds = parsed
            parsed_mapping = map_generated_turns_to_parsed_messages(
                message_log,
                parsed_bounds,
                student_length=student_length,
            )
            if parsed_mapping is not None:
                metrics["transcripts_recovered"] += 1
                return parsed_messages, parsed_mapping

        metrics["transcript_recovery_fallbacks"] += 1
        if structured:
            metrics["structured_transcripts_failed_closed"] += 1
            return messages, {}
        return messages, mapping

    def _render_teacher_transcript(
        self,
        *,
        messages: list[dict[str, str]],
        turn_structures: dict[int, AssistantTurnStructure],
        metrics: Counter[str],
    ) -> tuple[
        str,
        list[int],
        int | None,
        int | None,
        int | None,
        int | None,
    ]:
        teacher_tokenizer = self.context.teacher_tokenizer
        boundary_structure = next(
            (
                structure
                for structure in turn_structures.values()
                if structure.render_mode != "normal"
            ),
            None,
        )
        preserve_log_index = None
        preserve_start = None
        preserve_content_end = None
        preserve_eot_position = None

        if (
            boundary_structure is None
            or boundary_structure.teacher_message_index is None
        ):
            teacher_text = apply_teacher_chat_template(teacher_tokenizer, messages)
            return (
                teacher_text,
                _tokenize_text(teacher_tokenizer, teacher_text),
                preserve_log_index,
                preserve_start,
                preserve_content_end,
                preserve_eot_position,
            )

        teacher_message_index = boundary_structure.teacher_message_index
        if boundary_structure.render_mode == "preserve_open":
            try:
                prefix_text = render_teacher_open_thinking_prefix(
                    teacher_tokenizer,
                    messages[:teacher_message_index],
                )
                prefix_ids = _tokenize_text(teacher_tokenizer, prefix_text)
                body_ids = _tokenize_text(
                    teacher_tokenizer,
                    boundary_structure.assistant_text,
                )
                teacher_text = prefix_text + boundary_structure.assistant_text
                teacher_ids = [*prefix_ids, *body_ids]
                preserve_log_index = boundary_structure.message_log_index
                preserve_start = len(prefix_ids)
                preserve_content_end = len(teacher_ids)
                if boundary_structure.has_student_im_end:
                    teacher_im_end_id = sampled_im_end_token_id(teacher_tokenizer)
                    if teacher_im_end_id is None:
                        raise ValueError(
                            "Teacher tokenizer has no <|im_end|> EOS token"
                        )
                    teacher_text += "<|im_end|>"
                    preserve_eot_position = len(teacher_ids)
                    teacher_ids.append(teacher_im_end_id)
                metrics["open_state_preserved"] += 1
                return (
                    teacher_text,
                    teacher_ids,
                    preserve_log_index,
                    preserve_start,
                    preserve_content_end,
                    preserve_eot_position,
                )
            except (KeyError, TypeError, ValueError, RuntimeError):
                metrics["teacher_open_render_failures"] += 1
                turn_structures[boundary_structure.message_log_index] = replace(
                    boundary_structure,
                    render_mode="masked",
                    structural_error="teacher_open_render",
                )

        teacher_text = apply_teacher_chat_template(
            teacher_tokenizer,
            messages[:teacher_message_index],
        )
        return (
            teacher_text,
            _tokenize_text(teacher_tokenizer, teacher_text),
            preserve_log_index,
            preserve_start,
            preserve_content_end,
            preserve_eot_position,
        )

    def _prepare_sample(
        self,
        *,
        student_row_ids: list[int],
        student_length: int,
        message_log: Sequence[dict[str, Any]],
        metrics: Counter[str],
    ) -> _PreparedSample:
        student_tokenizer = self.context.student_tokenizer
        teacher_tokenizer = self.context.teacher_tokenizer
        messages, teacher_message_index_by_log_index = self._recover_transcript(
            student_row_ids=student_row_ids,
            student_length=student_length,
            message_log=message_log,
            metrics=metrics,
        )

        student_template = getattr(student_tokenizer, "chat_template", None)
        expects_thinking_state = bool(
            isinstance(student_template, str)
            and "<think>" in student_template
            and "</think>" in student_template
        )
        student_im_end_id = sampled_im_end_token_id(student_tokenizer)
        teacher_im_end_id = sampled_im_end_token_id(teacher_tokenizer)
        turn_structures = _classify_assistant_turn_structures(
            tokenizer=student_tokenizer,
            student_row_ids=student_row_ids,
            student_length=student_length,
            message_log=message_log,
            recovered_messages=messages,
            teacher_message_index_by_log_index=teacher_message_index_by_log_index,
            expects_thinking_state=expects_thinking_state,
            student_im_end_id=student_im_end_id,
            missing_think_close_policy=self.context.missing_think_close_policy,
        )
        for structure in turn_structures.values():
            if (
                structure.initial_thinking_state == "open"
                and "</think>" not in structure.assistant_text
            ):
                metrics["missing_think_close_detected"] += 1
            if structure.render_mode == "masked":
                if structure.structural_error == "causal_suffix":
                    metrics["causal_suffix_masked"] += 1
                else:
                    metrics["malformed_structure_masked"] += 1

        (
            teacher_text,
            teacher_ids,
            preserve_log_index,
            preserve_start,
            preserve_content_end,
            preserve_eot_position,
        ) = self._render_teacher_transcript(
            messages=messages,
            turn_structures=turn_structures,
            metrics=metrics,
        )
        if len(teacher_ids) > self.context.teacher_max_length:
            metrics["teacher_transcripts_too_long"] += 1
            return _PreparedSample(teacher_ids=(0,), turns=())

        trim_teacher_content = teacher_template_trims_assistant_content(
            teacher_tokenizer
        )
        turns: list[_PreparedTurn] = []
        student_cursor = 0
        teacher_search_start = 0
        for message_index, message in enumerate(message_log):
            raw_token_ids = message.get("token_ids")
            if raw_token_ids is None:
                continue
            message_token_ids = _to_int_list(raw_token_ids)
            message_start = student_cursor
            message_end = message_start + len(message_token_ids)
            student_cursor = message_end
            if not (
                message.get("role") == "assistant"
                and "generation_logprobs" in message
                and message_token_ids
                and message_end <= student_length
            ):
                continue

            structure = turn_structures.get(message_index)
            if structure is None or structure.render_mode == "masked":
                metrics["turns_skipped"] += 1
                continue

            assistant_text = student_tokenizer.decode(
                message_token_ids,
                skip_special_tokens=True,
            )
            if not assistant_text:
                metrics["turns_skipped"] += 1
                continue

            preserve_open_turn = structure.render_mode == "preserve_open"
            trim_this_turn = trim_teacher_content and not preserve_open_turn
            transformed_text, had_think = qwen3_assistant_content_transform(
                assistant_text,
                trim_outer_whitespace=trim_this_turn,
            )
            student_alignment_text = assistant_text
            student_local_ids, student_eot_local_index = split_sampled_assistant_eot(
                student_tokenizer,
                message_token_ids,
            )
            has_student_eot = student_eot_local_index is not None
            if has_student_eot:
                metrics["assistant_eot_expected"] += 1
            student_global_start = message_start
            student_offsets_override: list[tuple[int, int]] | None = None
            offset_frame_compatible = True

            if trim_this_turn:
                trim_result = trim_outer_whitespace_with_offsets_for_alignment(
                    self.context.student_alignment_tokenizer,
                    student_local_ids,
                    student_alignment_text,
                )
                if trim_result is None:
                    offset_frame_compatible = False
                else:
                    (
                        student_local_ids,
                        dropped_outer_prefix_tokens,
                        _dropped_outer_suffix_tokens,
                        student_alignment_text,
                        student_offsets_override,
                    ) = trim_result
                    student_global_start += dropped_outer_prefix_tokens
                    if student_eot_local_index is not None:
                        student_eot_local_index -= dropped_outer_prefix_tokens

            if had_think:
                trim_characters = qwen3_leading_think_prefix_len(student_alignment_text)
                (
                    student_local_ids,
                    dropped_prefix_tokens,
                    student_alignment_text,
                ) = trim_leading_text_tokens_for_alignment(
                    self.context.student_alignment_tokenizer,
                    student_local_ids,
                    student_alignment_text,
                    trim_characters,
                )
                if dropped_prefix_tokens:
                    student_global_start += dropped_prefix_tokens
                    if student_eot_local_index is not None:
                        student_eot_local_index -= dropped_prefix_tokens
                    if student_offsets_override is not None:
                        student_offsets_override = [
                            (start - trim_characters, end - trim_characters)
                            for start, end in student_offsets_override[
                                dropped_prefix_tokens:
                            ]
                        ]
                if not student_local_ids or not student_alignment_text:
                    metrics["turns_skipped"] += 1
                    continue

            teacher_assistant_ids, teacher_assistant_offsets = (
                self._teacher_tokenize_with_offsets(transformed_text, metrics)
            )
            if (
                not teacher_assistant_ids
                or teacher_assistant_offsets is None
                or len(teacher_assistant_ids) != len(teacher_assistant_offsets)
            ):
                metrics["teacher_offset_failures"] += 1
                metrics["turns_skipped"] += 1
                continue

            teacher_message_index = teacher_message_index_by_log_index.get(
                message_index
            )
            if teacher_message_index is None:
                metrics["teacher_message_mapping_failures"] += 1
                metrics["turns_skipped"] += 1
                continue

            teacher_start = -1
            teacher_needle_end = -1
            teacher_end_position = None
            if preserve_open_turn:
                if (
                    preserve_log_index != message_index
                    or preserve_start is None
                    or preserve_content_end is None
                    or preserve_start < teacher_search_start
                    or teacher_ids[preserve_start:preserve_content_end]
                    != teacher_assistant_ids
                ):
                    metrics["teacher_open_span_failures"] += 1
                    metrics["turns_skipped"] += 1
                    continue
                teacher_start = preserve_start
                teacher_content_end = preserve_content_end
                teacher_needle_end = teacher_content_end
                teacher_end_position = preserve_eot_position
            else:
                needle_trim = min(2, len(teacher_assistant_ids) // 4)
                search_needle = (
                    teacher_assistant_ids[
                        needle_trim : len(teacher_assistant_ids) - needle_trim
                    ]
                    if needle_trim > 0
                    else teacher_assistant_ids
                )
                message_search_floor = teacher_message_search_floor(
                    teacher_tokenizer,
                    messages,
                    teacher_message_index,
                    teacher_text,
                    teacher_ids,
                )
                if message_search_floor is None:
                    metrics["teacher_message_boundary_failures"] += 1
                    metrics["turns_skipped"] += 1
                    continue
                for position in range(
                    max(teacher_search_start, message_search_floor),
                    len(teacher_ids) - len(search_needle) + 1,
                ):
                    if (
                        teacher_ids[position : position + len(search_needle)]
                        == search_needle
                    ):
                        teacher_start = position - needle_trim
                        teacher_needle_end = position + len(search_needle)
                        break
                if teacher_start < 0:
                    metrics["teacher_span_not_found"] += 1
                    metrics["turns_skipped"] += 1
                    continue
                teacher_start = max(0, teacher_start)
                teacher_content_end = min(
                    len(teacher_ids),
                    teacher_start + len(teacher_assistant_ids),
                )
                if has_student_eot:
                    teacher_end_position = nearest_token_position(
                        teacher_ids,
                        teacher_im_end_id,
                        expected_position=teacher_content_end,
                        minimum_position=max(teacher_start, teacher_needle_end),
                    )
                    if teacher_end_position is not None:
                        teacher_content_end = teacher_end_position

            teacher_local_ids = teacher_ids[teacher_start:teacher_content_end]
            teacher_local_offsets = teacher_assistant_offsets[: len(teacher_local_ids)]
            if not teacher_local_ids or len(teacher_local_ids) != len(
                teacher_local_offsets
            ):
                metrics["teacher_local_span_failures"] += 1
                metrics["turns_skipped"] += 1
                continue

            student_offsets = student_offsets_override
            if student_offsets is None and offset_frame_compatible:
                student_offsets = self._student_offsets(
                    student_alignment_text,
                    student_local_ids,
                )
            if student_offsets is not None and had_think:
                char_mapping = build_char_mapping(
                    student_alignment_text,
                    transformed_text,
                )
                if char_mapping is None:
                    student_offsets = None
                    metrics["think_char_mapping_failures"] += 1
                else:
                    student_offsets = [
                        (char_mapping[start], char_mapping[end])
                        for start, end in student_offsets
                    ]
            if student_offsets is None:
                metrics["student_offset_failures"] += 1
                metrics["turns_skipped"] += 1
                continue

            template_only_indices: set[int] = set()
            if (
                self.context.exclude_proven_template_only_teacher_tokens
                and had_think
                and teacher_local_ids == teacher_assistant_ids[: len(teacher_local_ids)]
            ):
                provenance = proven_template_only_teacher_token_indices(
                    student_alignment_text,
                    transformed_text,
                    teacher_local_offsets,
                )
                if provenance is None:
                    metrics["template_provenance_turns_unproven"] += 1
                else:
                    template_only_indices, mixed_indices = provenance
                    metrics["template_provenance_turns_proven"] += 1
                    metrics["template_only_teacher_tokens_identified"] += len(
                        template_only_indices
                    )
                    metrics["template_mixed_teacher_tokens_retained"] += len(
                        mixed_indices
                    )

            try:
                pairs = align_token_ids(
                    student_local_ids,
                    teacher_local_ids,
                    aligner=self.context.aligner,
                    method=self.context.alignment_method,
                    student_offsets=student_offsets,
                    teacher_offsets=teacher_local_offsets,
                )
            except (TypeError, ValueError, RuntimeError, AssertionError):
                metrics["alignment_failures"] += 1
                metrics["turns_skipped"] += 1
                continue

            teacher_turn_end = teacher_content_end
            if (
                has_student_eot
                and student_eot_local_index is not None
                and teacher_im_end_id is not None
                and teacher_end_position is not None
            ):
                pairs.append(
                    AlignmentPair(
                        s_tokens=[],
                        t_tokens=[],
                        s_start=student_eot_local_index,
                        s_end=student_eot_local_index + 1,
                        t_start=teacher_end_position - teacher_start,
                        t_end=teacher_end_position - teacher_start + 1,
                        is_correct=True,
                    )
                )
                teacher_turn_end = teacher_end_position + 1
                metrics["assistant_eot_aligned"] += 1
            teacher_search_start = teacher_turn_end

            has_teacher_prefix, _teacher_prefix_tail = teacher_prefix_before_span(
                teacher_tokenizer,
                teacher_ids,
                teacher_start,
                student_alignment_text,
            )
            masked_pair_indices: set[int] = set()
            if self.context.mask_first_teacher_prefix_chunk:
                masked_index = first_teacher_prefix_chunk_to_mask(
                    pairs,
                    has_teacher_prefix,
                )
                if masked_index is not None:
                    masked_pair_indices.add(masked_index)
                    metrics["teacher_prefix_chunks_masked"] += 1

            turns.append(
                _PreparedTurn(
                    student_global_start=student_global_start,
                    teacher_global_start=teacher_start,
                    pairs=tuple(pairs),
                    masked_pair_indices=frozenset(masked_pair_indices),
                    template_only_teacher_indices=frozenset(template_only_indices),
                )
            )
            metrics["turns_aligned"] += 1

        if turns:
            metrics["samples_with_valid_alignment"] += 1
        else:
            metrics["samples_fully_masked"] += 1
        return _PreparedSample(teacher_ids=tuple(teacher_ids), turns=tuple(turns))

    def _project(
        self,
        *,
        prepared: Sequence[_PreparedSample],
        teacher_logprobs: torch.Tensor,
        student_sequence_length: int,
        metrics: Counter[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(prepared)
        projected = torch.zeros(
            batch_size,
            student_sequence_length,
            dtype=torch.float32,
        )
        valid_mask = torch.zeros(
            batch_size,
            student_sequence_length,
            dtype=torch.bool,
        )
        for row, sample in enumerate(prepared):
            for turn in sample.turns:
                for pair_index, pair in enumerate(turn.pairs):
                    if pair_index in turn.masked_pair_indices:
                        continue
                    if pair.s_start < 0:
                        metrics["teacher_only_orphan_pairs_masked"] += 1
                        continue
                    if pair.t_start < 0:
                        metrics["student_only_orphan_pairs_masked"] += 1
                        continue
                    if pair.s_end <= pair.s_start or pair.t_end <= pair.t_start:
                        metrics["invalid_alignment_span_pairs_masked"] += 1
                        continue
                    if not pair.is_correct:
                        metrics["incorrect_alignment_pairs_masked"] += 1
                        continue
                    student_start = turn.student_global_start + pair.s_start
                    student_end = turn.student_global_start + pair.s_end
                    teacher_start = turn.teacher_global_start + pair.t_start
                    teacher_end = turn.teacher_global_start + pair.t_end
                    if (
                        student_start < 0
                        or student_end > student_sequence_length
                        or teacher_start < 0
                        or teacher_end > teacher_logprobs.shape[1]
                    ):
                        metrics["projection_bounds_failures"] += 1
                        continue

                    retained_teacher_positions = [
                        turn.teacher_global_start + teacher_position
                        for teacher_position in range(pair.t_start, pair.t_end)
                        if teacher_position not in turn.template_only_teacher_indices
                    ]
                    excluded_count = (
                        pair.t_end - pair.t_start - len(retained_teacher_positions)
                    )
                    if excluded_count:
                        metrics["template_only_teacher_tokens_excluded"] += (
                            excluded_count
                        )
                    if not retained_teacher_positions:
                        metrics["template_only_chunks_fully_excluded"] += 1
                        continue
                    if excluded_count:
                        metrics["template_only_chunks_adjusted"] += 1

                    chunk_sum = teacher_logprobs[
                        row,
                        retained_teacher_positions,
                    ].sum(dtype=torch.float32)
                    student_width = student_end - student_start
                    projected[row, student_start:student_end] = chunk_sum / float(
                        student_width
                    )
                    valid_mask[row, student_start:student_end] = True

        metrics["valid_tokens"] = int(valid_mask.sum().item())
        return projected, valid_mask

    def score(
        self,
        *,
        input_ids: torch.Tensor,
        message_logs: Sequence[Sequence[dict[str, Any]]],
        input_lengths: torch.Tensor | None = None,
    ) -> MOPDTeacherScoreResult:
        """Score one routed physical-teacher group with one batched forward."""
        started_at = time.perf_counter()
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B, S], got {input_ids.shape}")
        batch_size, student_sequence_length = input_ids.shape
        if len(message_logs) != batch_size:
            raise ValueError(
                "message_logs batch size differs from input_ids: "
                f"{len(message_logs)} vs {batch_size}"
            )
        if input_lengths is None:
            lengths = torch.full(
                (batch_size,),
                student_sequence_length,
                dtype=torch.long,
            )
        else:
            if input_lengths.ndim != 1 or input_lengths.shape[0] != batch_size:
                raise ValueError(
                    "input_lengths must be [B] and match input_ids; got "
                    f"{input_lengths.shape} for B={batch_size}"
                )
            lengths = input_lengths.detach().to(device="cpu", dtype=torch.long)
        if bool(((lengths < 0) | (lengths > student_sequence_length)).any()):
            raise ValueError("input_lengths entries must lie in [0, S]")

        metrics: Counter[str] = Counter()
        # Keep observability schemas stable even on a clean batch.
        for metric_name in (
            "samples_with_valid_alignment",
            "samples_fully_masked",
            "turns_aligned",
            "turns_skipped",
            "valid_tokens",
            "teacher_calls",
            "dp_padding_rows",
            "teacher_inference_seconds",
            "incorrect_alignment_pairs_masked",
            "teacher_only_orphan_pairs_masked",
            "student_only_orphan_pairs_masked",
            "invalid_alignment_span_pairs_masked",
            "alignment_failures",
            "teacher_offset_failures",
            "student_offset_failures",
            "teacher_prefix_chunks_masked",
            "template_only_teacher_tokens_excluded",
            "template_only_chunks_fully_excluded",
            "malformed_structure_masked",
            "teacher_transcripts_too_long",
            "projection_bounds_failures",
        ):
            metrics[metric_name] = 0
        metrics["samples"] = batch_size
        if batch_size == 0:
            empty_scores = torch.zeros(
                0,
                student_sequence_length,
                dtype=torch.float32,
            )
            empty_mask = torch.zeros_like(empty_scores, dtype=torch.bool)
            return MOPDTeacherScoreResult(
                logprobs=empty_scores,
                valid_mask=empty_mask,
                elapsed_seconds=time.perf_counter() - started_at,
                metrics={f"mopd/{key}": float(value) for key, value in metrics.items()},
            )

        prepared: list[_PreparedSample] = []
        for row in range(batch_size):
            student_length = int(lengths[row].item())
            student_row_ids = _to_int_list(
                input_ids[row, :student_length].detach().to(device="cpu")
            )
            try:
                sample = self._prepare_sample(
                    student_row_ids=student_row_ids,
                    student_length=student_length,
                    message_log=message_logs[row],
                    metrics=metrics,
                )
            except Exception as error:
                # Tokenizer plugins can raise implementation-specific errors.
                # Isolation at the sample boundary is required: an unprovable
                # transcript is masked, while the rest of the group is scored.
                metrics["sample_prepare_failures"] += 1
                metrics[f"sample_prepare_failure/{type(error).__name__}"] += 1
                sample = _PreparedSample(teacher_ids=(0,), turns=())
            prepared.append(sample)

        teacher_pad_id = self.context.teacher_tokenizer.pad_token_id
        if teacher_pad_id is None:
            teacher_pad_id = 0
        max_teacher_length = max(len(sample.teacher_ids) for sample in prepared)
        teacher_input_ids = torch.full(
            (batch_size, max_teacher_length),
            int(teacher_pad_id),
            dtype=torch.long,
        )
        teacher_input_lengths = torch.zeros(batch_size, dtype=torch.long)
        for row, sample in enumerate(prepared):
            teacher_ids = sample.teacher_ids or (int(teacher_pad_id),)
            teacher_input_ids[row, : len(teacher_ids)] = torch.tensor(
                teacher_ids,
                dtype=torch.long,
            )
            teacher_input_lengths[row] = len(teacher_ids)

        dp_size = int(
            self.teacher_group.sharding_annotations.get_axis_size("data_parallel")
        )
        if dp_size < 1:
            raise ValueError(
                f"teacher data-parallel size must be positive, got {dp_size}"
            )
        remainder = batch_size % dp_size
        if remainder:
            padding_rows = dp_size - remainder
            teacher_input_ids = torch.cat(
                [
                    teacher_input_ids,
                    teacher_input_ids[-1:].expand(padding_rows, -1),
                ],
                dim=0,
            )
            teacher_input_lengths = torch.cat(
                [
                    teacher_input_lengths,
                    teacher_input_lengths[-1:].expand(padding_rows),
                ],
                dim=0,
            )
            metrics["dp_padding_rows"] = padding_rows

        teacher_batch = BatchedDataDict(
            {
                "input_ids": teacher_input_ids,
                "input_lengths": teacher_input_lengths,
            }
        )
        inference_started_at = time.perf_counter()
        raw_output = self.teacher_group.get_logprobs(teacher_batch)
        inference_seconds = time.perf_counter() - inference_started_at
        metrics["teacher_calls"] = 1
        raw_logprobs = raw_output["reference_logprobs"]
        if not isinstance(raw_logprobs, torch.Tensor):
            raise TypeError("teacher reference_logprobs must be a tensor")
        if raw_logprobs.ndim != 2:
            raise ValueError(
                f"teacher reference_logprobs must be [B, S], got {raw_logprobs.shape}"
            )
        if raw_logprobs.shape[0] < batch_size:
            raise ValueError(
                "teacher returned fewer rows than requested: "
                f"{raw_logprobs.shape[0]} < {batch_size}"
            )
        if raw_logprobs.shape[1] < max_teacher_length:
            raise ValueError(
                "teacher returned a shorter sequence than requested: "
                f"{raw_logprobs.shape[1]} < {max_teacher_length}"
            )
        teacher_logprobs = (
            raw_logprobs[:batch_size]
            .detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
        )

        projected, valid_mask = self._project(
            prepared=prepared,
            teacher_logprobs=teacher_logprobs,
            student_sequence_length=student_sequence_length,
            metrics=metrics,
        )
        elapsed_seconds = time.perf_counter() - started_at
        metric_values = {
            f"mopd/{key}": float(value) for key, value in sorted(metrics.items())
        }
        metric_values["mopd/teacher_inference_seconds"] = inference_seconds
        return MOPDTeacherScoreResult(
            logprobs=projected,
            valid_mask=valid_mask,
            elapsed_seconds=elapsed_seconds,
            metrics=metric_values,
        )


def build_mopd_teacher_scorer(
    *,
    student_tokenizer: PreTrainedTokenizerBase,
    teacher_group: Any,
    cross_tokenizer_config: Any,
    student_tokenizer_config: Any | None = None,
    student_alignment_tokenizer: PreTrainedTokenizerBase | None = None,
    teacher_tokenizer: PreTrainedTokenizerBase | None = None,
    aligner: TokenAligner | None = None,
) -> MOPDTeacherScorer:
    """Build one collector-cached scorer for a physical teacher group."""
    return MOPDTeacherScorer.from_config(
        student_tokenizer=student_tokenizer,
        teacher_group=teacher_group,
        cross_tokenizer_config=cross_tokenizer_config,
        student_tokenizer_config=student_tokenizer_config,
        student_alignment_tokenizer=student_alignment_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        aligner=aligner,
    )


def score_mopd_teacher_group(
    scorer: MOPDTeacherScorer,
    *,
    input_ids: torch.Tensor,
    message_logs: Sequence[Sequence[dict[str, Any]]],
    input_lengths: torch.Tensor | None = None,
) -> MOPDTeacherScoreResult:
    """Score one already-routed group through its cached scorer."""
    return scorer.score(
        input_ids=input_ids,
        message_logs=message_logs,
        input_lengths=input_lengths,
    )


__all__ = [
    "MOPDTeacherScoreResult",
    "MOPDTeacherScorer",
    "MOPDTeacherScoringContext",
    "build_mopd_teacher_scorer",
    "load_mopd_alignment_tokenizer",
    "load_mopd_student_alignment_tokenizer",
    "score_mopd_teacher_group",
    "validate_cross_tokenizer_preflight",
    "validate_mopd_student_alignment_preflight",
]
