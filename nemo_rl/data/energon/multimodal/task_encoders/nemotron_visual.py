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

import hashlib
import importlib
import json
import logging
import math
import random
import re
import threading
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import torch
from megatron.energon import edataclass
from megatron.energon.av import AVDecoder
from PIL import Image
from torchvision.transforms import ToPILImage

from nemo_rl.data.energon.multimodal.task_encoders.generic_sft import (
    _normalize_messages,
)
from nemo_rl.data.energon.multimodal.task_encoders.nemotron_tokenization import (
    MM_MARKER,
    NoTrainableTokensError,
    tokenize_nemotron_conversation,
    validate_text_content,
)
from nemo_rl.data.energon.multimodal.types import (
    CanonicalSFTSample,
    EncodedSFTSample,
    MediaRef,
)
from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    resolve_to_image,
)

# The marker sits between two special tokens, so encoding segment-by-segment
# cannot merge across the splice boundary.
COMPACT_IMAGE_PLACEHOLDER = f"<img>{MM_MARKER}</img>"

# Serializes the temporary av_open rebind below. Loader workers are separate
# forked processes, so each holds its own copy and this is uncontended in
# practice; it only matters if a caller ever decodes from multiple threads in
# one process (num_workers=0 runs the loader inside the training process).
_ENERGON_AV_OPEN_LOCK = threading.Lock()

logger = logging.getLogger(__name__)

_CLIP_PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)
_TO_PIL_IMAGE = ToPILImage()

# Thinking-tag spacing follows the Nemotron rules in Megatron-LM revision
# 6822175d92a40e0528be905aee50f5930cfa0c98, multimodal/data_loading/task_encoder.py.


def _metadata(ref: MediaRef) -> dict[str, str | int | float | bool | None]:
    return dict(ref.metadata)


def _required_metadata_int(ref: MediaRef, key: str) -> int:
    value = _metadata(ref).get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"Nemotron {ref.modality} media requires positive integer metadata "
            f"{key!r}; got {value!r}."
        )
    return value


def _required_metadata_float(ref: MediaRef, key: str) -> float:
    value = _metadata(ref).get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(
            f"Nemotron {ref.modality} media requires positive numeric metadata "
            f"{key!r}; got {value!r}."
        )
    return float(value)


def _optional_metadata_float(ref: MediaRef, key: str) -> float | None:
    value = _metadata(ref).get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Nemotron {ref.modality} media metadata {key!r} must be numeric; "
            f"got {value!r}."
        )
    return float(value)


@dataclass(frozen=True)
class NemotronVisualPlan:
    """Exact Megatron visual transform selected during pre-encoding."""

    message_index: int
    media_index: int
    modality: str
    patch_sizes: tuple[tuple[int, int], ...]
    embedding_widths: tuple[int, ...]
    frame_timestamps: tuple[float, ...] = ()

    @property
    def num_embeddings(self) -> int:
        return sum(self.embedding_widths)

    @property
    def num_frames(self) -> int:
        return len(self.frame_timestamps) if self.modality == "video" else 1


@edataclass
class NemotronEncodedSFTSample(EncodedSFTSample):
    """Encoded sample that retains the visual plan used for length selection."""

    visual_plans: tuple[NemotronVisualPlan, ...]


@dataclass(frozen=True)
class _VideoSelection:
    timestamps: tuple[float, ...]
    aug_scale_frames_up: float


@dataclass(frozen=True)
class _VisualFrameSpec:
    occurrence_index: int
    width: int
    height: int
    is_video: bool
    aug_scale_frames_up: float = 1.0


def _radio_token_count(*, height: int, width: int, patch_dim: int) -> int:
    """Return RADIO features after the fixed 2x2 spatial pixel shuffle.

    This matches ``_pixel_shuffled_token_count`` in the Apache-2.0
    Megatron-Bridge source at revision 8c46dc4259080c510b7455f43e836fdff222c5d3.
    """
    if patch_dim <= 0:
        raise ValueError("patch_dim must be greater than zero.")
    if height % patch_dim or width % patch_dim:
        raise ValueError(
            f"Processed image {height}x{width} is not divisible by "
            f"patch_dim={patch_dim}."
        )
    patch_rows = height // patch_dim
    patch_columns = width // patch_dim
    if patch_rows % 2 or patch_columns % 2:
        raise ValueError(
            f"Processed image {height}x{width} produces a patch grid that is "
            "not divisible by the 2x2 RADIO pixel shuffle."
        )
    return patch_rows * patch_columns // 4


def _video_prompt(
    ref: MediaRef,
    *,
    temporal_patch_size: int,
    frame_timestamps: Sequence[float] | None = None,
) -> tuple[str, int]:
    """Build the Megatron version-2 temporal prompt for sampled frames."""
    if frame_timestamps is None:
        frame_count = _required_metadata_int(ref, "sampled_num_frames")
        sampled_fps = _required_metadata_float(ref, "sampled_fps")
        frame_timestamps = tuple(
            frame_index / sampled_fps for frame_index in range(frame_count)
        )
    else:
        frame_timestamps = tuple(float(value) for value in frame_timestamps)
        frame_count = len(frame_timestamps)
    tubelet_count = frame_count // temporal_patch_size
    lines = ["This is a video:"]
    for frame_start in range(0, frame_count, temporal_patch_size):
        frame_end = min(frame_start + temporal_patch_size, frame_count)
        timestamps = [
            f"{'Frame' if frame_index == frame_start else 'frame'} "
            f"{frame_index + 1} sampled at {frame_timestamps[frame_index]:.2f} seconds"
            for frame_index in range(frame_start, frame_end)
        ]
        lines.append(" and ".join(timestamps) + f": {COMPACT_IMAGE_PLACEHOLDER}")
    return "\n".join(lines) + "\n", tubelet_count


def _clean_thinking_trace(match: re.Match[str], *, ultra_format: bool) -> str:
    content = match.group(1).strip()
    if not content:
        return "<think></think>"
    closing_prefix = "" if ultra_format else "\n"
    return f"<think>\n{content}{closing_prefix}</think>"


def _normalize_assistant_thinking(
    message: dict[str, Any],
    *,
    sample: CanonicalSFTSample,
    message_index: int,
    prompt_format: str,
    thinking_trace_format: str,
    relax_thinking_trace_check: bool,
) -> None:
    if (
        message["role"] != "assistant"
        or sample.__subflavors__.get("skip_chat_template", False)
        or relax_thinking_trace_check
    ):
        return
    content = message["content"]
    if any(part["type"] != "text" for part in content):
        raise ValueError("Nemotron assistant turns cannot contain visual media.")
    text = "".join(str(part.get("text", "")) for part in content)
    start_count = text.count("<think>")
    end_count = text.count("</think>")
    if start_count == 0 and end_count == 0:
        prefix = (
            "<think></think>"
            if prompt_format == "nemotron6-moe"
            else "<think></think>\n\n"
        )
        message["content"] = [{"type": "text", "text": prefix + text.strip()}]
        return
    error: str | None = None
    if start_count != 1 or end_count != 1:
        error = (
            "Nemotron assistant turns require exactly one matched pair of <think> "
            "tags."
        )
    elif text.find("<think>") > text.find("</think>"):
        error = "Nemotron assistant </think> appears before <think>."
    if error is not None:
        subflavors = getattr(sample, "__subflavors__", {})
        logger.error(
            "[NEMOTRON_THINKING_TRACE_DIAG]\n"
            "reason=%s\n"
            "sample_key=%r\n"
            "subflavors=%r\n"
            "assistant_message_index=%d\n"
            "think_start_count=%d\n"
            "think_end_count=%d\n"
            "trajectory=\n%s",
            error,
            sample.__key__,
            subflavors,
            message_index,
            start_count,
            end_count,
            json.dumps(sample.messages, ensure_ascii=False, indent=2, default=repr),
        )
        raise ValueError(
            f"{error} Sample key: {sample.__key__!r}; "
            f"subflavors: {subflavors!r}; assistant message index: {message_index}."
        )

    ultra_format = thinking_trace_format == "ultra"
    normalized = re.sub(
        r"<think>(.*?)</think>",
        lambda match: _clean_thinking_trace(match, ultra_format=ultra_format),
        text,
        flags=re.DOTALL,
    )
    if ultra_format:
        separator = "</think>"
    elif prompt_format == "nemotron6-moe":
        separator = "</think>\n"
    else:
        separator = "</think>\n\n"
    normalized = re.sub(r"</think>\s*", separator, normalized)
    message["content"] = [{"type": "text", "text": normalized}]


def _render_compact_messages(
    sample: CanonicalSFTSample,
    *,
    temporal_patch_size: int,
    prompt_format: str,
    thinking_trace_format: str,
    relax_thinking_trace_check: bool,
    video_timestamps: dict[int, tuple[float, ...]] | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[int, MediaRef]]]:
    # materialize=False; see _render_omni_messages. This renderer discards the
    # payload the same way.
    messages = _normalize_messages(sample, materialize=False)
    occurrences: list[tuple[int, MediaRef]] = []
    media_index = 0
    for message_index, message in enumerate(messages):
        _normalize_assistant_thinking(
            message,
            sample=sample,
            message_index=message_index,
            prompt_format=prompt_format,
            thinking_trace_format=thinking_trace_format,
            relax_thinking_trace_check=relax_thinking_trace_check,
        )
        content = message["content"]
        rendered_content: list[dict[str, str]] = []
        for part in content:
            part_type = part["type"]
            if part_type == "text":
                text = str(part.get("text", ""))
                validate_text_content(text, sample_key=sample.__key__)
                rendered_content.append({"type": "text", "text": text})
                continue
            if part_type not in {"image", "video", "video_frame"}:
                raise ValueError(
                    f"Nemotron visual encoding does not support {part_type!r} media."
                )
            if media_index >= len(sample.media):
                raise ValueError(
                    "Nemotron message media exceeds the canonical media list."
                )
            ref = sample.media[media_index]
            if ref.modality != part_type:
                raise ValueError(
                    f"Nemotron media order mismatch: expected {part_type!r}, got "
                    f"{ref.modality!r}."
                )
            if ref.modality in {"image", "video_frame"}:
                placeholder = COMPACT_IMAGE_PLACEHOLDER
            else:
                placeholder, _ = _video_prompt(
                    ref,
                    temporal_patch_size=temporal_patch_size,
                    frame_timestamps=(video_timestamps or {}).get(media_index),
                )
            rendered_content.append({"type": "text", "text": placeholder})
            occurrences.append((message_index, ref))
            media_index += 1
        message["content"] = rendered_content
    if media_index != len(sample.media):
        raise ValueError(
            f"Nemotron messages reference {media_index}/{len(sample.media)} media items."
        )
    return messages, occurrences


def _token_id(tokenizer: Any, token: str) -> int:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        raise TypeError("Nemotron tokenizer must expose convert_tokens_to_ids().")
    token_id = convert(token)
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if (
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        or token_id == unknown_id
    ):
        raise ValueError(f"Nemotron tokenizer does not define token {token!r}.")
    return token_id


def _tokenize_nemotron_sample(
    sample: CanonicalSFTSample,
    messages: list[dict[str, Any]],
    *,
    processor: Any,
    prompt_format: str,
) -> list[dict[str, Any]]:
    if sample.tools is not None:
        raise ValueError(
            "The Megatron Nemotron training tokenizer does not pass a tools "
            "argument to the chat template."
        )
    explicit_loss = sample.__subflavors__.get("loss_mask_mode") == (
        "explicit_assistant_turns"
    )
    assistant_turn_loss: list[bool] | None = [] if explicit_loss else None
    for message in messages:
        loss = message.get("train_on_message")
        if message["role"] == "assistant":
            if explicit_loss:
                if type(loss) is not bool:
                    raise ValueError(
                        "Explicit Nemotron assistant loss requires a boolean "
                        "train_on_message on every assistant turn."
                    )
                assistant_turn_loss.append(loss)
        elif loss not in (None, False):
            raise ValueError(
                "Nemotron train_on_message is only valid on assistant turns."
            )

    return tokenize_nemotron_conversation(
        messages,
        processor=processor,
        prompt_format=prompt_format,
        skip_chat_template=bool(
            sample.__subflavors__.get("skip_chat_template", False)
        ),
        train_only_on_last_assistant_turn=bool(
            sample.__subflavors__.get("train_only_on_last_assistant_turn", False)
        ),
        tool_response_as_turn_boundary=bool(
            sample.__subflavors__.get("tool_response_as_turn_boundary", False)
        ),
        assistant_turn_loss=assistant_turn_loss,
        complete_conversation=sample.messages,
    )


def _expand_visual_placeholders(
    message_log: list[dict[str, Any]],
    occurrences: Sequence[tuple[int, tuple[int, ...]]],
    *,
    image_token_id: int,
) -> None:
    """Replace each compact visual token with its measured feature width."""
    widths_by_message: defaultdict[int, list[int]] = defaultdict(list)
    for message_index, widths in occurrences:
        widths_by_message[message_index].extend(widths)

    for message_index, widths in widths_by_message.items():
        token_ids = message_log[message_index].get("token_ids")
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1:
            raise ValueError(
                "Nemotron tokenized messages require one-dimensional token_ids."
            )
        token_loss_mask = message_log[message_index].get("token_loss_mask")
        if (
            not isinstance(token_loss_mask, torch.Tensor)
            or token_loss_mask.ndim != 1
            or token_loss_mask.shape != token_ids.shape
        ):
            raise ValueError(
                "Nemotron tokenized messages require token_loss_mask aligned "
                "with token_ids."
            )
        # Positions are recorded at splice time, so a literal "<image>" or even
        # "<img><image></img>" written in prose is never mistaken for one.
        placeholder_positions = list(
            message_log[message_index]["visual_placeholder_positions"]
        )
        if len(placeholder_positions) != len(widths):
            raise ValueError(
                f"Nemotron message {message_index} has "
                f"{len(placeholder_positions)} compact visual placeholders for "
                f"{len(widths)} projected visual rows."
            )

        token_pieces: list[torch.Tensor] = []
        mask_pieces: list[torch.Tensor] = []
        start = 0
        for position, width in zip(placeholder_positions, widths, strict=True):
            if width <= 0:
                raise ValueError("Nemotron visual expansion widths must be positive.")
            token_pieces.append(token_ids[start:position])
            token_pieces.append(
                torch.full(
                    (width,),
                    image_token_id,
                    dtype=token_ids.dtype,
                    device=token_ids.device,
                )
            )
            mask_pieces.append(token_loss_mask[start:position])
            mask_pieces.append(token_loss_mask[position].expand(width))
            start = position + 1
        token_pieces.append(token_ids[start:])
        mask_pieces.append(token_loss_mask[start:])
        message_log[message_index]["token_ids"] = torch.cat(token_pieces)
        message_log[message_index]["token_loss_mask"] = torch.cat(mask_pieces)


def _truncate_message_log(
    message_log: list[dict[str, Any]],
    *,
    max_text_tokens: int,
    sample: CanonicalSFTSample,
) -> tuple[int, int]:
    """Truncate the flat token stream and its precomputed loss mask together."""
    original_length = sum(len(message["token_ids"]) for message in message_log)
    if max_text_tokens <= 0:
        raise ValueError(
            f"Nemotron sample {sample.__key__!r} reserves the full sequence for "
            "visual embeddings and leaves no room for text tokens."
        )
    if original_length <= max_text_tokens:
        return original_length, original_length

    remaining = max_text_tokens
    for message in message_log:
        token_ids = message.get("token_ids")
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1:
            raise ValueError(
                "Nemotron tokenized messages require one-dimensional token_ids."
            )
        token_loss_mask = message.get("token_loss_mask")
        if (
            not isinstance(token_loss_mask, torch.Tensor)
            or token_loss_mask.ndim != 1
            or token_loss_mask.shape != token_ids.shape
        ):
            raise ValueError(
                "Nemotron tokenized messages require token_loss_mask aligned "
                "with token_ids."
            )
        kept = min(len(token_ids), remaining)
        message["token_ids"] = token_ids[:kept]
        message["token_loss_mask"] = token_loss_mask[:kept]
        positions = message.get("visual_placeholder_positions")
        if positions is not None:
            # Truncation keeps a prefix, so a surviving position keeps its
            # index; only placeholders past the cut are dropped.
            message["visual_placeholder_positions"] = [
                position for position in positions if position < kept
            ]
        remaining -= kept

    truncated_length = max_text_tokens - remaining
    has_trainable_tokens = any(
        bool(message["token_loss_mask"].any()) for message in message_log
    )
    if not has_trainable_tokens:
        raise NoTrainableTokensError(
            "All trainable assistant tokens were removed by Nemotron sequence "
            f"truncation for sample {sample.__key__!r}; subflavors: "
            f"{sample.__subflavors__!r}; original text length: {original_length}; "
            f"truncated text length: {truncated_length}."
        )
    return original_length, truncated_length


class _NemotronVisualProcessorAdapter:
    """Split Nemotron visual cost prediction from media processing.

    Image rows may provide raw ``height`` and ``width`` metadata. Prepared image
    rows may instead provide ``processed_height``, ``processed_width``, and
    ``num_tiles``. Video rows must provide ``processed_height``,
    ``processed_width``, ``sampled_num_frames``, and ``sampled_fps`` metadata.
    """

    def __init__(
        self,
        *,
        processor: Any,
        max_sequence_length: int,
        packing_sequence_length: int | None = None,
        patch_dim: int = 16,
        temporal_patch_size: int = 2,
        prompt_format: str = "nemotron-h-5p5-reasoning",
        thinking_trace_format: str = "normalized",
        relax_thinking_trace_check: bool = False,
        video_min_num_frames: int = 8,
        video_max_num_frames: int = 32,
        video_default_fps: int = 2,
        video_frame_temporal_jitter: bool = False,
        video_aug_scale_frames_up: int | None = None,
        video_aug_scale_resolution_up: int | None = None,
        video_aug_scale_resolution_only: bool = False,
        allow_large_videos: bool = False,
        tiling_augment_prob: float = 0.4,
        video_decode_thread_count: int = 8,
        add_bos: bool = False,
        add_eos: bool = False,
        add_generation_prompt: bool = False,
    ) -> None:
        if patch_dim <= 0:
            raise ValueError("patch_dim must be greater than zero.")
        if temporal_patch_size <= 0:
            raise ValueError("temporal_patch_size must be greater than zero.")
        if video_min_num_frames < temporal_patch_size:
            raise ValueError(
                "video_min_num_frames must be at least temporal_patch_size."
            )
        if video_max_num_frames < video_min_num_frames:
            raise ValueError(
                "video_max_num_frames must be at least video_min_num_frames."
            )
        if video_max_num_frames % temporal_patch_size:
            raise ValueError(
                "video_max_num_frames must be divisible by temporal_patch_size."
            )
        if video_default_fps <= 0:
            raise ValueError("video_default_fps must be greater than zero.")
        if video_decode_thread_count < 0:
            raise ValueError("video_decode_thread_count must be non-negative.")
        if prompt_format not in {"nemotron-h-5p5-reasoning", "nemotron6-moe"}:
            raise ValueError(f"Unsupported Nemotron prompt format {prompt_format!r}.")
        # "default" is the legacy nemo-rl spelling of the reference's "normalized";
        # both select the same non-ultra newline behavior.
        if thinking_trace_format not in {"default", "normalized", "ultra"}:
            raise ValueError(
                f"Unsupported Nemotron thinking trace format {thinking_trace_format!r}."
            )
        # Megatron budgets tile planning against decoder_seq_length but truncates
        # against packing_seq_length, falling back to decoder_seq_length when unset.
        if packing_sequence_length is not None and packing_sequence_length <= 0:
            raise ValueError("packing_sequence_length must be greater than zero.")
        if (
            packing_sequence_length is not None
            and packing_sequence_length > max_sequence_length
        ):
            raise ValueError(
                "packing_sequence_length must not exceed max_sequence_length."
            )
        if not hasattr(processor, "apply_chat_template") or not hasattr(
            processor, "tokenizer"
        ):
            raise TypeError("Nemotron visual SFT requires a Hugging Face processor.")
        if add_bos or add_eos or add_generation_prompt:
            raise ValueError(
                "Megatron Nemotron training tokenization requires add_bos, "
                "add_eos, and add_generation_prompt to be false."
            )
        self.processor = processor
        self.max_sequence_length = max_sequence_length
        self.packing_sequence_length = packing_sequence_length or max_sequence_length
        self.patch_dim = patch_dim
        self.temporal_patch_size = temporal_patch_size
        self.prompt_format = prompt_format
        self.thinking_trace_format = thinking_trace_format
        self.relax_thinking_trace_check = relax_thinking_trace_check
        self.video_min_num_frames = video_min_num_frames
        self.video_max_num_frames = video_max_num_frames
        self.video_default_fps = video_default_fps
        self.video_frame_temporal_jitter = video_frame_temporal_jitter
        self.video_aug_scale_frames_up = video_aug_scale_frames_up
        self.video_aug_scale_resolution_up = video_aug_scale_resolution_up
        self.video_aug_scale_resolution_only = video_aug_scale_resolution_only
        self.allow_large_videos = allow_large_videos
        self.tiling_augment_prob = tiling_augment_prob
        # Decode-speed knob only: it does not change which frames are selected,
        # so it is deliberately left out of the fingerprint below to avoid
        # invalidating checkpoint resume.
        self.video_decode_thread_count = video_decode_thread_count
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.add_generation_prompt = add_generation_prompt
        fingerprint_data = {
            "processor_class": type(processor).__name__,
            "processor_name": getattr(processor, "name_or_path", None),
            "patch_dim": patch_dim,
            "temporal_patch_size": temporal_patch_size,
            "prompt_format": prompt_format,
            "thinking_trace_format": thinking_trace_format,
            "relax_thinking_trace_check": relax_thinking_trace_check,
            "video_min_num_frames": video_min_num_frames,
            "video_max_num_frames": video_max_num_frames,
            "video_default_fps": video_default_fps,
            "video_frame_temporal_jitter": video_frame_temporal_jitter,
            "video_aug_scale_frames_up": video_aug_scale_frames_up,
            "video_aug_scale_resolution_up": video_aug_scale_resolution_up,
            "video_aug_scale_resolution_only": video_aug_scale_resolution_only,
            "allow_large_videos": allow_large_videos,
            "tiling_augment_prob": tiling_augment_prob,
            "max_sequence_length": max_sequence_length,
            "packing_sequence_length": self.packing_sequence_length,
            "add_bos": add_bos,
            "add_eos": add_eos,
            "add_generation_prompt": add_generation_prompt,
        }
        encoded = json.dumps(fingerprint_data, sort_keys=True, default=str).encode()
        self._fingerprint = hashlib.sha256(encoded).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def encode(self, sample: CanonicalSFTSample) -> EncodedSFTSample:
        """Provide the generic adapter method as the lazy pre-encode operation."""
        return self.preencode(sample)

    def _select_video_frames(self, ref: MediaRef) -> _VideoSelection:
        """Select frame timestamps with the Megatron reference algorithm."""
        duration = _required_metadata_float(ref, "video_duration")
        num_source_frames = _required_metadata_int(ref, "video_num_frames")
        start_time = _optional_metadata_float(ref, "start_time")
        end_time = _optional_metadata_float(ref, "end_time")
        clip_start = 0.0 if start_time is None else start_time
        if start_time is not None:
            duration = (
                end_time - start_time
                if end_time is not None
                else duration - start_time
            )
        elif end_time is not None:
            duration = end_time
        if duration <= 0:
            raise ValueError(
                f"Nemotron video clip duration must be positive; got {duration}."
            )

        scale_values = [1.0]
        if self.video_aug_scale_frames_up is not None:
            scale_values.extend(
                float(value)
                for value in range(2, self.video_aug_scale_frames_up + 1)
            )
        if self.video_aug_scale_resolution_up is not None:
            scale_values.extend(
                1.0 / value
                for value in range(2, self.video_aug_scale_resolution_up + 1)
            )
        aug_scale_frames_up = random.choice(scale_values)

        if num_source_frames < self.video_min_num_frames:
            sample_num_frames = num_source_frames
            aug_scale_frames_up = 1.0
            effective_max_num_frames = self.video_max_num_frames
        else:
            if self.video_aug_scale_resolution_only:
                effective_max_num_frames = self.video_max_num_frames
                effective_fps = self.video_default_fps
            else:
                effective_max_num_frames = max(
                    self.video_min_num_frames,
                    int(self.video_max_num_frames * aug_scale_frames_up),
                )
                effective_fps = max(
                    1,
                    int(self.video_default_fps * aug_scale_frames_up),
                )
            sample_num_frames = min(
                max(int(effective_fps * duration), self.video_min_num_frames),
                effective_max_num_frames,
            )

        if sample_num_frames % self.temporal_patch_size:
            rounded_down = (
                sample_num_frames // self.temporal_patch_size
            ) * self.temporal_patch_size
            rounded_up = rounded_down + self.temporal_patch_size
            if (
                rounded_up <= num_source_frames
                and rounded_up <= effective_max_num_frames
            ):
                sample_num_frames = rounded_up
            else:
                sample_num_frames = max(self.temporal_patch_size, rounded_down)
        if sample_num_frames % self.temporal_patch_size:
            raise ValueError(
                f"Sampled video frame count {sample_num_frames} is not divisible "
                f"by temporal_patch_size={self.temporal_patch_size}."
            )

        segment_size = float(duration - 1) / sample_num_frames
        timestamps = segment_size * (
            torch.arange(sample_num_frames, dtype=torch.float32) + 0.5
        )
        if self.video_frame_temporal_jitter:
            jitter_size = segment_size * 0.5
            timestamps += (
                torch.rand(len(timestamps), dtype=torch.float32)
                * (jitter_size * 2)
                - jitter_size
            )
            timestamps = torch.clamp(timestamps, 0, duration)
        timestamps += clip_start
        return _VideoSelection(
            timestamps=tuple(float(value) for value in timestamps.tolist()),
            aug_scale_frames_up=aug_scale_frames_up,
        )

    def _video_selections(
        self, sample: CanonicalSFTSample
    ) -> dict[int, _VideoSelection]:
        selections: dict[int, _VideoSelection] = {}
        # The reference reads --allow-large-videos off args, not off the sample.
        # Keep honoring the per-sample subflavor so existing datasets still work.
        allow_large_videos = self.allow_large_videos or bool(
            sample.__subflavors__.get("allow_large_videos", False)
        )
        for media_index, ref in enumerate(sample.media):
            if ref.modality != "video":
                continue
            duration = _required_metadata_float(ref, "video_duration")
            start_time = _optional_metadata_float(ref, "start_time")
            end_time = _optional_metadata_float(ref, "end_time")
            clip_duration = (
                (end_time if end_time is not None else duration)
                - (start_time if start_time is not None else 0.0)
            )
            if not allow_large_videos and clip_duration > 600:
                raise ValueError(f"Video is too large: {ref.value}")
            selections[media_index] = self._select_video_frames(ref)
        return selections

    def _plan_visual_occurrences(
        self,
        occurrences: Sequence[tuple[int, MediaRef]],
        *,
        video_selections: dict[int, _VideoSelection],
        num_tokens_available: int,
        data_augment: bool,
        media_indexes: Sequence[int] | None = None,
    ) -> tuple[NemotronVisualPlan, ...]:
        """Port Megatron dynamic-resolution planning and joint budgeting."""
        image_processor = getattr(self.processor, "image_processor", None)
        min_num_patches = getattr(image_processor, "min_num_patches", None)
        max_num_patches = getattr(image_processor, "max_num_patches", None)
        downsample_factor = getattr(image_processor, "_downsample_factor", None)
        video_target_patches = getattr(
            image_processor, "video_target_num_patches", None
        )
        video_maintain_aspect_ratio = bool(
            getattr(image_processor, "video_maintain_aspect_ratio", False)
        )
        for name, value in (
            ("min_num_patches", min_num_patches),
            ("max_num_patches", max_num_patches),
            ("_downsample_factor", downsample_factor),
            ("video_target_num_patches", video_target_patches),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(
                    "Nemotron image processor must provide a positive integer "
                    f"{name}; got {value!r}."
                )
        processor_patch_size = getattr(image_processor, "patch_size", None)
        if processor_patch_size != self.patch_dim:
            raise ValueError(
                "Nemotron image processor patch size does not match task encoder: "
                f"{processor_patch_size!r} != {self.patch_dim}."
            )

        frames: list[_VisualFrameSpec] = []
        for occurrence_index, (_, ref) in enumerate(occurrences):
            media_index = (
                occurrence_index
                if media_indexes is None
                else media_indexes[occurrence_index]
            )
            if ref.modality == "image":
                frames.append(
                    _VisualFrameSpec(
                        occurrence_index=occurrence_index,
                        width=_required_metadata_int(ref, "width"),
                        height=_required_metadata_int(ref, "height"),
                        is_video=False,
                    )
                )
                continue
            if ref.modality == "video_frame":
                timestamp = _optional_metadata_float(ref, "timestamp")
                if timestamp is None:
                    raise ValueError(
                        "Nemotron video_frame media requires numeric timestamp "
                        "metadata."
                    )
                aug_scale = _optional_metadata_float(
                    ref, "video_aug_scale_frames_up"
                )
                frames.append(
                    _VisualFrameSpec(
                        occurrence_index=occurrence_index,
                        width=_required_metadata_int(ref, "video_width"),
                        height=_required_metadata_int(ref, "video_height"),
                        is_video=True,
                        aug_scale_frames_up=(
                            1.0 if aug_scale is None else aug_scale
                        ),
                    )
                )
                continue
            if ref.modality != "video":
                continue
            selection = video_selections[media_index]
            frames.extend(
                _VisualFrameSpec(
                    occurrence_index=occurrence_index,
                    width=_required_metadata_int(ref, "video_width"),
                    height=_required_metadata_int(ref, "video_height"),
                    is_video=True,
                    aug_scale_frames_up=selection.aug_scale_frames_up,
                )
                for _ in selection.timestamps
            )
        if not frames:
            return ()

        patch_budget = num_tokens_available * downsample_factor**2
        num_images = sum(not frame.is_video for frame in frames)
        num_video_frames = len(frames) - num_images
        if self.temporal_patch_size > 1 and num_video_frames:
            if not num_images:
                patch_budget *= self.temporal_patch_size
            else:
                patch_budget = int(
                    patch_budget
                    * (
                        num_images / len(frames)
                        + num_video_frames
                        / len(frames)
                        * self.temporal_patch_size
                    )
                )
        patch_budget = max(patch_budget, min_num_patches * len(frames))
        per_frame_budgets = [
            max(min(patch_budget, max_num_patches), min_num_patches)
            for _ in frames
        ]

        def process_frame(
            frame: _VisualFrameSpec, available: int
        ) -> tuple[tuple[int, int], int]:
            closest_h = round(frame.height / self.patch_dim + 0.5)
            closest_w = round(frame.width / self.patch_dim + 0.5)
            factor = min(math.sqrt(available / (closest_h * closest_w)), 1.0)
            target_h = math.floor(factor * closest_h)
            target_w = math.floor(factor * closest_w)
            if (
                available > min_num_patches
                and target_h * target_w < min_num_patches
            ):
                up_factor = math.sqrt(
                    min_num_patches / max(target_h * target_w, 1)
                )
                up_h = math.ceil(up_factor * target_h)
                up_w = math.ceil(up_factor * target_w)
                if available > up_h * up_w:
                    target_h, target_w = up_h, up_w

            if data_augment and random.random() < self.tiling_augment_prob:
                minimum_side_patches = 32
                if random.random() < 0.5:
                    if target_w > minimum_side_patches and target_h > minimum_side_patches:
                        if random.random() < 0.5:
                            target_w -= minimum_side_patches
                        else:
                            target_h -= minimum_side_patches
                    elif target_w > minimum_side_patches:
                        target_w -= minimum_side_patches
                    elif target_h > minimum_side_patches:
                        target_h -= minimum_side_patches
                elif target_w * target_h < available:
                    if random.random() < 0.5:
                        target_w += minimum_side_patches
                    else:
                        target_h += minimum_side_patches

            divisor = downsample_factor
            for dimension in ("height", "width"):
                value = target_h if dimension == "height" else target_w
                other = target_w if dimension == "height" else target_h
                remainder = value % divisor
                if remainder:
                    increase = divisor - remainder
                    value = (
                        value + increase
                        if (value + increase) * other <= available
                        else max(divisor, value - remainder)
                    )
                if dimension == "height":
                    target_h = value
                else:
                    target_w = value

            if frame.is_video:
                target = int(video_target_patches / frame.aug_scale_frames_up)
                if video_maintain_aspect_ratio:
                    aspect = frame.width / frame.height
                    target_h = max(1, round(math.sqrt(target / aspect)))
                    target_w = max(1, round(math.sqrt(target * aspect)))
                else:
                    side = int(math.sqrt(target))
                    if side * side != target and frame.aug_scale_frames_up == 1.0:
                        raise ValueError(
                            "video_target_num_patches must be square when aspect "
                            "ratio preservation is disabled."
                        )
                    target_h = target_w = max(1, side)
                h_down = target_h - target_h % divisor
                w_down = target_w - target_w % divisor
                h_up = h_down if target_h % divisor == 0 else h_down + divisor
                w_up = w_down if target_w % divisor == 0 else w_down + divisor
                if h_up * w_up <= target:
                    target_h, target_w = h_up, w_up
                else:
                    target_h = max(divisor, h_down)
                    target_w = max(divisor, w_down)
            return (target_w, target_h), target_w * target_h

        patch_sizes: list[tuple[int, int]] = []
        for _ in range(10):
            results = [
                process_frame(frame, available)
                for frame, available in zip(frames, per_frame_budgets, strict=True)
            ]
            patch_sizes = [result[0] for result in results]
            counts = [result[1] for result in results]
            total = sum(counts)
            if total <= patch_budget:
                break
            scale = patch_budget / total
            scaled = [
                max(min_num_patches, int(count * scale)) for count in counts
            ]
            per_frame_budgets = (
                scaled
                if any(
                    new < old
                    for new, old in zip(scaled, per_frame_budgets, strict=True)
                )
                else [min_num_patches] * len(frames)
            )

        frame_sizes_by_occurrence: defaultdict[int, list[tuple[int, int]]] = (
            defaultdict(list)
        )
        for frame, size in zip(frames, patch_sizes, strict=True):
            frame_sizes_by_occurrence[frame.occurrence_index].append(size)

        plans: list[NemotronVisualPlan] = []
        for occurrence_index, (message_index, ref) in enumerate(occurrences):
            media_index = (
                occurrence_index
                if media_indexes is None
                else media_indexes[occurrence_index]
            )
            sizes = tuple(frame_sizes_by_occurrence[occurrence_index])
            if ref.modality == "image":
                width, height = sizes[0]
                embedding_widths = (
                    _radio_token_count(
                        height=height * self.patch_dim,
                        width=width * self.patch_dim,
                        patch_dim=self.patch_dim,
                    ),
                )
                timestamps: tuple[float, ...] = ()
            elif ref.modality == "video":
                selection = video_selections[media_index]
                timestamps = selection.timestamps
                frame_embeddings = [
                    _radio_token_count(
                        height=height * self.patch_dim,
                        width=width * self.patch_dim,
                        patch_dim=self.patch_dim,
                    )
                    for width, height in sizes
                ]
                embedding_widths = tuple(
                    frame_embeddings[index]
                    for index in range(0, len(frame_embeddings), self.temporal_patch_size)
                )
            else:
                width, height = sizes[0]
                embedding_widths = (
                    _radio_token_count(
                        height=height * self.patch_dim,
                        width=width * self.patch_dim,
                        patch_dim=self.patch_dim,
                    ),
                )
                timestamp = _optional_metadata_float(ref, "timestamp")
                assert timestamp is not None
                timestamps = (timestamp,)
            plans.append(
                NemotronVisualPlan(
                    message_index=message_index,
                    media_index=media_index,
                    modality=ref.modality,
                    patch_sizes=sizes,
                    embedding_widths=embedding_widths,
                    frame_timestamps=timestamps,
                )
            )
        return tuple(plans)

    def preencode(self, sample: CanonicalSFTSample) -> EncodedSFTSample:
        """Tokenize compact placeholders and retain the exact visual plan."""
        video_selections = self._video_selections(sample)
        messages, occurrences = _render_compact_messages(
            sample,
            temporal_patch_size=self.temporal_patch_size,
            prompt_format=self.prompt_format,
            thinking_trace_format=self.thinking_trace_format,
            relax_thinking_trace_check=self.relax_thinking_trace_check,
            video_timestamps={
                index: selection.timestamps
                for index, selection in video_selections.items()
            },
        )
        message_log = _tokenize_nemotron_sample(
            sample,
            messages,
            processor=self.processor,
            prompt_format=self.prompt_format,
        )
        flat_occurrences = [(0, ref) for _, ref in occurrences]
        length = sum(len(message["token_ids"]) for message in message_log)
        visual_plans = self._plan_visual_occurrences(
            flat_occurrences,
            video_selections=video_selections,
            num_tokens_available=self.max_sequence_length - length - 4,
            data_augment=bool(sample.__subflavors__.get("data_augment", False)),
        )
        visual_embeddings = sum(plan.num_embeddings for plan in visual_plans)
        compact_placeholders = sum(
            len(plan.embedding_widths) for plan in visual_plans
        )
        max_text_tokens = (
            self.packing_sequence_length - visual_embeddings + compact_placeholders
        )
        image_tokens_before_truncation = sum(
            len(message["visual_placeholder_positions"])
            for message in message_log
        )
        # Separate the two ways the placeholder count can disagree with the plan.
        # Conflating them blames truncation for samples that were never truncated.
        #
        # A literal "<image>" in prose no longer reaches here: placeholders are
        # counted by splice position, and validate_text_content rejects the row
        # at render time. So this fires only on a renderer/planner disagreement.
        if image_tokens_before_truncation != compact_placeholders:
            raise ValueError(
                f"Nemotron sample {sample.__key__!r} rendered "
                f"{image_tokens_before_truncation} visual placeholders but "
                f"pre-encoding planned {compact_placeholders} from "
                f"{len(visual_plans)} visual media items."
            )
        original_length, length = _truncate_message_log(
            message_log,
            max_text_tokens=max_text_tokens,
            sample=sample,
        )
        remaining_placeholders = sum(
            len(message["visual_placeholder_positions"])
            for message in message_log
        )
        if remaining_placeholders != image_tokens_before_truncation:
            raise ValueError(
                f"Nemotron truncation removed visual placeholders from sample "
                f"{sample.__key__!r}: expected {image_tokens_before_truncation}, "
                f"found {remaining_placeholders}; original text length: "
                f"{original_length}; max text tokens: {max_text_tokens}."
            )
        packing_cost = length + visual_embeddings - compact_placeholders
        # The fingerprint alone. The Megatron reference packs its whole buffer
        # with one knapsack call and no grouping (task_encoder.py:1365), so any
        # extra key here only fragments packs.
        #
        # This used to add the model-input tensor names, which put samples WITH
        # media and samples WITHOUT media in separate partitions. Each
        # partition then got its own balanced_greedy_knapsack `+delta`
        # allocation. Measured on 19 video rows: 18 media samples gave 6 bins
        # and 1 text-only row gave a 7th, against the reference's 6.
        #
        # Dropping the names is safe because batch() never stacks media into a
        # batch tensor -- pixel_values/imgs_sizes/num_frames stay inside the
        # per-message dicts of message_log, so a pack that mixes media and
        # non-media sources is the ragged list it already was.
        #
        # The fingerprint stays. It hashes the encoder settings, so samples
        # encoded under different settings still cannot share a pack.
        return NemotronEncodedSFTSample.derive_from(
            sample,
            message_log=message_log,
            length=length,
            packing_cost=packing_cost,
            loss_multiplier=1.0,
            group_key=(self.fingerprint,),
            sample_key=sample.__key__,
            pending_sample=sample,
            visual_plans=visual_plans,
        )

    def _decode_video_clips(
        self, av_decoder: Any, targets: Sequence[float]
    ) -> list[Any]:
        """Decode the selected frames, optionally with worker-local ffmpeg threads.

        Energon's ``initialize_av_container`` pins every codec context to
        ``thread_type="NONE"`` so that no threads exist before the torch loader
        forks its workers. That leaves each video decoding on a single core,
        which is the dominant cost when a step has to fetch video. Rebinding
        ``av_open`` for the duration of one ``get_clips`` call re-enables FRAME
        threading inside the already-forked worker, and Energon's own indexing,
        seeking, and frame choice are untouched -- the frames returned are the
        same ones, decoded faster.
        """
        if not targets:
            return []

        def get_clips() -> list[Any]:
            return list(
                av_decoder.get_clips(
                    video_clip_ranges=[
                        (timestamp, timestamp) for timestamp in targets
                    ],
                    video_unit="seconds",
                ).video_clips
            )

        thread_count = self.video_decode_thread_count
        if thread_count <= 0:
            return get_clips()

        decoder_module = importlib.import_module(AVDecoder.__module__)
        original_av_open = getattr(decoder_module, "av_open", None)
        if not callable(original_av_open):
            # Older/newer Energon builds may not expose the hook; decoding
            # unthreaded is correct, just slower.
            return get_clips()

        def threaded_av_open(*args: Any, **kwargs: Any) -> Any:
            container = original_av_open(*args, **kwargs)
            try:
                for media_stream in container.streams:
                    if media_stream.type != "video":
                        continue
                    codec_context = media_stream.codec_context
                    if codec_context is None:
                        continue
                    codec_context.thread_type = "FRAME"
                    codec_context.thread_count = thread_count
            except Exception:
                container.close()
                raise
            return container

        with _ENERGON_AV_OPEN_LOCK:
            decoder_module.av_open = threaded_av_open
            try:
                clips = get_clips()
            finally:
                decoder_module.av_open = original_av_open
        # Some codecs hold delayed frames under FRAME threading and return
        # short. Retry that one video through the exact unthreaded path.
        if len(clips) < len(targets):
            clips = get_clips()
        return clips

    def _process_media(
        self,
        ref: MediaRef,
        plan: NemotronVisualPlan,
        sample: CanonicalSFTSample,
    ) -> tuple[dict[str, PackedTensor], int]:
        if ref.modality != plan.modality:
            raise ValueError(
                f"Nemotron visual plan expected {plan.modality!r}, got "
                f"{ref.modality!r}."
            )
        if ref.modality == "image":
            value = (
                ref.value.get(sample)
                if callable(getattr(ref.value, "get", None))
                else ref.value
            )
            if isinstance(value, (tuple, list)):
                if not value:
                    raise ValueError("Nemotron image payload is empty.")
                value = value[0]
            if isinstance(value, (bytes, bytearray, memoryview)):
                with Image.open(BytesIO(bytes(value))) as image:
                    images = [image.convert("RGB")]
            elif isinstance(value, str):
                images = [resolve_to_image(value)]
            elif isinstance(value, Image.Image):
                images = [value]
            else:
                raise ValueError(
                    "Nemotron image payload must decode to PIL.Image; got "
                    f"{type(value).__name__}."
                )
        else:
            value = (
                ref.value.get(sample)
                if callable(getattr(ref.value, "get", None))
                else ref.value
            )
            if isinstance(value, (tuple, list)):
                if not value:
                    raise ValueError("Nemotron video payload is empty.")
                value = value[0]
            if isinstance(value, (bytes, bytearray, memoryview)):
                value = AVDecoder(BytesIO(bytes(value)))
            if callable(getattr(value, "get_clips", None)):
                clips = self._decode_video_clips(value, plan.frame_timestamps)
                if not clips:
                    raise ValueError("Unable to decode any selected video frame.")
                if len(clips) < len(plan.frame_timestamps):
                    clips.extend([clips[-1]] * (len(plan.frame_timestamps) - len(clips)))
                raw_frames = [
                    torch.as_tensor(clip)[0]
                    for clip in clips[: len(plan.frame_timestamps)]
                ]
            elif isinstance(value, Image.Image):
                images = [value.convert("RGB")]
                raw_frames = []
            else:
                tensor = torch.as_tensor(value)
                if ref.modality == "video_frame" and tensor.ndim == 3:
                    raw_frames = [tensor]
                elif tensor.ndim != 4:
                    raise ValueError(
                        "Nemotron video payload must be an AVDecoder or a frame "
                        f"tensor; got {tuple(tensor.shape)}."
                    )
                else:
                    frame_index = _optional_metadata_float(ref, "frame_index")
                    if ref.modality == "video_frame" and frame_index is not None:
                        indexes = (frame_index,)
                    else:
                        fps = _required_metadata_float(ref, "video_fps")
                        indexes = tuple(
                            timestamp * fps for timestamp in plan.frame_timestamps
                        )
                    raw_frames = [
                        tensor[
                            min(
                                max(round(index), 0),
                                tensor.shape[0] - 1,
                            )
                        ]
                        for index in indexes
                    ]
            if not isinstance(value, Image.Image):
                images = []
                for frame in raw_frames:
                    frame = torch.as_tensor(frame).detach().cpu()
                    if frame.ndim != 3:
                        raise ValueError(
                            "Nemotron decoded video frames must be three-dimensional."
                        )
                    images.append(_TO_PIL_IMAGE(frame).convert("RGB"))

        if len(images) != len(plan.patch_sizes):
            raise ValueError(
                f"Nemotron {ref.modality} plan has {len(plan.patch_sizes)} frames "
                f"but decoding produced {len(images)}."
            )

        mean = torch.tensor(_CLIP_PIXEL_MEAN, dtype=torch.float32)[:, None, None]
        std = torch.tensor(_CLIP_PIXEL_STD, dtype=torch.float32)[:, None, None]
        pixels: list[torch.Tensor] = []
        sizes: list[tuple[int, int]] = []
        for image, (patch_width, patch_height) in zip(
            images, plan.patch_sizes, strict=True
        ):
            width = patch_width * self.patch_dim
            height = patch_height * self.patch_dim
            resized = image.convert("RGB").resize((width, height))
            array = np.array(resized, copy=True)
            tensor = (
                torch.from_numpy(array)
                .permute(2, 0, 1)
                .to(dtype=torch.float32)
                .div_(255)
            )
            pixels.append((tensor - mean) / std)
            sizes.append((height, width))

        pixel_values = torch.stack(pixels)
        packed_inputs = {
            # This is where the model's images actually enter the packing
            # pipeline, so the configured mode has to be applied here -- not
            # only on the processor path in nemo_rl/data/processors.py.
            "pixel_values": PackedTensor(
                pixel_values,
                dim_to_pack=0,
                preprocess_mode="patchify",
                preprocess_kwargs={"patch_dim": self.patch_dim},
            ),
            "imgs_sizes": PackedTensor(
                torch.tensor(sizes, dtype=torch.int32),
                dim_to_pack=0,
            ),
            "num_frames": PackedTensor(
                torch.tensor([plan.num_frames], dtype=torch.long),
                dim_to_pack=0,
            ),
        }
        return packed_inputs, plan.num_embeddings

    def postencode(self, sample: EncodedSFTSample) -> EncodedSFTSample:
        """Apply the visual plan selected before packing."""
        pending_sample = sample.pending_sample
        if pending_sample is None:
            raise ValueError(
                f"Nemotron sample {sample.sample_key!r} has no pending canonical media."
            )
        visual_plans = getattr(sample, "visual_plans", None)
        if not isinstance(visual_plans, tuple):
            raise ValueError(
                f"Nemotron sample {sample.sample_key!r} has no saved visual plan."
            )
        message_log = deepcopy(sample.message_log)
        inputs_by_message: defaultdict[int, defaultdict[str, list[PackedTensor]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        visual_occurrences: list[tuple[int, tuple[int, ...]]] = []
        for plan in visual_plans:
            ref = pending_sample.media[plan.media_index]
            media_inputs, _ = self._process_media(ref, plan, pending_sample)
            for key, value in media_inputs.items():
                inputs_by_message[plan.message_index][key].append(value)
            visual_occurrences.append((plan.message_index, plan.embedding_widths))

        # Expansion is driven by the saved plan alone, exactly as apply_params()
        # is in the Megatron reference, so the expanded width cannot drift from
        # the packing cost predicted for it.
        image_token_id = _token_id(self.processor.tokenizer, "<image>")
        _expand_visual_placeholders(
            message_log,
            visual_occurrences,
            image_token_id=image_token_id,
        )
        expanded_length = sum(len(message["token_ids"]) for message in message_log)
        for message_index, keyed_inputs in inputs_by_message.items():
            message_log[message_index].update(
                {
                    key: PackedTensor.merge_segments(values)
                    for key, values in keyed_inputs.items()
                }
            )
        return sample.__class__.derive_from(
            sample,
            message_log=message_log,
            length=expanded_length,
            packing_cost=sample.packing_cost,
            loss_multiplier=sample.loss_multiplier,
            group_key=sample.group_key,
            sample_key=sample.sample_key,
            pending_sample=None,
            visual_plans=(),
        )


__all__ = [
    "COMPACT_IMAGE_PLACEHOLDER",
    "NemotronEncodedSFTSample",
    "NemotronVisualPlan",
]
