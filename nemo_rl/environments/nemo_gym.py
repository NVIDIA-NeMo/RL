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
import copy
import json
import math
import os
import subprocess
import sys
from collections import Counter
from collections.abc import AsyncGenerator
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, NotRequired, Optional, TypedDict
from urllib.parse import unquote, urlparse

import numpy as np
import ray
import torch
import torch.nn.functional as F
from PIL import Image
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from transformers import AutoConfig, PreTrainedTokenizerBase

from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    encode_images_in_examples,
    extract_multimodal_model_inputs,
    get_dim_to_pack_along,
    get_multimodal_keys_from_processor,
    resolve_to_image,
    uses_image_placeholder,
)
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_GYM_PORT_RANGE_HIGH,
    DEFAULT_GYM_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.models.policy import TokenizerConfig
from nemo_rl.utils.routed_experts_codec import decode_routed_experts
from nemo_rl.utils.timer import Timer
from nemo_rl.utils.venvs import create_local_venv_on_each_node

# Kept local (not imported from models.generation) so the gym actor stays free of
# generation-module imports. Must cover every name resolve_routed_experts_dtype
# can produce.
_ROUTED_EXPERTS_DTYPES = {
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
}

DEFAULT_INVALID_TOOL_CALL_PATTERNS = [
    "<tool_call>",
    "</tool_call>",
    "<function_call>",
    "</function_call>",
]
DEFAULT_THINKING_TAGS = ["<think>", "</think>"]


def _has_nan_generation_logprobs(result: dict) -> bool:
    """Return whether a postprocessed rollout contains NaN policy logprobs."""
    return any(
        message.get("generation_logprobs") is not None
        and torch.isnan(message["generation_logprobs"]).any()
        for message in result["message_log"]
    )


def get_nemo_gym_uv_cache_dir() -> str | None:
    """Return the uv cache directory inside a container, or None outside one.

    Inside a container (NRL_CONTAINER=1), returns the uv cache location so Gym
    stores its caches in the expected shared path. Returns None outside a
    container, meaning the caller should omit this arg and let Gym create the
    cache locally (the default when you may not be able to write to /opt).
    """
    if not os.environ.get("NRL_CONTAINER"):
        return None
    return subprocess.check_output(["uv", "cache", "dir"]).decode().strip()


def get_nemo_gym_venv_dir() -> str | None:
    """Return the NeMo Gym venv directory from NEMO_GYM_VENV_DIR, or None.

    Returns the value of NEMO_GYM_VENV_DIR if set, otherwise None. When None
    the caller should omit this arg and let Gym create venvs locally (the
    default when a container is not used since you may not be able to write
    to /opt).
    """
    return os.environ.get("NEMO_GYM_VENV_DIR")


class NemoGymConfig(TypedDict):
    model_name: str
    base_urls: List[str]
    initial_global_config_dict: Dict[str, Any]
    # Port range for Gym HTTP servers (head server + subprocess servers).
    # Defaults to DEFAULT_GYM_PORT_RANGE_LOW/HIGH (5000-5999) from
    # nemo_rl.distributed.virtual_cluster.  See the port layout there.
    port_range_low: NotRequired[int]
    port_range_high: NotRequired[int]
    invalid_tool_call_patterns: NotRequired[
        List[str] | None
    ]  # Substrings in assistant text content that indicate an invalid tool call
    thinking_tags: NotRequired[
        List[str] | None
    ]  # Thinking tags to check for malformed usage
    require_routed_experts: NotRequired[
        bool
    ]  # Require Gym output items to carry R3 routed_experts
    routed_experts_dtype: NotRequired[
        str
    ]  # Carry dtype name for routed_experts tensors ("int8"/"int16"/"int32"), resolved from the model's expert count
    # Forwarded from policy.tokenizer.use_fastokens so rollout actors patch their
    # tokenizer consistently with the driver. Defaults to off when absent.
    use_fastokens: NotRequired[bool]
    # Multimodal fields (populated by `setup_nemo_gym_config` when VLM is enabled).
    tokenizer_config: NotRequired[
        Optional[TokenizerConfig]
    ]  # For processor reconstruction inside the actor


def _detect_invalid_tool_call_and_malformed_thinking(
    output_item_dict: dict[str, Any],
    invalid_tool_call_patterns: list[str] | None = None,
    thinking_tags: list[str] | None = None,
) -> tuple[bool, bool]:
    """Flag a NeMo-Gym output item as an invalid tool call / malformed thinking.

    Inspects the final output item of a model turn. For a final *content*
    message, any thinking tag is malformed (thinking should never leak into the
    answer); for a *reasoning* summary, only a repeated tag (count > 1) is
    malformed (a single pair is expected). A textual tool-call pattern in either
    indicates an invalid (unexecuted) tool call.

    Returns:
        (is_invalid_tool_call, has_malformed_thinking).
    """
    invalid_tool_call_patterns = (
        invalid_tool_call_patterns or DEFAULT_INVALID_TOOL_CALL_PATTERNS
    )
    thinking_tags = thinking_tags or DEFAULT_THINKING_TAGS

    is_output_message = (
        "content" in output_item_dict
        and len(output_item_dict["content"]) > 0
        and "text" in output_item_dict["content"][0]
    )
    # NeMo-Gym only attaches generation_token_ids to the last output item of a
    # model call (see vllm_model/app.py postprocess_chat_response). So this item
    # is guaranteed to be the final thing the model produced for this turn.
    # If it's a reasoning item, the model output only reasoning (no content/tool calls).
    is_reasoning_message = (
        output_item_dict.get("type") == "reasoning"
        and len(output_item_dict.get("summary", [])) > 0
        and "text" in output_item_dict["summary"][0]
    )

    is_invalid_tool_call = False
    has_malformed_thinking = False
    if is_output_message:
        assistant_message_content = output_item_dict["content"][0]["text"]
        if any(
            pattern in assistant_message_content
            for pattern in invalid_tool_call_patterns
        ):
            is_invalid_tool_call = True
        if any(tag in assistant_message_content for tag in thinking_tags):
            has_malformed_thinking = True
    elif is_reasoning_message:
        assistant_message_content = output_item_dict["summary"][0]["text"]
        if any(
            pattern in assistant_message_content
            for pattern in invalid_tool_call_patterns
        ):
            is_invalid_tool_call = True
        if any(assistant_message_content.count(tag) > 1 for tag in thinking_tags):
            has_malformed_thinking = True

    return is_invalid_tool_call, has_malformed_thinking


########################################
# Multimodal helpers
########################################


# WARNING: A function-call output beginning with HTTP(S) is accepted here and
# passed to ``resolve_to_image``, which performs an outbound request during
# postprocessing even when the tool result is not actually an image.
_IMAGE_SRC_PREFIXES = ("data:image/", "http://", "https://", "file://")


def _looks_like_image_src(src: str) -> bool:
    """True when ``src`` plausibly points at an image the loader can open.

    Guards against tool responses (e.g. ``{"x": 0.65, "y": 0.83}`` from a
    click tool) that are strings but not image URLs. Without this, the
    indexer forwards the JSON payload to ``resolve_to_image`` → PIL.open,
    which treats it as a filesystem path and raises ``FileNotFoundError``.
    """
    return src.startswith(_IMAGE_SRC_PREFIXES)


_VIDEO_CONTENT_TYPES = {"input_video", "video", "video_url"}
_IMAGE_CONTENT_TYPES = {"input_image", "image", "image_url"}
_AUDIO_CONTENT_TYPES = {"input_audio", "audio", "audio_url"}


def _get_content_part_url(part: dict[str, Any], *keys: str) -> str:
    """Return a string media source from a Responses/Chat content part."""
    for key in keys:
        value = part.get(key)
        if isinstance(value, dict):
            value = value.get("url") or value.get("path")
        if isinstance(value, str) and value:
            return value
    return ""


def _resolve_local_video_path(source: str) -> str:
    """Resolve a local video source and reject unsupported remote schemes."""
    parsed = urlparse(source)
    if parsed.scheme == "file":
        source = unquote(parsed.path)
    elif parsed.scheme:
        raise ValueError(
            "NeMo RL Gym video training currently supports local paths and "
            f"file:// URLs; received scheme {parsed.scheme!r}."
        )

    path = Path(source).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Gym video paths must be absolute, got {source!r}.")
    if not path.is_file():
        raise FileNotFoundError(f"Gym video file does not exist: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Gym video file is not readable: {path}")
    return str(path.resolve())


def _extract_static_video_messages(
    nemo_gym_example: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None] | None:
    """Convert one-video Responses input into HF multimodal chat messages.

    A video may be represented either by one native video content part or by a
    sequence of cached ``input_image`` parts carrying ``_is_video_frame``.  The
    latter is the on-disk frame-cache format used by the video Gym recipes.
    """
    response_input = nemo_gym_example.get("responses_create_params", {}).get(
        "input", []
    )
    if isinstance(response_input, str):
        return None

    video_sources: list[str] = []
    cached_frame_sources: list[str] = []
    has_still_images = False
    hf_messages: list[dict[str, Any]] = []
    for item in response_input:
        if not isinstance(item, dict) or "role" not in item:
            continue
        content = item.get("content", "")
        if isinstance(content, str):
            hf_messages.append({"role": item["role"], "content": content})
            continue
        if not isinstance(content, list):
            continue

        hf_content: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "input_text":
                hf_content.append({"type": "text", "text": part["text"]})
            elif part_type in _VIDEO_CONTENT_TYPES:
                source = _get_content_part_url(part, "video_url", "video", "url")
                if not source:
                    raise ValueError(f"{part_type} requires a non-empty video URL")
                video_sources.append(source)
                hf_content.append({"type": "video", "video": source})
            elif part_type in _IMAGE_CONTENT_TYPES:
                if not part.get("_is_video_frame"):
                    has_still_images = True
                    continue
                source = _get_content_part_url(part, "image_url", "image", "url")
                if not source:
                    raise ValueError(
                        "Cached Gym video frames require a non-empty image URL."
                    )
                cached_frame_sources.append(str(part.get("_video_source") or ""))
                hf_content.append(
                    {
                        "type": "image",
                        "image": resolve_to_image(source),
                        "_is_video_frame": True,
                        "_video_source": part.get("_video_source"),
                        "_video_frame_index": part.get("_video_frame_index"),
                        "_video_fps": part.get("_video_fps"),
                    }
                )
            elif part_type in _AUDIO_CONTENT_TYPES:
                raise ValueError(
                    "The initial Gym video contract does not support audio or "
                    "audio+video inputs."
                )
            else:
                raise ValueError(
                    f"Unsupported Gym multimodal content type: {part_type!r}"
                )
        hf_messages.append({"role": item["role"], "content": hf_content})

    if not video_sources and not cached_frame_sources:
        return None
    if has_still_images:
        raise ValueError(
            "Gym video training does not support mixing still images and "
            "video frames in one row."
        )
    if video_sources and cached_frame_sources:
        raise ValueError(
            "Gym video training does not support mixing native video and "
            "predecoded video frames in one row."
        )
    if len(video_sources) != 1:
        if video_sources:
            raise ValueError(
                "Gym video training requires exactly one video per row; "
                f"received {len(video_sources)}."
            )
        frame_groups = {source for source in cached_frame_sources if source}
        if len(frame_groups) > 1:
            raise ValueError(
                "Gym video training requires cached frames from exactly one "
                f"video per row; received {len(frame_groups)} sources."
            )
        return hf_messages, None
    return hf_messages, _resolve_local_video_path(video_sources[0])


def _json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _metadata_extra_body(nemo_gym_example: dict[str, Any]) -> dict[str, Any]:
    metadata = nemo_gym_example.get("responses_create_params", {}).get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    return _json_mapping(metadata.get("extra_body", {}))


def _chat_template_kwargs_for_processor(
    nemo_gym_example: dict[str, Any],
) -> dict[str, Any]:
    extra_body = _metadata_extra_body(nemo_gym_example)
    processor_kwargs: dict[str, Any] = {}
    chat_template_kwargs = _json_mapping(extra_body.get("chat_template_kwargs", {}))
    if chat_template_kwargs:
        processor_kwargs["chat_template_kwargs"] = chat_template_kwargs
    enable_thinking = extra_body.get(
        "enable_thinking", chat_template_kwargs.get("enable_thinking")
    )
    if enable_thinking is not None:
        processor_kwargs["enable_thinking"] = enable_thinking
    return processor_kwargs


def _deep_merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _inject_vllm_mm_processor_kwargs(
    nemo_gym_example: dict[str, Any],
    mm_processor_kwargs: dict[str, Any],
) -> None:
    params = nemo_gym_example.setdefault("responses_create_params", {})
    if not isinstance(params, dict):
        raise TypeError("responses_create_params must be a dict")
    metadata = params.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("responses_create_params.metadata must be a dict")

    original_extra_body = metadata.get("extra_body", "{}")
    extra_body = _json_mapping(original_extra_body)
    extra_body = _deep_merge_dict(
        extra_body, {"mm_processor_kwargs": mm_processor_kwargs}
    )
    metadata["extra_body"] = (
        json.dumps(extra_body) if isinstance(original_extra_body, str) else extra_body
    )


def _remove_vllm_mm_processor_kwargs(
    nemo_gym_example: dict[str, Any], names: set[str]
) -> None:
    params = nemo_gym_example.get("responses_create_params", {})
    if not isinstance(params, dict):
        return
    metadata = params.get("metadata", {})
    if not isinstance(metadata, dict) or "extra_body" not in metadata:
        return

    original_extra_body = metadata["extra_body"]
    extra_body = _json_mapping(original_extra_body)
    mm_processor_kwargs = extra_body.get("mm_processor_kwargs")
    if not isinstance(mm_processor_kwargs, dict):
        return
    for name in names:
        mm_processor_kwargs.pop(name, None)
    if not mm_processor_kwargs:
        extra_body.pop("mm_processor_kwargs", None)
    metadata["extra_body"] = (
        json.dumps(extra_body) if isinstance(original_extra_body, str) else extra_body
    )


def _replace_cached_video_frames_with_native_video(
    nemo_gym_example: dict[str, Any],
) -> None:
    """Replace cached image parts with one lossless native-video manifest."""
    input_items = nemo_gym_example.get("responses_create_params", {}).get("input", [])
    if not isinstance(input_items, list):
        raise TypeError("responses_create_params.input must be a list")

    frame_paths = []
    video_sources = set()
    for item in input_items:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or not part.get("_is_video_frame"):
                continue
            frame_path = _get_content_part_url(part, "image_url", "image", "url")
            if not frame_path:
                raise ValueError(
                    "Cached Gym video frames require a non-empty image URL."
                )
            frame_paths.append(frame_path)
            video_source = part.get("_video_source")
            if video_source:
                video_sources.add(str(video_source))

    if not frame_paths:
        raise ValueError("Cached Gym video request contains no frame paths.")
    if len(video_sources) != 1:
        raise ValueError(
            "Cached Gym video frames require exactly one non-empty _video_source; "
            f"received {len(video_sources)}."
        )

    from nemo_rl.models.generation.vllm.utils import (
        build_cached_video_frame_data_url,
    )

    video_url = build_cached_video_frame_data_url(frame_paths)
    inserted_video = False
    for item in input_items:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        converted_content = []
        for part in content:
            if isinstance(part, dict) and part.get("_is_video_frame"):
                if not inserted_video:
                    converted_content.append(
                        {
                            "type": "input_video",
                            "video_url": {"url": video_url},
                        }
                    )
                    inserted_video = True
                continue
            converted_content.append(part)
        item["content"] = converted_content

    if not inserted_video:
        raise ValueError("Failed to insert cached Gym video manifest.")


def _ensure_vllm_video_placeholder_target(
    nemo_gym_example: dict[str, Any],
) -> None:
    # Keep a token boundary after vLLM's literal replacement target. Some BPE
    # tokenizers merge ``>\n`` into one token, so ``<video>\n`` does not contain
    # the standalone token sequence vLLM searches for before its text fallback.
    # A normal space keeps the target independently tokenized while preserving
    # the same rendered prompt semantics.
    video_target_prefix = "<video> "
    input_items = nemo_gym_example.get("responses_create_params", {}).get("input", [])
    if not isinstance(input_items, list):
        return

    for item in input_items:
        if not isinstance(item, dict):
            continue
        content = item.get("content", "")
        if isinstance(content, str) and "<video>" in content:
            return
        if isinstance(content, list):
            for part in content:
                if isinstance(part, str) and "<video>" in part:
                    return
                if (
                    isinstance(part, dict)
                    and isinstance(part.get("text"), str)
                    and "<video>" in part["text"]
                ):
                    return

    for item in input_items:
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content", "")
        if isinstance(content, str):
            item["content"] = (
                f"{video_target_prefix}{content}" if content else "<video>"
            )
            return
        if not isinstance(content, list):
            continue
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") in ("input_text", "text")
                and isinstance(part.get("text", ""), str)
            ):
                text = part.get("text", "")
                part["text"] = f"{video_target_prefix}{text}" if text else "<video>"
                return
        content.append({"type": "input_text", "text": "<video>"})
        return


def _strip_local_media_metadata(nemo_gym_example: dict[str, Any]) -> None:
    input_items = nemo_gym_example.get("responses_create_params", {}).get("input", [])
    if not isinstance(input_items, list):
        return
    for item in input_items:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            for key in list(part):
                if key.startswith("_"):
                    part.pop(key)


def _compute_dynamic_prompt_length(
    processor: Any,
    messages: list[dict[str, Any]],
    template_kwargs: dict[str, Any],
) -> int | None:
    try:
        rendered = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        if not isinstance(rendered, str):
            return None
        tokenized = processor.tokenizer(
            rendered.replace("<image>", ""),
            add_special_tokens=False,
        )
        input_ids = getattr(tokenized, "input_ids", None)
        if input_ids is None and isinstance(tokenized, dict):
            input_ids = tokenized.get("input_ids")
        return len(input_ids) if input_ids is not None else None
    except Exception as exc:
        print(
            "WARNING: failed to compute dynamic video prompt length: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


def _video_to_image_content(
    video_path: str,
    *,
    num_frames: int,
    temporal_patch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from nemo_rl.models.generation.vllm.utils import (
        load_video_frames_with_metadata,
    )

    frames, metadata = load_video_frames_with_metadata(
        video_path,
        num_frames=num_frames,
        temporal_patch_size=temporal_patch_size,
    )
    frame_indices = list((metadata or {}).get("frames_indices", []))
    fps = float((metadata or {}).get("fps") or 0.0)
    items = []
    for frame_idx, frame in enumerate(frames):
        image = Image.fromarray(frame).convert("RGB")
        sampled_frame_idx = (
            int(frame_indices[frame_idx])
            if frame_idx < len(frame_indices)
            else frame_idx
        )
        items.append(
            {
                "type": "image",
                "image": image,
                "_is_video_frame": True,
                "_video_source": video_path,
                "_video_frame_index": sampled_frame_idx,
                "_video_fps": fps,
            }
        )
    return items, metadata


_NEMOTRON_VIDEO_PROCESSOR_NAMES = frozenset(
    {
        "NemotronNanoVLV2Processor",
        "NemotronH_Nano_Omni_Reasoning_V3Processor",
    }
)


def _required_config_value(config: Any, name: str) -> Any:
    if isinstance(config, dict):
        if name not in config:
            raise ValueError(f"Nemotron video config is missing {name!r}.")
        return config[name]
    if not hasattr(config, name):
        raise ValueError(f"Nemotron video config is missing {name!r}.")
    return getattr(config, name)


@lru_cache(maxsize=8)
def _load_nemotron_video_model_config(model_name: str) -> Any:
    return AutoConfig.from_pretrained(model_name, trust_remote_code=True)


def _nemotron_video_target_resolution(
    *,
    original_width: int,
    original_height: int,
    target_num_patches: int,
    patch_size: int,
    downsample_ratio: float,
    maintain_aspect_ratio: bool,
) -> tuple[int, int]:
    """Return the SFT/vLLM-compatible ``(width, height)`` for a video frame."""
    if target_num_patches <= 0:
        raise ValueError("video_target_num_patches must be positive.")
    if patch_size <= 0:
        raise ValueError("Nemotron patch_size must be positive.")
    if not 0 < downsample_ratio <= 1:
        raise ValueError(
            f"Nemotron downsample_ratio must be in (0, 1], got {downsample_ratio}."
        )

    if maintain_aspect_ratio:
        aspect_ratio = original_width / max(original_height, 1)
        patch_height = round(math.sqrt(target_num_patches / aspect_ratio))
        patch_width = round(math.sqrt(target_num_patches * aspect_ratio))
    else:
        side = math.isqrt(target_num_patches)
        patch_height = patch_width = side

    patch_height = max(1, patch_height)
    patch_width = max(1, patch_width)
    required_divisor = int(round(1 / downsample_ratio))
    if required_divisor > 1:
        height_remainder = patch_height % required_divisor
        width_remainder = patch_width % required_divisor
        height_up = patch_height + (
            required_divisor - height_remainder if height_remainder else 0
        )
        width_up = patch_width + (
            required_divisor - width_remainder if width_remainder else 0
        )
        height_down = patch_height - height_remainder
        width_down = patch_width - width_remainder
        if height_up * width_up <= target_num_patches:
            patch_height, patch_width = height_up, width_up
        else:
            patch_height = max(required_divisor, height_down)
            patch_width = max(required_divisor, width_down)

    return patch_width * patch_size, patch_height * patch_size


def _flatten_nemotron_video_frame_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """Replace locally decoded frame items with ordered ``<image>`` markers."""
    flattened_messages = []
    frames = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            flattened_messages.append(message)
            continue

        flattened_parts = []
        for part in content:
            if not isinstance(part, dict):
                flattened_parts.append(str(part))
                continue
            part_type = part.get("type")
            if part_type in ("image", "image_url") and part.get("_is_video_frame"):
                image = part.get("image")
                if not isinstance(image, Image.Image):
                    raise ValueError(
                        "Nemotron video frames must be decoded PIL images, "
                        f"got {type(image).__name__}."
                    )
                frames.append(image)
                flattened_parts.append("<image>")
            elif part_type == "text":
                flattened_parts.append(str(part.get("text", "")))
            elif part_type in ("image", "image_url"):
                raise ValueError(
                    "Nemotron Gym video preprocessing does not support mixing "
                    "still images with video frames."
                )
        flattened_messages.append(
            {
                **message,
                "content": "\n".join(part for part in flattened_parts if part),
            }
        )
    return flattened_messages, frames


def _render_nemotron_video_prompt(
    processor: Any,
    messages: list[dict[str, Any]],
    template_kwargs: dict[str, Any],
) -> str:
    render_kwargs = copy.deepcopy(template_kwargs)
    nested_template_kwargs = render_kwargs.pop("chat_template_kwargs", {})
    if isinstance(nested_template_kwargs, dict):
        for name, value in nested_template_kwargs.items():
            render_kwargs.setdefault(name, value)
    rendered = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **render_kwargs,
    )
    if not isinstance(rendered, str):
        raise TypeError(
            "Nemotron tokenizer.apply_chat_template must return a string when "
            f"tokenize=False, got {type(rendered).__name__}."
        )
    return rendered


def _expand_nemotron_video_placeholders(
    rendered_text: str,
    *,
    embeddings_per_tubelet: list[int],
    temporal_patch_size: int,
) -> str:
    """Expand primary tubelet frames and retain secondary frames as bare tokens.

    The current Megatron-Bridge model consumes only image tokens inside
    ``<img>...</img>`` wrappers. vLLM-authored Nemotron prompts retain a bare
    ``<image>`` for secondary frames in each temporal tubelet, so reproducing
    that shape here keeps the pre-rollout datum compatible with the native
    expanded-sequence path. The rollout manager later replaces these provisional
    token IDs with the exact token IDs returned by vLLM.
    """
    if temporal_patch_size < 1:
        raise ValueError("video_temporal_patch_size must be at least 1.")
    parts = rendered_text.split("<image>")
    frame_count = len(parts) - 1
    expected_tubelets = math.ceil(frame_count / temporal_patch_size)
    if len(embeddings_per_tubelet) != expected_tubelets:
        raise ValueError(
            "Rendered Nemotron video prompt/frame mismatch: "
            f"found {frame_count} <image> markers for "
            f"{len(embeddings_per_tubelet)} tubelets; expected "
            f"{expected_tubelets} tubelets with temporal patch size "
            f"{temporal_patch_size}."
        )

    expanded = parts[0]
    for frame_index, suffix in enumerate(parts[1:]):
        if frame_index % temporal_patch_size == 0:
            tubelet_index = frame_index // temporal_patch_size
            replacement = (
                "<img>" + "<image>" * embeddings_per_tubelet[tubelet_index] + "</img>"
            )
        else:
            replacement = "<image>"
        expanded += replacement + suffix
    return expanded


def _process_nemotron_video_frames(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    template_kwargs: dict[str, Any],
    temporal_patch_size: int,
    target_num_patches: int,
    maintain_aspect_ratio: bool,
) -> dict[str, torch.Tensor]:
    """Port the source branch's dynamic video-frame preprocessing contract."""
    model_name = getattr(processor.tokenizer, "name_or_path", None)
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(
            "Nemotron video preprocessing requires tokenizer.name_or_path."
        )
    model_config = _load_nemotron_video_model_config(model_name)
    patch_size = int(_required_config_value(model_config, "patch_size"))
    downsample_ratio = float(_required_config_value(model_config, "downsample_ratio"))
    norm_mean = torch.tensor(
        _required_config_value(model_config, "norm_mean"), dtype=torch.float32
    ).view(3, 1, 1)
    norm_std = torch.tensor(
        _required_config_value(model_config, "norm_std"), dtype=torch.float32
    ).view(3, 1, 1)

    flattened_messages, frames = _flatten_nemotron_video_frame_messages(messages)
    if not frames:
        raise ValueError("Nemotron video preprocessing received no decoded frames.")

    rendered_text = _render_nemotron_video_prompt(
        processor, flattened_messages, template_kwargs
    )
    pixel_values = []
    image_sizes = []
    embeddings_per_frame = []
    for frame in frames:
        target_width, target_height = _nemotron_video_target_resolution(
            original_width=frame.width,
            original_height=frame.height,
            target_num_patches=target_num_patches,
            patch_size=patch_size,
            downsample_ratio=downsample_ratio,
            maintain_aspect_ratio=maintain_aspect_ratio,
        )
        frame_array = np.array(
            frame.convert("RGB") if frame.mode != "RGB" else frame,
            dtype=np.uint8,
            copy=True,
        )
        frame_tensor = torch.from_numpy(frame_array).permute(2, 0, 1).unsqueeze(0)
        if frame_tensor.shape[-2:] != (target_height, target_width):
            frame_tensor = F.interpolate(
                frame_tensor,
                size=(target_height, target_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        normalized = frame_tensor.squeeze(0) / 255.0
        normalized = (normalized - norm_mean) / norm_std
        pixel_values.append(normalized.contiguous())
        image_sizes.append([target_height, target_width])
        embeddings_per_frame.append(
            int((target_height // patch_size) * downsample_ratio)
            * int((target_width // patch_size) * downsample_ratio)
        )

    if len(set(map(tuple, image_sizes))) != 1:
        raise ValueError(
            "All frames from one Nemotron Gym video must resolve to the same "
            f"target size, got {sorted(set(map(tuple, image_sizes)))}."
        )
    embeddings_per_tubelet = [
        embeddings_per_frame[frame_index]
        for frame_index in range(0, len(frames), temporal_patch_size)
    ]
    expanded_text = _expand_nemotron_video_placeholders(
        rendered_text,
        embeddings_per_tubelet=embeddings_per_tubelet,
        temporal_patch_size=temporal_patch_size,
    )
    text_inputs = processor.tokenizer(
        expanded_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    return {
        **dict(text_inputs),
        "pixel_values": torch.stack(pixel_values),
        "imgs_sizes": torch.tensor(image_sizes, dtype=torch.int32),
    }


def _make_overlength_filtered_video_example(
    nemo_gym_example: dict[str, Any],
) -> dict[str, Any]:
    filtered = copy.deepcopy(nemo_gym_example)
    params = filtered.setdefault("responses_create_params", {})
    params["input"] = [
        {
            "role": "user",
            "type": "message",
            "content": [
                {
                    "type": "input_text",
                    "text": "This sample was filtered because its prompt is too long.",
                }
            ],
        }
    ]
    return filtered


def nemo_gym_example_to_video_datum_spec(
    nemo_gym_example: dict[str, Any],
    *,
    processor: Any,
    max_seq_length: int | None,
    idx: int,
    task_name: str,
    data_config: Any | None = None,
) -> dict[str, Any] | None:
    """Preprocess static Gym video with vLLM-equivalent frame sampling.

    The raw video remains in the outbound Gym request. Cached frames are sent as
    one native-video manifest so vLLM consumes the same lossless RGB frames as
    policy preprocessing. Those tensors are reattached to vLLM-authored prompt
    token IDs after the rollout.
    """
    extracted = _extract_static_video_messages(nemo_gym_example)
    if extracted is None:
        return None
    hf_messages, video_path = extracted

    num_frames = int(getattr(data_config, "num_frames", 8))
    temporal_patch_size = int(getattr(data_config, "video_temporal_patch_size", 1))
    if video_path is not None:
        frame_items, _video_metadata = _video_to_image_content(
            video_path,
            num_frames=num_frames,
            temporal_patch_size=temporal_patch_size,
        )
        for message in hf_messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            expanded_content = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "video":
                    expanded_content.extend(frame_items)
                else:
                    expanded_content.append(part)
            message["content"] = expanded_content
    else:
        frame_items = [
            part
            for message in hf_messages
            for part in message.get("content", [])
            if isinstance(part, dict)
            and part.get("type") in ("image", "image_url")
            and part.get("_is_video_frame")
        ]
        if not frame_items:
            raise ValueError("Cached Gym video preprocessing received no frames.")

    template_kwargs = _chat_template_kwargs_for_processor(nemo_gym_example)
    video_target_num_patches = getattr(data_config, "video_target_num_patches", None)
    maintain_aspect_ratio = bool(
        getattr(data_config, "video_maintain_aspect_ratio", True)
    )
    if type(processor).__name__ in _NEMOTRON_VIDEO_PROCESSOR_NAMES:
        if video_target_num_patches is None:
            raise ValueError(
                "Nemotron Gym video data requires video_target_num_patches."
            )
        processed = _process_nemotron_video_frames(
            processor,
            hf_messages,
            template_kwargs=template_kwargs,
            temporal_patch_size=temporal_patch_size,
            target_num_patches=int(video_target_num_patches),
            maintain_aspect_ratio=maintain_aspect_ratio,
        )
    else:
        processor_kwargs: dict[str, Any] = {
            "video_flags": [True] * len(frame_items),
            "video_temporal_patch_size": temporal_patch_size,
            "video_maintain_aspect_ratio": maintain_aspect_ratio,
        }
        if video_target_num_patches is not None:
            processor_kwargs["video_target_num_patches"] = video_target_num_patches
        if max_seq_length is not None:
            prompt_length = _compute_dynamic_prompt_length(
                processor, hf_messages, template_kwargs
            )
            if prompt_length is not None:
                processor_kwargs["num_tokens_available"] = (
                    max_seq_length
                    - prompt_length
                    - int(getattr(data_config, "min_generation_tokens", 2000))
                )
        processed = dict(
            processor.apply_chat_template(
                hf_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **template_kwargs,
                **processor_kwargs,
            )
        )
    user_message: dict[str, Any] = {
        "role": "user",
        "content": "",
        "token_ids": processed["input_ids"][0],
    }
    if "imgs_sizes" in processed and "num_frames" not in processed:
        processed["num_frames"] = torch.tensor([len(frame_items)], dtype=torch.int32)
    user_message.update(extract_multimodal_model_inputs(processor, processed))
    if "num_frames" in processed:
        user_message["num_frames"] = PackedTensor(
            processed["num_frames"].to(dtype=torch.int32),
            dim_to_pack=get_dim_to_pack_along(processor, "num_frames"),
        )

    length = len(user_message["token_ids"])
    loss_multiplier = 1.0
    extra_env_info = copy.deepcopy(nemo_gym_example)
    is_nemotron_video = type(processor).__name__ in _NEMOTRON_VIDEO_PROCESSOR_NAMES
    if is_nemotron_video and video_path is None:
        _replace_cached_video_frames_with_native_video(extra_env_info)
    _strip_local_media_metadata(extra_env_info)
    _ensure_vllm_video_placeholder_target(extra_env_info)
    # vLLM 0.20's native Nano-Nemotron processor consumes the video modality
    # directly and reads its temporal/dynamic-resolution settings from the
    # checkpoint config. Its constructor rejects the legacy
    # ``video_as_images`` kwarg. Keep that compatibility path only for
    # processors that still expect frame-as-image grouping.
    if is_nemotron_video:
        _remove_vllm_mm_processor_kwargs(
            extra_env_info, {"max_num_tiles", "video_as_images"}
        )
    else:
        mm_processor_kwargs: dict[str, Any] = {"video_as_images": True}
        if video_target_num_patches is not None:
            mm_processor_kwargs["max_num_tiles"] = 1
        _inject_vllm_mm_processor_kwargs(extra_env_info, mm_processor_kwargs)

    if max_seq_length is not None and length >= max_seq_length:
        for key, value in list(user_message.items()):
            if isinstance(value, PackedTensor):
                user_message[key] = PackedTensor.empty_like(value)
        user_message["token_ids"] = user_message["token_ids"][: min(4, max_seq_length)]
        length = len(user_message["token_ids"])
        loss_multiplier = 0.0
        extra_env_info = _make_overlength_filtered_video_example(nemo_gym_example)

    return {
        "message_log": [user_message],
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": task_name,
    }


def reattach_static_multimodal_payload(
    target_message_log: list[dict[str, Any]],
    source_message_log: list[dict[str, Any]] | None,
) -> None:
    """Attach driver-side PackedTensor payloads to the first rollout user turn."""
    if not source_message_log:
        return
    payload = {
        key: value
        for message in source_message_log
        for key, value in message.items()
        if isinstance(value, PackedTensor)
    }
    if not payload:
        return
    for message in target_message_log:
        if message.get("role") == "user":
            message.update(payload)
            return
    raise ValueError(
        "Cannot attach the static multimodal payload: Gym returned no user message."
    )


def _extract_input_images_from_message(item: dict) -> list[Image.Image]:
    """Pull PIL images out of a non-assistant Responses-API item.

    Handles both content-list items (user / tool messages carrying
    ``input_image``/``image``/``image_url`` parts) and ``function_call_output``
    items whose ``output`` field is an image data URL. Tool outputs that are
    non-image strings (e.g. structured JSON returned by tools like
    ``click(x, y)``) contribute zero images to the bucket.
    """
    images: list[Image.Image] = []
    if item.get("type") == "function_call_output":
        src = item.get("output")
        if isinstance(src, str) and _looks_like_image_src(src):
            images.append(resolve_to_image(src))
        return images
    content = item.get("content") or []
    if not isinstance(content, list):
        return images
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") not in ("input_image", "image", "image_url"):
            continue
        src = part.get("image") or part.get("image_url") or part.get("url")
        if src is None:
            continue
        if isinstance(src, dict):
            src = src.get("url")
        if src is None:
            continue
        images.append(resolve_to_image(src))
    return images


def _index_per_turn_images(
    output: list[dict],
    input_messages: list[dict] | None = None,
) -> list[list[Image.Image]]:
    """Bin server-returned images by the trainable turn that saw them.

    Walks the Responses-API items in order and flushes ``pending`` into a
    per-turn bucket each time it hits an item carrying truthy
    ``generation_token_ids`` — matching the exact gate that
    ``_postprocess_nemo_gym_to_nemo_rl_result`` uses to decide which items
    become trainable turns. Every other item (user turns, tool messages,
    ``function_call_output``, non-trainable reasoning) contributes its images
    to ``pending`` for the next trainable turn. This ensures the returned list
    has one entry per trainable turn, aligned with the postprocess loop's
    ``turn_idx`` even when the trainable item's role is not ``assistant``
    (e.g. a reasoning-only response, or a ``function_call``).

    ``input_messages`` is the initial ``responses_create_params.input`` list —
    images there (e.g. a single-shot user prompt for tool-based envs like
    circle-click) are consumed by the first trainable turn's tokenized prompt
    and must land in the first bucket. Agents like ``gym_v_agent`` that keep
    ``input`` empty and inject observations as ``function_call_output`` items
    are unaffected — the seed is a no-op when ``input_messages`` is empty.
    """
    per_turn: list[list[Image.Image]] = []
    pending: list[Image.Image] = []
    for item in input_messages or ():
        if isinstance(item, dict) and item.get("role") != "assistant":
            pending.extend(_extract_input_images_from_message(item))
    for item in output:
        if item.get(
            "generation_token_ids"
        ):  # trainable turn; empty generation_token_ids is skipped by the postprocess loop and must not consume a bucket
            per_turn.append(pending)
            pending = []
        elif item.get("role") != "assistant":
            pending.extend(_extract_input_images_from_message(item))
    return per_turn


def _attach_multimodal_data_to_user_message(
    user_message: dict,
    *,
    images: list[Image.Image],
    processor: Any,
) -> None:
    """Attach per-turn multimodal tensors to ``user_message``.

    The processor is only invoked to extract multimodal tensors (pixel_values,
    imgs_sizes, num_patches, etc.); its text output is discarded — vLLM's
    tokens remain the trajectory. We therefore feed it the minimal placeholder
    text it needs to count image regions: one ``processor.image_token`` per
    image. Passing the vLLM-decoded text does not work because that text
    already contains expanded ``<img>...<image>*N...</img>`` regions, and the
    processor would try to re-expand every embedded ``<image>``.
    """
    if not images or processor is None:
        return
    image_token = getattr(processor, "image_token", "<image>")
    processed = processor(
        text=image_token * len(images),
        images=images,
        return_tensors="pt",
    )
    uses_placeholder = uses_image_placeholder(processor)
    multimodal_keys = list(get_multimodal_keys_from_processor(processor))
    # Historical checkpoints may emit dynamic image tiles without imgs_sizes.
    # Mirror the media-metadata handling in vlm_hf_data_processor.
    if (
        uses_placeholder
        and "pixel_values" in processed
        and "imgs_sizes" not in processed
        and processed["pixel_values"].ndim == 4
    ):
        pixel_values = processed["pixel_values"]
        num_tiles, _, height, width = pixel_values.shape
        processed["imgs_sizes"] = torch.tensor(
            [[height, width]] * num_tiles, dtype=torch.long
        )

    # imgs_sizes / num_frames are not always declared in model_input_names by
    # bundled image processors. RADIO uses temporal patching even for still
    # images and requires one num_frames=1 entry per image/tile.
    if "imgs_sizes" in processed and "imgs_sizes" not in multimodal_keys:
        multimodal_keys.append("imgs_sizes")
    if "imgs_sizes" in processed and "num_frames" not in processed:
        processed["num_frames"] = torch.ones(
            len(processed["imgs_sizes"]), dtype=torch.long
        )
    if "num_frames" in processed and "num_frames" not in multimodal_keys:
        multimodal_keys.append("num_frames")
    for key in multimodal_keys:
        if key not in processed:
            continue
        value = processed[key]
        if key == "imgs_sizes":
            value = value.to(dtype=torch.int32)
        user_message[key] = PackedTensor(
            value,
            dim_to_pack=get_dim_to_pack_along(processor, key),
            pad_to_max_shape=uses_placeholder and key == "pixel_values",
        )


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class NemoGym(EnvironmentInterface):
    """This environment class isn't really used for training. It's really meant as an integration wrapper around NeMo-Gym that hooks into the existing NeMo RL resource management via ray. So there is still one source of truth for resource management in NeMo RL."""

    def __init__(self, cfg: NemoGymConfig):
        self.cfg = cfg
        # Reconstruct the processor inside the actor (rather than serializing it
        # per rollout call) for full-trajectory multimodal postprocessing.
        self._processor: Optional[Any] = None
        tokenizer_config = cfg.get("tokenizer_config")
        if tokenizer_config:
            from nemo_rl.algorithms.utils import get_tokenizer

            self._processor = get_tokenizer(tokenizer_config, get_processor=True)
            # _attach_multimodal_data_to_user_message assumes a placeholder-style
            # processor (imgs_sizes / num_frames reconstruction + pad_to_max_shape
            # PackedTensor build). A non-placeholder VLM would silently produce
            # wrong multimodal tensors — fail at actor construction instead.
            assert uses_image_placeholder(self._processor), (
                "NemoGym multimodal path assumes a placeholder-style processor "
                "(see _PLACEHOLDER_STYLE_PROCESSOR_NAMES in nemo_rl/data/multimodal_utils.py); "
                f"got {type(self._processor).__name__}. Update "
                "_attach_multimodal_data_to_user_message before enabling."
            )

    def _spinup(self) -> None:
        """Start the NeMo-Gym head server and rollout collection helper.

        Deferred from __init__ so the actor can be created cheaply (and
        scheduled onto reserved nodes) and spun up explicitly once the vLLM
        server URLs are available, overlapping with vLLM model loading.
        """
        self.node_ip = _get_node_ip_local()
        _gym_port_low = self.cfg.get("port_range_low", DEFAULT_GYM_PORT_RANGE_LOW)
        _gym_port_high = self.cfg.get("port_range_high", DEFAULT_GYM_PORT_RANGE_HIGH)
        self.head_server_port = _get_free_port_local(_gym_port_low, _gym_port_high)

        from nemo_gym.cli import GlobalConfigDictParserConfig, RunHelper
        from nemo_gym.rollout_collection import RolloutCollectionHelper
        from nemo_gym.server_utils import HEAD_SERVER_KEY_NAME, BaseServerConfig
        from omegaconf import DictConfig

        RELATIVE_PATH = "nemo_rl/environments/nemo_gym.py"
        assert __file__.endswith(RELATIVE_PATH)

        # Make a shallow copy so that NeMo-RL-side keys we pop or add below
        # do not mutate the caller's config dict (config.env["nemo_gym"]).
        initial_global_config_dict = dict(
            self.cfg.get("initial_global_config_dict") or {}
        )
        # Strip NeMo-RL-only training knobs that must not be forwarded to the
        # NeMo-Gym server (same pattern as the pops in run_grpo_nemo_gym.py).
        initial_global_config_dict.pop("effort_levels", None)
        # Policy information
        initial_global_config_dict["policy_model_name"] = self.cfg["model_name"]
        initial_global_config_dict["policy_api_key"] = (
            "dummy_key"  # No key necessary for training.
        )
        initial_global_config_dict["policy_base_url"] = self.cfg["base_urls"]
        # In multinode runs, Gym-managed service configs must advertise a real node IP
        # rather than falling back to localhost, or remote workers will connect to
        # their own loopback interface instead of the actor-hosted service.
        initial_global_config_dict.setdefault("default_host", self.node_ip)

        _gym_port_low = self.cfg.get("port_range_low", DEFAULT_GYM_PORT_RANGE_LOW)
        _gym_port_high = self.cfg.get("port_range_high", DEFAULT_GYM_PORT_RANGE_HIGH)
        if (
            _gym_port_low < DEFAULT_GYM_PORT_RANGE_LOW
            or _gym_port_high > DEFAULT_GYM_PORT_RANGE_HIGH
        ):
            print(
                f"WARNING: Gym port range [{_gym_port_low}, {_gym_port_high}) is outside "
                f"the default [{DEFAULT_GYM_PORT_RANGE_LOW}, {DEFAULT_GYM_PORT_RANGE_HIGH}). "
                f"Check the port layout in virtual_cluster.py for conflicts."
            )
        initial_global_config_dict["port_range_low"] = _gym_port_low
        initial_global_config_dict["port_range_high"] = _gym_port_high

        initial_global_config_dict.setdefault(
            "global_aiohttp_connector_limit_per_host", 16_384
        )
        initial_global_config_dict.setdefault("global_aiohttp_connector_limit", 65_536)
        print(
            f"""Set global_aiohttp_connector_limit_per_host={initial_global_config_dict["global_aiohttp_connector_limit_per_host"]} and global_aiohttp_connector_limit={initial_global_config_dict["global_aiohttp_connector_limit"]}.
Depending on your data shape, you may want to change these values."""
        )

        # Get Ray head node address if Ray is initialized
        assert ray.is_initialized(), (
            "Ray must be initialized before using NeMo-Gym environment"
        )
        ray_context = ray.get_runtime_context()
        assert ray_context.gcs_address, "Ray must have a GCS address"

        initial_global_config_dict["ray_head_node_address"] = ray_context.gcs_address
        print(f"Ray head node address: {ray_context.gcs_address}")

        # Head server
        initial_global_config_dict[HEAD_SERVER_KEY_NAME] = {
            "host": "0.0.0.0",
            "port": self.head_server_port,
        }

        self.rh = RunHelper()
        self.rh.start(
            global_config_dict_parser_config=GlobalConfigDictParserConfig(
                dotenv_path=Path(__file__.removesuffix(RELATIVE_PATH)).absolute()
                / "nemo_gym_env.yaml",
                initial_global_config_dict=DictConfig(initial_global_config_dict),
                skip_load_from_cli=True,
            )
        )

        # Setup for rollout collection
        self.head_server_config = BaseServerConfig(
            host=self.node_ip,
            port=self.head_server_port,
        )
        self.rch = RolloutCollectionHelper()

    async def run_rollouts(
        self,
        nemo_gym_examples: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        timer_prefix: str,
    ) -> AsyncGenerator[tuple[int, dict, dict | None], None]:
        """Stream postprocessed rollouts as NeMo-Gym tasks complete."""
        if not nemo_gym_examples:
            raise ValueError("NeMo-Gym rollout batch must not be empty")

        from nemo_rl.utils.fastokens import maybe_patch_fastokens

        maybe_patch_fastokens(bool(self.cfg.get("use_fastokens")))

        timer = Timer()
        counts_left = Counter(row["agent_ref"]["name"] for row in nemo_gym_examples)

        # For multimodal runs, replace local filesystem image paths in the
        # examples with base64 data URLs before shipping to vLLM. No-op when
        # examples carry no `input_image` items (text-only case).
        encode_images_in_examples(nemo_gym_examples)

        timer.start("_run_rollouts_total")
        nemo_gym_result_iterator = self.rch.run_examples(
            examples=nemo_gym_examples, head_server_config=self.head_server_config
        )

        num_results = 0
        for task in nemo_gym_result_iterator:
            with timer.time(label=f"{timer_prefix}/await_results"):
                try:
                    nemo_gym_row, nemo_gym_result = await task
                except Exception as error:
                    if hasattr(error, "response_content"):
                        print(
                            "EXCEPTION RESULT",
                            error.response_content,
                            file=sys.stderr,
                        )
                    raise

            with timer.time(label=f"{timer_prefix}/postprocess_results"):
                nemo_rl_result = self._postprocess_nemo_gym_to_nemo_rl_result(
                    nemo_gym_result, tokenizer
                )
                if _has_nan_generation_logprobs(nemo_rl_result):
                    raise RuntimeError("Generation logprobs contain NaN")

            num_results += 1
            timing_metrics = None
            if num_results == len(nemo_gym_examples):
                timer.stop("_run_rollouts_total")
                timing_metrics = timer.get_timing_metrics("sum")
                total_time = timing_metrics.pop("_run_rollouts_total")
                timing_metrics[f"{timer_prefix}/postprocess_results_pct"] = (
                    100
                    * timing_metrics[f"{timer_prefix}/postprocess_results"]
                    / total_time
                )

            agent_name = nemo_gym_row["agent_ref"]["name"]
            counts_left[agent_name] -= 1
            if counts_left[agent_name] <= 0:
                counts_left.pop(agent_name)
            if num_results % 10 == 0 and counts_left:
                top_left = counts_left.most_common(5)
                top_left_str = "\n".join(
                    f"{index + 1}. {name}: {count}"
                    for index, (name, count) in enumerate(top_left)
                )
                print(
                    "Top 5 NeMo Gym agent refs left in this rollout batch: "
                    f"{top_left_str}",
                    file=sys.stderr,
                )

            yield nemo_gym_row["_rowidx"], nemo_rl_result, timing_metrics

    def _postprocess_nemo_gym_to_nemo_rl_result(
        self,
        nemo_gym_result: dict,
        tokenizer: PreTrainedTokenizerBase,
    ) -> dict:
        assert isinstance(nemo_gym_result, dict), (
            f"Hit a non-successful response when querying NeMo Gym for rollouts: {nemo_gym_result}"
        )

        processor = getattr(self, "_processor", None)
        per_turn_images = (
            _index_per_turn_images(
                nemo_gym_result["response"]["output"],
                input_messages=nemo_gym_result.get("responses_create_params", {}).get(
                    "input"
                ),
            )
            if processor is not None
            else []
        )
        turn_idx = 0

        nemo_rl_message_log = []
        seen_token_ids: List[int] = []
        batch_decode_items = []
        for output_item_dict in nemo_gym_result["response"]["output"]:
            # Nemo RL really only has two types of messages: assistant and not assistant since that is all that it is concerned with (i.e. to train or not to train)
            # Here we map all the trainable messages to assistant and all the non-trainable messages to user.
            # Eventually we can maybe be smarter about this, but this is functional for now.

            # Note that NeMo-Gym will only return token ids on "assistant" messages and not other message types.
            # Also skip if generation_token_ids is present but empty, e.g. all-EOS generation stripped to [] — torch.tensor([]) defaults to float32 and breaks batch dtype consistency.
            if (
                "generation_token_ids" not in output_item_dict
                or not output_item_dict["generation_token_ids"]
            ):
                continue

            assert (
                seen_token_ids
                == output_item_dict["prompt_token_ids"][: len(seen_token_ids)]
            ), f"""Non-contiguous messages found! This may be a tokenization issue where certain tokens are combined when messages are concatenated, or it may be due to part of the chat history being truncated (like if super long history is truncated or if reasoning is stripped out).
Seen token IDs: {seen_token_ids}
Output prompt token IDs: {output_item_dict["prompt_token_ids"]}
output prompt token ids till seen: {output_item_dict["prompt_token_ids"][: len(seen_token_ids)]}
"""

            prompt_token_ids = output_item_dict.pop("prompt_token_ids")
            generation_token_ids = output_item_dict.pop("generation_token_ids")
            generation_log_probs = output_item_dict.pop("generation_log_probs")
            routed_experts_raw = output_item_dict.pop("routed_experts", None)
            new_prompt_token_ids = prompt_token_ids[len(seen_token_ids) :]

            routed_experts = None
            if routed_experts_raw is not None:
                routed_experts_dtype = _ROUTED_EXPERTS_DTYPES[
                    self.cfg.get("routed_experts_dtype", "int16")
                ]
                routed_experts = decode_routed_experts(
                    routed_experts_raw, dtype=routed_experts_dtype
                )
                if routed_experts.dim() != 3:
                    raise ValueError(
                        "NeMo Gym returned routed_experts with invalid shape. "
                        "Expected [tokens, num_moe_layers, topk], got "
                        f"{tuple(routed_experts.shape)}."
                    )
                expected_tokens = len(prompt_token_ids) + len(generation_token_ids)
                if routed_experts.shape[0] < expected_tokens:
                    raise ValueError(
                        "NeMo Gym returned too few routed_experts rows for a "
                        "trainable output item: "
                        f"routes={routed_experts.shape[0]}, expected_at_least="
                        f"{expected_tokens}."
                    )
            elif self.cfg.get("require_routed_experts", False):
                raise ValueError(
                    "policy.router_replay.enabled=true requires NeMo Gym output "
                    "items to include routed_experts, but the field was missing. "
                    "Make sure the Gym repo includes routed_experts propagation "
                    "and the NeMo-RL vLLM OpenAI-compatible server is configured "
                    "with enable_return_routed_experts."
                )

            # The next prompt prefill supplies the real route for the previous
            # turn's final token, whose decode route was padded.
            if routed_experts is not None and seen_token_ids:
                previous_routes = nemo_rl_message_log[-1].get("routed_experts")
                if isinstance(previous_routes, torch.Tensor):
                    previous_routes[-1] = routed_experts[len(seen_token_ids) - 1]

            prompt_start = len(seen_token_ids)
            prompt_end = len(prompt_token_ids)
            generation_start = prompt_end
            generation_end = prompt_end + len(generation_token_ids)

            user_message = {
                "role": "user",
                "content": "",
                "token_ids": torch.tensor(new_prompt_token_ids),
            }
            if routed_experts is not None:
                user_message["routed_experts"] = routed_experts[prompt_start:prompt_end]
            nemo_rl_message_log.append(user_message)

            if processor is not None:
                images_this_turn = (
                    per_turn_images[turn_idx] if turn_idx < len(per_turn_images) else []
                )
                _attach_multimodal_data_to_user_message(
                    user_message,
                    images=images_this_turn,
                    processor=processor,
                )
            # Valid tool calls go through the structured API (tool_calls field) and get
            # executed by NeMo-Gym. If tool call patterns appear in the text content instead,
            # the call was invalid and never executed — flag it so training can penalize it.
            is_invalid_tool_call, has_malformed_thinking = (
                _detect_invalid_tool_call_and_malformed_thinking(
                    output_item_dict,
                    invalid_tool_call_patterns=self.cfg.get(
                        "invalid_tool_call_patterns"
                    ),
                    thinking_tags=self.cfg.get("thinking_tags"),
                )
            )

            assistant_message = {
                "role": "assistant",
                "content": "",
                "token_ids": torch.tensor(generation_token_ids),
                "generation_logprobs": torch.tensor(generation_log_probs),
                "is_invalid_tool_call": is_invalid_tool_call,
                "has_malformed_thinking": has_malformed_thinking,
            }
            if routed_experts is not None:
                assistant_message["routed_experts"] = routed_experts[
                    generation_start:generation_end
                ]
            nemo_rl_message_log.append(assistant_message)

            seen_token_ids.extend(new_prompt_token_ids)
            seen_token_ids.extend(generation_token_ids)

            # We pop to remove larger tensors from logging.
            batch_decode_items.append(
                (output_item_dict, prompt_token_ids, generation_token_ids)
            )
            turn_idx += 1

        if batch_decode_items:
            prompt_strs = tokenizer.batch_decode(
                [item[1] for item in batch_decode_items]
            )
            generation_strs = tokenizer.batch_decode(
                [item[2] for item in batch_decode_items]
            )

            for (output_item_dict, _, _), prompt_str, generation_str in zip(
                batch_decode_items, prompt_strs, generation_strs
            ):
                output_item_dict["prompt_str"] = prompt_str
                output_item_dict["generation_str"] = generation_str

        if not nemo_rl_message_log:
            input_messages = nemo_gym_result["responses_create_params"]["input"]
            try:
                prompt_token_ids = tokenizer.apply_chat_template(
                    input_messages, tokenize=True
                )
                prompt_len_str = f"{len(prompt_token_ids)} tokens"
            except Exception as e:
                prompt_len_str = (
                    f"<unknown — apply_chat_template failed: {type(e).__name__}: {e}>"
                )
            output_item_types = [
                o.get("type") for o in nemo_gym_result["response"]["output"]
            ]
            raise ValueError(
                f"NeMo Gym returned a result with no generation data. "
                f"Possible causes: (1) the prompt for the first turn already exceeds the vLLM max_model_len, "
                f"so vLLM rejected the request before any tokens could be generated; "
                f"(2) all response output items were reasoning/tool-call items with no assistant generation.\n"
                f"  Prompt length: {prompt_len_str}.\n"
                f"  response.output item types ({len(output_item_types)} items): {output_item_types}.\n"
                f"  → If (1): increase `policy.max_total_sequence_length` and `policy.generation.vllm_cfg.max_model_len` "
                f"above the prompt length above.\n"
                f"  → If (2): inspect why no assistant content was produced for this rollout."
            )

        return {
            "message_log": nemo_rl_message_log,
            "input_message_log": nemo_rl_message_log[:1],
            "full_result": nemo_gym_result,
        }

    def shutdown(self) -> None:
        self.rh.shutdown()

    def step(self, message_log_batch, metadata):
        # This is not used since NeMo-Gym will handle the rollouts entirely.
        raise NotImplementedError

    def global_post_process_and_metrics(self, batch):
        # Similar to the step function, this is not used.
        raise NotImplementedError


def extract_reward_components(nemo_gym_result: dict) -> Dict[str, float] | None:
    """Return per-component rewards from a NeMo Gym verify result, or None.

    Single-reward NeMo Gym environments return only a scalar ``reward``. Multi-reward
    environments additionally return ``reward_components``: a mapping of
    component-name -> score. These are surfaced as ``reward/<name>`` batch keys and
    consumed by GDPO (see ``nemo_rl.algorithms.advantage_estimator.GDPOAdvantageEstimator``).

    Returns ``None`` when the environment is single-reward (no ``reward_components``),
    so callers fall back to the scalar ``reward`` path unchanged.
    """
    components = nemo_gym_result.get("reward_components")
    if not components:
        return None
    return {str(name): float(score) for name, score in components.items()}


def build_reward_component_columns(
    component_dicts: List[Dict[str, float] | None],
) -> Dict[str, torch.Tensor]:
    """Build ``reward/<name>`` batch columns from per-sample reward-component dicts.

    Takes the union of component names across the batch in sorted (deterministic) order
    and, for each, emits a ``reward/<name>`` tensor with one entry per sample. A
    component absent on a given sample is filled with ``0.0`` so every column covers all
    samples (the per-prompt baseline requires each component present for all responses).

    Keys are prefixed ``reward/`` so they are exactly what
    ``nemo_rl.algorithms.utils.get_gdpo_reward_component_keys`` selects (it matches
    ``startswith("reward/")`` and sorts by name); the name carries the component identity,
    so no positional index is needed. Returns an empty dict when no sample has components.
    """
    component_names = sorted(
        {name for c in component_dicts if c is not None for name in c}
    )
    return {
        f"reward/{name}": torch.tensor(
            [c[name] if c is not None and name in c else 0.0 for c in component_dicts]
        )
        for name in component_names
    }


def validate_reward_components_match_scalar(nemo_gym_results: List[dict]) -> None:
    """Assert each multi-reward result sets ``reward == sum(reward_components)``.

    A multi-reward verifier must set the scalar ``reward`` to the sum of its
    ``reward_components`` so single-reward (GRPO) consumers and GDPO read the same
    aggregate. We keep the verifier's scalar ``reward`` as ``total_reward`` rather than
    silently overwriting it with the component sum, so a verifier that violates this
    contract must be surfaced here instead of masked.

    Raises ``ValueError`` on the first violating result. A no-op for single-reward
    results (those without ``reward_components``).
    """
    for idx, result in enumerate(nemo_gym_results):
        components = extract_reward_components(result)
        if components is None:
            continue
        scalar_reward = float(result["reward"])
        component_sum = sum(components.values())
        if not math.isclose(scalar_reward, component_sum, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(
                f"NeMo Gym verify result {idx} has reward={scalar_reward} but its "
                f"reward_components sum to {component_sum} ({components}). A multi-reward "
                "verifier must set reward = sum(reward_components.values()) so single-reward "
                "(GRPO) consumers and GDPO read the same aggregate."
            )


########################################
# Global config utils
########################################


def setup_nemo_gym_config(config, tokenizer) -> None:
    generation_config = config.policy["generation"]

    # Enable the http server. Requires both async engine and the expose_http_server flag
    generation_config["vllm_cfg"]["async_engine"] = True
    generation_config["vllm_cfg"]["expose_http_server"] = True

    # Stop strings or token ids are not supported
    generation_config["stop_strings"] = None
    generation_config["stop_token_ids"] = None

    # For VLM runs, plumb the tokenizer config into the gym env config so the
    # NemoGym actor can reconstruct the processor inside itself (needed for
    # multi-turn multimodal postprocessing).
    if config.policy.get("is_vlm"):
        env_cfg = config.env.setdefault("nemo_gym", {})
        env_cfg.setdefault("tokenizer_config", dict(config.policy["tokenizer"]))


def spinup_nemo_gym_actor(
    env_configs: dict[str, Any],
    base_urls: list[str],
    model_name: str,
    *,
    enable_router_replay: bool,
    routed_experts_dtype: str,
    use_fastokens: bool,
) -> Any:
    """Spin up the NeMo-Gym actor against the given generation server URLs.

    When env_configs["nemo_gym"]["num_gpu_nodes"] > 0, the actor is scheduled
    with soft NodeAffinity to the current Ray node so its colocated GPU
    resources land where the caller expects.

    Args:
        env_configs: The master_config.env mapping; env_configs["nemo_gym"] supplies
            the Gym global config plus NeMo-RL detection knobs (invalid_tool_call_patterns,
            thinking_tags, num_gpu_nodes).
        base_urls: Per-DP-rank OpenAI-compatible server base URLs from the generation backend.
        model_name: Served model name the Gym rollouts should target.
        enable_router_replay: Sets require_routed_experts on the NemoGymConfig.
        routed_experts_dtype: Dtype name for R3 routed_experts tensors ("int8"/"int16"/"int32"),
            resolved by the caller from the model's expert count.
        use_fastokens: Forwarded from policy.tokenizer.use_fastokens so the rollout actor
            patches its tokenizer consistently with the driver.

    Returns:
        The spun-up NemoGym Ray actor handle (_spinup already awaited).
    """
    nemo_gym_dict = dict(env_configs["nemo_gym"])

    # NeMo-RL-side detection knobs are top-level NemoGymConfig fields
    # (where the detector reads them), not part of Gym's global config.
    invalid_tool_call_patterns = nemo_gym_dict.pop("invalid_tool_call_patterns", None)
    thinking_tags = nemo_gym_dict.pop("thinking_tags", None)
    tokenizer_config = nemo_gym_dict.pop("tokenizer_config", None)

    # Pass prebuilt cache + venv dirs through the global config so the gym reuses
    # image-baked venvs instead of rebuilding them.
    uv_cache_dir = get_nemo_gym_uv_cache_dir()
    if uv_cache_dir is not None:
        nemo_gym_dict.setdefault("uv_cache_dir", uv_cache_dir)
    uv_venv_dir = get_nemo_gym_venv_dir()
    if uv_venv_dir is not None:
        nemo_gym_dict.setdefault("uv_venv_dir", uv_venv_dir)

    nemo_gym_cfg = NemoGymConfig(
        model_name=model_name,
        base_urls=base_urls,
        invalid_tool_call_patterns=invalid_tool_call_patterns,
        thinking_tags=thinking_tags,
        tokenizer_config=tokenizer_config,
        require_routed_experts=enable_router_replay,
        routed_experts_dtype=routed_experts_dtype,
        use_fastokens=use_fastokens,
        initial_global_config_dict=nemo_gym_dict,
    )

    nemo_gym_py_exec = get_actor_python_env("nemo_rl.environments.nemo_gym.NemoGym")
    if nemo_gym_py_exec.startswith("uv"):
        nemo_gym_py_exec = create_local_venv_on_each_node(
            nemo_gym_py_exec, "nemo_rl.environments.nemo_gym.NemoGym"
        )

    nemo_gym_opts: dict[str, Any] = {}
    if nemo_gym_dict.get("num_gpu_nodes", 0):
        nemo_gym_opts["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=True,
        )
    nemo_gym_opts["runtime_env"] = {
        "py_executable": nemo_gym_py_exec,
        "env_vars": {
            **os.environ,
            "VIRTUAL_ENV": nemo_gym_py_exec,
            "UV_PROJECT_ENVIRONMENT": nemo_gym_py_exec,
        },
    }

    actor = NemoGym.options(**nemo_gym_opts).remote(nemo_gym_cfg)
    ray.get(actor._spinup.remote())
    return actor
