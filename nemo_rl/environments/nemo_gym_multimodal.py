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

import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar, cast
from urllib.parse import unquote, urlparse

import torch
from PIL import Image

from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.multimodal_utils import (
    AUDIO_CONTENT_TYPES,
    IMAGE_CONTENT_TYPES,
    VIDEO_CONTENT_TYPES,
    PackedTensor,
    extract_input_media_sources_from_responses_messages,
    extract_multimodal_model_inputs,
    get_dim_to_pack_along,
    get_responses_content_part_url,
    image_to_data_url,
    media_sources_equal,
    resolve_to_image,
    video_path_to_data_url,
)
from nemo_rl.environments.nemo_gym_request import (
    _chat_template_kwargs_for_processor,
    _deep_merge_dict,
    _json_mapping,
)
from nemo_rl.environments.nemotron_utils import (
    NEMOTRON_VIDEO_PROCESSOR_NAMES,
    process_nemotron_video_frames,
)
from nemo_rl.models.generation.vllm.video_utils import (
    VideoSamplingStyle,
    build_cached_video_frame_data_url,
    build_cached_video_frame_metadata,
    load_video_frames_with_metadata,
)

_NEMO_GYM_IMAGE_ENCODE_MAX_WORKERS = 8


def _encode_single_image_source(source: str) -> str:
    """Resolve, encode, and close one local image source."""
    # `closing` (not a bare `with`): PIL's Image.__exit__ is a no-op, so only an
    # explicit close() releases the buffer.
    with closing(resolve_to_image(source)) as image:
        return image_to_data_url(image)


def normalize_media_in_examples(nemo_gym_examples: list[dict]) -> list[dict]:
    """Replace local media paths in NeMo Gym examples with data URLs."""
    local_image_sources: dict[str, None] = {}
    local_video_sources: dict[str, None] = {}
    pending_mutations: list[
        tuple[dict, tuple[str, str, str], str, str, bool, Any, str]
    ] = []
    for example in nemo_gym_examples:
        input_items = example.get("responses_create_params", {}).get("input", [])
        if not isinstance(input_items, list):
            continue
        for item in input_items:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in IMAGE_CONTENT_TYPES:
                    source_keys = ("image_url", "image", "url")
                    canonical_type = "input_image"
                    canonical_key = "image_url"
                    is_image = True
                elif part_type in VIDEO_CONTENT_TYPES:
                    source_keys = ("video_url", "video", "url")
                    canonical_type = "input_video"
                    canonical_key = "video_url"
                    is_image = False
                else:
                    continue

                present_keys = [key for key in source_keys if key in part]
                if (
                    not present_keys
                    and part_type == "input_image"
                    and "file_id" in part
                ):
                    continue
                if len(present_keys) != 1:
                    raise ValueError(
                        f"{part_type} requires exactly one of {source_keys}"
                    )

                source = part[present_keys[0]]
                nested_detail = (
                    source.get("detail") if isinstance(source, dict) else None
                )
                url = (
                    source.get("url") or source.get("path", "")
                    if isinstance(source, dict)
                    else source
                )
                if not isinstance(url, str) or not url:
                    raise ValueError(f"{part_type} requires a non-empty media URL")
                if not url.startswith(("http://", "https://", "data:")):
                    if is_image:
                        local_image_sources.setdefault(url, None)
                    else:
                        local_video_sources.setdefault(url, None)

                pending_mutations.append(
                    (
                        part,
                        source_keys,
                        canonical_type,
                        canonical_key,
                        is_image,
                        nested_detail,
                        url,
                    )
                )

    sources = list(local_image_sources)
    encoded_by_source: dict[str, str] = {}
    if sources:
        with ThreadPoolExecutor(
            max_workers=_NEMO_GYM_IMAGE_ENCODE_MAX_WORKERS
        ) as executor:
            encoded_by_source = dict(
                zip(
                    sources,
                    executor.map(_encode_single_image_source, sources),
                    strict=True,
                )
            )

    # Encode each unique video once. A video shared by G generations then points
    # every part at the same string, instead of G separate base64 copies of the
    # same file. Kept sequential: these payloads are large enough that encoding
    # several at once would spike driver memory.
    encoded_video_by_source: dict[str, str] = {
        source: video_path_to_data_url(source) for source in local_video_sources
    }

    # Apply mutations only after every local source was encoded successfully.
    for (
        part,
        source_keys,
        canonical_type,
        canonical_key,
        is_image,
        nested_detail,
        url,
    ) in pending_mutations:
        for key in source_keys:
            if key != canonical_key:
                part.pop(key, None)
        part["type"] = canonical_type
        encoded = encoded_by_source if is_image else encoded_video_by_source
        part[canonical_key] = encoded.get(url, url)
        if is_image and nested_detail is not None:
            part.setdefault("detail", nested_detail)
    return nemo_gym_examples


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
        if isinstance(src, dict):
            src = src.get("url")
        if src is not None:
            images.append(resolve_to_image(src))
    return images


def _is_trainable_output_item(item: dict) -> bool:
    """Report whether an output item becomes a trainable assistant turn.

    The postprocess loop skips items whose ``generation_token_ids`` is missing
    *or* empty, so per-turn image binning has to use the same predicate or the
    two walks disagree and every later turn gets the wrong images.
    """
    return bool(item.get("generation_token_ids"))


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
        if _is_trainable_output_item(item):
            per_turn.append(pending)
            pending = []
        elif item.get("role") != "assistant":
            pending.extend(_extract_input_images_from_message(item))
    return per_turn


def _without_initial_media_sources(
    messages: Any, initial_sources: list[Any]
) -> tuple[Any, bool]:
    """Copy Responses messages and remove one ordered copy of initial images and videos."""
    if not isinstance(messages, list):
        return messages, False

    filtered = deepcopy(messages)
    remaining_sources = list(initial_sources)
    for message in filtered:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue

        filtered_content = []
        for part in content:
            part_sources = extract_input_media_sources_from_responses_messages(
                [{"content": [part]}]
            )
            if (
                remaining_sources
                and len(part_sources) == 1
                and media_sources_equal(part_sources[0], remaining_sources[0])
            ):
                remaining_sources.pop(0)
                continue
            filtered_content.append(part)
        message["content"] = filtered_content

    return filtered, not remaining_sources


_VideoConfigValue = TypeVar("_VideoConfigValue")
_LOCAL_VIDEO_METADATA_KEYS = frozenset(
    {
        "_is_video_frame",
        "_video_source",
        "_video_frame_index",
        "_video_fps",
    }
)


def _require_video_config_value(
    value: _VideoConfigValue | None, field_name: str
) -> _VideoConfigValue:
    if value is None:
        raise ValueError(
            f"Gym video preprocessing requires data.{field_name} to be configured."
        )
    return value


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
    sequence of cached ``input_image`` parts carrying ``_is_video_frame``. The
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
            elif part_type in VIDEO_CONTENT_TYPES:
                source = get_responses_content_part_url(
                    part, "video_url", "video", "url"
                )
                if not source:
                    raise ValueError(f"{part_type} requires a non-empty video URL")
                video_sources.append(source)
                hf_content.append({"type": "video", "video": source})
            elif part_type in IMAGE_CONTENT_TYPES:
                if not part.get("_is_video_frame"):
                    has_still_images = True
                    continue
                source = get_responses_content_part_url(
                    part, "image_url", "image", "url"
                )
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
            elif part_type in AUDIO_CONTENT_TYPES:
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
        if len(frame_groups) != 1:
            raise ValueError(
                "Gym video training requires cached frames from exactly one "
                f"video per row; received {len(frame_groups)} sources."
            )
        cached_metadata = build_cached_video_frame_metadata(len(cached_frame_sources))
        cached_frame_position = 0
        for message in hf_messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") in ("image", "image_url")
                    and part.get("_is_video_frame")
                ):
                    part["_video_frame_index"] = cached_metadata["frames_indices"][
                        cached_frame_position
                    ]
                    part["_video_fps"] = cached_metadata["fps"]
                    cached_frame_position += 1
        return hf_messages, None
    return hf_messages, _resolve_local_video_path(video_sources[0])


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

    extra_body = _json_mapping(
        metadata.get("extra_body", "{}"),
        field_name="responses_create_params.metadata.extra_body",
    )
    extra_body = _deep_merge_dict(
        extra_body, {"mm_processor_kwargs": mm_processor_kwargs}
    )
    metadata["extra_body"] = json.dumps(extra_body)


def _remove_vllm_mm_processor_kwargs(
    nemo_gym_example: dict[str, Any], names: set[str]
) -> None:
    params = nemo_gym_example.get("responses_create_params", {})
    if not isinstance(params, dict):
        return
    metadata = params.get("metadata", {})
    if not isinstance(metadata, dict) or "extra_body" not in metadata:
        return

    extra_body = _json_mapping(
        metadata["extra_body"],
        field_name="responses_create_params.metadata.extra_body",
    )
    mm_processor_kwargs = extra_body.get("mm_processor_kwargs")
    if not isinstance(mm_processor_kwargs, dict):
        return
    for name in names:
        mm_processor_kwargs.pop(name, None)
    if not mm_processor_kwargs:
        extra_body.pop("mm_processor_kwargs", None)
    metadata["extra_body"] = json.dumps(extra_body)


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
            frame_path = get_responses_content_part_url(
                part, "image_url", "image", "url"
            )
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
            for key in _LOCAL_VIDEO_METADATA_KEYS:
                part.pop(key, None)


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
    sampling_style: VideoSamplingStyle,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frames, metadata = load_video_frames_with_metadata(
        video_path,
        num_frames=num_frames,
        temporal_patch_size=temporal_patch_size,
        sampling_style=sampling_style,
    )
    frame_indices = list((metadata or {}).get("frames_indices", []))
    fps = float((metadata or {}).get("fps") or 0.0)
    items = []
    for frame_idx in range(frames.shape[0]):
        frame = frames[frame_idx]
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
    data_config: TaskDataSpec | None = None,
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

    if data_config is None:
        raise ValueError("Gym video preprocessing requires a data configuration.")
    num_frames = int(_require_video_config_value(data_config.num_frames, "num_frames"))
    temporal_patch_size = int(
        _require_video_config_value(
            data_config.video_temporal_patch_size,
            "video_temporal_patch_size",
        )
    )
    maintain_aspect_ratio = bool(
        _require_video_config_value(
            data_config.video_maintain_aspect_ratio,
            "video_maintain_aspect_ratio",
        )
    )
    if video_path is not None:
        sampling_style = cast(
            VideoSamplingStyle,
            _require_video_config_value(
                data_config.video_sampling_style,
                "video_sampling_style",
            ),
        )
        frame_items, _video_metadata = _video_to_image_content(
            video_path,
            num_frames=num_frames,
            temporal_patch_size=temporal_patch_size,
            sampling_style=sampling_style,
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
    video_target_num_patches = data_config.video_target_num_patches
    if type(processor).__name__ in NEMOTRON_VIDEO_PROCESSOR_NAMES:
        if video_target_num_patches is None:
            raise ValueError(
                "Nemotron Gym video data requires video_target_num_patches."
            )
        processed = process_nemotron_video_frames(
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
            min_generation_tokens = int(
                _require_video_config_value(
                    data_config.min_generation_tokens,
                    "min_generation_tokens",
                )
            )
            prompt_length = _compute_dynamic_prompt_length(
                processor, hf_messages, template_kwargs
            )
            if prompt_length is not None:
                processor_kwargs["num_tokens_available"] = (
                    max_seq_length - prompt_length - min_generation_tokens
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
    is_nemotron_video = type(processor).__name__ in NEMOTRON_VIDEO_PROCESSOR_NAMES
    if is_nemotron_video and video_path is None:
        _replace_cached_video_frames_with_native_video(extra_env_info)
    _strip_local_media_metadata(extra_env_info)
    # vLLM's native Nano-Nemotron processor consumes the video modality
    # directly and reads its temporal/dynamic-resolution settings from the
    # checkpoint config. Its constructor rejects the legacy ``video_as_images``
    # kwarg. Keep that compatibility path only for processors that still expect
    # frame-as-image grouping.
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
