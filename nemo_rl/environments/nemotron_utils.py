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
import math
from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig

NEMOTRON_VIDEO_PROCESSOR_NAMES = frozenset(
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
def load_nemotron_video_model_config(model_name: str) -> Any:
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


def _resize_and_normalize_nemotron_video_frame(
    frame: Image.Image,
    *,
    target_height: int,
    target_width: int,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
) -> torch.Tensor:
    """Resize one frame with the same numeric order as stock vLLM."""
    frame_array = np.array(
        frame.convert("RGB") if frame.mode != "RGB" else frame,
        dtype=np.uint8,
        copy=True,
    )
    frame_tensor = (
        torch.from_numpy(np.expand_dims(frame_array, axis=0))
        .permute(0, 3, 1, 2)
        .to(dtype=torch.float32)
    )
    if frame_tensor.shape[-2:] != (target_height, target_width):
        frame_tensor = F.interpolate(
            frame_tensor,
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    normalized = (frame_tensor / 255.0 - norm_mean) / norm_std
    return normalized.squeeze(0).contiguous()


def _flatten_nemotron_video_frame_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[Image.Image], list[int], float]:
    """Replace locally decoded frame items with ordered ``<image>`` markers."""
    flattened_messages = []
    frames = []
    frame_indices = []
    frame_fps: float | None = None
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
                frame_index = part.get("_video_frame_index")
                fps = part.get("_video_fps")
                if type(frame_index) is not int or frame_index < 0:
                    raise ValueError(
                        "Nemotron video frames require a non-negative "
                        "_video_frame_index."
                    )
                fps_value = float(fps) if isinstance(fps, (int, float)) else 0.0
                if not math.isfinite(fps_value) or fps_value <= 0:
                    raise ValueError(
                        "Nemotron video frames require a positive _video_fps."
                    )
                if frame_fps is None:
                    frame_fps = fps_value
                elif not math.isclose(frame_fps, fps_value):
                    raise ValueError(
                        "All frames from one Nemotron video must use one fps value."
                    )
                frame_indices.append(frame_index)
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
    return flattened_messages, frames, frame_indices, frame_fps or 0.0


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
    frame_indices: list[int],
    fps: float,
    temporal_patch_size: int,
) -> str:
    """Match vLLM's timestamped one-wrapper-per-tubelet video replacement."""
    if temporal_patch_size < 1:
        raise ValueError("video_temporal_patch_size must be at least 1.")
    if fps <= 0:
        raise ValueError("Nemotron video placeholder expansion requires positive fps.")
    parts = rendered_text.split("<image>")
    frame_count = len(parts) - 1
    if len(frame_indices) != frame_count:
        raise ValueError(
            "Rendered Nemotron video prompt/frame-index mismatch: "
            f"found {frame_count} markers and {len(frame_indices)} indices."
        )
    expected_tubelets = math.ceil(frame_count / temporal_patch_size)
    if len(embeddings_per_tubelet) != expected_tubelets:
        raise ValueError(
            "Rendered Nemotron video prompt/frame mismatch: "
            f"found {frame_count} <image> markers for "
            f"{len(embeddings_per_tubelet)} tubelets; expected "
            f"{expected_tubelets} tubelets with temporal patch size "
            f"{temporal_patch_size}."
        )
    if any(fragment.strip() for fragment in parts[1:-1]):
        raise ValueError(
            "Nemotron video frame placeholders must form one contiguous block."
        )

    tubelet_replacements = []
    frame_duration_ms = int(1000.0 / fps)
    for tubelet_index, first_frame in enumerate(
        range(0, frame_count, temporal_patch_size)
    ):
        descriptions = []
        for offset in range(temporal_patch_size):
            frame_position = first_frame + offset
            if frame_position >= frame_count:
                break
            frame_label = "Frame" if offset == 0 else "frame"
            timestamp = frame_indices[frame_position] * frame_duration_ms / 1000.0
            descriptions.append(
                f"{frame_label} {frame_position + 1} sampled at {timestamp:.2f} seconds"
            )
        wrapper = "<img>" + "<image>" * embeddings_per_tubelet[tubelet_index] + "</img>"
        tubelet_replacements.append(" and ".join(descriptions) + ": " + wrapper)

    replacement = "\n".join(tubelet_replacements)
    return parts[0] + replacement + parts[-1]


def process_nemotron_video_frames(
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
    model_config = load_nemotron_video_model_config(model_name)
    patch_size = int(_required_config_value(model_config, "patch_size"))
    downsample_ratio = float(_required_config_value(model_config, "downsample_ratio"))
    norm_mean = torch.tensor(
        _required_config_value(model_config, "norm_mean"), dtype=torch.float32
    ).view(3, 1, 1)
    norm_std = torch.tensor(
        _required_config_value(model_config, "norm_std"), dtype=torch.float32
    ).view(3, 1, 1)

    flattened_messages, frames, frame_indices, frame_fps = (
        _flatten_nemotron_video_frame_messages(messages)
    )
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
        pixel_values.append(
            _resize_and_normalize_nemotron_video_frame(
                frame,
                target_height=target_height,
                target_width=target_width,
                norm_mean=norm_mean,
                norm_std=norm_std,
            )
        )
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
        frame_indices=frame_indices,
        fps=frame_fps,
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
