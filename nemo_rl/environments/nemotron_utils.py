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
from collections.abc import Sequence
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig

from nemo_rl.data.multimodal_utils import (
    _image_num_tokens_from_processed,
    uses_image_placeholder,
)

NEMOTRON_VIDEO_PROCESSOR_NAMES = frozenset(
    {
        "NemotronNanoVLV2Processor",
        "NemotronH_Nano_Omni_Reasoning_V3Processor",
        "NemotronH_Omni_Reasoning_V3Processor",
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


# --- Rollout/train image-tiling parity helpers (Nemotron placeholder-style
# processors). The rollout engine expands each image to <img><image>*N</img>;
# these helpers read those runs back as the authoritative per-image budgets
# and force the training-side processor to reproduce them exactly.


def image_placeholder_token_ids_from_tokenizer(
    tokenizer: Any,
    *,
    image_token: str = "<image>",
    image_start_token: str = "<img>",
    image_end_token: str = "</img>",
) -> "tuple[int, int, int] | None":
    """(image_token_id, image_start_id, image_end_id), or None if unavailable."""
    if tokenizer is not None and not hasattr(tokenizer, "convert_tokens_to_ids"):
        # Callers sometimes hold a processor rather than a bare tokenizer.
        tokenizer = getattr(tokenizer, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "convert_tokens_to_ids"):
        return None
    ids = tokenizer.convert_tokens_to_ids(
        [image_token, image_start_token, image_end_token]
    )
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if any(i is None or (unk_id is not None and i == unk_id) for i in ids):
        return None
    return int(ids[0]), int(ids[1]), int(ids[2])


def get_image_placeholder_token_ids(processor: Any) -> "tuple[int, int, int] | None":
    """(image_token_id, image_start_id, image_end_id), or None if unavailable."""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or not uses_image_placeholder(processor):
        return None
    return image_placeholder_token_ids_from_tokenizer(
        tokenizer,
        image_token=getattr(processor, "image_token", "<image>"),
        image_start_token=getattr(processor, "image_start_token", "<img>"),
        image_end_token=getattr(processor, "image_end_token", "</img>"),
    )


def supports_image_placeholder_run_parity(processor: Any) -> bool:
    """Whether rollout tokens can be parsed into per-image placeholder runs."""
    return get_image_placeholder_token_ids(processor) is not None


def count_image_placeholder_runs(
    token_ids: "Sequence[int] | torch.Tensor", processor: Any
) -> list[int]:
    """Per-image placeholder run lengths from rollout token ids, in order.

    The rollout engine (and this processor) expand each image to
    ``<img><image>*N</img>``; each run's N is the exact number of projected
    media features the model expects for that image.
    """
    placeholder_ids = get_image_placeholder_token_ids(processor)
    if placeholder_ids is None:
        raise ValueError(
            f"{type(processor).__name__} does not expose image placeholder "
            "tokens; cannot count image placeholder runs."
        )
    return placeholder_runs_from_token_ids(token_ids, placeholder_ids)


def placeholder_runs_from_token_ids(
    token_ids: "Sequence[int] | torch.Tensor",
    placeholder_ids: "tuple[int, int, int]",
) -> list[int]:
    """Per-media placeholder run lengths from token ids, in order."""
    image_id, start_id, end_id = placeholder_ids
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    runs: list[int] = []
    current: "int | None" = None
    for token in token_ids:
        if token == start_id:
            if current is not None:
                raise ValueError("Nested <img> region in rollout tokens.")
            current = 0
        elif token == end_id:
            if current is None:
                raise ValueError("Unmatched </img> in rollout tokens.")
            runs.append(current)
            current = None
        elif token == image_id:
            if current is None:
                raise ValueError(
                    "Image placeholder token outside an <img>...</img> region "
                    "in rollout tokens."
                )
            current += 1
    if current is not None:
        raise ValueError("Unterminated <img> region in rollout tokens.")
    return runs


def _is_static_video_turn(message: dict[str, Any]) -> bool:
    """Whether a source user turn carries a statically-preprocessed video.

    Still-image turns also carry ``num_frames`` (one ``1`` per image, added by
    the image attach path); a video turn is the one whose frame count exceeds 1.
    """
    num_frames = message.get("num_frames")
    tensors = getattr(num_frames, "tensors", None)
    if tensors is None:
        return False
    return any(
        tensor is not None and bool((tensor > 1).any()) for tensor in tensors
    )


def verify_static_video_media_alignment(
    source_message: dict[str, Any],
    target_message: dict[str, Any],
    tokenizer: Any,
) -> None:
    """Verify rollout tokens match the static video tensors before attach.

    The static video datum expands its video to one ``<img><image>*k</img>``
    run per tubelet, and its tensors project exactly ``sum(k)`` features. The
    rollout engine performs its own expansion; if the two disagree, training
    would crash deep in Megatron with an unattributable "media alignment
    failed". Compare the run structures here and raise a diagnostic that names
    the mismatch (tubelet count vs per-tubelet token count) instead.

    A no-op for non-video turns or when placeholder ids cannot be resolved.
    """
    if not _is_static_video_turn(source_message):
        return
    placeholder_ids = image_placeholder_token_ids_from_tokenizer(tokenizer)
    if placeholder_ids is None:
        return
    source_token_ids = source_message.get("token_ids")
    target_token_ids = target_message.get("token_ids")
    if source_token_ids is None or target_token_ids is None:
        return
    source_runs = placeholder_runs_from_token_ids(source_token_ids, placeholder_ids)
    target_runs = placeholder_runs_from_token_ids(target_token_ids, placeholder_ids)
    if source_runs == target_runs:
        return
    first_mismatch = next(
        (
            i
            for i, (s, t) in enumerate(zip(source_runs, target_runs))
            if s != t
        ),
        min(len(source_runs), len(target_runs)),
    )
    num_frames = source_message.get("num_frames")
    frame_counts = [
        tensor.tolist()
        for tensor in getattr(num_frames, "tensors", [])
        if tensor is not None
    ]
    imgs_sizes = source_message.get("imgs_sizes")
    frame_sizes = [
        tensor[0].tolist() if len(tensor) else None
        for tensor in getattr(imgs_sizes, "tensors", [])
        if tensor is not None
    ]
    raise ValueError(
        "Rollout/video media alignment failed: the rollout's placeholder "
        "expansion disagrees with the statically-preprocessed video tensors "
        "about to be attached for training. "
        f"static: {len(source_runs)} tubelet runs, "
        f"{sum(source_runs)} placeholder tokens total, "
        f"run lengths {sorted(set(source_runs))}; "
        f"rollout: {len(target_runs)} tubelet runs, "
        f"{sum(target_runs)} placeholder tokens total, "
        f"run lengths {sorted(set(target_runs))}; "
        f"first mismatching run index {first_mismatch} "
        f"(static {source_runs[first_mismatch] if first_mismatch < len(source_runs) else 'absent'} "
        f"vs rollout {target_runs[first_mismatch] if first_mismatch < len(target_runs) else 'absent'}); "
        f"num_frames={frame_counts}, frame size (h, w)={frame_sizes}. "
        "Equal run counts with different lengths mean vLLM resolved a "
        "different per-frame token grid (target_num_patches / aspect-ratio "
        "settings); different run counts mean a frame/tubelet sampling "
        "mismatch. Refusing to train on misaligned media."
    )


def predicted_static_image_num_tokens(
    processor: Any, image_sizes: list[tuple[int, int]]
) -> "list[int] | None":
    """Predict per-image token counts of the processor's static image path.

    Mirrors the budget selection in the checkpoint's image processor
    ``_preprocess`` (image path) and reuses its own ``_compute_target_patches``
    for the grid math, so it costs no pixel work. Returns None when the
    processor does not expose the needed hooks (callers must then assume the
    static output is unverified and attach rollout-matched media themselves).
    """
    image_processor = getattr(processor, "image_processor", None)
    required = (
        "max_model_len",
        "min_num_patches",
        "max_num_patches",
        "_compute_target_patches",
    )
    if image_processor is None or not all(
        hasattr(image_processor, name) for name in required
    ):
        return None
    downsample = getattr(image_processor, "_downsample_factor", None)
    if not downsample:
        ratio = getattr(image_processor, "downsample_ratio", None)
        if not ratio:
            return None
        downsample = int(round(1.0 / ratio))
    budget = (image_processor.max_model_len - 4) * downsample**2
    budget = max(budget, image_processor.min_num_patches * len(image_sizes))
    max_patches = image_processor.max_num_patches
    max_budget = max_patches if (max_patches and max_patches > 0) else float("inf")
    per_image_budget = max(min(budget, max_budget), image_processor.min_num_patches)
    num_tokens = []
    for width, height in image_sizes:
        shim = SimpleNamespace(width=width, height=height)
        grid_w, grid_h = image_processor._compute_target_patches(shim, per_image_budget)
        num_tokens.append((grid_w * grid_h) // downsample**2)
    return num_tokens


def _closest_aspect_grid(
    target_patches: int, height: int, width: int, divisor: int
) -> tuple[int, int]:
    """Aspect-closest (grid_h, grid_w), both divisible by ``divisor``, whose product is exactly ``target_patches``."""
    units = divisor * divisor
    if target_patches <= 0 or target_patches % units:
        raise ValueError(
            f"Rollout image placeholder run implies {target_patches} patches, "
            f"which is not a positive multiple of the pixel-shuffle factor "
            f"squared ({units}); the rollout tokens do not describe a valid "
            "image tile."
        )
    base = target_patches // units
    aspect = math.log(max(height, 1) / max(width, 1))
    best: "tuple[float, int, int] | None" = None
    for low in range(1, math.isqrt(base) + 1):
        if base % low:
            continue
        high = base // low
        for h_units, w_units in ((low, high), (high, low)):
            grid_h, grid_w = h_units * divisor, w_units * divisor
            score = abs(math.log(grid_h / grid_w) - aspect)
            if best is None or score < best[0]:
                best = (score, grid_h, grid_w)
    assert best is not None
    return best[1], best[2]


def _process_single_image_at_num_tokens(
    processor: Any, image: Image.Image, num_tokens: int
) -> dict[str, Any]:
    """Run the processor on one image, pinned to an exact media token count."""
    image_processor = processor.image_processor
    image_token = getattr(processor, "image_token", "<image>")
    downsample = getattr(image_processor, "_downsample_factor", None) or int(
        round(1.0 / image_processor.downsample_ratio)
    )

    def run_pinned(single_image: Image.Image) -> dict[str, Any]:
        # The image-path per-image budget is clamp((max_model_len-4)*ds^2, ...),
        # so max_model_len = num_tokens + 4 requests exactly num_tokens*ds^2
        # patches. Instance mutation is the only knob: the wrapper's
        # ImagesKwargs silently ignore unknown kwargs. Both attach call sites
        # run this on a single thread with no awaits in between (same pattern
        # as the processor's own _is_video_mode flag).
        original = image_processor.max_model_len
        try:
            image_processor.max_model_len = num_tokens + 4
            return dict(
                processor(text=image_token, images=[single_image], return_tensors=None)
            )
        finally:
            image_processor.max_model_len = original

    processed = run_pinned(image)
    if _image_num_tokens_from_processed(processed) == [num_tokens]:
        return processed

    # Grid rounding is not idempotent for every (size, budget); force the exact
    # grid by pre-resizing to it. An even grid of exactly num_tokens*ds^2
    # patches passes _compute_target_patches unchanged under the pinned budget.
    grid_h, grid_w = _closest_aspect_grid(
        num_tokens * downsample * downsample, image.height, image.width, downsample
    )
    patch = image_processor.patch_size
    resized = image.resize((grid_w * patch, grid_h * patch), Image.BICUBIC)
    processed = run_pinned(resized)
    actual = _image_num_tokens_from_processed(processed)
    if actual != [num_tokens]:
        raise ValueError(
            "Cannot match the rollout's image tiling: the rollout expanded a "
            f"{image.width}x{image.height} image to {num_tokens} placeholder "
            f"tokens, but the training processor produced {actual[0]} tokens "
            f"even when pinned to a {grid_h}x{grid_w} patch grid. Refusing to "
            "train on misaligned media."
        )
    return processed


def reprocess_images_at_rollout_budgets(
    processor: Any, images: list[Image.Image], expected_num_tokens: list[int]
) -> dict[str, Any]:
    """Re-run the processor per image at the rollout's exact per-image budget."""
    parts = [
        _process_single_image_at_num_tokens(processor, image, count)
        for image, count in zip(images, expected_num_tokens, strict=True)
    ]
    pixel_values: list[torch.Tensor] = []
    imgs_sizes: list[list[int]] = []
    input_ids: list[torch.Tensor] = []
    for part in parts:
        tile = part["pixel_values"]
        if isinstance(tile, list):
            tile = torch.as_tensor(tile[0])
        else:
            tile = torch.as_tensor(tile)
            if tile.ndim == 4:
                tile = tile[0]
        pixel_values.append(tile)
        imgs_sizes.extend(
            [int(h), int(w)]
            for h, w in torch.as_tensor(part["imgs_sizes"]).reshape(-1, 2).tolist()
        )
        input_ids.append(torch.as_tensor(part["input_ids"]).reshape(1, -1))
    merged: dict[str, Any] = {
        "input_ids": torch.cat(input_ids, dim=1),
        "imgs_sizes": torch.tensor(imgs_sizes, dtype=torch.long),
        "num_tokens": list(expected_num_tokens),
        "num_patches": [1] * len(images),
    }
    merged["pixel_values"] = (
        pixel_values[0].unsqueeze(0) if len(pixel_values) == 1 else pixel_values
    )
    return merged
