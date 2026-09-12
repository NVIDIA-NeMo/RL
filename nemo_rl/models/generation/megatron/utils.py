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

from dataclasses import replace
from typing import Any, Callable

import torch
from megatron.core.inference.config import ImageProcessingConfig, VideoProcessingConfig
from megatron.core.inference.utils import device_memory_summary


def sample_vision_tensors(data, index: int):
    """Return one sample's vision tensors from RL PackedTensors."""
    from nemo_rl.data.multimodal_utils import PackedTensor

    pixel_values = data.get("pixel_values")
    imgs_sizes = data.get("imgs_sizes")
    packed_num_frames = data.get("num_frames")
    if pixel_values is None and imgs_sizes is None:
        if packed_num_frames is not None:
            raise ValueError("num_frames was provided without vision tensors.")
        return None, None, None
    if pixel_values is None or imgs_sizes is None:
        raise ValueError(
            "Megatron image generation requires both pixel_values and imgs_sizes."
        )
    if not isinstance(pixel_values, PackedTensor) or not isinstance(
        imgs_sizes, PackedTensor
    ):
        raise TypeError(
            "Megatron image generation expects pixel_values and imgs_sizes "
            "as per-sample PackedTensor values."
        )
    if packed_num_frames is not None and not isinstance(
        packed_num_frames, PackedTensor
    ):
        raise TypeError(
            "Megatron video generation expects num_frames as a "
            "per-sample PackedTensor value."
        )

    # `.tensors` is the physical segment list and only matches logical row
    # indices while media are not deduplicated.
    for name, packed in (
        ("pixel_values", pixel_values),
        ("imgs_sizes", imgs_sizes),
        ("num_frames", packed_num_frames),
    ):
        if packed is not None and packed._row_offsets is not None:
            raise ValueError(
                f"Megatron generation cannot index deduplicated {name}; "
                "set deduplicate_multimodal_data=false (it is only supported "
                "for the vLLM backend)."
            )

    imgs = pixel_values.tensors[index]
    sizes = imgs_sizes.tensors[index]
    num_frames = (
        packed_num_frames.tensors[index] if packed_num_frames is not None else None
    )
    if imgs is None and sizes is None:
        return None, None, None
    if imgs is None or sizes is None:
        raise ValueError(
            "Megatron image generation requires matching per-sample "
            "pixel_values and imgs_sizes."
        )
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(0)
    if sizes.ndim == 1:
        sizes = sizes.unsqueeze(0)
    if num_frames is not None:
        num_frames = num_frames.to(dtype=torch.int32).reshape(-1)
    return imgs, sizes, num_frames


def build_prompt_and_multimodal_data(
    data,
    index: int,
    *,
    supports_modality: Callable[[str], bool],
    sample_tensors: Callable = sample_vision_tensors,
):
    """Build one pre-expanded token prompt and optional MCore media payload."""
    length = int(data["input_lengths"][index].item())
    prompt = data["input_ids"][index, :length].tolist()
    imgs, imgs_sizes, num_frames = sample_tensors(data, index)
    if imgs is None:
        return prompt, None

    assert imgs_sizes is not None
    is_video = num_frames is not None and bool(torch.any(num_frames > 1).item())
    modality = "video" if is_video else "image"
    if not supports_modality(modality):
        raise ValueError(
            f"The configured megatron_inference_wrapper does not support "
            f"{modality} inputs."
        )
    if is_video:
        if int(num_frames.sum().item()) != int(imgs_sizes.shape[0]):
            raise ValueError(
                "Video num_frames must partition imgs_sizes exactly: "
                f"sum(num_frames)={int(num_frames.sum().item())}, "
                f"imgs_sizes={imgs_sizes.shape[0]}."
            )
        modality_data = {
            "imgs": imgs,
            "imgs_sizes": imgs_sizes,
            "num_frames": num_frames,
        }
    else:
        modality_data = {"imgs": imgs, "imgs_sizes": imgs_sizes}
    return prompt, {
        modality: modality_data,
        "media_tokens_preexpanded": True,
    }


def build_image_preprocessing_config(
    image_processor: Any,
    *,
    dynamic_resolution: bool | None = None,
    vision_model_type: str | None = None,
) -> ImageProcessingConfig:
    """Translate an HF image processor to an MCore config.

    Args:
        image_processor: HF image processor to read patch/normalization fields from.
        dynamic_resolution: Override for `ImageProcessingConfig.dynamic_resolution`.
            `None` leaves MCore's own default in place.
        vision_model_type: Override for `ImageProcessingConfig.vision_model_type`.
            `None` leaves MCore's own default in place.
    """

    def read(*names: str) -> Any:
        for name in names:
            value = getattr(image_processor, name, None)
            if value is not None:
                return value
        return None

    patch_dim = read("patch_size", "patch_dim")
    if isinstance(patch_dim, dict):
        patch_dim = patch_dim.get("height", patch_dim.get("width"))
    min_patches = read("min_num_patches")
    max_patches = read("max_num_patches")
    pixel_mean = read("norm_mean", "image_mean")
    pixel_std = read("norm_std", "image_std")

    if (
        patch_dim is None
        or min_patches is None
        or max_patches is None
        or pixel_mean is None
        or pixel_std is None
    ):
        missing = [
            name
            for name, value in (
                ("patch_size", patch_dim),
                ("min_num_patches", min_patches),
                ("max_num_patches", max_patches),
                ("norm_mean", pixel_mean),
                ("norm_std", pixel_std),
            )
            if value is None
        ]
        raise ValueError(
            f"{type(image_processor).__name__} does not expose {', '.join(missing)}, "
            "so MCore cannot preprocess raw images the way this model's data "
            "pipeline does."
        )

    downsample_ratio = read("downsample_ratio")
    if downsample_ratio is not None:
        merge_size = int(round(1.0 / float(downsample_ratio)))
    else:
        merge_size = int(read("merge_size", "spatial_merge_size") or 1)

    return ImageProcessingConfig(
        patch_dim=int(patch_dim),
        **(
            {}
            if dynamic_resolution is None
            else {"dynamic_resolution": dynamic_resolution}
        ),
        **(
            {}
            if vision_model_type is None
            else {"vision_model_type": vision_model_type}
        ),
        use_tiling=False,
        pixel_shuffle=merge_size > 1,
        spatial_merge_size=merge_size,
        dynamic_resolution_min_patches=int(min_patches),
        dynamic_resolution_max_patches=int(max_patches),
        pixel_mean=[float(value) for value in pixel_mean],
        pixel_std=[float(value) for value in pixel_std],
    )


def build_video_preprocessing_config(
    image_config: ImageProcessingConfig | None,
    generation_config: dict[str, Any],
    *,
    frame_manifest_magic: bytes,
) -> VideoProcessingConfig | None:
    """Build video preprocessing when explicitly enabled by generation config."""
    video_num_frames = generation_config.get("video_num_frames")
    if image_config is None or video_num_frames is None:
        return None

    # Video configs.
    video_kwargs: dict[str, Any] = {}
    if "video_temporal_patch_size" in generation_config:
        video_kwargs["temporal_patch_size"] = int(
            generation_config["video_temporal_patch_size"]
        )
    if "video_maintain_aspect_ratio" in generation_config:
        video_kwargs["video_maintain_aspect_ratio"] = bool(
            generation_config["video_maintain_aspect_ratio"]
        )

    target_num_patches = generation_config.get("video_target_num_patches")
    if target_num_patches is not None:
        image_config = replace(
            image_config,
            dynamic_resolution_max_patches=int(target_num_patches),
        )

    return VideoProcessingConfig(
        image_config=image_config,
        num_frames=int(video_num_frames),
        frame_manifest_magic=frame_manifest_magic,
        **video_kwargs,
    )


def resolve_torch_dtype(val):
    """Convert a value to `torch.dtype`."""
    if isinstance(val, torch.dtype):
        return val
    if isinstance(val, str):
        name = val.replace("torch.", "")
        dtype = getattr(torch, name, None)
        if isinstance(dtype, torch.dtype):
            return dtype
    raise ValueError(
        f"Cannot resolve torch dtype from {val!r} (type {type(val).__name__}). "
        f"Expected a torch.dtype or a string like 'torch.float32' / 'float32'."
    )


def log_gpu_memory(tag: str) -> None:
    """Print a one-line GPU-memory summary for the calling rank."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    print(f"[GPU Rank {rank}] {tag} | {device_memory_summary()}")
