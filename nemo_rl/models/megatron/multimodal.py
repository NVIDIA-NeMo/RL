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

import math
from typing import TYPE_CHECKING, Any, Optional

import torch
from einops import rearrange
from nemo_rl.data.multimodal_utils import PackedTensor

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams


def _get_num_embeddings_from_sizes(
    imgs_sizes: torch.Tensor,
    patch_dim: int,
    downsample_ratio: float,
    class_token_len: int = 0,
) -> torch.Tensor:
    patches_per_image = (imgs_sizes[:, 0] // patch_dim) * (
        imgs_sizes[:, 1] // patch_dim
    )
    seq_len = patches_per_image + class_token_len
    return (seq_len * (downsample_ratio**2)).int()


def is_llava_model(model) -> bool:
    """Check if the model is a LLaVA model.

    Args:
        model: The model to check

    Returns:
        True if the model is a LLaVA model, False otherwise
    """
    # Handle wrapped models (e.g., DDP, Float16Module)
    actual_model = model
    while hasattr(actual_model, "module"):
        actual_model = actual_model.module
    # Check for core LLaVAModel
    try:
        from megatron.core.models.multimodal.llava_model import LLaVAModel

        if isinstance(actual_model, LLaVAModel):
            return True
    except ImportError:
        pass
    # Check for MIMO-based LLaVA models (Megatron-Bridge)
    # These wrap a LLaVA-style model but aren't a direct LLaVAModel subclass.
    # Detect by checking for the llava_model attribute or image token handling.
    if hasattr(actual_model, "llava_model"):
        return True
    if hasattr(actual_model, "img_start_token_id") and hasattr(
        actual_model, "img_end_token_id"
    ):
        return True
    config = getattr(actual_model, "config", None)
    if (
        config is not None
        and hasattr(config, "img_start_token_id")
        and hasattr(config, "img_end_token_id")
    ):
        return True
    return False


def _get_model_config(model: Any) -> tuple[int, float, int, bool, Optional[int]]:
    """Extract the vision expansion parameters from a wrapped LLaVA model."""
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model

    vision_model = getattr(inner, "vision_model", None)
    patch_dim = getattr(vision_model, "patch_dim", 16)

    downsample_ratio = getattr(inner, "downsample_ratio", None)
    if downsample_ratio is None:
        pixel_shuffle = getattr(inner, "_pixel_shuffle", False)
        conv_merging = getattr(inner, "_use_conv_merging", False)
        downsample_ratio = 1.0
        if pixel_shuffle:
            downsample_ratio *= 0.5
        if conv_merging:
            downsample_ratio *= 0.5
        if not pixel_shuffle and not conv_merging:
            downsample_ratio = 0.5

    drop_vision_class_token = getattr(inner, "_drop_vision_class_token", True)
    class_token_len = (
        0
        if drop_vision_class_token
        else getattr(
            inner,
            "_class_token_len",
            getattr(vision_model, "class_token_len", 1),
        )
    )
    dynamic_resolution = bool(
        getattr(
            inner,
            "dynamic_resolution",
            getattr(
                inner,
                "_dynamic_resolution",
                getattr(vision_model, "dynamic_resolution", False),
            ),
        )
    )
    static_img_seq_len = (
        getattr(inner, "img_seq_len", None) if not dynamic_resolution else None
    )

    return (
        int(patch_dim),
        float(downsample_ratio),
        int(class_token_len),
        bool(dynamic_resolution),
        None if static_img_seq_len is None else int(static_img_seq_len),
    )


def _resolve_packed_per_sample(
    value: Any,
    batch_size: int,
    counts: Optional[list[int]] = None,
) -> list[Optional[torch.Tensor]]:
    """Resolve PackedTensor/flat tensors to one tensor per logical sample."""
    if value is None:
        return [None] * batch_size
    if isinstance(value, PackedTensor):
        if getattr(value, "_dedup_indices", None) is not None:
            resolved = [value.tensors[i] for i in value._dedup_indices[:batch_size]]
        else:
            resolved = list(value.tensors[:batch_size])
        return resolved + [None] * max(0, batch_size - len(resolved))
    if isinstance(value, torch.Tensor):
        if value.ndim >= 3 and value.shape[0] >= batch_size:
            return [value[b] for b in range(batch_size)]
        if value.ndim == 2 and counts is not None:
            resolved: list[Optional[torch.Tensor]] = []
            offset = 0
            for count in counts:
                if count > 0 and offset + count <= value.shape[0]:
                    resolved.append(value[offset : offset + count])
                    offset += count
                else:
                    resolved.append(None)
            return resolved
        return [value] + [None] * max(0, batch_size - 1)
    if isinstance(value, (list, tuple)):
        resolved = [
            item if isinstance(item, torch.Tensor) else torch.tensor(item)
            for item in value[:batch_size]
        ]
        return resolved + [None] * max(0, batch_size - len(resolved))
    return [None] * batch_size


def _as_int_tensor(value: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value
    else:
        result = torch.tensor(value, dtype=torch.int32)
    result = result.to(dtype=torch.int32)
    if device is not None:
        result = result.to(device=device)
    return result


def _tubelet_img_sizes(
    row_imgs_sizes: torch.Tensor,
    row_num_frames: Optional[torch.Tensor],
    temporal_patch_size: int,
) -> torch.Tensor:
    """Mirror RADIO temporal grouping: one size entry per image/tubelet."""
    if temporal_patch_size <= 1 or row_num_frames is None:
        return row_imgs_sizes

    frame_groups = _as_int_tensor(row_num_frames).flatten().tolist()
    if (
        not frame_groups
        or sum(max(0, int(frames)) for frames in frame_groups) > row_imgs_sizes.shape[0]
    ):
        return row_imgs_sizes

    tubelet_sizes: list[torch.Tensor] = []
    offset = 0
    for frames_value in frame_groups:
        frames = max(0, int(frames_value))
        if frames == 0:
            continue
        if frames <= 1:
            tubelet_sizes.extend(row_imgs_sizes[offset : offset + frames])
        else:
            for frame_idx in range(0, frames, temporal_patch_size):
                source_idx = offset + frame_idx
                if source_idx < row_imgs_sizes.shape[0]:
                    tubelet_sizes.append(row_imgs_sizes[source_idx])
        offset += frames

    if offset < row_imgs_sizes.shape[0]:
        tubelet_sizes.extend(row_imgs_sizes[offset:])

    if not tubelet_sizes:
        return row_imgs_sizes
    return torch.stack(tubelet_sizes, dim=0)


def _expand_tubelet_values_to_frame_placeholders(
    values: list[int],
    row_num_frames: Optional[torch.Tensor],
    num_placeholders: int,
    temporal_patch_size: int,
    *,
    secondary_value: int = 0,
) -> Optional[list[int]]:
    """Expand tubelet-level values to raw frame placeholder positions."""
    if temporal_patch_size <= 1 or row_num_frames is None or num_placeholders <= 0:
        return None

    frame_groups = _as_int_tensor(row_num_frames).flatten().tolist()
    frame_groups = [max(0, int(frames)) for frames in frame_groups]
    if not frame_groups:
        return None

    raw_frame_count = sum(frame_groups)
    if raw_frame_count != num_placeholders:
        recovered_frame_groups = [
            frames if frames <= 1 else frames * temporal_patch_size
            for frames in frame_groups
        ]
        recovered_raw_count = sum(recovered_frame_groups)
        if recovered_raw_count == num_placeholders:
            frame_groups = recovered_frame_groups
            raw_frame_count = sum(frame_groups)

    if raw_frame_count != num_placeholders:
        return None

    expected_tubelet_count = 0
    for frames in frame_groups:
        if frames <= 1:
            expected_tubelet_count += frames
        else:
            expected_tubelet_count += math.ceil(frames / temporal_patch_size)
    if expected_tubelet_count != len(values):
        return None

    expanded: list[int] = []
    value_idx = 0
    for frames in frame_groups:
        if frames <= 0:
            continue
        if frames <= 1:
            expanded.append(int(values[value_idx]))
            value_idx += 1
            continue
        for frame_idx in range(frames):
            if frame_idx % temporal_patch_size == 0:
                expanded.append(int(values[value_idx]))
                value_idx += 1
            else:
                expanded.append(int(secondary_value))

    if value_idx != len(values) or len(expanded) != num_placeholders:
        return None
    return expanded


def _raw_placeholder_count_from_num_frames(
    frames: Optional[torch.Tensor],
    placeholder_count: int,
    temporal_patch_size: int,
) -> int:
    if frames is None:
        return placeholder_count
    frame_groups = [
        max(0, int(value)) for value in _as_int_tensor(frames).flatten().tolist()
    ]
    explicit_count = sum(frame_groups)
    if explicit_count == placeholder_count or temporal_patch_size <= 1:
        return explicit_count

    recovered_count = sum(
        value if value <= 1 else value * temporal_patch_size for value in frame_groups
    )
    if recovered_count == placeholder_count:
        return placeholder_count
    return explicit_count


def _visual_embed_counts_for_placeholders(
    row_imgs_sizes: Optional[torch.Tensor],
    row_num_frames: Optional[torch.Tensor],
    num_placeholders: int,
    patch_dim: int,
    downsample_ratio: float,
    class_token_len: int,
    temporal_patch_size: int,
) -> list[int]:
    if row_imgs_sizes is None or num_placeholders <= 0:
        return []
    row_imgs_sizes = _as_int_tensor(row_imgs_sizes)
    if row_imgs_sizes.numel() == 0:
        return []
    if row_imgs_sizes.ndim == 1:
        row_imgs_sizes = row_imgs_sizes.view(-1, 2)

    grouped_sizes = _tubelet_img_sizes(
        row_imgs_sizes,
        row_num_frames,
        temporal_patch_size,
    )
    embeds = _get_num_embeddings_from_sizes(
        grouped_sizes,
        patch_dim,
        downsample_ratio,
        class_token_len,
    ).tolist()

    expanded_embeds = _expand_tubelet_values_to_frame_placeholders(
        [int(value) for value in embeds],
        row_num_frames,
        num_placeholders,
        temporal_patch_size,
        secondary_value=0,
    )
    if expanded_embeds is not None:
        embeds = expanded_embeds

    if len(embeds) != num_placeholders and row_imgs_sizes.shape[0] > num_placeholders:
        if row_imgs_sizes.shape[0] % num_placeholders == 0:
            group = row_imgs_sizes.shape[0] // num_placeholders
            inferred_sizes = row_imgs_sizes[::group]
            embeds = _get_num_embeddings_from_sizes(
                inferred_sizes,
                patch_dim,
                downsample_ratio,
                class_token_len,
            ).tolist()

    return [max(0, int(value)) for value in embeds[:num_placeholders]]


def compute_vision_expansion(
    imgs_sizes_per_sample: list[Optional[torch.Tensor]],
    num_image_placeholders_per_sample: list[int],
    patch_dim: int,
    downsample_ratio: float,
    class_token_len: int = 0,
    num_frames_per_sample: Optional[list[Optional[torch.Tensor]]] = None,
    temporal_patch_size: int = 1,
) -> list[int]:
    """Compute extra tokens produced when collapsed image placeholders expand."""
    expansions: list[int] = []
    for b, (imgs_sizes, n_placeholders) in enumerate(
        zip(imgs_sizes_per_sample, num_image_placeholders_per_sample)
    ):
        if imgs_sizes is None or n_placeholders == 0:
            expansions.append(0)
            continue
        embeds = _visual_embed_counts_for_placeholders(
            imgs_sizes,
            num_frames_per_sample[b] if num_frames_per_sample is not None else None,
            n_placeholders,
            patch_dim,
            downsample_ratio,
            class_token_len,
            temporal_patch_size,
        )
        expansions.append(max(0, int(sum(embeds)) - n_placeholders))
    return expansions


def compute_expanded_lengths(
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    imgs_sizes: Any,
    image_token_id: Optional[int],
    patch_dim: int = 16,
    downsample_ratio: float = 0.5,
    class_token_len: int = 0,
    num_frames: Any = None,
    temporal_patch_size: int = 1,
    max_length: Optional[int] = None,
    img_start_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Return per-sample expanded lengths for vision-aware packing."""
    batch_size = input_ids.shape[0]
    expanded = input_lengths.clone().to(torch.int64)

    if image_token_id is None or imgs_sizes is None:
        if max_length is not None:
            expanded.clamp_(max=max_length)
        return expanded

    if img_start_token_id is not None:
        num_placeholders = [
            int((input_ids[b] == img_start_token_id).sum().item())
            for b in range(batch_size)
        ]
    else:
        num_placeholders = [
            int((input_ids[b] == image_token_id).sum().item())
            for b in range(batch_size)
        ]

    collapse_savings = [0] * batch_size
    if img_start_token_id is not None:
        for b in range(batch_size):
            raw_image_count = int((input_ids[b] == image_token_id).sum().item())
            collapse_savings[b] = max(0, raw_image_count - num_placeholders[b])

    if isinstance(imgs_sizes, torch.Tensor):
        counts = None
        if img_start_token_id is not None:
            counts = [
                int((input_ids[b] == image_token_id).sum().item())
                for b in range(batch_size)
            ]
        imgs_sizes_per_sample = _resolve_packed_per_sample(
            imgs_sizes, batch_size, counts=counts
        )
    else:
        imgs_sizes_per_sample = _resolve_packed_per_sample(imgs_sizes, batch_size)

    num_frames_per_sample = (
        _resolve_packed_per_sample(num_frames, batch_size)
        if num_frames is not None
        else None
    )
    expansions = compute_vision_expansion(
        imgs_sizes_per_sample,
        num_placeholders,
        patch_dim,
        downsample_ratio,
        class_token_len,
        num_frames_per_sample=num_frames_per_sample,
        temporal_patch_size=temporal_patch_size,
    )

    for b in range(batch_size):
        expanded[b] = int(input_lengths[b].item()) - collapse_savings[b] + expansions[b]

    if max_length is not None:
        expanded.clamp_(max=max_length)
    return expanded


def _trim_image_data_for_truncated_sample(
    new_data_dict: dict[str, Any],
    b: int,
    surviving_image_count: int,
    original_image_count: int,
    temporal_patch_size: int,
) -> None:
    """Trim per-sample image tensors after text truncation drops image groups."""

    def _packed_logical_items(packed: PackedTensor) -> list[Optional[torch.Tensor]]:
        return (
            [packed.tensors[j] for j in packed._dedup_indices]
            if packed._dedup_indices is not None
            else list(packed.tensors)
        )

    original_raw_count: Optional[int] = None
    surviving_media_count = max(0, int(surviving_image_count))
    trimmed_frame_groups: Optional[list[int]] = None
    packed_num_frames = new_data_dict.get("num_frames")
    if isinstance(packed_num_frames, PackedTensor):
        per_sample_frames = _packed_logical_items(packed_num_frames)
        if 0 <= b < len(per_sample_frames) and per_sample_frames[b] is not None:
            frame_groups = _frame_groups_from_value(per_sample_frames[b])
            if frame_groups:
                original_raw_count = sum(frame_groups)
                trimmed_frame_groups, surviving_media_count = (
                    _trim_frame_groups_to_placeholder_count(
                        frame_groups,
                        surviving_image_count,
                        original_image_count,
                        temporal_patch_size,
                    )
                )

    tiles_to_keep_pvf: Optional[int] = None
    if "pixel_values_flat" in new_data_dict and "image_num_patches" in new_data_dict:
        image_num_patches = new_data_dict["image_num_patches"]
        if isinstance(image_num_patches, PackedTensor):
            per_sample = _packed_logical_items(image_num_patches)
            if 0 <= b < len(per_sample) and per_sample[b] is not None:
                kept_image_num_patches = _slice_tensor_to_media_count(
                    per_sample[b],
                    surviving_media_count,
                    dim_to_pack=image_num_patches.dim_to_pack,
                    original_count=original_raw_count,
                )
                if kept_image_num_patches is not None:
                    tiles_to_keep_pvf = int(kept_image_num_patches.sum().item())

    for key in ("pixel_values", "imgs_sizes", "image_num_patches"):
        packed = new_data_dict.get(key)
        if not isinstance(packed, PackedTensor):
            continue
        per_sample = _packed_logical_items(packed)
        if not (0 <= b < len(per_sample)) or per_sample[b] is None:
            continue
        old = per_sample[b]
        kept = _slice_tensor_to_media_count(
            old,
            surviving_media_count,
            dim_to_pack=packed.dim_to_pack,
            original_count=original_raw_count,
        )
        if kept is old:
            continue
        per_sample[b] = kept
        new_data_dict[key] = PackedTensor(per_sample, dim_to_pack=packed.dim_to_pack)

    if isinstance(packed_num_frames, PackedTensor) and trimmed_frame_groups is not None:
        per_sample_frames = _packed_logical_items(packed_num_frames)
        if 0 <= b < len(per_sample_frames) and per_sample_frames[b] is not None:
            old_frames = per_sample_frames[b]
            if surviving_media_count <= 0 or not trimmed_frame_groups:
                per_sample_frames[b] = None
            else:
                per_sample_frames[b] = torch.tensor(
                    trimmed_frame_groups,
                    dtype=old_frames.dtype,
                    device=old_frames.device,
                )
            new_data_dict["num_frames"] = PackedTensor(
                per_sample_frames, dim_to_pack=packed_num_frames.dim_to_pack
            )

    if tiles_to_keep_pvf is not None:
        packed_flat = new_data_dict.get("pixel_values_flat")
        if isinstance(packed_flat, PackedTensor):
            per_sample_flat = _packed_logical_items(packed_flat)
            if 0 <= b < len(per_sample_flat) and per_sample_flat[b] is not None:
                old_flat = per_sample_flat[b]
                if tiles_to_keep_pvf <= 0:
                    per_sample_flat[b] = None
                else:
                    kept_flat = _slice_tensor_to_media_count(
                        old_flat,
                        tiles_to_keep_pvf,
                        dim_to_pack=packed_flat.dim_to_pack,
                    )
                    if kept_flat is old_flat:
                        return
                    per_sample_flat[b] = kept_flat
                if per_sample_flat[b] is old_flat:
                    return
                new_data_dict["pixel_values_flat"] = PackedTensor(
                    per_sample_flat, dim_to_pack=packed_flat.dim_to_pack
                )


def truncate_for_expanded_budget(
    data_dict: dict[str, Any],
    max_seq_length: int,
    patch_dim: int = 16,
    downsample_ratio: float = 0.5,
    class_token_len: int = 0,
    pad_token_id: int = 0,
    image_token_id: Optional[int] = None,
    img_start_token_id: Optional[int] = None,
    temporal_patch_size: int = 1,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Truncate text tails so post-vision expanded sequence length fits."""
    input_ids = data_dict["input_ids"]
    input_lengths = data_dict.get("input_lengths")
    batch_size = input_ids.shape[0]
    truncated_mask = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    if image_token_id is None or input_lengths is None:
        return data_dict, truncated_mask

    imgs_sizes = data_dict.get("imgs_sizes")
    if img_start_token_id is not None:
        num_placeholders = [
            int((input_ids[b] == img_start_token_id).sum().item())
            for b in range(batch_size)
        ]
    else:
        num_placeholders = [
            int((input_ids[b] == image_token_id).sum().item())
            for b in range(batch_size)
        ]

    if imgs_sizes is None or not any(num_placeholders):
        return data_dict, truncated_mask

    if isinstance(imgs_sizes, torch.Tensor):
        counts = None
        if img_start_token_id is not None:
            counts = [
                int((input_ids[b] == image_token_id).sum().item())
                for b in range(batch_size)
            ]
        imgs_sizes_per_sample = _resolve_packed_per_sample(
            imgs_sizes, batch_size, counts=counts
        )
    else:
        imgs_sizes_per_sample = _resolve_packed_per_sample(imgs_sizes, batch_size)

    num_frames = data_dict.get("num_frames")
    num_frames_per_sample = (
        _resolve_packed_per_sample(num_frames, batch_size)
        if num_frames is not None
        else None
    )
    expansions = compute_vision_expansion(
        imgs_sizes_per_sample,
        num_placeholders,
        patch_dim,
        downsample_ratio,
        class_token_len,
        num_frames_per_sample=num_frames_per_sample,
        temporal_patch_size=temporal_patch_size,
    )

    collapse_savings = [0] * batch_size
    if img_start_token_id is not None:
        for b in range(batch_size):
            raw_image_count = int((input_ids[b] == image_token_id).sum().item())
            collapse_savings[b] = max(0, raw_image_count - num_placeholders[b])

    samples_to_truncate: list[tuple[int, int, int, int, int]] = []
    for b in range(batch_size):
        valid_len = int(input_lengths[b].item())
        expanded_len = valid_len - collapse_savings[b] + expansions[b]
        if expanded_len > max_seq_length:
            max_collapsed_len = max(
                0, max_seq_length - expansions[b] + collapse_savings[b]
            )
            if max_collapsed_len < valid_len:
                samples_to_truncate.append(
                    (b, valid_len, max_collapsed_len, expansions[b], expanded_len)
                )

    if not samples_to_truncate:
        return data_dict, truncated_mask

    new_data_dict = data_dict.copy()
    new_data_dict["input_ids"] = new_data_dict["input_ids"].clone()
    new_data_dict["input_lengths"] = new_data_dict["input_lengths"].clone()
    for key in (
        "token_mask",
        "advantages",
        "generation_logprobs",
        "reference_policy_logprobs",
        "prev_logprobs",
    ):
        if key in new_data_dict and new_data_dict[key].dim() >= 2:
            new_data_dict[key] = new_data_dict[key].clone()

    group_count_token_id = (
        img_start_token_id if img_start_token_id is not None else image_token_id
    )
    for (
        b,
        valid_len,
        max_collapsed_len,
        _expansion,
        _expanded_len,
    ) in samples_to_truncate:
        truncated_mask[b] = True
        new_data_dict["input_ids"][b, max_collapsed_len:] = pad_token_id
        new_data_dict["input_lengths"][b] = max_collapsed_len
        for key in (
            "token_mask",
            "advantages",
            "generation_logprobs",
            "reference_policy_logprobs",
            "prev_logprobs",
        ):
            if key in new_data_dict and new_data_dict[key].dim() >= 2:
                new_data_dict[key][b, max_collapsed_len:] = 0

        if group_count_token_id is not None:
            pre_ids = data_dict["input_ids"][b, :valid_len]
            post_ids = new_data_dict["input_ids"][b, :max_collapsed_len]
            original_n_imgs = int((pre_ids == group_count_token_id).sum().item())
            surviving_n_imgs = int((post_ids == group_count_token_id).sum().item())
            if surviving_n_imgs < original_n_imgs:
                _trim_image_data_for_truncated_sample(
                    new_data_dict,
                    b,
                    surviving_n_imgs,
                    original_n_imgs,
                    temporal_patch_size,
                )

    return new_data_dict, truncated_mask


def collapse_multimodal_tokens(data_dict: Any, model: Any) -> Any:
    """Collapse N image tokens to 1 token per image for Megatron LLaVA forward pass.

    vLLM uses N tokens per image (1:1 token-to-embedding), while Megatron uses 1 token
    per image/tile (1:N via imgs_sizes). This collapses <img><image>×N</img> to <img><image></img>.

    Processes the full padded sequence (not just valid content) so that after model forward,
    output length matches padded input length. Padding tokens (zeros) won't match image token
    IDs, so only content region gets collapsed while padding is preserved.
    """
    image_token_ids = _get_image_token_ids(model)
    image_payload_keys = (
        "pixel_values",
        "pixel_values_flat",
        "image_num_patches",
        "imgs_sizes",
    )
    has_image_payload = any(key in data_dict for key in image_payload_keys)
    if image_token_ids is None or not has_image_payload:
        return data_dict

    input_ids = data_dict["input_ids"]
    input_lengths = data_dict.get("input_lengths")
    token_mask = data_dict.get("token_mask")
    img_start_id, img_end_id = image_token_ids
    batch_size = input_ids.shape[0]

    # Check if image payload exists without image tokens. This happens when
    # all samples in a micro-batch were discarded (overlong).
    img_start_count = (input_ids == img_start_id).sum().item()
    img_end_count = (input_ids == img_end_id).sum().item()
    image_token_index = _get_image_token_index(model)

    if img_start_count == 0 and img_end_count == 0:
        if image_token_index is not None:
            has_collapsed_placeholders = (input_ids == image_token_index).any().item()
            if has_collapsed_placeholders:
                return data_dict
        # Drop stale multimodal keys and treat the batch as text-only.
        for key in image_payload_keys:
            data_dict.pop(key, None)
        return data_dict

    collapsed_list = []
    new_lengths = []
    tokens_removed_per_sample = []
    all_keep_masks = []

    for b in range(batch_size):
        # Process full padded sequence, not just valid content
        # Padding tokens (zeros) won't match image token IDs, so only content gets collapsed
        sample = input_ids[b]
        full_len = sample.shape[0]
        valid_len = input_lengths[b].item() if input_lengths is not None else full_len

        keep_mask = torch.ones(full_len, dtype=torch.bool, device=input_ids.device)
        for start_pos in (sample == img_start_id).nonzero(as_tuple=True)[0]:
            # The policy can emit literal media-control tokens in its response.
            # They are ordinary generated vocabulary tokens without visual payload,
            # so only collapse wrappers that came from the prompt (token_mask=0).
            if torch.is_tensor(token_mask):
                row_mask = token_mask if token_mask.ndim == 1 else token_mask[b]
                if bool(row_mask[start_pos].item()):
                    continue

            end_matches = (sample[start_pos:] == img_end_id).nonzero(as_tuple=True)[0]
            if len(end_matches) == 0:
                raise ValueError(
                    "Malformed multimodal token sequence: found <img> token without a "
                    f"matching </img> token (batch_index={b}, start_pos={start_pos.item()})."
                )
            end_pos = end_matches[0] + start_pos
            keep_mask[start_pos + 2 : end_pos] = False

        collapsed_list.append(sample[keep_mask])
        all_keep_masks.append(keep_mask)
        tokens_removed = full_len - keep_mask.sum().item()
        tokens_removed_per_sample.append(tokens_removed)
        # Actual content length = original content - tokens removed (from content region)
        new_lengths.append(valid_len - tokens_removed)

    max_collapsed_len = max(len(c) for c in collapsed_list)
    collapsed_ids = torch.zeros(
        batch_size, max_collapsed_len, dtype=input_ids.dtype, device=input_ids.device
    )
    for b, collapsed in enumerate(collapsed_list):
        collapsed_ids[b, : len(collapsed)] = collapsed

    new_data_dict = data_dict.copy()
    new_data_dict["input_ids"] = collapsed_ids
    if input_lengths is not None:
        new_data_dict["input_lengths"] = torch.tensor(
            new_lengths, dtype=input_lengths.dtype, device=input_lengths.device
        )
    new_data_dict["tokens_removed_per_sample"] = torch.tensor(
        tokens_removed_per_sample, dtype=torch.int64, device=input_ids.device
    )
    new_data_dict["_collapse_keep_mask"] = torch.stack(all_keep_masks)
    new_data_dict["vision_expansion_per_sample"] = _compute_vision_expansion_tensor(
        new_data_dict,
        collapsed_ids,
        model,
        image_token_index=image_token_index,
        img_start_id=img_start_id,
        device=input_ids.device,
    )

    return new_data_dict


def _get_image_token_ids(model) -> Optional[tuple[int, int]]:
    """Extract <img> and </img> token IDs from Megatron model."""
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model

    for obj in [inner, getattr(inner, "config", None)]:
        if obj is None:
            continue
        start = getattr(obj, "img_start_token_id", None)
        end = getattr(obj, "img_end_token_id", None)
        if start is not None and end is not None:
            return start, end
    return None


def _get_image_token_index(model) -> Optional[int]:
    """Extract the image placeholder token index used inside <img> wrappers."""
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model

    for obj in [inner, getattr(inner, "config", None)]:
        if obj is None:
            continue
        image_token_index = getattr(obj, "image_token_index", None)
        if image_token_index is not None:
            return image_token_index
    return None


def _get_media_image_token_mask(
    input_ids: Optional[torch.Tensor],
    model: Any,
    image_token_index: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Return image-token positions that should be consumed as visual media."""
    if input_ids is None:
        return None
    if image_token_index is None:
        image_token_index = _get_image_token_index(model)
    if image_token_index is None:
        return None

    ids = input_ids
    squeeze = False
    if ids.ndim == 1:
        ids = ids.view(1, -1)
        squeeze = True

    image_mask = ids == image_token_index
    image_token_ids = _get_image_token_ids(model)
    if image_token_ids is None or not image_mask.any().item():
        return image_mask[0] if squeeze else image_mask

    img_start_id, img_end_id = image_token_ids
    prev_is_start = torch.zeros_like(image_mask)
    next_is_end = torch.zeros_like(image_mask)
    prev_is_start[:, 1:] = ids[:, :-1] == img_start_id
    next_is_end[:, :-1] = ids[:, 1:] == img_end_id
    wrapped_mask = image_mask & prev_is_start & next_is_end

    row_has_wrapped = wrapped_mask.any(dim=-1, keepdim=True)
    media_mask = torch.where(row_has_wrapped, wrapped_mask, image_mask)
    return media_mask[0] if squeeze else media_mask


def _get_video_temporal_patch_size(model) -> int:
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model
    vision_model = getattr(inner, "vision_model", None)
    return int(
        getattr(
            inner,
            "temporal_patch_dim",
            getattr(
                inner,
                "_video_temporal_patch_size",
                getattr(vision_model, "temporal_patch_dim", 1),
            ),
        )
    )


def _compute_vision_expansion_tensor(
    data_dict: Any,
    collapsed_ids: torch.Tensor,
    model: Any,
    *,
    image_token_index: Optional[int],
    img_start_id: int,
    device: torch.device,
) -> torch.Tensor:
    batch_size = collapsed_ids.shape[0]
    placeholder_id = (
        image_token_index if image_token_index is not None else img_start_id
    )
    media_image_mask = _get_media_image_token_mask(
        collapsed_ids,
        model,
        image_token_index=placeholder_id,
    )
    num_placeholders = [
        int(media_image_mask[b].sum().item())
        if media_image_mask is not None
        else int((collapsed_ids[b] == placeholder_id).sum().item())
        for b in range(batch_size)
    ]

    (
        patch_dim,
        downsample_ratio,
        class_token_len,
        dynamic_resolution,
        static_img_seq_len,
    ) = _get_model_config(model)

    if not dynamic_resolution and static_img_seq_len is not None:
        expansions = [
            max(0, count * static_img_seq_len - count) for count in num_placeholders
        ]
        return torch.tensor(expansions, dtype=torch.int64, device=device)

    temporal_patch_size = _get_video_temporal_patch_size(model)
    num_frames_per_sample = _resolve_packed_per_sample(
        data_dict.get("num_frames"), batch_size
    )
    frame_counts = []
    for frames, placeholders in zip(num_frames_per_sample, num_placeholders):
        frame_counts.append(
            _raw_placeholder_count_from_num_frames(
                frames,
                placeholders,
                temporal_patch_size,
            )
        )

    imgs_sizes_per_sample = _resolve_packed_per_sample(
        data_dict.get("imgs_sizes"),
        batch_size,
        counts=frame_counts,
    )
    expansions = compute_vision_expansion(
        imgs_sizes_per_sample,
        num_placeholders,
        patch_dim,
        downsample_ratio,
        class_token_len,
        num_frames_per_sample=num_frames_per_sample,
        temporal_patch_size=temporal_patch_size,
    )
    return torch.tensor(expansions, dtype=torch.int64, device=device)


def _get_sound_token_index(model) -> Optional[int]:
    """Extract the sound placeholder token index from Megatron model."""
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model
    return getattr(inner, "sound_token_index", None)


def _get_sound_feature_extractor(model):
    """Get the FastConformer feature extractor from the model's sound config."""
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model
    sound_model = getattr(inner, "sound_model", None)
    if sound_model is None:
        return None
    sound_config = getattr(sound_model, "config", None)
    if sound_config is None:
        return None
    from megatron.core.models.huggingface.fastconformer.feature_extraction_fastconformer import (
        FastConformerFeatureExtractor,
    )

    num_mel_bins = getattr(sound_config, "num_mel_bins", 128)
    sampling_rate = getattr(sound_config, "sampling_rate", 16000)
    hop_length = getattr(sound_config, "hop_length", 160)
    win_length = getattr(sound_config, "win_length", 400)
    n_fft = getattr(sound_config, "n_fft", 512)
    return FastConformerFeatureExtractor(
        feature_size=num_mel_bins,
        sampling_rate=sampling_rate,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
    )


def _resolve_first_float(value: Any, default: float) -> float:
    """Resolve batched/list metadata to a scalar float."""
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return float(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        for item in value:
            if item is None:
                continue
            return _resolve_first_float(item, default)
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def split_audio_into_clips(
    waveform: torch.Tensor,
    *,
    sampling_rate: int = 16000,
    clip_duration_s: float = 30.0,
    clip_min_duration_s: float = 0.1,
) -> list[torch.Tensor]:
    """Split one waveform using the same clip rule as vLLM/SFT Parakeet."""
    if waveform.ndim != 1:
        waveform = waveform.reshape(-1)

    clip_target_samples = max(1, int(round(clip_duration_s * sampling_rate)))
    tail_min_samples = max(1, int(round(clip_min_duration_s * sampling_rate)))
    audio_len = int(waveform.shape[0])
    effective_len = max(audio_len, tail_min_samples)
    num_full_clips, remainder = divmod(effective_len, clip_target_samples)

    clip_sizes = [clip_target_samples] * num_full_clips
    if remainder > 0:
        clip_sizes.append(max(remainder, tail_min_samples))
    if not clip_sizes:
        clip_sizes = [tail_min_samples]

    target_len = sum(clip_sizes)
    if audio_len < target_len:
        waveform = torch.nn.functional.pad(waveform, (0, target_len - audio_len))

    clips: list[torch.Tensor] = []
    offset = 0
    for clip_size in clip_sizes:
        clips.append(waveform[offset : offset + clip_size])
        offset += clip_size
    return clips


def prepare_multimodal_data(
    multimodal_data: dict,
    model,
    device: torch.device,
    input_ids: Optional[torch.Tensor] = None,
) -> None:
    """Prepare pixel_values and sound_clips for Megatron forward pass."""
    _prepare_image_data(multimodal_data, model, device, input_ids=input_ids)
    _prepare_sound_data(multimodal_data, model, device)


def _count_prompt_image_placeholders_per_sample(
    input_ids: Optional[torch.Tensor],
    model: Any,
) -> Optional[list[int]]:
    image_token_index = _get_image_token_index(model)
    if image_token_index is None:
        return None
    media_image_mask = _get_media_image_token_mask(
        input_ids,
        model,
        image_token_index=image_token_index,
    )
    if media_image_mask is None:
        return None
    if media_image_mask.ndim == 1:
        media_image_mask = media_image_mask.view(1, -1)
    return [
        int(value) for value in media_image_mask.sum(dim=-1).to(torch.int64).tolist()
    ]


def _frame_groups_from_value(value: Any) -> Optional[list[int]]:
    if value is None:
        return None
    try:
        groups = _as_int_tensor(value).flatten().tolist()
    except Exception:
        return None
    return [max(0, int(frames)) for frames in groups]


def _tubelet_count_from_frame_groups(
    frame_groups: Optional[list[int]],
    temporal_patch_size: int,
) -> Optional[int]:
    if frame_groups is None:
        return None
    if temporal_patch_size <= 1:
        return sum(frame_groups)
    return sum(
        frames if frames <= 1 else math.ceil(frames / temporal_patch_size)
        for frames in frame_groups
    )


def _trim_frame_groups_to_placeholder_count(
    frame_groups: Optional[list[int]],
    surviving_placeholder_count: int,
    original_placeholder_count: Optional[int],
    temporal_patch_size: int,
) -> tuple[Optional[list[int]], int]:
    """Trim frame metadata to the media placeholders kept in the prompt."""
    if not frame_groups:
        return frame_groups, max(0, int(surviving_placeholder_count))

    surviving_placeholder_count = max(0, int(surviving_placeholder_count))
    temporal_patch_size = max(1, int(temporal_patch_size))
    raw_frame_count = sum(frame_groups)
    tubelet_count = _tubelet_count_from_frame_groups(frame_groups, temporal_patch_size)
    count_is_tubelet = (
        temporal_patch_size > 1
        and tubelet_count is not None
        and original_placeholder_count == tubelet_count
        and raw_frame_count != tubelet_count
    )

    trimmed_groups: list[int] = []
    remaining = surviving_placeholder_count
    for frames_value in frame_groups:
        if remaining <= 0:
            break
        frames = max(0, int(frames_value))
        if frames <= 0:
            continue

        if count_is_tubelet:
            placeholder_span = (
                frames if frames <= 1 else math.ceil(frames / temporal_patch_size)
            )
            kept_placeholders = min(remaining, placeholder_span)
            if kept_placeholders <= 0:
                break
            if frames <= 1:
                kept_frames = kept_placeholders
            elif kept_placeholders == placeholder_span:
                kept_frames = frames
            else:
                kept_frames = min(frames, kept_placeholders * temporal_patch_size)
            remaining -= kept_placeholders
        else:
            kept_frames = min(frames, remaining)
            remaining -= kept_frames

        if kept_frames > 0:
            trimmed_groups.append(int(kept_frames))

    return trimmed_groups, sum(trimmed_groups)


def _slice_tensor_to_media_count(
    tensor: torch.Tensor,
    keep_count: int,
    *,
    dim_to_pack: int = 0,
    original_count: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Slice the tensor axis that carries per-frame/per-image media entries."""
    keep_count = max(0, int(keep_count))
    if keep_count <= 0:
        return None
    if tensor.ndim == 0:
        return tensor

    normalized_pack_dim = dim_to_pack % tensor.ndim
    axes: list[int] = [normalized_pack_dim]
    if original_count is not None:
        axes.extend(
            axis
            for axis, size in enumerate(tensor.shape)
            if axis not in axes and int(size) == int(original_count)
        )
    axes.extend(
        axis
        for axis, size in enumerate(tensor.shape)
        if axis not in axes and int(size) > keep_count
    )

    for axis in axes:
        size = int(tensor.shape[axis])
        if size == keep_count:
            return tensor
        if size > keep_count:
            index = [slice(None)] * tensor.ndim
            index[axis] = slice(0, keep_count)
            return tensor[tuple(index)]
    return tensor


def _infer_num_tile_splits_for_llava(
    prompt_counts: list[int],
    num_frames_per_sample: list[Optional[torch.Tensor]],
    temporal_patch_size: int,
    total_tiles: int,
) -> Optional[list[int]]:
    if len(prompt_counts) == 1:
        return [total_tiles]

    frame_based_counts: list[int] = []
    for frames in num_frames_per_sample:
        tile_count = _tubelet_count_from_frame_groups(
            _frame_groups_from_value(frames),
            temporal_patch_size,
        )
        if tile_count is None:
            break
        frame_based_counts.append(tile_count)
    if (
        len(frame_based_counts) == len(prompt_counts)
        and sum(frame_based_counts) == total_tiles
    ):
        return frame_based_counts

    if temporal_patch_size > 1:
        prompt_based_counts = [
            count if count <= 1 else math.ceil(count / temporal_patch_size)
            for count in prompt_counts
        ]
        if sum(prompt_based_counts) == total_tiles:
            return prompt_based_counts

    total_prompt = sum(prompt_counts)
    if total_tiles > 0 and total_prompt % total_tiles == 0:
        ratio = total_prompt // total_tiles
        if 1 < ratio <= 8:
            ratio_based_counts = [
                count if count <= 1 else math.ceil(count / ratio)
                for count in prompt_counts
            ]
            if sum(ratio_based_counts) == total_tiles:
                return ratio_based_counts

    return None


def _expand_num_image_tiles_for_llava_placeholders(
    num_tiles: torch.Tensor,
    num_frames_value: Any,
    input_ids: Optional[torch.Tensor],
    model: Any,
    temporal_patch_size: int,
) -> torch.Tensor:
    prompt_counts = _count_prompt_image_placeholders_per_sample(input_ids, model)
    if prompt_counts is None:
        return num_tiles

    total_prompt_placeholders = sum(prompt_counts)
    total_tiles = int(num_tiles.numel())
    if total_prompt_placeholders <= 0 or total_prompt_placeholders == total_tiles:
        return num_tiles
    if total_prompt_placeholders < total_tiles or total_tiles <= 0:
        return num_tiles

    num_frames_per_sample = (
        _resolve_packed_per_sample(num_frames_value, len(prompt_counts))
        if num_frames_value is not None
        else [None] * len(prompt_counts)
    )
    tile_splits = _infer_num_tile_splits_for_llava(
        prompt_counts,
        num_frames_per_sample,
        temporal_patch_size,
        total_tiles,
    )
    if tile_splits is None or sum(tile_splits) != total_tiles:
        return num_tiles

    values = [int(value) for value in num_tiles.detach().cpu().tolist()]
    expanded_values: list[int] = []
    offset = 0
    changed = False
    for prompt_count, tile_count, frames in zip(
        prompt_counts,
        tile_splits,
        num_frames_per_sample,
    ):
        tile_values = values[offset : offset + tile_count]
        offset += tile_count
        if prompt_count == tile_count:
            expanded_values.extend(tile_values)
            continue

        expanded = None
        if temporal_patch_size > 1:
            expanded = _expand_tubelet_values_to_frame_placeholders(
                tile_values,
                frames,
                prompt_count,
                temporal_patch_size,
                secondary_value=0,
            )

        if (
            expanded is None
            and tile_count > 0
            and prompt_count > tile_count
            and prompt_count % tile_count == 0
        ):
            ratio = prompt_count // tile_count
            if 1 < ratio <= 8:
                expanded = []
                for value in tile_values:
                    expanded.append(value)
                    expanded.extend([0] * (ratio - 1))

        if expanded is None or len(expanded) != prompt_count:
            return num_tiles
        expanded_values.extend(expanded)
        changed = True

    if (
        not changed
        or offset != total_tiles
        or len(expanded_values) != total_prompt_placeholders
    ):
        return num_tiles

    return torch.tensor(expanded_values, dtype=num_tiles.dtype, device=num_tiles.device)


def _expand_num_image_tiles_from_frame_metadata(
    num_tiles: torch.Tensor,
    num_frames_value: Any,
    temporal_patch_size: int,
) -> torch.Tensor:
    if temporal_patch_size <= 1 or num_frames_value is None:
        return num_tiles

    frames_per_sample = _resolve_packed_per_sample(num_frames_value, 1)
    frame_groups = _frame_groups_from_value(frames_per_sample[0])
    if not frame_groups:
        return num_tiles

    total_tiles = int(num_tiles.numel())
    raw_frame_count = sum(frame_groups)
    if total_tiles <= 0 or raw_frame_count <= total_tiles:
        return num_tiles

    expected_tubelets = _tubelet_count_from_frame_groups(
        frame_groups,
        temporal_patch_size,
    )
    if expected_tubelets != total_tiles:
        return num_tiles

    expanded = _expand_tubelet_values_to_frame_placeholders(
        [int(value) for value in num_tiles.detach().cpu().tolist()],
        torch.tensor(frame_groups, dtype=torch.int32),
        raw_frame_count,
        temporal_patch_size,
        secondary_value=0,
    )
    if expanded is None or len(expanded) != raw_frame_count:
        return num_tiles

    return torch.tensor(expanded, dtype=num_tiles.dtype, device=num_tiles.device)


def _trim_video_media_to_prompt_placeholders(
    multimodal_data: dict,
    images: torch.Tensor,
    input_ids: Optional[torch.Tensor],
    model: Any,
    temporal_patch_size: int,
) -> torch.Tensor:
    """Drop video frames whose Conv3D/tubelet placeholders are absent from input_ids."""
    if temporal_patch_size <= 1:
        return images

    prompt_counts = _count_prompt_image_placeholders_per_sample(input_ids, model)
    if prompt_counts is None or len(prompt_counts) != 1:
        return images

    prompt_count = int(prompt_counts[0])
    if prompt_count <= 0:
        return images

    frame_groups = _frame_groups_from_value(multimodal_data.get("num_frames"))
    if not frame_groups:
        return images

    original_raw_count = sum(frame_groups)
    original_tubelet_count = _tubelet_count_from_frame_groups(
        frame_groups,
        temporal_patch_size,
    )
    if (
        original_tubelet_count is None
        or prompt_count >= original_tubelet_count
        or original_raw_count == original_tubelet_count
    ):
        return images

    trimmed_frame_groups, keep_raw_count = _trim_frame_groups_to_placeholder_count(
        frame_groups,
        prompt_count,
        original_tubelet_count,
        temporal_patch_size,
    )
    if (
        keep_raw_count <= 0
        or keep_raw_count >= original_raw_count
        or not trimmed_frame_groups
    ):
        return images

    kept_images = _slice_tensor_to_media_count(
        images,
        keep_raw_count,
        dim_to_pack=0,
        original_count=original_raw_count,
    )
    if kept_images is None:
        return images

    imgs_sizes = multimodal_data.get("imgs_sizes")
    if torch.is_tensor(imgs_sizes):
        kept_sizes = _slice_tensor_to_media_count(
            imgs_sizes,
            keep_raw_count,
            dim_to_pack=0,
            original_count=original_raw_count,
        )
        if kept_sizes is not None:
            multimodal_data["imgs_sizes"] = kept_sizes

    num_frames_value = multimodal_data.get("num_frames")
    if torch.is_tensor(num_frames_value):
        multimodal_data["num_frames"] = torch.tensor(
            trimmed_frame_groups,
            dtype=num_frames_value.dtype,
            device=num_frames_value.device,
        )

    return kept_images


def _prepare_image_data(
    multimodal_data: dict,
    model,
    device: torch.device,
    input_ids: Optional[torch.Tensor] = None,
) -> None:
    """Prepare pixel_values for Megatron forward (patchification for dynamic resolution)."""
    if "pixel_values" not in multimodal_data and "pixel_values_flat" in multimodal_data:
        images = multimodal_data.pop("pixel_values_flat").to(torch.bfloat16)
        num_tiles = multimodal_data.pop("image_num_patches", None)
        if num_tiles is None:
            num_tiles = torch.ones(images.shape[0], dtype=torch.int, device=device)
        elif not isinstance(num_tiles, torch.Tensor):
            num_tiles = torch.tensor(num_tiles, dtype=torch.int, device=device)
        else:
            num_tiles = num_tiles.to(device=device, dtype=torch.int)

        multimodal_data["images"] = images
        multimodal_data["num_image_tiles"] = num_tiles
        return

    if "pixel_values" not in multimodal_data:
        # LLaVAModel requires images, imgs_sizes, and num_image_tiles; pass empty tensors
        # num_image_tiles must be empty to match images count, even if input_ids has image tokens
        multimodal_data["images"] = torch.empty(0, dtype=torch.bfloat16, device=device)
        multimodal_data["imgs_sizes"] = torch.empty(
            0, 2, dtype=torch.int32, device=device
        )
        multimodal_data["num_image_tiles"] = torch.empty(
            0, dtype=torch.int, device=device
        )
        return

    images = multimodal_data.pop("pixel_values").to(torch.bfloat16)

    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model

    dynamic_res = bool(
        getattr(
            inner,
            "dynamic_resolution",
            getattr(
                inner,
                "_dynamic_resolution",
                getattr(
                    getattr(inner, "vision_model", None), "dynamic_resolution", False
                ),
            ),
        )
    )
    has_imgs_sizes = "imgs_sizes" in multimodal_data
    imgs_sizes = multimodal_data.get("imgs_sizes")

    if dynamic_res and has_imgs_sizes:
        patch_dim = getattr(inner.vision_model, "patch_dim", 16)
        temporal_patch_size = _get_video_temporal_patch_size(inner)
        images = _trim_video_media_to_prompt_placeholders(
            multimodal_data,
            images,
            input_ids,
            inner,
            temporal_patch_size,
        )
        # imgs_sizes contains actual pixel dimensions for cropping
        # RADIO uses these to compute patch counts for position encoding
        # LLaVAModel._preprocess_data applies pixel_shuffle reduction internally
        images, num_tiles, vision_params = _patchify_for_dynamic_resolution(
            images, multimodal_data["imgs_sizes"], patch_dim
        )
        num_frames_value = multimodal_data.get("num_frames")
        prompt_counts = _count_prompt_image_placeholders_per_sample(input_ids, inner)
        num_tiles = _expand_num_image_tiles_for_llava_placeholders(
            num_tiles,
            num_frames_value,
            input_ids,
            inner,
            temporal_patch_size,
        )
        if prompt_counts is None:
            num_tiles = _expand_num_image_tiles_from_frame_metadata(
                num_tiles,
                num_frames_value,
                temporal_patch_size,
            )

        multimodal_data["num_image_tiles"] = num_tiles
        multimodal_data["vision_packed_seq_params"] = vision_params

        # When temporal compression is enabled (video_temporal_patch_size > 1),
        # RADIO requires num_frames to distinguish images (1 frame) from videos.
        # For image-only data, default to 1 frame per image.
        if temporal_patch_size > 1 and "num_frames" not in multimodal_data:
            num_images = len(multimodal_data["imgs_sizes"])
            multimodal_data["num_frames"] = torch.ones(
                num_images, dtype=torch.int32, device=device
            )
    elif dynamic_res and not has_imgs_sizes:
        raise AssertionError(
            "dynamic_resolution=True but imgs_sizes not provided in multimodal_data. "
            "The data pipeline must supply imgs_sizes when dynamic_resolution is enabled, "
            "otherwise the model output length will not match the input length."
        )

    multimodal_data["images"] = images


def _prepare_sound_data(multimodal_data: dict, model, device: torch.device) -> None:
    """Prepare sound_clips for Megatron forward.

    Raw waveforms are split from the flat concatenated tensor, padded into a 2D batch,
    then converted to log-mel spectrograms via FastConformerFeatureExtractor.
    The BridgeSoundEncoder expects mel features [batch, frames, mel_bins], not raw audio.
    """
    clip_duration_value = multimodal_data.pop("sound_clip_duration", None)
    clip_min_duration_value = multimodal_data.pop("sound_clip_min_duration", None)

    if "sound_clips" not in multimodal_data:
        return

    flat_waveform = multimodal_data.pop("sound_clips")
    lengths = multimodal_data.pop("sound_length")

    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, dtype=torch.int32)
    lengths = lengths.to(device)

    if lengths.numel() == 0 or flat_waveform.numel() == 0:
        return

    feature_extractor = _get_sound_feature_extractor(model)
    sampling_rate = int(getattr(feature_extractor, "sampling_rate", 16000))
    clip_duration_s = _resolve_first_float(clip_duration_value, 30.0)
    clip_min_duration_s = _resolve_first_float(clip_min_duration_value, 0.1)

    original_clips = torch.split(flat_waveform, lengths.tolist())
    split_clips: list[torch.Tensor] = []
    for clip in original_clips:
        split_clips.extend(
            split_audio_into_clips(
                clip.to(dtype=torch.float32, device=device),
                sampling_rate=sampling_rate,
                clip_duration_s=clip_duration_s,
                clip_min_duration_s=clip_min_duration_s,
            )
        )

    lengths = torch.tensor(
        [clip.shape[0] for clip in split_clips], dtype=torch.int32, device=device
    )
    max_len = int(lengths.max().item())
    padded = torch.zeros(len(split_clips), max_len, dtype=torch.float32, device=device)
    for i, clip in enumerate(split_clips):
        padded[i, : clip.shape[0]] = clip

    if feature_extractor is not None:
        result = feature_extractor(
            raw_speech=padded,
            audio_lengths=lengths.long(),
            sampling_rate=feature_extractor.sampling_rate,
            device=str(device),
        )
        mel_features = result["input_features"].to(dtype=torch.bfloat16, device=device)
        hop_length = feature_extractor.hop_length
        mel_lengths = torch.tensor(
            [int(wl) // hop_length for wl in lengths.tolist()],
            dtype=torch.int32,
            device=device,
        )
        max_canonical_len = int(mel_lengths.max().item())
        mel_features = mel_features[:, :max_canonical_len, :]
        multimodal_data["sound_clips"] = mel_features
        multimodal_data["sound_length"] = mel_lengths
    else:
        multimodal_data["sound_clips"] = padded.to(dtype=torch.bfloat16)
        multimodal_data["sound_length"] = lengths


def _patchify_for_dynamic_resolution(
    images: torch.Tensor,
    imgs_sizes: torch.Tensor,
    patch_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, "PackedSeqParams"]:
    """Convert images to packed patches for dynamic resolution RADIO vision encoder."""
    from megatron.core.packed_seq_params import PackedSeqParams

    def to_patches(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
        img = img[:, :h, :w]
        py, px = h // patch_dim, w // patch_dim
        return rearrange(
            img,
            "c (py yy) (px xx) -> (py px) (c yy xx)",
            py=py,
            yy=patch_dim,
            px=px,
            xx=patch_dim,
        )

    patches_list = [
        to_patches(img, *imgs_sizes[i].tolist()) for i, img in enumerate(images)
    ]

    cu_seqlens = [0]
    for p in patches_list:
        cu_seqlens.append(cu_seqlens[-1] + p.shape[0])

    max_seqlen = max(p.shape[0] for p in patches_list)
    return (
        torch.cat(patches_list, dim=0).unsqueeze(0),
        torch.ones(len(images), dtype=torch.int, device=images.device),
        PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor(
                cu_seqlens, dtype=torch.int32, device=images.device
            ),
            cu_seqlens_kv=torch.tensor(
                cu_seqlens, dtype=torch.int32, device=images.device
            ),
            max_seqlen_q=torch.tensor(
                max_seqlen, dtype=torch.int32, device=images.device
            ),
            max_seqlen_kv=torch.tensor(
                max_seqlen, dtype=torch.int32, device=images.device
            ),
        ),
    )


def remap_expanded_logits_to_collapsed(
    expanded_logits: torch.Tensor,
    collapsed_input_ids: torch.Tensor,
    model,
    multimodal_data: dict,
) -> torch.Tensor:
    """Map Megatron's image-expanded logits back to collapsed token positions."""
    image_token_index = _get_image_token_index(model)
    if image_token_index is None:
        return expanded_logits

    batch_size, collapsed_len = collapsed_input_ids.shape
    expanded_len = expanded_logits.shape[1]
    if expanded_len == collapsed_len:
        return expanded_logits

    imgs_sizes = multimodal_data.get("imgs_sizes")
    if imgs_sizes is None or imgs_sizes.numel() == 0:
        return expanded_logits

    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model

    patch_dim = getattr(inner.vision_model, "patch_dim", 16)
    pixel_shuffle = getattr(inner, "_pixel_shuffle", False)
    conv_merging = getattr(inner, "_use_conv_merging", False)
    drop_cls = getattr(inner, "_drop_vision_class_token", True)
    cls_len = (
        0
        if drop_cls
        else getattr(
            inner,
            "_class_token_len",
            getattr(getattr(inner, "vision_model", None), "class_token_len", 1),
        )
    )

    per_img_embeds = (
        torch.prod(imgs_sizes // patch_dim, dim=-1, dtype=torch.int32) + cls_len
    )
    if pixel_shuffle:
        per_img_embeds = (per_img_embeds * (0.5**2)).int()
    if conv_merging:
        per_img_embeds = (per_img_embeds * (0.5**2)).int()

    result_list = []
    image_offset = 0
    for b in range(batch_size):
        ids = collapsed_input_ids[b]
        img_positions = (ids == image_token_index).nonzero(as_tuple=True)[0]

        if len(img_positions) == 0:
            result_list.append(expanded_logits[b, :collapsed_len])
            continue

        mapping = torch.arange(collapsed_len, device=ids.device, dtype=torch.long)
        for local_img_idx, img_pos in enumerate(img_positions):
            embed_idx = image_offset + local_img_idx
            if embed_idx >= len(per_img_embeds):
                break
            expansion = int(per_img_embeds[embed_idx].item()) - 1
            mapping[img_pos:] += expansion
        image_offset += len(img_positions)

        mapping = mapping.clamp(0, expanded_len - 1)
        result_list.append(expanded_logits[b, mapping])

    return torch.stack(result_list)
