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

from typing import Optional

import torch
from einops import rearrange
from megatron.core.packed_seq_params import PackedSeqParams
from nemo_rl.data.multimodal_utils import PackedTensor


def is_llava_model(model) -> bool:
    """Check if the model is a LLaVA model.
    
    Args:
        model: The model to check
        
    Returns:
        True if the model is a LLaVA model, False otherwise
    """
    # Handle wrapped models (e.g., DDP, Float16Module)
    actual_model = model
    while hasattr(actual_model, 'module'):
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
    if hasattr(actual_model, 'llava_model'):
        return True
    if hasattr(actual_model, 'img_start_token_id') and hasattr(actual_model, 'img_end_token_id'):
        return True
    config = getattr(actual_model, 'config', None)
    if config is not None and hasattr(config, 'img_start_token_id') and hasattr(config, 'img_end_token_id'):
        return True
    return False



def collapse_multimodal_tokens(data_dict: dict, model) -> dict:
    """Collapse N image tokens to 1 token per image for Megatron LLaVA forward pass.

    vLLM uses N tokens per image (1:1 token-to-embedding), while Megatron uses 1 token
    per image/tile (1:N via imgs_sizes). This collapses <img><image>×N</img> to <img><image></img>.

    Processes the full padded sequence (not just valid content) so that after model forward,
    output length matches padded input length. Padding tokens (zeros) won't match image token
    IDs, so only content region gets collapsed while padding is preserved.
    """
    image_token_ids = _get_image_token_ids(model)
    if image_token_ids is None or "pixel_values" not in data_dict:
        return data_dict

    input_ids = data_dict["input_ids"]
    input_lengths = data_dict.get("input_lengths")
    img_start_id, img_end_id = image_token_ids
    batch_size = input_ids.shape[0]

    # Check if pixel_values key exists without image tokens. This happens when
    # all samples in a micro-batch were discarded (overlong).
    img_start_count = (input_ids == img_start_id).sum().item()
    img_end_count = (input_ids == img_end_id).sum().item()

    if img_start_count == 0 and img_end_count == 0 and "pixel_values" in data_dict:
        # Drop the stale multimodal keys and treat the batch as text-only.
        del data_dict["pixel_values"]
        if "imgs_sizes" in data_dict:
            del data_dict["imgs_sizes"]
        return data_dict

    original_seq_len = input_ids.shape[1]
    has_imgs_sizes = "imgs_sizes" in data_dict

    collapsed_list = []
    new_lengths = []
    tokens_removed_per_sample = []

    for b in range(batch_size):
        # Process full padded sequence, not just valid content
        # Padding tokens (zeros) won't match image token IDs, so only content gets collapsed
        sample = input_ids[b]
        full_len = sample.shape[0]
        valid_len = input_lengths[b].item() if input_lengths is not None else full_len

        keep_mask = torch.ones(full_len, dtype=torch.bool, device=input_ids.device)
        for start_pos in (sample == img_start_id).nonzero(as_tuple=True)[0]:
            end_matches = (sample[start_pos:] == img_end_id).nonzero(as_tuple=True)[0]
            if len(end_matches) == 0:
                raise ValueError(
                    "Malformed multimodal token sequence: found <img> token without a "
                    f"matching </img> token (batch_index={b}, start_pos={start_pos.item()})."
                )
            end_pos = end_matches[0] + start_pos
            keep_mask[start_pos + 2 : end_pos] = False

        collapsed_list.append(sample[keep_mask])
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
    stored_tokens_removed_per_sample = list(tokens_removed_per_sample)

    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model

    if not getattr(inner, "_dynamic_resolution", True):
        static_img_seq_len = getattr(inner, "img_seq_len", None)
        if static_img_seq_len is not None:
            # Preserve the local physical removal count for collapsed lengths,
            # but store the fixed static-resolution expansion delta because
            # downstream packing/SP code already consumes this tensor in
            # expansion space.
            static_img_seq_len = int(static_img_seq_len)
            for b in range(batch_size):
                num_images = int((collapsed_ids[b] == img_start_id).sum().item())
                stored_tokens_removed_per_sample[b] = (
                    num_images * (static_img_seq_len - 1)
                )

    new_data_dict["tokens_removed_per_sample"] = torch.tensor(
        stored_tokens_removed_per_sample, dtype=torch.int64, device=input_ids.device
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


def prepare_multimodal_data(multimodal_data: dict, model, device: torch.device) -> None:
    """Prepare pixel_values and sound_clips for Megatron forward pass."""
    _prepare_image_data(multimodal_data, model, device)
    _prepare_sound_data(multimodal_data, model, device)


def _prepare_image_data(multimodal_data: dict, model, device: torch.device) -> None:
    """Prepare pixel_values for Megatron forward (patchification for dynamic resolution)."""
    if "pixel_values" not in multimodal_data:
        # LLaVAModel requires images, imgs_sizes, and num_image_tiles; pass empty tensors
        # num_image_tiles must be empty to match images count, even if input_ids has image tokens
        multimodal_data["images"] = torch.empty(0, dtype=torch.bfloat16, device=device)
        multimodal_data["imgs_sizes"] = torch.empty(0, 2, dtype=torch.int32, device=device)
        multimodal_data["num_image_tiles"] = torch.empty(0, dtype=torch.int, device=device)
        return

    images = multimodal_data.pop("pixel_values").to(torch.bfloat16)

    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model

    dynamic_res = getattr(inner, "_dynamic_resolution", False)
    has_imgs_sizes = "imgs_sizes" in multimodal_data
    imgs_sizes = multimodal_data.get("imgs_sizes")

    if dynamic_res and has_imgs_sizes:
        patch_dim = getattr(inner.vision_model, "patch_dim", 16)
        # imgs_sizes contains actual pixel dimensions for cropping
        # RADIO uses these to compute patch counts for position encoding
        # LLaVAModel._preprocess_data applies pixel_shuffle reduction internally
        images, num_tiles, vision_params = _patchify_for_dynamic_resolution(
            images, multimodal_data["imgs_sizes"], patch_dim
        )
        multimodal_data["num_image_tiles"] = num_tiles
        multimodal_data["vision_packed_seq_params"] = vision_params

        # When temporal compression is enabled (video_temporal_patch_size > 1),
        # RADIO requires num_frames to distinguish images (1 frame) from videos.
        # For image-only data, default to 1 frame per image.
        temporal_patch_size = getattr(inner, "_video_temporal_patch_size", 1)
        if temporal_patch_size > 1 and "num_frames" not in multimodal_data:
            num_images = len(multimodal_data["imgs_sizes"])
            multimodal_data["num_frames"] = torch.ones(num_images, dtype=torch.int32, device=device)
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
    if "sound_clips" not in multimodal_data:
        return

    flat_waveform = multimodal_data.pop("sound_clips")
    lengths = multimodal_data.pop("sound_length")

    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, dtype=torch.int32)
    lengths = lengths.to(device)

    if lengths.numel() == 0 or flat_waveform.numel() == 0:
        return

    clips = torch.split(flat_waveform, lengths.tolist())
    max_len = int(lengths.max().item())
    num_clips = len(clips)
    padded = torch.zeros(num_clips, max_len, dtype=torch.float32, device=device)
    for i, clip in enumerate(clips):
        padded[i, : clip.shape[0]] = clip.to(dtype=torch.float32, device=device)

    feature_extractor = _get_sound_feature_extractor(model)
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
            dtype=torch.int32, device=device,
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
) -> tuple[torch.Tensor, torch.Tensor, PackedSeqParams]:
    """Convert images to packed patches for dynamic resolution RADIO vision encoder."""

    def to_patches(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
        img = img[:, :h, :w]
        py, px = h // patch_dim, w // patch_dim
        return rearrange(
            img, "c (py yy) (px xx) -> (py px) (c yy xx)", py=py, yy=patch_dim, px=px, xx=patch_dim
        )

    patches_list = [to_patches(img, *imgs_sizes[i].tolist()) for i, img in enumerate(images)]

    cu_seqlens = [0]
    for p in patches_list:
        cu_seqlens.append(cu_seqlens[-1] + p.shape[0])

    max_seqlen = max(p.shape[0] for p in patches_list)
    return (
        torch.cat(patches_list, dim=0).unsqueeze(0),
        torch.ones(len(images), dtype=torch.int, device=images.device),
        PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor(cu_seqlens, dtype=torch.int32, device=images.device),
            cu_seqlens_kv=torch.tensor(cu_seqlens, dtype=torch.int32, device=images.device),
            max_seqlen_q=torch.tensor(max_seqlen, dtype=torch.int32, device=images.device),
            max_seqlen_kv=torch.tensor(max_seqlen, dtype=torch.int32, device=images.device),
        ),
    )
