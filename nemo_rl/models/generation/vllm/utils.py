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

import base64
from io import BytesIO
from typing import Any, Optional

from PIL import Image

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import GenerationDatumSpec


def load_image(image: str) -> Image.Image:
    """Load an image from file path or base64 string."""
    if image.startswith("data:"):
        _, encoded = image.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
    return Image.open(image).convert("RGB")


def _format_prompts_from_compact_payload(
    data: BatchedDataDict[GenerationDatumSpec],
    compact: dict[str, Any],
    start_idx: int,
    end_idx: int,
) -> list[dict[str, Any]]:
    """Reconstruct per-row vLLM prompt dicts from a compact multimodal payload.

    Unique images are decoded once and reused across rows, avoiding N-redundant
    PIL.Image construction for repeated prompts.
    """
    input_ids = data["input_ids"]
    input_lengths = data["input_lengths"]
    row_use_token_prompt: list[bool] = compact["row_use_token_prompt"]
    row_content_idx: list[int] = compact["row_content_idx"]
    row_image_ref_indices: list[list[int]] = compact["row_image_ref_indices"]
    unique_contents: list[str] = compact["unique_contents"]
    unique_images: list[str] = compact["unique_images"]

    # Decode each unique image once, build a cache indexed by unique image index
    pil_image_cache: dict[int, Image.Image] = {}

    def _get_pil(idx: int) -> Image.Image:
        if idx not in pil_image_cache:
            pil_image_cache[idx] = load_image(unique_images[idx])
        return pil_image_cache[idx]

    def _get_regular_prompt(index: int) -> dict[str, Any]:
        valid_length = input_lengths[index].item()
        valid_ids = (
            input_ids[index, :valid_length]
            if valid_length > 0
            else input_ids[index, :0]
        )
        return {"prompt_token_ids": valid_ids.tolist()}

    prompts: list[dict[str, Any]] = []
    for i in range(start_idx, end_idx):
        if row_use_token_prompt[i]:
            prompts.append(_get_regular_prompt(i))
            continue

        content = unique_contents[row_content_idx[i]]
        img_refs = row_image_ref_indices[i]
        if not img_refs:
            prompts.append(_get_regular_prompt(i))
            continue

        pil_images = [_get_pil(ref) for ref in img_refs]
        prompt_dict: dict[str, Any] = {"prompt": content}
        prompt_dict["multi_modal_data"] = {
            "image": pil_images[0] if len(pil_images) == 1 else pil_images
        }
        prompts.append(prompt_dict)

    return prompts


def format_prompt_for_vllm_generation(
    data: BatchedDataDict[GenerationDatumSpec], sample_idx: Optional[int] = None
) -> list[dict[str, Any]]:
    """Format a list of prompts for vllm generation (which requires a specific format for its own `generate` method).

    See https://docs.vllm.ai/en/v0.9.1/features/multimodal_inputs.html for prompt format for multimodal inputs.
    """
    # Prepare prompts for vLLM (removing padding)
    prompts = []

    input_ids = data["input_ids"]
    batch_size = input_ids.shape[0]
    input_lengths = data["input_lengths"]

    # if sample_idx is None, return list of all prompts for the entire batch
    # else, return the prompt for the single sample specified by sample_idx
    return_all = sample_idx is None
    if sample_idx is None:
        start_idx = 0
        end_idx = batch_size
    else:
        start_idx = sample_idx
        end_idx = sample_idx + 1

    def _get_regular_prompt(index: int):
        valid_length = input_lengths[index].item()
        valid_ids = (
            input_ids[index, :valid_length]
            if valid_length > 0
            else input_ids[index, :0]
        )
        token_ids = valid_ids.tolist()
        return {"prompt_token_ids": token_ids}

    # Check if compact multimodal payload is present (deduplicate_multimodal_data path)
    if "vllm_mm_compact_payload" in data:
        compact = data["vllm_mm_compact_payload"]
        prompts = _format_prompts_from_compact_payload(
            data, compact, start_idx, end_idx
        )
        return prompts if return_all else prompts[0]

    # Check if this is VLM generation by looking for message_log with images
    # Support for videos/audio/etc. can be added here
    # if 'message_log' in data and any('images' in msg for msg in data['message_log']):
    if "vllm_content" in data:
        # VLM generation using content and multi_modal_data
        for i in range(start_idx, end_idx):
            msg = data["vllm_content"][i]
            # if msg is None, this conversation had no multimodal content, fallback to regular prompt
            if msg is None:
                prompts.append(_get_regular_prompt(i))
                continue
            # init prompt dict
            prompt_dict = {"prompt": msg}
            # add additional data if present
            images = data.get("vllm_images", None)
            if images is None or len(images[i]) == 0 or images[i][0] == "__noimage__":
                prompts.append(_get_regular_prompt(i))
                continue
            else:
                pil_images = [load_image(image) for image in images[i]]
                prompt_dict["multi_modal_data"] = {
                    "image": pil_images[0] if len(pil_images) == 1 else pil_images
                }
            prompts.append(prompt_dict)
    else:
        # Regular LLM generation using token_ids
        for i in range(start_idx, end_idx):
            # Use input_lengths to get only valid tokens (not padding)
            prompts.append(_get_regular_prompt(i))

    return prompts if return_all else prompts[0]

