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

"""Training-side Nano VL processor matching vLLM dynamic image tiling."""

import math
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BatchFeature, PretrainedConfig
from transformers.processing_utils import ProcessorMixin

IMG_INPUT_TAG = "<image>"
IMG_START = "<img>"
IMG_END = "</img>"
IMG_CONTEXT = "<image>"


def _flatten_images(images: Any) -> list[Image.Image]:
    if images is None:
        return []
    if isinstance(images, Image.Image):
        return [images]
    if isinstance(images, (list, tuple)):
        flattened: list[Image.Image] = []
        for item in images:
            flattened.extend(_flatten_images(item))
        return flattened
    return [images]


class DynamicResolutionProcessor(ProcessorMixin):
    """Process Nano VL images with the same dynamic tiler used by vLLM.

    The exported Nano checkpoint's HF processor still uses legacy 512px
    InternVL tiles. vLLM selects its RADIO dynamic-resolution path from
    ``vision_config.args``. This wrapper makes policy preprocessing use that
    same path and preserves image encounter order in interleaved SAV prompts.
    """

    attributes = ["tokenizer"]
    tokenizer_class = "PreTrainedTokenizerFast"
    model_input_names = ["pixel_values", "imgs_sizes"]
    image_token = IMG_CONTEXT

    def __init__(
        self,
        tokenizer: Any,
        config: PretrainedConfig,
        *,
        chat_template: Optional[str] = None,
    ) -> None:
        super().__init__(tokenizer, chat_template=chat_template)
        self.config = config
        vision_config = config.vision_config
        vision_args = getattr(vision_config, "args", {}) or {}
        self.patch_size = int(getattr(vision_config, "patch_size", 16))
        self.min_num_patches = int(vision_args.get("min_num_patches", 1024))
        configured_max = int(vision_args.get("max_num_patches", 13312))
        self.max_num_patches: Union[int, float] = (
            configured_max if configured_max > 0 else float("inf")
        )
        self.downsample_ratio = float(getattr(config, "downsample_ratio", 0.5))
        reduction_factor = 1 / self.downsample_ratio
        if reduction_factor != 2.0:
            raise ValueError(
                "Nano dynamic resolution currently requires downsample_ratio=0.5; "
                f"got {self.downsample_ratio}."
            )
        # vLLM's DynamicResolutionImageTiler hardcodes PIXEL_SHUFFLE=True and
        # CONV_MERGING=False for this model, independent of the nullable HF key.
        self.reduction_factor = int(reduction_factor)

        default_mean = [0.48145466, 0.4578275, 0.40821073]
        default_std = [0.26862954, 0.26130258, 0.27577711]
        norm_mean = getattr(config, "norm_mean", None) or vision_args.get(
            "norm_mean"
        ) or default_mean
        norm_std = getattr(config, "norm_std", None) or vision_args.get(
            "norm_std"
        ) or default_std
        self.norm_mean = torch.tensor(norm_mean, dtype=torch.float32).view(3, 1, 1)
        self.norm_std = torch.tensor(norm_std, dtype=torch.float32).view(3, 1, 1)

        print(
            f"[{type(self).__name__}] vLLM-compatible dynamic images: "
            f"patch_size={self.patch_size} min_num_patches={self.min_num_patches} "
            f"max_num_patches={self.max_num_patches} "
            f"downsample_ratio={self.downsample_ratio}",
            flush=True,
        )

    @staticmethod
    def conversation_preprocessor(message: dict[str, Any]) -> dict[str, Any]:
        """Flatten structured content without changing interleaved ordering."""
        content = message.get("content")
        if not isinstance(content, list):
            return message

        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = item.get("type", "")
            if item_type in ("image", "image_url", "input_image"):
                parts.append(IMG_INPUT_TAG)
            elif item_type in ("video", "video_url", "input_video"):
                parts.append("<video>")
            elif item_type in ("audio", "audio_url", "input_audio"):
                parts.append("<so_embedding>")
            elif item_type in ("text", "input_text"):
                parts.append(str(item.get("text", "")))
        return {**message, "content": "\n".join(parts)}

    def compute_num_embeddings(self, height: int, width: int) -> int:
        num_patches = (height // self.patch_size) * (width // self.patch_size)
        return num_patches // (self.reduction_factor**2)

    def _process_media(
        self, image: Image.Image, patch_budget: Union[int, float]
    ) -> tuple[int, int, int]:
        orig_width, orig_height = image.size
        closest_patch_height = round(orig_height / self.patch_size + 0.5)
        closest_patch_width = round(orig_width / self.patch_size + 0.5)
        patches = closest_patch_height * closest_patch_width
        factor = min(math.sqrt(patch_budget / patches), 1.0)
        target_patch_height = max(1, math.floor(factor * closest_patch_height))
        target_patch_width = max(1, math.floor(factor * closest_patch_width))

        target_patches = target_patch_height * target_patch_width
        if patch_budget > self.min_num_patches and target_patches < self.min_num_patches:
            up_factor = math.sqrt(self.min_num_patches / target_patches)
            target_patch_height = math.ceil(up_factor * target_patch_height)
            target_patch_width = math.ceil(up_factor * target_patch_width)

        # Match vLLM's pixel-shuffle grid constraint.
        for dimension in ("height", "width"):
            if dimension == "height":
                value, other = target_patch_height, target_patch_width
            else:
                value, other = target_patch_width, target_patch_height
            remainder = value % 2
            if remainder:
                increment = 2 - remainder
                if (value + increment) * other <= patch_budget:
                    value += increment
                else:
                    value = max(2, value - remainder)
            if dimension == "height":
                target_patch_height = value
            else:
                target_patch_width = value

        patch_count = target_patch_height * target_patch_width
        return target_patch_height, target_patch_width, patch_count

    def _compute_params(
        self, images: list[Image.Image], num_tokens_available: int
    ) -> list[tuple[int, int]]:
        # vLLM expands the post-shuffle LLM budget back to pre-shuffle patches.
        patch_budget = max(
            int(num_tokens_available) * (self.reduction_factor**2),
            self.min_num_patches * len(images),
        )
        per_image = [
            max(min(patch_budget, self.max_num_patches), self.min_num_patches)
            for _ in images
        ]
        for _ in range(10):
            resolved = [
                self._process_media(image, budget)
                for image, budget in zip(images, per_image)
            ]
            patch_counts = [item[2] for item in resolved]
            total = sum(patch_counts)
            if total <= patch_budget:
                return [(item[0], item[1]) for item in resolved]
            scale = patch_budget / total
            scaled = [
                max(self.min_num_patches, int(count * scale))
                for count in patch_counts
            ]
            if not any(new < old for new, old in zip(scaled, per_image)):
                per_image = [self.min_num_patches] * len(images)
            else:
                per_image = scaled
        raise ValueError("Nano dynamic image token budgeting did not converge.")

    def _preprocess_image(
        self, image: Image.Image, patch_height: int, patch_width: int
    ) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        tensor = (
            torch.from_numpy(array)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .to(dtype=torch.float32)
        )
        target_size = (
            patch_height * self.patch_size,
            patch_width * self.patch_size,
        )
        if tuple(tensor.shape[-2:]) != target_size:
            tensor = F.interpolate(
                tensor,
                size=target_size,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        return (
            (tensor.squeeze(0) / 255.0 - self.norm_mean) / self.norm_std
        ).contiguous()

    def __call__(
        self,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        text: Optional[Union[str, list[str]]] = None,
        **kwargs: Any,
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You have to specify text.")
        texts = text if isinstance(text, list) else [text]
        flat_images = _flatten_images(images)
        max_num_patches = kwargs.pop("max_num_patches", None)
        num_tokens_available = kwargs.pop("num_tokens_available", None)
        # These are handled by the dedicated native-video path. Accept them so
        # callers can share a processor contract without changing CAPRL behavior.
        kwargs.pop("video_flags", None)
        kwargs.pop("video_temporal_patch_size", None)
        kwargs.pop("video_target_num_patches", None)
        kwargs.pop("video_maintain_aspect_ratio", None)

        if max_num_patches is not None:
            per_image = [int(max_num_patches)] * len(flat_images)
            sizes = [
                self._process_media(image, budget)[:2]
                for image, budget in zip(flat_images, per_image)
            ]
        elif num_tokens_available is not None:
            sizes = self._compute_params(flat_images, int(num_tokens_available))
        else:
            sizes = [
                self._process_media(image, self.max_num_patches)[:2]
                for image in flat_images
            ]

        pixel_values = [
            self._preprocess_image(image, patch_height, patch_width)
            for image, (patch_height, patch_width) in zip(flat_images, sizes)
        ]
        image_sizes = [
            [patch_height * self.patch_size, patch_width * self.patch_size]
            for patch_height, patch_width in sizes
        ]

        expanded_texts: list[str] = []
        for item in texts:
            parts = item.split(IMG_INPUT_TAG)
            if len(parts) - 1 != len(image_sizes):
                raise ValueError(
                    f"Found {len(parts) - 1} image placeholders for "
                    f"{len(image_sizes)} images."
                )
            expanded = parts[0]
            for (height, width), suffix in zip(image_sizes, parts[1:]):
                num_embeddings = self.compute_num_embeddings(height, width)
                expanded += IMG_START + IMG_CONTEXT * num_embeddings + IMG_END + suffix
            expanded_texts.append(expanded)

        text_inputs = self.tokenizer(
            expanded_texts,
            return_tensors=kwargs.get("return_tensors"),
            add_special_tokens=kwargs.get("add_special_tokens", False),
        )
        result = BatchFeature(data=dict(text_inputs))
        if pixel_values:
            max_height = max(value.shape[-2] for value in pixel_values)
            max_width = max(value.shape[-1] for value in pixel_values)
            result["pixel_values"] = torch.stack(
                [
                    F.pad(
                        value,
                        (
                            0,
                            max_width - value.shape[-1],
                            0,
                            max_height - value.shape[-2],
                        ),
                    )
                    for value in pixel_values
                ]
            )
            result["imgs_sizes"] = torch.tensor(image_sizes, dtype=torch.int32)
        return result

    def apply_chat_template(
        self, conversation: list[dict[str, Any]], tokenize: bool = True, **kwargs: Any
    ) -> Any:
        images: list[Image.Image] = []
        messages: list[dict[str, Any]] = []
        for message in conversation:
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") in (
                        "image",
                        "image_url",
                        "input_image",
                    ):
                        image = item.get("image")
                        if isinstance(image, Image.Image):
                            images.append(image)
                messages.append(self.conversation_preprocessor(message))
            else:
                messages.append(message)

        add_generation_prompt = kwargs.pop("add_generation_prompt", False)
        nested = dict(kwargs.pop("chat_template_kwargs", {}) or {})
        enable_thinking = kwargs.pop("enable_thinking", None)
        if enable_thinking is not None:
            nested["enable_thinking"] = enable_thinking
        render_kwargs = dict(nested)
        render_kwargs["add_generation_prompt"] = add_generation_prompt
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, **render_kwargs
        )
        if not tokenize:
            return rendered
        return self(text=rendered, images=images or None, **kwargs)

    def batch_decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.decode(*args, **kwargs)


def is_dynamic_resolution_model(config: PretrainedConfig) -> bool:
    # This wrapper implements the Nano VL V2 image-only contract. Super/Omni
    # configs also carry RADIO patch bounds but require their native processor
    # (and, for Omni, audio hooks), so do not classify from vision args alone.
    if getattr(config, "model_type", None) != "NemotronH_Nano_VL_V2":
        return False
    vision_args = getattr(getattr(config, "vision_config", None), "args", None)
    return bool(
        vision_args
        and "min_num_patches" in vision_args
        and "max_num_patches" in vision_args
    )
