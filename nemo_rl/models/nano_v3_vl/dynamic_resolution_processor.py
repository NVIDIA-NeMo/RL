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
from typing import Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import BatchFeature, PretrainedConfig
from transformers.processing_utils import ProcessorMixin

# Configure PIL to handle large images without warnings
# This prevents DecompressionBombWarning for legitimate large images
Image.MAX_IMAGE_PIXELS = None

# Incoming prompt tags
IMG_INPUT_TAG = "<image>"
# Preprocessed prompt placeholders
IMG_START = "<img>"
IMG_END = "</img>"
IMG_CONTEXT = "<image>"


def _flatten_images(images):
    """Recursively flatten nested lists of images into a flat list."""
    if images is None:
        return []
    if isinstance(images, Image.Image):
        return [images]
    if isinstance(images, list):
        result = []
        for item in images:
            result.extend(_flatten_images(item))
        return result
    return [images]


class DynamicResolutionProcessor(ProcessorMixin):
    """Custom processor for dynamic resolution VLMs (Nano v3 VL).

    Bypasses HF's static tiling (InternVL-style) and implements
    dynamic resolution similar to vLLM's DynamicResolutionImageTiler.

    This processor:
    - Resizes images to dynamic dimensions (multiples of patch_size)
    - Uses min_num_patches/max_num_patches constraints from config
    - Emits N <image> tokens per image matching vLLM's convention
    - Returns imgs_sizes for variable resolution handling
    - Megatron collapses N→1 tokens before forward pass
    """

    attributes = ["tokenizer"]
    tokenizer_class = "PreTrainedTokenizerFast"
    model_input_names = ["pixel_values", "imgs_sizes"]

    def __init__(
        self,
        tokenizer,
        config: PretrainedConfig,
        *,
        chat_template: Optional[str] = None,
    ):
        super().__init__(tokenizer, chat_template=chat_template)
        self.config = config

        vision_args = getattr(config.vision_config, "args", {}) or {}
        self.patch_size = getattr(config.vision_config, "patch_size", 16)
        self.min_num_patches = vision_args.get("min_num_patches", 1024)
        self.max_num_patches = vision_args.get("max_num_patches", 13312)
        self.downsample_ratio = getattr(config, "downsample_ratio", 0.5)
        self.pixel_shuffle = getattr(config, "pixel_shuffle", True)

        norm_mean = vision_args.get("norm_mean", [0.48145466, 0.4578275, 0.40821073])
        norm_std = vision_args.get("norm_std", [0.26862954, 0.26130258, 0.27577711])
        self.norm_mean = torch.tensor(norm_mean)
        self.norm_std = torch.tensor(norm_std)

    def compute_num_embeddings(self, height: int, width: int) -> int:
        """Compute number of image embeddings for given dimensions.

        This must match vLLM's DynamicResolutionImageTiler._get_num_embeddings().
        Formula: (height // patch_size) * (width // patch_size) // downsample_ratio²
        """
        reduction_factor = int(1 / self.downsample_ratio)
        num_patches = (height // self.patch_size) * (width // self.patch_size)
        return num_patches // (reduction_factor ** 2)

    def compute_target_resolution(self, image: Image.Image) -> tuple[int, int]:
        """Compute dynamic target resolution for an image.

        Ported from vLLM's DynamicResolutionImageTiler.process_media()
        """
        orig_width, orig_height = image.size
        closest_patch_height = round(orig_height / self.patch_size + 0.5)
        closest_patch_width = round(orig_width / self.patch_size + 0.5)
        patches = closest_patch_height * closest_patch_width

        factor = min(math.sqrt(self.max_num_patches / patches), 1.0)
        target_patch_height = math.floor(factor * closest_patch_height)
        target_patch_width = math.floor(factor * closest_patch_width)

        if target_patch_height * target_patch_width < self.min_num_patches:
            up_factor = math.sqrt(
                self.min_num_patches / (target_patch_height * target_patch_width)
            )
            target_patch_height = math.ceil(up_factor * target_patch_height)
            target_patch_width = math.ceil(up_factor * target_patch_width)

        if self.pixel_shuffle:
            required_divisor = 2
            rem_h = target_patch_height % required_divisor
            if rem_h != 0:
                inc_h = required_divisor - rem_h
                if (target_patch_height + inc_h) * target_patch_width <= self.max_num_patches:
                    target_patch_height += inc_h
                else:
                    target_patch_height = max(required_divisor, target_patch_height - rem_h)

            rem_w = target_patch_width % required_divisor
            if rem_w != 0:
                inc_w = required_divisor - rem_w
                if target_patch_height * (target_patch_width + inc_w) <= self.max_num_patches:
                    target_patch_width += inc_w
                else:
                    target_patch_width = max(required_divisor, target_patch_width - rem_w)

        target_height = target_patch_height * self.patch_size
        target_width = target_patch_width * self.patch_size
        return target_height, target_width

    def preprocess_image(
        self, image: Image.Image
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        target_h, target_w = self.compute_target_resolution(image)
        resized = image.resize((target_w, target_h), Image.BICUBIC)

        tensor = transforms.ToTensor()(resized)
        tensor = (tensor - self.norm_mean.view(3, 1, 1)) / self.norm_std.view(3, 1, 1)

        return tensor, (target_h, target_w)

    def _add_image_placeholders(
        self,
        text: list[str],
        imgs_sizes_list: list[list[int]],
    ) -> list[str]:
        if len(imgs_sizes_list) == 0:
            return text

        results_lst = []
        for t in text:
            parts = t.split(IMG_INPUT_TAG)
            assert len(parts) - 1 == len(imgs_sizes_list), (
                f"Number of {IMG_INPUT_TAG} tokens ({len(parts) - 1}) "
                f"doesn't match number of images ({len(imgs_sizes_list)})"
            )
            result = parts[0]
            for (h, w), part in zip(imgs_sizes_list, parts[1:]):
                num_embeddings = self.compute_num_embeddings(h, w)
                image_placeholder = IMG_START + IMG_CONTEXT * num_embeddings + IMG_END
                result += image_placeholder + part
            results_lst.append(result)
        return results_lst

    def __call__(
        self,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        text: Optional[Union[str, list[str]]] = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You have to specify text.")

        if not isinstance(text, list):
            text = [text]

        pixel_values_list: list[torch.Tensor] = []
        imgs_sizes_list: list[list[int]] = []

        if images is not None:
            flat_images = _flatten_images(images)

            for image in flat_images:
                if not isinstance(image, Image.Image):
                    raise ValueError(f"Expected PIL Image, got {type(image)}")
                pv, (h, w) = self.preprocess_image(image)
                pixel_values_list.append(pv)
                imgs_sizes_list.append([h, w])

        processed_text = self._add_image_placeholders(text, imgs_sizes_list)

        text_inputs = self.tokenizer(
            processed_text,
            return_tensors=kwargs.get("return_tensors"),
            add_special_tokens=kwargs.get("add_special_tokens", False),
        )

        result = BatchFeature(data=dict(text_inputs))

        if pixel_values_list:
            max_h = max(s[0] for s in imgs_sizes_list)
            max_w = max(s[1] for s in imgs_sizes_list)
            padded_pvs = []
            for pv, (h, w) in zip(pixel_values_list, imgs_sizes_list):
                pad_h = max_h - h
                pad_w = max_w - w
                if pad_h > 0 or pad_w > 0:
                    pv = F.pad(pv, (0, pad_w, 0, pad_h), value=0)
                padded_pvs.append(pv)

            result["pixel_values"] = torch.stack(padded_pvs)
            # imgs_sizes: actual pixel dimensions (NOT scaled by downsample_ratio)
            # RADIO uses imgs_sizes to compute patch counts for position encoding
            # LLaVAModel._preprocess_data applies pixel_shuffle reduction INTERNALLY
            # E.g., for 448x640 image: RADIO gets 1120 patches, model reduces to 280 embeddings
            result["imgs_sizes"] = torch.tensor(imgs_sizes_list, dtype=torch.int32)

        return result

    def apply_chat_template(self, conversation, tokenize=True, **kwargs):
        """Override to handle multimodal content lists in messages.

        The base Jinja2 chat template renders list content via the template
        (which now handles <image> tags correctly), but for tokenize=True
        we need to also process PIL images into pixel_values by calling
        self() after rendering the text.
        """
        images = []
        for msg in conversation:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        img = item.get("image")
                        if isinstance(img, Image.Image):
                            images.append(img)

        if not tokenize:
            return super().apply_chat_template(
                conversation, tokenize=False, **kwargs
            )

        add_generation_prompt = kwargs.pop("add_generation_prompt", False)
        rendered_text = super().apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return self(
            text=rendered_text,
            images=images or None,
            **kwargs,
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


def is_dynamic_resolution_model(config: PretrainedConfig) -> bool:
    """Check if model uses dynamic resolution (not static InternVL tiling)."""
    if not hasattr(config, "vision_config"):
        return False
    vision_args = getattr(config.vision_config, "args", None)
    if vision_args is None:
        return False
    return "min_num_patches" in vision_args and "max_num_patches" in vision_args
