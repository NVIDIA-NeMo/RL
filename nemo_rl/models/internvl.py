# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_internvl_chat.py
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import PretrainedConfig
from transformers.image_processing_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

IMG_START = '<img>'
IMG_END = '</img>'
IMG_CONTEXT = '<IMG_CONTEXT>'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def convert_image_mode(img: Image.Image, mode: str) -> Image.Image:
    """Converts PIL image mode if needed."""
    return img if img.mode == mode else img.convert(mode)


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
def build_transform(input_size: int):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose([
        T.Lambda(lambda img: convert_image_mode(img, 'RGB')),
        T.Resize((input_size, input_size),
                 interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    *,
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def resolve_internvl_min_max_num(
    *,
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    dynamic_image_size: bool,
    use_thumbnail: bool,
) -> tuple[int, int]:
    min_dynamic_patch = min_dynamic_patch if dynamic_image_size else 1
    max_dynamic_patch = max_dynamic_patch if dynamic_image_size else 1

    if use_thumbnail and max_dynamic_patch != 1:
        max_dynamic_patch += 1

    return min_dynamic_patch, max_dynamic_patch


def get_internvl_target_ratios(
    min_num: int,
    max_num: int,
) -> list[tuple[int, int]]:
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1)
                     for j in range(1, n + 1) if min_num <= i * j <= max_num}
    return sorted(target_ratios, key=lambda x: x[0] * x[1])


def calculate_internvl_targets(
    *,
    orig_width: int,
    orig_height: int,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> tuple[int, int, int]:
    aspect_ratio = orig_width / orig_height

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        width=orig_width,
        height=orig_height,
        image_size=image_size,
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # add thumbnail image if num_blocks != 1
    if use_thumbnail and blocks != 1:
        blocks += 1

    return blocks, target_width, target_height


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
def dynamic_preprocess_internvl(
    image: Image.Image,
    *,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> list[Image.Image]:
    orig_width, orig_height = image.size

    # calculate the number of blocks without thumbnail
    blocks, target_width, target_height = calculate_internvl_targets(
        orig_width=orig_width,
        orig_height=orig_height,
        target_ratios=target_ratios,
        image_size=image_size,
        use_thumbnail=False,
    )

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
def image_to_pixel_values_internvl(
    image: Image.Image,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
) -> torch.Tensor:
    target_ratios = get_internvl_target_ratios(min_num, max_num)

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess_internvl(
        image,
        target_ratios=target_ratios,
        image_size=input_size,
        use_thumbnail=use_thumbnail,
    )

    pixel_values = torch.stack([transform(image) for image in images])
    return pixel_values


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
def video_to_pixel_values_internvl(
    video: npt.NDArray,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
) -> torch.Tensor:
    target_ratios = get_internvl_target_ratios(min_num, max_num)

    transform = build_transform(input_size=input_size)
    frames_list = list[Image.Image]()
    for frame in video:
        pil_frame = dynamic_preprocess_internvl(
            Image.fromarray(frame, mode="RGB"),
            target_ratios=target_ratios,
            image_size=input_size,
            use_thumbnail=use_thumbnail,
        )
        assert len(pil_frame) == 1
        frames_list.extend(pil_frame)

    pixel_values = torch.stack([transform(image) for image in frames_list])
    return pixel_values


class InternVLProcessor(ProcessorMixin):
    """HuggingFace Processor for InternVL with image and video support."""

    attributes = ["tokenizer"]
    model_input_names = ["pixel_values", "image_flags"]


    def __init__(
        self,
        tokenizer,
        config: PretrainedConfig,
        *,
        chat_template=None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.chat_template = chat_template or tokenizer.chat_template

        image_size: int = config.vision_config.image_size
        patch_size: int = config.vision_config.patch_size

        self.image_size = image_size
        self.min_dynamic_patch = int(config.min_dynamic_patch)
        self.max_dynamic_patch = int(config.max_dynamic_patch)
        self.dynamic_image_size = bool(config.dynamic_image_size)
        self.use_thumbnail: bool = bool(config.use_thumbnail)
        self.image_seq_length = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio ** 2)
        )

        self.start_image_token = IMG_START
        self.end_image_token = IMG_END
        self.image_token = IMG_CONTEXT
        self.video_placeholder = "<video>"
        self.video_context_token = IMG_CONTEXT

    def _make_batch_input(self, input_item: Optional[Union[object, list[object]]] = None):
        if input_item is None:
            input_item = []
        if not isinstance(input_item, list):
            input_item = [input_item]
        return input_item

    def _resolve_min_max_num(
        self,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        use_thumbnail: Optional[bool] = None,
    ) -> tuple[int, int]:
        if min_dynamic_patch is None:
            min_dynamic_patch = self.min_dynamic_patch
        if max_dynamic_patch is None:
            max_dynamic_patch = self.max_dynamic_patch
        if dynamic_image_size is None:
            dynamic_image_size = self.dynamic_image_size
        if use_thumbnail is None:
            use_thumbnail = self.use_thumbnail
        return resolve_internvl_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
        )

    def _images_to_pixel_values_lst(
        self,
        images: list[Image.Image],
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> list[torch.Tensor]:
        min_num, max_num = self._resolve_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=False,
        )
        flat_images: list[Image.Image] = []
        for item in images:
            if isinstance(item, (list, tuple)):
                flat_images.extend(item)
            else:
                flat_images.append(item)
        return [
            image_to_pixel_values_internvl(
                image,
                input_size=self.image_size,
                min_num=min_num,
                max_num=max_num,
                use_thumbnail=self.use_thumbnail,
            )
            for image in flat_images
        ]

    def _videos_to_pixel_values_lst(
        self,
        videos: list[npt.NDArray],
        *,
        dynamic_image_size: Optional[bool] = None,
    ) -> list[torch.Tensor]:
        min_num, max_num = self._resolve_min_max_num(
            min_dynamic_patch=1,
            max_dynamic_patch=1,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=False,
        )
        return [
            video_to_pixel_values_internvl(
                video,
                input_size=self.image_size,
                min_num=min_num,
                max_num=max_num,
                use_thumbnail=False,
            )
            for video in videos
        ]

    def _insert_media_placeholders(
        self,
        text: list[str],
        image_pixel_values: Optional[torch.Tensor],
        video_pixel_values: Optional[torch.Tensor],
        image_num_patches: list[int],
        video_num_patches: list[int],
        image_num_patches_indices: np.ndarray,
        video_num_patches_indices: np.ndarray,
        video_patch_indices: np.ndarray,
    ):
        image_index = 0
        video_index = 0
        processed_text = []
        image_video_patches: list[torch.Tensor] = []
        for prompt in text:
            new_prompt = prompt
            replace_strings: list[str] = []
            while "<image>" in new_prompt or self.video_placeholder in new_prompt:
                if "<image>" in new_prompt and (
                    self.video_placeholder not in new_prompt
                    or new_prompt.index("<image>") < new_prompt.index(self.video_placeholder)
                ):
                    start_index = image_num_patches_indices[image_index - 1] if image_index > 0 else 0
                    end_index = image_num_patches_indices[image_index]
                    if image_pixel_values is not None:
                        image_video_patches.append(image_pixel_values[start_index:end_index])
                    replace_strings.append(
                        f"{self.start_image_token}{self.image_token * (self.image_seq_length * image_num_patches[image_index])}{self.end_image_token}"
                    )
                    new_prompt = new_prompt.replace("<image>", "<placeholder>", 1)
                    image_index += 1
                else:
                    current_patch_index = video_patch_indices[video_index]
                    end_patch_index = video_patch_indices[video_index + 1]
                    start_index = video_num_patches_indices[current_patch_index]
                    end_index = video_num_patches_indices[end_patch_index]
                    if video_pixel_values is not None:
                        image_video_patches.append(video_pixel_values[start_index:end_index])
                    num_patches = list(video_num_patches[current_patch_index:end_patch_index])
                    repl_features = self.video_context_token * self.image_seq_length
                    repl_features_with_sep = f"{self.start_image_token}{repl_features}{self.end_image_token}"
                    video_prompt = "".join(
                        f"Frame{i + 1}: {repl_features_with_sep}" for i in range(len(num_patches))
                    )
                    replace_strings.append(video_prompt)
                    new_prompt = new_prompt.replace(self.video_placeholder, "<placeholder>", 1)
                    video_index += 1
            while "<placeholder>" in new_prompt:
                replace_str = replace_strings.pop(0)
                new_prompt = new_prompt.replace("<placeholder>", replace_str, 1)
            processed_text.append(new_prompt)
        return processed_text, image_video_patches, image_index, video_index

    def __call__(
        self,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        text: Optional[Union[str, list[str]]] = None,
        videos: Optional[Union[npt.NDArray, list[npt.NDArray]]] = None,
        *,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You have to specify text.")

        if not isinstance(text, (list, tuple)):
            text = [text]

        image_num_patches: list[int] = []
        image_pixel_values: Optional[torch.Tensor] = None
        image_num_patches_indices = np.array([0])
        if images is not None:
            images_list = self._make_batch_input(images)
            pixel_values_lst = self._images_to_pixel_values_lst(images_list)
            image_num_patches = [len(item) for item in pixel_values_lst]
            image_pixel_values = torch.cat(pixel_values_lst) if len(pixel_values_lst) > 0 else None
            image_num_patches_indices = np.cumsum(image_num_patches)

        video_num_patches: list[int] = []
        video_pixel_values: Optional[torch.Tensor] = None
        video_patch_indices = np.array([0])
        video_num_patches_indices = np.array([0])
        num_frames_per_video: Optional[np.ndarray] = None
        if videos is not None:
            videos_list = self._make_batch_input(videos)
            pixel_values_lst_video = self._videos_to_pixel_values_lst(videos_list)
            if len(pixel_values_lst_video) > 0:
                frames_per_video = np.array([pv.shape[0] for pv in pixel_values_lst_video], dtype=int)
                num_frames_per_video = frames_per_video
                video_patch_indices = np.empty(len(pixel_values_lst_video) + 1, int)
                video_patch_indices[0] = 0
                video_patch_indices[1:] = np.cumsum(frames_per_video)
                video_num_patches = [1] * int(frames_per_video.sum())
                video_num_patches_indices = np.empty(len(video_num_patches) + 1, int)
                video_num_patches_indices[0] = 0
                video_num_patches_indices[1:] = np.cumsum(video_num_patches)
                video_pixel_values = torch.cat(pixel_values_lst_video)

        image_videos_inputs: dict[str, torch.Tensor] = {}
        if images is not None or videos is not None:
            processed_text, image_video_patches, image_index, video_index = self._insert_media_placeholders(
                list(text),
                image_pixel_values,
                video_pixel_values,
                image_num_patches,
                video_num_patches,
                image_num_patches_indices,
                video_num_patches_indices,
                video_patch_indices,
            )
            if images is not None:
                images_list_for_count = self._make_batch_input(images)
                expected_image_count = sum(
                    len(x) if isinstance(x, (list, tuple)) else 1 for x in images_list_for_count
                )
                if image_index != expected_image_count:
                    raise ValueError("Number of image placeholders in the prompt does not match the number of images.")
            if videos is not None and num_frames_per_video is not None and video_index != len(num_frames_per_video):
                raise ValueError("Number of video placeholders in the prompt does not match the number of videos.")

            text = processed_text
            if len(image_video_patches) > 0:
                pixel_values = torch.cat(image_video_patches)
                image_videos_inputs = {
                    "pixel_values": pixel_values,
                    "image_flags": torch.ones(pixel_values.shape[0], 1, dtype=torch.long),
                }

        tokenizer_allowed_keys = {
            "text_pair",
            "add_special_tokens",
            "padding",
            "truncation",
            "max_length",
            "stride",
            "is_split_into_words",
            "pad_to_multiple_of",
            "return_token_type_ids",
            "return_attention_mask",
            "return_overflowing_tokens",
            "return_special_tokens_mask",
            "return_offsets_mapping",
            "return_length",
            "verbose",
        }
        tokenizer_kwargs = {k: v for k, v in kwargs.items() if k in tokenizer_allowed_keys}
        text_inputs = self.tokenizer(text, **tokenizer_kwargs)
        return BatchFeature(data={**text_inputs, **image_videos_inputs}, tensor_type=return_tensors)
