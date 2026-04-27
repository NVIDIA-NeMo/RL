from typing import List, Optional, Union

from PIL import Image
import torch
from transformers.image_processing_base import BatchFeature
from transformers.image_processing_utils_fast import BaseImageProcessorFast
from transformers.image_utils import (make_list_of_images,
                                      get_image_type, ImageInput, ImageType)
from transformers.utils import TensorType
import torchvision.transforms as T



class NemotronNanoVLV2ImageProcessor(BaseImageProcessorFast):
    model_input_names = ["pixel_values", "image_flags"]

    def __init__(self,
        image_size=512,
        patch_size=16,
        max_num_tiles=12,
        use_thumbnail=True,
        downsample_ratio=0.5,
        norm_mean=None,
        norm_std=None,
        do_rescale=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.max_num_tiles = max_num_tiles
        self.use_thumbnail = use_thumbnail
        self.downsample_ratio = downsample_ratio
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.do_rescale = do_rescale

    def _process_image(
        self,
        image: ImageInput,
        **kwargs,
    ) -> Image.Image:
        image_type = get_image_type(image)
        if image_type not in [ImageType.PIL]:
            raise ValueError(f"Unsupported input image type {image_type}. Only PIL images supported")
        return image

    def _preprocess(
        self,
        images: List[Image.Image],
        image_size: int = None,
        max_num_tiles: int = None,
        use_thumbnail: bool = None,
        do_rescale: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> List[torch.Tensor]:
        image_size = image_size if image_size is not None else self.image_size
        max_num_tiles = max_num_tiles if max_num_tiles is not None else self.max_num_tiles
        use_thumbnail = use_thumbnail if use_thumbnail is not None else self.use_thumbnail
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale

        images = make_list_of_images(images)

        all_patches = []
        num_patches = []
        for image in images:
            patches = dynamic_preprocess(image, image_size, max_num_tiles, use_thumbnail)
            all_patches.extend(patches)
            num_patches.append(len(patches))

        pixel_values = torch.stack(all_patches, dim=0)
        norm_mean = torch.Tensor(self.norm_mean).view(1, 3, 1, 1)
        norm_std = torch.Tensor(self.norm_std).view(1, 3, 1, 1)
        pixel_values = (pixel_values - norm_mean) / norm_std
        image_inputs = {
            "pixel_values": pixel_values,
            "image_flags": torch.ones(pixel_values.shape[0], 1, dtype=torch.long),
            "num_patches": torch.tensor(num_patches),
        }
        return BatchFeature(data=image_inputs, tensor_type=return_tensors)


def get_internvl_target_ratios(
    min_num: int,
    max_num: int,
) -> list[tuple[int, int]]:
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1)
                     for j in range(1, n + 1) if min_num <= i * j <= max_num}
    return sorted(target_ratios, key=lambda x: x[0] * x[1])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_factor = float('-inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        factor_based_on_area_n_ratio = min(
            (ratio[0]*ratio[1]*image_size*image_size)/ area, 0.6
            )* min(
                target_aspect_ratio/aspect_ratio, aspect_ratio/target_aspect_ratio)
        if factor_based_on_area_n_ratio > best_factor:
            best_factor = factor_based_on_area_n_ratio
            best_ratio = ratio
    return best_ratio


def calculate_targets(
    orig_width: int,
    orig_height: int,
    target_ratios: list[tuple[int, int]],
    image_size: int,
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
    tiles = target_aspect_ratio[0] * target_aspect_ratio[1]

    return tiles, target_width, target_height


def dynamic_preprocess(image, image_size=512, max_num_tiles=12, use_thumbnail=True):
    orig_width, orig_height = image.size
    target_ratios = get_internvl_target_ratios(1, max_num_tiles)

    blocks, target_width, target_height = calculate_targets(
        orig_width,
        orig_height,
        target_ratios,
        image_size
    )
    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    processed_images = [img.convert('RGB') if img.mode != 'RGB' else img for img in processed_images]
    processed_images = [T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)(img) for img in processed_images]
    processed_images = [T.ToTensor()(img) for img in processed_images]

    return processed_images
