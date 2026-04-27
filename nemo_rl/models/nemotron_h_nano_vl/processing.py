from typing import Any, Optional, Union

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import BatchEncoding, ProcessorMixin, TensorType
from transformers.image_processing_utils_fast import divide_to_patches

from .image_processing import NemotronNanoVLV2ImageProcessor

# Configure PIL to handle large images without warnings
# This prevents DecompressionBombWarning for legitimate large images
Image.MAX_IMAGE_PIXELS = None

# incoming prompt tags
IMG_INPUT_TAG = "<image>"
# preprocessed prompt placeholders
IMG_START = "<img>"
IMG_END = "</img>"
IMG_CONTEXT = "<image>"


class NemotronNanoVLV2Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "NemotronNanoVLV2ImageProcessor"
    tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(
        self,
        image_processor: NemotronNanoVLV2ImageProcessor = None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT)
        self.image_size = image_processor.image_size
        self.num_image_token = int(
            (image_processor.image_size // image_processor.patch_size) ** 2
            * (image_processor.downsample_ratio**2)
        )

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ) -> BatchEncoding:
        text, images = [self._make_batch_input(x) for x in (text, images)]
        if images:
            image_inputs = self.image_processor(images)
            text = self._add_image_placeholders(text, image_inputs)
        else:
            image_inputs = {}
        text_inputs = self.tokenizer(text, add_special_tokens=False)
        return BatchEncoding(
            data=dict(**text_inputs, **image_inputs), tensor_type=return_tensors
        )

    def _add_image_placeholders(
        self,
        text: list[str],
        image_inputs: dict[str, torch.Tensor],
    ) -> list[str]:
        image_num_patches = image_inputs["num_patches"].tolist()
        if len(image_num_patches) == 0:
            return text

        results_lst = []

        for t in text:
            parts = t.split(IMG_INPUT_TAG)
            assert len(parts) - 1 == len(image_num_patches), (
                f"Number of {IMG_INPUT_TAG} tokens ({len(parts) - 1}) doesn't match num_patches_list length ({len(image_num_patches)})"
            )

            result = parts[0]
            for num_tiles, part in zip(image_num_patches, parts[1:]):
                feature_size = num_tiles * self.num_image_token
                image_placeholder_features = IMG_CONTEXT * feature_size
                image_placeholder = IMG_START + image_placeholder_features + IMG_END
                result += image_placeholder + part
            results_lst.append(result)
        return results_lst

    def _make_batch_input(self, input_item: Optional[Union[Any, list[Any]]] = None):
        if input_item is None:
            input_item = []
        if not isinstance(input_item, list):
            input_item = [input_item]
        return input_item
