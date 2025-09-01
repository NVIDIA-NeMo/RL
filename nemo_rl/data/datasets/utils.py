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
import io
from typing import Union

import torch
from PIL import Image
from transformers import AutoProcessor, PreTrainedTokenizerBase

TokenizerType = Union[PreTrainedTokenizerBase, AutoProcessor]


def assert_no_double_bos(token_ids: torch.Tensor, tokenizer: TokenizerType) -> None:
    """Assert that there are no double starting BOS tokens in the message.

    Args:
        token_ids: List of token IDs
        tokenizer: Tokenizer
    """
    if tokenizer.bos_token_id is not None:
        token_ids_list = token_ids.tolist()
        if len(token_ids_list) > 1:
            assert not (
                token_ids_list[0] == tokenizer.bos_token_id
                and token_ids_list[1] == tokenizer.bos_token_id
            ), "Found double BOS token in the first two positions of the message."
    else:
        # `name_or_path` is not available for AutoProcessor, temp fix in get_tokenizer
        print(
            f"skip assert_start_single_bos since Tokenizer {tokenizer.name_or_path} has no BOS token"
        )


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Converts a PIL Image object to a base64 encoded string.

    Args:
        image: The PIL Image object to convert.
        format: The image format (e.g., "PNG", "JPEG"). Defaults to "PNG".

    Returns:
        A base64 encoded string representation of the image.
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"
