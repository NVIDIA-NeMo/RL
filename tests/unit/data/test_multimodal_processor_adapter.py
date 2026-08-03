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

import pytest
import torch
from PIL import Image

from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    extract_multimodal_model_inputs,
    process_multimodal_chat,
    register_multimodal_processor_adapter,
)


def _messages() -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (2, 2))},
                {"type": "text", "text": "describe"},
            ],
        }
    ]


class _ImageProcessor:
    model_input_names = ["pixel_values", "image_grid_thw"]


class _Tokenizer:
    model_input_names = ["input_ids", "attention_mask"]


class StandardProcessor:
    image_processor = _ImageProcessor()
    tokenizer = _Tokenizer()
    model_input_names = [
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
    ]

    def __init__(self):
        self.tokenized_messages = None

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, **kwargs
    ):
        assert add_generation_prompt
        if not tokenize:
            return "rendered"
        self.tokenized_messages = messages
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.ones(2, 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "token_type_ids": torch.tensor([[0, 1, 1]]),
            "mm_token_type_ids": torch.tensor([[0, 1, 1]]),
        }


def test_standard_hf_adapter_and_model_input_extraction():
    processor = StandardProcessor()
    formatted, processed = process_multimodal_chat(
        processor, _messages(), add_generation_prompt=True
    )
    model_inputs = extract_multimodal_model_inputs(processor, processed)

    assert formatted == "rendered"
    assert processor.tokenized_messages[0]["content"][0]["type"] == "image"
    assert isinstance(
        processor.tokenized_messages[0]["content"][0]["image"], Image.Image
    )
    assert isinstance(model_inputs["pixel_values"], PackedTensor)
    assert isinstance(model_inputs["image_grid_thw"], PackedTensor)
    assert model_inputs["token_type_ids"].tolist() == [0, 1, 1]
    assert model_inputs["mm_token_type_ids"].tolist() == [0, 1, 1]


def test_smolvlm_inputs_pack_along_dimension_one():
    class SmolVLMProcessor(StandardProcessor):
        pass

    processor = SmolVLMProcessor()
    _, processed = process_multimodal_chat(
        processor, _messages(), add_generation_prompt=True
    )
    model_inputs = extract_multimodal_model_inputs(processor, processed)
    assert model_inputs["pixel_values"].dim_to_pack == 1
    assert model_inputs["image_grid_thw"].dim_to_pack == 1


def test_registered_nemotron_placeholder_adapter():
    class NemotronNanoVLV2Processor(StandardProcessor):
        image_token = "<image>"

        def __init__(self):
            self.call = None

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"] == "<image>\ndescribe"
            return messages[0]["content"]

        def __call__(self, *, text, images, return_tensors):
            self.call = (text, images, return_tensors)
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "pixel_values": torch.ones(1, 3, 2, 2),
            }

    processor = NemotronNanoVLV2Processor()
    formatted, _ = process_multimodal_chat(
        processor, _messages(), add_generation_prompt=True
    )
    assert formatted == "<image>\ndescribe"
    assert processor.call[0] == formatted
    assert len(processor.call[1]) == 1


def test_custom_processor_adapter_registration():
    class CustomProcessor(StandardProcessor):
        pass

    class CustomAdapter:
        def process(self, processor, messages, *, add_generation_prompt):
            assert add_generation_prompt
            return "custom", {
                "input_ids": torch.tensor([[4, 5]]),
                "pixel_values": torch.ones(1, 3, 2, 2),
            }

    register_multimodal_processor_adapter("CustomProcessor", CustomAdapter())
    formatted, processed = process_multimodal_chat(
        CustomProcessor(), _messages(), add_generation_prompt=True
    )
    assert formatted == "custom"
    assert processed["input_ids"].tolist() == [[4, 5]]


def test_images_without_visual_model_inputs_fail_loudly():
    class MissingVisualProcessor(StandardProcessor):
        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt, **kwargs
        ):
            if not tokenize:
                return "rendered"
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    with pytest.raises(ValueError, match="returned no visual model inputs"):
        process_multimodal_chat(
            MissingVisualProcessor(), _messages(), add_generation_prompt=True
        )


@pytest.mark.parametrize("key", ["token_type_ids", "mm_token_type_ids"])
def test_malformed_sequence_auxiliary_length_fails_loudly(key):
    processor = StandardProcessor()
    processed = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        key: torch.tensor([[0, 1]]),
    }
    with pytest.raises(ValueError, match=f"{key!r} has length 2"):
        extract_multimodal_model_inputs(processor, processed)
