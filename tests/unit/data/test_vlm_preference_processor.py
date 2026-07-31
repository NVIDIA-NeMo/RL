# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import torch
from PIL import Image

import nemo_rl.data.processors as processors
from nemo_rl.data.interfaces import TaskDataSpec


class _Tokenizer:
    model_input_names = ["input_ids"]

    def get_vocab(self):
        return {"<image>": 9}


class _ImageProcessor:
    model_input_names = ["pixel_values"]


class NemotronH_Nano_Omni_Reasoning_V3Processor:
    tokenizer = _Tokenizer()
    image_processor = _ImageProcessor()
    model_input_names = ["input_ids", "pixel_values"]
    image_token = "<image>"
    bos_token = "<bos>"
    eos_token = "<eos>"

    def __init__(self):
        self.saw_explicit_image_placeholder = False
        self.chat_template_kwargs = []

    def apply_chat_template(self, messages, **kwargs):
        self.chat_template_kwargs.append(kwargs)
        return "".join(
            f"{message['role']}:{message['content']}\n" for message in messages
        )

    def __call__(self, text, images=None, **kwargs):
        del kwargs
        text = text[0] if isinstance(text, list) else text
        if images:
            self.saw_explicit_image_placeholder = "<image>" in text
            return {
                "input_ids": torch.tensor([[9, 10]]),
                "pixel_values": torch.ones(1, 3, 16, 24),
                # The processor can report an unpadded crop that differs from
                # the pixel tensor's padded spatial shape.
                "imgs_sizes": torch.tensor([[15, 23]]),
            }
        return {"input_ids": torch.tensor([[11]])}


def test_vlm_preference_processor_adds_nemotron_omni_media_metadata():
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    result = processors.vlm_preference_preprocessor(
        {
            "context": [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": Image.new("RGB", (24, 16))}],
                }
            ],
            "completions": [
                {
                    "rank": 0,
                    "completion": [{"role": "assistant", "content": "chosen"}],
                },
                {
                    "rank": 1,
                    "completion": [{"role": "assistant", "content": "rejected"}],
                },
            ],
        },
        TaskDataSpec(task_name="mmpr"),
        processor,
        max_seq_length=128,
        idx=7,
    )

    assert processor.saw_explicit_image_placeholder
    assert all(
        "add_special_tokens" not in kwargs for kwargs in processor.chat_template_kwargs
    )
    assert result["idx"] == 7
    assert result["length_chosen"] == 3
    for message_log in (
        result["message_log_chosen"],
        result["message_log_rejected"],
    ):
        message = message_log[0]
        assert message["pixel_values"].pad_to_max_shape
        assert message["imgs_sizes"].as_tensor().tolist() == [[15, 23]]
        assert message["num_frames"].as_tensor().tolist() == [1]
