# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real Super processor and Gym image attachment before startup."""

from typing import Any

from PIL import Image
import torch

from nemo_rl.data.multimodal_utils import (
    attach_image_model_inputs_to_message,
    uses_image_placeholder,
)


def check_processor_image_contract(processor: Any) -> None:
    """Check single, equal-size, and ragged images without altering rollout IDs."""
    assert uses_image_placeholder(processor), type(processor).__name__
    for sizes in [[(64, 96)], [(64, 96), (64, 96)], [(64, 96), (160, 64)]]:
        images = [
            Image.new("RGB", size, (40 + i * 80, 50, 90))
            for i, size in enumerate(sizes)
        ]
        tokens = torch.tensor([7, 11], dtype=torch.long)
        message = {"token_ids": tokens, "content": "preserved rollout content"}
        attach_image_model_inputs_to_message(
            message,
            images=images,
            processor=processor,
            pad_dynamic_image_shapes=True,
        )
        assert message["token_ids"] is tokens
        assert message["content"] == "preserved rollout content"
        pixels = message["pixel_values"].as_tensor()
        image_sizes = message["imgs_sizes"].as_tensor()
        assert pixels.shape[:2] == (len(images), 3)
        assert image_sizes.shape == (len(images), 2)
        assert message["num_frames"].as_tensor().tolist() == [1] * len(images)
        assert torch.isfinite(pixels).all()
        for index, image in enumerate(images):
            single = processor(
                text=processor.image_token, images=[image], return_tensors="pt"
            )
            expected = single["pixel_values"][0]
            height, width = expected.shape[-2:]
            assert image_sizes[index].tolist() == [height, width]
            torch.testing.assert_close(
                pixels[index, :, :height, :width], expected, rtol=0, atol=0
            )
            image_tokens = int((single["input_ids"] == processor.image_token_id).sum())
            assert image_tokens == int(single["num_tokens"][0]) > 0
        print(
            f"IMAGE_TOOLS_PROCESSOR_IMAGES_OK count={len(images)} sizes={image_sizes.tolist()}",
            flush=True,
        )
