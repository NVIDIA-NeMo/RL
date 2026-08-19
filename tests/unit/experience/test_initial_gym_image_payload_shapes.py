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

"""The initial Gym payload must honour pad_dynamic_image_shapes.

This path only runs when ``grpo.deduplicate_multimodal_data`` is true: the
driver attaches the prompt's image tensors once, before
``repeat_interleave(..., share_immutable_media=True)`` fans the prompt out. The
per-turn attach inside the NeMo-Gym actor already reads the flag, so without it
here the same prompt is processed under different rules on the two paths.

It bites only for a prompt carrying more than one image at differing
resolutions, which is exactly what a dynamic-resolution processor returns as a
ragged CHW list.
"""

import pytest
import torch
from PIL import Image

import nemo_rl.experience.rollouts as rollouts_mod
from nemo_rl.data.multimodal_utils import image_to_data_url
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class NemotronNanoVLV2Processor:
    """Stand-in for a processor that emits per-image resolutions.

    The name matters: ``uses_image_placeholder`` dispatches on the class name,
    so this exercises the same branch the real processor does.
    """

    image_token = "<image>"
    # The extractor unions these to decide which keys are model inputs.
    model_input_names = ["input_ids"]

    class _ImageProcessor:
        model_input_names = ["pixel_values", "imgs_sizes"]

    class _Tokenizer:
        # Subtracted from the union, leaving the media keys.
        model_input_names = ["input_ids"]

    image_processor = _ImageProcessor()
    tokenizer = _Tokenizer()

    def __call__(self, *, text, images, return_tensors):
        tiles = [
            torch.zeros(3, image.height, image.width, dtype=torch.float32)
            for image in images
        ]
        # One placeholder token per image, as the real processors emit.
        token_ids = [0] * len(images)
        if return_tensors == "pt":
            # What a real processor does for "pt": stack into one tensor. Images
            # of differing resolutions cannot stack, so this raises -- the
            # failure pad_dynamic_image_shapes exists to avoid.
            return {
                "pixel_values": torch.stack(tiles),
                "input_ids": torch.tensor([token_ids]),
            }
        # return_tensors=None hands everything back as plain lists, including
        # the ragged per-image pixel values this mode is requested for.
        return {
            "pixel_values": [tile.tolist() for tile in tiles],
            "input_ids": [token_ids],
        }


def _batch_with_two_differently_sized_images() -> BatchedDataDict:
    urls = [
        image_to_data_url(Image.new("RGB", (2, 3), color="red")),
        image_to_data_url(Image.new("RGB", (4, 5), color="blue")),
    ]
    return BatchedDataDict(
        {
            "message_log": [[{"role": "user", "content": ""}]],
            "extra_env_info": [
                {
                    "responses_create_params": {
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_image", "image_url": url}
                                    for url in urls
                                ],
                            }
                        ]
                    }
                }
            ],
        }
    )


def test_without_the_flag_a_mixed_resolution_prompt_fails_to_stack():
    """Documents the gap: the default asks the processor to stack the images.

    This is what the dedup path did before the flag was threaded, because the
    call site passed neither value and the parameter defaults to off.
    """
    batch = _batch_with_two_differently_sized_images()

    with pytest.raises(
        RuntimeError, match="stack expects each tensor to be equal size"
    ):
        rollouts_mod.attach_initial_nemo_gym_image_payloads(
            batch, NemotronNanoVLV2Processor()
        )


def test_with_the_flag_the_prompt_is_padded_and_keeps_its_true_sizes():
    """The padded tensor is one shape, but imgs_sizes stays per-image.

    Padding is a batching convenience; the model needs the unpadded extents to
    crop each image back out, so they must be read before the pad.
    """
    batch = _batch_with_two_differently_sized_images()

    rollouts_mod.attach_initial_nemo_gym_image_payloads(
        batch,
        NemotronNanoVLV2Processor(),
        pad_dynamic_image_shapes=True,
    )

    user_message = batch["message_log"][0][0]
    pixel_values = user_message["pixel_values"].as_tensor()
    # Padded up to the larger image, one row per image.
    assert pixel_values.shape[0] == 2
    assert pixel_values.shape[-2:] == torch.Size([5, 4])
    # The true per-image extents survive the pad: (height, width). Read off the
    # unpadded tiles, so the smaller image still reports 3x2 rather than 5x4.
    assert user_message["imgs_sizes"].as_tensor().tolist() == [[3, 2], [5, 4]]


# --------------------------------------------------------------------------
# reading the flag out of the config
# --------------------------------------------------------------------------


class _Config:
    def __init__(self, env):
        self.env = env


def test_flag_is_read_from_the_nemo_gym_env_config():
    from nemo_rl.algorithms.grpo import get_pad_dynamic_image_shapes

    assert (
        get_pad_dynamic_image_shapes(
            _Config({"nemo_gym": {"pad_dynamic_image_shapes": True}})
        )
        is True
    )


def test_flag_defaults_off_when_unset_or_absent():
    """A run without NeMo-Gym must not fabricate a value for it."""
    from nemo_rl.algorithms.grpo import get_pad_dynamic_image_shapes

    assert get_pad_dynamic_image_shapes(_Config({"nemo_gym": {}})) is False
    assert get_pad_dynamic_image_shapes(_Config({})) is False
