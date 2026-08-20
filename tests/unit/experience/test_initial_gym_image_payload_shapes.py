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
from nemo_rl.environments.nemo_gym import get_pad_dynamic_image_shapes


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

    def __init__(self):
        # Records what the caller asked for, so a test can assert the choice
        # directly instead of inferring it from an exception type.
        self.return_tensors_seen = []

    def __call__(self, *, text, images, return_tensors):
        self.return_tensors_seen.append(return_tensors)
        tiles = [
            torch.zeros(3, image.height, image.width, dtype=torch.float32)
            for image in images
        ]
        # One placeholder token per image, as the real processors emit.
        token_ids = [0] * len(images)
        # The real image processor always reports per-image extents, so the
        # derive-them-here fallback in _stack_ragged_pixel_values is dead in
        # production. Emit them, so this exercises the branch production takes.
        imgs_sizes = [[tile.shape[-2], tile.shape[-1]] for tile in tiles]
        all_same_shape = len({tuple(size) for size in imgs_sizes}) == 1
        if return_tensors == "pt" and all_same_shape:
            # BatchFeature converts every key, not just pixel_values.
            return {
                "pixel_values": torch.stack(tiles),
                "imgs_sizes": torch.tensor(imgs_sizes, dtype=torch.long),
                "input_ids": torch.tensor([token_ids]),
            }
        if return_tensors == "pt":
            # What the real processor does: it does NOT stack here. It hands
            # BatchFeature a ragged list, and transformers re-raises the torch
            # error as ValueError. Reproducing the real type matters -- a test
            # expecting RuntimeError would not catch the production failure.
            raise ValueError(
                "Unable to convert output 'pixel_values' (type: list) to tensor: "
                "stack expects each tensor to be equal size, but got "
                f"{list(tiles[0].shape)} at entry 0 and {list(tiles[1].shape)} at entry 1"
            )
        return {
            "pixel_values": [tile.tolist() for tile in tiles],
            "imgs_sizes": imgs_sizes,
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


def test_without_the_flag_a_mixed_resolution_prompt_is_asked_to_stack():
    """Documents the gap: the flag off asks the processor for stacked tensors.

    This is what the dedup path did before the flag was threaded, because the
    call site passed nothing and the parameter defaulted to off.

    The assertion is on the ``return_tensors`` the processor was handed, not on
    the exception: production raises ``ValueError`` from ``BatchFeature``, with
    the torch ``RuntimeError`` only as ``__cause__``, so matching on the torch
    message would pass against a fake while missing the real failure.
    """
    batch = _batch_with_two_differently_sized_images()
    processor = NemotronNanoVLV2Processor()

    with pytest.raises(ValueError, match="Unable to convert output"):
        rollouts_mod.attach_initial_nemo_gym_image_payloads(
            batch, processor, env_config={}
        )

    assert processor.return_tensors_seen == ["pt"]


def test_with_the_flag_the_processor_is_asked_for_ragged_output():
    """The flag-on multi-image case must request return_tensors=None."""
    batch = _batch_with_two_differently_sized_images()
    processor = NemotronNanoVLV2Processor()

    rollouts_mod.attach_initial_nemo_gym_image_payloads(
        batch,
        processor,
        env_config={"nemo_gym": {"pad_dynamic_image_shapes": True}},
    )

    assert processor.return_tensors_seen == [None]


def test_equal_resolution_multi_image_is_unchanged_by_the_flag():
    """Turning the flag on must not perturb the homogeneous case.

    Multi-image prompts at one resolution are the common shape; the ragged
    branch keys on image *count*, not on the resolutions differing, so they
    change code path when the flag flips. The payload must not change with it.
    """

    def _payload(env_config):
        batch = BatchedDataDict(
            {
                "message_log": [[{"role": "user", "content": ""}]],
                "extra_env_info": [
                    {
                        "responses_create_params": {
                            "input": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "input_image",
                                            "image_url": image_to_data_url(
                                                Image.new("RGB", (4, 4), color=colour)
                                            ),
                                        }
                                        for colour in ("red", "blue")
                                    ],
                                }
                            ]
                        }
                    }
                ],
            }
        )
        rollouts_mod.attach_initial_nemo_gym_image_payloads(
            batch, NemotronNanoVLV2Processor(), env_config=env_config
        )
        return batch["message_log"][0][0]

    off = _payload({})
    on = _payload({"nemo_gym": {"pad_dynamic_image_shapes": True}})

    assert off.keys() == on.keys()
    for key in ("pixel_values", "imgs_sizes"):
        assert torch.equal(off[key].as_tensor(), on[key].as_tensor()), key


def test_with_the_flag_the_prompt_is_padded_and_keeps_its_true_sizes():
    """The padded tensor is one shape, but imgs_sizes stays per-image.

    Padding is a batching convenience; the model needs the unpadded extents to
    crop each image back out, so they must be read before the pad.
    """
    batch = _batch_with_two_differently_sized_images()

    rollouts_mod.attach_initial_nemo_gym_image_payloads(
        batch,
        NemotronNanoVLV2Processor(),
        env_config={"nemo_gym": {"pad_dynamic_image_shapes": True}},
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


def test_flag_is_read_from_the_nemo_gym_env_config():
    assert (
        get_pad_dynamic_image_shapes({"nemo_gym": {"pad_dynamic_image_shapes": True}})
        is True
    )


def test_flag_defaults_off_when_unset_or_absent():
    """A run without NeMo-Gym must not fabricate a value for it."""
    assert get_pad_dynamic_image_shapes({"nemo_gym": {}}) is False
    assert get_pad_dynamic_image_shapes({}) is False
    assert get_pad_dynamic_image_shapes({"nemo_gym": None}) is False
    # The value a user writes to turn a recipe's `true` back off.
    assert (
        get_pad_dynamic_image_shapes({"nemo_gym": {"pad_dynamic_image_shapes": False}})
        is False
    )


# --------------------------------------------------------------------------
# the wiring: the flag must survive the trip from the config to the processor
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_config, expected_return_tensors",
    [
        ({"nemo_gym": {"pad_dynamic_image_shapes": True}}, None),
        ({"nemo_gym": {"pad_dynamic_image_shapes": False}}, "pt"),
        ({}, "pt"),
    ],
)
def test_env_config_reaches_the_processor(env_config, expected_return_tensors):
    """The helper resolves the flag from env config, not from a caller literal.

    This is the regression the fix addresses: the call sites previously passed
    nothing, so the prompt was processed under the wrong rules. Resolving inside
    the helper means there is no per-call-site value left to get wrong.
    """
    batch = _batch_with_two_differently_sized_images()
    processor = NemotronNanoVLV2Processor()

    if expected_return_tensors == "pt":
        with pytest.raises(ValueError, match="Unable to convert output"):
            rollouts_mod.attach_initial_nemo_gym_image_payloads(
                batch, processor, env_config=env_config
            )
    else:
        rollouts_mod.attach_initial_nemo_gym_image_payloads(
            batch, processor, env_config=env_config
        )

    assert processor.return_tensors_seen == [expected_return_tensors]
