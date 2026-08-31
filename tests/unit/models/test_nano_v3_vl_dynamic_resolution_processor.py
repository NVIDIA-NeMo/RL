# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from types import SimpleNamespace

from PIL import Image

from nemo_rl.models.nano_v3_vl import (
    DynamicResolutionProcessor,
    is_dynamic_resolution_model,
)


def test_conversation_preprocessor_preserves_interleaved_image_order() -> None:
    message = {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "before"},
            {"type": "input_image", "image": Image.new("RGB", (8, 8))},
            {"type": "input_text", "text": "between"},
            {"type": "input_image", "image": Image.new("RGB", (8, 8))},
        ],
    }

    processed = DynamicResolutionProcessor.conversation_preprocessor(message)

    assert processed["content"] == "before\n<image>\nbetween\n<image>"


def test_dynamic_resolution_model_detection_is_nano_v2_specific() -> None:
    vision_config = SimpleNamespace(
        args={"min_num_patches": 1024, "max_num_patches": 2040}
    )
    nano_config = SimpleNamespace(
        model_type="NemotronH_Nano_VL_V2", vision_config=vision_config
    )
    super_config = SimpleNamespace(
        model_type="NemotronH_Omni_Reasoning_V3", vision_config=vision_config
    )

    assert is_dynamic_resolution_model(nano_config)
    assert not is_dynamic_resolution_model(super_config)


def test_dynamic_tiler_matches_even_pixel_shuffle_grid() -> None:
    processor = object.__new__(DynamicResolutionProcessor)
    processor.patch_size = 16
    processor.min_num_patches = 16

    patch_height, patch_width, patch_count = processor._process_media(
        Image.new("RGB", (160, 96)), patch_budget=80
    )

    assert patch_height % 2 == 0
    assert patch_width % 2 == 0
    assert patch_count == patch_height * patch_width
    assert patch_count <= 80
