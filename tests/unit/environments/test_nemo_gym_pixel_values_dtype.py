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
"""env.nemo_gym.pixel_values_dtype: rollout image tensors stored in a smaller dtype to cut host memory."""

import pytest
import torch
from PIL import Image

import nemo_rl.data.multimodal_utils as multimodal_utils
from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    attach_image_model_inputs_to_message,
)
from nemo_rl.environments.nemo_gym import (
    _build_gym_actor_config,
    get_pixel_values_dtype,
    resolve_pixel_values_dtype,
)


def test_resolve_pixel_values_dtype():
    assert resolve_pixel_values_dtype(None) is None
    assert resolve_pixel_values_dtype("bfloat16") is torch.bfloat16
    assert resolve_pixel_values_dtype("float32") is torch.float32
    with pytest.raises(ValueError, match="pixel_values_dtype"):
        resolve_pixel_values_dtype("int8")


def test_get_pixel_values_dtype_reads_env_config():
    assert get_pixel_values_dtype({}) is None
    assert get_pixel_values_dtype({"nemo_gym": {}}) is None
    assert (
        get_pixel_values_dtype({"nemo_gym": {"pixel_values_dtype": "bfloat16"}})
        is torch.bfloat16
    )


def _build(nemo_gym_dict):
    return _build_gym_actor_config(
        nemo_gym_dict,
        base_urls=["http://x"],
        model_name="m",
        enable_router_replay=False,
        use_fastokens=False,
    )


def test_build_gym_actor_config_keeps_pixel_values_dtype_out_of_gym_config():
    config = _build({"pixel_values_dtype": "bfloat16", "some_gym_key": 1})
    assert config["pixel_values_dtype"] == "bfloat16"
    assert "pixel_values_dtype" not in config["initial_global_config_dict"]
    assert config["initial_global_config_dict"]["some_gym_key"] == 1
    assert "pixel_values_dtype" not in _build({})


def test_build_gym_actor_config_rejects_unknown_pixel_values_dtype():
    with pytest.raises(ValueError, match="pixel_values_dtype"):
        _build({"pixel_values_dtype": "fp8"})


def test_attach_casts_pixel_tensors_but_not_image_metadata(monkeypatch):
    pixels = torch.rand(1, 3, 4, 4, dtype=torch.float32)
    sizes = torch.tensor([[4, 4]], dtype=torch.long)
    monkeypatch.setattr(
        multimodal_utils,
        "extract_multimodal_model_inputs",
        lambda processor, processed: {
            "pixel_values": PackedTensor([pixels], dim_to_pack=0),
            "imgs_sizes": PackedTensor([sizes], dim_to_pack=0),
        },
    )
    images = [Image.new("RGB", (4, 4))]

    def processor(**kwargs):
        return {}

    message = {}
    attach_image_model_inputs_to_message(
        message, images=images, processor=processor, pixel_dtype=torch.bfloat16
    )
    assert message["pixel_values"].tensors[0].dtype == torch.bfloat16
    assert torch.allclose(message["pixel_values"].tensors[0].float(), pixels, atol=1e-2)
    assert message["imgs_sizes"].tensors[0].dtype == torch.long

    unchanged = {}
    attach_image_model_inputs_to_message(unchanged, images=images, processor=processor)
    assert unchanged["pixel_values"].tensors[0].dtype == torch.float32


def test_initial_payload_attach_uses_env_pixel_values_dtype(monkeypatch):
    import nemo_rl.experience.rollouts as rollouts_mod

    seen = []

    def fake_attach(
        message, *, images, processor, pad_dynamic_image_shapes=False, pixel_dtype=None
    ):
        seen.append(pixel_dtype)
        message["pixel_values"] = PackedTensor([torch.ones(1, 3, 2, 2)], dim_to_pack=0)

    monkeypatch.setattr(
        rollouts_mod, "attach_image_model_inputs_to_message", fake_attach
    )
    monkeypatch.setattr(
        rollouts_mod, "resolve_to_image", lambda source: Image.new("RGB", (2, 2))
    )

    class _Processor:
        image_processor = object()

    batch = {
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
                                    "image_url": "data:image/png;base64,xx",
                                }
                            ],
                        }
                    ]
                }
            }
        ],
    }
    rollouts_mod.attach_initial_nemo_gym_image_payloads(
        batch, _Processor(), env_config={"nemo_gym": {"pixel_values_dtype": "bfloat16"}}
    )
    assert seen == [torch.bfloat16]
