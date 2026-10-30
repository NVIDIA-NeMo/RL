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

"""CUA baseline and FD update tests against explicit reference calculations."""

import copy
import pickle
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from cua_test_utils import create_optimizer
from logra.cua import CuaConfig, TaskBaseline
from torch import Tensor


class SimpleChatTemplate:
    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        text = "".join(f"{row['role']}:{row['content']}|" for row in messages)
        return text + ("assistant:" if add_generation_prompt else "")


def test_turn_capture_preserves_exact_prefix_and_retains_compacted_segments() -> None:
    from logra.osworld import TurnRecorder

    recorder = TurnRecorder(SimpleChatTemplate())
    first = {"messages": [{"role": "user", "content": "image one"}]}
    reply = {
        "content": "action one",
        "prompt_token_ids": [4, 5],
        "generation_token_ids": [6, 7],
        "generation_log_probs": [-0.2, -0.3],
        "routed_experts": torch.tensor([[[0]], [[1]], [[2]], [[-1]]]),
    }
    recorder.record(first, reply)
    assert recorder.cached_response(first) == {"content": "action one"}
    second = {
        "messages": first["messages"]
        + [
            {"role": "assistant", "content": "reformatted text"},
            {"role": "user", "content": "image two"},
        ]
    }
    assert recorder.request_body(second)["required_prefix_token_ids"] == [4, 5, 6, 7]
    second_reply = {
        "content": "action two",
        "prompt_token_ids": [4, 5, 6, 7, 8],
        "generation_token_ids": [9],
        "generation_log_probs": [-0.4],
        "routed_experts": torch.full((6, 1, 1), 3),
    }
    recorder.record(second, second_reply)
    assert len(recorder.segments) == 1
    assert recorder.segments[0].token_mask.tolist() == [0, 0, 1, 1, 0, 1]
    assert recorder.segments[0].routed_experts.flatten().tolist() == [0, 1, 2, 3, 3, 3]
    compacted = copy.deepcopy(second)
    compacted["messages"][0]["content"] = "old image omitted"
    assert "required_prefix_token_ids" not in recorder.request_body(compacted)
    recorder.record(
        compacted,
        {
            **second_reply,
            "prompt_token_ids": [11, 12, 13],
            "routed_experts": torch.zeros((4, 1, 1), dtype=torch.long),
        },
    )
    assert len(recorder.segments) == 2
    assert sum(x.token_mask.sum().item() for x in recorder.segments) == 4
    # An engine violating the requested prefix must fail without changing data.
    third = {
        "messages": compacted["messages"]
        + [
            {"role": "assistant", "content": "action"},
            {"role": "user", "content": "new image"},
        ]
    }
    with pytest.raises(ValueError, match="preserve"):
        recorder.record(third, second_reply)
    assert len(recorder.segments) == 2


def test_image_positions_follow_exact_multiturn_tokens() -> None:
    from dataclasses import dataclass

    from logra.osworld import align_image_placeholders

    @dataclass
    class Span:
        offset: int
        length: int
        is_embed: torch.Tensor | None = None

    # The rendered history adds tokens around both earlier assistant replies.
    # A single offset for the newest screenshot would leave image two wrong.
    prompt = {
        "prompt_token_ids": [7, 18, 18, 5, 5, 5, 18, 18, 6, 6, 6, 18, 18, 9],
        "mm_placeholders": {"image": [Span(1, 2), Span(6, 2), Span(11, 2)]},
    }
    exact = [7, 18, 18, 5, 18, 18, 6, 18, 18, 9]
    align_image_placeholders(prompt, exact)
    assert [span.offset for span in prompt["mm_placeholders"]["image"]] == [1, 4, 7]
    assert all(
        exact[span.offset : span.offset + span.length] == [18, 18]
        for span in prompt["mm_placeholders"]["image"]
    )
    align_image_placeholders({}, [1, 2, 3])


def test_image_alignment_rejects_changed_media_geometry() -> None:
    from dataclasses import dataclass

    from logra.osworld import align_image_placeholders

    @dataclass
    class Span:
        offset: int
        length: int
        is_embed: torch.Tensor | None = None

    prompt = {
        "prompt_token_ids": [7, 18, 18, 9],
        "mm_placeholders": {"image": [Span(1, 2)]},
    }
    with pytest.raises(ValueError, match="disagree"):
        align_image_placeholders(prompt, [7, 18, 9])
    assert prompt["mm_placeholders"]["image"][0].offset == 1
    with pytest.raises(ValueError, match="image placeholders only"):
        align_image_placeholders({"mm_placeholders": {"video": [Span(1, 2)]}}, [18])


def test_image_alignment_preserves_native_boundary_tokens_and_mask() -> None:
    from dataclasses import dataclass

    from logra.osworld import align_image_placeholders

    @dataclass
    class Span:
        offset: int
        length: int
        is_embed: torch.Tensor

    mask = torch.tensor([False, True, True, False])
    image = [17, 18, 18, 19]
    prompt = {
        "prompt_token_ids": [1] + image + [5, 5, 5] + image + [7, 7] + image + [9],
        "mm_placeholders": {
            "image": [Span(1, 4, mask), Span(8, 4, mask), Span(14, 4, mask)]
        },
    }
    exact = [1] + image + [5] + image + [7] + image + [9]
    align_image_placeholders(prompt, exact)
    spans = prompt["mm_placeholders"]["image"]
    assert [span.offset for span in spans] == [1, 6, 11]
    assert all(span.is_embed is mask for span in spans)
    # Matching feature counts cannot silently accept a missing boundary marker.
    bad = {
        "prompt_token_ids": [1] + image + [9],
        "mm_placeholders": {"image": [Span(1, 4, mask)]},
    }
    with pytest.raises(ValueError, match="boundary tokens"):
        align_image_placeholders(bad, [1, 17, 18, 18, 20, 9])


def test_http_image_parts_preserve_inline_order_without_changing_history() -> None:
    from logra.osworld import TurnRecorder

    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,fixture"}}
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Task:\n"},
                    image,
                    {"type": "text", "text": "\nFeedback:"},
                    image,
                ],
            }
        ]
    }
    before = copy.deepcopy(payload)
    body = TurnRecorder(SimpleChatTemplate()).request_body(payload)
    assert body["messages"][0]["content"] == [
        {"type": "text", "text": "Task:\n<image>\nFeedback:<image>"},
        image,
        image,
    ]
    assert payload == before


class _ImageProcessor:
    """Tiny deterministic processor exercising native PackedTensor handling."""

    image_token_id = 99
    model_input_names = ["input_ids", "pixel_values"]
    tokenizer = SimpleNamespace(model_input_names=["input_ids"], pad_token_id=0)

    def __init__(self) -> None:
        self.calls = 0
        self.forbidden = False

    def __call__(
        self, *, text: str, images: list[Any], return_tensors: str
    ) -> dict[str, Tensor]:
        assert not self.forbidden, "The driver repeated actor-side image processing"
        self.calls += 1
        assert text == "<image>" * len(images)
        assert return_tensors == "pt"
        return {
            "input_ids": torch.full((1, len(images)), self.image_token_id),
            "pixel_values": torch.tensor(
                [image.getpixel((0, 0)) for image in images], dtype=torch.float32
            ).reshape(len(images), 3, 1, 1),
        }


@pytest.mark.parametrize("compress", [False, True])
def test_actor_prepared_media_preserves_batch_after_serialization(
    tmp_path: Path,
    compress: bool,
) -> None:
    from logra.osworld import CuaSegment, DesktopTask, Episode, make_training_batch
    from PIL import Image

    paths = [tmp_path / "red.png", tmp_path / "blue.png"]
    for path, colour in zip(paths, [(255, 0, 0), (0, 0, 255)], strict=True):
        Image.new("RGB", (1, 1), colour).save(path)
    episodes = []
    for index, ordered in enumerate((paths, paths[::-1])):
        segment = CuaSegment(
            torch.tensor([99, 5, 99, 6]),
            torch.tensor([0.0, 1.0, 0.0, 1.0]),
            torch.tensor([0.0, -0.2, 0.0, -0.3]),
            torch.arange(4).reshape(4, 1, 1),
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": str(path)}}
                        for path in ordered
                    ],
                }
            ],
        )
        episodes.append(
            Episode(
                DesktopTask(str(index), str(index), "test", {}),
                "v1",
                float(index == 0),
                True,
                None,
                [segment],
            )
        )
    baseline = TaskBaseline(CuaConfig())
    baseline.global_rate = 0.5
    reference = make_training_batch(
        copy.deepcopy(episodes), copy.deepcopy(baseline), _ImageProcessor(), 4
    )
    processor = _ImageProcessor()
    for episode in episodes:
        for segment in episode.segments:
            segment.prepare_media(processor)
            if compress:
                pixel_refs = [
                    weakref.ref(tensor)
                    for tensor in segment.media["pixel_values"].tensors
                ]
                segment.compress_media()
                assert segment.media is None
                assert all(reference() is None for reference in pixel_refs)
    # Ray transports these completed segments between the actor and driver.
    transported = pickle.loads(pickle.dumps(episodes))
    processor.forbidden = True
    actual = make_training_batch(transported, baseline, processor, 4)
    assert processor.calls == 2
    for key, value in reference.data.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
    expected_media = reference.get_multimodal_dict(as_tensors=True)
    actual_media = actual.get_multimodal_dict(as_tensors=True)
    assert actual_media.keys() == expected_media.keys()
    for key, value in expected_media.items():
        torch.testing.assert_close(actual_media[key], value, rtol=0, atol=0)
        assert torch.equal(
            actual_media[key].contiguous().view(torch.uint8),
            value.contiguous().view(torch.uint8),
        )

    def update(batch: Any, pixels: Tensor) -> tuple[Tensor, Tensor]:
        net, optimizer = create_optimizer()
        features = pixels.reshape(len(batch["input_lengths"]), -1)[:, [0, 2]] / 255
        features.requires_grad_()
        weights = batch["advantages"].sum(1) * batch["sample_mask"]
        (net(features).square().mean(1) * weights).sum().backward()
        optimizer.synchronize_and_clip(None, None)
        sketch = optimizer.layers[0].sketch.clone()
        assert sketch.any()
        optimizer.step()
        return sketch, net[0].weight.detach().clone()

    expected_sketch, expected_weight = update(reference, expected_media["pixel_values"])
    actual_sketch, actual_weight = update(actual, actual_media["pixel_values"])
    assert torch.equal(actual_sketch, expected_sketch)
    assert torch.equal(actual_weight, expected_weight)
    assert actual["sample_mask"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert actual["advantages"][0, 1].item() == 0.5
    assert actual["advantages"][1, 1].item() == -0.5
    assert actual_media["pixel_values"].flatten().tolist() == [
        255.0,
        0.0,
        0.0,
        0.0,
        0.0,
        255.0,
        0.0,
        0.0,
        255.0,
        255.0,
        0.0,
        0.0,
        255.0,
        0.0,
        0.0,
        0.0,
        0.0,
        255.0,
        255.0,
        0.0,
        0.0,
        0.0,
        0.0,
        255.0,
    ]


def test_media_preparation_rejects_image_token_mismatch(tmp_path: Path) -> None:
    from logra.osworld import CuaSegment
    from PIL import Image

    path = tmp_path / "image.png"
    Image.new("RGB", (1, 1), (1, 2, 3)).save(path)
    segment = CuaSegment(
        torch.tensor([5, 6]),
        torch.ones(2),
        torch.zeros(2),
        torch.zeros(2, 1, 1, dtype=torch.long),
        [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": str(path)}}],
            }
        ],
    )
    with pytest.raises(ValueError, match="image-token counts"):
        segment.prepare_media(_ImageProcessor())
    assert segment.media is None


@pytest.mark.parametrize("compress", [False, True])
def test_media_preserves_screenshots_after_resolution_change(
    tmp_path: Path, compress: bool
) -> None:
    """A display-setting action must not make earlier screenshots untrainable."""
    from logra.osworld import CuaSegment
    from PIL import Image
    from transformers.feature_extraction_utils import BatchFeature

    class NemotronNanoVLV2Processor:
        image_token_id = 99
        model_input_names = ["input_ids", "pixel_values"]
        tokenizer = SimpleNamespace(model_input_names=["input_ids"], pad_token_id=0)

        def __call__(self, *, text, images, return_tensors):
            return BatchFeature(
                {
                    "input_ids": [[99] * len(images)],
                    "pixel_values": [
                        torch.full(
                            (3, image.height, image.width),
                            float(image.getpixel((0, 0))[0]),
                        ).numpy()
                        for image in images
                    ],
                },
                tensor_type=return_tensors,
            )

    paths = [tmp_path / "before.png", tmp_path / "after.png"]
    for colour, (path, size) in enumerate(
        zip(paths, [(16, 32), (32, 16)], strict=True), start=1
    ):
        Image.new("RGB", size, (colour,) * 3).save(path)
    segment = CuaSegment(
        torch.tensor([99, 99, 7]),
        torch.tensor([0.0, 0.0, 1.0]),
        torch.zeros(3),
        torch.zeros(3, 1, 1, dtype=torch.long),
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": str(path)}}
                    for path in paths
                ],
            }
        ],
    )
    media = segment.prepare_media(NemotronNanoVLV2Processor())
    transported = pickle.loads(pickle.dumps(media))
    pixels = transported["pixel_values"].as_tensor()
    assert pixels.shape == (1, 4, 768)
    torch.testing.assert_close(pixels[:, :2], torch.ones(1, 2, 768))
    torch.testing.assert_close(pixels[:, 2:], torch.full((1, 2, 768), 2.0))
    assert transported["imgs_sizes"].as_tensor().tolist() == [[32, 16], [16, 32]]
    assert transported["num_frames"].as_tensor().tolist() == [1, 1]

    # AutoModel consumes padded NCHW images and crops each using imgs_sizes.
    # Checking only the default patchified representation misses this boundary.
    model_pixels = transported["pixel_values"].as_tensor(mode="pad_to_max_shape")
    assert model_pixels.shape == (2, 3, 32, 32)
    torch.testing.assert_close(model_pixels[0, :, :32, :16], torch.ones(3, 32, 16))
    torch.testing.assert_close(
        model_pixels[1, :, :16, :32], torch.full((3, 16, 32), 2.0)
    )

    # A conversation with multiple image sizes is still one training row.
    # Exercise the same native sharding boundary that rejected the real batch.
    from logra.osworld import DesktopTask, Episode, make_training_batch

    ordinary = copy.deepcopy(segment)
    ordinary.media = None
    ordinary.messages[0]["content"][1]["image_url"]["url"] = str(paths[0])
    processor = NemotronNanoVLV2Processor()
    ordinary_media = ordinary.prepare_media(processor)
    episodes = [
        Episode(
            DesktopTask("mixed", "mixed", "test", {}), "v1", 1.0, True, None, [segment]
        ),
        Episode(
            DesktopTask("same", "same", "test", {}), "v1", 0.0, True, None, [ordinary]
        ),
    ]
    if compress:
        for episode in episodes:
            for prepared_segment in episode.segments:
                prepared_segment.compress_media()
    episodes = pickle.loads(pickle.dumps(episodes))
    batch = make_training_batch(episodes, TaskBaseline(CuaConfig()), processor, 4)
    shards = batch.shard_by_batch_size(shards=4)
    assert batch["sample_mask"].tolist() == [1.0, 1.0, 0.0, 0.0]
    for shard, expected in zip(
        shards, [transported, ordinary_media, transported, transported], strict=True
    ):
        assert len(shard["input_lengths"]) == 1
        for key, value in shard.get_multimodal_dict(as_tensors=True).items():
            torch.testing.assert_close(value, expected[key].as_tensor(), rtol=0, atol=0)
        model_media = shard.get_multimodal_dict(
            as_tensors=True, pixel_preprocess_mode="pad_to_max_shape"
        )
        assert model_media["pixel_values"].ndim == 4
        torch.testing.assert_close(
            model_media["pixel_values"],
            expected["pixel_values"].as_tensor(mode="pad_to_max_shape"),
            rtol=0,
            atol=0,
        )
