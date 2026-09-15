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

# ruff: noqa: E402

import re
from dataclasses import dataclass
from io import BytesIO

import pytest

pytest.importorskip("megatron.energon")
pytest.importorskip("megatron.core")

pytestmark = pytest.mark.mcore

import torch
from PIL import Image

from nemo_rl.data.energon.multimodal.task_encoders.generic_sft import (
    HFMultimodalSFTProcessorAdapter,
)
from nemo_rl.data.energon.multimodal.task_encoders.nemotron_multimodal import (
    NemotronMultiModalProcessorAdapter,
    NemotronMultiModalTaskEncoder,
)
from nemo_rl.data.energon.multimodal.task_encoders.nemotron_visual import (
    COMPACT_IMAGE_PLACEHOLDER,
)
from nemo_rl.data.energon.multimodal.types import (
    CanonicalSFTSample,
    MediaRef,
    freeze_media_metadata,
)
from nemo_rl.data.llm_message_utils import message_log_to_flat_messages
from nemo_rl.data.multimodal_utils import PackedTensor


@dataclass(frozen=True)
class _FakeMedia:
    name: str
    height: int
    width: int
    marker: float
    frames: int = 1
    tiles: int = 1


class _FakeTokenizer:
    bos_token = None
    eos_token = None
    pad_token_id = 0
    model_input_names = ["input_ids", "attention_mask"]
    unk_token_id = -1
    _special_ids = {"<image>": 803}
    _pattern = re.compile("(" + "|".join(map(re.escape, _special_ids)) + ")")

    def convert_tokens_to_ids(self, token):
        return self._special_ids.get(token, self.unk_token_id)

    def __call__(self, text, **kwargs):
        del kwargs
        value = text[0] if isinstance(text, list) else text
        ids = []
        for part in self._pattern.split(value):
            if not part:
                continue
            if part in self._special_ids:
                ids.append(self._special_ids[part])
            else:
                ids.extend(ord(character) % 251 + 1 for character in part)
        token_ids = torch.tensor(
            [ids],
            dtype=torch.long,
        )
        return {
            "input_ids": token_ids,
            "attention_mask": torch.ones_like(token_ids),
        }


class _FakeVisualProcessor:
    model_input_names = ["pixel_values", "imgs_sizes", "num_frames"]
    patch_size = 16
    _downsample_factor = 2
    min_num_patches = 1
    max_num_patches = 4_096
    max_model_len = 4_096

    def _compute_target_patches(self, image, tokens_available):
        assert tokens_available > 0
        return image.width // self.patch_size, image.height // self.patch_size


class NemotronH_Nano_Omni_Reasoning_V3Processor:
    tokenizer = _FakeTokenizer()
    image_processor = _FakeVisualProcessor()
    model_input_names = [
        "input_ids",
        "attention_mask",
        "pixel_values",
        "imgs_sizes",
        "num_frames",
    ]
    bos_token = None
    eos_token = None
    name_or_path = "fake-nemotron-omni"

    def __init__(self) -> None:
        self.visual_calls: list[tuple[str, object]] = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is False

        def render_content(content):
            if isinstance(content, str):
                return content
            return "".join(part.get("text", "") for part in content)

        return "".join(
            f"<{message['role']}>{render_content(message.get('content', ''))}"
            f"</{message['role']}>"
            for message in messages
        )

    def __call__(self, *, text, images=None, videos=None, **kwargs):
        text_output = self.tokenizer(text, **kwargs)
        if images is None and videos is None:
            return text_output
        if images is not None and videos is not None:
            raise AssertionError(
                "The focused fixture processes one media item at a time."
            )
        modality = "image" if images is not None else "video"
        payload = (images if images is not None else videos)[0]
        self.visual_calls.append((modality, payload))
        rows = (
            getattr(payload, "tiles", 1)
            if modality == "image"
            else getattr(payload, "frames")
        )
        marker = getattr(payload, "marker", 4.0)
        return {
            **text_output,
            "pixel_values": torch.full(
                (rows, 3, payload.height, payload.width),
                marker,
            ),
            "imgs_sizes": torch.tensor(
                [[payload.height, payload.width]] * rows,
                dtype=torch.long,
            ),
            "num_frames": torch.tensor(
                [1] * rows if modality == "image" else [payload.frames],
                dtype=torch.long,
            ),
        }


def _metadata(
    *,
    height: int,
    width: int,
    num_tiles: int | None = None,
    sampled_num_frames: int | None = None,
    sampled_fps: float | None = None,
):
    values = {
        "processed_height": height,
        "processed_width": width,
    }
    if num_tiles is not None:
        values["num_tiles"] = num_tiles
    if sampled_num_frames is not None:
        values["sampled_num_frames"] = sampled_num_frames
    if sampled_fps is not None:
        values["sampled_fps"] = sampled_fps
    return freeze_media_metadata(values)


def _sample(
    content: list[dict[str, object]] | str,
    media: list[MediaRef],
) -> CanonicalSFTSample:
    return CanonicalSFTSample(
        __key__="nemotron-0",
        __restore_key__=("nemotron-0",),
        messages=[
            {"role": "user", "content": content},
            {"role": "assistant", "content": "done"},
        ],
        media=media,
        tools=None,
    )


def _encoder(
    processor: NemotronH_Nano_Omni_Reasoning_V3Processor,
    *,
    temporal_patch_size: int = 2,
) -> NemotronMultiModalTaskEncoder:
    adapter = NemotronMultiModalProcessorAdapter(
        processor=processor,
        max_sequence_length=16_384,
        patch_dim=16,
        temporal_patch_size=temporal_patch_size,
    )
    return NemotronMultiModalTaskEncoder(
        adapter=adapter,
        cooker_functions=[],
        packer=None,
        include_source_ids=False,
    )


def test_loader_hf_adapter_builds_nemotron_visual_adapter() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    hf_adapter = HFMultimodalSFTProcessorAdapter(
        processor=processor,
        max_sequence_length=4_096,
        add_bos=False,
        add_eos=False,
        add_generation_prompt=True,
    )

    encoder = NemotronMultiModalTaskEncoder(
        adapter=hf_adapter,
        cooker_functions=[],
        packer=None,
        include_source_ids=False,
    )

    assert isinstance(encoder.adapter, NemotronMultiModalProcessorAdapter)
    assert encoder.adapter.processor is processor
    assert encoder.adapter.max_sequence_length == 4_096
    assert encoder.adapter.add_generation_prompt is True
    assert encoder.decoder is None


def test_text_only_sample_keeps_equal_compact_and_expanded_costs() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor)
    sample = _sample("describe the result", [])

    preencoded = encoder.preencode_sample(sample)
    assert preencoded.packing_cost == preencoded.length
    assert preencoded.pending_sample is sample
    assert processor.visual_calls == []

    postencoded = encoder.postencode_sample(preencoded)
    assert postencoded.pending_sample is None
    assert postencoded.length == postencoded.packing_cost
    assert [message["role"] for message in postencoded.message_log] == [
        message["role"] for message in preencoded.message_log
    ]
    assert all(
        torch.equal(after["token_ids"], before["token_ids"])
        for after, before in zip(
            postencoded.message_log,
            preencoded.message_log,
            strict=True,
        )
    )
    assert processor.visual_calls == []


def test_assistant_thinking_trace_is_normalized_before_tokenization() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor)
    sample = _sample("question", [])
    sample.messages[1]["content"] = "<think>  reason  </think>  answer"

    preencoded = encoder.preencode_sample(sample)

    assert (
        "<think>\nreason\n</think>\n\nanswer"
        in preencoded.message_log[1]["content"][0]["text"]
    )


def test_preformatted_sample_skips_thinking_check_and_chat_template() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor)
    sample = _sample("<|im_start|>assistant\n<think>\n", [])
    sample.messages[1]["content"] = "reasoning</think>answer<|im_end|>\n"
    sample.__subflavors__["skip_chat_template"] = True

    preencoded = encoder.preencode_sample(sample)

    expected = processor.tokenizer(
        "<|im_start|>assistant\n<think>\nreasoning</think>answer<|im_end|>\n"
    )["input_ids"][0]
    actual = torch.cat(
        [message["token_ids"] for message in preencoded.message_log]
    )
    assert torch.equal(actual, expected)


def test_invalid_thinking_trace_logs_complete_conversation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor)
    sample = _sample("question from the failing sample", [])
    sample.messages[1]["content"] = "<think>first</think><think>second</think>answer"

    with caplog.at_level(
        "ERROR",
        logger="nemo_rl.data.energon.multimodal.task_encoders.nemotron_visual",
    ):
        with pytest.raises(ValueError, match="Sample key: 'nemotron-0'"):
            encoder.preencode_sample(sample)

    assert "[NEMOTRON_THINKING_TRACE_DIAG]" in caplog.text
    assert "assistant_message_index=1" in caplog.text
    assert "think_start_count=2" in caplog.text
    assert "think_end_count=2" in caplog.text
    assert '"content": "question from the failing sample"' in caplog.text
    assert '"content": "<think>first</think><think>second</think>answer"' in caplog.text


def test_raw_image_dimensions_predict_processed_size_without_loading_media() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor)
    image = _FakeMedia("image-a", height=64, width=32, marker=1.0)
    sample = _sample(
        [{"type": "image", "media_index": 0}],
        [
            MediaRef(
                "image",
                image,
                freeze_media_metadata({"width": 32, "height": 64}),
            )
        ],
    )

    preencoded = encoder.preencode_sample(sample)

    assert preencoded.packing_cost == preencoded.length + 1
    assert processor.visual_calls == []


def test_wds_image_bytes_are_decoded_only_after_selection() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor)
    buffer = BytesIO()
    Image.new("RGB", (32, 64), color=(10, 20, 30)).save(buffer, format="PNG")
    sample = _sample(
        [{"type": "image", "media_index": 0}],
        [
            MediaRef(
                "image",
                buffer.getvalue(),
                _metadata(height=64, width=32, num_tiles=1),
            )
        ],
    )

    preencoded = encoder.preencode_sample(sample)
    assert processor.visual_calls == []

    encoder.postencode_sample(preencoded)
    assert len(processor.visual_calls) == 1
    modality, decoded = processor.visual_calls[0]
    assert modality == "image"
    assert isinstance(decoded, Image.Image)
    assert decoded.mode == "RGB"
    assert decoded.size == (32, 64)


def test_wds_video_bytes_are_decoded_only_after_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_rl.data.energon.multimodal.task_encoders import nemotron_visual

    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor)
    decoded = _FakeMedia(
        "video-a",
        height=64,
        width=64,
        marker=3.0,
        frames=2,
    )
    decode_calls: list[tuple[object, str]] = []
    monkeypatch.setattr(
        nemotron_visual,
        "decode_selected_av_bytes",
        lambda value, *, modality: decode_calls.append((value, modality)) or decoded,
    )
    sample = _sample(
        [{"type": "video", "media_index": 0}],
        [
            MediaRef(
                "video",
                b"raw-video-member",
                _metadata(
                    height=64,
                    width=64,
                    sampled_num_frames=2,
                    sampled_fps=1.0,
                ),
            )
        ],
    )
    preencoded = encoder.preencode_sample(sample)
    assert decode_calls == []
    assert processor.visual_calls == []

    encoder.postencode_sample(preencoded)
    assert decode_calls == [(b"raw-video-member", "video")]
    assert processor.visual_calls == [("video", decoded)]


def test_image_cost_is_predicted_before_processing_and_checked_after() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor)
    image = _FakeMedia("image-a", height=64, width=32, marker=1.0)
    sample = _sample(
        [
            {"type": "text", "text": "inspect "},
            {"type": "image", "media_index": 0},
        ],
        [
            MediaRef(
                "image",
                image,
                _metadata(height=64, width=32, num_tiles=1),
            )
        ],
    )

    preencoded = encoder.preencode_sample(sample)
    assert preencoded.packing_cost == preencoded.length + 1
    assert processor.visual_calls == []

    postencoded = encoder.postencode_sample(preencoded)
    flat = message_log_to_flat_messages(postencoded.message_log)
    assert processor.visual_calls == [("image", image)]
    assert isinstance(flat["pixel_values"], PackedTensor)
    assert torch.equal(
        flat["imgs_sizes"].as_tensor(),
        torch.tensor([[64, 32]], dtype=torch.int32),
    )
    assert postencoded.pending_sample is None
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    assert (
        sum(
            int((message["token_ids"] == image_token_id).sum())
            for message in postencoded.message_log
        )
        == 2
    )
    assert postencoded.length == postencoded.packing_cost


def test_multiple_images_keep_source_order() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor)
    first = _FakeMedia("first", height=64, width=32, marker=1.0)
    second = _FakeMedia("second", height=64, width=64, marker=2.0)
    sample = _sample(
        [
            {"type": "image", "media_index": 0},
            {"type": "text", "text": " then "},
            {"type": "image", "media_index": 1},
        ],
        [
            MediaRef(
                "image",
                first,
                _metadata(height=64, width=32, num_tiles=1),
            ),
            MediaRef(
                "image",
                second,
                _metadata(height=64, width=64, num_tiles=1),
            ),
        ],
    )

    preencoded = encoder.preencode_sample(sample)
    assert preencoded.packing_cost == preencoded.length + 4

    postencoded = encoder.postencode_sample(preencoded)
    flat = message_log_to_flat_messages(postencoded.message_log)
    pixels = flat["pixel_values"].as_tensor()
    assert processor.visual_calls == [("image", first), ("image", second)]
    assert pixels is not None
    assert float(pixels[0, :, :, : first.width].mean()) == 1.0
    assert float(pixels[1].mean()) == 2.0
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    assert (
        sum(
            int((message["token_ids"] == image_token_id).sum())
            for message in postencoded.message_log
        )
        == 6
    )


def test_video_temporal_cost_uses_tubelets_without_opening_media_early() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor, temporal_patch_size=2)
    video = _FakeMedia(
        "video-a",
        height=64,
        width=64,
        marker=3.0,
        frames=3,
    )
    sample = _sample(
        [{"type": "video", "media_index": 0}],
        [
            MediaRef(
                "video",
                video,
                _metadata(
                    height=64,
                    width=64,
                    sampled_num_frames=3,
                    sampled_fps=2.0,
                ),
            )
        ],
    )

    preencoded = encoder.preencode_sample(sample)
    compact_content = preencoded.message_log[0]["content"][0]["text"]
    assert compact_content.count(COMPACT_IMAGE_PLACEHOLDER) == 2
    assert preencoded.packing_cost == preencoded.length + 6
    assert processor.visual_calls == []

    postencoded = encoder.postencode_sample(preencoded)
    flat = message_log_to_flat_messages(postencoded.message_log)
    assert processor.visual_calls == [("video", video)]
    assert torch.equal(flat["num_frames"].as_tensor(), torch.tensor([3]))
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    assert (
        sum(
            int((message["token_ids"] == image_token_id).sum())
            for message in postencoded.message_log
        )
        == 8
    )
    assert postencoded.length == postencoded.packing_cost


def test_postencode_rejects_changed_image_expansion() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor)
    image = _FakeMedia("image-a", height=64, width=32, marker=1.0)
    sample = _sample(
        [{"type": "image", "media_index": 0}],
        [
            MediaRef(
                "image",
                image,
                _metadata(height=64, width=64, num_tiles=1),
            )
        ],
    )
    preencoded = encoder.preencode_sample(sample)

    with pytest.raises(ValueError, match="expansion changed"):
        encoder.postencode_sample(preencoded)
