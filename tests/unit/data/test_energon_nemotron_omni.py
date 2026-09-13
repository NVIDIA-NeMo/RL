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

import pytest

pytest.importorskip("megatron.energon")
pytest.importorskip("megatron.core")

pytestmark = pytest.mark.mcore

import soundfile as sf
import torch

from nemo_rl.data.energon.multimodal.cookers.nemotron import (
    cook_general_conversations_jsonl,
)
from nemo_rl.data.energon.multimodal.task_encoders.generic_sft import (
    HFMultimodalSFTProcessorAdapter,
)
from nemo_rl.data.energon.multimodal.task_encoders.nemotron_multimodal import (
    SOUND_END,
    SOUND_PLACEHOLDER,
    SOUND_START,
    NemotronMultiModalProcessorAdapter,
    NemotronMultiModalTaskEncoder,
)
from nemo_rl.data.energon.multimodal.types import (
    CanonicalSFTSample,
    MediaRef,
    freeze_media_metadata,
)
from nemo_rl.data.llm_message_utils import message_log_to_flat_messages
from nemo_rl.data.multimodal_utils import PackedTensor


@dataclass(frozen=True)
class _FakeImage:
    name: str
    height: int = 64
    width: int = 32


class _LazyAudio:
    def __init__(self, waveform):
        self.waveform = waveform
        self.reads = 0

    def get(self):
        self.reads += 1
        return self.waveform


class _FakeTokenizer:
    bos_token = None
    eos_token = None
    unk_token_id = -1
    pad_token_id = 0
    model_input_names = ["input_ids", "attention_mask"]
    _special_ids = {
        SOUND_START: 800,
        SOUND_PLACEHOLDER: 801,
        SOUND_END: 802,
        "<image>": 803,
    }
    _pattern = re.compile(
        "(" + "|".join(re.escape(token) for token in _special_ids) + ")"
    )

    def convert_tokens_to_ids(self, token):
        return self._special_ids.get(token, self.unk_token_id)

    def __call__(self, text, **kwargs):
        del kwargs
        value = text[0] if isinstance(text, list) else text
        token_ids = []
        for part in self._pattern.split(value):
            if not part:
                continue
            if part in self._special_ids:
                token_ids.append(self._special_ids[part])
            else:
                token_ids.extend(ord(character) % 251 + 1 for character in part)
        tensor = torch.tensor([token_ids], dtype=torch.long)
        return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}


class _FakeFeatureExtractor:
    sampling_rate = 16_000
    hop_length = 160
    model_input_names = ["sound_clips", "sound_length"]

    def __init__(self, events, *, extra_valid_frame=False):
        self.events = events
        self.extra_valid_frame = extra_valid_frame

    def __call__(
        self,
        waveform,
        *,
        sampling_rate,
        return_tensors,
        return_attention_mask,
    ):
        assert sampling_rate == self.sampling_rate
        assert return_tensors == "pt"
        assert return_attention_mask is True
        self.events.append(("audio", int(waveform.shape[0])))
        valid_frames = int(waveform.shape[0]) // self.hop_length
        if self.extra_valid_frame:
            valid_frames += 1
        physical_frames = valid_frames + 1
        return {
            "input_features": torch.arange(
                physical_frames * 4,
                dtype=torch.float32,
            ).reshape(1, physical_frames, 4),
            "attention_mask": torch.tensor(
                [[1] * valid_frames + [0]],
                dtype=torch.long,
            ),
        }


class _FakeVisualProcessor:
    model_input_names = ["pixel_values", "imgs_sizes", "num_frames"]


class NemotronH_Nano_Omni_Reasoning_V3Processor:
    bos_token = None
    eos_token = None
    name_or_path = "fake-nemotron-omni"
    model_input_names = [
        "input_ids",
        "attention_mask",
        "pixel_values",
        "imgs_sizes",
        "num_frames",
        "sound_clips",
        "sound_length",
    ]

    def __init__(self, *, extra_valid_frame=False):
        self.events = []
        self.tokenizer = _FakeTokenizer()
        self.image_processor = _FakeVisualProcessor()
        self.feature_extractor = _FakeFeatureExtractor(
            self.events,
            extra_valid_frame=extra_valid_frame,
        )
        self.audio_subsampling_factor = 8

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is False

        def render(content):
            if isinstance(content, str):
                return content
            return "".join(part.get("text", "") for part in content)

        return "".join(
            f"<{message['role']}>{render(message.get('content', ''))}"
            f"</{message['role']}>"
            for message in messages
        )

    def __call__(self, *, text, images=None, videos=None, **kwargs):
        text_output = self.tokenizer(text, **kwargs)
        payload = (images if images is not None else videos)[0]
        modality = "image" if images is not None else "video"
        self.events.append((modality, payload.name))
        rows = 1
        return {
            **text_output,
            "pixel_values": torch.ones(rows, 3, payload.height, payload.width),
            "imgs_sizes": torch.tensor([[payload.height, payload.width]]),
            "num_frames": torch.ones(rows, dtype=torch.long),
        }


def _audio_ref(
    waveform,
    *,
    sampling_rate: int = 16_000,
    num_samples: int | None = None,
) -> MediaRef:
    if num_samples is None:
        num_samples = len(waveform)
    return MediaRef(
        "audio",
        waveform,
        freeze_media_metadata(
            {
                "audio_num_samples": num_samples,
                "audio_sample_rate": sampling_rate,
            }
        ),
    )


def _image_ref(image: _FakeImage) -> MediaRef:
    return MediaRef(
        "image",
        image,
        freeze_media_metadata(
            {
                "processed_height": image.height,
                "processed_width": image.width,
                "num_tiles": 1,
            }
        ),
    )


def _sample(content, media):
    return CanonicalSFTSample(
        __key__="omni-0",
        __restore_key__=("omni-0",),
        messages=[
            {"role": "user", "content": content},
            {"role": "assistant", "content": "done"},
        ],
        media=media,
        tools=None,
    )


def test_loader_hf_adapter_builds_nemotron_omni_adapter() -> None:
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


def test_pinned_processor_audio_fields_build_feature_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    del processor.feature_extractor
    processor.audio_sampling_rate = 16_000
    processor.audio_hop_length = 160
    processor.audio_num_mel_bins = 128
    created = []

    def build_feature_extractor(*, feature_size, sampling_rate, hop_length):
        created.append((feature_size, sampling_rate, hop_length))
        return _FakeFeatureExtractor(processor.events)

    monkeypatch.setattr(
        "transformers.ParakeetFeatureExtractor",
        build_feature_extractor,
    )

    encoder = _encoder(processor)

    assert created == [(128, 16_000, 160)]
    assert encoder.adapter.target_sampling_rate == 16_000
    assert encoder.adapter.hop_length == 160


def _encoder(
    processor,
    *,
    clip_duration_seconds=60.0,
    max_sequence_length=16_384,
) -> NemotronMultiModalTaskEncoder:
    adapter = NemotronMultiModalProcessorAdapter(
        processor=processor,
        max_sequence_length=max_sequence_length,
        patch_dim=16,
        temporal_patch_size=2,
        audio_clip_duration_seconds=clip_duration_seconds,
    )
    return NemotronMultiModalTaskEncoder(
        adapter=adapter,
        cooker_functions=[],
        packer=None,
        include_source_ids=False,
    )


def test_oversized_text_sample_is_truncated_to_sequence_length() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor, max_sequence_length=80)
    sample = _sample("short question", [])
    sample.messages[1]["content"] = "<think>reason</think>" + "a" * 200

    preencoded = encoder.preencode_sample(sample)

    assert preencoded.length == 80
    assert preencoded.packing_cost == 80
    assert sum(
        len(message["token_ids"])
        for message in preencoded.message_log
        if message["role"] == "assistant"
    ) > 0


def test_audio_width_is_predicted_without_processing_payload() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor)
    waveform = torch.linspace(-1.0, 1.0, 3_200)
    lazy_audio = _LazyAudio(waveform)
    sample = _sample(
        [{"type": "audio", "media_index": 0}],
        [_audio_ref(lazy_audio, num_samples=len(waveform))],
    )

    preencoded = encoder.preencode_sample(sample)
    sound_token_id = processor.tokenizer.convert_tokens_to_ids(SOUND_PLACEHOLDER)
    assert processor.events == []
    assert (
        sum(
            int((message["token_ids"] == sound_token_id).sum())
            for message in preencoded.message_log
        )
        == 3
    )
    assert preencoded.packing_cost == preencoded.length
    assert preencoded.pending_sample is sample
    assert lazy_audio.reads == 0

    postencoded = encoder.postencode_sample(preencoded)
    flat = message_log_to_flat_messages(postencoded.message_log)
    assert processor.events == [("audio", 3_200)]
    assert isinstance(flat["sound_clips"], PackedTensor)
    assert flat["sound_clips"].as_tensor().shape == (1, 21, 4)
    assert torch.equal(flat["sound_length"].as_tensor(), torch.tensor([20]))
    assert postencoded.pending_sample is None
    assert lazy_audio.reads == 1


def test_audio_clips_preserve_each_valid_length_and_total_embedding_count() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor, clip_duration_seconds=0.125)
    waveform = torch.zeros(4_000)
    sample = _sample(
        [{"type": "audio", "media_index": 0}],
        [_audio_ref(waveform)],
    )

    preencoded = encoder.preencode_sample(sample)
    sound_token_id = processor.tokenizer.convert_tokens_to_ids(SOUND_PLACEHOLDER)
    assert (
        sum(
            int((message["token_ids"] == sound_token_id).sum())
            for message in preencoded.message_log
        )
        == 4
    )

    postencoded = encoder.postencode_sample(preencoded)
    flat = message_log_to_flat_messages(postencoded.message_log)
    assert processor.events == [("audio", 2_000), ("audio", 2_000)]
    assert flat["sound_clips"].as_tensor().shape == (2, 13, 4)
    assert torch.equal(
        flat["sound_length"].as_tensor(),
        torch.tensor([12, 12]),
    )


def test_short_final_clip_is_padded_to_minimum_duration() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor, clip_duration_seconds=0.125)
    waveform = torch.zeros(2_100)
    sample = _sample(
        [{"type": "audio", "media_index": 0}],
        [_audio_ref(waveform)],
    )

    preencoded = encoder.preencode_sample(sample)
    sound_token_id = processor.tokenizer.convert_tokens_to_ids(SOUND_PLACEHOLDER)
    assert (
        sum(
            int((message["token_ids"] == sound_token_id).sum())
            for message in preencoded.message_log
        )
        == 4
    )

    postencoded = encoder.postencode_sample(preencoded)
    flat = message_log_to_flat_messages(postencoded.message_log)
    assert processor.events == [("audio", 2_000), ("audio", 1_600)]
    assert torch.equal(
        flat["sound_length"].as_tensor(),
        torch.tensor([12, 10]),
    )


def test_mixed_image_audio_order_and_fields_are_preserved() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor)
    first = _FakeImage("first")
    second = _FakeImage("second")
    waveform = torch.zeros(3_200)
    sample = _sample(
        [
            {"type": "image", "media_index": 0},
            {"type": "audio", "media_index": 1},
            {"type": "image", "media_index": 2},
        ],
        [_image_ref(first), _audio_ref(waveform), _image_ref(second)],
    )

    preencoded = encoder.preencode_sample(sample)
    assert processor.events == []
    assert preencoded.packing_cost == preencoded.length + 2

    postencoded = encoder.postencode_sample(preencoded)
    flat = message_log_to_flat_messages(postencoded.message_log)
    assert processor.events == [
        ("image", "first"),
        ("audio", 3_200),
        ("image", "second"),
    ]
    assert flat["pixel_values"].as_tensor().shape == (2, 3, 64, 32)
    assert flat["sound_clips"].as_tensor().shape == (1, 21, 4)
    assert postencoded.pending_sample is None
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    assert (
        sum(
            int((message["token_ids"] == image_token_id).sum())
            for message in postencoded.message_log
        )
        == 4
    )
    assert postencoded.length == postencoded.packing_cost


def test_assistant_thinking_trace_is_normalized_for_omni() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor)
    sample = _sample("question", [])
    sample.messages[1]["content"] = "<think>  reason  </think>  answer"

    preencoded = encoder.preencode_sample(sample)

    assert (
        "<think>\nreason\n</think>\n\nanswer"
        in preencoded.message_log[1]["content"][0]["text"]
    )


def test_jsonl_audio_path_is_loaded_only_after_selection(tmp_path) -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor)
    audio_path = tmp_path / "selected.wav"
    sf.write(audio_path, torch.zeros(3_200).numpy(), 16_000)
    cooked = cook_general_conversations_jsonl(
        {
            "__key__": "audio-path-0",
            "__restore_key__": ("audio-path-0",),
            "__subflavors__": {},
            "json": {
                "audio": {
                    "path": str(audio_path),
                    "metadata": {
                        "audio_num_samples": 3_200,
                        "audio_sample_rate": 16_000,
                    },
                },
                "conversations": [
                    {"from": "user", "value": "<audio>"},
                    {"from": "assistant", "value": "transcript"},
                ],
            },
        }
    )

    preencoded = encoder.preencode_sample(cooked)
    assert processor.events == []

    postencoded = encoder.postencode_sample(preencoded)
    flat = message_log_to_flat_messages(postencoded.message_log)
    assert processor.events == [("audio", 3_200)]
    assert torch.equal(flat["sound_length"].as_tensor(), torch.tensor([20]))


def test_postencode_rejects_audio_frame_prediction_change() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor(extra_valid_frame=True)
    encoder = _encoder(processor)
    sample = _sample(
        [{"type": "audio", "media_index": 0}],
        [_audio_ref(torch.zeros(3_200))],
    )
    preencoded = encoder.preencode_sample(sample)

    with pytest.raises(ValueError, match="frame count changed"):
        encoder.postencode_sample(preencoded)
    assert preencoded.pending_sample is sample


def test_preencode_requires_exact_audio_sample_metadata() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor)
    sample = _sample(
        [{"type": "audio", "media_index": 0}],
        [
            MediaRef(
                "audio",
                torch.zeros(3_200),
                freeze_media_metadata({"audio_sample_rate": 16_000}),
            )
        ],
    )

    with pytest.raises(ValueError, match="audio_num_samples"):
        encoder.preencode_sample(sample)
    assert processor.events == []


def test_duration_metadata_predicts_exact_source_sample_count() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    encoder = _encoder(processor)
    waveform = torch.zeros(3_200)
    sample = _sample(
        [{"type": "audio", "media_index": 0}],
        [
            MediaRef(
                "audio",
                waveform,
                freeze_media_metadata(
                    {
                        "audio_duration": 0.2,
                        "audio_sample_rate": 16_000,
                    }
                ),
            )
        ],
    )

    preencoded = encoder.preencode_sample(sample)
    postencoded = encoder.postencode_sample(preencoded)
    flat = message_log_to_flat_messages(postencoded.message_log)
    assert torch.equal(flat["sound_length"].as_tensor(), torch.tensor([20]))


def test_adapter_requires_processor_audio_sizing_settings() -> None:
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    processor.feature_extractor.hop_length = None

    with pytest.raises(ValueError, match="sampling_rate and hop_length"):
        _encoder(processor)
