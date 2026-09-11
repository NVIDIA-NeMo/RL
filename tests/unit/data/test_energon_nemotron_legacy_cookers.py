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

import pytest

pytest.importorskip("megatron.energon")
pytest.importorskip("megatron.core")

pytestmark = pytest.mark.mcore

from nemo_rl.data.energon.multimodal.cookers import nemotron
from nemo_rl.data.energon.multimodal.cookers.nemotron import (
    EMPTY_SYSTEM_CONTENT,
    LEGACY_SYSTEM_CONTENT,
    cook_audio_conversation_jsonl,
    cook_nano_openai_messages_jsonl,
    cook_nano_openai_messages_offline_packed_jsonl,
    cook_omcat_legacy_conversation_monolithic,
)
from nemo_rl.data.energon.multimodal.model_families import (
    get_supported_model_families,
)


def _sample(payload, **members):
    return {
        "__key__": "sample-0",
        "__restore_key__": ("sample-0",),
        "__subflavors__": {},
        "json": payload,
        **members,
    }


def _message_text(message):
    return message["content"][0]["text"]


def test_nano_cooker_normalizes_chatml_thinking_and_roles():
    open_think = "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    cooked = cook_nano_openai_messages_jsonl(
        _sample(
            {
                "messages": [
                    {"role": "human", "content": "Question" + open_think},
                    {"role": "gpt", "content": "\n</think>  Answer   "},
                    {
                        "role": "function",
                        "content": [{"type": "text", "text": "", "content": "result"}],
                    },
                ]
            }
        )
    )

    assert [message["role"] for message in cooked.messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert _message_text(cooked.messages[0]) == EMPTY_SYSTEM_CONTENT
    assert _message_text(cooked.messages[1]) == (
        "Question<|im_end|>\n<|im_start|>assistant\n<think></think>"
    )
    assert _message_text(cooked.messages[2]) == "Answer\n"
    assert _message_text(cooked.messages[3]) == "result"


def test_nano_cooker_rewrites_legacy_system_and_rejects_repeated_system():
    cooked = cook_nano_openai_messages_jsonl(
        _sample(
            {
                "messages": [
                    {"role": "system", "content": LEGACY_SYSTEM_CONTENT},
                    {"role": "assistant", "content": "done"},
                ]
            }
        )
    )
    assert _message_text(cooked.messages[0]) == EMPTY_SYSTEM_CONTENT
    assert _message_text(cooked.messages[1]) == "done\n"

    with pytest.raises(ValueError, match="only one leading system"):
        cook_nano_openai_messages_jsonl(
            _sample(
                {
                    "messages": [
                        {"role": "system", "content": "one"},
                        {"role": "user", "content": "question"},
                        {"role": "system", "content": "two"},
                    ]
                }
            )
        )


def test_nano_offline_packed_cooker_preserves_conversation_boundaries():
    cooked = cook_nano_openai_messages_offline_packed_jsonl(
        _sample(
            {
                "messages": [
                    {"role": "system", "content": "system one"},
                    {"role": "assistant", "content": "answer one"},
                    {"role": "system", "content": "system two"},
                    {"role": "assistant", "content": "answer two"},
                ]
            }
        )
    )

    assert [message["role"] for message in cooked.messages] == [
        "system",
        "assistant",
        "system",
        "assistant",
    ]
    assert cooked.__subflavors__["offline_packed_messages"] is True


def test_audio_conversation_cooker_preserves_multi_audio_order_and_metadata():
    audio_one, audio_two = object(), object()
    cooked = cook_audio_conversation_jsonl(
        _sample(
            {
                "id": "audio-0",
                "sound_0": {
                    "value": audio_one,
                    "metadata": {"audio_duration": 1.25},
                },
                "sound_1": {
                    "value": audio_two,
                    "metadata": {"audio_duration": 2.5},
                },
                "conversations": [
                    {"from": "human", "value": "Listen: <sound>"},
                    {"from": "gpt", "value": "Transcript"},
                ],
            }
        )
    )

    assert [media.modality for media in cooked.media] == ["audio", "audio"]
    assert [media.value for media in cooked.media] == [audio_one, audio_two]
    assert [media.metadata for media in cooked.media] == [
        (("audio_duration", 1.25),),
        (("audio_duration", 2.5),),
    ]
    assert cooked.messages[0]["content"] == [
        {"type": "text", "text": "Listen: "},
        {"type": "audio", "media_index": 0},
        {"type": "audio", "media_index": 1},
    ]


def test_audio_conversation_cooker_expands_video_sound_in_visual_audio_order():
    video, audio = object(), object()
    cooked = cook_audio_conversation_jsonl(
        _sample(
            {
                "vis_video_0": {
                    "value": video,
                    "metadata": {"width": 32, "height": 16},
                },
                "vis_sound_0": {
                    "value": audio,
                    "metadata": {"audio_duration": 1.0},
                },
                "conversations": [
                    {"from": "human", "value": "<video-sound>"},
                    {"from": "gpt", "value": "done"},
                ],
            }
        )
    )

    assert [media.modality for media in cooked.media] == ["video", "audio"]
    assert [media.value for media in cooked.media] == [video, audio]


def test_omcat_legacy_cooker_maps_alias_tag_to_monolithic_member(monkeypatch):
    audio = object()
    monkeypatch.setattr(
        nemotron,
        "_primary_media",
        lambda sample, member, **kwargs: (sample[member], ()),
    )
    payload = {
        "speech": "legacy-field-value-is-not-the-member",
        "conversations": [
            {"from": "human", "value": "Transcribe <audio>"},
            {"from": "gpt", "value": "hello"},
        ],
    }
    cooked = cook_omcat_legacy_conversation_monolithic(_sample(payload, FLAC=audio))

    assert payload["speech"] == "legacy-field-value-is-not-the-member"
    assert cooked.media[0].modality == "audio"
    assert cooked.media[0].value is audio
    assert cooked.messages[0]["content"] == [
        {"type": "text", "text": "Transcribe "},
        {"type": "audio", "media_index": 0},
    ]


def test_omcat_legacy_cooker_rejects_undefined_video_sound_mapping():
    with pytest.raises(ValueError, match="no defined member mapping"):
        cook_omcat_legacy_conversation_monolithic(
            _sample(
                {
                    "video-sound": "legacy",
                    "conversations": [
                        {"from": "human", "value": "<video-sound>"},
                        {"from": "gpt", "value": "done"},
                    ],
                },
                mp4=object(),
            )
        )


def test_legacy_cookers_declare_nemotron_support():
    for cooker in (
        cook_audio_conversation_jsonl,
        cook_nano_openai_messages_jsonl,
        cook_nano_openai_messages_offline_packed_jsonl,
        cook_omcat_legacy_conversation_monolithic,
    ):
        assert get_supported_model_families(cooker) == frozenset({"nemotron"})
