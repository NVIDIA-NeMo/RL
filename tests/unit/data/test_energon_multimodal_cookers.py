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

import sqlite3
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

pytest.importorskip("megatron.energon")
pytest.importorskip("megatron.core")

pytestmark = pytest.mark.mcore

from megatron.energon import Cooker

from nemo_rl.data.energon.multimodal.cookers.generic import cook_conversation
from nemo_rl.data.energon.multimodal.cookers.nemotron import (
    GRANARY_ENGLISH_PROMPT,
    _load_media_metadata,
    _media_metadata,
    cook_general_conversations_jsonl,
    cook_general_conversations_webdataset,
    cook_granary_english_jsonl,
    cook_granary_english_webdataset,
    cook_nemotron_conversation,
)
from nemo_rl.data.energon.multimodal.model_families import (
    get_supported_model_families,
)


def _sample(payload, **members):
    return {
        "__key__": "sample-0",
        "__restore_key__": ("sample-0",),
        "__subflavors__": {},
        "__sources__": (),
        "json": payload,
        **members,
    }


class _FakeMediaStore:
    def get_path(self):
        return "/data/media"

    def get_media_metadata(self, path):
        return SimpleNamespace(width=80, height=40)


class _FakeCache:
    def get_lazy(self, store, path):
        return ("lazy", store.get_path(), path)


def test_nemotron_conversation_cooker_loads_fragment_media_from_aux_store():
    sample = _sample(
        {
            "conversation": [
                {
                    "sender": "user",
                    "fragments": [
                        {"t": "image", "value": "data/example.png"},
                        {"t": "text", "value": "Describe it."},
                    ],
                },
                {
                    "sender": "assistant",
                    "fragments": [{"t": "text", "value": "A diagram."}],
                },
            ]
        }
    )

    cooked = cook_nemotron_conversation(
        sample,
        cache=_FakeCache(),
        media_source=_FakeMediaStore(),
    )

    assert cooked.media[0].value == ("lazy", "/data/media", "data/example.png")
    assert cooked.media[0].metadata == (("height", 40), ("width", 80))
    assert cooked.messages[0]["content"] == [
        {"type": "image", "media_index": 0},
        {"type": "text", "text": "Describe it."},
    ]
    assert cooked.__sources__[-1].index == "data/example.png"
    assert Cooker(cook_nemotron_conversation).need_cache


def test_generic_cooker_freezes_explicit_media_metadata_without_opening_media():
    media_value = object()
    cooked = cook_conversation(
        _sample(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "media_index": 0}],
                    }
                ],
                "media": [
                    {
                        "type": "image",
                        "value": media_value,
                        "metadata": {"width": 32, "height": 16},
                    }
                ],
            }
        )
    )

    assert cooked.media[0].value is media_value
    assert cooked.media[0].metadata == (("height", 16), ("width", 32))
    with pytest.raises(FrozenInstanceError):
        cooked.media[0].metadata = ()


def test_general_wds_cooker_preserves_aliases_order_and_metadata():
    image, video, speech = object(), object(), object()
    cooked = cook_general_conversations_webdataset(
        _sample(
            {
                "images": {
                    "member": "sample-0.picture.png",
                    "metadata": {"width": 64, "height": 32},
                },
                "video-sound": {
                    "member": "sample-0.clip.mp4",
                    "metadata": {"video_num_frames": 8, "video_fps": 2.0},
                },
                "speech": {
                    "member": "sample-0.voice.flac",
                    "metadata": {"audio_duration": 1.5, "audio_sample_rate": 16000},
                },
                "conversations": [
                    {
                        "from": "human",
                        "value": "See <images>, watch <video-sound>, hear <speech>.",
                    },
                    {"from": "gpt", "value": "Done."},
                ],
            },
            png=image,
            mp4=video,
            flac=speech,
        )
    )

    assert [message["role"] for message in cooked.messages] == ["user", "assistant"]
    assert [ref.modality for ref in cooked.media] == [
        "image",
        "video",
        "audio",
        "audio",
    ]
    assert [ref.value for ref in cooked.media] == [image, video, video, speech]
    assert cooked.media[0].metadata == (("height", 32), ("width", 64))
    media_indexes = [
        part["media_index"]
        for part in cooked.messages[0]["content"]
        if "media_index" in part
    ]
    assert media_indexes == [
        0,
        1,
        2,
        3,
    ]


def test_general_jsonl_cooker_keeps_lazy_paths_and_rejects_source_errors():
    cooked = cook_general_conversations_jsonl(
        _sample(
            {
                "audio": {
                    "path": "/media/voice.wav",
                    "metadata": {"audio_duration": 2.0},
                },
                "conversations": [
                    {"from": "user", "value": "<audio>"},
                    {"from": "assistant", "value": "Transcript"},
                ],
            }
        )
    )
    assert cooked.media[0].value == "/media/voice.wav"
    assert cooked.media[0].metadata == (("audio_duration", 2.0),)
    assert cooked.messages[0]["content"] == [
        {"type": "audio", "media_index": 0},
        {"type": "text", "text": " "},
    ]

    with pytest.raises(ValueError, match="Retrieved 0/1 image"):
        cook_general_conversations_jsonl(
            _sample(
                {
                    "image": "unused.jpg",
                    "conversations": [{"from": "user", "value": "text only"}],
                }
            )
        )
    with pytest.raises(ValueError, match="has no 'image' media field"):
        cook_general_conversations_jsonl(
            _sample({"conversations": [{"from": "user", "value": "<image>"}]})
        )


def test_general_cookers_receive_aux_media_and_apply_source_loss_mask():
    sample = _sample(
        {
            "image": "example.png",
            "conversations": [
                {"from": "human", "value": "<image> first"},
                {"from": "gpt", "value": "draft"},
                {"from": "human", "value": "revise"},
                {"from": "gpt", "value": "final"},
            ],
        }
    )
    sample["__subflavors__"] = {"train_only_on_last_assistant_turn": True}

    cooked = cook_general_conversations_jsonl(
        sample,
        cache=_FakeCache(),
        primary=_FakeMediaStore(),
        media_source=_FakeMediaStore(),
    )

    assert cooked.media[0].value == ("lazy", "/data/media", "example.png")
    assert [message["train_on_message"] for message in cooked.messages] == [
        False,
        False,
        False,
        True,
    ]
    assert Cooker(cook_general_conversations_jsonl).need_cache
    assert Cooker(cook_general_conversations_jsonl).need_primary


@pytest.mark.parametrize(
    ("cooker", "payload", "members", "expected_value"),
    [
        (
            cook_granary_english_webdataset,
            {
                "audio_filepath": {
                    "member": "sample-0.audio.flac",
                    "metadata": {
                        "audio_num_samples": 32000,
                        "audio_sample_rate": 16000,
                    },
                },
                "text": "Hello world.",
            },
            {"flac": "decoded-later"},
            "decoded-later",
        ),
        (
            cook_granary_english_jsonl,
            {
                "audio_filepath": {
                    "path": "/media/audio.flac",
                    "metadata": {
                        "audio_num_samples": 32000,
                        "audio_sample_rate": 16000,
                    },
                },
                "text": "Hello world.",
            },
            {},
            "/media/audio.flac",
        ),
    ],
)
def test_granary_cookers_build_the_fixed_asr_conversation(
    cooker, payload, members, expected_value
):
    cooked = cooker(_sample(payload, **members))

    assert cooked.media[0].modality == "audio"
    assert cooked.media[0].value == expected_value
    assert cooked.media[0].metadata == (
        ("audio_num_samples", 32000),
        ("audio_sample_rate", 16000),
    )
    assert cooked.messages == [
        {
            "role": "user",
            "content": [
                {"type": "audio", "media_index": 0},
                {
                    "type": "text",
                    "text": GRANARY_ENGLISH_PROMPT.removeprefix("<audio>"),
                },
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Hello world."}]},
    ]


def test_new_cookers_declare_nemotron_support():
    for cooker in (
        cook_general_conversations_jsonl,
        cook_general_conversations_webdataset,
        cook_granary_english_jsonl,
        cook_granary_english_webdataset,
        cook_nemotron_conversation,
    ):
        assert get_supported_model_families(cooker) == frozenset({"nemotron"})


@pytest.mark.parametrize(
    "error",
    [
        sqlite3.OperationalError("unable to open database file"),
        sqlite3.DatabaseError("file is not a database"),
    ],
)
def test_media_metadata_falls_back_when_sqlite_index_is_unreadable(error):
    """A sqlite failure means "no prepared metadata", not "drop the sample".

    Energon backs ``get_media_metadata`` with a sqlite index, and ``sqlite3.Error``
    descends from ``Exception`` rather than ``OSError``. Read-only DSS caches raise
    ``OperationalError`` here, which used to escape ``_media_metadata`` and make
    Energon discard the whole sample instead of decoding the media to derive the
    dimensions.
    """

    class _UnreadableIndexStore(_FakeMediaStore):
        def get_media_metadata(self, path):
            raise error

    assert _media_metadata(_UnreadableIndexStore(), "data/example.png") == ()


def test_media_metadata_warns_once_per_unprepared_store(capsys):
    class _UnpreparedStore(_FakeMediaStore):
        def get_path(self):
            return "/data/unprepared-media"

        def get_media_metadata(self, path):
            raise RuntimeError("metadata index is missing")

    store = _UnpreparedStore()
    assert _media_metadata(store, "data/first.png") == ()
    assert _media_metadata(store, "data/second.png") == ()

    output = capsys.readouterr().out
    assert output.count("WARNING: Dataset /data/unprepared-media") == 1
    assert "slow metadata for data/first.png" in output
    assert "data/second.png" not in output


def test_missing_media_metadata_is_probed_from_seekable_file(
    monkeypatch, tmp_path
):
    media_path = tmp_path / "sample.mp4"
    media_path.write_bytes(b"video")

    class _UnpreparedStore(_FakeMediaStore):
        def get_path(self):
            return str(tmp_path)

        def get_media_metadata(self, path):
            raise RuntimeError("metadata index is missing")

    class _ProbeDecoder:
        def __init__(self, stream):
            assert stream.seekable()

        def get_metadata(self):
            return SimpleNamespace(video_duration=10.0, video_num_frames=20)

    monkeypatch.setattr(
        "nemo_rl.data.energon.multimodal.cookers.nemotron.AVDecoder",
        _ProbeDecoder,
    )

    assert _load_media_metadata(
        _UnpreparedStore(),
        media_path.name,
        modality="video",
        derive_missing_metadata=True,
    ) == (("video_duration", 10.0), ("video_num_frames", 20))
