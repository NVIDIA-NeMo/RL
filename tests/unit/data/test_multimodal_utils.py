# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

from nemo_rl.data import multimodal_utils


def test_load_decord_reports_decord2_install_hint(monkeypatch):
    def import_without_decord(name):
        assert name == "decord"
        raise ImportError("No module named 'decord'")

    monkeypatch.setattr(
        multimodal_utils.importlib, "import_module", import_without_decord
    )

    with pytest.raises(ImportError, match="decord2 is required"):
        multimodal_utils._load_decord()


def test_audio_fallback_reports_missing_decord(monkeypatch):
    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("audio load failed")

    def raise_missing_decord():
        raise ImportError("decord2 is required")

    monkeypatch.setattr(multimodal_utils, "load_audio", raise_runtime_error)
    monkeypatch.setattr(multimodal_utils, "_load_decord", raise_missing_decord)

    with pytest.raises(ImportError, match="decord2 is required"):
        multimodal_utils.load_media_from_message(
            {"content": [{"type": "audio", "audio": "audio.wav"}]},
            multimodal_load_kwargs={"audio": {"sampling_rate": 16000}},
        )


def test_audio_fallback_uses_decord(monkeypatch):
    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("audio load failed")

    class FakeAudio:
        def asnumpy(self):
            return ["channel0", "channel1"]

    class FakeDecord:
        class AudioReader:
            def __init__(self, path, *, sample_rate, mono):
                assert path == "audio.wav"
                assert sample_rate == 16000
                assert mono is True

            def __getitem__(self, index):
                assert index == slice(None)
                return FakeAudio()

    monkeypatch.setattr(multimodal_utils, "load_audio", raise_runtime_error)
    monkeypatch.setattr(multimodal_utils, "_load_decord", lambda: FakeDecord)

    loaded_media = multimodal_utils.load_media_from_message(
        {"content": [{"type": "audio", "audio": "audio.wav"}]},
        processor=object(),
        multimodal_load_kwargs={"audio": {"sampling_rate": 16000}},
    )

    assert loaded_media["audio"] == ["channel0"]
