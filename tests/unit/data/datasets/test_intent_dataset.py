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

"""Tests for the IntentTrain / IntentBench dataset loader.

These tests validate the v1 audio+video contract: every yielded sample
carries one ``{type:video}`` content item AND one ``{type:audio}`` content
item AND a text prompt. The independent-streams shape is what lets the
chat template emit both ``<|VIDEO|>`` and ``<|AUDIO|>`` placeholders so
vLLM rollouts can populate ``multi_modal_data["video"]`` and
``multi_modal_data["audio"]`` (see Round 1 BitLesson
``BL-20260428-omni-use-audio-in-video``).

The tests use a fabricated manifest + zip + .mp4 so they do not pull the
~16 GB IntentTrain / IntentBench archives from HuggingFace.
"""

import json
import os
import wave
import zipfile
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest


def _write_silent_mp4(path: str, duration_seconds: float = 1.0) -> None:
    """Encode a silent stereo WAV-in-MP4 container for tests.

    decord.AudioReader can decode common MP4 audio containers; encoding a
    real mp4 from scratch in a unit test is awkward, so we use ffmpeg via
    a subprocess if available, else skip the test.
    """
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not available; cannot fabricate intent video")

    sample_rate = 16000
    n_samples = int(duration_seconds * sample_rate)
    wav_path = path + ".wav"
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((np.zeros(n_samples, dtype=np.int16)).tobytes())

    # Encode WAV + black video frames into an mp4 with both streams.
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=size=64x64:rate=4:duration={duration_seconds}",
        "-i",
        wav_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(wav_path)


def _build_fake_intent_snapshot(
    snapshot_dir: str,
    manifest_filename: str,
    relpath: str = "social_iq/sample_001.mp4",
) -> dict[str, Any]:
    """Populate ``snapshot_dir`` with one .mp4 + manifest + videos.zip sentinel."""
    os.makedirs(
        os.path.join(snapshot_dir, "videos", os.path.dirname(relpath)), exist_ok=True
    )
    video_path = os.path.join(snapshot_dir, "videos", relpath)
    _write_silent_mp4(video_path, duration_seconds=1.0)

    manifest = [
        {
            "problem": "Are the participants confident?",
            "problem_type": "multiple choice",
            "options": ["A. Yes", "B. No"],
            "answer": "A",
            "data_type": "video",
            "path": relpath,
        },
        # negative-filter sample: should be dropped by allowed_problem_types
        {
            "problem": "How do you feel?",
            "problem_type": "free-form",
            "options": [],
            "answer": "Happy",
            "data_type": "video",
            "path": relpath,
        },
    ]
    with open(os.path.join(snapshot_dir, manifest_filename), "w") as f:
        json.dump(manifest, f)

    # IntentDataset uses a videos.zip sentinel as proxy for "extracted";
    # write an empty marker so the extraction step is a no-op when the
    # videos/ tree already exists from this fixture.
    with zipfile.ZipFile(os.path.join(snapshot_dir, "videos.zip"), "w") as zf:
        zf.writestr("placeholder", b"")
    sentinel_path = os.path.join(snapshot_dir, ".intent_videos_extracted")
    with open(sentinel_path, "w") as f:
        f.write("ok\n")

    return {
        "video_path": video_path,
        "manifest_path": os.path.join(snapshot_dir, manifest_filename),
    }


class TestIntentDatasetIndependentStreams:
    """Sample-shape contract: one video item + one audio item + text."""

    def test_intent_train_sample_carries_video_and_audio_items(self, tmp_path):
        from nemo_rl.data.datasets.response_datasets.intent import IntentTrainDataset

        snapshot_dir = tmp_path / "intent_train_snapshot"
        snapshot_dir.mkdir()
        _build_fake_intent_snapshot(
            str(snapshot_dir), manifest_filename="emer_rewrite.json"
        )

        # IntentTrain class normally requires both emer_rewrite.json AND
        # social_iq_v2_rewrite.json; provide the second as an empty list.
        with open(snapshot_dir / "social_iq_v2_rewrite.json", "w") as f:
            json.dump([], f)

        with (
            patch(
                "nemo_rl.data.datasets.response_datasets.intent.snapshot_download",
                return_value=str(snapshot_dir),
            ),
            patch(
                "nemo_rl.data.datasets.response_datasets.intent.get_huggingface_cache_path",
                return_value=None,
            ),
        ):
            ds = IntentTrainDataset(allowed_problem_types=["multiple choice"])

        assert ds.task_name == "intent-train"
        assert len(ds.dataset) == 1, (
            "free-form sample should be filtered out by allow-list"
        )

        formatted = ds.format_data(ds.dataset[0])
        user_content = formatted["messages"][0]["content"]
        type_counts: dict[str, int] = {}
        for item in user_content:
            type_counts[item["type"]] = type_counts.get(item["type"], 0) + 1

        assert type_counts.get("video", 0) == 1, (
            f"expected exactly one video item, got types={type_counts}"
        )
        assert type_counts.get("audio", 0) == 1, (
            f"expected exactly one audio item, got types={type_counts}"
        )
        assert type_counts.get("text", 0) == 1, (
            f"expected exactly one text item, got types={type_counts}"
        )

        audio_item = next(c for c in user_content if c["type"] == "audio")
        assert isinstance(audio_item["audio"], np.ndarray)
        assert audio_item["audio"].ndim == 1
        assert audio_item["audio"].dtype == np.float32

        video_item = next(c for c in user_content if c["type"] == "video")
        assert os.path.isfile(video_item["video"])

    def test_intent_invalid_split_raises(self):
        from nemo_rl.data.datasets.response_datasets.intent import IntentDataset

        with pytest.raises(ValueError, match="Invalid split"):
            IntentDataset(split="test")
