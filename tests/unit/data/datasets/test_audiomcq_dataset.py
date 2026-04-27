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

import json
import os
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from nemo_rl.data.datasets.response_datasets import audiomcq as audiomcq_module
from nemo_rl.data.datasets.response_datasets.audiomcq import (
    AUDIOMCQ_MANIFEST,
    AudioMCQDataset,
)


def _write_wav(path: str, sample_rate: int = 16000, duration_s: float = 0.05) -> None:
    n = int(sample_rate * duration_s)
    samples = np.zeros(n, dtype=np.float32)
    sf.write(path, samples, sample_rate)


def _build_fixture_snapshot(
    tmp_path,
    rows: list[dict[str, Any]],
    write_audio_for: list[int] | None = None,
) -> str:
    """Write a data.jsonl manifest under tmp_path plus optional .wav fixtures."""
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    manifest_path = snapshot_root / AUDIOMCQ_MANIFEST
    with open(manifest_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    if write_audio_for is None:
        write_audio_for = list(range(len(rows)))
    for idx in write_audio_for:
        rel = rows[idx]["audio_path"]
        absolute = snapshot_root / rel
        absolute.parent.mkdir(parents=True, exist_ok=True)
        _write_wav(str(absolute))
    return str(snapshot_root)


def _patch_snapshot(monkeypatch, snapshot_root: str) -> None:
    monkeypatch.setattr(
        audiomcq_module, "_resolve_snapshot_root", lambda: snapshot_root
    )


def _row(
    idx: int,
    *,
    source: str = "AudioCaps",
    ext: str = "wav",
    answer: str = "Water splashing continuously",
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = {
        "source_dataset": source,
        "id": str(idx),
        "question_type": "sound",
        "audio_path": f"{source}/sample_{idx}.{ext}",
        "question": f"Question {idx}?",
        "answer": answer,
        "choices": [
            "Water is pouring steadily",
            answer,
            "Water is boiling rapidly",
            "Water is dripping slowly",
        ],
    }
    if extras:
        base.update(extras)
    return base


class TestAudioMCQRegistry:
    def test_registered_under_audiomcq_key(self):
        from nemo_rl.data.datasets.response_datasets import DATASET_REGISTRY

        assert DATASET_REGISTRY["audiomcq"] is AudioMCQDataset

    def test_task_name_is_audiomcq(self):
        assert AudioMCQDataset.task_name == "audiomcq"

    def test_invalid_split_raises(self, monkeypatch, tmp_path):
        rows = [_row(0)]
        snapshot_root = _build_fixture_snapshot(tmp_path, rows)
        _patch_snapshot(monkeypatch, snapshot_root)
        with pytest.raises(ValueError, match="Invalid split"):
            AudioMCQDataset(split="test")

    def test_load_response_dataset_round_trip(self, monkeypatch, tmp_path):
        rows = [_row(i) for i in range(3)]
        snapshot_root = _build_fixture_snapshot(tmp_path, rows)
        _patch_snapshot(monkeypatch, snapshot_root)

        from nemo_rl.data.datasets import load_response_dataset

        wrapper = load_response_dataset(
            {
                "dataset_name": "audiomcq",
                "split": "train",
                "max_samples": 2,
                "processor": "vlm_hf_data_processor",
            }
        )
        assert wrapper.task_name == "audiomcq"
        assert wrapper.preprocessor is not None
        assert len(wrapper.dataset) == 2
        sample = wrapper.preprocessor(wrapper.dataset[0])
        assert sample["task_name"] == "audiomcq"
        assert "messages" in sample

    def test_unsupported_dataset_name_raises(self):
        from nemo_rl.data.datasets import load_response_dataset

        with pytest.raises(ValueError, match="Unsupported"):
            load_response_dataset({"dataset_name": "audiomcq_strong", "split": "train"})


class TestAudioMCQConstruction:
    def test_minimal_load_succeeds(self, monkeypatch, tmp_path):
        rows = [_row(i) for i in range(3)]
        snapshot_root = _build_fixture_snapshot(tmp_path, rows)
        _patch_snapshot(monkeypatch, snapshot_root)

        dataset = AudioMCQDataset(split="train", max_samples=2)

        assert dataset.task_name == "audiomcq"
        assert len(dataset.dataset) == 2
        assert dataset.preprocessor is not None
        assert all(r["task_name"] == "audiomcq" for r in dataset.dataset)

    def test_max_samples_is_seed_deterministic(self, monkeypatch, tmp_path):
        rows = [_row(i) for i in range(20)]
        snapshot_root = _build_fixture_snapshot(tmp_path, rows)
        _patch_snapshot(monkeypatch, snapshot_root)

        ds_a = AudioMCQDataset(split="train", max_samples=5, seed=123)
        ds_b = AudioMCQDataset(split="train", max_samples=5, seed=123)
        ds_c = AudioMCQDataset(split="train", max_samples=5, seed=456)

        ids_a = [r["id"] for r in ds_a.dataset]
        ids_b = [r["id"] for r in ds_b.dataset]
        ids_c = [r["id"] for r in ds_c.dataset]
        assert ids_a == ids_b
        assert ids_a != ids_c

    def test_defensive_strong_filter_drops_weak_rows(self, monkeypatch, tmp_path):
        rows = [
            _row(0, extras={"audio-contribution": "strong"}),
            _row(1, extras={"audio-contribution": "weak"}),
            _row(2, extras={"audio-contribution": "strong"}),
        ]
        snapshot_root = _build_fixture_snapshot(tmp_path, rows)
        _patch_snapshot(monkeypatch, snapshot_root)

        dataset = AudioMCQDataset(split="train")

        kept_ids = sorted(r["id"] for r in dataset.dataset)
        assert kept_ids == ["0", "2"]

    def test_max_samples_applied_after_defensive_filter(self, monkeypatch, tmp_path):
        # 5 strong rows, 5 weak rows. With max_samples=4 and the filter
        # applied AFTER, we'd see at most 4 strong rows (never weak). With
        # max_samples applied BEFORE, the truncated set could contain weak
        # rows that the filter would then drop, leaving fewer than 4.
        rows = []
        for i in range(5):
            rows.append(_row(i, extras={"audio-contribution": "strong"}))
        for i in range(5, 10):
            rows.append(_row(i, extras={"audio-contribution": "weak"}))
        snapshot_root = _build_fixture_snapshot(tmp_path, rows)
        _patch_snapshot(monkeypatch, snapshot_root)

        dataset = AudioMCQDataset(split="train", max_samples=4, seed=1)
        ids = [r["id"] for r in dataset.dataset]
        assert len(ids) == 4
        assert all(int(i) < 5 for i in ids), (
            "weak rows leaked through; max_samples must be applied AFTER the "
            f"audio-contribution filter. Got ids={ids}"
        )

    def test_eager_probe_raises_when_head_audio_missing(self, monkeypatch, tmp_path):
        rows = [_row(0), _row(1)]
        snapshot_root = _build_fixture_snapshot(tmp_path, rows, write_audio_for=[])
        _patch_snapshot(monkeypatch, snapshot_root)

        with pytest.raises(RuntimeError) as excinfo:
            AudioMCQDataset(split="train", max_samples=1)

        msg = str(excinfo.value)
        assert "AudioCaps" in msg
        assert snapshot_root in msg

    def test_eager_probe_does_not_decode_audio(self, monkeypatch, tmp_path):
        rows = [_row(0)]
        snapshot_root = _build_fixture_snapshot(tmp_path, rows)
        _patch_snapshot(monkeypatch, snapshot_root)

        called = {"sf_read": 0}
        original_read = sf.read

        def counting_read(*args, **kwargs):
            called["sf_read"] += 1
            return original_read(*args, **kwargs)

        monkeypatch.setattr(audiomcq_module.sf, "read", counting_read)

        AudioMCQDataset(split="train", max_samples=1)
        assert called["sf_read"] == 0

    def test_split_validation_size_creates_val_dataset(self, monkeypatch, tmp_path):
        rows = [_row(i) for i in range(10)]
        snapshot_root = _build_fixture_snapshot(tmp_path, rows)
        _patch_snapshot(monkeypatch, snapshot_root)

        dataset = AudioMCQDataset(split="train", split_validation_size=0.5, seed=42)
        assert dataset.val_dataset is not None
        assert len(dataset.val_dataset) > 0
        assert len(dataset.dataset) > 0

    def test_validation_split_is_seed_reproducible(self, monkeypatch, tmp_path):
        rows = [_row(i) for i in range(20)]
        snapshot_root = _build_fixture_snapshot(tmp_path, rows)
        _patch_snapshot(monkeypatch, snapshot_root)

        ds_a = AudioMCQDataset(split="train", split_validation_size=0.4, seed=99)
        ds_b = AudioMCQDataset(split="train", split_validation_size=0.4, seed=99)
        ds_c = AudioMCQDataset(split="train", split_validation_size=0.4, seed=200)

        train_ids_a = sorted(r["id"] for r in ds_a.dataset)
        train_ids_b = sorted(r["id"] for r in ds_b.dataset)
        train_ids_c = sorted(r["id"] for r in ds_c.dataset)
        val_ids_a = sorted(r["id"] for r in ds_a.val_dataset)
        val_ids_b = sorted(r["id"] for r in ds_b.val_dataset)

        assert train_ids_a == train_ids_b
        assert val_ids_a == val_ids_b
        # Different seed should produce a materially different split.
        assert train_ids_a != train_ids_c

    def test_split_validation_returns_held_out_slice_not_train(
        self, monkeypatch, tmp_path
    ):
        # Regression for round-1 review #2: split="validation" used to load
        # the train manifest split and then drop the held-out slice on the
        # floor, returning training rows. The wrapper must now route correctly.
        rows = [_row(i) for i in range(10)]
        snapshot_root = _build_fixture_snapshot(tmp_path, rows)
        _patch_snapshot(monkeypatch, snapshot_root)

        train_ds = AudioMCQDataset(split="train", split_validation_size=0.2, seed=42)
        val_ds = AudioMCQDataset(split="validation", split_validation_size=0.2, seed=42)

        train_ids = set(r["id"] for r in train_ds.dataset)
        held_out_ids = set(r["id"] for r in train_ds.val_dataset)
        val_loader_ids = set(r["id"] for r in val_ds.dataset)

        assert val_loader_ids == held_out_ids, (
            "split='validation' must return the held-out slice, not the training slice"
        )
        assert train_ids.isdisjoint(val_loader_ids), (
            "validation IDs must not overlap with training IDs"
        )

    def test_validation_split_requires_size(self, monkeypatch, tmp_path):
        rows = [_row(i) for i in range(5)]
        snapshot_root = _build_fixture_snapshot(tmp_path, rows)
        _patch_snapshot(monkeypatch, snapshot_root)

        with pytest.raises(ValueError, match="split_validation_size > 0"):
            AudioMCQDataset(split="validation", split_validation_size=0.0)


class TestAudioMCQFormatData:
    def test_format_data_emits_avqa_shape(self, monkeypatch, tmp_path):
        rows = [_row(0)]
        snapshot_root = _build_fixture_snapshot(tmp_path, rows)
        _patch_snapshot(monkeypatch, snapshot_root)

        dataset = AudioMCQDataset(split="train", max_samples=1)
        formatted = dataset.preprocessor(dataset.dataset[0])

        assert "messages" in formatted
        assert formatted["task_name"] == "audiomcq"
        assert formatted["choices"] == rows[0]["choices"]

        messages = formatted["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == rows[0]["answer"]

        user_content = messages[0]["content"]
        types = [c["type"] for c in user_content]
        assert types == ["audio", "text"]
        assert isinstance(user_content[0]["audio"], np.ndarray)
        assert user_content[0]["audio"].ndim == 1
        assert "<answer> </answer>" in user_content[1]["text"]

    def test_format_data_resamples_to_16khz(self, monkeypatch, tmp_path):
        # Build a fixture with a 32 kHz wav file.
        snapshot_root = tmp_path / "snapshot"
        snapshot_root.mkdir()
        rel = "AudioCaps/sample_0.wav"
        absolute = snapshot_root / rel
        absolute.parent.mkdir(parents=True, exist_ok=True)
        n = 32000  # 1 second @ 32kHz
        sf.write(str(absolute), np.zeros(n, dtype=np.float32), 32000)
        manifest = snapshot_root / AUDIOMCQ_MANIFEST
        rows = [_row(0)]
        with open(manifest, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        _patch_snapshot(monkeypatch, str(snapshot_root))

        dataset = AudioMCQDataset(split="train", max_samples=1)
        formatted = dataset.preprocessor(dataset.dataset[0])

        audio = formatted["messages"][0]["content"][0]["audio"]
        # 32k samples @ 32kHz resampled to 16kHz -> ~16k samples
        assert abs(len(audio) - 16000) <= 1

    def test_format_data_missing_file_raises_filenotfound(self, monkeypatch, tmp_path):
        # Two rows; head exists (eager probe ok) but second is missing.
        rows = [_row(0), _row(1, source="Tacos", ext="mp3")]
        snapshot_root = _build_fixture_snapshot(tmp_path, rows, write_audio_for=[0])
        _patch_snapshot(monkeypatch, snapshot_root)

        # Force deterministic head ordering: pre-shuffle yields different head,
        # so we instead patch shuffle to no-op for this test.
        monkeypatch.setattr(
            audiomcq_module.Dataset,
            "shuffle",
            lambda self, **kwargs: self,
        )
        dataset = AudioMCQDataset(split="train")

        # row 1 (Tacos) has no audio file on disk
        target = next(r for r in dataset.dataset if r["source_dataset"] == "Tacos")
        with pytest.raises(FileNotFoundError) as excinfo:
            dataset.preprocessor(target)
        msg = str(excinfo.value)
        assert "Tacos" in msg
        # The error message must include the resolved on-disk path so the
        # user can immediately identify which file is missing.
        expected_resolved = os.path.join(snapshot_root, target["audio_path"])
        assert expected_resolved in msg, (
            f"FileNotFoundError must include the resolved path "
            f"{expected_resolved!r}; got: {msg!r}"
        )


class _FakeFeatureExtractor:
    sampling_rate = 16000
    model_input_names: list[str] = []


class _FakeTokenizer:
    bos_token_id = None
    model_input_names = ["input_ids"]


class _FakeProcessor:
    """Minimal AutoProcessor stand-in for vlm_hf_data_processor."""

    feature_extractor = _FakeFeatureExtractor()
    tokenizer = _FakeTokenizer()
    model_input_names = ["input_ids"]

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        if tokenize:
            import torch as _torch

            return {"input_ids": _torch.tensor([[1, 2, 3, 4]])}
        return "<chat>fake</chat>"


class TestAudioMCQProcessor:
    def test_vlm_hf_data_processor_returns_audiomcq_datum_spec(
        self, monkeypatch, tmp_path
    ):
        rows = [_row(0)]
        snapshot_root = _build_fixture_snapshot(tmp_path, rows)
        _patch_snapshot(monkeypatch, snapshot_root)

        dataset = AudioMCQDataset(split="train", max_samples=1)
        formatted = dataset.preprocessor(dataset.dataset[0])

        from nemo_rl.data.interfaces import TaskDataSpec
        from nemo_rl.data.processors import vlm_hf_data_processor

        result = vlm_hf_data_processor(
            datum_dict=formatted,
            task_data_spec=TaskDataSpec(task_name="audiomcq"),
            processor=_FakeProcessor(),
            max_seq_length=4096,
            idx=0,
        )
        assert result["task_name"] == "audiomcq"
        assert "message_log" in result
        assert "extra_env_info" in result
        assert result["extra_env_info"]["ground_truth"] == rows[0]["answer"]
        assert result["extra_env_info"]["choices"] == rows[0]["choices"]

    def test_dispatcher_rejects_unknown_task_name(self):
        from nemo_rl.data.interfaces import TaskDataSpec
        from nemo_rl.data.processors import vlm_hf_data_processor

        bogus_datum = {
            "task_name": "definitely-not-a-task",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": "hello"},
            ],
        }
        with pytest.raises(ValueError, match="No data processor for task"):
            vlm_hf_data_processor(
                datum_dict=bogus_datum,
                task_data_spec=TaskDataSpec(task_name="definitely-not-a-task"),
                processor=_FakeProcessor(),
                max_seq_length=4096,
                idx=0,
            )
